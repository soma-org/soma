// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the inference-related indexer pipelines:
//!
//!   - `soma_providers` — `RegisterProvider` / `UpdateProvider`
//!   - `soma_channels` — `OpenChannel` / `Settle` / `RequestClose` /
//!     `TopUp` / `WithdrawAfterTimeout`
//!   - `soma_channel_events` — append-only event log fueling
//!     `provider_reputation`
//!
//! These tests boot a `TestCluster` (real chain), an `OffchainCluster`
//! (real indexer-alt + Postgres), submit transactions, and assert on
//! the resulting Postgres rows.
//!
//! Requires Postgres on PATH. Run with:
//!   `cargo test -p indexer-e2e-tests --test inference_pipeline_e2e -- --ignored --nocapture`

use std::ops::DerefMut;
use std::time::Duration;

use diesel::prelude::*;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::*;
use indexer_e2e_tests::OffchainCluster;
use indexer_framework::IndexerArgs;
use soma_keys::keystore::AccountKeystore as _;
use test_cluster::TestClusterBuilder;
use types::base::SomaAddress;
use types::channel::Voucher;
use types::crypto::{GenericSignature, Signature};
use types::effects::TransactionEffectsAPI;
use types::intent::{Intent, IntentMessage, IntentScope};
use types::object::{CoinType, ObjectID};
use types::provider::Provider;
use types::transaction::{
    OpenChannelArgs, RegisterProviderArgs, RequestCloseArgs, SettleArgs, TopUpArgs,
    TransactionKind, UpdateProviderArgs,
};

const STATUS_OPEN: i16 = 0;
const STATUS_CLOSING: i16 = 1;

/// Sign a `Voucher` with the channel's `authorized_signer` key.
/// Mirrors what the proxy does per request, but synchronous since
/// these tests don't go through the HTTP path.
async fn sign_voucher(
    test_cluster: &test_cluster::TestCluster,
    signer: SomaAddress,
    channel_id: ObjectID,
    cumulative_amount: u64,
) -> GenericSignature {
    let voucher = Voucher::new(channel_id, cumulative_amount);
    let sig: Signature = test_cluster
        .wallet
        .config
        .keystore
        .sign_secure::<Voucher>(
            &signer,
            &voucher,
            Intent::soma_app(IntentScope::PaymentVoucher),
        )
        .await
        .expect("voucher signing succeeds");
    sig.into()
}

/// Submit `OpenChannel` and return the new channel's ObjectID.
async fn open_channel(
    test_cluster: &test_cluster::TestCluster,
    payer: SomaAddress,
    payee: SomaAddress,
    deposit_amount: u64,
) -> ObjectID {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payer,
        TransactionKind::OpenChannel(OpenChannelArgs {
            payee,
            authorized_signer: payer,
            token: CoinType::Usdc,
            deposit_amount,
        }),
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok(), "OpenChannel must succeed");
    let created = response.effects.created();
    let chan_oref = created
        .iter()
        .find(|(_oref, owner)| owner.is_shared())
        .expect("OpenChannel creates a shared Channel object");
    chan_oref.0.0
}

/// Provider registration helper.
async fn register_provider(
    test_cluster: &test_cluster::TestCluster,
    signer: SomaAddress,
    endpoint: &str,
) {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        signer,
        TransactionKind::RegisterProvider(RegisterProviderArgs {
            endpoint: endpoint.to_string(),
        }),
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok(), "RegisterProvider must succeed");
}

/// Endpoint-update helper.
async fn update_provider(
    test_cluster: &test_cluster::TestCluster,
    signer: SomaAddress,
    endpoint: &str,
) {
    let provider_id = Provider::derive_id(signer);
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        signer,
        TransactionKind::UpdateProvider(UpdateProviderArgs {
            provider_id,
            endpoint: endpoint.to_string(),
        }),
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok(), "UpdateProvider must succeed");
}

/// Submit `Settle` for an existing channel + voucher.
async fn submit_settle(
    test_cluster: &test_cluster::TestCluster,
    payee: SomaAddress,
    channel_id: ObjectID,
    cumulative_amount: u64,
    voucher_signature: GenericSignature,
) {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payee,
        TransactionKind::Settle(SettleArgs {
            channel_id,
            cumulative_amount,
            voucher_signature,
        }),
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok(), "Settle must succeed");
}

/// Submit `TopUp` to refill an existing channel.
async fn submit_top_up(
    test_cluster: &test_cluster::TestCluster,
    payer: SomaAddress,
    channel_id: ObjectID,
    amount: u64,
) {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payer,
        TransactionKind::TopUp(TopUpArgs {
            channel_id,
            coin_type: CoinType::Usdc,
            amount,
        }),
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok(), "TopUp must succeed");
}

async fn submit_request_close(
    test_cluster: &test_cluster::TestCluster,
    payer: SomaAddress,
    channel_id: ObjectID,
) {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payer,
        TransactionKind::RequestClose(RequestCloseArgs { channel_id }),
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok(), "RequestClose must succeed");
}

/// Latest checkpoint sequence from the indexer's view, used as the
/// catch-up target after each tx batch.
async fn current_cp(test_cluster: &test_cluster::TestCluster) -> u64 {
    test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_checkpoint_store()
            .get_highest_executed_checkpoint_seq_number()
            .ok()
            .flatten()
            .unwrap_or(0)
    })
}

/// `RegisterProvider` → row in `soma_providers`. `UpdateProvider`
/// changes endpoint and bumps `last_update_cp`.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_provider_registry_indexed() {
    let _ = tracing_subscriber::fmt::try_init();

    let ingestion_dir = tempfile::tempdir().unwrap();
    let ingestion_path = ingestion_dir.path().to_path_buf();

    let test_cluster = TestClusterBuilder::new()
        .with_data_ingestion_dir(ingestion_path.clone())
        .build()
        .await;
    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(&ingestion_path, IndexerArgs::default(), &registry)
        .await
        .expect("OffchainCluster boot");

    let signer = test_cluster.wallet.get_addresses()[0];

    register_provider(&test_cluster, signer, "https://prov-v1.example").await;
    let cp1 = current_cp(&test_cluster).await;
    cluster
        .wait_for_indexer(cp1, Duration::from_secs(60))
        .await
        .expect("indexer reaches cp1");

    let mut conn = cluster.db().connect().await.unwrap();
    let row: (Vec<u8>, String, i64) = soma_providers::table
        .select((
            soma_providers::address,
            soma_providers::endpoint,
            soma_providers::last_update_cp,
        ))
        .filter(soma_providers::address.eq(signer.to_vec()))
        .first(conn.deref_mut())
        .await
        .expect("provider row present after RegisterProvider");
    assert_eq!(row.0, signer.to_vec());
    assert_eq!(row.1, "https://prov-v1.example");
    let cp_at_register = row.2;

    update_provider(&test_cluster, signer, "https://prov-v2.example").await;
    let cp2 = current_cp(&test_cluster).await;
    cluster
        .wait_for_indexer(cp2, Duration::from_secs(60))
        .await
        .expect("indexer reaches cp2");

    let row: (String, i64) = soma_providers::table
        .select((soma_providers::endpoint, soma_providers::last_update_cp))
        .filter(soma_providers::address.eq(signer.to_vec()))
        .first(conn.deref_mut())
        .await
        .unwrap();
    assert_eq!(row.0, "https://prov-v2.example", "endpoint updated");
    assert!(row.1 > cp_at_register, "last_update_cp must advance on update");
}

/// Full channel lifecycle: open → top_up → settle → settle (twice) →
/// request_close. Verify `soma_channels` mirrors state and
/// `soma_channel_events` accumulates one row per op with correct
/// kind+delta.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_channel_lifecycle_indexed() {
    let _ = tracing_subscriber::fmt::try_init();

    let ingestion_dir = tempfile::tempdir().unwrap();
    let ingestion_path = ingestion_dir.path().to_path_buf();

    let test_cluster = TestClusterBuilder::new()
        .with_data_ingestion_dir(ingestion_path.clone())
        .build()
        .await;
    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(&ingestion_path, IndexerArgs::default(), &registry)
        .await
        .expect("OffchainCluster boot");

    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    // 1. Open
    let initial_deposit = 1_000_000u64;
    let channel_id = open_channel(&test_cluster, payer, payee, initial_deposit).await;
    let cp_open = current_cp(&test_cluster).await;
    cluster
        .wait_for_indexer(cp_open, Duration::from_secs(60))
        .await
        .expect("indexer reaches cp_open");

    let mut conn = cluster.db().connect().await.unwrap();
    let chan: (Vec<u8>, Vec<u8>, Vec<u8>, String, i64, i64, i16) = soma_channels::table
        .select((
            soma_channels::channel_id,
            soma_channels::payer,
            soma_channels::payee,
            soma_channels::token,
            soma_channels::deposit,
            soma_channels::settled_amount,
            soma_channels::status,
        ))
        .filter(soma_channels::channel_id.eq(channel_id.to_vec()))
        .first(conn.deref_mut())
        .await
        .expect("channel row after Open");
    assert_eq!(chan.1, payer.to_vec());
    assert_eq!(chan.2, payee.to_vec());
    assert_eq!(chan.3, "USDC");
    assert_eq!(chan.4, initial_deposit as i64, "deposit");
    assert_eq!(chan.5, 0, "settled_amount starts at 0");
    assert_eq!(chan.6, STATUS_OPEN);

    // Event log: exactly one `open` row with delta == initial_deposit.
    let events: Vec<(String, i64)> = soma_channel_events::table
        .select((soma_channel_events::kind, soma_channel_events::delta))
        .filter(soma_channel_events::channel_id.eq(channel_id.to_vec()))
        .order(soma_channel_events::cp_sequence_number.asc())
        .load(conn.deref_mut())
        .await
        .unwrap();
    assert_eq!(events, vec![("open".to_string(), initial_deposit as i64)]);

    // 2. TopUp +500_000.
    let top_up_amount = 500_000u64;
    submit_top_up(&test_cluster, payer, channel_id, top_up_amount).await;
    let cp_top_up = current_cp(&test_cluster).await;
    cluster
        .wait_for_indexer(cp_top_up, Duration::from_secs(60))
        .await
        .expect("indexer reaches cp_top_up");

    let chan: (i64, i64) = soma_channels::table
        .select((soma_channels::deposit, soma_channels::settled_amount))
        .filter(soma_channels::channel_id.eq(channel_id.to_vec()))
        .first(conn.deref_mut())
        .await
        .unwrap();
    assert_eq!(chan.0, (initial_deposit + top_up_amount) as i64, "deposit grew");
    assert_eq!(chan.1, 0);

    // Event log gained a `top_up` row with delta == top_up_amount.
    let events: Vec<(String, i64)> = soma_channel_events::table
        .select((soma_channel_events::kind, soma_channel_events::delta))
        .filter(soma_channel_events::channel_id.eq(channel_id.to_vec()))
        .order(soma_channel_events::cp_sequence_number.asc())
        .then_order_by(soma_channel_events::tx_sequence_number.asc())
        .load(conn.deref_mut())
        .await
        .unwrap();
    assert_eq!(
        events,
        vec![
            ("open".to_string(), initial_deposit as i64),
            ("top_up".to_string(), top_up_amount as i64),
        ],
    );

    // 3. Two settles — cumulative=300, then cumulative=700. Each
    //    settle's delta is the *increment* in settled_amount, not the
    //    absolute cumulative.
    let sig1 = sign_voucher(&test_cluster, payer, channel_id, 300).await;
    submit_settle(&test_cluster, payee, channel_id, 300, sig1).await;
    let cp_s1 = current_cp(&test_cluster).await;
    cluster.wait_for_indexer(cp_s1, Duration::from_secs(60)).await.unwrap();

    let sig2 = sign_voucher(&test_cluster, payer, channel_id, 700).await;
    submit_settle(&test_cluster, payee, channel_id, 700, sig2).await;
    let cp_s2 = current_cp(&test_cluster).await;
    cluster.wait_for_indexer(cp_s2, Duration::from_secs(60)).await.unwrap();

    let chan: (i64, i64, i16) = soma_channels::table
        .select((
            soma_channels::deposit,
            soma_channels::settled_amount,
            soma_channels::status,
        ))
        .filter(soma_channels::channel_id.eq(channel_id.to_vec()))
        .first(conn.deref_mut())
        .await
        .unwrap();
    assert_eq!(chan.1, 700, "settled_amount equals last cumulative");
    assert_eq!(
        chan.0,
        (initial_deposit + top_up_amount - 700) as i64,
        "deposit decreased by total settled"
    );
    assert_eq!(chan.2, STATUS_OPEN);

    // Event log: settle deltas are 300 (first) and 400 (700-300).
    let settles: Vec<i64> = soma_channel_events::table
        .select(soma_channel_events::delta)
        .filter(soma_channel_events::channel_id.eq(channel_id.to_vec()))
        .filter(soma_channel_events::kind.eq("settle"))
        .order(soma_channel_events::cp_sequence_number.asc())
        .then_order_by(soma_channel_events::tx_sequence_number.asc())
        .load(conn.deref_mut())
        .await
        .unwrap();
    assert_eq!(settles, vec![300, 400], "settle deltas");

    // 4. RequestClose flips status → CLOSING and stamps
    // `close_requested_at_ms`.
    submit_request_close(&test_cluster, payer, channel_id).await;
    let cp_rc = current_cp(&test_cluster).await;
    cluster.wait_for_indexer(cp_rc, Duration::from_secs(60)).await.unwrap();

    let chan: (i16, Option<i64>) = soma_channels::table
        .select((
            soma_channels::status,
            soma_channels::close_requested_at_ms,
        ))
        .filter(soma_channels::channel_id.eq(channel_id.to_vec()))
        .first(conn.deref_mut())
        .await
        .unwrap();
    assert_eq!(chan.0, STATUS_CLOSING);
    assert!(chan.1.is_some() && chan.1.unwrap() > 0);

    // 5. Subsequent TopUp clears the close timer (per the executor's
    // semantics: TopUp clears `close_requested_at_ms`).
    submit_top_up(&test_cluster, payer, channel_id, 50_000).await;
    let cp_tu2 = current_cp(&test_cluster).await;
    cluster.wait_for_indexer(cp_tu2, Duration::from_secs(60)).await.unwrap();

    let chan: (i16, Option<i64>) = soma_channels::table
        .select((
            soma_channels::status,
            soma_channels::close_requested_at_ms,
        ))
        .filter(soma_channels::channel_id.eq(channel_id.to_vec()))
        .first(conn.deref_mut())
        .await
        .unwrap();
    assert_eq!(chan.0, STATUS_OPEN, "TopUp clears the close timer");
    assert!(chan.1.is_none());

    tracing::info!("test_channel_lifecycle_indexed passed");
}

// `WithdrawAfterTimeout` indexer mapping (status flips to `withdrawn`,
// row preserved post-deletion) is covered by the executor-side msim
// tests in `e2e-tests/tests/channel_tests.rs::channel_full_lifecycle`,
// which exercises the same effects shape that drives the indexer
// handler. A non-msim version would need either a 10-minute grace
// period wait (the production default) or a protocol-config override
// the standard `TestClusterBuilder` doesn't expose. Adding the
// override builder is a separate refactor; the indexer handler's
// withdraw branch is already covered by the unit-level handler tests.
