// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for per-(provider, model) on-chain offerings.
//!
//! Exercises the full submission path through wallet → consensus →
//! executor → fullnode read for the three new tx kinds
//! (`RegisterOffering`, `UpdateOffering`, `DeactivateOffering`).
//! Tests verify the executor's invariants survive the network: the
//! offering must reference an active entry in the protocol-config
//! `ModelRegistry`, only the registering provider can update, and
//! `OpenChannel` against an unknown / inactive `model_id` is
//! rejected.

use test_cluster::{TestCluster, TestClusterBuilder};
use types::base::SomaAddress;
use types::channel::Channel;
use types::effects::TransactionEffectsAPI as _;
use types::object::{Object, ObjectID};
use types::offering::Offering;
use types::transaction::{
    DeactivateOfferingArgs, OpenChannelArgs, RegisterOfferingArgs, TransactionKind,
    UpdateOfferingArgs,
};
use utils::logging::init_tracing;

/// A model_id that is guaranteed to exist in the protocol-config
/// ModelRegistry at MIN protocol version — the sole launch model.
const VALID_MODEL: &str = "google/gemma-4-31b-it";

fn read_offering(test_cluster: &TestCluster, id: ObjectID) -> Option<Offering> {
    test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_object_store().get_object(&id).as_ref().and_then(Object::as_offering)
    })
}

async fn submit_register(
    test_cluster: &TestCluster,
    signer: SomaAddress,
    args: RegisterOfferingArgs,
) -> Result<bool, String> {
    let tx_data =
        e2e_tests::stateless_tx_data(test_cluster, signer, TransactionKind::RegisterOffering(args));
    match test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
    {
        Ok(r) => Ok(r.effects.status().is_ok()),
        Err(e) => Err(format!("{:?}", e)),
    }
}

async fn submit_update(
    test_cluster: &TestCluster,
    signer: SomaAddress,
    args: UpdateOfferingArgs,
) -> Result<bool, String> {
    let tx_data =
        e2e_tests::stateless_tx_data(test_cluster, signer, TransactionKind::UpdateOffering(args));
    match test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
    {
        Ok(r) => Ok(r.effects.status().is_ok()),
        Err(e) => Err(format!("{:?}", e)),
    }
}

async fn submit_deactivate(
    test_cluster: &TestCluster,
    signer: SomaAddress,
    args: DeactivateOfferingArgs,
) -> Result<bool, String> {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        signer,
        TransactionKind::DeactivateOffering(args),
    );
    match test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
    {
        Ok(r) => Ok(r.effects.status().is_ok()),
        Err(e) => Err(format!("{:?}", e)),
    }
}

/// Happy path: provider registers an offering for a registered model,
/// updates it, and deactivates it. Each step lands on chain and the
/// fullnode reflects the new state.
#[tokio::test]
async fn offering_lifecycle_register_update_deactivate() {
    init_tracing();
    let test_cluster = TestClusterBuilder::new().build().await;
    let signer = test_cluster.wallet.get_addresses()[0];

    // 1. Register.
    let ok = submit_register(
        &test_cluster,
        signer,
        RegisterOfferingArgs {
            model_id: VALID_MODEL.to_string(),
            prompt_micros_per_1k: 3_000,
            completion_micros_per_1k: 15_000,
            cache_read_micros_per_1k: 300,
            cache_write_micros_per_1k: 3_000,
            request_micros: 0,
            ttft_bound_ms: 1_500,
            ttot_bound_ms: 50,
        },
    )
    .await
    .expect("submit OK");
    assert!(ok, "register should succeed");

    let id = Offering::derive_id(signer, VALID_MODEL);
    let o = read_offering(&test_cluster, id).expect("offering created");
    assert_eq!(o.provider(), signer);
    assert_eq!(o.model_id(), VALID_MODEL);
    assert_eq!(o.prompt_micros_per_1k(), 3_000);
    assert_eq!(o.completion_micros_per_1k(), 15_000);
    assert_eq!(o.ttft_bound_ms(), 1_500);
    assert!(o.active());

    // 2. Update prices.
    let ok = submit_update(
        &test_cluster,
        signer,
        UpdateOfferingArgs {
            offering_id: id,
            model_id: VALID_MODEL.to_string(),
            prompt_micros_per_1k: 2_500,
            completion_micros_per_1k: 12_000,
            cache_read_micros_per_1k: 250,
            cache_write_micros_per_1k: 2_500,
            request_micros: 0,
            ttft_bound_ms: 1_200,
            ttot_bound_ms: 40,
        },
    )
    .await
    .expect("submit OK");
    assert!(ok, "update should succeed");

    let o = read_offering(&test_cluster, id).expect("offering still exists");
    assert_eq!(o.prompt_micros_per_1k(), 2_500);
    assert_eq!(o.ttft_bound_ms(), 1_200);
    assert!(o.active());

    // 3. Deactivate. Row stays, `active=false`.
    let ok = submit_deactivate(
        &test_cluster,
        signer,
        DeactivateOfferingArgs { offering_id: id, model_id: VALID_MODEL.to_string() },
    )
    .await
    .expect("submit OK");
    assert!(ok, "deactivate should succeed");

    let o = read_offering(&test_cluster, id).expect("offering still on chain");
    assert!(!o.active());
}

/// Registering an offering against an unknown `model_id` is rejected
/// — the executor checks the protocol-config ModelRegistry.
#[tokio::test]
async fn offering_register_rejects_unknown_model() {
    init_tracing();
    let test_cluster = TestClusterBuilder::new().build().await;
    let signer = test_cluster.wallet.get_addresses()[0];

    let ok = submit_register(
        &test_cluster,
        signer,
        RegisterOfferingArgs {
            model_id: "definitely/not-a-real-model".to_string(),
            prompt_micros_per_1k: 1,
            completion_micros_per_1k: 1,
            cache_read_micros_per_1k: 0,
            cache_write_micros_per_1k: 0,
            request_micros: 0,
            ttft_bound_ms: 1_000,
            ttot_bound_ms: 100,
        },
    )
    .await
    .expect("submit OK");
    assert!(!ok, "unknown model_id must be rejected");
}

/// `OpenChannel` against a (payee, model_id) pair with no offering
/// is rejected by the channel executor.
///
/// Driven by the `NotYetCreated` shared-input path: the offering
/// doesn't exist on chain, so the input loader passes the absence
/// through as a `NotYetCreated` marker, the tx lands cleanly, and
/// the executor surfaces `ChannelOfferingMissing` as the on-chain
/// failure.
#[tokio::test]
async fn open_channel_rejects_when_no_offering() {
    init_tracing();
    let test_cluster = TestClusterBuilder::new().build().await;
    let payer = test_cluster.wallet.get_addresses()[0];
    let payee = test_cluster.wallet.get_addresses()[1];
    // Note: no RegisterOffering — payee has no menu for this model.
    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        payer,
        TransactionKind::OpenChannel(OpenChannelArgs {
            payee,
            authorized_signer: payer,
            token: types::object::CoinType::Usdc,
            deposit_amount: 100_000,
            model_id: VALID_MODEL.to_string(),
        }),
    );
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .expect("submission completes");
    assert!(!r.effects.status().is_ok(), "OpenChannel without an active offering must be rejected");
}

/// Localnet integration: provider registers an offering at price X →
/// buyer opens a channel against it → channel materializes carrying
/// the snapshot → provider updates the offering to price Y → existing
/// channel still reflects X (snapshot invariant). This is the
/// "channel-as-oracle" property: mutating an offering can't
/// retroactively change settlement math for an already-open channel.
#[tokio::test]
async fn channel_snapshots_offering_at_open() {
    init_tracing();
    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    // 1. Provider registers offering at price X.
    let prompt_x = 5_000u64;
    let completion_x = 25_000u64;
    let ttft_x = 1_000u32;
    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        payee,
        TransactionKind::RegisterOffering(RegisterOfferingArgs {
            model_id: VALID_MODEL.to_string(),
            prompt_micros_per_1k: prompt_x,
            completion_micros_per_1k: completion_x,
            cache_read_micros_per_1k: 500,
            cache_write_micros_per_1k: 5_000,
            request_micros: 0,
            ttft_bound_ms: ttft_x,
            ttot_bound_ms: 80,
        }),
    );
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .expect("register submits");
    assert!(r.effects.status().is_ok(), "register at price X");

    // Give the validators a moment to propagate the new shared
    // object to the scheduler's shared-input resolver.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // 2. Buyer opens a channel against the offering.
    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        payer,
        TransactionKind::OpenChannel(OpenChannelArgs {
            payee,
            authorized_signer: payer,
            token: types::object::CoinType::Usdc,
            deposit_amount: 1_000_000,
            model_id: VALID_MODEL.to_string(),
        }),
    );
    let resp = test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .expect("open submits");
    assert!(resp.effects.status().is_ok(), "open should succeed");

    // Find the new Channel ID from effects.
    let created = resp.effects.created();
    let channel_id = created
        .iter()
        .find_map(|((id, _, _), _)| {
            test_cluster.fullnode_handle.soma_node.with(|node| {
                node.state()
                    .get_object_store()
                    .get_object(id)
                    .as_ref()
                    .and_then(|o| o.as_channel())
                    .map(|_| *id)
            })
        })
        .expect("a Channel object was created");

    // Channel should reflect price X.
    let ch: Channel = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| {
            node.state()
                .get_object_store()
                .get_object(&channel_id)
                .as_ref()
                .and_then(Object::as_channel)
        })
        .expect("channel object readable");
    assert_eq!(ch.model_id(), VALID_MODEL);
    assert_eq!(ch.prompt_micros_per_1k(), prompt_x);
    assert_eq!(ch.completion_micros_per_1k(), completion_x);
    assert_eq!(ch.ttft_bound_ms(), ttft_x);

    // 3. Provider updates offering to price Y.
    let prompt_y = 9_000u64;
    let completion_y = 45_000u64;
    let offering_id = Offering::derive_id(payee, VALID_MODEL);
    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        payee,
        TransactionKind::UpdateOffering(UpdateOfferingArgs {
            offering_id,
            model_id: VALID_MODEL.to_string(),
            prompt_micros_per_1k: prompt_y,
            completion_micros_per_1k: completion_y,
            cache_read_micros_per_1k: 900,
            cache_write_micros_per_1k: 9_000,
            request_micros: 0,
            ttft_bound_ms: 2_000,
            ttot_bound_ms: 100,
        }),
    );
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .expect("update submits");
    assert!(r.effects.status().is_ok(), "update at price Y");

    // 4. Existing channel must still carry price X — offering update
    //    does NOT retroactively change settlement math.
    let ch_after = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| {
            node.state()
                .get_object_store()
                .get_object(&channel_id)
                .as_ref()
                .and_then(Object::as_channel)
        })
        .expect("channel still on chain");
    assert_eq!(
        ch_after.prompt_micros_per_1k(),
        prompt_x,
        "channel snapshot must survive offering update"
    );
    assert_eq!(ch_after.completion_micros_per_1k(), completion_x);
    assert_eq!(ch_after.ttft_bound_ms(), ttft_x);

    // 5. But the offering itself shows price Y.
    let o = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| {
            node.state()
                .get_object_store()
                .get_object(&offering_id)
                .as_ref()
                .and_then(Object::as_offering)
        })
        .expect("offering still on chain");
    assert_eq!(o.prompt_micros_per_1k(), prompt_y);
    assert_eq!(o.completion_micros_per_1k(), completion_y);
}

/// Regression for the version-poisoning chain-halt DoS: a failed
/// `OpenChannel` that declared the payee's `ProviderInbox` as a mutable
/// shared input bumps the inbox's `next_version` without materializing
/// the object, then a subsequent `OpenChannel` for the same payee gets
/// the bumped (non-initial) version. Previously the input loader
/// `panic!`d on this combination (`transaction_input_loader.rs:226`)
/// and every validator replayed the panic on boot → cluster
/// `CrashLoopBackOff`. The fix surfaces the missing-object case as
/// `NotYetCreated` regardless of the assigned version when the object
/// has never been materialized on chain, so the second OpenChannel
/// executes cleanly and the inbox is lazy-created at the bumped
/// version.
#[tokio::test]
async fn open_channel_after_failed_open_for_same_payee_does_not_panic() {
    init_tracing();
    let test_cluster = TestClusterBuilder::new().build().await;
    let payer = test_cluster.wallet.get_addresses()[0];
    let payee = test_cluster.wallet.get_addresses()[1];

    // 1. Register a valid offering for (payee, VALID_MODEL) so the
    // second open below has something to snapshot.
    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        payee,
        TransactionKind::RegisterOffering(RegisterOfferingArgs {
            model_id: VALID_MODEL.to_string(),
            prompt_micros_per_1k: 3_000,
            completion_micros_per_1k: 15_000,
            cache_read_micros_per_1k: 300,
            cache_write_micros_per_1k: 3_000,
            request_micros: 0,
            ttft_bound_ms: 1_500,
            ttot_bound_ms: 50,
        }),
    );
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .expect("register submits");
    assert!(r.effects.status().is_ok(), "register offering must succeed");

    // 2. First OpenChannel against `payee` that FAILS at the executor
    // (self-payee — `payer == payee` is rejected). This still goes
    // through consensus, so the scheduler bumps the payee's
    // ProviderInbox `next_version` even though the executor aborts
    // without materializing the inbox object.
    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        payee,
        TransactionKind::OpenChannel(OpenChannelArgs {
            payee, // self-payee → executor rejects
            authorized_signer: payee,
            token: types::object::CoinType::Usdc,
            deposit_amount: 50_000,
            model_id: VALID_MODEL.to_string(),
        }),
    );
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .expect("failed-open submits cleanly");
    assert!(
        !r.effects.status().is_ok(),
        "self-payee OpenChannel must fail at the executor (this is the trigger that bumps next_version)"
    );

    // 3. Second OpenChannel for the SAME payee from a different payer.
    // The scheduler hands this tx the bumped, non-initial version for
    // `ProviderInbox(payee)`. Before the fix this would `panic!` in
    // `read_objects_for_execution` because the object didn't exist
    // at the bumped version AND the `version == initial` escape
    // hatch didn't fire. After the fix the input loader detects
    // "object never existed at any version" and surfaces
    // `NotYetCreated(bumped_version)`, the executor lazy-creates the
    // inbox at the lamport-incremented version, the channel opens
    // normally, and no validator panics.
    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        payer,
        TransactionKind::OpenChannel(OpenChannelArgs {
            payee,
            authorized_signer: payer,
            token: types::object::CoinType::Usdc,
            deposit_amount: 100_000,
            model_id: VALID_MODEL.to_string(),
        }),
    );
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .expect("second open submits cleanly (no validator panic)");
    assert!(
        r.effects.status().is_ok(),
        "second OpenChannel must succeed against a poisoned ProviderInbox version: status={:?}",
        r.effects.status()
    );

    // 4. Chain liveness check: a third tx after the poison-then-recover
    // sequence must still land — proves validators kept producing
    // checkpoints (i.e. they did not all crash on the second
    // OpenChannel and recover only via restart).
    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        payee,
        TransactionKind::DeactivateOffering(DeactivateOfferingArgs {
            offering_id: Offering::derive_id(payee, VALID_MODEL),
            model_id: VALID_MODEL.to_string(),
        }),
    );
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .expect("post-recovery tx lands");
    assert!(
        r.effects.status().is_ok(),
        "chain must still advance after the poison-recovery sequence"
    );
}
