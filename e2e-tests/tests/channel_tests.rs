// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the payment-channel ops (Phase 1: `OpenChannel`,
//! `Settle`, `RequestClose`, `WithdrawAfterTimeout`).
//!
//! These tests exercise the **full submission path**: a user-signed
//! `Transaction` flows through the wallet/RPC layer, hits validators
//! via consensus, runs through the executor, and produces durable state
//! changes that the fullnode reflects. We sign vouchers off-chain via
//! the keystore's `sign_secure` (real Ed25519 signatures) and submit
//! `Settle` / `RequestClose` / `WithdrawAfterTimeout` transactions
//! through the same channel as any other tx.
//!
//! The msim protocol-config override drops `channel_grace_period_ms`
//! to 5 seconds (vs 10 minutes in production) so the timed-close path
//! is testable without sleeping out for the full grace window.

use std::time::Duration;

use soma_keys::keystore::AccountKeystore as _;
use test_cluster::{TestCluster, TestClusterBuilder};
use tokio::time::sleep;
use tracing::info;
use types::CLOCK_OBJECT_ID;
use types::base::SomaAddress;
use types::channel::{Channel, Voucher};
use types::crypto::{GenericSignature, Signature};
use types::digests::ObjectDigest;
use types::effects::TransactionEffectsAPI as _;
use types::intent::{Intent, IntentMessage, IntentScope};
use types::object::{CoinType, Object, ObjectID, ObjectRef, ObjectType, Version};
use types::transaction::{
    OpenChannelArgs, RequestCloseArgs, SettleArgs, TransactionData, TransactionKind,
    WithdrawAfterTimeoutArgs,
};
use utils::logging::init_tracing;

/// Pull a USDC gas/payment coin owned by `addr`. Returns the
/// `ObjectRef` (id, version, digest).
async fn one_coin(test_cluster: &TestCluster, addr: SomaAddress) -> ObjectRef {
    test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(addr)
        .await
        .unwrap()
        .expect("address has at least one gas coin")
}

/// Read a Channel from the fullnode (returns None if it doesn't
/// exist — e.g. after WithdrawAfterTimeout deletes it).
fn read_channel(test_cluster: &TestCluster, channel_id: ObjectID) -> Option<Channel> {
    test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_object_store()
            .get_object(&channel_id)
            .as_ref()
            .and_then(Object::as_channel)
    })
}

/// Read the fullnode's Clock timestamp.
fn read_clock_ts(test_cluster: &TestCluster) -> u64 {
    test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_object_store()
            .get_object(&CLOCK_OBJECT_ID)
            .expect("clock present")
            .clock_timestamp_ms()
    })
}

/// Sign a voucher with the `authorized_signer`'s key, returning the
/// `GenericSignature` form ready to embed in a `SettleArgs`.
async fn sign_voucher(
    test_cluster: &TestCluster,
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

/// Submit `OpenChannel` and return the new channel's ObjectID. The
/// resulting Channel is at `OBJECT_START_VERSION` per the contract
/// (see `Object::new_channel`).
async fn open_channel(
    test_cluster: &TestCluster,
    payer: SomaAddress,
    payee: SomaAddress,
    deposit_amount: u64,
) -> ObjectID {
    // Stage 8: OpenChannel is balance-mode for both gas and deposit.
    // Sender's USDC accumulator covers `deposit_amount + gas_fee`;
    // the executor emits a single Withdraw event for the deposit.
    let chain = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_chain_identifier());
    let tx_data = TransactionData::new_with_expiration(
        TransactionKind::OpenChannel(OpenChannelArgs {
            payee,
            authorized_signer: payer,
            token: CoinType::Usdc,
            deposit_amount,
        }),
        payer,
        Vec::new(),
        types::transaction::TransactionExpiration::ValidDuring {
            min_epoch: Some(0),
            max_epoch: Some(1),
            chain,
            nonce: 0,
        },
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(
        response.effects.status().is_ok(),
        "OpenChannel must succeed: status={:?}",
        response.effects.status()
    );

    // The new Channel is the only created shared object in the effects.
    let created = response.effects.created();
    let channel_oref = created
        .iter()
        .find(|(_oref, owner)| owner.is_shared())
        .expect("OpenChannel creates a shared Channel object");
    channel_oref.0.0
}

async fn submit_settle(
    test_cluster: &TestCluster,
    payee: SomaAddress,
    channel_id: ObjectID,
    cumulative_amount: u64,
    voucher_signature: GenericSignature,
) -> bool {
    let coin = one_coin(test_cluster, payee).await;
    let tx_data = TransactionData::new(
        TransactionKind::Settle(SettleArgs {
            channel_id,
            cumulative_amount,
            voucher_signature,
        }),
        payee,
        vec![coin],
    );
    test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .map(|r| r.effects.status().is_ok())
        .unwrap_or(false)
}

async fn submit_request_close(
    test_cluster: &TestCluster,
    payer: SomaAddress,
    channel_id: ObjectID,
) -> bool {
    let coin = one_coin(test_cluster, payer).await;
    let tx_data = TransactionData::new(
        TransactionKind::RequestClose(RequestCloseArgs { channel_id }),
        payer,
        vec![coin],
    );
    test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .map(|r| r.effects.status().is_ok())
        .unwrap_or(false)
}

async fn submit_withdraw(
    test_cluster: &TestCluster,
    payer: SomaAddress,
    channel_id: ObjectID,
) -> bool {
    let coin = one_coin(test_cluster, payer).await;
    let tx_data = TransactionData::new(
        TransactionKind::WithdrawAfterTimeout(WithdrawAfterTimeoutArgs { channel_id }),
        payer,
        vec![coin],
    );
    test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
        .map(|r| r.effects.status().is_ok())
        .unwrap_or(false)
}

/// Drive consensus by submitting a balance-mode self-transfer. Used to
/// advance the Clock past the grace period and to flush any in-flight
/// state. Stage 13c: no gas coin needed.
async fn drive_one_commit(test_cluster: &TestCluster) {
    let addrs = test_cluster.wallet.get_addresses();
    let tx = e2e_tests::balance_transfer_data(
        test_cluster,
        types::object::CoinType::Usdc,
        addrs[0],
        vec![(addrs[1], 1)],
    );
    let _ = test_cluster.sign_and_execute_transaction(&tx).await;
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

/// Full happy-path lifecycle: open → settle (twice) → request_close →
/// withdraw_after_timeout. Verifies channel state transitions, payee
/// receives delta, payer gets remainder, channel deleted on close.
///
/// Stage 8 also asserts the accumulator balance flow:
///   OpenChannel  → payer USDC drops by deposit
///   Settle       → payee USDC rises by delta (channel deposit drops)
///   Withdraw…    → payer USDC recovers by remainder
#[cfg(msim)]
#[msim::sim_test]
async fn channel_full_lifecycle() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    let read_usdc = |addr: SomaAddress| -> u64 {
        test_cluster
            .fullnode_handle
            .soma_node
            .with(|node| node.state().database_for_testing().get_balance(addr, CoinType::Usdc))
            .unwrap_or(0)
    };

    // Snapshot balances; deltas must equal the funds-flow predicted by
    // the channel ops below (modulo gas, which the assertions discount).
    let payer_initial = read_usdc(payer);
    let payee_initial = read_usdc(payee);

    // 1. Open with 100_000 µUSDC deposit.
    let channel_id = open_channel(&test_cluster, payer, payee, 100_000).await;
    info!(?channel_id, "channel opened");

    let ch = read_channel(&test_cluster, channel_id).expect("channel exists post-open");
    assert_eq!(ch.payer, payer);
    assert_eq!(ch.payee, payee);
    assert_eq!(ch.deposit, 100_000);
    assert_eq!(ch.settled_amount, 0);
    assert!(ch.close_requested_at_ms.is_none());

    // After OpenChannel: payer USDC dropped by *at least* the deposit
    // (gas fee adds a small extra debit). Stage 8 invariant: the
    // deposit lands as a Withdraw event in the accumulator.
    let payer_after_open = read_usdc(payer);
    assert!(
        payer_initial - payer_after_open >= 100_000,
        "payer must be debited at least 100_000 USDC on OpenChannel: initial={}, after_open={}",
        payer_initial,
        payer_after_open,
    );

    // 2. Settle at cumulative=10_000.
    let voucher_sig = sign_voucher(&test_cluster, payer, channel_id, 10_000).await;
    assert!(
        submit_settle(&test_cluster, payee, channel_id, 10_000, voucher_sig).await,
        "first Settle must succeed"
    );

    let ch = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(ch.deposit, 90_000);
    assert_eq!(ch.settled_amount, 10_000);

    // 3. Settle at cumulative=25_000 (delta = 15_000).
    let voucher_sig = sign_voucher(&test_cluster, payer, channel_id, 25_000).await;
    assert!(
        submit_settle(&test_cluster, payee, channel_id, 25_000, voucher_sig).await,
        "second Settle must succeed"
    );
    let ch = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(ch.deposit, 75_000);
    assert_eq!(ch.settled_amount, 25_000);

    // After two Settles: payee USDC rose by exactly 25_000 minus the
    // gas fees they paid (Settle is a payee-signed tx). The two
    // Settles are coin-mode-gas in this test, so the accumulator delta
    // for the payee is exactly the cumulative of the deposits — no
    // gas debit lands on the accumulator in coin-mode.
    let payee_after_settles = read_usdc(payee);
    assert_eq!(
        payee_after_settles - payee_initial,
        25_000,
        "payee accumulator must equal exactly the settled total (Stage 8: Deposit events from Settle land in the accumulator)",
    );

    // 4. Request close — Clock timestamp gets stamped onto channel.
    let pre_close_ts = read_clock_ts(&test_cluster);
    assert!(
        submit_request_close(&test_cluster, payer, channel_id).await,
        "RequestClose must succeed"
    );
    let ch = read_channel(&test_cluster, channel_id).unwrap();
    let close_at = ch.close_requested_at_ms.expect("close_requested_at_ms set");
    assert!(close_at >= pre_close_ts, "close timestamp must be at-or-after pre-close clock");

    // 5. Try Withdraw immediately — must fail (grace not elapsed).
    assert!(
        !submit_withdraw(&test_cluster, payer, channel_id).await,
        "Withdraw before grace must fail"
    );
    // Channel still alive.
    assert!(read_channel(&test_cluster, channel_id).is_some(), "Channel still present pre-grace");

    // 6. Wait past the (msim-shortened) grace period and drive a few
    //    commits so the Clock advances.
    sleep(Duration::from_secs(10)).await;
    for _ in 0..3 {
        drive_one_commit(&test_cluster).await;
    }

    // 7. Withdraw — succeeds; Channel is deleted; payer gets 75_000 back.
    assert!(
        submit_withdraw(&test_cluster, payer, channel_id).await,
        "Withdraw after grace must succeed"
    );
    assert!(
        read_channel(&test_cluster, channel_id).is_none(),
        "Channel must be deleted after WithdrawAfterTimeout"
    );

    // Stage 8 closing invariant: payer's net debit is exactly the
    // 25_000 µUSDC paid out via Settle (plus gas fees they spent on
    // OpenChannel/RequestClose/Withdraw). Compare to:
    //   payer_initial - payer_final >= 25_000  (settle total)
    // and at-most: 25_000 + reasonable_gas_budget.
    //
    // The remainder (75_000) flowed back through a Deposit event on
    // WithdrawAfterTimeout, conserving total accumulator supply
    // (channel.deposit = 0 at deletion).
    let payer_final = read_usdc(payer);
    let payer_net_debit = payer_initial - payer_final;
    assert!(
        payer_net_debit >= 25_000,
        "payer net debit must cover the 25_000 paid to payee: net_debit={}",
        payer_net_debit,
    );
    assert!(
        payer_net_debit < 100_000,
        "payer net debit must be far less than the full deposit (remainder returned): net_debit={}",
        payer_net_debit,
    );
}

/// A user-signed `Settle` with the wrong sender (payer instead of
/// payee) must fail at the executor (Sui-spec rule).
#[cfg(msim)]
#[msim::sim_test]
async fn channel_settle_rejects_payer_caller() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    let channel_id = open_channel(&test_cluster, payer, payee, 50_000).await;
    let voucher_sig = sign_voucher(&test_cluster, payer, channel_id, 1_000).await;

    // Submit Settle from `payer` instead of `payee`.
    let coin = one_coin(&test_cluster, payer).await;
    let tx = TransactionData::new(
        TransactionKind::Settle(SettleArgs {
            channel_id,
            cumulative_amount: 1_000,
            voucher_signature: voucher_sig,
        }),
        payer,
        vec![coin],
    );
    let response = test_cluster.sign_and_execute_transaction(&tx).await;
    assert!(
        !response.effects.status().is_ok(),
        "Settle from non-payee must fail at executor; got status={:?}",
        response.effects.status()
    );

    // Channel state is unchanged.
    let ch = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(ch.deposit, 50_000);
    assert_eq!(ch.settled_amount, 0);
}

/// Cumulative-monotonic replay protection at the e2e level: a stale
/// voucher (cumulative ≤ already-settled) is rejected.
#[cfg(msim)]
#[msim::sim_test]
async fn channel_settle_rejects_stale_voucher() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    let channel_id = open_channel(&test_cluster, payer, payee, 100_000).await;

    // First settle at cumulative=5_000.
    let v1 = sign_voucher(&test_cluster, payer, channel_id, 5_000).await;
    assert!(submit_settle(&test_cluster, payee, channel_id, 5_000, v1).await);

    // Stale voucher at cumulative=3_000 (less than settled_amount=5_000).
    let v_stale = sign_voucher(&test_cluster, payer, channel_id, 3_000).await;
    let ok = submit_settle(&test_cluster, payee, channel_id, 3_000, v_stale).await;
    assert!(!ok, "stale voucher must be rejected");

    let ch = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(ch.deposit, 95_000);
    assert_eq!(ch.settled_amount, 5_000);
}

/// All validators must agree on Channel state. Strong invariant:
/// validators at the same Channel version must hold byte-identical
/// channel data.
#[cfg(msim)]
#[msim::sim_test]
async fn channel_state_agrees_across_validators() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    let channel_id = open_channel(&test_cluster, payer, payee, 100_000).await;
    let voucher_sig = sign_voucher(&test_cluster, payer, channel_id, 30_000).await;
    assert!(submit_settle(&test_cluster, payee, channel_id, 30_000, voucher_sig).await);

    // Let state-sync settle so every validator catches up.
    sleep(Duration::from_secs(1)).await;

    // Snapshot Channel from every validator.
    let snapshots: Vec<(Channel, Version)> = test_cluster
        .swarm
        .validator_node_handles()
        .into_iter()
        .map(|h| {
            h.with(|node| {
                let obj = node
                    .state()
                    .get_object_store()
                    .get_object(&channel_id)
                    .expect("Channel present on validator");
                (obj.as_channel().unwrap(), obj.version())
            })
        })
        .collect();
    assert_eq!(snapshots.len(), 4);

    // For any two validators at the same Channel version, the data
    // must be identical (BCS-byte-equal).
    let mut by_version: std::collections::BTreeMap<Version, Channel> =
        std::collections::BTreeMap::new();
    for (ch, v) in &snapshots {
        if let Some(prev) = by_version.insert(*v, ch.clone()) {
            assert_eq!(
                prev, *ch,
                "validators at Channel version {:?} disagree on data",
                v
            );
        }
    }
    info!(snapshot_count = snapshots.len(), "all validators agree on channel state per version");
}

/// Two independent channels can be opened and settled in the same
/// cluster without interfering with each other (per-channel scheduling
/// works correctly).
#[cfg(msim)]
#[msim::sim_test]
async fn channels_independent_no_cross_interference() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer_a = addrs[0];
    let payee_a = addrs[1];
    let payer_b = addrs[2];
    let payee_b = addrs[3];

    let chan_a = open_channel(&test_cluster, payer_a, payee_a, 50_000).await;
    let chan_b = open_channel(&test_cluster, payer_b, payee_b, 80_000).await;
    assert_ne!(chan_a, chan_b);

    // Settle on each.
    let v_a = sign_voucher(&test_cluster, payer_a, chan_a, 10_000).await;
    assert!(submit_settle(&test_cluster, payee_a, chan_a, 10_000, v_a).await);

    let v_b = sign_voucher(&test_cluster, payer_b, chan_b, 30_000).await;
    assert!(submit_settle(&test_cluster, payee_b, chan_b, 30_000, v_b).await);

    let ch_a = read_channel(&test_cluster, chan_a).unwrap();
    let ch_b = read_channel(&test_cluster, chan_b).unwrap();
    assert_eq!(ch_a.deposit, 40_000);
    assert_eq!(ch_a.settled_amount, 10_000);
    assert_eq!(ch_b.deposit, 50_000);
    assert_eq!(ch_b.settled_amount, 30_000);

    // Cross-channel voucher must not verify: a voucher signed for
    // chan_a presented as a Settle on chan_b is rejected. The
    // signature includes channel_id, so the executor catches this.
    let cross_voucher = sign_voucher(&test_cluster, payer_a, chan_a, 5_000).await;
    let ok = submit_settle(&test_cluster, payee_a, chan_b, 5_000, cross_voucher).await;
    assert!(!ok, "voucher signed for chan_a must not validate on chan_b");
}

/// Tampering with the cumulative amount in a forwarded voucher must
/// invalidate the signature and the executor must reject the Settle.
#[cfg(msim)]
#[msim::sim_test]
async fn channel_settle_rejects_invalid_signature() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    let channel_id = open_channel(&test_cluster, payer, payee, 100_000).await;

    // Sign a voucher for 1_000 but submit Settle claiming 9_999. The
    // signature won't match the IntentMessage<Voucher{channel_id, 9_999}>.
    let real_sig = sign_voucher(&test_cluster, payer, channel_id, 1_000).await;
    let ok = submit_settle(&test_cluster, payee, channel_id, 9_999, real_sig).await;
    assert!(!ok, "Settle with mismatched cumulative_amount must be rejected");

    let ch = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(ch.deposit, 100_000, "no payment made on rejected Settle");
    assert_eq!(ch.settled_amount, 0);
}
