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
    OpenChannelArgs, RegisterOfferingArgs, RequestCloseArgs, SettleArgs, TopUpArgs,
    TransactionData, TransactionKind, WithdrawAfterTimeoutArgs,
};
use utils::logging::init_tracing;

// Stage 13c: `one_coin` removed — channel txs are balance-mode, so
// they need no per-tx gas coin. Helpers use `stateless_tx_data` to
// build a tx with empty `gas_payment` and a `ValidDuring` window.

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
    let voucher = Voucher::new_amount_only(channel_id, cumulative_amount);
    let sig: Signature = test_cluster
        .wallet
        .config
        .keystore
        .sign_secure::<Voucher>(&signer, &voucher, Intent::soma_app(IntentScope::PaymentVoucher))
        .await
        .expect("voucher signing succeeds");
    sig.into()
}

/// Register a baseline offering for `payee` against the test model. Idempotent: re-registering
/// after the first call leaves the existing on-chain row alone (the executor rejects with
/// `OfferingAlreadyExists`, which we treat as success since it means the precondition for
/// OpenChannel is already met).
async fn register_default_offering(test_cluster: &TestCluster, payee: SomaAddress) {
    use types::offering::Offering;
    let offering_id = Offering::derive_id(payee, "anthropic/claude-sonnet-4.6");
    if test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_object_store().get_object(&offering_id).is_some())
    {
        return;
    }
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payee,
        TransactionKind::RegisterOffering(RegisterOfferingArgs {
            model_id: "anthropic/claude-sonnet-4.6".to_string(),
            prompt_micros_per_1k: 3_000,
            completion_micros_per_1k: 15_000,
            cache_read_micros_per_1k: 300,
            cache_write_micros_per_1k: 3_000,
            request_micros: 0,
            ttft_bound_ms: 1_500,
            ttot_bound_ms: 50,
        }),
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(
        response.effects.status().is_ok(),
        "RegisterOffering must succeed: status={:?}",
        response.effects.status()
    );
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
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payer,
        TransactionKind::OpenChannel(OpenChannelArgs {
            payee,
            authorized_signer: payer,
            token: CoinType::Usdc,
            deposit_amount,
            model_id: "anthropic/claude-sonnet-4.6".to_string(),
        }),
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(
        response.effects.status().is_ok(),
        "OpenChannel must succeed: status={:?}",
        response.effects.status()
    );

    // Find the Channel object among the tx's created shared objects.
    // OpenChannel may also lazily-create a `ProviderInbox` (the first time a
    // payer opens against this payee), so we can't blindly pick the first
    // shared object — match on the on-chain `ObjectType` instead.
    let created = response.effects.created();
    let channel_oref = created
        .iter()
        .find(|((id, _, _), owner)| {
            owner.is_shared()
                && test_cluster.fullnode_handle.soma_node.with(|node| {
                    node.state()
                        .get_object_store()
                        .get_object(id)
                        .map(|o| *o.type_() == ObjectType::Channel)
                        .unwrap_or(false)
                })
        })
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
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payee,
        TransactionKind::Settle(SettleArgs {
            channel_id,
            cumulative_amount,
            cumulative_prompt_tokens: 0,
            cumulative_completion_tokens: 0,
            cumulative_cache_read_tokens: 0,
            cumulative_cache_write_tokens: 0,
            cumulative_requests: 0,
            voucher_signature,
        }),
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
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payer,
        TransactionKind::RequestClose(RequestCloseArgs { channel_id }),
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
    // Look up the channel's payee — required to declare the
    // per-payee `ProviderInbox` shared input on the tx.
    let payee = read_channel(test_cluster, channel_id).expect("channel exists").payee();
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payer,
        TransactionKind::WithdrawAfterTimeout(WithdrawAfterTimeoutArgs { channel_id, payee }),
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

    // Provider-side setup before reading balances so the offering tx's gas
    // doesn't pollute the per-test invariant (payee delta == settled - settle-gas).
    register_default_offering(&test_cluster, payee).await;

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
    assert_eq!(ch.payer(), payer);
    assert_eq!(ch.payee(), payee);
    assert_eq!(ch.deposit(), 100_000);
    assert_eq!(ch.settled_amount(), 0);
    assert!(ch.close_requested_at_ms().is_none());

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
    assert_eq!(ch.deposit(), 90_000);
    assert_eq!(ch.settled_amount(), 10_000);

    // 3. Settle at cumulative=25_000 (delta = 15_000).
    let voucher_sig = sign_voucher(&test_cluster, payer, channel_id, 25_000).await;
    assert!(
        submit_settle(&test_cluster, payee, channel_id, 25_000, voucher_sig).await,
        "second Settle must succeed"
    );
    let ch = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(ch.deposit(), 75_000);
    assert_eq!(ch.settled_amount(), 25_000);

    // After two Settles: payee USDC rose by 25_000 (the cumulative
    // settled amount) minus gas fees. Stage 13c: gas is balance-
    // mode, so each Settle debits the payee's USDC accumulator for
    // its fee in addition to the +Deposit delta. Settle's
    // fee_units = 1 (catch-all branch in TransactionKind::fee_units),
    // so total gas across two Settles is 2 × unit_fee.
    let unit_fee = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().fee_parameters().unit_fee
    });
    let expected_gas = 2 * unit_fee;
    let payee_after_settles = read_usdc(payee);
    assert_eq!(
        payee_after_settles - payee_initial,
        25_000 - expected_gas,
        "payee accumulator delta must equal settled total ({}) minus exact gas ({}) over 2 Settles",
        25_000,
        expected_gas,
    );

    // 4. Request close — Clock timestamp gets stamped onto channel.
    let pre_close_ts = read_clock_ts(&test_cluster);
    assert!(
        submit_request_close(&test_cluster, payer, channel_id).await,
        "RequestClose must succeed"
    );
    let ch = read_channel(&test_cluster, channel_id).unwrap();
    let close_at = ch.close_requested_at_ms().expect("close_requested_at_ms set");
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

    register_default_offering(&test_cluster, payee).await;
    let channel_id = open_channel(&test_cluster, payer, payee, 50_000).await;
    let voucher_sig = sign_voucher(&test_cluster, payer, channel_id, 1_000).await;

    // Submit Settle from `payer` instead of `payee`.
    let tx = e2e_tests::stateless_tx_data(
        &test_cluster,
        payer,
        TransactionKind::Settle(SettleArgs {
            channel_id,
            cumulative_amount: 1_000,
            cumulative_prompt_tokens: 0,
            cumulative_completion_tokens: 0,
            cumulative_cache_read_tokens: 0,
            cumulative_cache_write_tokens: 0,
            cumulative_requests: 0,
            voucher_signature: voucher_sig,
        }),
    );
    let response = test_cluster.sign_and_execute_transaction(&tx).await;
    assert!(
        !response.effects.status().is_ok(),
        "Settle from non-payee must fail at executor; got status={:?}",
        response.effects.status()
    );

    // Channel state is unchanged.
    let ch = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(ch.deposit(), 50_000);
    assert_eq!(ch.settled_amount(), 0);
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

    register_default_offering(&test_cluster, payee).await;
    let channel_id = open_channel(&test_cluster, payer, payee, 100_000).await;

    // First settle at cumulative=5_000.
    let v1 = sign_voucher(&test_cluster, payer, channel_id, 5_000).await;
    assert!(submit_settle(&test_cluster, payee, channel_id, 5_000, v1).await);

    // Stale voucher at cumulative=3_000 (less than settled_amount=5_000).
    let v_stale = sign_voucher(&test_cluster, payer, channel_id, 3_000).await;
    let ok = submit_settle(&test_cluster, payee, channel_id, 3_000, v_stale).await;
    assert!(!ok, "stale voucher must be rejected");

    let ch = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(ch.deposit(), 95_000);
    assert_eq!(ch.settled_amount(), 5_000);
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

    register_default_offering(&test_cluster, payee).await;
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
            assert_eq!(prev, *ch, "validators at Channel version {:?} disagree on data", v);
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

    register_default_offering(&test_cluster, payee_a).await;
    register_default_offering(&test_cluster, payee_b).await;
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
    assert_eq!(ch_a.deposit(), 40_000);
    assert_eq!(ch_a.settled_amount(), 10_000);
    assert_eq!(ch_b.deposit(), 50_000);
    assert_eq!(ch_b.settled_amount(), 30_000);

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

    register_default_offering(&test_cluster, payee).await;
    let channel_id = open_channel(&test_cluster, payer, payee, 100_000).await;

    // Sign a voucher for 1_000 but submit Settle claiming 9_999. The
    // signature won't match the IntentMessage<Voucher{channel_id, 9_999}>.
    let real_sig = sign_voucher(&test_cluster, payer, channel_id, 1_000).await;
    let ok = submit_settle(&test_cluster, payee, channel_id, 9_999, real_sig).await;
    assert!(!ok, "Settle with mismatched cumulative_amount must be rejected");

    let ch = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(ch.deposit(), 100_000, "no payment made on rejected Settle");
    assert_eq!(ch.settled_amount(), 0);
}

/// Reproduction target for the testnet fullnode fork at tx
/// `GixFsWyC1i8SdJwCqQbcfi7dKzLBsvUqb2dH7YFoTDz` (2026-05-25). The
/// payer there (`0xcf0b…387c8`) ran a relay that submitted **bursts
/// of 4 simultaneous TopUps** — one per channel — every minute, all
/// landing in the same checkpoint. Each TopUp emits an
/// `AccumulatorWriteV1` against the payer's `(USDC,)` accumulator;
/// the cp builder aggregates the four into one Settlement
/// `Withdraw`. Hypothesis (a) from the post-mortem: if anything in
/// that aggregation or in `apply_settlement_to_object_inputs` is
/// non-deterministic when N>1 events share an accumulator id, the
/// fullnode's local execution diverges from the validator quorum
/// and `authority.rs:1129`'s `fork detected!` fires.
///
/// This test exercises that exact workload locally. It must:
///   * have all 4 TopUps succeed,
///   * not trigger `fork detected!` on the fullnode,
///   * leave the payer's USDC accumulator debited by **exactly**
///     `4 * topup_amount + 4 * unit_fee`.
///
/// Any divergence here pins down hypothesis (a). A clean pass leaves
/// the prod drift attributable to (b) — historical crashed-mid-write
/// state divergence — or (c) — an undetected fork in a callsite the
/// detector isn't installed on.
#[cfg(msim)]
#[msim::sim_test]
async fn channel_concurrent_topups_aggregate_correctly() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    register_default_offering(&test_cluster, payee).await;

    let read_usdc = |addr: SomaAddress| -> u64 {
        test_cluster
            .fullnode_handle
            .soma_node
            .with(|node| node.state().database_for_testing().get_balance(addr, CoinType::Usdc))
            .unwrap_or(0)
    };

    // Four channels — same payer, same payee, same coin — so each
    // TopUp targets the SAME `(payer, USDC)` BalanceAccumulator. This
    // is the only configuration that exercises N>1 events on one
    // accumulator id in a single checkpoint.
    let mut channel_ids = Vec::with_capacity(4);
    for _ in 0..4 {
        channel_ids.push(open_channel(&test_cluster, payer, payee, 1_000_000).await);
    }

    let pre_topup = read_usdc(payer);

    let topup_amount = 100_000u64;
    // Build + sign all four txs up front so the concurrent submission
    // window contains only `execute_transaction_may_fail` calls — the
    // tighter that window, the better the chance all four land in the
    // same checkpoint.
    let mut signed_txs = Vec::with_capacity(4);
    for cid in channel_ids.iter().copied() {
        let tx = e2e_tests::stateless_tx_data(
            &test_cluster,
            payer,
            TransactionKind::TopUp(TopUpArgs {
                channel_id: cid,
                coin_type: CoinType::Usdc,
                amount: topup_amount,
            }),
        );
        signed_txs.push(test_cluster.wallet.sign_transaction(&tx).await);
    }
    let wallet = &test_cluster.wallet;
    let topup_futures = signed_txs.into_iter().map(|signed| async move {
        wallet
            .execute_transaction_may_fail(signed)
            .await
            .map(|r| r.effects.status().is_ok())
            .unwrap_or(false)
    });
    let results: Vec<bool> = futures::future::join_all(topup_futures).await;
    assert!(
        results.iter().all(|ok| *ok),
        "all 4 concurrent TopUps must succeed; got: {:?}",
        results
    );

    // The fullnode would have panicked already if the settlement
    // forked, so reaching here without a panic is itself the first
    // assertion. Run one more commit so the post-state we read below
    // includes the settlement effects (settlement applies inside the
    // commit that includes the TopUps, but state-sync to the fullnode
    // catches up a beat later).
    drive_one_commit(&test_cluster).await;

    let post_topup = read_usdc(payer);
    let unit_fee = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().fee_parameters().unit_fee
    });

    // Each TopUp debits the payer's USDC by `amount + 1*unit_fee`
    // (TopUp's fee_units = 1). The `drive_one_commit` after the
    // burst is a BalanceTransfer of 1 µUSDC to a single recipient,
    // whose fee_units = 1 + n_recipients = 2 (see
    // `TransactionKind::fee_units` in types/src/transaction.rs). So
    // the expected debit on the payer is:
    //   4 * (topup_amount + unit_fee)   — the 4 TopUps
    //   +  1                            — the BalanceTransfer principal
    //   +  2 * unit_fee                 — the BalanceTransfer gas
    let expected_debit = 4 * (topup_amount + unit_fee) + 1 + 2 * unit_fee;
    let actual_debit = pre_topup - post_topup;
    info!(
        "concurrent_topups: pre_usdc={} post_usdc={} actual_debit={} expected_debit={} \
         (4 * ({} + {}) + 1 + 2 * {})",
        pre_topup, post_topup, actual_debit, expected_debit, topup_amount, unit_fee, unit_fee,
    );
    // Drift would manifest as a non-equal debit (off by the residue
    // amount that the cp builder mis-aggregated).
    assert_eq!(
        actual_debit, expected_debit,
        "payer's USDC accumulator debit must equal 4*(topup+gas) + drive-commit gas exactly — \
         any inequality is the aggregation/settlement drift bug"
    );

    // Per-channel sanity: each channel got exactly one top-up.
    for cid in &channel_ids {
        let ch = read_channel(&test_cluster, *cid).expect("channel exists post-topup");
        assert_eq!(
            ch.deposit(),
            1_000_000 + topup_amount,
            "channel {:?} deposit must equal initial + one topup",
            cid,
        );
    }
}
