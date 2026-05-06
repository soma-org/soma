// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! E2E test: full proxy↔provider payment-channel flow.
//!
//! Simulates an inference marketplace session where:
//!
//!   payer ("proxy")  ←→  payee ("provider")
//!
//! drives traffic through a long-running on-chain payment channel.
//! Each "request" bumps the voucher's cumulative_amount; the provider
//! periodically redeems via on-chain `Settle`; the payer tops up
//! when escrow runs low; eventually the payer closes via
//! `RequestClose` → `WithdrawAfterTimeout`.
//!
//! Why no real HTTP: the inference crate's `RunningTab` voucher
//! format embeds HTTP-level binding (method/path/body_sha) for
//! per-request anti-replay, which is incompatible with the on-chain
//! `IntentMessage<Voucher>` format. Reconciling those is its own
//! protocol-design task. This test exercises the **on-chain channel
//! mechanics** end-to-end with the same shape an inference proxy
//! would drive — voucher signing via `sdk::channel::sign_voucher`,
//! provider verification via `sdk::channel::verify_voucher`, on-chain
//! settlement, top-ups, close.
//!
//! Edge cases covered:
//!   - happy path multi-Settle + multi-TopUp + close
//!   - stale voucher rejected
//!   - non-payee Settle rejected
//!   - overspend rejected
//!   - bad signature rejected
//!   - non-payer TopUp rejected
//!   - wrong-coin-type TopUp rejected
//!   - zero-amount TopUp rejected
//!   - TopUp clears pending RequestClose
//!   - non-payer RequestClose rejected
//!   - WithdrawAfterTimeout before grace rejected
//!   - WithdrawAfterTimeout without RequestClose rejected
//!   - typed error variants surface correctly
//!   - balance accounting: payer net-debit = total-settled (modulo gas)
//!     payee net-credit = total-settled (modulo gas)

#![cfg(msim)]

use std::time::Duration;

use sdk::channel as sdk_channel;
use soma_keys::keystore::AccountKeystore as _;
use test_cluster::{TestCluster, TestClusterBuilder};
use tokio::time::sleep;
use tracing::info;
use types::CLOCK_OBJECT_ID;
use types::base::SomaAddress;
use types::channel::Channel;
use types::crypto::GenericSignature;
use types::effects::{ExecutionFailureStatus, ExecutionStatus, TransactionEffectsAPI as _};
use types::object::{CoinType, Object, ObjectID};
use types::transaction::{
    OpenChannelArgs, RequestCloseArgs, SettleArgs, TopUpArgs, TransactionKind,
    WithdrawAfterTimeoutArgs,
};
use utils::logging::init_tracing;

// ---------------------------------------------------------------------
// Test fixtures + helpers
// ---------------------------------------------------------------------

/// One in-memory record of every voucher the "proxy" has issued.
/// Mirrors what an inference proxy would persist between requests.
struct RunningSession {
    channel_id: ObjectID,
    payer: SomaAddress,
    payee: SomaAddress,
    cumulative: u64,
    /// Latest voucher signature — what the provider would settle if
    /// it redeemed right now.
    latest_sig: Option<GenericSignature>,
    /// Total realized cost ever attributed by the proxy. For
    /// post-test sanity checks.
    total_authorized: u64,
}

impl RunningSession {
    fn new(channel_id: ObjectID, payer: SomaAddress, payee: SomaAddress) -> Self {
        Self {
            channel_id,
            payer,
            payee,
            cumulative: 0,
            latest_sig: None,
            total_authorized: 0,
        }
    }

    /// Authorize an additional `amount` of spend by signing a fresh
    /// voucher whose cumulative = previous + amount. Mirrors the
    /// per-request bump an inference proxy does.
    async fn authorize_more(
        &mut self,
        test_cluster: &TestCluster,
        amount: u64,
    ) -> GenericSignature {
        let new_cum = sdk_channel::next_cumulative(self.cumulative, amount)
            .expect("voucher bump must not overflow");
        let sig = sdk_channel::sign_voucher(
            &test_cluster.wallet.config.keystore,
            &self.payer,
            self.channel_id,
            new_cum,
        )
        .await
        .expect("voucher signing succeeds");
        self.cumulative = new_cum;
        self.total_authorized = self.total_authorized.saturating_add(amount);
        self.latest_sig = Some(sig.clone());
        sig
    }
}

/// Read a Channel from the fullnode — None if the channel has been
/// deleted (closed).
fn read_channel(test_cluster: &TestCluster, channel_id: ObjectID) -> Option<Channel> {
    test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_object_store()
            .get_object(&channel_id)
            .as_ref()
            .and_then(Object::as_channel)
    })
}

fn read_clock_ts(test_cluster: &TestCluster) -> u64 {
    test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_object_store()
            .get_object(&CLOCK_OBJECT_ID)
            .expect("clock present")
            .clock_timestamp_ms()
    })
}

fn read_usdc(test_cluster: &TestCluster, addr: SomaAddress) -> u64 {
    test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().database_for_testing().get_balance(addr, CoinType::Usdc))
        .unwrap_or(0)
}

/// Submit `OpenChannel` and return the new channel's id (predicted
/// client-side via `sdk::channel::predicted_channel_id` — matches the
/// chain's derive_id).
async fn open_channel(
    test_cluster: &TestCluster,
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
    assert!(
        response.effects.status().is_ok(),
        "OpenChannel must succeed: status={:?}",
        response.effects.status()
    );

    // Predict the channel id from the open-tx digest. Must match the
    // executor's derive_id(tx_digest, 0).
    let predicted = sdk_channel::predicted_channel_id(*response.effects.transaction_digest());

    let created = response.effects.created();
    let channel_oref = created
        .iter()
        .find(|(_oref, owner)| owner.is_shared())
        .expect("OpenChannel creates a shared Channel object");
    assert_eq!(
        channel_oref.0.0, predicted,
        "predicted channel id must match the on-chain derive_id"
    );
    predicted
}

async fn submit_settle(
    test_cluster: &TestCluster,
    payee: SomaAddress,
    channel_id: ObjectID,
    cumulative_amount: u64,
    voucher_signature: GenericSignature,
) -> Result<ExecutionStatus, anyhow::Error> {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payee,
        TransactionKind::Settle(SettleArgs {
            channel_id,
            cumulative_amount,
            voucher_signature,
        }),
    );
    let signed = test_cluster.wallet.sign_transaction(&tx_data).await;
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(signed)
        .await
        .map_err(|e| anyhow::anyhow!("settle exec error: {e}"))?;
    Ok(r.effects.status().clone())
}

async fn submit_top_up(
    test_cluster: &TestCluster,
    payer: SomaAddress,
    channel_id: ObjectID,
    coin_type: CoinType,
    amount: u64,
) -> Result<ExecutionStatus, anyhow::Error> {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payer,
        TransactionKind::TopUp(TopUpArgs { channel_id, coin_type, amount }),
    );
    let signed = test_cluster.wallet.sign_transaction(&tx_data).await;
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(signed)
        .await
        .map_err(|e| anyhow::anyhow!("top_up exec error: {e}"))?;
    Ok(r.effects.status().clone())
}

async fn submit_request_close(
    test_cluster: &TestCluster,
    payer: SomaAddress,
    channel_id: ObjectID,
) -> Result<ExecutionStatus, anyhow::Error> {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payer,
        TransactionKind::RequestClose(RequestCloseArgs { channel_id }),
    );
    let signed = test_cluster.wallet.sign_transaction(&tx_data).await;
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(signed)
        .await
        .map_err(|e| anyhow::anyhow!("request_close exec error: {e}"))?;
    Ok(r.effects.status().clone())
}

async fn submit_withdraw(
    test_cluster: &TestCluster,
    payer: SomaAddress,
    channel_id: ObjectID,
) -> Result<ExecutionStatus, anyhow::Error> {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payer,
        TransactionKind::WithdrawAfterTimeout(WithdrawAfterTimeoutArgs { channel_id }),
    );
    let signed = test_cluster.wallet.sign_transaction(&tx_data).await;
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(signed)
        .await
        .map_err(|e| anyhow::anyhow!("withdraw exec error: {e}"))?;
    Ok(r.effects.status().clone())
}

/// Drive consensus by submitting a tiny self-transfer. Used to push
/// the Clock past the grace boundary for timeout tests.
async fn drive_one_commit(test_cluster: &TestCluster) {
    let addrs = test_cluster.wallet.get_addresses();
    let tx = e2e_tests::balance_transfer_data(
        test_cluster,
        CoinType::Usdc,
        addrs[0],
        vec![(addrs[1], 1)],
    );
    let _ = test_cluster.sign_and_execute_transaction(&tx).await;
}

fn assert_failure_kind(status: &ExecutionStatus, expected: &str) {
    match status {
        ExecutionStatus::Success => panic!("expected failure ({expected}), got success"),
        ExecutionStatus::Failure { error } => {
            let dbg = format!("{:?}", error);
            assert!(
                dbg.contains(expected),
                "expected error to contain {expected:?}, got {dbg}"
            );
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

/// Full long-running session: open → many settles → top up → many
/// more settles → close → withdraw. Verifies funds flow, channel
/// state at each transition, and the overall conservation invariant.
#[msim::sim_test]
async fn proxy_provider_long_running_session() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0]; // "proxy"
    let payee = addrs[1]; // "provider"

    let payer_initial = read_usdc(&test_cluster, payer);
    let payee_initial = read_usdc(&test_cluster, payee);

    // 1. Open the channel with a small starter deposit.
    let initial_deposit = 50_000;
    let channel_id = open_channel(&test_cluster, payer, payee, initial_deposit).await;
    info!(?channel_id, "channel opened");
    let ch = read_channel(&test_cluster, channel_id).expect("channel exists post-open");
    assert_eq!(ch.deposit, initial_deposit);
    assert_eq!(ch.settled_amount, 0);
    assert!(ch.close_requested_at_ms.is_none());

    // 2. Run 5 simulated requests; provider settles after each.
    let request_costs: [u64; 5] = [3_000, 7_500, 4_200, 12_000, 8_300];
    let mut session = RunningSession::new(channel_id, payer, payee);
    let mut total_settled = 0u64;
    for cost in request_costs {
        let sig = session.authorize_more(&test_cluster, cost).await;

        // Provider verifies before redeeming (cheap pre-check).
        let ch = read_channel(&test_cluster, channel_id).expect("channel exists");
        let voucher = types::channel::Voucher::new(channel_id, session.cumulative);
        sdk_channel::verify_voucher(&ch, voucher, &sig).expect("voucher verifies in-process");

        let status = submit_settle(&test_cluster, payee, channel_id, session.cumulative, sig)
            .await
            .expect("settle tx executes");
        assert!(status.is_ok(), "Settle must succeed: {status:?}");
        total_settled = session.cumulative;

        let updated = read_channel(&test_cluster, channel_id).unwrap();
        assert_eq!(updated.settled_amount, total_settled);
        assert_eq!(updated.deposit, initial_deposit - total_settled);
    }
    assert_eq!(total_settled, request_costs.iter().sum::<u64>());
    info!(total_settled, "after first round of settles");

    // 3. Top up — channel deposit grows by the amount, channel keeps
    //    running. (Run 3 more requests after to prove it works.)
    let topup_amount = 100_000;
    let status = submit_top_up(
        &test_cluster, payer, channel_id, CoinType::Usdc, topup_amount,
    )
    .await
    .expect("topup tx executes");
    assert!(status.is_ok(), "TopUp must succeed: {status:?}");
    let after_topup = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(
        after_topup.deposit,
        initial_deposit - total_settled + topup_amount,
        "deposit must grow by the top-up amount",
    );

    let more_costs: [u64; 3] = [5_000, 9_500, 11_200];
    for cost in more_costs {
        let sig = session.authorize_more(&test_cluster, cost).await;
        let status = submit_settle(&test_cluster, payee, channel_id, session.cumulative, sig)
            .await
            .expect("settle tx executes");
        assert!(status.is_ok(), "post-topup Settle must succeed: {status:?}");
    }
    total_settled = session.cumulative;
    let final_total: u64 = request_costs.iter().sum::<u64>() + more_costs.iter().sum::<u64>();
    assert_eq!(total_settled, final_total);

    // 4. Close path: payer requests close, waits grace, withdraws.
    let pre_close_ts = read_clock_ts(&test_cluster);
    let status = submit_request_close(&test_cluster, payer, channel_id).await.unwrap();
    assert!(status.is_ok(), "RequestClose must succeed: {status:?}");
    let ch = read_channel(&test_cluster, channel_id).unwrap();
    let close_at = ch.close_requested_at_ms.expect("close_requested_at_ms set");
    assert!(close_at >= pre_close_ts);

    // Withdraw too early — must fail with ChannelGraceNotElapsed.
    let bad_status = submit_withdraw(&test_cluster, payer, channel_id).await.unwrap();
    assert_failure_kind(&bad_status, "ChannelGraceNotElapsed");

    // Wait past grace + drive a few commits so Clock advances.
    sleep(Duration::from_secs(10)).await;
    for _ in 0..3 {
        drive_one_commit(&test_cluster).await;
    }

    let status = submit_withdraw(&test_cluster, payer, channel_id).await.unwrap();
    assert!(status.is_ok(), "Withdraw must succeed after grace: {status:?}");
    assert!(read_channel(&test_cluster, channel_id).is_none(), "channel deleted on withdraw");

    // 5. Funds-flow invariants. The proxy's net debit equals the
    //    total amount the provider settled, plus gas spent on
    //    OpenChannel + TopUp + RequestClose + WithdrawAfterTimeout.
    //    The provider's net credit is total_settled minus its own gas
    //    on each Settle.
    let payer_final = read_usdc(&test_cluster, payer);
    let payee_final = read_usdc(&test_cluster, payee);
    let payer_net_debit = payer_initial - payer_final;
    let payee_net_credit = payee_final - payee_initial;

    // Sanity bounds: payer settled `total_settled` µUSDC plus some gas.
    assert!(
        payer_net_debit >= total_settled,
        "payer net debit {} must cover total_settled {}",
        payer_net_debit,
        total_settled,
    );
    // Payer paid at most: total_settled + (4 fixed-fee txs). With
    // unit_fee around 1000µUSDC and op fee_units = 1, that's a few
    // thousand µUSDC of gas tops. Verify we're not overshooting by
    // an order of magnitude.
    assert!(
        payer_net_debit <= total_settled + 100_000,
        "payer net debit {} should be close to total_settled {}",
        payer_net_debit,
        total_settled,
    );
    // Payee gained `total_settled` minus 8 Settle gas fees.
    assert!(
        payee_net_credit > 0,
        "payee net credit {} must be positive",
        payee_net_credit,
    );
    assert!(
        payee_net_credit <= total_settled,
        "payee net credit {} cannot exceed total_settled {}",
        payee_net_credit,
        total_settled,
    );

    info!(
        total_settled,
        payer_net_debit,
        payee_net_credit,
        "long-running session closed cleanly",
    );
}

/// Edge-case sweep: every channel error path the executor enforces.
/// One test, one channel, one cluster — exercises each typed
/// `ExecutionFailureStatus::Channel*` variant in turn so a future
/// regression in the error-categorization layer surfaces here.
#[msim::sim_test]
async fn channel_typed_error_sweep() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];
    let third_party = addrs[2];

    let channel_id = open_channel(&test_cluster, payer, payee, 100_000).await;
    let mut session = RunningSession::new(channel_id, payer, payee);

    // 1. Settle by non-payee → ChannelCallerNotPayee.
    let sig = session.authorize_more(&test_cluster, 1_000).await;
    {
        let tx_data = e2e_tests::stateless_tx_data(
            &test_cluster,
            payer,
            TransactionKind::Settle(SettleArgs {
                channel_id,
                cumulative_amount: session.cumulative,
                voucher_signature: sig.clone(),
            }),
        );
        let r = test_cluster.sign_and_execute_transaction(&tx_data).await;
        assert_failure_kind(&r.effects.status().clone(), "ChannelCallerNotPayee");
    }

    // Land the legitimate Settle so subsequent tests have a non-zero
    // settled_amount to attack with.
    let status = submit_settle(&test_cluster, payee, channel_id, session.cumulative, sig.clone())
        .await
        .unwrap();
    assert!(status.is_ok());

    // 2. Stale voucher (cumulative <= settled) → ChannelVoucherNotMonotonic.
    {
        // Build a stale voucher manually (cumulative < settled_amount).
        let stale_sig = sdk_channel::sign_voucher(
            &test_cluster.wallet.config.keystore,
            &payer,
            channel_id,
            500, // less than the 1_000 we just settled
        )
        .await
        .unwrap();
        let status = submit_settle(&test_cluster, payee, channel_id, 500, stale_sig)
            .await
            .unwrap();
        assert_failure_kind(&status, "ChannelVoucherNotMonotonic");
    }

    // 3. Overspend → ChannelOverspend (cumulative > deposit + settled).
    {
        let huge = 999_999_999;
        let huge_sig = sdk_channel::sign_voucher(
            &test_cluster.wallet.config.keystore,
            &payer,
            channel_id,
            huge,
        )
        .await
        .unwrap();
        let status = submit_settle(&test_cluster, payee, channel_id, huge, huge_sig)
            .await
            .unwrap();
        assert_failure_kind(&status, "ChannelOverspend");
    }

    // 4. Bad signature (signed by a different key) →
    //    ChannelInvalidVoucherSignature.
    {
        // Sign with the third party's key — wrong signer.
        let bad_sig = sdk_channel::sign_voucher(
            &test_cluster.wallet.config.keystore,
            &third_party,
            channel_id,
            session.cumulative + 1_000,
        )
        .await
        .unwrap();
        let status = submit_settle(
            &test_cluster,
            payee,
            channel_id,
            session.cumulative + 1_000,
            bad_sig,
        )
        .await
        .unwrap();
        assert_failure_kind(&status, "ChannelInvalidVoucherSignature");
    }

    // 5. TopUp by non-payer → ChannelCallerNotPayer.
    {
        let status = submit_top_up(
            &test_cluster, payee, channel_id, CoinType::Usdc, 1_000,
        )
        .await
        .unwrap();
        assert_failure_kind(&status, "ChannelCallerNotPayer");
    }

    // 6. TopUp with wrong coin_type → ChannelCoinTypeMismatch.
    {
        let status = submit_top_up(
            &test_cluster, payer, channel_id, CoinType::Soma, 1_000,
        )
        .await
        .unwrap();
        assert_failure_kind(&status, "ChannelCoinTypeMismatch");
    }

    // 7. TopUp with zero amount → ChannelAmountZero.
    {
        let status = submit_top_up(
            &test_cluster, payer, channel_id, CoinType::Usdc, 0,
        )
        .await
        .unwrap();
        assert_failure_kind(&status, "ChannelAmountZero");
    }

    // 8. RequestClose by non-payer → ChannelCallerNotPayer.
    {
        let tx_data = e2e_tests::stateless_tx_data(
            &test_cluster,
            payee,
            TransactionKind::RequestClose(RequestCloseArgs { channel_id }),
        );
        let r = test_cluster.sign_and_execute_transaction(&tx_data).await;
        assert_failure_kind(&r.effects.status().clone(), "ChannelCallerNotPayer");
    }

    // 9. WithdrawAfterTimeout without prior RequestClose →
    //    ChannelNoCloseRequest.
    {
        let status = submit_withdraw(&test_cluster, payer, channel_id).await.unwrap();
        assert_failure_kind(&status, "ChannelNoCloseRequest");
    }

    // 10. RequestClose now succeeds; second RequestClose →
    //     ChannelCloseAlreadyPending.
    {
        let status = submit_request_close(&test_cluster, payer, channel_id).await.unwrap();
        assert!(status.is_ok());
        let again = submit_request_close(&test_cluster, payer, channel_id).await.unwrap();
        assert_failure_kind(&again, "ChannelCloseAlreadyPending");
    }

    // 11. WithdrawAfterTimeout before grace → ChannelGraceNotElapsed.
    {
        let status = submit_withdraw(&test_cluster, payer, channel_id).await.unwrap();
        assert_failure_kind(&status, "ChannelGraceNotElapsed");
    }

    // 12. TopUp with a pending close pending — must succeed AND
    //     clear the close_requested_at_ms (turning the channel back
    //     into a long-running one).
    {
        let status = submit_top_up(
            &test_cluster, payer, channel_id, CoinType::Usdc, 5_000,
        )
        .await
        .unwrap();
        assert!(status.is_ok(), "TopUp must succeed even with close pending");
        let ch = read_channel(&test_cluster, channel_id).unwrap();
        assert!(
            ch.close_requested_at_ms.is_none(),
            "TopUp must clear pending close_requested_at_ms",
        );
    }

    info!("typed-error sweep covered every Channel* failure variant");
}

/// `predicted_channel_id` agrees with the on-chain id without the
/// fullnode round-trip. The proxy can submit the next tx (Settle /
/// TopUp / etc.) immediately after sending OpenChannel without
/// waiting for indexing.
#[msim::sim_test]
async fn predicted_channel_id_matches_on_chain() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        payer,
        TransactionKind::OpenChannel(OpenChannelArgs {
            payee,
            authorized_signer: payer,
            token: CoinType::Usdc,
            deposit_amount: 10_000,
        }),
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    let predicted = sdk_channel::predicted_channel_id(*response.effects.transaction_digest());
    let actual = response
        .effects
        .created()
        .iter()
        .find(|(_oref, owner)| owner.is_shared())
        .expect("OpenChannel creates a shared channel")
        .0
        .0;
    assert_eq!(predicted, actual);
    info!(?predicted, "predicted == on-chain channel id");
}

/// A single Settle credits the payee's accumulator immediately —
/// no per-session "Close" needed. This is the long-running-channel
/// invariant.
#[msim::sim_test]
async fn settle_credits_payee_immediately_no_close_needed() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    let payee_before = read_usdc(&test_cluster, payee);
    let channel_id = open_channel(&test_cluster, payer, payee, 100_000).await;

    let sig = sdk_channel::sign_voucher(
        &test_cluster.wallet.config.keystore,
        &payer,
        channel_id,
        25_000,
    )
    .await
    .unwrap();
    let status = submit_settle(&test_cluster, payee, channel_id, 25_000, sig)
        .await
        .unwrap();
    assert!(status.is_ok());

    let payee_after = read_usdc(&test_cluster, payee);
    // payee gained 25_000 minus its single Settle gas.
    assert!(
        payee_after > payee_before,
        "payee accumulator must increase on Settle ({}→{})",
        payee_before,
        payee_after,
    );
    let credit = payee_after - payee_before;
    assert!(credit > 20_000, "credit {credit} must be near 25_000 settled");
    assert!(credit <= 25_000, "credit {credit} cannot exceed settled");

    // Channel still alive — no Close needed.
    let ch = read_channel(&test_cluster, channel_id).expect("channel still present after Settle");
    assert_eq!(ch.settled_amount, 25_000);
    assert_eq!(ch.deposit, 100_000 - 25_000);
}

/// Cross-epoch lifecycle. Verifies the three properties that matter
/// when a channel outlives a reconfiguration:
///   1. A voucher signed in epoch N still verifies + settles in epoch N+1
///      — voucher signatures aren't epoch-bound, only the carrying tx is
///      (via `ValidDuring`).
///   2. `Settle` carrying that voucher succeeds after a reconfig.
///   3. `RequestClose` in epoch N + `WithdrawAfterTimeout` in epoch N+1
///      both work — the protocol grace timer is wall-clock-based, not
///      epoch-based.
#[msim::sim_test]
async fn channel_survives_reconfiguration() {
    init_tracing();

    // Short epochs so reconfig doesn't dominate runtime. Test
    // protocol-config sets `channel_grace_period_ms = 5_000` so we
    // wait that long after RequestClose to withdraw.
    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(15_000)
        .build()
        .await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let payee = addrs[1];

    // 1. Open + sign a voucher in epoch 0.
    let deposit = 100_000;
    let channel_id = open_channel(&test_cluster, payer, payee, deposit).await;
    let mut session = RunningSession::new(channel_id, payer, payee);
    let cum1: u64 = 12_345;
    let sig1 = session.authorize_more(&test_cluster, cum1).await;
    info!(cum1, "signed pre-reconfig voucher in epoch 0");

    // 2. Force a reconfiguration to epoch 1.
    let epoch_before = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().epoch_store_for_testing().epoch());
    test_cluster.trigger_reconfiguration().await;
    let epoch_after = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().epoch_store_for_testing().epoch());
    assert_eq!(epoch_after, epoch_before + 1, "reconfig must advance epoch");

    // 3. Pre-reconfig voucher still verifies in-process. (The on-chain
    //    voucher is over `(channel_id, cumulative_amount)` only — no
    //    epoch field — so this is just a sanity check that nothing
    //    changed about Channel.authorized_signer across the reconfig.)
    let ch = read_channel(&test_cluster, channel_id).expect("channel survives reconfig");
    let voucher = types::channel::Voucher::new(channel_id, cum1);
    sdk_channel::verify_voucher(&ch, voucher, &sig1)
        .expect("pre-reconfig voucher must verify post-reconfig");

    // 4. Submit Settle in the new epoch using that pre-reconfig voucher.
    let status = submit_settle(&test_cluster, payee, channel_id, cum1, sig1)
        .await
        .expect("settle tx executes post-reconfig");
    assert!(status.is_ok(), "Settle in new epoch must succeed: {status:?}");
    let updated = read_channel(&test_cluster, channel_id).unwrap();
    assert_eq!(updated.settled_amount, cum1);
    assert_eq!(updated.deposit, deposit - cum1);

    // 5. Sign + settle a NEW voucher entirely within the new epoch
    //    (proves the signing path also still works post-reconfig).
    let cum2 = cum1 + 7_777;
    let sig2 = session.authorize_more(&test_cluster, 7_777).await;
    let status = submit_settle(&test_cluster, payee, channel_id, cum2, sig2)
        .await
        .expect("post-reconfig settle executes");
    assert!(status.is_ok(), "Settle of post-reconfig voucher: {status:?}");
    assert_eq!(
        read_channel(&test_cluster, channel_id).unwrap().settled_amount,
        cum2,
    );

    // 6. RequestClose in epoch N+1, then trigger another reconfiguration
    //    so the timer crosses an epoch boundary, then WithdrawAfterTimeout.
    let status = submit_request_close(&test_cluster, payer, channel_id)
        .await
        .expect("request_close executes");
    assert!(status.is_ok(), "RequestClose: {status:?}");
    assert!(
        read_channel(&test_cluster, channel_id)
            .unwrap()
            .close_requested_at_ms
            .is_some(),
        "close timer must be armed",
    );

    // Reconfigure once more — the close timer should NOT reset across
    // epoch transitions.
    test_cluster.trigger_reconfiguration().await;
    assert!(
        read_channel(&test_cluster, channel_id)
            .unwrap()
            .close_requested_at_ms
            .is_some(),
        "close timer must survive reconfig",
    );

    // Drive Clock past the grace boundary. The reconfiguration above
    // already consumed wall-clock time; pump a few commits to be sure.
    let close_ts = read_channel(&test_cluster, channel_id)
        .unwrap()
        .close_requested_at_ms
        .unwrap();
    // 5_000ms = test protocol-config grace period.
    while read_clock_ts(&test_cluster) < close_ts + 5_000 {
        drive_one_commit(&test_cluster).await;
        sleep(Duration::from_millis(200)).await;
    }

    let status = submit_withdraw(&test_cluster, payer, channel_id)
        .await
        .expect("withdraw executes");
    assert!(
        status.is_ok(),
        "WithdrawAfterTimeout across reconfig: {status:?}"
    );
    assert!(
        read_channel(&test_cluster, channel_id).is_none(),
        "channel object must be deleted after withdraw",
    );
}
