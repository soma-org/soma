// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Stage 7 end-to-end tests: balance-mode value transfer via the
//! account-balance accumulator.
//!
//! Submits a stateless [`TransactionKind::BalanceTransfer`] (no owned
//! gas coin, `TransactionExpiration::ValidDuring` populated) and
//! verifies:
//!
//! 1. Validators accept and execute the tx.
//! 2. Settlement at commit boundary moves balances atomically — the
//!    sender's USDC drops by `(transfer_total + gas_fee)`; each
//!    recipient's USDC increases by their share.
//! 3. Underfunded transfers are dropped by the reservation pre-pass
//!    before execution.
//!
//! The on-chain proof that BalanceTransfer is wired through the full
//! pipeline (Wire → reservation → executor → BalanceEvent → Settlement
//! → accumulator).

use test_cluster::TestClusterBuilder;
use tracing::info;
use types::base::SomaAddress;
use types::effects::TransactionEffectsAPI;
use types::object::CoinType;
use types::transaction::{
    BalanceTransferArgs, TransactionData, TransactionExpiration, TransactionKind,
};
use utils::logging::init_tracing;

fn read_usdc(test_cluster: &test_cluster::TestCluster, address: SomaAddress) -> u64 {
    test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().database_for_testing().get_balance(address, CoinType::Usdc))
        .unwrap_or(0)
}

fn unit_fee(test_cluster: &test_cluster::TestCluster) -> u64 {
    test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().fee_parameters().unit_fee
    })
}

/// Happy path: stateless USDC BalanceTransfer to two recipients. Sender's
/// balance drops by (sum of transfers + fee); each recipient gains
/// exactly their declared amount. Atomic — Settlement applies the net
/// delta in a single batch.
#[cfg(msim)]
#[msim::sim_test]
async fn test_balance_transfer_two_recipients_succeeds() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    let sender = test_cluster.get_addresses()[0];
    let recipient_a = SomaAddress::random();
    let recipient_b = SomaAddress::random();

    let chain =
        test_cluster.fullnode_handle.soma_node.with(|node| node.state().get_chain_identifier());

    let initial_sender = read_usdc(&test_cluster, sender);
    let initial_a = read_usdc(&test_cluster, recipient_a);
    let initial_b = read_usdc(&test_cluster, recipient_b);
    assert!(initial_sender > 0, "sender must start with USDC for balance-mode gas + transfer");
    assert_eq!(initial_a, 0, "fresh recipient must start at 0");
    assert_eq!(initial_b, 0, "fresh recipient must start at 0");

    let amount_a: u64 = 250_000;
    let amount_b: u64 = 750_000;
    let total_transfer = amount_a + amount_b;

    let tx_data = TransactionData::new_with_expiration(
        TransactionKind::BalanceTransfer(BalanceTransferArgs {
            coin_type: CoinType::Usdc,
            transfers: vec![(recipient_a, amount_a), (recipient_b, amount_b)],
        }),
        sender,
        Vec::new(), // EMPTY gas_payment → balance-mode
        TransactionExpiration::ValidDuring {
            min_epoch: Some(0),
            max_epoch: Some(1),
            chain,
            nonce: 0,
        },
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(
        response.effects.status().is_ok(),
        "BalanceTransfer must succeed: {:?}",
        response.effects.status(),
    );

    let final_sender = read_usdc(&test_cluster, sender);
    let final_a = read_usdc(&test_cluster, recipient_a);
    let final_b = read_usdc(&test_cluster, recipient_b);

    info!(
        sender = ?(initial_sender, final_sender),
        a = ?(initial_a, final_a),
        b = ?(initial_b, final_b),
        "BalanceTransfer balances",
    );

    // Recipients receive exactly their declared amount — no fee
    // skimming or rounding.
    assert_eq!(final_a, amount_a, "recipient A balance must equal transferred amount");
    assert_eq!(final_b, amount_b, "recipient B balance must equal transferred amount");

    // Sender pays transfer total + balance-mode gas fee. fee_units for
    // BalanceTransfer is `1 + recipients.len() = 3` (see
    // TransactionKind::fee_units).
    let fee = unit_fee(&test_cluster).saturating_mul(3);
    let expected_debit = total_transfer + fee;
    assert_eq!(
        initial_sender - final_sender,
        expected_debit,
        "sender debit must equal transfer total + gas fee (got delta {}, expected {})",
        initial_sender - final_sender,
        expected_debit,
    );
}

/// Release regression for bug #1's `ConsensusFundsReservations` gate.
///
/// The gate records each accepted withdrawal in `reserved` and must RELEASE it
/// once that commit's settlement persists (the released amount is by then baked
/// into the store balance). If release failed to fire, `reserved` would
/// double-count and wrongly block later legitimate spends from the same sender.
///
/// This test sizes two SEQUENTIAL transfers (each fully finalized+settled
/// before the next is submitted) so the second is affordable from the real,
/// reduced balance but would be REJECTED if the first's reservation were still
/// held: with each transfer ≈40% of the balance, after tx1 settles the store
/// holds ≈60%, but `store - reserved_tx1` would be only ≈20% < 40%. Both
/// succeeding proves the settlement hook releases reservations correctly.
#[cfg(msim)]
#[msim::sim_test]
async fn test_sequential_transfers_release_reservations() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    let sender = test_cluster.get_addresses()[0];
    let recipient = SomaAddress::random();
    let chain =
        test_cluster.fullnode_handle.soma_node.with(|node| node.state().get_chain_identifier());

    let initial = read_usdc(&test_cluster, sender);
    assert!(initial > 10_000_000, "sender needs a sizable balance for this test");

    // Each transfer ≈40% of the starting balance. Sum (≈80%) is affordable
    // sequentially (each settles, freeing its reservation) but `store - reserved`
    // would block the second if release were broken.
    let amount = (initial / 100) * 40;

    let make_tx = |nonce: u32| {
        TransactionData::new_with_expiration(
            TransactionKind::BalanceTransfer(BalanceTransferArgs {
                coin_type: CoinType::Usdc,
                transfers: vec![(recipient, amount)],
            }),
            sender,
            Vec::new(),
            TransactionExpiration::ValidDuring {
                min_epoch: Some(0),
                max_epoch: Some(1),
                chain,
                nonce,
            },
        )
    };

    // tx1: finalize + settle (sign_and_execute waits for effects, which for
    // balance-mode means settlement landed — cp signing blocks on it).
    let r1 = test_cluster.sign_and_execute_transaction(&make_tx(0)).await;
    assert!(r1.effects.status().is_ok(), "first transfer must succeed: {:?}", r1.effects.status());
    let after_first = read_usdc(&test_cluster, sender);

    // tx2: must succeed against the reduced real balance. If the gate failed to
    // release tx1's reservation, the effective balance would be
    // `after_first - amount` < amount and this would be dropped/rejected.
    let r2 = test_cluster.sign_and_execute_transaction(&make_tx(1)).await;
    assert!(
        r2.effects.status().is_ok(),
        "second sequential transfer must succeed once tx1's reservation is released \
         (got {:?}); a failure here means the settlement release hook did not fire",
        r2.effects.status(),
    );

    let after_second = read_usdc(&test_cluster, sender);
    assert_eq!(read_usdc(&test_cluster, recipient), amount * 2, "recipient got both transfers");
    assert!(after_second < after_first, "sender balance must drop after the second transfer");
    info!(initial, after_first, after_second, amount, "sequential transfers released correctly");
}

/// Deposit-then-withdraw end-to-end (residual 1 of bug #1).
///
/// An account receives funds (a deposit) and then spends them — including more
/// than it held before the deposit. The funds-withdraw scheduler must let the
/// recipient spend the received funds: it tracks the deposit via the settlement
/// delta and (if the spend is scheduled before the deposit settles) holds it
/// Pending until settlement, then approves it. Under the OLD synchronous gate
/// this cross-account deposit→withdraw could be dropped/forked on settlement
/// timing; now it resolves deterministically.
///
/// Here: A is funded with just enough for gas + a small base; a large transfer
/// lands in A; then A forwards (base + most of the deposit) to B. The forward
/// is only affordable because of the deposit — proving received funds are
/// spendable.
#[cfg(msim)]
#[msim::sim_test]
async fn test_deposit_then_withdraw_spends_received_funds() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    use fastcrypto::ed25519::Ed25519KeyPair;

    let funder = test_cluster.get_addresses()[0];
    // A is a fresh, wallet-unmanaged account; sign its own spend manually.
    let (a, a_kp): (SomaAddress, Ed25519KeyPair) = types::crypto::get_key_pair();
    let b = SomaAddress::random();
    let chain =
        test_cluster.fullnode_handle.soma_node.with(|node| node.state().get_chain_identifier());

    let fee = unit_fee(&test_cluster);
    // Seed A with enough for its forward's gas (fee_units = 1 + 1 recipient = 2)
    // plus a tiny base, but NOT enough to forward a large amount — that must
    // come from the deposit.
    let a_seed = fee * 2 + 1_000;
    let deposit = 5_000_000u64;

    let mk = |sender, recipient, amount, nonce| {
        TransactionData::new_with_expiration(
            TransactionKind::BalanceTransfer(BalanceTransferArgs {
                coin_type: CoinType::Usdc,
                transfers: vec![(recipient, amount)],
            }),
            sender,
            Vec::new(),
            TransactionExpiration::ValidDuring {
                min_epoch: Some(0),
                max_epoch: Some(1),
                chain,
                nonce,
            },
        )
    };

    // Fund A's gas base (from the wallet-managed funder).
    let r = test_cluster.sign_and_execute_transaction(&mk(funder, a, a_seed, 0)).await;
    assert!(r.effects.status().is_ok(), "seed A: {:?}", r.effects.status());
    // Deposit a large amount into A.
    let r = test_cluster.sign_and_execute_transaction(&mk(funder, a, deposit, 1)).await;
    assert!(r.effects.status().is_ok(), "deposit to A: {:?}", r.effects.status());

    assert_eq!(read_usdc(&test_cluster, a), a_seed + deposit, "A holds seed + deposit");

    // A forwards most of the deposit to B — only possible because the deposit
    // was credited and is spendable. Signed with A's own key.
    let forward = deposit; // more than A's pre-deposit balance
    let signed =
        types::transaction::Transaction::from_data_and_signer(mk(a, b, forward, 0), vec![&a_kp]);
    let r = test_cluster
        .wallet
        .execute_transaction_may_fail(signed)
        .await
        .expect("A's forward should finalize");
    assert!(
        r.effects.status().is_ok(),
        "A must be able to spend received funds (deposit-then-withdraw): {:?}",
        r.effects.status()
    );
    assert_eq!(read_usdc(&test_cluster, b), forward, "B received the forwarded deposit");
    info!(a_seed, deposit, forward, "deposit-then-withdraw spent received funds");
}

/// A self-transfer (sender == recipient) is rejected at execution
/// time by the executor's invariant — it's always a no-op or a wallet
/// bug. The tx still consumes its gas (by Soma's failed-effect
/// convention), but no transfer balance moves.
#[cfg(msim)]
#[msim::sim_test]
async fn test_balance_transfer_self_recipient_rejected() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    let sender = test_cluster.get_addresses()[0];
    let chain =
        test_cluster.fullnode_handle.soma_node.with(|node| node.state().get_chain_identifier());

    let initial_sender = read_usdc(&test_cluster, sender);

    let tx_data = TransactionData::new_with_expiration(
        TransactionKind::BalanceTransfer(BalanceTransferArgs {
            coin_type: CoinType::Usdc,
            transfers: vec![(sender, 100)],
        }),
        sender,
        Vec::new(),
        TransactionExpiration::ValidDuring {
            min_epoch: Some(0),
            max_epoch: Some(1),
            chain,
            nonce: 0,
        },
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(
        !response.effects.status().is_ok(),
        "self-transfer must NOT succeed: {:?}",
        response.effects.status(),
    );

    // Failed-effect path still charges gas, but the transfer total
    // should NOT have moved. Easy invariant: net change must be exactly
    // the gas fee (not gas + transfer).
    let final_sender = read_usdc(&test_cluster, sender);
    let fee = unit_fee(&test_cluster).saturating_mul(2); // 1 + 1 recipient
    assert_eq!(
        initial_sender - final_sender,
        fee,
        "self-transfer must consume gas only, not transfer amount",
    );
}

/// Underfunded BalanceTransfer is rejected — fast.
///
/// F7/F23: the tx must NOT reach neither execution nor settlement, and
/// the submitter must NOT be left waiting out the finality timeout.
/// The best-effort funds check in early validation rejects it before
/// consensus with a non-retriable `InsufficientBalance` error; the
/// consensus reservation pre-pass remains the deterministic net for
/// the concurrent / cumulative case. Either way the recipient must see
/// no balance change.
#[cfg(msim)]
#[msim::sim_test]
async fn test_balance_transfer_underfunded_dropped_by_prepass() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    let chain =
        test_cluster.fullnode_handle.soma_node.with(|node| node.state().get_chain_identifier());

    // Synthesize a fresh, unfunded address. Sign manually and submit
    // via the may-fail wallet path.
    use fastcrypto::ed25519::Ed25519KeyPair;
    let (unfunded_sender, kp): (SomaAddress, Ed25519KeyPair) = types::crypto::get_key_pair();
    let recipient = SomaAddress::random();

    assert_eq!(read_usdc(&test_cluster, unfunded_sender), 0);
    assert_eq!(read_usdc(&test_cluster, recipient), 0);

    let tx_data = TransactionData::new_with_expiration(
        TransactionKind::BalanceTransfer(BalanceTransferArgs {
            coin_type: CoinType::Usdc,
            transfers: vec![(recipient, 1_000_000)],
        }),
        unfunded_sender,
        Vec::new(),
        TransactionExpiration::ValidDuring {
            min_epoch: Some(0),
            max_epoch: Some(1),
            chain,
            nonce: 0,
        },
    );
    let signed_tx = types::transaction::Transaction::from_data_and_signer(tx_data, vec![&kp]);

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        test_cluster.wallet.execute_transaction_may_fail(signed_tx),
    )
    .await;

    match result {
        // Acceptable: the tx finalized as a failed effect.
        Ok(Ok(response)) => {
            assert!(
                !response.effects.status().is_ok(),
                "underfunded BalanceTransfer must NOT succeed",
            );
        }
        // Expected: early validation rejects the underfunded tx fast,
        // before consensus, with a clear insufficient-balance reason.
        Ok(Err(e)) => {
            let msg = format!("{e:#}").to_lowercase();
            assert!(
                msg.contains("insufficient") && msg.contains("balance"),
                "expected an insufficient-balance rejection, got: {e:#}",
            );
            info!("underfunded tx rejected fast at submission: {e:#}");
        }
        // F23 regression: a pre-pass-dropped tx that produces no
        // observable outcome leaves the client waiting out the full
        // finality window.
        Err(_) => panic!(
            "underfunded tx must be rejected fast — it instead timed out \
             waiting for finality (F23 regression)",
        ),
    }

    // Critical: the recipient must NOT have been credited — Settlement
    // never saw a Deposit event for this tx. Without the pre-pass, an
    // accumulator underflow on the sender would leave the system in an
    // inconsistent state.
    assert_eq!(
        read_usdc(&test_cluster, recipient),
        0,
        "recipient must not be credited by an underfunded transfer",
    );
}

/// F22: `simulate_transaction` must model the consensus reservation
/// pre-pass. The balance-transfer executor has no funds check of its
/// own — sufficiency is enforced only by the pre-pass, which simulation
/// bypasses. Without modelling it, simulating an unspendable transfer
/// wrongly reports "Would Succeed". This verifies an over-balance
/// transfer simulates as a failure while an affordable one still
/// simulates as success.
#[cfg(msim)]
#[msim::sim_test]
async fn test_simulate_over_balance_transfer_reports_failure() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    let sender = test_cluster.get_addresses()[0];
    let recipient = SomaAddress::random();
    let chain =
        test_cluster.fullnode_handle.soma_node.with(|node| node.state().get_chain_identifier());

    let balance = read_usdc(&test_cluster, sender);
    assert!(balance > 0, "sender must start with USDC");

    let make_tx = |amount: u64, nonce: u32| {
        TransactionData::new_with_expiration(
            TransactionKind::BalanceTransfer(BalanceTransferArgs {
                coin_type: CoinType::Usdc,
                transfers: vec![(recipient, amount)],
            }),
            sender,
            Vec::new(),
            TransactionExpiration::ValidDuring {
                min_epoch: Some(0),
                max_epoch: Some(1),
                chain,
                nonce,
            },
        )
    };

    // Over-balance transfer: simulation must report a failure.
    let over_tx = make_tx(balance.saturating_add(1_000_000_000), 0);
    let over_result = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .simulate_transaction(over_tx, types::transaction_executor::TransactionChecks::Enabled)
            .expect("simulation itself must not error")
    });
    assert!(
        over_result.execution_result.is_err(),
        "simulating an over-balance transfer must report failure, got: {:?}",
        over_result.execution_result,
    );
    assert!(
        !over_result.effects.status().is_ok(),
        "over-balance simulation effects must carry a failure status",
    );
    info!("over-balance simulation correctly failed: {:?}", over_result.execution_result);

    // Control: a small, affordable transfer still simulates as success.
    let ok_tx = make_tx(1_000, 1);
    let ok_result = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .simulate_transaction(ok_tx, types::transaction_executor::TransactionChecks::Enabled)
            .expect("simulation itself must not error")
    });
    assert!(
        ok_result.execution_result.is_ok(),
        "an affordable transfer must still simulate as success, got: {:?}",
        ok_result.execution_result,
    );
}
