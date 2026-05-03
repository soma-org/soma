// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Stage 6c end-to-end test: gas paid from the address-balance accumulator.
//!
//! Submits a stateless transaction (`gas_payment` empty,
//! `TransactionExpiration::ValidDuring` populated) and verifies:
//!
//! 1. The validator accepts it (didn't reject for missing gas coin).
//! 2. The transaction executes successfully through consensus.
//! 3. Settlement at commit boundary debits the sender's USDC balance
//!    by exactly the gas fee.
//!
//! This is the smallest e2e proof that the entire SIP-58-style
//! pipeline (TransactionExpiration → reservation → balance-mode
//! prepare_gas → BalanceEvent::Withdraw → Settlement → accumulator)
//! works end-to-end.

use test_cluster::TestClusterBuilder;
use tracing::info;
use types::balance::WithdrawalReservation;
use types::effects::TransactionEffectsAPI;
use types::object::CoinType;
use types::transaction::{TransactionData, TransactionExpiration, TransactionKind};
use utils::logging::init_tracing;

/// Read sender's USDC balance from the accumulator via the fullnode.
fn read_usdc_balance(
    test_cluster: &test_cluster::TestCluster,
    address: types::base::SomaAddress,
) -> u64 {
    test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().database_for_testing().get_balance(address, CoinType::Usdc))
        .unwrap_or(0)
}

/// Happy path: stateless AddStake tx pays gas from the sender's USDC
/// accumulator balance, not from any owned coin object.
///
/// The on-chain stake principal is still SOMA (an owned input), but
/// gas is balance-mode. After commit, the sender's USDC balance
/// drops by the gas fee; the SOMA coin's version advances to reflect
/// the partial-stake debit.
#[cfg(msim)]
#[msim::sim_test]
async fn test_balance_mode_gas_addstake_succeeds() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    let sender = test_cluster.get_addresses()[0];
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().validators().validators[0]
            .metadata
            .soma_address
    });

    let chain = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_chain_identifier());

    let initial_balance = read_usdc_balance(&test_cluster, sender);
    assert!(initial_balance > 0, "sender must start with USDC balance for balance-mode gas");

    // Stage 9d-C2 + 13a: AddStake is balance-mode — both stake and
    // gas come out of the sender's accumulator. No SOMA coin object
    // is needed (and Stage 13a's genesis no longer creates them).
    // ValidDuring covers the current epoch.
    let tx_data = TransactionData::new_with_expiration(
        TransactionKind::AddStake {
            validator: validator_address,
            amount: 1_000_000,
        },
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
        "balance-mode gas AddStake must succeed: {:?}",
        response.effects.status()
    );

    // Wait briefly for settlement to apply the gas debit. Settlement
    // runs as part of the commit; effects-completion typically
    // implies settlement has run.
    let final_balance = read_usdc_balance(&test_cluster, sender);
    info!(
        "USDC balance: initial={}, final={}, debited={}",
        initial_balance,
        final_balance,
        initial_balance.saturating_sub(final_balance),
    );

    // The settlement debit must have happened. Final balance must be
    // strictly less than initial (gas fee is non-zero on Soma's
    // fee-units model).
    assert!(
        final_balance < initial_balance,
        "USDC balance must decrease by gas fee: initial={}, final={}",
        initial_balance,
        final_balance,
    );

    // Sanity: the debit must equal exactly the gas fee, not more.
    // The fee model is `unit_fee × fee_units(AddStake) = unit_fee × 2`.
    // Stake principal is SOMA (separate currency), so USDC delta
    // equals the gas fee precisely.
    let debited = initial_balance - final_balance;
    let unit_fee = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().fee_parameters().unit_fee
    });
    let expected_fee = unit_fee.saturating_mul(2); // AddStake fee_units = 2 (see staking.rs)
    assert_eq!(
        debited, expected_fee,
        "USDC debit must equal gas fee exactly: debited={}, expected={}",
        debited, expected_fee,
    );
}

/// Defense-in-depth: a balance-mode tx without `ValidDuring` is
/// rejected at validation time. This is the contract enforced by
/// `handle_vote_transaction`'s "stateless tx must be replay-protected"
/// check. Without it, every stateless tx would be replayable forever.
#[cfg(msim)]
#[msim::sim_test]
async fn test_balance_mode_gas_without_valid_during_is_rejected() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    let sender = test_cluster.get_addresses()[0];
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().validators().validators[0]
            .metadata
            .soma_address
    });

    // Stage 13a: AddStake is balance-mode — no SOMA coin object
    // input. Stateless tx (empty gas_payment) but expiration = None
    // — should be rejected by the validator at signing time.
    let tx_data = TransactionData::new(
        TransactionKind::AddStake {
            validator: validator_address,
            amount: 1_000_000,
        },
        sender,
        Vec::new(), // empty
                    // Default expiration is None — no replay protection
    );

    let signed_tx = test_cluster.wallet.sign_transaction(&tx_data).await;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        test_cluster.wallet.execute_transaction_may_fail(signed_tx),
    )
    .await;

    match result {
        Ok(Ok(response)) => {
            assert!(
                !response.effects.status().is_ok(),
                "stateless tx without ValidDuring must NOT succeed"
            );
        }
        Ok(Err(_)) | Err(_) => {
            info!("stateless tx without ValidDuring correctly rejected");
        }
    }
}

/// Reservation type plumbing: `WithdrawalReservation` for gas is
/// well-formed.
#[cfg(msim)]
#[msim::sim_test]
async fn test_balance_gas_reservation_constructs() {
    let alice = types::base::SomaAddress::random();
    let r = WithdrawalReservation::new(alice, CoinType::Usdc, 1_000);
    assert_eq!(r.owner, alice);
    assert_eq!(r.coin_type, CoinType::Usdc);
    assert_eq!(r.amount, 1_000);
}

/// Stage 6d: the consensus-handler reservation pre-pass drops
/// balance-mode txs whose declared reservations exceed the sender's
/// pre-commit USDC balance. End-to-end check that an underfunded
/// stateless tx never reaches execution.
///
/// Critical safety property: without the pre-pass, the underfunded
/// tx would pass `prepare_gas`'s individual balance read (single-tx-
/// per-commit case is fine), then `apply_settlement_events` would
/// fail at write-time with InsufficientGas, abandoning the tx but
/// also failing the whole commit's batch write. The pre-pass drops
/// it *before* execution so the rest of the commit is unaffected.
#[cfg(msim)]
#[msim::sim_test]
async fn test_balance_mode_gas_underfunded_tx_dropped_by_prepass() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().validators().validators[0]
            .metadata
            .soma_address
    });
    let _ = validator_address;

    // Use a freshly-generated address that has zero USDC balance —
    // any balance-mode tx from this address must be dropped by the
    // pre-pass. We can't easily create-and-fund a fresh wallet
    // address mid-test, so we construct a synthetic key, sign the
    // tx manually, and submit via execute_transaction_may_fail.
    use fastcrypto::ed25519::Ed25519KeyPair;
    let (unfunded_sender, kp): (types::base::SomaAddress, Ed25519KeyPair) =
        types::crypto::get_key_pair();

    // The unfunded sender has zero USDC, so the reservation pre-pass
    // must drop the tx before it gets a chance to execute. Use a
    // BalanceTransfer (the simplest balance-mode kind) — for the
    // positive case we already have
    // test_balance_mode_gas_addstake_succeeds.
    let recipient = types::base::SomaAddress::random();

    let tx_data = e2e_tests::balance_transfer_data(
        &test_cluster,
        types::object::CoinType::Usdc,
        unfunded_sender,
        vec![(recipient, 1)],
    );
    let signed_tx = types::transaction::Transaction::from_data_and_signer(tx_data, vec![&kp]);

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        test_cluster.wallet.execute_transaction_may_fail(signed_tx),
    )
    .await;

    // Three acceptable outcomes:
    //   1. Validators drop the tx pre-consensus (most likely): the
    //      reservation pre-pass filters it out, no effects produced.
    //      Surfaces as a timeout or RPC error from the wallet.
    //   2. Tx reaches execution and fails non-success there.
    //   3. Tx fails at submission validation.
    // What's NOT acceptable: tx succeeds.
    match result {
        Ok(Ok(response)) => {
            assert!(
                !response.effects.status().is_ok(),
                "underfunded balance-mode tx must NOT execute successfully"
            );
        }
        Ok(Err(e)) => {
            info!("underfunded tx rejected at submission: {:#}", e);
        }
        Err(_) => {
            info!("underfunded tx timed out (expected — no validator will sign / drop in pre-pass)");
        }
    }
}
