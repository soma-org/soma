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

    // Get a SOMA coin to stake. Gas is balance-mode; only the stake
    // principal needs an owned input.
    let (stake_coin, _) = test_cluster
        .wallet
        .get_richest_soma_coin(sender)
        .await
        .unwrap()
        .expect("sender must have a SOMA coin to stake");

    // Build a stateless tx: empty gas_payment + ValidDuring covering
    // the current epoch. The validator's handle_vote_transaction
    // requires this combination for non-system txs without owned
    // gas coins.
    let tx_data = TransactionData::new_with_expiration(
        TransactionKind::AddStake {
            address: validator_address,
            coin_ref: stake_coin,
            amount: Some(1_000_000),
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

    let (stake_coin, _) = test_cluster
        .wallet
        .get_richest_soma_coin(sender)
        .await
        .unwrap()
        .expect("sender must have a SOMA coin");

    // Stateless tx (empty gas_payment) but expiration = None.
    // This should be rejected by the validator at signing time.
    let tx_data = TransactionData::new(
        TransactionKind::AddStake {
            address: validator_address,
            coin_ref: stake_coin,
            amount: Some(1_000_000),
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
/// well-formed. Stage 6d will wire this into the scheduler pre-pass;
/// for now we just verify the type constructs cleanly.
#[cfg(msim)]
#[msim::sim_test]
async fn test_balance_gas_reservation_constructs() {
    let alice = types::base::SomaAddress::random();
    let r = WithdrawalReservation::new(alice, CoinType::Usdc, 1_000);
    assert_eq!(r.owner, alice);
    assert_eq!(r.coin_type, CoinType::Usdc);
    assert_eq!(r.amount, 1_000);
}
