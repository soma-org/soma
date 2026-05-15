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

    let chain = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_chain_identifier());

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
    let chain = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_chain_identifier());

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

/// Underfunded BalanceTransfer is dropped by the reservation pre-pass
/// (Stage 6d). Reaches neither execution nor settlement. The recipient
/// must see no balance change.
#[cfg(msim)]
#[msim::sim_test]
async fn test_balance_transfer_underfunded_dropped_by_prepass() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    let chain = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_chain_identifier());

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
        Ok(Ok(response)) => {
            assert!(
                !response.effects.status().is_ok(),
                "underfunded BalanceTransfer must NOT succeed",
            );
        }
        Ok(Err(e)) => info!("underfunded tx rejected at submission: {:#}", e),
        Err(_) => info!("underfunded tx dropped by pre-pass (timeout)"),
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
