// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Security regression test for the "revert leak": a transaction that
//! emits a balance/accumulator event and then FAILS must not have that
//! event applied (a failed tx must not move balances beyond its gas fee).
//!
//! `revert_non_gas_changes` (authority/src/execution/mod.rs) historically
//! restored only `written_objects`/`deleted_object_ids`, leaving the
//! emitted `balance_events` / `delegation_events` / `accumulator_writes`
//! in the failed tx's effects — which the per-commit Settlement then
//! applied unconditionally, minting/burning balance on a FAILED tx.
//!
//! Live trigger used here: `execute_bridge_withdraw` emits the burn
//! `Split` for the withdrawal amount BEFORE the fallible
//! `total_usdc_supply.checked_sub(amount)`. Genesis allocates USDC
//! *balances* for tests but never bumps `total_usdc_supply` (USDC is a
//! bridged asset; see genesis_builder), so a withdrawal of genesis USDC
//! underflows the supply counter and the tx fails — exercising the leak.
//!
//! EXPECTED (secure): the withdraw tx fails AND the account keeps its USDC
//! (only the small gas fee is deducted).
//! FAILURE (vulnerable): the withdraw amount is burned despite the failed
//! tx.

use test_cluster::TestClusterBuilder;
use tracing::info;
use types::base::SomaAddress;
use types::bridge::BridgeChainId;
use types::effects::TransactionEffectsAPI;
use types::object::CoinType;
use types::transaction::{BridgeWithdrawArgs, TransactionKind};
use utils::logging::init_tracing;

fn read_usdc(test_cluster: &test_cluster::TestCluster, address: SomaAddress) -> u64 {
    test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().database_for_testing().get_balance(address, CoinType::Usdc))
        .unwrap_or(0)
}

/// A `BridgeWithdraw` that fails (supply underflow) must NOT burn the
/// withdrawal amount — only the gas fee may be deducted.
#[cfg(msim)]
#[msim::sim_test]
async fn failed_bridge_withdraw_does_not_burn_balance() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    // Default genesis funds account[0] with USDC (balance only — genesis
    // never bumps total_usdc_supply).
    let account = test_cluster.get_addresses()[0];
    let before = read_usdc(&test_cluster, account);
    assert!(before > 0, "account must start with USDC (balance-mode gas + withdraw)");

    // Withdraw far less than the balance, so the (buggy) leaked burn shows
    // up as a measurable balance drop rather than underflowing settlement.
    let amount: u64 = 100_000_000; // 100 USDC
    assert!(before > amount * 2, "fund must exceed the withdraw amount");

    let kind = TransactionKind::BridgeWithdraw(BridgeWithdrawArgs {
        amount,
        recipient_eth_address: [0u8; 20],
        target_chain: BridgeChainId::EthSepolia,
    });
    let tx_data = e2e_tests::stateless_tx_data(&test_cluster, account, kind);
    let tx = test_cluster.wallet.sign_transaction(&tx_data).await;

    let resp = test_cluster.wallet.execute_transaction_may_fail(tx).await;

    // The withdraw must fail (genesis total_usdc_supply == 0 → underflow).
    // If it unexpectedly succeeded, the test premise is invalid.
    match &resp {
        Ok(r) => {
            info!("withdraw status: {:?}", r.effects.status());
            assert!(
                !r.effects.status().is_ok(),
                "test premise invalid: BridgeWithdraw unexpectedly SUCCEEDED \
                 (genesis must leave total_usdc_supply == 0)"
            );
        }
        Err(e) => info!("withdraw rejected/failed: {e}"),
    }

    let after = read_usdc(&test_cluster, account);
    let drop = before.saturating_sub(after);
    info!(before, after, drop, amount, "balance around failed withdraw");

    // SECURITY: the failed withdraw must not have burned the amount. A
    // secure chain deducts only the (small) gas fee, far below `amount`.
    assert!(
        drop < amount,
        "VULNERABLE: failed BridgeWithdraw burned {drop} USDC (>= withdraw amount {amount}) \
         from a FAILED transaction — revert leak",
    );
}
