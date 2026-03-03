// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for object lock behavior after transaction failures.
//!
//! Tests:
//! 1. test_lock_persists_after_insufficient_gas - Demonstrates that a failed
//!    InsufficientGas transaction permanently locks the gas coin within the epoch
//!    on a 3-validator network.

use test_cluster::TestClusterBuilder;
use tracing::info;
use types::base::SomaAddress;
use types::config::genesis_config::SHANNONS_PER_SOMA;
use types::effects::TransactionEffectsAPI;
use types::transaction::{TransactionData, TransactionKind};
use utils::logging::init_tracing;

// ===================================================================
// Test 1: Lock persists after InsufficientGas failure
//
// On a 3-validator network (quorum = 2), when a transaction fails during
// gas preparation (InsufficientGas), the gas coin's lock is never released
// because error_result() creates an empty InnerTemporaryStore that discards
// gas coin mutations. The gas coin version never advances, so subsequent
// transactions using that coin hit ObjectLockConflict.
//
// Steps:
// 1. Transfer a tiny amount (10 shannons) to create a dust coin
// 2. Use the dust coin as gas for a TransferCoin → fails InsufficientGas
// 3. Attempt another tx with the same dust coin → ObjectLockConflict (bug)
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_lock_persists_after_insufficient_gas() {
    init_tracing();

    // Use 3 validators — quorum is 2, so 2 locked validators = permanent lock
    let test_cluster = TestClusterBuilder::new().with_num_validators(3).build().await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    // Step 1: Get a gas coin and transfer a tiny amount to create a dust coin
    let gas_coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(sender, Some(2)).await.unwrap();
    assert!(gas_coins.len() >= 2, "Sender needs at least 2 coins");

    let main_coin = gas_coins[0];
    let funding_gas = gas_coins[1];

    info!("Main coin: {} v{}", main_coin.0, main_coin.1.value());
    info!("Funding gas: {} v{}", funding_gas.0, funding_gas.1.value());

    // Transfer 10 shannons to sender's own address to create a dust coin
    // We use the main_coin as the transfer coin with amount=10, and funding_gas as gas
    let dust_amount: u64 = 10; // 10 shannons — well below any base fee
    let create_dust_tx = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: main_coin,
            amount: Some(dust_amount),
            recipient: sender, // Send to self to create a new small coin
        },
        sender,
        vec![funding_gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&create_dust_tx).await;
    assert!(
        response.effects.status().is_ok(),
        "Dust coin creation should succeed: {:?}",
        response.effects.status()
    );

    // Find the created dust coin (the new coin sent to sender with dust_amount)
    let created = response.effects.created();
    info!("Created {} objects", created.len());

    // The created coin is the one owned by sender (there should be exactly one created)
    let dust_coin_id = created
        .iter()
        .find_map(|(oref, owner)| {
            if matches!(owner, types::object::Owner::AddressOwner(addr) if *addr == sender) {
                Some(oref)
            } else {
                None
            }
        })
        .expect("Should have created a dust coin for sender");

    info!("Dust coin created: {} v{}", dust_coin_id.0, dust_coin_id.1.value());

    // Get an updated ref for the funding gas coin (its version changed)
    let updated_coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(sender, Some(10)).await.unwrap();

    let updated_funding_gas = updated_coins
        .iter()
        .find(|c| c.0 == funding_gas.0)
        .expect("Funding gas should still exist");

    info!("Updated funding gas: {} v{}", updated_funding_gas.0, updated_funding_gas.1.value());

    // Step 2: Use the dust coin as gas for a transaction — should fail with InsufficientGas
    // The dust coin has only 10 shannons, which is below the base fee
    let fail_tx = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: *dust_coin_id, // Use dust coin as the transfer coin AND gas
            amount: Some(1),     // Try to transfer 1 shannon
            recipient,
        },
        sender,
        vec![*dust_coin_id], // Use dust coin as gas payment
    );

    let fail_tx_signed = test_cluster.wallet.sign_transaction(&fail_tx).await;
    let fail_result = test_cluster.wallet.execute_transaction_may_fail(fail_tx_signed).await;

    match &fail_result {
        Ok(resp) => {
            info!("First dust tx completed with effects status: {:?}", resp.effects.status());
            // Should have InsufficientGas in effects
            assert!(resp.effects.status().is_err(), "Transaction with dust gas should fail");
        }
        Err(e) => {
            info!("First dust tx failed at orchestrator level: {}", e);
        }
    }

    // Step 3: Verify the fix — the dust coin should have been properly mutated
    // (version advanced) by into_effects(). Since the partial fee deduction took
    // all 10 shannons, the coin was deleted (balance → 0). This means:
    // - The object version advanced (lock on old version is consumed)
    // - The coin no longer exists (get_gas_objects won't find it)
    //
    // If the bug persists (error_result discards mutations), the dust coin would
    // still exist at the same version with the lock blocking any new transaction.

    let updated_coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(sender, Some(20)).await.unwrap();

    let dust_still_exists = updated_coins.iter().any(|c| c.0 == dust_coin_id.0);

    if dust_still_exists {
        // Dust coin still exists — it may have been partially deducted but not fully consumed.
        // Try using it in a DIFFERENT transaction to verify no ObjectLockConflict.
        let current_dust_ref = updated_coins.iter().find(|c| c.0 == dust_coin_id.0).unwrap();

        info!(
            "Dust coin still exists after failed tx: {} v{} (was v{})",
            current_dust_ref.0,
            current_dust_ref.1.value(),
            dust_coin_id.1.value()
        );

        // Version should have advanced if the fix works
        assert!(
            current_dust_ref.1.value() > dust_coin_id.1.value(),
            "Dust coin version should have advanced after gas preparation failure. \
             Old: v{}, Current: v{}. This indicates error_result() discarded gas mutations.",
            dust_coin_id.1.value(),
            current_dust_ref.1.value()
        );

        // Try a different transaction with the updated ref — should not hit ObjectLockConflict
        let retry_tx = TransactionData::new(
            TransactionKind::TransferCoin {
                coin: *current_dust_ref,
                amount: Some(1),
                recipient: sender,
            },
            sender,
            vec![*current_dust_ref],
        );

        let retry_tx_signed = test_cluster.wallet.sign_transaction(&retry_tx).await;
        let retry_result = test_cluster.wallet.execute_transaction_may_fail(retry_tx_signed).await;

        match &retry_result {
            Ok(resp) => {
                info!("Retry tx effects status: {:?}", resp.effects.status());
            }
            Err(e) => {
                let err_str = format!("{}", e);
                assert!(
                    !err_str.contains("ObjectLockConflict") && !err_str.contains("already locked"),
                    "BUG: Dust coin still locked after fix. Error: {}",
                    e
                );
                info!("Retry tx failed with non-lock error (acceptable): {}", e);
            }
        }
    } else {
        // Dust coin was deleted (balance went to 0 after partial fee deduction).
        // This is the expected outcome: into_effects() preserved the gas coin mutations,
        // the version advanced, and the coin was consumed. The old lock is irrelevant.
        info!(
            "SUCCESS: Dust coin {} was properly consumed (deleted after partial gas deduction). \
             Lock released via version advancement.",
            dust_coin_id.0
        );
    }
}
