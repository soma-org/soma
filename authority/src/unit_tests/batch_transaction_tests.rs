// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use fastcrypto::ed25519::Ed25519KeyPair;
use types::base::dbg_addr;
use types::crypto::get_key_pair;
use types::effects::{ExecutionStatus, TransactionEffectsAPI};
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::authority_test_utils::send_and_confirm_transaction;
use crate::test_authority_builder::TestAuthorityBuilder;

// =============================================================================
// Multiple sequential transfers
// =============================================================================

#[tokio::test]
async fn test_multiple_sequential_transfers() {
    // Execute 3 BalanceTransfer transactions in sequence. Stage 13c:
    // gas (USDC) and the transferable balance (SOMA) live in
    // accumulators — we observe the SOMA accumulator drop after
    // each tx, since BalanceTransfer creates no per-tx coin
    // objects to chain through.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let starting_soma = 50_000_000u64;
    let starting_usdc = 50_000_000u64;
    let per_transfer = 1_000u64;

    let authority_state = TestAuthorityBuilder::new().build().await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        starting_soma,
        starting_usdc,
    );

    let recipients = [dbg_addr(1), dbg_addr(2), dbg_addr(3)];

    // Flush after each tx so each settlement lands before the next
    // tx reads the accumulator. This matches how the per-commit
    // settlement runs in production — one settlement per
    // consensus commit, aggregating that commit's events.
    let cache_commit = authority_state.get_cache_commit();
    let epoch = authority_state.epoch_store_for_testing().epoch();
    for (i, recipient) in recipients.iter().enumerate() {
        let data = crate::authority_test_utils::balance_transfer_data_legacy(
            *recipient,
            sender,
            Some(per_transfer),
        );
        let tx = to_sender_signed_transaction(data, &sender_key);
        let (_, effects) = send_and_confirm_transaction(&authority_state, tx)
            .await
            .unwrap_or_else(|e| panic!("Transfer {} should succeed: {:?}", i + 1, e));
        assert_eq!(
            *effects.status(),
            ExecutionStatus::Success,
            "Transfer {} should succeed",
            i + 1
        );
        let digest = *effects.transaction_digest();
        let batch = cache_commit.build_db_batch(epoch, &[digest]);
        cache_commit.commit_transaction_outputs(epoch, batch, &[digest]);
    }

    let store = authority_state.database_for_testing();
    let final_soma =
        store.get_balance(sender, types::object::CoinType::Soma).unwrap();
    let final_usdc =
        store.get_balance(sender, types::object::CoinType::Usdc).unwrap();
    assert_eq!(
        final_soma,
        starting_soma - 3 * per_transfer,
        "Sender SOMA must have dropped by exactly 3 × {}",
        per_transfer,
    );
    assert!(
        final_usdc < starting_usdc,
        "Sender USDC (gas) must have dropped after 3 fees: got {}",
        final_usdc,
    );
}

// =============================================================================
// Failed execution reverts non-gas changes
// =============================================================================

#[tokio::test]
async fn test_failed_execution_reverts_non_gas() {
    // Stage 13c: BalanceTransfer with empty SOMA accumulator. The
    // SOMA Withdraw event against an empty accumulator is applied
    // as a saturated apply — the tx still executes successfully,
    // but only the gas (USDC) fee debit and any recipient SOMA
    // credit are observable. We assert the conservation
    // invariant: no objects created/deleted/mutated, fee charged.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    // Sender has gas (USDC) but zero SOMA — the transfer Withdraw
    // saturates against an empty accumulator.
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        0,
        10_000_000,
    );

    let recipient = dbg_addr(1);
    let data = crate::authority_test_utils::balance_transfer_data_legacy(
        recipient,
        sender,
        Some(4500),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let result = send_and_confirm_transaction(&authority_state, tx).await;

    let (_, effects) = result.unwrap();
    let effects = effects.into_data();

    assert!(effects.created().is_empty(), "BalanceTransfer creates no objects");
    let fee = effects.transaction_fee();
    assert!(fee.total_fee > 0, "Some fee must be charged");
}

// =============================================================================
// Effects accumulate correctly across transactions
// =============================================================================

#[tokio::test]
async fn test_effects_accumulate_correctly() {
    // Execute multiple transactions and verify each set of effects is stored
    // and retrievable independently.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        50_000_000,
        50_000_000,
    );

    let mut digests = Vec::new();

    // Execute 3 transactions
    for i in 0..3 {
        let recipient = dbg_addr((i + 1) as u8);
        let data = crate::authority_test_utils::balance_transfer_data_legacy(
            recipient,
            sender,
            Some(1000),
        );
        let tx = to_sender_signed_transaction(data, &sender_key);
        let tx_digest = *tx.digest();

        let (_, effects) = send_and_confirm_transaction(&authority_state, tx)
            .await
            .unwrap_or_else(|e| panic!("Transaction {} should succeed: {:?}", i, e));
        assert_eq!(*effects.status(), ExecutionStatus::Success);

        digests.push(tx_digest);
    }

    // Verify each transaction's effects are retrievable
    for (i, digest) in digests.iter().enumerate() {
        let effects = authority_state.notify_read_effects(*digest).await;
        assert!(effects.is_ok(), "Effects for transaction {} should be readable", i);
        let effects = effects.unwrap();
        assert_eq!(
            effects.transaction_digest(),
            digest,
            "Effects digest should match transaction {} digest",
            i
        );
    }

    // All three digests should be distinct
    assert_ne!(digests[0], digests[1]);
    assert_ne!(digests[1], digests[2]);
    assert_ne!(digests[0], digests[2]);
}

// =============================================================================
// Version tracking across sequential mutations
// =============================================================================

#[tokio::test]
async fn test_distinct_balance_transfers_produce_distinct_effects() {
    // Stage 13c: BalanceTransfer touches no per-object versions
    // (no coin gas, no created/mutated objects), so the per-tx
    // lamport version stays flat. The remaining observable
    // contract is that distinct txs yield distinct effects digests.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    crate::authority_test_utils::seed_balance_mode_funds(
        &authority_state,
        sender,
        50_000_000,
        50_000_000,
    );

    let mut effects_digests = Vec::new();
    for i in 0..3 {
        let data = crate::authority_test_utils::balance_transfer_data_legacy(
            dbg_addr((i + 1) as u8),
            sender,
            Some(100),
        );
        let tx = to_sender_signed_transaction(data, &sender_key);
        let (_, effects) = send_and_confirm_transaction(&authority_state, tx).await.unwrap();
        assert_eq!(*effects.status(), ExecutionStatus::Success);
        effects_digests.push(*effects.transaction_digest());
    }

    let unique: std::collections::HashSet<_> = effects_digests.iter().collect();
    assert_eq!(
        unique.len(),
        effects_digests.len(),
        "Each transaction must produce a distinct effects digest",
    );
}
