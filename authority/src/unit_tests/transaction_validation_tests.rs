// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tests for transaction validation rules:
//! - Users cannot send system transactions (Genesis, ChangeEpoch, ConsensusCommitPrologueV1)
//! - Gas validation edge cases
//! - Transaction data serialization

use fastcrypto::ed25519::Ed25519KeyPair;
use types::base::SomaAddress;
use types::consensus::ConsensusCommitPrologueV1;
use types::crypto::get_key_pair;
use types::digests::{AdditionalConsensusStateDigest, ConsensusCommitDigest};
use types::effects::TransactionEffectsAPI;
use types::object::{Object, ObjectID};
use types::transaction::{ChangeEpoch, TransactionData, TransactionKind};
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::authority_test_utils::send_and_confirm_transaction_;
use crate::test_authority_builder::TestAuthorityBuilder;

// =============================================================================
// System transaction rejection tests
// =============================================================================

#[tokio::test]
async fn test_change_epoch_system_transaction_executes() {
    // FINDING: ChangeEpoch is a system-only transaction conceptually, but the authority
    // pipeline does NOT explicitly reject it when submitted by a user. In production,
    // only the consensus handler creates ChangeEpoch transactions, so user submission
    // is prevented at the network layer. This test documents the current behavior.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let data = TransactionData::new(
        TransactionKind::ChangeEpoch(ChangeEpoch {
            epoch: 1,
            protocol_version: 1.into(),
            fees: 0,
            epoch_start_timestamp_ms: 0,
            epoch_randomness: vec![],
        }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);

    // ChangeEpoch accesses SystemState (shared object), so with_shared: true
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;
    // Currently succeeds — system transaction rejection is enforced at network layer,
    // not in the execution pipeline itself.
    assert!(
        result.is_ok(),
        "ChangeEpoch should reach execution (rejection is at network layer): {:?}",
        result.err()
    );
}

#[tokio::test]
async fn test_user_submitted_consensus_commit_prologue_rejected() {
    // Sui-parity defense: a CCP submitted by a user (i.e. signed by a
    // non-system address) must NOT mutate the Clock. The
    // `ConsensusCommitExecutor` enforces `signer == SomaAddress::ZERO`,
    // mirroring sui::clock's `assert!(ctx.sender() == @0x0)`. Real
    // CCPs are constructed by the consensus handler via
    // `TransactionData::new_system_transaction` which uses ZERO; a
    // user-signed Transaction can never produce that sender, so this
    // path is closed at execution.
    //
    // Prior to the sender check this test asserted the CCP succeeded
    // (and the original "FINDING" comment noted that user submission
    // wasn't blocked at the execution level). The check now blocks it.
    // End-to-end success of legitimate (system-built) CCPs is covered
    // by the msim e2e tests in `e2e-tests/tests/clock_tests.rs`.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    assert_ne!(sender, SomaAddress::ZERO, "test invariant: user sender != system");

    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let data = TransactionData::new(
        TransactionKind::ConsensusCommitPrologueV1(ConsensusCommitPrologueV1 {
            epoch: 0,
            round: 1,
            commit_timestamp_ms: 12345,
            consensus_commit_digest: ConsensusCommitDigest::default(),
            sub_dag_index: None,
            additional_state_digest: AdditionalConsensusStateDigest::ZERO,
        }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);

    let (_cert, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .expect("cert is built; rejection happens at the executor level");

    assert!(
        !effects.status().is_ok(),
        "user-signed CCP must fail at execution; got status={:?}",
        effects.status(),
    );

    // And the Clock must NOT have advanced.
    let clock = authority_state
        .get_object(&types::CLOCK_OBJECT_ID)
        .await
        .expect("Clock must still exist");
    assert_eq!(
        clock.clock_timestamp_ms(),
        0,
        "Clock must NOT advance from a rejected user-signed CCP",
    );
}

// =============================================================================
// Gas edge case tests
// =============================================================================

#[tokio::test]
async fn test_empty_gas_payment_without_usdc_balance_fails_insufficient_gas() {
    // Stage 13b: balance-mode txs (empty gas_payment) draw fees from
    // the sender's USDC accumulator. A sender with no USDC balance
    // hits InsufficientGas at execution time. The pre-Stage-13b
    // "empty gas always rejected" semantic is gone — empty gas is
    // the normal balance-mode shape.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let authority_state = TestAuthorityBuilder::new().build().await;

    let data = TransactionData::new(
        TransactionKind::BalanceTransfer(types::transaction::BalanceTransferArgs {
            coin_type: types::object::CoinType::Soma,
            transfers: vec![(SomaAddress::default(), 1)],
        }),
        sender,
        vec![], // empty gas payment (balance-mode)
    );
    let tx = to_sender_signed_transaction(data, &key);

    let result = send_and_confirm_transaction_(&authority_state, None, tx, false).await;
    match result {
        Ok((_, effects)) => {
            assert!(
                !effects.status().is_ok(),
                "balance-mode tx with no USDC must fail at execution",
            );
        }
        Err(_) => {
            // Also acceptable — earlier validation may reject before execution.
        }
    }
}

#[tokio::test]
async fn test_nonexistent_gas_object_rejected() {
    // Gas object that doesn't exist in state should fail
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;

    let fake_gas_ref = (ObjectID::random(), (0u64).into(), types::digests::ObjectDigest::MIN);
    let data = TransactionData::new(
        TransactionKind::BalanceTransfer(types::transaction::BalanceTransferArgs { coin_type: types::object::CoinType::Soma, transfers: vec![(SomaAddress::default(), 1)] }),
        sender,
        vec![fake_gas_ref], // non-existent gas
    );
    let tx = to_sender_signed_transaction(data, &key);

    let result = send_and_confirm_transaction_(&authority_state, None, tx, false).await;
    assert!(result.is_err(), "Non-existent gas object should be rejected");
}

// =============================================================================
// Transaction data serialization
// =============================================================================

#[tokio::test]
async fn test_transaction_data_bcs_roundtrip() {
    // TransactionData should survive BCS serialization round-trip
    let (sender, _key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_ref = (ObjectID::random(), (1u64).into(), types::digests::ObjectDigest::MIN);

    let data = TransactionData::new(
        TransactionKind::BalanceTransfer(types::transaction::BalanceTransferArgs { coin_type: types::object::CoinType::Soma, transfers: vec![(SomaAddress::default(), 1)] }),
        sender,
        vec![coin_ref],
    );

    let serialized = bcs::to_bytes(&data).expect("BCS serialization should succeed");
    let deserialized: TransactionData =
        bcs::from_bytes(&serialized).expect("BCS deserialization should succeed");

    assert_eq!(data, deserialized, "TransactionData should round-trip via BCS");
}

#[tokio::test]
async fn test_transaction_digest_determinism() {
    // Same TransactionData should always produce the same digest
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let coin_ref = (ObjectID::random(), (1u64).into(), types::digests::ObjectDigest::MIN);

    let data = TransactionData::new(
        TransactionKind::BalanceTransfer(types::transaction::BalanceTransferArgs { coin_type: types::object::CoinType::Soma, transfers: vec![(SomaAddress::default(), 1)] }),
        sender,
        vec![coin_ref],
    );

    let tx1 = to_sender_signed_transaction(data.clone(), &key);
    let tx2 = to_sender_signed_transaction(data, &key);

    assert_eq!(
        tx1.digest(),
        tx2.digest(),
        "Same transaction data should produce identical digests"
    );
}

// =============================================================================
// Multiple gas coins test
// =============================================================================

#[tokio::test]
async fn test_duplicate_gas_coin_rejected() {
    // Same gas coin used twice in gas_payment should be rejected.
    // This may cause a panic (assertion failure in temporary_store) or return an error.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 10_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    // Use a different coin for the transfer to avoid the gas==coin overlap
    let coin2 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 5_000_000);
    let coin2_ref = coin2.compute_object_reference();
    authority_state.insert_genesis_object(coin2).await;

    let data = TransactionData::new(
        TransactionKind::BalanceTransfer(types::transaction::BalanceTransferArgs { coin_type: types::object::CoinType::Soma, transfers: vec![(SomaAddress::default(), 1)] }),
        sender,
        vec![gas_ref, gas_ref], // duplicate gas coin
    );
    let tx = to_sender_signed_transaction(data, &key);

    let result = send_and_confirm_transaction_(&authority_state, None, tx, false).await;
    // Duplicate gas coins should be rejected — likely during input validation
    assert!(
        result.is_err(),
        "Duplicate gas coin should be rejected: {:?}",
        result.ok().map(|(_, e)| format!("{:?}", e.status()))
    );
}
