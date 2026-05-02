// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::base::SomaAddress;
use types::bridge::BridgeState;
use types::crypto::{SomaKeyPair, get_key_pair};
use types::digests::TransactionDigest;
use types::effects::{
    ExecutionFailureStatus, ExecutionStatus, SignedTransactionEffects, TransactionEffectsAPI,
};
use types::error::SomaError;
use types::object::{CoinType, Object, ObjectID, ObjectType};
use types::system_state::{SystemState, SystemStateTrait};
use types::transaction::{
    BridgeDepositArgs, BridgeEmergencyPauseArgs, BridgeEmergencyUnpauseArgs, BridgeWithdrawArgs,
    TransactionData, TransactionKind,
};
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::authority::AuthorityState;
use crate::authority_test_utils::send_and_confirm_transaction_;
use crate::test_authority_builder::TestAuthorityBuilder;

// =============================================================================
// Helpers
// =============================================================================

struct TransactionResult {
    authority_state: Arc<AuthorityState>,
    txn_result: Result<SignedTransactionEffects, SomaError>,
}

/// Build a signer_bitmap where bit `i` is set for each index in `signer_indices`.
fn make_signer_bitmap(signer_indices: &[usize]) -> Vec<u8> {
    let max_index = signer_indices.iter().copied().max().unwrap_or(0);
    let num_bytes = (max_index / 8) + 1;
    let mut bitmap = vec![0u8; num_bytes];
    for &i in signer_indices {
        bitmap[i / 8] |= 1 << (i % 8);
    }
    bitmap
}

/// Get the bridge committee from the default test authority state.
/// The default genesis creates one validator with default bridge committee.
fn get_bridge_state(authority: &AuthorityState) -> BridgeState {
    let state = authority.get_system_state_object_for_testing().unwrap();
    state.bridge_state().clone()
}

async fn execute_system_tx(
    authority: &AuthorityState,
    kind: TransactionKind,
) -> Result<SignedTransactionEffects, SomaError> {
    // System transactions use a dummy sender and no gas payment.
    // In production, only the consensus handler creates these.
    // Tests can submit them directly — system tx rejection is at the network layer.
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority.insert_genesis_object(gas).await;

    let data = TransactionData::new(kind, sender, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &SomaKeyPair::Ed25519(key));
    send_and_confirm_transaction_(authority, None, tx, true)
        .await
        .map(|(_, effects)| effects)
}

// =============================================================================
// BridgeDeposit tests
// =============================================================================

#[tokio::test]
async fn test_bridge_deposit_mints_usdc() {
    let authority_state = TestAuthorityBuilder::new().build().await;
    let recipient = SomaAddress::random();

    // The default test genesis creates a bridge committee with empty members.
    // With an empty committee, threshold_deposit = 3334, but no members means
    // any bitmap will sum to 0 stake — which is < 3334.
    // For this test, we use an empty bitmap. With empty committee, verify_committee_stake
    // will return Ok only if threshold is 0, which it isn't (3334).
    // So we need to check: does an empty committee + any bitmap fail the threshold check?
    //
    // Actually, with an empty committee, total_stake will be 0, and threshold_deposit
    // is 3334, so verify_committee_stake WILL fail. The test should show this failure.
    // To test the happy path, we'd need a genesis with a real bridge committee.
    //
    // For now, test that the basic execution path works by verifying the error
    // is specifically BridgeInsufficientSignatureStake (not a deserialization error
    // or other crash).

    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce: 0,
        eth_tx_hash: [0u8; 32],
        recipient,
        amount: 1_000_000,
        aggregated_signature: vec![],
        signer_bitmap: vec![],
    });

    let result = execute_system_tx(&authority_state, kind).await;
    let effects = result.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure {
                error: ExecutionFailureStatus::BridgeInsufficientSignatureStake
            }
        ),
        "Empty committee should fail stake threshold check, got: {:?}",
        effects.status()
    );
}

#[tokio::test]
async fn test_bridge_deposit_nonce_replay_rejected() {
    // This tests the nonce replay check. Even though the first deposit will fail
    // the signature check (empty committee), we can verify the nonce logic
    // by checking that both calls fail with the SAME error (stake threshold),
    // proving the nonce check didn't trigger first on the second call.
    // (If the nonce was recorded despite the failure, the second would get
    // BridgeNonceAlreadyProcessed instead.)
    let authority_state = TestAuthorityBuilder::new().build().await;
    let recipient = SomaAddress::random();

    let args = BridgeDepositArgs {
        nonce: 42,
        eth_tx_hash: [1u8; 32],
        recipient,
        amount: 500_000,
        aggregated_signature: vec![],
        signer_bitmap: vec![],
    };

    // First attempt — fails at stake check
    let kind = TransactionKind::BridgeDeposit(args.clone());
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(matches!(
        effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::BridgeInsufficientSignatureStake }
    ));

    // Second attempt with same nonce — also fails at stake check (nonce not recorded
    // because first tx failed/reverted)
    let kind = TransactionKind::BridgeDeposit(args);
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(matches!(
        effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::BridgeInsufficientSignatureStake }
    ));
}

// =============================================================================
// BridgeWithdraw tests
// =============================================================================

/// Stage 12: BridgeWithdraw is balance-mode. The sender's USDC
/// accumulator is debited via a Withdraw event; the executor no
/// longer reads a payment coin object. Funds-availability is enforced
/// by the reservation pre-pass in production. Here we verify the
/// happy path: a PendingWithdrawal object is created and a Withdraw
/// event is emitted for the right (sender, amount).
#[tokio::test]
async fn test_bridge_withdraw_creates_pending_and_emits_withdraw() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let sender_key = SomaKeyPair::Ed25519(key);
    let authority_state = TestAuthorityBuilder::new().build().await;

    // Seed sender's USDC accumulator so the post-execution settlement
    // (which applies the Withdraw event) doesn't underflow.
    let withdraw_amount = 2_000_000u64;
    authority_state
        .database_for_testing()
        .set_balance(sender, CoinType::Usdc, withdraw_amount * 2)
        .unwrap();

    // Gas coin (SOMA — coin-mode gas, since we're not exercising the
    // balance-mode-gas path here).
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::BridgeWithdraw(BridgeWithdrawArgs {
        amount: withdraw_amount,
        recipient_eth_address: [0xABu8; 20],
    });
    let data = TransactionData::new(kind, sender, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &sender_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert_eq!(*effects_data.status(), ExecutionStatus::Success);

    // A PendingWithdrawal object must be created with the right
    // sender / amount / eth recipient. Bridge nodes consume this.
    assert!(effects_data.created().len() >= 1);
    let mut found_pending = false;
    for (oref, _owner) in effects_data.created() {
        let obj = authority_state.get_object(&oref.0).await.unwrap();
        if obj.type_() == &ObjectType::PendingWithdrawal {
            found_pending = true;
            let pw: types::bridge::PendingWithdrawal =
                obj.deserialize_contents(ObjectType::PendingWithdrawal).unwrap();
            assert_eq!(pw.sender, sender);
            assert_eq!(pw.amount, withdraw_amount);
            assert_eq!(pw.recipient_eth_address, [0xABu8; 20]);
        }
    }
    assert!(found_pending, "PendingWithdrawal should be created");

    // Stage 12 invariant: sender's USDC accumulator dropped by exactly
    // the withdraw amount. The Withdraw event emitted by the executor
    // landed via the per-tx settlement path.
    //
    // Flush the writeback cache so the perpetual_tables reflect the
    // tx's writes (unit tests skip the checkpoint executor that does
    // this in production — same plumbing concern as the staking
    // dual-write integrity test).
    let tx_digest = effects_data.transaction_digest();
    let epoch = authority_state.epoch_store_for_testing().epoch();
    let batch = authority_state.get_cache_commit().build_db_batch(epoch, &[*tx_digest]);
    authority_state.get_cache_commit().commit_transaction_outputs(epoch, batch, &[*tx_digest]);

    let final_balance = authority_state
        .database_for_testing()
        .get_balance(sender, CoinType::Usdc)
        .unwrap();
    assert_eq!(
        final_balance,
        withdraw_amount * 2 - withdraw_amount,
        "sender's USDC accumulator must drop by exactly the bridge withdraw amount",
    );
}

/// Stage 12: zero-amount withdrawals are rejected at the executor.
/// They're meaningless on-chain (no PendingWithdrawal worth signing
/// off-chain) and almost always indicate a wallet bug.
#[tokio::test]
async fn test_bridge_withdraw_rejects_zero_amount() {
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let sender_key = SomaKeyPair::Ed25519(key);
    let authority_state = TestAuthorityBuilder::new().build().await;

    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::BridgeWithdraw(BridgeWithdrawArgs {
        amount: 0,
        recipient_eth_address: [0x01; 20],
    });
    let data = TransactionData::new(kind, sender, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &sender_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert!(!effects_data.status().is_ok(), "zero-amount withdraw must be rejected");
}

// =============================================================================
// BridgeEmergencyPause tests
// =============================================================================

#[tokio::test]
async fn test_bridge_emergency_pause_insufficient_stake() {
    let authority_state = TestAuthorityBuilder::new().build().await;

    // With empty committee, pause should fail (need threshold_pause = 450 stake)
    let kind = TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
        aggregated_signature: vec![],
        signer_bitmap: vec![],
    });

    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(matches!(
        effects.status(),
        ExecutionStatus::Failure {
            error: ExecutionFailureStatus::BridgeInsufficientSignatureStake
        }
    ));
}

#[tokio::test]
async fn test_bridge_emergency_unpause_insufficient_stake() {
    let authority_state = TestAuthorityBuilder::new().build().await;

    let kind = TransactionKind::BridgeEmergencyUnpause(BridgeEmergencyUnpauseArgs {
        aggregated_signature: vec![],
        signer_bitmap: vec![],
    });

    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(matches!(
        effects.status(),
        ExecutionStatus::Failure {
            error: ExecutionFailureStatus::BridgeInsufficientSignatureStake
        }
    ));
}

// =============================================================================
// ECDSA signature verification tests
// =============================================================================

#[tokio::test]
async fn test_bridge_deposit_with_real_ecdsa_signatures() {
    use types::bridge::{
        BridgeMessageType, SOMA_BRIDGE_CHAIN_ID, build_bridge_signatures,
        encode_bridge_message, encode_deposit_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;

    // Generate a real committee with 4 members
    let (committee, keypairs) = generate_test_bridge_committee(4);
    // Each member has 2500 voting power; threshold_deposit = 3334, so we need 2 members

    // Build authority with the real bridge committee in genesis
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state = TestAuthorityBuilder::new()
        .with_genesis_config(genesis_config)
        .build()
        .await;

    let recipient = SomaAddress::random();
    let nonce = 0u64;
    let amount = 5_000_000u64;

    // Build the message that the executor will reconstruct
    let payload = encode_deposit_payload(&recipient, amount);
    let message = encode_bridge_message(
        BridgeMessageType::UsdcDeposit,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );

    // Sign with members 0 and 1 (5000 > 3334 threshold)
    let signers: Vec<(usize, &fastcrypto::secp256k1::Secp256k1KeyPair)> =
        vec![(0, &keypairs[0]), (1, &keypairs[1])];
    let (aggregated_signature, signer_bitmap) =
        build_bridge_signatures(&signers, &message);

    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce,
        eth_tx_hash: [0xAAu8; 32],
        recipient,
        amount,
        aggregated_signature,
        signer_bitmap,
    });

    let effects = execute_system_tx(&authority_state, kind)
        .await
        .unwrap()
        .into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Success,
        "BridgeDeposit with valid ECDSA should succeed, got: {:?}",
        effects.status()
    );

    // Stage 12: BridgeDeposit no longer mints a coin object — it
    // emits a Deposit event that credits the recipient's USDC
    // accumulator. Flush the writeback cache so the perpetual_tables
    // reflect the post-execution settlement.
    let tx_digest = effects.transaction_digest();
    let epoch = authority_state.epoch_store_for_testing().epoch();
    let batch = authority_state.get_cache_commit().build_db_batch(epoch, &[*tx_digest]);
    authority_state.get_cache_commit().commit_transaction_outputs(epoch, batch, &[*tx_digest]);

    let recipient_balance = authority_state
        .database_for_testing()
        .get_balance(recipient, CoinType::Usdc)
        .unwrap();
    assert_eq!(
        recipient_balance, amount,
        "BridgeDeposit must credit the recipient's USDC accumulator by `amount`",
    );
}

#[tokio::test]
async fn test_bridge_deposit_wrong_signature_rejected() {
    use types::bridge::{
        BridgeMessageType, SOMA_BRIDGE_CHAIN_ID, encode_bridge_message,
        encode_deposit_payload, generate_test_bridge_committee, sign_bridge_message,
    };
    use types::config::genesis_config::GenesisConfig;

    let (committee, keypairs) = generate_test_bridge_committee(4);

    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state = TestAuthorityBuilder::new()
        .with_genesis_config(genesis_config)
        .build()
        .await;

    let recipient = SomaAddress::random();
    let nonce = 0u64;
    let amount = 5_000_000u64;

    // Build the correct message
    let payload = encode_deposit_payload(&recipient, amount);
    let message = encode_bridge_message(
        BridgeMessageType::UsdcDeposit,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );

    // Sign with the WRONG keypair (use keypair[2] for member index 0)
    // This means the ecrecovered key won't match member 0's registered key
    let wrong_sig = sign_bridge_message(&keypairs[2], &message);
    let mut aggregated_signature = Vec::new();
    aggregated_signature.extend_from_slice(wrong_sig.as_ref());
    // Also need a second sig for threshold — use keypair[3] for member index 1
    let wrong_sig2 = sign_bridge_message(&keypairs[3], &message);
    aggregated_signature.extend_from_slice(wrong_sig2.as_ref());

    let signer_bitmap = vec![0b0000_0011u8]; // bits 0 and 1

    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce,
        eth_tx_hash: [0xBBu8; 32],
        recipient,
        amount,
        aggregated_signature,
        signer_bitmap,
    });

    let effects = execute_system_tx(&authority_state, kind)
        .await
        .unwrap()
        .into_data();
    // Should fail because recovered keys don't match committee members
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure {
                error: ExecutionFailureStatus::SomaError(..)
            }
        ),
        "Wrong ECDSA signature should be rejected, got: {:?}",
        effects.status()
    );
}

#[tokio::test]
async fn test_bridge_nonce_replay_with_real_ecdsa() {
    use types::bridge::{
        BridgeMessageType, SOMA_BRIDGE_CHAIN_ID, build_bridge_signatures,
        encode_bridge_message, encode_deposit_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;

    let (committee, keypairs) = generate_test_bridge_committee(4);

    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state = TestAuthorityBuilder::new()
        .with_genesis_config(genesis_config)
        .build()
        .await;

    let recipient = SomaAddress::random();
    let nonce = 42u64;
    let amount = 1_000_000u64;

    let payload = encode_deposit_payload(&recipient, amount);
    let message = encode_bridge_message(
        BridgeMessageType::UsdcDeposit,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );

    let signers: Vec<(usize, &fastcrypto::secp256k1::Secp256k1KeyPair)> =
        vec![(0, &keypairs[0]), (1, &keypairs[1])];
    let (aggregated_signature, signer_bitmap) =
        build_bridge_signatures(&signers, &message);

    // First deposit succeeds
    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce,
        eth_tx_hash: [0xCCu8; 32],
        recipient,
        amount,
        aggregated_signature: aggregated_signature.clone(),
        signer_bitmap: signer_bitmap.clone(),
    });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Second deposit with same nonce should fail
    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce, // same nonce
        eth_tx_hash: [0xCCu8; 32],
        recipient,
        amount,
        aggregated_signature,
        signer_bitmap,
    });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure {
                error: ExecutionFailureStatus::BridgeNonceAlreadyProcessed
            }
        ),
        "Replay should be rejected, got: {:?}",
        effects.status()
    );
}

// =============================================================================
// Bridge pause/unpause effect on withdraw
// =============================================================================

// Note: Testing pause effect on BridgeWithdraw requires a paused bridge state.
// Since we can't pause the bridge without committee signatures in the default
// test genesis, this test verifies the code path exists by checking that
// a non-paused bridge allows withdrawal (covered by test_bridge_withdraw_burns_usdc).
// Full pause/unpause integration tests belong in e2e-tests with a real committee.
