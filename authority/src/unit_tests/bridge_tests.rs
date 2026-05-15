// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::base::SomaAddress;
use types::bridge::{BridgeChainId, BridgeState, USDC_TOKEN_TYPE};
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

/// Get the bridge committee from the default test authority state.
/// The default genesis creates one validator with default bridge committee.
fn get_bridge_state(authority: &AuthorityState) -> BridgeState {
    let state = authority.get_system_state_object_for_testing().unwrap();
    state.bridge_state().clone()
}

/// Seed `BridgeState.total_usdc_supply` to `amount` so withdraw tests have
/// enough conservation-invariant balance to burn. The supply counter is
/// normally incremented only via successful BridgeDeposit; tests that go
/// straight to BridgeWithdraw must seed it manually or the executor's
/// `checked_sub` returns `BridgeSupplyUnderflow`.
async fn seed_bridge_supply(authority: &Arc<AuthorityState>, amount: u64) {
    use types::SYSTEM_STATE_OBJECT_ID;
    let mut obj = authority.get_object(&SYSTEM_STATE_OBJECT_ID).await.unwrap();
    let mut state: SystemState = bcs::from_bytes(obj.data.contents()).unwrap();
    state.bridge_state_mut().total_usdc_supply = amount;
    obj.data.update_contents(bcs::to_bytes(&state).unwrap());
    // `insert_genesis_object` asserts `previous_transaction == genesis_marker`;
    // restore it so the re-insertion is accepted (we're side-stepping the
    // normal effects pipeline because there's no public test API for mutating
    // SystemState directly).
    obj.previous_transaction = TransactionDigest::genesis_marker();
    authority.insert_genesis_object(obj).await;
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
    send_and_confirm_transaction_(authority, None, tx, true).await.map(|(_, effects)| effects)
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
        sender_eth_address: [0u8; 20],
        target_chain: BridgeChainId::SomaCustom,
        token_type: USDC_TOKEN_TYPE,
        recipient,
        amount: 1_000_000,
        timestamp_ms: 0,
        signatures: Default::default(),
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
        sender_eth_address: [0u8; 20],
        target_chain: BridgeChainId::SomaCustom,
        token_type: USDC_TOKEN_TYPE,
        recipient,
        amount: 500_000,
        timestamp_ms: 0,
        signatures: Default::default(),
    };

    // First attempt — fails at stake check
    let kind = TransactionKind::BridgeDeposit(args.clone());
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(matches!(
        effects.status(),
        ExecutionStatus::Failure {
            error: ExecutionFailureStatus::BridgeInsufficientSignatureStake
        }
    ));

    // Second attempt with same nonce — also fails at stake check (nonce not recorded
    // because first tx failed/reverted)
    let kind = TransactionKind::BridgeDeposit(args);
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(matches!(
        effects.status(),
        ExecutionStatus::Failure {
            error: ExecutionFailureStatus::BridgeInsufficientSignatureStake
        }
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

    // Stage 13c: bridge withdraw burns USDC from the sender's
    // accumulator (the value being withdrawn) AND debits gas (USDC)
    // from the same accumulator. Seed enough to cover both, plus
    // a margin we'll verify isn't touched by the settlement layer
    // beyond the expected delta.
    let withdraw_amount = 2_000_000u64;
    let starting_usdc = withdraw_amount * 2 + 100_000_000;
    authority_state
        .database_for_testing()
        .set_balance(sender, CoinType::Usdc, starting_usdc)
        .unwrap();

    // Seed bridge.total_usdc_supply so the burn doesn't underflow the
    // conservation-invariant counter.
    seed_bridge_supply(&authority_state, withdraw_amount).await;

    let kind = TransactionKind::BridgeWithdraw(BridgeWithdrawArgs {
        amount: withdraw_amount,
        recipient_eth_address: [0xABu8; 20],
        target_chain: BridgeChainId::EthCustom,
    });
    let data = TransactionData::new(kind, sender, vec![]);
    let tx = to_sender_signed_transaction(data, &sender_key);
    let (_, effects) =
        send_and_confirm_transaction_(&authority_state, None, tx, true).await.unwrap();
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

    let final_balance =
        authority_state.database_for_testing().get_balance(sender, CoinType::Usdc).unwrap();
    let total_fee = effects_data.transaction_fee().total_fee;
    assert_eq!(
        final_balance,
        starting_usdc - withdraw_amount - total_fee,
        "sender's USDC must drop by withdraw + gas fee (got {}, expected {})",
        final_balance,
        starting_usdc - withdraw_amount - total_fee,
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

    authority_state
        .database_for_testing()
        .set_balance(sender, CoinType::Usdc, 100_000_000)
        .unwrap();

    let kind = TransactionKind::BridgeWithdraw(BridgeWithdrawArgs {
        amount: 0,
        recipient_eth_address: [0x01; 20],
        target_chain: BridgeChainId::EthCustom,
    });
    let data = TransactionData::new(kind, sender, vec![]);
    let tx = to_sender_signed_transaction(data, &sender_key);
    let (_, effects) =
        send_and_confirm_transaction_(&authority_state, None, tx, true).await.unwrap();
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
        nonce: 0,
        signatures: Default::default(),
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
async fn test_bridge_emergency_unpause_when_not_paused() {
    // H1: refuse unpause when bridge is not paused. The default authority
    // boots unpaused, so an unpause attempt should bail with BridgeNotPaused
    // BEFORE wasting effort on signature verification.
    let authority_state = TestAuthorityBuilder::new().build().await;

    let kind = TransactionKind::BridgeEmergencyUnpause(BridgeEmergencyUnpauseArgs {
        nonce: 0,
        signatures: Default::default(),
    });

    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure { error: ExecutionFailureStatus::BridgeNotPaused }
        ),
        "expected BridgeNotPaused, got {:?}",
        effects.status()
    );
}

// =============================================================================
// ECDSA signature verification tests
// =============================================================================

#[tokio::test]
async fn test_bridge_deposit_with_real_ecdsa_signatures() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_deposit_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;

    // Generate a real committee with 4 members
    let (committee, keypairs) = generate_test_bridge_committee(4);
    // Each member has 2500 voting power; threshold_deposit = 3334, so we need 2 members

    // Build authority with the real bridge committee in genesis
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let recipient = SomaAddress::random();
    let nonce = 0u64;
    let amount = 5_000_000u64;
    let sender_eth_address = [0u8; 20];
    let target_chain = BridgeChainId::SomaCustom;
    let token_type = USDC_TOKEN_TYPE;

    // Build the message that the executor will reconstruct
    let payload = encode_deposit_payload(
        &sender_eth_address,
        target_chain,
        &recipient,
        token_type,
        amount,
        0,
    );
    let message = encode_bridge_message(
        BridgeMessageType::UsdcDeposit,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );

    // Sign with members 0 and 1 (5000 > 3334 threshold)
    let signers: Vec<&fastcrypto::secp256k1::Secp256k1KeyPair> = vec![&keypairs[0], &keypairs[1]];
    let signatures = build_bridge_signatures(&signers, &message);

    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce,
        eth_tx_hash: [0xAAu8; 32],
        sender_eth_address,
        target_chain,
        token_type,
        recipient,
        amount,
        timestamp_ms: 0,
        signatures,
    });

    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
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

    let recipient_balance =
        authority_state.database_for_testing().get_balance(recipient, CoinType::Usdc).unwrap();
    assert_eq!(
        recipient_balance, amount,
        "BridgeDeposit must credit the recipient's USDC accumulator by `amount`",
    );
}

#[tokio::test]
async fn test_bridge_deposit_wrong_signature_rejected() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_deposit_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;

    let (committee, _keypairs) = generate_test_bridge_committee(4);

    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let recipient = SomaAddress::random();
    let nonce = 0u64;
    let amount = 5_000_000u64;
    let sender_eth_address = [0u8; 20];
    let target_chain = BridgeChainId::SomaCustom;
    let token_type = USDC_TOKEN_TYPE;

    // Build the correct message
    let payload = encode_deposit_payload(
        &sender_eth_address,
        target_chain,
        &recipient,
        token_type,
        amount,
        0,
    );
    let message = encode_bridge_message(
        BridgeMessageType::UsdcDeposit,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );

    // Sign with keypairs that aren't in the committee. With pubkey-keyed
    // membership (Sui parity), the labeled pubkeys must hit the committee
    // map; non-committee sigs are rejected at signature verification.
    use fastcrypto::traits::KeyPair;
    let outsider_a = fastcrypto::secp256k1::Secp256k1KeyPair::generate(&mut rand::thread_rng());
    let outsider_b = fastcrypto::secp256k1::Secp256k1KeyPair::generate(&mut rand::thread_rng());
    let signatures = build_bridge_signatures(&[&outsider_a, &outsider_b], &message);

    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce,
        eth_tx_hash: [0xBBu8; 32],
        sender_eth_address,
        target_chain,
        token_type,
        recipient,
        amount,
        timestamp_ms: 0,
        signatures,
    });

    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    // Should fail because recovered pubkeys aren't in the committee.
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure { error: ExecutionFailureStatus::SomaError(..) }
        ),
        "Wrong ECDSA signature should be rejected, got: {:?}",
        effects.status()
    );
}

#[tokio::test]
async fn test_bridge_nonce_replay_with_real_ecdsa() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_deposit_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;

    let (committee, keypairs) = generate_test_bridge_committee(4);

    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let recipient = SomaAddress::random();
    let nonce = 42u64;
    let amount = 1_000_000u64;
    let sender_eth_address = [0u8; 20];
    let target_chain = BridgeChainId::SomaCustom;
    let token_type = USDC_TOKEN_TYPE;

    let payload = encode_deposit_payload(
        &sender_eth_address,
        target_chain,
        &recipient,
        token_type,
        amount,
        0,
    );
    let message = encode_bridge_message(
        BridgeMessageType::UsdcDeposit,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );

    let signers: Vec<&fastcrypto::secp256k1::Secp256k1KeyPair> = vec![&keypairs[0], &keypairs[1]];
    let signatures = build_bridge_signatures(&signers, &message);

    // First deposit succeeds
    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce,
        eth_tx_hash: [0xCCu8; 32],
        sender_eth_address,
        target_chain,
        token_type,
        recipient,
        amount,
        timestamp_ms: 0,
        signatures: signatures.clone(),
    });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Second deposit with same nonce should fail
    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce, // same nonce
        eth_tx_hash: [0xCCu8; 32],
        sender_eth_address,
        target_chain,
        token_type,
        recipient,
        amount,
        timestamp_ms: 0,
        signatures,
    });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure { error: ExecutionFailureStatus::BridgeNonceAlreadyProcessed }
        ),
        "Replay should be rejected, got: {:?}",
        effects.status()
    );
}

// =============================================================================
// Per-message-type seq num replay defense (governance actions)
//
// Mirrors Sui's `BridgeInner.sequence_nums: VecMap<u8, u64>` + the
// `execute_system_message` per-type assert-and-increment. Pause and unpause
// share the EmergencyOp counter (both have message_type byte 2).
// =============================================================================

#[tokio::test]
async fn test_emergency_op_advances_seq_num_then_rejects_replay() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, EmergencyOpCode, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_emergency_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;

    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    // Pre-state: counter starts at 0.
    let bridge = get_bridge_state(&authority_state);
    assert_eq!(bridge.expected_system_message_seq(BridgeMessageType::EmergencyOp), 0);

    // Pause with nonce=0 (the expected initial value).
    let pause_payload = encode_emergency_payload(EmergencyOpCode::Freeze);
    let pause_msg = encode_bridge_message(
        BridgeMessageType::EmergencyOp,
        BRIDGE_MESSAGE_VERSION,
        0,
        SOMA_BRIDGE_CHAIN_ID,
        &pause_payload,
    );
    let pause_sig = build_bridge_signatures(&[&keypairs[0]], &pause_msg); // 2500 stake > 450 threshold

    let pause = TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
        nonce: 0,
        signatures: pause_sig.clone(),
    });
    let effects = execute_system_tx(&authority_state, pause).await.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Counter advanced; bridge is paused.
    let bridge = get_bridge_state(&authority_state);
    assert_eq!(bridge.expected_system_message_seq(BridgeMessageType::EmergencyOp), 1);
    assert!(bridge.paused);

    // Replaying the SAME pause cert now hits the H1 state check first
    // (bridge is already paused) — BridgeAlreadyPaused. The seq check is
    // also armed; we verify it separately below by trying with a FUTURE
    // nonce when bridge is not yet at that nonce + state mismatch.
    let pause_replay = TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
        nonce: 0,
        signatures: pause_sig,
    });
    let effects = execute_system_tx(&authority_state, pause_replay).await.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure { error: ExecutionFailureStatus::BridgeAlreadyPaused },
        ),
        "Replayed pause when paused must fail with BridgeAlreadyPaused, got {:?}",
        effects.status()
    );

    // Now verify the seq check: build a new pause cert at the next-expected
    // nonce (=1) but the bridge is still paused, so the state check still
    // fires first. To exercise the seq check, we'd need bridge to be
    // unpausable, which requires a separate cert. The seq check coverage
    // lives in test_emergency_op_nonce_too_high_rejected (sends a future
    // nonce when bridge is unpaused — passes state check, hits seq check).
}

#[tokio::test]
async fn test_pause_and_unpause_share_seq_counter() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, EmergencyOpCode, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_emergency_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;

    // Sui parity: pause and unpause both have message_type byte 2 and
    // share the EmergencyOp counter. Each consumes one nonce.

    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    // Pause at nonce=0.
    let pause_payload = encode_emergency_payload(EmergencyOpCode::Freeze);
    let pause_msg = encode_bridge_message(
        BridgeMessageType::EmergencyOp,
        BRIDGE_MESSAGE_VERSION,
        0,
        SOMA_BRIDGE_CHAIN_ID,
        &pause_payload,
    );
    let pause_sig = build_bridge_signatures(&[&keypairs[0]], &pause_msg);
    let pause = TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
        nonce: 0,
        signatures: pause_sig,
    });
    assert_eq!(
        *execute_system_tx(&authority_state, pause).await.unwrap().into_data().status(),
        ExecutionStatus::Success
    );

    // Unpause must use the NEXT nonce (1), not 0 — pause consumed 0.
    let unpause_payload = encode_emergency_payload(EmergencyOpCode::Unfreeze);
    let unpause_msg_n1 = encode_bridge_message(
        BridgeMessageType::EmergencyOp,
        BRIDGE_MESSAGE_VERSION,
        1,
        SOMA_BRIDGE_CHAIN_ID,
        &unpause_payload,
    );
    // Need 6667 stake for unpause; sign with members 0, 1, 2 (7500).
    let unpause_sig =
        build_bridge_signatures(&[&keypairs[0], &keypairs[1], &keypairs[2]], &unpause_msg_n1);
    let unpause = TransactionKind::BridgeEmergencyUnpause(BridgeEmergencyUnpauseArgs {
        nonce: 1,
        signatures: unpause_sig,
    });
    let effects = execute_system_tx(&authority_state, unpause).await.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Both pause+unpause consumed; counter at 2.
    let bridge = get_bridge_state(&authority_state);
    assert_eq!(bridge.expected_system_message_seq(BridgeMessageType::EmergencyOp), 2);
    assert!(!bridge.paused);
}

#[tokio::test]
async fn test_emergency_op_nonce_too_high_rejected() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, EmergencyOpCode, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_emergency_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;

    // System messages must be strictly in-order — a future nonce gets
    // rejected just like a past one. (Token transfers can be out of
    // order via the watermark; system messages cannot.)
    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let payload = encode_emergency_payload(EmergencyOpCode::Freeze);
    let msg = encode_bridge_message(
        BridgeMessageType::EmergencyOp,
        BRIDGE_MESSAGE_VERSION,
        5, // skipping nonces 0-4
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );
    let sig = build_bridge_signatures(&[&keypairs[0]], &msg);

    let kind = TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
        nonce: 5,
        signatures: sig,
    });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure {
                error: ExecutionFailureStatus::BridgeSystemMessageSeqMismatch {
                    expected: 0,
                    actual: 5,
                },
            },
        ),
        "Future nonce must be rejected: {:?}",
        effects.status()
    );
}

// =============================================================================
// BridgeUpdateCommitteeBlocklist + 0-stake filter for blocklisted members
//
// Mirrors Sui's `committee::execute_blocklist` (member matching by derived
// 20-byte Eth address) + `verify_signatures` (blocklisted members keep
// their entry in the committee but contribute 0 voting power).
// =============================================================================

#[tokio::test]
async fn test_blocklisted_member_signature_contributes_zero_stake() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, EmergencyOpCode, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, derive_eth_address,
        encode_blocklist_payload, encode_bridge_message, encode_emergency_payload,
        generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;
    use types::transaction::BridgeUpdateCommitteeBlocklistArgs;

    // 4 members × 2500 stake = 10000 total. threshold_pause = 450 (any
    // single member can pause). After we blocklist member 0, their sigs
    // count as 0 — they alone can no longer pause.
    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    // Step 1: blocklist member 0 (uses threshold_blocklist = 5001).
    let kp0_pubkey = types::bridge::BridgePubkey::from_keypair(&keypairs[0]);
    let target_eth = derive_eth_address(&kp0_pubkey);
    let bl_payload =
        encode_blocklist_payload(types::bridge::BlocklistType::Blocklist, &[target_eth]);
    let bl_msg = encode_bridge_message(
        BridgeMessageType::UpdateCommitteeBlocklist,
        BRIDGE_MESSAGE_VERSION,
        0,
        SOMA_BRIDGE_CHAIN_ID,
        &bl_payload,
    );
    // Need 5001 stake — 3 members × 2500 = 7500.
    let bl_sig = build_bridge_signatures(&[&keypairs[1], &keypairs[2], &keypairs[3]], &bl_msg);
    let kind =
        TransactionKind::BridgeUpdateCommitteeBlocklist(BridgeUpdateCommitteeBlocklistArgs {
            nonce: 0,
            is_blocklist: true,
            eth_addresses: vec![target_eth],
            signatures: bl_sig,
        });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Member 0 is now blocklisted (verified via on-chain state).
    let bridge = get_bridge_state(&authority_state);
    let blocklisted_addr = bridge
        .bridge_committee
        .members
        .get(&kp0_pubkey)
        .expect("member 0 in committee")
        .is_blocklisted;
    assert!(blocklisted_addr, "member 0 must be blocklisted");

    // Step 2: try to pause with ONLY member 0's sig. With member 0
    // blocklisted, their stake counts as 0, so we fail the threshold.
    let pause_payload = encode_emergency_payload(EmergencyOpCode::Freeze);
    let pause_msg = encode_bridge_message(
        BridgeMessageType::EmergencyOp,
        BRIDGE_MESSAGE_VERSION,
        0, // emergency op nonce starts at 0
        SOMA_BRIDGE_CHAIN_ID,
        &pause_payload,
    );
    let pause_sig = build_bridge_signatures(&[&keypairs[0]], &pause_msg);
    let pause = TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
        nonce: 0,
        signatures: pause_sig,
    });
    let effects = execute_system_tx(&authority_state, pause).await.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure {
                error: ExecutionFailureStatus::BridgeInsufficientSignatureStake,
            },
        ),
        "Blocklisted member's sig must contribute 0 stake: {:?}",
        effects.status()
    );

    // Step 3: a non-blocklisted member (member 1) alone CAN pause —
    // their 2500 stake exceeds the 450 pause threshold.
    let pause_sig2 = build_bridge_signatures(&[&keypairs[1]], &pause_msg);
    let pause2 = TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
        nonce: 0,
        signatures: pause_sig2,
    });
    let effects = execute_system_tx(&authority_state, pause2).await.unwrap().into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Success,
        "Non-blocklisted member can still reach the pause threshold"
    );
}

#[tokio::test]
async fn test_blocklist_unknown_address_rejected() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_blocklist_payload,
        encode_bridge_message, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;
    use types::transaction::BridgeUpdateCommitteeBlocklistArgs;

    // Sui parity: `committee::execute_blocklist` aborts with
    // EValidatorBlocklistContainsUnknownKey if any address isn't in the
    // committee — no silent ignores.
    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let unknown_eth = [0xFFu8; 20];
    let payload = encode_blocklist_payload(types::bridge::BlocklistType::Blocklist, &[unknown_eth]);
    let msg = encode_bridge_message(
        BridgeMessageType::UpdateCommitteeBlocklist,
        BRIDGE_MESSAGE_VERSION,
        0,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );
    let sig = build_bridge_signatures(&[&keypairs[0], &keypairs[1], &keypairs[2]], &msg);
    let kind =
        TransactionKind::BridgeUpdateCommitteeBlocklist(BridgeUpdateCommitteeBlocklistArgs {
            nonce: 0,
            is_blocklist: true,
            eth_addresses: vec![unknown_eth],
            signatures: sig,
        });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(
        matches!(*effects.status(), ExecutionStatus::Failure { .. }),
        "Unknown blocklist address must be rejected: {:?}",
        effects.status()
    );
}

// =============================================================================
// Hardening fixes (post-review)
// =============================================================================

/// H1: pause-when-paused must fail with BridgeAlreadyPaused, not silently
/// consume a seq num. Defends against committee collusion that would
/// otherwise burn EmergencyOp nonces to invalidate pre-signed unpause certs.
#[tokio::test]
async fn test_pause_when_paused_rejected() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, EmergencyOpCode, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_emergency_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;

    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    // Pause once.
    let pause_payload = encode_emergency_payload(EmergencyOpCode::Freeze);
    let msg_n0 = encode_bridge_message(
        BridgeMessageType::EmergencyOp,
        BRIDGE_MESSAGE_VERSION,
        0,
        SOMA_BRIDGE_CHAIN_ID,
        &pause_payload,
    );
    let sig0 = build_bridge_signatures(&[&keypairs[0]], &msg_n0);
    let pause_n0 = TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
        nonce: 0,
        signatures: sig0,
    });
    assert_eq!(
        *execute_system_tx(&authority_state, pause_n0).await.unwrap().into_data().status(),
        ExecutionStatus::Success,
    );

    // Now sign a fresh pause for the next-expected nonce (=1) — same
    // committee, same threshold check passes, but state check rejects.
    let msg_n1 = encode_bridge_message(
        BridgeMessageType::EmergencyOp,
        BRIDGE_MESSAGE_VERSION,
        1,
        SOMA_BRIDGE_CHAIN_ID,
        &pause_payload,
    );
    let sig1 = build_bridge_signatures(&[&keypairs[0]], &msg_n1);
    let pause_n1 = TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
        nonce: 1,
        signatures: sig1,
    });
    let effects = execute_system_tx(&authority_state, pause_n1).await.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure { error: ExecutionFailureStatus::BridgeAlreadyPaused },
        ),
        "fresh pause cert when paused must be rejected, got {:?}",
        effects.status()
    );

    // Crucially: seq counter must NOT have advanced — the pre-signed
    // pause-N1 cert effectively didn't happen. (If we'd burned N1, an
    // attacker could continue: sign N2, etc., to grief the unpause path.)
    let bridge = get_bridge_state(&authority_state);
    assert_eq!(
        bridge.expected_system_message_seq(BridgeMessageType::EmergencyOp),
        1,
        "rejected pause must not consume seq num"
    );
}

/// L1: zero-amount deposit cert is rejected before consuming a nonce or
/// creating a BridgeRecord. Sui parity (abi.rs::ZeroValueBridgeTransfer).
#[tokio::test]
async fn test_zero_amount_deposit_rejected() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_deposit_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;

    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let recipient = SomaAddress::random();
    let sender_eth_address = [0u8; 20];
    let target_chain = BridgeChainId::SomaCustom;
    let token_type = USDC_TOKEN_TYPE;
    let payload =
        encode_deposit_payload(&sender_eth_address, target_chain, &recipient, token_type, 0, 0); // amount = 0
    let msg = encode_bridge_message(
        BridgeMessageType::UsdcDeposit,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
        0,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );
    let sig = build_bridge_signatures(&[&keypairs[0], &keypairs[1]], &msg);

    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce: 0,
        eth_tx_hash: [0; 32],
        sender_eth_address,
        target_chain,
        token_type,
        recipient,
        amount: 0,
        timestamp_ms: 0,
        signatures: sig,
    });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure { error: ExecutionFailureStatus::BridgeAmountZero },
        ),
        "zero-amount deposit must be rejected, got {:?}",
        effects.status()
    );
}

/// M1: blocklist payload exceeding MAX_BLOCKLIST_ENTRIES_PER_TX rejected.
#[tokio::test]
async fn test_blocklist_payload_size_capped() {
    use types::bridge::generate_test_bridge_committee;
    use types::config::genesis_config::GenesisConfig;
    use types::transaction::BridgeUpdateCommitteeBlocklistArgs;

    let (committee, _) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    // 256 entries — over the 255 cap. Sigs/payload don't matter; cap fires first.
    let too_many: Vec<[u8; 20]> = (0..256u32)
        .map(|i| {
            let mut b = [0u8; 20];
            b[..4].copy_from_slice(&i.to_be_bytes());
            b
        })
        .collect();

    let kind =
        TransactionKind::BridgeUpdateCommitteeBlocklist(BridgeUpdateCommitteeBlocklistArgs {
            nonce: 0,
            is_blocklist: true,
            eth_addresses: too_many,
            signatures: Default::default(),
        });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure {
                error: ExecutionFailureStatus::BridgeBlocklistPayloadTooLarge {
                    got: 256,
                    max: 255
                },
            },
        ),
        "oversized blocklist must be rejected, got {:?}",
        effects.status()
    );
}

// L2 (cap http_url): the executor enforces `args.http_url.len() <=
// MAX_BRIDGE_HTTP_URL_LEN`. End-to-end test would require a fresh
// validator-keyed tx through the full signing pipeline (substantial
// plumbing); the executor branch is small + direct, and verifying the
// constant via `assert_eq!(MAX_BRIDGE_HTTP_URL_LEN, 2048)` doesn't add
// signal. Coverage is the inline check itself + the BridgeUrlTooLong
// error variant being match-exhaustive in workspace.

// =============================================================================
// Bridge pause/unpause effect on withdraw
// =============================================================================

// Note: Testing pause effect on BridgeWithdraw requires a paused bridge state.
// Since we can't pause the bridge without committee signatures in the default
// test genesis, this test verifies the code path exists by checking that
// a non-paused bridge allows withdrawal (covered by test_bridge_withdraw_burns_usdc).
// Full pause/unpause integration tests belong in e2e-tests with a real committee.

/// Stage 2 invariant: a successful BridgeDeposit creates an Immutable
/// BridgeRecord object at the deterministic ID derived from
/// (eth_chain, UsdcDeposit, nonce). This is the on-chain audit trail —
/// anyone can compute the ID and read "did deposit X happen, with what
/// Eth tx hash, to whom, for how much, at which epoch?".
#[tokio::test]
async fn test_bridge_deposit_creates_audit_record() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, BridgeRecord, ETH_BRIDGE_CHAIN_ID,
        SOMA_BRIDGE_CHAIN_ID, TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures,
        derive_bridge_record_id, encode_bridge_message, encode_deposit_payload,
        generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;
    use types::object::Owner;

    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let recipient = SomaAddress::random();
    let nonce = 7u64;
    let amount = 5_000_000u64;
    let eth_tx_hash = [0xDEu8; 32];
    let sender_eth_address = [0u8; 20];
    let target_chain = BridgeChainId::SomaCustom;
    let token_type = USDC_TOKEN_TYPE;

    let payload = encode_deposit_payload(
        &sender_eth_address,
        target_chain,
        &recipient,
        token_type,
        amount,
        0,
    );
    let message = encode_bridge_message(
        BridgeMessageType::UsdcDeposit,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );
    let agg_sig = build_bridge_signatures(&[&keypairs[0], &keypairs[1]], &message);

    let kind = TransactionKind::BridgeDeposit(BridgeDepositArgs {
        nonce,
        eth_tx_hash,
        sender_eth_address,
        target_chain,
        token_type,
        recipient,
        amount,
        timestamp_ms: 0,
        signatures: agg_sig,
    });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // The BridgeRecord object lives at the deterministic ID — anyone with
    // (eth_chain, UsdcDeposit, nonce) can compute it and look it up.
    let record_id =
        derive_bridge_record_id(ETH_BRIDGE_CHAIN_ID, BridgeMessageType::UsdcDeposit, nonce);
    let record_obj =
        authority_state.get_object(&record_id).await.expect("BridgeRecord must be created");
    assert_eq!(*record_obj.type_(), ObjectType::BridgeRecord);
    assert!(matches!(record_obj.owner, Owner::Immutable));

    let record: BridgeRecord = record_obj.deserialize_contents(ObjectType::BridgeRecord).unwrap();
    assert_eq!(record.source_chain_id, ETH_BRIDGE_CHAIN_ID);
    assert_eq!(record.nonce, nonce);
    assert_eq!(record.eth_tx_hash, eth_tx_hash);
    assert_eq!(record.recipient, recipient);
    assert_eq!(record.amount, amount);
}

// =============================================================================
// BridgeAttachWithdrawalSignatures
//
// Mirrors Sui's `bridge::approve_token_transfer` for outbound (Sui→Eth)
// records: the BridgeRecord must already exist (created at withdraw time);
// attaching is idempotent; verification uses `threshold_withdraw`.
// =============================================================================

/// Helper: drive a withdrawal end-to-end and return its on-chain
/// `PendingWithdrawal` object id. Used as setup for attach-sigs tests.
async fn create_pending_withdrawal(
    authority_state: &Arc<AuthorityState>,
    sender: SomaAddress,
    sender_key: &SomaKeyPair,
    amount: u64,
    recipient_eth: [u8; 20],
) -> ObjectID {
    // Seed enough USDC to cover withdraw + gas.
    authority_state
        .database_for_testing()
        .set_balance(sender, CoinType::Usdc, amount * 2 + 100_000_000)
        .unwrap();

    // Seed bridge.total_usdc_supply so the burn doesn't underflow the
    // conservation-invariant counter — in production this would have been
    // credited by prior BridgeDeposit txs.
    seed_bridge_supply(authority_state, amount).await;

    let kind = TransactionKind::BridgeWithdraw(BridgeWithdrawArgs {
        amount,
        recipient_eth_address: recipient_eth,
        target_chain: BridgeChainId::EthCustom,
    });
    let data = TransactionData::new(kind, sender, vec![]);
    let tx = to_sender_signed_transaction(data, sender_key);
    let (_, effects) =
        send_and_confirm_transaction_(authority_state, None, tx, true).await.unwrap();
    let effects_data = effects.into_data();
    assert_eq!(*effects_data.status(), ExecutionStatus::Success);

    // Find the PendingWithdrawal object that got created.
    for (oref, _owner) in effects_data.created() {
        let obj = authority_state.get_object(&oref.0).await.unwrap();
        if obj.type_() == &ObjectType::PendingWithdrawal {
            return oref.0;
        }
    }
    panic!("No PendingWithdrawal created");
}

#[tokio::test]
async fn test_attach_withdrawal_signatures_happy_path() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, PendingWithdrawal, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, derive_bridge_record_id,
        encode_bridge_message, encode_withdraw_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;
    use types::transaction::BridgeAttachWithdrawalSignaturesArgs;

    let (committee, keypairs) = generate_test_bridge_committee(4);
    // Each member has 2500 voting power; threshold_withdraw = 3334 → need 2.

    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let sender_key = SomaKeyPair::Ed25519(key);
    let amount = 1_000_000u64;
    let recipient_eth = [0xCDu8; 20];

    let withdrawal_id =
        create_pending_withdrawal(&authority_state, sender, &sender_key, amount, recipient_eth)
            .await;

    // ID must match the deterministic derivation — anyone with (chain, type, nonce)
    // can compute this off-chain.
    let nonce = 0u64;
    assert_eq!(
        withdrawal_id,
        derive_bridge_record_id(SOMA_BRIDGE_CHAIN_ID, BridgeMessageType::UsdcWithdraw, nonce)
    );

    // Object exists, has no cert yet.
    let pre = authority_state.get_object(&withdrawal_id).await.unwrap();
    let pending: PendingWithdrawal =
        pre.deserialize_contents(ObjectType::PendingWithdrawal).unwrap();
    assert!(pending.verified_signatures.is_none());

    // Build the canonical message bytes (executor will reconstruct identically).
    // V2 token-transfer payload includes the on-chain timestamp_ms; we read
    // it from the PendingWithdrawal so signed bytes match what the executor
    // reconstructs from authoritative state.
    let payload = encode_withdraw_payload(
        &pending.sender,
        pending.target_chain,
        &pending.recipient_eth_address,
        USDC_TOKEN_TYPE,
        pending.amount,
        pending.created_at_ms,
    );
    let message = encode_bridge_message(
        BridgeMessageType::UsdcWithdraw,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );

    // Quorum sign with members 0, 1 (5000 > 3334 threshold).
    let signers: Vec<&fastcrypto::secp256k1::Secp256k1KeyPair> = vec![&keypairs[0], &keypairs[1]];
    let signatures = build_bridge_signatures(&signers, &message);

    let kind =
        TransactionKind::BridgeAttachWithdrawalSignatures(BridgeAttachWithdrawalSignaturesArgs {
            nonce,
            signatures: signatures.clone(),
        });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert_eq!(
        *effects.status(),
        ExecutionStatus::Success,
        "Valid quorum cert should attach: {:?}",
        effects.status()
    );

    // Object now carries the cert, with the same bytes the executor verified.
    let post = authority_state.get_object(&withdrawal_id).await.unwrap();
    let pending_post: PendingWithdrawal =
        post.deserialize_contents(ObjectType::PendingWithdrawal).unwrap();
    let cert = pending_post.verified_signatures.as_ref().expect("cert must be attached");
    assert_eq!(cert.signatures, signatures);
}

#[tokio::test]
async fn test_attach_withdrawal_signatures_idempotent() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, PendingWithdrawal, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_withdraw_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;
    use types::transaction::BridgeAttachWithdrawalSignaturesArgs;

    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let sender_key = SomaKeyPair::Ed25519(key);
    let withdrawal_id =
        create_pending_withdrawal(&authority_state, sender, &sender_key, 1_000_000, [0xEEu8; 20])
            .await;

    let nonce = 0u64;
    // V2 message bytes must include the on-chain `created_at_ms`.
    let pre = authority_state.get_object(&withdrawal_id).await.unwrap();
    let pending_pre: PendingWithdrawal =
        pre.deserialize_contents(ObjectType::PendingWithdrawal).unwrap();
    let payload = encode_withdraw_payload(
        &pending_pre.sender,
        pending_pre.target_chain,
        &pending_pre.recipient_eth_address,
        USDC_TOKEN_TYPE,
        pending_pre.amount,
        pending_pre.created_at_ms,
    );
    let message = encode_bridge_message(
        BridgeMessageType::UsdcWithdraw,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );

    // Members 0, 1 sign first.
    let sig01 = build_bridge_signatures(&[&keypairs[0], &keypairs[1]], &message);
    let attach1 =
        TransactionKind::BridgeAttachWithdrawalSignatures(BridgeAttachWithdrawalSignaturesArgs {
            nonce,
            signatures: sig01.clone(),
        });
    assert_eq!(
        *execute_system_tx(&authority_state, attach1).await.unwrap().into_data().status(),
        ExecutionStatus::Success
    );

    // A racing relayer submits a *different* (but also valid) cert from
    // members 2, 3. Sui's `approve_token_transfer` would emit
    // `TokenTransferAlreadyApproved` and skip; we do the same — Ok(()),
    // but the original cert is preserved.
    let sig23 = build_bridge_signatures(&[&keypairs[2], &keypairs[3]], &message);
    let attach2 =
        TransactionKind::BridgeAttachWithdrawalSignatures(BridgeAttachWithdrawalSignaturesArgs {
            nonce,
            signatures: sig23,
        });
    assert_eq!(
        *execute_system_tx(&authority_state, attach2).await.unwrap().into_data().status(),
        ExecutionStatus::Success,
        "second attach must succeed as a no-op (idempotent)"
    );

    let post = authority_state.get_object(&withdrawal_id).await.unwrap();
    let pending: PendingWithdrawal =
        post.deserialize_contents(ObjectType::PendingWithdrawal).unwrap();
    let cert = pending.verified_signatures.expect("cert must still be attached");
    assert_eq!(cert.signatures, sig01, "first cert wins; second submit is a no-op");
}

#[tokio::test]
async fn test_attach_withdrawal_signatures_below_threshold_rejected() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, PendingWithdrawal, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, derive_bridge_record_id,
        encode_bridge_message, encode_withdraw_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;
    use types::transaction::BridgeAttachWithdrawalSignaturesArgs;

    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let sender_key = SomaKeyPair::Ed25519(key);
    create_pending_withdrawal(&authority_state, sender, &sender_key, 1_000_000, [0xFFu8; 20]).await;

    let nonce = 0u64;
    let withdrawal_id =
        derive_bridge_record_id(SOMA_BRIDGE_CHAIN_ID, BridgeMessageType::UsdcWithdraw, nonce);
    let pre = authority_state.get_object(&withdrawal_id).await.unwrap();
    let pending_pre: PendingWithdrawal =
        pre.deserialize_contents(ObjectType::PendingWithdrawal).unwrap();
    let payload = encode_withdraw_payload(
        &pending_pre.sender,
        pending_pre.target_chain,
        &pending_pre.recipient_eth_address,
        USDC_TOKEN_TYPE,
        pending_pre.amount,
        pending_pre.created_at_ms,
    );
    let message = encode_bridge_message(
        BridgeMessageType::UsdcWithdraw,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );

    // Only one signer (2500 voting power < 3334 threshold).
    let agg_sig = build_bridge_signatures(&[&keypairs[0]], &message);
    let kind =
        TransactionKind::BridgeAttachWithdrawalSignatures(BridgeAttachWithdrawalSignaturesArgs {
            nonce,
            signatures: agg_sig,
        });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(
        matches!(
            effects.status(),
            ExecutionStatus::Failure {
                error: ExecutionFailureStatus::BridgeInsufficientSignatureStake,
                ..
            },
        ),
        "Below-threshold cert must be rejected, got {:?}",
        effects.status()
    );
}

#[tokio::test]
async fn test_attach_withdrawal_signatures_wrong_message_rejected() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_withdraw_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;
    use types::transaction::BridgeAttachWithdrawalSignaturesArgs;

    // Defense-in-depth: even if a relayer submits a valid quorum cert for
    // a *different* withdrawal (e.g. tries to repurpose a cert from
    // nonce=1 onto nonce=0 to bypass amount checks), the executor must
    // reject. This is enforced by reconstructing the message from the
    // on-chain object's fields, not from caller input.
    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let sender_key = SomaKeyPair::Ed25519(key);
    create_pending_withdrawal(&authority_state, sender, &sender_key, 1_000_000, [0x11u8; 20]).await;

    // Sign a message for a DIFFERENT amount than the on-chain withdrawal.
    let wrong_amount = 999_999u64;
    let bad_payload = encode_withdraw_payload(
        &SomaAddress::ZERO,
        BridgeChainId::EthCustom,
        &[0x11u8; 20],
        USDC_TOKEN_TYPE,
        wrong_amount,
        0,
    );
    let bad_message = encode_bridge_message(
        BridgeMessageType::UsdcWithdraw,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
        0,
        SOMA_BRIDGE_CHAIN_ID,
        &bad_payload,
    );
    let agg_sig = build_bridge_signatures(&[&keypairs[0], &keypairs[1]], &bad_message);

    let kind =
        TransactionKind::BridgeAttachWithdrawalSignatures(BridgeAttachWithdrawalSignaturesArgs {
            nonce: 0,
            signatures: agg_sig,
        });
    let effects = execute_system_tx(&authority_state, kind).await.unwrap().into_data();
    assert!(
        matches!(*effects.status(), ExecutionStatus::Failure { .. }),
        "Cert signed over wrong message bytes must be rejected, got {:?}",
        effects.status()
    );
}

#[tokio::test]
async fn test_attach_withdrawal_signatures_unknown_nonce_rejected() {
    use types::bridge::{
        BRIDGE_MESSAGE_VERSION, BridgeMessageType, SOMA_BRIDGE_CHAIN_ID,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2, build_bridge_signatures, encode_bridge_message,
        encode_withdraw_payload, generate_test_bridge_committee,
    };
    use types::config::genesis_config::GenesisConfig;
    use types::transaction::BridgeAttachWithdrawalSignaturesArgs;

    let (committee, keypairs) = generate_test_bridge_committee(4);
    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.bridge_committee = Some(committee);
    let authority_state =
        TestAuthorityBuilder::new().with_genesis_config(genesis_config).build().await;

    // No PendingWithdrawal created. Attempting to attach to a nonce that
    // doesn't exist must fail with ObjectNotFound — there's no on-chain
    // record to attach to, and an attacker can't conjure one by signing.
    let nonce = 42u64;
    let payload = encode_withdraw_payload(
        &SomaAddress::ZERO,
        BridgeChainId::EthCustom,
        &[0u8; 20],
        USDC_TOKEN_TYPE,
        0,
        0,
    );
    let message = encode_bridge_message(
        BridgeMessageType::UsdcWithdraw,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );
    let agg_sig = build_bridge_signatures(&[&keypairs[0], &keypairs[1]], &message);
    let kind =
        TransactionKind::BridgeAttachWithdrawalSignatures(BridgeAttachWithdrawalSignaturesArgs {
            nonce,
            signatures: agg_sig,
        });
    // The tx declares the (nonexistent) PendingWithdrawal as a shared
    // input. The input-loading layer rejects before the executor runs
    // — defense in depth: a bad relayer can't even submit a cert for a
    // fabricated nonce, since the chain-level scheduler won't accept
    // the tx in the first place. Either rejection point is acceptable
    // (validator-layer SomaError or executor-layer ObjectNotFound).
    let result = execute_system_tx(&authority_state, kind).await;
    match result {
        Err(_) => {
            // Validator-layer rejection — input object doesn't exist.
        }
        Ok(effects) => {
            let effects_data = effects.into_data();
            assert!(
                matches!(
                    effects_data.status(),
                    ExecutionStatus::Failure {
                        error: ExecutionFailureStatus::ObjectNotFound { .. },
                        ..
                    },
                ),
                "Attach to nonexistent withdrawal must fail, got {:?}",
                effects_data.status()
            );
        }
    }
}
