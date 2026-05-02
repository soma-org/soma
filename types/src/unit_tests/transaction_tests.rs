// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use fastcrypto::ed25519::Ed25519KeyPair;

use crate::base::SomaAddress;
use crate::consensus::ConsensusCommitPrologueV1;
use crate::crypto::{default_hash, get_key_pair};
use crate::digests::{
    AdditionalConsensusStateDigest, ConsensusCommitDigest, ObjectDigest,
};
use crate::envelope::Message;
use crate::error::SomaError;
use crate::intent::{Intent, IntentMessage};
use crate::object::{ObjectID, ObjectRef, Version};
use crate::transaction::*;
use crate::unit_tests::utils::to_sender_signed_transaction;
use crate::{SYSTEM_STATE_OBJECT_ID, SYSTEM_STATE_OBJECT_SHARED_VERSION};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a random ObjectRef suitable for testing.
fn random_object_ref() -> ObjectRef {
    (ObjectID::random(), Version::from_u64(1), ObjectDigest::random())
}

/// Create a simple TransferCoin TransactionData for testing.
fn make_transfer_coin_data() -> (TransactionData, SomaAddress) {
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let coin_ref = random_object_ref();
    let data = TransactionData::new_transfer_coin(recipient, sender, Some(1000), coin_ref);
    (data, sender)
}

/// Helper to create TransactionData for system transactions.
/// Uses TransactionData::new with default sender and empty gas,
/// since TransactionData::new_system_transaction is private.
fn make_system_tx_data(kind: TransactionKind) -> TransactionData {
    assert!(kind.is_system_tx(), "kind must be a system transaction");
    TransactionData::new(kind, SomaAddress::default(), vec![])
}

// ---------------------------------------------------------------------------
// 1. BCS round-trip for TransactionData
// ---------------------------------------------------------------------------

#[test]
fn test_transaction_data_bcs_roundtrip() {
    let (data, sender) = make_transfer_coin_data();

    let bytes = bcs::to_bytes(&data).expect("BCS serialization should succeed");
    let decoded: TransactionData =
        bcs::from_bytes(&bytes).expect("BCS deserialization should succeed");

    assert_eq!(data, decoded);
    assert_eq!(decoded.sender(), sender);
    assert!(!decoded.is_system_tx());
}

// ---------------------------------------------------------------------------
// 2. Digest determinism
// ---------------------------------------------------------------------------

#[test]
fn test_transaction_digest_determinism() {
    let (data, _) = make_transfer_coin_data();
    let digest1 = data.digest();
    let digest2 = data.digest();
    assert_eq!(digest1, digest2, "Same TransactionData must produce identical digests");
}

// ---------------------------------------------------------------------------
// 3. Different data produces different digests
// ---------------------------------------------------------------------------

#[test]
fn test_transaction_digest_different_for_different_data() {
    let (data1, _) = make_transfer_coin_data();
    let (data2, _) = make_transfer_coin_data(); // different random sender/coin
    assert_ne!(
        data1.digest(),
        data2.digest(),
        "Different TransactionData should produce different digests"
    );
}

// ---------------------------------------------------------------------------
// 4. Signed transaction round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_signed_transaction() {
    let (sender, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let recipient = SomaAddress::random();
    let coin_ref = random_object_ref();
    let data = TransactionData::new_transfer_coin(recipient, sender, Some(500), coin_ref);

    let tx = Transaction::from_data_and_signer(data.clone(), vec![&kp]);

    // Verify the data round-trips through the envelope
    let inner_data = tx.data().transaction_data();
    assert_eq!(*inner_data, data);
    assert_eq!(inner_data.sender(), sender);

    // Digest from the envelope should match
    let envelope_digest = *tx.digest();
    assert_eq!(envelope_digest, data.digest());
}

// ---------------------------------------------------------------------------
// 5. TransactionKind classification methods
// ---------------------------------------------------------------------------

#[test]
fn test_transaction_kind_classification() {
    // System transactions
    let genesis = TransactionKind::Genesis(GenesisTransaction { objects: vec![] });
    assert!(genesis.is_system_tx());
    assert!(!genesis.is_validator_tx());
    assert!(!genesis.is_staking_tx());

    let ccp = TransactionKind::ConsensusCommitPrologueV1(ConsensusCommitPrologueV1 {
        epoch: 0,
        round: 1,
        sub_dag_index: None,
        commit_timestamp_ms: 100,
        consensus_commit_digest: ConsensusCommitDigest::new([0; 32]),
        additional_state_digest: AdditionalConsensusStateDigest::new([0; 32]),
    });
    assert!(ccp.is_system_tx());

    let change_epoch = TransactionKind::ChangeEpoch(ChangeEpoch {
        epoch: 1,
        epoch_start_timestamp_ms: 1000,
        protocol_version: protocol_config::ProtocolVersion::MIN,
        fees: 0,
        epoch_randomness: vec![],
    });
    assert!(change_epoch.is_system_tx());
    assert!(change_epoch.is_end_of_epoch_tx());

    // Validator transactions
    let add_val = TransactionKind::AddValidator(AddValidatorArgs {
        pubkey_bytes: vec![1],
        network_pubkey_bytes: vec![2],
        worker_pubkey_bytes: vec![3],
        net_address: vec![],
        p2p_address: vec![],
        primary_address: vec![],
        proxy_address: vec![],
        proof_of_possession: vec![],
    });
    assert!(add_val.is_validator_tx());
    assert!(!add_val.is_system_tx());

    let remove_val =
        TransactionKind::RemoveValidator(RemoveValidatorArgs { pubkey_bytes: vec![1] });
    assert!(remove_val.is_validator_tx());

    let report_val = TransactionKind::ReportValidator { reportee: SomaAddress::random() };
    assert!(report_val.is_validator_tx());

    let undo_report_val = TransactionKind::UndoReportValidator { reportee: SomaAddress::random() };
    assert!(undo_report_val.is_validator_tx());

    let set_commission = TransactionKind::SetCommissionRate { new_rate: 500 };
    assert!(set_commission.is_validator_tx());

    let update_meta =
        TransactionKind::UpdateValidatorMetadata(UpdateValidatorMetadataArgs::default());
    assert!(update_meta.is_validator_tx());

    // Staking transactions
    let add_stake = TransactionKind::AddStake {
        address: SomaAddress::random(),
        coin_ref: random_object_ref(),
        amount: Some(1000),
    };
    assert!(add_stake.is_staking_tx());
    assert!(!add_stake.is_validator_tx());
    assert!(!add_stake.is_system_tx());

    let withdraw_stake = TransactionKind::WithdrawStake { staked_soma: random_object_ref() };
    assert!(withdraw_stake.is_staking_tx());

    // Coin/object transactions should not match any category
    let transfer_coin = TransactionKind::Transfer {
        coins: vec![random_object_ref()],
        amounts: Some(100).map(|a| vec![a]),
        recipients: vec![SomaAddress::random()],
    };
    assert!(!transfer_coin.is_system_tx());
    assert!(!transfer_coin.is_validator_tx());
    assert!(!transfer_coin.is_staking_tx());
}

// ---------------------------------------------------------------------------
// 6. System transactions have no gas
// ---------------------------------------------------------------------------

#[test]
fn test_system_tx_has_no_gas() {
    // Genesis
    let genesis_data =
        make_system_tx_data(TransactionKind::Genesis(GenesisTransaction { objects: vec![] }));
    assert!(genesis_data.gas().is_empty());
    assert!(genesis_data.is_system_tx());

    // ConsensusCommitPrologueV1
    let ccp_data = make_system_tx_data(TransactionKind::ConsensusCommitPrologueV1(
        ConsensusCommitPrologueV1 {
            epoch: 0,
            round: 1,
            sub_dag_index: None,
            commit_timestamp_ms: 100,
            consensus_commit_digest: ConsensusCommitDigest::new([0; 32]),
            additional_state_digest: AdditionalConsensusStateDigest::new([0; 32]),
        },
    ));
    assert!(ccp_data.gas().is_empty());
    assert!(ccp_data.is_system_tx());

    // ChangeEpoch
    let change_epoch_data = make_system_tx_data(TransactionKind::ChangeEpoch(ChangeEpoch {
        epoch: 1,
        epoch_start_timestamp_ms: 0,
        protocol_version: protocol_config::ProtocolVersion::MIN,
        fees: 0,
        epoch_randomness: vec![],
    }));
    assert!(change_epoch_data.gas().is_empty());
    assert!(change_epoch_data.is_system_tx());
}

// ---------------------------------------------------------------------------
// 7. User transactions have gas
// ---------------------------------------------------------------------------

#[test]
fn test_user_tx_has_gas() {
    let (data, _) = make_transfer_coin_data();
    assert!(!data.gas().is_empty(), "TransferCoin should have gas payment");
    assert!(!data.is_system_tx());

    // AddStake
    let sender = SomaAddress::random();
    let coin_ref = random_object_ref();
    let gas_ref = random_object_ref();
    let add_stake_data = TransactionData::new(
        TransactionKind::AddStake { address: SomaAddress::random(), coin_ref, amount: Some(1000) },
        sender,
        vec![gas_ref],
    );
    assert!(!add_stake_data.gas().is_empty(), "AddStake should have gas payment");
}

// ---------------------------------------------------------------------------
// 8. All TransactionKind variants BCS round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_all_tx_kinds_bcs_roundtrip() {
    let kinds: Vec<TransactionKind> = vec![
        // System
        TransactionKind::Genesis(GenesisTransaction { objects: vec![] }),
        TransactionKind::ConsensusCommitPrologueV1(ConsensusCommitPrologueV1 {
            epoch: 1,
            round: 2,
            sub_dag_index: Some(3),
            commit_timestamp_ms: 12345,
            consensus_commit_digest: ConsensusCommitDigest::new([1; 32]),
            additional_state_digest: AdditionalConsensusStateDigest::new([2; 32]),
        }),
        TransactionKind::ChangeEpoch(ChangeEpoch {
            epoch: 5,
            epoch_start_timestamp_ms: 99999,
            protocol_version: protocol_config::ProtocolVersion::MIN,
            fees: 42,
            epoch_randomness: vec![0xAA, 0xBB],
        }),
        // Validator management
        TransactionKind::AddValidator(AddValidatorArgs {
            pubkey_bytes: vec![10],
            network_pubkey_bytes: vec![20],
            worker_pubkey_bytes: vec![30],
            net_address: vec![40],
            p2p_address: vec![50],
            primary_address: vec![60],
            proxy_address: vec![70],
            proof_of_possession: vec![80],
        }),
        TransactionKind::RemoveValidator(RemoveValidatorArgs { pubkey_bytes: vec![10] }),
        TransactionKind::ReportValidator { reportee: SomaAddress::random() },
        TransactionKind::UndoReportValidator { reportee: SomaAddress::random() },
        TransactionKind::UpdateValidatorMetadata(UpdateValidatorMetadataArgs::default()),
        TransactionKind::SetCommissionRate { new_rate: 100 },
        // Coin/object
        TransactionKind::Transfer {
        coins: vec![random_object_ref()],
        amounts: Some(500).map(|a| vec![a]),
        recipients: vec![SomaAddress::random()],
    },
        TransactionKind::Transfer {
            coins: vec![random_object_ref(), random_object_ref()],
            amounts: Some(vec![100, 200]),
            recipients: vec![SomaAddress::random(), SomaAddress::random()],
        },
        TransactionKind::MergeCoins {
            coins: vec![random_object_ref(), random_object_ref()],
        },
        TransactionKind::TransferObjects {
            objects: vec![random_object_ref()],
            recipient: SomaAddress::random(),
        },
        // Staking
        TransactionKind::AddStake {
            address: SomaAddress::random(),
            coin_ref: random_object_ref(),
            amount: Some(1000),
        },
        TransactionKind::WithdrawStake { staked_soma: random_object_ref() },
        // Bridge
        TransactionKind::BridgeDeposit(crate::transaction::BridgeDepositArgs {
            nonce: 1,
            eth_tx_hash: [0u8; 32],
            recipient: SomaAddress::random(),
            amount: 1000,
            aggregated_signature: vec![],
            signer_bitmap: vec![],
        }),
        TransactionKind::BridgeWithdraw(crate::transaction::BridgeWithdrawArgs {
            amount: 500,
            recipient_eth_address: [0u8; 20],
        }),
        TransactionKind::BridgeEmergencyPause(crate::transaction::BridgeEmergencyPauseArgs {
            aggregated_signature: vec![],
            signer_bitmap: vec![],
        }),
        TransactionKind::BridgeEmergencyUnpause(crate::transaction::BridgeEmergencyUnpauseArgs {
            aggregated_signature: vec![],
            signer_bitmap: vec![],
        }),
    ];

    assert_eq!(
        kinds.len(),
        19,
        "Expected 19 TransactionKind variants; if a new variant was added, update this test"
    );

    for (i, kind) in kinds.iter().enumerate() {
        let bytes =
            bcs::to_bytes(kind).unwrap_or_else(|e| panic!("BCS ser failed for variant {i}: {e}"));
        let decoded: TransactionKind = bcs::from_bytes(&bytes)
            .unwrap_or_else(|e| panic!("BCS deser failed for variant {i}: {e}"));
        assert_eq!(*kind, decoded, "BCS round-trip failed for variant {i}");
    }
}

// ---------------------------------------------------------------------------
// 9. SenderSignedData::new wraps correctly
// ---------------------------------------------------------------------------

#[test]
fn test_sender_signed_data_new() {
    let (data, _) = make_transfer_coin_data();
    let ssd = SenderSignedData::new(data.clone(), vec![]);

    assert_eq!(*ssd.transaction_data(), data);
    assert!(ssd.tx_signatures().is_empty());

    // Intent message should wrap the data with the correct intent
    let intent_msg = ssd.intent_message();
    assert_eq!(intent_msg.intent, Intent::soma_transaction());
    assert_eq!(intent_msg.value, data);
}

// ---------------------------------------------------------------------------
// 10. ChangeEpoch transaction classification
// ---------------------------------------------------------------------------

#[test]
fn test_change_epoch_transaction() {
    let change_epoch = ChangeEpoch {
        epoch: 10,
        epoch_start_timestamp_ms: 1_000_000,
        protocol_version: protocol_config::ProtocolVersion::MIN,
        fees: 500,
        epoch_randomness: vec![1, 2, 3],
    };
    let data = make_system_tx_data(TransactionKind::ChangeEpoch(change_epoch.clone()));

    assert!(data.is_system_tx());
    assert!(data.kind().is_end_of_epoch_tx());
    assert!(data.kind().is_epoch_change());
    assert!(!data.is_genesis_tx());
    assert!(!data.is_consensus_commit_prologue());
    assert_eq!(data.sender(), SomaAddress::default());
    assert!(data.gas().is_empty());
}

// ---------------------------------------------------------------------------
// 11. ConsensusCommitPrologueV1 transaction classification
// ---------------------------------------------------------------------------

#[test]
fn test_consensus_commit_prologue_transaction() {
    let ccp = ConsensusCommitPrologueV1 {
        epoch: 3,
        round: 42,
        sub_dag_index: None,
        commit_timestamp_ms: 999,
        consensus_commit_digest: ConsensusCommitDigest::new([7; 32]),
        additional_state_digest: AdditionalConsensusStateDigest::new([8; 32]),
    };
    let data = make_system_tx_data(TransactionKind::ConsensusCommitPrologueV1(ccp));

    assert!(data.is_system_tx());
    assert!(data.is_consensus_commit_prologue());
    assert!(!data.is_genesis_tx());
    assert!(!data.kind().is_end_of_epoch_tx());
    assert_eq!(data.sender(), SomaAddress::default());
}

// ---------------------------------------------------------------------------
// 12. Genesis transaction classification
// ---------------------------------------------------------------------------

#[test]
fn test_genesis_transaction() {
    let data =
        make_system_tx_data(TransactionKind::Genesis(GenesisTransaction { objects: vec![] }));

    assert!(data.is_system_tx());
    assert!(data.is_genesis_tx());
    assert!(!data.is_consensus_commit_prologue());
    assert!(!data.kind().is_end_of_epoch_tx());
    assert_eq!(data.sender(), SomaAddress::default());
    assert!(data.gas().is_empty());
}

// ---------------------------------------------------------------------------
// 13. shared_input_objects returns SystemState for validator/staking/model/submission txs
// ---------------------------------------------------------------------------

#[test]
fn test_shared_input_objects() {
    // Validator tx -> SystemState only
    let add_val = TransactionKind::AddValidator(AddValidatorArgs {
        pubkey_bytes: vec![],
        network_pubkey_bytes: vec![],
        worker_pubkey_bytes: vec![],
        net_address: vec![],
        p2p_address: vec![],
        primary_address: vec![],
        proxy_address: vec![],
        proof_of_possession: vec![],
    });
    let shared: Vec<_> = add_val.shared_input_objects().collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].id, SYSTEM_STATE_OBJECT_ID);

    // Staking tx -> SystemState only
    let add_stake = TransactionKind::AddStake {
        address: SomaAddress::random(),
        coin_ref: random_object_ref(),
        amount: Some(1000),
    };
    let shared: Vec<_> = add_stake.shared_input_objects().collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].id, SYSTEM_STATE_OBJECT_ID);

    // Genesis -> no shared input objects
    let genesis = TransactionKind::Genesis(GenesisTransaction { objects: vec![] });
    let shared: Vec<_> = genesis.shared_input_objects().collect();
    assert!(shared.is_empty());

    // TransferCoin -> no shared input objects
    let transfer = TransactionKind::Transfer {
        coins: vec![random_object_ref()],
        amounts: Some(100).map(|a| vec![a]),
        recipients: vec![SomaAddress::random()],
    };
    let shared: Vec<_> = transfer.shared_input_objects().collect();
    assert!(shared.is_empty());
}

// ---------------------------------------------------------------------------
// 14. input_objects deduplicates gas coin from inputs
// ---------------------------------------------------------------------------

#[test]
fn test_input_objects_no_duplicates() {
    // TransferCoin uses the coin as both the input and gas payment.
    // TransactionData::new_transfer_coin sets gas_payment = [coin_ref], same as
    // the coin inside TransferCoin. The input_objects() method should deduplicate.
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let coin_ref = random_object_ref();
    let data = TransactionData::new_transfer_coin(recipient, sender, Some(100), coin_ref);

    let inputs = data.input_objects().expect("input_objects should succeed");
    // The coin appears once as ImmOrOwnedObject, and since the gas_payment contains
    // the same ObjectID, it should NOT be added again.
    let coin_count = inputs.iter().filter(|inp| inp.object_id() == coin_ref.0).count();
    assert_eq!(coin_count, 1, "Gas coin should not be duplicated in input_objects");
}

// ---------------------------------------------------------------------------
// 15. contains_shared_object
// ---------------------------------------------------------------------------

#[test]
fn test_contains_shared_object() {
    // TransferCoin does NOT touch shared state
    let transfer = TransactionKind::Transfer {
        coins: vec![random_object_ref()],
        amounts: Some(100).map(|a| vec![a]),
        recipients: vec![SomaAddress::random()],
    };
    assert!(!transfer.contains_shared_object(), "TransferCoin should not contain shared objects");

    // AddStake touches SystemState
    let add_stake = TransactionKind::AddStake {
        address: SomaAddress::random(),
        coin_ref: random_object_ref(),
        amount: Some(1000),
    };
    assert!(
        add_stake.contains_shared_object(),
        "AddStake should contain shared objects (SystemState)"
    );

    // Genesis does NOT touch shared state
    let genesis = TransactionKind::Genesis(GenesisTransaction { objects: vec![] });
    assert!(!genesis.contains_shared_object(), "Genesis should not contain shared objects");
}

// ---------------------------------------------------------------------------
// 16. TransactionData::execution_parts
// ---------------------------------------------------------------------------

#[test]
fn test_execution_parts() {
    let (data, sender) = make_transfer_coin_data();
    let (kind, exec_sender, gas) = data.execution_parts();

    assert_eq!(&kind, data.kind());
    assert_eq!(exec_sender, sender);
    assert_eq!(gas, data.gas());
}

// ---------------------------------------------------------------------------
// 17. Signed transaction signature verification
// ---------------------------------------------------------------------------

#[test]
fn test_verify_sender_signed_transaction() {
    let (sender, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let recipient = SomaAddress::random();
    let coin_ref = random_object_ref();
    let data = TransactionData::new_transfer_coin(recipient, sender, Some(100), coin_ref);

    let tx = Transaction::from_data_and_signer(data, vec![&kp]);
    assert!(tx.verify_signature_for_testing().is_ok(), "Valid signature should pass verification");
}

// ---------------------------------------------------------------------------
// 18. SenderSignedData digest matches TransactionData digest
// ---------------------------------------------------------------------------

#[test]
fn test_sender_signed_data_digest() {
    let (data, _) = make_transfer_coin_data();
    let ssd = SenderSignedData::new(data.clone(), vec![]);

    // The SenderSignedData's Message::digest() should be the same as TransactionData::digest()
    let ssd_digest = <SenderSignedData as Message>::digest(&ssd);
    let data_digest = data.digest();
    assert_eq!(ssd_digest, data_digest);
}

// ---------------------------------------------------------------------------
// 19. TransactionKey
// ---------------------------------------------------------------------------

#[test]
fn test_transaction_key() {
    let (data, _) = make_transfer_coin_data();
    let (_, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let tx = to_sender_signed_transaction(data.clone(), &kp);

    let key = tx.key();
    match key {
        TransactionKey::Digest(d) => {
            assert_eq!(d, data.digest());
        }
    }
    assert_eq!(*key.unwrap_digest(), data.digest());
    assert!(key.as_digest().is_some());
}

// ---------------------------------------------------------------------------
// 20. VerifiedTransaction system transaction constructors
// ---------------------------------------------------------------------------

#[test]
fn test_verified_transaction_system_constructors() {
    // Genesis
    let vtx_genesis = VerifiedTransaction::new_genesis_transaction(vec![]);
    let genesis_data = vtx_genesis.data().transaction_data();
    assert!(genesis_data.is_genesis_tx());
    assert!(genesis_data.is_system_tx());
    assert!(genesis_data.gas().is_empty());

    // CCP
    let vtx_ccp = VerifiedTransaction::new_consensus_commit_prologue(
        1,
        10,
        12345,
        ConsensusCommitDigest::new([0; 32]),
        AdditionalConsensusStateDigest::new([0; 32]),
    );
    let ccp_data = vtx_ccp.data().transaction_data();
    assert!(ccp_data.is_consensus_commit_prologue());
    assert!(ccp_data.is_system_tx());

    // ChangeEpoch
    let vtx_epoch = VerifiedTransaction::new_change_epoch_transaction(
        5,
        protocol_config::ProtocolVersion::MIN,
        100,
        999999,
        vec![],
    );
    let epoch_data = vtx_epoch.data().transaction_data();
    assert!(epoch_data.kind().is_end_of_epoch_tx());
    assert!(epoch_data.is_system_tx());
}

// ---------------------------------------------------------------------------
// 21. full_message_digest is deterministic
// ---------------------------------------------------------------------------

#[test]
fn test_full_message_digest_deterministic() {
    let (data, _) = make_transfer_coin_data();
    let ssd = SenderSignedData::new(data, vec![]);

    let digest1 = ssd.full_message_digest();
    let digest2 = ssd.full_message_digest();
    assert_eq!(digest1, digest2, "full_message_digest should be deterministic");
}

// ---------------------------------------------------------------------------
// 22. TransactionData::new_transfer and new_pay_coins constructors
// ---------------------------------------------------------------------------

#[test]
fn test_transaction_data_constructors() {
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let obj_ref = random_object_ref();
    let gas_ref = random_object_ref();

    // new_transfer
    let transfer_data = TransactionData::new_transfer(recipient, obj_ref, sender, vec![gas_ref]);
    assert_eq!(transfer_data.sender(), sender);
    assert_eq!(transfer_data.gas(), vec![gas_ref]);
    match transfer_data.kind() {
        TransactionKind::TransferObjects { objects, recipient: r } => {
            assert_eq!(objects.len(), 1);
            assert_eq!(objects[0], obj_ref);
            assert_eq!(*r, recipient);
        }
        _ => panic!("Expected TransferObjects kind"),
    }

    // new_pay_coins
    let coin1 = random_object_ref();
    let coin2 = random_object_ref();
    let r1 = SomaAddress::random();
    let r2 = SomaAddress::random();
    let pay_data = TransactionData::new_pay_coins(
        vec![coin1, coin2],
        Some(vec![100, 200]),
        vec![r1, r2],
        sender,
    );
    assert_eq!(pay_data.sender(), sender);
    // Gas payment should be the first coin
    assert_eq!(pay_data.gas(), vec![coin1]);
    match pay_data.kind() {
        TransactionKind::Transfer { coins, amounts, recipients } => {
            assert_eq!(coins.len(), 2);
            assert_eq!(*amounts, Some(vec![100, 200]));
            assert_eq!(recipients.len(), 2);
        }
        _ => panic!("Expected Transfer kind"),
    }
}

// ---------------------------------------------------------------------------
// 23. Requires system state
// ---------------------------------------------------------------------------

#[test]
fn test_requires_system_state() {
    // Validator tx requires system state
    let add_val = TransactionKind::AddValidator(AddValidatorArgs {
        pubkey_bytes: vec![],
        network_pubkey_bytes: vec![],
        worker_pubkey_bytes: vec![],
        net_address: vec![],
        p2p_address: vec![],
        primary_address: vec![],
        proxy_address: vec![],
        proof_of_possession: vec![],
    });
    assert!(add_val.requires_system_state());

    // Staking requires system state
    let add_stake = TransactionKind::AddStake {
        address: SomaAddress::random(),
        coin_ref: random_object_ref(),
        amount: None,
    };
    assert!(add_stake.requires_system_state());

    // ChangeEpoch requires system state (is_epoch_change)
    let epoch = TransactionKind::ChangeEpoch(ChangeEpoch {
        epoch: 1,
        epoch_start_timestamp_ms: 0,
        protocol_version: protocol_config::ProtocolVersion::MIN,
        fees: 0,
        epoch_randomness: vec![],
    });
    assert!(epoch.requires_system_state());

    // Transfer does NOT require system state
    let transfer = TransactionKind::Transfer {
        coins: vec![random_object_ref()],
        amounts: Some(100).map(|a| vec![a]),
        recipients: vec![SomaAddress::random()],
    };
    assert!(!transfer.requires_system_state());

    // Genesis does NOT require system state
    let genesis = TransactionKind::Genesis(GenesisTransaction { objects: vec![] });
    assert!(!genesis.requires_system_state());
}

// ---------------------------------------------------------------------------
// 24. input_objects for system transactions
// ---------------------------------------------------------------------------

#[test]
fn test_input_objects_system_tx() {
    // Genesis has no input objects
    let genesis = TransactionKind::Genesis(GenesisTransaction { objects: vec![] });
    let inputs = genesis.input_objects().expect("should succeed");
    assert!(inputs.is_empty(), "Genesis should have no input objects");

    // CCP declares Clock as a mutable shared input — that's how the
    // prologue executor mutates the wall-clock timestamp each commit.
    let ccp = TransactionKind::ConsensusCommitPrologueV1(ConsensusCommitPrologueV1 {
        epoch: 0,
        round: 1,
        sub_dag_index: None,
        commit_timestamp_ms: 0,
        consensus_commit_digest: ConsensusCommitDigest::new([0; 32]),
        additional_state_digest: AdditionalConsensusStateDigest::new([0; 32]),
    });
    let inputs = ccp.input_objects().expect("should succeed");
    assert_eq!(inputs.len(), 1, "CCP must declare exactly Clock as input");
    match &inputs[0] {
        crate::transaction::InputObjectKind::SharedObject { id, mutable, .. } => {
            assert_eq!(*id, crate::CLOCK_OBJECT_ID, "CCP input must be the Clock object");
            assert!(*mutable, "CCP must declare Clock as mutable");
        }
        other => panic!("CCP input must be a SharedObject, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 25. input_objects for various user transactions
// ---------------------------------------------------------------------------

#[test]
fn test_input_objects_user_txs() {
    // TransferCoin: should have the coin as ImmOrOwnedObject
    let coin_ref = random_object_ref();
    let transfer = TransactionKind::Transfer {
        coins: vec![coin_ref],
        amounts: Some(100).map(|a| vec![a]),
        recipients: vec![SomaAddress::random()],
    };
    let inputs = transfer.input_objects().expect("should succeed");
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].object_id(), coin_ref.0);

    // AddStake: should have SystemState (shared) + coin_ref (owned)
    let coin_ref2 = random_object_ref();
    let add_stake = TransactionKind::AddStake {
        address: SomaAddress::random(),
        coin_ref: coin_ref2,
        amount: Some(1000),
    };
    let inputs = add_stake.input_objects().expect("should succeed");
    assert_eq!(inputs.len(), 2);
    // First is SystemState shared object
    assert!(inputs[0].is_shared_object());
    assert_eq!(inputs[0].object_id(), SYSTEM_STATE_OBJECT_ID);
    // Second is the coin
    assert!(!inputs[1].is_shared_object());
    assert_eq!(inputs[1].object_id(), coin_ref2.0);

}

// ---------------------------------------------------------------------------
// Channel tx kind input shapes
// ---------------------------------------------------------------------------

#[test]
fn test_input_objects_open_channel() {
    // Stage 8: OpenChannel debits the deposit from the sender's
    // accumulator balance, not a coin object. Input set is empty —
    // funds-availability is enforced by the reservation pre-pass via
    // `TransactionData::reservations()`, not by reading a coin input.
    let kind = TransactionKind::OpenChannel(OpenChannelArgs {
        payee: SomaAddress::random(),
        authorized_signer: SomaAddress::random(),
        token: crate::object::CoinType::Usdc,
        deposit_amount: 1_000,
    });
    let inputs = kind.input_objects().expect("OpenChannel inputs build");
    assert!(inputs.is_empty(), "OpenChannel has no owned inputs in balance-mode");
    assert_eq!(kind.shared_input_objects().count(), 0);
}

#[test]
fn test_input_objects_settle() {
    // Settle: declares the Channel as a mutable shared input only.
    let channel_id = ObjectID::random();
    let kind = TransactionKind::Settle(SettleArgs {
        channel_id,
        cumulative_amount: 100,
        voucher_signature: dummy_voucher_signature(),
    });
    let inputs = kind.input_objects().expect("Settle inputs build");
    assert_eq!(inputs.len(), 1);
    assert!(inputs[0].is_shared_object());
    assert_eq!(inputs[0].object_id(), channel_id);
    assert!(inputs[0].is_mutable());

    let shared: Vec<_> = kind.shared_input_objects().collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].id, channel_id);
    assert!(shared[0].mutable);
}

#[test]
fn test_input_objects_request_close() {
    // RequestClose: Channel (mutable shared) + Clock (immutable shared).
    let channel_id = ObjectID::random();
    let kind = TransactionKind::RequestClose(RequestCloseArgs { channel_id });
    let inputs = kind.input_objects().expect("RequestClose inputs build");
    assert_eq!(inputs.len(), 2);
    let ids: Vec<_> = inputs.iter().map(|i| i.object_id()).collect();
    assert!(ids.contains(&crate::CLOCK_OBJECT_ID), "Clock must be declared");
    assert!(ids.contains(&channel_id), "Channel must be declared");

    // Clock must be immutable.
    let clock_input = inputs.iter().find(|i| i.object_id() == crate::CLOCK_OBJECT_ID).unwrap();
    assert!(!clock_input.is_mutable(), "Clock must be read-only for user txs");
    // Channel must be mutable.
    let channel_input = inputs.iter().find(|i| i.object_id() == channel_id).unwrap();
    assert!(channel_input.is_mutable(), "Channel is mutated by RequestClose");
}

#[test]
fn test_input_objects_withdraw_after_timeout() {
    // WithdrawAfterTimeout: Channel (mutable) + Clock (read) + SystemState (read).
    let channel_id = ObjectID::random();
    let kind =
        TransactionKind::WithdrawAfterTimeout(WithdrawAfterTimeoutArgs { channel_id });
    let inputs = kind.input_objects().expect("WithdrawAfterTimeout inputs build");
    assert_eq!(inputs.len(), 3);

    let ids: Vec<_> = inputs.iter().map(|i| i.object_id()).collect();
    assert!(ids.contains(&channel_id));
    assert!(ids.contains(&crate::CLOCK_OBJECT_ID));
    assert!(ids.contains(&SYSTEM_STATE_OBJECT_ID));

    // Channel mutable, Clock read-only, SystemState read-only.
    for input in &inputs {
        let id = input.object_id();
        let expected_mut = id == channel_id;
        assert_eq!(
            input.is_mutable(),
            expected_mut,
            "object {} mutability should be {}",
            id,
            expected_mut
        );
    }
}

/// Helper: a syntactically-valid GenericSignature for shape tests. The
/// signature does not need to verify since we're testing input-object
/// declarations, not execution.
fn dummy_voucher_signature() -> crate::crypto::GenericSignature {
    use fastcrypto::ed25519::Ed25519KeyPair;
    let (_, kp): (SomaAddress, Ed25519KeyPair) = crate::crypto::get_key_pair();
    let voucher = crate::channel::Voucher::new(ObjectID::ZERO, 0);
    let intent_msg = crate::intent::IntentMessage::new(
        crate::intent::Intent::soma_app(crate::intent::IntentScope::PaymentVoucher),
        voucher,
    );
    crate::crypto::Signature::new_secure(&intent_msg, &kp).into()
}

// ---------------------------------------------------------------------------
// 26. SenderSignedData serialized_size
// ---------------------------------------------------------------------------

#[test]
fn test_sender_signed_data_serialized_size() {
    let (data, _) = make_transfer_coin_data();
    let ssd = SenderSignedData::new(data, vec![]);
    let size = ssd.serialized_size().expect("serialized_size should succeed");
    assert!(size > 0, "serialized_size should be non-zero");

    // Verify consistency with actual BCS serialization
    let bytes = bcs::to_bytes(&ssd).expect("BCS serialization should succeed");
    assert_eq!(size, bytes.len());
}

// ---------------------------------------------------------------------------
// 27. TransactionData signers
// ---------------------------------------------------------------------------

#[test]
fn test_transaction_data_signers() {
    let (data, sender) = make_transfer_coin_data();
    let signers = data.signers();
    assert_eq!(signers.len(), 1);
    assert_eq!(signers.head, sender);
}

// ---------------------------------------------------------------------------
// 28. SharedInputObject constants and methods
// ---------------------------------------------------------------------------

#[test]
fn test_shared_input_object() {
    let sys = SharedInputObject::SYSTEM_OBJ;
    assert_eq!(sys.id(), SYSTEM_STATE_OBJECT_ID);
    assert_eq!(sys.id_and_version(), (SYSTEM_STATE_OBJECT_ID, SYSTEM_STATE_OBJECT_SHARED_VERSION));
    assert!(sys.mutable);

    let custom = SharedInputObject {
        id: ObjectID::random(),
        initial_shared_version: Version::from_u64(5),
        mutable: false,
    };
    assert_eq!(custom.id(), custom.id);
    assert!(!custom.mutable);
    let (_id, ver) = custom.into_id_and_version();
    assert_eq!(ver, Version::from_u64(5));
}

// ---------------------------------------------------------------------------
// 29. InputObjectKind methods
// ---------------------------------------------------------------------------

#[test]
fn test_input_object_kind() {
    let obj_ref = random_object_ref();
    let owned = InputObjectKind::ImmOrOwnedObject(obj_ref);
    assert_eq!(owned.object_id(), obj_ref.0);
    assert_eq!(owned.version(), Some(obj_ref.1));
    assert!(!owned.is_shared_object());
    assert!(owned.is_mutable());

    let shared = InputObjectKind::SharedObject {
        id: ObjectID::random(),
        initial_shared_version: Version::from_u64(1),
        mutable: true,
    };
    assert!(shared.is_shared_object());
    assert!(shared.is_mutable());
    assert_eq!(shared.version(), None);

    let shared_immut = InputObjectKind::SharedObject {
        id: ObjectID::random(),
        initial_shared_version: Version::from_u64(1),
        mutable: false,
    };
    assert!(!shared_immut.is_mutable());
}

// ---------------------------------------------------------------------------
// 30. Envelope contains_shared_object / is_consensus_tx
// ---------------------------------------------------------------------------

#[test]
fn test_envelope_shared_object_methods() {
    let sender = SomaAddress::random();
    let gas_ref = random_object_ref();

    // AddStake transaction -> touches shared SystemState
    let add_stake_data = TransactionData::new(
        TransactionKind::AddStake {
            address: SomaAddress::random(),
            coin_ref: random_object_ref(),
            amount: Some(1000),
        },
        sender,
        vec![gas_ref],
    );
    let ssd = SenderSignedData::new(add_stake_data, vec![]);
    let tx = Transaction::new(ssd);
    assert!(tx.contains_shared_object());
    assert!(tx.is_consensus_tx());

    // TransferCoin -> no shared objects
    let transfer_data = TransactionData::new_transfer_coin(
        SomaAddress::random(),
        sender,
        Some(100),
        random_object_ref(),
    );
    let ssd2 = SenderSignedData::new(transfer_data, vec![]);
    let tx2 = Transaction::new(ssd2);
    assert!(!tx2.contains_shared_object());
    assert!(!tx2.is_consensus_tx());
}

// ---------------------------------------------------------------------------
// Stage 2: TransactionData::reservations() placeholder contract
// ---------------------------------------------------------------------------
//
// The scheduler's reservation pre-pass (Stage 4 + 6d) calls
// `tx.reservations(unit_fee)` for every tx in a commit. Coin-mode txs
// (non-empty gas_payment) never declare reservations — replay/
// insufficiency is caught at execution. Balance-mode txs (empty
// gas_payment + ValidDuring) declare a USDC reservation for the gas
// fee.

const TEST_UNIT_FEE: u64 = 1_000;

#[test]
fn test_reservations_transfer_coin_is_empty() {
    // Coin-mode tx (gas_payment = vec![coin_ref]) — no reservation.
    let (data, _) = make_transfer_coin_data();
    assert!(
        data.reservations(TEST_UNIT_FEE).is_empty(),
        "Coin-mode TransferCoin must not declare gas reservation"
    );
}

#[test]
fn test_reservations_genesis_is_empty() {
    let data = make_system_tx_data(TransactionKind::Genesis(GenesisTransaction {
        objects: vec![],
    }));
    assert!(data.reservations(TEST_UNIT_FEE).is_empty());
}

#[test]
fn test_reservations_change_epoch_is_empty() {
    let data = make_system_tx_data(TransactionKind::ChangeEpoch(ChangeEpoch {
        epoch: 1,
        epoch_start_timestamp_ms: 1000,
        protocol_version: protocol_config::ProtocolVersion::MIN,
        fees: 0,
        epoch_randomness: vec![],
    }));
    assert!(data.reservations(TEST_UNIT_FEE).is_empty());
}

#[test]
fn test_reservations_balance_mode_returns_gas_reservation() {
    // Stage 6d: balance-mode tx (empty gas_payment) declares
    // a USDC reservation = unit_fee × kind.fee_units().
    use crate::balance::WithdrawalReservation;
    use crate::object::CoinType;

    let chain = fresh_chain_id();
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let coin_ref = random_object_ref();

    // Transfer with 1 input + 1 output → fee_units = 2.
    let data = TransactionData::new_with_expiration(
        TransactionKind::Transfer {
            coins: vec![coin_ref],
            amounts: Some(vec![1]),
            recipients: vec![recipient],
        },
        sender,
        Vec::new(), // empty gas_payment → balance-mode
        TransactionExpiration::ValidDuring {
            min_epoch: Some(0),
            max_epoch: Some(1),
            chain,
            nonce: 0,
        },
    );

    let reservations = data.reservations(TEST_UNIT_FEE);
    assert_eq!(reservations.len(), 1, "balance-mode tx must declare exactly one gas reservation");
    assert_eq!(
        reservations[0],
        WithdrawalReservation::new(sender, CoinType::Usdc, TEST_UNIT_FEE * 2),
        "gas reservation is unit_fee × kind.fee_units()"
    );
}

#[test]
fn test_reservations_balance_mode_zero_unit_fee_skips_reservation() {
    // If unit_fee is 0 (e.g., a chain-wide fee waiver), we don't emit
    // a zero-amount reservation. Cleaner for the scheduler and tests.
    let chain = fresh_chain_id();
    let data = TransactionData::new_with_expiration(
        TransactionKind::Transfer {
            coins: vec![random_object_ref()],
            amounts: Some(vec![1]),
            recipients: vec![SomaAddress::random()],
        },
        SomaAddress::random(),
        Vec::new(),
        TransactionExpiration::ValidDuring {
            min_epoch: Some(0),
            max_epoch: Some(0),
            chain,
            nonce: 0,
        },
    );
    assert!(data.reservations(0).is_empty());
}

// ---------------------------------------------------------------------------
// Stage 5.5a: TransactionExpiration / replay-protection declaration
// ---------------------------------------------------------------------------
//
// Structural validation of the expiration declaration. The "is the digest
// in the executed cache?" check happens elsewhere and is tested in the
// authority crate (Stage 5.5b/c).

use crate::digests::ChainIdentifier;

fn fresh_chain_id() -> ChainIdentifier {
    // Build a deterministic non-default chain id for tests.
    crate::digests::CheckpointDigest::new([7u8; 32]).into()
}

#[test]
fn test_expiration_default_is_none() {
    let (data, _) = make_transfer_coin_data();
    assert!(matches!(data.expiration(), TransactionExpiration::None));
    assert!(!data.expiration().is_replay_protected());
}

#[test]
fn test_expiration_none_check_passes() {
    // `None` always passes structural check — replay protection comes
    // from owned-input version-bumps for these.
    let (data, _) = make_transfer_coin_data();
    let chain = fresh_chain_id();
    data.check_expiration(0, &chain).expect("None expiration is always structurally valid");
    data.check_expiration(999, &chain).expect("None expiration ignores epoch");
}

#[test]
fn test_expiration_valid_during_within_window_passes() {
    let chain = fresh_chain_id();
    let sender = SomaAddress::random();
    let data = TransactionData::new_with_expiration(
        TransactionKind::Transfer {
            coins: vec![random_object_ref()],
            amounts: None,
            recipients: vec![SomaAddress::random()],
        },
        sender,
        vec![random_object_ref()],
        TransactionExpiration::ValidDuring {
            min_epoch: Some(5),
            max_epoch: Some(6),
            chain,
            nonce: 42,
        },
    );

    data.check_expiration(5, &chain).expect("at min_epoch");
    data.check_expiration(6, &chain).expect("at max_epoch");
    assert!(data.expiration().is_replay_protected());
}

#[test]
fn test_expiration_valid_during_premature_rejected() {
    let chain = fresh_chain_id();
    let data = TransactionData::new_with_expiration(
        TransactionKind::Transfer {
            coins: vec![random_object_ref()],
            amounts: None,
            recipients: vec![SomaAddress::random()],
        },
        SomaAddress::random(),
        vec![random_object_ref()],
        TransactionExpiration::ValidDuring {
            min_epoch: Some(5),
            max_epoch: Some(6),
            chain,
            nonce: 0,
        },
    );
    let err = data.check_expiration(4, &chain).expect_err("epoch < min must reject");
    assert!(matches!(err, SomaError::TransactionExpired { current_epoch: 4, .. }));
}

#[test]
fn test_expiration_valid_during_expired_rejected() {
    let chain = fresh_chain_id();
    let data = TransactionData::new_with_expiration(
        TransactionKind::Transfer {
            coins: vec![random_object_ref()],
            amounts: None,
            recipients: vec![SomaAddress::random()],
        },
        SomaAddress::random(),
        vec![random_object_ref()],
        TransactionExpiration::ValidDuring {
            min_epoch: Some(5),
            max_epoch: Some(6),
            chain,
            nonce: 0,
        },
    );
    let err = data.check_expiration(7, &chain).expect_err("epoch > max must reject");
    assert!(matches!(err, SomaError::TransactionExpired { current_epoch: 7, .. }));
}

#[test]
fn test_expiration_chain_mismatch_rejected() {
    let chain_a = fresh_chain_id();
    let chain_b: ChainIdentifier = crate::digests::CheckpointDigest::new([9u8; 32]).into();
    assert_ne!(chain_a, chain_b);
    let data = TransactionData::new_with_expiration(
        TransactionKind::Transfer {
            coins: vec![random_object_ref()],
            amounts: None,
            recipients: vec![SomaAddress::random()],
        },
        SomaAddress::random(),
        vec![random_object_ref()],
        TransactionExpiration::ValidDuring {
            min_epoch: Some(5),
            max_epoch: Some(5),
            chain: chain_a,
            nonce: 0,
        },
    );
    let err = data.check_expiration(5, &chain_b).expect_err("cross-chain replay must reject");
    assert!(matches!(err, SomaError::InvalidChainId { .. }));
}

#[test]
fn test_expiration_oversized_window_rejected() {
    // Width > 2 epochs would unbound the digest cache.
    let chain = fresh_chain_id();
    let data = TransactionData::new_with_expiration(
        TransactionKind::Transfer {
            coins: vec![random_object_ref()],
            amounts: None,
            recipients: vec![SomaAddress::random()],
        },
        SomaAddress::random(),
        vec![random_object_ref()],
        TransactionExpiration::ValidDuring {
            min_epoch: Some(5),
            max_epoch: Some(7), // width 3 — too wide
            chain,
            nonce: 0,
        },
    );
    let err = data.check_expiration(5, &chain).expect_err("width > 2 must reject");
    assert!(matches!(err, SomaError::UnsupportedFeatureError { .. }));
    assert!(!data.expiration().is_replay_protected(), "oversized window is not replay-protected");
}

#[test]
fn test_expiration_missing_min_or_max_rejected() {
    let chain = fresh_chain_id();
    let cases = [
        (None, Some(5)),
        (Some(5), None),
        (None, None),
    ];
    for (min_epoch, max_epoch) in cases {
        let data = TransactionData::new_with_expiration(
            TransactionKind::Transfer {
                coins: vec![random_object_ref()],
                amounts: None,
                recipients: vec![SomaAddress::random()],
            },
            SomaAddress::random(),
            vec![random_object_ref()],
            TransactionExpiration::ValidDuring { min_epoch, max_epoch, chain, nonce: 0 },
        );
        let err = data.check_expiration(5, &chain).expect_err("missing bound must reject");
        assert!(matches!(err, SomaError::UnsupportedFeatureError { .. }));
        assert!(!data.expiration().is_replay_protected());
    }
}

#[test]
fn test_expiration_nonce_distinguishes_otherwise_identical_txs() {
    // Different nonces produce different tx digests. This is the
    // mechanism that lets clients legitimately re-send "the same"
    // logical tx without colliding in the digest cache.
    let chain = fresh_chain_id();
    let sender = SomaAddress::random();
    let kind = TransactionKind::Transfer {
        coins: vec![random_object_ref()],
        amounts: None,
        recipients: vec![SomaAddress::random()],
    };
    let gas = vec![random_object_ref()];

    let d1 = TransactionData::new_with_expiration(
        kind.clone(),
        sender,
        gas.clone(),
        TransactionExpiration::ValidDuring {
            min_epoch: Some(5),
            max_epoch: Some(6),
            chain,
            nonce: 1,
        },
    );
    let d2 = TransactionData::new_with_expiration(
        kind,
        sender,
        gas,
        TransactionExpiration::ValidDuring {
            min_epoch: Some(5),
            max_epoch: Some(6),
            chain,
            nonce: 2,
        },
    );
    assert_ne!(d1.digest(), d2.digest(), "different nonces must produce different digests");
}

/// Critical: signing a tx with `ValidDuring` must produce a signature
/// that round-trips through verification. If the BCS encoding of the
/// new `expiration` field doesn't get included in the signed payload
/// — or the verification path skips it — signatures break for every
/// stateless tx. Catches Stage 5.5a regression.
#[test]
fn test_signed_transaction_with_valid_during_expiration() {
    let (sender, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let chain = fresh_chain_id();
    let data = TransactionData::new_with_expiration(
        TransactionKind::Transfer {
            coins: vec![random_object_ref()],
            amounts: Some(vec![1]),
            recipients: vec![SomaAddress::random()],
        },
        sender,
        Vec::new(),
        TransactionExpiration::ValidDuring {
            min_epoch: Some(0),
            max_epoch: Some(1),
            chain,
            nonce: 99,
        },
    );

    let tx = Transaction::from_data_and_signer(data.clone(), vec![&kp]);
    let inner_data = tx.data().transaction_data();
    assert_eq!(*inner_data, data, "TransactionData must round-trip through signed envelope");
    assert!(matches!(
        inner_data.expiration(),
        TransactionExpiration::ValidDuring { nonce: 99, .. }
    ));

    // The actual signature verification path (used by validators).
    crate::transaction::verify_sender_signed_data_message_signatures(tx.data())
        .expect("ValidDuring tx signature must verify");
}

/// Mimic what the e2e wallet does (`keystore.sign_secure` ⇒
/// `Signature::new_secure(&IntentMessage::new(intent, msg), key)`)
/// vs. what the validator does (BCS-hash IntentMessage<TransactionData>
/// and verify). Bytes signed and bytes verified must be identical for
/// any sig to validate.
///
/// This is a no-op-looking but load-bearing test: if it passes here
/// but the e2e fails, the divergence has to be in the test-cluster
/// chain_identifier or some msim quirk — NOT in the bytes themselves.
#[test]
fn test_e2e_wallet_signing_path_for_valid_during() {
    use crate::crypto::Signature;

    let (sender, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let chain = fresh_chain_id();
    let data = TransactionData::new_with_expiration(
        TransactionKind::Transfer {
            coins: vec![random_object_ref()],
            amounts: Some(vec![1]),
            recipients: vec![SomaAddress::random()],
        },
        sender,
        Vec::new(),
        TransactionExpiration::ValidDuring {
            min_epoch: Some(0),
            max_epoch: Some(1),
            chain,
            nonce: 0,
        },
    );

    // Sign mimicking the e2e wallet path: build IntentMessage<&T>,
    // sign with the keypair, wrap in Transaction::from_data.
    let sig = Signature::new_secure(
        &IntentMessage::new(Intent::soma_transaction(), &data),
        &kp,
    );
    let tx = Transaction::from_data(data.clone(), vec![sig]);

    // Verify via the validator's path. Same data, same intent → same bytes.
    crate::transaction::verify_sender_signed_data_message_signatures(tx.data())
        .expect("e2e-wallet-equivalent signing path must produce verifying signature");
}

#[test]
fn test_expiration_bcs_roundtrip() {
    let chain = fresh_chain_id();
    let data = TransactionData::new_with_expiration(
        TransactionKind::Transfer {
            coins: vec![random_object_ref()],
            amounts: None,
            recipients: vec![SomaAddress::random()],
        },
        SomaAddress::random(),
        vec![random_object_ref()],
        TransactionExpiration::ValidDuring {
            min_epoch: Some(10),
            max_epoch: Some(11),
            chain,
            nonce: 999_999,
        },
    );

    let bytes = bcs::to_bytes(&data).expect("BCS serialize");
    let decoded: TransactionData = bcs::from_bytes(&bytes).expect("BCS deserialize");
    assert_eq!(data, decoded);
    assert_eq!(data.digest(), decoded.digest());
}
