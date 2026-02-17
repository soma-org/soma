use crate::{
    SYSTEM_STATE_OBJECT_ID, SYSTEM_STATE_OBJECT_SHARED_VERSION,
    base::SomaAddress,
    challenge::ChallengeId,
    checksum::Checksum,
    consensus::ConsensusCommitPrologue,
    crypto::{
        DecryptionKey, default_hash, get_key_pair,
    },
    digests::{
        AdditionalConsensusStateDigest, ConsensusCommitDigest, DataCommitment,
        ModelWeightsCommitment, ModelWeightsUrlCommitment, ObjectDigest,
    },
    envelope::Message,
    intent::{Intent, IntentMessage},
    metadata::{Manifest, ManifestV1, Metadata, MetadataV1},
    model::{ModelId, ModelWeightsManifest},
    object::{ObjectID, ObjectRef, Version},
    submission::SubmissionManifest,
    target::TargetId,
    tensor::SomaTensor,
    transaction::*,
    unit_tests::utils::to_sender_signed_transaction,
};
use fastcrypto::ed25519::Ed25519KeyPair;

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

/// Create a dummy Manifest for model / submission tests.
fn dummy_manifest() -> Manifest {
    let url = url::Url::parse("https://example.com/weights.bin").unwrap();
    let metadata = Metadata::V1(MetadataV1::new(Checksum::default(), 1024));
    Manifest::V1(ManifestV1::new(url, metadata))
}

/// Create a dummy ModelWeightsManifest.
fn dummy_model_weights_manifest() -> ModelWeightsManifest {
    ModelWeightsManifest {
        manifest: dummy_manifest(),
        decryption_key: DecryptionKey::new([0u8; 32]),
    }
}

/// Create a dummy SubmissionManifest.
fn dummy_submission_manifest() -> SubmissionManifest {
    SubmissionManifest::new(dummy_manifest())
}

/// Create a dummy SomaTensor (a single-element embedding).
fn dummy_tensor() -> SomaTensor {
    SomaTensor::new(vec![1.0, 2.0, 3.0], vec![3])
}

/// Create a dummy scalar SomaTensor for distance scores.
fn dummy_scalar_tensor() -> SomaTensor {
    SomaTensor::new(vec![0.5], vec![1])
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
    assert!(!genesis.is_model_tx());
    assert!(!genesis.is_submission_tx());
    assert!(!genesis.is_challenge_tx());

    let ccp = TransactionKind::ConsensusCommitPrologue(ConsensusCommitPrologue {
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
    });
    assert!(add_val.is_validator_tx());
    assert!(!add_val.is_system_tx());

    let remove_val = TransactionKind::RemoveValidator(RemoveValidatorArgs {
        pubkey_bytes: vec![1],
    });
    assert!(remove_val.is_validator_tx());

    let report_val = TransactionKind::ReportValidator {
        reportee: SomaAddress::random(),
    };
    assert!(report_val.is_validator_tx());

    let undo_report_val = TransactionKind::UndoReportValidator {
        reportee: SomaAddress::random(),
    };
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

    let withdraw_stake = TransactionKind::WithdrawStake {
        staked_soma: random_object_ref(),
    };
    assert!(withdraw_stake.is_staking_tx());

    // Model transactions
    let model_id = ModelId::random();
    let commit_model = TransactionKind::CommitModel(CommitModelArgs {
        model_id,
        weights_url_commitment: ModelWeightsUrlCommitment::random(),
        weights_commitment: ModelWeightsCommitment::random(),
        architecture_version: 1,
        stake_amount: 1000,
        commission_rate: 100,
        staking_pool_id: ObjectID::random(),
    });
    assert!(commit_model.is_model_tx());
    assert!(!commit_model.is_staking_tx());

    let reveal_model = TransactionKind::RevealModel(RevealModelArgs {
        model_id,
        weights_manifest: dummy_model_weights_manifest(),
        embedding: dummy_tensor(),
    });
    assert!(reveal_model.is_model_tx());

    let deactivate = TransactionKind::DeactivateModel { model_id };
    assert!(deactivate.is_model_tx());

    let report_model = TransactionKind::ReportModel { model_id };
    assert!(report_model.is_model_tx());

    let undo_report_model = TransactionKind::UndoReportModel { model_id };
    assert!(undo_report_model.is_model_tx());

    // Submission transactions
    let target_id: TargetId = ObjectID::random();
    let submit_data = TransactionKind::SubmitData(SubmitDataArgs {
        target_id,
        data_commitment: DataCommitment::random(),
        data_manifest: dummy_submission_manifest(),
        model_id,
        embedding: dummy_tensor(),
        distance_score: dummy_scalar_tensor(),
        bond_coin: random_object_ref(),
    });
    assert!(submit_data.is_submission_tx());
    assert!(!submit_data.is_model_tx());

    let claim_rewards = TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id });
    assert!(claim_rewards.is_submission_tx());

    let report_sub = TransactionKind::ReportSubmission {
        target_id,
        challenger: None,
    };
    assert!(report_sub.is_submission_tx());

    let undo_report_sub = TransactionKind::UndoReportSubmission { target_id };
    assert!(undo_report_sub.is_submission_tx());

    // Challenge transactions
    let challenge_id: ChallengeId = ObjectID::random();
    let init_challenge = TransactionKind::InitiateChallenge(InitiateChallengeArgs {
        target_id,
        bond_coin: random_object_ref(),
    });
    assert!(init_challenge.is_challenge_tx());
    assert!(!init_challenge.is_submission_tx());

    let report_chal = TransactionKind::ReportChallenge { challenge_id };
    assert!(report_chal.is_challenge_tx());

    let undo_report_chal = TransactionKind::UndoReportChallenge { challenge_id };
    assert!(undo_report_chal.is_challenge_tx());

    let claim_bond = TransactionKind::ClaimChallengeBond { challenge_id };
    assert!(claim_bond.is_challenge_tx());

    // Coin/object transactions should not match any category
    let transfer_coin = TransactionKind::TransferCoin {
        coin: random_object_ref(),
        amount: Some(100),
        recipient: SomaAddress::random(),
    };
    assert!(!transfer_coin.is_system_tx());
    assert!(!transfer_coin.is_validator_tx());
    assert!(!transfer_coin.is_staking_tx());
    assert!(!transfer_coin.is_model_tx());
    assert!(!transfer_coin.is_submission_tx());
    assert!(!transfer_coin.is_challenge_tx());
}

// ---------------------------------------------------------------------------
// 6. System transactions have no gas
// ---------------------------------------------------------------------------

#[test]
fn test_system_tx_has_no_gas() {
    // Genesis
    let genesis_data = make_system_tx_data(TransactionKind::Genesis(GenesisTransaction {
        objects: vec![],
    }));
    assert!(genesis_data.gas().is_empty());
    assert!(genesis_data.is_system_tx());

    // ConsensusCommitPrologue
    let ccp_data = make_system_tx_data(TransactionKind::ConsensusCommitPrologue(
        ConsensusCommitPrologue {
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
    let change_epoch_data =
        make_system_tx_data(TransactionKind::ChangeEpoch(ChangeEpoch {
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
        TransactionKind::AddStake {
            address: SomaAddress::random(),
            coin_ref,
            amount: Some(1000),
        },
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
    let model_id = ModelId::random();
    let target_id: TargetId = ObjectID::random();
    let challenge_id: ChallengeId = ObjectID::random();

    let kinds: Vec<TransactionKind> = vec![
        // System
        TransactionKind::Genesis(GenesisTransaction { objects: vec![] }),
        TransactionKind::ConsensusCommitPrologue(ConsensusCommitPrologue {
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
        }),
        TransactionKind::RemoveValidator(RemoveValidatorArgs {
            pubkey_bytes: vec![10],
        }),
        TransactionKind::ReportValidator {
            reportee: SomaAddress::random(),
        },
        TransactionKind::UndoReportValidator {
            reportee: SomaAddress::random(),
        },
        TransactionKind::UpdateValidatorMetadata(UpdateValidatorMetadataArgs::default()),
        TransactionKind::SetCommissionRate { new_rate: 100 },
        // Coin/object
        TransactionKind::TransferCoin {
            coin: random_object_ref(),
            amount: Some(500),
            recipient: SomaAddress::random(),
        },
        TransactionKind::PayCoins {
            coins: vec![random_object_ref(), random_object_ref()],
            amounts: Some(vec![100, 200]),
            recipients: vec![SomaAddress::random(), SomaAddress::random()],
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
        TransactionKind::WithdrawStake {
            staked_soma: random_object_ref(),
        },
        // Model
        TransactionKind::CommitModel(CommitModelArgs {
            model_id,
            weights_url_commitment: ModelWeightsUrlCommitment::random(),
            weights_commitment: ModelWeightsCommitment::random(),
            architecture_version: 1,
            stake_amount: 5000,
            commission_rate: 200,
            staking_pool_id: ObjectID::random(),
        }),
        TransactionKind::RevealModel(RevealModelArgs {
            model_id,
            weights_manifest: dummy_model_weights_manifest(),
            embedding: dummy_tensor(),
        }),
        TransactionKind::CommitModelUpdate(CommitModelUpdateArgs {
            model_id,
            weights_url_commitment: ModelWeightsUrlCommitment::random(),
            weights_commitment: ModelWeightsCommitment::random(),
        }),
        TransactionKind::RevealModelUpdate(RevealModelUpdateArgs {
            model_id,
            weights_manifest: dummy_model_weights_manifest(),
            embedding: dummy_tensor(),
        }),
        TransactionKind::AddStakeToModel {
            model_id,
            coin_ref: random_object_ref(),
            amount: Some(500),
        },
        TransactionKind::SetModelCommissionRate {
            model_id,
            new_rate: 300,
        },
        TransactionKind::DeactivateModel { model_id },
        TransactionKind::ReportModel { model_id },
        TransactionKind::UndoReportModel { model_id },
        // Submission
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: dummy_submission_manifest(),
            model_id,
            embedding: dummy_tensor(),
            distance_score: dummy_scalar_tensor(),
            bond_coin: random_object_ref(),
        }),
        TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id }),
        TransactionKind::ReportSubmission {
            target_id,
            challenger: Some(SomaAddress::random()),
        },
        TransactionKind::UndoReportSubmission { target_id },
        // Challenge
        TransactionKind::InitiateChallenge(InitiateChallengeArgs {
            target_id,
            bond_coin: random_object_ref(),
        }),
        TransactionKind::ReportChallenge { challenge_id },
        TransactionKind::UndoReportChallenge { challenge_id },
        TransactionKind::ClaimChallengeBond { challenge_id },
    ];

    assert_eq!(
        kinds.len(),
        31,
        "Expected 31 TransactionKind variants; if a new variant was added, update this test"
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
// 11. ConsensusCommitPrologue transaction classification
// ---------------------------------------------------------------------------

#[test]
fn test_consensus_commit_prologue_transaction() {
    let ccp = ConsensusCommitPrologue {
        epoch: 3,
        round: 42,
        sub_dag_index: None,
        commit_timestamp_ms: 999,
        consensus_commit_digest: ConsensusCommitDigest::new([7; 32]),
        additional_state_digest: AdditionalConsensusStateDigest::new([8; 32]),
    };
    let data = make_system_tx_data(TransactionKind::ConsensusCommitPrologue(ccp));

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
    let data = make_system_tx_data(TransactionKind::Genesis(GenesisTransaction {
        objects: vec![],
    }));

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
    let model_id = ModelId::random();
    let target_id: TargetId = ObjectID::random();
    let challenge_id: ChallengeId = ObjectID::random();

    // Validator tx -> SystemState only
    let add_val = TransactionKind::AddValidator(AddValidatorArgs {
        pubkey_bytes: vec![],
        network_pubkey_bytes: vec![],
        worker_pubkey_bytes: vec![],
        net_address: vec![],
        p2p_address: vec![],
        primary_address: vec![],
        proxy_address: vec![],
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

    // Model tx -> SystemState only
    let commit_model = TransactionKind::CommitModel(CommitModelArgs {
        model_id,
        weights_url_commitment: ModelWeightsUrlCommitment::random(),
        weights_commitment: ModelWeightsCommitment::random(),
        architecture_version: 1,
        stake_amount: 1000,
        commission_rate: 100,
        staking_pool_id: ObjectID::random(),
    });
    let shared: Vec<_> = commit_model.shared_input_objects().collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].id, SYSTEM_STATE_OBJECT_ID);

    // Submission tx -> SystemState + Target
    let submit_data = TransactionKind::SubmitData(SubmitDataArgs {
        target_id,
        data_commitment: DataCommitment::random(),
        data_manifest: dummy_submission_manifest(),
        model_id,
        embedding: dummy_tensor(),
        distance_score: dummy_scalar_tensor(),
        bond_coin: random_object_ref(),
    });
    let shared: Vec<_> = submit_data.shared_input_objects().collect();
    assert_eq!(shared.len(), 2);
    assert_eq!(shared[0].id, SYSTEM_STATE_OBJECT_ID);
    assert_eq!(shared[1].id, target_id);
    assert!(shared[1].mutable);

    // Challenge tx -> SystemState + Challenge
    let report_chal = TransactionKind::ReportChallenge { challenge_id };
    let shared: Vec<_> = report_chal.shared_input_objects().collect();
    assert_eq!(shared.len(), 2);
    assert_eq!(shared[0].id, SYSTEM_STATE_OBJECT_ID);
    assert_eq!(shared[1].id, challenge_id);

    // Genesis -> no shared input objects
    let genesis = TransactionKind::Genesis(GenesisTransaction { objects: vec![] });
    let shared: Vec<_> = genesis.shared_input_objects().collect();
    assert!(shared.is_empty());

    // TransferCoin -> no shared input objects
    let transfer = TransactionKind::TransferCoin {
        coin: random_object_ref(),
        amount: Some(100),
        recipient: SomaAddress::random(),
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
    let coin_count = inputs
        .iter()
        .filter(|inp| inp.object_id() == coin_ref.0)
        .count();
    assert_eq!(coin_count, 1, "Gas coin should not be duplicated in input_objects");
}

// ---------------------------------------------------------------------------
// 15. contains_shared_object
// ---------------------------------------------------------------------------

#[test]
fn test_contains_shared_object() {
    // TransferCoin does NOT touch shared state
    let transfer = TransactionKind::TransferCoin {
        coin: random_object_ref(),
        amount: Some(100),
        recipient: SomaAddress::random(),
    };
    assert!(
        !transfer.contains_shared_object(),
        "TransferCoin should not contain shared objects"
    );

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

    // SubmitData touches SystemState + Target
    let submit = TransactionKind::SubmitData(SubmitDataArgs {
        target_id: ObjectID::random(),
        data_commitment: DataCommitment::random(),
        data_manifest: dummy_submission_manifest(),
        model_id: ModelId::random(),
        embedding: dummy_tensor(),
        distance_score: dummy_scalar_tensor(),
        bond_coin: random_object_ref(),
    });
    assert!(
        submit.contains_shared_object(),
        "SubmitData should contain shared objects"
    );

    // ReportChallenge touches SystemState + Challenge
    let report_chal = TransactionKind::ReportChallenge {
        challenge_id: ObjectID::random(),
    };
    assert!(
        report_chal.contains_shared_object(),
        "ReportChallenge should contain shared objects"
    );

    // Genesis does NOT touch shared state
    let genesis = TransactionKind::Genesis(GenesisTransaction { objects: vec![] });
    assert!(
        !genesis.contains_shared_object(),
        "Genesis should not contain shared objects"
    );
}

// ---------------------------------------------------------------------------
// 16. TransactionData::execution_parts
// ---------------------------------------------------------------------------

#[test]
fn test_execution_parts() {
    let (data, sender) = make_transfer_coin_data();
    let (kind, exec_sender, gas) = data.execution_parts();

    assert_eq!(kind, data.kind);
    assert_eq!(exec_sender, sender);
    assert_eq!(gas, data.gas_payment);
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
    assert!(
        tx.verify_signature_for_testing().is_ok(),
        "Valid signature should pass verification"
    );
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
        TransactionKind::PayCoins {
            coins,
            amounts,
            recipients,
        } => {
            assert_eq!(coins.len(), 2);
            assert_eq!(*amounts, Some(vec![100, 200]));
            assert_eq!(recipients.len(), 2);
        }
        _ => panic!("Expected PayCoins kind"),
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
    });
    assert!(add_val.requires_system_state());

    // Staking requires system state
    let add_stake = TransactionKind::AddStake {
        address: SomaAddress::random(),
        coin_ref: random_object_ref(),
        amount: None,
    };
    assert!(add_stake.requires_system_state());

    // Model tx requires system state
    let deactivate = TransactionKind::DeactivateModel {
        model_id: ModelId::random(),
    };
    assert!(deactivate.requires_system_state());

    // Submission tx requires system state
    let claim = TransactionKind::ClaimRewards(ClaimRewardsArgs {
        target_id: ObjectID::random(),
    });
    assert!(claim.requires_system_state());

    // Challenge tx requires system state
    let claim_bond = TransactionKind::ClaimChallengeBond {
        challenge_id: ObjectID::random(),
    };
    assert!(claim_bond.requires_system_state());

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
    let transfer = TransactionKind::TransferCoin {
        coin: random_object_ref(),
        amount: Some(100),
        recipient: SomaAddress::random(),
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

    // CCP has no input objects
    let ccp = TransactionKind::ConsensusCommitPrologue(ConsensusCommitPrologue {
        epoch: 0,
        round: 1,
        sub_dag_index: None,
        commit_timestamp_ms: 0,
        consensus_commit_digest: ConsensusCommitDigest::new([0; 32]),
        additional_state_digest: AdditionalConsensusStateDigest::new([0; 32]),
    });
    let inputs = ccp.input_objects().expect("should succeed");
    assert!(
        inputs.is_empty(),
        "CCP should have no input objects (not validator/staking/model/submission/challenge)"
    );
}

// ---------------------------------------------------------------------------
// 25. input_objects for various user transactions
// ---------------------------------------------------------------------------

#[test]
fn test_input_objects_user_txs() {
    // TransferCoin: should have the coin as ImmOrOwnedObject
    let coin_ref = random_object_ref();
    let transfer = TransactionKind::TransferCoin {
        coin: coin_ref,
        amount: Some(100),
        recipient: SomaAddress::random(),
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

    // SubmitData: SystemState + bond_coin + target (shared)
    let target_id = ObjectID::random();
    let bond_coin = random_object_ref();
    let submit = TransactionKind::SubmitData(SubmitDataArgs {
        target_id,
        data_commitment: DataCommitment::random(),
        data_manifest: dummy_submission_manifest(),
        model_id: ModelId::random(),
        embedding: dummy_tensor(),
        distance_score: dummy_scalar_tensor(),
        bond_coin,
    });
    let inputs = submit.input_objects().expect("should succeed");
    assert_eq!(inputs.len(), 3);
    // Should contain system state, bond coin, and target
    let ids: Vec<ObjectID> = inputs.iter().map(|i| i.object_id()).collect();
    assert!(ids.contains(&SYSTEM_STATE_OBJECT_ID));
    assert!(ids.contains(&bond_coin.0));
    assert!(ids.contains(&target_id));
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
    assert_eq!(
        sys.id_and_version(),
        (SYSTEM_STATE_OBJECT_ID, SYSTEM_STATE_OBJECT_SHARED_VERSION)
    );
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
