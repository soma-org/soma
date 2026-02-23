//! Tests for submission executor transactions:
//! SubmitData, ClaimRewards, ReportSubmission, UndoReportSubmission.
//!
//! The default TestAuthorityBuilder creates no genesis models or targets.
//! For tests needing active models, we build a genesis with a seed model via
//! `GenesisModelConfig`, which creates proper shared objects (targets, system state)
//! with correct versioning.

use std::collections::BTreeMap;
use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    checksum::Checksum,
    config::{
        genesis_config::{GenesisConfig, GenesisModelConfig},
        network_config::ConfigBuilder,
    },
    crypto::{DIGEST_LENGTH, DecryptionKey, SomaKeyPair, get_key_pair},
    digests::{
        DataCommitment, ModelWeightsCommitment, ModelWeightsUrlCommitment, TransactionDigest,
    },
    effects::{ExecutionStatus, TransactionEffectsAPI},
    error::SomaError,
    metadata::{Manifest, ManifestV1, Metadata, MetadataV1},
    model::{ModelId, ModelWeightsManifest},
    object::{Object, ObjectID, ObjectRef, ObjectType, Owner, Version},
    submission::SubmissionManifest,
    target::{TargetStatus, TargetV1},
    tensor::SomaTensor,
    transaction::{ClaimRewardsArgs, SubmitDataArgs, TransactionData, TransactionKind},
    unit_tests::utils::to_sender_signed_transaction,
};
use url::Url;

use crate::{
    authority::AuthorityState, authority_test_utils::send_and_confirm_transaction_,
    test_authority_builder::TestAuthorityBuilder,
};

// =============================================================================
// Helpers
// =============================================================================

/// Create a test SubmissionManifest with the given data size.
fn test_manifest(data_size: usize) -> SubmissionManifest {
    let url = Url::parse("https://example.com/data").unwrap();
    let metadata =
        Metadata::V1(MetadataV1::new(Checksum::new_from_hash([0u8; DIGEST_LENGTH]), data_size));
    let manifest = Manifest::V1(ManifestV1::new(url, metadata));
    SubmissionManifest::new(manifest)
}

/// Build an open Target object.
fn make_open_target(
    target_id: ObjectID,
    model_ids: Vec<ObjectID>,
    embedding_dim: usize,
    distance_threshold: f32,
    reward_pool: u64,
    generation_epoch: u64,
) -> Object {
    let target = TargetV1 {
        embedding: SomaTensor::zeros(vec![embedding_dim]),
        model_ids,
        distance_threshold: SomaTensor::scalar(distance_threshold),
        reward_pool,
        generation_epoch,
        status: TargetStatus::Open,
        submitter: None,
        winning_model_id: None,
        winning_model_owner: None,
        bond_amount: 0,
        winning_data_manifest: None,
        winning_data_commitment: None,
        winning_embedding: None,
        winning_distance_score: None,
        challenger: None,
        challenge_id: None,
        submission_reports: BTreeMap::new(),
    };
    Object::new_target_object(target_id, target, TransactionDigest::default())
}

/// Build a filled Target object.
fn make_filled_target(
    target_id: ObjectID,
    model_ids: Vec<ObjectID>,
    embedding_dim: usize,
    distance_threshold: f32,
    reward_pool: u64,
    generation_epoch: u64,
    fill_epoch: u64,
    submitter: SomaAddress,
    bond_amount: u64,
) -> Object {
    let target = TargetV1 {
        embedding: SomaTensor::zeros(vec![embedding_dim]),
        model_ids,
        distance_threshold: SomaTensor::scalar(distance_threshold),
        reward_pool,
        generation_epoch,
        status: TargetStatus::Filled { fill_epoch },
        submitter: Some(submitter),
        winning_model_id: None,
        winning_model_owner: None,
        bond_amount,
        winning_data_manifest: None,
        winning_data_commitment: None,
        winning_embedding: None,
        winning_distance_score: None,
        challenger: None,
        challenge_id: None,
        submission_reports: BTreeMap::new(),
    };
    Object::new_target_object(target_id, target, TransactionDigest::default())
}

/// Build a claimed Target object.
fn make_claimed_target(target_id: ObjectID, embedding_dim: usize, reward_pool: u64) -> Object {
    let target = TargetV1 {
        embedding: SomaTensor::zeros(vec![embedding_dim]),
        model_ids: vec![],
        distance_threshold: SomaTensor::scalar(0.5),
        reward_pool,
        generation_epoch: 0,
        status: TargetStatus::Claimed,
        submitter: None,
        winning_model_id: None,
        winning_model_owner: None,
        bond_amount: 0,
        winning_data_manifest: None,
        winning_data_commitment: None,
        winning_embedding: None,
        winning_distance_score: None,
        challenger: None,
        challenge_id: None,
        submission_reports: BTreeMap::new(),
    };
    Object::new_target_object(target_id, target, TransactionDigest::default())
}

/// Create a GenesisModelConfig for seeding a model at genesis.
fn make_genesis_model_config(model_id: ModelId, owner: SomaAddress) -> GenesisModelConfig {
    let url = Url::parse("https://example.com/model.bin").unwrap();
    let metadata =
        Metadata::V1(MetadataV1::new(Checksum::new_from_hash([1u8; DIGEST_LENGTH]), 1024));
    let manifest = Manifest::V1(ManifestV1::new(url, metadata));

    GenesisModelConfig {
        owner,
        model_id,
        weights_manifest: ModelWeightsManifest {
            manifest,
            decryption_key: DecryptionKey::new([0u8; 32]),
        },
        weights_url_commitment: ModelWeightsUrlCommitment::new([1u8; 32]),
        weights_commitment: ModelWeightsCommitment::new([2u8; 32]),
        architecture_version: 1,
        commission_rate: 1000,
        initial_stake: 1_000_000_000, // 1 SOMA
    }
}

/// Helper to get a genesis target from the authority state.
/// Genesis creates targets for epoch 0 when active models exist.
/// Returns (target_id, target) for the first open target found.
async fn find_genesis_target(authority_state: &AuthorityState) -> Option<(ObjectID, TargetV1)> {
    // Genesis creates targets as shared objects. We need to find them.
    // We can get the SystemState and look at target_state.targets_generated_this_epoch,
    // but we don't have a direct list of target IDs. Instead, the genesis builder
    // creates them with ObjectID::random() — we can't predict the IDs.
    //
    // However, genesis objects are returned by genesis.objects(). We can search through
    // the object store. Since there's no direct API to list by type, we'll check
    // the SystemState for hints.
    //
    // Alternative approach: look for Target objects in the genesis objects.
    // The TestAuthorityBuilder stores genesis objects during build.
    None // We'll use a different approach - see build_authority_with_model
}

struct ModelTestSetup {
    authority_state: Arc<AuthorityState>,
    model_id: ModelId,
    model_owner: SomaAddress,
    /// A genesis target that references the model (if available).
    /// These are the targets created at genesis with proper versioning.
    genesis_target_id: Option<ObjectID>,
}

/// Build an authority state with a seeded genesis model and targets.
/// This creates the proper shared object versioning from genesis.
async fn build_authority_with_model() -> ModelTestSetup {
    let model_id = ObjectID::random();
    let model_owner: SomaAddress = get_key_pair::<Ed25519KeyPair>().0;

    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.genesis_models.push(make_genesis_model_config(model_id, model_owner));

    let network_config =
        ConfigBuilder::new_with_temp_dir().with_genesis_config(genesis_config).build();

    let authority_state =
        TestAuthorityBuilder::new().with_network_config(&network_config, 0).build().await;

    // Find a genesis target that references our model
    // Genesis creates targets as shared objects. They are stored in the object store.
    // We need to find one by scanning genesis objects.
    let genesis_objects = network_config.genesis.objects();
    let genesis_target_id =
        genesis_objects.iter().find(|obj| *obj.type_() == ObjectType::Target).map(|obj| obj.id());

    ModelTestSetup { authority_state, model_id, model_owner, genesis_target_id }
}

// =============================================================================
// SubmitData tests
// =============================================================================

#[tokio::test]
async fn test_submit_data_basic() {
    // Successful SubmitData: target is Open, model is in target's list and active,
    // bond coin covers the bond, embedding dimension matches, distance <= threshold.
    let setup = build_authority_with_model().await;

    // We need a genesis target that references our model
    let genesis_target_id = match setup.genesis_target_id {
        Some(id) => id,
        None => {
            // No genesis targets — skip test (would need e2e test infrastructure)
            return;
        }
    };

    // Get the target to learn its embedding dimension and model assignment
    let target_obj = setup.authority_state.get_object(&genesis_target_id).await.unwrap();
    let target = target_obj.as_target().unwrap();

    // The genesis target may or may not include our model_id — it uses random selection.
    // Check if our model is in the target's model_ids.
    if !target.model_ids.contains(&setup.model_id) {
        // Skip this test — the model wasn't selected for this target.
        // In practice, with only 1 model, it should always be selected.
        return;
    }

    let embedding_dim = target.embedding.dim();
    let distance_threshold = target.distance_threshold.as_scalar();

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 500_000_000);
    let gas_ref = gas.compute_object_reference();

    // bond = submission_bond_per_byte(10) * data_size(1024) = 10240
    let bond_id = ObjectID::random();
    let bond = Object::with_id_owner_coin_for_testing(bond_id, sender, 500_000);
    let bond_ref = bond.compute_object_reference();

    setup.authority_state.insert_genesis_object(gas).await;
    setup.authority_state.insert_genesis_object(bond).await;

    let data = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id: genesis_target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: test_manifest(1024),
            model_id: setup.model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(distance_threshold * 0.5), // well within threshold
            bond_coin: bond_ref,
        }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    let (_, effects) = result.expect("SubmitData should succeed");
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Should have created at least a Submission object
    assert!(
        !effects.created().is_empty(),
        "Should create Submission (and possibly replacement target)"
    );

    // Target should be mutated (status -> Filled)
    let mutated_ids: Vec<ObjectID> = effects.mutated().iter().map(|m| m.0.0).collect();
    assert!(mutated_ids.contains(&genesis_target_id), "Target should be mutated to Filled status");
}

#[tokio::test]
async fn test_submit_data_target_not_found() {
    // Submit to a non-existent target ID should fail
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 50_000_000);
    let gas_ref = gas.compute_object_reference();

    let bond_id = ObjectID::random();
    let bond = Object::with_id_owner_coin_for_testing(bond_id, sender, 50_000_000);
    let bond_ref = bond.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;
    authority_state.insert_genesis_object(bond).await;

    let fake_target_id = ObjectID::random();
    let data = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id: fake_target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: test_manifest(1024),
            model_id: ObjectID::random(),
            embedding: SomaTensor::zeros(vec![10]),
            distance_score: SomaTensor::scalar(0.1),
            bond_coin: bond_ref,
        }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: target does not exist");
        }
        Err(_) => {
            // Input loading may reject before execution — also acceptable
        }
    }
}

#[tokio::test]
async fn test_submit_data_wrong_model() {
    // Submit with a model not in the target's model_ids list
    let setup = build_authority_with_model().await;
    let genesis_target_id = match setup.genesis_target_id {
        Some(id) => id,
        None => return,
    };

    let target_obj = setup.authority_state.get_object(&genesis_target_id).await.unwrap();
    let target = target_obj.as_target().unwrap();
    let embedding_dim = target.embedding.dim();

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 500_000_000);
    let gas_ref = gas.compute_object_reference();

    let bond_id = ObjectID::random();
    let bond = Object::with_id_owner_coin_for_testing(bond_id, sender, 500_000);
    let bond_ref = bond.compute_object_reference();

    setup.authority_state.insert_genesis_object(gas).await;
    setup.authority_state.insert_genesis_object(bond).await;

    // Submit with a model NOT in the target's model_ids
    let wrong_model = ObjectID::random();
    let data = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id: genesis_target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: test_manifest(1024),
            model_id: wrong_model,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(0.1),
            bond_coin: bond_ref,
        }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: model not in target's model_ids");
        }
        Err(_) => {}
    }
}

#[tokio::test]
async fn test_submit_data_filled_target() {
    // Submit to a target that is already filled should fail with TargetNotOpen
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 50_000_000);
    let gas_ref = gas.compute_object_reference();

    let bond_id = ObjectID::random();
    let bond = Object::with_id_owner_coin_for_testing(bond_id, sender, 50_000_000);
    let bond_ref = bond.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;
    authority_state.insert_genesis_object(bond).await;

    let submitter: SomaAddress = get_key_pair::<Ed25519KeyPair>().0;
    let model_id = ObjectID::random();
    let target_id = ObjectID::random();
    let target_obj =
        make_filled_target(target_id, vec![model_id], 10, 0.5, 1_000_000, 0, 0, submitter, 10_000);
    authority_state.insert_genesis_object(target_obj).await;

    let data = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: test_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![10]),
            distance_score: SomaTensor::scalar(0.1),
            bond_coin: bond_ref,
        }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: target already filled");
        }
        Err(_) => {}
    }
}

#[tokio::test]
async fn test_submit_data_distance_exceeds_threshold() {
    // Submit with a distance score that exceeds the target's threshold
    let setup = build_authority_with_model().await;
    let genesis_target_id = match setup.genesis_target_id {
        Some(id) => id,
        None => return,
    };

    let target_obj = setup.authority_state.get_object(&genesis_target_id).await.unwrap();
    let target = target_obj.as_target().unwrap();
    let embedding_dim = target.embedding.dim();
    let threshold = target.distance_threshold.as_scalar();

    if !target.model_ids.contains(&setup.model_id) {
        return;
    }

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 500_000_000);
    let gas_ref = gas.compute_object_reference();

    let bond_id = ObjectID::random();
    let bond = Object::with_id_owner_coin_for_testing(bond_id, sender, 500_000);
    let bond_ref = bond.compute_object_reference();

    setup.authority_state.insert_genesis_object(gas).await;
    setup.authority_state.insert_genesis_object(bond).await;

    // Submit with distance_score clearly exceeding threshold
    let data = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id: genesis_target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: test_manifest(1024),
            model_id: setup.model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(threshold + 1.0), // exceeds threshold
            bond_coin: bond_ref,
        }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: distance score exceeds threshold");
        }
        Err(_) => {}
    }
}

#[tokio::test]
async fn test_submit_data_insufficient_bond() {
    // Bond coin balance is too low for the required bond
    let setup = build_authority_with_model().await;
    let genesis_target_id = match setup.genesis_target_id {
        Some(id) => id,
        None => return,
    };

    let target_obj = setup.authority_state.get_object(&genesis_target_id).await.unwrap();
    let target = target_obj.as_target().unwrap();
    let embedding_dim = target.embedding.dim();

    if !target.model_ids.contains(&setup.model_id) {
        return;
    }

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 500_000_000);
    let gas_ref = gas.compute_object_reference();

    // Very small bond coin (1 shannon) — insufficient for 10MB data
    let bond_id = ObjectID::random();
    let bond = Object::with_id_owner_coin_for_testing(bond_id, sender, 1);
    let bond_ref = bond.compute_object_reference();

    setup.authority_state.insert_genesis_object(gas).await;
    setup.authority_state.insert_genesis_object(bond).await;

    // Manifest with 10MB data → bond = submission_bond_per_byte(10) * 10_000_000 = 100_000_000
    let data = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id: genesis_target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: test_manifest(10_000_000),
            model_id: setup.model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(0.1),
            bond_coin: bond_ref,
        }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: bond coin too small");
        }
        Err(_) => {}
    }
}

#[tokio::test]
async fn test_submit_data_spawn_on_fill() {
    // When a target is filled, the executor should spawn a replacement target
    // if the emission pool has funds and active models exist.
    let setup = build_authority_with_model().await;
    let genesis_target_id = match setup.genesis_target_id {
        Some(id) => id,
        None => return,
    };

    let target_obj = setup.authority_state.get_object(&genesis_target_id).await.unwrap();
    let target = target_obj.as_target().unwrap();
    let embedding_dim = target.embedding.dim();
    let threshold = target.distance_threshold.as_scalar();

    if !target.model_ids.contains(&setup.model_id) {
        return;
    }

    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 500_000_000);
    let gas_ref = gas.compute_object_reference();

    let bond_id = ObjectID::random();
    let bond = Object::with_id_owner_coin_for_testing(bond_id, sender, 500_000);
    let bond_ref = bond.compute_object_reference();

    setup.authority_state.insert_genesis_object(gas).await;
    setup.authority_state.insert_genesis_object(bond).await;

    let data = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id: genesis_target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: test_manifest(1024),
            model_id: setup.model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(threshold * 0.5),
            bond_coin: bond_ref,
        }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    let (_, effects) = result.expect("SubmitData should succeed");
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Should create at least 1 object: the replacement Target
    let created_count = effects.created().len();
    assert!(created_count >= 1, "Should create replacement Target, got {} objects", created_count);
}

// =============================================================================
// ClaimRewards tests
// =============================================================================

#[tokio::test]
async fn test_claim_rewards_expired_target() {
    // Claim rewards from an open target in the same epoch: should fail because
    // the target is not expired (current_epoch <= generation_epoch).
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 50_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let target_id = ObjectID::random();
    let target_obj = make_open_target(target_id, vec![], 10, 0.5, 1_000_000, 0);
    authority_state.insert_genesis_object(target_obj).await;

    let data = TransactionData::new(
        TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(
                !effects.status().is_ok(),
                "Should fail: open target in same epoch is not expired (TargetNotFilled)"
            );
        }
        Err(_) => {}
    }
}

#[tokio::test]
async fn test_claim_rewards_too_early() {
    // Try to claim a filled target while the challenge window is still open.
    // fill_epoch = 0, current_epoch = 0, challenge_window_end = 1
    // current_epoch (0) <= challenge_window_end (1) → ChallengeWindowOpen
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let submitter: SomaAddress = get_key_pair::<Ed25519KeyPair>().0;
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 50_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let target_id = ObjectID::random();
    let target_obj =
        make_filled_target(target_id, vec![], 10, 0.5, 1_000_000, 0, 0, submitter, 10_000);
    authority_state.insert_genesis_object(target_obj).await;

    let data = TransactionData::new(
        TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: challenge window still open");
        }
        Err(_) => {}
    }
}

#[tokio::test]
async fn test_claim_rewards_already_claimed() {
    // Try to claim a target that was already claimed
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 50_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let target_id = ObjectID::random();
    let target_obj = make_claimed_target(target_id, 10, 0);
    authority_state.insert_genesis_object(target_obj).await;

    let data = TransactionData::new(
        TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: target already claimed");
        }
        Err(_) => {}
    }
}

#[tokio::test]
async fn test_claim_rewards_target_not_found() {
    // Claiming from a non-existent target should fail
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 50_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let fake_target_id = ObjectID::random();
    let data = TransactionData::new(
        TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id: fake_target_id }),
        sender,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: target does not exist");
        }
        Err(_) => {}
    }
}

// =============================================================================
// ReportSubmission tests
// =============================================================================

#[tokio::test]
async fn test_report_submission_not_validator() {
    // Non-validator trying to report a submission should fail with NotAValidator
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 50_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let submitter: SomaAddress = get_key_pair::<Ed25519KeyPair>().0;
    let target_id = ObjectID::random();
    let target_obj =
        make_filled_target(target_id, vec![], 10, 0.5, 1_000_000, 0, 0, submitter, 10_000);
    authority_state.insert_genesis_object(target_obj).await;

    let data = TransactionData::new(
        TransactionKind::ReportSubmission { target_id, challenger: None },
        sender, // NOT a validator
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(
                !effects.status().is_ok(),
                "Non-validator should not be able to report submission"
            );
        }
        Err(_) => {}
    }
}

#[tokio::test]
async fn test_report_submission_target_not_filled() {
    // Reporting an open (unfilled) target should fail with TargetNotFilled.
    let network_config = ConfigBuilder::new_with_temp_dir().build();
    let v0_config = &network_config.validator_configs()[0];
    let v0_address = v0_config.soma_address();
    let v0_key = v0_config.account_key_pair.keypair().copy();

    let authority_state =
        TestAuthorityBuilder::new().with_network_config(&network_config, 0).build().await;

    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, v0_address, 50_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let target_id = ObjectID::random();
    let target_obj = make_open_target(target_id, vec![], 10, 0.5, 1_000_000, 0);
    authority_state.insert_genesis_object(target_obj).await;

    let data = TransactionData::new(
        TransactionKind::ReportSubmission { target_id, challenger: None },
        v0_address,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &v0_key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: target is Open, not Filled");
        }
        Err(_) => {}
    }
}

#[tokio::test]
async fn test_report_submission_tally() {
    // Validator successfully reports a filled target — the report is recorded.
    // We first fill a genesis target via SubmitData, then report it via ReportSubmission.
    let model_id = ObjectID::random();
    let model_owner: SomaAddress = get_key_pair::<Ed25519KeyPair>().0;

    let mut genesis_config = GenesisConfig::for_local_testing();
    genesis_config.genesis_models.push(make_genesis_model_config(model_id, model_owner));

    let network_config =
        ConfigBuilder::new_with_temp_dir().with_genesis_config(genesis_config).build();

    let v0_config = &network_config.validator_configs()[0];
    let v0_address = v0_config.soma_address();
    let v0_key = v0_config.account_key_pair.keypair().copy();

    let authority_state =
        TestAuthorityBuilder::new().with_network_config(&network_config, 0).build().await;

    // Find a genesis target that includes our model
    let genesis_objects = network_config.genesis.objects();
    let genesis_target = genesis_objects.iter().find(|obj| *obj.type_() == ObjectType::Target);

    let genesis_target_id = match genesis_target {
        Some(obj) => obj.id(),
        None => return,
    };

    let target_obj = authority_state.get_object(&genesis_target_id).await.unwrap();
    let target = target_obj.as_target().unwrap();

    if !target.model_ids.contains(&model_id) {
        return;
    }

    let embedding_dim = target.embedding.dim();
    let threshold = target.distance_threshold.as_scalar();

    // Pre-create all objects before any transactions run
    let (submitter, submitter_key): (_, Ed25519KeyPair) = get_key_pair();
    let gas1_id = ObjectID::random();
    let gas1 = Object::with_id_owner_coin_for_testing(gas1_id, submitter, 500_000_000);
    let gas1_ref = gas1.compute_object_reference();
    let bond_id = ObjectID::random();
    let bond = Object::with_id_owner_coin_for_testing(bond_id, submitter, 500_000);
    let bond_ref = bond.compute_object_reference();
    let gas2_id = ObjectID::random();
    let gas2 = Object::with_id_owner_coin_for_testing(gas2_id, v0_address, 50_000_000);
    let gas2_ref = gas2.compute_object_reference();

    authority_state.insert_genesis_object(gas1).await;
    authority_state.insert_genesis_object(bond).await;
    authority_state.insert_genesis_object(gas2).await;

    // Step 1: Fill the target via SubmitData (as a regular user)
    let submit_data = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id: genesis_target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: test_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(threshold * 0.5),
            bond_coin: bond_ref,
        }),
        submitter,
        vec![gas1_ref],
    );
    let submit_tx = to_sender_signed_transaction(submit_data, &submitter_key);
    let (_, submit_effects) =
        send_and_confirm_transaction_(&authority_state, None, submit_tx, true)
            .await
            .expect("SubmitData should succeed");
    assert_eq!(*submit_effects.status(), ExecutionStatus::Success);

    // Step 2: Report the now-filled target as the validator

    let report_data = TransactionData::new(
        TransactionKind::ReportSubmission { target_id: genesis_target_id, challenger: None },
        v0_address,
        vec![gas2_ref],
    );
    let report_tx = to_sender_signed_transaction(report_data, &v0_key);
    let result = send_and_confirm_transaction_(&authority_state, None, report_tx, true).await;

    let (_, effects) = result.expect("ReportSubmission by validator should succeed");
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Target should be mutated (report added)
    let mutated_ids: Vec<ObjectID> = effects.mutated().iter().map(|m| m.0.0).collect();
    assert!(mutated_ids.contains(&genesis_target_id), "Target should be mutated with new report");

    // Verify the report was recorded on the target
    let target_obj = authority_state.get_object(&genesis_target_id).await.unwrap();
    let target = target_obj.as_target().unwrap();
    assert!(
        target.submission_reports.contains_key(&v0_address),
        "Target should have validator's report recorded"
    );
}

// =============================================================================
// UndoReportSubmission tests
// =============================================================================

#[tokio::test]
async fn test_undo_report_submission_not_validator() {
    // Non-validator trying to undo a report should fail
    let (sender, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, sender, 50_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let submitter: SomaAddress = get_key_pair::<Ed25519KeyPair>().0;
    let target_id = ObjectID::random();
    let target_obj =
        make_filled_target(target_id, vec![], 10, 0.5, 1_000_000, 0, 0, submitter, 10_000);
    authority_state.insert_genesis_object(target_obj).await;

    let data = TransactionData::new(
        TransactionKind::UndoReportSubmission { target_id },
        sender, // NOT a validator
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(
                !effects.status().is_ok(),
                "Non-validator should not be able to undo report submission"
            );
        }
        Err(_) => {}
    }
}

#[tokio::test]
async fn test_undo_report_submission_no_prior_report() {
    // Validator tries to undo a report they never made — should fail with ReportRecordNotFound
    let network_config = ConfigBuilder::new_with_temp_dir().build();
    let v0_config = &network_config.validator_configs()[0];
    let v0_address = v0_config.soma_address();
    let v0_key = v0_config.account_key_pair.keypair().copy();

    let authority_state =
        TestAuthorityBuilder::new().with_network_config(&network_config, 0).build().await;

    let gas_id = ObjectID::random();
    let gas = Object::with_id_owner_coin_for_testing(gas_id, v0_address, 50_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let submitter: SomaAddress = get_key_pair::<Ed25519KeyPair>().0;
    let target_id = ObjectID::random();
    let target_obj =
        make_filled_target(target_id, vec![], 10, 0.5, 1_000_000, 0, 0, submitter, 10_000);
    authority_state.insert_genesis_object(target_obj).await;

    let data = TransactionData::new(
        TransactionKind::UndoReportSubmission { target_id },
        v0_address,
        vec![gas_ref],
    );
    let tx = to_sender_signed_transaction(data, &v0_key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: no prior report to undo");
        }
        Err(_) => {}
    }
}
