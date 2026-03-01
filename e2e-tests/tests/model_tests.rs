// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use test_cluster::TestClusterBuilder;
use tracing::info;
use types::base::SomaAddress;
use types::checksum::Checksum;
use types::config::genesis_config::{GenesisModelConfig, SHANNONS_PER_SOMA};
use types::crypto::DecryptionKey;
use types::digests::{DecryptionKeyCommitment, EmbeddingCommitment, ModelWeightsCommitment};
use types::effects::TransactionEffectsAPI;
use types::metadata::{Manifest, ManifestV1, Metadata, MetadataV1};
use types::model::ModelId;
use types::object::ObjectID;
use types::system_state::SystemStateTrait as _;
use types::tensor::SomaTensor;
use types::transaction::{CommitModelArgs, RevealModelArgs, TransactionData, TransactionKind};
use url::Url;
use utils::logging::init_tracing;

// ===== Helpers =====

fn make_manifest(url_str: &str) -> Manifest {
    let url = Url::parse(url_str).expect("Invalid URL");
    let metadata = Metadata::V1(MetadataV1::new(Checksum::new_from_hash([1u8; 32]), 1024));
    Manifest::V1(ManifestV1::new(url, metadata))
}

fn make_genesis_model_config(
    owner: SomaAddress,
    model_id: ModelId,
    initial_stake: u64,
) -> GenesisModelConfig {
    let url_str = format!("https://example.com/models/{}", model_id);
    let manifest = make_manifest(&url_str);
    let weights_commitment = ModelWeightsCommitment::new([0xBB; 32]);

    GenesisModelConfig {
        owner,
        model_id,
        manifest,
        decryption_key: DecryptionKey::new([0xAA; 32]),
        weights_commitment,
        architecture_version: 1,
        embedding_commitment: EmbeddingCommitment::new([0u8; 32]),
        decryption_key_commitment: DecryptionKeyCommitment::new([0u8; 32]),
        embedding: SomaTensor::zeros(vec![768]),
        commission_rate: 0,
        initial_stake,
    }
}

// ===================================================================
// Test 1: Genesis model bootstrap
//
// Verifies that a seed model configured at genesis is correctly
// created as active in the SystemState, with correct pool balance,
// staking pool mapping, and total_model_stake.
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_genesis_model_bootstrap() {
    init_tracing();

    // Use an arbitrary address as the model owner (doesn't need to transact)
    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let model_id = ObjectID::from_bytes([0x42; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;

    let model_config = make_genesis_model_config(model_owner, model_id, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    // Query system state from the fullnode
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    // Model should be in active_models (genesis models skip commit-reveal)
    assert!(
        system_state.model_registry().active_models.contains_key(&model_id),
        "Genesis model should be in active_models"
    );

    // Verify model fields
    let model = system_state.model_registry().active_models.get(&model_id).unwrap();
    assert_eq!(model.owner, model_owner);
    assert_eq!(model.architecture_version, 1);
    assert_eq!(model.commission_rate, 0);
    assert!(model.is_active());
    assert!(model.decryption_key.is_some());

    // Pool balance should reflect initial_stake
    assert_eq!(model.staking_pool.soma_balance, initial_stake);

    // total_model_stake should match
    assert_eq!(system_state.model_registry().total_model_stake, initial_stake);

    // Staking pool mapping should exist
    let pool_id = model.staking_pool.id;
    assert_eq!(system_state.model_registry().staking_pool_mappings.get(&pool_id), Some(&model_id),);

    // No pending or inactive models
    assert!(system_state.model_registry().pending_models.is_empty());
    assert!(system_state.model_registry().inactive_models.is_empty());

    info!("test_genesis_model_bootstrap passed");
}

// ===================================================================
// Test 2: Full commit-reveal round trip through consensus
//
// Exercises the complete model lifecycle via real transactions:
//   CommitModel → epoch change → RevealModel
// Verifies executor dispatch, BCS serialization through consensus,
// epoch-boundary model state handling, and state persistence.
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_model_commit_reveal_round_trip() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    // Use the first pre-generated wallet address as model owner
    // (already funded and its keypair is in the wallet keystore)
    let model_owner = test_cluster.get_addresses()[0];

    let stake_amount = 5 * SHANNONS_PER_SOMA;

    let url_str = "https://example.com/models/test-model";
    let manifest = make_manifest(url_str);
    let weights_commitment = ModelWeightsCommitment::new([0xBB; 32]);
    let embedding = SomaTensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], vec![10]);
    let embedding_commitment = {
        use fastcrypto::hash::HashFunction as _;
        let emb_bytes = bcs::to_bytes(&embedding).unwrap();
        let mut hasher = types::crypto::DefaultHash::default();
        hasher.update(&emb_bytes);
        let hash = hasher.finalize();
        EmbeddingCommitment::new(hash.as_ref().try_into().unwrap())
    };
    let decryption_key = DecryptionKey::new([0xAA; 32]);
    let decryption_key_commitment = {
        use fastcrypto::hash::HashFunction as _;
        let mut hasher = types::crypto::DefaultHash::default();
        hasher.update(decryption_key.as_bytes());
        let hash = hasher.finalize();
        DecryptionKeyCommitment::new(hash.as_ref().try_into().unwrap())
    };

    // ----- Step 1: Submit CommitModel transaction -----

    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(model_owner)
        .await
        .unwrap()
        .expect("Model owner should have a gas object");

    let commit_tx_data = TransactionData::new(
        TransactionKind::CommitModel(CommitModelArgs {
            manifest: manifest.clone(),
            weights_commitment,
            architecture_version: 1,
            embedding_commitment,
            decryption_key_commitment,
            stake_amount,
            commission_rate: 0,
        }),
        model_owner,
        vec![gas_object],
    );

    let commit_response = test_cluster.sign_and_execute_transaction(&commit_tx_data).await;

    assert!(
        commit_response.effects.status().is_ok(),
        "CommitModel transaction should succeed: {:?}",
        commit_response.effects.status()
    );

    info!("CommitModel transaction succeeded");

    // Verify model is in pending_models -- model_id is auto-generated by the executor,
    // so discover it by finding the newly created pending model owned by model_owner.
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    assert_eq!(
        system_state.model_registry().pending_models.len(),
        1,
        "There should be exactly one pending model after CommitModel"
    );

    let (&model_id, pending_model) =
        system_state.model_registry().pending_models.iter().next().unwrap();

    assert!(
        !system_state.model_registry().active_models.contains_key(&model_id),
        "Model should NOT be in active_models yet"
    );

    assert!(pending_model.is_committed());
    assert_eq!(pending_model.owner, model_owner);
    assert_eq!(pending_model.commit_epoch, 0);

    // ----- Step 2: Trigger epoch reconfiguration -----

    test_cluster.trigger_reconfiguration().await;

    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    // Model should still be in pending_models (reveal window is now open)
    assert!(
        system_state.model_registry().pending_models.contains_key(&model_id),
        "Model should still be in pending_models after first reconfig"
    );

    info!("Epoch 1 reached, model still pending — reveal window open");

    // ----- Step 3: Submit RevealModel transaction -----

    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(model_owner)
        .await
        .unwrap()
        .expect("Model owner should have a gas object for reveal");

    let reveal_tx_data = TransactionData::new(
        TransactionKind::RevealModel(RevealModelArgs {
            model_id,
            decryption_key,
            embedding: embedding.clone(),
        }),
        model_owner,
        vec![gas_object],
    );

    let reveal_response = test_cluster.sign_and_execute_transaction(&reveal_tx_data).await;

    assert!(
        reveal_response.effects.status().is_ok(),
        "RevealModel transaction should succeed: {:?}",
        reveal_response.effects.status()
    );

    info!("RevealModel transaction succeeded");

    // ----- Step 4: Verify model is now active -----

    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    assert!(
        !system_state.model_registry().pending_models.contains_key(&model_id),
        "Model should no longer be in pending_models after reveal"
    );
    assert!(
        system_state.model_registry().active_models.contains_key(&model_id),
        "Model should be in active_models after reveal"
    );

    let active_model = system_state.model_registry().active_models.get(&model_id).unwrap();
    assert!(active_model.is_active());
    assert!(active_model.decryption_key.is_some());
    assert_eq!(active_model.owner, model_owner);
    assert_eq!(active_model.architecture_version, 1);
    assert_eq!(active_model.staking_pool.soma_balance, stake_amount);

    // total_model_stake should reflect the newly active model
    assert_eq!(system_state.model_registry().total_model_stake, stake_amount,);

    // Staking pool mapping should exist
    let pool_id = active_model.staking_pool.id;
    assert_eq!(system_state.model_registry().staking_pool_mappings.get(&pool_id), Some(&model_id),);

    info!("test_model_commit_reveal_round_trip passed");
}
