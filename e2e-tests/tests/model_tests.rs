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
use types::system_state::SystemStateTrait as _;
use types::tensor::SomaTensor;
use types::transaction::{
    CommitModelArgs, CreateModelArgs, RevealModelArgs, TransactionData, TransactionKind,
};
use url::Url;
use utils::logging::init_tracing;

// ===== Helpers =====

fn make_manifest(url_str: &str) -> Manifest {
    let url = Url::parse(url_str).expect("Invalid URL");
    let metadata = Metadata::V1(MetadataV1::new(Checksum::new_from_hash([1u8; 32]), 1024));
    Manifest::V1(ManifestV1::new(url, metadata))
}

fn make_genesis_model_config(owner: SomaAddress, initial_stake: u64) -> GenesisModelConfig {
    let manifest = make_manifest("https://example.com/models/genesis");
    GenesisModelConfig {
        owner,
        manifest,
        decryption_key: DecryptionKey::new([0xAA; 32]),
        weights_commitment: ModelWeightsCommitment::new([0xBB; 32]),
        architecture_version: 1,
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
    let initial_stake = 5 * SHANNONS_PER_SOMA;

    let model_config = make_genesis_model_config(model_owner, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    // Query system state from the fullnode
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    // Discover the auto-assigned model_id from active models
    let (&model_id, active_model) = system_state
        .model_registry()
        .active_models()
        .next()
        .expect("Genesis model should be active");
    assert_eq!(active_model.owner, model_owner);
    assert_eq!(active_model.architecture_version, 1);
    assert_eq!(active_model.commission_rate, 0);
    assert!(active_model.decryption_key != DecryptionKey::new([0u8; 32]));

    // Pool balance should reflect initial_stake
    assert_eq!(active_model.staking_pool.soma_balance, initial_stake);

    // total_model_stake should match
    assert_eq!(system_state.model_registry().total_model_stake, initial_stake);

    // Staking pool mapping should exist
    let pool_id = active_model.staking_pool.id;
    assert_eq!(system_state.model_registry().staking_pool_mappings.get(&pool_id), Some(&model_id),);

    // No pending or inactive models
    assert_eq!(system_state.model_registry().pending_models().count(), 0);
    assert_eq!(system_state.model_registry().inactive_models().count(), 0);

    info!("test_genesis_model_bootstrap passed");
}

// ===================================================================
// Test 2: Full create-commit-reveal round trip through consensus
//
// Exercises the complete model lifecycle via real transactions:
//   CreateModel → CommitModel → epoch change → RevealModel
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
    let embedding =
        SomaTensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], vec![10]);
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

    // ----- Step 1a: Submit CreateModel transaction -----

    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(model_owner)
        .await
        .unwrap()
        .expect("Model owner should have a gas object");

    let create_tx_data = TransactionData::new(
        TransactionKind::CreateModel(CreateModelArgs {
            architecture_version: 1,
            stake_amount,
            commission_rate: 0,
        }),
        model_owner,
        vec![gas_object],
    );

    let create_response = test_cluster.sign_and_execute_transaction(&create_tx_data).await;

    assert!(
        create_response.effects.status().is_ok(),
        "CreateModel transaction should succeed: {:?}",
        create_response.effects.status()
    );

    info!("CreateModel transaction succeeded");

    // Verify model is in Created state -- model_id is auto-generated by the executor,
    // so discover it by finding the newly created model owned by model_owner.
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    assert_eq!(
        system_state.model_registry().models.len(),
        1,
        "There should be exactly one model after CreateModel"
    );

    let (&model_id, model) = system_state.model_registry().models.iter().next().unwrap();

    assert!(model.is_created(), "Model should be in Created state");
    assert_eq!(model.owner(), model_owner);

    // ----- Step 1b: Submit CommitModel transaction -----

    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(model_owner)
        .await
        .unwrap()
        .expect("Model owner should have a gas object for commit");

    let commit_tx_data = TransactionData::new(
        TransactionKind::CommitModel(CommitModelArgs {
            model_id,
            manifest: manifest.clone(),
            weights_commitment,
            embedding_commitment,
            decryption_key_commitment,
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

    // Verify model is now in Pending state
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    let model = system_state.model_registry().models.get(&model_id).unwrap();
    assert!(model.is_pending(), "Model should be in Pending state after CommitModel");
    assert!(!model.is_active(), "Model should NOT be active yet");
    let pending_model = model.as_pending().unwrap();
    assert_eq!(pending_model.owner, model_owner);
    assert_eq!(pending_model.commit_epoch, 0);

    // ----- Step 2: Trigger epoch reconfiguration -----

    test_cluster.trigger_reconfiguration().await;

    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    // Model should still be in Pending state (reveal window is now open)
    assert!(
        system_state.model_registry().models.get(&model_id).unwrap().is_pending(),
        "Model should still be pending after first reconfig"
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

    let model = system_state.model_registry().models.get(&model_id).unwrap();
    assert!(!model.is_pending(), "Model should no longer be pending after reveal");
    assert!(model.is_active(), "Model should be active after reveal");

    let active_model = model.as_active().unwrap();
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
