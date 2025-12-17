use e2e_tests::integration_helpers::{
    setup_integrated_encoder_validator_test, verify_encoder_committee_sync,
};
use rand::rngs::OsRng;
use rpc::proto::soma::InitiateShardWorkRequest;
use std::time::Duration;
use test_cluster::TestCluster;
use tokio::time::sleep;
use tracing::info;
use types::checksum::Checksum;
use types::crypto::{get_key_pair_from_rng, KeypairTraits, NetworkKeyPair};
use types::digests::TransactionDigest;
use types::effects::{self, TransactionEffects, TransactionEffectsAPI as _};
use types::full_checkpoint_content::ObjectSet;
use types::metadata::{DownloadMetadata, Metadata, MetadataAPI as _, MetadataV1, ObjectPath};
use types::shard::{Shard, ShardAuthToken};
use types::shard_crypto::digest::Digest;
use types::shard_crypto::keys::EncoderKeyPair;
use types::{
    base::SomaAddress,
    config::encoder_config::{EncoderConfig, EncoderGenesisConfigBuilder},
    crypto::SomaKeyPair,
    transaction::{AddEncoderArgs, Transaction, TransactionData, TransactionKind},
};
use utils::logging::init_tracing;

const ENCODER_STARTING_STAKE: u64 = 1_000_000_000_000_000;

#[cfg(msim)]
#[msim::sim_test]
async fn test_integrated_encoder_validator_system() {
    init_tracing();

    let initial_validators = 4;
    let initial_encoders = 4;

    // Generate keypairs for the new encoder (NOT the full genesis config)
    let mut rng = OsRng;
    let new_encoder_keypair = EncoderKeyPair::new(get_key_pair_from_rng(&mut rng).1);
    let new_account_keypair = SomaKeyPair::Ed25519(get_key_pair_from_rng(&mut rng).1);
    let new_network_keypair = NetworkKeyPair::new(get_key_pair_from_rng(&mut rng).1);

    // Get the address from the account keypair
    let new_encoder_address: SomaAddress = (&new_account_keypair.public()).into();

    // Set up test with the new encoder address as a candidate
    let (mut test_cluster, mut encoder_cluster) = setup_integrated_encoder_validator_test(
        initial_validators,
        initial_encoders,
        [new_encoder_address], // Include this address to receive gas at genesis
    )
    .await;

    // Verify initial encoder committee size
    let initial_size = test_cluster.get_encoder_committee_size();
    assert_eq!(
        initial_size, initial_encoders,
        "Initial encoder committee should have {} members",
        initial_encoders
    );

    // NOW create the encoder config - this will create and upload the probe
    let encoder_config = test_cluster
        .create_new_encoder_config(
            new_encoder_keypair.clone(),
            new_account_keypair.copy(),
            new_network_keypair.clone(),
        )
        .await;

    // Register the new encoder
    execute_add_encoder_transaction(&mut test_cluster, &encoder_config, new_encoder_address).await;

    // Stake the encoder
    execute_add_stake_transaction(
        new_account_keypair.copy(), // Use the keypair directly
        &mut test_cluster,
        new_encoder_address,
        ENCODER_STARTING_STAKE,
    )
    .await;

    // Wait for epoch 1 to process the encoder registration
    // NOTE: Triggering reconfiguration manually does not guarantee encoders will poll properly.
    test_cluster.wait_for_epoch(Some(1)).await;

    // Verify the new encoder is in the committee
    let new_size = test_cluster.get_encoder_committee_size();
    assert_eq!(
        new_size,
        initial_encoders + 1,
        "Encoder committee should have one more member after reconfiguration"
    );

    // Start the new encoder and verify sync
    let _encoder_handle = encoder_cluster.spawn_new_encoder(encoder_config).await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    verify_encoder_committee_sync(&encoder_cluster, initial_encoders + 1);

    let data_size = 1024 * 10; // 10KB of data
    let mut data = vec![0u8; data_size];
    rand::RngCore::fill_bytes(&mut rng, &mut data);

    // Upload data to test object server and get metadata
    let (metadata, download_metadata) = test_cluster.upload_test_data(&data).await;

    info!(
        checksum = %metadata.checksum(),
        size = metadata.size(),
        url = %download_metadata.url(),
        "Uploaded test data to object server"
    );

    // Embed data and wait for completion - single high-level call
    let (exec_response, completion) = test_cluster
        .wallet
        .embed_data_and_wait(
            &new_account_keypair,
            new_encoder_address,
            download_metadata,
            Duration::from_secs(60),
        )
        .await
        .expect("EmbedData and shard completion should succeed");

    info!(
        tx_digest = ?exec_response.effects.transaction_digest(),
        checkpoint = exec_response.checkpoint_sequence_number,
        shard_id = %completion.shard_id,
        winner_tx = %completion.winner_tx_digest,
        signers = completion.signers.len(),
        "Shard encoding completed successfully"
    );

    // Fetch the shard to get embedding metadata
    let client = test_cluster.wallet.get_client().await.unwrap();
    let shard = client
        .get_shard(completion.shard_id)
        .await
        .expect("Should fetch completed shard");

    // TODO: check that embeddings are available
    // assert!(
    //     shard.embeddings_download_metadata.is_some(),
    //     "Embeddings should be available"
    // );

    assert!(
        !completion.signers.is_empty(),
        "ReportWinner should have at least one signer"
    );
}

/// Execute EmbedData transaction
async fn execute_embed_data(
    signer: SomaKeyPair,
    test_cluster: &mut TestCluster,
    address: SomaAddress,
    download_metadata: DownloadMetadata,
) -> (TransactionEffects, ObjectSet) {
    // Get gas object for the transaction
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(address)
        .await
        .unwrap()
        .expect("Can't get gas object for encoder address");

    // Create EmbedData transaction
    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::EmbedData {
                download_metadata,
                coin_ref: gas_object,
                target_ref: None, // TODO: test with target
            },
            address,
            vec![gas_object],
        ),
        vec![&signer],
    );

    info!(
        tx_digest = ?tx.digest(),
        "Executing EmbedData transaction for {}",
        address
    );

    // Execute and wait for finalization
    let response = test_cluster.execute_transaction(tx).await;

    (response.effects, response.objects)
}

/// Execute AddEncoder transaction
async fn execute_add_encoder_transaction(
    test_cluster: &mut TestCluster,
    encoder_config: &EncoderConfig,
    encoder_address: SomaAddress,
) {
    // Get gas object for the transaction
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(encoder_address)
        .await
        .unwrap()
        .expect("Can't get gas object for encoder address");

    // Create and execute AddEncoder transaction
    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::AddEncoder(AddEncoderArgs {
                encoder_pubkey_bytes: bcs::to_bytes(
                    &encoder_config.encoder_keypair.encoder_keypair().public(),
                )
                .unwrap(),
                network_pubkey_bytes: bcs::to_bytes(&encoder_config.network_public_key()).unwrap(),
                external_network_address: bcs::to_bytes(&encoder_config.external_network_address)
                    .unwrap(),
                internal_network_address: bcs::to_bytes(&encoder_config.internal_network_address)
                    .unwrap(),
                object_server_address: bcs::to_bytes(&encoder_config.object_address).unwrap(),
                probe: bcs::to_bytes(&encoder_config.probe).unwrap(),
            }),
            encoder_address,
            vec![gas_object],
        ),
        vec![encoder_config.account_keypair.keypair()], // Sign with keypair
    );

    tracing::info!(?tx, "Executing add encoder tx for {}", encoder_address);
    let response = test_cluster.execute_transaction(tx).await;
}

/// Execute AddStake transaction for an encoder
async fn execute_add_stake_transaction(
    signer: SomaKeyPair,
    test_cluster: &mut TestCluster,
    address: SomaAddress,
    stake_amount: u64,
) {
    // Get gas object for the transaction
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(address)
        .await
        .unwrap()
        .expect("Can't get gas object for address");

    // Create and execute AddStake transaction
    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::AddStakeToEncoder {
                encoder_address: address,
                coin_ref: gas_object,
                amount: Some(stake_amount),
            },
            address,
            vec![gas_object],
        ),
        vec![&signer],
    );

    tracing::info!(?tx, "Executing stake encoder tx for {}", address);
    let response = test_cluster.execute_transaction(tx).await;
}

/// Initiate shard work after an EmbedData transaction has been checkpointed
async fn initiate_shard_work(
    test_cluster: &TestCluster,
    tx_digest: &TransactionDigest,
    checkpoint_seq: u64,
) -> Result<rpc::proto::soma::Shard, anyhow::Error> {
    let client = test_cluster.wallet.get_client().await?;

    let request = InitiateShardWorkRequest::default()
        .with_checkpoint_seq(checkpoint_seq)
        .with_tx_digest(tx_digest.to_string());

    let response = client
        .initiate_shard_work(request)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initiate shard work: {}", e))?;

    let shard = response
        .shard
        .ok_or_else(|| anyhow::anyhow!("No shard returned in response"))?;

    info!(
        tx_digest = %tx_digest,
        checkpoint = checkpoint_seq,
        "Initiated shard work"
    );

    Ok(shard)
}

/// Execute EmbedData transaction and wait for it to be checkpointed
/// Returns effects, objects, and the checkpoint sequence number
async fn execute_embed_data_and_wait(
    signer: SomaKeyPair,
    test_cluster: &mut TestCluster,
    address: SomaAddress,
    download_metadata: DownloadMetadata,
) -> (TransactionEffects, ObjectSet, u64) {
    // Get gas object for the transaction
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(address)
        .await
        .unwrap()
        .expect("Can't get gas object for encoder address");

    // Create EmbedData transaction
    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::EmbedData {
                download_metadata,
                coin_ref: gas_object,
                target_ref: None, // TODO: test with target
            },
            address,
            vec![gas_object],
        ),
        vec![&signer],
    );

    info!(
        tx_digest = ?tx.digest(),
        "Executing EmbedData transaction for {}",
        address
    );

    // Execute and wait for checkpointing (ensures we have checkpoint_seq)
    let response = test_cluster
        .wallet
        .execute_transaction_and_wait_for_indexing(tx)
        .await
        .expect("Transaction should execute and be checkpointed");

    (
        response.effects,
        response.objects,
        response.checkpoint_sequence_number,
    )
}
