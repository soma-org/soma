use e2e_tests::integration_helpers::{
    extract_shard_input_id, setup_integrated_encoder_validator_test, wait_for_shard_completion,
};
use rand::rngs::OsRng;
use rpc::proto::soma::InitiateShardWorkRequest;
use std::time::Duration;
use test_cluster::TestCluster;
use tokio::time::sleep;
use tracing::info;
use types::checksum::Checksum;
use types::crypto::{KeypairTraits, NetworkKeyPair};
use types::digests::TransactionDigest;
use types::effects::{self, TransactionEffects, TransactionEffectsAPI as _};
use types::full_checkpoint_content::ObjectSet;
use types::metadata::{DownloadMetadata, Metadata, MetadataAPI as _, MetadataV1, ObjectPath};
use types::shard::{Shard, ShardAuthToken};
use types::shard_crypto::digest::Digest;
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

    // Generate a new encoder for later addition
    let mut rng = OsRng;
    let new_encoder_genesis = EncoderGenesisConfigBuilder::new().build(&mut rng);
    let new_encoder_address = (&new_encoder_genesis.account_key_pair.public()).into();

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

    // Create the encoder config for the new encoder
    let encoder_config = test_cluster.create_new_encoder_config(
        new_encoder_genesis.encoder_key_pair.clone(),
        new_encoder_genesis.account_key_pair.copy(),
        new_encoder_genesis.network_key_pair.clone(),
    );

    // Register the new encoder
    execute_add_encoder_transaction(&mut test_cluster, &encoder_config, new_encoder_address).await;

    // Stake the encoder
    execute_add_stake_transaction(
        new_encoder_genesis.account_key_pair.copy(),
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

    // Start the new encoder in the encoder cluster
    let _encoder_handle = encoder_cluster.spawn_new_encoder(encoder_config).await;

    // Wait for a moment to allow the encoder to sync
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Verify all encoders are in sync with the new committee
    for handle in encoder_cluster.all_encoder_handles() {
        handle.with(|node| {
            let context = node.context.clone();
            let inner_context = context.inner();
            let epoch = inner_context.current_epoch;
            let committees = inner_context
                .committees(epoch)
                .expect("Should have committees data");

            assert_eq!(
                committees.encoder_committee.size(),
                initial_encoders + 1,
                "All encoders should see correct committee size"
            );
        });
    }

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

    // Execute EmbedData transaction AND wait for it to be checkpointed
    // This is important because initiate_shard_work needs the checkpoint_seq
    // TODO: modify wait for tx to get sequence number
    let (effects, objects, checkpoint_seq) = execute_embed_data_and_wait(
        new_encoder_genesis.account_key_pair.copy(),
        &mut test_cluster,
        new_encoder_address,
        download_metadata.clone(),
    )
    .await;

    let tx_digest = effects.transaction_digest();

    info!(
        tx_digest = ?tx_digest,
        checkpoint = checkpoint_seq,
        status = ?effects.status(),
        "EmbedData transaction executed and checkpointed"
    );

    // Extract the ShardInput object ID from the effects
    let shard_input_id = extract_shard_input_id(&effects)
        .expect("Should find ShardInput object in EmbedData effects");

    info!(shard_input = %shard_input_id, "Created ShardInput object");

    // Initiate shard work with the transaction digest and checkpoint sequence
    let shard = initiate_shard_work(&test_cluster, tx_digest, checkpoint_seq)
        .await
        .expect("Should initiate shard work");

    info!("Shard work initiated, shard: {:?}", shard);

    // Get the client for subscription
    let client = test_cluster
        .wallet
        .get_client()
        .await
        .expect("Should get RPC client");

    // Wait for the shard to complete (ReportWinner transaction)
    let completion_info =
        wait_for_shard_completion(&client, &shard_input_id, Duration::from_secs(60))
            .await
            .expect("Should receive ReportWinner transaction");

    info!(
        winner_tx = %completion_info.winner_tx_digest,
        checkpoint = completion_info.checkpoint_sequence,
        signers = ?completion_info.signers,
        "Shard encoding completed successfully"
    );

    // Verify we got a valid completion
    assert!(
        !completion_info.signers.is_empty(),
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
                // probe_address: bcs::to_bytes(&encoder_config.probe_address).unwrap(),
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
    // TODO: Get the checkpoint sequence, the checkpoint_seq should be available after waiting for indexing
    let response = test_cluster
        .wallet
        .execute_transaction_and_wait_for_indexing(tx)
        .await
        .expect("Transaction should execute and be checkpointed");

    (response.effects, response.objects, 0)
}
