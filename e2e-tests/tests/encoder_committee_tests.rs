use e2e_tests::integration_helpers::setup_integrated_encoder_validator_test;
use rand::rngs::OsRng;
use std::time::Duration;
use test_cluster::TestCluster;
use tokio::time::sleep;
use tracing::info;
use types::checksum::Checksum;
use types::crypto::{KeypairTraits, NetworkKeyPair};
use types::metadata::{
    DownloadableMetadata, DownloadableMetadataV1, Metadata, MetadataCommitment, MetadataV1,
    ObjectPath,
};
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
    tokio::time::sleep(Duration::from_secs(3)).await;

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

    // TODO: define real Metadata and commitment
    let size_in_bytes = 1;
    let fullnode_config = test_cluster
        .fullnode_handle
        .soma_node
        .state()
        .config
        .clone();

    let tls_key = fullnode_config.network_key_pair();
    let address = fullnode_config.network_address;

    let checksum = Checksum::default();
    let epoch = 1;
    let object_path = ObjectPath::Inputs(epoch, checksum);

    let metadata = MetadataV1::new(object_path, size_in_bytes);
    let metadata_commitment = MetadataCommitment::new(Metadata::V1(metadata), [0u8; 32]);
    let digest = metadata_commitment
        .digest()
        .expect("Could not create Digest for MetadataCommitment");

    let shard = execute_embed_data_transaction(
        new_encoder_genesis.account_key_pair.copy(),
        &mut test_cluster,
        new_encoder_address,
        digest,
        size_in_bytes,
    )
    .await
    .expect("Could not get shard auth token from transaction execution");

    sleep(Duration::from_secs(5)).await;

    for handle in encoder_cluster.all_encoder_handles() {
        handle.with(|node| {
            let enc_key = node.get_config().encoder_keypair.encoder_keypair().public();
            if shard.encoders().contains(&enc_key) {
                let store = node.get_store_for_testing();

                let result = store.get_aggregate_score(&shard);

                assert!(
                    result.is_ok(),
                    "Failed to get agg sig for shard: {:?}",
                    result.err()
                );

                info!("Final agg sig: {:?}", result.unwrap().0);
            }
        });
    }
}

/// Execute EmbedData transaction
async fn execute_embed_data_transaction(
    signer: SomaKeyPair,
    test_cluster: &mut TestCluster,
    address: SomaAddress,
    digest: Digest<MetadataCommitment>,
    data_size_bytes: u64,
) -> Option<Shard> {
    // Get gas object for the transaction
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(address)
        .await
        .unwrap()
        .expect("Can't get gas object for encoder address");

    // Create and execute AddEncoder transaction
    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::EmbedData {
                digest: digest,
                data_size_bytes,
                coin_ref: gas_object,
            },
            address,
            vec![gas_object],
        ),
        vec![&signer], // Sign with keypair
    );

    tracing::info!(?tx, "Executing embed data tx for {}", address);
    let response = test_cluster.execute_transaction(tx).await;

    response.shard
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
                object_server_address: bcs::to_bytes(&encoder_config.external_object_address)
                    .unwrap(),
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
