use std::num::NonZeroUsize;

use test_cluster::{TestCluster, TestClusterBuilder};
use test_encoder_cluster::{
    config::EncoderConfig, swarm::EncoderSwarm, TestEncoderCluster, TestEncoderClusterBuilder,
};
use types::{
    base::SomaAddress,
    config::genesis_config::{AccountConfig, DEFAULT_GAS_AMOUNT},
    crypto::SomaKeyPair,
    transaction::{AddEncoderArgs, Transaction, TransactionData, TransactionKind},
};

const ENCODER_STARTING_STAKE: u64 = 1_000_000_000_000_000;

/// Sets up an integrated test environment with validators and encoders.
pub async fn setup_integrated_encoder_validator_test(
    num_validators: usize,
    num_encoders: usize,
) -> (TestCluster, TestEncoderCluster) {
    // Step 1: Use EncoderSwarmBuilder to generate encoder configs
    let encoder_swarm_builder =
        EncoderSwarm::builder().committee_size(NonZeroUsize::new(num_encoders).unwrap());

    // Get the encoder configurations
    let encoder_configs = encoder_swarm_builder.generate_configs();

    // Step 2: Extract encoder addresses for account setup
    let encoder_addresses: Vec<SomaAddress> = encoder_configs
        .iter()
        .map(|config| (&config.account_keypair.keypair().public()).into())
        .collect();

    // Step 3: Build TestCluster with accounts for encoders
    let mut test_cluster_builder = TestClusterBuilder::new().with_num_validators(num_validators);

    // Add gas accounts for each encoder
    let encoder_accounts = encoder_addresses
        .into_iter()
        .map(|address| AccountConfig {
            address: Some(address),
            gas_amounts: vec![ENCODER_STARTING_STAKE * 10], // Enough gas for transactions
        })
        .collect::<Vec<_>>();

    test_cluster_builder = test_cluster_builder.with_accounts(encoder_accounts);

    // Set epoch duration
    test_cluster_builder = test_cluster_builder.with_epoch_duration_ms(1000);

    // Build and start TestCluster
    let mut test_cluster = test_cluster_builder.build().await;

    // Step 4: Register encoders in the TestCluster
    for config in &encoder_configs {
        let encoder_address = (&config.account_keypair.keypair().public()).into();

        // Execute AddEncoder transaction
        execute_add_encoder_transaction(&mut test_cluster, config, encoder_address).await;

        // Execute AddStake transaction
        execute_add_stake_transaction(
            config.account_keypair.keypair().copy(),
            &mut test_cluster,
            encoder_address,
            ENCODER_STARTING_STAKE, // Stake amount
        )
        .await;
    }

    // Step 5: Advance epoch to process encoder registrations
    test_cluster.trigger_reconfiguration().await;

    // Step 6: Extract necessary parameters from TestCluster for encoders
    let genesis = test_cluster.get_genesis();
    let committee_with_network = genesis.committee_with_network();
    let committee = committee_with_network.committee();

    // Get fullnode RPC address
    let validator_rpc_address = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.get_config().encoder_validator_address.clone());

    // Get epoch duration
    let epoch_duration = test_cluster
        .swarm
        .config()
        .genesis
        .system_object()
        .parameters
        .epoch_duration_ms;

    // Step 7: Update the encoder configs and build TestEncoderCluster
    let updated_encoder_configs = encoder_configs
        .into_iter()
        .map(|mut config| {
            // Update config with TestCluster parameters
            config.validator_rpc_address = validator_rpc_address.clone();
            config.genesis_committee = committee.clone();
            config.epoch_duration_ms = epoch_duration;
            config
        })
        .collect::<Vec<_>>();

    // Build encoder cluster with the updated configs
    let encoder_cluster = TestEncoderClusterBuilder::new()
        .with_encoders(updated_encoder_configs)
        .build()
        .await;

    (test_cluster, encoder_cluster)
}

/// Execute AddEncoder transaction
async fn execute_add_encoder_transaction(
    test_cluster: &mut TestCluster,
    encoder_config: &EncoderConfig,
    encoder_address: SomaAddress,
) {
    // Get gas object for the transaction
    let gas_object = test_cluster
        .get_gas_objects_owned_by_address(encoder_address, Some(1))
        .await
        .expect("Can't get gas object for encoder address");

    // Create and execute AddEncoder transaction
    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::AddEncoder(AddEncoderArgs {
                encoder_pubkey_bytes: bcs::to_bytes(
                    &encoder_config.encoder_keypair.encoder_keypair().public(),
                )
                .unwrap(),
                network_pubkey_bytes: bcs::to_bytes(&encoder_config.peer_public_key()).unwrap(),
                net_address: bcs::to_bytes(&encoder_config.internal_network_address).unwrap(),
                object_server_address: bcs::to_bytes(&encoder_config.object_address).unwrap(),
                // probe_address: bcs::to_bytes(&encoder_config.probe_address).unwrap(),
            }),
            encoder_address,
            gas_object,
        ),
        vec![encoder_config.account_keypair.keypair()], // Sign with keypair
    );

    tracing::info!(?tx, "Executing add encoder tx for {}", encoder_address);
    test_cluster.execute_transaction(tx).await;
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
        .get_gas_objects_owned_by_address(address, Some(1))
        .await
        .expect("Can't get gas object for address");

    // Create and execute AddStake transaction
    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::AddStakeToEncoder {
                encoder_address: address,
                coin_ref: gas_object[0],
                amount: Some(stake_amount),
            },
            address,
            gas_object,
        ),
        vec![&signer],
    );

    tracing::info!(?tx, "Executing stake encoder tx for {}", address);
    test_cluster.execute_transaction(tx).await;
}
