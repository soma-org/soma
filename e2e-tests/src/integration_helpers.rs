use test_cluster::{TestCluster, TestClusterBuilder};
use test_encoder_cluster::{TestEncoderCluster, TestEncoderClusterBuilder};
use types::{
    base::SomaAddress,
    config::genesis_config::{AccountConfig, DEFAULT_GAS_AMOUNT},
};

/// Sets up an integrated test environment with validators and encoders.
pub async fn setup_integrated_encoder_validator_test(
    num_validators: usize,
    num_encoders: usize,
    encoder_candidates: impl IntoIterator<Item = SomaAddress>,
) -> (TestCluster, TestEncoderCluster) {
    // Create accounts for encoders with appropriate gas amounts
    let encoder_candidate_accounts = encoder_candidates
        .into_iter()
        .map(|address| AccountConfig {
            address: Some(address),
            gas_amounts: vec![DEFAULT_GAS_AMOUNT * 10], // Enough gas for transactions
        })
        .collect::<Vec<_>>();

    // Step 1: Build TestCluster with genesis encoders
    let test_cluster_builder = TestClusterBuilder::new()
        .with_num_validators(num_validators)
        .with_num_encoders(num_encoders)
        .with_accounts(encoder_candidate_accounts)
        .with_epoch_duration_ms(10 * 1000); // 10s

    // Build and start TestCluster
    let test_cluster = test_cluster_builder.build().await;

    // Step 2: Get the formatted encoder configs from TestCluster
    let encoder_configs = test_cluster.get_encoder_configs_for_encoder_cluster();

    // Step 3: Build TestEncoderCluster with the configs from TestCluster
    let encoder_cluster = TestEncoderClusterBuilder::new()
        .with_encoders(encoder_configs)
        .with_shared_object_store(test_cluster.object_store())
        .build()
        .await;

    (test_cluster, encoder_cluster)
}
