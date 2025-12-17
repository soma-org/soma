use std::time::Duration;
use test_cluster::{TestCluster, TestClusterBuilder};
use test_encoder_cluster::{TestEncoderCluster, TestEncoderClusterBuilder};
use types::{
    base::SomaAddress,
    config::genesis_config::{AccountConfig, DEFAULT_GAS_AMOUNT},
};

// Re-export SDK types for convenience in tests
pub use rpc::api::client::{ShardCompletionInfo, ShardError};
pub use sdk::SomaClient;

/// Sets up an integrated test environment with validators and encoders.
pub async fn setup_integrated_encoder_validator_test(
    num_validators: usize,
    num_encoders: usize,
    encoder_candidates: impl IntoIterator<Item = SomaAddress>,
) -> (TestCluster, TestEncoderCluster) {
    let encoder_candidate_accounts = encoder_candidates
        .into_iter()
        .map(|address| AccountConfig {
            address: Some(address),
            gas_amounts: vec![DEFAULT_GAS_AMOUNT * 10],
        })
        .collect::<Vec<_>>();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(num_validators)
        .with_num_encoders(num_encoders)
        .with_accounts(encoder_candidate_accounts)
        .with_epoch_duration_ms(10 * 1000)
        .build()
        .await;

    let encoder_configs = test_cluster.get_encoder_configs_for_encoder_cluster();

    let encoder_cluster = TestEncoderClusterBuilder::new()
        .with_encoders(encoder_configs)
        .with_shared_object_store(test_cluster.object_store())
        .build()
        .await;

    (test_cluster, encoder_cluster)
}

/// Verify all encoders in a cluster see the expected committee size
pub fn verify_encoder_committee_sync(encoder_cluster: &TestEncoderCluster, expected_size: usize) {
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
                expected_size,
                "All encoders should see correct committee size"
            );
        });
    }
}
