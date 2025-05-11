use e2e_tests::integration_helpers::setup_integrated_encoder_validator_test;
use encoder::types::shard_verifier::ShardAuthToken;
use std::time::Duration;
use test_encoder_cluster::{create_valid_test_token, TestEncoderClusterBuilder};
use tokio::time::sleep;
use tracing::info;
use utils::logging::init_tracing;

#[cfg(msim)]
#[msim::sim_test]
async fn test_integrated_encoder_validator_system() {
    init_tracing();

    // Set up integrated clusters with 4 validators and 4 encoders
    let (test_cluster, encoder_cluster) = setup_integrated_encoder_validator_test(
        4, // num_validators
        4, // num_encoders
    )
    .await;

    // Verify encoder committee sync
    encoder_cluster
        .all_encoder_handles()
        .first()
        .unwrap()
        .with(|node| {
            let context = node.context.clone();
            let inner_context = context.inner();

            // Check the committee information matches what's in the test cluster
            let committees = inner_context.committees(1).unwrap(); // For epoch 1
            let encoder_committee = &committees.encoder_committee;

            // Verify encoder committee members match what we registered
            assert_eq!(encoder_committee.size(), 4);
        });

    // Create a valid token for a shard transaction
    let token = create_valid_test_token();

    // Get the shard based on the token
    let shard = encoder_cluster.get_shard_from_token(&token).unwrap();

    info!("Shard contains these encoders: {:?}", shard.encoders());

    // Send a transaction to the shard members
    let result = encoder_cluster
        .send_to_shard_members(&token, Duration::from_secs(5))
        .await;

    assert!(
        result.is_ok(),
        "Failed to send input to shard members: {:?}",
        result.err()
    );

    sleep(Duration::from_secs(30)).await;

    // // Trigger epoch change to test reconfiguration
    // test_cluster.trigger_reconfiguration().await;

    // // Verify encoders update their committee information
    // // ...

    // Cleanup
    test_cluster.stop_all_validators().await;
    encoder_cluster.stop_all_encoders().await;
}
