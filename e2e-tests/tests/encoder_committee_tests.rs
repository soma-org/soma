use encoder::types::shard_verifier::ShardAuthToken;
use std::time::Duration;
use test_encoder_cluster::TestEncoderClusterBuilder;
use tokio::time::sleep;
use utils::logging::init_tracing;

#[cfg(msim)]
#[msim::sim_test]
async fn test_encoder_cluster() {
    init_tracing();

    let cluster = TestEncoderClusterBuilder::new()
        .with_num_encoders(4)
        .build()
        .await;

    // Start all encoders
    cluster.start_all_encoders().await;

    // Create and send input to all encoders in one step
    cluster
        .send_to_all_encoders(Duration::from_secs(2))
        .await
        .expect("Failed to send input to all encoders");

    sleep(Duration::from_secs(60)).await;
}
