use encoder::types::shard_verifier::ShardAuthToken;
use std::time::Duration;
use test_encoder_cluster::TestEncoderClusterBuilder;
use tokio::time::sleep;
use utils::logging::init_tracing;

#[cfg(msim)]
#[msim::sim_test]
async fn test_encoder_cluster() {
    use test_encoder_cluster::create_valid_test_token;

    init_tracing();

    let cluster = TestEncoderClusterBuilder::new()
        .with_num_encoders(20)
        .build()
        .await;

    // Start all encoders
    cluster.start_all_encoders().await;

    let token = create_valid_test_token();

    // Get the shard determined by this token
    let shard = cluster.get_shard_from_token(&token).unwrap();

    println!("Shard contains these encoders: {:?}", shard.encoders());

    // Send input only to the encoders in the shard
    match cluster
        .send_to_shard_members(&token, Duration::from_secs(5))
        .await
    {
        Ok(_) => println!("Successfully sent input to all shard members"),
        Err(errors) => println!("Failed to send input to some shard members: {:?}", errors),
    }

    sleep(Duration::from_secs(60)).await;
}
