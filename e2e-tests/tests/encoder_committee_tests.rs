use std::time::Duration;
use test_encoder_cluster::TestEncoderClusterBuilder;
use tokio::time::sleep;
use utils::logging::init_tracing;

#[cfg(msim)]
#[msim::sim_test]
async fn test_encoder_cluster() {
    init_tracing();

    let test_cluster = TestEncoderClusterBuilder::new().build().await;

    sleep(Duration::from_millis(10000)).await;
}
