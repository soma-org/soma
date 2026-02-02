use std::time::Duration;
use test_cluster::{TestCluster, TestClusterBuilder};
use types::{
    base::SomaAddress,
    config::genesis_config::{AccountConfig, DEFAULT_GAS_AMOUNT},
};

// Re-export SDK types for convenience in tests
pub use rpc::api::client::{ShardCompletionInfo, ShardError};
pub use sdk::SomaClient;