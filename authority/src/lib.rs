pub mod adapter;
pub mod aggregator;
pub mod authority_test_utils;
pub mod cache;
pub mod checkpoints;
pub mod client;
pub mod consensus_quarantine;
pub mod consensus_store_pruner;
pub mod encoder_client;
pub mod epoch_store;
pub mod epoch_store_pruner;
pub mod execution;
pub mod execution_driver;
pub mod global_state_hasher;
pub mod handler;
pub mod manager;
pub mod orchestrator;
pub mod quorum_driver;
pub mod reconfiguration;
pub mod rpc_index;
pub mod rpc_store;
pub mod safe_client;
pub mod server;
pub mod service;
pub mod shared_obj_version_manager;
pub mod stake_aggregator;
pub mod start_epoch;
pub mod state;
pub mod state_sync_store;
pub mod store;
pub mod store_pruner;
pub mod store_tables;
pub mod test_authority_builder;
pub mod throughput;
pub mod tx_input_loader;
pub mod tx_manager;
pub mod tx_validator;

#[cfg(test)]
#[path = "unit_tests/pay_coin_tests.rs"]
mod pay_coin_tests;

// Tonic generated RPC stubs.
pub mod tonic_gen {
    include!("proto/validator.Validator.rs");
}
