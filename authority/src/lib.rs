pub mod aggregator;
pub mod authority;
pub mod authority_client;
pub mod authority_per_epoch_store;
pub mod authority_per_epoch_store_pruner;
pub mod authority_server;
pub mod authority_store;
pub mod authority_store_pruner;
pub mod authority_store_tables;
pub mod authority_test_utils;
pub mod backpressure_manager;
pub mod cache;
pub mod checkpoints;
pub mod consensus_adapter;
pub mod consensus_handler;
pub mod consensus_manager;
pub mod consensus_output_api;
pub mod consensus_quarantine;
pub mod consensus_store_pruner;
pub mod consensus_tx_status_cache;
pub mod encoder_client;
pub mod execution;
pub mod execution_driver;
pub mod execution_scheduler;
pub mod fallback_fetch;
pub mod global_state_hasher;
pub mod mysticeti_adapter;
pub mod orchestrator;
pub mod quorum_driver;
pub mod reconfiguration;
pub mod rpc_index;
pub mod safe_client;
pub mod server;
pub mod shared_obj_version_manager;
pub mod signature_verifier;
pub mod stake_aggregator;
pub mod start_epoch;
pub mod storage;
pub mod submitted_transaction_cache;
pub mod test_authority_builder;
pub mod transaction_reject_reason_cache;
pub mod tx_input_loader;
pub mod tx_validator;
pub mod validator_tx_finalizer;

#[cfg(test)]
#[path = "unit_tests/pay_coin_tests.rs"]
mod pay_coin_tests;

// Tonic generated RPC stubs.
pub mod tonic_gen {
    include!("proto/validator.Validator.rs");
}
