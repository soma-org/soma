#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::needless_bool)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::unnecessary_unwrap)]

pub mod audit_service;
pub mod authority;
pub mod fullnode_proxy;
pub mod proxy_server;
pub mod authority_aggregator;
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
pub mod consensus_validator;
pub mod execution;
pub mod execution_driver;
pub mod execution_scheduler;
pub mod fallback_fetch;
pub mod global_state_hasher;
pub mod mysticeti_adapter;
pub mod reconfiguration;
pub mod rpc_index;
pub mod safe_client;
pub mod server;
pub mod shared_obj_version_manager;
pub mod signature_verifier;
pub mod stake_aggregator;
pub mod start_epoch;
pub mod status_aggregator;
pub mod storage;
pub mod submitted_transaction_cache;
pub mod test_authority_builder;
pub mod transaction_checks;
pub mod transaction_driver;
pub mod transaction_input_loader;
pub mod transaction_orchestrator;
pub mod transaction_reject_reason_cache;
pub mod validator_client_monitor;
pub mod validator_tx_finalizer;

#[cfg(test)]
#[path = "unit_tests/pay_coin_tests.rs"]
mod pay_coin_tests;

#[cfg(test)]
#[path = "unit_tests/gas_tests.rs"]
mod gas_tests;

#[cfg(test)]
#[path = "unit_tests/transfer_coin_tests.rs"]
mod transfer_coin_tests;

#[cfg(test)]
#[path = "unit_tests/staking_tests.rs"]
mod staking_tests;

#[cfg(test)]
#[path = "unit_tests/authority_tests.rs"]
mod authority_tests;

#[cfg(test)]
#[path = "unit_tests/validator_tests.rs"]
mod validator_tests;

#[cfg(test)]
#[path = "unit_tests/model_tests.rs"]
mod model_tests;

#[cfg(test)]
#[path = "unit_tests/submission_tests.rs"]
mod submission_tests;

#[cfg(test)]
#[path = "unit_tests/transaction_validation_tests.rs"]
mod transaction_validation_tests;

#[cfg(test)]
#[path = "unit_tests/epoch_tests.rs"]
mod epoch_tests;

#[cfg(test)]
#[path = "unit_tests/server_tests.rs"]
mod server_tests;

#[cfg(test)]
#[path = "unit_tests/batch_transaction_tests.rs"]
mod batch_transaction_tests;

#[cfg(test)]
#[path = "unit_tests/epoch_store_tests.rs"]
mod epoch_store_tests;

#[cfg(test)]
#[path = "unit_tests/execution_driver_tests.rs"]
mod execution_driver_tests;

#[cfg(test)]
#[path = "unit_tests/batch_verification_tests.rs"]
mod batch_verification_tests;

#[cfg(test)]
#[path = "unit_tests/consensus_tests.rs"]
mod consensus_tests;

#[cfg(test)]
pub(crate) mod consensus_test_utils;

// Tonic generated RPC stubs.
pub mod tonic_gen {
    include!("proto/validator.Validator.rs");
}
