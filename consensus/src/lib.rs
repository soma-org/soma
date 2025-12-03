/// Consensus modules.
mod ancestor;
mod authority_node;
mod authority_service;
mod base_committer;
mod block_manager;
mod block_verifier;
mod commit_consumer;
mod commit_finalizer;
mod commit_observer;
mod commit_syncer;
mod commit_vote_monitor;
mod core;
mod core_thread;
mod dag_state;
mod leader_schedule;
mod leader_timeout;
mod linearizer;
mod network;
mod proposed_block_handler;
mod round_prober;
mod round_tracker;
mod subscriber;
mod synchronizer;
mod threshold_clock;
mod transaction;
mod transaction_certifier;
mod universal_committer;

/// Consensus test utilities.
#[cfg(test)]
mod test_dag;
mod test_dag_builder;
#[cfg(test)]
mod test_dag_parser;

pub use authority_node::{ConsensusAuthority, NetworkType};
pub use commit_consumer::{CommitConsumerArgs, CommitConsumerMonitor};
pub use transaction::{
    BlockStatus, ClientError, TransactionClient, TransactionVerifier, ValidationError,
};

// Exported API for simtests.
#[cfg(msim)]
pub use network::tonic_network::to_socket_addr;
#[cfg(msim)]
pub use transaction::NoopTransactionVerifier;
