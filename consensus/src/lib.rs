// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

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

#[cfg(test)]
pub(crate) mod commit_test_fixture;
#[cfg(test)]
#[path = "tests/randomized_tests.rs"]
mod randomized_tests;
/// Consensus test utilities.
#[cfg(test)]
mod test_dag;
mod test_dag_builder;
#[cfg(test)]
mod test_dag_parser;

pub use authority_node::{ConsensusAuthority, NetworkType};
pub use commit_consumer::{CommitConsumerArgs, CommitConsumerMonitor};
// Exported API for simtests.
#[cfg(any(test, msim))]
pub use network::tonic_network::to_socket_addr;
#[cfg(any(test, msim))]
pub use transaction::NoopTransactionVerifier;
pub use transaction::{
    BlockStatus, ClientError, TransactionClient, TransactionVerifier, ValidationError,
};

/// Simtests: integration tests using the msim deterministic simulator.
#[cfg(all(test, msim))]
#[path = "simtests/consensus_tests.rs"]
mod simtests;
