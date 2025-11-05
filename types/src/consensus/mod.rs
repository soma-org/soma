//! # Consensus Types Module
//!
//! ## Overview
//! This module defines the core data structures and types used by the Soma blockchain's consensus
//! mechanism. It provides the foundation for Byzantine Fault Tolerant (BFT) agreement between
//! validators in the network.
//!
//! ## Responsibilities
//! - Define consensus transaction types and their serialization formats
//! - Provide structures for consensus blocks and commits
//! - Define interfaces for epoch management and validator set changes
//! - Support verification of consensus messages and state transitions
//! - Facilitate communication between consensus and other system components
//!
//! ## Component Relationships
//! - Used by the Consensus module to structure and process consensus messages
//! - Consumed by the Authority module to execute transactions in consensus order
//! - Provides input to the Node module for epoch transitions and reconfiguration
//! - Interfaces with the P2P module for network message propagation
//!
//! ## Key Workflows
//! 1. Consensus transaction submission and processing
//! 2. Block creation, verification, and commitment
//! 3. Epoch transitions and validator set changes
//! 4. Consensus state synchronization between nodes
//!
//! ## Design Patterns
//! - Type-safe enums for different consensus message types
//! - Trait-based interfaces for component interaction
//! - Immutable data structures for consensus state representation
//! - Test utilities for consensus verification

use crate::{
    accumulator::Accumulator,
    base::{AuthorityName, ConciseableName},
    committee::NetworkingCommittee,
    crypto::AuthorityPublicKeyBytes,
    digests::{ConsensusCommitDigest, ECMHLiveObjectSetDigest, TransactionDigest},
    encoder_committee::EncoderCommittee,
    state_sync::CommitTimestamp,
    transaction::CertifiedTransaction,
};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Formatter};
use validator_set::ValidatorSet;

pub mod block;
pub mod block_verifier;
pub mod commit;
pub mod committee;
pub mod context;
pub mod output;
pub mod stake_aggregator;
pub mod transaction;
pub mod validator_set;

/// # ConsensusTransaction
///
/// Represents a transaction that is processed by the consensus mechanism.
///
/// ## Purpose
/// Wraps different types of consensus messages (user transactions, end-of-publish markers)
/// in a common structure that can be processed by the consensus protocol.
///
/// ## Lifecycle
/// 1. Created by authorities when submitting transactions to consensus
/// 2. Processed by the consensus protocol to establish a total order
/// 3. Executed by the authority state after consensus commitment
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConsensusTransaction {
    /// The specific type of consensus transaction
    pub kind: ConsensusTransactionKind,
}

/// # ConsensusTransactionKind
///
/// Defines the different types of transactions that can be processed by the consensus mechanism.
///
/// ## Purpose
/// Distinguishes between user-submitted transactions and system-generated messages
/// that are needed for consensus operation.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ConsensusTransactionKind {
    /// A user-submitted transaction that has been certified by a quorum of validators
    UserTransaction(Box<CertifiedTransaction>),

    /// A message indicating that an authority has no more transactions to publish in this epoch
    /// Used for consensus liveness and epoch boundary detection
    EndOfPublish(AuthorityName),
    // CheckpointSignature(Box<CheckpointSignatureMessage>),
}

/// # ConsensusTransactionKey
///
/// A unique identifier for consensus transactions that can be used for deduplication and lookup.
///
/// ## Purpose
/// Provides a way to uniquely identify and reference consensus transactions,
/// which is essential for tracking transaction status and preventing duplicates.
#[derive(Serialize, Deserialize, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub enum ConsensusTransactionKey {
    /// Key for a user transaction certificate, identified by its digest
    Certificate(TransactionDigest),

    /// Key for an end-of-publish message, identified by the authority that sent it
    EndOfPublish(AuthorityName),
    // CheckpointSignature(AuthorityName, CheckpointSequenceNumber),
}

impl Debug for ConsensusTransactionKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Certificate(digest) => write!(f, "Certificate({:?})", digest),
            // Self::CheckpointSignature(name, seq) => {
            //     write!(f, "CheckpointSignature({:?}, {:?})", name.concise(), seq)
            // }
            Self::EndOfPublish(name) => write!(f, "EndOfPublish({:?})", name.concise()),
        }
    }
}

impl ConsensusTransaction {
    pub fn new_certificate_message(
        // authority: &AuthorityName,
        certificate: CertifiedTransaction,
    ) -> Self {
        Self {
            kind: ConsensusTransactionKind::UserTransaction(Box::new(certificate)),
        }
    }

    // pub fn new_checkpoint_signature_message(data: CheckpointSignatureMessage) -> Self {
    //     Self {
    //         kind: ConsensusTransactionKind::CheckpointSignature(Box::new(data)),
    //     }
    // }

    pub fn new_end_of_publish(authority: AuthorityName) -> Self {
        Self {
            kind: ConsensusTransactionKind::EndOfPublish(authority),
        }
    }

    pub fn new_mysticeti_certificate(
        round: u64,
        offset: u64,
        certificate: CertifiedTransaction,
    ) -> Self {
        Self {
            kind: ConsensusTransactionKind::UserTransaction(Box::new(certificate)),
        }
    }

    pub fn key(&self) -> ConsensusTransactionKey {
        match &self.kind {
            ConsensusTransactionKind::UserTransaction(cert) => {
                ConsensusTransactionKey::Certificate(*cert.digest())
            }
            // ConsensusTransactionKind::CheckpointSignature(data) => {
            //     ConsensusTransactionKey::CheckpointSignature(
            //         data.summary.auth_sig().authority,
            //         data.summary.sequence_number,
            //     )
            // }
            ConsensusTransactionKind::EndOfPublish(authority) => {
                ConsensusTransactionKey::EndOfPublish(*authority)
            }
        }
    }

    pub fn is_user_certificate(&self) -> bool {
        matches!(self.kind, ConsensusTransactionKind::UserTransaction(_))
    }

    pub fn is_end_of_publish(&self) -> bool {
        matches!(self.kind, ConsensusTransactionKind::EndOfPublish(_))
    }
}

/// # ConsensusCommitPrologue
///
/// Contains metadata about a consensus commit that is used to establish
/// the context for transaction execution.
///
/// ## Purpose
/// Provides essential information about the consensus state at the time of commit,
/// including timing information and cryptographic verification data.
///
/// ## Usage
/// Used by the authority state to properly sequence and timestamp transactions
/// during execution after consensus commitment.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct ConsensusCommitPrologue {
    /// Epoch of the commit prologue transaction
    pub epoch: u64,

    /// Consensus round of the commit
    pub round: u64,

    /// The sub DAG index of the consensus commit
    /// This field will be populated if there are multiple consensus commits per round
    pub sub_dag_index: Option<u64>,

    /// Unix timestamp from consensus (in milliseconds)
    pub commit_timestamp_ms: CommitTimestamp,

    /// Digest of consensus output for verification
    pub consensus_commit_digest: ConsensusCommitDigest,
}

/// # EndOfEpochAPI
///
/// Interface for accessing information about the next epoch during epoch transitions.
///
/// ## Purpose
/// Provides a way for the consensus mechanism to access information about the next epoch,
/// which is essential for coordinating epoch transitions and validator set changes.
///
/// ## Implementation
/// Implemented by components that manage epoch state, such as AuthorityPerEpochStore.
pub trait EndOfEpochAPI: Send + Sync + 'static {
    /// Returns the committee for the next epoch if one has been computed, along with the epoch state digest
    ///
    /// ## Returns
    /// - `Some((validator_set, digest, epoch))` if the next epoch state has been computed
    /// - `None` if the next epoch state has not yet been computed
    fn get_next_epoch_state(
        &self,
    ) -> Option<(
        ValidatorSet,
        EncoderCommittee,
        NetworkingCommittee,
        ECMHLiveObjectSetDigest,
        u64,
    )>;
}

/// # TestEpochStore
///
/// A simple implementation of EndOfEpochAPI for testing purposes.
///
/// ## Purpose
/// Provides a way to test epoch transition logic without requiring a full
/// implementation of the authority state and epoch store.
pub struct TestEpochStore {
    /// The next epoch state, if one has been computed
    pub next_epoch_state: Option<(
        ValidatorSet,
        EncoderCommittee,
        NetworkingCommittee,
        ECMHLiveObjectSetDigest,
        u64,
    )>,
}

impl TestEpochStore {
    pub fn new() -> Self {
        Self {
            next_epoch_state: None,
        }
    }

    pub fn set_next_epoch_state(
        &mut self,
        state: (
            ValidatorSet,
            EncoderCommittee,
            NetworkingCommittee,
            ECMHLiveObjectSetDigest,
            u64,
        ),
    ) {
        self.next_epoch_state = Some(state);
    }
}

impl EndOfEpochAPI for TestEpochStore {
    fn get_next_epoch_state(
        &self,
    ) -> Option<(
        ValidatorSet,
        EncoderCommittee,
        NetworkingCommittee,
        ECMHLiveObjectSetDigest,
        u64,
    )> {
        self.next_epoch_state.clone()
    }
}
