use std::fmt::Display;

use protocol_config::ProtocolConfig;

use crate::committee::AuthorityIndex;
use crate::consensus::block::BlockRef;
use crate::consensus::{block::BlockAPI, commit::CommittedSubDag};
use crate::{consensus::ConsensusTransaction, digests::ConsensusCommitDigest};

/// A list of tuples of:
/// (certificate origin authority index, all transactions corresponding to the certificate).
/// For each transaction, returns the serialized transaction and the deserialized transaction.
type ConsensusOutputTransactions<'a> = Vec<(AuthorityIndex, Vec<(&'a [u8], ConsensusTransaction)>)>;

pub trait ConsensusOutputAPI: Display {
    fn leader_round(&self) -> u64;
    fn leader_author_index(&self) -> AuthorityIndex;
    fn leader_block_ref(&self) -> BlockRef;

    /// Returns epoch UNIX timestamp in milliseconds
    fn commit_timestamp_ms(&self) -> u64;

    /// Returns a unique global index for each committed sub-dag.
    fn commit_sub_dag_index(&self) -> u64;

    /// Returns all transactions in the commit.
    fn transactions(&self) -> ConsensusOutputTransactions<'_>;

    /// Returns the digest of consensus output.
    fn consensus_digest(&self, protocol_config: &ProtocolConfig) -> ConsensusCommitDigest;
}

impl ConsensusOutputAPI for CommittedSubDag {
    fn leader_round(&self) -> u64 {
        self.leader.round as u64
    }

    fn leader_author_index(&self) -> AuthorityIndex {
        AuthorityIndex::new_for_test(self.leader.author.value() as u32)
    }

    fn commit_timestamp_ms(&self) -> u64 {
        // TODO: Enforce ordered timestamp in Mysticeti.
        self.timestamp_ms
    }

    fn commit_sub_dag_index(&self) -> u64 {
        self.commit_ref.index.into()
    }

    fn leader_block_ref(&self) -> BlockRef {
        self.leader
    }

    fn transactions(&self) -> ConsensusOutputTransactions {
        self.blocks
            .iter()
            .map(|block| {
                let round = block.round();
                let author = AuthorityIndex::new_for_test(block.author().value() as u32);
                let transactions: Vec<_> = block
                    .transactions()
                    .iter()
                    .flat_map(|tx| {
                        let transaction = bcs::from_bytes::<ConsensusTransaction>(tx.data());
                        match transaction {
                            Ok(transaction) => Some((tx.data(), transaction)),
                            Err(err) => {
                                tracing::error!(
                                    "Failed to deserialize sequenced consensus transaction(this \
                                     should not happen) {} from {author} at {round}",
                                    err
                                );
                                None
                            }
                        }
                    })
                    .collect();
                (author, transactions)
            })
            .collect()
    }

    fn consensus_digest(&self, protocol_config: &ProtocolConfig) -> ConsensusCommitDigest {
        ConsensusCommitDigest::new(self.commit_ref.digest.into_inner())
    }
}
