use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use store::TypedStoreError;

use crate::{
    accumulator::CommitIndex,
    balance_change::{derive_balance_changes, BalanceChange},
    base::SomaAddress,
    committee::{Committee, EpochId},
    consensus::{
        commit::{CommitAPI as _, CommitDigest, CommittedSubDag},
        output::ConsensusOutputAPI as _,
        ConsensusTransactionKind,
    },
    digests::TransactionDigest,
    effects::TransactionEffects,
    object::{Object, ObjectID, ObjectType, Version},
    transaction::{TransactionData, VerifiedTransaction},
};

use super::{object_store::ObjectStore, storage_error::Result};

pub trait ReadStore: ReadCommitteeStore + ObjectStore + Send + Sync {
    //
    // Commit Getters
    //

    fn get_latest_commit(&self) -> Result<CommittedSubDag>;

    /// Get the highest synced commit. This is the highest commit that has been synced from
    /// state-sync.
    fn get_highest_synced_commit(&self) -> Result<CommittedSubDag>;

    /// Lowest available commit for which transaction data can be requested.
    fn get_lowest_available_commit(&self) -> Result<CommitIndex>;

    fn get_commit_by_digest(&self, digest: &CommitDigest) -> Option<CommittedSubDag>;

    fn get_commit_by_index(&self, index: CommitIndex) -> Option<CommittedSubDag>;

    fn get_last_commit_index_of_epoch(&self, epoch: EpochId) -> Option<CommitIndex>;

    // TODO: Fix this
    // fn get_checkpoint_data(&self, committed_sub_dag: &CommittedSubDag) -> Result<Checkpoint> {
    //     let all_tx_digests: HashSet<_> = committed_sub_dag
    //         .transactions()
    //         .iter()
    //         .flat_map(|(_, authority_transactions)| {
    //             authority_transactions
    //                 .iter()
    //                 .filter_map(|(_, transaction)| {
    //                     if let ConsensusTransactionKind::UserTransaction(cert_tx) =
    //                         &transaction.kind
    //                     {
    //                         Some(*cert_tx.digest())
    //                     } else {
    //                         None
    //                     }
    //                 })
    //         })
    //         .collect();

    //     // Convert to Vec to maintain consistent ordering
    //     let tx_digests: Vec<_> = all_tx_digests.into_iter().collect();

    //     // Fetch all transactions
    //     let transactions = self
    //         .multi_get_transactions(&tx_digests)?
    //         .into_iter()
    //         .zip(&tx_digests)
    //         .map(|(maybe_tx, digest)| {
    //             maybe_tx.ok_or_else(|| {
    //                 // Create appropriate error for missing transaction
    //                 crate::storage::storage_error::Error::custom(format!(
    //                     "Missing transaction for digest: {}",
    //                     digest
    //                 ))
    //             })
    //         })
    //         .collect::<Result<Vec<_>>>()?;

    //     // Fetch all effects
    //     let effects = self
    //         .multi_get_transaction_effects(&tx_digests)?
    //         .into_iter()
    //         .zip(&tx_digests)
    //         .map(|(maybe_effects, digest)| {
    //             maybe_effects.ok_or_else(|| {
    //                 crate::storage::storage_error::Error::custom(format!(
    //                     "Missing effects for digest: {}",
    //                     digest
    //                 ))
    //             })
    //         })
    //         .collect::<Result<Vec<_>>>()?;

    //     // Build ExecutedTransaction objects
    //     let executed_transactions = transactions
    //         .into_iter()
    //         .zip(effects)
    //         .map(|(tx, fx)| ExecutedTransaction {
    //             transaction: tx.transaction_data().clone(),
    //             effects: fx,
    //         })
    //         .collect();

    //     Ok(Checkpoint {
    //         commit_index: committed_sub_dag.commit_ref.index,
    //         timestamp_ms: committed_sub_dag.timestamp_ms,
    //         transactions: executed_transactions,
    //     })
    // }

    //
    // Transaction Getters
    //

    fn get_transaction(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Result<Option<Arc<VerifiedTransaction>>>;

    fn multi_get_transactions(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Result<Vec<Option<Arc<VerifiedTransaction>>>> {
        tx_digests
            .iter()
            .map(|digest| self.get_transaction(digest))
            .collect::<Result<Vec<_>, _>>()
    }

    fn get_transaction_effects(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Result<Option<TransactionEffects>>;

    fn multi_get_transaction_effects(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Result<Vec<Option<TransactionEffects>>> {
        tx_digests
            .iter()
            .map(|digest| self.get_transaction_effects(digest))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<T: ReadStore + ?Sized> ReadStore for &T {
    fn get_latest_commit(&self) -> Result<CommittedSubDag> {
        (*self).get_latest_commit()
    }

    fn get_highest_synced_commit(&self) -> Result<CommittedSubDag> {
        (*self).get_highest_synced_commit()
    }

    fn get_lowest_available_commit(&self) -> Result<CommitIndex> {
        (*self).get_lowest_available_commit()
    }

    fn get_commit_by_digest(&self, digest: &CommitDigest) -> Option<CommittedSubDag> {
        (*self).get_commit_by_digest(digest)
    }

    fn get_commit_by_index(&self, index: CommitIndex) -> Option<CommittedSubDag> {
        (*self).get_commit_by_index(index)
    }

    fn get_last_commit_index_of_epoch(&self, epoch: EpochId) -> Option<CommitIndex> {
        (*self).get_last_commit_index_of_epoch(epoch)
    }

    fn get_transaction(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Result<Option<Arc<VerifiedTransaction>>> {
        (*self).get_transaction(tx_digest)
    }

    fn multi_get_transactions(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Result<Vec<Option<Arc<VerifiedTransaction>>>> {
        (*self).multi_get_transactions(tx_digests)
    }

    fn get_transaction_effects(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Result<Option<TransactionEffects>> {
        (*self).get_transaction_effects(tx_digest)
    }

    fn multi_get_transaction_effects(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> Result<Vec<Option<TransactionEffects>>> {
        (*self).multi_get_transaction_effects(tx_digests)
    }
}

pub trait ReadCommitteeStore: Send + Sync {
    fn get_committee(&self, epoch: EpochId) -> Result<Option<Arc<Committee>>>;
}

impl<T: ReadCommitteeStore + ?Sized> ReadCommitteeStore for &T {
    fn get_committee(&self, epoch: EpochId) -> Result<Option<Arc<Committee>>> {
        (*self).get_committee(epoch)
    }
}

/// Trait used to provide functionality to the REST API service.
///
/// It extends both ObjectStore and ReadStore by adding functionality that may require more
/// detailed underlying databases or indexes to support.
pub trait RpcStateReader: ObjectStore + ReadStore + Send + Sync {
    // fn get_chain_identifier(&self) -> Result<ChainIdentifier>;

    // Get a handle to an instance of the RpcIndexes
    fn indexes(&self) -> Option<&dyn RpcIndexes>;
}

pub trait RpcIndexes: Send + Sync {
    fn get_epoch_info(&self, epoch: EpochId) -> Result<Option<EpochInfo>>;

    fn get_transaction_info(&self, digest: &TransactionDigest) -> Result<Option<TransactionInfo>>;

    fn owned_objects_iter(
        &self,
        owner: SomaAddress,
        object_type: Option<ObjectType>,
        cursor: Option<OwnedObjectInfo>,
    ) -> Result<Box<dyn Iterator<Item = Result<OwnedObjectInfo, TypedStoreError>> + '_>>;

    fn get_balance(&self, owner: &SomaAddress) -> Result<Option<u64>>;
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct OwnedObjectInfo {
    pub owner: SomaAddress,
    pub object_type: ObjectType,
    pub balance: Option<u64>,
    pub object_id: ObjectID,
    pub version: Version,
}

#[derive(Clone, Serialize, Deserialize, Eq, PartialEq, Debug)]
pub struct TransactionInfo {
    pub commit: u64,
    pub balance_changes: Vec<BalanceChange>,
    pub object_types: HashMap<ObjectID, ObjectType>,
}

impl TransactionInfo {
    pub fn new(
        effects: &TransactionEffects,
        input_objects: &[Object],
        output_objects: &[Object],
        commit: u64,
    ) -> TransactionInfo {
        let balance_changes = derive_balance_changes(effects, input_objects, output_objects);

        let object_types = input_objects
            .iter()
            .chain(output_objects)
            .map(|object| (object.id(), ObjectType::from(object)))
            .collect();

        TransactionInfo {
            commit,
            balance_changes,
            object_types,
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize, Eq, PartialEq, Debug)]
pub struct EpochInfo {
    pub epoch: u64,
    // pub protocol_version: Option<u64>,
    pub start_timestamp_ms: Option<u64>,
    pub end_timestamp_ms: Option<u64>,
    pub start_checkpoint: Option<u64>,
    pub end_checkpoint: Option<u64>,
    // TODO: pub reference_byte_price: Option<u64>,
    pub system_state: Option<crate::system_state::SystemState>,
}

#[derive(Default, Copy, Clone, Debug, Eq, PartialEq)]
pub struct BalanceInfo {
    pub balance: u64,
}
