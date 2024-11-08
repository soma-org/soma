use std::sync::Arc;

use crate::{
    committee::{Committee, EpochId},
    digests::TransactionDigest,
    effects::TransactionEffects,
    transaction::VerifiedTransaction,
};

use super::{object_store::ObjectStore, storage_error::Result};

pub trait ReadStore: ObjectStore {
    //
    // Committee Getters
    //

    fn get_committee(&self, epoch: EpochId) -> Result<Option<Arc<Committee>>>;

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
