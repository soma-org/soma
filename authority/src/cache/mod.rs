use std::sync::Arc;

use futures::{future::BoxFuture, FutureExt};
use tracing::debug;
use types::{
    committee::EpochId,
    digests::{TransactionDigest, TransactionEffectsDigest},
    effects::TransactionEffects,
    error::SomaResult,
    transaction::VerifiedTransaction,
    tx_outputs::TransactionOutputs,
};
use writeback_cache::WritebackCache;

use crate::store::AuthorityStore;

pub(crate) mod writeback_cache;

#[derive(Clone)]
pub struct ExecutionCacheTraitPointers {
    pub transaction_cache_reader: Arc<dyn TransactionCacheRead>,
    pub cache_writer: Arc<dyn ExecutionCacheWrite>,
    // pub backing_store: Arc<dyn BackingStore + Send + Sync>,
    // pub reconfig_api: Arc<dyn ExecutionCacheReconfigAPI>,
    // pub accumulator_store: Arc<dyn AccumulatorStore>,
    // pub state_sync_store: Arc<dyn StateSyncAPI>,
    pub cache_commit: Arc<dyn ExecutionCacheCommit>,
}

impl ExecutionCacheTraitPointers {
    pub fn new<T>(cache: Arc<T>) -> Self
    where
        T: TransactionCacheRead
            + ExecutionCacheWrite
            // + BackingStore
            // + ExecutionCacheReconfigAPI
            // + AccumulatorStore
            // + StateSyncAPI
            + ExecutionCacheCommit
            + 'static,
    {
        Self {
            transaction_cache_reader: cache.clone(),
            cache_writer: cache.clone(),
            // backing_store: cache.clone(),
            // reconfig_api: cache.clone(),
            // accumulator_store: cache.clone(),
            // state_sync_store: cache.clone(),
            cache_commit: cache.clone(),
        }
    }
}

pub fn build_execution_cache(store: &Arc<AuthorityStore>) -> ExecutionCacheTraitPointers {
    ExecutionCacheTraitPointers::new(WritebackCache::new(store.clone()).into())
}

pub trait TransactionCacheRead: Send + Sync {
    fn multi_get_transaction_blocks(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<Arc<VerifiedTransaction>>>>;

    fn get_transaction_block(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<Arc<VerifiedTransaction>>> {
        self.multi_get_transaction_blocks(&[*digest])
            .map(|mut blocks| {
                blocks
                    .pop()
                    .expect("multi-get must return correct number of items")
            })
    }

    fn multi_get_executed_effects_digests(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<TransactionEffectsDigest>>>;

    fn is_tx_already_executed(&self, digest: &TransactionDigest) -> SomaResult<bool> {
        self.multi_get_executed_effects_digests(&[*digest])
            .map(|mut digests| {
                digests
                    .pop()
                    .expect("multi-get must return correct number of items")
                    .is_some()
            })
    }

    fn multi_get_executed_effects(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<TransactionEffects>>> {
        let effects_digests = self.multi_get_executed_effects_digests(digests)?;
        assert_eq!(effects_digests.len(), digests.len());

        let mut results = vec![None; digests.len()];
        let mut fetch_digests = Vec::with_capacity(digests.len());
        let mut fetch_indices = Vec::with_capacity(digests.len());

        for (i, digest) in effects_digests.into_iter().enumerate() {
            if let Some(digest) = digest {
                fetch_digests.push(digest);
                fetch_indices.push(i);
            }
        }

        let effects = self.multi_get_effects(&fetch_digests)?;
        for (i, effects) in fetch_indices.into_iter().zip(effects.into_iter()) {
            results[i] = effects;
        }

        Ok(results)
    }

    fn get_executed_effects(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        self.multi_get_executed_effects(&[*digest])
            .map(|mut effects| {
                effects
                    .pop()
                    .expect("multi-get must return correct number of items")
            })
    }

    fn multi_get_effects(
        &self,
        digests: &[TransactionEffectsDigest],
    ) -> SomaResult<Vec<Option<TransactionEffects>>>;

    fn get_effects(
        &self,
        digest: &TransactionEffectsDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        self.multi_get_effects(&[*digest]).map(|mut effects| {
            effects
                .pop()
                .expect("multi-get must return correct number of items")
        })
    }

    fn notify_read_executed_effects_digests<'a>(
        &'a self,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult<Vec<TransactionEffectsDigest>>>;

    fn notify_read_executed_effects<'a>(
        &'a self,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult<Vec<TransactionEffects>>> {
        async move {
            let digests = self.notify_read_executed_effects_digests(digests).await?;
            debug!("notify_read_executed_effects: {:?}", digests);
            // once digests are available, effects must be present as well
            self.multi_get_effects(&digests).map(|effects| {
                effects
                    .into_iter()
                    .map(|e| e.expect("digests must exist"))
                    .collect()
            })
        }
        .boxed()
    }
}

pub trait ExecutionCacheWrite: Send + Sync {
    /// Write the output of a transaction.
    fn write_transaction_outputs(
        &self,
        epoch_id: EpochId,
        tx_outputs: Arc<TransactionOutputs>,
    ) -> BoxFuture<'_, SomaResult>;
}

pub trait ExecutionCacheCommit: Send + Sync {
    /// Durably commit the outputs of the given transactions to the database.
    /// Will be called by CheckpointExecutor to ensure that transaction outputs are
    /// written durably before marking a checkpoint as finalized.
    fn commit_transaction_outputs<'a>(
        &'a self,
        epoch: EpochId,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult>;

    /// Durably commit transactions (but not their outputs) to the database.
    /// Called before writing a locally built checkpoint to the CheckpointStore, so that
    /// the inputs of the checkpoint cannot be lost.
    /// These transactions are guaranteed to be final unless this validator
    /// forks (i.e. constructs a checkpoint which will never be certified). In this case
    /// some non-final transactions could be left in the database.
    ///
    /// This is an intermediate solution until we delay commits to the epoch db. After
    /// we have done that, crash recovery will be done by re-processing consensus commits
    /// and pending_consensus_transactions, and this method can be removed.
    fn persist_transactions<'a>(
        &'a self,
        digests: &'a [TransactionDigest],
    ) -> BoxFuture<'a, SomaResult>;
}
