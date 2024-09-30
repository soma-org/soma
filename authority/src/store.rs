use std::{collections::HashMap, sync::Arc};

use tokio::sync::{RwLockReadGuard, RwLockWriteGuard};
use tracing::{debug, info, instrument, trace};
use types::{
    committee::{Committee, EpochId},
    digests::{TransactionDigest, TransactionEffectsDigest},
    effects::{TransactionEffects, TransactionEffectsAPI},
    envelope::Message,
    error::SomaResult,
    genesis::Genesis,
    node_config::NodeConfig,
    system_state::SystemStateTrait,
    transaction::VerifiedTransaction,
    tx_outputs::TransactionOutputs,
};

use crate::{start_epoch::EpochStartConfiguration, store_tables::AuthorityPerpetualTables};

pub struct AuthorityStore {
    pub(crate) perpetual_tables: Arc<AuthorityPerpetualTables>,
    // pub(crate) root_state_notify_read: NotifyRead<EpochId, (CheckpointSequenceNumber, Accumulator)>,
}

pub type ExecutionLockReadGuard<'a> = RwLockReadGuard<'a, EpochId>;
pub type ExecutionLockWriteGuard<'a> = RwLockWriteGuard<'a, EpochId>;

impl AuthorityStore {
    /// Open an authority store by directory path.
    /// If the store is empty, initialize it using genesis.
    pub async fn open(
        perpetual_tables: Arc<AuthorityPerpetualTables>,
        genesis: &Genesis,
        config: &NodeConfig,
    ) -> SomaResult<Arc<Self>> {
        let epoch_start_configuration = if perpetual_tables.database_is_empty()? {
            info!("Creating new epoch start config from genesis");

            let epoch_start_configuration =
                EpochStartConfiguration::new(genesis.system_state().into_epoch_start_state());
            perpetual_tables.set_epoch_start_configuration(&epoch_start_configuration)?;
            epoch_start_configuration
        } else {
            info!("Loading epoch start config from DB");
            perpetual_tables
                .epoch_start_configuration
                .read()
                .get(&())
                .expect("Epoch start configuration must be set in non-empty DB")
                .clone()
        };
        let cur_epoch = perpetual_tables.get_recovery_epoch_at_restart()?;
        info!("Epoch start config: {:?}", epoch_start_configuration);
        info!("Cur epoch: {:?}", cur_epoch);
        let this = Self::open_inner(genesis, perpetual_tables).await?;
        Ok(this)
    }

    /// Returns true if there are no objects in the database
    pub fn database_is_empty(&self) -> SomaResult<bool> {
        self.perpetual_tables.database_is_empty()
    }

    pub async fn open_with_committee_for_testing(
        perpetual_tables: Arc<AuthorityPerpetualTables>,
        committee: &Committee,
        genesis: &Genesis,
    ) -> SomaResult<Arc<Self>> {
        // TODO: Since we always start at genesis, the committee should be technically the same
        // as the genesis committee.
        assert_eq!(committee.epoch, 0);
        Self::open_inner(genesis, perpetual_tables).await
    }

    async fn open_inner(
        genesis: &Genesis,
        perpetual_tables: Arc<AuthorityPerpetualTables>,
    ) -> SomaResult<Arc<Self>> {
        let store = Arc::new(Self { perpetual_tables });

        // Only initialize an empty database.
        if store
            .database_is_empty()
            .expect("Database read should not fail at init.")
        {
            // insert txn and effects of genesis
            let transaction = VerifiedTransaction::new_unchecked(genesis.transaction().clone());

            store
                .perpetual_tables
                .transactions
                .write()
                .insert(
                    *transaction.digest(),
                    transaction.serializable_ref().clone(),
                )
                .unwrap();

            store
                .perpetual_tables
                .effects
                .write()
                .insert(genesis.effects().digest(), genesis.effects().clone())
                .unwrap();
            // We don't insert the effects to executed_effects yet because the genesis tx hasn't but will be executed.
            // This is important for fullnodes to be able to generate indexing data right now.
        }

        Ok(store)
    }

    pub fn get_recovery_epoch_at_restart(&self) -> SomaResult<EpochId> {
        self.perpetual_tables.get_recovery_epoch_at_restart()
    }

    pub fn get_effects(
        &self,
        effects_digest: &TransactionEffectsDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        Ok(self
            .perpetual_tables
            .effects
            .read()
            .get(effects_digest)
            .cloned())
    }

    /// Returns true if we have an effects structure for this transaction digest
    pub fn effects_exists(&self, effects_digest: &TransactionEffectsDigest) -> SomaResult<bool> {
        Ok(self
            .perpetual_tables
            .effects
            .read()
            .contains_key(effects_digest))
    }

    pub fn multi_get_effects<'a>(
        &self,
        effects_digests: impl Iterator<Item = &'a TransactionEffectsDigest>,
    ) -> SomaResult<Vec<Option<TransactionEffects>>> {
        let read_guard = self.perpetual_tables.effects.read();
        Ok(effects_digests
            .map(|key| read_guard.get(key).cloned())
            .collect())
    }

    pub fn get_executed_effects(
        &self,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        let executed_effects_read = self.perpetual_tables.executed_effects.read();
        let effects_digest = executed_effects_read.get(tx_digest);
        match effects_digest {
            Some(digest) => Ok(self.perpetual_tables.effects.read().get(&digest).cloned()),
            None => Ok(None),
        }
    }

    /// Given a list of transaction digests, returns a list of the corresponding effects only if they have been
    /// executed. For transactions that have not been executed, None is returned.
    pub fn multi_get_executed_effects_digests(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<TransactionEffectsDigest>>> {
        let read_guard = self.perpetual_tables.executed_effects.read();

        Ok(digests
            .iter()
            .map(|key| read_guard.get(key).cloned())
            .collect())
    }

    /// Given a list of transaction digests, returns a list of the corresponding effects only if they have been
    /// executed. For transactions that have not been executed, None is returned.
    pub fn multi_get_executed_effects(
        &self,
        digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<TransactionEffects>>> {
        let read_guard = self.perpetual_tables.executed_effects.read();
        let executed_effects_digests: Vec<Option<TransactionEffectsDigest>> = digests
            .iter()
            .map(|key| read_guard.get(key).cloned())
            .collect();
        let effects = self.multi_get_effects(executed_effects_digests.iter().flatten())?;
        let mut tx_to_effects_map = effects
            .into_iter()
            .flatten()
            .map(|effects| (*effects.transaction_digest(), effects))
            .collect::<HashMap<_, _>>();
        Ok(digests
            .iter()
            .map(|digest| tx_to_effects_map.remove(digest))
            .collect())
    }

    pub fn is_tx_already_executed(&self, digest: &TransactionDigest) -> SomaResult<bool> {
        Ok(self
            .perpetual_tables
            .executed_effects
            .read()
            .contains_key(digest))
    }

    pub fn set_epoch_start_configuration(
        &self,
        epoch_start_configuration: &EpochStartConfiguration,
    ) -> SomaResult {
        self.perpetual_tables
            .set_epoch_start_configuration(epoch_start_configuration)?;
        Ok(())
    }

    pub fn get_epoch_start_configuration(&self) -> SomaResult<Option<EpochStartConfiguration>> {
        Ok(self
            .perpetual_tables
            .epoch_start_configuration
            .read()
            .get(&())
            .cloned())
    }

    pub fn insert_transaction_and_effects(
        &self,
        transaction: &VerifiedTransaction,
        transaction_effects: &TransactionEffects,
    ) -> Result<(), TypedStoreError> {
        self.perpetual_tables.transactions.write().insert(
            *transaction.digest(),
            transaction.serializable_ref().clone(),
        );

        self.perpetual_tables
            .effects
            .write()
            .insert(transaction_effects.digest(), transaction_effects.clone());
        Ok(())
    }

    pub fn multi_get_transaction_blocks(
        &self,
        tx_digests: &[TransactionDigest],
    ) -> SomaResult<Vec<Option<VerifiedTransaction>>> {
        let read_guard = self.perpetual_tables.transactions.read();
        Ok(tx_digests
            .iter()
            .map(|key| read_guard.get(key).cloned().map(|v| v.into()))
            .collect())
    }

    pub fn get_transaction_block(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Option<VerifiedTransaction> {
        self.perpetual_tables
            .transactions
            .read()
            .get(tx_digest)
            .cloned()
            .map(|v| v.into())
    }

    /// Updates the state resulting from the execution of a certificate.
    #[instrument(level = "debug", skip_all)]
    pub async fn write_transaction_outputs(
        &self,
        epoch_id: EpochId,
        tx_outputs: &[Arc<TransactionOutputs>],
    ) -> SomaResult {
        for outputs in tx_outputs {
            let TransactionOutputs {
                transaction,
                effects,
            } = outputs as &TransactionOutputs;

            // Store the certificate indexed by transaction digest
            let transaction_digest = transaction.digest();
            self.perpetual_tables
                .transactions
                .write()
                .insert(*transaction_digest, transaction.serializable_ref().clone());

            let effects_digest = effects.digest();

            self.perpetual_tables
                .effects
                .write()
                .insert(effects_digest, effects.clone());

            self.perpetual_tables
                .executed_effects
                .write()
                .insert(*transaction_digest, effects_digest);

            debug!(effects_digest = ?effects.digest(), "commit_certificate finished");
        }

        trace!(
            "committed transactions: {:?}",
            tx_outputs
                .iter()
                .map(|tx| tx.transaction.digest())
                .collect::<Vec<_>>()
        );

        Ok(())
    }

    /// Commits transactions only to the db. Called by checkpoint builder. See
    /// ExecutionCache::commit_transactions for more info
    pub(crate) fn commit_transactions(
        &self,
        transactions: &[(TransactionDigest, VerifiedTransaction)],
    ) -> SomaResult {
        for (digest, transaction) in transactions {
            self.perpetual_tables
                .transactions
                .write()
                .insert(*digest, transaction.serializable_ref().clone());
        }
        Ok(())
    }
}

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Ord, PartialOrd)]
pub enum TypedStoreError {
    #[error("rocksdb error: {0}")]
    RocksDBError(String),
    #[error("(de)serialization error: {0}")]
    SerializationError(String),
    #[error("the column family {0} was not registered with the database")]
    UnregisteredColumn(String),
    #[error("a batch operation can't operate across databases")]
    CrossDBBatch,
    #[error("Transaction should be retried")]
    RetryableTransactionError,
}
