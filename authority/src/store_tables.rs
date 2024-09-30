use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use parking_lot::RwLock;
use types::{
    committee::EpochId,
    digests::{TransactionDigest, TransactionEffectsDigest},
    effects::TransactionEffects,
    error::SomaResult,
    system_state::EpochStartSystemStateTrait,
    transaction::TrustedTransaction,
};

use crate::start_epoch::{EpochStartConfigTrait, EpochStartConfiguration};

/// AuthorityPerpetualTables contains data that must be preserved from one epoch to the next.
pub struct AuthorityPerpetualTables {
    /// This is a map between the transaction digest and the corresponding transaction that's known to be
    /// executable. This means that it may have been executed locally, or it may have been synced through
    /// state-sync but hasn't been executed yet.
    pub(crate) transactions: RwLock<BTreeMap<TransactionDigest, TrustedTransaction>>,

    /// A map between the transaction digest of a certificate to the effects of its execution.
    /// We store effects into this table in two different cases:
    /// 1. When a transaction is synced through state_sync, we store the effects here. These effects
    ///     are known to be final in the network, but may not have been executed locally yet.
    /// 2. When the transaction is executed locally on this node, we store the effects here. This means that
    ///     it's possible to store the same effects twice (once for the synced transaction, and once for the executed).
    ///
    /// It's also possible for the effects to be reverted if the transaction didn't make it into the epoch.
    pub(crate) effects: RwLock<BTreeMap<TransactionEffectsDigest, TransactionEffects>>,

    /// Transactions that have been executed locally on this node. We need this table since the `effects` table
    /// doesn't say anything about the execution status of the transaction on this node. When we wait for transactions
    /// to be executed, we wait for them to appear in this table. When we revert transactions, we remove them from both
    /// tables.
    pub(crate) executed_effects: RwLock<BTreeMap<TransactionDigest, TransactionEffectsDigest>>,

    /// Parameters of the system fixed at the epoch start
    pub(crate) epoch_start_configuration: RwLock<BTreeMap<(), EpochStartConfiguration>>,
}

impl AuthorityPerpetualTables {
    pub fn path(parent_path: &Path) -> PathBuf {
        parent_path.join("perpetual")
    }

    pub fn open(parent_path: &Path) -> Self {
        // Self::open_tables_read_write(Self::path(parent_path))
        Self {
            transactions: RwLock::new(BTreeMap::new()),
            effects: RwLock::new(BTreeMap::new()),
            executed_effects: RwLock::new(BTreeMap::new()),
            epoch_start_configuration: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn get_recovery_epoch_at_restart(&self) -> SomaResult<EpochId> {
        Ok(self
            .epoch_start_configuration
            .read()
            .get(&())
            .expect("Must have current epoch.")
            .epoch_start_state()
            .epoch())
    }

    pub fn set_epoch_start_configuration(
        &self,
        epoch_start_configuration: &EpochStartConfiguration,
    ) -> SomaResult {
        self.epoch_start_configuration
            .write()
            .insert((), epoch_start_configuration.clone());
        Ok(())
    }

    pub fn get_transaction(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<TrustedTransaction>> {
        let transaction_read = self.transactions.read();
        let Some(transaction) = transaction_read.get(digest) else {
            return Ok(None);
        };
        Ok(Some(transaction.clone()))
    }

    pub fn get_effects(
        &self,
        digest: &TransactionDigest,
    ) -> SomaResult<Option<TransactionEffects>> {
        let executed_effects_read = self.executed_effects.read();
        let Some(effect_digest) = executed_effects_read.get(digest) else {
            return Ok(None);
        };
        Ok(self.effects.read().get(&effect_digest).cloned())
    }

    pub fn database_is_empty(&self) -> SomaResult<bool> {
        Ok(self.transactions.read().is_empty()
            && self.effects.read().is_empty()
            && self.executed_effects.read().is_empty()
            && self.epoch_start_configuration.read().is_empty())
    }
}
