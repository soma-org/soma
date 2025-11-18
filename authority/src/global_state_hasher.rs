use fastcrypto::hash::MultisetHash as _;
use std::sync::Arc;
use tracing::debug;
use types::checkpoints::{CheckpointSequenceNumber, ECMHLiveObjectSetDigest, GlobalStateHash};
use types::committee::EpochId;
use types::digests::ObjectDigest;
use types::effects::{TransactionEffects, TransactionEffectsAPI as _};
use types::error::SomaResult;
use types::object::{LiveObject, ObjectID};
use types::storage::object_store::ObjectStore;
use types::storage::shared_in_memory_store::SharedInMemoryStore;

use crate::epoch_store::AuthorityPerEpochStore;

pub struct GlobalStateHasher {
    store: Arc<dyn GlobalStateHashStore>,
}

pub trait GlobalStateHashStore: ObjectStore + Send + Sync {
    fn get_root_state_hash_for_epoch(
        &self,
        epoch: EpochId,
    ) -> SomaResult<Option<(CheckpointSequenceNumber, GlobalStateHash)>>;

    fn get_root_state_hash_for_highest_epoch(
        &self,
    ) -> SomaResult<Option<(EpochId, (CheckpointSequenceNumber, GlobalStateHash))>>;

    fn insert_state_hash_for_epoch(
        &self,
        epoch: EpochId,
        checkpoint_seq_num: &CheckpointSequenceNumber,
        acc: &GlobalStateHash,
    ) -> SomaResult;

    fn iter_live_object_set(
        &self,
        include_wrapped_tombstone: bool,
    ) -> Box<dyn Iterator<Item = LiveObject> + '_>;

    fn iter_cached_live_object_set_for_testing(
        &self,
        include_wrapped_tombstone: bool,
    ) -> Box<dyn Iterator<Item = LiveObject> + '_> {
        self.iter_live_object_set(include_wrapped_tombstone)
    }
}

impl GlobalStateHashStore for SharedInMemoryStore {
    fn get_root_state_hash_for_epoch(
        &self,
        _epoch: EpochId,
    ) -> SomaResult<Option<(CheckpointSequenceNumber, GlobalStateHash)>> {
        unreachable!("not used for testing")
    }

    fn get_root_state_hash_for_highest_epoch(
        &self,
    ) -> SomaResult<Option<(EpochId, (CheckpointSequenceNumber, GlobalStateHash))>> {
        unreachable!("not used for testing")
    }

    fn insert_state_hash_for_epoch(
        &self,
        _epoch: EpochId,
        _checkpoint_seq_num: &CheckpointSequenceNumber,
        _acc: &GlobalStateHash,
    ) -> SomaResult {
        unreachable!("not used for testing")
    }

    fn iter_live_object_set(
        &self,
        _include_wrapped_tombstone: bool,
    ) -> Box<dyn Iterator<Item = LiveObject> + '_> {
        unreachable!("not used for testing")
    }
}

pub fn accumulate_effects<T, S>(effects: &[TransactionEffects]) -> GlobalStateHash
where
    S: std::ops::Deref<Target = T>,
    T: GlobalStateHashStore + ?Sized,
{
    let mut acc = GlobalStateHash::default();

    // process insertions to the set
    acc.insert_all(
        effects
            .iter()
            .flat_map(|fx| {
                fx.all_changed_objects()
                    .into_iter()
                    .map(|(object_ref, _, _)| object_ref.2)
            })
            .collect::<Vec<ObjectDigest>>(),
    );

    // process modified objects to the set
    acc.remove_all(
        effects
            .iter()
            .flat_map(|fx| {
                fx.old_object_metadata()
                    .into_iter()
                    .map(|(object_ref, _owner)| object_ref.2)
            })
            .collect::<Vec<ObjectDigest>>(),
    );

    acc
}

impl GlobalStateHasher {
    pub fn new(store: Arc<dyn GlobalStateHashStore>) -> Self {
        Self { store }
    }

    /// Accumulates the effects of a single checkpoint and persists the hasher.
    pub fn accumulate_checkpoint(
        &self,
        effects: &[TransactionEffects],
        checkpoint_seq_num: CheckpointSequenceNumber,
        epoch_store: &AuthorityPerEpochStore,
    ) -> SomaResult<GlobalStateHash> {
        if let Some(acc) = epoch_store.get_state_hash_for_checkpoint(&checkpoint_seq_num)? {
            return Ok(acc);
        }

        let acc = self.accumulate_effects(effects, epoch_store.protocol_config());

        epoch_store.insert_state_hash_for_checkpoint(&checkpoint_seq_num, &acc)?;
        debug!("Accumulated checkpoint {}", checkpoint_seq_num);

        epoch_store
            .checkpoint_state_notify_read
            .notify(&checkpoint_seq_num, &acc);

        Ok(acc)
    }

    pub fn accumulate_cached_live_object_set_for_testing(
        &self,
        include_wrapped_tombstone: bool,
    ) -> GlobalStateHash {
        Self::accumulate_live_object_set_impl(
            self.store
                .iter_cached_live_object_set_for_testing(include_wrapped_tombstone),
        )
    }

    /// Returns the result of accumulating the live object set, without side effects
    pub fn accumulate_live_object_set(&self, include_wrapped_tombstone: bool) -> GlobalStateHash {
        Self::accumulate_live_object_set_impl(
            self.store.iter_live_object_set(include_wrapped_tombstone),
        )
    }

    fn accumulate_live_object_set_impl(iter: impl Iterator<Item = LiveObject>) -> GlobalStateHash {
        let mut acc = GlobalStateHash::default();
        iter.for_each(|live_object| {
            Self::accumulate_live_object(&mut acc, &live_object);
        });
        acc
    }

    pub fn accumulate_live_object(acc: &mut GlobalStateHash, live_object: &LiveObject) {
        match live_object {
            LiveObject::Normal(object) => {
                acc.insert(object.compute_object_reference().2);
            }
        }
    }

    pub fn digest_live_object_set(
        &self,
        include_wrapped_tombstone: bool,
    ) -> ECMHLiveObjectSetDigest {
        let acc = self.accumulate_live_object_set(include_wrapped_tombstone);
        acc.digest().into()
    }

    pub async fn digest_epoch(
        &self,
        epoch_store: Arc<AuthorityPerEpochStore>,
        last_checkpoint_of_epoch: CheckpointSequenceNumber,
    ) -> SomaResult<ECMHLiveObjectSetDigest> {
        Ok(self
            .accumulate_epoch(epoch_store, last_checkpoint_of_epoch)?
            .digest()
            .into())
    }

    pub async fn wait_for_previous_running_root(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        checkpoint_seq_num: CheckpointSequenceNumber,
    ) -> SomaResult {
        assert!(checkpoint_seq_num > 0);

        // Check if this is the first checkpoint of the new epoch, in which case
        // there is nothing to wait for.
        if self
            .store
            .get_root_state_hash_for_highest_epoch()?
            .map(|(_, (last_checkpoint_prev_epoch, _))| last_checkpoint_prev_epoch)
            == Some(checkpoint_seq_num - 1)
        {
            return Ok(());
        }

        // There is an edge case here where checkpoint_seq_num is 1. This means the previous
        // checkpoint is the genesis checkpoint. CheckpointExecutor is guaranteed to execute
        // and accumulate the genesis checkpoint, so this will resolve.
        epoch_store
            .notify_read_running_root(checkpoint_seq_num - 1)
            .await?;
        Ok(())
    }

    fn get_prior_root(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        checkpoint_seq_num: CheckpointSequenceNumber,
    ) -> SomaResult<GlobalStateHash> {
        if checkpoint_seq_num == 0 {
            return Ok(GlobalStateHash::default());
        }

        if let Some(prior_running_root) =
            epoch_store.get_running_root_state_hash(checkpoint_seq_num - 1)?
        {
            return Ok(prior_running_root);
        }

        if let Some((last_checkpoint_prev_epoch, prev_acc)) = self
            .store
            .get_root_state_hash_for_epoch(epoch_store.epoch() - 1)?
        {
            if last_checkpoint_prev_epoch == checkpoint_seq_num - 1 {
                return Ok(prev_acc);
            }
        }

        panic!(
            "Running root state hasher must exist for checkpoint {}",
            checkpoint_seq_num - 1
        );
    }

    // Accumulate the running root.
    // The previous checkpoint must be accumulated before calling this function, or it will panic.
    pub fn accumulate_running_root(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        checkpoint_seq_num: CheckpointSequenceNumber,
        checkpoint_acc: Option<GlobalStateHash>,
    ) -> SomaResult {
        tracing::info!(
            "accumulating running root for checkpoint {}",
            checkpoint_seq_num
        );

        // Idempotency.
        if epoch_store
            .get_running_root_state_hash(checkpoint_seq_num)?
            .is_some()
        {
            debug!(
                "accumulate_running_root {:?} {:?} already exists",
                epoch_store.epoch(),
                checkpoint_seq_num
            );
            return Ok(());
        }

        let mut running_root = self.get_prior_root(epoch_store, checkpoint_seq_num)?;

        let checkpoint_acc = checkpoint_acc.unwrap_or_else(|| {
            epoch_store
                .get_state_hash_for_checkpoint(&checkpoint_seq_num)
                .expect("Failed to get checkpoint accumulator from disk")
                .expect("Expected checkpoint accumulator to exist")
        });
        running_root.union(&checkpoint_acc);
        epoch_store.insert_running_root_state_hash(&checkpoint_seq_num, &running_root)?;
        debug!(
            "Accumulated checkpoint {} to running root accumulator",
            checkpoint_seq_num,
        );
        Ok(())
    }

    pub fn accumulate_epoch(
        &self,
        epoch_store: Arc<AuthorityPerEpochStore>,
        last_checkpoint_of_epoch: CheckpointSequenceNumber,
    ) -> SomaResult<GlobalStateHash> {
        let running_root = epoch_store
            .get_running_root_state_hash(last_checkpoint_of_epoch)?
            .expect("Expected running root accumulator to exist up to last checkpoint of epoch");

        self.store.insert_state_hash_for_epoch(
            epoch_store.epoch(),
            &last_checkpoint_of_epoch,
            &running_root,
        )?;
        debug!(
            "Finalized root state hash for epoch {} (up to checkpoint {})",
            epoch_store.epoch(),
            last_checkpoint_of_epoch
        );
        Ok(running_root.clone())
    }
}
