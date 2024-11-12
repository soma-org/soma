use std::sync::Arc;

use fastcrypto::hash::MultisetHash;
use tracing::{debug, info};
use types::{
    accumulator::Accumulator,
    committee::EpochId,
    digests::{ECMHLiveObjectSetDigest, ObjectDigest},
    effects::{TransactionEffects, TransactionEffectsAPI},
    error::SomaResult,
    storage::object_store::ObjectStore,
};

use crate::{epoch_store::AuthorityPerEpochStore, store_tables::LiveObject};

pub type CheckpointSequenceNumber = u64;

pub struct StateAccumulator {
    store: Arc<dyn AccumulatorStore>,
}

pub trait AccumulatorStore: ObjectStore + Send + Sync {
    fn get_root_state_accumulator_for_epoch(
        &self,
        epoch: EpochId,
    ) -> SomaResult<Option<(CheckpointSequenceNumber, Accumulator)>>;

    fn get_root_state_accumulator_for_highest_epoch(
        &self,
    ) -> SomaResult<Option<(EpochId, (CheckpointSequenceNumber, Accumulator))>>;

    fn insert_state_accumulator_for_epoch(
        &self,
        epoch: EpochId,
        checkpoint_seq_num: &CheckpointSequenceNumber,
        acc: &Accumulator,
    ) -> SomaResult;

    fn iter_live_object_set(&self) -> Box<dyn Iterator<Item = LiveObject> + '_>;
}

impl StateAccumulator {
    pub fn new(store: Arc<dyn AccumulatorStore>) -> Self {
        Self { store }
    }

    pub fn accumulate_effects(&self, effects: Vec<TransactionEffects>) -> Accumulator {
        let mut acc = Accumulator::default();

        let inserts = effects
            .iter()
            .flat_map(|fx| {
                fx.all_changed_objects()
                    .into_iter()
                    .map(|(object_ref, _)| object_ref.2)
            })
            .collect::<Vec<ObjectDigest>>();
        let removals = effects
            .iter()
            .flat_map(|fx| {
                fx.old_object_metadata()
                    .into_iter()
                    .map(|object_ref| object_ref.2)
            })
            .collect::<Vec<ObjectDigest>>();

        // process insertions to the set
        acc.insert_all(inserts);

        // process modified objects to the set
        acc.remove_all(removals);

        acc
    }

    fn accumulate_live_object_set_impl(iter: impl Iterator<Item = LiveObject>) -> Accumulator {
        let mut acc = Accumulator::default();
        iter.for_each(|live_object| {
            Self::accumulate_live_object(&mut acc, &live_object);
        });
        acc
    }

    pub fn accumulate_live_object(acc: &mut Accumulator, live_object: &LiveObject) {
        match live_object {
            LiveObject::Normal(object) => {
                acc.insert(object.compute_object_reference().2);
            }
        }
    }

    pub fn digest_live_object_set(&self) -> ECMHLiveObjectSetDigest {
        let acc = self.accumulate_live_object_set();
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

    /// Returns the result of accumulating the live object set, without side effects
    pub fn accumulate_live_object_set(&self) -> Accumulator {
        Self::accumulate_live_object_set_impl(self.store.iter_live_object_set())
    }

    /// Accumulates the effects of a single checkpoint and persists the accumulator.
    pub fn accumulate_checkpoint(
        &self,
        effects: Vec<TransactionEffects>,
        checkpoint_seq_num: CheckpointSequenceNumber,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<Accumulator> {
        if let Some(acc) = epoch_store.get_state_hash_for_checkpoint(&checkpoint_seq_num)? {
            return Ok(acc);
        }

        let acc = self.accumulate_effects(effects.clone());

        epoch_store.insert_state_hash_for_checkpoint(&checkpoint_seq_num, &acc)?;

        Ok(acc)
    }

    /// Unions all checkpoint accumulators at the end of the epoch to generate the
    /// root state hash and persists it to db. This function is idempotent. Can be called on
    /// non-consecutive epochs, e.g. to accumulate epoch 3 after having last
    /// accumulated epoch 1.
    pub async fn accumulate_running_root(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        checkpoint_seq_num: CheckpointSequenceNumber,
        checkpoint_acc: Option<Accumulator>,
    ) -> SomaResult {
        tracing::info!(
            "accumulating running root for checkpoint {}",
            checkpoint_seq_num
        );

        // For the last checkpoint of the epoch, this function will be called once by the
        // checkpoint builder, and again by checkpoint executor.
        //
        // Normally this is fine, since the notify_read_running_root(checkpoint_seq_num - 1) will
        // work normally. But if there is only one checkpoint in the epoch, that call will hang
        // forever, since the previous checkpoint belongs to the previous epoch.
        if epoch_store
            .get_running_root_accumulator(&checkpoint_seq_num)?
            .is_some()
        {
            debug!(
                "accumulate_running_root {:?} {:?} already exists",
                epoch_store.epoch(),
                checkpoint_seq_num
            );
            return Ok(());
        }

        let mut running_root = if checkpoint_seq_num == 0 {
            // we're at genesis and need to start from scratch
            Accumulator::default()
        } else if epoch_store
            .get_highest_running_root_accumulator()?
            .is_none()
        {
            // we're at the beginning of a new epoch and need to
            // bootstrap from the previous epoch's root state hash. Because this
            // should only occur at beginning of epoch, we shouldn't have to worry
            // about race conditions on reading the highest running root accumulator.
            if let Some((prev_epoch, (last_checkpoint_prev_epoch, prev_acc))) =
                self.store.get_root_state_accumulator_for_highest_epoch()?
            {
                if last_checkpoint_prev_epoch != checkpoint_seq_num - 1 {
                    epoch_store
                        .notify_read_running_root(checkpoint_seq_num - 1)
                        .await?
                } else {
                    assert_eq!(
                        prev_epoch + 1,
                        epoch_store.epoch(),
                        "Expected highest existing root state hash to be for previous epoch",
                    );
                    prev_acc
                }
            } else {
                // Rare edge case where we manage to somehow lag in checkpoint execution from genesis
                // such that the end of epoch checkpoint is built before we execute any checkpoints.
                assert_eq!(
                    epoch_store.epoch(),
                    0,
                    "Expected epoch to be 0 if previous root state hash does not exist"
                );
                epoch_store
                    .notify_read_running_root(checkpoint_seq_num - 1)
                    .await?
            }
        } else {
            epoch_store
                .notify_read_running_root(checkpoint_seq_num - 1)
                .await?
        };

        let checkpoint_acc = checkpoint_acc.unwrap_or_else(|| {
            epoch_store
                .get_state_hash_for_checkpoint(&checkpoint_seq_num)
                .expect("Failed to get checkpoint accumulator from disk")
                .expect("Expected checkpoint accumulator to exist")
        });
        running_root.union(&checkpoint_acc);
        epoch_store.insert_running_root_accumulator(&checkpoint_seq_num, &running_root)?;
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
    ) -> SomaResult<Accumulator> {
        let running_root = epoch_store
            .get_running_root_accumulator(&last_checkpoint_of_epoch)?
            .expect("Expected running root accumulator to exist up to last checkpoint of epoch");

        self.store.insert_state_accumulator_for_epoch(
            epoch_store.epoch(),
            &last_checkpoint_of_epoch,
            &running_root,
        )?;
        debug!(
            "Finalized root state hash for epoch {} (up to checkpoint {}): {}",
            epoch_store.epoch(),
            last_checkpoint_of_epoch,
            running_root.clone().digest(),
        );
        Ok(running_root.clone())
    }
}
