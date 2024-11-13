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

pub type CommitIndex = u64;

pub struct StateAccumulator {
    store: Arc<dyn AccumulatorStore>,
}

pub trait AccumulatorStore: ObjectStore + Send + Sync {
    fn get_root_state_accumulator_for_commit(
        &self,
        commit: CommitIndex,
    ) -> SomaResult<Option<Accumulator>>;

    fn get_root_state_accumulator_for_highest_commit(
        &self,
    ) -> SomaResult<Option<(CommitIndex, Accumulator)>>;

    fn insert_state_accumulator_for_commit(
        &self,
        commit: &CommitIndex,
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

    /// Returns the result of accumulating the live object set, without side effects
    pub fn accumulate_live_object_set(&self) -> Accumulator {
        Self::accumulate_live_object_set_impl(self.store.iter_live_object_set())
    }

    /// Accumulates the effects of a single checkpoint and persists the accumulator.
    pub fn accumulate_commit(
        &self,
        effects: Vec<TransactionEffects>,
        commit: CommitIndex,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<Accumulator> {
        if let Some(acc) = epoch_store.get_state_hash_for_commit(&commit)? {
            return Ok(acc);
        }

        let acc = self.accumulate_effects(effects.clone());

        debug!(
            "Accumulated effects for commit {} to state hash: {}",
            commit,
            acc.digest()
        );

        epoch_store.insert_state_hash_for_commit(&commit, &acc)?;

        Ok(acc)
    }

    /// Unions all commit accumulators to generate the
    /// root state hash and persists it to db. This function is idempotent. Can be called on
    /// non-consecutive commits, e.g. to accumulate commit 3 after having last
    /// accumulated commit 1.
    pub async fn accumulate_running_root(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        commit: CommitIndex,
        commit_acc: Option<Accumulator>,
    ) -> SomaResult {
        tracing::info!("accumulating running root for commit {}", commit);

        // For the last checkpoint of the epoch, this function will be called once by the
        // checkpoint builder, and again by checkpoint executor.
        //
        // Normally this is fine, since the notify_read_running_root(checkpoint_seq_num - 1) will
        // work normally. But if there is only one checkpoint in the epoch, that call will hang
        // forever, since the previous checkpoint belongs to the previous epoch.
        if epoch_store.get_running_root_accumulator(&commit)?.is_some() {
            debug!(
                "accumulate_running_root {:?} {:?} already exists",
                epoch_store.epoch(),
                commit
            );
            return Ok(());
        }

        let mut running_root = if commit == 0 {
            // we're at genesis and need to start from scratch
            Accumulator::default()
        } else if let Some((commit, acc)) = epoch_store.get_highest_running_root_accumulator()? {
            acc
        } else {
            // we're at the beginning of a new epoch and need to
            // bootstrap from the previous epoch's root state hash. Because this
            // should only occur at beginning of epoch, we shouldn't have to worry
            // about race conditions on reading the highest running root accumulator.
            if let Some((prev_commit, prev_acc)) =
                self.store.get_root_state_accumulator_for_highest_commit()?
            {
                prev_acc
            } else {
                epoch_store.notify_read_running_root(commit - 1).await?
            }
        };

        let commit_acc = commit_acc.unwrap_or_else(|| {
            epoch_store
                .get_state_hash_for_commit(&commit)
                .expect("Failed to get commit accumulator from disk")
                .expect("Expected commit accumulator to exist")
        });

        running_root.union(&commit_acc);
        epoch_store.insert_running_root_accumulator(&commit, &running_root)?;

        self.store
            .insert_state_accumulator_for_commit(&commit, &running_root)?;

        debug!(
            "Finalized root state hash for epoch (up to commit {}): {}",
            commit,
            running_root.clone().digest(),
        );
        // debug!("Accumulated commit {} to running root accumulator", commit,);
        Ok(())
    }

    pub fn finalize_running_root(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        commit: CommitIndex,
    ) -> SomaResult<Accumulator> {
        let running_root = epoch_store
            .get_running_root_accumulator(&commit)?
            .expect("Expected running root accumulator to exist up to last checkpoint of epoch");

        self.store
            .insert_state_accumulator_for_commit(&commit, &running_root)?;
        debug!(
            "Finalized root state hash for epoch (up to commit {}): {}",
            commit,
            running_root.clone().digest(),
        );
        Ok(running_root.clone())
    }
}
