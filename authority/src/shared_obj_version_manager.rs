// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};

use either::Either;
use tracing::trace;
use types::base::ConsensusObjectSequenceKey;
use types::digests::TransactionDigest;
use types::effects::{TransactionEffects, TransactionEffectsAPI as _};
use types::error::SomaResult;
use types::object::Version;
use types::storage::{
    ObjectKey, transaction_non_shared_input_object_keys, transaction_receiving_object_keys,
};
use types::transaction::{SharedInputObject, TransactionKey, VerifiedExecutableTransaction};

use crate::authority_per_epoch_store::{AuthorityPerEpochStore, CancelConsensusCertificateReason};
use crate::cache::ObjectCacheRead;

pub struct SharedObjVerManager {}

/// Version assignments for a single transaction
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssignedVersions {
    pub shared_object_versions: Vec<(ConsensusObjectSequenceKey, Version)>,
}

impl AssignedVersions {
    pub fn new(shared_object_versions: Vec<(ConsensusObjectSequenceKey, Version)>) -> Self {
        Self { shared_object_versions }
    }

    pub fn iter(&self) -> impl Iterator<Item = &(ConsensusObjectSequenceKey, Version)> {
        self.shared_object_versions.iter()
    }

    pub fn as_slice(&self) -> &[(ConsensusObjectSequenceKey, Version)] {
        &self.shared_object_versions
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct AssignedTxAndVersions(pub Vec<(TransactionKey, AssignedVersions)>);

impl AssignedTxAndVersions {
    pub fn new(assigned_versions: Vec<(TransactionKey, AssignedVersions)>) -> Self {
        Self(assigned_versions)
    }

    pub fn into_map(self) -> HashMap<TransactionKey, AssignedVersions> {
        self.0.into_iter().collect()
    }
}

/// A wrapper around things that can be scheduled for execution by the assigning of
/// shared object versions.
#[derive(Clone)]
pub enum Schedulable<T = VerifiedExecutableTransaction> {
    Transaction(T),
    /// Stage 14c.5: per-checkpoint settlement placeholder. Mirrors
    /// Sui's `Schedulable::AccumulatorSettlement`. This is NOT a real
    /// transaction — it's a sync marker that the [`SettlementScheduler`]
    /// recognizes and routes to its dedicated queue. The queue runner
    /// awaits the in-batch user-tx effects, builds the actual
    /// settlement transactions via [`crate::accumulators::AccumulatorSettlementTxBuilder`],
    /// and dispatches them to the regular [`ExecutionScheduler`].
    ///
    /// The placeholder carries enough information for the settlement
    /// scheduler to:
    ///   - identify itself in the cache (`settlement_key`),
    ///   - know which user-tx effects to await (`tx_keys`), and
    ///   - tag the resulting settlement txs with the right
    ///     checkpoint sequence number (`checkpoint_seq`).
    ///
    /// [`SettlementScheduler`]: crate::execution_scheduler::settlement_scheduler::SettlementScheduler
    /// [`ExecutionScheduler`]: crate::execution_scheduler::ExecutionScheduler
    AccumulatorSettlement(Box<SettlementBatchInfo>),
}

/// Stage 14c.5: payload of [`Schedulable::AccumulatorSettlement`].
///
/// Constructed by the consensus handler when it processes a commit
/// that contains balance-touching user txs. The settlement scheduler
/// uses this to await user-tx effects, then to construct + dispatch
/// the actual settlement transactions.
#[derive(Clone, Debug)]
pub struct SettlementBatchInfo {
    /// Stable, unique key for this settlement batch. Used by the
    /// epoch_store's notify-read mechanism (consensus handler signals
    /// at enqueue time; settlement scheduler awaits before building
    /// settlement). Also used as the cache key downstream.
    pub settlement_key: TransactionKey,

    /// Digests of every user transaction in this batch. The
    /// settlement scheduler awaits these to complete via the cache's
    /// `notify_read_executed_effects` before walking each tx's
    /// `effects.accumulator_events()` to build the settlement.
    ///
    /// We carry digests directly (not `TransactionKey`s) because the
    /// consensus handler knows them at schedulable-construction time
    /// — no extra notify-read mapping is needed.
    pub tx_digests: Vec<TransactionDigest>,

    /// Checkpoint sequence number this settlement belongs to. Baked
    /// into the settlement transaction's tx-data (alongside epoch +
    /// commit metadata) so consecutive empty/identical settlements
    /// at different checkpoints don't collide in the executed-digest
    /// cache.
    pub checkpoint_seq: u64,

    /// Pre-computed shared-object version assignments for the
    /// settlement transaction itself. The placeholder doesn't have
    /// shared inputs of its own (it's a marker), but the resulting
    /// settlement txs may — those inherit this assignment.
    pub assigned_versions: AssignedVersions,
}

impl From<VerifiedExecutableTransaction> for Schedulable<VerifiedExecutableTransaction> {
    fn from(tx: VerifiedExecutableTransaction) -> Self {
        Schedulable::Transaction(tx)
    }
}

// AsTx is like Deref, in that it allows us to use either refs or values in Schedulable.
// Deref does not work because it conflicts with the impl of Deref for VerifiedExecutableTransaction.
pub trait AsTx {
    fn as_tx(&self) -> &VerifiedExecutableTransaction;
}

impl AsTx for VerifiedExecutableTransaction {
    fn as_tx(&self) -> &VerifiedExecutableTransaction {
        self
    }
}

impl AsTx for &'_ VerifiedExecutableTransaction {
    fn as_tx(&self) -> &VerifiedExecutableTransaction {
        self
    }
}

impl Schedulable<&'_ VerifiedExecutableTransaction> {
    // Cannot use the blanket ToOwned trait impl because it just calls clone.
    pub fn to_owned_schedulable(&self) -> Schedulable<VerifiedExecutableTransaction> {
        match self {
            Schedulable::Transaction(tx) => Schedulable::Transaction((*tx).clone()),
            Schedulable::AccumulatorSettlement(info) => {
                Schedulable::AccumulatorSettlement(info.clone())
            }
        }
    }
}

impl<T> Schedulable<T> {
    pub fn as_tx(&self) -> Option<&VerifiedExecutableTransaction>
    where
        T: AsTx,
    {
        match self {
            Schedulable::Transaction(tx) => Some(tx.as_tx()),
            // AccumulatorSettlement is a placeholder, not a real tx —
            // shared-input-version assignment, executed-digest cache
            // hits, etc. don't apply to it. Callers that hit this
            // branch should be filtering it out at the
            // SettlementScheduler boundary.
            Schedulable::AccumulatorSettlement(_) => None,
        }
    }

    pub fn shared_input_objects(
        &self,
        epoch_store: &AuthorityPerEpochStore,
    ) -> Box<dyn Iterator<Item = SharedInputObject> + '_>
    where
        T: AsTx,
    {
        match self {
            Schedulable::Transaction(tx) => Box::new(tx.as_tx().shared_input_objects()),
            // No shared inputs on the placeholder. The actual
            // settlement transactions the scheduler builds may
            // declare them, but those go through the regular
            // ExecutionScheduler with their own assignment.
            Schedulable::AccumulatorSettlement(_) => Box::new(std::iter::empty()),
        }
    }

    pub fn non_shared_input_object_keys(&self) -> Vec<ObjectKey>
    where
        T: AsTx,
    {
        match self {
            Schedulable::Transaction(tx) => transaction_non_shared_input_object_keys(tx.as_tx())
                .expect("Transaction input should have been verified"),
            Schedulable::AccumulatorSettlement(_) => Vec::new(),
        }
    }

    pub fn receiving_object_keys(&self) -> Vec<ObjectKey>
    where
        T: AsTx,
    {
        match self {
            Schedulable::Transaction(tx) => transaction_receiving_object_keys(tx.as_tx()),
            Schedulable::AccumulatorSettlement(_) => Vec::new(),
        }
    }

    pub fn key(&self) -> TransactionKey
    where
        T: AsTx,
    {
        match self {
            Schedulable::Transaction(tx) => tx.as_tx().key(),
            Schedulable::AccumulatorSettlement(info) => info.settlement_key,
        }
    }
}

#[must_use]
#[derive(Default, Eq, PartialEq, Debug)]
pub struct ConsensusSharedObjVerAssignment {
    pub shared_input_next_versions: HashMap<ConsensusObjectSequenceKey, Version>,
    pub assigned_versions: AssignedTxAndVersions,
}

impl SharedObjVerManager {
    pub fn assign_versions_from_consensus<'a, T>(
        epoch_store: &AuthorityPerEpochStore,
        cache_reader: &dyn ObjectCacheRead,
        assignables: impl Iterator<Item = &'a Schedulable<T>> + Clone,
    ) -> SomaResult<ConsensusSharedObjVerAssignment>
    where
        T: AsTx + 'a,
    {
        let mut shared_input_next_versions = get_or_init_versions(
            assignables.clone().flat_map(|a| a.shared_input_objects(epoch_store)),
            epoch_store,
            cache_reader,
        )?;
        let mut assigned_versions = Vec::new();
        for assignable in assignables {
            let cert_assigned_versions = Self::assign_versions_for_certificate(
                epoch_store,
                assignable,
                &mut shared_input_next_versions,
            );
            assigned_versions.push((assignable.key(), cert_assigned_versions));
        }

        Ok(ConsensusSharedObjVerAssignment {
            shared_input_next_versions,
            assigned_versions: AssignedTxAndVersions::new(assigned_versions),
        })
    }

    pub fn assign_versions_from_effects(
        certs_and_effects: &[(&VerifiedExecutableTransaction, &TransactionEffects)],
        epoch_store: &AuthorityPerEpochStore,
        cache_reader: &dyn ObjectCacheRead,
    ) -> AssignedTxAndVersions {
        // We don't care about the results since we can use effects to assign versions.
        // But we must call it to make sure whenever a consensus object is touched the first time
        // during an epoch, either through consensus or through checkpoint executor,
        // its next version must be initialized. This is because we initialize the next version
        // of a consensus object in an epoch by reading the current version from the object store.
        // This must be done before we mutate it the first time, otherwise we would be initializing
        // it with the wrong version.
        let _ = get_or_init_versions(
            certs_and_effects
                .iter()
                .flat_map(|(cert, _)| cert.transaction_data().shared_input_objects().into_iter()),
            epoch_store,
            cache_reader,
        );
        let mut assigned_versions = Vec::new();
        for (cert, effects) in certs_and_effects {
            let initial_version_map: BTreeMap<_, _> = cert
                .transaction_data()
                .shared_input_objects()
                .into_iter()
                .map(|input| input.into_id_and_version())
                .collect();
            let cert_assigned_versions: Vec<_> = effects
                .input_shared_objects()
                .into_iter()
                .map(|iso| {
                    let (id, version) = iso.id_and_version();
                    let initial_version = initial_version_map
                        .get(&id)
                        .expect("transaction must have all inputs from effects");
                    ((id, *initial_version), version)
                })
                .collect();
            let tx_key = cert.key();
            trace!(
                ?tx_key,
                ?cert_assigned_versions,
                "assigned consensus object versions from effects"
            );
            assigned_versions.push((tx_key, AssignedVersions::new(cert_assigned_versions)));
        }
        AssignedTxAndVersions::new(assigned_versions)
    }

    pub fn assign_versions_for_certificate(
        epoch_store: &AuthorityPerEpochStore,
        assignable: &Schedulable<impl AsTx>,
        shared_input_next_versions: &mut HashMap<ConsensusObjectSequenceKey, Version>,
    ) -> AssignedVersions {
        let shared_input_objects: Vec<_> = assignable.shared_input_objects(epoch_store).collect();
        let non_shared_input_keys = assignable.non_shared_input_object_keys();
        let receiving_object_keys = assignable.receiving_object_keys();

        let assigned = assign_versions_for_shared_inputs_inner(
            &shared_input_objects,
            &non_shared_input_keys,
            &receiving_object_keys,
            shared_input_next_versions,
        );

        let tx_key = assignable.key();
        trace!(?tx_key, assigned_versions = ?assigned, "locking shared objects");

        assigned
    }
}

/// Pure version-assignment logic, decoupled from `AuthorityPerEpochStore`.
///
/// Read-only shared inputs (`mutable: false`) deliberately do NOT bump
/// their `next_version` entry, so subsequent readers in the same commit
/// window see the same version and can be scheduled in parallel. This is
/// the Sui-Clock parallelism property: `0x6` is mutated by the prologue
/// once per commit, then thousands of user transactions read it without
/// contention.
///
/// Mutable shared inputs bump `next_version` to `lamport_increment(all
/// inputs of this tx)` so the next transaction sees the post-mutation
/// version.
pub(crate) fn assign_versions_for_shared_inputs_inner(
    shared_input_objects: &[SharedInputObject],
    non_shared_input_keys: &[ObjectKey],
    receiving_object_keys: &[ObjectKey],
    shared_input_next_versions: &mut HashMap<ConsensusObjectSequenceKey, Version>,
) -> AssignedVersions {
    if shared_input_objects.is_empty() {
        return AssignedVersions::new(vec![]);
    }

    let mut input_object_keys: Vec<ObjectKey> = non_shared_input_keys.to_vec();
    input_object_keys.extend_from_slice(receiving_object_keys);

    let mut assigned_versions = Vec::with_capacity(shared_input_objects.len());
    let mut mutated_keys: Vec<ConsensusObjectSequenceKey> =
        Vec::with_capacity(shared_input_objects.len());

    for SharedInputObject { id, initial_shared_version, mutable } in shared_input_objects {
        let key = (*id, *initial_shared_version);
        let assigned_version = *shared_input_next_versions
            .get(&key)
            .expect("shared input must be in next_versions map");
        assigned_versions.push((key, assigned_version));
        input_object_keys.push(ObjectKey(*id, assigned_version));
        if *mutable {
            mutated_keys.push(key);
        }
    }

    let next_version = Version::lamport_increment(input_object_keys.iter().map(|obj| obj.1));
    assert!(next_version.is_valid(), "Assigned version must be valid. Got {:?}", next_version);

    for key in &mutated_keys {
        shared_input_next_versions
            .insert(*key, next_version)
            .expect("Object must exist in shared_input_next_versions.");
    }

    AssignedVersions::new(assigned_versions)
}

fn get_or_init_versions<'a>(
    shared_input_objects: impl Iterator<Item = SharedInputObject> + 'a,
    epoch_store: &AuthorityPerEpochStore,
    cache_reader: &dyn ObjectCacheRead,
) -> SomaResult<HashMap<ConsensusObjectSequenceKey, Version>> {
    let mut shared_input_objects: Vec<_> =
        shared_input_objects.map(|so| so.into_id_and_version()).collect();

    shared_input_objects.sort();
    shared_input_objects.dedup();

    epoch_store.get_or_init_next_object_versions(&shared_input_objects, cache_reader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::object::ObjectID;
    use types::{
        CLOCK_OBJECT_ID, CLOCK_OBJECT_SHARED_VERSION, SYSTEM_STATE_OBJECT_ID,
        SYSTEM_STATE_OBJECT_SHARED_VERSION,
    };

    fn clock_read() -> SharedInputObject {
        SharedInputObject::CLOCK_OBJ_READ
    }

    fn clock_mut() -> SharedInputObject {
        SharedInputObject::CLOCK_OBJ_MUT
    }

    fn system_state_mut() -> SharedInputObject {
        SharedInputObject::SYSTEM_OBJ
    }

    /// Read-only refs to Clock must NOT bump `next_version` — this is the
    /// parallelism property. Two consecutive reader transactions see the
    /// same Clock version.
    #[test]
    fn read_only_shared_input_does_not_bump_next_version() {
        let key = (CLOCK_OBJECT_ID, CLOCK_OBJECT_SHARED_VERSION);
        let mut next = HashMap::new();
        next.insert(key, Version::from_u64(5));

        let assigned1 = assign_versions_for_shared_inputs_inner(
            &[clock_read()],
            &[],
            &[],
            &mut next,
        );
        assert_eq!(assigned1.as_slice(), &[(key, Version::from_u64(5))]);
        assert_eq!(next[&key], Version::from_u64(5), "first read must not bump");

        let assigned2 = assign_versions_for_shared_inputs_inner(
            &[clock_read()],
            &[],
            &[],
            &mut next,
        );
        assert_eq!(
            assigned2.as_slice(),
            &[(key, Version::from_u64(5))],
            "second reader sees the same version as the first"
        );
        assert_eq!(next[&key], Version::from_u64(5), "second read must not bump either");
    }

    /// Mutable refs to a shared object DO bump `next_version` so the next
    /// transaction sees the post-mutation state.
    #[test]
    fn mutable_shared_input_bumps_next_version() {
        let key = (CLOCK_OBJECT_ID, CLOCK_OBJECT_SHARED_VERSION);
        let mut next = HashMap::new();
        next.insert(key, Version::from_u64(5));

        let assigned = assign_versions_for_shared_inputs_inner(
            &[clock_mut()],
            &[],
            &[],
            &mut next,
        );
        assert_eq!(assigned.as_slice(), &[(key, Version::from_u64(5))]);
        // lamport_increment over the single shared input version 5 → 6
        assert_eq!(next[&key], Version::from_u64(6), "mutable ref must bump");
    }

    /// Sui-Clock pattern: prologue mutates Clock (bumps), then many readers
    /// see the new version without further bumps.
    #[test]
    fn prologue_then_many_readers_serialize_only_on_prologue() {
        let key = (CLOCK_OBJECT_ID, CLOCK_OBJECT_SHARED_VERSION);
        let mut next = HashMap::new();
        next.insert(key, Version::from_u64(10));

        // Prologue: mutates Clock at v=10 → bumps to 11
        let prologue = assign_versions_for_shared_inputs_inner(
            &[clock_mut()],
            &[],
            &[],
            &mut next,
        );
        assert_eq!(prologue.as_slice(), &[(key, Version::from_u64(10))]);
        assert_eq!(next[&key], Version::from_u64(11));

        // Now N readers all see v=11 without bumping
        for _ in 0..1000 {
            let r = assign_versions_for_shared_inputs_inner(
                &[clock_read()],
                &[],
                &[],
                &mut next,
            );
            assert_eq!(r.as_slice(), &[(key, Version::from_u64(11))]);
        }
        assert_eq!(
            next[&key],
            Version::from_u64(11),
            "1000 reads must not bump Clock's next_version"
        );

        // Next prologue mutates v=11 → bumps to 12
        let prologue2 = assign_versions_for_shared_inputs_inner(
            &[clock_mut()],
            &[],
            &[],
            &mut next,
        );
        assert_eq!(prologue2.as_slice(), &[(key, Version::from_u64(11))]);
        assert_eq!(next[&key], Version::from_u64(12));
    }

    /// Read-only access to one shared object must not interfere with
    /// mutation tracking of another shared object in the same tx.
    #[test]
    fn mixed_mutable_and_readonly_shared_inputs() {
        let clock_key = (CLOCK_OBJECT_ID, CLOCK_OBJECT_SHARED_VERSION);
        let ss_key = (SYSTEM_STATE_OBJECT_ID, SYSTEM_STATE_OBJECT_SHARED_VERSION);
        let mut next = HashMap::new();
        next.insert(clock_key, Version::from_u64(5));
        next.insert(ss_key, Version::from_u64(7));

        // Tx mutates SystemState, reads Clock.
        let assigned = assign_versions_for_shared_inputs_inner(
            &[system_state_mut(), clock_read()],
            &[],
            &[],
            &mut next,
        );
        // Both inputs assigned at their current versions
        assert!(assigned.as_slice().contains(&(ss_key, Version::from_u64(7))));
        assert!(assigned.as_slice().contains(&(clock_key, Version::from_u64(5))));

        // SystemState bumped (max(5,7)+1 = 8); Clock unchanged
        assert_eq!(next[&ss_key], Version::from_u64(8), "SystemState must bump");
        assert_eq!(next[&clock_key], Version::from_u64(5), "Clock must not bump");
    }

    /// Empty shared input list short-circuits without panicking.
    #[test]
    fn no_shared_inputs_returns_empty_assignment() {
        let mut next = HashMap::new();
        let assigned =
            assign_versions_for_shared_inputs_inner(&[], &[], &[], &mut next);
        assert!(assigned.as_slice().is_empty());
    }

    /// Subsequent mutators see the bumped version, so updates serialize as
    /// expected (lamport-correct).
    #[test]
    fn consecutive_mutators_chain_versions() {
        let key = (CLOCK_OBJECT_ID, CLOCK_OBJECT_SHARED_VERSION);
        let mut next = HashMap::new();
        next.insert(key, Version::from_u64(1));

        let a = assign_versions_for_shared_inputs_inner(
            &[clock_mut()],
            &[],
            &[],
            &mut next,
        );
        assert_eq!(a.as_slice()[0].1, Version::from_u64(1));
        assert_eq!(next[&key], Version::from_u64(2));

        let b = assign_versions_for_shared_inputs_inner(
            &[clock_mut()],
            &[],
            &[],
            &mut next,
        );
        assert_eq!(b.as_slice()[0].1, Version::from_u64(2));
        assert_eq!(next[&key], Version::from_u64(3));
    }

    /// Lamport increment must include non-shared and receiving object
    /// versions, not just shared ones.
    #[test]
    fn lamport_increment_uses_all_input_versions() {
        let key = (CLOCK_OBJECT_ID, CLOCK_OBJECT_SHARED_VERSION);
        let mut next = HashMap::new();
        next.insert(key, Version::from_u64(2));

        // Owned input at version 100 should dominate the lamport increment.
        let other_id = ObjectID::random();
        let owned = ObjectKey(other_id, Version::from_u64(100));

        let _ = assign_versions_for_shared_inputs_inner(
            &[clock_mut()],
            std::slice::from_ref(&owned),
            &[],
            &mut next,
        );
        // max(2, 100) + 1 = 101
        assert_eq!(next[&key], Version::from_u64(101));
    }
}
