use std::collections::{BTreeMap, HashMap, HashSet};

use either::Either;
use tracing::trace;
use types::{
    base::ConsensusObjectSequenceKey,
    digests::TransactionDigest,
    effects::{TransactionEffects, TransactionEffectsAPI as _},
    error::SomaResult,
    object::Version,
    storage::{
        transaction_non_shared_input_object_keys, transaction_receiving_object_keys, ObjectKey,
    },
    transaction::{SharedInputObject, TransactionKey, VerifiedExecutableTransaction},
};

use crate::{
    authority_per_epoch_store::{AuthorityPerEpochStore, CancelConsensusCertificateReason},
    cache::ObjectCacheRead,
};

pub struct SharedObjVerManager {}

/// Version assignments for a single transaction
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssignedVersions {
    pub shared_object_versions: Vec<(ConsensusObjectSequenceKey, Version)>,
}

impl AssignedVersions {
    pub fn new(shared_object_versions: Vec<(ConsensusObjectSequenceKey, Version)>) -> Self {
        Self {
            shared_object_versions,
        }
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
        }
    }

    pub fn shared_input_objects(
        &self,
        epoch_store: &AuthorityPerEpochStore,
    ) -> impl Iterator<Item = SharedInputObject> + '_
    where
        T: AsTx,
    {
        match self {
            Schedulable::Transaction(tx) => tx.as_tx().shared_input_objects(),
        }
    }

    pub fn non_shared_input_object_keys(&self) -> Vec<ObjectKey>
    where
        T: AsTx,
    {
        match self {
            Schedulable::Transaction(tx) => transaction_non_shared_input_object_keys(tx.as_tx())
                .expect("Transaction input should have been verified"),
        }
    }

    pub fn receiving_object_keys(&self) -> Vec<ObjectKey>
    where
        T: AsTx,
    {
        match self {
            Schedulable::Transaction(tx) => transaction_receiving_object_keys(tx.as_tx()),
        }
    }

    pub fn key(&self) -> TransactionKey
    where
        T: AsTx,
    {
        match self {
            Schedulable::Transaction(tx) => tx.as_tx().key(),
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
            assignables
                .clone()
                .flat_map(|a| a.shared_input_objects(epoch_store)),
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

        if shared_input_objects.is_empty() {
            // No shared object used by this transaction. No need to assign versions.
            return AssignedVersions::new(vec![]);
        }

        let tx_key = assignable.key();

        let mut input_object_keys = assignable.non_shared_input_object_keys();
        let mut assigned_versions = Vec::with_capacity(shared_input_objects.len());

        // Record receiving object versions towards the shared version computation.
        let receiving_object_keys = assignable.receiving_object_keys();
        input_object_keys.extend(receiving_object_keys);

        for (
            SharedInputObject {
                id,
                initial_shared_version,
                mutable,
            },
            assigned_version,
        ) in shared_input_objects.iter().map(|obj| {
            (
                obj,
                *shared_input_next_versions
                    .get(&obj.id_and_version())
                    .unwrap(),
            )
        }) {
            assigned_versions.push(((*id, *initial_shared_version), assigned_version));
            input_object_keys.push(ObjectKey(*id, assigned_version));
        }

        let next_version = Version::lamport_increment(input_object_keys.iter().map(|obj| obj.1));
        assert!(
            next_version.is_valid(),
            "Assigned version must be valid. Got {:?}",
            next_version
        );

        // Update the next version for the shared objects.
        assigned_versions.iter().for_each(|(id, version)| {
            assert!(
                version.is_valid(),
                "Assigned version must be a valid version."
            );
            shared_input_next_versions
                .insert(*id, next_version)
                .expect("Object must exist in shared_input_next_versions.");
        });

        trace!(
            ?tx_key,
            ?assigned_versions,
            ?next_version,
            "locking shared objects"
        );

        AssignedVersions::new(assigned_versions)
    }
}

fn get_or_init_versions<'a>(
    shared_input_objects: impl Iterator<Item = SharedInputObject> + 'a,
    epoch_store: &AuthorityPerEpochStore,
    cache_reader: &dyn ObjectCacheRead,
) -> SomaResult<HashMap<ConsensusObjectSequenceKey, Version>> {
    let mut shared_input_objects: Vec<_> = shared_input_objects
        .map(|so| so.into_id_and_version())
        .collect();

    shared_input_objects.sort();
    shared_input_objects.dedup();

    epoch_store.get_or_init_next_object_versions(&shared_input_objects, cache_reader)
}
