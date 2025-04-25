use std::collections::{BTreeMap, HashMap, HashSet};

use tracing::trace;
use types::{
    base::ConsensusObjectSequenceKey,
    digests::TransactionDigest,
    effects::{TransactionEffects, TransactionEffectsAPI},
    error::SomaResult,
    object::Version,
    storage::{
        transaction_non_shared_input_object_keys, transaction_receiving_object_keys, ObjectKey,
    },
    transaction::{
        SenderSignedData, SharedInputObject, TransactionKey, VerifiedExecutableTransaction,
    },
};

use crate::{
    cache::ObjectCacheRead,
    epoch_store::{AuthorityPerEpochStore, CancelConsensusCertificateReason},
};

pub struct SharedObjVerManager {}

pub type AssignedTxAndVersions = Vec<(TransactionKey, Vec<(ConsensusObjectSequenceKey, Version)>)>;

#[must_use]
#[derive(Default)]
pub struct ConsensusSharedObjVerAssignment {
    pub shared_input_next_versions: HashMap<ConsensusObjectSequenceKey, Version>,
    pub assigned_versions: AssignedTxAndVersions,
}

impl SharedObjVerManager {
    pub fn assign_versions_from_consensus(
        epoch_store: &AuthorityPerEpochStore,
        cache_reader: &dyn ObjectCacheRead,
        certificates: &[VerifiedExecutableTransaction],
        cancelled_txns: &BTreeMap<TransactionDigest, CancelConsensusCertificateReason>,
    ) -> SomaResult<ConsensusSharedObjVerAssignment> {
        let mut shared_input_next_versions = get_or_init_versions(
            certificates.iter().map(|cert| cert.data()),
            epoch_store,
            cache_reader,
        )?;
        let mut assigned_versions = Vec::new();

        for cert in certificates {
            if !cert.contains_shared_object() {
                continue;
            }
            let cert_assigned_versions = Self::assign_versions_for_certificate(
                cert,
                &mut shared_input_next_versions,
                cancelled_txns,
            );
            assigned_versions.push((cert.key(), cert_assigned_versions));
        }

        Ok(ConsensusSharedObjVerAssignment {
            shared_input_next_versions,
            assigned_versions,
        })
    }

    pub fn assign_versions_from_effects(
        certs_and_effects: &[(&VerifiedExecutableTransaction, &TransactionEffects)],
        epoch_store: &AuthorityPerEpochStore,
        cache_reader: &dyn ObjectCacheRead,
    ) -> AssignedTxAndVersions {
        // We don't care about the results since we can use effects to assign versions.
        // But we must call it to make sure whenever a shared object is touched the first time
        // during an epoch, either through consensus or through checkpoint executor,
        // its next version must be initialized. This is because we initialize the next version
        // of a shared object in an epoch by reading the current version from the object store.
        // This must be done before we mutate it the first time, otherwise we would be initializing
        // it with the wrong version.
        let _ = get_or_init_versions(
            certs_and_effects.iter().map(|(cert, _)| cert.data()),
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
                "locking shared objects from effects"
            );
            assigned_versions.push((tx_key, cert_assigned_versions));
        }
        assigned_versions
    }

    pub fn assign_versions_for_certificate(
        cert: &VerifiedExecutableTransaction,
        shared_input_next_versions: &mut HashMap<ConsensusObjectSequenceKey, Version>,
        cancelled_txns: &BTreeMap<TransactionDigest, CancelConsensusCertificateReason>,
    ) -> Vec<(ConsensusObjectSequenceKey, Version)> {
        let tx_digest = cert.digest();

        // Check if the transaction is cancelled due to congestion.
        let cancellation_info = cancelled_txns.get(tx_digest);
        let congested_objects_info: Option<HashSet<_>> =
            if let Some(CancelConsensusCertificateReason::CongestionOnObjects(congested_objects)) =
                &cancellation_info
            {
                Some(congested_objects.iter().cloned().collect())
            } else {
                None
            };
        let txn_cancelled = cancellation_info.is_some();

        // Make an iterator to update the locks of the transaction's shared objects.
        let shared_input_objects: Vec<_> = cert.shared_input_objects().collect();

        let mut input_object_keys = transaction_non_shared_input_object_keys(cert)
            .expect("Transaction input should have been verified");
        let mut assigned_versions = Vec::with_capacity(shared_input_objects.len());
        let mut is_mutable_input = Vec::with_capacity(shared_input_objects.len());
        // Record receiving object versions towards the shared version computation.
        let receiving_object_keys = transaction_receiving_object_keys(cert);
        input_object_keys.extend(receiving_object_keys);

        if txn_cancelled {
            // For cancelled transaction due to congestion, assign special versions to all shared objects.
            // Note that new lamport version does not depend on any shared objects.
            for SharedInputObject {
                id,
                initial_shared_version,
                ..
            } in shared_input_objects.iter()
            {
                let assigned_version = match cancellation_info {
                    Some(CancelConsensusCertificateReason::CongestionOnObjects(_)) => {
                        if congested_objects_info
                            .as_ref()
                            .is_some_and(|info| info.contains(id))
                        {
                            Version::CONGESTED
                        } else {
                            Version::CANCELLED_READ
                        }
                    }
                    None => unreachable!("cancelled transaction should have cancellation info"),
                };
                assigned_versions.push(((*id, *initial_shared_version), assigned_version));
                is_mutable_input.push(false);
            }
        } else {
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
                is_mutable_input.push(*mutable);
            }
        }

        let next_version = Version::lamport_increment(input_object_keys.iter().map(|obj| obj.1));
        assert!(
            next_version.is_valid(),
            "Assigned version must be valid. Got {:?}",
            next_version
        );

        if !txn_cancelled {
            // Update the next version for the shared objects.
            assigned_versions
                .iter()
                .zip(is_mutable_input)
                .filter_map(|((id, _), mutable)| {
                    if mutable {
                        Some((*id, next_version))
                    } else {
                        None
                    }
                })
                .for_each(|(id, version)| {
                    assert!(
                        version.is_valid(),
                        "Assigned version must be a valid version."
                    );
                    shared_input_next_versions
                        .insert(id, version)
                        .expect("Object must exist in shared_input_next_versions.");
                });
        }

        trace!(
            ?tx_digest,
            ?assigned_versions,
            ?next_version,
            ?txn_cancelled,
            "locking shared objects"
        );

        assigned_versions
    }
}

fn get_or_init_versions<'a>(
    transactions: impl Iterator<Item = &'a SenderSignedData>,
    epoch_store: &AuthorityPerEpochStore,
    cache_reader: &dyn ObjectCacheRead,
) -> SomaResult<HashMap<ConsensusObjectSequenceKey, Version>> {
    let mut shared_input_objects: Vec<_> = transactions
        .flat_map(|tx| {
            tx.transaction_data()
                .shared_input_objects()
                .into_iter()
                .map(|so| so.into_id_and_version())
        })
        .collect();

    shared_input_objects.sort();
    shared_input_objects.dedup();

    epoch_store.get_or_init_next_object_versions(&shared_input_objects, cache_reader)
}
