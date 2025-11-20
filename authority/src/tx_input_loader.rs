use once_cell::unsync::OnceCell;
use std::{collections::HashMap, sync::Arc};

use itertools::izip;
use tracing::{debug, instrument};
use types::{
    base::FullObjectID,
    committee::EpochId,
    digests::TransactionDigest,
    error::{SomaError, SomaResult},
    object::{ObjectRef, Version},
    storage::{FullObjectKey, ObjectKey},
    transaction::{
        InputObjectKind, InputObjects, ObjectReadResult, ObjectReadResultKind,
        ReceivingObjectReadResult, ReceivingObjectReadResultKind, ReceivingObjects, TransactionKey,
    },
};

use crate::{
    authority_per_epoch_store::{AuthorityPerEpochStore, CertLockGuard},
    cache::ObjectCacheRead,
};

pub(crate) struct TransactionInputLoader {
    cache: Arc<dyn ObjectCacheRead>,
}

impl TransactionInputLoader {
    pub fn new(cache: Arc<dyn ObjectCacheRead>) -> Self {
        Self { cache }
    }
}

impl TransactionInputLoader {
    /// Read the inputs for a transaction that the validator was asked to sign.
    ///
    /// tx_digest is provided so that the inputs can be cached with the tx_digest and returned with
    /// a single hash map lookup when notify_read_objects_for_execution is called later.
    /// TODO: implement this caching
    #[instrument(level = "debug", skip_all)]
    pub fn read_objects_for_signing(
        &self,
        _tx_digest_for_caching: Option<&TransactionDigest>,
        input_object_kinds: &[InputObjectKind],
        receiving_objects: &[ObjectRef],
        epoch_id: EpochId,
    ) -> SomaResult<(InputObjects, ReceivingObjects)> {
        let mut input_results = vec![None; input_object_kinds.len()];
        let mut object_refs = Vec::with_capacity(input_object_kinds.len());
        let mut fetch_indices = Vec::with_capacity(input_object_kinds.len());

        for (i, kind) in input_object_kinds.iter().enumerate() {
            match kind {
                InputObjectKind::SharedObject {
                    id,
                    initial_shared_version,
                    ..
                } => match self.cache.get_object(id)? {
                    Some(object) => {
                        input_results[i] = Some(ObjectReadResult::new(*kind, object.into()))
                    }
                    None => {
                        if let Some((version, digest)) =
                            self.cache.get_last_shared_object_deletion_info(
                                FullObjectID::new(*id, Some(*initial_shared_version)),
                                epoch_id,
                            )
                        {
                            input_results[i] = Some(ObjectReadResult {
                                input_object_kind: *kind,
                                object: ObjectReadResultKind::DeletedSharedObject(version, digest),
                            });
                        } else {
                            return Err(SomaError::from(kind.object_not_found_error()));
                        }
                    }
                },
                InputObjectKind::ImmOrOwnedObject(objref) => {
                    object_refs.push(*objref);
                    fetch_indices.push(i);
                }
            }
        }

        let objects = self
            .cache
            .multi_get_objects_with_more_accurate_error_return(&object_refs)?;
        assert_eq!(objects.len(), object_refs.len());
        for (index, object) in fetch_indices.into_iter().zip(objects.into_iter()) {
            input_results[index] = Some(ObjectReadResult {
                input_object_kind: input_object_kinds[index],
                object: ObjectReadResultKind::Object(object),
            });
        }

        let receiving_results =
            self.read_receiving_objects_for_signing(receiving_objects, epoch_id)?;

        Ok((
            input_results
                .into_iter()
                .map(Option::unwrap)
                .collect::<Vec<_>>()
                .into(),
            receiving_results,
        ))
    }

    /// Read the inputs for a transaction that is ready to be executed.
    ///
    /// epoch_store is used to resolve the versions of any shared input objects.
    ///
    /// This function panics if any inputs are not available, as TransactionManager should already
    /// have verified that the transaction is ready to be executed.
    ///
    /// The tx_digest is provided here to support the following optimization (not yet implemented):
    /// All the owned input objects will likely have been loaded during transaction signing, and
    /// can be stored as a group with the transaction_digest as the key, allowing the lookup to
    /// proceed with only a single hash map lookup. (additional lookups may be necessary for shared
    /// inputs, since the versions are not known at signing time). Receiving objects could be
    /// cached, but only with appropriate invalidation logic for when an object is received by a
    /// different tx first.
    #[instrument(level = "debug", skip_all)]
    pub fn read_objects_for_execution(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        tx_key: &TransactionKey,
        _tx_lock: &CertLockGuard, // see below for why this is needed
        input_object_kinds: &[InputObjectKind],
        epoch_id: EpochId,
    ) -> SomaResult<InputObjects> {
        let assigned_shared_versions_cell: OnceCell<Option<HashMap<_, _>>> = OnceCell::new();

        let mut results = vec![None; input_object_kinds.len()];
        let mut object_keys = Vec::with_capacity(input_object_kinds.len());
        let mut fetches = Vec::with_capacity(input_object_kinds.len());

        for (i, input) in input_object_kinds.iter().enumerate() {
            match input {
                InputObjectKind::ImmOrOwnedObject(objref) => {
                    object_keys.push(objref.into());
                    fetches.push((i, input));
                }
                InputObjectKind::SharedObject {
                    id,
                    initial_shared_version,
                    ..
                } => {
                    let assigned_shared_versions = assigned_shared_versions_cell
                        .get_or_init(|| {
                            epoch_store
                                .get_assigned_shared_object_versions(tx_key)
                                .map(|versions| versions.into_iter().collect())
                        })
                        .as_ref()
                        .unwrap_or_else(|| {
                            // Important to hold the _tx_lock here - otherwise it would be possible
                            // for a concurrent execution of the same tx to enter this point after
                            // the first execution has finished and the assigned shared versions
                            // have been deleted.
                            panic!(
                                "Failed to get assigned shared versions for transaction {tx_key:?}"
                            );
                        });

                    let initial_shared_version = *initial_shared_version;
                    // If we find a set of assigned versions but an object is missing, it indicates
                    // a serious inconsistency:
                    let version: &Version = assigned_shared_versions
                        .get(&(*id, initial_shared_version))
                        .unwrap_or_else(|| {
                            panic!(
                                "Shared object version should have been assigned. key: \
                                 {tx_key:?}, obj id: {id:?}"
                            )
                        });
                    if version.is_cancelled() {
                        // Do not need to fetch shared object for cancelled transaction.
                        results[i] = Some(ObjectReadResult {
                            input_object_kind: *input,
                            object: ObjectReadResultKind::CancelledTransactionSharedObject(
                                *version,
                            ),
                        })
                    } else {
                        object_keys.push(ObjectKey(*id, *version));
                        fetches.push((i, input));
                    }
                }
            }
        }

        let objects = self.cache.multi_get_objects_by_key(&object_keys)?;

        assert!(objects.len() == object_keys.len() && objects.len() == fetches.len());

        for (object, key, (index, input)) in izip!(
            objects.into_iter(),
            object_keys.into_iter(),
            fetches.into_iter()
        ) {
            results[index] = Some(match (object, input) {
                (Some(obj), input_object_kind) => ObjectReadResult {
                    input_object_kind: *input_object_kind,
                    object: obj.into(),
                },
                (
                    None,
                    InputObjectKind::SharedObject {
                        id,
                        initial_shared_version,
                        ..
                    },
                ) => {
                    assert!(key.1.is_valid());
                    // Check if the object was deleted by a concurrently certified tx
                    let version = key.1;
                    if let Some(dependency) =
                        self.cache.get_deleted_shared_object_previous_tx_digest(
                            FullObjectKey::new(
                                FullObjectID::new(*id, Some(*initial_shared_version)),
                                version,
                            ),
                            epoch_id,
                        )
                    {
                        ObjectReadResult {
                            input_object_kind: *input,
                            object: ObjectReadResultKind::DeletedSharedObject(version, dependency),
                        }
                    } else {
                        // Check if this is a shared object with an assigned version
                        let assigned_shared_versions = assigned_shared_versions_cell
                            .get_or_init(|| {
                                epoch_store
                                    .get_assigned_shared_object_versions(tx_key)
                                    .map(|versions| versions.into_iter().collect())
                            })
                            .as_ref()
                            .unwrap_or_else(|| {
                                panic!(
                                    "Failed to get assigned shared versions for transaction \
                                     {tx_key:?}"
                                );
                            });

                        // If this specific object at this specific version is in the assigned versions,
                        // we're in the circular dependency situation where the transaction needs this
                        // object version but that version won't exist until after the transaction executes
                        if assigned_shared_versions.get(&(*id, *initial_shared_version))
                            == Some(&version)
                        {
                            debug!(
                                "Creating special placeholder for shared object with assigned \
                                 version. tx={tx_key:?}, object={id:?}, version={version:?}"
                            );

                            // Use CancelledTransactionSharedObject as a placeholder that won't
                            // trigger validation errors in downstream code
                            ObjectReadResult {
                                input_object_kind: *input,
                                object: ObjectReadResultKind::CancelledTransactionSharedObject(
                                    version,
                                ),
                            }
                        } else {
                            // Normal case - the object should exist but doesn't
                            panic!(
                                "All dependencies of tx {tx_key:?} should have been executed now, \
                                 but Shared Object id: {}, version: {} is absent in epoch \
                                 {epoch_id}",
                                *id,
                                version.value()
                            );
                        }
                    }
                }
                _ => panic!(
                    "All dependencies of tx {tx_key:?} should have been executed now, but obj \
                     {key:?} is absent"
                ),
            });
        }

        Ok(results
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>()
            .into())
    }
}

// private methods
impl TransactionInputLoader {
    fn read_receiving_objects_for_signing(
        &self,
        receiving_objects: &[ObjectRef],
        epoch_id: EpochId,
    ) -> SomaResult<ReceivingObjects> {
        let mut receiving_results = Vec::with_capacity(receiving_objects.len());
        for objref in receiving_objects {
            // Note: the digest is checked later in check_transaction_input
            let (object_id, version, _) = objref;

            // TODO: Add support for receiving ConsensusV2 objects. For now this assumes fastpath.
            if self.cache.have_received_object_at_version(
                FullObjectKey::new(FullObjectID::new(*object_id, None), *version),
                epoch_id,
            ) {
                receiving_results.push(ReceivingObjectReadResult::new(
                    *objref,
                    ReceivingObjectReadResultKind::PreviouslyReceivedObject,
                ));
                continue;
            }

            let Some(object) = self.cache.get_object(object_id)? else {
                return Err(SomaError::ObjectNotFound {
                    object_id: *object_id,
                    version: Some(*version),
                }
                .into());
            };

            receiving_results.push(ReceivingObjectReadResult::new(*objref, object.into()));
        }
        Ok(receiving_results.into())
    }
}
