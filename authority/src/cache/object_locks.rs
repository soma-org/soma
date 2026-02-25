// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use core::panic;

use crate::{authority_per_epoch_store::AuthorityPerEpochStore, authority_store::LockDetails};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry as DashMapEntry;
use tracing::{debug, error, info, instrument, trace};
use types::{
    digests::TransactionDigest,
    error::{SomaError, SomaResult},
    object::{Object, ObjectID, ObjectRef},
    storage::object_store::ObjectStore,
    transaction::VerifiedSignedTransaction,
};

use super::writeback_cache::WritebackCache;

type RefCount = usize;

pub(super) struct ObjectLocks {
    // When acquire transaction locks, lock entries are briefly inserted into this map. The map
    // exists to provide atomic test-and-set operations on the locks. After all locks have been inserted
    // into the map, they are written to the db, and then all locks are removed from the map.
    //
    // After a transaction has been executed, newly created objects are available to be locked.
    // But, because of crash recovery, we cannot rule out that a lock may already exist in the db for
    // those objects. Therefore we do a db read for each object we are locking.
    //
    // TODO: find a strategy to allow us to avoid db reads for each object.
    locked_transactions: DashMap<ObjectRef, (RefCount, LockDetails)>,
}

impl ObjectLocks {
    pub fn new() -> Self {
        Self { locked_transactions: DashMap::new() }
    }

    pub(crate) fn get_transaction_lock(
        &self,
        obj_ref: &ObjectRef,
        epoch_store: &AuthorityPerEpochStore,
    ) -> SomaResult<Option<LockDetails>> {
        // We don't consult the in-memory state here. We are only interested in state that
        // has been committed to the db. This is because in memory state is reverted
        // if the transaction is not successfully locked.
        epoch_store.tables()?.get_locked_transaction(obj_ref)
    }

    /// Attempts to atomically test-and-set a transaction lock on an object.
    /// If the lock is already set to a conflicting transaction, an error is returned.
    /// If the lock is not set, or is already set to the same transaction, the lock is
    /// set.
    pub(crate) fn try_set_transaction_lock(
        &self,
        obj_ref: &ObjectRef,
        new_lock: LockDetails,
        epoch_store: &AuthorityPerEpochStore,
    ) -> SomaResult {
        // entry holds a lock on the dashmap shard, so this function operates atomicly
        let entry = self.locked_transactions.entry(*obj_ref);

        // TODO: currently, the common case for this code is that we will miss the cache
        // and read from the db. It is difficult to implement negative caching, since we
        // may have restarted, in which case there could be locks in the db that we do
        // not have in the cache. We may want to explore strategies for proving there
        // cannot be a lock in the db that we do not know about. Two possibilities are:
        //
        // 1. Read all locks into memory at startup (and keep them there). The lifetime
        //    of locks is relatively short in the common case, so this might be feasible.
        // 2. Find some strategy to distinguish between the cases where we are re-executing
        //    old transactions after restarting vs executing transactions that we have never
        //    seen before. The output objects of novel transactions cannot previously have
        //    been locked on this validator.
        //
        // Solving this is not terribly important as it is not in the execution path, and
        // hence only improves the latency of transaction signing, not transaction execution
        let prev_lock = match entry {
            DashMapEntry::Vacant(vacant) => {
                let tables = epoch_store.tables()?;
                if let Some(lock_details) = tables.get_locked_transaction(obj_ref)? {
                    trace!("read lock from db: {:?}", lock_details);
                    vacant.insert((1, lock_details.clone()));
                    lock_details
                } else {
                    trace!("set lock: {:?}", new_lock);
                    vacant.insert((1, new_lock.clone()));
                    new_lock.clone()
                }
            }
            DashMapEntry::Occupied(mut occupied) => {
                occupied.get_mut().0 += 1;
                occupied.get().1.clone()
            }
        };

        if prev_lock != new_lock {
            debug!("lock conflict detected for {:?}: {:?} != {:?}", obj_ref, prev_lock, new_lock);
            Err(SomaError::ObjectLockConflict {
                obj_ref: *obj_ref,
                pending_transaction: prev_lock.tx_digest,
            }
            .into())
        } else {
            Ok(())
        }
    }

    pub(crate) fn clear(&self) {
        info!("clearing old transaction locks");
        self.locked_transactions.clear();
    }

    fn verify_live_object(obj_ref: &ObjectRef, live_object: &Object) -> SomaResult {
        debug_assert_eq!(obj_ref.0, live_object.id());
        if obj_ref.1 != live_object.version() {
            debug!(
                "object version unavailable for consumption: {:?} (current: {:?})",
                obj_ref,
                live_object.version()
            );
            return Err(SomaError::ObjectVersionUnavailableForConsumption {
                provided_obj_ref: *obj_ref,
                current_version: live_object.version(),
            });
        }

        let live_digest = live_object.digest();
        if obj_ref.2 != live_digest {
            return Err(SomaError::InvalidObjectDigest {
                object_id: obj_ref.0,
                expected_digest: live_digest,
            });
        }

        Ok(())
    }

    fn clear_cached_locks(&self, locks: &[(ObjectRef, LockDetails)]) {
        for (obj_ref, lock) in locks {
            let entry = self.locked_transactions.entry(*obj_ref);
            let mut occupied = match entry {
                DashMapEntry::Vacant(_) => {
                    debug!("lock must exist for object: {:?}", obj_ref);
                    continue;
                }
                DashMapEntry::Occupied(occupied) => occupied,
            };

            if occupied.get().1 == *lock {
                occupied.get_mut().0 -= 1;
                if occupied.get().0 == 0 {
                    trace!("clearing lock: {:?}", lock);
                    occupied.remove();
                }
            } else {
                // this is impossible because the only case in which we overwrite a
                // lock is when the lock is from a previous epoch. but we are holding
                // execution_lock, so the epoch cannot have changed.
                panic!("lock was changed since we set it");
            }
        }
    }

    fn multi_get_objects_must_exist(
        cache: &WritebackCache,
        object_ids: &[ObjectID],
    ) -> SomaResult<Vec<Object>> {
        let objects = cache.multi_get_objects(object_ids);
        let mut result = Vec::with_capacity(objects.len());
        for (i, object) in objects.into_iter().enumerate() {
            if let Some(object) = object {
                result.push(object);
            } else {
                return Err(SomaError::ObjectNotFound { object_id: object_ids[i], version: None });
            }
        }
        Ok(result)
    }

    #[instrument(level = "debug", skip_all)]
    pub(crate) fn acquire_transaction_locks(
        &self,
        cache: &WritebackCache,
        epoch_store: &AuthorityPerEpochStore,
        owned_input_objects: &[ObjectRef],
        tx_digest: TransactionDigest,
        signed_transaction: Option<VerifiedSignedTransaction>,
    ) -> SomaResult {
        let object_ids = owned_input_objects.iter().map(|o| o.0).collect::<Vec<_>>();
        let live_objects = Self::multi_get_objects_must_exist(cache, &object_ids)?;

        // Only live objects can be locked
        for (obj_ref, live_object) in owned_input_objects.iter().zip(live_objects.iter()) {
            Self::verify_live_object(obj_ref, live_object)?;
        }

        let mut locks_to_write: Vec<(_, LockDetails)> =
            Vec::with_capacity(owned_input_objects.len());

        // Sort the objects before locking. This is not required by the protocol (since it's okay to
        // reject any equivocating tx). However, this does prevent a confusing error on the client.
        // Consider the case:
        //   TX1: [o1, o2];
        //   TX2: [o2, o1];
        // If two threads race to acquire these locks, they might both acquire the first object, then
        // error when trying to acquire the second. The error returned to the client would say that there
        // is a conflicting tx on that object, but in fact neither object was locked and the tx was never
        // signed. If one client then retries, they will succeed (counterintuitively).
        let owned_input_objects = {
            let mut o = owned_input_objects.to_vec();
            o.sort_by_key(|o| o.0);
            o
        };

        let epoch = epoch_store.epoch();
        let lock = LockDetails { tx_digest, epoch };

        // Note that this function does not have to operate atomically. If there are two racing threads,
        // then they are either trying to lock the same transaction (in which case both will succeed),
        // or they are trying to lock the same object in two different transactions, in which case
        // the sender has equivocated, and we are under no obligation to help them form a cert.
        for obj_ref in owned_input_objects.iter() {
            match self.try_set_transaction_lock(obj_ref, lock.clone(), epoch_store) {
                Ok(()) => locks_to_write.push((*obj_ref, lock.clone())),
                Err(e) => {
                    // revert all pending writes and return error
                    // Note that reverting is not required for liveness, since a well formed and un-equivocating
                    // txn cannot fail to acquire locks.
                    // However, reverting is easy enough to do in this implementation that we do it anyway.
                    self.clear_cached_locks(&locks_to_write);
                    return Err(e);
                }
            }
        }

        // commit all writes to DB
        epoch_store
            .tables()?
            .write_transaction_locks(signed_transaction, locks_to_write.iter().cloned())?;

        // remove pending locks from unbounded storage
        self.clear_cached_locks(&locks_to_write);

        Ok(())
    }
}
