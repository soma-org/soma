use fastcrypto::bls12381::min_sig;
use parking_lot::RwLock;
use shared::{checksum::Checksum, crypto::EncryptionKey, digest::Digest, signed::Signed};
use std::{
    collections::BTreeMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::{
    error::{ShardError, ShardResult},
    types::{
        certified::Certified,
        encoder_committee::{EncoderIndex, Epoch},
        shard::Shard,
        shard_commit::ShardCommit,
    },
};

use super::Store;

/// In-memory storage for testing.
#[allow(unused)]
pub(crate) struct MemStore {
    inner: RwLock<Inner>,
}

#[allow(unused)]
struct Inner {
    // EPOCH, SHARD_REF, SLOT
    commit_digests: BTreeMap<
        (Epoch, Digest<Shard>, EncoderIndex),
        Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    >,
    // EPOCH, SHARD_REF, COMMITTER
    committers: BTreeMap<
        (Epoch, Digest<Shard>, EncoderIndex),
        Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    >,
    // EPOCH, SHARD_REF, SLOT
    certified_commits: BTreeMap<
        (Epoch, Digest<Shard>, EncoderIndex),
        Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    >,

    // EPOCH, SHARD_REF, SLOT
    reveals: BTreeMap<(Epoch, Digest<Shard>, EncoderIndex), (EncryptionKey, Checksum)>,
    first_commit_timestamp_ms: BTreeMap<(Epoch, Digest<Shard>), u64>,
    first_reveal_timestamp_ms: BTreeMap<(Epoch, Digest<Shard>), u64>,
}

impl MemStore {
    pub(crate) const fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                commit_digests: BTreeMap::new(),
                committers: BTreeMap::new(),
                certified_commits: BTreeMap::new(),
                reveals: BTreeMap::new(),
                first_commit_timestamp_ms: BTreeMap::new(),
                first_reveal_timestamp_ms: BTreeMap::new(),
            }),
        }
    }
}

impl Store for MemStore {
    fn lock_signed_commit_digest(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
        slot: EncoderIndex,
        committer: EncoderIndex,
        digest: Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let slot_key = (epoch, shard_ref, slot);
        let committer_key = (epoch, shard_ref, committer);
        let mut inner = self.inner.write();

        // Check both conditions first
        let committer_check = inner.committers.get(&committer_key);
        let slot_check = inner.commit_digests.get(&slot_key);

        match (committer_check, slot_check) {
            // Committer has committed before
            (Some(existing_digest), _) if existing_digest != &digest => {
                return Err(ShardError::ConflictingCommit(
                    "existing commit from committer".to_string(),
                ));
            }
            // Slot is taken with different commit
            (_, Some(existing)) if existing != &digest => {
                return Err(ShardError::ConflictingCommit(
                    "slot already has commit".to_string(),
                ));
            }
            // If we made it here, either there are no existing commits
            // or the existing commits match exactly
            (None, None) => {
                // Insert new commit
                inner.commit_digests.insert(slot_key, digest);
                inner.committers.insert(committer_key, digest);
            }
            // Everything matches, idempotent case
            _ => (),
        }
        Ok(())
    }
    fn atomic_certified_commit(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
        slot: EncoderIndex,
        certified_commit: Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    ) -> ShardResult<usize> {
        let slot_key = (epoch, shard_ref, slot);
        let shard_key = (epoch, shard_ref);

        let mut inner = self.inner.write();

        let was_inserted = match inner.certified_commits.get(&slot_key) {
            Some(existing_commit) => {
                if existing_commit == &certified_commit {
                    false
                } else {
                    return Err(ShardError::ConflictingCommit(
                        "slot already has different commit".to_string(),
                    ));
                }
            }
            None => {
                inner.certified_commits.insert(slot_key, certified_commit);
                true
            }
        };
        let start_key = (epoch, shard_ref, EncoderIndex::MIN);
        let end_key = (epoch, shard_ref, EncoderIndex::MAX);

        let count = inner.certified_commits.range(start_key..=end_key).count();

        if was_inserted && count == 1 {
            let current_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as u64;

            inner
                .first_commit_timestamp_ms
                .insert(shard_key, current_ms);
        }

        Ok(count)
    }
    fn get_certified_commit(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
        slot: EncoderIndex,
    ) -> ShardResult<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>> {
        let slot_key = (epoch, shard_ref, slot);
        let mut inner = self.inner.read();
        match inner.certified_commits.get(&slot_key) {
            Some(signed_commit) => return Ok(signed_commit.clone()),
            None => {
                return Err(ShardError::InvalidReveal("key does not exist".to_string()));
            }
        }
    }
    fn time_since_first_certified_commit(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
    ) -> Option<Duration> {
        let shard_key = (epoch, shard_ref);
        let inner = self.inner.read();
        let timestamp_ms = *inner.first_commit_timestamp_ms.get(&shard_key)?;

        let current_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_millis() as u64;

        Some(Duration::from_millis(current_ms - timestamp_ms))
    }
    fn atomic_reveal(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
        slot: EncoderIndex,
        key: EncryptionKey,
        checksum: Checksum,
    ) -> ShardResult<usize> {
        let slot_key = (epoch, shard_ref, slot);
        let shard_key = (epoch, shard_ref);
        let mut inner = self.inner.write();

        // Now insert the reveal if it doesn't exist or matches
        let was_inserted = match inner.reveals.get(&slot_key) {
            Some((existing_key, _)) => {
                if existing_key == &key {
                    false // No insertion needed, already exists
                } else {
                    return Err(ShardError::InvalidReveal(
                        "slot already has different reveal key".to_string(),
                    ));
                }
            }
            None => {
                inner.reveals.insert(slot_key, (key, checksum));
                true // New insertion happened
            }
        };

        // Count all reveals for this shard using range query
        let start_key = (epoch, shard_ref, EncoderIndex::MIN);
        let end_key = (epoch, shard_ref, EncoderIndex::MAX);
        let count = inner.reveals.range(start_key..=end_key).count();

        // If we inserted a reveal and the count is 1, this is the first reveal
        if was_inserted && count == 1 {
            let current_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as u64;

            inner
                .first_reveal_timestamp_ms
                .insert(shard_key, current_ms);
        }

        Ok(count)
    }
    fn time_since_first_reveal(&self, epoch: Epoch, shard_ref: Digest<Shard>) -> Option<Duration> {
        let shard_key = (epoch, shard_ref);
        let inner = self.inner.read();
        let timestamp_ms = *inner.first_reveal_timestamp_ms.get(&shard_key)?;

        let current_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_millis() as u64;

        Some(Duration::from_millis(current_ms - timestamp_ms))
    }
}
