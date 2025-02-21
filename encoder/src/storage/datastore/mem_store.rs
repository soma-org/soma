use fastcrypto::bls12381::min_sig;
use parking_lot::RwLock;
use shared::{
    crypto::AesKey,
    digest::Digest,
    metadata::{EncryptionAPI, Metadata, MetadataAPI},
    network_committee::NetworkingIndex,
    signed::Signed,
    verified::Verified,
};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    error::{ShardError, ShardResult},
    types::{
        certified::Certified,
        encoder_committee::{EncoderIndex, Epoch},
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_input::ShardInput,
        shard_reveal::ShardReveal,
        shard_verifier::ShardAuthToken,
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
    // epoch, shard reference, source encoder (slot)
    commits: BTreeMap<
        (Epoch, Digest<Shard>, EncoderIndex),
        (
            Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
            Metadata,
        ),
    >,
    committers: BTreeMap<
        (Epoch, Digest<Shard>, EncoderIndex),
        Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    >,
}

impl MemStore {
    // #[cfg(test)]
    pub(crate) const fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                commits: BTreeMap::new(),
                committers: BTreeMap::new(),
            }),
        }
    }
}

impl Store for MemStore {
    fn atomic_commit(
        &self,
        shard_ref: Digest<Shard>,
        signed_commit: Signed<ShardCommit, min_sig::BLS12381Signature>,
    ) -> ShardResult<()> {
        {
            let epoch = signed_commit.auth_token().epoch();
            let slot = signed_commit.slot();
            let committer = signed_commit.committer();
            let slot_key = (epoch, shard_ref, slot);
            let committer_key = (epoch, shard_ref, committer);
            let digest = Digest::new(&signed_commit).map_err(ShardError::DigestFailure)?;
            let metadata = signed_commit.commit().clone();
            let new_value = (digest.clone(), metadata);
            let mut inner = self.inner.write();

            // Check both conditions first
            let committer_check = inner.committers.get(&committer_key);
            let slot_check = inner.commits.get(&slot_key);

            match (committer_check, slot_check) {
                // Committer has committed before
                (Some(existing_digest), _) if existing_digest != &digest => {
                    return Err(ShardError::ConflictingCommit(
                        "existing commit from committer".to_string(),
                    ));
                }
                // Slot is taken with different commit
                (_, Some(existing)) if existing != &new_value => {
                    return Err(ShardError::ConflictingCommit(
                        "slot already has commit".to_string(),
                    ));
                }
                // If we made it here, either there are no existing commits
                // or the existing commits match exactly
                (None, None) => {
                    // Insert new commit
                    inner.commits.insert(slot_key, new_value);
                    inner.committers.insert(committer_key, digest);
                }
                // Everything matches, idempotent case
                _ => {}
            }
            Ok(())
        }
    }
    fn check_reveal(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
        slot: EncoderIndex,
        key_digest: Digest<AesKey>,
    ) -> ShardResult<()> {
        let slot_key = (epoch, shard_ref, slot);
        let mut inner = self.inner.read();
        match inner.commits.get(&slot_key) {
            Some((_, metadata)) => match metadata.encryption() {
                Some(encryption) => {
                    if encryption.key_digest() != key_digest {
                        return Err(ShardError::InvalidReveal("sss".to_string()));
                    }
                    return Ok(());
                }
                None => {
                    return Err(ShardError::InvalidReveal("sss".to_string()));
                }
            },
            None => {
                return Err(ShardError::InvalidReveal("sss".to_string()));
            }
        }
    }
}
