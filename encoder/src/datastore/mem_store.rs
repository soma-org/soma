use fastcrypto::bls12381::min_sig;
use parking_lot::RwLock;
use shared::{
    checksum::Checksum,
    crypto::{keys::EncoderPublicKey, EncryptionKey},
    digest::Digest,
    signed::Signed,
    verified::Verified,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
};

use crate::{
    error::{ShardError, ShardResult},
    types::{
        encoder_committee::{EncoderIndex, Epoch},
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_reveal::{ShardReveal, ShardRevealAPI},
        shard_scores::{ScoreSet, ShardScores}, // shard_votes::{CommitRound, RevealRound, ShardVotes, ShardVotesAPI},
    },
};

use super::Store;

/// In-memory storage for testing.
#[allow(unused)]
pub(crate) struct MemStore {
    inner: RwLock<Inner>,
}

type Encoder = EncoderPublicKey;
type Committer = EncoderPublicKey;

#[allow(unused)]
struct Inner {
    #[allow(clippy::type_complexity)]
    signed_commit_digests: BTreeMap<
        (Epoch, Digest<Shard>, Encoder),
        Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    >,
    #[allow(clippy::type_complexity)]
    shard_committers: BTreeMap<
        (Epoch, Digest<Shard>, Committer),
        Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    >,
    #[allow(clippy::type_complexity)]
    signed_commits:
        BTreeMap<(Epoch, Digest<Shard>, Encoder), Signed<ShardCommit, min_sig::BLS12381Signature>>,

    // EPOCH, SHARD_REF, SLOT
    reveals: BTreeMap<(Epoch, Digest<Shard>, EncoderIndex), (EncryptionKey, Checksum)>,
    // EPOCH, SHARD_REF
    first_commit_timestamp_ms: BTreeMap<(Epoch, Digest<Shard>), u64>,
    // EPOCH, SHARD_REF
    first_reveal_timestamp_ms: BTreeMap<(Epoch, Digest<Shard>), u64>,

    // EPOCH, SHARD_REF, SLOT -> EVAL SET VOTER
    commit_slot_accept_voters:
        BTreeMap<(Epoch, Digest<Shard>, EncoderIndex), BTreeSet<EncoderIndex>>,
    // EPOCH, SHARD_REF, SLOT -> EVAL SET VOTER
    commit_slot_reject_voters:
        BTreeMap<(Epoch, Digest<Shard>, EncoderIndex), BTreeSet<EncoderIndex>>,
    // EPOCH, SHARD_REF, SLOT -> FINALITY STATUS
    commit_slot_finality: BTreeMap<(Epoch, Digest<Shard>, EncoderIndex), SlotFinality>,

    // EPOCH, SHARD_REF, SLOT -> EVAL SET VOTER
    reveal_slot_accept_voters:
        BTreeMap<(Epoch, Digest<Shard>, EncoderIndex), BTreeSet<EncoderIndex>>,
    // EPOCH, SHARD_REF, SLOT -> EVAL SET VOTER
    reveal_slot_reject_voters:
        BTreeMap<(Epoch, Digest<Shard>, EncoderIndex), BTreeSet<EncoderIndex>>,
    reveal_slot_finality: BTreeMap<(Epoch, Digest<Shard>, EncoderIndex), SlotFinality>,

    #[allow(clippy::type_complexity)]
    scores: BTreeMap<
        (Epoch, Digest<Shard>, EncoderIndex),
        (
            Digest<ScoreSet>,
            Signed<ScoreSet, min_sig::BLS12381Signature>,
        ),
    >,
}

pub(crate) enum SlotFinality {
    Accepted,
    Rejected,
}
impl MemStore {
    pub(crate) const fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                signed_commit_digests: BTreeMap::new(),
                shard_committers: BTreeMap::new(),
                signed_commits: BTreeMap::new(),
                reveals: BTreeMap::new(),
                first_commit_timestamp_ms: BTreeMap::new(),
                first_reveal_timestamp_ms: BTreeMap::new(),
                commit_slot_accept_voters: BTreeMap::new(),
                commit_slot_reject_voters: BTreeMap::new(),
                commit_slot_finality: BTreeMap::new(),
                reveal_slot_accept_voters: BTreeMap::new(),
                reveal_slot_reject_voters: BTreeMap::new(),
                reveal_slot_finality: BTreeMap::new(),
                scores: BTreeMap::new(),
            }),
        }
    }
}

impl Store for MemStore {
    fn lock_signed_commit(
        &self,
        shard: &Shard,
        signed_commit: &Signed<ShardCommit, min_sig::BLS12381Signature>,
    ) -> ShardResult<()> {
        let epoch = signed_commit.auth_token().epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_commit.encoder();
        let committer = signed_commit.committer();

        let signed_commit_digest = Digest::new(signed_commit).map_err(ShardError::DigestFailure)?;

        let signed_commit_digests_key = (epoch, shard_digest, encoder.clone());
        let shard_committer_key = (epoch, shard_digest, committer.clone());

        let mut inner = self.inner.write();

        // Check both conditions first
        let slot_check = inner.signed_commit_digests.get(&signed_commit_digests_key);
        let committer_check = inner.shard_committers.get(&shard_committer_key);

        match (committer_check, slot_check) {
            // Committer has committed before
            (Some(existing_digest), _) if existing_digest != &signed_commit_digest => {
                return Err(ShardError::ConflictingCommit(
                    "existing commit from committer".to_string(),
                ));
            }
            // Slot is taken with different commit
            (_, Some(existing)) if existing != &signed_commit_digest => {
                return Err(ShardError::ConflictingCommit(
                    "slot already has commit".to_string(),
                ));
            }
            // If we made it here, either there are no existing commits
            // or the existing commits match exactly
            (None, None) => {
                // Insert new commit
                inner
                    .signed_commit_digests
                    .insert(signed_commit_digests_key, signed_commit_digest);
                inner
                    .shard_committers
                    .insert(shard_committer_key, signed_commit_digest);
            }
            // Everything matches, idempotent case
            _ => (),
        }
        Ok(())
    }
    fn add_signed_commit(
        &self,
        shard: &Shard,
        signed_commit: &Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    ) -> ShardResult<usize> {
        let epoch = signed_commit.auth_token().epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_commit.encoder();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let signed_commit = signed_commit.deref();

        let mut inner = self.inner.write();

        match inner.signed_commits.get(&encoder_key) {
            Some(existing_commit) => {
                if existing_commit != &signed_commit {
                    return Err(ShardError::ConflictingCommit(
                        "encoder has a different existing commit".to_string(),
                    ));
                }
            }
            None => {
                inner.signed_commits.insert(encoder_key, signed_commit);
            }
        };
        Ok(())

        // let start_key = (epoch, shard_digest, EncoderIndex::MIN);
        // let end_key = (epoch, shard_digest, EncoderIndex::MAX);

        // let count = inner.certified_commits.range(start_key..=end_key).count();

        // if was_inserted && count == 1 {
        //     let current_ms = SystemTime::now()
        //         .duration_since(UNIX_EPOCH)
        //         .expect("Time went backwards")
        //         .as_millis() as u64;

        //     inner
        //         .first_commit_timestamp_ms
        //         .insert(shard_key, current_ms);
        // }

        // Ok(count)
    }
    fn check_reveal_key(
        &self,
        shard: &Shard,
        signed_reveal: &Signed<ShardReveal, min_sig::BLS12381Signature>,
    ) -> ShardResult<()> {
        let epoch = signed_reveal.auth_token().epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_reveal.encoder();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let inner = self.inner.read();
        match inner.signed_commits.get(&encoder_key) {
            Some(signed_commits) => {
                let key_digest =
                    Digest::new(signed_reveal.key()).map_err(ShardError::DigestFailure)?;
                if signed_commits.reveal_key_digest() != key_digest {
                    Err(ShardError::InvalidReveal(
                        "encryption key digest did not match commmit".to_string(),
                    ))
                }
                Ok(())
            }
            None => Err(ShardError::NotFound(
                "encryption key for commit".to_string(),
            )),
        }
    }
    fn add_signed_reveal(
        &self,
        shard: &Shard,
        signed_reveal: &Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let epoch = signed_reveal.auth_token().epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_reveal.encoder();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let signed_reveal = signed_reveal.deref();

        let mut inner = self.inner.write();

        match inner.signed_commits.get(&encoder_key) {
            Some(existing_reveal) => {
                if existing_reveal != &signed_reveal {
                    return Err(ShardError::ConflictingCommit(
                        "encoder has a different existing signed reveal".to_string(),
                    ));
                }
            }
            None => {
                inner.signed_reveals.insert(encoder_key, signed_reveal);
            }
        };
        Ok(())
    }

    // fn get_filled_certified_commit_slots(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    // ) -> Vec<EncoderIndex> {
    //     let start_key = (epoch, shard_ref, EncoderIndex::MIN);
    //     let end_key = (epoch, shard_ref, EncoderIndex::MAX);
    //     let inner = self.inner.read();

    //     // Use range query to get all keys in the range and extract the EncoderIndex (slot)
    //     inner
    //         .certified_commits
    //         .range(start_key..=end_key)
    //         .map(|((_, _, slot), _)| *slot) // Extract the slot from the key tuple
    //         .collect::<Vec<EncoderIndex>>()
    // }
    // fn get_certified_commit(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     slot: EncoderIndex,
    // ) -> ShardResult<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>> {
    //     let slot_key = (epoch, shard_ref, slot);
    //     let inner = self.inner.read();
    //     match inner.certified_commits.get(&slot_key) {
    //         Some(signed_commit) => Ok(signed_commit.clone()),
    //         None => Err(ShardError::NotFound("key does not exist".to_string())),
    //     }
    // }
    // fn time_since_first_certified_commit(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    // ) -> Option<Duration> {
    //     let shard_key = (epoch, shard_ref);
    //     let inner = self.inner.read();
    //     let timestamp_ms = *inner.first_commit_timestamp_ms.get(&shard_key)?;

    //     let current_ms = SystemTime::now()
    //         .duration_since(UNIX_EPOCH)
    //         .ok()?
    //         .as_millis() as u64;

    //     Some(Duration::from_millis(current_ms - timestamp_ms))
    // }
    // fn atomic_reveal(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     slot: EncoderIndex,
    //     key: EncryptionKey,
    //     checksum: Checksum,
    // ) -> ShardResult<usize> {
    //     let slot_key = (epoch, shard_ref, slot);
    //     let shard_key = (epoch, shard_ref);
    //     let mut inner = self.inner.write();

    //     // Now insert the reveal if it doesn't exist or matches
    //     let was_inserted = match inner.reveals.get(&slot_key) {
    //         Some((existing_key, _)) => {
    //             if existing_key == &key {
    //                 false // No insertion needed, already exists
    //             } else {
    //                 return Err(ShardError::InvalidReveal(
    //                     "slot already has different reveal key".to_string(),
    //                 ));
    //             }
    //         }
    //         None => {
    //             inner.reveals.insert(slot_key, (key, checksum));
    //             true // New insertion happened
    //         }
    //     };

    //     // Count all reveals for this shard using range query
    //     let start_key = (epoch, shard_ref, EncoderIndex::MIN);
    //     let end_key = (epoch, shard_ref, EncoderIndex::MAX);
    //     let count = inner.reveals.range(start_key..=end_key).count();

    //     // If we inserted a reveal and the count is 1, this is the first reveal
    //     if was_inserted && count == 1 {
    //         let current_ms = SystemTime::now()
    //             .duration_since(UNIX_EPOCH)
    //             .expect("Time went backwards")
    //             .as_millis() as u64;

    //         inner
    //             .first_reveal_timestamp_ms
    //             .insert(shard_key, current_ms);
    //     }

    //     Ok(count)
    // }
    // fn get_reveal(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     slot: EncoderIndex,
    // ) -> ShardResult<(EncryptionKey, Checksum)> {
    //     let slot_key = (epoch, shard_ref, slot);
    //     let inner = self.inner.read();
    //     match inner.reveals.get(&slot_key) {
    //         Some(reveal) => Ok(reveal.clone()),
    //         None => Err(ShardError::InvalidReveal("key does not exist".to_string())),
    //     }
    // }
    // fn get_filled_reveal_slots(&self, epoch: Epoch, shard_ref: Digest<Shard>) -> Vec<EncoderIndex> {
    //     let start_key = (epoch, shard_ref, EncoderIndex::MIN);
    //     let end_key = (epoch, shard_ref, EncoderIndex::MAX);
    //     let inner = self.inner.read();

    //     inner
    //         .reveals
    //         .range(start_key..=end_key)
    //         .map(|((_, _, slot), _)| *slot) // Extract the slot from the key tuple
    //         .collect::<Vec<EncoderIndex>>()
    // }

    // fn time_since_first_reveal(&self, epoch: Epoch, shard_ref: Digest<Shard>) -> Option<Duration> {
    //     let shard_key = (epoch, shard_ref);
    //     let inner = self.inner.read();
    //     let timestamp_ms = *inner.first_reveal_timestamp_ms.get(&shard_key)?;

    //     let current_ms = SystemTime::now()
    //         .duration_since(UNIX_EPOCH)
    //         .ok()?
    //         .as_millis() as u64;

    //     Some(Duration::from_millis(current_ms - timestamp_ms))
    // }
    // fn add_commit_vote(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     shard: Shard,
    //     vote: ShardVotes<CommitRound>,
    // ) -> ShardResult<(usize, usize)> {
    //     let mut inner = self.inner.write();
    //     let voter = vote.voter();
    //     let rejects = vote.rejects();

    //     // Get the evaluation set and quorum threshold from the shard
    //     let evaluation_set = shard.evaluation_set();
    //     let evaluation_set_size = shard.evaluation_size();

    //     let quorum = shard.evaluation_quorum_threshold() as usize;
    //     let accept_threshold = quorum; // 2f + 1
    //     let reject_threshold = (evaluation_set_size - quorum) + 1; // (N - quorum) + 1

    //     let start_key = (epoch, shard_ref, EncoderIndex::MIN);
    //     let end_key = (epoch, shard_ref, EncoderIndex::MAX);

    //     // Process votes for each slot in the evaluation set
    //     for &slot in &evaluation_set {
    //         let slot_key = (epoch, shard_ref, slot);

    //         // Skip if the slot is already finalized, we'll count it later via range query
    //         if inner.commit_slot_finality.contains_key(&slot_key) {
    //             continue;
    //         }

    //         // Determine if the voter accepts or rejects this slot
    //         let is_reject = rejects.contains(&slot);
    //         if is_reject {
    //             inner
    //                 .commit_slot_reject_voters
    //                 .entry(slot_key)
    //                 .or_default()
    //                 .insert(voter);
    //         } else {
    //             inner
    //                 .commit_slot_accept_voters
    //                 .entry(slot_key)
    //                 .or_default()
    //                 .insert(voter);
    //         }

    //         // Count current votes for this slot
    //         let accept_count = inner
    //             .commit_slot_accept_voters
    //             .get(&slot_key)
    //             .map(|voters| voters.len())
    //             .unwrap_or(0);
    //         let reject_count = inner
    //             .commit_slot_reject_voters
    //             .get(&slot_key)
    //             .map(|voters| voters.len())
    //             .unwrap_or(0);

    //         // Check for finality
    //         if accept_count >= accept_threshold {
    //             inner
    //                 .commit_slot_finality
    //                 .insert(slot_key, SlotFinality::Accepted);
    //         } else if reject_count >= reject_threshold {
    //             inner
    //                 .commit_slot_finality
    //                 .insert(slot_key, SlotFinality::Rejected);
    //         }
    //     }

    //     // Use a single range query to count both finalized and accepted slots
    //     let (total_finalized_slots, total_accepted_slots) = inner
    //         .commit_slot_finality
    //         .range(start_key..=end_key)
    //         .fold((0, 0), |(finalized, accepted), (_, finality)| {
    //             let finalized = finalized + 1;
    //             let accepted = if matches!(finality, SlotFinality::Accepted) {
    //                 accepted + 1
    //             } else {
    //                 accepted
    //             };
    //             (finalized, accepted)
    //         });

    //     Ok((total_finalized_slots, total_accepted_slots))
    // }
    // fn add_reveal_vote(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     shard: Shard,
    //     vote: ShardVotes<RevealRound>,
    // ) -> ShardResult<(usize, usize)> {
    //     let mut inner = self.inner.write();
    //     let voter = vote.voter();
    //     let rejects = vote.rejects();

    //     // Get the evaluation set and quorum threshold from the shard
    //     let evaluation_set = shard.evaluation_set();
    //     let evaluation_set_size = shard.evaluation_size();

    //     let quorum = shard.evaluation_quorum_threshold() as usize;
    //     let accept_threshold = quorum; // 2f + 1
    //     let reject_threshold = (evaluation_set_size - quorum) + 1; // (N - quorum) + 1

    //     let start_key = (epoch, shard_ref, EncoderIndex::MIN);
    //     let end_key = (epoch, shard_ref, EncoderIndex::MAX);

    //     // Process votes for each slot in the evaluation set
    //     for &slot in &evaluation_set {
    //         let slot_key = (epoch, shard_ref, slot);

    //         // Skip if the slot is already finalized, we'll count it later via range query
    //         if inner.reveal_slot_finality.contains_key(&slot_key) {
    //             continue;
    //         }

    //         // Determine if the voter accepts or rejects this slot
    //         let is_reject = rejects.contains(&slot);
    //         if is_reject {
    //             inner
    //                 .reveal_slot_reject_voters
    //                 .entry(slot_key)
    //                 .or_default()
    //                 .insert(voter);
    //         } else {
    //             inner
    //                 .reveal_slot_accept_voters
    //                 .entry(slot_key)
    //                 .or_default()
    //                 .insert(voter);
    //         }

    //         // Count current votes for this slot
    //         let accept_count = inner
    //             .reveal_slot_accept_voters
    //             .get(&slot_key)
    //             .map(|voters| voters.len())
    //             .unwrap_or(0);
    //         let reject_count = inner
    //             .reveal_slot_reject_voters
    //             .get(&slot_key)
    //             .map(|voters| voters.len())
    //             .unwrap_or(0);

    //         // Check for finality
    //         if accept_count >= accept_threshold {
    //             inner
    //                 .reveal_slot_finality
    //                 .insert(slot_key, SlotFinality::Accepted);
    //         } else if reject_count >= reject_threshold {
    //             inner
    //                 .reveal_slot_finality
    //                 .insert(slot_key, SlotFinality::Rejected);
    //         }
    //     }

    //     // Use a single range query to count both finalized and accepted slots
    //     let (total_finalized_slots, total_accepted_slots) = inner
    //         .reveal_slot_finality
    //         .range(start_key..=end_key)
    //         .fold((0, 0), |(finalized, accepted), (_, finality)| {
    //             let finalized = finalized + 1;
    //             let accepted = if matches!(finality, SlotFinality::Accepted) {
    //                 accepted + 1
    //             } else {
    //                 accepted
    //             };
    //             (finalized, accepted)
    //         });

    //     Ok((total_finalized_slots, total_accepted_slots))
    // }

    // fn get_accepted_finalized_reveal_slots(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    // ) -> ShardResult<Vec<EncoderIndex>> {
    //     let start_key = (epoch, shard_ref, EncoderIndex::MIN);
    //     let end_key = (epoch, shard_ref, EncoderIndex::MAX);
    //     let inner = self.inner.read();

    //     // Collect slots where reveal votes are finalized as Accepted
    //     let accepted_slots = inner
    //         .reveal_slot_finality
    //         .range(start_key..=end_key)
    //         .filter_map(|((_, _, slot), finality)| {
    //             if matches!(finality, SlotFinality::Accepted) {
    //                 Some(*slot)
    //             } else {
    //                 None
    //             }
    //         })
    //         .collect::<Vec<EncoderIndex>>();

    //     Ok(accepted_slots)
    // }

    // fn add_scores(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     evaluator: EncoderIndex,
    //     signed_scores: Signed<ScoreSet, min_sig::BLS12381Signature>,
    // ) -> ShardResult<Vec<(EncoderIndex, Signed<ScoreSet, min_sig::BLS12381Signature>)>> {
    //     let scores_digest =
    //         Digest::new(&signed_scores.clone().into_inner()).map_err(ShardError::DigestFailure)?;
    //     let scores_vote_key = (epoch, shard_ref, evaluator);
    //     let mut inner = self.inner.write();
    //     match inner.scores.get(&scores_vote_key) {
    //         Some((existing_digest, _)) => {
    //             if existing_digest != &scores_digest {
    //                 return Err(ShardError::ConflictingCommit(
    //                     "evaluator already has different scores".to_string(),
    //                 ));
    //             }
    //         }
    //         None => {
    //             inner
    //                 .scores
    //                 .insert(scores_vote_key, (scores_digest, signed_scores));
    //         }
    //     }

    //     let start_key = (epoch, shard_ref, EncoderIndex::MIN);
    //     let end_key = (epoch, shard_ref, EncoderIndex::MAX);

    //     Ok(inner
    //         .scores
    //         .range(start_key..=end_key)
    //         .filter_map(|((_, _, eval_idx), (digest, signed))| {
    //             if digest == &scores_digest {
    //                 Some((*eval_idx, signed.clone()))
    //             } else {
    //                 None
    //             }
    //         })
    //         .collect())
    // }
}
