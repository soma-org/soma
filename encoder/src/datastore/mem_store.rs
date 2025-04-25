use fastcrypto::bls12381::min_sig;
use parking_lot::RwLock;
use shared::{crypto::keys::EncoderPublicKey, digest::Digest, signed::Signed, verified::Verified};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    time::Instant,
};

use crate::{
    error::{ShardError, ShardResult},
    types::{
        encoder_committee::Epoch,
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_commit_votes::{ShardCommitVotes, ShardCommitVotesAPI},
        shard_reveal::{ShardReveal, ShardRevealAPI},
        shard_reveal_votes::{ShardRevealVotes, ShardRevealVotesAPI},
        shard_scores::{ShardScores, ShardScoresAPI},
    },
};

use super::{CommitVoteCounts, RevealVoteCounts, Store};

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

    signed_reveals:
        BTreeMap<(Epoch, Digest<Shard>, Encoder), Signed<ShardReveal, min_sig::BLS12381Signature>>,

    first_commit_time: BTreeMap<(Epoch, Digest<Shard>), Instant>,

    //      ////////////////////
    signed_commit_votes: BTreeMap<
        (Epoch, Digest<Shard>, Encoder),
        Signed<ShardCommitVotes, min_sig::BLS12381Signature>,
    >,
    #[allow(clippy::type_complexity)]
    commit_accept_digest_voters: BTreeMap<
        (
            Epoch,
            Digest<Shard>,
            EncoderPublicKey,
            Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        ),
        BTreeSet<EncoderPublicKey>,
    >,
    commit_reject_voters:
        BTreeMap<(Epoch, Digest<Shard>, EncoderPublicKey), BTreeSet<EncoderPublicKey>>,

    commit_votes_highest_digest: BTreeMap<(Epoch, Digest<Shard>, EncoderPublicKey), usize>,

    //      ////////////////////
    signed_reveal_votes: BTreeMap<
        (Epoch, Digest<Shard>, Encoder),
        Signed<ShardRevealVotes, min_sig::BLS12381Signature>,
    >,

    reveal_accept_voters:
        BTreeMap<(Epoch, Digest<Shard>, EncoderPublicKey), BTreeSet<EncoderPublicKey>>,

    reveal_reject_voters:
        BTreeMap<(Epoch, Digest<Shard>, EncoderPublicKey), BTreeSet<EncoderPublicKey>>,

    first_reveal_time: BTreeMap<(Epoch, Digest<Shard>), Instant>,

    signed_scores:
        BTreeMap<(Epoch, Digest<Shard>, Encoder), Signed<ShardScores, min_sig::BLS12381Signature>>,
}

impl MemStore {
    pub(crate) const fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                signed_commit_digests: BTreeMap::new(),
                shard_committers: BTreeMap::new(),
                signed_commits: BTreeMap::new(),
                signed_reveals: BTreeMap::new(),
                first_commit_time: BTreeMap::new(),
                signed_commit_votes: BTreeMap::new(),
                commit_accept_digest_voters: BTreeMap::new(),
                commit_reject_voters: BTreeMap::new(),
                commit_votes_highest_digest: BTreeMap::new(),
                signed_reveal_votes: BTreeMap::new(),
                reveal_accept_voters: BTreeMap::new(),
                reveal_reject_voters: BTreeMap::new(),
                signed_scores: BTreeMap::new(),
                first_reveal_time: BTreeMap::new(),
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
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_commit.encoder();
        let committer = signed_commit.committer();

        let signed_commit_digest = Digest::new(signed_commit).map_err(ShardError::DigestFailure)?;

        let signed_commit_digests_key = (epoch, shard_digest, encoder.clone());
        let shard_committer_key = (epoch, shard_digest, committer.clone());

        match (
            self.inner
                .read()
                .signed_commit_digests
                .get(&signed_commit_digests_key),
            self.inner.read().shard_committers.get(&shard_committer_key),
        ) {
            // Committer has committed before
            (Some(existing_digest), _) if existing_digest != &signed_commit_digest => {
                return Err(ShardError::Conflict(
                    "existing commit from committer".to_string(),
                ));
            }
            // Slot is taken with different commit
            (_, Some(existing)) if existing != &signed_commit_digest => {
                return Err(ShardError::Conflict(
                    "encoder already has commit".to_string(),
                ));
            }
            // If we made it here, either there are no existing commits
            // or the existing commits match exactly
            (None, None) => {
                // Insert new commit
                self.inner
                    .write()
                    .signed_commit_digests
                    .insert(signed_commit_digests_key, signed_commit_digest);
                self.inner
                    .write()
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
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_commit.encoder();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let signed_commit = signed_commit.deref();

        match self.inner.read().signed_commits.get(&encoder_key) {
            Some(existing_commit) => {
                if existing_commit != signed_commit {
                    return Err(ShardError::Conflict(
                        "encoder has a different existing commit".to_string(),
                    ));
                }
            }
            None => {
                self.inner
                    .write()
                    .signed_commits
                    .insert(encoder_key, signed_commit.clone());
            }
        };
        Ok(())
    }
    fn count_signed_commits(&self, shard: &Shard) -> ShardResult<usize> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let start_key = (epoch, shard_digest, EncoderPublicKey::MIN());
        let end_key = (epoch, shard_digest, EncoderPublicKey::MAX());

        let count = self
            .inner
            .read()
            .signed_commits
            .range(start_key..=end_key)
            .count();
        Ok(count)
    }
    fn get_signed_commits(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<ShardCommit, min_sig::BLS12381Signature>>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let start_key = (epoch, shard_digest, EncoderPublicKey::MIN());
        let end_key = (epoch, shard_digest, EncoderPublicKey::MAX());

        let commits: Vec<_> = self
            .inner
            .read()
            .signed_commits
            .range(start_key..=end_key)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(commits)
    }

    fn add_first_commit_time(&self, shard: &Shard) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let shard_key = (epoch, shard_digest);

        self.inner
            .write()
            .first_commit_time
            .insert(shard_key, Instant::now());
        Ok(())
    }
    fn get_first_commit_time(&self, shard: &Shard) -> ShardResult<Instant> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let shard_key = (epoch, shard_digest);

        let inner = self.inner.read();
        if let Some(instant) = inner.first_commit_time.get(&shard_key) {
            Ok(*instant)
        } else {
            Err(ShardError::NotFound("first commit time".to_string()))
        }
    }

    fn check_reveal_key(
        &self,
        shard: &Shard,
        signed_reveal: &Signed<ShardReveal, min_sig::BLS12381Signature>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_reveal.encoder();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let inner = self.inner.read();
        match inner.signed_commits.get(&encoder_key) {
            Some(signed_commits) => {
                let key_digest =
                    Digest::new(signed_reveal.key()).map_err(ShardError::DigestFailure)?;
                if signed_commits.reveal_key_digest()? != key_digest {
                    return Err(ShardError::Conflict(
                        "encryption key digest did not match commmit".to_string(),
                    ));
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
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_reveal.encoder();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let signed_reveal = signed_reveal.deref();

        match self.inner.read().signed_reveals.get(&encoder_key) {
            Some(existing_reveal) => {
                if existing_reveal != signed_reveal {
                    return Err(ShardError::Conflict(
                        "encoder has a different existing signed reveal".to_string(),
                    ));
                }
            }
            None => {
                self.inner
                    .write()
                    .signed_reveals
                    .insert(encoder_key, signed_reveal.clone());
            }
        };
        Ok(())
    }
    fn count_signed_reveal(&self, shard: &Shard) -> ShardResult<usize> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let start_key = (epoch, shard_digest, EncoderPublicKey::MIN());
        let end_key = (epoch, shard_digest, EncoderPublicKey::MAX());

        let count = self
            .inner
            .read()
            .signed_reveals
            .range(start_key..=end_key)
            .count();
        Ok(count)
    }
    fn get_signed_reveals(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<ShardReveal, min_sig::BLS12381Signature>>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let start_key = (epoch, shard_digest, EncoderPublicKey::MIN());
        let end_key = (epoch, shard_digest, EncoderPublicKey::MAX());

        let reveals: Vec<_> = self
            .inner
            .read()
            .signed_reveals
            .range(start_key..=end_key)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(reveals)
    }
    fn add_first_reveal_time(&self, shard: &Shard) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let shard_key = (epoch, shard_digest);

        self.inner
            .write()
            .first_reveal_time
            .insert(shard_key, Instant::now());
        Ok(())
    }
    fn get_first_reveal_time(&self, shard: &Shard) -> ShardResult<Instant> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let shard_key = (epoch, shard_digest);

        let inner = self.inner.read();
        if let Some(instant) = inner.first_reveal_time.get(&shard_key) {
            Ok(instant.to_owned())
        } else {
            Err(ShardError::NotFound("first reveal time".to_string()))
        }
    }
    fn add_commit_votes(
        &self,
        shard: &Shard,
        votes: &Verified<Signed<ShardCommitVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = votes.voter();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let votes = &**votes;

        match self.inner.read().signed_commit_votes.get(&encoder_key) {
            Some(existing) => {
                if existing != votes {
                    return Err(ShardError::Conflict(
                        "encoder has a different commit vote".to_string(),
                    ));
                }
            }
            None => {
                self.inner
                    .write()
                    .signed_commit_votes
                    .insert(encoder_key, votes.clone());
            }
        };
        for receiving_encoder in shard.encoders() {
            let receiving_encoder_key = (epoch, shard_digest, receiving_encoder.clone());
            match votes
                .accepts()
                .iter()
                .find(|(key, _)| key == &receiving_encoder)
                .map(|(_, digest)| digest)
            {
                Some(digest) => {
                    let receiving_encoder_digest_key = (
                        epoch,
                        shard_digest,
                        receiving_encoder.clone(),
                        digest.to_owned(),
                    );
                    self.inner
                        .write()
                        .commit_accept_digest_voters
                        .entry(receiving_encoder_digest_key.clone())
                        .or_default()
                        .insert(votes.voter().clone());

                    let this_digests_votes = self
                        .inner
                        .read()
                        .commit_accept_digest_voters
                        .get(&receiving_encoder_digest_key)
                        .map_or(0_usize, BTreeSet::len);

                    let current_highest_digest_votes = self
                        .inner
                        .read()
                        .commit_votes_highest_digest
                        .get(&receiving_encoder_key)
                        .copied()
                        .unwrap_or(0_usize);

                    if this_digests_votes > current_highest_digest_votes {
                        self.inner
                            .write()
                            .commit_votes_highest_digest
                            .insert(receiving_encoder_key, this_digests_votes);
                    }
                }
                None => {
                    self.inner
                        .write()
                        .commit_reject_voters
                        .entry(receiving_encoder_key)
                        .or_default()
                        .insert(votes.voter().clone());
                }
            }
        }
        Ok(())
    }
    fn count_commit_votes(&self, shard: &Shard) -> ShardResult<usize> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let start_key = (epoch, shard_digest, EncoderPublicKey::MIN());
        let end_key = (epoch, shard_digest, EncoderPublicKey::MAX());

        let count = self
            .inner
            .read()
            .signed_commit_votes
            .range(start_key..=end_key)
            .count();
        Ok(count)
    }
    fn get_commit_votes_for_encoder(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
        digest: Option<&Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>>,
    ) -> ShardResult<CommitVoteCounts> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let accepts: Option<usize>;
        if let Some(digest) = digest {
            let encoder_digest_key = (epoch, shard_digest, encoder.clone(), digest.to_owned());
            accepts = Some(
                self.inner
                    .read()
                    .commit_accept_digest_voters
                    .get(&encoder_digest_key)
                    .map(|voters| voters.len())
                    .unwrap_or(0),
            );
        } else {
            accepts = None;
        }

        let rejects = self
            .inner
            .read()
            .commit_reject_voters
            .get(&encoder_key)
            .map(|voters| voters.len())
            .unwrap_or(0);

        let highest = self
            .inner
            .read()
            .commit_votes_highest_digest
            .get(&encoder_key)
            .copied()
            .unwrap_or(0_usize);

        Ok(CommitVoteCounts::new(accepts, rejects, highest))
    }
    fn get_commit_votes(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<ShardCommitVotes, min_sig::BLS12381Signature>>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let start_key = (epoch, shard_digest, EncoderPublicKey::MIN());
        let end_key = (epoch, shard_digest, EncoderPublicKey::MAX());

        let commit_votes = self
            .inner
            .read()
            .signed_commit_votes
            .range(start_key..=end_key)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(commit_votes)
    }

    fn add_reveal_votes(
        &self,
        shard: &Shard,
        votes: &Verified<Signed<ShardRevealVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let voter_key = (epoch, shard_digest, votes.voter().clone());
        let votes = &**votes;
        match self.inner.read().signed_reveal_votes.get(&voter_key) {
            Some(existing) => {
                if existing != votes {
                    return Err(ShardError::Conflict(
                        "encoder has a different reveal vote".to_string(),
                    ));
                }
            }
            None => {
                self.inner
                    .write()
                    .signed_reveal_votes
                    .insert(voter_key, votes.clone());
            }
        };

        for receiving_encoder in shard.encoders() {
            let receiving_encoder_key = (epoch, shard_digest, receiving_encoder.clone());
            if votes.accepts().contains(&receiving_encoder) {
                // explicit accept vote
                self.inner
                    .write()
                    .reveal_accept_voters
                    .entry(receiving_encoder_key)
                    .or_default()
                    .insert(votes.voter().clone());
            } else {
                // implicit reject vote
                self.inner
                    .write()
                    .reveal_reject_voters
                    .entry(receiving_encoder_key)
                    .or_default()
                    .insert(votes.voter().clone());
            }
        }
        Ok(())
    }
    fn get_reveal_votes(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<ShardRevealVotes, min_sig::BLS12381Signature>>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let start_key = (epoch, shard_digest, EncoderPublicKey::MIN());
        let end_key = (epoch, shard_digest, EncoderPublicKey::MAX());

        let reveal_votes = self
            .inner
            .read()
            .signed_reveal_votes
            .range(start_key..=end_key)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(reveal_votes)
    }

    fn get_reveal_votes_for_encoder(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
    ) -> ShardResult<RevealVoteCounts> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder_key = (epoch, shard_digest, encoder.clone());

        let accepts = self
            .inner
            .read()
            .reveal_accept_voters
            .get(&encoder_key)
            .map(|voters| voters.len())
            .unwrap_or(0);

        let rejects = self
            .inner
            .read()
            .reveal_reject_voters
            .get(&encoder_key)
            .map(|voters| voters.len())
            .unwrap_or(0);

        Ok(RevealVoteCounts::new(accepts, rejects))
    }

    fn add_signed_scores(
        &self,
        shard: &Shard,
        scores: &Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = scores.evaluator();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let scores = scores.deref();
        match self.inner.read().signed_scores.get(&encoder_key) {
            Some(existing) => {
                if existing != scores {
                    return Err(ShardError::Conflict(
                        "encoder has a different reveal vote".to_string(),
                    ));
                }
            }
            None => {
                self.inner
                    .write()
                    .signed_scores
                    .insert(encoder_key, scores.clone());
            }
        };
        Ok(())
    }
    fn get_signed_scores(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<ShardScores, min_sig::BLS12381Signature>>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let start_key = (epoch, shard_digest, EncoderPublicKey::MIN());
        let end_key = (epoch, shard_digest, EncoderPublicKey::MAX());

        let scores = self
            .inner
            .read()
            .signed_scores
            .range(start_key..=end_key)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(scores)
    }
}
