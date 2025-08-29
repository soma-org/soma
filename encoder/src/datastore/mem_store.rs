use fastcrypto::bls12381::min_sig;
use parking_lot::RwLock;
use shared::{
    crypto::keys::{EncoderAggregateSignature, EncoderPublicKey},
    digest::Digest,
    encoder_committee::Epoch,
    error::{ShardError, ShardResult},
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    time::Instant,
};
use tracing::{info, warn};
use types::shard_score::{ShardScore, ShardScoreAPI};

use crate::types::{
    commit::{Commit, CommitAPI},
    commit_votes::{CommitVotes, CommitVotesAPI},
    reveal::{Reveal, RevealAPI},
};

use super::{CommitVoteCounts, Store};

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
        Digest<Signed<Commit, min_sig::BLS12381Signature>>,
    >,
    #[allow(clippy::type_complexity)]
    shard_committers: BTreeMap<
        (Epoch, Digest<Shard>, Committer),
        Digest<Signed<Commit, min_sig::BLS12381Signature>>,
    >,
    #[allow(clippy::type_complexity)]
    signed_commits:
        BTreeMap<(Epoch, Digest<Shard>, Encoder), Signed<Commit, min_sig::BLS12381Signature>>,

    signed_reveals:
        BTreeMap<(Epoch, Digest<Shard>, Encoder), Signed<Reveal, min_sig::BLS12381Signature>>,

    first_commit_time: BTreeMap<(Epoch, Digest<Shard>), Instant>,

    //      ////////////////////
    signed_commit_votes:
        BTreeMap<(Epoch, Digest<Shard>, Encoder), Signed<CommitVotes, min_sig::BLS12381Signature>>,
    #[allow(clippy::type_complexity)]
    commit_accept_digest_voters: BTreeMap<
        (
            Epoch,
            Digest<Shard>,
            EncoderPublicKey,
            Digest<Signed<Reveal, min_sig::BLS12381Signature>>,
        ),
        BTreeSet<EncoderPublicKey>,
    >,
    commit_reject_voters:
        BTreeMap<(Epoch, Digest<Shard>, EncoderPublicKey), BTreeSet<EncoderPublicKey>>,

    commit_votes_highest_digest: BTreeMap<(Epoch, Digest<Shard>, EncoderPublicKey), usize>,

    reveal_accept_voters:
        BTreeMap<(Epoch, Digest<Shard>, EncoderPublicKey), BTreeSet<EncoderPublicKey>>,

    reveal_reject_voters:
        BTreeMap<(Epoch, Digest<Shard>, EncoderPublicKey), BTreeSet<EncoderPublicKey>>,

    first_reveal_time: BTreeMap<(Epoch, Digest<Shard>), Instant>,

    signed_scores:
        BTreeMap<(Epoch, Digest<Shard>, Encoder), Signed<ShardScore, min_sig::BLS12381Signature>>,

    agg_scores:
        BTreeMap<(Epoch, Digest<Shard>), (EncoderAggregateSignature, Vec<EncoderPublicKey>)>,
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
                reveal_accept_voters: BTreeMap::new(),
                reveal_reject_voters: BTreeMap::new(),
                signed_scores: BTreeMap::new(),
                first_reveal_time: BTreeMap::new(),
                agg_scores: BTreeMap::new(),
            }),
        }
    }
}

impl Store for MemStore {
    fn lock_signed_commit(
        &self,
        shard: &Shard,
        signed_commit: &Signed<Commit, min_sig::BLS12381Signature>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_commit.author();
        let committer = signed_commit.author();

        let signed_commit_digest = Digest::new(signed_commit).map_err(ShardError::DigestFailure)?;

        let signed_commit_digests_key = (epoch, shard_digest, encoder.clone());
        let shard_committer_key = (epoch, shard_digest, committer.clone());

        let mut inner = self.inner.write();

        match (
            inner.signed_commit_digests.get(&signed_commit_digests_key),
            inner.shard_committers.get(&shard_committer_key),
        ) {
            // Committer has committed before
            (Some(existing_digest), _) if existing_digest != &signed_commit_digest => {
                warn!("Committer has committed before");
                return Err(ShardError::Conflict(
                    "existing commit from committer".to_string(),
                ));
            }
            // Slot is taken with different commit
            (_, Some(existing)) if existing != &signed_commit_digest => {
                warn!("Slot is taken with different commit");
                return Err(ShardError::Conflict(
                    "encoder already has commit".to_string(),
                ));
            }
            // If we made it here, either there are no existing commits
            // or the existing commits match exactly
            (None, None) => {
                info!("No existing commits, insert into database");
                // Insert new commit
                inner
                    .signed_commit_digests
                    .insert(signed_commit_digests_key, signed_commit_digest);
                info!("Inserted into signed commit digests");
                inner
                    .shard_committers
                    .insert(shard_committer_key, signed_commit_digest);
                info!("Inserted into shard committers");
            }
            // Everything matches, idempotent case
            _ => (),
        }
        Ok(())
    }
    fn add_signed_commit(
        &self,
        shard: &Shard,
        signed_commit: &Verified<Signed<Commit, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_commit.author();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let signed_commit = signed_commit.deref();

        let mut guard = self.inner.write();

        match guard.signed_commits.get(&encoder_key) {
            Some(existing_commit) => {
                // TODO: use digests to compare Shard message types
                // if existing_commit != signed_commit {
                //     return Err(ShardError::Conflict(
                //         "encoder has a different existing commit".to_string(),
                //     ));
                // }
            }
            None => {
                guard
                    .signed_commits
                    .insert(encoder_key, signed_commit.clone());
            }
        };
        Ok(())
    }
    fn count_signed_commits(&self, shard: &Shard) -> ShardResult<usize> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let count = self
            .inner
            .read()
            .signed_commits
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .count();

        Ok(count)
    }
    fn get_signed_commits(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<Commit, min_sig::BLS12381Signature>>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let commits: Vec<_> = self
            .inner
            .read()
            .signed_commits
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
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

    fn add_signed_reveal(
        &self,
        shard: &Shard,
        signed_reveal: &Verified<Signed<Reveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = signed_reveal.author();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let signed_reveal = signed_reveal.deref();

        let mut guard = self.inner.write();

        match guard.signed_reveals.get(&encoder_key) {
            Some(existing_reveal) => {
                // TODO: use digests to compare Shard message types
                // if existing_reveal != signed_reveal {
                //     return Err(ShardError::Conflict(
                //         "encoder has a different existing signed reveal".to_string(),
                //     ));
                // }
            }
            None => {
                guard
                    .signed_reveals
                    .insert(encoder_key, signed_reveal.clone());
            }
        };
        Ok(())
    }
    fn count_signed_reveal(&self, shard: &Shard) -> ShardResult<usize> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let count = self
            .inner
            .read()
            .signed_reveals
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .count();

        Ok(count)
    }
    fn get_signed_reveals(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<Reveal, min_sig::BLS12381Signature>>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let reveals: Vec<_> = self
            .inner
            .read()
            .signed_reveals
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(reveals)
    }

    fn get_encoder_signed_reveal(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
    ) -> ShardResult<Signed<Reveal, min_sig::BLS12381Signature>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder_key = (epoch, shard_digest, encoder.clone());

        if let Some(reveal) = self.inner.read().signed_reveals.get(&encoder_key) {
            Ok(reveal.clone())
        } else {
            Err(ShardError::NotFound("signed reveal".to_string()))
        }
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
        votes: &Verified<Signed<CommitVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = votes.author();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let votes = &**votes;

        let mut guard = self.inner.write();

        match guard.signed_commit_votes.get(&encoder_key) {
            Some(existing) => {
                // TODO: use digests to compare Shard message types
                // if existing != votes {
                //     return Err(ShardError::Conflict(
                //         "encoder has a different commit vote".to_string(),
                //     ));
                // }
            }
            None => {
                guard.signed_commit_votes.insert(encoder_key, votes.clone());
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
                    guard
                        .commit_accept_digest_voters
                        .entry(receiving_encoder_digest_key.clone())
                        .or_default()
                        .insert(votes.author().clone());

                    let this_digests_votes = guard
                        .commit_accept_digest_voters
                        .get(&receiving_encoder_digest_key)
                        .map_or(0_usize, BTreeSet::len);

                    let current_highest_digest_votes = guard
                        .commit_votes_highest_digest
                        .get(&receiving_encoder_key)
                        .copied()
                        .unwrap_or(0_usize);

                    if this_digests_votes > current_highest_digest_votes {
                        guard
                            .commit_votes_highest_digest
                            .insert(receiving_encoder_key, this_digests_votes);
                    }
                }
                None => {
                    guard
                        .commit_reject_voters
                        .entry(receiving_encoder_key)
                        .or_default()
                        .insert(votes.author().clone());
                }
            }
        }
        Ok(())
    }
    fn count_commit_votes(&self, shard: &Shard) -> ShardResult<usize> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let count = self
            .inner
            .read()
            .signed_commit_votes
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .count();

        Ok(count)
    }
    fn get_commit_votes_for_encoder(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
        digest: Option<&Digest<Signed<Reveal, min_sig::BLS12381Signature>>>,
    ) -> ShardResult<CommitVoteCounts> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let accepts: Option<usize>;

        let mut guard = self.inner.read();

        if let Some(digest) = digest {
            let encoder_digest_key = (epoch, shard_digest, encoder.clone(), digest.to_owned());
            accepts = Some(
                guard
                    .commit_accept_digest_voters
                    .get(&encoder_digest_key)
                    .map(|voters| voters.len())
                    .unwrap_or(0),
            );
        } else {
            accepts = None;
        }

        let rejects = guard
            .commit_reject_voters
            .get(&encoder_key)
            .map(|voters| voters.len())
            .unwrap_or(0);

        let highest = guard
            .commit_votes_highest_digest
            .get(&encoder_key)
            .copied()
            .unwrap_or(0_usize);

        Ok(CommitVoteCounts::new(accepts, rejects, highest))
    }
    fn get_commit_votes(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<CommitVotes, min_sig::BLS12381Signature>>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let commit_votes: Vec<_> = self
            .inner
            .read()
            .signed_commit_votes
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(commit_votes)
    }

    fn add_signed_score(
        &self,
        shard: &Shard,
        score: &Verified<Signed<ShardScore, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = score.author();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let scores = score.deref();

        let mut guard = self.inner.write();

        match guard.signed_scores.get(&encoder_key) {
            Some(existing) => {
                // TODO: use digests to compare Shard message types
                // if existing != scores {
                //     return Err(ShardError::Conflict(
                //         "encoder has a different signed score".to_string(),
                //     ));
                // }
            }
            None => {
                guard.signed_scores.insert(encoder_key, scores.clone());
            }
        };
        Ok(())
    }
    fn get_signed_scores(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<ShardScore, min_sig::BLS12381Signature>>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let scores: Vec<_> = self
            .inner
            .read()
            .signed_scores
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(scores)
    }
    fn add_aggregate_score(
        &self,
        shard: &Shard,
        agg_details: (EncoderAggregateSignature, Vec<EncoderPublicKey>),
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let shard_key = (epoch, shard_digest);

        let mut guard = self.inner.write();

        match guard.agg_scores.get(&shard_key) {
            Some(existing) => {
                // TODO: handle when an agg sig already exists for the shard
                // if existing != &agg_details {
                //     return Err(ShardError::Conflict(
                //         "encoder has a different aggregate score".to_string(),
                //     ));
                // }
            }
            None => {
                guard.agg_scores.insert(shard_key, agg_details);
            }
        };
        Ok(())
    }
    fn get_agg_score(
        &self,
        shard: &Shard,
    ) -> ShardResult<(EncoderAggregateSignature, Vec<EncoderPublicKey>)> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let shard_key = (epoch, shard_digest);

        let inner = self.inner.read();
        if let Some(agg_details) = inner.agg_scores.get(&shard_key) {
            Ok(agg_details.to_owned())
        } else {
            Err(ShardError::NotFound(
                "agg details (scores and evaluators)".to_string(),
            ))
        }
    }
}
