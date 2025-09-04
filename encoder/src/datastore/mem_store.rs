use parking_lot::RwLock;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    time::Instant,
};
use tracing::{info, warn};
use types::{
    committee::Epoch,
    error::{ShardError, ShardResult},
    shard::Shard,
    shard_crypto::{
        digest::Digest,
        keys::{EncoderAggregateSignature, EncoderPublicKey},
        verified::Verified,
    },
};

use crate::types::{
    commit::{Commit, CommitAPI},
    commit_votes::{CommitVotes, CommitVotesAPI},
    reveal::{Reveal, RevealAPI},
    score_vote::{ScoreVote, ScoreVoteAPI},
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
    commit_digests: BTreeMap<(Epoch, Digest<Shard>, Encoder), Digest<Commit>>,
    #[allow(clippy::type_complexity)]
    shard_committers: BTreeMap<(Epoch, Digest<Shard>, Committer), Digest<Commit>>,
    #[allow(clippy::type_complexity)]
    commits: BTreeMap<(Epoch, Digest<Shard>, Encoder), Commit>,

    reveals: BTreeMap<(Epoch, Digest<Shard>, Encoder), Reveal>,

    first_commit_time: BTreeMap<(Epoch, Digest<Shard>), Instant>,

    //      ////////////////////
    commit_votes: BTreeMap<(Epoch, Digest<Shard>, Encoder), CommitVotes>,
    #[allow(clippy::type_complexity)]
    commit_accept_digest_voters: BTreeMap<
        (Epoch, Digest<Shard>, EncoderPublicKey, Digest<Reveal>),
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

    score_votes: BTreeMap<(Epoch, Digest<Shard>, Encoder), ScoreVote>,

    agg_scores:
        BTreeMap<(Epoch, Digest<Shard>), (EncoderAggregateSignature, Vec<EncoderPublicKey>)>,
}

impl MemStore {
    pub(crate) const fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                commit_digests: BTreeMap::new(),
                shard_committers: BTreeMap::new(),
                commits: BTreeMap::new(),
                reveals: BTreeMap::new(),
                first_commit_time: BTreeMap::new(),
                commit_votes: BTreeMap::new(),
                commit_accept_digest_voters: BTreeMap::new(),
                commit_reject_voters: BTreeMap::new(),
                commit_votes_highest_digest: BTreeMap::new(),
                reveal_accept_voters: BTreeMap::new(),
                reveal_reject_voters: BTreeMap::new(),
                score_votes: BTreeMap::new(),
                first_reveal_time: BTreeMap::new(),
                agg_scores: BTreeMap::new(),
            }),
        }
    }
}

impl Store for MemStore {
    fn lock_commit(&self, shard: &Shard, commit: &Commit) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = commit.author();
        let committer = commit.author();

        let commit_digest = Digest::new(commit).map_err(ShardError::DigestFailure)?;

        let commit_digests_key = (epoch, shard_digest, encoder.clone());
        let shard_committer_key = (epoch, shard_digest, committer.clone());

        let mut inner = self.inner.write();

        match (
            inner.commit_digests.get(&commit_digests_key),
            inner.shard_committers.get(&shard_committer_key),
        ) {
            // Committer has committed before
            (Some(existing_digest), _) if existing_digest != &commit_digest => {
                warn!("Committer has committed before");
                return Err(ShardError::Conflict(
                    "existing commit from committer".to_string(),
                ));
            }
            // Slot is taken with different commit
            (_, Some(existing)) if existing != &commit_digest => {
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
                    .commit_digests
                    .insert(commit_digests_key, commit_digest);
                info!("Inserted into signed commit digests");
                inner
                    .shard_committers
                    .insert(shard_committer_key, commit_digest);
                info!("Inserted into shard committers");
            }
            // Everything matches, idempotent case
            _ => (),
        }
        Ok(())
    }
    fn add_commit(&self, shard: &Shard, commit: &Verified<Commit>) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = commit.author();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let commit = commit.deref();

        let mut guard = self.inner.write();

        match guard.commits.get(&encoder_key) {
            Some(existing_commit) => {
                // TODO: use digests to compare Shard message types
                // if existing_commit != commit {
                //     return Err(ShardError::Conflict(
                //         "encoder has a different existing commit".to_string(),
                //     ));
                // }
            }
            None => {
                guard.commits.insert(encoder_key, commit.clone());
            }
        };
        Ok(())
    }
    fn count_commits(&self, shard: &Shard) -> ShardResult<usize> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let count = self
            .inner
            .read()
            .commits
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .count();

        Ok(count)
    }
    fn get_commits(&self, shard: &Shard) -> ShardResult<Vec<Commit>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let commits: Vec<_> = self
            .inner
            .read()
            .commits
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

    fn add_reveal(&self, shard: &Shard, reveal: &Verified<Reveal>) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = reveal.author();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let reveal = reveal.deref();

        let mut guard = self.inner.write();

        match guard.reveals.get(&encoder_key) {
            Some(existing_reveal) => {
                // TODO: use digests to compare Shard message types
                // if existing_reveal != reveal {
                //     return Err(ShardError::Conflict(
                //         "encoder has a different existing signed reveal".to_string(),
                //     ));
                // }
            }
            None => {
                guard.reveals.insert(encoder_key, reveal.clone());
            }
        };
        Ok(())
    }
    fn count_reveal(&self, shard: &Shard) -> ShardResult<usize> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let count = self
            .inner
            .read()
            .reveals
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .count();

        Ok(count)
    }
    fn get_reveals(&self, shard: &Shard) -> ShardResult<Vec<Reveal>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let reveals: Vec<_> = self
            .inner
            .read()
            .reveals
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(reveals)
    }

    fn get_encoder_reveal(&self, shard: &Shard, encoder: &EncoderPublicKey) -> ShardResult<Reveal> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder_key = (epoch, shard_digest, encoder.clone());

        if let Some(reveal) = self.inner.read().reveals.get(&encoder_key) {
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
    fn add_commit_votes(&self, shard: &Shard, votes: &Verified<CommitVotes>) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = votes.author();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let votes = &**votes;

        let mut guard = self.inner.write();

        match guard.commit_votes.get(&encoder_key) {
            Some(existing) => {
                // TODO: use digests to compare Shard message types
                // if existing != votes {
                //     return Err(ShardError::Conflict(
                //         "encoder has a different commit vote".to_string(),
                //     ));
                // }
            }
            None => {
                guard.commit_votes.insert(encoder_key, votes.clone());
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
            .commit_votes
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .count();

        Ok(count)
    }
    fn get_commit_votes_for_encoder(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
        digest: Option<&Digest<Reveal>>,
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
    fn get_commit_votes(&self, shard: &Shard) -> ShardResult<Vec<CommitVotes>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let commit_votes: Vec<_> = self
            .inner
            .read()
            .commit_votes
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(commit_votes)
    }

    fn add_score_vote(&self, shard: &Shard, score_vote: &Verified<ScoreVote>) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = score_vote.author();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let score_vote = score_vote.deref();

        let mut guard = self.inner.write();

        match guard.score_votes.get(&encoder_key) {
            Some(existing) => {
                // TODO: use digests to compare Shard message types
                // if existing != scores {
                //     return Err(ShardError::Conflict(
                //         "encoder has a different signed score".to_string(),
                //     ));
                // }
            }
            None => {
                guard.score_votes.insert(encoder_key, score_vote.clone());
            }
        };
        Ok(())
    }
    fn get_score_vote(&self, shard: &Shard) -> ShardResult<Vec<ScoreVote>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let score_votes: Vec<_> = self
            .inner
            .read()
            .score_votes
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(score_votes)
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
