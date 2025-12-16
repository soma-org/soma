use parking_lot::RwLock;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::Deref,
    time::Instant,
};
use types::{
    committee::Epoch,
    error::{ShardError, ShardResult},
    metadata::DownloadMetadata,
    shard::{Input, Shard},
    shard_crypto::{
        digest::Digest,
        keys::{EncoderAggregateSignature, EncoderPublicKey},
        verified::Verified,
    },
    submission::{Submission, SubmissionAPI},
};

use crate::types::{
    commit_votes::{CommitVotes, CommitVotesAPI},
    report_vote::{ReportVote, ReportVoteAPI},
};

use super::{ShardStage, Store};

/// In-memory storage for testing.
#[allow(unused)]
pub(crate) struct MemStore {
    inner: RwLock<Inner>,
}

type Encoder = EncoderPublicKey;

#[allow(unused)]
pub(crate) struct Inner {
    shard_stage_messages: BTreeMap<(Epoch, Digest<Shard>, ShardStage), HashSet<EncoderPublicKey>>,
    shard_stage_dispatches: BTreeMap<(Epoch, Digest<Shard>, ShardStage), ()>,
    inputs: BTreeMap<(Epoch, Digest<Shard>), Verified<Input>>,
    submission_digests: BTreeMap<(Epoch, Digest<Shard>, Encoder), (Digest<Submission>, Instant)>,
    submissions: BTreeMap<(Epoch, Digest<Shard>, Digest<Submission>), (Submission, Instant)>,
    commit_votes: BTreeMap<(Epoch, Digest<Shard>, Encoder), CommitVotes>,
    accepted_commits: BTreeMap<(Epoch, Digest<Shard>, Encoder), Digest<Submission>>,
    report_votes: BTreeMap<(Epoch, Digest<Shard>, Encoder), ReportVote>,
    agg_scores:
        BTreeMap<(Epoch, Digest<Shard>), (EncoderAggregateSignature, Vec<EncoderPublicKey>)>,
}

impl MemStore {
    pub(crate) const fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                shard_stage_messages: BTreeMap::new(),
                shard_stage_dispatches: BTreeMap::new(),
                inputs: BTreeMap::new(),
                submission_digests: BTreeMap::new(),
                submissions: BTreeMap::new(),
                commit_votes: BTreeMap::new(),
                accepted_commits: BTreeMap::new(),
                report_votes: BTreeMap::new(),
                agg_scores: BTreeMap::new(),
            }),
        }
    }
}

impl Store for MemStore {
    fn add_shard_stage_message(
        &self,
        shard: &Shard,
        stage: ShardStage,
        encoder: &EncoderPublicKey,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest, stage);

        let mut guard = self.inner.write();
        let set = guard
            .shard_stage_messages
            .entry(key)
            .or_insert_with(HashSet::new);
        if !set.insert(encoder.clone()) {
            Err(ShardError::Conflict("encoder already in stage".to_string()))
        } else {
            Ok(())
        }
    }

    fn add_shard_stage_dispatch(&self, shard: &Shard, stage: ShardStage) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest, stage);

        let mut guard = self.inner.write();
        if guard.shard_stage_dispatches.insert(key, ()).is_none() {
            Ok(())
        } else {
            Err(ShardError::Conflict("stage already exists".to_string()))
        }
    }
    fn add_input(&self, shard: &Shard, input: Verified<Input>) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest);

        let mut guard = self.inner.write();

        match guard.inputs.get(&key) {
            Some(_existing) => Err(ShardError::Conflict(
                "encoder has existing input".to_string(),
            )),
            None => {
                guard.inputs.insert(key, input);
                Ok(())
            }
        }
    }
    fn get_input(&self, shard: &Shard) -> ShardResult<Verified<Input>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest);

        let inner = self.inner.read();
        if let Some(verified_input) = inner.inputs.get(&key) {
            Ok(verified_input.clone())
        } else {
            Err(ShardError::NotFound("input".to_string()))
        }
    }

    fn add_submission_digest(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
        submission_digest: Digest<Submission>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest, encoder.clone());
        let mut guard = self.inner.write();

        match guard.submission_digests.get(&key) {
            Some(_exists) => return Err(ShardError::RecvDuplicate),
            None => {
                guard
                    .submission_digests
                    .insert(key, (submission_digest, Instant::now()));
            }
        };
        Ok(())
    }

    fn get_submission_digest(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
    ) -> ShardResult<(Digest<Submission>, Instant)> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest, encoder.clone());
        let inner = self.inner.read();
        if let Some(submission_digest) = inner.submission_digests.get(&key) {
            Ok(submission_digest.to_owned())
        } else {
            Err(ShardError::NotFound("score set".to_string()))
        }
    }

    fn get_all_submission_digests(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<(EncoderPublicKey, Digest<Submission>, Instant)>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let submission_digests = self
            .inner
            .read()
            .submission_digests
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .map(|((_, _, encoder), (submission_digest, instant))| {
                (encoder.clone(), submission_digest.clone(), instant.clone())
            })
            .collect();

        Ok(submission_digests)
    }

    fn add_commit_votes(&self, shard: &Shard, votes: &Verified<CommitVotes>) -> ShardResult<()> {
        // TODO: change this to not store the commit votes type but rather just the accept messages
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = votes.author();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let votes = &**votes;

        let mut guard = self.inner.write();

        match guard.commit_votes.get(&encoder_key) {
            Some(_existing) => Err(ShardError::Conflict(
                "encoder has a different commit vote".to_string(),
            )),
            None => {
                guard.commit_votes.insert(encoder_key, votes.clone());
                Ok(())
            }
        }
    }

    fn get_all_commit_votes(&self, shard: &Shard) -> ShardResult<Vec<CommitVotes>> {
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
    fn add_accepted_commit(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
        submission_digest: Digest<Submission>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest, encoder.clone());

        let mut guard = self.inner.write();

        match guard.accepted_commits.get(&key) {
            Some(_existing) => Err(ShardError::Conflict(
                "encoder has a different commit vote".to_string(),
            )),
            None => {
                guard.accepted_commits.insert(key, submission_digest);
                Ok(())
            }
        }
    }

    fn get_all_accepted_commits(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<(EncoderPublicKey, Digest<Submission>)>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let accepted_commits = self
            .inner
            .read()
            .accepted_commits
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .map(|((_, _, encoder), submission_digest)| {
                (encoder.clone(), submission_digest.clone())
            })
            .collect();

        Ok(accepted_commits)
    }

    fn add_submission(&self, shard: &Shard, submission: Submission) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = submission.shard_digest().clone();
        let submission_digest = Digest::new(&submission).map_err(ShardError::DigestFailure)?;
        let key = (epoch, shard_digest, submission_digest);

        let mut guard = self.inner.write();

        match guard.submissions.get(&key) {
            Some(_) => {}
            None => {
                guard.submissions.insert(key, (submission, Instant::now()));
            }
        };
        Ok(())
    }

    fn get_submission(
        &self,
        shard: &Shard,
        submission_digest: Digest<Submission>,
    ) -> ShardResult<(Submission, Instant)> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest, submission_digest);
        let inner = self.inner.read();
        if let Some(submission) = inner.submissions.get(&key) {
            Ok(submission.to_owned())
        } else {
            Err(ShardError::NotFound("score set".to_string()))
        }
    }

    fn get_all_submissions(&self, shard: &Shard) -> ShardResult<Vec<(Submission, Instant)>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let submissions = self
            .inner
            .read()
            .submissions
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .map(|(_, (submission, instant))| (submission.clone(), instant.clone()))
            .collect();

        Ok(submissions)
    }

    fn add_report_vote(
        &self,
        shard: &Shard,
        report_vote: &Verified<ReportVote>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = report_vote.author();
        let encoder_key = (epoch, shard_digest, encoder.clone());
        let report_vote = report_vote.deref();

        let mut guard = self.inner.write();

        match guard.report_votes.get(&encoder_key) {
            Some(existing) => {
                // TODO: use digests to compare Shard message types
                // if existing != scores {
                //     return Err(ShardError::Conflict(
                //         "encoder has a different signed score".to_string(),
                //     ));
                // }
            }
            None => {
                guard.report_votes.insert(encoder_key, report_vote.clone());
            }
        };
        Ok(())
    }
    fn get_all_report_votes(&self, shard: &Shard) -> ShardResult<Vec<ReportVote>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let report_votes: Vec<_> = self
            .inner
            .read()
            .report_votes
            .iter()
            .filter(|((e, sd, _), _)| *e == epoch && *sd == shard_digest)
            .map(|(_, value)| value.clone())
            .collect();

        Ok(report_votes)
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
    fn get_aggregate_score(
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
