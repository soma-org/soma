use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use store::DBMapUtils;
use store::{
    rocks::{default_db_options, DBMap},
    Map,
};
use tracing::{debug, error};

use crate::types::{
    commit_votes::{CommitVotes, CommitVotesAPI},
    report_vote::{ReportVote, ReportVoteAPI},
};
use types::{
    committee::Epoch,
    error::{ShardError, ShardResult},
    metadata::DownloadMetadata,
    shard::Shard,
    shard_crypto::{
        digest::Digest,
        keys::{EncoderAggregateSignature, EncoderPublicKey},
        verified::Verified,
    },
    submission::{Submission, SubmissionAPI},
};

use super::{ShardStage, Store};

// Helper struct to store timestamp with submissions
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TimestampedData<T> {
    data: T,
    timestamp_millis: u64,
}

impl<T> TimestampedData<T> {
    fn new(data: T) -> Self {
        let timestamp_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        Self {
            data,
            timestamp_millis,
        }
    }

    fn instant(&self) -> Instant {
        // Convert stored timestamp back to an Instant
        // Note: This is approximate as we're losing some precision
        Instant::now()
            - Duration::from_millis(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_millis() as u64
                    - self.timestamp_millis,
            )
    }
}

/// Persistent storage with RocksDB for encoder data.
#[derive(DBMapUtils)]
pub struct RocksDBStore {
    /// Stores encoders that have sent messages at each shard stage
    shard_stage_messages: DBMap<(Epoch, Digest<Shard>, ShardStage), HashSet<EncoderPublicKey>>,

    /// Stores dispatch markers for shard stages
    shard_stage_dispatches: DBMap<(Epoch, Digest<Shard>, ShardStage), ()>,

    input_download_metadata: DBMap<(Epoch, Digest<Shard>), DownloadMetadata>,

    /// Stores submission digests with timestamps
    submission_digests:
        DBMap<(Epoch, Digest<Shard>, EncoderPublicKey), TimestampedData<Digest<Submission>>>,

    /// Stores actual submissions with metadata
    submissions: DBMap<
        (Epoch, Digest<Shard>, Digest<Submission>),
        TimestampedData<(Bytes, DownloadMetadata)>,
    >,

    /// Stores commit votes
    commit_votes: DBMap<(Epoch, Digest<Shard>, EncoderPublicKey), Bytes>,

    /// Stores accepted commits
    accepted_commits: DBMap<(Epoch, Digest<Shard>, EncoderPublicKey), Digest<Submission>>,

    /// Stores report votes
    report_votes: DBMap<(Epoch, Digest<Shard>, EncoderPublicKey), Bytes>,

    /// Stores aggregate scores with evaluators
    agg_scores: DBMap<(Epoch, Digest<Shard>), (EncoderAggregateSignature, Vec<EncoderPublicKey>)>,
}

impl RocksDBStore {
    const SHARD_STAGE_MESSAGES_CF: &'static str = "shard_stage_messages";
    const SHARD_STAGE_DISPATCHES_CF: &'static str = "shard_stage_dispatches";
    const INPUT_DOWNLOAD_METADATA_CF: &'static str = "input_download_metadata";
    const SUBMISSION_DIGESTS_CF: &'static str = "submission_digests";
    const SUBMISSIONS_CF: &'static str = "submissions";
    const COMMIT_VOTES_CF: &'static str = "commit_votes";
    const ACCEPTED_COMMITS_CF: &'static str = "accepted_commits";
    const REPORT_VOTES_CF: &'static str = "report_votes";
    const AGG_SCORES_CF: &'static str = "agg_scores";

    /// Creates a new instance of RocksDB storage.
    pub fn new(path: &str) -> Self {
        // Encoder data has moderate write throughput and is read frequently during processing
        let db_options = default_db_options();
        let cf_options = default_db_options().optimize_for_point_lookup(64);

        let column_families = vec![
            (
                Self::SHARD_STAGE_MESSAGES_CF.to_string(),
                cf_options.clone(),
            ),
            (
                Self::SHARD_STAGE_DISPATCHES_CF.to_string(),
                cf_options.clone(),
            ),
            (
                Self::INPUT_DOWNLOAD_METADATA_CF.to_string(),
                cf_options.clone(),
            ),
            (Self::SUBMISSION_DIGESTS_CF.to_string(), cf_options.clone()),
            (
                Self::SUBMISSIONS_CF.to_string(),
                default_db_options()
                    .optimize_for_write_throughput()
                    .set_block_options(512, 128 << 10),
            ), // Larger blocks for submissions
            (Self::COMMIT_VOTES_CF.to_string(), cf_options.clone()),
            (Self::ACCEPTED_COMMITS_CF.to_string(), cf_options.clone()),
            (Self::REPORT_VOTES_CF.to_string(), cf_options.clone()),
            (Self::AGG_SCORES_CF.to_string(), cf_options.clone()),
        ];

        let column_family_options =
            store::rocks::DBMapTableConfigMap::new(column_families.into_iter().collect());

        Self::open_tables_read_write(
            path.into(),
            Some(db_options.options),
            Some(column_family_options),
        )
    }
}

impl Store for RocksDBStore {
    fn add_shard_stage_message(
        &self,
        shard: &Shard,
        stage: ShardStage,
        encoder: &EncoderPublicKey,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest, stage);

        // Get existing set or create new one
        let mut encoders = self
            .shard_stage_messages
            .get(&key)?
            .unwrap_or_else(HashSet::new);

        if !encoders.insert(encoder.clone()) {
            return Err(ShardError::Conflict("encoder already in stage".to_string()));
        }

        self.shard_stage_messages.insert(&key, &encoders)?;
        Ok(())
    }

    fn add_shard_stage_dispatch(&self, shard: &Shard, stage: ShardStage) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest, stage);

        if self.shard_stage_dispatches.contains_key(&key)? {
            Err(ShardError::Conflict("stage already exists".to_string()))
        } else {
            self.shard_stage_dispatches.insert(&key, &())?;
            Ok(())
        }
    }
    fn add_input_download_metadata(
        &self,
        shard: &Shard,
        download_metadata: DownloadMetadata,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest);

        if self.input_download_metadata.contains_key(&key)? {
            return Err(ShardError::RecvDuplicate);
        }

        self.input_download_metadata
            .insert(&key, &download_metadata)?;
        Ok(())
    }

    fn get_input_download_metadata(&self, shard: &Shard) -> ShardResult<DownloadMetadata> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest);

        self.input_download_metadata
            .get(&key)?
            .ok_or_else(|| ShardError::NotFound("submission digest".to_string()))
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

        if self.submission_digests.contains_key(&key)? {
            return Err(ShardError::RecvDuplicate);
        }

        let timestamped = TimestampedData::new(submission_digest);
        self.submission_digests.insert(&key, &timestamped)?;
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

        self.submission_digests
            .get(&key)?
            .map(|timestamped| (timestamped.data, timestamped.instant()))
            .ok_or_else(|| ShardError::NotFound("submission digest".to_string()))
    }

    fn get_all_submission_digests(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<(EncoderPublicKey, Digest<Submission>, Instant)>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let mut results = Vec::new();
        let lower_bound = (epoch, shard_digest, EncoderPublicKey::MIN());
        let upper_bound = (epoch, shard_digest, EncoderPublicKey::MAX());

        for item in self
            .submission_digests
            .safe_iter_with_bounds(Some(lower_bound), Some(upper_bound))
        {
            let ((e, sd, encoder), timestamped) = item?;
            if e == epoch && sd == shard_digest {
                results.push((encoder, timestamped.data, timestamped.instant()));
            }
        }

        Ok(results)
    }

    fn add_commit_votes(&self, shard: &Shard, votes: &Verified<CommitVotes>) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = votes.author();
        let key = (epoch, shard_digest, encoder.clone());

        if self.commit_votes.contains_key(&key)? {
            return Err(ShardError::Conflict(
                "encoder has a different commit vote".to_string(),
            ));
        }

        let serialized =
            bcs::to_bytes(&**votes).map_err(|e| ShardError::SerializationFailure(e.to_string()))?;
        self.commit_votes.insert(&key, &Bytes::from(serialized))?;
        Ok(())
    }

    fn get_all_commit_votes(&self, shard: &Shard) -> ShardResult<Vec<CommitVotes>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let mut results = Vec::new();
        let lower_bound = (epoch, shard_digest, EncoderPublicKey::MIN());
        let upper_bound = (epoch, shard_digest, EncoderPublicKey::MAX());

        for item in self
            .commit_votes
            .safe_iter_with_bounds(Some(lower_bound), Some(upper_bound))
        {
            let ((e, sd, _), bytes) = item?;
            if e == epoch && sd == shard_digest {
                let votes: CommitVotes = bcs::from_bytes(&bytes)
                    .map_err(|e| ShardError::SerializationFailure(e.to_string()))?;
                results.push(votes);
            }
        }

        Ok(results)
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

        if self.accepted_commits.contains_key(&key)? {
            return Err(ShardError::Conflict(
                "encoder has a different commit vote".to_string(),
            ));
        }

        self.accepted_commits.insert(&key, &submission_digest)?;
        Ok(())
    }

    fn get_all_accepted_commits(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<(EncoderPublicKey, Digest<Submission>)>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let mut results = Vec::new();
        let lower_bound = (epoch, shard_digest, EncoderPublicKey::MIN());
        let upper_bound = (epoch, shard_digest, EncoderPublicKey::MAX());

        for item in self
            .accepted_commits
            .safe_iter_with_bounds(Some(lower_bound), Some(upper_bound))
        {
            let ((e, sd, encoder), submission_digest) = item?;
            if e == epoch && sd == shard_digest {
                results.push((encoder, submission_digest));
            }
        }

        Ok(results)
    }

    fn add_submission(
        &self,
        shard: &Shard,
        submission: Submission,
        embedding_download_metadata: DownloadMetadata,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = submission.shard_digest();
        let submission_digest = Digest::new(&submission).map_err(ShardError::DigestFailure)?;
        let key = (epoch, shard_digest, submission_digest);

        if self.submissions.contains_key(&key)? {
            return Ok(()); // Already exists, skip
        }

        let serialized = bcs::to_bytes(&submission)
            .map_err(|e| ShardError::SerializationFailure(e.to_string()))?;
        let timestamped =
            TimestampedData::new((Bytes::from(serialized), embedding_download_metadata));
        self.submissions.insert(&key, &timestamped)?;
        Ok(())
    }

    fn get_submission(
        &self,
        shard: &Shard,
        submission_digest: Digest<Submission>,
    ) -> ShardResult<(Submission, Instant, DownloadMetadata)> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest, submission_digest);

        self.submissions
            .get(&key)?
            .map(|timestamped| {
                let submission: Submission = bcs::from_bytes(&timestamped.data.0)
                    .map_err(|e| ShardError::SerializationFailure(e.to_string()))?;
                Ok((submission, timestamped.instant(), timestamped.data.1))
            })
            .ok_or_else(|| ShardError::NotFound("submission".to_string()))?
    }

    fn get_all_submissions(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<(Submission, Instant, DownloadMetadata)>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let mut results = Vec::new();
        let lower_bound = (epoch, shard_digest, Digest::<Submission>::MIN);
        let upper_bound = (epoch, shard_digest, Digest::<Submission>::MAX);

        for item in self
            .submissions
            .safe_iter_with_bounds(Some(lower_bound), Some(upper_bound))
        {
            let ((e, sd, _), timestamped) = item?;
            if e == epoch && sd == shard_digest {
                let submission: Submission = bcs::from_bytes(&timestamped.data.0)
                    .map_err(|e| ShardError::SerializationFailure(e.to_string()))?;
                results.push((submission, timestamped.instant(), timestamped.data.1));
            }
        }

        Ok(results)
    }

    fn add_report_vote(
        &self,
        shard: &Shard,
        report_vote: &Verified<ReportVote>,
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let encoder = report_vote.author();
        let key = (epoch, shard_digest, encoder.clone());

        // For now, we allow overwriting existing report votes
        let serialized = bcs::to_bytes(&**report_vote)
            .map_err(|e| ShardError::SerializationFailure(e.to_string()))?;
        self.report_votes.insert(&key, &Bytes::from(serialized))?;
        Ok(())
    }

    fn get_all_report_votes(&self, shard: &Shard) -> ShardResult<Vec<ReportVote>> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;

        let mut results = Vec::new();
        let lower_bound = (epoch, shard_digest, EncoderPublicKey::MIN());
        let upper_bound = (epoch, shard_digest, EncoderPublicKey::MAX());

        for item in self
            .report_votes
            .safe_iter_with_bounds(Some(lower_bound), Some(upper_bound))
        {
            let ((e, sd, _), bytes) = item?;
            if e == epoch && sd == shard_digest {
                let vote: ReportVote = bcs::from_bytes(&bytes)
                    .map_err(|e| ShardError::SerializationFailure(e.to_string()))?;
                results.push(vote);
            }
        }

        Ok(results)
    }

    fn add_aggregate_score(
        &self,
        shard: &Shard,
        agg_details: (EncoderAggregateSignature, Vec<EncoderPublicKey>),
    ) -> ShardResult<()> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest);

        // For now, we allow overwriting existing aggregate scores
        self.agg_scores.insert(&key, &agg_details)?;
        Ok(())
    }

    fn get_aggregate_score(
        &self,
        shard: &Shard,
    ) -> ShardResult<(EncoderAggregateSignature, Vec<EncoderPublicKey>)> {
        let epoch = shard.epoch();
        let shard_digest = shard.digest()?;
        let key = (epoch, shard_digest);

        self.agg_scores
            .get(&key)?
            .ok_or_else(|| ShardError::NotFound("agg details (scores and evaluators)".to_string()))
    }
}
