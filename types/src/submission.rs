use crate::evaluation::{EmbeddingDigest, ProbeSet, Score};
use crate::metadata::Metadata;
use crate::{shard::Shard, shard_crypto::digest::Digest, shard_crypto::keys::EncoderPublicKey};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
pub trait SubmissionAPI {
    fn encoder(&self) -> &EncoderPublicKey;
    fn shard_digest(&self) -> Digest<Shard>;
    fn metadata(&self) -> &Metadata;
    fn probe_set(&self) -> &ProbeSet;
    fn score(&self) -> &Score;
    fn summary_digest(&self) -> &EmbeddingDigest;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(SubmissionAPI)]
pub enum Submission {
    V1(SubmissionV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SubmissionV1 {
    encoder: EncoderPublicKey,
    shard_digest: Digest<Shard>,
    metadata: Metadata,
    probe_set: ProbeSet,
    score: Score,
    summary_digest: EmbeddingDigest,
}

impl SubmissionV1 {
    pub fn new(
        encoder: EncoderPublicKey,
        score: Score,
        summary_digest: EmbeddingDigest,
        probe_set: ProbeSet,
        metadata: Metadata,
        shard_digest: Digest<Shard>,
    ) -> Self {
        Self {
            encoder,
            score,
            summary_digest,
            probe_set,
            metadata,
            shard_digest,
        }
    }
}

impl SubmissionAPI for SubmissionV1 {
    fn encoder(&self) -> &EncoderPublicKey {
        &self.encoder
    }
    fn score(&self) -> &Score {
        &self.score
    }
    fn summary_digest(&self) -> &EmbeddingDigest {
        &self.summary_digest
    }
    fn probe_set(&self) -> &ProbeSet {
        &self.probe_set
    }
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    fn shard_digest(&self) -> Digest<Shard> {
        self.shard_digest.clone()
    }
}
