use enum_dispatch::enum_dispatch;
use evaluation::{EvaluationScore, ProbeSet, SummaryEmbedding};
use serde::{Deserialize, Serialize};
use shared::{crypto::keys::EncoderPublicKey, digest::Digest, shard::Shard};

#[enum_dispatch]
pub trait ScoreSetAPI {
    fn winner(&self) -> &EncoderPublicKey;
    fn score(&self) -> &EvaluationScore;
    fn summary_embedding(&self) -> &SummaryEmbedding;
    fn probe_set(&self) -> &ProbeSet;
    fn shard_digest(&self) -> Digest<Shard>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ScoreSetAPI)]
pub enum ScoreSet {
    V1(ScoreSetV1),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScoreSetV1 {
    winner: EncoderPublicKey,
    score: EvaluationScore,
    summary_embedding: SummaryEmbedding,
    probe_set: ProbeSet,
    shard_digest: Digest<Shard>,
}
impl ScoreSetV1 {
    pub fn new(
        winner: EncoderPublicKey,
        score: EvaluationScore,
        summary_embedding: SummaryEmbedding,
        probe_set: ProbeSet,
        shard_digest: Digest<Shard>,
    ) -> Self {
        Self {
            winner,
            score,
            summary_embedding,
            probe_set,
            shard_digest,
        }
    }
}

impl ScoreSetAPI for ScoreSetV1 {
    fn winner(&self) -> &EncoderPublicKey {
        &self.winner
    }
    fn score(&self) -> &EvaluationScore {
        &self.score
    }
    fn summary_embedding(&self) -> &SummaryEmbedding {
        &self.summary_embedding
    }
    fn probe_set(&self) -> &ProbeSet {
        &self.probe_set
    }
    fn shard_digest(&self) -> Digest<Shard> {
        self.shard_digest.clone()
    }
}
