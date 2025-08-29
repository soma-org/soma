use crate::finality::FinalityProof;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::{metadata::MetadataCommitment, shard::Shard};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardAuthToken {
    pub finality_proof: FinalityProof,
    pub metadata_commitment: MetadataCommitment,
    pub shard: Shard,
}

impl ShardAuthToken {
    pub fn new(
        finality_proof: FinalityProof,
        metadata_commitment: MetadataCommitment,
        shard: Shard,
    ) -> Self {
        Self {
            finality_proof,
            metadata_commitment,
            shard,
        }
    }
    pub fn metadata_commitment(&self) -> MetadataCommitment {
        self.metadata_commitment.clone()
    }

    pub fn epoch(&self) -> u64 {
        self.finality_proof.consensus_finality.leader_block.epoch
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[enum_dispatch(ShardInputAPI)]
pub enum ShardInput {
    V1(ShardInputV1),
}

#[enum_dispatch]
pub trait ShardInputAPI {
    fn auth_token(&self) -> &ShardAuthToken;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ShardInputV1 {
    auth_token: ShardAuthToken,
}

impl ShardInputV1 {
    pub fn new(auth_token: ShardAuthToken) -> Self {
        Self { auth_token }
    }
}

impl ShardInputAPI for ShardInputV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
}
