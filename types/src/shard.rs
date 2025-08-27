use serde::{Deserialize, Serialize};
use shared::{metadata::MetadataCommitment, shard::Shard};

use crate::finality::FinalityProof;

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
}
