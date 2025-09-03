use crate::finality::FinalityProof;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::{error::SharedResult, metadata::MetadataCommitment, shard::Shard};

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
#[enum_dispatch(InputAPI)]
pub enum Input {
    V1(InputV1),
}

#[enum_dispatch]
pub trait InputAPI {
    fn auth_token(&self) -> &ShardAuthToken;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputV1 {
    auth_token: ShardAuthToken,
}

impl InputV1 {
    pub fn new(auth_token: ShardAuthToken) -> Self {
        Self { auth_token }
    }
}

impl InputAPI for InputV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
}

pub fn verify_input(input: &Input, shard: &Shard) -> SharedResult<()> {
    // TODO: need to fix this to work with the correct signature
    // input.verify_signature(Scope::Input, .author().inner())?;
    Ok(())
}
