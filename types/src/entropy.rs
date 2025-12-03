use crate::{
    consensus::block::BlockRef,
    digests::CheckpointDigest,
    error::{SomaError, SomaResult},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use vdf::{
    class_group::{discriminant::DISCRIMINANT_3072, QuadraticForm},
    vdf::{wesolowski::DefaultVDF, VDF},
};

type Iterations = u64;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct CheckpointEntropy(pub Bytes);

impl CheckpointEntropy {
    pub fn new(bytes: Bytes) -> Self {
        Self(bytes)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct CheckpointEntropyProof(pub Bytes);
impl CheckpointEntropyProof {
    pub fn new(bytes: Bytes) -> Self {
        Self(bytes)
    }
}

pub trait EntropyAPI: Send + Sync + Sized + 'static {
    fn get_entropy(
        &mut self,
        checkpoint_digest: &CheckpointDigest,
        iterations: Iterations,
    ) -> SomaResult<(CheckpointEntropy, CheckpointEntropyProof)>;

    fn verify_entropy(
        &mut self,
        checkpoint_digest: &CheckpointDigest,
        entropy: &CheckpointEntropy,
        proof: &CheckpointEntropyProof,
        iterations: Iterations,
    ) -> SomaResult<()>;
}

pub struct SimpleVDF {
    vdf: Arc<DefaultVDF>,
}

impl SimpleVDF {
    pub fn new(iterations: Iterations) -> Self {
        Self {
            vdf: Arc::new(DefaultVDF::new(DISCRIMINANT_3072.clone(), iterations)),
        }
    }

    pub fn get_entropy(
        &self,
        checkpoint_digest: &CheckpointDigest,
    ) -> SomaResult<(CheckpointEntropy, CheckpointEntropyProof)> {
        let input = QuadraticForm::hash_to_group_with_default_parameters(
            checkpoint_digest.inner(),
            &DISCRIMINANT_3072,
        )
        .map_err(|e| SomaError::FailedVDF(e.to_string()))?;

        let (output, proof) = self
            .vdf
            .evaluate(&input)
            .map_err(|e| SomaError::FailedVDF(e.to_string()))?;

        let entropy_bytes = bcs::to_bytes(&output).expect("BCS serialization should not fail");

        let entropy = CheckpointEntropy::new(Bytes::copy_from_slice(&entropy_bytes));
        let proof_bytes = bcs::to_bytes(&proof).expect("BCS serialization should not fail");

        let proof = CheckpointEntropyProof::new(Bytes::copy_from_slice(&proof_bytes));
        Ok((entropy, proof))
    }

    pub fn verify_entropy(
        &self,
        checkpoint_digest: &CheckpointDigest,
        entropy: &CheckpointEntropy,
        proof: &CheckpointEntropyProof,
    ) -> SomaResult<()> {
        let input = QuadraticForm::hash_to_group_with_default_parameters(
            checkpoint_digest.inner(),
            &DISCRIMINANT_3072,
        )
        .map_err(|e| SomaError::FailedVDF(e.to_string()))?;

        let entropy: QuadraticForm =
            bcs::from_bytes(&entropy.0).expect("BCS serialization should not fail");

        let proof: QuadraticForm =
            bcs::from_bytes(&proof.0).expect("BCS serialization should not fail");

        self.vdf
            .verify(&input, &entropy, &proof)
            .map_err(|e| SomaError::FailedVDF(e.to_string()))
    }
}
