use crate::{
    consensus::block::BlockRef,
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
pub struct BlockEntropy(pub Bytes);

impl BlockEntropy {
    pub fn new(bytes: Bytes) -> Self {
        Self(bytes)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct BlockEntropyProof(pub Bytes);
impl BlockEntropyProof {
    pub fn new(bytes: Bytes) -> Self {
        Self(bytes)
    }
}

pub trait EntropyAPI: Send + Sync + Sized + 'static {
    fn get_entropy(
        &mut self,
        block_ref: &BlockRef,
        iterations: Iterations,
    ) -> SomaResult<(BlockEntropy, BlockEntropyProof)>;

    fn verify_entropy(
        &mut self,
        block_ref: &BlockRef,
        block_entropy: &BlockEntropy,
        block_entropy_proof: &BlockEntropyProof,
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
        block_ref: BlockRef,
    ) -> SomaResult<(BlockEntropy, BlockEntropyProof)> {
        let seed = bcs::to_bytes(&(block_ref)).expect("BCS serialization should not fail");
        let input = QuadraticForm::hash_to_group_with_default_parameters(&seed, &DISCRIMINANT_3072)
            .map_err(|e| SomaError::FailedVDF(e.to_string()))?;

        let (output, proof) = self
            .vdf
            .evaluate(&input)
            .map_err(|e| SomaError::FailedVDF(e.to_string()))?;

        let entropy_bytes = bcs::to_bytes(&output).expect("BCS serialization should not fail");

        let entropy = BlockEntropy::new(Bytes::copy_from_slice(&entropy_bytes));
        let proof_bytes = bcs::to_bytes(&proof).expect("BCS serialization should not fail");

        let proof = BlockEntropyProof::new(Bytes::copy_from_slice(&proof_bytes));
        Ok((entropy, proof))
    }

    pub fn verify_entropy(
        &self,
        block_ref: BlockRef,
        tx_entropy: &BlockEntropy,
        tx_entropy_proof: &BlockEntropyProof,
    ) -> SomaResult<()> {
        let seed = bcs::to_bytes(&(block_ref)).expect("BCS serialization should not fail");
        let input = QuadraticForm::hash_to_group_with_default_parameters(&seed, &DISCRIMINANT_3072)
            .map_err(|e| SomaError::FailedVDF(e.to_string()))?;

        let entropy: QuadraticForm =
            bcs::from_bytes(&tx_entropy.0).expect("BCS serialization should not fail");

        let proof: QuadraticForm =
            bcs::from_bytes(&tx_entropy_proof.0).expect("BCS serialization should not fail");

        self.vdf
            .verify(&input, &entropy, &proof)
            .map_err(|e| SomaError::FailedVDF(e.to_string()))
    }
}
