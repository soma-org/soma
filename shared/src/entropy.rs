use std::sync::Arc;

use bytes::Bytes;
use fastcrypto_vdf::{
    class_group::{discriminant::DISCRIMINANT_3072, QuadraticForm},
    vdf::{wesolowski::DefaultVDF, VDF},
};
use serde::{Deserialize, Serialize};

use crate::{
    block::{BlockRef, Epoch},
    error::{SharedError, SharedResult},
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BlockEntropyOutput(Bytes);
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BlockEntropyProof(Bytes);

type EntropyIterations = u64;

pub trait EntropyAPI {
    fn get_entropy(
        &self,
        epoch: Epoch,
        block_ref: BlockRef,
    ) -> SharedResult<(BlockEntropyOutput, BlockEntropyProof)>;

    fn verify_entropy(
        &self,
        epoch: Epoch,
        block_ref: BlockRef,
        tx_entropy: &BlockEntropyOutput,
        tx_entropy_proof: &BlockEntropyProof,
    ) -> SharedResult<()>;
}

pub struct EntropyVDF {
    vdf: DefaultVDF,
}

impl EntropyVDF {
    pub fn new(iterations: EntropyIterations) -> Self {
        Self {
            vdf: DefaultVDF::new(DISCRIMINANT_3072.clone(), iterations),
        }
    }
}

impl EntropyAPI for EntropyVDF {
    fn get_entropy(
        &self,
        epoch: Epoch,
        block_ref: BlockRef,
    ) -> SharedResult<(BlockEntropyOutput, BlockEntropyProof)> {
        let seed = bcs::to_bytes(&(epoch, block_ref)).map_err(SharedError::SerializationFailure)?;
        let input = QuadraticForm::hash_to_group_with_default_parameters(&seed, &DISCRIMINANT_3072)
            .map_err(|e| SharedError::FailedVDF(e.to_string()))?;

        let (output, proof) = self
            .vdf
            .evaluate(&input)
            .map_err(|e| SharedError::FailedVDF(e.to_string()))?;

        let entropy_bytes =
            bcs::to_bytes(&output).map_err(|e| SharedError::SerializationFailure(e))?;

        let entropy = BlockEntropyOutput(Bytes::copy_from_slice(&entropy_bytes));
        let proof_bytes =
            bcs::to_bytes(&proof).map_err(|e| SharedError::SerializationFailure(e))?;

        let proof = BlockEntropyProof(Bytes::copy_from_slice(&proof_bytes));
        Ok((entropy, proof))
    }

    fn verify_entropy(
        &self,
        epoch: Epoch,
        block_ref: BlockRef,
        tx_entropy: &BlockEntropyOutput,
        tx_entropy_proof: &BlockEntropyProof,
    ) -> SharedResult<()> {
        let seed = bcs::to_bytes(&(epoch, block_ref)).map_err(SharedError::SerializationFailure)?;
        let input = QuadraticForm::hash_to_group_with_default_parameters(&seed, &DISCRIMINANT_3072)
            .map_err(|e| SharedError::FailedVDF(e.to_string()))?;

        let entropy: QuadraticForm =
            bcs::from_bytes(&tx_entropy.0).map_err(SharedError::MalformedType)?;

        let proof: QuadraticForm =
            bcs::from_bytes(&tx_entropy_proof.0).map_err(SharedError::MalformedType)?;

        self.vdf
            .verify(&input, &entropy, &proof)
            .map_err(|e| SharedError::FailedVDF(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use crate::{authority_committee::AuthorityIndex, digest::Digest};

    use super::*;

    #[test]
    fn test_entropy_vdf_e2e() {
        // Initialize VDF with same parameters as the reference test
        let iterations: EntropyIterations = 1;
        let vdf = EntropyVDF::new(iterations);

        // Create test epoch and block reference
        let epoch: Epoch = 1;
        let block_ref = BlockRef::new(
            1,
            AuthorityIndex::new_for_test(1),
            Digest::new_from_bytes(Bytes::from("digest")),
        );

        // Generate entropy and proof
        let (entropy, proof) = vdf.get_entropy(epoch, block_ref).unwrap();

        // Verify the generated entropy
        assert!(vdf
            .verify_entropy(epoch, block_ref, &entropy, &proof)
            .is_ok());

        // Negative test - verify with wrong epoch should fail
        let wrong_epoch: Epoch = 2;
        assert!(vdf
            .verify_entropy(wrong_epoch, block_ref, &entropy, &proof)
            .is_err());

        // Negative test - verify with wrong block_ref should fail
        let wrong_block_ref = BlockRef::new(
            2,
            AuthorityIndex::new_for_test(2),
            Digest::new_from_bytes(Bytes::from("wrong digest")),
        );
        assert!(vdf
            .verify_entropy(epoch, wrong_block_ref, &entropy, &proof)
            .is_err());

        // Negative test - verify with wrong entropy should fail
        let wrong_entropy = BlockEntropyOutput(Bytes::from(vec![1, 2, 3, 4]));
        assert!(vdf
            .verify_entropy(epoch, block_ref, &wrong_entropy, &proof)
            .is_err());
    }
}
