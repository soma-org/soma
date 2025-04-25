use std::convert::Infallible;

use bytes::Bytes;
use quick_cache::unsync::Cache;
use serde::{Deserialize, Serialize};
use vdf::{
    class_group::{discriminant::DISCRIMINANT_3072, QuadraticForm},
    vdf::{wesolowski::DefaultVDF, VDF},
};

use crate::{
    block::{BlockRef, Epoch},
    error::{SharedError, SharedResult},
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BlockEntropy(Bytes);

impl BlockEntropy {
    pub fn new(bytes: Bytes) -> Self {
        Self(bytes)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BlockEntropyProof(Bytes);
impl BlockEntropyProof {
    pub fn new(bytes: Bytes) -> Self {
        Self(bytes)
    }
}
type Iterations = u64;

pub trait EntropyAPI: Send + Sync + Sized + 'static {
    fn get_entropy(
        &mut self,
        epoch: Epoch,
        block_ref: BlockRef,
        iterations: Iterations,
    ) -> SharedResult<(BlockEntropy, BlockEntropyProof)>;

    fn verify_entropy(
        &mut self,
        epoch: Epoch,
        block_ref: BlockRef,
        block_entropy: &BlockEntropy,
        block_entropy_proof: &BlockEntropyProof,
        iterations: Iterations,
    ) -> SharedResult<()>;
}

pub struct EntropyVDF {
    vdfs: Cache<Iterations, DefaultVDF>,
}

impl EntropyVDF {
    pub fn new(cache_capacity: usize) -> Self {
        Self {
            vdfs: Cache::new(cache_capacity),
        }
    }
}

impl EntropyAPI for EntropyVDF {
    fn get_entropy(
        &mut self,
        epoch: Epoch,
        block_ref: BlockRef,
        iterations: Iterations,
    ) -> SharedResult<(BlockEntropy, BlockEntropyProof)> {
        let seed = bcs::to_bytes(&(epoch, block_ref)).map_err(SharedError::SerializationFailure)?;
        let input = QuadraticForm::hash_to_group_with_default_parameters(&seed, &DISCRIMINANT_3072)
            .map_err(|e| SharedError::FailedVDF(e.to_string()))?;

        let vdf = self
            .vdfs
            .get_or_insert_with(&iterations, || -> Result<DefaultVDF, Infallible> {
                Ok(DefaultVDF::new(DISCRIMINANT_3072.clone(), iterations))
            })
            .unwrap()
            .unwrap();

        let (output, proof) = vdf
            .evaluate(&input)
            .map_err(|e| SharedError::FailedVDF(e.to_string()))?;

        let entropy_bytes = bcs::to_bytes(&output).map_err(SharedError::SerializationFailure)?;

        let entropy = BlockEntropy(Bytes::copy_from_slice(&entropy_bytes));
        let proof_bytes = bcs::to_bytes(&proof).map_err(SharedError::SerializationFailure)?;

        let proof = BlockEntropyProof(Bytes::copy_from_slice(&proof_bytes));
        Ok((entropy, proof))
    }

    fn verify_entropy(
        &mut self,
        epoch: Epoch,
        block_ref: BlockRef,
        tx_entropy: &BlockEntropy,
        tx_entropy_proof: &BlockEntropyProof,
        iterations: Iterations,
    ) -> SharedResult<()> {
        let seed = bcs::to_bytes(&(epoch, block_ref)).map_err(SharedError::SerializationFailure)?;
        let input = QuadraticForm::hash_to_group_with_default_parameters(&seed, &DISCRIMINANT_3072)
            .map_err(|e| SharedError::FailedVDF(e.to_string()))?;

        let entropy: QuadraticForm =
            bcs::from_bytes(&tx_entropy.0).map_err(SharedError::MalformedType)?;

        let proof: QuadraticForm =
            bcs::from_bytes(&tx_entropy_proof.0).map_err(SharedError::MalformedType)?;

        let vdf = self
            .vdfs
            .get_or_insert_with(&iterations, || -> Result<DefaultVDF, Infallible> {
                Ok(DefaultVDF::new(DISCRIMINANT_3072.clone(), iterations))
            })
            .unwrap()
            .unwrap();

        vdf.verify(&input, &entropy, &proof)
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
        let cache_capacity = 2_usize;
        let iterations: Iterations = 1;
        let vdf = EntropyVDF::new(cache_capacity);

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
        let wrong_entropy = BlockEntropy(Bytes::from(vec![1, 2, 3, 4]));
        assert!(vdf
            .verify_entropy(epoch, block_ref, &wrong_entropy, &proof)
            .is_err());
    }
}
