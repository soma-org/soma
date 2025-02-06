use bytes::Bytes;
use fastcrypto_vdf::{
    class_group::{discriminant::DISCRIMINANT_3072, QuadraticForm},
    vdf::{wesolowski::DefaultVDF, VDF},
};

use crate::{
    block::BlockHeader,
    digest::Digest,
    error::{SharedError, SharedResult},
    signed::Signed,
    transaction::SignedTransaction,
};

pub struct TransactionEntropy(Bytes);
pub struct TransactionEntropyProof(Bytes);

type EntropyIterations = u64;

trait EntropyAPI {
    fn get_entropy(
        &self,
        header_digest: Digest<Signed<BlockHeader>>,
        tx_digest: Digest<SignedTransaction>,
    ) -> SharedResult<(TransactionEntropy, TransactionEntropyProof)>;

    fn verify_entropy(
        &self,
        header_digest: Digest<Signed<BlockHeader>>,
        tx_digest: Digest<SignedTransaction>,
        tx_entropy: TransactionEntropy,
        tx_entropy_proof: TransactionEntropyProof,
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
        header_digest: Digest<Signed<BlockHeader>>,
        tx_digest: Digest<SignedTransaction>,
    ) -> SharedResult<(TransactionEntropy, TransactionEntropyProof)> {
        let mut seed = Vec::with_capacity(64);
        seed.extend_from_slice(header_digest.as_ref());
        seed.extend_from_slice(tx_digest.as_ref());

        let input = QuadraticForm::hash_to_group_with_default_parameters(&seed, &DISCRIMINANT_3072)
            .map_err(|e| SharedError::FailedVDF(e.to_string()))?;

        let (output, proof) = self
            .vdf
            .evaluate(&input)
            .map_err(|e| SharedError::FailedVDF(e.to_string()))?;

        let entropy_bytes =
            bcs::to_bytes(&output).map_err(|e| SharedError::SerializationFailure(e))?;

        let entropy = TransactionEntropy(Bytes::copy_from_slice(&entropy_bytes));
        let proof_bytes =
            bcs::to_bytes(&proof).map_err(|e| SharedError::SerializationFailure(e))?;

        let proof = TransactionEntropyProof(Bytes::copy_from_slice(&proof_bytes));
        Ok((entropy, proof))
    }

    fn verify_entropy(
        &self,
        header_digest: Digest<Signed<BlockHeader>>,
        tx_digest: Digest<SignedTransaction>,
        tx_entropy: TransactionEntropy,
        tx_entropy_proof: TransactionEntropyProof,
    ) -> SharedResult<()> {
        let mut seed = Vec::with_capacity(64);
        seed.extend_from_slice(header_digest.as_ref());
        seed.extend_from_slice(tx_digest.as_ref());

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
