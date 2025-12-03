use fastcrypto::{
    hash::HashFunction,
    traits::{Signer, VerifyingKey},
};
use serde::{Deserialize, Serialize};

use crate::{
    committee::NetworkMetadata,
    crypto::{
        AuthorityKeyPair, AuthorityPublicKey, AuthorityPublicKeyBytes, AuthoritySignature,
        DefaultHash as DefaultHashFunction, DIGEST_LENGTH,
    },
    error::{ConsensusError, ConsensusResult},
    intent::{Intent, IntentMessage, IntentScope},
};

/// Type for next epoch's validator set
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValidatorSet(pub Vec<(AuthorityPublicKeyBytes, u64, NetworkMetadata)>);

/// Digest of validator set, used for signing
#[derive(Serialize, Deserialize)]
pub struct ValidatorSetDigest([u8; DIGEST_LENGTH]);

impl ValidatorSet {
    pub fn compute_digest(&self) -> ConsensusResult<ValidatorSetDigest> {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(bcs::to_bytes(self).map_err(ConsensusError::SerializationFailure)?);
        Ok(ValidatorSetDigest(hasher.finalize().into()))
    }

    pub fn sign(&self, keypair: &AuthorityKeyPair) -> ConsensusResult<AuthoritySignature> {
        let digest = self.compute_digest()?;
        let message = bcs::to_bytes(&to_validator_set_intent(digest))
            .map_err(ConsensusError::SerializationFailure)?;
        Ok(keypair.sign(&message))
    }

    pub fn verify_signature(
        &self,
        signature: &AuthoritySignature,
        public_key: &AuthorityPublicKey,
    ) -> ConsensusResult<()> {
        let digest = self.compute_digest()?;
        let message = bcs::to_bytes(&to_validator_set_intent(digest))
            .map_err(ConsensusError::SerializationFailure)?;
        public_key
            .verify(&message, signature)
            .map_err(ConsensusError::SignatureVerificationFailure)
    }
}

/// Wrap a ValidatorSetDigest in the intent message
pub fn to_validator_set_intent(digest: ValidatorSetDigest) -> IntentMessage<ValidatorSetDigest> {
    IntentMessage::new(Intent::soma_app(IntentScope::ValidatorSet), digest)
}
