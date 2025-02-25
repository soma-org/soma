use super::encoder_committee::{EncoderCommittee, EncoderIndex};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::{EncoderAggregateSignature, EncoderPublicKey},
    digest::Digest,
    error::{SharedError, SharedResult},
    network_committee::NetworkingIndex,
    scope::{Scope, ScopedMessage},
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(CertifiedAPI)]
pub enum Certified<T: Serialize + PartialEq + Eq> {
    V1(CertifiedV1<T>),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct CertifiedV1<T: Serialize + PartialEq + Eq> {
    inner: T,
    indices: Vec<EncoderIndex>,
    aggregate_signature: EncoderAggregateSignature,
}

impl<T: Serialize + PartialEq + Eq> Certified<T> {
    /// new constructs a new transaction certificate
    pub(crate) const fn new_v1(
        inner: T,
        indices: Vec<EncoderIndex>,
        aggregate_signature: EncoderAggregateSignature,
    ) -> Certified<T> {
        Certified::V1(CertifiedV1 {
            inner,
            indices,
            aggregate_signature,
        })
    }
}

#[enum_dispatch]
pub trait CertifiedAPI {
    fn indices(&self) -> Vec<EncoderIndex>;
    fn aggregate_signature(&self) -> &EncoderAggregateSignature;
    fn verify_quorum(&self, scope: Scope, committee: &EncoderCommittee) -> SharedResult<()>;
}

impl<T: Serialize + PartialEq + Eq> CertifiedAPI for CertifiedV1<T> {
    fn indices(&self) -> Vec<EncoderIndex> {
        self.indices.clone()
    }
    fn aggregate_signature(&self) -> &EncoderAggregateSignature {
        &self.aggregate_signature
    }

    fn verify_quorum(&self, scope: Scope, committee: &EncoderCommittee) -> SharedResult<()> {
        let unique_indices: std::collections::HashSet<_> = self.indices.iter().cloned().collect();

        // Get evaluation quorum threshold from committee
        let threshold = committee.evaluation_quorum_threshold();

        // Check if we have enough unique indices to meet quorum
        if unique_indices.len() < threshold as usize {
            return Err(SharedError::ValidationError(format!(
                "got: {} unique indices, needed: {}",
                unique_indices.len(),
                threshold
            )));
        }

        // Proceed with signature verification using unique indices only
        let inner_digest = Digest::new(&self.inner)?;
        let message = bcs::to_bytes(&ScopedMessage::new(scope, inner_digest))
            .map_err(SharedError::SerializationFailure)?;

        // Collect public keys only for unique indices
        let certifier_keys: Vec<EncoderPublicKey> = unique_indices
            .iter()
            .map(|index| committee.encoder(index.clone()).encoder_key.clone())
            .collect();

        let _ = self
            .aggregate_signature
            .verify(&certifier_keys, &message)
            .map_err(SharedError::SignatureVerificationFailure)?;

        Ok(())
    }
}

impl<T: Serialize + PartialEq + Eq> std::ops::Deref for Certified<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Certified::V1(cert) => &cert.inner,
        }
    }
}
