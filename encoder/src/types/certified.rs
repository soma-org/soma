use super::encoder_committee::EncoderIndex;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::{crypto::keys::EncoderAggregateSignature, network_committee::NetworkingIndex};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(CertifiedAPI)]
pub enum Certified<T> {
    V1(CertifiedV1<T>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CertifiedV1<T> {
    inner: T,
    indices: Vec<EncoderIndex>,
    aggregate_signature: EncoderAggregateSignature,
}

impl<T> Certified<T> {
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
trait CertifiedAPI {
    fn indices(&self) -> Vec<EncoderIndex>;
    fn aggregate_signature(&self) -> &EncoderAggregateSignature;
}

impl<T> CertifiedAPI for CertifiedV1<T> {
    fn indices(&self) -> Vec<EncoderIndex> {
        self.indices.clone()
    }
    fn aggregate_signature(&self) -> &EncoderAggregateSignature {
        &self.aggregate_signature
    }
}

impl<T> std::ops::Deref for Certified<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Certified::V1(cert) => &cert.inner,
        }
    }
}
