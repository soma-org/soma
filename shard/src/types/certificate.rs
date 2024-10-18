use super::encoder_committee::EncoderIndex;
use super::network_committee::NetworkingIndex;
use super::{block::SignedBlockHeader, transaction::SignedTransaction};
use crate::crypto::keys::ProtocolKeySignature;
use crate::crypto::{DefaultHashFunction, DIGEST_LENGTH};
use crate::error::{ShardError, ShardResult};
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardCertificateAPI)]
pub enum ShardCertificate<T> {
    V1(ShardCertificateV1<T>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardCertificateV1<T> {
    inner: T,
    //TODO: improve referencing to specific encoders
    // TODO figure out how to reference shards in a safer more specific way?
    indices: Vec<NetworkingIndex>,
    aggregate_signature: ProtocolKeySignature,
}

impl<T> ShardCertificate<T> {
    /// new constructs a new transaction certificate
    pub(crate) const fn new_v1(
        inner: T,
        indices: Vec<NetworkingIndex>,
        aggregate_signature: ProtocolKeySignature,
    ) -> ShardCertificateV1<T> {
        ShardCertificateV1 {
            inner,
            indices,
            aggregate_signature,
        }
    }
}

#[enum_dispatch]
trait ShardCertificateAPI {
    /// Returns the signed transaction
    fn signed_transaction(&self) -> &SignedTransaction;
    /// Returns the round
    fn signed_block_header(&self) -> &SignedBlockHeader;
    /// Returns the author authority index. Dependent on epoch.
    fn certificate_signature(&self) -> &ProtocolKeySignature;
}

impl<T> ShardCertificateAPI for ShardCertificateV1<T> {
    fn signed_transaction(&self) -> &SignedTransaction {
        &self.signed_transaction
    }
    fn signed_block_header(&self) -> &SignedBlockHeader {
        &self.signed_block_header
    }
    fn certificate_signature(&self) -> &ProtocolKeySignature {
        &self.certificate_signature
    }
}
