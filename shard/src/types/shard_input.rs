use crate::{
    crypto::{
        keys::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey},
        DefaultHashFunction, DIGEST_LENGTH,
    },
    error::{ShardError, ShardResult},
};

use std::ops::Deref;
use std::sync::Arc;

use crate::types::{
    manifest::Manifest, modality::Modality, shard::ShardSecret,
    transaction_certificate::TransactionCertificate,
};
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};

use super::scope::{Scope, ScopedMessage};
use std::{
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[enum_dispatch(ShardInputAPI)]
pub enum ShardInput {
    V1(ShardInputV1),
}

#[enum_dispatch]
pub trait ShardInputAPI {
    fn transaction_certificate(&self) -> &TransactionCertificate;
    fn shard_secret(&self) -> &ShardSecret;
    fn manifest(&self) -> &Manifest;
    fn modality(&self) -> &Modality;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ShardInputV1 {
    transaction_certificate: TransactionCertificate,
    shard_secret: ShardSecret,
    manifest: Manifest,
    modality: Modality,
}

impl ShardInputV1 {
    pub(crate) fn new(
        transaction_certificate: TransactionCertificate,
        shard_secret: ShardSecret,
        manifest: Manifest,
        modality: Modality,
    ) -> Self {
        Self {
            transaction_certificate,
            shard_secret,
            manifest,
            modality,
        }
    }
}

impl ShardInputAPI for ShardInputV1 {
    fn transaction_certificate(&self) -> &TransactionCertificate {
        &self.transaction_certificate
    }
    fn shard_secret(&self) -> &ShardSecret {
        &self.shard_secret
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn modality(&self) -> &Modality {
        &self.modality
    }
}

macros::generate_signed_type!(ShardInput);
macros::generate_digest_type!(SignedShardInput);
macros::generate_verified_type!(SignedShardInput);
