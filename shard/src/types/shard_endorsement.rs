use crate::{
    crypto::{
        keys::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey},
        DefaultHashFunction, DIGEST_LENGTH,
    },
    error::{ShardError, ShardResult},
};

use std::ops::Deref;
use std::sync::Arc;

use crate::types::manifest::Manifest;
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};

use crate::types::{
    scope::{Scope, ScopedMessage},
    score::Score,
    shard::ShardRef,
};

use std::{
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardEndorsementAPI)]
pub enum ShardEndorsement {
    V1(ShardEndorsementV1),
}

#[enum_dispatch]
pub trait ShardEndorsementAPI {
    fn scores(&self) -> &[Score];
    fn manifest(&self) -> &Manifest;
    fn shard_ref(&self) -> &ShardRef;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardEndorsementV1 {
    scores: Vec<Score>,
    manifest: Manifest,
    shard_ref: ShardRef,
}

impl ShardEndorsementV1 {
    pub(crate) fn new(scores: Vec<Score>, manifest: Manifest, shard_ref: ShardRef) -> Self {
        Self {
            scores,
            manifest,
            shard_ref,
        }
    }
}

impl ShardEndorsementAPI for ShardEndorsementV1 {
    fn scores(&self) -> &[Score] {
        &self.scores
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
}

macros::generate_signed_type!(ShardEndorsement);
macros::generate_digest_type!(SignedShardEndorsement);
macros::generate_verified_type!(SignedShardEndorsement);
