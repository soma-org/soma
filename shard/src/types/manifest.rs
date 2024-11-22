use crate::crypto::{DefaultHashFunction, DIGEST_LENGTH};
use crate::error::{ShardError, ShardResult};
use crate::types::checksum::Checksum;
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
    sync::Arc,
};
use url::Url;

use super::modality::Modality;

/// size of a chunk in bytes
type SizeInBytes = u64;
type SizeInElements = u64;

/// `ManifestAPI` describes the API for interacting with versioned Manifests
#[enum_dispatch]
pub(crate) trait ManifestAPI {
    fn modality(&self) -> Modality;
    fn compression(&self) -> Compression;
    fn encryption(&self) -> Option<Encryption>;
    fn batches(&self) -> &[Batch];
}

// TODO: make these enums better
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub enum Compression {
    ZSTD,
}
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub enum Encryption {
    Aes256Ctr64LE,
}

/// Versioned manifest type
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ManifestAPI)]
pub enum Manifest {
    V1(ManifestV1),
}

impl Manifest {
    /// new constructs a new transaction certificate
    fn new_v1(
        modality: Modality,
        compression: Compression,
        encryption: Option<Encryption>,
        batches: Vec<Batch>,
    ) -> Manifest {
        Manifest::V1(ManifestV1 {
            modality,
            compression,
            encryption,
            batches,
        })
    }
}
/// Version 1 of the manifest. Adding versioning here because while not currently sent over the wire,
/// it is reasonable to assume that it may be.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct ManifestV1 {
    modality: Modality,
    compression: Compression,
    encryption: Option<Encryption>,
    batches: Vec<Batch>,
}

impl ManifestAPI for ManifestV1 {
    fn modality(&self) -> Modality {
        self.modality
    }
    fn compression(&self) -> Compression {
        self.compression
    }
    fn encryption(&self) -> Option<Encryption> {
        self.encryption
    }
    fn batches(&self) -> &[Batch] {
        &self.batches
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BatchIndex(u32);

/// The API to interact with versioned chunks
#[enum_dispatch]
pub(crate) trait BatchAPI {
    /// Returns the signed transaction
    fn checksum(&self) -> Checksum;
    fn shape(&self) -> &[SizeInElements];
    fn download_size(&self) -> SizeInBytes;
    fn batch_index(&self) -> BatchIndex;
}

/// Version switch for the chunk versions
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(BatchAPI)]
pub(crate) enum Batch {
    V1(BatchV1),
}

/// First version of the chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchV1 {
    checksum: Checksum,
    shape: Vec<SizeInElements>,
    download_size: SizeInBytes,
    batch_index: BatchIndex,
}

impl BatchAPI for BatchV1 {
    fn checksum(&self) -> Checksum {
        self.checksum
    }
    fn shape(&self) -> &[SizeInElements] {
        &self.shape
    }
    fn download_size(&self) -> SizeInBytes {
        self.download_size
    }
    fn batch_index(&self) -> BatchIndex {
        self.batch_index.clone()
    }
}
