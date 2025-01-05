use crate::types::checksum::Checksum;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use super::modality::Modality;

type SizeInBytes = usize;
type SizeInElements = usize;

#[enum_dispatch]
pub(crate) trait DataAPI {
    fn modality(&self) -> Modality;
    fn compression(&self) -> Compression;
    fn encryption(&self) -> Option<Encryption>;
    fn checksum(&self) -> Checksum;
    fn shape(&self) -> &[SizeInElements];
    fn download_size(&self) -> SizeInBytes;
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
#[enum_dispatch(DataAPI)]
pub enum Data {
    V1(DataV1),
}

impl Data {
    /// new constructs a new transaction certificate
    pub(crate) fn new_v1(
        modality: Modality,
        compression: Compression,
        encryption: Option<Encryption>,
        checksum: Checksum,
        shape: Vec<SizeInElements>,
        download_size: SizeInBytes,
    ) -> Data {
        Data::V1(DataV1 {
            modality,
            compression,
            encryption,
            checksum,
            shape,
            download_size,
        })
    }
}
/// Version 1 of the manifest. Adding versioning here because while not currently sent over the wire,
/// it is reasonable to assume that it may be.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct DataV1 {
    modality: Modality,
    compression: Compression,
    encryption: Option<Encryption>,
    checksum: Checksum,
    shape: Vec<SizeInElements>,
    download_size: SizeInBytes,
}

impl DataAPI for DataV1 {
    fn modality(&self) -> Modality {
        self.modality
    }
    fn compression(&self) -> Compression {
        self.compression
    }
    fn encryption(&self) -> Option<Encryption> {
        self.encryption
    }
    fn checksum(&self) -> Checksum {
        self.checksum
    }
    fn shape(&self) -> &[SizeInElements] {
        &self.shape
    }
    fn download_size(&self) -> SizeInBytes {
        self.download_size
    }
}
