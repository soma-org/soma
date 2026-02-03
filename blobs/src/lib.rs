use object_store::path::Path;
use serde::{Deserialize, Serialize};
use std::fmt;
use types::{checksum::Checksum, committee::Epoch};

// pub mod services;
pub mod buffer;
pub mod downloader;
pub mod readers;

/// Cloud providers typically require a minimum multipart part size except for the last part
pub(crate) const MIN_PART_SIZE: u64 = 5 * 1024 * 1024;
/// Cloud providers typically have a max multipart part size
pub(crate) const MAX_PART_SIZE: u64 = 5 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum BlobPath {
    Data(Epoch, Checksum),
    Weights(Epoch, Checksum),
}

impl BlobPath {
    pub fn path(&self) -> Path {
        match self {
            Self::Data(epoch, checksum) => {
                Path::from(format!("epochs/{}/data/{}", epoch, checksum))
            }
            Self::Weights(epoch, checksum) => {
                Path::from(format!("epochs/{}/weights/{}", epoch, checksum))
            }
        }
    }
}

impl fmt::Display for BlobPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.path().as_ref())
    }
}
