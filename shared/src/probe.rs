use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::metadata::Metadata;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Probe {
    weights: Bytes,
}

impl Probe {
    pub fn new(weights: Bytes) -> Self {
        Self { weights }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProbeMetadata {
    metadata: Metadata,
}

impl ProbeMetadata {
    pub fn new_for_test(bytes: &[u8]) -> Self {
        ProbeMetadata {
            metadata: Metadata::new_for_test(bytes),
        }
    }
}

impl std::ops::Deref for ProbeMetadata {
    type Target = Metadata;
    fn deref(&self) -> &Self::Target {
        &self.metadata
    }
}
