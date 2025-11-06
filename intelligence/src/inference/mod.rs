use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use types::committee::Epoch;
use types::evaluation::{EmbeddingDigest, ProbeSet};
use types::metadata::{DownloadMetadata, ObjectPath};
pub mod core_processor;
pub mod messaging;
pub mod module;

#[enum_dispatch]
pub(crate) trait InferenceInputAPI {
    fn epoch(&self) -> Epoch;
    fn object_path(&self) -> &ObjectPath;
    fn download_metadata(&self) -> &DownloadMetadata;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct InferenceInputV1 {
    epoch: Epoch,
    object_path: ObjectPath,
    download_metadata: DownloadMetadata,
}

impl InferenceInputV1 {
    pub fn new(epoch: Epoch, object_path: ObjectPath, download_metadata: DownloadMetadata) -> Self {
        Self {
            epoch,
            object_path,
            download_metadata,
        }
    }
}

impl InferenceInputAPI for InferenceInputV1 {
    fn epoch(&self) -> Epoch {
        self.epoch
    }
    fn object_path(&self) -> &ObjectPath {
        &self.object_path
    }
    fn download_metadata(&self) -> &DownloadMetadata {
        &self.download_metadata
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(InferenceInputAPI)]
pub enum InferenceInput {
    V1(InferenceInputV1),
}

#[enum_dispatch]
pub trait InferenceOutputAPI {
    fn download_metadata(&self) -> &DownloadMetadata;
    fn probe_set(&self) -> &ProbeSet;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InferenceOutputV1 {
    download_metadata: DownloadMetadata,
    probe_set: ProbeSet,
}

impl InferenceOutputV1 {
    pub fn new(download_metadata: DownloadMetadata, probe_set: ProbeSet) -> Self {
        Self {
            download_metadata,
            probe_set,
        }
    }
}
impl InferenceOutputAPI for InferenceOutputV1 {
    fn download_metadata(&self) -> &DownloadMetadata {
        &self.download_metadata
    }
    fn probe_set(&self) -> &ProbeSet {
        &self.probe_set
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(InferenceOutputAPI)]
pub enum InferenceOutput {
    V1(InferenceOutputV1),
}
