use async_trait::async_trait;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use types::committee::Epoch;
use types::error::IntelligenceResult;
use types::evaluation::{EmbeddingDigest, ProbeSet};
use types::metadata::{DownloadMetadata, Metadata, ObjectPath};
use types::shard::Shard;
pub mod json_client;
pub mod mock_client;

#[async_trait]
pub trait InferenceClient: Send + Sync + Sized + 'static {
    async fn call(
        &self,
        input: InferenceInput,
        timeout: Duration,
    ) -> IntelligenceResult<InferenceOutput>;
}

#[enum_dispatch]
pub(crate) trait InferenceInputAPI {
    fn epoch(&self) -> Epoch;
    fn download_metadata(&self) -> &DownloadMetadata;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct InferenceInputV1 {
    epoch: Epoch,
    download_metadata: DownloadMetadata,
}

impl InferenceInputV1 {
    pub fn new(epoch: Epoch, download_metadata: DownloadMetadata) -> Self {
        Self {
            epoch,
            download_metadata,
        }
    }
}

impl InferenceInputAPI for InferenceInputV1 {
    fn epoch(&self) -> Epoch {
        self.epoch
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
    fn summary_digest(&self) -> &EmbeddingDigest;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InferenceOutputV1 {
    download_metadata: DownloadMetadata,
    probe_set: ProbeSet,
    summary_digest: EmbeddingDigest,
}

impl InferenceOutputV1 {
    pub fn new(
        download_metadata: DownloadMetadata,
        probe_set: ProbeSet,
        summary_digest: EmbeddingDigest,
    ) -> Self {
        Self {
            download_metadata,
            probe_set,
            summary_digest,
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
    fn summary_digest(&self) -> &EmbeddingDigest {
        &self.summary_digest
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(InferenceOutputAPI)]
pub enum InferenceOutput {
    V1(InferenceOutputV1),
}
