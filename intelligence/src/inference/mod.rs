use async_trait::async_trait;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use types::committee::Epoch;
use types::error::IntelligenceResult;
use types::evaluation::ProbeSet;
use types::metadata::Metadata;
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
    fn metadata(&self) -> Metadata;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct InferenceInputV1 {
    epoch: Epoch,
    metadata: Metadata,
}

impl InferenceInputV1 {
    pub fn new(epoch: Epoch, metadata: Metadata) -> Self {
        Self { epoch, metadata }
    }
}

impl InferenceInputAPI for InferenceInputV1 {
    fn epoch(&self) -> Epoch {
        self.epoch.clone()
    }
    fn metadata(&self) -> Metadata {
        self.metadata.clone()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(InferenceInputAPI)]
pub enum InferenceInput {
    V1(InferenceInputV1),
}

#[enum_dispatch]
pub trait InferenceOutputAPI {
    fn metadata(&self) -> Metadata;
    fn probe_set(&self) -> ProbeSet;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InferenceOutputV1 {
    metadata: Metadata,
    probe_set: ProbeSet,
}

impl InferenceOutputV1 {
    pub fn new(metadata: Metadata, probe_set: ProbeSet) -> Self {
        Self {
            metadata,
            probe_set,
        }
    }
}
impl InferenceOutputAPI for InferenceOutputV1 {
    fn metadata(&self) -> Metadata {
        self.metadata.clone()
    }
    fn probe_set(&self) -> ProbeSet {
        self.probe_set.clone()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(InferenceOutputAPI)]
pub enum InferenceOutput {
    V1(InferenceOutputV1),
}
