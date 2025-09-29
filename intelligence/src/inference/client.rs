use async_trait::async_trait;
use objects::storage::ObjectStorage;
use std::{sync::Arc, time::Duration};
use types::evaluation::{ProbeSet, ProbeSetV1};
use types::{
    checksum::Checksum,
    error::InferenceResult,
    metadata::{Metadata, MetadataV1},
};

use super::{InferenceInput, InferenceOutput, InferenceOutputV1};

#[async_trait]
pub trait InferenceClient: Send + Sync + Sized + 'static {
    async fn call(
        &self,
        input: InferenceInput,
        timeout: Duration,
    ) -> InferenceResult<InferenceOutput>;
}

pub struct MockInferenceClient<S: ObjectStorage> {
    storage: Arc<S>,
}

impl<S: ObjectStorage> MockInferenceClient<S> {
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl<S: ObjectStorage> InferenceClient for MockInferenceClient<S> {
    async fn call(
        &self,
        input: InferenceInput,
        timeout: Duration,
    ) -> InferenceResult<InferenceOutput> {
        Ok(InferenceOutput::V1(InferenceOutputV1 {
            embeddings: Metadata::V1(MetadataV1::new(
                Checksum::new_from_bytes(b"mock"),
                b"mock".len(),
            )),
            probe_set: ProbeSet::V1(ProbeSetV1::new(vec![])),
        }))
    }
}
