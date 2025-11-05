use crate::inference::{InferenceClient, InferenceInput, InferenceOutput};
use async_trait::async_trait;
use object_store::ObjectStore;
use std::{sync::Arc, time::Duration};
use types::error::IntelligenceResult;

pub struct MockInferenceClient<S: ObjectStore> {
    storage: Arc<S>,
}

impl<S: ObjectStore> MockInferenceClient<S> {
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl<S: ObjectStore> InferenceClient for MockInferenceClient<S> {
    async fn call(
        &self,
        input: InferenceInput,
        timeout: Duration,
    ) -> IntelligenceResult<InferenceOutput> {
        unimplemented!();
        // Ok(InferenceOutput::V1(InferenceOutputV1 {
        //     tensors: Metadata::V1(MetadataV1::new(
        //         Checksum::new_from_bytes(b"mock"),
        //         b"mock".len(),
        //     )),
        //     probe_set: ProbeSet::V1(ProbeSetV1::new(vec![])),
        // }))
    }
}
