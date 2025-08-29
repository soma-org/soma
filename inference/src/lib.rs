use enum_dispatch::enum_dispatch;
use evaluation::ProbeSet;
use serde::{Deserialize, Serialize};
use shared::metadata::Metadata;
pub mod client;

#[enum_dispatch]
pub(crate) trait InferenceInputAPI {
    fn input(&self) -> Metadata;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct InferenceInputV1 {
    input: Metadata,
}

impl InferenceInputV1 {
    pub fn new(input: Metadata) -> Self {
        Self { input }
    }
}

impl InferenceInputAPI for InferenceInputV1 {
    fn input(&self) -> Metadata {
        self.input.clone()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(InferenceInputAPI)]
pub enum InferenceInput {
    V1(InferenceInputV1),
}

#[enum_dispatch]
pub trait InferenceOutputAPI {
    fn embeddings(&self) -> Metadata;
    fn probe_set(&self) -> ProbeSet;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InferenceOutputV1 {
    embeddings: Metadata,
    probe_set: ProbeSet,
}

impl InferenceOutputAPI for InferenceOutputV1 {
    fn embeddings(&self) -> Metadata {
        self.embeddings.clone()
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
