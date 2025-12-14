pub mod json_client;
use async_trait::async_trait;
use enum_dispatch::enum_dispatch;
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use types::committee::Epoch;
use types::error::{InferenceError, InferenceResult};
use types::metadata::ObjectPath;
use types::shard_crypto::keys::EncoderPublicKey;

#[enum_dispatch]
pub(crate) trait ModuleInputAPI {
    fn epoch(&self) -> Epoch;
    fn input_path(&self) -> &ObjectPath;
    fn output_path(&self) -> &ObjectPath;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModuleInputV1 {
    epoch: Epoch,
    input_path: ObjectPath,
    output_path: ObjectPath,
}

impl ModuleInputV1 {
    pub fn new(epoch: Epoch, input_path: ObjectPath, output_path: ObjectPath) -> Self {
        Self {
            epoch,
            input_path,
            output_path,
        }
    }
}

impl ModuleInputAPI for ModuleInputV1 {
    fn epoch(&self) -> Epoch {
        self.epoch
    }
    fn input_path(&self) -> &ObjectPath {
        &self.input_path
    }
    fn output_path(&self) -> &ObjectPath {
        &self.output_path
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ModuleInputAPI)]
pub enum ModuleInput {
    V1(ModuleInputV1),
}

#[enum_dispatch]
pub trait ModuleOutputAPI {
    fn probe_encoder(&self) -> &EncoderPublicKey;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModuleOutputV1 {
    probe_encoder: EncoderPublicKey,
}

impl ModuleOutputV1 {
    pub fn new(probe_encoder: EncoderPublicKey) -> Self {
        Self { probe_encoder }
    }
}
impl ModuleOutputAPI for ModuleOutputV1 {
    fn probe_encoder(&self) -> &EncoderPublicKey {
        &self.probe_encoder
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ModuleOutputAPI)]
pub enum ModuleOutput {
    V1(ModuleOutputV1),
}

#[async_trait]
pub trait ModuleClient<S>: Send + Sync + Sized + 'static
where
    S: ObjectStore + 'static,
{
    async fn call(&self, input: ModuleInput, timeout: Duration) -> InferenceResult<ModuleOutput>;
}

pub struct MockModule<S: ObjectStore> {
    probe_encoder: EncoderPublicKey,
    storage: Arc<S>,
}

impl<S: ObjectStore> MockModule<S> {
    pub fn new(probe_encoder: EncoderPublicKey, storage: Arc<S>) -> Self {
        Self {
            probe_encoder,
            storage,
        }
    }
}

#[async_trait]
impl<S: ObjectStore> ModuleClient<S> for MockModule<S> {
    async fn call(&self, input: ModuleInput, timeout: Duration) -> InferenceResult<ModuleOutput> {
        if let ObjectPath::Inputs(epoch, shard_digest, checksum) = input.input_path().clone() {
            self.storage
                .copy(&input.input_path().path(), &input.output_path().path())
                .await
                .map_err(InferenceError::ObjectStoreError)?;
            Ok(ModuleOutput::V1(ModuleOutputV1::new(
                self.probe_encoder.clone(),
            )))
        } else {
            Err(InferenceError::ValidationError(
                "incorrect object path type".to_string(),
            ))
        }
    }
}
