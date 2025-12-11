use std::sync::Arc;

use async_trait::async_trait;
use enum_dispatch::enum_dispatch;
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use types::committee::Epoch;
use types::error::{InferenceError, InferenceResult};
use types::evaluation::EmbeddingDigest;
use types::metadata::{Metadata, MetadataV1, ObjectPath};
use types::shard_crypto::keys::EncoderPublicKey;

#[enum_dispatch]
pub(crate) trait ModuleInputAPI {
    fn epoch(&self) -> Epoch;
    fn input_object_path(&self) -> &ObjectPath;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModuleInputV1 {
    epoch: Epoch,
    input_object_path: ObjectPath,
}

impl ModuleInputV1 {
    pub fn new(epoch: Epoch, input_object_path: ObjectPath) -> Self {
        Self {
            epoch,
            input_object_path,
        }
    }
}

impl ModuleInputAPI for ModuleInputV1 {
    fn epoch(&self) -> Epoch {
        self.epoch
    }
    fn input_object_path(&self) -> &ObjectPath {
        &self.input_object_path
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ModuleInputAPI)]
pub enum ModuleInput {
    V1(ModuleInputV1),
}

#[enum_dispatch]
pub trait ModuleOutputAPI {
    fn object_path(&self) -> &ObjectPath;
    fn metadata(&self) -> &Metadata;
    fn probe_encoder(&self) -> &EncoderPublicKey;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModuleOutputV1 {
    object_path: ObjectPath,
    metadata: Metadata,
    probe_encoder: EncoderPublicKey,
}

impl ModuleOutputV1 {
    pub fn new(
        object_path: ObjectPath,
        metadata: Metadata,
        probe_encoder: EncoderPublicKey,
    ) -> Self {
        Self {
            object_path,
            metadata,
            probe_encoder,
        }
    }
}
impl ModuleOutputAPI for ModuleOutputV1 {
    fn object_path(&self) -> &ObjectPath {
        &self.object_path
    }
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
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
pub trait ModuleClient: Send + Sync + Sized + 'static {
    async fn call(
        &self,
        input: ModuleInput,
        storage: Arc<dyn ObjectStore>,
    ) -> InferenceResult<ModuleOutput>;
}

pub struct MockModule {
    probe_encoder: EncoderPublicKey,
}

impl MockModule {
    pub fn new(probe_encoder: EncoderPublicKey) -> Self {
        Self { probe_encoder }
    }
}

#[async_trait]
impl ModuleClient for MockModule {
    async fn call(
        &self,
        input: ModuleInput,
        storage: Arc<dyn ObjectStore>,
    ) -> InferenceResult<ModuleOutput> {
        if let ObjectPath::Inputs(epoch, shard_digest, checksum) = input.input_object_path().clone()
        {
            let object_path = ObjectPath::Embeddings(epoch, shard_digest, checksum);
            storage
                .copy(&input.input_object_path().path(), &object_path.path())
                .await
                .map_err(InferenceError::ObjectStoreError)?;
            let head = storage
                .head(&object_path.path())
                .await
                .map_err(InferenceError::ObjectStoreError)?;
            let metadata = Metadata::V1(MetadataV1::new(object_path.checksum(), head.size));
            Ok(ModuleOutput::V1(ModuleOutputV1::new(
                object_path,
                metadata,
                self.probe_encoder.clone(),
            )))
        } else {
            Err(InferenceError::ValidationError(
                "incorrect object path type".to_string(),
            ))
        }
    }
}
