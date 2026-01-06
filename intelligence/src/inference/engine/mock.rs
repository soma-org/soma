use std::{ time::Duration};
use async_trait::async_trait;
use fastcrypto::hash::HashFunction;
use objects::stores::{PersistentStore, memory::PersistentInMemoryStore};
use types::{checksum::Checksum, crypto::DefaultHash, error::InferenceResult, metadata::{ Metadata, MetadataV1, ObjectPath}, shard_crypto::keys::EncoderPublicKey};
use crate::inference::{InferenceInput, InferenceInputAPI, InferenceOutput, InferenceOutputV1, engine::InferenceEngineAPI};
use object_store::ObjectStore;

pub struct MockInferenceEngine {
    probe_encoder: EncoderPublicKey,
    persistent_store: PersistentInMemoryStore,
}

impl MockInferenceEngine {
    pub fn new(probe_encoder: EncoderPublicKey, persistent_store: PersistentInMemoryStore) -> Self {
        Self {
            probe_encoder,
            persistent_store,
        }
    }
}

#[async_trait]
impl InferenceEngineAPI for MockInferenceEngine {
    async fn call(&self, input: InferenceInput, timeout: Duration) -> InferenceResult<InferenceOutput> {
        let mock_data = vec![1u8; 10_000];
        let mut hasher = DefaultHash::new();
        hasher.update(&mock_data);
        let checksum = Checksum::new_from_hash(hasher.finalize().into());
        let metadata = Metadata::V1(MetadataV1::new(checksum, mock_data.len()));

        let output_object_path = ObjectPath::Embeddings(input.epoch(), input.shard_digest().clone(), checksum);
        self.persistent_store.object_store()
            .put(&output_object_path.path(), mock_data.into())
            .await
            .unwrap();

        let output_download_metadata = self.persistent_store.download_metadata(output_object_path.clone(), metadata).await.unwrap();

        Ok(InferenceOutput::V1(InferenceOutputV1::new(
            input.input_download_metadata().clone(),
            input.input_object_path().clone(),
            output_download_metadata,
            output_object_path,
            self.probe_encoder.clone()
        )))

    }
}