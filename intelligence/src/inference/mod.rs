use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use types::committee::Epoch;
use types::metadata::{DownloadMetadata, ObjectPath};
use types::shard::Shard;
use types::shard_crypto::digest::Digest;
use types::shard_crypto::keys::EncoderPublicKey;
pub mod core_processor;
pub mod messaging;
pub mod module;

#[enum_dispatch]
pub(crate) trait InferenceInputAPI {
    fn epoch(&self) -> Epoch;
    fn shard_digest(&self) -> &Digest<Shard>;
    fn input_download_metadata(&self) -> &DownloadMetadata;
    fn input_object_path(&self) -> &ObjectPath;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct InferenceInputV1 {
    epoch: Epoch,
    shard_digest: Digest<Shard>,
    input_download_metadata: DownloadMetadata,
    input_object_path: ObjectPath,
}

impl InferenceInputV1 {
    pub fn new(
        epoch: Epoch,
        shard_digest: Digest<Shard>,
        input_download_metadata: DownloadMetadata,
        input_object_path: ObjectPath,
    ) -> Self {
        Self {
            epoch,
            shard_digest,
            input_download_metadata,
            input_object_path,
        }
    }
}

impl InferenceInputAPI for InferenceInputV1 {
    fn epoch(&self) -> Epoch {
        self.epoch
    }
    fn shard_digest(&self) -> &Digest<Shard> {
        &self.shard_digest
    }
    fn input_download_metadata(&self) -> &DownloadMetadata {
        &self.input_download_metadata
    }
    fn input_object_path(&self) -> &ObjectPath {
        &self.input_object_path
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(InferenceInputAPI)]
pub enum InferenceInput {
    V1(InferenceInputV1),
}

#[enum_dispatch]
pub trait InferenceOutputAPI {
    fn output_download_metadata(&self) -> &DownloadMetadata;
    fn output_object_path(&self) -> &ObjectPath;
    fn probe_encoder(&self) -> &EncoderPublicKey;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InferenceOutputV1 {
    output_download_metadata: DownloadMetadata,
    output_object_path: ObjectPath,
    probe_encoder: EncoderPublicKey,
}

impl InferenceOutputV1 {
    pub fn new(
        output_download_metadata: DownloadMetadata,
        output_object_path: ObjectPath,
        probe_encoder: EncoderPublicKey,
    ) -> Self {
        Self {
            output_download_metadata,
            output_object_path,
            probe_encoder,
        }
    }
}
impl InferenceOutputAPI for InferenceOutputV1 {
    fn output_download_metadata(&self) -> &DownloadMetadata {
        &self.output_download_metadata
    }
    fn output_object_path(&self) -> &ObjectPath {
        &self.output_object_path
    }
    fn probe_encoder(&self) -> &EncoderPublicKey {
        &self.probe_encoder
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(InferenceOutputAPI)]
pub enum InferenceOutput {
    V1(InferenceOutputV1),
}
