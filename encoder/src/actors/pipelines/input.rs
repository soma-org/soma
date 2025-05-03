use std::{collections::HashSet, sync::Arc, time::Duration};

use crate::{
    actors::{
        pipelines::broadcast::BroadcastAction,
        workers::{
            compression::{CompressionProcessor, CompressorInput},
            downloader::{Downloader, DownloaderInput},
            encryption::{EncryptionInput, EncryptionProcessor},
            model::ModelProcessor,
            storage::{StorageProcessor, StorageProcessorInput},
        },
        ActorHandle, ActorMessage, Processor,
    },
    compression::zstd_compressor::ZstdCompressor,
    core::{internal_broadcaster::Broadcaster, shard_tracker::ShardTracker},
    encryption::aes_encryptor::Aes256Ctr64LEEncryptor,
    error::{ShardError, ShardResult},
    intelligence::model::Model,
    messaging::EncoderInternalNetworkClient,
    types::{
        encoder_committee::EncoderIndex,
        shard::Shard,
        shard_input::{ShardInput, ShardInputAPI},
        shard_verifier::ShardAuthToken,
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use model::{client::ModelClient, ModelInput, ModelInputV1};
use objects::{
    error::ObjectError,
    networking::ObjectNetworkClient,
    storage::{ObjectPath, ObjectStorage},
};
use probe::messaging::ProbeClient;
use rand::{rngs::OsRng, RngCore};
use shared::{
    checksum::Checksum,
    crypto::{keys::EncoderKeyPair, Aes256IV, Aes256Key, EncryptionKey},
    digest::Digest,
    metadata::{
        CompressionAPI, CompressionAlgorithmV1, CompressionV1, EncryptionV1, Metadata, MetadataAPI,
    },
    scope::Scope,
    signed::Signed,
    verified::Verified,
};
use tracing::debug;

use super::broadcast::BroadcastProcessor;

pub(crate) struct InputProcessor<
    C: EncoderInternalNetworkClient,
    O: ObjectNetworkClient,
    M: ModelClient,
    S: ObjectStorage,
    P: ProbeClient,
> {
    downloader: ActorHandle<Downloader<O, S>>,
    compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
    broadcast_handle: ActorHandle<BroadcastProcessor<C, S, P>>,
    model_client: Arc<M>,
    encryptor: ActorHandle<EncryptionProcessor<Aes256Ctr64LEEncryptor>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    storage: ActorHandle<StorageProcessor<S>>,
}

impl<
        C: EncoderInternalNetworkClient,
        O: ObjectNetworkClient,
        M: ModelClient,
        S: ObjectStorage,
        P: ProbeClient,
    > InputProcessor<C, O, M, S, P>
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        downloader: ActorHandle<Downloader<O, S>>,
        compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
        broadcast_handle: ActorHandle<BroadcastProcessor<C, S, P>>,
        model_client: Arc<M>,
        encryptor: ActorHandle<EncryptionProcessor<Aes256Ctr64LEEncryptor>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        storage: ActorHandle<StorageProcessor<S>>,
    ) -> Self {
        Self {
            downloader,
            compressor,
            broadcast_handle,
            model_client,
            encryptor,
            encoder_keypair,
            storage,
        }
    }
}

#[async_trait]
impl<
        C: EncoderInternalNetworkClient,
        O: ObjectNetworkClient,
        M: ModelClient,
        S: ObjectStorage,
        P: ProbeClient,
    > Processor for InputProcessor<C, O, M, S, P>
{
    type Input = (
        Shard,
        Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let keypair = self.encoder_keypair.inner().copy();
        let result: ShardResult<()> = async {
            let (shard, verified_signed_input) = msg.input;
            let epoch = shard.epoch();
            // let metadata = auth_token.metadata_commitment().metadata();
            // // TODO: need to change this to handle network index?
            // let downloader_input =
            //     DownloaderInput::new(epoch, EncoderIndex::default(), metadata.clone());

            // let download_result = self
            //     .downloader
            //     .process(downloader_input, msg.cancellation.clone())
            //     .await?;

            // let uncompressed_size = metadata
            //     .compression()
            //     .ok_or(ShardError::MissingCompressionMetadata)?
            //     .uncompressed_size();

            // let decompressed = self
            //     .compressor
            //     .process(
            //         CompressorInput::Decompress(download_result, uncompressed_size),
            //         msg.cancellation.clone(),
            //     )
            //     .await?;

            // let array: ArrayD<f32> =
            //     bcs::from_bytes(&decompressed).map_err(ShardError::MalformedType)?;
            // let model_output = self.model.process(array, msg.cancellation.clone()).await?;

            // let shape = model_output.shape();

            // Create compressible data (repeated pattern)
            //

            // TODO: use config for model timeout
            let model_input = ModelInput::V1(ModelInputV1::new(Bytes::from(
                "mock input, not used at all".to_string(),
            )));

            let timeout = Duration::from_secs(1);
            let model_output = self
                .model_client
                .call(model_input, timeout)
                .await
                .map_err(ShardError::ModelError)?;

            let serialized = serde_json::to_string(&model_output).expect("Serialization failed");
            let message = bcs::to_bytes(&serialized).map_err(ShardError::MalformedType);
            debug!("bcs: {:?}", message);
            let message = message?;
            let model_bytes = Bytes::from(message);
            debug!("model bytes: {:?}", model_bytes);

            let uncompressed_size = model_bytes.len();
            let compressed_embeddings = self
                .compressor
                .process(
                    CompressorInput::Compress(model_bytes),
                    msg.cancellation.clone(),
                )
                .await?;

            debug!("Passed compression");
            let encoder_public_key = self.encoder_keypair.public().clone();
            debug!("Encoder public key: {:?}", encoder_public_key);
            debug!("Shard encoders: {:?}", shard.encoders());
            debug!(
                "Is encoder in shard: {}",
                shard.contains(&encoder_public_key)
            );

            let signed_shard =
                Signed::new(shard.clone(), Scope::EncryptionKey, &keypair.private()).unwrap();
            let signature_bytes = signed_shard.raw_signature();

            let mut key_bytes = [0u8; 32];
            key_bytes.copy_from_slice(&signature_bytes[..32]); // Use only the first 32 bytes

            let mut iv_bytes = [0u8; 16];
            iv_bytes.copy_from_slice(&signature_bytes[..16]); // Use only the first 16 bytes

            let key = EncryptionKey::Aes256(Aes256IV {
                iv: iv_bytes,
                key: Aes256Key::from(key_bytes),
            });

            let encrypted_embeddings = self
                .encryptor
                .process(
                    EncryptionInput::Encrypt(key.clone(), compressed_embeddings),
                    msg.cancellation.clone(),
                )
                .await?;
            let download_size = encrypted_embeddings.len();

            let checksum = Checksum::new_from_bytes(&encrypted_embeddings);
            let path = ObjectPath::new(checksum.to_string()).map_err(ShardError::ObjectError)?;

            self.storage
                .process(
                    StorageProcessorInput::Store(path, encrypted_embeddings),
                    msg.cancellation.clone(),
                )
                .await?;

            let key_digest = Digest::new(&key).map_err(ShardError::DigestFailure)?;

            let commit_data = Metadata::new_v1(
                Some(CompressionV1::new(
                    CompressionAlgorithmV1::ZSTD,
                    uncompressed_size,
                )),
                Some(EncryptionV1::Aes256Ctr64LE(key_digest)),
                checksum,
                download_size,
            );

            self.broadcast_handle
                .process(
                    BroadcastAction::Commit(
                        shard,
                        verified_signed_input.auth_token().clone(),
                        None,
                        commit_data,
                    ),
                    msg.cancellation.clone(),
                )
                .await?;

            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
