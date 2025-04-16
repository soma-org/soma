use std::{collections::HashSet, sync::Arc};

use crate::{
    actors::{
        workers::{
            broadcaster::BroadcasterProcessor,
            compression::{CompressionProcessor, CompressorInput},
            downloader::{Downloader, DownloaderInput},
            encryption::{EncryptionInput, EncryptionProcessor},
            model::ModelProcessor,
            storage::{StorageProcessor, StorageProcessorInput},
        },
        ActorHandle, ActorMessage, Processor,
    },
    compression::zstd_compressor::ZstdCompressor,
    encryption::aes_encryptor::Aes256Ctr64LEEncryptor,
    error::{ShardError, ShardResult},
    intelligence::model::Model,
    messaging::EncoderInternalNetworkClient,
    types::{
        encoder_committee::EncoderIndex, shard::Shard, shard_input::ShardInput,
        shard_verifier::ShardAuthToken,
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use ndarray::ArrayD;
use objects::{
    networking::ObjectNetworkClient,
    storage::{ObjectPath, ObjectStorage},
};
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

pub(crate) struct InputProcessor<
    O: ObjectNetworkClient,
    M: Model,
    E: EncoderInternalNetworkClient,
    S: ObjectStorage,
> {
    downloader: ActorHandle<Downloader<O, S>>,
    compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
    model: ActorHandle<ModelProcessor<M>>,
    encryptor: ActorHandle<EncryptionProcessor<Aes256Ctr64LEEncryptor>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    broadcaster: ActorHandle<BroadcasterProcessor<E>>,
    storage: ActorHandle<StorageProcessor<S>>,
}

impl<O: ObjectNetworkClient, M: Model, E: EncoderInternalNetworkClient, S: ObjectStorage>
    InputProcessor<O, M, E, S>
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        downloader: ActorHandle<Downloader<O, S>>,
        compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
        model: ActorHandle<ModelProcessor<M>>,
        encryptor: ActorHandle<EncryptionProcessor<Aes256Ctr64LEEncryptor>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        broadcaster: ActorHandle<BroadcasterProcessor<E>>,
        storage: ActorHandle<StorageProcessor<S>>,
    ) -> Self {
        Self {
            downloader,
            compressor,
            model,
            encryptor,
            encoder_keypair,
            broadcaster,
            storage,
        }
    }
}

#[async_trait]
impl<O: ObjectNetworkClient, M: Model, E: EncoderInternalNetworkClient, S: ObjectStorage> Processor
    for InputProcessor<O, M, E, S>
{
    type Input = (
        Shard,
        Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let keypair = self.encoder_keypair.inner().copy();
        let result: ShardResult<()> = async {
            // let (auth_token, shard, verified_signed_input) = msg.input;
            // let epoch = auth_token.epoch();
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
            // let model_bytes = bcs::to_bytes(&model_output)
            //     .map_err(|e| ShardError::SerializationFailure(e.to_string()))?;
            // let model_bytes = Bytes::copy_from_slice(&model_bytes);

            // let uncompressed_size = model_bytes.len();
            // let compressed_embeddings = self
            //     .compressor
            //     .process(
            //         CompressorInput::Compress(model_bytes),
            //         msg.cancellation.clone(),
            //     )
            //     .await?;

            // let signed_shard =
            //     Signed::new(shard.clone(), Scope::EncryptionKey, &keypair.private()).unwrap();
            // let signature_bytes = signed_shard.raw_signature();

            // let mut key_bytes = [0u8; 32];
            // key_bytes.copy_from_slice(&signature_bytes);

            // let mut iv_bytes = [0u8; 16];
            // iv_bytes.copy_from_slice(&signature_bytes);

            // let key = EncryptionKey::Aes256(Aes256IV {
            //     iv: iv_bytes,
            //     key: Aes256Key::from(key_bytes),
            // });

            // let encrypted_embeddings = self
            //     .encryptor
            //     .process(
            //         EncryptionInput::Encrypt(key.clone(), compressed_embeddings),
            //         msg.cancellation.clone(),
            //     )
            //     .await?;
            // let download_size = encrypted_embeddings.len();

            // let checksum = Checksum::new_from_bytes(&encrypted_embeddings);
            // // TODO: generate a path using the checksum of encrypted data, or make the storage actor generate the path and return it.
            // let path = ObjectPath::new(checksum.to_string())?;

            // self.storage
            //     .process(
            //         StorageProcessorInput::Store(path, encrypted_embeddings),
            //         msg.cancellation.clone(),
            //     )
            //     .await?;

            // let key_digest = Digest::new(&key).map_err(ShardError::DigestFailure)?;

            // let commit_data = Metadata::new_v1(
            //     Some(CompressionV1::new(
            //         CompressionAlgorithmV1::ZSTD,
            //         uncompressed_size,
            //     )),
            //     Some(EncryptionV1::Aes256Ctr64LE(key_digest)),
            //     checksum,
            //     download_size,
            // );

            // let inference_set = shard.inference_set(); // Vec<EncoderIndex>
            // let evaluation_set = shard.evaluation_set(); // Vec<EncoderIndex>

            // // Combine into a HashSet to deduplicate
            // let mut peers_set: HashSet<EncoderIndex> = inference_set.into_iter().collect();
            // peers_set.extend(evaluation_set);

            // // Convert back to Vec
            // let peers: Vec<EncoderIndex> = peers_set.into_iter().collect();
            // // TODO: need to certify
            // // let _ = self
            // //     .broadcaster
            // //     .process(
            // //         (
            // //             auth_token,
            // //             shard,
            // //             BroadcastType::RevealKey(epoch, shard_ref, self.own_index),
            // //             peers,
            // //         ),
            // //         msg.cancellation.clone(),
            // //     )
            // //     .await;
            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
