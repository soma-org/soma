use std::{sync::Arc, time::Duration};

use crate::{
    actors::workers::{
        compression::{CompressionProcessor, CompressorInput},
        downloader::Downloader,
        encryption::{EncryptionInput, EncryptionProcessor},
        storage::{StorageHandle, StorageProcessor},
    },
    compression::zstd_compressor::ZstdCompressor,
    core::internal_broadcaster::Broadcaster,
    encryption::aes_encryptor::Aes256Ctr64LEEncryptor,
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::shard_commit::ShardCommit,
};
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use model::{client::ModelClient, ModelInput, ModelInputV1};
use objects::{
    networking::ObjectNetworkClient,
    storage::{ObjectPath, ObjectStorage},
};
use probe::messaging::ProbeClient;
use shared::{
    actors::{ActorHandle, ActorMessage, Processor},
    checksum::Checksum,
    crypto::{
        keys::{EncoderKeyPair, PeerPublicKey},
        Aes256IV, Aes256Key, EncryptionKey,
    },
    digest::Digest,
    error::{ShardError, ShardResult},
    metadata::{CompressionAlgorithmV1, CompressionV1, EncryptionV1, Metadata},
    probe::ProbeMetadata,
    scope::Scope,
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use soma_network::multiaddr::Multiaddr;
use tracing::{debug, info};
use types::shard::{ShardInput, ShardInputAPI};

use super::commit::CommitProcessor;

pub(crate) struct InputProcessor<
    C: EncoderInternalNetworkClient,
    O: ObjectNetworkClient,
    M: ModelClient,
    S: ObjectStorage,
    P: ProbeClient,
> {
    downloader: ActorHandle<Downloader<O, S>>,
    compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
    broadcaster: Arc<Broadcaster<C>>,
    model_client: Arc<M>,
    encryptor: ActorHandle<EncryptionProcessor<Aes256Ctr64LEEncryptor>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    storage: StorageHandle<S>,
    commit_pipeline: ActorHandle<CommitProcessor<C, O, S, P>>,
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
        broadcaster: Arc<Broadcaster<C>>,
        model_client: Arc<M>,
        encryptor: ActorHandle<EncryptionProcessor<Aes256Ctr64LEEncryptor>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        storage: ActorHandle<StorageProcessor<S>>,
        commit_pipeline: ActorHandle<CommitProcessor<C, O, S, P>>,
    ) -> Self {
        Self {
            downloader,
            compressor,
            broadcaster,
            model_client,
            encryptor,
            encoder_keypair,
            storage: StorageHandle::new(storage),
            commit_pipeline,
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
    // TODO the way input is passing in the encoders own probe metadata, peer public key, and multiaddress is not clean
    type Input = (
        Shard,
        Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
        ProbeMetadata,
        PeerPublicKey,
        Multiaddr,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let keypair = self.encoder_keypair.inner().copy();
        let result: ShardResult<()> = async {
            let (shard, verified_signed_input, probe_metadata, peer, address) = msg.input;
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
                .store(path, encrypted_embeddings, msg.cancellation.clone())
                .await?;
            // self.storage
            //     .process(
            //         StorageProcessorInput::Store(path, encrypted_embeddings),
            //         msg.cancellation.clone(),
            //     )
            //     .await?;

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
            info!("Handling commit in BroadcastProcessor");
            let inner_keypair = self.encoder_keypair.inner().copy();

            // Create signed route if provided
            // let signed_route = route.map(|r| {
            //     Signed::new(r, Scope::ShardCommitRoute, &inner_keypair.copy().private()).unwrap()
            // });

            // Create commit
            let commit = ShardCommit::new_v1(
                verified_signed_input.auth_token().clone(),
                self.encoder_keypair.public(),
                None,
                commit_data,
            );

            // Sign the commit
            let signed_commit =
                Signed::new(commit, Scope::ShardCommit, &inner_keypair.private()).unwrap();
            let verified = Verified::from_trusted(signed_commit).unwrap();

            // SEND TO PIPELINE
            // NOTE: ONLY DO THIS IF YOU ARE A MEMBER OF THE SHARD
            // ROUTED NODES WILL NOT

            self.commit_pipeline
                .process(
                    (
                        shard.clone(),
                        verified.clone(),
                        probe_metadata,
                        peer,
                        address,
                    ),
                    msg.cancellation.clone(),
                )
                .await?;
            info!("Broadcasting to other nodes");
            // Broadcast to other encoders
            self.broadcaster
                .broadcast(
                    verified.clone(),
                    shard.encoders(),
                    |client, peer, verified_type| async move {
                        client
                            .send_commit(&peer, &verified_type, MESSAGE_TIMEOUT)
                            .await?;
                        Ok(())
                    },
                )
                .await?;

            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
