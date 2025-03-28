use std::{collections::HashSet, sync::Arc, time::Duration};

use crate::{
    actors::{
        workers::{
            broadcaster::{BroadcastType, BroadcasterProcessor},
            compression::{CompressionProcessor, CompressorInput},
            encryption::{EncryptionInput, EncryptionProcessor},
            storage::{StorageProcessor, StorageProcessorInput, StorageProcessorOutput},
        },
        ActorHandle, ActorMessage, Processor,
    },
    compression::zstd_compressor::ZstdCompressor,
    core::slot_tracker::SlotTracker,
    encryption::aes_encryptor::Aes256Ctr64LEEncryptor,
    error::{ShardError, ShardResult},
    messaging::tonic::internal::EncoderInternalTonicClient,
    storage::{
        datastore::Store,
        object::{filesystem::FilesystemObjectStorage, ObjectPath},
    },
    types::{
        encoder_committee::EncoderIndex,
        shard::Shard,
        shard_reveal::{ShardReveal, ShardRevealAPI},
        shard_verifier::ShardAuthToken,
    },
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use quick_cache::sync::Cache;
use shared::{
    checksum::Checksum,
    digest::Digest,
    metadata::{CompressionAPI, Metadata, MetadataAPI},
    signed::Signed,
    verified::Verified,
};

pub(crate) struct RevealProcessor {
    cache: Cache<Digest<Shard>, ()>,
    store: Arc<dyn Store>,
    slot_tracker: SlotTracker,
    broadcaster: ActorHandle<BroadcasterProcessor<EncoderInternalTonicClient>>,
    storage: ActorHandle<StorageProcessor<FilesystemObjectStorage>>,
    compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
    encryptor: ActorHandle<EncryptionProcessor<Aes256Ctr64LEEncryptor>>,
}

impl RevealProcessor {
    pub(crate) fn new(
        cache_size: usize,
        store: Arc<dyn Store>,
        slot_tracker: SlotTracker,
        broadcaster: ActorHandle<BroadcasterProcessor<EncoderInternalTonicClient>>,
        storage: ActorHandle<StorageProcessor<FilesystemObjectStorage>>,
        compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
        encryptor: ActorHandle<EncryptionProcessor<Aes256Ctr64LEEncryptor>>,
    ) -> Self {
        Self {
            cache: Cache::new(cache_size),
            store,
            slot_tracker,
            broadcaster,
            storage,
            compressor,
            encryptor,
        }
    }
}

#[async_trait]
impl Processor for RevealProcessor {
    type Input = (
        ShardAuthToken,
        Shard,
        Metadata,
        Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            // TODO: check/mark cache
            let (auth_token, shard, metadata, verified_reveal) = msg.input;
            let slot = verified_reveal.slot();
            let epoch = verified_reveal.auth_token().epoch();

            let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
            let data_path = ObjectPath::from_checksum(metadata.checksum());

            let uncompressed_size = metadata
                .compression()
                .ok_or(ShardError::MissingCompressionMetadata)?
                .uncompressed_size();

            let encrypted_bytes = match self
                .storage
                .process(
                    StorageProcessorInput::Get(data_path),
                    msg.cancellation.clone(),
                )
                .await?
            {
                StorageProcessorOutput::Get(bytes) => bytes,
                _ => return Err(ShardError::MissingData),
            };
            let decrypted_bytes = self
                .encryptor
                .process(
                    EncryptionInput::Encrypt(verified_reveal.key().clone(), encrypted_bytes),
                    msg.cancellation.clone(),
                )
                .await?;

            let embedding = self
                .compressor
                .process(
                    CompressorInput::Decompress(decrypted_bytes, uncompressed_size),
                    msg.cancellation.clone(),
                )
                .await?;

            let embedding_checksum = Checksum::new_from_bytes(&embedding);
            let embedding_path = ObjectPath::from_checksum(embedding_checksum);
            let _ = self
                .storage
                .process(
                    StorageProcessorInput::Store(embedding_path, embedding),
                    msg.cancellation.clone(),
                )
                .await?;

            let count = self.store.atomic_reveal(
                epoch,
                shard_ref,
                slot,
                verified_reveal.key().to_owned(),
                embedding_checksum,
            )?;
            if count == shard.minimum_inference_size() as usize {
                let duration = self
                    .store
                    .time_since_first_reveal(epoch, shard_ref)
                    .unwrap_or(Duration::from_secs(60));
                // TODO: make this cleaner should be built into the shard
                let inference_set = shard.inference_set(); // Vec<EncoderIndex>
                let evaluation_set = shard.evaluation_set(); // Vec<EncoderIndex>

                // Combine into a HashSet to deduplicate
                let mut peers_set: HashSet<EncoderIndex> = inference_set.into_iter().collect();
                peers_set.extend(evaluation_set);

                // Convert back to Vec
                let peers: Vec<EncoderIndex> = peers_set.into_iter().collect();
                let shard_clone = shard.clone();
                let broadcaster = self.broadcaster.clone();
                self.slot_tracker
                    .start_reveal_vote_timer(shard_ref, duration, move || async move {
                        let _ = broadcaster
                            .process(
                                (
                                    auth_token,
                                    shard_clone,
                                    BroadcastType::RevealVote(epoch, shard_ref),
                                    peers,
                                ),
                                msg.cancellation.clone(),
                            )
                            .await;
                    })
                    .await;
            }
            if count == shard.inference_size() {
                self.slot_tracker.trigger_reveal_vote(shard_ref).await;
            }

            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
