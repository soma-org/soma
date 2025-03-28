use std::{collections::HashSet, ops::Deref, sync::Arc, time::Duration};

use crate::{
    actors::{
        workers::{
            broadcaster::{BroadcastType, BroadcasterProcessor},
            compression::{CompressionProcessor, CompressorInput},
            downloader::{Downloader, DownloaderInput},
            storage::{StorageProcessor, StorageProcessorInput},
        },
        ActorHandle, ActorMessage, Processor,
    },
    compression::zstd_compressor::ZstdCompressor,
    core::slot_tracker::SlotTracker,
    error::{ShardError, ShardResult},
    messaging::EncoderInternalNetworkClient,
    networking::object::{http_network::ObjectHttpClient, ObjectNetworkClient},
    storage::{
        datastore::Store,
        object::{filesystem::FilesystemObjectStorage, ObjectPath, ObjectStorage},
    },
    types::{
        certified::Certified,
        encoder_committee::EncoderIndex,
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_verifier::ShardAuthToken,
    },
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use quick_cache::sync::Cache;
use shared::{
    digest::Digest,
    metadata::{CompressionAPI, MetadataAPI},
    probe::ProbeMetadata,
    signed::Signed,
    verified::Verified,
};

pub(crate) struct CertifiedCommitProcessor<
    E: EncoderInternalNetworkClient,
    O: ObjectNetworkClient,
    S: ObjectStorage,
> {
    cache: Cache<Digest<Shard>, ()>,
    store: Arc<dyn Store>,
    slot_tracker: SlotTracker,
    broadcaster: ActorHandle<BroadcasterProcessor<E>>,
    downloader: ActorHandle<Downloader<O>>,
    compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
    storage: ActorHandle<StorageProcessor<S>>,
}

impl<E: EncoderInternalNetworkClient, O: ObjectNetworkClient, S: ObjectStorage>
    CertifiedCommitProcessor<E, O, S>
{
    pub(crate) fn new(
        cache_size: usize,
        store: Arc<dyn Store>,
        slot_tracker: SlotTracker,
        broadcaster: ActorHandle<BroadcasterProcessor<E>>,
        downloader: ActorHandle<Downloader<O>>,
        compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
        storage: ActorHandle<StorageProcessor<S>>,
    ) -> Self {
        Self {
            cache: Cache::new(cache_size),
            store,
            slot_tracker,
            broadcaster,
            downloader,
            compressor,
            storage,
        }
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient, O: ObjectNetworkClient, S: ObjectStorage> Processor
    for CertifiedCommitProcessor<E, O, S>
{
    type Input = (
        ShardAuthToken,
        Shard,
        ProbeMetadata,
        Verified<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            // TODO: check/mark cache
            let (shard_auth_token, shard, probe_metadata, verified_certified_commit) = msg.input;
            let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;

            let slot = verified_certified_commit.slot();
            let epoch = verified_certified_commit.auth_token().epoch();
            let commit_metadata = verified_certified_commit.commit();
            let data_path: ObjectPath = ObjectPath::from_checksum(commit_metadata.checksum());
            let downloader_input = DownloaderInput::new(
                epoch,
                verified_certified_commit.committer(),
                commit_metadata.clone(),
            );
            // download the commit and store
            let encrypted_embedding_bytes = self
                .downloader
                .process(downloader_input, msg.cancellation.clone())
                .await?;
            let _ = self
                .storage
                .process(
                    StorageProcessorInput::Store(data_path.clone(), encrypted_embedding_bytes),
                    msg.cancellation.clone(),
                )
                .await?;
            // download the corresponding probe and store
            let probe_path: ObjectPath = ObjectPath::from_checksum(probe_metadata.checksum());
            let probe_bytes = self
                .downloader
                .process(
                    DownloaderInput::new(
                        epoch,
                        verified_certified_commit.committer(),
                        probe_metadata.deref().clone(),
                    ),
                    msg.cancellation.clone(),
                )
                .await?;

            let uncompressed_size = probe_metadata.compression().unwrap().uncompressed_size();
            let decompressed_probe_bytes = self
                .compressor
                .process(
                    CompressorInput::Decompress(probe_bytes, uncompressed_size),
                    msg.cancellation.clone(),
                )
                .await?;

            let _ = self
                .storage
                .process(
                    StorageProcessorInput::Store(probe_path, decompressed_probe_bytes),
                    msg.cancellation.clone(),
                )
                .await?;

            let count = self.store.atomic_certified_commit(
                epoch,
                shard_ref,
                slot,
                verified_certified_commit.deref().to_owned(),
            )?;

            // TODO: only do this if you are a eval set member

            if count == shard.minimum_inference_size() as usize {
                let duration = self
                    .store
                    .time_since_first_certified_commit(epoch, shard_ref)
                    .unwrap_or(Duration::from_secs(60));

                let peers = shard.shard_set();

                let broadcaster = self.broadcaster.clone();
                let shard_clone = shard.clone();
                self.slot_tracker
                    .start_commit_vote_timer(shard_ref, duration, move || async move {
                        let _ = broadcaster
                            .process(
                                (
                                    shard_auth_token,
                                    shard_clone,
                                    BroadcastType::CommitVote(epoch, shard_ref),
                                    peers,
                                ),
                                msg.cancellation.clone(),
                            )
                            .await;
                    })
                    .await;
            }
            if count == shard.inference_size() {
                self.slot_tracker.trigger_commit_vote(shard_ref).await;
            }

            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
