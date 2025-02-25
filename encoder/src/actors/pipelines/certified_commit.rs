use std::{ops::Deref, sync::Arc, time::Duration};

use crate::{
    actors::{
        workers::{
            compression::{CompressionProcessor, CompressorInput},
            downloader::{Downloader, DownloaderInput},
            storage::{StorageProcessor, StorageProcessorInput},
        },
        ActorHandle, ActorMessage, Processor,
    },
    compression::zstd_compressor::ZstdCompressor,
    core::slot_tracker::SlotTracker,
    error::{ShardError, ShardResult},
    networking::object::http_network::ObjectHttpClient,
    storage::{
        datastore::Store,
        object::{filesystem::FilesystemObjectStorage, ObjectPath},
    },
    types::{
        certified::Certified,
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
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

pub(crate) struct CertifiedCommitProcessor {
    cache: Cache<Digest<Shard>, ()>,
    store: Arc<dyn Store>,
    slot_tracker: SlotTracker,
    downloader: ActorHandle<Downloader<ObjectHttpClient>>,
    compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
    storage: ActorHandle<StorageProcessor<FilesystemObjectStorage>>,
}

impl CertifiedCommitProcessor {
    pub(crate) fn new(
        cache_size: usize,
        store: Arc<dyn Store>,
        slot_tracker: SlotTracker,
        downloader: ActorHandle<Downloader<ObjectHttpClient>>,
        compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
        storage: ActorHandle<StorageProcessor<FilesystemObjectStorage>>,
    ) -> Self {
        Self {
            cache: Cache::new(cache_size),
            store,
            slot_tracker,
            downloader,
            compressor,
            storage,
        }
    }
}

#[async_trait]
impl Processor for CertifiedCommitProcessor {
    type Input = (
        Shard,
        ProbeMetadata,
        Verified<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            // TODO: check/mark cache
            let (shard, probe_metadata, verified_certified_commit) = msg.input;
            let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;

            let slot = verified_certified_commit.slot();
            let epoch = verified_certified_commit.auth_token().epoch();
            let commit_metadata = verified_certified_commit.commit();
            let data_path: ObjectPath = ObjectPath::from_checksum(commit_metadata.checksum());
            let downloader_input =
                DownloaderInput::new(verified_certified_commit.committer(), data_path.clone());
            // download the commit and store
            let encrypted_embedding_bytes = self
                .downloader
                .process(downloader_input, msg.cancellation.clone())
                .await?;
            let _ = self
                .storage
                .process(
                    StorageProcessorInput::Store(data_path, encrypted_embedding_bytes),
                    msg.cancellation.clone(),
                )
                .await?;
            // download the corresponding probe and store
            let probe_path: ObjectPath = ObjectPath::from_checksum(probe_metadata.checksum());
            let probe_bytes = self
                .downloader
                .process(
                    DownloaderInput::new(verified_certified_commit.committer(), probe_path.clone()),
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

            if count == shard.minimum_inference_size() as usize {
                let duration = self
                    .store
                    .time_since_first_certified_commit(epoch, shard_ref)
                    .unwrap_or(Duration::from_secs(60));
                self.slot_tracker
                    .start_commit_vote_timer(shard_ref, duration)
                    .await;
            }
            if count == shard.inference_size() {
                self.slot_tracker.trigger_commit_vote(shard_ref).await;
            }

            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
