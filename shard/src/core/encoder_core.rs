// use crate::networking::messaging::MESSAGE_TIMEOUT;
use crate::{
    actors::{
        blob::{StorageProcessor, StorageProcessorInput},
        compression::{Compressor, CompressorInput},
        downloader::{Downloader, DownloaderInput},
        encryption::{EncryptionInput, Encryptor},
        model::ModelProcessor,
        ActorHandle,
    },
    crypto::AesKey,
    error::{ShardError, ShardResult},
    intelligence::model::Model,
    networking::messaging::EncoderNetworkClient,
    storage::blob::{compression::ZstdCompressor, encryption::AesEncryptor, BlobPath, BlobStorage},
    types::{
        certificate::ShardCertificate,
        checksum::Checksum,
        data::DataAPI,
        manifest::{Batch, BatchAPI, Compression, Encryption, Manifest, ManifestAPI},
        network_committee::NetworkingIndex,
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_completion_proof::ShardCompletionProof,
        shard_endorsement::ShardEndorsement,
        shard_input::{ShardInput, ShardInputAPI},
        shard_removal::ShardRemoval,
        shard_reveal::ShardReveal,
        signed::Signed,
        verified::Verified,
    },
};
use bytes::Bytes;
use ndarray::ArrayD;
use rand::rngs::OsRng;
use rand::RngCore;
use std::sync::Arc;
use tokio::{sync::Semaphore, task::JoinSet};
use tokio_util::sync::CancellationToken;

use super::broadcaster::Broadcaster;

pub struct EncoderCore<C: EncoderNetworkClient, M: Model, B: BlobStorage> {
    shard_input_semaphore: Arc<Semaphore>,
    shard_commit_certificate_semaphore: Arc<Semaphore>,
    shard_reveal_certificate_semaphore: Arc<Semaphore>,
    shard_removal_certificate_semaphore: Arc<Semaphore>,
    shard_endorsement_certificate_semaphore: Arc<Semaphore>,
    shard_completion_proof_semaphore: Arc<Semaphore>,
    client: Arc<C>,
    broadcaster: Arc<Broadcaster<C>>,
    downloader: ActorHandle<Downloader>,
    // TODO: potentially change this to be a generic
    encryptor: ActorHandle<Encryptor<AesKey, AesEncryptor>>,
    // TODO: potentially change this to be a generic
    compressor: ActorHandle<Compressor<ZstdCompressor>>,
    model: ActorHandle<ModelProcessor<M>>,
    storage: ActorHandle<StorageProcessor<B>>,
}

impl<C, M, B> EncoderCore<C, M, B>
where
    C: EncoderNetworkClient,
    M: Model,
    B: BlobStorage,
{
    pub fn new(
        max_concurrent_tasks: usize,
        client: Arc<C>,
        broadcaster: Arc<Broadcaster<C>>,
        downloader: ActorHandle<Downloader>,
        encryptor: ActorHandle<Encryptor<AesKey, AesEncryptor>>,
        compressor: ActorHandle<Compressor<ZstdCompressor>>,
        model: ActorHandle<ModelProcessor<M>>,
        storage: ActorHandle<StorageProcessor<B>>,
    ) -> Self {
        Self {
            shard_input_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            shard_commit_certificate_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            shard_reveal_certificate_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            shard_removal_certificate_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            shard_endorsement_certificate_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            shard_completion_proof_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            client,
            broadcaster,
            downloader,
            encryptor,
            compressor,
            model,
            storage,
        }
    }

    pub async fn process_shard_input(
        &self,
        shard: Shard,
        shard_input: Verified<Signed<ShardInput>>,
    ) -> ShardResult<()> {
        let downloader = self.downloader.clone();
        let compressor = self.compressor.clone();
        let model = self.model.clone();
        let encryptor = self.encryptor.clone();
        let storage = self.storage.clone();
        let broadcaster = self.broadcaster.clone();
        // println!("{:?}", shard_input);
        // TODO: look up or create cancellation token for the shard
        let cancellation = CancellationToken::new();
        let data = shard_input.data();

        // TODO: sign the shard and hash the signature. Simple, easily reproducible, random, secure
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        let key = AesKey::from(key);

        // Process the batch through each actor in sequence
        // TODO: fix the networking index peer
        let downloader_input = DownloaderInput::new(
            NetworkingIndex::default(),
            BlobPath::from_checksum(data.checksum()),
        );

        let download_result = downloader
            .process(downloader_input, cancellation.clone())
            .await?;

        let decompressed = compressor
            .process(
                CompressorInput::Decompress(download_result),
                cancellation.clone(),
            )
            .await?;
        let array: ArrayD<f32> =
            bcs::from_bytes(&decompressed).map_err(ShardError::MalformedType)?;
        let model_output = model.process(array, cancellation.clone()).await?;

        let shape = model_output.shape();
        let model_bytes = bcs::to_bytes(&model_output).map_err(ShardError::SerializationFailure)?;
        let model_bytes = Bytes::copy_from_slice(&model_bytes);
        let compressed_embeddings = compressor
            .process(CompressorInput::Compress(model_bytes), cancellation.clone())
            .await?;
        let encrypted_embeddings = encryptor
            .process(
                EncryptionInput::Encrypt(key, compressed_embeddings),
                cancellation.clone(),
            )
            .await?;
        // figure out the hash?

        let download_size = encrypted_embeddings.len();

        let checksum = Checksum::new_from_bytes(&encrypted_embeddings);
        // TODO: generate a path using the checksum of encrypted data, or make the storage actor generate the path and return it.
        let path = BlobPath::new(checksum.to_string())?;

        storage
            .process(
                StorageProcessorInput::Store(path, encrypted_embeddings),
                cancellation.clone(),
            )
            .await?;

        // let new_manifest = Manifest::new_v1(
        //     shard_input.modality().to_owned(),
        //     Compression::ZSTD,
        //     Some(Encryption::Aes256Ctr64LE),
        //     new_batches,
        // );
        // create shard commit type

        // should I move this function defition elsewhere?
        // async fn network_fn<C: EncoderNetworkClient>(client: Arc<C>, peer: NetworkingIndex) -> ShardResult<Si> {

        // }

        // let signatures = broadcaster.collect_signatures(new_manifest, peers, network_fn).await?;

        // package into a commit
        // broadcast to everyone in the shard

        Ok(())
    }

    pub async fn process_shard_commit_certificate(
        &self,
        shard_commit_certificate: Verified<ShardCertificate<Signed<ShardCommit>>>,
    ) {
        // if let Ok(permit) = self
        //     .shard_commit_certificate_semaphore
        //     .clone()
        //     .acquire_owned()
        //     .await
        // {
        //     let downloader = self.downloader.clone();
        //     let compressor = self.compressor.clone();
        //     let storage = self.storage.clone();

        //     // TODO: fix this
        //     let peer = NetworkingIndex::default();

        //     tokio::spawn(async move {
        //         let manifest = shard_commit_certificate.manifest();
        //         let cancellation = CancellationToken::new();
        //         let mut set: JoinSet<ShardResult<()>> = JoinSet::new();

        //         let probe_downloader_handle = downloader.clone();
        //         let probe_compressor_handle: ActorHandle<Compressor<ZstdCompressor>> =
        //             compressor.clone();
        //         let probe_storage_handle = storage.clone();
        //         let probe_cancellation = cancellation.clone();
        //         // TODO: fix this
        //         let probe_checksum = Checksum::default();
        //         let probe_path = BlobPath::from_checksum(probe_checksum);

        //         set.spawn(async move {
        //             let downloaded_bytes = probe_downloader_handle
        //                 .send(
        //                     DownloaderInput::new(peer, probe_path.clone()),
        //                     probe_cancellation.clone(),
        //                 )
        //                 .await?;
        //             // maybe cancel here instead?

        //             let decompressed_probe_bytes = probe_compressor_handle
        //                 .send(
        //                     CompressorInput::Decompress(downloaded_bytes),
        //                     probe_cancellation.clone(),
        //                 )
        //                 .await?;

        //             let _ = probe_storage_handle
        //                 .send(
        //                     StorageProcessorInput::Store(probe_path, decompressed_probe_bytes),
        //                     probe_cancellation.clone(),
        //                 )
        //                 .await?;
        //             Ok(())
        //         });

        //         let batches = manifest.batches().to_owned();

        //         for batch in batches {
        //             let storage = storage.clone();
        //             let downloader = downloader.clone();
        //             let cancellation = cancellation.clone();

        //             set.spawn(async move {
        //                 let path = BlobPath::from_checksum(batch.checksum());

        //                 let downloader_input = DownloaderInput::new(peer, path.clone());

        //                 let download_result = downloader
        //                     .send(downloader_input, cancellation.clone())
        //                     .await?;
        //                 let _ = storage
        //                     .send(
        //                         StorageProcessorInput::Store(path, download_result),
        //                         cancellation.clone(),
        //                     )
        //                     .await?;

        //                 Ok(())
        //             });
        //         }

        //         // download the commit data
        //         // download the probe
        //         // mark complete when those steps are done
        //         // track the completion of these things and trigger actions after a certain timeout for instance
        //         // downloader.send(DownloaderInput, cancellation)

        //         println!("{:?}", shard_commit_certificate);
        //         drop(permit);
        //     });
    }

    pub async fn process_shard_reveal_certificate(
        &self,
        shard_reveal_certificate: Verified<ShardCertificate<Signed<ShardReveal>>>,
    ) {
        if let Ok(permit) = self
            .shard_reveal_certificate_semaphore
            .clone()
            .acquire_owned()
            .await
        {
            tokio::spawn(async move {
                // decrypt the data
                // decompress the data
                // mark complete when data has been applied to every probe and store the intermediate responses
                println!("{:?}", shard_reveal_certificate);
                drop(permit);
            });
        }
    }

    pub async fn process_shard_removal_certificate(
        &self,
        shard_removal_certificate: Verified<ShardCertificate<ShardRemoval>>,
    ) {
        if let Ok(permit) = self
            .shard_removal_certificate_semaphore
            .clone()
            .acquire_owned()
            .await
        {
            tokio::spawn(async move {
                println!("{:?}", shard_removal_certificate);
                drop(permit);
            });
        }
    }

    pub async fn process_shard_endorsement_certificate(
        &self,
        shard_endorsement_certificate: Verified<ShardCertificate<Signed<ShardEndorsement>>>,
    ) {
        if let Ok(permit) = self
            .shard_endorsement_certificate_semaphore
            .clone()
            .acquire_owned()
            .await
        {
            tokio::spawn(async move {
                println!("{:?}", shard_endorsement_certificate);
                drop(permit);
            });
        }
    }

    pub async fn process_shard_completion_proof(
        &self,
        shard_completion_proof: Verified<ShardCompletionProof>,
    ) {
        if let Ok(permit) = self
            .shard_completion_proof_semaphore
            .clone()
            .acquire_owned()
            .await
        {
            tokio::spawn(async move {
                println!("{:?}", shard_completion_proof);
                drop(permit);
            });
        }
    }
}

impl<C: EncoderNetworkClient, M: Model, B: BlobStorage> Clone for EncoderCore<C, M, B> {
    fn clone(&self) -> Self {
        Self {
            shard_input_semaphore: Arc::clone(&self.shard_input_semaphore),
            shard_commit_certificate_semaphore: Arc::clone(
                &self.shard_commit_certificate_semaphore,
            ),
            shard_reveal_certificate_semaphore: Arc::clone(
                &self.shard_reveal_certificate_semaphore,
            ),
            shard_removal_certificate_semaphore: Arc::clone(
                &self.shard_removal_certificate_semaphore,
            ),
            shard_endorsement_certificate_semaphore: Arc::clone(
                &self.shard_endorsement_certificate_semaphore,
            ),
            shard_completion_proof_semaphore: Arc::clone(&self.shard_completion_proof_semaphore),
            client: Arc::clone(&self.client),
            downloader: self.downloader.clone(),
            encryptor: self.encryptor.clone(),
            compressor: self.compressor.clone(),
            model: self.model.clone(),
            storage: self.storage.clone(),
            broadcaster: self.broadcaster.clone(),
        }
    }
}
