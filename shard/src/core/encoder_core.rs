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
    storage::blob::{
        compression::ZstdCompressor, encryption::AesEncryptor, BlobEncryption, BlobPath,
        BlobStorage,
    },
    types::{
        certificate::ShardCertificate,
        manifest::ManifestAPI,
        network_committee::NetworkingIndex,
        shard_commit::ShardCommit,
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

pub struct EncoderCore<C: EncoderNetworkClient, M: Model, B: BlobStorage> {
    shard_input_semaphore: Arc<Semaphore>,
    shard_commit_certificate_semaphore: Arc<Semaphore>,
    shard_reveal_certificate_semaphore: Arc<Semaphore>,
    shard_removal_certificate_semaphore: Arc<Semaphore>,
    shard_endorsement_certificate_semaphore: Arc<Semaphore>,
    shard_completion_proof_semaphore: Arc<Semaphore>,
    client: Arc<C>,
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
            downloader,
            encryptor,
            compressor,
            model,
            storage,
        }
    }

    pub async fn process_shard_input(&self, shard_input: Verified<Signed<ShardInput>>) {
        if let Ok(permit) = self.shard_input_semaphore.clone().acquire_owned().await {
            let downloader = self.downloader.clone();
            let compressor = self.compressor.clone();
            let model = self.model.clone();
            let encryptor = self.encryptor.clone();
            let storage = self.storage.clone();
            tokio::spawn(async move {
                // println!("{:?}", shard_input);
                let cancellation = CancellationToken::new();
                let manifest = shard_input.manifest();
                let batches = manifest.batches().to_owned();

                let mut set: JoinSet<ShardResult<()>> = JoinSet::new();

                // TODO: sign the shard and hash the signature. Simple, easily reproducible, random, secure
                let mut key = [0u8; 32];
                OsRng.fill_bytes(&mut key);
                let key = AesKey::from(key);

                for batch in batches {
                    // Clone the references to actors/services needed for this batch
                    let cancellation = cancellation.clone();
                    let downloader = downloader.clone();
                    let compressor = compressor.clone();
                    let model = model.clone();
                    let encryptor = encryptor.clone();
                    let storage = storage.clone();

                    set.spawn(async move {
                        // Process the batch through each actor in sequence
                        let downloader_input =
                            DownloaderInput::new(NetworkingIndex::default(), batch.to_owned());
                        let download_result = downloader
                            .send(downloader_input, cancellation.clone())
                            .await?;
                        let decompressed = compressor
                            .send(
                                CompressorInput::Decompress(download_result),
                                cancellation.clone(),
                            )
                            .await?;
                        let array: ArrayD<f32> =
                            bcs::from_bytes(&decompressed).map_err(ShardError::MalformedType)?;
                        let model_output = model.send(array, cancellation.clone()).await?;
                        let model_bytes = bcs::to_bytes(&model_output)
                            .map_err(ShardError::SerializationFailure)?;
                        let model_bytes = Bytes::copy_from_slice(&model_bytes);
                        let compressed_embeddings = compressor
                            .send(CompressorInput::Compress(model_bytes), cancellation.clone())
                            .await?;
                        let encrypted_embeddings = encryptor
                            .send(
                                EncryptionInput::Encrypt(key, compressed_embeddings),
                                cancellation.clone(),
                            )
                            .await?;
                        // TODO: generate a path using the checksum of encrypted data, or make the storage actor generate the path and return it.
                        let path = BlobPath::new("change me".to_string())?;
                        let checksum = storage
                            .send(
                                StorageProcessorInput::Store(path, encrypted_embeddings),
                                cancellation.clone(),
                            )
                            .await?;
                        // return checksums to package into a commit
                        Ok(())
                    });
                }

                // Collect all results
                // let mut all_results = Vec::new();
                // while let Some(result) = set.join_next().await {
                //     all_results.push(result??);
                // }

                // package into a commit
                // broadcast to everyone in the shard

                drop(permit);
            });
        }
    }

    pub async fn process_shard_commit_certificate(
        &self,
        shard_commit_certificate: Verified<ShardCertificate<Signed<ShardCommit>>>,
    ) {
        if let Ok(permit) = self
            .shard_commit_certificate_semaphore
            .clone()
            .acquire_owned()
            .await
        {
            tokio::spawn(async move {
                println!("{:?}", shard_commit_certificate);
                drop(permit);
            });
        }
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
        }
    }
}
