// use crate::networking::messaging::MESSAGE_TIMEOUT;
use crate::{
    actors::{
        workers::{
            compression::{CompressionProcessor, CompressorInput},
            downloader::{Downloader, DownloaderInput},
            encryption::{EncryptionInput, EncryptionProcessor},
            model::ModelProcessor,
            storage::{StorageProcessor, StorageProcessorInput, StorageProcessorOutput},
        },
        ActorHandle,
    },
    compression::zstd_compressor::ZstdCompressor,
    encryption::aes_encryptor::Aes256Ctr64LEEncryptor,
    error::{ShardError, ShardResult},
    intelligence::model::Model,
    networking::{
        messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
        object::ObjectNetworkClient,
    },
    storage::object::{ObjectPath, ObjectStorage},
    types::{
        certified::Certified,
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_input::{ShardInput, ShardInputAPI},
        shard_reveal::{ShardReveal, ShardRevealAPI},
    },
};
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use ndarray::ArrayD;
use rand::rngs::OsRng;
use rand::RngCore;
use shared::{
    checksum::Checksum,
    crypto::{keys::ProtocolKeyPair, AesKey},
    digest::Digest,
    metadata::{
        CompressionAPI, CompressionAlgorithmV1, CompressionV1, EncryptionV1, Metadata, MetadataAPI,
    },
    network_committee::NetworkingIndex,
    scope::Scope,
    signed::{Signature, Signed},
    verified::Verified,
};
use std::sync::Arc;
use tokio::{sync::Semaphore, task::JoinSet};
use tokio_util::sync::CancellationToken;

use super::broadcaster::Broadcaster;

pub struct EncoderCore<
    C: EncoderInternalNetworkClient,
    M: Model,
    B: ObjectStorage,
    BC: ObjectNetworkClient,
> {
    client: Arc<C>,
    broadcaster: Arc<Broadcaster<C>>,
    downloader: ActorHandle<Downloader<BC>>,
    // TODO: potentially change this to be a generic
    encryptor: ActorHandle<EncryptionProcessor<AesKey, Aes256Ctr64LEEncryptor>>,
    // TODO: potentially change this to be a generic
    compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
    model: ActorHandle<ModelProcessor<M>>,
    storage: ActorHandle<StorageProcessor<B>>,
    keypair: Arc<ProtocolKeyPair>,
}

impl<C, M, B, BC> EncoderCore<C, M, B, BC>
where
    C: EncoderInternalNetworkClient,
    M: Model,
    B: ObjectStorage,
    BC: ObjectNetworkClient,
{
    pub fn new(
        client: Arc<C>,
        broadcaster: Arc<Broadcaster<C>>,
        downloader: ActorHandle<Downloader<BC>>,
        encryptor: ActorHandle<EncryptionProcessor<AesKey, Aes256Ctr64LEEncryptor>>,
        compressor: ActorHandle<CompressionProcessor<ZstdCompressor>>,
        model: ActorHandle<ModelProcessor<M>>,
        storage: ActorHandle<StorageProcessor<B>>,
        keypair: Arc<ProtocolKeyPair>,
    ) -> Self {
        Self {
            client,
            broadcaster,
            downloader,
            encryptor,
            compressor,
            model,
            storage,
            keypair,
        }
    }

    pub async fn process_shard_input(
        &self,
        peer: NetworkingIndex,
        shard: Shard,
        shard_input: Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // // println!("{:?}", shard_input);
        // // TODO: look up or create cancellation token for the shard
        // let cancellation = CancellationToken::new();
        // let data = shard_input.data();

        // let uncompressed_size = data.compression().map(|c| c.uncompressed_size()).unwrap();

        // // TODO: sign the shard and hash the signature. Simple, easily reproducible, random, secure
        // let mut key = [0u8; 32];
        // OsRng.fill_bytes(&mut key);
        // let key = AesKey::from(key);

        // // Process the batch through each actor in sequence
        // // TODO: fix the networking index peer
        // let downloader_input =
        //     DownloaderInput::new(peer, ObjectPath::from_checksum(data.checksum()));

        // let download_result = self
        //     .downloader
        //     .process(downloader_input, cancellation.clone())
        //     .await?;

        // let decompressed = self
        //     .compressor
        //     .process(
        //         CompressorInput::Decompress(download_result, uncompressed_size),
        //         cancellation.clone(),
        //     )
        //     .await?;
        // let array: ArrayD<f32> =
        //     bcs::from_bytes(&decompressed).map_err(ShardError::MalformedType)?;
        // let model_output = self.model.process(array, cancellation.clone()).await?;

        // let shape = model_output.shape();
        // let model_bytes = bcs::to_bytes(&model_output).map_err(ShardError::SerializationFailure)?;
        // let model_bytes = Bytes::copy_from_slice(&model_bytes);
        // let uncompressed_size = model_bytes.len();
        // let compressed_embeddings = self
        //     .compressor
        //     .process(CompressorInput::Compress(model_bytes), cancellation.clone())
        //     .await?;
        // let encrypted_embeddings = self
        //     .encryptor
        //     .process(
        //         EncryptionInput::Encrypt(key, compressed_embeddings),
        //         cancellation.clone(),
        //     )
        //     .await?;
        // // figure out the hash?

        // let download_size = encrypted_embeddings.len();

        // let checksum = Checksum::new_from_bytes(&encrypted_embeddings);
        // // TODO: generate a path using the checksum of encrypted data, or make the storage actor generate the path and return it.
        // let path = ObjectPath::new(checksum.to_string())?;

        // self.storage
        //     .process(
        //         StorageProcessorInput::Store(path, encrypted_embeddings),
        //         cancellation.clone(),
        //     )
        //     .await?;

        // // TODO: remove unwraps
        // let key_digest = Digest::new(&key).unwrap();

        // let commit_data = Metadata::new_v1(
        //     Some(CompressionV1::new(
        //         CompressionAlgorithmV1::ZSTD,
        //         uncompressed_size,
        //     )),
        //     Some(EncryptionV1::Aes256Ctr64LE(key_digest)),
        //     checksum,
        //     shape.to_vec(),
        //     download_size,
        // );
        // todo!();
        // let shard_commit = ShardCommit::new_v1(shard.shard_ref().clone(), commit_data);
        // // TODO: remove unwraps
        // let signed_commit = Signed::new(shard_commit, Scope::ShardCommit, &self.keypair).unwrap();
        // // TODO: remove unwraps
        // let verified_signed_commit = Verified::from_trusted(signed_commit).unwrap();

        // async fn get_shard_commit_signatures<C: EncoderInternalNetworkClient>(
        //     client: Arc<C>,
        //     peer: NetworkingIndex,
        //     shard_commit: Verified<Signed<ShardCommit>>,
        // ) -> ShardResult<Verified<Signature<Signed<ShardCommit>>>> {
        //     let signature_bytes = client
        //         .get_shard_commit_signature(peer, &shard_commit, MESSAGE_TIMEOUT)
        //         .await?;
        //     let signature: Signature<Signed<ShardCommit>> =
        //         bcs::from_bytes(&signature_bytes).map_err(ShardError::MalformedType)?;
        //     let verification_fn = |_signature: &Signature<Signed<ShardCommit>>| {
        //         // TODO: actually verify the signature
        //         unimplemented!()
        //     };
        //     // TODO: remove unwraps
        //     let verified_signature =
        //         Verified::new(signature, signature_bytes, verification_fn).unwrap();
        //     Ok(verified_signature)
        // }

        // let signatures = self
        //     .broadcaster
        //     .collect_signatures(
        //         verified_signed_commit,
        //         shard.encoders(),
        //         get_shard_commit_signatures,
        //     )
        //     .await?;

        // async fn send_shard_commit_certificates<C: EncoderInternalNetworkClient>(
        //     client: Arc<C>,
        //     peer: NetworkingIndex,
        //     shard_commit_certificate: Verified<Certified<Signed<ShardCommit>>>,
        // ) -> ShardResult<()> {
        //     let _ = client
        //         .send_shard_commit_certificate(peer, &shard_commit_certificate, MESSAGE_TIMEOUT)
        //         .await?;
        //     Ok(())
        // }

        // // // TODO: add a binder that converts a
        // // let shard_commit_certificate = Verified::new_from_trusted(ShardCertificate::new_v1(
        // //     inner,
        // //     indices,
        // //     aggregate_signature,
        // // ))?;

        // // let _ = self
        // //     .broadcaster
        // //     .broadcast(
        // //         shard_commit_certificate,
        // //         shard.members(),
        // //         send_shard_commit_certificates,
        // //     )
        // //     .await?;

        // // WHAT IS LEFT:
        // // 1. convert commit to certificate
        // // 2. broadcast the certificate

        Ok(())
    }

    pub async fn process_shard_commit_certificate(
        &self,
        peer: NetworkingIndex,
        shard: Shard,
        shard_commit_certificate: Verified<
            Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        >,
    ) -> ShardResult<()> {
        // unimplemented!();
        // let data = shard_commit_certificate.data();
        // let cancellation = CancellationToken::new();

        // // TODO: fix this
        // let probe_checksum = Checksum::default();
        // let probe_path = ObjectPath::from_checksum(probe_checksum);

        // let probe_bytes = self
        //     .downloader
        //     .process(
        //         DownloaderInput::new(peer, probe_path.clone()),
        //         cancellation.clone(),
        //     )
        //     .await?;

        // let PROBE_SIZE = 1024_usize * 10;
        // // TODO: fix this

        // let decompressed_probe_bytes = self
        //     .compressor
        //     .process(
        //         CompressorInput::Decompress(probe_bytes, PROBE_SIZE),
        //         cancellation.clone(),
        //     )
        //     .await?;

        // let _ = self
        //     .storage
        //     .process(
        //         StorageProcessorInput::Store(probe_path, decompressed_probe_bytes),
        //         cancellation.clone(),
        //     )
        //     .await?;

        // let data_path: ObjectPath = ObjectPath::from_checksum(data.checksum());

        // let downloader_input = DownloaderInput::new(peer, data_path.clone());

        // let encrypted_embedding_bytes = self
        //     .downloader
        //     .process(downloader_input, cancellation.clone())
        //     .await?;
        // let _ = self
        //     .storage
        //     .process(
        //         StorageProcessorInput::Store(data_path, encrypted_embedding_bytes),
        //         cancellation.clone(),
        //     )
        //     .await?;

        // // TODO:
        // // 1. mark completed
        // // 2. when all completed, reveal your own encryption key
        // // 3. check if reveal already exists, if yes, proceed to processing the reveal

        Ok(())
    }

    pub async fn process_shard_reveal_certificate(
        &self,
        peer: NetworkingIndex,
        shard: Shard,
        encrypted_data_checksum: Checksum,
        shard_reveal_certificate: Verified<
            Certified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
        >,
    ) -> ShardResult<()> {
        let cancellation = CancellationToken::new();
        let data_path: ObjectPath = ObjectPath::from_checksum(encrypted_data_checksum);
        let DATA_SIZE = 1024_usize * 10;
        // TODO: need to

        if let StorageProcessorOutput::Get(encrypted_bytes) = self
            .storage
            .process(StorageProcessorInput::Get(data_path), cancellation.clone())
            .await?
        {
            let decrypted_bytes = self
                .encryptor
                .process(
                    EncryptionInput::Encrypt(
                        shard_reveal_certificate.key().clone(),
                        encrypted_bytes,
                    ),
                    cancellation.clone(),
                )
                .await?;

            let embedding = self
                .compressor
                .process(
                    CompressorInput::Decompress(decrypted_bytes, DATA_SIZE),
                    cancellation.clone(),
                )
                .await?;
        }

        // decrypt the data
        // decompress the data
        // mark complete when data has been applied to every probe and store the intermediate responses

        // schedule the embedding to be probed by all

        // when all have been completed, trigger the calculation of the final endorsement and broadcast to all
        println!("{:?}", shard_reveal_certificate);
        Ok(())
    }

    // pub async fn process_shard_removal_certificate(
    //     &self,
    //     shard_removal_certificate: Verified<Certified<ShardRemoval>>,
    // ) {
    //     println!("{:?}", shard_removal_certificate);
    // }

    // pub async fn process_shard_endorsement_certificate(
    //     &self,
    //     shard_endorsement_certificate: Verified<
    //         Certified<Signed<ShardEndorsement, min_sig::BLS12381Signature>>,
    //     >,
    // ) {
    //     // print the endorsement certificate
    //     // just reach this point
    //     println!("{:?}", shard_endorsement_certificate);
    // }

    // pub async fn process_shard_completion_proof(
    //     &self,
    //     shard_completion_proof: Verified<ShardCompletionProof>,
    // ) {
    //     println!("{:?}", shard_completion_proof);
    // }
}

impl<C: EncoderInternalNetworkClient, M: Model, B: ObjectStorage, BC: ObjectNetworkClient> Clone
    for EncoderCore<C, M, B, BC>
{
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            downloader: self.downloader.clone(),
            encryptor: self.encryptor.clone(),
            compressor: self.compressor.clone(),
            model: self.model.clone(),
            storage: self.storage.clone(),
            broadcaster: self.broadcaster.clone(),
            keypair: self.keypair.clone(),
        }
    }
}
