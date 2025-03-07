use std::{path::Path, sync::Arc};

use quick_cache::sync::Cache;
use shared::{
    crypto::keys::{EncoderKeyPair, NetworkKeyPair},
    digest::Digest,
    entropy::EntropyVDF,
};

use crate::{
    actors::{
        pipelines::{
            certified_commit::CertifiedCommitProcessor, commit_votes::CommitVotesProcessor,
            evaluation::EvaluationProcessor, reveal::RevealProcessor,
            reveal_votes::RevealVotesProcessor, scores::ScoresProcessor,
        },
        workers::{
            broadcaster::BroadcasterProcessor, compression::CompressionProcessor, downloader,
            encryption::EncryptionProcessor, model::ModelProcessor, storage::StorageProcessor,
            vdf::VDFProcessor,
        },
        ActorManager,
    },
    compression::zstd_compressor::ZstdCompressor,
    encryption::aes_encryptor::Aes256Ctr64LEEncryptor,
    intelligence::model::python::PythonInterpreter,
    networking::{
        messaging::{
            tonic_network::{EncoderInternalTonicClient, EncoderInternalTonicManager},
            EncoderInternalNetworkManager,
        },
        object::{
            http_network::{ObjectHttpClient, ObjectHttpManager},
            DirectNetworkService, ObjectNetworkManager,
        },
    },
    storage::{datastore::mem_store::MemStore, object::filesystem::FilesystemObjectStorage},
    types::{encoder_context::EncoderContext, shard_verifier},
};

use self::{
    downloader::Downloader,
    shard_verifier::{ShardAuthToken, ShardVerifier, VerificationStatus},
    slot_tracker::SlotTracker,
};

use super::{
    broadcaster::Broadcaster, encoder_core::EncoderCore, encoder_service::EncoderInternalService,
    pipeline_dispatcher::PipelineDispatcher, slot_tracker,
};

// pub struct Encoder(EncoderNode<ActorPipelineDispatcher<EncoderTonicClient, PythonModule, FilesystemObjectStorage, ObjectHttpClient>, EncoderTonicManager>);

// impl Encoder {
//     pub async fn start(
//         encoder_context: Arc<EncoderContext>,
//         network_keypair: NetworkKeyPair,
//         protocol_keypair: ProtocolKeyPair,
//         project_root: &Path,
//         entry_point: &Path,
//     ) -> Self {
//         let encoder_node: EncoderNode<ActorPipelineDispatcher<EncoderTonicClient, PythonModule, FilesystemObjectStorage, ObjectHttpClient>, EncoderTonicManager> =
//             EncoderNode::start(
//                 encoder_context,
//                 network_keypair,
//                 protocol_keypair,
//                 project_root,
//                 entry_point,
//             )
//             .await;
//         Self(encoder_node)
//     }
//     pub async fn stop(self) {
//         self.0.stop().await;
//     }
// }

pub struct EncoderNode {
    network_manager: EncoderInternalTonicManager,
}

impl EncoderNode {
    pub(crate) async fn start(
        encoder_context: Arc<EncoderContext>,
        network_keypair: NetworkKeyPair,
        encoder_keypair: EncoderKeyPair,
        project_root: &Path,
        entry_point: &Path,
    ) -> Self {
        let mut network_manager =
            EncoderInternalTonicManager::new(encoder_context.clone(), network_keypair);

        let messaging_client = <EncoderInternalTonicManager as EncoderInternalNetworkManager<
            EncoderInternalService<
                PipelineDispatcher<
                    EncoderInternalTonicClient,
                    ObjectHttpClient,
                    FilesystemObjectStorage,
                >,
            >,
        >>::client(&network_manager);

        // let messaging_client = network_manager.client();

        let blob_storage = Arc::new(FilesystemObjectStorage::new("base_path"));
        let blob_network_service: DirectNetworkService<FilesystemObjectStorage> =
            DirectNetworkService::new(blob_storage.clone());
        let mut blob_network_manager: ObjectHttpManager<
            DirectNetworkService<FilesystemObjectStorage>,
        > = ObjectHttpManager::new(encoder_context.clone()).unwrap();
        // tokio::spawn(async move {
        //     blob_network_manager.start(Arc::new(blob_network_service)).await
        // });

        let blob_client = blob_network_manager.client();
        let encoder_keypair = Arc::new(encoder_keypair);

        let default_buffer = 100_usize;
        let default_concurrency = 100_usize;

        let broadcaster = Arc::new(Broadcaster::new(
            encoder_context.clone(),
            messaging_client.clone(),
        ));

        let download_processor = Downloader::new(default_concurrency, blob_client.clone());
        let downloader_manager = ActorManager::new(default_buffer, download_processor);
        let downloader_handle = downloader_manager.handle();

        let encryptor_processor: EncryptionProcessor<Aes256Ctr64LEEncryptor> =
            EncryptionProcessor::new(Arc::new(Aes256Ctr64LEEncryptor::new()));
        let encryptor_manager = ActorManager::new(default_buffer, encryptor_processor);
        let encryptor_handle = encryptor_manager.handle();

        let compressor_processor = CompressionProcessor::new(Arc::new(ZstdCompressor::new()));
        let compressor_manager = ActorManager::new(default_buffer, compressor_processor);
        let compressor_handle = compressor_manager.handle();

        let python_interpreter = PythonInterpreter::new(project_root).unwrap();
        let model = python_interpreter.new_module(entry_point).unwrap();

        let model_processor = ModelProcessor::new(model, None);
        let model_manager = ActorManager::new(default_buffer, model_processor);
        let model_handle = model_manager.handle();

        let storage_processor = StorageProcessor::new(blob_storage, None);
        let storage_manager = ActorManager::new(default_buffer, storage_processor);
        let storage_handle = storage_manager.handle();

        let core = EncoderCore::new(
            messaging_client.clone(),
            broadcaster,
            downloader_handle.clone(),
            encryptor_handle.clone(),
            compressor_handle.clone(),
            model_handle,
            storage_handle.clone(),
            encoder_keypair.clone(),
        );
        let vdf = EntropyVDF::new(1);
        let vdf_processor = VDFProcessor::new(vdf, 1);
        let vdf_handle = ActorManager::new(1, vdf_processor).handle();
        let store = Arc::new(MemStore::new());

        let broadcaster = Broadcaster::new(encoder_context.clone(), messaging_client);
        let broadcast_processor = BroadcasterProcessor::new(
            default_concurrency,
            broadcaster,
            store.clone(),
            encoder_context.own_encoder_index,
            encoder_keypair.clone(),
        );
        let broadcaster_handle = ActorManager::new(default_buffer, broadcast_processor).handle();

        let evaluation_processor =
            EvaluationProcessor::new(store.clone(), broadcaster_handle.clone());
        let evaluation_handle = ActorManager::new(default_buffer, evaluation_processor).handle();

        let slot_tracker = SlotTracker::new(100);
        let certified_commit_processor = CertifiedCommitProcessor::new(
            100,
            store.clone(),
            slot_tracker.clone(),
            broadcaster_handle.clone(),
            downloader_handle.clone(),
            compressor_handle.clone(),
            storage_handle.clone(),
        );
        let commit_votes_processor = CommitVotesProcessor::new(
            store.clone(),
            encoder_context.own_encoder_index,
            broadcaster_handle.clone(),
        );
        let reveal_processor = RevealProcessor::new(
            100,
            store.clone(),
            slot_tracker,
            broadcaster_handle.clone(),
            storage_handle.clone(),
            compressor_handle.clone(),
            encryptor_handle.clone(),
        );
        let reveal_votes_processor = RevealVotesProcessor::new(
            store.clone(),
            encoder_context.own_encoder_index,
            evaluation_handle,
        );

        let scores_processor =
            ScoresProcessor::new(store.clone(), encoder_context.own_encoder_index);

        let certified_commit_manager =
            ActorManager::new(default_buffer, certified_commit_processor);
        let commit_votes_manager = ActorManager::new(default_buffer, commit_votes_processor);
        let reveal_manager = ActorManager::new(default_buffer, reveal_processor);
        let reveal_votes_manager = ActorManager::new(default_buffer, reveal_votes_processor);
        let scores_manager = ActorManager::new(default_buffer, scores_processor);

        let certified_commit_handle = certified_commit_manager.handle();
        let commit_votes_handle = commit_votes_manager.handle();
        let reveal_handle = reveal_manager.handle();
        let reveal_votes_handle = reveal_votes_manager.handle();
        let scores_handle = scores_manager.handle();

        let pipeline_dispatcher = PipelineDispatcher::new(
            certified_commit_handle,
            commit_votes_handle,
            reveal_handle,
            reveal_votes_handle,
            scores_handle,
        );
        let cache: Cache<Digest<ShardAuthToken>, VerificationStatus> = Cache::new(64);
        let verifier = ShardVerifier::new(cache);

        let network_service = Arc::new(EncoderInternalService::new(
            encoder_context,
            pipeline_dispatcher,
            vdf_handle,
            verifier,
            store,
            encoder_keypair,
        ));
        network_manager.start(network_service).await;
        Self { network_manager }
    }

    pub(crate) async fn stop(mut self) {
        <EncoderInternalTonicManager as EncoderInternalNetworkManager<
            EncoderInternalService<
                PipelineDispatcher<
                    EncoderInternalTonicClient,
                    ObjectHttpClient,
                    FilesystemObjectStorage,
                >,
            >,
        >>::stop(&mut self.network_manager)
        .await;
    }
}
