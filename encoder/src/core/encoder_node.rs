use std::{fs::File, marker::PhantomData, path::Path, sync::Arc};


use crate::{
    actors::{
        pipelines::shard_input,
        workers::{
            compression::CompressionProcessor, downloader, encryption::EncryptionProcessor,
            model::ModelProcessor, storage::StorageProcessor,
        },
        ActorManager,
    },
    compression::zstd_compressor::ZstdCompressor,
    crypto::{keys::NetworkKeyPair, AesKey},
    encryption::{aes_encryptor::Aes256Ctr64LEEncryptor, Encryptor},
    intelligence::model::{
        python::{PythonInterpreter, PythonModule},
        Model,
    },
    networking::{
        blob::{
            http_network::{ObjectHttpClient, ObjectHttpManager},
            DirectNetworkService, ObjectNetworkClient, ObjectNetworkManager, ObjectNetworkService,
        },
        messaging::{
            tonic_network::{EncoderTonicClient, EncoderTonicManager},
            EncoderNetworkClient, EncoderNetworkManager,
        },
    },
    storage::{
        datastore::mem_store::MemStore,
        object::{filesystem::FilesystemObjectStorage, ObjectStorage},
    },
    types::{context::EncoderContext, shard},
    ProtocolKeyPair,
};

use self::{downloader::Downloader, shard_input::ShardInputProcessor};

use super::{
    broadcaster::Broadcaster,
    encoder_core::EncoderCore,
    encoder_service::EncoderService,
    pipeline_dispatcher::{ActorPipelineDispatcher, PipelineDispatcher},
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

pub struct EncoderNode
{
    network_manager: EncoderTonicManager,
}

impl EncoderNode {
    pub(crate) async fn start(
        encoder_context: Arc<EncoderContext>,
        network_keypair: NetworkKeyPair,
        protocol_keypair: ProtocolKeyPair,
        project_root: &Path,
        entry_point: &Path,
    ) -> Self {
        let mut network_manager= EncoderTonicManager::new(encoder_context.clone(), network_keypair);

        let messaging_client = <EncoderTonicManager as EncoderNetworkManager<EncoderService<ActorPipelineDispatcher<EncoderTonicClient, PythonModule, FilesystemObjectStorage, ObjectHttpClient>, MemStore>>>::client(&network_manager);

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
        let protocol_keypair: Arc<ProtocolKeyPair> = Arc::new(protocol_keypair);

        let default_buffer = 100_usize;
        let default_concurrency = 100_usize;

        let broadcaster = Arc::new(Broadcaster::new(
            encoder_context.clone(),
            messaging_client.clone(),
        ));

        let download_processor = Downloader::new(default_concurrency, blob_client.clone());
        let downloader_manager = ActorManager::new(default_buffer, download_processor);
        let downloader_handle = downloader_manager.handle();

        let encryptor_processor: EncryptionProcessor<AesKey, Aes256Ctr64LEEncryptor> =
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
            messaging_client,
            broadcaster,
            downloader_handle,
            encryptor_handle,
            compressor_handle,
            model_handle,
            storage_handle,
            protocol_keypair.clone(),
        );

        let shard_input_processor = ShardInputProcessor::new(core, default_concurrency);
        let shard_input_manager = ActorManager::new(default_buffer, shard_input_processor);
        let shard_input_handle = shard_input_manager.handle();

        let pipeline_dispatcher = ActorPipelineDispatcher::new(shard_input_handle);

        let store = Arc::new(MemStore::new());
        let network_service = Arc::new(EncoderService::new(
            encoder_context,
            Arc::new(pipeline_dispatcher),
            store,
            protocol_keypair,
        ));
        network_manager.start(network_service).await;
        Self {
            network_manager,
        }
    }

    pub(crate) async fn stop(mut self) {
        <EncoderTonicManager as EncoderNetworkManager<EncoderService<
        ActorPipelineDispatcher<
            EncoderTonicClient,
            PythonModule,
            FilesystemObjectStorage,
            ObjectHttpClient
        >,
        MemStore
    >>>::stop(&mut self.network_manager).await;

    }
}
