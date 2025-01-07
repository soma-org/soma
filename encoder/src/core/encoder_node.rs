use std::{path::Path, sync::Arc};

use crate::{
    actors::{
        compression::Compressor, downloader, encryption::Encryptor, model::ModelProcessor,
        storage::StorageProcessor, ActorManager,
    },
    crypto::{keys::NetworkKeyPair, AesKey},
    intelligence::model::python::{PythonInterpreter, PythonModule},
    networking::{
        blob::{
            http_network::ObjectHttpManager, ObjectNetworkManager, ObjectNetworkService,
            DirectNetworkService,
        },
        messaging::{tonic_network::EncoderTonicManager, EncoderNetworkManager},
    },
    storage::{
        blob::{
            compression::ZstdCompressor, encryption::AesEncryptor,
            filesystem::FilesystemObjectStorage,
        },
        datastore::mem_store::MemStore,
    },
    types::context::EncoderContext,
    ProtocolKeyPair,
};

use self::downloader::Downloader;

use super::{
    broadcaster::Broadcaster,
    encoder_core::EncoderCore,
    encoder_service::EncoderService,
    task_manager::{ChannelTaskDispatcher, TaskManagerHandle},
};

pub struct Encoder(EncoderNode<EncoderTonicManager>);

impl Encoder {
    pub async fn start(
        encoder_context: Arc<EncoderContext>,
        network_keypair: NetworkKeyPair,
        protocol_keypair: ProtocolKeyPair,
        project_root: &Path,
        entry_point: &Path,
    ) -> Self {
        let encoder_node: EncoderNode<EncoderTonicManager> = EncoderNode::start(
            encoder_context,
            network_keypair,
            protocol_keypair,
            project_root,
            entry_point,
        )
        .await;
        Self(encoder_node)
    }
    pub async fn stop(self) {
        self.0.stop().await;
    }
}

pub(crate) struct EncoderNode<N>
where
    N: EncoderNetworkManager<EncoderService<ChannelTaskDispatcher, MemStore>>,
{
    task_manager_handle: TaskManagerHandle,
    network_manager: N,
}

impl<N> EncoderNode<N>
where
    N: EncoderNetworkManager<EncoderService<ChannelTaskDispatcher, MemStore>>,
{
    pub(crate) async fn start(
        encoder_context: Arc<EncoderContext>,
        network_keypair: NetworkKeyPair,
        protocol_keypair: ProtocolKeyPair,
        project_root: &Path,
        entry_point: &Path,
    ) -> Self {
        let mut network_manager = N::new(encoder_context.clone(), network_keypair);
        let messaging_client: Arc<
            <N as EncoderNetworkManager<EncoderService<ChannelTaskDispatcher, MemStore>>>::Client,
        > = network_manager.client();

        let blob_storage = Arc::new(FilesystemObjectStorage::new("base_path"));
        let blob_network_service: DirectNetworkService<FilesystemObjectStorage> =
            DirectNetworkService::new(blob_storage.clone());
        let mut blob_network_manager: ObjectHttpManager<DirectNetworkService<FilesystemObjectStorage>> =
            ObjectHttpManager::new(encoder_context.clone()).unwrap();
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

        let encryptor_processor: Encryptor<AesKey, AesEncryptor> =
            Encryptor::new(Arc::new(AesEncryptor::new()));
        let encryptor_manager = ActorManager::new(default_buffer, encryptor_processor);
        let encryptor_handle = encryptor_manager.handle();

        let compressor_processor = Compressor::new(Arc::new(ZstdCompressor::new()));
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
        let (task_dispatcher, task_manager_handle) = ChannelTaskDispatcher::start(core);
        let task_dispatcher = Arc::new(task_dispatcher);
        let store = Arc::new(MemStore::new());
        let network_service = Arc::new(EncoderService::new(
            encoder_context,
            task_dispatcher,
            store,
            protocol_keypair,
        ));
        network_manager.start(network_service).await;
        Self {
            task_manager_handle,
            network_manager,
        }
    }

    pub(crate) async fn stop(mut self) {
        self.network_manager.stop().await;
    }
}
