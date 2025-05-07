use std::{future::Future, path::Path, sync::Arc};

use model::client::{MockModelClient, ModelClient};
use objects::{
    networking::{
        http_network::{ObjectHttpClient, ObjectHttpManager},
        ObjectNetworkManager, ObjectNetworkService,
    },
    storage::{filesystem::FilesystemObjectStorage, memory::MemoryObjectStore},
};
use probe::messaging::service::MockProbeService;
use probe::messaging::tonic::{ProbeTonicClient, ProbeTonicManager};
use probe::messaging::ProbeManager;
use quick_cache::sync::Cache;
use shared::{
    crypto::keys::{EncoderKeyPair, PeerKeyPair},
    digest::Digest,
    entropy::EntropyVDF,
};
use soma_network::multiaddr::Multiaddr;
use soma_tls::AllowPublicKeys;
use tokio::sync::Semaphore;
use tracing::{info, warn};
use types::committee::Committee;

use crate::{
    actors::{
        pipelines::{
            broadcast::BroadcastProcessor, commit::CommitProcessor,
            commit_votes::CommitVotesProcessor, evaluation::EvaluationProcessor,
            input::InputProcessor, reveal::RevealProcessor, reveal_votes::RevealVotesProcessor,
            scores::ScoresProcessor,
        },
        workers::{
            compression::CompressionProcessor, downloader, encryption::EncryptionProcessor,
            model::ModelProcessor, storage::StorageProcessor, vdf::VDFProcessor,
        },
        ActorManager,
    },
    compression::zstd_compressor::ZstdCompressor,
    datastore::{mem_store::MemStore, Store},
    encryption::aes_encryptor::Aes256Ctr64LEEncryptor,
    intelligence::model::python::{PythonInterpreter, PythonModule},
    messaging::{
        external_service::EncoderExternalService,
        internal_service::EncoderInternalService,
        tonic::{
            external::EncoderExternalTonicManager,
            internal::{ConnectionsInfo, EncoderInternalTonicClient, EncoderInternalTonicManager},
            NetworkingInfo,
        },
        EncoderExternalNetworkManager, EncoderInternalNetworkManager,
    },
    types::{context::Context, parameters::Parameters, shard_verifier},
};

use self::{
    downloader::Downloader,
    shard_verifier::{ShardAuthToken, ShardVerifier},
};

use super::{
    encoder_validator_client::EncoderValidatorClient,
    internal_broadcaster::Broadcaster,
    pipeline_dispatcher::{ExternalPipelineDispatcher, InternalPipelineDispatcher},
    shard_tracker::ShardTracker,
};

#[cfg(msim)]
use msim::task::NodeId;
#[cfg(msim)]
use simulator::SimState;

// pub struct Encoder(EncoderNode<ActorInternalPipelineDispatcher<EncoderTonicClient, PythonModule, FilesystemObjectStorage, ObjectHttpClient>, EncoderTonicManager>);

// impl Encoder {
//     pub async fn start(
//         context: Arc<Context>,
//         network_keypair: NetworkKeyPair,
//         protocol_keypair: ProtocolKeyPair,
//         project_root: &Path,
//         entry_point: &Path,
//     ) -> Self {
//         let encoder_node: EncoderNode<ActorInternalPipelineDispatcher<EncoderTonicClient, PythonModule, FilesystemObjectStorage, ObjectHttpClient>, EncoderTonicManager> =
//             EncoderNode::start(
//                 context,
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
    internal_network_manager: EncoderInternalTonicManager,
    external_network_manager: EncoderExternalTonicManager,
    object_network_manager: ObjectHttpManager,
    probe_network_manager: ProbeTonicManager,
    store: Arc<dyn Store>,
    pub context: Context,
    object_storage: Arc<MemoryObjectStore>,
    /// Client for fetching committees from validator nodes
    validator_client: Option<Arc<tokio::sync::Mutex<EncoderValidatorClient>>>,

    #[cfg(msim)]
    sim_state: SimState,
}

impl EncoderNode {
    pub async fn start(
        context: Context,
        encoder_keypair: EncoderKeyPair,
        networking_info: NetworkingInfo,
        parameters: Arc<Parameters>,
        object_parameters: Arc<objects::parameters::Parameters>,
        probe_parameters: Arc<probe::parameters::Parameters>,
        peer_keypair: PeerKeyPair,
        internal_address: Multiaddr,
        external_address: Multiaddr,
        object_address: Multiaddr,
        probe_address: Multiaddr,
        allower: AllowPublicKeys,
        connections_info: ConnectionsInfo,
        project_root: &Path,
        entry_point: &Path,
        validator_rpc_address: Option<types::multiaddr::Multiaddr>,
        genesis_committee: Option<Committee>,
    ) -> Self {
        let mut internal_network_manager = EncoderInternalTonicManager::new(
            networking_info.clone(),
            parameters.clone(),
            peer_keypair.clone(),
            internal_address.clone(),
            allower.clone(),
            connections_info,
        );

        let messaging_client = <EncoderInternalTonicManager as EncoderInternalNetworkManager<
            EncoderInternalService<
                InternalPipelineDispatcher<
                    EncoderInternalTonicClient,
                    ObjectHttpClient,
                    FilesystemObjectStorage,
                    ProbeTonicClient,
                >,
            >,
        >>::client(&internal_network_manager);

        let object_storage = Arc::new(MemoryObjectStore::new_for_test());
        let object_network_service: ObjectNetworkService<MemoryObjectStore> =
            ObjectNetworkService::new(object_storage.clone());

        let mut object_network_manager = <ObjectHttpManager as ObjectNetworkManager<
            MemoryObjectStore,
        >>::new(
            peer_keypair.clone(), object_parameters, allower.clone()
        )
        .unwrap();

        object_network_manager
            .start(&object_address, object_network_service)
            .await;

        let object_client = <ObjectHttpManager as ObjectNetworkManager<MemoryObjectStore>>::client(
            &object_network_manager,
        );

        ///////////////////////////
        let mut probe_network_manager =
            ProbeTonicManager::new(probe_parameters.clone(), probe_address.clone());

        let probe_service = Arc::new(MockProbeService::new());
        probe_network_manager.start(probe_service).await;

        let probe_client = Arc::new(
            ProbeTonicClient::new(probe_address.clone(), probe_parameters.clone())
                .await
                .unwrap(),
        );
        ////////////////////////////

        let encoder_keypair = Arc::new(encoder_keypair);

        let default_buffer = 100_usize;
        let default_concurrency = 100_usize;

        let download_processor = Downloader::new(
            default_concurrency,
            object_client.clone(),
            object_storage.clone(),
        );
        let downloader_manager = ActorManager::new(default_buffer, download_processor);
        let downloader_handle = downloader_manager.handle();

        let encryptor_processor: EncryptionProcessor<Aes256Ctr64LEEncryptor> =
            EncryptionProcessor::new(Arc::new(Aes256Ctr64LEEncryptor::new()));
        let encryptor_manager = ActorManager::new(default_buffer, encryptor_processor);
        let encryptor_handle = encryptor_manager.handle();

        let compressor_processor = CompressionProcessor::new(Arc::new(ZstdCompressor::new()));
        let compressor_manager = ActorManager::new(default_buffer, compressor_processor);
        let compressor_handle = compressor_manager.handle();

        // let python_interpreter = PythonInterpreter::new(project_root).unwrap();
        // let model = python_interpreter.new_module(entry_point).unwrap();

        // let model_processor = ModelProcessor::new(model, None);
        // let model_manager = ActorManager::new(default_buffer, model_processor);
        // let model_handle = model_manager.handle();

        let storage_processor = StorageProcessor::new(object_storage.clone(), None);
        let storage_manager = ActorManager::new(default_buffer, storage_processor);
        let storage_handle = storage_manager.handle();

        let vdf = EntropyVDF::new(1);
        let vdf_processor = VDFProcessor::new(vdf, 1);
        let vdf_handle = ActorManager::new(1, vdf_processor).handle();
        let store = Arc::new(MemStore::new());

        let broadcaster = Arc::new(Broadcaster::new(
            messaging_client.clone(),
            Arc::new(Semaphore::new(default_concurrency)),
            encoder_keypair.public(),
        ));

        let shard_tracker = Arc::new(ShardTracker::new(
            Arc::new(Semaphore::new(default_concurrency)),
            store.clone(),
            encoder_keypair.clone(),
        ));

        let broadcast_processor = BroadcastProcessor::new(
            broadcaster.clone(),
            store.clone(),
            encoder_keypair.clone(),
            shard_tracker.clone(),
        );
        let broadcast_manager = ActorManager::new(default_buffer, broadcast_processor);
        let broadcast_handle = broadcast_manager.handle();

        let evaluation_processor = EvaluationProcessor::new(
            store.clone(),
            broadcast_handle.clone(),
            encoder_keypair.clone(),
            storage_handle.clone(),
            probe_client,
        );
        let evaluation_handle = ActorManager::new(default_buffer, evaluation_processor).handle();

        // Now update ShardTracker with evaluation_handle and broadcast handle
        shard_tracker.set_broadcast_handle(broadcast_handle.clone());
        shard_tracker.set_evaluation_handle(evaluation_handle);

        let model_client = Arc::new(MockModelClient {});

        let input_processor = InputProcessor::new(
            downloader_handle.clone(),
            compressor_handle,
            broadcast_handle,
            model_client,
            // model_handle,
            encryptor_handle,
            encoder_keypair.clone(),
            storage_handle.clone(),
        );

        let input_handle = ActorManager::new(default_buffer, input_processor).handle();

        let commit_processor = CommitProcessor::new(
            store.clone(),
            shard_tracker.clone(),
            downloader_handle.clone(),
        );

        let commit_votes_processor =
            CommitVotesProcessor::new(store.clone(), shard_tracker.clone());

        let reveal_processor = RevealProcessor::new(store.clone(), shard_tracker.clone());
        let reveal_votes_processor =
            RevealVotesProcessor::new(store.clone(), shard_tracker.clone());

        let scores_processor = ScoresProcessor::new(store.clone(), shard_tracker.clone());

        let commit_manager = ActorManager::new(default_buffer, commit_processor);
        let commit_votes_manager = ActorManager::new(default_buffer, commit_votes_processor);
        let reveal_manager = ActorManager::new(default_buffer, reveal_processor);
        let reveal_votes_manager = ActorManager::new(default_buffer, reveal_votes_processor);
        let scores_manager = ActorManager::new(default_buffer, scores_processor);

        let commit_handle = commit_manager.handle();
        let commit_votes_handle = commit_votes_manager.handle();
        let reveal_handle = reveal_manager.handle();
        let reveal_votes_handle = reveal_votes_manager.handle();
        let scores_handle = scores_manager.handle();

        let pipeline_dispatcher = InternalPipelineDispatcher::new(
            commit_handle,
            commit_votes_handle,
            reveal_handle,
            reveal_votes_handle,
            scores_handle,
        );
        let verifier = Arc::new(ShardVerifier::new(
            100,
            vdf_handle,
            encoder_keypair.public(),
        ));

        let internal_network_service = Arc::new(EncoderInternalService::new(
            context.clone(),
            store.clone(),
            pipeline_dispatcher,
            verifier.clone(),
        ));
        internal_network_manager
            .start(internal_network_service)
            .await;

        // Now create the external manager and service
        let mut external_network_manager = EncoderExternalTonicManager::new(
            parameters.clone(),
            peer_keypair.clone(),
            external_address.clone(),
            allower.clone(),
        );

        // Create the external service using the same context and verifier
        let external_network_service = Arc::new(EncoderExternalService::new(
            Arc::new(context.clone()),
            ExternalPipelineDispatcher::new(input_handle),
            verifier,
        ));

        // Start the external manager with the service
        external_network_manager
            .start(external_network_service)
            .await;

        // Initialize validator client if configuration is provided
        let validator_client =
            if let (Some(address), Some(committee)) = (validator_rpc_address, genesis_committee) {
                match EncoderValidatorClient::new(&address, committee).await {
                    Ok(client) => {
                        info!("Successfully connected to validator node for committee updates");
                        Some(Arc::new(tokio::sync::Mutex::new(client)))
                    }
                    Err(e) => {
                        warn!("Failed to create validator client: {}", e);
                        None
                    }
                }
            } else {
                None
            };

        Self {
            internal_network_manager,
            external_network_manager,
            object_network_manager,
            store,
            object_storage,
            context,
            probe_network_manager,
            validator_client,
            #[cfg(msim)]
            sim_state: Default::default(),
        }
    }

    pub(crate) async fn stop(mut self) {
        <EncoderInternalTonicManager as EncoderInternalNetworkManager<
            EncoderInternalService<
                InternalPipelineDispatcher<
                    EncoderInternalTonicClient,
                    ObjectHttpClient,
                    MemoryObjectStore,
                    ProbeTonicClient,
                >,
            >,
        >>::stop(&mut self.internal_network_manager)
        .await;

        <EncoderExternalTonicManager as EncoderExternalNetworkManager<
            EncoderExternalService<
                ExternalPipelineDispatcher<
                    EncoderInternalTonicClient,
                    ObjectHttpClient,
                    MockModelClient,
                    MemoryObjectStore,
                    ProbeTonicClient,
                >,
            >,
        >>::stop(&mut self.external_network_manager)
        .await;

        // TODO: self.object_network_manager.stop().await;
    }
}

/// Wrap EncoderNode to allow correct access to EncoderNode in simulator tests.
pub struct EncoderNodeHandle {
    node: Option<Arc<EncoderNode>>,
    shutdown_on_drop: bool,
}

impl EncoderNodeHandle {
    pub fn new(node: Arc<EncoderNode>) -> Self {
        Self {
            node: Some(node),
            shutdown_on_drop: false,
        }
    }

    pub fn inner(&self) -> &Arc<EncoderNode> {
        self.node.as_ref().unwrap()
    }

    pub fn with<T>(&self, cb: impl FnOnce(&EncoderNode) -> T) -> T {
        let _guard = self.guard();
        cb(self.inner())
    }

    pub fn store(&self) -> Arc<dyn Store> {
        self.with(|soma_node| soma_node.store.clone())
    }

    pub fn object_storage(&self) -> Arc<MemoryObjectStore> {
        self.with(|soma_node| soma_node.object_storage.clone())
    }

    pub fn shutdown_on_drop(&mut self) {
        self.shutdown_on_drop = true;
    }
}

impl Clone for EncoderNodeHandle {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            shutdown_on_drop: false,
        }
    }
}

#[cfg(not(msim))]
impl EncoderNodeHandle {
    // Must return something to silence lints above at `let _guard = ...`
    fn guard(&self) -> u32 {
        0
    }

    pub async fn with_async<'a, F, R, T>(&'a self, cb: F) -> T
    where
        F: FnOnce(&'a EncoderNode) -> R,
        R: Future<Output = T>,
    {
        cb(self.inner()).await
    }
}

#[cfg(msim)]
impl EncoderNodeHandle {
    fn guard(&self) -> msim::runtime::NodeEnterGuard {
        self.inner().sim_state.sim_node.enter_node()
    }

    pub async fn with_async<'a, F, R, T>(&'a self, cb: F) -> T
    where
        F: FnOnce(&'a EncoderNode) -> R,
        R: Future<Output = T>,
    {
        let fut = cb(self.node.as_ref().unwrap());
        self.inner()
            .sim_state
            .sim_node
            .await_future_in_node(fut)
            .await
    }
}

#[cfg(msim)]
impl Drop for EncoderNodeHandle {
    fn drop(&mut self) {
        if self.shutdown_on_drop {
            let node_id = self.inner().sim_state.sim_node.id();
            msim::runtime::Handle::try_current().map(|h| h.delete_node(node_id));
        }
    }
}

impl From<Arc<EncoderNode>> for EncoderNodeHandle {
    fn from(node: Arc<EncoderNode>) -> Self {
        EncoderNodeHandle::new(node)
    }
}

#[cfg(msim)]
mod simulator {
    use super::*;
    use std::sync::atomic::AtomicBool;
    pub(super) struct SimState {
        pub sim_node: msim::runtime::NodeHandle,
    }

    impl Default for SimState {
        fn default() -> Self {
            Self {
                sim_node: msim::runtime::NodeHandle::current(),
            }
        }
    }
}
