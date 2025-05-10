use std::{future::Future, path::Path, sync::Arc};

use model::client::MockModelClient;
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
use shared::{
    authority_committee::AuthorityCommittee,
    crypto::keys::{EncoderKeyPair, EncoderPublicKey, PeerKeyPair},
    entropy::EntropyVDF,
    probe::ProbeMetadata,
};
use soma_network::multiaddr::Multiaddr;
use soma_tls::AllowPublicKeys;
use tokio::sync::{Mutex, Semaphore};
use tracing::{error, info, warn};
use types::committee::Committee;

use crate::{
    actors::{
        pipelines::{
            commit::CommitProcessor, commit_votes::CommitVotesProcessor,
            evaluation::EvaluationProcessor, finality::FinalityProcessor, input::InputProcessor,
            reveal::RevealProcessor, reveal_votes::RevealVotesProcessor, scores::ScoresProcessor,
        },
        workers::{
            compression::CompressionProcessor, downloader, encryption::EncryptionProcessor,
            storage::StorageProcessor, vdf::VDFProcessor,
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
    sync::{
        committee_sync_manager::CommitteeSyncManager,
        encoder_validator_client::EncoderValidatorClient,
    },
    types::{
        context::{Committees, Context, InnerContext},
        encoder_committee::{Encoder, EncoderCommittee},
        parameters::Parameters,
        shard_verifier,
    },
};

use self::{
    downloader::Downloader,
    shard_verifier::{ShardAuthToken, ShardVerifier},
};

use super::{
    internal_broadcaster::Broadcaster,
    pipeline_dispatcher::{ExternalPipelineDispatcher, InternalPipelineDispatcher},
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
    committee_sync_manager: Arc<CommitteeSyncManager>,

    #[cfg(msim)]
    sim_state: SimState,
}

impl EncoderNode {
    pub async fn start(
        encoder_keypair: EncoderKeyPair,
        parameters: Arc<Parameters>,
        object_parameters: Arc<objects::parameters::Parameters>,
        probe_parameters: Arc<probe::parameters::Parameters>,
        peer_keypair: PeerKeyPair,
        internal_address: Multiaddr,
        external_address: Multiaddr,
        object_address: Multiaddr,
        probe_address: Multiaddr,
        project_root: &Path,
        entry_point: &Path,
        validator_rpc_address: types::multiaddr::Multiaddr,
        genesis_committee: Committee,
        epoch_duration_ms: u64,
    ) -> Self {
        let networking_info = NetworkingInfo::default();
        let connections_info = ConnectionsInfo::default();
        let allower = AllowPublicKeys::default();

        // Create minimal default context with empty committees
        // This creates a Context with epoch 0 and minimal valid structures
        let context = Self::create_default_context(encoder_keypair.public());

        let mut internal_network_manager = EncoderInternalTonicManager::new(
            networking_info.clone(),
            parameters.clone(),
            peer_keypair.clone(),
            internal_address.clone(),
            allower.clone(),
            connections_info.clone(),
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

        let recv_dedup_cache_capacity: usize = 1000;
        let send_dedup_cache_capacity: usize = 100;
        let finality_processor = FinalityProcessor::new(store.clone(), recv_dedup_cache_capacity);
        let finality_handle = ActorManager::new(default_buffer, finality_processor).handle();

        let scores_processor = ScoresProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            finality_handle.clone(),
            recv_dedup_cache_capacity,
            send_dedup_cache_capacity,
        );
        let scores_handle = ActorManager::new(default_buffer, scores_processor).handle();

        let evaluation_processor = EvaluationProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            storage_handle.clone(),
            scores_handle.clone(),
            probe_client,
            recv_dedup_cache_capacity,
        );
        let evaluation_handle = ActorManager::new(default_buffer, evaluation_processor).handle();

        let reveal_votes_processor = RevealVotesProcessor::new(
            store.clone(),
            evaluation_handle.clone(),
            recv_dedup_cache_capacity,
            send_dedup_cache_capacity,
        );
        let reveal_votes_handle =
            ActorManager::new(default_buffer, reveal_votes_processor).handle();

        let reveal_processor = RevealProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            reveal_votes_handle.clone(),
            recv_dedup_cache_capacity,
            send_dedup_cache_capacity,
        );
        let reveal_handle = ActorManager::new(default_buffer, reveal_processor).handle();

        let commit_votes_processor = CommitVotesProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            reveal_handle.clone(),
            recv_dedup_cache_capacity,
            send_dedup_cache_capacity,
        );
        let commit_votes_handle =
            ActorManager::new(default_buffer, commit_votes_processor).handle();

        let commit_processor = CommitProcessor::new(
            store.clone(),
            downloader_handle.clone(),
            broadcaster.clone(),
            commit_votes_handle.clone(),
            encoder_keypair.clone(),
            recv_dedup_cache_capacity,
            send_dedup_cache_capacity,
        );
        let commit_handle = ActorManager::new(default_buffer, commit_processor).handle();

        let model_client = Arc::new(MockModelClient {});

        let input_processor = InputProcessor::new(
            downloader_handle.clone(),
            compressor_handle,
            broadcaster.clone(),
            model_client,
            encryptor_handle,
            encoder_keypair.clone(),
            storage_handle.clone(),
            commit_handle.clone(),
        );

        let input_handle = ActorManager::new(default_buffer, input_processor).handle();

        let pipeline_dispatcher = InternalPipelineDispatcher::new(
            commit_handle,
            commit_votes_handle,
            reveal_handle,
            reveal_votes_handle,
            scores_handle,
            finality_handle,
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

        info!(
            "Creating validator client connecting to {}",
            validator_rpc_address
        );
        let validator_client =
            match EncoderValidatorClient::new(&validator_rpc_address, genesis_committee).await {
                Ok(client) => {
                    info!("Successfully connected to validator node for committee updates");
                    Arc::new(Mutex::new(client))
                }
                Err(e) => {
                    error!("Failed to create validator client: {}", e);
                    panic!("Validator client initialization failed: {}", e);
                }
            };

        // Initialize and start the committee sync manager
        let committee_sync_manager = CommitteeSyncManager::new(
            validator_client.clone(),
            context.clone(),
            networking_info.clone(),
            connections_info.clone(),
            allower.clone(),
            epoch_duration_ms,
            encoder_keypair.public(),
        );

        let committee_sync_manager = committee_sync_manager.start().await;

        Self {
            internal_network_manager,
            external_network_manager,
            object_network_manager,
            store,
            object_storage,
            context,
            probe_network_manager,
            committee_sync_manager,
            #[cfg(msim)]
            sim_state: Default::default(),
        }
    }

    // TODO: make this more robust
    fn create_default_context(own_encoder_key: EncoderPublicKey) -> Context {
        // Create minimal valid committees with just our own encoder
        let (authority_committee, _) = AuthorityCommittee::local_test_committee(
            0,       // Genesis epoch
            vec![1], // Minimal valid stake
        );

        // Create minimal encoder committee with just ourselves
        let test_probe = ProbeMetadata::new_for_test(&[0u8; 32]);
        let encoder = Encoder {
            voting_power: 10000, // Total voting power
            encoder_key: own_encoder_key.clone(),
            probe: test_probe,
        };

        // Create encoder committee with minimal valid configuration
        let encoder_committee = EncoderCommittee::new(
            0, // Genesis epoch
            1, // Minimal shard size
            1, // Minimal quorum threshold
            vec![encoder],
        );

        // Create committees with our encoder as index 0
        let committees = Committees::new(
            0, // Genesis epoch
            authority_committee,
            encoder_committee,
            1, // Minimal valid VDF iterations
        );

        // Create inner context with current and previous committees the same
        let inner_context = InnerContext::new(
            [committees.clone(), committees], // Same committee for current and previous
            0,                                // Genesis epoch
            own_encoder_key,
            std::collections::HashMap::new(), // Empty object servers map
        );

        // Create and return the context
        Context::new(inner_context)
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

        self.committee_sync_manager.stop();

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
