use std::{collections::BTreeSet, future::Future, sync::Arc};

use evaluation::messaging::service::MockEvaluationService;
use evaluation::messaging::tonic::{EvaluationTonicClient, EvaluationTonicManager};
use evaluation::messaging::EvaluationManager;
use fastcrypto::traits::KeyPair;
use inference::client::MockInferenceClient;
use objects::networking::downloader::Downloader;
use objects::{
    networking::{
        http_network::{ObjectHttpClient, ObjectHttpManager},
        ObjectNetworkManager, ObjectNetworkService,
    },
    storage::{filesystem::FilesystemObjectStorage, memory::MemoryObjectStore},
};
use soma_tls::AllowPublicKeys;
use tokio::sync::{Mutex, Semaphore};
use tracing::{error, info, warn};
use types::actors::ActorManager;
use types::multiaddr::Multiaddr;
use types::shard_crypto::keys::{EncoderPublicKey, PeerKeyPair};
use types::{config::encoder_config::EncoderConfig, system_state::SystemStateTrait};

use crate::pipelines::clean_up::CleanUpProcessor;
use crate::{
    datastore::{mem_store::MemStore, Store},
    messaging::{
        external_service::EncoderExternalService,
        internal_service::EncoderInternalService,
        tonic::{
            external::EncoderExternalTonicManager,
            internal::{EncoderInternalTonicClient, EncoderInternalTonicManager},
        },
        EncoderExternalNetworkManager, EncoderInternalNetworkManager,
    },
    pipelines::{
        commit::CommitProcessor, commit_votes::CommitVotesProcessor,
        evaluation::EvaluationProcessor, input::InputProcessor, reveal::RevealProcessor,
        score_vote::ScoreVoteProcessor,
    },
    sync::{
        committee_sync_manager::CommitteeSyncManager,
        encoder_validator_client::EncoderValidatorClient,
    },
    types::context::{Committees, Context, InnerContext},
};
use types::shard_networking::EncoderNetworkingInfo;

use super::{
    internal_broadcaster::Broadcaster,
    pipeline_dispatcher::{ExternalPipelineDispatcher, InternalPipelineDispatcher},
};
use types::shard_verifier::ShardVerifier;

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
    config: EncoderConfig,
    internal_network_manager: EncoderInternalTonicManager,
    external_network_manager: EncoderExternalTonicManager,
    object_network_manager: ObjectHttpManager,
    evaluation_network_manager: EvaluationTonicManager,
    downloader_manager: ActorManager<Downloader<ObjectHttpClient, MemoryObjectStore>>,
    clean_up_manager: ActorManager<CleanUpProcessor>,
    score_vote_manager: ActorManager<ScoreVoteProcessor<EncoderInternalTonicClient>>,
    evaluation_manager: ActorManager<
        EvaluationProcessor<
            ObjectHttpClient,
            EncoderInternalTonicClient,
            MemoryObjectStore,
            EvaluationTonicClient,
        >,
    >,
    reveal_manager: ActorManager<
        RevealProcessor<
            ObjectHttpClient,
            EncoderInternalTonicClient,
            MemoryObjectStore,
            EvaluationTonicClient,
        >,
    >,
    commit_votes_manager: ActorManager<
        CommitVotesProcessor<
            ObjectHttpClient,
            EncoderInternalTonicClient,
            MemoryObjectStore,
            EvaluationTonicClient,
        >,
    >,
    commit_manager: ActorManager<
        CommitProcessor<
            ObjectHttpClient,
            EncoderInternalTonicClient,
            MemoryObjectStore,
            EvaluationTonicClient,
        >,
    >,
    input_manager: ActorManager<
        InputProcessor<
            EncoderInternalTonicClient,
            ObjectHttpClient,
            MockInferenceClient<MemoryObjectStore>,
            MemoryObjectStore,
            EvaluationTonicClient,
        >,
    >,
    store: Arc<dyn Store>,
    pub context: Context,
    object_storage: Arc<MemoryObjectStore>,
    committee_sync_manager: Arc<CommitteeSyncManager>,

    #[cfg(msim)]
    sim_state: SimState,
}

impl EncoderNode {
    pub async fn start(config: EncoderConfig) -> Self {
        let encoder_keypair = config.encoder_keypair.encoder_keypair().clone();
        let peer_keypair = PeerKeyPair::new(config.peer_keypair.keypair().inner().copy());
        let parameters = Arc::new(types::parameters::Parameters::default());
        let internal_address: Multiaddr = config
            .internal_network_address
            .to_string()
            .parse()
            .expect("Valid multiaddr");
        let external_address: Multiaddr = config
            .external_network_address
            .to_string()
            .parse()
            .expect("Valid multiaddr");
        let object_address: Multiaddr = config
            .object_address
            .to_string()
            .parse()
            .expect("Valid multiaddr");
        let evaluation_address: Multiaddr = config
            .evaluation_address
            .to_string()
            .parse()
            .expect("Valid multiaddr");

        let (context, networking_info, allower) =
            create_context_from_genesis(&config, encoder_keypair.public());

        let mut internal_network_manager = EncoderInternalTonicManager::new(
            networking_info.clone(),
            parameters.clone(),
            peer_keypair.clone(),
            internal_address.clone(),
            allower.clone(),
        );

        let messaging_client = <EncoderInternalTonicManager as EncoderInternalNetworkManager<
            EncoderInternalService<
                InternalPipelineDispatcher<
                    EncoderInternalTonicClient,
                    ObjectHttpClient,
                    FilesystemObjectStorage,
                    EvaluationTonicClient,
                >,
            >,
        >>::client(&internal_network_manager);

        let object_storage = Arc::new(MemoryObjectStore::new_for_test());
        let object_network_service: ObjectNetworkService<MemoryObjectStore> =
            ObjectNetworkService::new(object_storage.clone());

        let mut object_network_manager =
            <ObjectHttpManager as ObjectNetworkManager<MemoryObjectStore>>::new(
                peer_keypair.clone(),
                config.object_parameters.clone(),
                allower.clone(),
            )
            .unwrap();

        object_network_manager
            .start(&object_address, object_network_service)
            .await;

        let object_client = <ObjectHttpManager as ObjectNetworkManager<MemoryObjectStore>>::client(
            &object_network_manager,
        );

        ///////////////////////////
        let mut evaluation_network_manager = EvaluationTonicManager::new(
            config.evaluation_parameters.clone(),
            evaluation_address.clone(),
        );

        let evaluation_service = Arc::new(MockEvaluationService::new());
        evaluation_network_manager.start(evaluation_service).await;

        let evaluation_client = Arc::new(
            EvaluationTonicClient::new(
                evaluation_address.clone(),
                config.evaluation_parameters.clone(),
            )
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

        // TODO: Remove - VDF handle is created in ShardVerifier
        // let vdf = EntropyVDF::new(1);
        // let vdf_processor = VDFProcessor::new(vdf, 1);
        // let vdf_handle = ActorManager::new(1, vdf_processor).handle();
        let store = Arc::new(MemStore::new());

        let broadcaster = Arc::new(Broadcaster::new(
            messaging_client.clone(),
            Arc::new(Semaphore::new(default_concurrency)),
            encoder_keypair.public(),
        ));

        let recv_dedup_cache_capacity: usize = 1000;
        let send_dedup_cache_capacity: usize = 100;
        let clean_up_processor = CleanUpProcessor::new(store.clone(), recv_dedup_cache_capacity);
        let clean_up_manager = ActorManager::new(default_buffer, clean_up_processor);
        let clean_up_handle = clean_up_manager.handle();

        let score_vote_processor = ScoreVoteProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            clean_up_handle.clone(),
            recv_dedup_cache_capacity,
            send_dedup_cache_capacity,
        );
        let score_vote_manager = ActorManager::new(default_buffer, score_vote_processor);
        let score_vote_handle = score_vote_manager.handle();

        let evaluation_processor = EvaluationProcessor::new(
            store.clone(),
            downloader_handle.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            object_storage.clone(),
            score_vote_handle.clone(),
            evaluation_client.clone(),
            recv_dedup_cache_capacity,
        );
        let evaluation_manager = ActorManager::new(default_buffer, evaluation_processor);
        let evaluation_handle = evaluation_manager.handle();

        let reveal_processor = RevealProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            evaluation_handle.clone(),
            recv_dedup_cache_capacity,
            send_dedup_cache_capacity,
        );
        let reveal_manager = ActorManager::new(default_buffer, reveal_processor);
        let reveal_handle = reveal_manager.handle();

        let commit_votes_processor = CommitVotesProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            reveal_handle.clone(),
            recv_dedup_cache_capacity,
            send_dedup_cache_capacity,
        );
        let commit_votes_manager = ActorManager::new(default_buffer, commit_votes_processor);
        let commit_votes_handle = commit_votes_manager.handle();

        let commit_processor = CommitProcessor::new(
            store.clone(),
            broadcaster.clone(),
            commit_votes_handle.clone(),
            encoder_keypair.clone(),
            recv_dedup_cache_capacity,
            send_dedup_cache_capacity,
        );
        let commit_manager = ActorManager::new(default_buffer, commit_processor);
        let commit_handle = commit_manager.handle();

        let inference_client = Arc::new(MockInferenceClient::new(object_storage.clone()));

        let input_processor = InputProcessor::new(
            store.clone(),
            downloader_handle.clone(),
            broadcaster.clone(),
            inference_client,
            evaluation_client,
            encoder_keypair.clone(),
            object_storage.clone(),
            commit_handle.clone(),
        );
        let input_manager = ActorManager::new(default_buffer, input_processor);
        let input_handle = input_manager.handle();

        let pipeline_dispatcher = InternalPipelineDispatcher::new(
            commit_handle,
            commit_votes_handle,
            reveal_handle,
            score_vote_handle,
        );
        let verifier = Arc::new(ShardVerifier::new(100));

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
            config.validator_rpc_address
        );
        let validator_client = match EncoderValidatorClient::new(
            &config.validator_rpc_address,
            config.genesis.committee().unwrap(),
        )
        .await
        {
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
            allower.clone(),
            config.genesis.system_object().epoch_start_timestamp_ms(),
            config.epoch_duration_ms,
            encoder_keypair.public(),
        );

        let committee_sync_manager = committee_sync_manager.start().await;

        Self {
            config,
            internal_network_manager,
            external_network_manager,
            object_network_manager,
            store,
            object_storage,
            context,
            evaluation_network_manager,
            committee_sync_manager,
            downloader_manager,
            clean_up_manager,
            score_vote_manager,
            evaluation_manager,
            reveal_manager,
            commit_votes_manager,
            commit_manager,
            input_manager,
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
                    EvaluationTonicClient,
                >,
            >,
        >>::stop(&mut self.internal_network_manager)
        .await;

        <EncoderExternalTonicManager as EncoderExternalNetworkManager<
            EncoderExternalService<
                ExternalPipelineDispatcher<
                    EncoderInternalTonicClient,
                    ObjectHttpClient,
                    MockInferenceClient<MemoryObjectStore>,
                    MemoryObjectStore,
                    EvaluationTonicClient,
                >,
            >,
        >>::stop(&mut self.external_network_manager)
        .await;

        self.committee_sync_manager.stop();

        <ObjectHttpManager as ObjectNetworkManager<MemoryObjectStore>>::stop(
            &mut self.object_network_manager,
        )
        .await;

        // TODO: Replace mock with real service
        <EvaluationTonicManager as EvaluationManager<MockEvaluationService>>::stop(
            &mut self.evaluation_network_manager,
        )
        .await;

        self.downloader_manager.shutdown();
        self.clean_up_manager.shutdown();
        self.score_vote_manager.shutdown();
        self.evaluation_manager.shutdown();
        self.reveal_manager.shutdown();
        self.commit_votes_manager.shutdown();
        self.commit_manager.shutdown();
        self.input_manager.shutdown();
    }

    pub fn get_config(&self) -> &EncoderConfig {
        &self.config
    }

    pub fn get_store_for_testing(&self) -> Arc<dyn Store> {
        self.store.clone()
    }
}

fn create_context_from_genesis(
    config: &EncoderConfig,
    own_encoder_key: EncoderPublicKey,
) -> (Context, EncoderNetworkingInfo, AllowPublicKeys) {
    let networking_info = EncoderNetworkingInfo::default();
    let allower = AllowPublicKeys::default();

    // Extract validator committee from genesis
    let authority_committee = match config.genesis.committee() {
        Ok(committee) => committee,
        Err(e) => {
            warn!("Failed to extract committee from genesis: {}", e);
            panic!("Failed to extract committee from genesis: {}", e);
        }
    };

    // Convert EncoderCommittee from genesis to our internal ShardCommittee format
    let genesis_encoder_committee = config.genesis.encoder_committee();

    // Extract peer keys and network addresses for network info
    let (initial_networking_info, object_servers) =
        EncoderValidatorClient::extract_network_info(&config.genesis.encoder_committee(), None);

    // Create committees struct with proper genesis parameters
    let committees = Committees::new(
        0, // Genesis epoch
        authority_committee,
        genesis_encoder_committee,
        config.genesis.networking_committee(),
        1, // TODO: Default VDF iterations, adjust as needed
    );

    // Create inner context with our committees
    let inner_context = InnerContext::new(
        [committees.clone(), committees], // Same committee for current and previous in genesis
        0,                                // Genesis epoch
        own_encoder_key,
        object_servers, // Initialize with object servers from genesis
    );

    // Update the NetworkingInfo
    if !initial_networking_info.is_empty() {
        info!(
            "Initializing networking info with {} entries from genesis",
            initial_networking_info.len()
        );
        networking_info.update(initial_networking_info.clone());
    }

    // Update allowed public keys
    let mut allowed_keys = BTreeSet::new();
    for (encoder_public_key, (peer_key, address)) in initial_networking_info {
        allowed_keys.insert(peer_key.clone().into_inner());
    }

    if !allowed_keys.is_empty() {
        info!(
            "Initializing allowed public keys with {} entries from genesis",
            allowed_keys.len()
        );
        allower.update(allowed_keys);
    }

    // Create and return the context
    (Context::new(inner_context), networking_info, allower)
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
