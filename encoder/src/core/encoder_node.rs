use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;
use std::{collections::BTreeSet, future::Future, sync::Arc};

use fastcrypto::traits::KeyPair;
use intelligence::evaluation::core_processor::EvaluationCoreProcessor;
use intelligence::evaluation::messaging::service::EvaluationNetworkService;
use intelligence::evaluation::messaging::tonic::{EvaluationTonicClient, EvaluationTonicManager};
use intelligence::evaluation::messaging::{EvaluationManager, EvaluationService};
use intelligence::evaluation::{EvaluatorClient, MockEvaluator};
use intelligence::inference::core_processor::InferenceCoreProcessor;
use intelligence::inference::messaging::service::InferenceNetworkService;
use intelligence::inference::messaging::tonic::{InferenceTonicClient, InferenceTonicManager};
use intelligence::inference::messaging::InferenceServiceManager;
use intelligence::inference::module::MockModule;
use object_store::memory::InMemory;
use objects::downloader::ObjectDownloader;
use objects::readers::url::ObjectHttpClient;
use objects::services::signed_url::ObjectServiceUrlGenerator;
use objects::services::{ObjectService, ObjectServiceManager};
use objects::stores::memory::{EphemeralInMemoryStore, PersistentInMemoryStore};
use objects::stores::EphemeralStore;
use objects::MIN_PART_SIZE;
use rand::Rng;
use sdk::client_config::{SomaClientConfig, SomaEnv};
use sdk::wallet_context::WalletContext;
use soma_keys::keystore::{AccountKeystore, FileBasedKeystore, Keystore};
use soma_tls::AllowPublicKeys;
use tokio::sync::{Mutex, Semaphore};
use tracing::{error, info, warn};
use types::actors::ActorManager;
use types::base::SomaAddress;
use types::config::{SOMA_CLIENT_CONFIG, SOMA_KEYSTORE_FILENAME};
use types::crypto::NetworkKeyPair;
use types::evaluation::{Score, ScoreV1};
use types::multiaddr::Multiaddr;
use types::p2p::to_host_port_str;
use types::parameters::HttpParameters;
use types::shard_crypto::keys::EncoderPublicKey;
use types::{
    config::{encoder_config::EncoderConfig, Config},
    system_state::SystemStateTrait,
};
use url::Url;

use super::{
    internal_broadcaster::Broadcaster,
    pipeline_dispatcher::{ExternalPipelineDispatcher, InternalPipelineDispatcher},
};
use crate::datastore::rocksdb_store::RocksDBStore;
use crate::pipelines::clean_up::CleanUpProcessor;
use crate::{
    datastore::Store,
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
        evaluation::EvaluationProcessor, input::InputProcessor, report_vote::ReportVoteProcessor,
        reveal::RevealProcessor,
    },
    sync::{
        committee_sync_manager::CommitteeSyncManager,
        encoder_validator_client::EncoderValidatorClient,
    },
    types::context::{Committees, Context, InnerContext},
};
use tokio::sync::RwLock;
use types::shard_networking::EncoderNetworkingInfo;
use types::shard_verifier::ShardVerifier;

#[cfg(msim)]
use msim::task::NodeId;
#[cfg(msim)]
use simulator::SimState;

pub struct EncoderNode {
    config: EncoderConfig,
    internal_network_manager: EncoderInternalTonicManager,
    external_network_manager: EncoderExternalTonicManager,
    object_service_manager: ObjectServiceManager,
    inference_network_manager: InferenceTonicManager,
    evaluation_network_manager: EvaluationTonicManager,
    clean_up_manager: ActorManager<CleanUpProcessor>,
    report_vote_manager: ActorManager<ReportVoteProcessor<EncoderInternalTonicClient>>,
    evaluation_manager:
        ActorManager<EvaluationProcessor<EncoderInternalTonicClient, EvaluationTonicClient>>,
    reveal_manager:
        ActorManager<RevealProcessor<EncoderInternalTonicClient, EvaluationTonicClient>>,
    commit_votes_manager:
        ActorManager<CommitVotesProcessor<EncoderInternalTonicClient, EvaluationTonicClient>>,
    commit_manager:
        ActorManager<CommitProcessor<EncoderInternalTonicClient, EvaluationTonicClient>>,
    input_manager: ActorManager<
        InputProcessor<EncoderInternalTonicClient, EvaluationTonicClient, InferenceTonicClient>,
    >,
    store: Arc<dyn Store>,
    pub context: Context,
    object_storage: Arc<InMemory>,
    committee_sync_manager: Arc<CommitteeSyncManager>,
    wallet_context: Arc<RwLock<WalletContext>>,

    #[cfg(msim)]
    sim_state: SimState,
}

impl EncoderNode {
    pub async fn start(config: EncoderConfig, working_dir: PathBuf) -> Self {
        let encoder_keypair = config.encoder_keypair.encoder_keypair().clone();
        let network_keypair = NetworkKeyPair::new(config.network_keypair.keypair().inner().copy());
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
        let inference_address: Multiaddr = config
            .inference_address
            .to_string()
            .parse()
            .expect("Valid multiaddr");
        let evaluation_address: Multiaddr = config
            .evaluation_address
            .to_string()
            .parse()
            .expect("Valid multiaddr");

        let (context, networking_info, allower) =
            create_context_from_genesis(&config, encoder_keypair.public(), network_keypair.clone());

        let mut internal_network_manager = EncoderInternalTonicManager::new(
            networking_info.clone(),
            parameters.clone(),
            network_keypair.clone(),
            internal_address.clone(),
            allower.clone(),
        );

        let messaging_client = <EncoderInternalTonicManager as EncoderInternalNetworkManager<
            EncoderInternalService<
                InternalPipelineDispatcher<EncoderInternalTonicClient, EvaluationTonicClient>,
            >,
        >>::client(&internal_network_manager);

        let object_storage = Arc::new(InMemory::new());

        let mut object_service_manager = ObjectServiceManager::new(
            network_keypair.clone(),
            config.object_parameters.clone(),
            allower.clone(),
        )
        .unwrap();

        let object_service = ObjectService::new(object_storage.clone(), network_keypair.public());

        object_service_manager
            .start(&object_address, object_service.clone())
            .await;

        let object_server_url = Url::from_str(&format!(
            "https://{}",
            to_host_port_str(&object_address).unwrap()
        ))
        .unwrap();

        let download_metadata_generator = Arc::new(ObjectServiceUrlGenerator::new(
            object_server_url,
            network_keypair.clone(),
            //TODO change the timeout to be one epoch?
            Duration::from_secs(3600),
        ));

        let ephemeral_store = EphemeralInMemoryStore::new(object_storage.clone());
        let persistent_store =
            PersistentInMemoryStore::new(object_storage.clone(), download_metadata_generator);

        ///////////////////////////
        ////////////////////////////

        let encoder_keypair = Arc::new(encoder_keypair);

        let wallet_context = Arc::new(RwLock::new(
            init_wallet_context(&config, &working_dir)
                .await
                .expect("Failed to initialize wallet context"),
        ));

        let default_buffer = 100_usize;
        let default_concurrency = 100_usize;
        // 5gb
        let max_size: u64 = 5 * 1024 * 1024 * 1024;

        let concurrency = Arc::new(Semaphore::new(default_concurrency));
        let chunk_size = MIN_PART_SIZE;
        let ns_per_byte = 40;
        let object_downloader =
            Arc::new(ObjectDownloader::new(concurrency, chunk_size, ns_per_byte).unwrap());

        let object_http_client =
            ObjectHttpClient::new(network_keypair.clone(), Arc::new(HttpParameters::default()))
                .unwrap();

        let module_client = Arc::new(MockModule::new());

        let inference_core_processor = InferenceCoreProcessor::new(
            persistent_store.clone(),
            ephemeral_store.clone(),
            module_client,
            object_downloader.clone(),
            object_http_client.clone(),
        );

        let inference_processor_manager =
            ActorManager::new(default_buffer, inference_core_processor);
        let inference_processor_handle = inference_processor_manager.handle();

        let inference_service = Arc::new(InferenceNetworkService::new(inference_processor_handle));
        let mut inference_network_manager = InferenceTonicManager::new(
            Arc::new(parameters.tonic.clone()),
            inference_address.clone(),
        );

        inference_network_manager.start(inference_service).await;
        let inference_client = Arc::new(
            InferenceTonicClient::new(
                inference_address.clone(),
                Arc::new(parameters.tonic.clone()),
            )
            .await
            .unwrap(),
        );

        let evaluator_client = Arc::new(MockEvaluator::new(Score::V1(ScoreV1::new(
            rand::thread_rng().gen(),
        ))));

        let evaluation_core_processor = EvaluationCoreProcessor::new(
            persistent_store.clone(),
            ephemeral_store,
            evaluator_client,
            object_downloader.clone(),
            object_http_client.clone(),
        );

        let evaluation_processor_manager =
            ActorManager::new(default_buffer, evaluation_core_processor);
        let evaluation_processor_handle = evaluation_processor_manager.handle();

        let evaluation_service =
            Arc::new(EvaluationNetworkService::new(evaluation_processor_handle));
        let mut evaluation_network_manager = EvaluationTonicManager::new(
            config.evaluation_parameters.clone(),
            evaluation_address.clone(),
        );

        evaluation_network_manager.start(evaluation_service).await;

        let evaluation_client = Arc::new(
            EvaluationTonicClient::new(
                evaluation_address.clone(),
                config.evaluation_parameters.clone(),
            )
            .await
            .unwrap(),
        );
        let store = Arc::new(RocksDBStore::new(config.db_path().to_str().unwrap()));

        let broadcaster = Arc::new(Broadcaster::new(
            messaging_client.clone(),
            Arc::new(Semaphore::new(default_concurrency)),
            encoder_keypair.public(),
        ));

        let clean_up_processor = CleanUpProcessor::new(store.clone());
        let clean_up_manager = ActorManager::new(default_buffer, clean_up_processor);
        let clean_up_handle = clean_up_manager.handle();

        let report_vote_processor = ReportVoteProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            clean_up_handle.clone(),
            wallet_context.clone(),
        );
        let report_vote_manager = ActorManager::new(default_buffer, report_vote_processor);
        let report_vote_handle = report_vote_manager.handle();

        let evaluation_processor = EvaluationProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            report_vote_handle.clone(),
            evaluation_client.clone(),
            context.clone(),
        );
        let evaluation_manager = ActorManager::new(default_buffer, evaluation_processor);
        let evaluation_handle = evaluation_manager.handle();

        let reveal_processor = RevealProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            evaluation_handle.clone(),
        );
        let reveal_manager = ActorManager::new(default_buffer, reveal_processor);
        let reveal_handle = reveal_manager.handle();

        let commit_votes_processor = CommitVotesProcessor::new(
            store.clone(),
            broadcaster.clone(),
            encoder_keypair.clone(),
            reveal_handle.clone(),
            context.clone(),
        );
        let commit_votes_manager = ActorManager::new(default_buffer, commit_votes_processor);
        let commit_votes_handle = commit_votes_manager.handle();

        let commit_processor = CommitProcessor::new(
            store.clone(),
            broadcaster.clone(),
            commit_votes_handle.clone(),
            encoder_keypair.clone(),
        );
        let commit_manager = ActorManager::new(default_buffer, commit_processor);
        let commit_handle = commit_manager.handle();

        let input_processor = InputProcessor::new(
            store.clone(),
            broadcaster.clone(),
            inference_client,
            evaluation_client,
            encoder_keypair.clone(),
            commit_handle.clone(),
            context.clone(),
        );
        let input_manager = ActorManager::new(default_buffer, input_processor);
        let input_handle = input_manager.handle();

        let pipeline_dispatcher = InternalPipelineDispatcher::new(
            commit_handle,
            commit_votes_handle,
            reveal_handle,
            report_vote_handle,
        );
        let verifier = Arc::new(ShardVerifier::new(100));

        let internal_network_service = Arc::new(EncoderInternalService::new(
            context.clone(),
            pipeline_dispatcher,
            verifier.clone(),
        ));
        internal_network_manager
            .start(internal_network_service)
            .await;

        // Now create the external manager and service
        let mut external_network_manager = EncoderExternalTonicManager::new(
            parameters.clone(),
            network_keypair.clone(),
            external_address.clone(),
            allower.clone(),
        );

        // Create the external service using the same context and verifier
        let external_network_service = Arc::new(EncoderExternalService::new(
            context.clone(),
            ExternalPipelineDispatcher::new(input_handle),
            verifier,
            persistent_store,
        ));

        // Start the external manager with the service
        external_network_manager
            .start(external_network_service)
            .await;

        info!(
            "Creating validator sync client connecting to {}",
            config.validator_sync_address
        );
        let validator_client = match EncoderValidatorClient::new(
            &config.validator_sync_address,
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
            object_service_manager,
            store,
            object_storage,
            context,
            inference_network_manager,
            evaluation_network_manager,
            committee_sync_manager,
            clean_up_manager,
            report_vote_manager,
            evaluation_manager,
            reveal_manager,
            commit_votes_manager,
            commit_manager,
            input_manager,
            wallet_context,
            #[cfg(msim)]
            sim_state: Default::default(),
        }
    }

    pub(crate) async fn stop(mut self) {
        <EncoderInternalTonicManager as EncoderInternalNetworkManager<
            EncoderInternalService<
                InternalPipelineDispatcher<EncoderInternalTonicClient, EvaluationTonicClient>,
            >,
        >>::stop(&mut self.internal_network_manager)
        .await;

        <EncoderExternalTonicManager as EncoderExternalNetworkManager<
            EncoderExternalService<
                ExternalPipelineDispatcher<
                    EncoderInternalTonicClient,
                    EvaluationTonicClient,
                    InferenceTonicClient,
                >,
                PersistentInMemoryStore,
            >,
        >>::stop(&mut self.external_network_manager)
        .await;

        self.committee_sync_manager.stop();

        self.object_service_manager.stop().await;

        <InferenceTonicManager as InferenceServiceManager<
            InferenceNetworkService<
                InMemory,
                InMemory,
                PersistentInMemoryStore,
                EphemeralInMemoryStore,
                MockModule,
            >,
        >>::stop(&mut self.inference_network_manager)
        .await;

        <EvaluationTonicManager as EvaluationManager<
            EvaluationNetworkService<
                InMemory,
                InMemory,
                PersistentInMemoryStore,
                EphemeralInMemoryStore,
                MockEvaluator,
            >,
        >>::stop(&mut self.evaluation_network_manager)
        .await;

        self.clean_up_manager.shutdown();
        self.report_vote_manager.shutdown();
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

async fn init_wallet_context(
    config: &EncoderConfig,
    working_dir: &PathBuf,
) -> Result<WalletContext, anyhow::Error> {
    std::fs::create_dir_all(working_dir)?;

    let client_config_path = working_dir.join(SOMA_CLIENT_CONFIG);
    let keystore_path = working_dir.join(SOMA_KEYSTORE_FILENAME);

    // Create keystore first
    let mut keystore = FileBasedKeystore::load_or_create(&keystore_path)?;

    // Import the account keypair from EncoderConfig
    let account_kp = (*(config.account_keypair.keypair())).copy();
    let address = SomaAddress::from(&account_kp.public());
    keystore.add_key(Some("encoder-account".to_string()), account_kp)?;

    // Create the client config
    let env = SomaEnv {
        alias: "localnet".to_string(),
        rpc: config.rpc_address.to_string(),
        internal_object_address: config.rpc_address.to_string(), // TODO: replace this with rpc's internal object address
        basic_auth: None,
    };

    let soma_client_config = SomaClientConfig {
        keystore: Keystore::File(keystore),
        external_keys: None,
        envs: vec![env],
        active_env: Some("localnet".to_string()),
        active_address: Some(address),
    };

    // Save the config
    soma_client_config.save(&client_config_path)?;
    // Create WalletContext from the saved config
    WalletContext::new(&client_config_path)
}

fn create_context_from_genesis(
    config: &EncoderConfig,
    own_encoder_key: EncoderPublicKey,
    own_network_keypair: NetworkKeyPair,
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
        own_network_keypair,
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

    pub fn object_storage(&self) -> Arc<InMemory> {
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
