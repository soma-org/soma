use anyhow::{anyhow, Result};
use arc_swap::ArcSwap;
use authority::{
    adapter::{ConsensusAdapter, SubmitToConsensus},
    aggregator::AuthorityAggregator,
    cache::build_execution_cache,
    client::NetworkAuthorityClient,
    commit::{executor::CommitExecutor, CommitStore},
    consensus_store_pruner::ConsensusStorePruner,
    encoder_client::EncoderClientService,
    epoch_store::AuthorityPerEpochStore,
    handler::ConsensusHandlerInitializer,
    manager::{ConsensusClient, ConsensusManager, ConsensusManagerTrait},
    orchestrator::TransactionOrchestrator,
    reconfiguration::ReconfigurationInitiator,
    rpc_index::RpcIndexStore,
    rpc_store::RestReadStore,
    server::ServerBuilder,
    service::ValidatorService,
    start_epoch::{EpochStartConfigTrait, EpochStartConfiguration},
    state::{self, AuthorityState},
    state_accumulator::StateAccumulator,
    state_sync_store::StateSyncStore,
    store::AuthorityStore,
    store_pruner::{ObjectsCompactionFilter, PrunerWatermarks},
    store_tables::{
        AuthorityPerpetualTables, AuthorityPerpetualTablesOptions, AuthorityPrunerTables,
    },
    throughput::{
        ConsensusThroughputCalculator, ConsensusThroughputProfiler, ThroughputProfileRanges,
    },
    tonic_gen::validator_server::ValidatorServer,
    tx_validator::TxValidator,
};
use encoder_validator_api::{
    service::EncoderValidatorService,
    tonic_gen::encoder_validator_api_server::EncoderValidatorApiServer,
};
use futures::TryFutureExt;
use object_store::memory::InMemory;
use objects::networking::{
    external_service::ExternalObjectServiceManager, internal_service::InternalObjectServiceManager,
    DownloadService, ObjectServiceManager as _,
};
use p2p::{
    builder::{DiscoveryHandle, P2pBuilder, StateSyncHandle},
    tonic_gen::p2p_server::P2pServer,
};
use parking_lot::RwLock;
use rpc::api::subscription::SubscriptionService;
use soma_tls::AllowPublicKeys;
use store::rocks::default_db_options;
use tower::ServiceBuilder;

use std::{
    collections::{BTreeSet, HashMap},
    sync::{Arc, Weak},
    time::{Duration, SystemTime},
};
use tokio::{
    sync::{broadcast, mpsc::Sender, Mutex},
    task::JoinHandle,
    time::sleep,
};
use tracing::{error_span, info, warn, Instrument};
use types::{
    base::{AuthorityName, SomaAddress},
    checkpoint::Checkpoint,
    client::Config,
    committee::Committee,
    config::node_config::{ConsensusConfig, NodeConfig},
    consensus::context::{Clock, Context},
    crypto::KeypairTraits,
    effects::TransactionEffects,
    encoder_committee::EncoderCommittee,
    error::{SomaError, SomaResult},
    object::ObjectRef,
    p2p::{
        active_peers::{self, ActivePeers},
        channel_manager::{ChannelManager, ChannelManagerRequest},
    },
    parameters::{HttpParameters, Parameters},
    peer_id::PeerId,
    protocol::ProtocolConfig,
    quorum_driver::{ExecuteTransactionRequest, ExecuteTransactionRequestType},
    shard::{Shard, ShardAuthToken},
    storage::{
        committee_store::CommitteeStore,
        consensus::{mem_store::MemStore, rocksdb_store::RocksDBStore, ConsensusStore},
        write_store::WriteStore,
    },
    system_state::{
        epoch_start::{EpochStartSystemState, EpochStartSystemStateTrait},
        SystemState, SystemStateTrait,
    },
    tls::AllowedPublicKeys,
    transaction::{
        CertificateProof, ExecutableTransaction, Transaction, VerifiedExecutableTransaction,
    },
};

#[cfg(msim)]
use msim::task::NodeId;
#[cfg(msim)]
use simulator::SimState;

pub mod handle;

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

pub struct ValidatorComponents {
    validator_server_handle: JoinHandle<Result<()>>,
    consensus_manager: ConsensusManager,
    consensus_store_pruner: ConsensusStorePruner,
    consensus_adapter: Arc<ConsensusAdapter>,
}

pub struct P2pComponents {
    channel_manager_tx: Sender<ChannelManagerRequest>,
    discovery_handle: DiscoveryHandle,
    state_sync_handle: StateSyncHandle,
}

#[derive(Default)]
struct HttpServers {
    #[allow(unused)]
    http: Option<soma_http::ServerHandle>,
    #[allow(unused)]
    https: Option<soma_http::ServerHandle>,
}

pub struct SomaNode {
    config: NodeConfig,
    validator_components: Mutex<Option<ValidatorComponents>>,
    /// Broadcast channel to send the starting system state for the next epoch.
    end_of_epoch_channel: broadcast::Sender<SystemState>,
    // Channel to allow signaling upstream to shutdown node
    // shutdown_channel_tx: broadcast::Sender<Option<RunWithRange>>,
    // Broadcast channel to notify state-sync for new validator peers.
    // trusted_peer_change_tx: watch::Sender<TrustedPeerChangeEvent>,
    state: Arc<AuthorityState>,
    transaction_orchestrator: Option<Arc<TransactionOrchestrator<NetworkAuthorityClient>>>,
    state_sync_handle: StateSyncHandle,
    commit_store: Arc<CommitStore>,
    accumulator: Mutex<Option<Arc<StateAccumulator>>>,
    consensus_store: Arc<dyn ConsensusStore>,
    // connection_monitor_status: Arc<ConnectionMonitorStatus>,
    // AuthorityAggregator of the network, created at start and beginning of each epoch.
    auth_agg: Arc<ArcSwap<AuthorityAggregator<NetworkAuthorityClient>>>,
    encoder_validator_server_handle: Mutex<Option<JoinHandle<Result<()>>>>,
    encoder_client_service: Option<Arc<EncoderClientService>>,
    http_servers: HttpServers,
    object_managers: Option<(InternalObjectServiceManager, ExternalObjectServiceManager)>,
    allower: AllowPublicKeys,

    subscription_service_checkpoint_sender: Option<tokio::sync::mpsc::Sender<Checkpoint>>,

    #[cfg(msim)]
    sim_state: SimState,
}

impl SomaNode {
    pub async fn start(config: NodeConfig) -> Result<Arc<SomaNode>> {
        Self::start_async(config).await
    }

    pub async fn start_async(config: NodeConfig) -> Result<Arc<SomaNode>> {
        let is_validator = config.consensus_config().is_some();
        let is_full_node = !is_validator;

        info!(node =? config.protocol_public_key(),
            "Initializing soma-node listening on {}", config.network_address
        );

        let genesis = config.genesis().clone();

        let secret = Arc::pin(config.protocol_key_pair().copy());
        let genesis_committee = genesis.committee()?;
        let committee_store = Arc::new(CommitteeStore::new(
            config.db_path().join("epochs"),
            &genesis_committee,
            None,
        ));

        let pruner_watermarks = Arc::new(PrunerWatermarks::default());

        let mut pruner_db = None;
        if config
            .authority_store_pruning_config
            .enable_compaction_filter
        {
            pruner_db = Some(Arc::new(AuthorityPrunerTables::open(
                &config.db_path().join("store"),
            )));
        }
        let compaction_filter = pruner_db.clone().map(|db| ObjectsCompactionFilter::new(db));

        // By default, only enable write stall on validators for perpetual db.
        // TODO: let enable_write_stall = config.enable_db_write_stall.unwrap_or(is_validator);
        let perpetual_tables_options = AuthorityPerpetualTablesOptions {
            enable_write_stall: true,
            compaction_filter,
        };
        let perpetual_tables = Arc::new(AuthorityPerpetualTables::open(
            &config.db_path().join("store"),
            Some(perpetual_tables_options),
        ));
        let is_genesis = perpetual_tables
            .database_is_empty()
            .expect("Database read should not fail at init.");
        let store = AuthorityStore::open(perpetual_tables, &genesis, &config).await?;

        let cur_epoch = store.get_recovery_epoch_at_restart()?;
        let committee = committee_store
            .get_committee(&cur_epoch)?
            .expect("Committee of the current epoch must exist");
        let epoch_start_configuration = store
            .get_epoch_start_configuration()?
            .expect("EpochStartConfiguration of the current epoch must exist");

        let cache_traits = build_execution_cache(&store);

        let auth_agg = {
            Arc::new(ArcSwap::new(Arc::new(
                AuthorityAggregator::new_from_epoch_start_state(
                    epoch_start_configuration.epoch_start_state(),
                    &committee_store,
                ),
            )))
        };

        info!("creating commit store");

        let commit_store = CommitStore::new(&config.db_path().join("commits"));

        let epoch_options = default_db_options().optimize_db_for_write_throughput(4);
        let epoch_store = AuthorityPerEpochStore::new(
            config.protocol_public_key(),
            committee.clone(),
            &config.db_path().join("store"),
            Some(epoch_options.options),
            epoch_start_configuration,
            commit_store
                .get_highest_executed_commit_index()
                .expect("commit store read cannot fail")
                .unwrap_or(0),
        )?;

        info!("created epoch store");

        info!("creating long term consensus store");
        let store_path = config.consensus_db_path();
        let consensus_store = Arc::new(RocksDBStore::new(store_path.as_path().to_str().unwrap()));

        info!("creating state sync store");
        let state_sync_store = StateSyncStore::new(
            cache_traits.clone(),
            committee_store.clone(),
            commit_store.clone(),
            consensus_store.clone(),
        );

        let rpc_index = if is_full_node && config.rpc().is_some_and(|rpc| rpc.enable_indexing()) {
            Some(Arc::new(
                RpcIndexStore::new(&config.db_path(), &store, &commit_store).await,
            ))
        } else {
            None
        };

        // let (trusted_peer_change_tx, trusted_peer_change_rx) = watch::channel(Default::default());
        let P2pComponents {
            channel_manager_tx,
            discovery_handle,
            state_sync_handle,
        } = Self::create_p2p_network(
            &config,
            state_sync_store.clone(),
            // trusted_peer_change_rx,
        )?;

        // We must explicitly send this instead of relying on the initial value to trigger
        // watch value change, so that state-sync is able to process it.
        // send_trusted_peer_change(
        //     &config,
        //     &trusted_peer_change_tx,
        //     epoch_store.epoch_start_state(),
        // )
        // .expect("Initial trusted peers must be set");

        // info!("start state archival");
        // // Start archiving local state to remote store
        // let state_archive_handle =
        //     Self::start_state_archival(&config, &prometheus_registry, state_sync_store.clone())
        //         .await?;

        let accumulator = Arc::new(StateAccumulator::new(
            cache_traits.accumulator_store.clone(),
        ));

        info!("create authority state");
        let authority_name = config.protocol_public_key();
        let state = AuthorityState::new(
            authority_name,
            secret,
            epoch_store.clone(),
            committee_store.clone(),
            config.clone(),
            cache_traits.clone(),
            accumulator.clone(),
            rpc_index,
            commit_store.clone(),
            store.clone(),
            pruner_db,
            pruner_watermarks,
        )
        .await;

        commit_store.insert_genesis_commit(genesis.commit());
        // ensure genesis txn was executed
        if epoch_store.epoch() == 0 {
            let txn = &genesis.transaction();
            let span = error_span!("genesis_txn", tx_digest = ?txn.digest());
            let transaction = VerifiedExecutableTransaction::new_unchecked(
                ExecutableTransaction::new_from_data_and_sig(
                    genesis.transaction().data().clone(),
                    CertificateProof::new_system(0),
                ),
            );

            let tx_digests = &[txn.digest().clone()];

            state
                .try_execute_immediately(&transaction, None, Some(0), &epoch_store)
                .instrument(span)
                .await
                .unwrap();

            state
                .get_cache_commit()
                .commit_transaction_outputs(0, tx_digests)
                .await
                .expect("commit_transaction_outputs cannot fail");

            epoch_store
                .handle_committed_transactions(0, tx_digests)
                .expect("cannot fail");
        }

        // checkpoint_store
        //     .reexecute_local_checkpoints(&state, &epoch_store)
        //     .await;

        let (end_of_epoch_channel, end_of_epoch_receiver) =
            broadcast::channel(config.end_of_epoch_broadcast_channel_capacity);

        // let authority_names_to_peer_ids = epoch_store
        //     .epoch_start_state()
        //     .get_authority_names_to_peer_ids();

        // let authority_names_to_peer_ids = ArcSwap::from_pointee(authority_names_to_peer_ids);

        // let (_connection_monitor_handle, connection_statuses) =
        //     narwhal_network::connectivity::ConnectionMonitor::spawn(
        //         p2p_network.downgrade(),
        //         network_connection_metrics,
        //         HashMap::new(),
        //         None,
        //     );

        // let connection_monitor_status = ConnectionMonitorStatus {
        //     connection_statuses,
        //     authority_names_to_peer_ids,
        // };

        // let connection_monitor_status = Arc::new(connection_monitor_status);

        let validator_components = if state.is_validator(&epoch_store) && is_validator {
            let components = Self::construct_validator_components(
                config.clone(),
                state.clone(),
                committee,
                epoch_store.clone(),
                state_sync_handle.clone(),
                Arc::downgrade(&accumulator),
                consensus_store.clone(),
            )
            .await?;
            // This is only needed during cold start.
            components.consensus_adapter.submit_recovered(&epoch_store);

            Some(components)
        } else {
            None
        };

        // setup shutdown channel
        // let (shutdown_channel, _) = broadcast::channel::<Option<RunWithRange>>(1);

        // TODO: for production make this configurable to use either Filesystem or Bucket
        let object_storage = Arc::new(InMemory::new());

        let encoder_client_service = if is_full_node {
            // Only fullnodes send to encoders, not validators
            Some(Arc::new(EncoderClientService::new(
                config.protocol_key_pair().copy(),
                config.network_key_pair(),
            )))
        } else {
            None
        };

        let transaction_orchestrator = if is_full_node {
            Some(Arc::new(TransactionOrchestrator::new_with_encoder_client(
                auth_agg.load_full(),
                state.clone(),
                end_of_epoch_receiver,
                encoder_client_service.clone(),
                Some(object_storage.clone()),
            )))
        } else {
            None
        };

        let (http_servers, subscription_service_checkpoint_sender) = build_http_servers(
            state.clone(),
            state_sync_store,
            &transaction_orchestrator.clone(),
            &config,
        )
        .await?;

        let encoder_validator_server_handle = if is_full_node {
            info!("Starting encoder validator service for fullnode");
            Some(
                Self::start_grpc_encoder_service(&config, state.clone(), commit_store.clone())
                    .await?,
            )
        } else {
            None
        };

        let genesis_encoder_committee = genesis.encoder_committee();
        let mut encoder_committee_keys = BTreeSet::new();

        for (key, _) in genesis_encoder_committee.members() {
            if let Some(metadata) = genesis_encoder_committee.network_metadata.get(&key) {
                let peer_key = metadata.network_key.clone().into_inner();
                encoder_committee_keys.insert(peer_key);
            }
        }

        let allower = AllowPublicKeys::new(encoder_committee_keys);

        let object_managers = if is_full_node {
            Some(Self::start_object_services(&config, allower.clone(), object_storage).await?)
        } else {
            None
        };

        let node = Self {
            config,
            validator_components: Mutex::new(validator_components),
            end_of_epoch_channel,
            state,
            transaction_orchestrator,
            auth_agg, // shutdown_channel_tx: shutdown_channel,
            accumulator: Mutex::new(Some(accumulator)),
            state_sync_handle,
            commit_store,
            consensus_store,
            encoder_validator_server_handle: Mutex::new(encoder_validator_server_handle),
            encoder_client_service,
            http_servers,
            object_managers,
            allower,
            subscription_service_checkpoint_sender,
            // connection_monitor_status,
            #[cfg(msim)]
            sim_state: Default::default(),
        };

        info!("SomaNode started!");
        let node = Arc::new(node);
        let node_copy = node.clone();

        tokio::spawn(async move {
            let result = Self::monitor_reconfiguration(node_copy).await;
            if let Err(error) = result {
                warn!("Reconfiguration finished with error {:?}", error);
            }
        });

        Ok(node)
    }

    // Init reconfig process by starting to reject user certs
    pub async fn close_epoch(&self, epoch_store: &Arc<AuthorityPerEpochStore>) -> SomaResult {
        info!("close_epoch (current epoch = {})", epoch_store.epoch());
        self.validator_components
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| SomaError::from("Node is not a validator"))?
            .consensus_adapter
            .close_epoch(epoch_store);
        Ok(())
    }

    fn create_p2p_network(
        config: &NodeConfig,
        state_sync_store: StateSyncStore,
    ) -> Result<P2pComponents> {
        let (discovery, state_sync, p2p_server) = P2pBuilder::new()
            .config(config.p2p_config.clone())
            .store(state_sync_store)
            .archive_config(config.state_archive_read_config.clone())
            .build();

        let own_address = config
            .p2p_config
            .external_address
            .clone()
            .expect("External address must be set");

        let active_peers = ActivePeers::new(1000);
        let (channel_manager, channel_manager_tx) = ChannelManager::new(
            own_address,
            config.network_key_pair().clone(),
            p2p_server,
            active_peers.clone(),
        );

        let peer_event_receiver = channel_manager.subscribe();

        tokio::spawn(channel_manager.start());

        info!("P2p network started");

        let discovery_handle = discovery.start(
            active_peers.clone(),
            channel_manager_tx.clone(),
            config.network_key_pair().clone(),
        );
        let state_sync_handle = state_sync.start(active_peers, peer_event_receiver);

        Ok(P2pComponents {
            channel_manager_tx,
            discovery_handle,
            state_sync_handle,
        })
    }

    async fn construct_validator_components(
        config: NodeConfig,
        state: Arc<AuthorityState>,
        committee: Arc<Committee>,
        epoch_store: Arc<AuthorityPerEpochStore>,
        state_sync_handle: StateSyncHandle,
        accumulator: Weak<StateAccumulator>,
        consensus_store: Arc<dyn ConsensusStore>,
    ) -> Result<ValidatorComponents> {
        let mut config_clone = config.clone();
        let consensus_config = config_clone
            .consensus_config
            .as_mut()
            .ok_or_else(|| anyhow!("Validator is missing consensus config"))?;

        let client = Arc::new(ConsensusClient::new());
        let consensus_adapter = Arc::new(Self::construct_consensus_adapter(
            &committee,
            consensus_config,
            state.name,
            // connection_monitor_status.clone(),
            epoch_store.protocol_config().clone(),
            client.clone(),
        ));
        let consensus_manager = ConsensusManager::new(
            &config,
            consensus_config,
            client,
            state.get_accumulator_store().clone(),
            consensus_adapter.clone(),
            consensus_store.clone(),
            state.clone_committee_store(),
        );

        // This only gets started up once, not on every epoch. (Make call to remove every epoch.)
        let consensus_store_pruner = ConsensusStorePruner::new(
            consensus_store.clone(),
            consensus_config.db_retention_epochs(),
            consensus_config.db_pruner_period(),
        );

        let validator_server_handle =
            Self::start_grpc_validator_service(&config, state.clone(), consensus_adapter.clone())
                .await?;

        // Starts an overload monitor that monitors the execution of the authority.
        // Don't start the overload monitor when max_load_shedding_percentage is 0.
        // let validator_overload_monitor_handle = if config
        //     .authority_overload_config
        //     .max_load_shedding_percentage
        //     > 0
        // {
        //     let authority_state = Arc::downgrade(&state);
        //     let overload_config = config.authority_overload_config.clone();
        //     fail_point!("starting_overload_monitor");
        //     Some(spawn_monitored_task!(overload_monitor(
        //         authority_state,
        //         overload_config,
        //     )))
        // } else {
        //     None
        // };

        Self::start_epoch_specific_validator_components(
            &config,
            state.clone(),
            consensus_adapter,
            epoch_store,
            state_sync_handle,
            consensus_manager,
            consensus_store_pruner,
            accumulator,
            validator_server_handle,
        )
        .await
    }

    async fn start_epoch_specific_validator_components(
        config: &NodeConfig,
        state: Arc<AuthorityState>,
        consensus_adapter: Arc<ConsensusAdapter>,
        epoch_store: Arc<AuthorityPerEpochStore>,
        state_sync_handle: StateSyncHandle,
        consensus_manager: ConsensusManager,
        consensus_store_pruner: ConsensusStorePruner,
        accumulator: Weak<StateAccumulator>,
        validator_server_handle: JoinHandle<Result<()>>,
    ) -> Result<ValidatorComponents> {
        let throughput_calculator = Arc::new(ConsensusThroughputCalculator::new(None));

        let throughput_profiler = Arc::new(ConsensusThroughputProfiler::new(
            throughput_calculator.clone(),
            None,
            None,
            ThroughputProfileRanges::new_with_default(),
        ));

        consensus_adapter.swap_throughput_profiler(throughput_profiler);

        let consensus_handler_initializer = ConsensusHandlerInitializer::new(
            state.clone(),
            epoch_store.clone(),
            throughput_calculator,
            state_sync_handle,
        );

        consensus_manager
            .start(
                config,
                epoch_store.clone(),
                consensus_handler_initializer,
                TxValidator::new(epoch_store.clone()),
            )
            .await;

        Ok(ValidatorComponents {
            validator_server_handle,
            consensus_manager,
            consensus_store_pruner,
            consensus_adapter,
        })
    }

    fn construct_consensus_adapter(
        committee: &Committee,
        consensus_config: &ConsensusConfig,
        authority: AuthorityName,
        // connection_monitor_status: Arc<ConnectionMonitorStatus>,
        protocol_config: ProtocolConfig,
        consensus_client: Arc<dyn SubmitToConsensus>,
    ) -> ConsensusAdapter {
        // The consensus adapter allows the authority to send user certificates through consensus.

        ConsensusAdapter::new(
            consensus_client,
            authority,
            // connection_monitor_status,
            consensus_config.max_pending_transactions(),
            consensus_config.max_pending_transactions() * 2 / committee.num_members(),
            consensus_config.max_submit_position,
            consensus_config.submit_delay_step_override(),
            protocol_config,
        )
    }

    async fn start_grpc_validator_service(
        config: &NodeConfig,
        state: Arc<AuthorityState>,
        consensus_adapter: Arc<ConsensusAdapter>,
    ) -> Result<tokio::task::JoinHandle<Result<()>>> {
        let validator_service = ValidatorService::new(state.clone(), consensus_adapter);

        let server_conf = Config::new();
        let mut server_builder = ServerBuilder::from_config(&server_conf);

        server_builder = server_builder.add_service(ValidatorServer::new(validator_service));

        let server = server_builder
            .bind(config.consensus_config().unwrap().address())
            .await
            .map_err(|err| anyhow!(err.to_string()))?;
        let local_addr = server.local_addr();
        info!("Listening to traffic on {local_addr}");
        let grpc_server = tokio::spawn(server.serve().map_err(Into::into));

        Ok(grpc_server)
    }

    async fn start_grpc_encoder_service(
        config: &NodeConfig,
        state: Arc<AuthorityState>,
        commit_store: Arc<CommitStore>,
    ) -> Result<tokio::task::JoinHandle<Result<()>>> {
        let encoder_validator_service =
            EncoderValidatorService::new(state.clone(), commit_store.clone());

        let server_conf = Config::new();
        let mut server_builder = ServerBuilder::from_config(&server_conf);

        server_builder =
            server_builder.add_service(EncoderValidatorApiServer::new(encoder_validator_service));

        let server = server_builder
            .bind(&config.encoder_validator_address())
            .await
            .map_err(|err| anyhow!(err.to_string()))?;
        let local_addr = server.local_addr();
        info!("Encoder validator service listening on {local_addr}");
        let grpc_server = tokio::spawn(server.serve().map_err(Into::into));

        Ok(grpc_server)
    }

    async fn start_object_services(
        config: &NodeConfig,
        allower: AllowPublicKeys,
        object_storage: Arc<InMemory>,
    ) -> Result<(InternalObjectServiceManager, ExternalObjectServiceManager)> {
        // TODO: for production make this configurable to use either Filesystem or Bucket
        let params = Arc::new(HttpParameters::default());

        let download_service =
            DownloadService::new(object_storage.clone(), config.network_key_pair().public());

        let mut external_object_manager =
            ExternalObjectServiceManager::new(config.network_key_pair(), params.clone(), allower)?;

        let mut internal_object_manager =
            InternalObjectServiceManager::new(config.network_key_pair(), params)?;

        external_object_manager
            .start(&config.external_object_address, download_service.clone())
            .await;
        internal_object_manager
            .start(&config.internal_object_address, download_service)
            .await;

        info!("Started internal and external object servers");

        Ok((internal_object_manager, external_object_manager))
    }

    async fn run_epoch(&self, epoch_duration: Duration) -> u64 {
        loop {
            // Wait for the specified epoch duration
            sleep(epoch_duration).await;

            // Get the current timestamp in milliseconds
            let timestamp_ms = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as u64;

            // Return the timestamp
            return timestamp_ms;
        }
    }

    /// This function awaits the completion of checkpoint execution of the current epoch,
    /// after which it iniitiates reconfiguration of the entire system.
    pub async fn monitor_reconfiguration(self: Arc<Self>) -> Result<()> {
        loop {
            let mut accumulator_guard = self.accumulator.lock().await;
            let accumulator = accumulator_guard.take().unwrap();

            let mut commit_executor = CommitExecutor::new(
                self.state_sync_handle.subscribe_to_synced_commits(),
                self.commit_store.clone(),
                self.state.clone(),
                accumulator.clone(),
                self.subscription_service_checkpoint_sender.clone(),
            );

            let cur_epoch_store = self.state.load_epoch_store_one_call_per_task();

            let stop_condition = commit_executor.run_epoch(cur_epoch_store.clone()).await;
            drop(commit_executor);

            // Safe to call because we are in the middle of reconfiguration.
            let latest_system_state = self
                .state
                .get_object_cache_reader()
                .get_system_state_object()
                .expect("Read System State object cannot fail");

            if let Err(err) = self.end_of_epoch_channel.send(latest_system_state.clone()) {
                if self.state.is_fullnode(&cur_epoch_store) {
                    warn!(
                        "Failed to send end of epoch notification to subscriber: {:?}",
                        err
                    );
                }
            }

            if let Some(encoder_client) = &self.encoder_client_service {
                let encoder_committee = latest_system_state.get_current_epoch_encoder_committee();
                info!("Updating encoder committee after reconfiguration");
                encoder_client.update_encoder_committee(&encoder_committee);
            }

            let new_epoch_start_state = latest_system_state.clone().into_epoch_start_state();

            self.auth_agg.store(Arc::new(
                self.auth_agg
                    .load()
                    .recreate_with_new_epoch_start_state(&new_epoch_start_state),
            ));

            let next_epoch_committee = new_epoch_start_state.get_committee();
            let next_epoch = next_epoch_committee.epoch();
            assert_eq!(cur_epoch_store.epoch() + 1, next_epoch);

            let new_encoder_committee = latest_system_state.get_current_epoch_encoder_committee();
            let mut encoder_committee_keys = BTreeSet::new();

            for (key, _) in new_encoder_committee.members() {
                if let Some(metadata) = new_encoder_committee.network_metadata.get(&key) {
                    let peer_key = metadata.network_key.clone().into_inner();
                    encoder_committee_keys.insert(peer_key);
                }
            }

            // Update the allower with new encoder committee keys
            self.allower.update(encoder_committee_keys);

            info!(
                next_epoch,
                "Finished epoch. About to reconfigure the system."
            );

            // The following code handles 4 different cases, depending on whether the node
            // was a validator in the previous epoch, and whether the node is a validator
            // in the new epoch.

            let new_validator_components = if let Some(ValidatorComponents {
                validator_server_handle,
                consensus_manager,
                consensus_store_pruner,
                consensus_adapter,
            }) = self.validator_components.lock().await.take()
            {
                info!("Reconfiguring the validator.");
                // Stop the old checkpoint service.
                // drop(checkpoint_service_exit);

                consensus_manager.shutdown().await;

                let new_epoch_store = self
                    .reconfigure_state(
                        &self.state,
                        &cur_epoch_store,
                        next_epoch_committee.clone(),
                        new_epoch_start_state,
                    )
                    .await;

                let new_accumulator = Arc::new(StateAccumulator::new(
                    self.state.get_accumulator_store().clone(),
                ));
                let weak_accumulator = Arc::downgrade(&new_accumulator);
                *accumulator_guard = Some(new_accumulator);

                consensus_store_pruner.prune(next_epoch).await;

                if self.state.is_validator(&new_epoch_store)
                    && self.config.consensus_config().is_some()
                {
                    // Only restart consensus if this node is still a validator in the new epoch.
                    Some(
                        Self::start_epoch_specific_validator_components(
                            &self.config,
                            self.state.clone(),
                            consensus_adapter,
                            new_epoch_store.clone(),
                            self.state_sync_handle.clone(),
                            consensus_manager,
                            consensus_store_pruner,
                            weak_accumulator,
                            validator_server_handle,
                        )
                        .await?,
                    )
                } else {
                    info!("This node is no longer a validator after reconfiguration");
                    None
                }
            } else {
                let new_epoch_store = self
                    .reconfigure_state(
                        &self.state,
                        &cur_epoch_store,
                        next_epoch_committee.clone(),
                        new_epoch_start_state,
                    )
                    .await;

                let new_accumulator = Arc::new(StateAccumulator::new(
                    self.state.get_accumulator_store().clone(),
                ));
                let weak_accumulator = Arc::downgrade(&new_accumulator);
                *accumulator_guard = Some(new_accumulator);

                if self.state.is_validator(&new_epoch_store)
                    && self.config.consensus_config().is_some()
                {
                    info!("Promoting the node from fullnode to validator, starting grpc server");

                    Some(
                        Self::construct_validator_components(
                            self.config.clone(),
                            self.state.clone(),
                            Arc::new(next_epoch_committee.clone()),
                            new_epoch_store.clone(),
                            self.state_sync_handle.clone(),
                            weak_accumulator,
                            self.consensus_store.clone(),
                        )
                        .await?,
                    )
                } else {
                    None
                }
            };
            *self.validator_components.lock().await = new_validator_components;

            // Force releasing current epoch store DB handle, because the
            // Arc<AuthorityPerEpochStore> may linger.
            cur_epoch_store.release_db_handles();

            if cfg!(msim)
                && !matches!(
                    self.config
                        .authority_store_pruning_config
                        .num_epochs_to_retain_for_commits(),
                    None | Some(u64::MAX) | Some(0)
                )
            {
                self.state
                    .prune_commits_for_eligible_epochs_for_testing(self.config.clone())
                    .await?;
            }

            info!("Reconfiguration finished");
        }
    }

    async fn shutdown(&self) {
        if let Some(validator_components) = &*self.validator_components.lock().await {
            validator_components.consensus_manager.shutdown().await;
        }

        if let Some(handle) = self.encoder_validator_server_handle.lock().await.take() {
            handle.abort();
        }
    }

    async fn reconfigure_state(
        &self,
        state: &Arc<AuthorityState>,
        cur_epoch_store: &AuthorityPerEpochStore,
        next_epoch_committee: Committee,
        next_epoch_start_system_state: EpochStartSystemState,
    ) -> Arc<AuthorityPerEpochStore> {
        let next_epoch = next_epoch_committee.epoch();

        let last_commit = self
            .commit_store
            .get_epoch_last_commit(cur_epoch_store.epoch())
            .expect("Error loading last checkpoint for current epoch")
            .expect("Could not load last checkpoint for current epoch");

        let epoch_start_configuration = EpochStartConfiguration::new(next_epoch_start_system_state);

        let new_epoch_store = self
            .state
            .reconfigure(
                cur_epoch_store,
                next_epoch_committee,
                epoch_start_configuration,
                last_commit.commit_ref.index,
            )
            .await
            .expect("Reconfigure authority state cannot fail");
        info!(next_epoch, "Node State has been reconfigured");
        assert_eq!(next_epoch, new_epoch_store.epoch());

        new_epoch_store
    }

    pub fn state(&self) -> Arc<AuthorityState> {
        self.state.clone()
    }

    // Testing-only API to start epoch close process.
    // For production code, please use the non-testing version.
    pub async fn close_epoch_for_testing(&self) -> SomaResult {
        let epoch_store = self.state.epoch_store_for_testing();
        self.close_epoch(&epoch_store).await
    }

    pub fn subscribe_to_epoch_change(&self) -> broadcast::Receiver<SystemState> {
        self.end_of_epoch_channel.subscribe()
    }

    /// Clone an AuthorityAggregator currently used in this node's
    /// QuorumDriver, if the node is a fullnode. After reconfig,
    /// QuorumDriver builds a new AuthorityAggregator. The caller
    /// of this function will mostly likely want to call this again
    /// to get a fresh one.
    pub fn clone_authority_aggregator(
        &self,
    ) -> Option<Arc<AuthorityAggregator<NetworkAuthorityClient>>> {
        self.transaction_orchestrator
            .as_ref()
            .map(|to| to.clone_authority_aggregator())
    }

    // TODO: TEST CLUSTER HELPERS - MOVE THESE TO WALLET CONTEXT / RPC / SDK

    pub async fn execute_transaction(
        &self,
        transaction: Transaction,
    ) -> SomaResult<(TransactionEffects, Option<Shard>)> {
        let (response, _) = self
            .transaction_orchestrator
            .as_ref()
            .expect("Node is not a fullnode")
            .execute_transaction_block(
                ExecuteTransactionRequest {
                    transaction,
                    include_input_objects: true,
                    include_output_objects: true,
                },
                ExecuteTransactionRequestType::WaitForLocalExecution,
                None,
            )
            .await
            .unwrap();
        Ok((response.effects.effects, response.shard))
    }

    pub fn get_config(&self) -> &NodeConfig {
        &self.config
    }
}

async fn build_http_servers(
    state: Arc<AuthorityState>,
    store: StateSyncStore, //RocksDbStore,
    transaction_orchestrator: &Option<Arc<TransactionOrchestrator<NetworkAuthorityClient>>>,
    config: &NodeConfig,
) -> Result<(HttpServers, Option<tokio::sync::mpsc::Sender<Checkpoint>>)> {
    // Validators do not expose these APIs
    if config.consensus_config().is_some() {
        return Ok((HttpServers::default(), None));
    }

    let mut router = axum::Router::new();

    let (subscription_service_checkpoint_sender, subscription_service_handle) =
        SubscriptionService::build();
    let rpc_router = {
        let mut rpc_service =
            rpc::api::RpcService::new(Arc::new(RestReadStore::new(state.clone(), store)));
        // rpc_service.with_server_version(server_version);

        if let Some(config) = config.rpc.clone() {
            rpc_service.with_config(config);
        }

        rpc_service.with_subscription_service(subscription_service_handle);

        if let Some(transaction_orchestrator) = transaction_orchestrator {
            rpc_service.with_executor(transaction_orchestrator.clone())
        }

        rpc_service.into_router().await
    };

    let layers = ServiceBuilder::new()
        .map_request(|mut request: axum::http::Request<_>| {
            if let Some(connect_info) = request.extensions().get::<soma_http::ConnectInfo>() {
                let axum_connect_info = axum::extract::ConnectInfo(connect_info.remote_addr);
                request.extensions_mut().insert(axum_connect_info);
            }
            request
        })
        // .layer(axum::middleware::from_fn(server_timing_middleware))
        // Setup a permissive CORS policy
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_methods([http::Method::GET, http::Method::POST])
                .allow_origin(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any),
        );

    router = router.merge(rpc_router).layer(layers);

    let https = if let Some((tls_config, https_address)) = config
        .rpc()
        .and_then(|config| config.tls_config().map(|tls| (tls, config.https_address())))
    {
        let https = soma_http::Builder::new()
            .tls_single_cert(tls_config.cert(), tls_config.key())
            .and_then(|builder| builder.serve(https_address, router.clone()))
            .map_err(|e| anyhow::anyhow!(e))?;

        info!(
            https_address =? https.local_addr(),
            "HTTPS rpc server listening on {}",
            https.local_addr()
        );

        Some(https)
    } else {
        None
    };

    let http = soma_http::Builder::new()
        .serve(&config.rpc_address, router)
        .map_err(|e| anyhow::anyhow!(e))?;

    info!(
        http_address =? http.local_addr(),
        "HTTP rpc server listening on {}",
        http.local_addr()
    );

    Ok((
        HttpServers {
            http: Some(http),
            https,
        },
        Some(subscription_service_checkpoint_sender),
    ))
}
