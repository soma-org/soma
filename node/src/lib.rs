use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use authority::{
    authority::{AuthorityState, ExecutionEnv},
    authority_aggregator::AuthorityAggregator,
    authority_client::NetworkAuthorityClient,
    authority_per_epoch_store::AuthorityPerEpochStore,
    authority_server::ValidatorService,
    authority_store::AuthorityStore,
    authority_store_pruner::{ObjectsCompactionFilter, PrunerWatermarks},
    authority_store_tables::{
        AuthorityPerpetualTables, AuthorityPerpetualTablesOptions, AuthorityPrunerTables,
    },
    backpressure_manager::BackpressureManager,
    cache::build_execution_cache,
    checkpoints::{
        checkpoint_executor::{CheckpointExecutor, StopReason},
        CheckpointService, CheckpointStore, SendCheckpointToStateSync, SubmitCheckpointToConsensus,
    },
    consensus_adapter::{ConsensusAdapter, ConsensusClient},
    consensus_handler::ConsensusHandlerInitializer,
    consensus_manager::{ConsensusManager, UpdatableConsensusClient},
    consensus_store_pruner::ConsensusStorePruner,
    consensus_validator::TxValidator,
    encoder_client::EncoderClientService,
    execution_scheduler::SchedulingSource,
    global_state_hasher::GlobalStateHasher,
    reconfiguration::ReconfigurationInitiator,
    rpc_index::RpcIndexStore,
    server::{ServerBuilder, TLS_SERVER_NAME},
    shared_obj_version_manager::Schedulable,
    start_epoch::{EpochStartConfigTrait, EpochStartConfiguration},
    storage::{RestReadStore, RocksDbStore},
    tonic_gen::validator_server::ValidatorServer,
    transaction_orchestrator::TransactionOrchestrator,
    validator_tx_finalizer::ValidatorTxFinalizer,
};
use encoder_validator_api::{
    service::EncoderValidatorService,
    tonic_gen::encoder_validator_api_server::EncoderValidatorApiServer,
};
use futures::{future::BoxFuture, TryFutureExt};
use parking_lot::RwLock;
use rpc::api::{subscription::SubscriptionService, ServerVersion};

use store::rocks::default_db_options;
use sync::builder::{DiscoveryHandle, P2pBuilder, StateSyncHandle};
use tower::ServiceBuilder;

use protocol_config::ProtocolConfig;
use std::{
    collections::{BTreeSet, HashMap},
    future::Future,
    str::FromStr as _,
    sync::{Arc, Weak},
    time::{Duration, SystemTime},
};
use tokio::{
    sync::{broadcast, mpsc::Sender, oneshot, Mutex},
    task::JoinHandle,
    time::sleep,
};
use tracing::{debug, error, error_span, info, warn, Instrument};
use types::{
    base::AuthorityName,
    client::Config,
    committee::Committee,
    config::node_config::{
        ConsensusConfig, ForkCrashBehavior, ForkRecoveryConfig, NodeConfig, RunWithRange,
    },
    consensus::{AuthorityCapabilities, ConsensusTransaction, ConsensusTransactionKind},
    crypto::KeypairTraits,
    digests::{ChainIdentifier, CheckpointDigest, TransactionDigest, TransactionEffectsDigest},
    error::{SomaError, SomaResult},
    full_checkpoint_content::Checkpoint,
    storage::committee_store::CommitteeStore,
    supported_protocol_versions::SupportedProtocolVersions,
    sync::{
        active_peers::{self, ActivePeers},
        channel_manager::{ChannelManager, ChannelManagerRequest},
    },
    system_state::{
        epoch_start::{EpochStartSystemState, EpochStartSystemStateTrait},
        SystemState, SystemStateTrait,
    },
    transaction::{VerifiedCertificate, VerifiedExecutableTransaction},
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
    validator_server_handle: SpawnOnce,
    consensus_manager: Arc<ConsensusManager>,
    consensus_store_pruner: ConsensusStorePruner,
    consensus_adapter: Arc<ConsensusAdapter>,
}

enum SpawnOnce {
    // Mutex is only needed to make SpawnOnce Send
    Unstarted(oneshot::Receiver<()>, Mutex<BoxFuture<'static, ()>>),
    #[allow(unused)]
    Started(JoinHandle<()>),
}

impl SpawnOnce {
    pub fn new(
        ready_rx: oneshot::Receiver<()>,
        future: impl Future<Output = ()> + Send + 'static,
    ) -> Self {
        Self::Unstarted(ready_rx, Mutex::new(Box::pin(future)))
    }

    pub async fn start(self) -> Self {
        match self {
            Self::Unstarted(ready_rx, future) => {
                let future = future.into_inner();
                let handle = tokio::spawn(future);
                ready_rx.await.unwrap();
                Self::Started(handle)
            }
            Self::Started(_) => self,
        }
    }
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

const DEFAULT_GRPC_CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

pub struct SomaNode {
    config: NodeConfig,
    validator_components: Mutex<Option<ValidatorComponents>>,

    /// The http servers responsible for serving RPC traffic (gRPC and JSON-RPC)
    #[allow(unused)]
    http_servers: HttpServers,

    state: Arc<AuthorityState>,
    transaction_orchestrator: Option<Arc<TransactionOrchestrator<NetworkAuthorityClient>>>,

    state_sync_handle: StateSyncHandle,
    checkpoint_store: Arc<CheckpointStore>,
    global_state_hasher: Mutex<Option<Arc<GlobalStateHasher>>>,

    /// Broadcast channel to send the starting system state for the next epoch.
    end_of_epoch_channel: broadcast::Sender<SystemState>,

    /// Broadcast channel to notify state-sync for new validator peers.
    // trusted_peer_change_tx: watch::Sender<TrustedPeerChangeEvent>,
    backpressure_manager: Arc<BackpressureManager>,

    // _db_checkpoint_handle: Option<tokio::sync::broadcast::Sender<()>>,
    #[cfg(msim)]
    sim_state: SimState,

    // _state_snapshot_uploader_handle: Option<broadcast::Sender<()>>,
    // Channel to allow signaling upstream to shutdown sui-node
    shutdown_channel_tx: broadcast::Sender<Option<RunWithRange>>,

    /// AuthorityAggregator of the network, created at start and beginning of each epoch.
    /// Use ArcSwap so that we could mutate it without taking mut reference.
    // TODO: Eventually we can make this auth aggregator a shared reference so that this
    // update will automatically propagate to other uses.
    auth_agg: Arc<ArcSwap<AuthorityAggregator<NetworkAuthorityClient>>>,

    subscription_service_checkpoint_sender: Option<tokio::sync::mpsc::Sender<Checkpoint>>,

    encoder_validator_server_handle: Mutex<Option<JoinHandle<Result<()>>>>,
    encoder_client_service: Option<Arc<EncoderClientService>>,
}

impl SomaNode {
    pub async fn start(config: NodeConfig) -> Result<Arc<SomaNode>> {
        Self::start_async(config, ServerVersion::new("soma-node", "unknown")).await
    }

    pub async fn start_async(
        config: NodeConfig,
        server_version: ServerVersion,
    ) -> Result<Arc<SomaNode>> {
        let mut config = config.clone();
        if config.supported_protocol_versions.is_none() {
            info!(
                "populating config.supported_protocol_versions with default {:?}",
                SupportedProtocolVersions::SYSTEM_DEFAULT
            );
            config.supported_protocol_versions = Some(SupportedProtocolVersions::SYSTEM_DEFAULT);
        }

        let run_with_range = config.run_with_range;
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
        let checkpoint_store = CheckpointStore::new(
            &config.db_path().join("checkpoints"),
            pruner_watermarks.clone(),
        );

        Self::check_and_recover_forks(
            &checkpoint_store,
            is_validator,
            config.fork_recovery.as_ref(),
        )
        .await?;

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

        let backpressure_manager =
            BackpressureManager::new_from_checkpoint_store(&checkpoint_store);

        let store = AuthorityStore::open(perpetual_tables, &genesis, &config).await?;

        let cur_epoch = store.get_recovery_epoch_at_restart()?;
        let committee = committee_store
            .get_committee(&cur_epoch)?
            .expect("Committee of the current epoch must exist");
        let epoch_start_configuration = store
            .get_epoch_start_configuration()?
            .expect("EpochStartConfiguration of the current epoch must exist");

        let cache_traits = build_execution_cache(
            &config.execution_cache,
            &store,
            backpressure_manager.clone(),
        );

        let auth_agg = {
            Arc::new(ArcSwap::new(Arc::new(
                AuthorityAggregator::new_from_epoch_start_state(
                    epoch_start_configuration.epoch_start_state(),
                    &committee_store,
                ),
            )))
        };

        let chain_id = ChainIdentifier::from(*genesis.checkpoint().digest());
        let chain = match config.chain_override_for_testing {
            Some(chain) => chain,
            None => ChainIdentifier::from(*genesis.checkpoint().digest()).chain(),
        };

        let epoch_options = default_db_options().optimize_db_for_write_throughput(4);
        let epoch_store = AuthorityPerEpochStore::new(
            config.protocol_public_key(),
            committee.clone(),
            &config.db_path().join("store"),
            Some(epoch_options.options),
            epoch_start_configuration,
            (chain_id, chain),
            checkpoint_store
                .get_highest_executed_checkpoint_seq_number()
                .expect("checkpoint store read cannot fail")
                .unwrap_or(0),
        )?;

        info!("created epoch store");

        let effective_buffer_stake = epoch_store.get_effective_buffer_stake_bps();
        let default_buffer_stake = epoch_store
            .protocol_config()
            .buffer_stake_for_protocol_upgrade_bps();
        if effective_buffer_stake != default_buffer_stake {
            warn!(
                ?effective_buffer_stake,
                ?default_buffer_stake,
                "buffer_stake_for_protocol_upgrade_bps is currently overridden"
            );
        }

        checkpoint_store.insert_genesis_checkpoint(
            genesis.checkpoint(),
            genesis.checkpoint_contents().clone(),
            &epoch_store,
        );

        info!("creating state sync store");
        let state_sync_store = RocksDbStore::new(
            cache_traits.clone(),
            committee_store.clone(),
            checkpoint_store.clone(),
        );

        let rpc_index = if is_full_node && config.rpc().is_some_and(|rpc| rpc.enable_indexing()) {
            Some(Arc::new(
                RpcIndexStore::new(&config.db_path(), &store, &checkpoint_store).await,
            ))
        } else {
            None
        };

        let chain_identifier = epoch_store.get_chain_identifier();

        // TOD:  let (trusted_peer_change_tx, trusted_peer_change_rx) = watch::channel(Default::default());
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
        // TODO: send_trusted_peer_change(
        //     &config,
        //     &trusted_peer_change_tx,
        //     epoch_store.epoch_start_state(),
        // )
        // .expect("Initial trusted peers must be set");

        //  info!("start snapshot upload");
        // TODO: Start uploading state snapshot to remote store
        // let state_snapshot_handle = Self::start_state_snapshot(
        //     &config,
        //     checkpoint_store.clone(),
        //     chain_identifier,
        // )?;

        // TODO: Start uploading db checkpoints to remote store
        // info!("start db checkpoint");
        // let (db_checkpoint_config, db_checkpoint_handle) = Self::start_db_checkpoint(
        //     &config,
        //     state_snapshot_handle.is_some(),
        // )?;

        let authority_name = config.protocol_public_key();
        let validator_tx_finalizer =
            Arc::new(ValidatorTxFinalizer::new(auth_agg.clone(), authority_name));

        info!("create authority state");
        let state = AuthorityState::new(
            authority_name,
            secret,
            config.supported_protocol_versions.unwrap(),
            store.clone(),
            cache_traits.clone(),
            epoch_store.clone(),
            committee_store.clone(),
            rpc_index,
            checkpoint_store.clone(),
            genesis.objects(),
            config.clone(),
            Some(validator_tx_finalizer),
            chain_identifier,
            pruner_db,
            pruner_watermarks,
        )
        .await;

        // ensure genesis txn was executed
        if epoch_store.epoch() == 0 {
            let txn = &genesis.transaction();
            let span = error_span!("genesis_txn", tx_digest = ?txn.digest());
            let transaction = types::transaction::VerifiedExecutableTransaction::new_unchecked(
                types::transaction::ExecutableTransaction::new_from_data_and_sig(
                    genesis.transaction().data().clone(),
                    types::transaction::CertificateProof::Checkpoint(0, 0),
                ),
            );
            state
                .try_execute_immediately(
                    &transaction,
                    ExecutionEnv::new().with_scheduling_source(SchedulingSource::NonFastPath),
                    &epoch_store,
                )
                .instrument(span)
                .await
                .unwrap();
        }

        //  if config
        //     .expensive_safety_check_config
        //     .enable_secondary_index_checks()
        //     && let Some(indexes) = state.indexes.clone()
        // {
        // TODO:    authority::verify_indexes::verify_indexes(
        //         state.get_global_state_hash_store().as_ref(),
        //         indexes,
        //     )
        //     .expect("secondary indexes are inconsistent");
        // }

        let (end_of_epoch_channel, end_of_epoch_receiver) =
            broadcast::channel(config.end_of_epoch_broadcast_channel_capacity);

        let encoder_client_service = if is_full_node {
            // Only fullnodes send to encoders, not validators
            Some(Arc::new(EncoderClientService::new(
                config.protocol_key_pair().copy(),
                config.network_key_pair(),
            )))
        } else {
            None
        };

        let transaction_orchestrator = if is_full_node && run_with_range.is_none() {
            Some(Arc::new(TransactionOrchestrator::new_with_auth_aggregator(
                auth_agg.load_full(),
                state.clone(),
                end_of_epoch_receiver,
                &config.db_path(),
                &config,
                encoder_client_service.clone(),
            )))
        } else {
            None
        };

        let (http_servers, subscription_service_checkpoint_sender) = build_http_servers(
            state.clone(),
            state_sync_store,
            &transaction_orchestrator.clone(),
            &config,
            server_version,
        )
        .await?;

        let global_state_hasher = Arc::new(GlobalStateHasher::new(
            cache_traits.global_state_hash_store.clone(),
        ));

        let authority_names_to_peer_ids = epoch_store
            .epoch_start_state()
            .get_authority_names_to_peer_ids();

        let authority_names_to_peer_ids = ArcSwap::from_pointee(authority_names_to_peer_ids);

        let validator_components = if state.is_validator(&epoch_store) && is_validator {
            let (components, _) = futures::join!(
                Self::construct_validator_components(
                    config.clone(),
                    state.clone(),
                    committee,
                    epoch_store.clone(),
                    checkpoint_store.clone(),
                    state_sync_handle.clone(),
                    Arc::downgrade(&global_state_hasher),
                    backpressure_manager.clone(),
                ),
                Self::reexecute_pending_consensus_certs(&epoch_store, &state,)
            );
            let mut components = components?;

            components.consensus_adapter.submit_recovered(&epoch_store);

            // Start the gRPC server
            components.validator_server_handle = components.validator_server_handle.start().await;

            Some(components)
        } else {
            None
        };

        // setup shutdown channel
        let (shutdown_channel, _) = broadcast::channel::<Option<RunWithRange>>(1);

        let encoder_validator_server_handle = if is_full_node {
            info!("Starting encoder validator service for fullnode");
            Some(
                Self::start_grpc_encoder_service(&config, state.clone(), checkpoint_store.clone())
                    .await?,
            )
        } else {
            None
        };

        let node = Self {
            config,
            validator_components: Mutex::new(validator_components),
            http_servers,
            state,
            transaction_orchestrator,
            state_sync_handle,
            checkpoint_store,
            global_state_hasher: Mutex::new(Some(global_state_hasher)),
            end_of_epoch_channel,
            backpressure_manager,
            shutdown_channel_tx: shutdown_channel,
            auth_agg,
            subscription_service_checkpoint_sender,

            encoder_validator_server_handle: Mutex::new(encoder_validator_server_handle),
            encoder_client_service,
            // connection_monitor_status,
            #[cfg(msim)]
            sim_state: Default::default(),
        };

        info!("SomaNode started!");
        let node = Arc::new(node);
        let node_copy = node.clone();

        tokio::spawn(async move {
            let result = Self::monitor_reconfiguration(node_copy, epoch_store).await;
            if let Err(error) = result {
                warn!("Reconfiguration finished with error {:?}", error);
            }
        });

        Ok(node)
    }

    pub fn subscribe_to_epoch_change(&self) -> broadcast::Receiver<SystemState> {
        self.end_of_epoch_channel.subscribe()
    }

    pub fn subscribe_to_shutdown_channel(&self) -> broadcast::Receiver<Option<RunWithRange>> {
        self.shutdown_channel_tx.subscribe()
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
        state_sync_store: RocksDbStore,
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
        checkpoint_store: Arc<CheckpointStore>,
        state_sync_handle: StateSyncHandle,
        global_state_hasher: Weak<GlobalStateHasher>,
        backpressure_manager: Arc<BackpressureManager>,
    ) -> Result<ValidatorComponents> {
        let mut config_clone = config.clone();
        let consensus_config = config_clone
            .consensus_config
            .as_mut()
            .ok_or_else(|| anyhow!("Validator is missing consensus config"))?;

        let client = Arc::new(UpdatableConsensusClient::new());
        let consensus_adapter = Arc::new(Self::construct_consensus_adapter(
            &committee,
            consensus_config,
            state.name,
            // connection_monitor_status.clone(),
            epoch_store.protocol_config().clone(),
            client.clone(),
            checkpoint_store.clone(),
        ));
        let consensus_manager = Arc::new(ConsensusManager::new(&config, consensus_config, client));

        // This only gets started up once, not on every epoch. (Make call to remove every epoch.)
        let consensus_store_pruner = ConsensusStorePruner::new(
            consensus_manager.get_storage_base_path(),
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
            checkpoint_store,
            epoch_store,
            state_sync_handle,
            consensus_manager,
            consensus_store_pruner,
            global_state_hasher,
            backpressure_manager,
            validator_server_handle,
        )
        .await
    }

    async fn start_epoch_specific_validator_components(
        config: &NodeConfig,
        state: Arc<AuthorityState>,
        consensus_adapter: Arc<ConsensusAdapter>,
        checkpoint_store: Arc<CheckpointStore>,
        epoch_store: Arc<AuthorityPerEpochStore>,
        state_sync_handle: StateSyncHandle,
        consensus_manager: Arc<ConsensusManager>,
        consensus_store_pruner: ConsensusStorePruner,
        state_hasher: Weak<GlobalStateHasher>,
        backpressure_manager: Arc<BackpressureManager>,
        validator_server_handle: SpawnOnce,
    ) -> Result<ValidatorComponents> {
        let checkpoint_service = Self::build_checkpoint_service(
            config,
            consensus_adapter.clone(),
            checkpoint_store.clone(),
            epoch_store.clone(),
            state.clone(),
            state_sync_handle,
            state_hasher,
        );

        // create a new map that gets injected into both the consensus handler and the consensus adapter
        // the consensus handler will write values forwarded from consensus, and the consensus adapter
        // will read the values to make decisions about which validator submits a transaction to consensus
        let low_scoring_authorities = Arc::new(ArcSwap::new(Arc::new(HashMap::new())));

        consensus_adapter.swap_low_scoring_authorities(low_scoring_authorities.clone());

        let consensus_handler_initializer = ConsensusHandlerInitializer::new(
            state.clone(),
            checkpoint_service.clone(),
            epoch_store.clone(),
            consensus_adapter.clone(),
            low_scoring_authorities,
            backpressure_manager,
        );

        info!("Starting consensus manager asynchronously");

        // Spawn consensus startup asynchronously to avoid blocking other components
        tokio::spawn({
            let config = config.clone();
            let epoch_store = epoch_store.clone();
            let tx_validator = TxValidator::new(state.clone(), checkpoint_service.clone());
            let consensus_manager = consensus_manager.clone();
            async move {
                consensus_manager
                    .start(
                        &config,
                        epoch_store,
                        consensus_handler_initializer,
                        tx_validator,
                    )
                    .await;
            }
        });
        let replay_waiter = consensus_manager.replay_waiter();

        info!("Spawning checkpoint service");
        let replay_waiter = if std::env::var("DISABLE_REPLAY_WAITER").is_ok() {
            None
        } else {
            Some(replay_waiter)
        };
        checkpoint_service
            .spawn(epoch_store.clone(), replay_waiter)
            .await;

        Ok(ValidatorComponents {
            validator_server_handle,
            consensus_manager,
            consensus_store_pruner,
            consensus_adapter,
        })
    }

    fn build_checkpoint_service(
        config: &NodeConfig,
        consensus_adapter: Arc<ConsensusAdapter>,
        checkpoint_store: Arc<CheckpointStore>,
        epoch_store: Arc<AuthorityPerEpochStore>,
        state: Arc<AuthorityState>,
        state_sync_handle: StateSyncHandle,
        state_hasher: Weak<GlobalStateHasher>,
    ) -> Arc<CheckpointService> {
        let epoch_start_timestamp_ms = epoch_store.epoch_start_state().epoch_start_timestamp_ms();
        let epoch_duration_ms = epoch_store.epoch_start_state().epoch_duration_ms();

        debug!(
            "Starting checkpoint service with epoch start timestamp {}
            and epoch duration {}",
            epoch_start_timestamp_ms, epoch_duration_ms
        );

        let checkpoint_output = Box::new(SubmitCheckpointToConsensus {
            sender: consensus_adapter,
            signer: state.secret.clone(),
            authority: config.protocol_public_key(),
            next_reconfiguration_timestamp_ms: epoch_start_timestamp_ms
                .checked_add(epoch_duration_ms)
                .expect("Overflow calculating next_reconfiguration_timestamp_ms"),
        });

        let certified_checkpoint_output = SendCheckpointToStateSync::new(state_sync_handle);
        let max_tx_per_checkpoint = max_tx_per_checkpoint(epoch_store.protocol_config());
        let max_checkpoint_size_bytes =
            epoch_store.protocol_config().max_checkpoint_size_bytes() as usize;

        CheckpointService::build(
            state.clone(),
            checkpoint_store,
            epoch_store,
            state.get_transaction_cache_reader().clone(),
            state_hasher,
            checkpoint_output,
            Box::new(certified_checkpoint_output),
            max_tx_per_checkpoint,
            max_checkpoint_size_bytes,
        )
    }

    fn construct_consensus_adapter(
        committee: &Committee,
        consensus_config: &ConsensusConfig,
        authority: AuthorityName,
        // connection_monitor_status: Arc<ConnectionMonitorStatus>,
        protocol_config: ProtocolConfig,
        consensus_client: Arc<dyn ConsensusClient>,
        checkpoint_store: Arc<CheckpointStore>,
    ) -> ConsensusAdapter {
        // The consensus adapter allows the authority to send user certificates through consensus.

        ConsensusAdapter::new(
            consensus_client,
            checkpoint_store,
            authority,
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
    ) -> Result<SpawnOnce> {
        let validator_service = ValidatorService::new(state.clone(), consensus_adapter);

        let mut server_conf = Config::new();
        server_conf.connect_timeout = Some(DEFAULT_GRPC_CONNECT_TIMEOUT);
        server_conf.http2_keepalive_interval = Some(DEFAULT_GRPC_CONNECT_TIMEOUT);
        server_conf.http2_keepalive_timeout = Some(DEFAULT_GRPC_CONNECT_TIMEOUT);
        // TODO: define load shed and global concurrency limit in config
        // server_conf.global_concurrency_limit = config.grpc_concurrency_limit;
        // server_conf.load_shed = config.grpc_load_shed;
        let mut server_builder = ServerBuilder::from_config(&server_conf);

        server_builder = server_builder.add_service(ValidatorServer::new(validator_service));

        let tls_config = soma_tls::create_rustls_server_config(
            config.network_key_pair().clone().private_key().into_inner(),
            TLS_SERVER_NAME.to_string(),
        );

        let network_address = config.network_address().clone();

        let (ready_tx, ready_rx) = oneshot::channel();

        Ok(SpawnOnce::new(ready_rx, async move {
            let server = server_builder
                .bind(&network_address, Some(tls_config))
                .await
                .unwrap_or_else(|err| panic!("Failed to bind to {network_address}: {err}"));
            let local_addr = server.local_addr();
            info!("Listening to traffic on {local_addr}");
            ready_tx.send(()).unwrap();
            if let Err(err) = server.serve().await {
                info!("Server stopped: {err}");
            }
            info!("Server stopped");
        }))
    }

    /// Re-executes pending consensus certificates, which may not have been committed to disk
    /// before the node restarted. This is necessary for the following reasons:
    ///
    /// 1. For any transaction for which we returned signed effects to a client, we must ensure
    ///    that we have re-executed the transaction before we begin accepting grpc requests.
    ///    Otherwise we would appear to have forgotten about the transaction.
    /// 2. While this is running, we are concurrently waiting for all previously built checkpoints
    ///    to be rebuilt. Since there may be dependencies in either direction (from checkpointed
    ///    consensus transactions to pending consensus transactions, or vice versa), we must
    ///    re-execute pending consensus transactions to ensure that both processes can complete.
    /// 3. Also note that for any pending consensus transactions for which we wrote a signed effects
    ///    digest to disk, we must re-execute using that digest as the expected effects digest,
    ///    to ensure that we cannot arrive at different effects than what we previously signed.
    async fn reexecute_pending_consensus_certs(
        epoch_store: &Arc<AuthorityPerEpochStore>,
        state: &Arc<AuthorityState>,
    ) {
        let mut pending_consensus_certificates = Vec::new();
        let mut additional_certs = Vec::new();

        for tx in epoch_store.get_all_pending_consensus_transactions() {
            match tx.kind {
                // Shared object txns cannot be re-executed at this point, because we must wait for
                // consensus replay to assign shared object versions.
                ConsensusTransactionKind::CertifiedTransaction(tx) if !tx.is_consensus_tx() => {
                    let tx = *tx;
                    // new_unchecked is safe because we never submit a transaction to consensus
                    // without verifying it
                    let tx = VerifiedExecutableTransaction::new_from_certificate(
                        VerifiedCertificate::new_unchecked(tx),
                    );
                    // we only need to re-execute if we previously signed the effects (which indicates we
                    // returned the effects to a client).
                    if let Some(fx_digest) = epoch_store
                        .get_signed_effects_digest(tx.digest())
                        .expect("db error")
                    {
                        pending_consensus_certificates.push((
                            Schedulable::Transaction(tx),
                            ExecutionEnv::new().with_expected_effects_digest(fx_digest),
                        ));
                    } else {
                        additional_certs.push((
                            Schedulable::Transaction(tx),
                            ExecutionEnv::new()
                                .with_scheduling_source(SchedulingSource::NonFastPath),
                        ));
                    }
                }
                _ => (),
            }
        }

        let digests = pending_consensus_certificates
            .iter()
            // unwrap_digest okay because only user certs are in pending_consensus_certificates
            .map(|(tx, _)| *tx.key().unwrap_digest())
            .collect::<Vec<_>>();

        info!(
            "reexecuting {} pending consensus certificates: {:?}",
            digests.len(),
            digests
        );

        state
            .execution_scheduler()
            .enqueue(pending_consensus_certificates, epoch_store);
        state
            .execution_scheduler()
            .enqueue(additional_certs, epoch_store);

        // If this times out, the validator will still almost certainly start up fine. But, it is
        // possible that it may temporarily "forget" about transactions that it had previously
        // executed. This could confuse clients in some circumstances. However, the transactions
        // are still in pending_consensus_certificates, so we cannot lose any finality guarantees.
        let timeout = if cfg!(msim) { 120 } else { 60 };
        if tokio::time::timeout(
            std::time::Duration::from_secs(timeout),
            state
                .get_transaction_cache_reader()
                .notify_read_executed_effects_digests(&digests),
        )
        .await
        .is_err()
        {
            // Log all the digests that were not executed to help debugging.
            let executed_effects_digests = state
                .get_transaction_cache_reader()
                .multi_get_executed_effects_digests(&digests);
            let pending_digests = digests
                .iter()
                .zip(executed_effects_digests.iter())
                .filter_map(|(digest, executed_effects_digest)| {
                    if executed_effects_digest.is_none() {
                        Some(digest)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            debug!(
                "Timed out waiting for effects digests to be executed: {:?}",
                pending_digests
            );
        }
    }

    pub fn state(&self) -> Arc<AuthorityState> {
        self.state.clone()
    }

    async fn start_grpc_encoder_service(
        config: &NodeConfig,
        state: Arc<AuthorityState>,
        checkpoint_store: Arc<CheckpointStore>,
    ) -> Result<tokio::task::JoinHandle<Result<()>>> {
        let encoder_validator_service =
            EncoderValidatorService::new(state.clone(), checkpoint_store.clone());

        let server_conf = Config::new();
        let mut server_builder = ServerBuilder::from_config(&server_conf);

        server_builder =
            server_builder.add_service(EncoderValidatorApiServer::new(encoder_validator_service));

        let tls_config = soma_tls::create_rustls_server_config(
            config.network_key_pair().clone().private_key().into_inner(),
            "soma-encoder-sync".to_string(),
        );

        let server = server_builder
            .bind(&config.encoder_validator_address(), Some(tls_config))
            .await
            .map_err(|err| anyhow!(err.to_string()))?;
        let local_addr = server.local_addr();
        info!("Encoder validator service listening on {local_addr}");
        let grpc_server = tokio::spawn(server.serve().map_err(Into::into));

        Ok(grpc_server)
    }

    /// This function awaits the completion of checkpoint execution of the current epoch,
    /// after which it initiates reconfiguration of the entire system.
    pub async fn monitor_reconfiguration(
        self: Arc<Self>,
        mut epoch_store: Arc<AuthorityPerEpochStore>,
    ) -> Result<()> {
        loop {
            let mut hasher_guard = self.global_state_hasher.lock().await;
            let hasher = hasher_guard.take().unwrap();
            info!(
                "Creating checkpoint executor for epoch {}",
                epoch_store.epoch()
            );
            let checkpoint_executor = CheckpointExecutor::new(
                epoch_store.clone(),
                self.checkpoint_store.clone(),
                self.state.clone(),
                hasher.clone(),
                self.backpressure_manager.clone(),
                self.config.checkpoint_executor_config.clone(),
                self.subscription_service_checkpoint_sender.clone(),
            );

            let run_with_range = self.config.run_with_range;

            let cur_epoch_store = self.state.load_epoch_store_one_call_per_task();

            // Advertise capabilities to committee, if we are a validator.
            if let Some(components) = &*self.validator_components.lock().await {
                // TODO: without this sleep, the consensus message is not delivered reliably.
                tokio::time::sleep(Duration::from_millis(1)).await;

                let config = cur_epoch_store.protocol_config();
                let supported_protocol_versions = self
                    .config
                    .supported_protocol_versions
                    .expect("Supported versions should be populated")
                    // no need to send digests of versions less than the current version
                    .truncate_below(config.version);

                let transaction =
                    ConsensusTransaction::new_capability_notification(AuthorityCapabilities::new(
                        self.state.name,
                        cur_epoch_store.get_chain_identifier().chain(),
                        supported_protocol_versions,
                    ));
                info!(?transaction, "submitting capabilities to consensus");
                components.consensus_adapter.submit(
                    transaction,
                    None,
                    &cur_epoch_store,
                    None,
                    None,
                )?;
            }

            let stop_condition = checkpoint_executor.run_epoch(run_with_range).await;

            if stop_condition == StopReason::RunWithRangeCondition {
                SomaNode::shutdown(&self).await;
                self.shutdown_channel_tx
                    .send(run_with_range)
                    .expect("RunWithRangeCondition met but failed to send shutdown message");
                return Ok(());
            }

            // Safe to call because we are in the middle of reconfiguration.
            let latest_system_state = self
                .state
                .get_object_cache_reader()
                .get_system_state_object()
                .expect("Read Soma System State object cannot fail");

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

            let new_epoch_start_state = latest_system_state.into_epoch_start_state();

            self.auth_agg.store(Arc::new(
                self.auth_agg
                    .load()
                    .recreate_with_new_epoch_start_state(&new_epoch_start_state),
            ));

            let next_epoch_committee = new_epoch_start_state.get_committee();
            let next_epoch = next_epoch_committee.epoch();
            assert_eq!(cur_epoch_store.epoch() + 1, next_epoch);

            info!(
                next_epoch,
                "Finished executing all checkpoints in epoch. About to reconfigure the system."
            );

            // TODO: let _ = send_trusted_peer_change(
            //     &self.config,
            //     &self.trusted_peer_change_tx,
            //     &new_epoch_start_state,
            // );

            let mut validator_components_lock_guard = self.validator_components.lock().await;

            // The following code handles 4 different cases, depending on whether the node
            // was a validator in the previous epoch, and whether the node is a validator
            // in the new epoch.
            let new_epoch_store = self
                .reconfigure_state(
                    &self.state,
                    &cur_epoch_store,
                    next_epoch_committee.clone(),
                    new_epoch_start_state,
                    hasher.clone(),
                )
                .await;

            let new_validator_components = if let Some(ValidatorComponents {
                validator_server_handle,
                consensus_manager,
                consensus_store_pruner,
                consensus_adapter,
            }) = validator_components_lock_guard.take()
            {
                info!("Reconfiguring the validator.");

                consensus_manager.shutdown().await;
                info!("Consensus has shut down.");

                info!("Epoch store finished reconfiguration.");

                // No other components should be holding a strong reference to state hasher
                // at this point. Confirm here before we swap in the new hasher.
                let new_hasher = Arc::new(GlobalStateHasher::new(
                    self.state.get_global_state_hash_store().clone(),
                ));
                let weak_hasher = Arc::downgrade(&new_hasher);
                *hasher_guard = Some(new_hasher);

                consensus_store_pruner.prune(next_epoch).await;

                if self.state.is_validator(&new_epoch_store) {
                    // Only restart consensus if this node is still a validator in the new epoch.
                    Some(
                        Self::start_epoch_specific_validator_components(
                            &self.config,
                            self.state.clone(),
                            consensus_adapter,
                            self.checkpoint_store.clone(),
                            new_epoch_store.clone(),
                            self.state_sync_handle.clone(),
                            consensus_manager,
                            consensus_store_pruner,
                            weak_hasher,
                            self.backpressure_manager.clone(),
                            validator_server_handle,
                        )
                        .await?,
                    )
                } else {
                    info!("This node is no longer a validator after reconfiguration");
                    None
                }
            } else {
                // No other components should be holding a strong reference to state hasher
                // at this point. Confirm here before we swap in the new hasher.

                let new_hasher = Arc::new(GlobalStateHasher::new(
                    self.state.get_global_state_hash_store().clone(),
                ));
                let weak_hasher = Arc::downgrade(&new_hasher);
                *hasher_guard = Some(new_hasher);

                if self.state.is_validator(&new_epoch_store) {
                    info!("Promoting the node from fullnode to validator, starting grpc server");

                    let mut components = Self::construct_validator_components(
                        self.config.clone(),
                        self.state.clone(),
                        Arc::new(next_epoch_committee.clone()),
                        new_epoch_store.clone(),
                        self.checkpoint_store.clone(),
                        self.state_sync_handle.clone(),
                        weak_hasher,
                        self.backpressure_manager.clone(),
                    )
                    .await?;

                    components.validator_server_handle =
                        components.validator_server_handle.start().await;

                    Some(components)
                } else {
                    None
                }
            };
            *validator_components_lock_guard = new_validator_components;

            // Force releasing current epoch store DB handle, because the
            // Arc<AuthorityPerEpochStore> may linger.
            cur_epoch_store.release_db_handles();

            if cfg!(msim)
                && !matches!(
                    self.config
                        .authority_store_pruning_config
                        .num_epochs_to_retain_for_checkpoints(),
                    None | Some(u64::MAX) | Some(0)
                )
            {
                self.state
                    .prune_checkpoints_for_eligible_epochs_for_testing(self.config.clone())
                    .await?;
            }

            epoch_store = new_epoch_store;
            info!("Reconfiguration finished");
        }
    }

    async fn shutdown(&self) {
        if let Some(validator_components) = &*self.validator_components.lock().await {
            validator_components.consensus_manager.shutdown().await;
        }
    }

    async fn reconfigure_state(
        &self,
        state: &Arc<AuthorityState>,
        cur_epoch_store: &AuthorityPerEpochStore,
        next_epoch_committee: Committee,
        next_epoch_start_system_state: EpochStartSystemState,
        global_state_hasher: Arc<GlobalStateHasher>,
    ) -> Arc<AuthorityPerEpochStore> {
        let next_epoch = next_epoch_committee.epoch();

        let last_checkpoint = self
            .checkpoint_store
            .get_epoch_last_checkpoint(cur_epoch_store.epoch())
            .expect("Error loading last checkpoint for current epoch")
            .expect("Could not load last checkpoint for current epoch");

        let last_checkpoint_seq = *last_checkpoint.sequence_number();

        assert_eq!(
            Some(last_checkpoint_seq),
            self.checkpoint_store
                .get_highest_executed_checkpoint_seq_number()
                .expect("Error loading highest executed checkpoint sequence number")
        );

        let epoch_start_configuration =
            EpochStartConfiguration::new(next_epoch_start_system_state, *last_checkpoint.digest());

        let new_epoch_store = self
            .state
            .reconfigure(
                cur_epoch_store,
                self.config.supported_protocol_versions.unwrap(),
                next_epoch_committee,
                epoch_start_configuration,
                global_state_hasher,
                &self.config.expensive_safety_check_config,
                last_checkpoint_seq,
            )
            .await
            .expect("Reconfigure authority state cannot fail");
        info!(next_epoch, "Node State has been reconfigured");
        assert_eq!(next_epoch, new_epoch_store.epoch());

        new_epoch_store
    }

    /// Check for previously detected forks and handle them appropriately.
    /// For validators with fork recovery config, clear the fork if it matches the recovery config.
    /// For all other cases, block node startup if a fork is detected.
    async fn check_and_recover_forks(
        checkpoint_store: &CheckpointStore,
        is_validator: bool,
        fork_recovery: Option<&ForkRecoveryConfig>,
    ) -> Result<()> {
        // Fork detection and recovery is only relevant for validators
        // Fullnodes should sync from validators and don't need fork checking
        if !is_validator {
            return Ok(());
        }

        // Try to recover from forks if recovery config is provided
        if let Some(recovery) = fork_recovery {
            Self::try_recover_checkpoint_fork(checkpoint_store, recovery)?;
            Self::try_recover_transaction_fork(checkpoint_store, recovery)?;
        }

        if let Some((checkpoint_seq, checkpoint_digest)) = checkpoint_store
            .get_checkpoint_fork_detected()
            .map_err(|e| {
                error!("Failed to check for checkpoint fork: {:?}", e);
                e
            })?
        {
            Self::handle_checkpoint_fork(checkpoint_seq, checkpoint_digest, fork_recovery).await?;
        }
        if let Some((tx_digest, expected_effects, actual_effects)) = checkpoint_store
            .get_transaction_fork_detected()
            .map_err(|e| {
                error!("Failed to check for transaction fork: {:?}", e);
                e
            })?
        {
            Self::handle_transaction_fork(
                tx_digest,
                expected_effects,
                actual_effects,
                fork_recovery,
            )
            .await?;
        }

        Ok(())
    }

    fn try_recover_checkpoint_fork(
        checkpoint_store: &CheckpointStore,
        recovery: &ForkRecoveryConfig,
    ) -> Result<()> {
        // If configured overrides include a checkpoint whose locally computed digest mismatches,
        // clear locally computed checkpoints from that sequence (inclusive).
        for (seq, expected_digest_str) in &recovery.checkpoint_overrides {
            let Ok(expected_digest) = CheckpointDigest::from_str(expected_digest_str) else {
                anyhow::bail!(
                    "Invalid checkpoint digest override for seq {}: {}",
                    seq,
                    expected_digest_str
                );
            };

            if let Some(local_summary) = checkpoint_store.get_locally_computed_checkpoint(*seq)? {
                let local_digest = types::envelope::Message::digest(&local_summary);
                if local_digest != expected_digest {
                    info!(
                        seq,
                        local = %Self::get_digest_prefix(local_digest),
                        expected = %Self::get_digest_prefix(expected_digest),
                        "Fork recovery: clearing locally_computed_checkpoints from {} due to digest mismatch",
                        seq
                    );
                    checkpoint_store
                        .clear_locally_computed_checkpoints_from(*seq)
                        .context(
                            "Failed to clear locally computed checkpoints from override seq",
                        )?;
                }
            }
        }

        if let Some((checkpoint_seq, checkpoint_digest)) =
            checkpoint_store.get_checkpoint_fork_detected()?
        {
            if recovery.checkpoint_overrides.contains_key(&checkpoint_seq) {
                info!(
                    "Fork recovery enabled: clearing checkpoint fork at seq {} with digest {:?}",
                    checkpoint_seq, checkpoint_digest
                );
                checkpoint_store
                    .clear_checkpoint_fork_detected()
                    .expect("Failed to clear checkpoint fork detected marker");
            }
        }
        Ok(())
    }

    /// Get a short prefix of a digest for metric labels
    fn get_digest_prefix(digest: impl std::fmt::Display) -> String {
        let digest_str = digest.to_string();
        if digest_str.len() >= 8 {
            digest_str[0..8].to_string()
        } else {
            digest_str
        }
    }

    fn try_recover_transaction_fork(
        checkpoint_store: &CheckpointStore,
        recovery: &ForkRecoveryConfig,
    ) -> Result<()> {
        if recovery.transaction_overrides.is_empty() {
            return Ok(());
        }

        if let Some((tx_digest, _, _)) = checkpoint_store.get_transaction_fork_detected()? {
            if recovery
                .transaction_overrides
                .contains_key(&tx_digest.to_string())
            {
                info!(
                    "Fork recovery enabled: clearing transaction fork for tx {:?}",
                    tx_digest
                );
                checkpoint_store
                    .clear_transaction_fork_detected()
                    .expect("Failed to clear transaction fork detected marker");
            }
        }
        Ok(())
    }

    fn get_current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    async fn handle_checkpoint_fork(
        checkpoint_seq: u64,
        checkpoint_digest: CheckpointDigest,
        fork_recovery: Option<&ForkRecoveryConfig>,
    ) -> Result<()> {
        let behavior = fork_recovery
            .map(|fr| fr.fork_crash_behavior)
            .unwrap_or_default();

        match behavior {
            ForkCrashBehavior::AwaitForkRecovery => {
                error!(
                    checkpoint_seq = checkpoint_seq,
                    checkpoint_digest = ?checkpoint_digest,
                    "Checkpoint fork detected! Node startup halted. Sleeping indefinitely."
                );
                futures::future::pending::<()>().await;
                unreachable!("pending() should never return");
            }
            ForkCrashBehavior::ReturnError => {
                error!(
                    checkpoint_seq = checkpoint_seq,
                    checkpoint_digest = ?checkpoint_digest,
                    "Checkpoint fork detected! Returning error."
                );
                Err(anyhow::anyhow!(
                    "Checkpoint fork detected! checkpoint_seq: {}, checkpoint_digest: {:?}",
                    checkpoint_seq,
                    checkpoint_digest
                ))
            }
        }
    }

    async fn handle_transaction_fork(
        tx_digest: TransactionDigest,
        expected_effects_digest: TransactionEffectsDigest,
        actual_effects_digest: TransactionEffectsDigest,
        fork_recovery: Option<&ForkRecoveryConfig>,
    ) -> Result<()> {
        let behavior = fork_recovery
            .map(|fr| fr.fork_crash_behavior)
            .unwrap_or_default();

        match behavior {
            ForkCrashBehavior::AwaitForkRecovery => {
                error!(
                    tx_digest = ?tx_digest,
                    expected_effects_digest = ?expected_effects_digest,
                    actual_effects_digest = ?actual_effects_digest,
                    "Transaction fork detected! Node startup halted. Sleeping indefinitely."
                );
                futures::future::pending::<()>().await;
                unreachable!("pending() should never return");
            }
            ForkCrashBehavior::ReturnError => {
                error!(
                    tx_digest = ?tx_digest,
                    expected_effects_digest = ?expected_effects_digest,
                    actual_effects_digest = ?actual_effects_digest,
                    "Transaction fork detected! Returning error."
                );
                Err(anyhow::anyhow!(
                    "Transaction fork detected! tx_digest: {:?}, expected_effects: {:?}, actual_effects: {:?}",
                    tx_digest,
                    expected_effects_digest,
                    actual_effects_digest
                ))
            }
        }
    }

    // Testing-only API to start epoch close process.
    // For production code, please use the non-testing version.
    pub async fn close_epoch_for_testing(&self) -> SomaResult {
        let epoch_store = self.state.epoch_store_for_testing();
        self.close_epoch(&epoch_store).await
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

    pub fn get_config(&self) -> &NodeConfig {
        &self.config
    }
}

async fn build_http_servers(
    state: Arc<AuthorityState>,
    store: RocksDbStore,
    transaction_orchestrator: &Option<Arc<TransactionOrchestrator<NetworkAuthorityClient>>>,
    config: &NodeConfig,
    server_version: ServerVersion,
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
        rpc_service.with_server_version(server_version);

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

#[cfg(not(test))]
fn max_tx_per_checkpoint(protocol_config: &ProtocolConfig) -> usize {
    protocol_config.max_transactions_per_checkpoint() as usize
}

#[cfg(test)]
fn max_tx_per_checkpoint(_: &ProtocolConfig) -> usize {
    2
}
