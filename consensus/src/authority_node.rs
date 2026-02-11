use std::{sync::Arc, time::Instant};

use crate::{
    CommitConsumerArgs,
    authority_service::AuthorityService,
    block_manager::BlockManager,
    block_verifier::SignedBlockVerifier,
    commit_observer::CommitObserver,
    commit_syncer::{CommitSyncer, CommitSyncerHandle},
    commit_vote_monitor::CommitVoteMonitor,
    core::{Core, CoreSignals},
    core_thread::{ChannelCoreThreadDispatcher, CoreThreadHandle},
    dag_state::DagState,
    leader_schedule::LeaderSchedule,
    leader_timeout::{LeaderTimeoutTask, LeaderTimeoutTaskHandle},
    network::{NetworkManager, tonic_network::TonicManager},
    proposed_block_handler::ProposedBlockHandler,
    round_prober::{RoundProber, RoundProberHandle},
    round_tracker::PeerRoundTracker,
    subscriber::Subscriber,
    synchronizer::{Synchronizer, SynchronizerHandle},
    transaction::{TransactionClient, TransactionConsumer, TransactionVerifier},
    transaction_certifier::TransactionCertifier,
};
use itertools::Itertools;
use parking_lot::RwLock;
use protocol_config::ProtocolConfig;
use tokio::task::JoinHandle;
use tracing::{info, warn};
use types::committee::{AuthorityIndex, Committee};
use types::consensus::context::{Clock, Context};
use types::crypto::{NetworkKeyPair, ProtocolKeyPair};
use types::parameters::Parameters;
use types::storage::consensus::rocksdb_store::RocksDBStore;

/// ConsensusAuthority is used by Sui to manage the lifetime of AuthorityNode.
/// It hides the details of the implementation from the caller, MysticetiManager.
#[allow(private_interfaces)]
pub enum ConsensusAuthority {
    WithTonic(AuthorityNode<TonicManager>),
}

impl ConsensusAuthority {
    #[allow(clippy::too_many_arguments)]
    pub async fn start(
        network_type: NetworkType,
        epoch_start_timestamp_ms: u64,
        own_index: AuthorityIndex,
        committee: Committee,
        parameters: Parameters,
        protocol_config: ProtocolConfig,
        protocol_keypair: ProtocolKeyPair,
        network_keypair: NetworkKeyPair,
        clock: Arc<Clock>,
        transaction_verifier: Arc<dyn TransactionVerifier>,
        commit_consumer: CommitConsumerArgs,
        // A counter that keeps track of how many times the authority node has been booted while the binary
        // or the component that is calling the `ConsensusAuthority` has been running. It's mostly useful to
        // make decisions on whether amnesia recovery should run or not. When `boot_counter` is 0, then `ConsensusAuthority`
        // will initiate the process of amnesia recovery if that's enabled in the parameters.
        boot_counter: u64,
    ) -> Self {
        match network_type {
            NetworkType::Tonic => {
                let authority = AuthorityNode::start(
                    epoch_start_timestamp_ms,
                    own_index,
                    committee,
                    parameters,
                    protocol_config,
                    protocol_keypair,
                    network_keypair,
                    clock,
                    transaction_verifier,
                    commit_consumer,
                    boot_counter,
                )
                .await;
                Self::WithTonic(authority)
            }
        }
    }

    pub async fn stop(self) {
        match self {
            Self::WithTonic(authority) => authority.stop().await,
        }
    }

    pub fn transaction_client(&self) -> Arc<TransactionClient> {
        match self {
            Self::WithTonic(authority) => authority.transaction_client(),
        }
    }

    #[cfg(test)]
    fn context(&self) -> &Arc<Context> {
        match self {
            Self::WithTonic(authority) => &authority.context,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NetworkType {
    Tonic,
}

pub(crate) struct AuthorityNode<N>
where
    N: NetworkManager<AuthorityService<ChannelCoreThreadDispatcher>>,
{
    context: Arc<Context>,
    start_time: Instant,
    transaction_client: Arc<TransactionClient>,
    synchronizer: Arc<SynchronizerHandle>,

    commit_syncer_handle: CommitSyncerHandle,
    round_prober_handle: RoundProberHandle,
    proposed_block_handler: JoinHandle<()>,
    leader_timeout_handle: LeaderTimeoutTaskHandle,
    core_thread_handle: CoreThreadHandle,
    subscriber: Subscriber<N::Client, AuthorityService<ChannelCoreThreadDispatcher>>,
    network_manager: N,
}

impl<N> AuthorityNode<N>
where
    N: NetworkManager<AuthorityService<ChannelCoreThreadDispatcher>>,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn start(
        epoch_start_timestamp_ms: u64,
        own_index: AuthorityIndex,
        committee: Committee,
        parameters: Parameters,
        protocol_config: ProtocolConfig,
        // To avoid accidentally leaking the private key, the protocol key pair should only be
        // kept in Core.
        protocol_keypair: ProtocolKeyPair,
        network_keypair: NetworkKeyPair,
        clock: Arc<Clock>,
        transaction_verifier: Arc<dyn TransactionVerifier>,
        commit_consumer: CommitConsumerArgs,
        boot_counter: u64,
    ) -> Self {
        assert!(committee.is_valid_index(own_index), "Invalid own index {}", own_index);
        let own_hostname = committee
            .authority_by_authority_index(own_index)
            .expect("Own index should be present in committee")
            .hostname
            .clone();
        info!(
            "Starting consensus authority {} {}, {:?}, epoch start timestamp {}, boot counter {}, replaying after commit index {}, consumer last processed commit index {}",
            own_index,
            own_hostname,
            protocol_config.version,
            epoch_start_timestamp_ms,
            boot_counter,
            commit_consumer.replay_after_commit_index,
            commit_consumer.consumer_last_processed_commit_index
        );
        info!(
            "Consensus authorities: {}",
            committee.authorities().map(|(i, a)| format!("{}: {}", i, a.hostname)).join(", ")
        );
        info!("Consensus parameters: {:?}", parameters);
        info!("Consensus committee: {:?}", committee);
        let context = Arc::new(Context::new(
            epoch_start_timestamp_ms,
            own_index,
            committee,
            parameters,
            protocol_config,
            clock,
        ));
        let start_time = Instant::now();

        let (tx_client, tx_receiver) = TransactionClient::new(context.clone());
        let tx_consumer = TransactionConsumer::new(tx_receiver, context.clone());

        let (core_signals, signals_receivers) = CoreSignals::new(context.clone());

        let mut network_manager = N::new(context.clone(), network_keypair);
        let network_client = network_manager.client();

        let store_path = context.parameters.db_path.as_path().to_str().unwrap();
        let store = Arc::new(RocksDBStore::new(store_path));
        let dag_state = Arc::new(RwLock::new(DagState::new(context.clone(), store.clone())));

        let block_verifier =
            Arc::new(SignedBlockVerifier::new(context.clone(), transaction_verifier));

        let transaction_certifier = TransactionCertifier::new(
            context.clone(),
            block_verifier.clone(),
            dag_state.clone(),
            commit_consumer.block_sender.clone(),
        );

        let mut proposed_block_handler = ProposedBlockHandler::new(
            context.clone(),
            signals_receivers.block_broadcast_receiver(),
            transaction_certifier.clone(),
        );

        let proposed_block_handler =
            tokio::spawn(async move { proposed_block_handler.run().await });

        let sync_last_known_own_block = boot_counter == 0
            && dag_state.read().highest_accepted_round() == 0
            && !context.parameters.sync_last_known_own_block_timeout.is_zero();
        info!("Sync last known own block: {sync_last_known_own_block}");

        let block_manager = BlockManager::new(context.clone(), dag_state.clone());

        let leader_schedule =
            Arc::new(LeaderSchedule::from_store(context.clone(), dag_state.clone()));

        let commit_consumer_monitor = commit_consumer.monitor();
        let commit_observer = CommitObserver::new(
            context.clone(),
            commit_consumer,
            dag_state.clone(),
            transaction_certifier.clone(),
            leader_schedule.clone(),
        )
        .await;

        let round_tracker = Arc::new(RwLock::new(PeerRoundTracker::new(context.clone())));

        let core = Core::new(
            context.clone(),
            leader_schedule,
            tx_consumer,
            transaction_certifier.clone(),
            block_manager,
            commit_observer,
            core_signals,
            protocol_keypair,
            dag_state.clone(),
            sync_last_known_own_block,
            round_tracker.clone(),
        );

        let (core_dispatcher, core_thread_handle) =
            ChannelCoreThreadDispatcher::start(context.clone(), &dag_state, core);
        let core_dispatcher = Arc::new(core_dispatcher);
        let leader_timeout_handle =
            LeaderTimeoutTask::start(core_dispatcher.clone(), &signals_receivers, context.clone());

        let commit_vote_monitor = Arc::new(CommitVoteMonitor::new(context.clone()));

        let synchronizer = Synchronizer::start(
            network_client.clone(),
            context.clone(),
            core_dispatcher.clone(),
            commit_vote_monitor.clone(),
            block_verifier.clone(),
            transaction_certifier.clone(),
            dag_state.clone(),
            sync_last_known_own_block,
        );

        let commit_syncer_handle = CommitSyncer::new(
            context.clone(),
            core_dispatcher.clone(),
            commit_vote_monitor.clone(),
            commit_consumer_monitor.clone(),
            block_verifier.clone(),
            transaction_certifier.clone(),
            network_client.clone(),
            dag_state.clone(),
        )
        .start();

        let round_prober_handle = RoundProber::new(
            context.clone(),
            core_dispatcher.clone(),
            round_tracker.clone(),
            dag_state.clone(),
            network_client.clone(),
        )
        .start();

        let network_service = Arc::new(AuthorityService::new(
            context.clone(),
            block_verifier,
            commit_vote_monitor,
            round_tracker.clone(),
            synchronizer.clone(),
            core_dispatcher,
            signals_receivers.block_broadcast_receiver(),
            transaction_certifier,
            dag_state.clone(),
            store,
        ));

        let subscriber = {
            let s = Subscriber::new(
                context.clone(),
                network_client,
                network_service.clone(),
                dag_state,
            );
            for (peer, _) in context.committee.authorities() {
                if peer != context.own_index {
                    s.subscribe(peer);
                }
            }
            s
        };

        network_manager.install_service(network_service).await;

        info!("Consensus authority started, took {:?}", start_time.elapsed());

        Self {
            context,
            start_time,
            transaction_client: Arc::new(tx_client),
            synchronizer,
            commit_syncer_handle,
            round_prober_handle,
            proposed_block_handler,
            leader_timeout_handle,
            core_thread_handle,
            subscriber,
            network_manager,
        }
    }

    pub(crate) async fn stop(mut self) {
        info!("Stopping authority. Total run time: {:?}", self.start_time.elapsed());

        // First shutdown components calling into Core.
        if let Err(e) = self.synchronizer.stop().await {
            if e.is_panic() {
                std::panic::resume_unwind(e.into_panic());
            }
            warn!("Failed to stop synchronizer when shutting down consensus: {:?}", e);
        };
        self.commit_syncer_handle.stop().await;
        self.round_prober_handle.stop().await;
        self.proposed_block_handler.abort();
        self.leader_timeout_handle.stop().await;
        // Shutdown Core to stop block productions and broadcast.
        self.core_thread_handle.stop().await;
        // Stop block subscriptions before stopping network server.
        self.subscriber.stop();
        self.network_manager.stop().await;
    }

    pub(crate) fn transaction_client(&self) -> Arc<TransactionClient> {
        self.transaction_client.clone()
    }
}
