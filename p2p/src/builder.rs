use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, Weak},
};

use crate::{
    server::P2pService,
    state_sync::{tx_verifier::TxVerifier, PeerHeights, StateSyncEventLoop, StateSyncMessage},
    tonic_gen::p2p_server::{P2p, P2pServer},
};
use fastcrypto::traits::ToFromBytes;
use parking_lot::RwLock;
use tap::Pipe;
use tokio::{
    sync::{broadcast, mpsc, oneshot},
    task::JoinSet,
};
use tracing::{debug, info};
use types::{
    accumulator::{self, AccumulatorStore},
    config::{p2p_config::P2pConfig, state_sync_config::StateSyncConfig},
    consensus::{
        block_verifier::{BlockVerifier, SignedBlockVerifier},
        commit::CommittedSubDag,
        context::{Clock, Context},
        transaction,
    },
    crypto::NetworkKeyPair,
    discovery::SignedNodeInfo,
    p2p::{
        active_peers::{self, ActivePeers},
        channel_manager::{self, ChannelManager, ChannelManagerRequest},
        PeerEvent,
    },
    parameters::Parameters,
    signature_verifier::SignatureVerifier,
    state_sync::{self},
    storage::{
        consensus::{mem_store::MemStore, ConsensusStore},
        write_store::WriteStore,
    },
};

use crate::discovery::{DiscoveryEventLoop, DiscoveryState};

/// A Handle to the Discovery subsystem. The Discovery system will be shutdown once its Handle has
/// been dropped.
pub struct DiscoveryHandle {
    _shutdown_handle: Arc<oneshot::Sender<()>>,
}

/// Handle to an unstarted discovery system
pub struct UnstartedDiscovery {
    pub(super) handle: DiscoveryHandle,
    pub(super) config: P2pConfig,
    pub(super) shutdown_handle: oneshot::Receiver<()>,
    pub(super) state: Arc<RwLock<DiscoveryState>>,
    pub(super) their_info_receiver: mpsc::Receiver<SignedNodeInfo>,
    // pub(super) trusted_peer_change_rx: watch::Receiver<TrustedPeerChangeEvent>,
}

impl UnstartedDiscovery {
    pub(super) fn build(
        self,
        active_peers: ActivePeers,
        channel_manager_tx: mpsc::Sender<ChannelManagerRequest>,
        keypair: NetworkKeyPair,
    ) -> (DiscoveryEventLoop, DiscoveryHandle) {
        let Self {
            handle,
            config,
            shutdown_handle,
            state,
            their_info_receiver,
        } = self;

        let discovery_config = config.discovery.clone().unwrap_or_default();
        let allowlisted_peers = Arc::new(
            discovery_config
                .allowlisted_peers
                .clone()
                .into_iter()
                .map(|ap| (ap.peer_id, ap.address))
                .chain(config.seed_peers.iter().filter_map(|peer| {
                    peer.peer_id
                        .map(|peer_id| (peer_id, Some(peer.address.clone())))
                }))
                .collect::<HashMap<_, _>>(),
        );

        (
            DiscoveryEventLoop::new(
                config,
                discovery_config,
                allowlisted_peers,
                active_peers,
                keypair,
                channel_manager_tx,
                // shutdown_handle,
                state,
                their_info_receiver,
                // trusted_peer_change_rx,
            ),
            handle,
        )
    }

    pub fn start(
        self,
        active_peers: ActivePeers,
        channel_manager_tx: mpsc::Sender<ChannelManagerRequest>,
        keypair: NetworkKeyPair,
    ) -> DiscoveryHandle {
        let (event_loop, handle) = self.build(active_peers, channel_manager_tx, keypair);
        tokio::spawn(event_loop.start());

        handle
    }
}

/// This handle can be cloned and shared. Once all copies of a StateSync system's Handle have been
/// dropped, the StateSync system will be gracefully shutdown.
#[derive(Clone, Debug)]
pub struct StateSyncHandle {
    sender: mpsc::Sender<StateSyncMessage>,
    commit_event_sender: broadcast::Sender<CommittedSubDag>,
}

impl StateSyncHandle {
    /// Send a newly minted commit from Consensus to StateSync so that it can be disseminated
    /// to other nodes on the network.
    ///
    /// # Invariant
    ///
    /// Consensus must only notify StateSync of new commits that have been fully committed to
    /// persistent storage. This includes CommitContents and all Transactions and
    /// TransactionEffects included therein.
    pub async fn send_commit(&self, commit: CommittedSubDag) {
        debug!("Sending commit from consensus to state sync: {}", commit);
        self.sender
            .send(StateSyncMessage::VerifiedCommit(Box::new(commit)))
            .await
            .expect("Could not send state sync handle commit")
    }

    /// Subscribe to the stream of commits that have been fully synchronized and downloaded.
    pub fn subscribe_to_synced_commits(&self) -> broadcast::Receiver<CommittedSubDag> {
        self.commit_event_sender.subscribe()
    }

    pub fn new_for_testing() -> Self {
        let (sender, _) = mpsc::channel(50);
        let (commit_event_sender, _) = broadcast::channel(50);

        Self {
            sender,
            commit_event_sender,
        }
    }
}

pub struct UnstartedStateSync<S> {
    pub(super) config: StateSyncConfig,
    pub(super) handle: StateSyncHandle,
    pub(super) mailbox: mpsc::Receiver<StateSyncMessage>,
    pub(super) store: S,
    pub(super) peer_heights: Arc<RwLock<PeerHeights>>,
    pub(super) commit_event_sender: broadcast::Sender<CommittedSubDag>,
    pub(super) block_verifier: Arc<SignedBlockVerifier>,
}

impl<S> UnstartedStateSync<S>
where
    S: ConsensusStore + WriteStore + Clone + Send + Sync + 'static,
{
    pub(super) fn build(
        self,
        active_peers: ActivePeers,
        peer_event_receiver: broadcast::Receiver<PeerEvent>,
    ) -> (StateSyncEventLoop<S>, StateSyncHandle) {
        let Self {
            config,
            handle,
            mailbox,
            store,
            peer_heights,
            commit_event_sender,
            block_verifier,
        } = self;

        (
            StateSyncEventLoop::new(
                config,
                mailbox,
                handle.sender.downgrade(),
                store,
                peer_heights,
                commit_event_sender,
                active_peers,
                peer_event_receiver,
                block_verifier,
            ),
            handle,
        )
    }

    pub fn start(
        self,
        active_peers: ActivePeers,
        peer_event_receiver: broadcast::Receiver<PeerEvent>,
    ) -> StateSyncHandle {
        let (event_loop, handle) = self.build(active_peers, peer_event_receiver);
        tokio::spawn(event_loop.start());

        handle
    }
}

/// Discovery & State Sync Service Builder.
pub struct P2pBuilder<S> {
    config: Option<P2pConfig>,
    store: Option<S>,
    // trusted_peer_change_rx: watch::Receiver<TrustedPeerChangeEvent>,
}

impl P2pBuilder<()> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            store: None,
            config: None,
        }
    }
}

impl<S> P2pBuilder<S> {
    pub fn store<NewStore>(self, store: NewStore) -> P2pBuilder<NewStore> {
        P2pBuilder {
            store: Some(store),
            config: self.config,
        }
    }

    pub fn config(mut self, config: P2pConfig) -> Self {
        self.config = Some(config);
        self
    }
}

impl<S> P2pBuilder<S>
where
    S: ConsensusStore + WriteStore + AccumulatorStore + Clone + Send + Sync + 'static,
{
    pub fn build(
        self,
    ) -> (
        UnstartedDiscovery,
        UnstartedStateSync<S>,
        P2pServer<P2pService<S>>,
    ) {
        let (discovery_builder, state_sync_builder, server) = self.build_internal();
        let server = P2pServer::new(server);

        (discovery_builder, state_sync_builder, server)
    }

    pub(super) fn build_internal(
        self,
    ) -> (UnstartedDiscovery, UnstartedStateSync<S>, P2pService<S>) {
        let store = self.store.unwrap();
        let store_ref = Arc::new(store.clone());
        let config = self.config.unwrap();
        let state_sync_config = config.state_sync.clone().unwrap();

        let (discovery_sender, discovery_receiver) = oneshot::channel();
        let discovery_handle = DiscoveryHandle {
            _shutdown_handle: Arc::new(discovery_sender),
        };

        let (state_sync_sender, mailbox) = mpsc::channel(state_sync_config.mailbox_capacity());
        let (their_info_sender, their_info_receiver) =
            mpsc::channel(state_sync_config.mailbox_capacity());
        let (commit_event_sender, _receiver) = broadcast::channel(
            state_sync_config.synced_commit_broadcast_channel_capacity() as usize,
        );
        // let weak_state_sync_sender = state_sync_sender.downgrade();

        let state_sync_handle = StateSyncHandle {
            sender: state_sync_sender.clone(),
            commit_event_sender: commit_event_sender.clone(),
        };
        let peer_heights =
            PeerHeights::new(state_sync_config.wait_interval_when_no_peer_to_sync_content())
                .pipe(RwLock::new)
                .pipe(Arc::new);

        let discovery_state = DiscoveryState {
            our_info: None,
            known_peers: HashMap::default(),
        }
        .pipe(RwLock::new)
        .pipe(Arc::new);

        // Dummy context with genesis committee
        let parameters = Parameters {
            ..Default::default()
        };
        let context = Arc::new(Context::new(
            None,
            (*store_ref.get_committee(0).unwrap().unwrap()).clone(),
            parameters,
            Arc::new(Clock::new()),
        ));

        let signature_verifier =
            SignatureVerifier::new(Arc::new(context.committee.clone()), Some(store_ref.clone()));
        let transaction_verifier = TxVerifier::new(Arc::new(signature_verifier));
        let block_verifier = Arc::new(SignedBlockVerifier::new(
            context.clone(),
            Arc::new(transaction_verifier),
            store_ref.clone(),
            Some(store_ref.clone()),
        ));

        let server = P2pService {
            discovery_state: discovery_state.clone(),
            store: store.clone(),
            peer_heights: peer_heights.clone(),
            state_sync_sender,
            discovery_sender: their_info_sender,
        };

        (
            UnstartedDiscovery {
                handle: discovery_handle,
                config,
                shutdown_handle: discovery_receiver,
                state: discovery_state,
                their_info_receiver,
            },
            UnstartedStateSync {
                config: state_sync_config,
                handle: state_sync_handle,
                mailbox,
                store,
                peer_heights,
                commit_event_sender,
                block_verifier,
            },
            server,
        )
    }
}
