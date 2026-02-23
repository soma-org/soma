// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

//
// Modified for the Soma project.

use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};

use crate::{
    server::P2pService,
    state_sync::{PeerHeights, StateSyncEventLoop, StateSyncMessage},
    tonic_gen::p2p_server::{P2p, P2pServer},
};
use parking_lot::RwLock;
use tap::Pipe;
use tokio::{
    sync::{broadcast, mpsc, oneshot},
    task::JoinSet,
};
use types::{
    checkpoints::VerifiedCheckpoint,
    config::{
        node_config::ArchiveReaderConfig, p2p_config::P2pConfig, state_sync_config::StateSyncConfig,
    },
    crypto::NetworkKeyPair,
    storage::write_store::WriteStore,
    sync::{
        PeerEvent, SignedNodeInfo, active_peers::ActivePeers,
        channel_manager::ChannelManagerRequest,
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
        let Self { handle, config, shutdown_handle, state, their_info_receiver } = self;

        let discovery_config = config.discovery.clone().unwrap_or_default();
        let allowlisted_peers = Arc::new(
            discovery_config
                .allowlisted_peers
                .clone()
                .into_iter()
                .map(|ap| (ap.peer_id, ap.address))
                .chain(config.seed_peers.iter().filter_map(|peer| {
                    peer.peer_id.map(|peer_id| (peer_id, Some(peer.address.clone())))
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
    checkpoint_event_sender: broadcast::Sender<VerifiedCheckpoint>,
}

impl StateSyncHandle {
    /// Send a newly minted checkpoint from Consensus to StateSync so that it can be disseminated
    /// to other nodes on the network.
    ///
    /// # Invariant
    ///
    /// Consensus must only notify StateSync of new checkpoints that have been fully committed to
    /// persistent storage. This includes CheckpointContents and all Transactions and
    /// TransactionEffects included therein.
    pub async fn send_checkpoint(&self, checkpoint: VerifiedCheckpoint) {
        self.sender.send(StateSyncMessage::VerifiedCheckpoint(Box::new(checkpoint))).await.unwrap()
    }

    /// Subscribe to the stream of checkpoints that have been fully synchronized and downloaded.
    pub fn subscribe_to_synced_checkpoints(&self) -> broadcast::Receiver<VerifiedCheckpoint> {
        self.checkpoint_event_sender.subscribe()
    }
}

pub struct UnstartedStateSync<S> {
    pub(super) config: StateSyncConfig,
    pub(super) handle: StateSyncHandle,
    pub(super) mailbox: mpsc::Receiver<StateSyncMessage>,
    pub(super) store: S,
    pub(super) peer_heights: Arc<RwLock<PeerHeights>>,
    pub(super) checkpoint_event_sender: broadcast::Sender<VerifiedCheckpoint>,
    pub(super) archive_config: Option<ArchiveReaderConfig>,
}

impl<S> UnstartedStateSync<S>
where
    S: WriteStore + Clone + Send + Sync + 'static,
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
            checkpoint_event_sender,
            archive_config,
        } = self;

        (
            StateSyncEventLoop {
                config,
                mailbox,
                weak_sender: handle.sender.downgrade(),
                tasks: JoinSet::new(),
                sync_checkpoint_summaries_task: None,
                sync_checkpoint_contents_task: None,

                store,
                peer_heights,
                checkpoint_event_sender,
                active_peers,

                sync_checkpoint_from_archive_task: None,
                archive_config,
            },
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
    archive_config: Option<ArchiveReaderConfig>,
    // trusted_peer_change_rx: watch::Receiver<TrustedPeerChangeEvent>,
}

impl P2pBuilder<()> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self { store: None, config: None, archive_config: None }
    }
}

impl<S> P2pBuilder<S> {
    pub fn store<NewStore>(self, store: NewStore) -> P2pBuilder<NewStore> {
        P2pBuilder { store: Some(store), config: self.config, archive_config: None }
    }

    pub fn config(mut self, config: P2pConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn archive_config(mut self, archive_config: Option<ArchiveReaderConfig>) -> Self {
        self.archive_config = archive_config;
        self
    }
}

impl<S> P2pBuilder<S>
where
    S: WriteStore + Clone + Send + Sync + 'static,
{
    pub fn build(self) -> (UnstartedDiscovery, UnstartedStateSync<S>, P2pServer<P2pService<S>>) {
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
        let discovery_handle = DiscoveryHandle { _shutdown_handle: Arc::new(discovery_sender) };

        let (state_sync_sender, mailbox) = mpsc::channel(state_sync_config.mailbox_capacity());
        let (their_info_sender, their_info_receiver) =
            mpsc::channel(state_sync_config.mailbox_capacity());
        let (checkpoint_event_sender, _receiver) =
            broadcast::channel(state_sync_config.synced_checkpoint_broadcast_channel_capacity());
        let weak_sender = state_sync_sender.downgrade();

        let state_sync_handle = StateSyncHandle {
            sender: state_sync_sender.clone(),
            checkpoint_event_sender: checkpoint_event_sender.clone(),
        };
        let peer_heights = PeerHeights {
            peers: HashMap::new(),
            unprocessed_checkpoints: HashMap::new(),
            sequence_number_to_digest: HashMap::new(),
            wait_interval_when_no_peer_to_sync_content: state_sync_config
                .wait_interval_when_no_peer_to_sync_content(),
        }
        .pipe(RwLock::new)
        .pipe(Arc::new);

        let discovery_state = DiscoveryState { our_info: None, known_peers: HashMap::default() }
            .pipe(RwLock::new)
            .pipe(Arc::new);

        let server = P2pService {
            discovery_state: discovery_state.clone(),
            store: store.clone(),
            peer_heights: peer_heights.clone(),
            state_sync_sender: weak_sender,
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
                checkpoint_event_sender,
                archive_config: self.archive_config,
            },
            server,
        )
    }
}
