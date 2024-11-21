use std::{collections::HashMap, sync::Arc};

use crate::tonic_gen::p2p_server::{P2p, P2pServer};
use fastcrypto::traits::ToFromBytes;
use parking_lot::RwLock;
use tap::Pipe;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinSet,
};
use types::{
    config::p2p_config::P2pConfig,
    crypto::NetworkKeyPair,
    p2p::{
        active_peers::{self, ActivePeers},
        channel_manager::{self, ChannelManager, ChannelManagerRequest},
    },
};

use super::{server::P2pService, DiscoveryEventLoop, DiscoveryState};

/// Discovery Service Builder.
pub struct DiscoveryBuilder {
    config: Option<P2pConfig>,
    // trusted_peer_change_rx: watch::Receiver<TrustedPeerChangeEvent>,
}

impl DiscoveryBuilder {
    pub fn new() -> Self {
        Self { config: None }
    }

    pub fn config(mut self, config: P2pConfig) -> Self {
        self.config = Some(config);
        self
    }
    pub fn build(self) -> (UnstartedDiscovery, P2pServer<P2pService>) {
        let discovery_config = self
            .config
            .clone()
            .and_then(|config| config.discovery)
            .unwrap_or_default();
        let (builder, server) = self.build_internal();
        let mut discovery_server = P2pServer::new(server);

        (builder, discovery_server)
    }

    pub(super) fn build_internal(self) -> (UnstartedDiscovery, P2pService) {
        let DiscoveryBuilder { config } = self;
        let config = config.unwrap();

        let (sender, receiver) = oneshot::channel();

        let handle = DiscoveryHandle {
            _shutdown_handle: Arc::new(sender),
        };

        let state = DiscoveryState {
            our_info: None,
            connected_peers: HashMap::default(),
            known_peers: HashMap::default(),
        }
        .pipe(RwLock::new)
        .pipe(Arc::new);

        let server = P2pService {
            discovery_state: state.clone(),
        };

        (
            UnstartedDiscovery {
                handle,
                config,
                shutdown_handle: receiver,
                state,
            },
            server,
        )
    }
}

/// Handle to an unstarted discovery system
pub struct UnstartedDiscovery {
    pub(super) handle: DiscoveryHandle,
    pub(super) config: P2pConfig,
    pub(super) shutdown_handle: oneshot::Receiver<()>,
    pub(super) state: Arc<RwLock<DiscoveryState>>,
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
            DiscoveryEventLoop {
                config,
                discovery_config: Arc::new(discovery_config),
                allowlisted_peers,
                active_peers,
                keypair,
                tasks: JoinSet::new(),
                pending_dials: Default::default(),
                dial_seed_peers_task: None,
                channel_manager_tx,
                // shutdown_handle,
                state,
                // trusted_peer_change_rx,
            },
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

/// A Handle to the Discovery subsystem. The Discovery system will be shutdown once its Handle has
/// been dropped.
pub struct DiscoveryHandle {
    _shutdown_handle: Arc<oneshot::Sender<()>>,
}
