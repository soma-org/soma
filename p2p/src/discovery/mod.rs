use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use fastcrypto::{ed25519::Ed25519PublicKey, traits::VerifyingKey};
use futures::{channel, StreamExt};
use parking_lot::RwLock;
use tokio::{
    sync::{mpsc, oneshot, watch},
    task::{AbortHandle, JoinSet},
};
use tonic::{Request, Response};
use tracing::{debug, info};
use types::{
    config::p2p_config::{DiscoveryConfig, P2pConfig, SeedPeer},
    crypto::NetworkKeyPair,
    discovery::{
        GetKnownPeersRequest, GetKnownPeersResponse, NodeInfo, SignedNodeInfo,
        VerifiedSignedNodeInfo,
    },
    multiaddr::Multiaddr,
    p2p::{active_peers::ActivePeers, channel_manager::ChannelManagerRequest, PeerEvent},
    peer_id::PeerId,
};

use crate::tonic_gen::p2p_client::P2pClient;

pub mod builder;
pub mod server;

/// The internal discovery state shared between the main event loop and the request handler
pub struct DiscoveryState {
    our_info: Option<SignedNodeInfo>,
    connected_peers: HashMap<PeerId, ()>,
    known_peers: HashMap<PeerId, VerifiedSignedNodeInfo>,
}

// #[derive(Clone, Debug, Default)]
// pub struct TrustedPeerChangeEvent {
//     pub new_peers: Vec<NodeInfo>,
// }

struct DiscoveryEventLoop {
    config: P2pConfig,
    discovery_config: Arc<DiscoveryConfig>,
    allowlisted_peers: Arc<HashMap<PeerId, Option<Multiaddr>>>,
    channel_manager_tx: mpsc::Sender<ChannelManagerRequest>,
    active_peers: ActivePeers, // Reference to ChannelManager's ActivePeers
    keypair: NetworkKeyPair,
    tasks: JoinSet<()>,
    pending_dials: HashMap<PeerId, AbortHandle>,
    dial_seed_peers_task: Option<AbortHandle>,
    state: Arc<RwLock<DiscoveryState>>,
    // trusted_peer_change_rx: watch::Receiver<TrustedPeerChangeEvent>,
}

const INTERVAL_PERIOD_MS: u64 = 5_000;
const PEER_QUERY_TIMEOUT_SECS: u64 = 10;

impl DiscoveryEventLoop {
    pub async fn start(mut self) {
        info!("Discovery started");

        self.construct_our_info();
        self.configure_preferred_peers().await;

        let mut interval = tokio::time::interval(Duration::from_millis(INTERVAL_PERIOD_MS));

        loop {
            tokio::select! {
                now = interval.tick() => {
                    let now_unix = now_unix();
                    self.handle_tick(now.into_std(), now_unix).await;
                }
                // Ok(()) = self.trusted_peer_change_rx.changed() => {
                //     let event = self.trusted_peer_change_rx.borrow_and_update().clone();
                //     self.handle_trusted_peer_change_event(event).await;
                // }
                Some(task_result) = self.tasks.join_next() => {
                    match task_result {
                        Ok(()) => {},
                        Err(e) => {
                            if e.is_cancelled() {
                                // avoid crashing on ungraceful shutdown
                            } else if e.is_panic() {
                                std::panic::resume_unwind(e.into_panic());
                            } else {
                                panic!("task failed: {e}");
                            }
                        },
                    };
                },
            }
        }
    }

    fn construct_our_info(&mut self) {
        if self.state.read().our_info.is_some() {
            return;
        }

        // Get external address from config
        let address = self
            .config
            .external_address
            .clone()
            .expect("External address must be set");

        let our_info = NodeInfo {
            peer_id: PeerId::from(self.keypair.public()), // Assuming you have From<&NetworkKeyPair> for PeerId
            address: address,
            timestamp_ms: now_unix(),
        }
        .sign(&self.keypair);

        self.state.write().our_info = Some(our_info);
    }

    async fn configure_preferred_peers(&mut self) {
        for (peer_id, address) in self
            .discovery_config
            .allowlisted_peers
            .iter()
            .map(|sp| (sp.peer_id, sp.address.clone()))
            .chain(self.config.seed_peers.iter().filter_map(|ap| {
                ap.peer_id
                    .map(|peer_id| (peer_id, Some(ap.address.clone())))
            }))
        {
            // Try to connect to preferred peer
            if let Some(address) = address {
                let (tx, rx) = oneshot::channel();
                if let Err(e) = self
                    .channel_manager_tx
                    .send(ChannelManagerRequest::Connect {
                        address: address.clone(),
                        peer_id,
                        response: tx,
                    })
                    .await
                {
                    debug!("Failed to send connect request for preferred peer: {}", e);
                    continue;
                }

                if let Ok(Ok(_)) = rx.await {
                    debug!("Connected to preferred peer {}", peer_id);
                }
            }
        }
    }

    fn update_our_info_timestamp(&mut self, now_unix: u64) {
        let state = &mut self.state.write();
        if let Some(our_info) = &state.our_info {
            let mut data = our_info.data().clone();
            data.timestamp_ms = now_unix;
            state.our_info = Some(data.sign(&self.keypair));
        }
    }

    async fn handle_tick(&mut self, _now: Instant, now_unix: u64) {
        self.update_our_info_timestamp(now_unix);

        self.tasks
            .spawn(query_connected_peers_for_their_known_peers(
                self.active_peers.clone(),
                self.discovery_config.clone(),
                self.state.clone(),
                self.allowlisted_peers.clone(),
            ));

        // Cull old peers older than a day
        self.state
            .write()
            .known_peers
            .retain(|_k, v| now_unix.saturating_sub(v.timestamp_ms) < ONE_DAY_MILLISECONDS);

        // Clean out the pending_dials
        self.pending_dials.retain(|_k, v| !v.is_finished());
        if let Some(abort_handle) = &self.dial_seed_peers_task {
            if abort_handle.is_finished() {
                self.dial_seed_peers_task = None;
            }
        }

        // Find eligible peers to connect to
        let state = self.state.read();
        let eligible = state
            .known_peers
            .iter()
            .filter(|(peer_id, info)| {
                !state.connected_peers.contains_key(peer_id) // We're not already connected
                    && !self.pending_dials.contains_key(peer_id) // There is no pending dial to this node
            })
            .map(|(k, v)| (*k, v.clone()))
            .collect::<Vec<_>>();

        drop(state);

        let number_of_connections = self.state.read().connected_peers.len();
        let number_to_dial = std::cmp::min(
            eligible.len(),
            self.discovery_config
                .target_concurrent_connections()
                .saturating_sub(number_of_connections),
        );

        // Use as_slice() to get a slice reference
        for (peer_id, info) in rand::seq::SliceRandom::choose_multiple(
            eligible.as_slice(),
            &mut rand::thread_rng(),
            number_to_dial,
        ) {
            let channel_manager_tx = self.channel_manager_tx.clone();
            let abort_handle = self.tasks.spawn(try_to_connect_to_peer(
                channel_manager_tx,
                info.data().clone(),
            ));
            self.pending_dials.insert(*peer_id, abort_handle);
        }

        // If we aren't connected to anything and we aren't presently trying to connect to anyone
        // we need to try the seed peers
        if self.dial_seed_peers_task.is_none()
            && self.state.read().connected_peers.is_empty()
            && self.pending_dials.is_empty()
            && !self.config.seed_peers.is_empty()
        {
            let channel_manager_tx = self.channel_manager_tx.clone();
            let abort_handle = self.tasks.spawn(try_to_connect_to_seed_peers(
                channel_manager_tx,
                self.config.seed_peers.clone(),
                self.discovery_config.target_concurrent_connections(),
            ));
            self.dial_seed_peers_task = Some(abort_handle);
        }
    }
}

fn now_unix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

async fn try_to_connect_to_peer(
    channel_manager_tx: mpsc::Sender<ChannelManagerRequest>,
    info: NodeInfo,
) {
    debug!("Connecting to peer {info:?}");
    let (tx, rx) = oneshot::channel();
    if let Err(e) = channel_manager_tx
        .send(ChannelManagerRequest::Connect {
            address: info.address.clone(),
            peer_id: info.peer_id,
            response: tx,
        })
        .await
    {
        debug!(
            "error sending connect request for {} at address '{}': {e}",
            info.peer_id.short_display(4),
            info.address
        );
        return;
    }

    match rx.await {
        Ok(Ok(_)) => return,
        Ok(Err(e)) => {
            debug!(
                "error connecting to {} at address '{}': {e}",
                info.peer_id.short_display(4),
                info.address
            );
        }
        Err(e) => {
            debug!("connect request cancelled: {e}");
        }
    }
}

async fn try_to_connect_to_seed_peers(
    channel_manager_tx: mpsc::Sender<ChannelManagerRequest>,
    seed_peers: Vec<SeedPeer>,
    max_concurrent: usize,
) {
    debug!(?seed_peers, "Connecting to seed peers");

    futures::stream::iter(
        seed_peers
            .into_iter()
            .map(|seed| (seed.clone(), seed.address)),
    )
    .for_each_concurrent(max_concurrent, |(seed, address)| {
        let value = channel_manager_tx.clone();
        async move {
            let (tx, rx) = oneshot::channel();
            if let Err(e) = value
                .send(ChannelManagerRequest::Connect {
                    address: address.clone(),
                    peer_id: seed.peer_id.expect("Seed peers must have peer_id"),
                    response: tx,
                })
                .await
            {
                debug!("error sending connect request for seed peer: {e}");
                return;
            }

            if let Err(e) = rx.await {
                debug!("connect request for seed peer cancelled: {e}");
            }
        }
    })
    .await;
}

async fn query_connected_peers_for_their_known_peers(
    active_peers: ActivePeers,
    config: Arc<DiscoveryConfig>,
    state: Arc<RwLock<DiscoveryState>>,
    allowlisted_peers: Arc<HashMap<PeerId, Option<Multiaddr>>>,
) {
    use rand::seq::IteratorRandom;

    // Get connected peers from active_peers
    let peers_to_query = active_peers
        .peers()
        .into_iter()
        .choose_multiple(&mut rand::thread_rng(), config.peers_to_query());

    info!(
        "Querying {} connected peers for their known peers",
        peers_to_query.len()
    );

    let found_peers = futures::stream::iter(peers_to_query)
        .map(|peer_id| {
            let channel = active_peers
                .get(&peer_id)
                .expect("Active peer should exist")
                .clone();
            let mut client = P2pClient::new(channel);
            async move {
                let mut request = Request::new(GetKnownPeersRequest {
                    timestamp_ms: now_unix(),
                });
                request.set_timeout(Duration::from_secs(PEER_QUERY_TIMEOUT_SECS));
                client
                    .get_known_peers(request)
                    .await
                    .ok()
                    .map(Response::into_inner)
                    .map(
                        |GetKnownPeersResponse {
                             own_info,
                             mut known_peers,
                         }| {
                            known_peers.push(own_info);
                            info!(
                                "Received {} peers from connected peer {}",
                                known_peers.len(),
                                peer_id.short_display(4)
                            );
                            known_peers
                        },
                    )
            }
        })
        .buffer_unordered(config.peers_to_query())
        .filter_map(std::future::ready)
        .flat_map(futures::stream::iter)
        .collect::<Vec<_>>()
        .await;

    info!("Found {} peers from connected peers", found_peers.len());

    update_known_peers(state, found_peers, allowlisted_peers);
}

fn update_known_peers(
    state: Arc<RwLock<DiscoveryState>>,
    found_peers: Vec<SignedNodeInfo>,
    allowlisted_peers: Arc<HashMap<PeerId, Option<Multiaddr>>>,
) {
    use std::collections::hash_map::Entry;

    let now_unix = now_unix();
    let our_peer_id = state.read().our_info.as_ref().unwrap().peer_id;
    let known_peers = &mut state.write().known_peers;

    // Only take the first MAX_PEERS_TO_SEND peers
    for peer_info in found_peers.into_iter().take(MAX_PEERS_TO_SEND) {
        // Skip peers whose timestamp is too far in the future from our clock
        // or that are too old
        if peer_info.timestamp_ms > now_unix.saturating_add(30 * 1_000) // 30 seconds
            || now_unix.saturating_sub(peer_info.timestamp_ms) > ONE_DAY_MILLISECONDS
        {
            continue;
        }

        if peer_info.peer_id == our_peer_id {
            continue;
        }

        // Verify peer signature
        let public_key = match Ed25519PublicKey::try_from(peer_info.peer_id) {
            Ok(pk) => pk,
            Err(_) => {
                debug!("Failed to convert PeerId to Ed25519PublicKey");
                continue;
            }
        };

        let msg = bcs::to_bytes(peer_info.data()).expect("BCS serialization should not fail");

        if let Err(e) = public_key.verify(&msg, peer_info.auth_sig()) {
            info!(
                "Discovery failed to verify signature for NodeInfo for peer {:?}: {e:?}",
                peer_info.peer_id
            );
            // TODO: consider denylisting the source of bad NodeInfo from future requests
            continue;
        }

        let verified_info = VerifiedSignedNodeInfo::new_from_verified(peer_info);

        match known_peers.entry(verified_info.peer_id) {
            Entry::Occupied(mut entry) => {
                let current_info = entry.get();
                if verified_info.timestamp_ms > current_info.timestamp_ms {
                    // Log if peer's address availability changed
                    debug!(
                        "Peer {} now has address information",
                        verified_info.peer_id.short_display(4)
                    );

                    entry.insert(verified_info);
                }
            }
            Entry::Vacant(entry) => {
                debug!(
                    "Discovered new peer {} with address",
                    verified_info.peer_id.short_display(4)
                );
                entry.insert(verified_info);
            }
        }
    }
}

const MAX_PEERS_TO_SEND: usize = 1000;
const MAX_ADDRESSES_PER_PEER: usize = 10;
const MAX_ADDRESS_LENGTH: usize = 1024;
const ONE_DAY_MILLISECONDS: u64 = 24 * 60 * 60 * 1_000;
