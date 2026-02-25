// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tokio::sync::broadcast;
use tonic_rustls::Channel;

use super::{DisconnectReason, PeerEvent};
use crate::crypto::NetworkPublicKey;
use crate::multiaddr::Multiaddr;
use crate::peer_id::PeerId;

#[derive(Clone)]
pub struct ActivePeers {
    inner: Arc<RwLock<ActivePeersInner>>,
}

#[derive(Debug, Clone)]
pub struct PeerState {
    pub channel: Channel,
    address: Multiaddr,
    pub public_key: NetworkPublicKey,
    connected_since: Instant,
}

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum ConnectionOrigin {
//     Inbound,
//     Outbound,
// }

struct ActivePeersInner {
    peers: HashMap<PeerId, PeerState>,
    event_sender: broadcast::Sender<PeerEvent>,
}

impl ActivePeers {
    pub fn new(broadcast_capacity: usize) -> Self {
        let (event_sender, _) = broadcast::channel(broadcast_capacity);
        Self {
            inner: Arc::new(RwLock::new(ActivePeersInner { peers: HashMap::new(), event_sender })),
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<PeerEvent> {
        self.inner.read().event_sender.subscribe()
    }

    pub fn peers(&self) -> Vec<PeerId> {
        self.inner.read().peers.keys().copied().collect()
    }

    pub fn get(&self, peer_id: &PeerId) -> Option<Channel> {
        self.inner.read().peers.get(peer_id).map(|state| state.channel.clone())
    }

    pub fn insert(
        &self,
        peer_id: PeerId,
        address: Multiaddr,
        channel: Channel,
        public_key: NetworkPublicKey,
    ) -> Option<Channel> {
        let mut inner = self.inner.write();
        let result = inner
            .peers
            .insert(
                peer_id,
                PeerState {
                    channel: channel.clone(),
                    address: address.clone(),
                    public_key,
                    connected_since: Instant::now(),
                },
            )
            .map(|old_state| old_state.channel);

        // Send event if this is a new peer
        if result.is_none() {
            let _ = inner.event_sender.send(PeerEvent::NewPeer { peer_id, address });
        }

        result
    }

    pub fn remove(&self, peer_id: &PeerId, reason: DisconnectReason) -> Option<Channel> {
        let mut inner = self.inner.write();
        let result = inner.peers.remove(peer_id).map(|state| state.channel);

        if result.is_some() {
            let _ = inner
                .event_sender
                .send(PeerEvent::LostPeer { peer_id: *peer_id, reason: reason.clone() });
        }

        result
    }

    pub fn len(&self) -> usize {
        self.inner.read().peers.len()
    }

    pub fn contains(&self, peer_id: &PeerId) -> bool {
        self.inner.read().peers.contains_key(peer_id)
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().peers.is_empty()
    }

    pub fn get_state(&self, peer_id: &PeerId) -> Option<PeerState> {
        self.inner.read().peers.get(peer_id).cloned()
    }
}
