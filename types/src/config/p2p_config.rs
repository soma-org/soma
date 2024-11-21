use std::{net::SocketAddr, num::NonZeroU32, time::Duration};

use serde::{Deserialize, Serialize};

use crate::{multiaddr::Multiaddr, peer_id::PeerId};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct P2pConfig {
    // /// The address that the p2p network will bind on.
    // pub listen_address: Option<Multiaddr>,
    /// The external address other nodes can use to reach this node.
    /// This will be shared with other peers through the discovery service
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_address: Option<Multiaddr>,
    /// SeedPeers are preferred and the node will always try to ensure a
    /// connection is established with these nodes.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub seed_peers: Vec<SeedPeer>,

    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub state_sync: Option<StateSyncConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discovery: Option<DiscoveryConfig>,
}

impl Default for P2pConfig {
    fn default() -> Self {
        Self {
            // listen_address: Default::default(),
            external_address: Default::default(),
            seed_peers: Default::default(),
            discovery: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct SeedPeer {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peer_id: Option<PeerId>,
    pub address: Multiaddr,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct AllowlistedPeer {
    pub peer_id: PeerId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<Multiaddr>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct DiscoveryConfig {
    /// Query peers for their latest checkpoint every interval period.
    ///
    /// If unspecified, this will default to `5,000` milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval_period_ms: Option<u64>,

    /// Target number of concurrent connections to establish.
    ///
    /// If unspecified, this will default to `4`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_concurrent_connections: Option<usize>,

    /// Number of peers to query each interval.
    ///
    /// Sets the number of peers, to be randomly selected, that are queried for their known peers
    /// each interval.
    ///
    /// If unspecified, this will default to `1`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peers_to_query: Option<usize>,

    /// Per-peer rate-limit (in requests/sec) for the GetKnownPeers RPC.
    ///
    /// If unspecified, this will default to no limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub get_known_peers_rate_limit: Option<NonZeroU32>,

    /// Like `seed_peers` in `P2pConfig`, allowlisted peers will awlays be allowed to establish
    /// connection with this node regardless of the concurrency limit.
    /// Unlike `seed_peers`, a node does not reach out to `allowlisted_peers` preferentially.
    /// It is also used to determine if a peer is accessible when its AccessType is Private.
    /// For example, a node will ignore a peer with Private AccessType if the peer is not in
    /// its `allowlisted_peers`. Namely, the node will not try to establish connections
    /// to this peer, nor advertise this peer's info to other peers in the network.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub allowlisted_peers: Vec<AllowlistedPeer>,
}

impl DiscoveryConfig {
    pub fn interval_period(&self) -> Duration {
        const INTERVAL_PERIOD_MS: u64 = 5_000; // 5 seconds

        Duration::from_millis(self.interval_period_ms.unwrap_or(INTERVAL_PERIOD_MS))
    }

    pub fn target_concurrent_connections(&self) -> usize {
        const TARGET_CONCURRENT_CONNECTIONS: usize = 4;

        self.target_concurrent_connections
            .unwrap_or(TARGET_CONCURRENT_CONNECTIONS)
    }

    pub fn peers_to_query(&self) -> usize {
        const PEERS_TO_QUERY: usize = 5;

        self.peers_to_query.unwrap_or(PEERS_TO_QUERY)
    }
}
