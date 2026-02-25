// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::net::{SocketAddr, SocketAddrV4, SocketAddrV6};

use fastcrypto::ed25519::Ed25519Signature;
use fastcrypto::traits::Signer;
use multiaddr::Protocol;
use serde::{Deserialize, Serialize};

use crate::checkpoints::{
    CertifiedCheckpointSummary, CheckpointSequenceNumber, FullCheckpointContents,
};
use crate::crypto::NetworkKeyPair;
use crate::digests::{CheckpointContentsDigest, CheckpointDigest, Digest};
use crate::envelope::{Envelope, Message, VerifiedEnvelope};
use crate::full_checkpoint_content::Checkpoint;
use crate::intent::IntentScope;
use crate::multiaddr::Multiaddr;
use crate::peer_id::PeerId;

// pub mod connection_manager;
// pub mod network;
#[cfg(feature = "tls")]
pub mod active_peers;
#[cfg(feature = "server")]
pub mod channel_manager;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetKnownPeersRequest {
    pub own_info: SignedNodeInfo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetKnownPeersResponse {
    pub own_info: SignedNodeInfo,
    pub known_peers: Vec<SignedNodeInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetCheckpointAvailabilityRequest {
    // This is needed to make gRPC happy.
    pub _unused: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetCheckpointAvailabilityResponse {
    pub highest_synced_checkpoint: CertifiedCheckpointSummary,
    pub lowest_available_checkpoint: CheckpointSequenceNumber,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PushCheckpointSummaryRequest {
    pub checkpoint: CertifiedCheckpointSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PushCheckpointSummaryResponse {
    // This is needed to make gRPC happy.
    pub _unused: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GetCheckpointSummaryRequest {
    Latest,
    ByDigest(CheckpointDigest),
    BySequenceNumber(CheckpointSequenceNumber),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetCheckpointSummaryResponse {
    pub checkpoint: Option<CertifiedCheckpointSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetCheckpointContentsRequest {
    pub digest: CheckpointContentsDigest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetCheckpointContentsResponse {
    pub contents: Option<FullCheckpointContents>,
}

/// The information necessary to dial another peer.
///
/// `NodeInfo` contains all the information that is shared with other nodes via the discovery
/// service to advertise how a node can be reached.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeInfo {
    pub peer_id: PeerId,
    pub address: Multiaddr,

    /// Creation time.
    ///
    /// This is used to determine which of two NodeInfo's from the same PeerId should be retained.
    pub timestamp_ms: u64,
    // pub access_type: AccessType,
}

impl NodeInfo {
    pub fn sign(self, keypair: &NetworkKeyPair) -> SignedNodeInfo {
        let msg = bcs::to_bytes(&self).expect("BCS serialization should not fail");
        let sig = keypair.clone().into_inner().sign(&msg);
        SignedNodeInfo::new_from_data_and_sig(self, sig)
    }
}

pub type SignedNodeInfo = Envelope<NodeInfo, Ed25519Signature>;

pub type VerifiedSignedNodeInfo = VerifiedEnvelope<NodeInfo, Ed25519Signature>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NodeInfoDigest(Digest);

impl NodeInfoDigest {
    pub const fn new(digest: [u8; 32]) -> Self {
        Self(Digest::new(digest))
    }
}

impl Message for NodeInfo {
    type DigestType = NodeInfoDigest;
    const SCOPE: IntentScope = IntentScope::DiscoveryPeers;

    fn digest(&self) -> Self::DigestType {
        unreachable!("NodeInfoDigest is not used")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerEvent {
    NewPeer { peer_id: PeerId, address: Multiaddr },
    LostPeer { peer_id: PeerId, reason: DisconnectReason },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisconnectReason {
    ConnectionLost,
    Shutdown,
    RequestedDisconnect,
    SimultaneousDialResolution,
}

pub fn to_host_port_str(addr: &Multiaddr) -> Result<String, &'static str> {
    let mut iter = addr.iter();

    match (iter.next(), iter.next()) {
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Dns(hostname)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", hostname, port))
        }

        _ => {
            tracing::warn!("unsupported multiaddr: '{addr}'");
            Err("invalid address")
        }
    }
}

/// Attempts to convert a multiaddr of the form `/[ip4,ip6]/{}/[udp,tcp]/{port}` into
/// a SocketAddr value.
pub fn to_socket_addr(addr: &Multiaddr) -> Result<SocketAddr, &'static str> {
    let mut iter = addr.iter();

    match (iter.next(), iter.next()) {
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Udp(port)))
        | (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(SocketAddr::V4(SocketAddrV4::new(ipaddr, port)))
        }

        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Udp(port)))
        | (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(SocketAddr::V6(SocketAddrV6::new(ipaddr, port, 0, 0)))
        }

        _ => {
            tracing::warn!("unsupported multiaddr: '{addr}'");
            Err("invalid address")
        }
    }
}
