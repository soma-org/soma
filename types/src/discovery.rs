use fastcrypto::{ed25519::Ed25519Signature, traits::Signer};
use serde::{Deserialize, Serialize};

use crate::{
    crypto::NetworkKeyPair,
    digests::Digest,
    envelope::{Envelope, Message, VerifiedEnvelope},
    intent::IntentScope,
    multiaddr::Multiaddr,
    peer_id::PeerId,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetKnownPeersRequest {
    pub own_info: SignedNodeInfo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetKnownPeersResponse {
    pub own_info: SignedNodeInfo,
    pub known_peers: Vec<SignedNodeInfo>,
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
