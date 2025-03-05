#![doc = include_str!("README.md")]
pub(crate) mod channel_pool;
pub(crate) mod tonic_network;

/// Includes the generated protobuf/gRPC service code
mod tonic_gen {
    include!(concat!(
        env!("OUT_DIR"),
        "/soma.EncoderInternalTonicService.rs"
    ));
}

use crate::types::certified::Certified;
use crate::types::encoder_committee::EncoderIndex;
use crate::types::shard_commit::ShardCommit;
use crate::types::shard_reveal::ShardReveal;
use crate::types::shard_scores::ShardScores;
use crate::types::shard_votes::{CommitRound, RevealRound, ShardVotes};
use crate::{error::ShardResult, types::encoder_context::EncoderContext};
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use multiaddr::Protocol;
use shared::crypto::keys::NetworkKeyPair;
use shared::multiaddr::Multiaddr;
use shared::serialized::Serialized;
use shared::signed::Signature;
use shared::{signed::Signed, verified::Verified};
use std::net::{SocketAddr, SocketAddrV4, SocketAddrV6};
use std::{sync::Arc, time::Duration};

/// Default message timeout for each request.
// TODO: make timeout configurable and tune the default timeout based on measured network latency
pub(crate) const MESSAGE_TIMEOUT: std::time::Duration = Duration::from_secs(60);

#[async_trait]
pub(crate) trait EncoderInternalNetworkClient: Send + Sync + Sized + 'static {
    async fn send_commit(
        &self,
        peer: EncoderIndex,
        commit: &Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<Bytes>;

    async fn send_certified_commit(
        &self,
        peer: EncoderIndex,
        certified_commit: &Verified<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_commit_votes(
        &self,
        peer: EncoderIndex,
        votes: &Verified<Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_reveal(
        &self,
        peer: EncoderIndex,
        reveal: &Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;
    async fn send_reveal_votes(
        &self,
        peer: EncoderIndex,
        votes: &Verified<Signed<ShardVotes<RevealRound>, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;
    async fn send_scores(
        &self,
        peer: EncoderIndex,
        scores: &Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;
}

#[async_trait]
pub(crate) trait EncoderInternalNetworkService: Send + Sync + Sized + 'static {
    async fn handle_send_commit(
        &self,
        peer: EncoderIndex,
        commit_bytes: Bytes,
    ) -> ShardResult<
        Serialized<
            Signature<Signed<ShardCommit, min_sig::BLS12381Signature>, min_sig::BLS12381Signature>,
        >,
    >;
    async fn handle_send_certified_commit(
        &self,
        peer: EncoderIndex,
        certified_commit_bytes: Bytes,
    ) -> ShardResult<()>;
    async fn handle_send_commit_votes(
        &self,
        peer: EncoderIndex,
        votes_bytes: Bytes,
    ) -> ShardResult<()>;
    async fn handle_send_reveal(&self, peer: EncoderIndex, reveal_bytes: Bytes) -> ShardResult<()>;
    async fn handle_send_reveal_votes(
        &self,
        peer: EncoderIndex,
        votes_bytes: Bytes,
    ) -> ShardResult<()>;
    async fn handle_send_scores(&self, peer: EncoderIndex, scores_bytes: Bytes) -> ShardResult<()>;
}

/// `EncoderNetworkManager` handles starting and stopping the network related services
/// The network manager also provides clients to other encoders in an efficient way.
pub(crate) trait EncoderInternalNetworkManager<S>: Send + Sync
where
    S: EncoderInternalNetworkService,
{
    /// type alias
    type Client: EncoderInternalNetworkClient;
    /// Creates a new manager by taking an encoder context and a network keypair.
    /// The network keypair is used for TLS authentication.
    fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self;
    /// Returns a client
    fn client(&self) -> Arc<Self::Client>;
    /// Starts the network services
    async fn start(&mut self, service: Arc<S>);
    /// Stops the network services
    async fn stop(&mut self);
}

/// Converts a multiaddress to a host:port string handling both TCP and UDP
pub(crate) fn to_host_port_str(addr: &Multiaddr) -> Result<String, &'static str> {
    let mut iter = addr.iter();

    match (iter.next(), iter.next()) {
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Udp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Udp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(format!("{}:{}", ipaddr, port))
        }
        (Some(Protocol::Dns(hostname)), Some(Protocol::Udp(port))) => {
            Ok(format!("{}:{}", hostname, port))
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
/// Converts multiaddress to socket address
pub(crate) fn to_socket_addr(addr: &Multiaddr) -> Result<SocketAddr, &'static str> {
    let mut iter = addr.iter();
    match (iter.next(), iter.next()) {
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Udp(port))) => {
            Ok(SocketAddr::V4(SocketAddrV4::new(ipaddr, port)))
        }
        (Some(Protocol::Ip4(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(SocketAddr::V4(SocketAddrV4::new(ipaddr, port)))
        }
        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Udp(port))) => {
            Ok(SocketAddr::V6(SocketAddrV6::new(ipaddr, port, 0, 0)))
        }
        (Some(Protocol::Ip6(ipaddr)), Some(Protocol::Tcp(port))) => {
            Ok(SocketAddr::V6(SocketAddrV6::new(ipaddr, port, 0, 0)))
        }
        _ => {
            tracing::warn!("unsupported multiaddr: '{addr}'");
            Err("invalid address")
        }
    }
}
