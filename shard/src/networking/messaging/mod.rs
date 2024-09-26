pub(crate) mod channel_pool;
pub(crate) mod encoder_tonic_service;
pub(crate) mod leader_tonic_service;
pub(crate) mod tonic;

mod tonic_gen {
    include!(concat!(env!("OUT_DIR"), "/shard.LeaderService.rs"));
    include!(concat!(env!("OUT_DIR"), "/shard.EncoderService.rs"));
}

use std::net::{SocketAddr, SocketAddrV4, SocketAddrV6};

use crate::types::multiaddr::{Multiaddr, Protocol};
use crate::{
    crypto::keys::NetworkKeyPair,
    error::ShardResult,
    types::{
        context::{EncoderContext, LeaderContext},
        network_committee::NetworkIdentityIndex,
        shard_commit::VerifiedSignedShardCommit,
        shard_endorsement::VerifiedSignedShardEndorsement,
        shard_input::VerifiedSignedShardInput,
        shard_selection::VerifiedSignedShardSelection,
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};

pub(crate) const MESSAGE_TIMEOUT: std::time::Duration = Duration::from_secs(60);

#[async_trait]
pub(crate) trait LeaderNetworkClient: Send + Sync + Sized + 'static {
    /// note: using network identity index because leaders can be any staked member
    /// and are not dependent on a specific modality
    async fn send_commit(
        &self,
        peer: NetworkIdentityIndex,
        commit: &VerifiedSignedShardCommit,
        timeout: Duration,
    ) -> ShardResult<()>;

    /// note: using network identity index because leaders can be any staked member
    /// and are not dependent on a specific modality
    async fn send_endorsement(
        &self,
        peer: NetworkIdentityIndex,
        endorsement: &VerifiedSignedShardEndorsement,
        timeout: Duration,
    ) -> ShardResult<()>;
}

#[async_trait]
pub(crate) trait EncoderNetworkClient: Send + Sync + Sized + 'static {
    async fn send_input(
        &self,
        peer: NetworkIdentityIndex,
        input: &VerifiedSignedShardInput,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_selection(
        &self,
        peer: NetworkIdentityIndex,
        selection: &VerifiedSignedShardSelection,
        timeout: Duration,
    ) -> ShardResult<()>;
}

#[async_trait]
pub(crate) trait LeaderNetworkService: Send + Sync + Sized + 'static {
    async fn handle_send_commit(
        &self,
        peer: NetworkIdentityIndex,
        commit: Bytes,
    ) -> ShardResult<()>;
    async fn handle_send_endorsement(
        &self,
        peer: NetworkIdentityIndex,
        endorsement: Bytes,
    ) -> ShardResult<()>;
}

#[async_trait]
pub(crate) trait EncoderNetworkService: Send + Sync + Sized + 'static {
    async fn handle_send_input(&self, peer: NetworkIdentityIndex, input: Bytes) -> ShardResult<()>;
    async fn handle_send_selection(
        &self,
        peer: NetworkIdentityIndex,
        selection: Bytes,
    ) -> ShardResult<()>;
}

pub(crate) trait LeaderNetworkManager<S>: Send + Sync
where
    S: LeaderNetworkService,
{
    type Client: EncoderNetworkClient;

    fn new(context: Arc<LeaderContext>, network_keypair: NetworkKeyPair) -> Self;
    fn encoder_client(&self) -> Arc<Self::Client>;
    async fn start(&mut self, service: Arc<S>);
    async fn stop(&mut self);
}

pub(crate) trait EncoderNetworkManager<S>: Send + Sync
where
    S: EncoderNetworkService,
{
    type Client: LeaderNetworkClient;

    fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self;
    fn leader_client(&self) -> Arc<Self::Client>;
    async fn start(&mut self, service: Arc<S>);
    async fn stop(&mut self);
}

fn to_host_port_str(addr: &Multiaddr) -> Result<String, &'static str> {
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
fn to_socket_addr(addr: &Multiaddr) -> Result<SocketAddr, &'static str> {
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
