pub(crate) mod channel_pool;
pub(crate) mod encoder_tonic_service;
pub(crate) mod tonic;

mod tonic_gen {
    include!(concat!(env!("OUT_DIR"), "/soma.EncoderService.rs"));
}

use std::net::{SocketAddr, SocketAddrV4, SocketAddrV6};

use crate::types::certificate::ShardCertificate;
use crate::types::manifest::Manifest;
use crate::types::multiaddr::{Multiaddr, Protocol};
use crate::types::shard::ShardRef;
// use crate::types::shard::ShardRef;
use crate::types::shard_commit::ShardCommit;
use crate::types::shard_endorsement::ShardEndorsement;
use crate::types::shard_input::ShardInput;
use crate::types::shard_reveal::ShardReveal;
use crate::types::{signed::Signed, verified::Verified};
use crate::ProtocolKeySignature;
use crate::{
    crypto::keys::NetworkKeyPair,
    error::ShardResult,
    types::{context::EncoderContext, network_committee::NetworkingIndex},
};
use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};

pub(crate) const MESSAGE_TIMEOUT: std::time::Duration = Duration::from_secs(60);

#[async_trait]
pub(crate) trait EncoderClient: Send + Sync + Sized + 'static {
    //TODO: ensure everything going over the wire is verified and versioned
    async fn send_shard_input(
        &self,
        peer: NetworkingIndex,
        input: &Verified<Signed<ShardInput>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn get_shard_input(
        &self,
        peer: NetworkingIndex,
        input: &Verified<ShardRef>,
        timeout: Duration,
    ) -> ShardResult<Bytes>;

    async fn send_probes(
        &self,
        peer: NetworkingIndex,
        probes: &Vec<Verified<ShardCertificate<Signed<Probe>>>>,
        timeout: Duration,
    );

    async fn get_probes(
        &self,
        peer: NetworkingIndex,
        slots: ShardSlots,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>>;

    async fn get_commit_signature(
        &self,
        peer: NetworkingIndex,
        commit: &Verified<Signed<ShardCommit>>,
        timeout: Duration,
    ) -> ShardResult<Bytes>;

    async fn send_certified_commits(
        &self,
        peer: NetworkingIndex,
        commits: &Vec<Verified<ShardCertificate<Signed<ShardCommit>>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn get_certified_commits(
        &self,
        peer: NetworkingIndex,
        // TODO: fix type and verify
        slots: ShardSlots,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>>;

    async fn get_reveal_signature(
        &self,
        peer: NetworkingIndex,
        reveal: &Verified<Signed<ShardReveal>>,
        timeout: Duration,
    ) -> ShardResult<Bytes>;

    async fn send_certified_reveals(
        &self,
        peer: NetworkingIndex,
        reveals: &Vec<Verified<ShardCertificate<Signed<ShardReveal>>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn get_certified_reveals(
        &self,
        peer: NetworkingIndex,
        // TODO: fix type and verify
        slots: ShardSlots,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>>;

    async fn send_commit_manifests(
        &self,
        peer: NetworkingIndex,
        commits: &Vec<Verified<Signed<Manifest>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn get_commit_manifests(
        &self,
        peer: NetworkingIndex,
        slots: ShardSlots,
        timeout: Duration,
    ) -> ShardResult<Vec<Bytes>>;

    async fn send_removal_signatures(
        &self,
        peer: NetworkingIndex,
        //TODO: fix type
        removals: &Vec<Verified<Signed<Removal>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_certified_removals(
        &self,
        peer: NetworkingIndex,
        // TODO: fix type
        removals: &Vec<Verified<ShardCertificate<Removal>>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_removal_set(
        &self,
        peer: NetworkingIndex,
        removal_set: &Verified<Signed<RemovalSet>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_endorsement(
        &self,
        peer: NetworkingIndex,
        removal_set: &Verified<Signed<ShardEndorsement>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_finality_proof(
        &self,
        peer: NetworkingIndex,
        finality_proof: &Verified<EmbeddingFinalityProof>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_delivery_proof(
        &self,
        peer: NetworkingIndex,
        delivery_proof: &Verified<EmbeddingDeliveryProof>,
        timeout: Duration,
    ) -> ShardResult<()>;
}

#[async_trait]
pub(crate) trait EncoderService: Send + Sync + Sized + 'static {
    async fn handle_send_input(&self, peer: NetworkingIndex, input: Bytes) -> ShardResult<()>;
    async fn handle_get_input(
        &self,
        peer: NetworkingIndex,
        shard_ref: Bytes,
    ) -> ShardResult<VerifiedSignedShardInput>;
    async fn handle_send_selection(
        &self,
        peer: NetworkingIndex,
        selection: Bytes,
    ) -> ShardResult<()>;
}

pub(crate) trait EncoderManager<S>: Send + Sync
where
    S: EncoderService,
{
    type Client: EncoderClient;

    fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self;
    fn client(&self) -> Arc<Self::Client>;
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
