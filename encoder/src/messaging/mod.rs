#![doc = include_str!("README.md")]

pub(crate) mod external_service;
pub(crate) mod internal_service;
mod network_tests;
pub mod tonic;

use crate::error::ShardResult;
use crate::types::parameters::Parameters;
use crate::types::shard_commit::ShardCommit;
use crate::types::shard_commit_votes::ShardCommitVotes;
use crate::types::shard_input::ShardInput;
use crate::types::shard_reveal::ShardReveal;
use crate::types::shard_reveal_votes::ShardRevealVotes;
use crate::types::shard_scores::ShardScores;
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::crypto::keys::{EncoderPublicKey, PeerKeyPair, PeerPublicKey};
use shared::{signed::Signed, verified::Verified};
use soma_network::multiaddr::Multiaddr;
use soma_tls::AllowPublicKeys;
use std::{sync::Arc, time::Duration};
use tonic::internal::ConnectionsInfo;
use tonic::NetworkingInfo;

/// Default message timeout for each request.
// TODO: make timeout configurable and tune the default timeout based on measured network latency
pub(crate) const MESSAGE_TIMEOUT: std::time::Duration = Duration::from_secs(60);

#[async_trait]
pub(crate) trait EncoderInternalNetworkClient: Send + Sync + Sized + 'static {
    async fn send_commit(
        &self,
        encoder: &EncoderPublicKey,
        commit: &Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_commit_votes(
        &self,
        encoder: &EncoderPublicKey,
        votes: &Verified<Signed<ShardCommitVotes, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_reveal(
        &self,
        encoder: &EncoderPublicKey,
        reveal: &Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;
    async fn send_reveal_votes(
        &self,
        encoder: &EncoderPublicKey,
        votes: &Verified<Signed<ShardRevealVotes, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;
    async fn send_scores(
        &self,
        encoder: &EncoderPublicKey,
        scores: &Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;
}

#[async_trait]
pub trait EncoderExternalNetworkClient: Send + Sync + Sized + 'static {
    async fn send_input(
        &self,
        encoder: &EncoderPublicKey,
        input: &Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()>;
}

#[async_trait]
pub(crate) trait EncoderInternalNetworkService: Send + Sync + Sized + 'static {
    async fn handle_send_commit(
        &self,
        encoder: &EncoderPublicKey,
        commit_bytes: Bytes,
    ) -> ShardResult<()>;
    async fn handle_send_commit_votes(
        &self,
        encoder: &EncoderPublicKey,
        votes_bytes: Bytes,
    ) -> ShardResult<()>;
    async fn handle_send_reveal(
        &self,
        encoder: &EncoderPublicKey,
        reveal_bytes: Bytes,
    ) -> ShardResult<()>;
    async fn handle_send_reveal_votes(
        &self,
        encoder: &EncoderPublicKey,
        votes_bytes: Bytes,
    ) -> ShardResult<()>;
    async fn handle_send_scores(
        &self,
        encoder: &EncoderPublicKey,
        scores_bytes: Bytes,
    ) -> ShardResult<()>;
}

#[async_trait]
pub(crate) trait EncoderExternalNetworkService: Send + Sync + Sized + 'static {
    async fn handle_send_input(&self, peer: &PeerPublicKey, input_bytes: Bytes) -> ShardResult<()>;
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
    fn new(
        networking_info: NetworkingInfo,
        parameters: Arc<Parameters>,
        peer_keypair: PeerKeyPair,
        address: Multiaddr,
        allower: AllowPublicKeys,
        connections_info: ConnectionsInfo,
    ) -> Self;
    /// Returns a client
    fn client(&self) -> Arc<Self::Client>;
    /// Starts the network services
    async fn start(&mut self, service: Arc<S>);
    /// Stops the network services
    async fn stop(&mut self);
}

pub(crate) trait EncoderExternalNetworkManager<S>: Send + Sync
where
    S: EncoderExternalNetworkService,
{
    /// Creates a new manager by taking an encoder context and a network keypair.
    /// The network keypair is used for TLS authentication.
    fn new(
        parameters: Arc<Parameters>,
        peer_keypair: PeerKeyPair,
        address: Multiaddr,
        allower: AllowPublicKeys,
    ) -> Self;
    /// Starts the network services
    async fn start(&mut self, service: Arc<S>);
    /// Stops the network services
    async fn stop(&mut self);
}
