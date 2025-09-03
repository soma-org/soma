#![doc = include_str!("README.md")]

pub(crate) mod external_service;
pub(crate) mod internal_service;
mod network_tests;
pub mod tonic;

use crate::types::commit::Commit;
use crate::types::commit_votes::CommitVotes;
use crate::types::input::Input;
use crate::types::reveal::Reveal;
use crate::types::score_vote::ScoreVote;
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::crypto::keys::{EncoderPublicKey, PeerKeyPair, PeerPublicKey};
use shared::error::ShardResult;
use shared::{signed::Signed, verified::Verified};
use soma_network::multiaddr::Multiaddr;
use soma_tls::AllowPublicKeys;
use std::{sync::Arc, time::Duration};
use types::parameters::Parameters;
use types::shard_networking::EncoderNetworkingInfo;

/// Default message timeout for each request.
// TODO: make timeout configurable and tune the default timeout based on measured network latency
pub(crate) const MESSAGE_TIMEOUT: std::time::Duration = Duration::from_secs(60);

#[async_trait]
pub(crate) trait EncoderInternalNetworkClient: Send + Sync + Sized + 'static {
    async fn send_commit(
        &self,
        encoder: &EncoderPublicKey,
        commit: &Verified<Commit>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_commit_votes(
        &self,
        encoder: &EncoderPublicKey,
        votes: &Verified<CommitVotes>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_reveal(
        &self,
        encoder: &EncoderPublicKey,
        reveal: &Verified<Reveal>,
        timeout: Duration,
    ) -> ShardResult<()>;

    async fn send_score_vote(
        &self,
        encoder: &EncoderPublicKey,
        scores: &Verified<ScoreVote>,
        timeout: Duration,
    ) -> ShardResult<()>;
}

#[async_trait]
pub trait EncoderExternalNetworkClient: Send + Sync + Sized + 'static {
    async fn send_input(
        &self,
        encoder: &EncoderPublicKey,
        input: &Verified<Input>,
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
    async fn handle_send_score_vote(
        &self,
        encoder: &EncoderPublicKey,
        score_vote_bytes: Bytes,
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
        networking_info: EncoderNetworkingInfo,
        parameters: Arc<Parameters>,
        peer_keypair: PeerKeyPair,
        address: Multiaddr,
        allower: AllowPublicKeys,
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
