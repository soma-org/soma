//! Tonic Network contains all the code related to tonic-specific code implementing the network client, service, and manager traits.
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::NetworkKeyPair, network_committee::NetworkingIndex, signed::Signed,
    verified::Verified,
};
use std::{io::Read, sync::Arc, time::Duration};
use tokio::sync::oneshot;
use tonic::{transport::Server, Request, Response};
use tower_http::add_extension::AddExtensionLayer;

use crate::{
    error::{ShardError, ShardResult},
    networking::messaging::{
        to_socket_addr,
        tonic_gen::{
            encoder_external_tonic_service_server::EncoderExternalTonicServiceServer,
            encoder_internal_tonic_service_server::EncoderInternalTonicServiceServer,
        },
    },
    types::{
        certified::Certified,
        encoder_context::EncoderContext,
        shard_commit::ShardCommit,
        shard_input::ShardInput,
        shard_reveal::ShardReveal,
        shard_scores::ShardScores,
        shard_votes::{CommitRound, RevealRound, ShardVotes},
    },
};
use tracing::info;

use crate::networking::messaging::tonic_gen::encoder_internal_tonic_service_client::EncoderInternalTonicServiceClient;

use super::{
    channel_pool::{Channel, ChannelPool},
    tonic_gen::{
        encoder_external_tonic_service_client::EncoderExternalTonicServiceClient,
        encoder_external_tonic_service_server::EncoderExternalTonicService,
        encoder_internal_tonic_service_server::EncoderInternalTonicService,
    },
    EncoderExternalNetworkClient, EncoderExternalNetworkManager, EncoderExternalNetworkService,
    EncoderIndex, EncoderInternalNetworkClient, EncoderInternalNetworkManager,
    EncoderInternalNetworkService,
};

// Implements Tonic RPC client for Encoders.
pub(crate) struct EncoderInternalTonicClient {
    /// network_keypair used for TLS
    network_keypair: NetworkKeyPair,
    /// channel pool for tonic channel reuse
    channel_pool: Arc<ChannelPool>,
}

/// Implments the core functionality of the encoder tonic client
impl EncoderInternalTonicClient {
    /// Creates a new encoder tonic client and establishes an arc'd channel pool
    pub(crate) fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self {
            network_keypair,
            channel_pool: Arc::new(ChannelPool::new(context)),
        }
    }

    /// returns an encoder client
    // TODO: re-introduce configuring limits to the client for safety
    async fn get_client(
        &self,
        peer: EncoderIndex,
        timeout: Duration,
    ) -> ShardResult<EncoderInternalTonicServiceClient<Channel>> {
        // let config = &self.context.parameters.tonic;
        let channel = self.channel_pool.get_channel(peer, timeout).await?;
        Ok(EncoderInternalTonicServiceClient::new(channel))
        // .max_encoding_message_size(config.message_size_limit)
        // .max_decoding_message_size(config.message_size_limit))
    }
}

#[async_trait]
impl EncoderInternalNetworkClient for EncoderInternalTonicClient {
    async fn send_commit(
        &self,
        peer: EncoderIndex,
        commit: &Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<Bytes> {
        let mut request = Request::new(SendCommitRequest {
            commit: commit.bytes(),
        });
        request.set_timeout(timeout);
        let response = self
            .get_client(peer, timeout)
            .await?
            .send_commit(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(response.into_inner().partial_signature)
    }

    async fn send_certified_commit(
        &self,
        peer: EncoderIndex,
        certified_commit: &Verified<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendCertifiedCommitRequest {
            certified_commit: certified_commit.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_certified_commit(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(())
    }

    async fn send_commit_votes(
        &self,
        peer: EncoderIndex,
        votes: &Verified<Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendCommitVotesRequest {
            votes: votes.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_commit_votes(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(())
    }

    async fn send_reveal(
        &self,
        peer: EncoderIndex,
        reveal: &Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendRevealRequest {
            reveal: reveal.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_reveal(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(())
    }
    async fn send_reveal_votes(
        &self,
        peer: EncoderIndex,
        votes: &Verified<Signed<ShardVotes<RevealRound>, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendRevealVotesRequest {
            votes: votes.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_reveal_votes(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(())
    }
    async fn send_scores(
        &self,
        peer: EncoderIndex,
        scores: &Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendScoresRequest {
            scores: scores.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_scores(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(())
    }
}

/// Proxies Tonic requests to `NetworkService` with actual handler implementation.
struct EncoderInternalTonicServiceProxy<S: EncoderInternalNetworkService> {
    /// Encoder context
    context: Arc<EncoderContext>,
    /// Encoder Network Service - this is typically the same even between different networking stacks. The trait
    /// makes testing easier.
    service: Arc<S>,
}

/// Implements a new method to create an encoder tonic service proxy
impl<S: EncoderInternalNetworkService> EncoderInternalTonicServiceProxy<S> {
    /// Creates the tonic service proxy using pre-established context and service
    const fn new(context: Arc<EncoderContext>, service: Arc<S>) -> Self {
        Self { context, service }
    }
}

/// Used to pack the networking index into each request. Using a new type
/// such that this can be extended in the future. May want to version this however?
#[derive(Clone, Debug)]
pub(crate) struct PeerInfo {
    /// networking index, verified using the TLS networking keypair
    pub(crate) encoder_index: EncoderIndex,
    pub(crate) networking_index: NetworkingIndex,
}

#[async_trait]
impl<S: EncoderInternalNetworkService> EncoderInternalTonicService
    for EncoderInternalTonicServiceProxy<S>
{
    async fn send_commit(
        &self,
        request: Request<SendCommitRequest>,
    ) -> Result<Response<SendCommitResponse>, tonic::Status> {
        let Some(peer) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.encoder_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let commit = request.into_inner().commit;

        let partial_signature = self
            .service
            .handle_send_commit(peer, commit)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendCommitResponse {
            partial_signature: partial_signature.bytes(),
        }))
    }
    async fn send_certified_commit(
        &self,
        request: Request<SendCertifiedCommitRequest>,
    ) -> Result<Response<SendCertifiedCommitResponse>, tonic::Status> {
        let Some(peer) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.encoder_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let certified_commit = request.into_inner().certified_commit;

        self.service
            .handle_send_certified_commit(peer, certified_commit)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendCertifiedCommitResponse {}))
    }
    async fn send_commit_votes(
        &self,
        request: Request<SendCommitVotesRequest>,
    ) -> Result<Response<SendCommitVotesResponse>, tonic::Status> {
        let Some(peer) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.encoder_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let votes = request.into_inner().votes;

        self.service
            .handle_send_commit_votes(peer, votes)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendCommitVotesResponse {}))
    }
    async fn send_reveal(
        &self,
        request: Request<SendRevealRequest>,
    ) -> Result<Response<SendRevealResponse>, tonic::Status> {
        let Some(peer) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.encoder_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let reveal = request.into_inner().reveal;

        self.service
            .handle_send_reveal(peer, reveal)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendRevealResponse {}))
    }
    async fn send_reveal_votes(
        &self,
        request: Request<SendRevealVotesRequest>,
    ) -> Result<Response<SendRevealVotesResponse>, tonic::Status> {
        let Some(peer) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.encoder_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let votes = request.into_inner().votes;

        self.service
            .handle_send_reveal_votes(peer, votes)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendRevealVotesResponse {}))
    }
    async fn send_scores(
        &self,
        request: Request<SendScoresRequest>,
    ) -> Result<Response<SendScoresResponse>, tonic::Status> {
        let Some(peer) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.encoder_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let scores = request.into_inner().scores;

        self.service
            .handle_send_scores(peer, scores)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendScoresResponse {}))
    }
}

/// Tonic specific manager type that contains a tonic specific client and
/// the oneshot tokio channel to trigger service shutdown.
pub struct EncoderInternalTonicManager {
    context: Arc<EncoderContext>,
    client: Arc<EncoderInternalTonicClient>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

/// Implementation of the encoder tonic manager that contains a new fn to create the type
// TODO: switch this to type state pattern
impl EncoderInternalTonicManager {
    /// Takes context, and network keypair and creates a new encoder tonic client
    pub fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self {
            context: context.clone(),
            client: Arc::new(EncoderInternalTonicClient::new(context, network_keypair)),
            shutdown_tx: None,
        }
    }
}

impl<S: EncoderInternalNetworkService> EncoderInternalNetworkManager<S>
    for EncoderInternalTonicManager
{
    type Client = EncoderInternalTonicClient;

    fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self::new(context, network_keypair)
    }

    fn client(&self) -> Arc<Self::Client> {
        self.client.clone()
    }

    /// if the network is running locally, then it uses the localhost address, otherwise
    /// it uses the zero address since it will be used in a hosted context where the service will
    /// be routed to using the IP address. The function starts a gRPC server taking a shutdown channel
    /// to allow the system to trigger shutdown from outside of the spawned tokio task.
    async fn start(&mut self, service: Arc<S>) {
        let encoder = self
            .context
            .encoder_committee
            .encoder(self.context.own_encoder_index);
        let own_encoder_index = self.context.own_encoder_index;
        let own_networking_index = self.context.own_network_index;
        // By default, bind to the unspecified address to allow the actual address to be assigned.
        // But bind to localhost if it is requested.
        let own_address = if encoder.address.is_localhost_ip() {
            encoder.address.clone()
        } else {
            encoder.address.with_zero_ip()
        };
        let own_address = to_socket_addr(&own_address).unwrap();
        let svc = EncoderInternalTonicServiceServer::new(EncoderInternalTonicServiceProxy::new(
            self.context.clone(),
            service,
        ));
        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        tokio::spawn(async move {
            // let leader_service = Server::builder().add_service(LeaderShardServiceServer::new(svc));

            let tower_layer = tower::ServiceBuilder::new()
                .layer(AddExtensionLayer::new(PeerInfo {
                    encoder_index: own_encoder_index,
                    networking_index: own_networking_index,
                }))
                .into_inner();

            Server::builder()
                .layer(tower_layer)
                .add_service(svc)
                .serve_with_shutdown(own_address, async {
                    rx.await.ok();
                })
                .await
                .unwrap();
        });
        info!("Binding tonic server to address {:?}", own_address);
    }

    async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}
// ////////////////////////////////////////////////////////////////////

#[derive(Clone, prost::Message)]
pub(crate) struct SendCommitRequest {
    #[prost(bytes = "bytes", tag = "1")]
    commit: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendCommitResponse {
    #[prost(bytes = "bytes", tag = "1")]
    partial_signature: Bytes,
}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct SendCertifiedCommitRequest {
    #[prost(bytes = "bytes", tag = "1")]
    certified_commit: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendCertifiedCommitResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct SendCommitVotesRequest {
    #[prost(bytes = "bytes", tag = "1")]
    votes: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendCommitVotesResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct SendRevealRequest {
    #[prost(bytes = "bytes", tag = "1")]
    reveal: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendRevealResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct SendRevealVotesRequest {
    #[prost(bytes = "bytes", tag = "1")]
    votes: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendRevealVotesResponse {}

// ////////////////////////////////////////////////////////////////////
#[derive(Clone, prost::Message)]
pub(crate) struct SendScoresRequest {
    #[prost(bytes = "bytes", tag = "1")]
    scores: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendScoresResponse {}

// Implements Tonic RPC client for Encoders.
pub(crate) struct EncoderExternalTonicClient {
    /// network_keypair used for TLS
    network_keypair: NetworkKeyPair,
    /// channel pool for tonic channel reuse
    channel_pool: Arc<ChannelPool>,
}

/// Implments the core functionality of the encoder tonic client
impl EncoderExternalTonicClient {
    /// Creates a new encoder tonic client and establishes an arc'd channel pool
    pub(crate) fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self {
            network_keypair,
            channel_pool: Arc::new(ChannelPool::new(context)),
        }
    }

    /// returns an encoder client
    // TODO: re-introduce configuring limits to the client for safety
    async fn get_client(
        &self,
        peer: EncoderIndex,
        timeout: Duration,
    ) -> ShardResult<EncoderExternalTonicServiceClient<Channel>> {
        // let config = &self.context.parameters.tonic;
        let channel = self.channel_pool.get_channel(peer, timeout).await?;
        Ok(EncoderExternalTonicServiceClient::new(channel))
        // .max_encoding_message_size(config.message_size_limit)
        // .max_decoding_message_size(config.message_size_limit))
    }
}

#[async_trait]
impl EncoderExternalNetworkClient for EncoderExternalTonicClient {
    async fn send_input(
        &self,
        peer: EncoderIndex,
        input: &Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendInputRequest {
            input: input.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_input(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(())
    }
}

/// Proxies Tonic requests to `NetworkService` with actual handler implementation.
struct EncoderExternalTonicServiceProxy<S: EncoderExternalNetworkService> {
    /// Encoder context
    context: Arc<EncoderContext>,
    /// Encoder Network Service - this is typically the same even between different networking stacks. The trait
    /// makes testing easier.
    service: Arc<S>,
}

/// Implements a new method to create an encoder tonic service proxy
impl<S: EncoderExternalNetworkService> EncoderExternalTonicServiceProxy<S> {
    /// Creates the tonic service proxy using pre-established context and service
    const fn new(context: Arc<EncoderContext>, service: Arc<S>) -> Self {
        Self { context, service }
    }
}

/// Used to pack the networking index into each request. Using a new type
/// such that this can be extended in the future. May want to version this however?
#[derive(Clone, Debug)]
pub(crate) struct ExternalPeerInfo {
    /// networking index, verified using the TLS networking keypair
    pub(crate) networking_index: NetworkingIndex,
}

#[async_trait]
impl<S: EncoderExternalNetworkService> EncoderExternalTonicService
    for EncoderExternalTonicServiceProxy<S>
{
    async fn send_input(
        &self,
        request: Request<SendInputRequest>,
    ) -> Result<Response<SendInputResponse>, tonic::Status> {
        let Some(peer) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.networking_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let scores = request.into_inner().input;

        self.service
            .handle_send_input(peer, scores)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendInputResponse {}))
    }
}

/// Tonic specific manager type that contains a tonic specific client and
/// the oneshot tokio channel to trigger service shutdown.
pub struct EncoderExternalTonicManager {
    context: Arc<EncoderContext>,
    client: Arc<EncoderExternalTonicClient>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

/// Implementation of the encoder tonic manager that contains a new fn to create the type
// TODO: switch this to type state pattern
impl EncoderExternalTonicManager {
    /// Takes context, and network keypair and creates a new encoder tonic client
    pub fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self {
            context: context.clone(),
            client: Arc::new(EncoderExternalTonicClient::new(context, network_keypair)),
            shutdown_tx: None,
        }
    }
}

impl<S: EncoderExternalNetworkService> EncoderExternalNetworkManager<S>
    for EncoderExternalTonicManager
{
    type Client = EncoderExternalTonicClient;

    fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self::new(context, network_keypair)
    }

    fn client(&self) -> Arc<Self::Client> {
        self.client.clone()
    }

    /// if the network is running locally, then it uses the localhost address, otherwise
    /// it uses the zero address since it will be used in a hosted context where the service will
    /// be routed to using the IP address. The function starts a gRPC server taking a shutdown channel
    /// to allow the system to trigger shutdown from outside of the spawned tokio task.
    async fn start(&mut self, service: Arc<S>) {
        let encoder = self
            .context
            .encoder_committee
            .encoder(self.context.own_encoder_index);
        let own_encoder_index = self.context.own_encoder_index;
        let own_networking_index = self.context.own_network_index;
        // By default, bind to the unspecified address to allow the actual address to be assigned.
        // But bind to localhost if it is requested.
        let own_address = if encoder.address.is_localhost_ip() {
            encoder.address.clone()
        } else {
            encoder.address.with_zero_ip()
        };
        let own_address = to_socket_addr(&own_address).unwrap();
        let svc = EncoderExternalTonicServiceServer::new(EncoderExternalTonicServiceProxy::new(
            self.context.clone(),
            service,
        ));
        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        tokio::spawn(async move {
            // let leader_service = Server::builder().add_service(LeaderShardServiceServer::new(svc));

            let tower_layer = tower::ServiceBuilder::new()
                .layer(AddExtensionLayer::new(PeerInfo {
                    encoder_index: own_encoder_index,
                    networking_index: own_networking_index,
                }))
                .into_inner();

            Server::builder()
                .layer(tower_layer)
                .add_service(svc)
                .serve_with_shutdown(own_address, async {
                    rx.await.ok();
                })
                .await
                .unwrap();
        });
        info!("Binding tonic server to address {:?}", own_address);
    }

    async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendInputRequest {
    #[prost(bytes = "bytes", tag = "1")]
    input: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendInputResponse {}
