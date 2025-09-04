use crate::messaging::tonic::generated::encoder_internal_tonic_service_client::EncoderInternalTonicServiceClient;
use crate::types::score_vote::ScoreVote;
use crate::{
    messaging::{
        tonic::generated::encoder_internal_tonic_service_server::{
            EncoderInternalTonicService, EncoderInternalTonicServiceServer,
        },
        EncoderInternalNetworkClient, EncoderInternalNetworkManager, EncoderInternalNetworkService,
        EncoderPublicKey,
    },
    types::{commit::Commit, commit_votes::CommitVotes, reveal::Reveal},
};
use async_trait::async_trait;
use axum::http;
use bytes::Bytes;
use soma_http::ServerHandle;
use soma_tls::AllowPublicKeys;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tonic::{codec::CompressionEncoding, Request, Response};
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, TraceLayer};
use tracing::{error, info, trace, warn};
use types::error::{ShardError, ShardResult};
use types::parameters::Parameters;
use types::shard_crypto::{
    keys::{PeerKeyPair, PeerPublicKey},
    verified::Verified,
};
use types::shard_networking::EncoderNetworkingInfo;
use types::{multiaddr::Multiaddr, p2p::to_socket_addr, shard_networking::CERTIFICATE_NAME};

use types::shard_networking::channel_pool::{Channel, ChannelPool};

// Implements Tonic RPC client for Encoders.
pub(crate) struct EncoderInternalTonicClient {
    networking_info: EncoderNetworkingInfo,
    own_peer_keypair: PeerKeyPair,
    parameters: Arc<Parameters>,
    channel_pool: Arc<ChannelPool>,
}

/// Implments the core functionality of the encoder tonic client
impl EncoderInternalTonicClient {
    /// Creates a new encoder tonic client and establishes an arc'd channel pool
    pub(crate) fn new(
        networking_info: EncoderNetworkingInfo,
        own_peer_keypair: PeerKeyPair,
        parameters: Arc<Parameters>,
        capacity: usize,
    ) -> Self {
        Self {
            networking_info,
            own_peer_keypair,
            parameters,
            channel_pool: Arc::new(ChannelPool::new(capacity)),
        }
    }

    /// returns an encoder client
    // TODO: re-introduce configuring limits to the client for safety
    async fn get_client(
        &self,
        encoder: &EncoderPublicKey,
        timeout: Duration,
    ) -> ShardResult<EncoderInternalTonicServiceClient<Channel>> {
        let config = &self.parameters.tonic;
        if let Some((peer_public_key, address)) = self.networking_info.encoder_to_tls(encoder) {
            let channel = self
                .channel_pool
                .get_channel(
                    &address,
                    peer_public_key,
                    &self.parameters.tonic,
                    self.own_peer_keypair.clone(),
                    timeout,
                )
                .await?;
            let mut client = EncoderInternalTonicServiceClient::new(channel)
                .max_encoding_message_size(config.message_size_limit)
                .max_decoding_message_size(config.message_size_limit);

            client = client
                .send_compressed(CompressionEncoding::Zstd)
                .accept_compressed(CompressionEncoding::Zstd);
            Ok(client)
        } else {
            Err(ShardError::NetworkClientConnection(
                "failed to get networking info for peer".to_string(),
            ))
        }
    }
}

#[async_trait]
impl EncoderInternalNetworkClient for EncoderInternalTonicClient {
    async fn send_commit(
        &self,
        encoder: &EncoderPublicKey,
        commit: &Verified<Commit>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendCommitRequest {
            commit: commit.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(encoder, timeout)
            .await?
            .send_commit(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        info!("Sent commit");
        Ok(())
    }

    async fn send_commit_votes(
        &self,
        encoder: &EncoderPublicKey,
        votes: &Verified<CommitVotes>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendCommitVotesRequest {
            votes: votes.bytes(),
        });
        request.set_timeout(timeout);
        info!("Sending commit votes");
        self.get_client(encoder, timeout)
            .await?
            .send_commit_votes(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        info!("Sent commit votes");
        Ok(())
    }

    async fn send_reveal(
        &self,
        encoder: &EncoderPublicKey,
        reveal: &Verified<Reveal>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendRevealRequest {
            reveal: reveal.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(encoder, timeout)
            .await?
            .send_reveal(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(())
    }
    async fn send_score_vote(
        &self,
        encoder: &EncoderPublicKey,
        score_vote: &Verified<ScoreVote>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendScoreVoteRequest {
            score_vote: score_vote.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(encoder, timeout)
            .await?
            .send_score_vote(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(())
    }
}

/// Proxies Tonic requests to `NetworkService` with actual handler implementation.
struct EncoderInternalTonicServiceProxy<S: EncoderInternalNetworkService> {
    /// Encoder Network Service - this is typically the same even between different networking stacks. The trait
    /// makes testing easier.
    service: Arc<S>,
}

/// Implements a new method to create an encoder tonic service proxy
impl<S: EncoderInternalNetworkService> EncoderInternalTonicServiceProxy<S> {
    /// Creates the tonic service proxy using pre-established context and service
    const fn new(service: Arc<S>) -> Self {
        Self { service }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EncoderInfo {
    pub(crate) peer: EncoderPublicKey,
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
            .get::<EncoderInfo>()
            .map(|p| p.peer.clone())
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let commit = request.into_inner().commit;

        self.service
            .handle_send_commit(&peer, commit)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendCommitResponse {}))
    }
    async fn send_commit_votes(
        &self,
        request: Request<SendCommitVotesRequest>,
    ) -> Result<Response<SendCommitVotesResponse>, tonic::Status> {
        let Some(peer) = request
            .extensions()
            .get::<EncoderInfo>()
            .map(|p| p.peer.clone())
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let votes = request.into_inner().votes;

        self.service
            .handle_send_commit_votes(&peer, votes)
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
            .get::<EncoderInfo>()
            .map(|p| p.peer.clone())
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let reveal = request.into_inner().reveal;

        self.service
            .handle_send_reveal(&peer, reveal)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendRevealResponse {}))
    }
    async fn send_score_vote(
        &self,
        request: Request<SendScoreVoteRequest>,
    ) -> Result<Response<SendScoreVoteResponse>, tonic::Status> {
        let Some(peer) = request
            .extensions()
            .get::<EncoderInfo>()
            .map(|p| p.peer.clone())
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let score_vote = request.into_inner().score_vote;

        self.service
            .handle_send_score_vote(&peer, score_vote)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendScoreVoteResponse {}))
    }
}

/// Tonic specific manager type that contains a tonic specific client and
/// the oneshot tokio channel to trigger service shutdown.
pub struct EncoderInternalTonicManager {
    parameters: Arc<Parameters>,
    peer_keypair: PeerKeyPair,
    address: Multiaddr,
    allower: AllowPublicKeys,
    networking_info: EncoderNetworkingInfo,
    client: Arc<EncoderInternalTonicClient>,
    server: Option<ServerHandle>,
}

/// Implementation of the encoder tonic manager that contains a new fn to create the type
// TODO: switch this to type state pattern
impl EncoderInternalTonicManager {
    /// Takes context, and network keypair and creates a new encoder tonic client
    pub fn new(
        networking_info: EncoderNetworkingInfo,
        parameters: Arc<Parameters>,
        peer_keypair: PeerKeyPair,
        address: Multiaddr,
        allower: AllowPublicKeys,
    ) -> Self {
        let channel_pool_capacaity = parameters.tonic.channel_pool_capacity;
        Self {
            parameters: parameters.clone(),
            peer_keypair: peer_keypair.clone(),
            address,
            allower,
            networking_info: networking_info.clone(),
            client: Arc::new(EncoderInternalTonicClient::new(
                networking_info,
                peer_keypair,
                parameters,
                channel_pool_capacaity,
            )),
            server: None,
        }
    }
}

impl<S: EncoderInternalNetworkService> EncoderInternalNetworkManager<S>
    for EncoderInternalTonicManager
{
    type Client = EncoderInternalTonicClient;

    fn new(
        networking_info: EncoderNetworkingInfo,
        parameters: Arc<Parameters>,
        peer_keypair: PeerKeyPair,
        address: Multiaddr,
        allower: AllowPublicKeys,
    ) -> Self {
        Self::new(networking_info, parameters, peer_keypair, address, allower)
    }

    fn client(&self) -> Arc<Self::Client> {
        self.client.clone()
    }

    /// if the network is running locally, then it uses the localhost address, otherwise
    /// it uses the zero address since it will be used in a hosted context where the service will
    /// be routed to using the IP address. The function starts a gRPC server taking a shutdown channel
    /// to allow the system to trigger shutdown from outside of the spawned tokio task.
    async fn start(&mut self, service: Arc<S>) {
        // By default, bind to the unspecified address to allow the actual address to be assigned.
        // But bind to localhost if it is requested.
        let own_address = if self.address.is_localhost_ip() {
            self.address.clone()
        } else {
            self.address.with_zero_ip()
        };
        let own_address = to_socket_addr(&own_address).unwrap();

        let config = &self.parameters.tonic;

        let service = EncoderInternalTonicServiceProxy::new(service);

        let networking_info = self.networking_info.clone();

        let layers = tower::ServiceBuilder::new()
            // Add a layer to extract a peer's PeerInfo from their TLS certs
            .map_request(move |mut request: http::Request<_>| {
                if let Some(peer_certificates) =
                    request.extensions().get::<soma_http::PeerCertificates>()
                {
                    if let Some(peer_info) =
                        encoder_info_from_certs(&networking_info, peer_certificates)
                    {
                        request.extensions_mut().insert(peer_info);
                    }
                }
                request
            })
            .layer(
                TraceLayer::new_for_grpc()
                    .make_span_with(DefaultMakeSpan::new().level(tracing::Level::TRACE))
                    .on_failure(DefaultOnFailure::new().level(tracing::Level::DEBUG)),
            )
            .layer_fn(|service| {
                types::shard_networking::grpc_timeout::GrpcTimeout::new(service, None)
            });

        let encoder_internal_service_server = EncoderInternalTonicServiceServer::new(service)
            .max_encoding_message_size(config.message_size_limit)
            .max_decoding_message_size(config.message_size_limit)
            .send_compressed(CompressionEncoding::Zstd)
            .accept_compressed(CompressionEncoding::Zstd);

        let encoder_internal_service = tonic::service::Routes::new(encoder_internal_service_server)
            .into_axum_router()
            .route_layer(layers);

        let tls_server_config = soma_tls::create_rustls_server_config_with_client_verifier(
            self.peer_keypair.clone().private_key().into_inner(),
            CERTIFICATE_NAME.to_string(),
            self.allower.clone(),
        );

        let http_config = soma_http::Config::default()
            .tcp_nodelay(true)
            .initial_connection_window_size(64 << 20)
            .initial_stream_window_size(32 << 20)
            .http2_keepalive_interval(Some(config.keepalive_interval))
            .http2_keepalive_timeout(Some(config.keepalive_interval))
            .accept_http1(false);

        // Create server
        //
        // During simtest crash/restart tests there may be an older instance of consensus running
        // that is bound to the TCP port of `own_address` that hasn't finished relinquishing
        // control of the port yet. So instead of crashing when the address is inuse, we will retry
        // for a short/reasonable period of time before giving up.
        let deadline = Instant::now() + Duration::from_secs(20);
        let server = loop {
            match soma_http::Builder::new()
                .config(http_config.clone())
                .tls_config(tls_server_config.clone())
                .serve(own_address, encoder_internal_service.clone())
            {
                Ok(server) => break server,
                Err(err) => {
                    warn!("Error starting internal encoder server: {err:?}");
                    if Instant::now() > deadline {
                        panic!("Failed to start internal encoder server within required deadline");
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        };

        info!("Server started at: {own_address}");
        self.server = Some(server);
    }

    async fn stop(&mut self) {
        if let Some(server) = self.server.take() {
            server.shutdown().await;
        }
    }
}

impl Drop for EncoderInternalTonicManager {
    fn drop(&mut self) {
        if let Some(server) = self.server.as_ref() {
            server.trigger_shutdown();
        }
    }
}

fn encoder_info_from_certs(
    networking_info: &EncoderNetworkingInfo,
    peer_certificates: &soma_http::PeerCertificates,
) -> Option<EncoderInfo> {
    let certs = peer_certificates.peer_certs();

    if certs.len() != 1 {
        trace!(
            "Unexpected number of certificates from TLS stream: {}",
            certs.len()
        );
        return None;
    }
    trace!("Received {} certificates", certs.len());
    let public_key = soma_tls::public_key_from_certificate(&certs[0])
        .map_err(|e| {
            trace!("Failed to extract public key from certificate: {e:?}");
            e
        })
        .ok()?;
    let client_public_key = PeerPublicKey::new(public_key);
    let Some(peer) = networking_info.tls_to_encoder(&client_public_key) else {
        error!("Failed to find the authority with public key {client_public_key:?}");
        return None;
    };
    Some(EncoderInfo { peer })
}

// ////////////////////////////////////////////////////////////////////

#[derive(Clone, prost::Message)]
pub(crate) struct SendCommitRequest {
    #[prost(bytes = "bytes", tag = "1")]
    commit: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendCommitResponse {}

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
pub(crate) struct SendScoreVoteRequest {
    #[prost(bytes = "bytes", tag = "1")]
    score_vote: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct SendScoreVoteResponse {}
