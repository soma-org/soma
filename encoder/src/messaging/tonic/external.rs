use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::{EncoderPublicKey, PeerKeyPair, PeerPublicKey},
    signed::Signed,
    verified::Verified,
};
use soma_http::ServerHandle;
use soma_network::{
    multiaddr::{to_socket_addr, Multiaddr},
    CERTIFICATE_NAME,
};
use soma_tls::AllowPublicKeys;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tonic::{codec::CompressionEncoding, Request, Response};
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, TraceLayer};

use crate::{
    error::{ShardError, ShardResult},
    messaging::{
        tonic::generated::encoder_external_tonic_service_server::EncoderExternalTonicServiceServer,
        EncoderExternalNetworkClient, EncoderExternalNetworkManager, EncoderExternalNetworkService,
    },
    types::{parameters::Parameters, shard_input::ShardInput},
};
use tracing::{info, trace, warn};

use super::{
    channel_pool::{Channel, ChannelPool},
    generated::{
        encoder_external_tonic_service_client::EncoderExternalTonicServiceClient,
        encoder_external_tonic_service_server::EncoderExternalTonicService,
    },
    NetworkingInfo,
};

// Implements Tonic RPC client for Encoders.
pub struct EncoderExternalTonicClient {
    networking_info: NetworkingInfo,
    own_peer_keypair: PeerKeyPair,
    parameters: Arc<Parameters>,
    channel_pool: Arc<ChannelPool>,
}
impl EncoderExternalTonicClient {
    /// Creates a new encoder tonic client and establishes an arc'd channel pool
    pub fn new(
        networking_info: NetworkingInfo,
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
    pub async fn get_client(
        &self,
        encoder: &EncoderPublicKey,
        timeout: Duration,
    ) -> ShardResult<EncoderExternalTonicServiceClient<Channel>> {
        let config = &self.parameters.tonic;
        if let Some((address, peer_public_key)) = self.networking_info.lookup(encoder) {
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
            let mut client = EncoderExternalTonicServiceClient::new(channel)
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
impl EncoderExternalNetworkClient for EncoderExternalTonicClient {
    async fn send_input(
        &self,
        encoder: &EncoderPublicKey,
        input: &Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendInputRequest {
            input: input.bytes(),
        });
        request.set_timeout(timeout);
        self.get_client(encoder, timeout)
            .await?
            .send_input(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("request failed: {e:?}")))?;
        Ok(())
    }
}

/// Proxies Tonic requests to `NetworkService` with actual handler implementation.
struct EncoderExternalTonicServiceProxy<S: EncoderExternalNetworkService> {
    /// Encoder Network Service - this is typically the same even between different networking stacks. The trait
    /// makes testing easier.
    service: Arc<S>,
}
/// Implements a new method to create an encoder tonic service proxy
impl<S: EncoderExternalNetworkService> EncoderExternalTonicServiceProxy<S> {
    /// Creates the tonic service proxy using pre-established context and service
    const fn new(service: Arc<S>) -> Self {
        Self { service }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PeerInfo {
    pub(crate) peer: PeerPublicKey,
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
            .map(|p| p.peer.clone())
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let input = request.into_inner().input;

        self.service
            .handle_send_input(&peer, input)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendInputResponse {}))
    }
}

/// Tonic specific manager type that contains a tonic specific client and
/// the oneshot tokio channel to trigger service shutdown.
pub struct EncoderExternalTonicManager {
    parameters: Arc<Parameters>,
    peer_keypair: PeerKeyPair,
    address: Multiaddr,
    allower: AllowPublicKeys,
    server: Option<ServerHandle>,
}

/// Implementation of the encoder tonic manager that contains a new fn to create the type
// TODO: switch this to type state pattern
impl EncoderExternalTonicManager {
    /// Takes context, and network keypair and creates a new encoder tonic client
    pub fn new(
        parameters: Arc<Parameters>,
        peer_keypair: PeerKeyPair,
        address: Multiaddr,
        allower: AllowPublicKeys,
    ) -> Self {
        Self {
            parameters,
            peer_keypair,
            address,
            allower,
            server: None,
        }
    }
}

impl<S: EncoderExternalNetworkService> EncoderExternalNetworkManager<S>
    for EncoderExternalTonicManager
{
    fn new(
        parameters: Arc<Parameters>,
        peer_keypair: PeerKeyPair,
        address: Multiaddr,
        allower: AllowPublicKeys,
    ) -> Self {
        Self::new(parameters, peer_keypair, address, allower)
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

        let service = EncoderExternalTonicServiceProxy::new(service);

        let layers = tower::ServiceBuilder::new()
            // Add a layer to extract a peer's PeerInfo from their TLS certs
            .map_request(move |mut request: http::Request<_>| {
                if let Some(peer_certificates) =
                    request.extensions().get::<soma_http::PeerCertificates>()
                {
                    if let Some(peer_info) = peer_info_from_certs(peer_certificates) {
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
            .layer_fn(|service| soma_network::grpc_timeout::GrpcTimeout::new(service, None));
        let encoder_external_service_server = EncoderExternalTonicServiceServer::new(service)
            .max_encoding_message_size(config.message_size_limit)
            .max_decoding_message_size(config.message_size_limit)
            .send_compressed(CompressionEncoding::Zstd)
            .accept_compressed(CompressionEncoding::Zstd);
        let encoder_external_service = tonic::service::Routes::new(encoder_external_service_server)
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

        let deadline = Instant::now() + Duration::from_secs(20);
        let server = loop {
            match soma_http::Builder::new()
                .config(http_config.clone())
                .tls_config(tls_server_config.clone())
                .serve(own_address, encoder_external_service.clone())
            {
                Ok(server) => break server,
                Err(err) => {
                    warn!("Error starting external encoder server: {err:?}");
                    if Instant::now() > deadline {
                        panic!("Failed to start external encoder server within required deadline");
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

impl Drop for EncoderExternalTonicManager {
    fn drop(&mut self) {
        if let Some(server) = self.server.as_ref() {
            server.trigger_shutdown();
        }
    }
}

fn peer_info_from_certs(peer_certificates: &soma_http::PeerCertificates) -> Option<PeerInfo> {
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
    let peer = PeerPublicKey::new(public_key);
    Some(PeerInfo { peer })
}

#[derive(Clone, prost::Message)]
pub struct SendInputRequest {
    #[prost(bytes = "bytes", tag = "1")]
    pub input: Bytes,
}

#[derive(Clone, prost::Message)]
pub struct SendInputResponse {}
