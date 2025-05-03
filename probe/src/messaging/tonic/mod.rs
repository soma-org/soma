mod generated {
    include!(concat!(env!("OUT_DIR"), "/soma.ProbeTonicService.rs"));
}
use super::{ProbeClient, ProbeManager, ProbeService};
use crate::error::{ProbeError, ProbeResult};
use crate::messaging::tonic::generated::probe_tonic_service_client::ProbeTonicServiceClient;
use crate::parameters::Parameters;
use crate::{ProbeInput, ProbeOutput};
use async_trait::async_trait;
use bytes::Bytes;
use generated::probe_tonic_service_server::{ProbeTonicService, ProbeTonicServiceServer};
use soma_http::ServerHandle;
use soma_network::multiaddr::{to_host_port_str, to_socket_addr, Multiaddr};
use std::{sync::Arc, time::Duration};
use tokio::time::Instant;
use tonic::codec::CompressionEncoding;
use tonic::{Request, Response};
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, TraceLayer};
use tracing::{debug, info, trace, warn};

pub(crate) type Channel = tower_http::trace::Trace<
    tonic::transport::Channel,
    tower_http::classify::SharedClassifier<tower_http::classify::GrpcErrorsAsFailures>,
>;

pub struct ProbeTonicClient {
    address: Multiaddr,
    parameters: Arc<Parameters>,
    channel: Channel,
}
impl ProbeTonicClient {
    pub async fn new(address: Multiaddr, parameters: Arc<Parameters>) -> ProbeResult<Self> {
        let config = parameters.tonic.clone();
        let address_string = to_host_port_str(&address).map_err(|e| {
            ProbeError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        let address_string = format!("https://{address_string}");
        let endpoint = tonic::transport::Channel::from_shared(address_string.clone())
            .map_err(|e| ProbeError::NetworkConfig(format!("Failed to create URI: {e}")))?
            .connect_timeout(config.connect_timeout)
            .initial_connection_window_size(Some(config.connection_buffer_size as u32))
            .initial_stream_window_size(Some(config.connection_buffer_size as u32 / 2))
            .keep_alive_while_idle(true)
            .keep_alive_timeout(config.keepalive_interval)
            .http2_keep_alive_interval(config.keepalive_interval)
            .user_agent("soma probe")
            .unwrap();

        let deadline = tokio::time::Instant::now() + config.connect_timeout;
        let channel = loop {
            trace!("Connecting to endpoint at {address_string}");
            match endpoint.connect().await {
                Ok(channel) => break channel,
                Err(e) => {
                    debug!("Failed to connect to endpoint at {address_string}: {e:?}");
                    if tokio::time::Instant::now() >= deadline {
                        return Err(ProbeError::NetworkClientConnection(format!(
                            "Timed out connecting to endpoint at {address_string}: {e:?}"
                        )));
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        };
        trace!("Connected to {address_string}");
        let channel = tower::ServiceBuilder::new()
            .layer(
                TraceLayer::new_for_grpc()
                    .make_span_with(DefaultMakeSpan::new().level(tracing::Level::TRACE))
                    .on_failure(DefaultOnFailure::new().level(tracing::Level::DEBUG)),
            )
            .service(channel);
        Ok(Self {
            address,
            parameters,
            channel,
        })
    }

    pub(crate) async fn get_client(&self) -> ProbeResult<ProbeTonicServiceClient<Channel>> {
        let config = self.parameters.tonic.clone();
        let client = ProbeTonicServiceClient::new(self.channel.clone())
            .max_encoding_message_size(config.message_size_limit)
            .max_decoding_message_size(config.message_size_limit)
            .send_compressed(CompressionEncoding::Zstd)
            .accept_compressed(CompressionEncoding::Zstd);
        Ok(client)
    }
}

#[async_trait]
impl ProbeClient for ProbeTonicClient {
    async fn probe(&self, probe_input: ProbeInput, timeout: Duration) -> ProbeResult<ProbeOutput> {
        let probe_input_bytes = Bytes::copy_from_slice(&bcs::to_bytes(&probe_input).unwrap());
        let mut request = Request::new(ProbeRequest {
            input: probe_input_bytes,
        });
        request.set_timeout(timeout);
        let probe_output_response = self
            .get_client()
            .await?
            .probe(request)
            .await
            .map_err(|e| ProbeError::NetworkRequest(format!("request failed: {e:?}")))?;

        let probe_output_bytes = probe_output_response.into_inner().output;

        let probe_output: ProbeOutput =
            bcs::from_bytes(&probe_output_bytes).map_err(ProbeError::MalformedType)?;
        Ok(probe_output)
    }
}

struct ProbeTonicServiceProxy<S: ProbeService> {
    service: Arc<S>,
}

impl<S: ProbeService> ProbeTonicServiceProxy<S> {
    const fn new(service: Arc<S>) -> Self {
        Self { service }
    }
}

#[async_trait]
impl<S: ProbeService> ProbeTonicService for ProbeTonicServiceProxy<S> {
    async fn probe(
        &self,
        request: Request<ProbeRequest>,
    ) -> Result<Response<ProbeResponse>, tonic::Status> {
        let input = request.into_inner().input;
        let output = self
            .service
            .handle_probe(input)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(ProbeResponse { output }))
    }
}

pub struct ProbeTonicManager {
    parameters: Arc<Parameters>,
    address: Multiaddr,
    server: Option<ServerHandle>,
}

impl ProbeTonicManager {
    /// Takes context, and network keypair and creates a new encoder tonic client
    pub fn new(parameters: Arc<Parameters>, address: Multiaddr) -> Self {
        Self {
            parameters,
            address,
            server: None,
        }
    }
}

impl<S: ProbeService> ProbeManager<S> for ProbeTonicManager {
    fn new(parameters: Arc<Parameters>, address: Multiaddr) -> Self {
        Self::new(parameters, address)
    }

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

        let service = ProbeTonicServiceProxy::new(service);

        let layers = tower::ServiceBuilder::new()
            .layer(
                TraceLayer::new_for_grpc()
                    .make_span_with(DefaultMakeSpan::new().level(tracing::Level::TRACE))
                    .on_failure(DefaultOnFailure::new().level(tracing::Level::DEBUG)),
            )
            .layer_fn(|service| soma_network::grpc_timeout::GrpcTimeout::new(service, None));
        let encoder_external_service_server = ProbeTonicServiceServer::new(service)
            .max_encoding_message_size(config.message_size_limit)
            .max_decoding_message_size(config.message_size_limit)
            .send_compressed(CompressionEncoding::Zstd)
            .accept_compressed(CompressionEncoding::Zstd);
        let encoder_external_service = tonic::service::Routes::new(encoder_external_service_server)
            .into_axum_router()
            .route_layer(layers);

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
                // .tls_config(tls_server_config.clone())
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

impl Drop for ProbeTonicManager {
    fn drop(&mut self) {
        if let Some(server) = self.server.as_ref() {
            server.trigger_shutdown();
        }
    }
}

#[derive(Clone, prost::Message)]
pub(crate) struct ProbeRequest {
    #[prost(bytes = "bytes", tag = "1")]
    input: Bytes,
}

#[derive(Clone, prost::Message)]
pub(crate) struct ProbeResponse {
    #[prost(bytes = "bytes", tag = "1")]
    output: Bytes,
}
