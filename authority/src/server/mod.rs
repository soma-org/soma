use std::convert::Infallible;
use std::time::Duration;

use eyre::{Result, eyre};

use http::HeaderName;
use tokio_rustls::rustls::ServerConfig;
use tonic::body::Body;
use tonic::codegen::http::HeaderValue;
use tonic::{
    codegen::http::{Request, Response},
    server::NamedService,
};
use tower::{Service, ServiceBuilder, ServiceExt};
use tower_http::propagate_header::PropagateHeaderLayer;
use tower_http::set_header::SetRequestHeaderLayer;
use types::grpc_timeout::GrpcTimeout;
use types::{
    client::Config,
    multiaddr::{Multiaddr, Protocol},
};

pub const DEFAULT_GRPC_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
pub static GRPC_ENDPOINT_PATH_HEADER: HeaderName = HeaderName::from_static("grpc-path-req");

pub struct ServerBuilder {
    config: Config,
    router: tonic::service::Routes,
    health_reporter: tonic_health::server::HealthReporter,
}

impl ServerBuilder {
    pub fn from_config(config: &Config) -> Self {
        let (health_reporter, health_service) = tonic_health::server::health_reporter();
        let router = tonic::service::Routes::new(health_service);

        Self {
            config: config.to_owned(),
            router,
            health_reporter,
        }
    }

    pub fn health_reporter(&self) -> tonic_health::server::HealthReporter {
        self.health_reporter.clone()
    }

    /// Add a new service to this Server.
    pub fn add_service<S>(mut self, svc: S) -> Self
    where
        S: Service<Request<Body>, Response = Response<Body>, Error = Infallible>
            + NamedService
            + Clone
            + Send
            + Sync
            + 'static,
        S::Future: Send + 'static,
    {
        self.router = self.router.add_service(svc);
        self
    }

    pub async fn bind(self, addr: &Multiaddr, tls_config: Option<ServerConfig>) -> Result<Server> {
        let http_config = self
            .config
            .http_config()
            // Temporarily continue allowing clients to connection without TLS even when the server
            // is configured with a tls_config
            .allow_insecure(true);

        let request_timeout = self
            .config
            .request_timeout
            .unwrap_or(DEFAULT_GRPC_REQUEST_TIMEOUT);

        fn add_path_to_request_header<T>(request: &Request<T>) -> Option<HeaderValue> {
            let path = request.uri().path();
            Some(HeaderValue::from_str(path).unwrap())
        }

        let limiting_layers = ServiceBuilder::new()
            .option_layer(
                self.config
                    .load_shed
                    .unwrap_or_default()
                    .then_some(tower::load_shed::LoadShedLayer::new()),
            )
            .option_layer(
                self.config
                    .global_concurrency_limit
                    .map(tower::limit::GlobalConcurrencyLimitLayer::new),
            );
        let route_layers = ServiceBuilder::new()
            .map_request(|mut request: http::Request<_>| {
                if let Some(connect_info) = request.extensions().get::<soma_http::ConnectInfo>() {
                    let tonic_connect_info = tonic::transport::server::TcpConnectInfo {
                        local_addr: Some(connect_info.local_addr),
                        remote_addr: Some(connect_info.remote_addr),
                    };
                    request.extensions_mut().insert(tonic_connect_info);
                }
                request
            })
            .layer(SetRequestHeaderLayer::overriding(
                GRPC_ENDPOINT_PATH_HEADER.clone(),
                add_path_to_request_header,
            ))
            .layer(PropagateHeaderLayer::new(GRPC_ENDPOINT_PATH_HEADER.clone()))
            .layer_fn(move |service| GrpcTimeout::new(service, request_timeout));

        let mut builder = soma_http::Builder::new().config(http_config);

        if let Some(tls_config) = tls_config {
            builder = builder.tls_config(tls_config);
        }

        let server_handle = builder
            .serve(
                addr,
                limiting_layers.service(
                    self.router
                        .into_axum_router()
                        .layer(route_layers)
                        .into_service()
                        .map_err(tower::BoxError::from),
                ),
            )
            .map_err(|e| eyre!(e))?;

        let local_addr = update_tcp_port_in_multiaddr(addr, server_handle.local_addr().port());
        Ok(Server {
            server: server_handle,
            local_addr,
            health_reporter: self.health_reporter,
        })
    }
}

/// TLS server name to use for the public Soma validator interface.
pub const TLS_SERVER_NAME: &str = "soma";

pub struct Server {
    server: soma_http::ServerHandle,
    local_addr: Multiaddr,
    health_reporter: tonic_health::server::HealthReporter,
}

impl Server {
    pub async fn serve(self) -> Result<(), tonic::transport::Error> {
        self.server.wait_for_shutdown().await;
        Ok(())
    }

    pub fn local_addr(&self) -> &Multiaddr {
        &self.local_addr
    }

    pub fn health_reporter(&self) -> tonic_health::server::HealthReporter {
        self.health_reporter.clone()
    }

    pub fn handle(&self) -> &soma_http::ServerHandle {
        &self.server
    }
}

fn update_tcp_port_in_multiaddr(addr: &Multiaddr, port: u16) -> Multiaddr {
    addr.replace(1, |protocol| {
        if let Protocol::Tcp(_) = protocol {
            Some(Protocol::Tcp(port))
        } else {
            panic!("expected tcp protocol at index 1");
        }
    })
    .expect("tcp protocol at index 1")
}
