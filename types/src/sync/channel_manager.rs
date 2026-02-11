use axum::http;
use bytes::Bytes;
use fastcrypto::ed25519::Ed25519PublicKey;
use http::{Request, Response};
use http_body::Body as HttpBody;
use http_body_util::BodyExt as _;
use hyper_util::{rt::TokioIo, service::TowerToHyperService};
use parking_lot::RwLock;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;
use tokio::{
    pin,
    sync::oneshot,
    sync::{broadcast, mpsc},
    task::JoinSet,
};
use tokio_rustls::TlsAcceptor;
use tonic::{
    Status,
    codegen::{Service, StdError},
};
use tonic::{server::NamedService, transport::Server};
use tonic_rustls::Channel;
use tower::ServiceBuilder;
use tower_http::ServiceBuilderExt;
use tracing::{debug, error, info, trace, warn};

use tonic::body::Body as TonicBody;

use crate::{
    crypto::{NetworkKeyPair, NetworkPublicKey},
    error::{SomaError, SomaResult},
    multiaddr::Multiaddr,
    peer_id::PeerId,
    sync::{to_host_port_str, to_socket_addr},
    tls::{
        create_rustls_client_config, create_rustls_server_config, public_key_from_certificate,
        verifier::AllowAll,
    },
};

use super::{DisconnectReason, PeerEvent, active_peers::ActivePeers};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONNECTIONS_BACKLOG: u32 = 1024;

#[derive(Debug)]
pub enum ChannelManagerRequest {
    Connect { address: Multiaddr, peer_id: PeerId, response: oneshot::Sender<SomaResult<PeerId>> },
    Disconnect { peer_id: PeerId, response: oneshot::Sender<SomaResult<()>> },
    Shutdown { response: oneshot::Sender<()> },
}

pub struct ChannelManager<S> {
    own_address: Multiaddr,
    network_keypair: NetworkKeyPair,
    mailbox: mpsc::Receiver<ChannelManagerRequest>,
    active_peers: ActivePeers,
    connection_handlers: JoinSet<()>,
    server_handle: Option<tokio::task::JoinHandle<()>>,
    service: S,
}

impl<S> ChannelManager<S>
where
    S: Clone + Send + Sync + 'static,
    S: Service<Request<hyper::body::Incoming>, Response = Response<TonicBody>, Error = Infallible>,
    S::Future: Send + 'static,
    S: NamedService,
{
    pub fn new(
        own_address: Multiaddr,
        network_keypair: NetworkKeyPair,
        service: S,
        active_peers: ActivePeers,
    ) -> (Self, mpsc::Sender<ChannelManagerRequest>) {
        let (sender, receiver) = mpsc::channel(1000);
        (
            Self {
                own_address,
                network_keypair,
                mailbox: receiver,
                active_peers,
                connection_handlers: JoinSet::new(),
                server_handle: None,
                service,
            },
            sender,
        )
    }

    pub fn subscribe(&self) -> broadcast::Receiver<PeerEvent> {
        self.active_peers.subscribe()
    }

    pub async fn start(mut self) {
        info!("ChannelManager started");

        if let Err(e) = self.setup_server().await {
            error!("Failed to setup server: {}", e);
            return;
        }

        let mut shutdown_notifier = None;

        loop {
            tokio::select! {
                Some(request) = self.mailbox.recv() => {
                    match request {
                        ChannelManagerRequest::Connect { address, peer_id, response } => {
                            self.handle_connect_request(address, peer_id, response).await;
                        }
                        ChannelManagerRequest::Disconnect { peer_id, response } => {
                            self.handle_disconnect_request(peer_id, response).await;
                        }
                        ChannelManagerRequest::Shutdown { response } => {
                            shutdown_notifier = Some(response);
                            break;
                        }
                    }
                }

                Some(result) = self.connection_handlers.join_next() => {
                    match result {
                        Ok(()) => {
                            trace!("Connection handler completed successfully");
                        }
                        Err(e) => {
                            warn!("Connection handler failed: {}", e);
                        }
                    }
                }

                else => {
                    warn!("ChannelManager mailbox closed");
                    break;
                }
            }
        }

        self.shutdown().await;

        if let Some(sender) = shutdown_notifier {
            let _ = sender.send(());
        }

        info!("ChannelManager stopped");
    }

    async fn setup_server(&mut self) -> SomaResult<()> {
        info!("Starting GRPC service");

        let own_address = to_socket_addr(&self.own_address)?;

        let tls_server_config = create_rustls_server_config(
            AllowAll,
            certificate_server_name(),
            self.network_keypair.clone(),
        );
        let tls_acceptor = TlsAcceptor::from(Arc::new(tls_server_config));

        let deadline = Instant::now() + Duration::from_secs(20);
        let listener = loop {
            if Instant::now() > deadline {
                panic!("Failed to start server: timeout");
            }

            cfg_if::cfg_if!(
                if #[cfg(msim)] {
                    match tokio::net::TcpListener::bind(own_address).await {
                        Ok(listener) => {
                            info!("Successfully bound p2p tonic server to address {:?}", own_address);
                            break listener;
                        },
                        Err(e) => {
                            warn!("Error binding to {own_address}: {e:?}");
                            panic!();
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                    }
                } else {
                    info!("Binding tonic server to address {:?}", own_address);
                    let socket = create_socket(&own_address);
                    match socket.bind(own_address) {
                        Ok(_) => {
                            info!(
                                "Successfully bound tonic server to address {:?}",
                                own_address
                            )
                        }
                        Err(e) => {
                            warn!("Error binding to {own_address}: {e:?}");
                            tokio::time::sleep(Duration::from_secs(1)).await;
                            continue;
                        }
                    };

                    match socket.listen(MAX_CONNECTIONS_BACKLOG) {
                        Ok(listener) => break listener,
                        Err(e) => {
                            warn!("Error listening at {own_address}: {e:?}");
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                    }
                }
            );
        };
        info!("Server listening on {}", own_address);

        let http_builder = Arc::new(
            hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                .http2_only(),
        );

        let service = self.service.clone();
        let active_peers = self.active_peers.clone();

        let server_handle = tokio::spawn(async move {
            loop {
                let (tcp_stream, peer_addr) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(e) => {
                        warn!("Failed to accept connection: {}", e);
                        continue;
                    }
                };

                trace!("Accepted new connection from {}", peer_addr);

                let tls_acceptor = tls_acceptor.clone();
                let service = service.clone();
                let http = http_builder.clone();
                let active_peers = active_peers.clone();

                tokio::spawn(async move {
                    if let Err(e) = handle_connection(
                        tcp_stream,
                        peer_addr,
                        tls_acceptor,
                        service,
                        http,
                        active_peers,
                    )
                    .await
                    {
                        warn!("Connection handler error for {}: {}", peer_addr, e);
                    }
                });
            }
        });

        self.server_handle = Some(server_handle);

        Ok(())
    }

    fn add_peer(
        &mut self,
        peer_id: PeerId,
        address: Multiaddr,
        channel: Channel,
        public_key: NetworkPublicKey,
    ) -> SomaResult<bool> {
        self.active_peers.insert(peer_id, address.clone(), channel, public_key);
        Ok(true)
    }

    async fn handle_disconnect_request(
        &mut self,
        peer_id: PeerId,
        response: oneshot::Sender<SomaResult<()>>,
    ) {
        let result = if let Some(_channel) =
            self.active_peers.remove(&peer_id, DisconnectReason::RequestedDisconnect)
        {
            Ok(())
        } else {
            Err(SomaError::PeerNotFound(peer_id))
        };

        let _ = response.send(result);
    }

    async fn handle_connect_request(
        &mut self,
        address: Multiaddr,
        peer_id: PeerId,
        response: oneshot::Sender<SomaResult<PeerId>>,
    ) {
        if let Some(_state) = self.active_peers.get_state(&peer_id) {
            let _ = response.send(Ok(peer_id));
            return;
        }

        let result = self.dial_peer(address, peer_id).await;
        let _ = response.send(result);
    }

    async fn dial_peer(&mut self, address: Multiaddr, peer_id: PeerId) -> SomaResult<PeerId> {
        let address_str = to_host_port_str(&address).map_err(|e| {
            SomaError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        let address_str = format!("https://{address_str}");

        let target_pubkey: Ed25519PublicKey = peer_id.into();

        let client_tls_config = create_rustls_client_config(
            target_pubkey.clone(),
            certificate_server_name(),
            self.network_keypair.clone(),
        );

        let endpoint = tonic_rustls::Channel::from_shared(address_str.clone())
            .map_err(|e| {
                SomaError::NetworkConfig(format!("Invalid address {}: {}", address_str, e))
            })?
            .connect_timeout(CONNECT_TIMEOUT)
            .keep_alive_while_idle(true)
            .user_agent("soma-p2p")
            .map_err(|e| SomaError::NetworkConfig(format!("Failed to set user agent: {}", e)))?
            .tls_config(client_tls_config)
            .map_err(|e| SomaError::NetworkConfig(format!("Faileded to configure tls: {}", e)))?;

        let deadline = tokio::time::Instant::now() + CONNECT_TIMEOUT;
        let channel = loop {
            trace!("Connecting to endpoint at {address_str}");
            match endpoint.connect().await {
                Ok(channel) => break channel,
                Err(e) => {
                    warn!("Failed to connect to endpoint at {address_str}: {e:?}");
                    if tokio::time::Instant::now() >= deadline {
                        return Err(SomaError::NetworkClientConnection(format!(
                            "Timed out connecting to endpoint at {address_str}: {e:?}"
                        )));
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        };
        trace!("Connected to {address_str}");

        match self.add_peer(peer_id, address, channel, target_pubkey.into())? {
            true => {
                debug!("Successfully established new connection with peer {}", peer_id);
                Ok(peer_id)
            }
            false => {
                debug!("Connection was established but not needed (simultaneous dial resolution)");
                Ok(peer_id)
            }
        }
    }

    async fn shutdown(&mut self) {
        info!("Starting ChannelManager shutdown");

        if let Some(handle) = self.server_handle.take() {
            info!("Shutting down server accept loop");
            handle.abort();
            match handle.await {
                Ok(_) => debug!("Server accept loop shut down cleanly"),
                Err(e) => warn!("Server accept loop shutdown error: {}", e),
            }
        }

        let peers_to_disconnect: Vec<PeerId> = self.active_peers.peers();
        for peer_id in peers_to_disconnect {
            debug!("Closing connection to peer {}", peer_id);
            if let Some(_channel) = self.active_peers.remove(&peer_id, DisconnectReason::Shutdown) {
                debug!("Removed peer {} from active peers", peer_id);
            }
        }

        info!("Waiting for {} connection handlers to complete", self.connection_handlers.len());
        self.connection_handlers.shutdown().await;
        debug!("All connection handlers completed");

        debug_assert!(self.active_peers.is_empty(), "ActivePeers should be empty after shutdown");
        debug_assert!(
            self.connection_handlers.is_empty(),
            "All connection handlers should be completed"
        );

        info!("ChannelManager shutdown complete");
    }
}

async fn handle_connection<S>(
    tcp_stream: tokio::net::TcpStream,
    peer_addr: std::net::SocketAddr,
    tls_acceptor: TlsAcceptor,
    service: S,
    http: Arc<hyper_util::server::conn::auto::Builder<hyper_util::rt::TokioExecutor>>,
    active_peers: ActivePeers,
) -> SomaResult<()>
where
    S: Service<Request<hyper::body::Incoming>, Response = Response<TonicBody>, Error = Infallible>
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
{
    let tls_stream = tls_acceptor
        .accept(tcp_stream)
        .await
        .map_err(|e| SomaError::NetworkServerConnection(format!("TLS accept error: {}", e)))?;

    let (peer_id, _client_public_key) = extract_peer_info_from_tls(&tls_stream)?;

    // Add peer info to requests without changing the body type
    let svc = tower::ServiceBuilder::new()
        .map_request(move |mut request: http::Request<hyper::body::Incoming>| {
            request.extensions_mut().insert(PeerInfo { peer_id });
            request
        })
        .service(service);

    let hyper_service = hyper_util::service::TowerToHyperService::new(svc);

    let io = hyper_util::rt::TokioIo::new(tls_stream);
    let connection = http.serve_connection(io, hyper_service);

    let result = connection.await;
    active_peers.remove(&peer_id, DisconnectReason::ConnectionLost);
    match result {
        Ok(()) => {
            trace!("Connection closed for {}", peer_addr);
        }
        Err(e) => {
            return Err(SomaError::NetworkServerConnection(format!(
                "Connection error for {}: {}",
                peer_addr, e
            )));
        }
    }

    Ok(())
}

fn extract_peer_info_from_tls(
    tls_stream: &tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
) -> SomaResult<(PeerId, NetworkPublicKey)> {
    let certs = tls_stream
        .get_ref()
        .1
        .peer_certificates()
        .ok_or_else(|| SomaError::NetworkServerConnection("No peer certificate".into()))?;

    if certs.len() != 1 {
        return Err(SomaError::NetworkServerConnection(format!(
            "Expected 1 certificate, got {}",
            certs.len()
        )));
    }

    let certificate_public_key = public_key_from_certificate(&certs[0]).map_err(|e| {
        SomaError::NetworkServerConnection(format!(
            "Failed to extract public key from certificate: {}",
            e
        ))
    })?;

    let client_public_key = NetworkPublicKey::new(certificate_public_key);
    let peer_id = PeerId::from(&client_public_key);

    Ok((peer_id, client_public_key))
}

fn certificate_server_name() -> String {
    "p2p".to_string()
}

fn create_socket(address: &SocketAddr) -> tokio::net::TcpSocket {
    let socket = if address.is_ipv4() {
        tokio::net::TcpSocket::new_v4()
    } else if address.is_ipv6() {
        tokio::net::TcpSocket::new_v6()
    } else {
        panic!("Invalid own address: {address:?}");
    }
    .unwrap_or_else(|e| panic!("Cannot create TCP socket: {e:?}"));

    if let Err(e) = socket.set_reuseaddr(true) {
        info!("Failed to set SO_REUSEADDR: {e:?}");
    }
    socket
}

#[derive(Clone, Debug)]
pub struct PeerInfo {
    pub peer_id: PeerId,
}
