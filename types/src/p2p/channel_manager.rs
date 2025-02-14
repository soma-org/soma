use axum::{http, Router};
use bytes::Bytes;
use fastcrypto::ed25519::Ed25519PublicKey;
use futures_util::future;
use http::{Request, Response};
use http_body::Body;
use http_body_util::combinators::UnsyncBoxBody;
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
    codegen::{Service, StdError},
    Status,
};
use tonic::{server::NamedService, transport::Server};
use tower::ServiceBuilder;
use tower_http::ServiceBuilderExt;
use tracing::{debug, error, info, trace, warn};

type BoxBody = UnsyncBoxBody<Bytes, Status>;

use crate::{
    crypto::{NetworkKeyPair, NetworkPublicKey},
    error::{SomaError, SomaResult},
    multiaddr::Multiaddr,
    p2p::{to_host_port_str, to_socket_addr},
    peer_id::PeerId,
    tls::{
        create_rustls_client_config, create_rustls_server_config, public_key_from_certificate,
        verifier::AllowAll,
    },
};

use super::{active_peers::ActivePeers, DisconnectReason, PeerEvent};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONNECTIONS_BACKLOG: u32 = 1024;

#[derive(Debug)]
pub enum ChannelManagerRequest {
    Connect {
        address: Multiaddr,
        peer_id: PeerId,
        response: oneshot::Sender<SomaResult<PeerId>>,
    },
    Disconnect {
        peer_id: PeerId,
        response: oneshot::Sender<SomaResult<()>>,
    },
    Shutdown {
        response: oneshot::Sender<()>,
    },
}

type Channel = tonic::transport::Channel;

pub struct ChannelManager<S> {
    own_address: Multiaddr,
    network_keypair: NetworkKeyPair,

    // Mailbox for external requests
    mailbox: mpsc::Receiver<ChannelManagerRequest>,

    // Active peer connections and state
    active_peers: ActivePeers,

    // Server connection management
    connection_handlers: JoinSet<()>,
    server_handle: Option<tokio::task::JoinHandle<()>>,

    // Service factory for peer handlers
    service: S,
    // event_sender: broadcast::Sender<PeerEvent>,
}

impl<S> ChannelManager<S>
where
    S: Clone + Send + Sync + 'static,
    S: Service<Request<BoxBody>, Response = Response<BoxBody>, Error = Infallible>,
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
        // let (event_sender, _) = broadcast::channel(1000);
        (
            Self {
                own_address,
                network_keypair,
                mailbox: receiver,
                active_peers,
                connection_handlers: JoinSet::new(),
                server_handle: None,
                service,
                // event_sender,
            },
            sender,
        )
    }

    pub fn subscribe(&self) -> broadcast::Receiver<PeerEvent> {
        self.active_peers.subscribe()
    }

    pub async fn start(mut self) {
        info!("ChannelManager started");

        // Initialize server
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

        // Create server service with your GRPC service implementation
        let server = Server::builder()
            .initial_connection_window_size(64 << 20)
            .initial_stream_window_size(32 << 20)
            .add_service(self.service.clone())
            .into_router();
        // self.server_service = Some(Arc::new(server));

        // Setup TLS acceptor
        let tls_server_config = create_rustls_server_config(
            AllowAll::default(),
            certificate_server_name(),
            self.network_keypair.clone(),
        );
        let tls_acceptor = TlsAcceptor::from(Arc::new(tls_server_config));

        // Create and bind TCP listener
        let deadline = Instant::now() + Duration::from_secs(20);
        let listener = loop {
            if Instant::now() > deadline {
                panic!("Failed to start server: timeout");
            }

            cfg_if::cfg_if!(
                if #[cfg(msim)] {
                    // msim does not have a working stub for TcpSocket. So create TcpListener directly.
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
                    // TODO: Try creating an ephemeral port to test the highest allowed send and recv buffer sizes.
                    // Create TcpListener via TCP socket.
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

        // Create HTTP builder
        let http = Arc::new(
            hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                .http2_only(),
        );

        // Create a channel for sending new connection handler tasks
        let (task_tx, mut task_rx) = mpsc::channel(100);

        // Spawn server accept loop
        let server_handle = tokio::spawn({
            let tls_acceptor = tls_acceptor.clone();
            let server = server.clone();
            let http = http.clone();
            let active_peers = self.active_peers.clone();

            async move {
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
                    let server = server.clone();
                    let http = http.clone();
                    let active_peers = active_peers.clone();

                    let task = async move {
                        if let Err(e) = handle_connection(
                            tcp_stream,
                            peer_addr,
                            tls_acceptor,
                            server,
                            http,
                            active_peers,
                        )
                        .await
                        {
                            warn!("Connection handler error for {}: {}", peer_addr, e);
                        }
                    };

                    if let Err(e) = task_tx.send(task).await {
                        warn!("Failed to send connection handler task: {}", e);
                    }
                }
            }
        });

        // Spawn task to receive and spawn connection handlers
        tokio::spawn({
            let mut connection_handlers = std::mem::take(&mut self.connection_handlers);
            async move {
                while let Some(task) = task_rx.recv().await {
                    connection_handlers.spawn(task);
                }
            }
        });

        // Store server handle
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
        // Add the new connection
        self.active_peers
            .insert(peer_id, address.clone(), channel, public_key);

        // let _ = self
        //     .event_sender
        //     .send(PeerEvent::NewPeer { peer_id, address });

        Ok(true)
    }

    async fn handle_disconnect_request(
        &mut self,
        peer_id: PeerId,
        response: oneshot::Sender<SomaResult<()>>,
    ) {
        let result = if let Some(channel) = self
            .active_peers
            .remove(&peer_id, DisconnectReason::RequestedDisconnect)
        {
            // let _ = self.event_sender.send(PeerEvent::LostPeer {
            //     peer_id,
            //     reason: DisconnectReason::RequestedDisconnect,
            // });
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
        // Check if we already have an active connection for this peer
        if let Some(state) = self.active_peers.get_state(&peer_id) {
            let _ = response.send(Ok(peer_id));
            return;
        }

        // Attempt to dial peer
        let result = self.dial_peer(address, peer_id).await;
        let _ = response.send(result);
    }

    async fn dial_peer(&mut self, address: Multiaddr, peer_id: PeerId) -> SomaResult<PeerId> {
        // Convert address to host:port string
        let address_str = to_host_port_str(&address).map_err(|e| {
            SomaError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        let address_str = format!("https://{address_str}");

        // Get target public key if we have peer_id
        let target_pubkey: Ed25519PublicKey = peer_id.into();

        // Create endpoint configuration
        let endpoint = tonic::transport::Channel::from_shared(address_str.clone())
            .map_err(|e| {
                SomaError::NetworkConfig(format!("Invalid address {}: {}", address_str, e))
            })?
            .connect_timeout(CONNECT_TIMEOUT)
            .keep_alive_while_idle(true)
            .user_agent("soma-p2p")
            .map_err(|e| SomaError::NetworkConfig(format!("Failed to create endpoint: {}", e)))?;

        // Setup TLS config
        let client_tls_config = create_rustls_client_config(
            target_pubkey.clone(),
            certificate_server_name(),
            self.network_keypair.clone(),
        );

        let https_connector = hyper_rustls::HttpsConnectorBuilder::new()
            .with_tls_config(client_tls_config)
            .https_only()
            .enable_http2()
            .build();

        // Attempt connection with retries until timeout
        let deadline = tokio::time::Instant::now() + CONNECT_TIMEOUT;
        let channel = loop {
            trace!("Connecting to endpoint at {address_str}");
            match endpoint
                .connect_with_connector(https_connector.clone())
                .await
            {
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

        // Attempt to add the peer
        match self.add_peer(peer_id, address, channel, target_pubkey.into())? {
            true => {
                debug!(
                    "Successfully established new connection with peer {}",
                    peer_id
                );
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

        // Stop accepting new connections by shutting down the server
        if let Some(handle) = self.server_handle.take() {
            info!("Shutting down server accept loop");
            handle.abort();
            match handle.await {
                Ok(_) => debug!("Server accept loop shut down cleanly"),
                Err(e) => warn!("Server accept loop shutdown error: {}", e),
            }
        }

        // Close all active peer connections
        let peers_to_disconnect: Vec<PeerId> = self.active_peers.peers();
        for peer_id in peers_to_disconnect {
            debug!("Closing connection to peer {}", peer_id);
            if let Some(channel) = self
                .active_peers
                .remove(&peer_id, DisconnectReason::Shutdown)
            {
                // let _ = self.event_sender.send(PeerEvent::LostPeer {
                //     peer_id,
                //     reason: DisconnectReason::Shutdown,
                // });
                debug!("Removed peer {} from active peers", peer_id);
            }
        }

        // Wait for all connection handlers to complete
        info!(
            "Waiting for {} connection handlers to complete",
            self.connection_handlers.len()
        );
        self.connection_handlers.shutdown().await;
        debug!("All connection handlers completed");

        // Verify cleanup
        debug_assert!(
            self.active_peers.is_empty(),
            "ActivePeers should be empty after shutdown"
        );
        debug_assert!(
            self.connection_handlers.is_empty(),
            "All connection handlers should be completed"
        );

        info!("ChannelManager shutdown complete");
    }
}

async fn handle_connection(
    tcp_stream: tokio::net::TcpStream,
    peer_addr: std::net::SocketAddr,
    tls_acceptor: TlsAcceptor,
    service: Router,
    http: Arc<hyper_util::server::conn::auto::Builder<hyper_util::rt::TokioExecutor>>,
    active_peers: ActivePeers,
) -> SomaResult<()> {
    // Accept TLS connection
    let tls_stream = tls_acceptor
        .accept(tcp_stream)
        .await
        .map_err(|e| SomaError::NetworkServerConnection(format!("TLS accept error: {}", e)))?;

    // Extract peer certificate and public key
    let (peer_id, client_public_key) = extract_peer_info_from_tls(&tls_stream)?;
    // TODO: check that client_public_key is in the list of allowed peers

    // TODO: Dial peer back here to get a channel and add to active peers

    // Setup service stack
    let svc = ServiceBuilder::new()
        .add_extension(PeerInfo { peer_id })
        .service(service.clone());

    pin! {
        let connection = http.serve_connection(TokioIo::new(tls_stream), TowerToHyperService::new(svc));
    }
    let mut has_shutdown = false;
    loop {
        tokio::select! {
            result = &mut connection => {
                // Remove peer - ActivePeers will handle the event emission internally
                active_peers.remove(&peer_id, DisconnectReason::ConnectionLost);
                match result {
                    Ok(()) => {
                        trace!("Connection closed for {}", peer_addr);
                        break;
                    }
                    Err(e) => {
                        return Err(SomaError::NetworkServerConnection(
                            format!("Connection error for {}: {}", peer_addr, e)
                        ));
                    }
                }
            }
            // Optional: Handle graceful shutdown
            // _ = shutdown_signal.recv() => {
            //     if !has_shutdown {
            //         connection.graceful_shutdown();
            //         has_shutdown = true;
            //     }
            // }
        }
    }

    Ok(())
}

// Helper function for extracting peer info from TLS stream
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
    let peer_id = PeerId::from(&client_public_key); // Assuming you have this conversion

    Ok((peer_id, client_public_key))
}

fn certificate_server_name() -> String {
    format!("p2p")
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
    // if let Err(e) = socket.set_nodelay(true) {
    //     info!("Failed to set TCP_NODELAY: {e:?}");
    // }
    if let Err(e) = socket.set_reuseaddr(true) {
        info!("Failed to set SO_REUSEADDR: {e:?}");
    }
    socket
}

#[derive(Clone, Debug)]
pub struct PeerInfo {
    pub peer_id: PeerId,
}
