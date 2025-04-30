use std::{
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use fastcrypto::hash::HashFunction;
use quick_cache::sync::Cache;
use reqwest::Client;
use shared::{
    checksum::Checksum,
    crypto::{
        keys::{PeerKeyPair, PeerPublicKey},
        DefaultHashFunction,
    },
    metadata::{Metadata, MetadataAPI},
};
use soma_http::{PeerCertificates, ServerHandle};
use soma_tls::{
    create_rustls_client_config, public_key_from_certificate, AllowPublicKeys, TlsConnectionInfo,
};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio_util::io::ReaderStream;
use tracing::{info, warn};
use url::Url;

use crate::{
    error::{ObjectError, ObjectResult},
    parameters::Parameters,
};
use soma_network::{
    multiaddr::{to_host_port_str, to_socket_addr, Multiaddr},
    CERTIFICATE_NAME,
};

use super::{
    ObjectNetworkClient, ObjectNetworkManager, ObjectNetworkService, ObjectPath, ObjectStorage,
};

// TODO: move this to parameters
const BASE_LATENCY_SECS: u64 = 2;
const MIN_BYTES_PER_SEC: u64 = 50_000; // 50 KB/s

fn calculate_timeout(bytes: usize) -> Duration {
    let seconds = BASE_LATENCY_SECS + (bytes as u64 / MIN_BYTES_PER_SEC);
    Duration::from_secs(seconds)
}

pub(crate) struct ClientPool {
    clients: Cache<PeerPublicKey, Client>,
    parameters: Arc<Parameters>,
    own_key: PeerKeyPair,
}

impl ClientPool {
    pub(crate) fn new(own_key: PeerKeyPair, parameters: Arc<Parameters>) -> Self {
        Self {
            clients: Cache::new(parameters.http2.client_pool_capacity),
            parameters,
            own_key,
        }
    }
    pub(crate) async fn get_client(&self, peer: &PeerPublicKey) -> ObjectResult<Client> {
        if let Some(client) = self.clients.get(peer) {
            return Ok(client);
        }
        let buffer_size = self.parameters.http2.connection_buffer_size;

        let tls_config = create_rustls_client_config(
            peer.clone().into_inner(),
            CERTIFICATE_NAME.to_string(),
            Some(self.own_key.clone().private_key().into_inner()),
        );
        let client = reqwest::Client::builder()
            .http2_prior_knowledge()
            .use_preconfigured_tls(tls_config)
            .http2_initial_connection_window_size(Some(buffer_size as u32))
            .http2_initial_stream_window_size(Some(buffer_size as u32 / 2))
            .http2_keep_alive_while_idle(true)
            .http2_keep_alive_interval(self.parameters.http2.keepalive_interval)
            .http2_keep_alive_timeout(self.parameters.http2.keepalive_interval)
            .connect_timeout(self.parameters.http2.connect_timeout)
            .user_agent("SOMA (Object Client)")
            .build()
            .map_err(|e| ObjectError::ReqwestError(e.to_string()))?;

        self.clients.insert(peer.clone(), client.clone());
        Ok(client)
    }
}

pub struct ObjectHttpClient {
    clients: ClientPool,
}

impl ObjectHttpClient {
    pub fn new(own_key: PeerKeyPair, parameters: Arc<Parameters>) -> ObjectResult<Self> {
        Ok(Self {
            clients: ClientPool::new(own_key, parameters),
        })
    }

    pub(crate) async fn get_client(&self, peer: &PeerPublicKey) -> ObjectResult<Client> {
        self.clients.get_client(peer).await
    }
}

#[async_trait]
impl ObjectNetworkClient for ObjectHttpClient {
    async fn download_object<W>(
        &self,
        writer: &mut W,
        peer: &PeerPublicKey,
        address: &Multiaddr,
        metadata: &Metadata,
    ) -> ObjectResult<()>
    where
        W: AsyncWrite + Unpin + Send,
    {
        let address = to_host_port_str(address).map_err(|e| {
            ObjectError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        // warn!("checksum: {}", metadata.checksum());
        let address = format!("https://{address}/{}", metadata.checksum());
        // warn!("address: {}", address);
        let url = Url::from_str(&address).map_err(|e| ObjectError::UrlParseError(e.to_string()))?;

        let timeout = calculate_timeout(metadata.size());
        // warn!("timeout: {:?}", timeout);

        let mut response = self
            .get_client(peer)
            .await?
            .get(url.clone())
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?;

        // warn!("response: {:?}", response);

        if !response.status().is_success() {
            return Err(ObjectError::NetworkRequest(format!(
                "http status: {}",
                response.status().as_u16()
            )));
        }
        let mut hasher = DefaultHashFunction::new();
        let mut total_size: usize = 0;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?
        {
            hasher.update(&chunk);
            total_size += chunk.len();
            writer
                .write_all(&chunk)
                .await
                .map_err(|e| ObjectError::WriteError(e.to_string()))?;
        }

        writer
            .flush()
            .await
            .map_err(|e| ObjectError::WriteError(e.to_string()))?;

        writer
            .shutdown()
            .await
            .map_err(|e| ObjectError::WriteError(e.to_string()))?;

        let checksum = Checksum::new_from_hash(hasher.finalize().into());

        if total_size != metadata.size() {
            return Err(ObjectError::VerificationError(format!(
                "Size mismatch: expected {}, got {}",
                metadata.size(),
                total_size
            )));
        }

        // Verify checksum
        if checksum != metadata.checksum() {
            return Err(ObjectError::VerificationError(format!(
                "Checksum mismatch: expected {}, got {}",
                metadata.checksum(),
                checksum
            )));
        }
        return Ok(());
    }
}
#[derive(Clone)]
struct ObjectHttpServiceProxy<S: ObjectStorage> {
    service: ObjectNetworkService<S>,
}

impl<S: ObjectStorage + Clone> ObjectHttpServiceProxy<S> {
    const fn new(service: ObjectNetworkService<S>) -> Self {
        Self { service }
    }

    fn router(self) -> Router {
        Router::new()
            .route("/:path", get(Self::download_object))
            .with_state(self)
    }

    pub async fn download_object(
        Path(path): Path<String>,
        peer_certificates: axum::Extension<PeerCertificates>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let pk = public_key_from_certificate(peer_certificates.peer_certs().first().unwrap())
            .ok()
            .unwrap();
        let path = ObjectPath::new(path).map_err(|_| StatusCode::BAD_REQUEST)?;
        warn!("path: {:?}", path);
        // instead handle this as unauthorized!
        let peer = PeerPublicKey::new(pk.clone());
        warn!("peer public key: {:?}", peer);

        let reader = service
            .handle_download_object(&peer, &path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)?;
        // Convert BufReader<fs::File> to a Stream of Bytes
        let stream = ReaderStream::new(reader);

        // Wrap the stream in StreamBody
        let body = Body::from_stream(stream);

        // Build the response
        let response = Response::builder()
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(body)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        Ok(response)
    }
}

pub struct ObjectHttpManager {
    client: Arc<ObjectHttpClient>,
    own_key: PeerKeyPair,
    allower: AllowPublicKeys,
    parameters: Arc<Parameters>,
    server: Option<ServerHandle>,
}

impl<S: ObjectStorage + Clone> ObjectNetworkManager<S> for ObjectHttpManager {
    type Client = ObjectHttpClient;

    fn new(
        own_key: PeerKeyPair,
        parameters: Arc<Parameters>,
        allower: AllowPublicKeys,
    ) -> ObjectResult<Self> {
        Ok(Self {
            client: Arc::new(ObjectHttpClient::new(own_key.clone(), parameters.clone())?),
            own_key,
            allower,
            parameters,
            server: None,
        })
    }

    fn client(&self) -> Arc<Self::Client> {
        self.client.clone()
    }

    async fn start(&mut self, address: &Multiaddr, service: ObjectNetworkService<S>) {
        let config = &self.parameters.http2;
        let own_address = if address.is_localhost_ip() {
            address.clone()
        } else {
            address.with_zero_ip()
        };

        let own_address = to_socket_addr(&own_address).unwrap();

        let tls_server_config = soma_tls::create_rustls_server_config_with_client_verifier(
            self.own_key.clone().private_key().into_inner(),
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
                .serve(
                    own_address,
                    ObjectHttpServiceProxy::new(service.clone()).router(),
                ) {
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

        info!("Object server started at: {own_address}");
        self.server = Some(server);
    }

    async fn stop(&mut self) {
        if let Some(server) = self.server.take() {
            server.shutdown().await;
        }
    }
}

impl Drop for ObjectHttpManager {
    fn drop(&mut self) {
        if let Some(server) = self.server.as_ref() {
            server.trigger_shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeSet,
        net::{TcpListener, TcpStream},
        sync::Arc,
        thread::sleep,
        time::Duration,
    };

    use bytes::Bytes;
    use rand::{rngs::OsRng, RngCore};
    use shared::{checksum::Checksum, crypto::keys::PeerKeyPair, metadata::Metadata};
    use soma_network::multiaddr::Multiaddr;
    use soma_tls::AllowPublicKeys;
    use tracing::warn;

    use crate::{
        networking::{ObjectNetworkClient, ObjectNetworkManager, ObjectNetworkService},
        parameters::Parameters,
        storage::{memory::MemoryObjectStore, ObjectPath, ObjectStorage},
    };

    use super::{ObjectHttpClient, ObjectHttpManager};
    fn get_available_local_address() -> Multiaddr {
        let host = "127.0.0.1";
        let port = get_available_port(host);
        format!("/ip4/{}/udp/{}", host, port).parse().unwrap()
    }
    fn get_available_port(host: &str) -> u16 {
        const MAX_PORT_RETRIES: u32 = 1000;

        for _ in 0..MAX_PORT_RETRIES {
            if let Ok(port) = get_ephemeral_port(host) {
                return port;
            }
        }

        panic!("Error: could not find an available port");
    }

    fn get_ephemeral_port(host: &str) -> std::io::Result<u16> {
        // Request a random available port from the OS
        let listener = TcpListener::bind((host, 0))?;
        let addr = listener.local_addr()?;

        // Create and accept a connection (which we'll promptly drop) in order to force the port
        // into the TIME_WAIT state, ensuring that the port will be reserved from some limited
        // amount of time (roughly 60s on some Linux systems)
        let _sender = TcpStream::connect(addr)?;
        let _incoming = listener.accept()?;

        Ok(addr.port())
    }

    #[tokio::test]
    async fn object_http_success() {
        tracing_subscriber::fmt::init();
        let parameters = Arc::new(Parameters::default());
        let mut rng = rand::thread_rng();
        let mut buffer = vec![0u8; 1024 * 1024];
        OsRng.fill_bytes(&mut buffer);
        let random_bytes = Bytes::from(buffer);

        let checksum = Checksum::new_from_bytes(&random_bytes);
        let download_size = random_bytes.len();
        let object_path = ObjectPath::new(checksum.to_string()).unwrap();
        let metadata = Metadata::new_v1(None, None, checksum, download_size);

        let client_keypair = PeerKeyPair::generate(&mut rng);
        let server_keypair = PeerKeyPair::generate(&mut rng);

        let client_object_storage = Arc::new(MemoryObjectStore::new_for_test());
        let server_object_storage = Arc::new(MemoryObjectStore::new_for_test());

        server_object_storage
            .put_object(&object_path, random_bytes.clone())
            .await
            .unwrap();

        let server_object_network_service: ObjectNetworkService<MemoryObjectStore> =
            ObjectNetworkService::new(server_object_storage.clone());

        let allower = AllowPublicKeys::new(BTreeSet::from([client_keypair
            .public()
            .into_inner()
            .clone()]));
        // allower.update(BTreeSet::from([client_public_key.clone()]));
        let mut server_object_network_manager =
            <ObjectHttpManager as ObjectNetworkManager<MemoryObjectStore>>::new(
                server_keypair.clone(),
                parameters.clone(),
                allower,
            )
            .unwrap();

        let address = get_available_local_address();
        server_object_network_manager
            .start(&address, server_object_network_service)
            .await;

        let object_client = ObjectHttpClient::new(client_keypair, parameters).unwrap();

        let mut writer = client_object_storage
            .get_object_writer(&object_path)
            .await
            .unwrap();

        object_client
            .download_object(&mut writer, &server_keypair.public(), &address, &metadata)
            .await
            .unwrap();

        let x = client_object_storage
            .get_object(&object_path)
            .await
            .unwrap();

        assert_eq!(random_bytes, x)
    }
}
