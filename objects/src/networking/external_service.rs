use std::{
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use quick_cache::sync::Cache;
use reqwest::Client;
use soma_http::{PeerCertificates, ServerHandle};
use soma_tls::{create_rustls_client_config, public_key_from_certificate, AllowPublicKeys};
use tracing::{info, warn};
use types::{
    checksum::Checksum,
    committee::{AuthorityIndex, Epoch},
    consensus::block::Round,
    crypto::{NetworkKeyPair, NetworkPublicKey},
    metadata::ObjectPath,
    parameters::HttpParameters,
    shard::Shard,
    shard_crypto::digest::Digest,
};
use types::{
    error::{ObjectError, ObjectResult},
    metadata::SignedParams,
};
use types::{
    metadata::{DownloadableMetadata, DownloadableMetadataAPI},
    shard_networking::CERTIFICATE_NAME,
};
use types::{multiaddr::Multiaddr, p2p::to_socket_addr};

use crate::networking::{ClientPool, ObjectService, ObjectServiceManager};

use super::{ObjectClient, ObjectStorage};

pub struct ExternalClientPool {
    clients: Cache<NetworkPublicKey, Client>,
    parameters: Arc<HttpParameters>,
    own_key: NetworkKeyPair,
}

impl ExternalClientPool {
    pub(crate) fn new(own_key: NetworkKeyPair, parameters: Arc<HttpParameters>) -> Self {
        Self {
            clients: Cache::new(parameters.client_pool_capacity),
            parameters,
            own_key,
        }
    }
}

#[async_trait]
impl ClientPool for ExternalClientPool {
    async fn get_client(
        &self,
        downloadable_metadata: &DownloadableMetadata,
    ) -> ObjectResult<Client> {
        if let Some(peer) = downloadable_metadata.peer() {
            if let Some(client) = self.clients.get(&peer) {
                return Ok(client);
            }
            let buffer_size = self.parameters.connection_buffer_size;

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
                .http2_keep_alive_interval(self.parameters.keepalive_interval)
                .http2_keep_alive_timeout(self.parameters.keepalive_interval)
                .connect_timeout(self.parameters.connect_timeout)
                .user_agent("SOMA (Object Client)")
                .build()
                .map_err(|e| ObjectError::ReqwestError(e.to_string()))?;

            self.clients.insert(peer.clone(), client.clone());
            Ok(client)
        } else {
            Err(ObjectError::NetworkRequest(
                "A peer is needed for mTLS".to_string(),
            ))
        }
    }

    fn calculate_timeout(&self, num_bytes: u64) -> Duration {
        let nanos = self
            .parameters
            .nanoseconds_per_byte
            .saturating_mul(num_bytes);
        self.parameters.connect_timeout + Duration::from_nanos(nanos)
    }
}

#[derive(Clone)]
struct ExternalObjectService<S: ObjectStorage> {
    service: ObjectService<S>,
}

impl<S: ObjectStorage + Clone> ExternalObjectService<S> {
    const fn new(service: ObjectService<S>) -> Self {
        Self { service }
    }

    fn router(self) -> Router {
        Router::new()
            .route(
                "/epochs/{epoch}/shards/{shard}/embeddings/{checksum}",
                get(Self::embeddings),
            )
            .route("/epochs/{epoch}/probes/{checksum}", get(Self::probes))
            .route("/epochs/{epoch}/inputs/{checksum}", get(Self::inputs))
            .route(
                "/epochs/{epoch}/rounds/{round}/authorities/{authority_index}/blocks/{checksum}",
                get(Self::blocks),
            )
            .with_state(self)
    }

    pub async fn embeddings(
        Path((epoch, shard, checksum)): Path<(Epoch, Digest<Shard>, Checksum)>,
        Query(params): Query<SignedParams>,
        peer_certificates: axum::Extension<PeerCertificates>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Embeddings(epoch, shard, checksum);
        Self::signed_download(service, path, params, peer_certificates).await
    }

    /// probes is a special case where an object server will share this without requiring any peer or signed url authentication
    pub async fn probes(
        Path((epoch, checksum)): Path<(Epoch, Checksum)>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Probes(epoch, checksum);
        service
            .handle_download_object(&path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)
    }

    pub async fn inputs(
        Path((epoch, checksum)): Path<(Epoch, Checksum)>,
        Query(params): Query<SignedParams>,
        peer_certificates: axum::Extension<PeerCertificates>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Inputs(epoch, checksum);
        Self::signed_download(service, path, params, peer_certificates).await
    }

    pub async fn blocks(
        Path((epoch, round, authority_index, checksum)): Path<(
            Epoch,
            Round,
            AuthorityIndex,
            Checksum,
        )>,
        Query(params): Query<SignedParams>,
        peer_certificates: axum::Extension<PeerCertificates>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Blocks(epoch, round, authority_index, checksum);
        Self::signed_download(service, path, params, peer_certificates).await
    }

    pub async fn signed_download(
        service: ObjectService<S>,
        path: ObjectPath,
        params: SignedParams,
        peer_certificates: axum::Extension<PeerCertificates>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let pk = public_key_from_certificate(peer_certificates.peer_certs().first().unwrap())
            .ok()
            .unwrap();
        let peer = NetworkPublicKey::new(pk.clone());

        if let Some(ref peers) = params.peers {
            if !peers.contains(&peer) {
                return Err(StatusCode::UNAUTHORIZED);
            }
        } else {
            return Err(StatusCode::UNAUTHORIZED);
        }

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .as_secs();

        if params.expires < current_time {
            return Err(StatusCode::GONE);
        }

        if params.prefix.is_empty() || !path.path().starts_with(&params.prefix) {
            return Err(StatusCode::BAD_REQUEST);
        }

        if !service.verify_signed_params(&params).is_ok() {
            return Err(StatusCode::UNAUTHORIZED);
        }

        service
            .handle_download_object(&path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)
    }
}

pub struct ExternalObjectServiceManager {
    client: Arc<ObjectClient<ExternalClientPool>>,
    own_key: NetworkKeyPair,
    allower: AllowPublicKeys,
    parameters: Arc<HttpParameters>,
    server: Option<ServerHandle>,
}

impl ExternalObjectServiceManager {
    pub fn new(
        own_key: NetworkKeyPair,
        parameters: Arc<HttpParameters>,
        allower: AllowPublicKeys,
    ) -> ObjectResult<Self> {
        Ok(Self {
            client: Arc::new(ObjectClient::new(ExternalClientPool::new(
                own_key.clone(),
                parameters.clone(),
            ))?),
            own_key,
            allower,
            parameters,
            server: None,
        })
    }
}

impl<S: ObjectStorage + Clone> ObjectServiceManager<S> for ExternalObjectServiceManager {
    type ClientPool = ExternalClientPool;

    fn client(&self) -> Arc<ObjectClient<Self::ClientPool>> {
        self.client.clone()
    }

    async fn start(&mut self, address: &Multiaddr, service: ObjectService<S>) {
        let config = &self.parameters;
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
                    ExternalObjectService::new(service.clone()).router(),
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

impl Drop for ExternalObjectServiceManager {
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
        time::Duration,
    };

    use bytes::Bytes;
    use rand::{rngs::OsRng, RngCore};
    use soma_tls::AllowPublicKeys;
    use types::{
        checksum::Checksum,
        metadata::{
            DownloadableMetadata, DownloadableMetadataV1, Metadata, MetadataV1, ObjectPath,
            SignedParams,
        },
        parameters::HttpParameters,
    };
    use types::{crypto::NetworkKeyPair, multiaddr::Multiaddr};

    use crate::{
        networking::{
            external_service::{ExternalClientPool, ExternalObjectServiceManager},
            ObjectClient, ObjectService, ObjectServiceManager,
        },
        storage::{memory::MemoryObjectStore, ObjectStorage},
    };

    fn get_available_local_address() -> Multiaddr {
        let host = "127.0.0.1";
        let port = get_available_port(host);
        format!("/ip4/{}/tcp/{}", host, port).parse().unwrap()
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
        let parameters = Arc::new(HttpParameters::default());
        let mut rng = rand::thread_rng();
        let mut buffer = vec![0u8; 1024 * 1024];
        OsRng.fill_bytes(&mut buffer);
        let random_bytes = Bytes::from(buffer);

        let address = get_available_local_address();
        let client_keypair = NetworkKeyPair::generate(&mut rng);
        let server_keypair = NetworkKeyPair::generate(&mut rng);
        let checksum = Checksum::new_from_bytes(&random_bytes);
        let download_size = random_bytes.len() as u64;
        let object_path = ObjectPath::Tmp(0, checksum);

        let metadata = Metadata::V1(MetadataV1::new(object_path.clone(), download_size));

        let params = SignedParams::new(
            object_path.path(),
            Some(vec![client_keypair.public()]),
            Duration::from_secs(10),
            &server_keypair,
        );

        let downloadable_metadata = DownloadableMetadata::V1(DownloadableMetadataV1::new(
            Some(server_keypair.public()),
            Some(params),
            address.clone(),
            metadata,
        ));

        let client_object_storage = Arc::new(MemoryObjectStore::new());
        let server_object_storage = Arc::new(MemoryObjectStore::new());

        server_object_storage
            .put_object(&object_path, random_bytes.clone())
            .await
            .unwrap();

        let server_object_network_service: ObjectService<MemoryObjectStore> =
            ObjectService::new(server_object_storage.clone(), server_keypair.public());

        let allower = AllowPublicKeys::new(BTreeSet::from([client_keypair
            .public()
            .into_inner()
            .clone()]));
        // allower.update(BTreeSet::from([client_public_key.clone()]));
        let mut server_object_network_manager =
            ExternalObjectServiceManager::new(server_keypair.clone(), parameters.clone(), allower)
                .unwrap();

        server_object_network_manager
            .start(&address, server_object_network_service)
            .await;

        let object_client =
            ObjectClient::new(ExternalClientPool::new(client_keypair, parameters)).unwrap();

        let mut writer = client_object_storage
            .get_object_writer(&object_path)
            .await
            .unwrap();

        object_client
            .download_object(&mut writer, &downloadable_metadata)
            .await
            .unwrap();

        let x = client_object_storage
            .get_object(&object_path)
            .await
            .unwrap();

        assert_eq!(random_bytes, x)
    }
}
