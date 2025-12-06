pub mod signed_url;
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header, HeaderMap, Response, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use object_store::{GetOptions, GetRange, ObjectStore};
use soma_http::{PeerCertificates, ServerHandle};
use soma_tls::{public_key_from_certificate, AllowPublicKeys};
use std::{
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tracing::{info, warn};
use types::{
    checksum::Checksum,
    committee::Epoch,
    crypto::{NetworkKeyPair, NetworkPublicKey},
    metadata::ObjectPath,
    multiaddr::Multiaddr,
    parameters::HttpParameters,
    shard::Shard,
    shard_crypto::digest::Digest,
    sync::to_socket_addr,
};
use types::{error::ObjectResult, shard_networking::CERTIFICATE_NAME};

use crate::services::signed_url::SignedParams;

pub struct ObjectService<S: ObjectStore> {
    object_store: Arc<S>,
    own_key: NetworkPublicKey,
}

impl<S: ObjectStore> Clone for ObjectService<S> {
    fn clone(&self) -> Self {
        Self {
            object_store: self.object_store.clone(),
            own_key: self.own_key.clone(),
        }
    }
}

impl<S: ObjectStore> ObjectService<S> {
    pub fn new(object_store: Arc<S>, own_key: NetworkPublicKey) -> Self {
        Self {
            object_store,
            own_key,
        }
    }

    fn router(self) -> Router {
        Router::new()
            .route(
                "/epochs/{epoch}/shards/{shard}/embeddings/{checksum}",
                get(Self::embeddings),
            )
            .route("/epochs/{epoch}/probes/{checksum}", get(Self::probes))
            .route(
                "/epochs/{epoch}/shards/{shard}/inputs/{checksum}",
                get(Self::inputs),
            )
            .with_state(self)
    }

    pub async fn embeddings(
        Path((epoch, shard, checksum)): Path<(Epoch, Digest<Shard>, Checksum)>,
        Query(params): Query<SignedParams>,
        peer_certificates: axum::Extension<PeerCertificates>,
        headers: HeaderMap,
        State(Self {
            object_store,
            own_key,
        }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Embeddings(epoch, shard, checksum);
        Self::serve_object(
            object_store,
            own_key,
            path,
            params,
            peer_certificates,
            headers,
        )
        .await
    }

    pub async fn probes(
        Path((epoch, checksum)): Path<(Epoch, Checksum)>,
        Query(params): Query<SignedParams>,
        peer_certificates: axum::Extension<PeerCertificates>,
        headers: HeaderMap,
        State(Self {
            object_store,
            own_key,
        }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Probes(epoch, checksum);
        Self::serve_object(
            object_store,
            own_key,
            path,
            params,
            peer_certificates,
            headers,
        )
        .await
    }

    pub async fn inputs(
        Path((epoch, shard, checksum)): Path<(Epoch, Digest<Shard>, Checksum)>,
        Query(params): Query<SignedParams>,
        peer_certificates: axum::Extension<PeerCertificates>,
        headers: HeaderMap,
        State(Self {
            object_store,
            own_key,
        }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Inputs(epoch, shard, checksum);
        Self::serve_object(
            object_store,
            own_key,
            path,
            params,
            peer_certificates,
            headers,
        )
        .await
    }

    pub async fn serve_object(
        object_store: Arc<S>,
        own_key: NetworkPublicKey,
        path: ObjectPath,
        params: SignedParams,
        peer_certificates: axum::Extension<PeerCertificates>,
        headers: HeaderMap,
    ) -> Result<impl IntoResponse, StatusCode> {
        // VERIFICATION ///////////////////////////////////////////////
        let pk = public_key_from_certificate(peer_certificates.peer_certs().first().unwrap())
            .ok()
            .unwrap();
        let _peer = NetworkPublicKey::new(pk.clone());

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .as_secs();
        if params.expires < now {
            return Err(StatusCode::GONE);
        }
        params
            .verify(&own_key)
            .map_err(|_| StatusCode::UNAUTHORIZED)?;

        // DATA LOADING ///////////////////////////////////////////////
        let head = object_store
            .head(&path.path())
            .await
            .map_err(|_| StatusCode::NOT_FOUND)?;
        let total_size = head.size;
        let range = Self::parse_range(&headers, total_size);

        let (body, status, content_range) = match range {
            Some((start, end)) => {
                let mut options = GetOptions::default();
                options.range = Some(GetRange::Bounded(start..end + 1));
                let reader = object_store
                    .get_opts(&path.path(), options)
                    .await
                    .map_err(|_| StatusCode::NOT_FOUND)?;
                let body = Body::from_stream(reader.into_stream());

                (
                    body,
                    StatusCode::PARTIAL_CONTENT,
                    format!("bytes {start}-{end}/{total_size}"),
                )
            }
            None => {
                let reader = object_store
                    .get(&path.path())
                    .await
                    .map_err(|_| StatusCode::NOT_FOUND)?;
                let body = Body::from_stream(reader.into_stream());

                (body, StatusCode::OK, format!("bytes */{total_size}"))
            }
        };

        println!("creating response");

        let resp = Response::builder()
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(header::ACCEPT_RANGES, "bytes")
            .header(header::CACHE_CONTROL, "public, max-age=31536000, immutable")
            .header(header::ETAG, path.checksum().to_string())
            .header(header::CONTENT_RANGE, content_range);

        let resp = resp
            .status(status)
            .body(body)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        println!("{:?}", resp);

        Ok(resp)
    }
    fn parse_range(headers: &HeaderMap, total_size: usize) -> Option<(usize, usize)> {
        let range = headers.get(header::RANGE)?.to_str().ok()?;
        let range = range.strip_prefix("bytes=")?;
        let mut parts = range.split('-');
        let start = parts.next()?.parse::<usize>().ok()?;
        let end = parts
            .next()?
            .parse::<usize>()
            .ok()
            .unwrap_or(total_size - 1);

        if start >= total_size || end >= total_size || start > end {
            return None;
        }
        Some((start, end))
    }
}

pub struct ObjectServiceManager {
    own_key: NetworkKeyPair,
    allower: AllowPublicKeys,
    parameters: Arc<HttpParameters>,
    server: Option<ServerHandle>,
}

impl ObjectServiceManager {
    pub fn new(
        own_key: NetworkKeyPair,
        parameters: Arc<HttpParameters>,
        allower: AllowPublicKeys,
    ) -> ObjectResult<Self> {
        Ok(Self {
            own_key,
            allower,
            parameters,
            server: None,
        })
    }

    /// Compute the bind address. In msim, we must bind to the actual IP.
    /// In production, we bind to 0.0.0.0 for non-localhost addresses.
    #[cfg(msim)]
    fn compute_bind_address(address: &Multiaddr) -> Multiaddr {
        // In msim, always use the actual address - can't bind to 0.0.0.0
        address.clone()
    }

    #[cfg(not(msim))]
    fn compute_bind_address(address: &Multiaddr) -> Multiaddr {
        if address.is_localhost_ip() {
            address.clone()
        } else {
            address.with_zero_ip()
        }
    }

    pub async fn start<S: ObjectStore>(&mut self, address: &Multiaddr, service: ObjectService<S>) {
        let config = &self.parameters;
        let own_address = Self::compute_bind_address(address);

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
                .serve(own_address, service.clone().router())
            {
                Ok(server) => break server,
                Err(err) => {
                    warn!("Error starting object server: {err:?}");
                    if Instant::now() > deadline {
                        panic!("Failed to start object server within required deadline");
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        };

        info!("Object server started at: {own_address}");
        self.server = Some(server);
    }

    pub async fn stop(&mut self) {
        if let Some(server) = self.server.take() {
            server.shutdown().await;
        }
    }
}

impl Drop for ObjectServiceManager {
    fn drop(&mut self) {
        if let Some(server) = self.server.as_ref() {
            server.trigger_shutdown();
        }
    }
}

#[cfg(test)]
mod http_tests {
    use crate::readers::url::ObjectHttpClient;
    use crate::{downloader::ObjectDownloader, readers::url::ObjectHttpReader, MIN_PART_SIZE};

    use super::*;
    use bytes::Bytes;
    use fastcrypto::hash::HashFunction;
    use object_store::memory::InMemory;
    use rand::{rngs::OsRng, RngCore};
    use soma_tls::AllowPublicKeys;
    use std::{collections::BTreeSet, net::SocketAddr};
    use std::{net::TcpListener, sync::Arc};
    use std::{net::TcpStream, time::Duration};
    use tokio::sync::Semaphore;
    use types::{
        checksum::Checksum,
        committee::get_available_local_address,
        committee::Epoch,
        crypto::{DefaultHash, NetworkKeyPair, NetworkPublicKey},
        error::ObjectError,
        metadata::{
            DownloadMetadata, Metadata, MetadataV1, MtlsDownloadMetadata, MtlsDownloadMetadataV1,
            ObjectPath,
        },
        parameters::HttpParameters,
        shard_crypto::digest::Digest,
        sync::to_host_port_str,
    };
    use url::Url;

    // --------------------------------------------------------------------- //
    //  Server start / stop helper
    // --------------------------------------------------------------------- //
    async fn spawn_server(
        store: Arc<InMemory>,
        server_kp: NetworkKeyPair,
        client_pk: NetworkPublicKey,
    ) -> (ObjectServiceManager, Multiaddr) {
        let addr = get_available_local_address();

        // Only the client is allowed to connect
        let allower = AllowPublicKeys::new(BTreeSet::from([client_pk.clone().into_inner()]));

        let params = Arc::new(HttpParameters::default());

        let object_service = ObjectService::new(store, server_kp.public().clone());

        let mut manager =
            ObjectServiceManager::new(server_kp.clone(), params.clone(), allower).unwrap();

        manager.start(&addr, object_service).await;

        (manager, addr)
    }

    fn init_tracing() {
        use tracing_subscriber::{fmt, EnvFilter};

        let _ = fmt()
            .with_env_filter(EnvFilter::from_default_env()) // Respects RUST_LOG
            .try_init(); // Ignores if already initialized
    }
    // --------------------------------------------------------------------- //
    //  Signed URL builder (includes `peers`)
    // --------------------------------------------------------------------- //
    fn signed_url(
        base: &str,
        path: &ObjectPath,
        server_kp: &NetworkKeyPair,
        timeout: Duration,
    ) -> Url {
        let prefix = path.path().to_string();
        let params = SignedParams::new(prefix.clone(), timeout, server_kp);

        let query = serde_urlencoded::to_string(&params).unwrap();
        Url::parse(&format!("{base}/{prefix}?{query}")).unwrap()
    }

    // --------------------------------------------------------------------- //
    //  Test: small object (single request)
    // --------------------------------------------------------------------- //
    #[tokio::test]
    async fn download_small_object_via_http() {
        init_tracing();
        let mut rng = rand::thread_rng();
        let client_kp = NetworkKeyPair::generate(&mut rng);
        let server_kp = NetworkKeyPair::generate(&mut rng);

        let source_store: Arc<InMemory> = Arc::new(InMemory::new());
        let (mut manager, address) = spawn_server(
            source_store.clone(),
            server_kp.clone(),
            client_kp.public().clone(),
        )
        .await;

        let base = to_host_port_str(&address).unwrap();
        let base = format!("https://{base}");
        let data_len = MIN_PART_SIZE / 2;
        let data = vec![1u8; data_len as usize];

        let mut h = DefaultHash::new();
        h.update(&data);
        let checksum = Checksum::new_from_hash(h.finalize().into());
        let meta = Metadata::V1(MetadataV1::new(checksum, data.len()));
        let path = ObjectPath::Probes(0, checksum);

        // Upload to server
        source_store
            .put(&path.path(), data.clone().into())
            .await
            .unwrap();

        // Signed URL
        let url = signed_url(&base, &path, &server_kp, Duration::from_secs(30));

        // Client with mTLS (uses ObjectHttpClient)
        let http_params = Arc::new(HttpParameters::default());
        let url_client = ObjectHttpClient::new(client_kp.clone(), http_params).unwrap();
        let dm = DownloadMetadata::Mtls(MtlsDownloadMetadata::V1(MtlsDownloadMetadataV1::new(
            server_kp.public(),
            url.clone(),
            meta.clone(),
        )));
        let client = url_client.get_client(&dm).await.unwrap();

        let reader = Arc::new(ObjectHttpReader::new(url, client));

        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let concurrency = Arc::new(Semaphore::new(5));
        let downloader = ObjectDownloader::new(concurrency, MIN_PART_SIZE, 40).unwrap();

        downloader
            .download(reader, dest_store.clone(), path.clone(), meta)
            .await
            .unwrap();

        let got = dest_store
            .get(&path.path())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(got.to_vec(), data);

        manager.stop().await;
    }

    // --------------------------------------------------------------------- //
    //  Test: large multipart object
    // --------------------------------------------------------------------- //
    #[tokio::test]
    async fn download_large_object_multipart_via_http() {
        init_tracing();
        let mut rng = rand::thread_rng();
        let client_kp = NetworkKeyPair::generate(&mut rng);
        let server_kp = NetworkKeyPair::generate(&mut rng);

        let source_store: Arc<InMemory> = Arc::new(InMemory::new());
        let (mut manager, address) = spawn_server(
            source_store.clone(),
            server_kp.clone(),
            client_kp.public().clone(),
        )
        .await;

        let base = to_host_port_str(&address).unwrap();
        let base = format!("https://{base}");
        let chunk = MIN_PART_SIZE as usize;
        let data_len = chunk * 3;
        let data = vec![2u8; data_len];

        let mut h = DefaultHash::new();
        h.update(&data);
        let checksum = Checksum::new_from_hash(h.finalize().into());
        let meta = Metadata::V1(MetadataV1::new(checksum, data.len()));
        let path = ObjectPath::Probes(1, checksum);

        source_store
            .put(&path.path(), data.clone().into())
            .await
            .unwrap();

        let url = signed_url(&base, &path, &server_kp, Duration::from_secs(30));

        let http_params = Arc::new(HttpParameters::default());
        let url_client = ObjectHttpClient::new(client_kp.clone(), http_params).unwrap();
        let dm = DownloadMetadata::Mtls(MtlsDownloadMetadata::V1(MtlsDownloadMetadataV1::new(
            server_kp.public(),
            url.clone(),
            meta.clone(),
        )));
        let client = url_client.get_client(&dm).await.unwrap();

        let reader = Arc::new(ObjectHttpReader::new(url, client));

        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let concurrency = Arc::new(Semaphore::new(3));
        let downloader = ObjectDownloader::new(concurrency, MIN_PART_SIZE, 40).unwrap();

        downloader
            .download(reader, dest_store.clone(), path.clone(), meta)
            .await
            .unwrap();

        let got = dest_store
            .get(&path.path())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(got.to_vec(), data);

        manager.stop().await;
    }

    // --------------------------------------------------------------------- //
    //  Test: existing object → skip download
    // --------------------------------------------------------------------- //
    #[tokio::test]
    async fn download_existing_object_skips_via_http() {
        let mut rng = rand::thread_rng();
        let client_kp = NetworkKeyPair::generate(&mut rng);
        let server_kp = NetworkKeyPair::generate(&mut rng);

        let source_store: Arc<InMemory> = Arc::new(InMemory::new());
        let (mut manager, address) = spawn_server(
            source_store.clone(),
            server_kp.clone(),
            client_kp.public().clone(),
        )
        .await;

        let base = to_host_port_str(&address).unwrap();
        let base = format!("https://{base}");
        let chunk = MIN_PART_SIZE as usize;
        let data = vec![3u8; chunk];

        let mut h = DefaultHash::new();
        h.update(&data);
        let checksum = Checksum::new_from_hash(h.finalize().into());
        let meta = Metadata::V1(MetadataV1::new(checksum, data.len()));
        let path = ObjectPath::Probes(2, checksum);

        source_store
            .put(&path.path(), data.clone().into())
            .await
            .unwrap();

        // Pre-populate destination so head() succeeds
        let dest_store: Arc<InMemory> = Arc::new(InMemory::new());
        dest_store
            .put(&path.path(), data.clone().into())
            .await
            .unwrap();

        let url = signed_url(&base, &path, &server_kp, Duration::from_secs(30));

        let http_params = Arc::new(HttpParameters::default());
        let url_client = ObjectHttpClient::new(client_kp.clone(), http_params).unwrap();
        let dm = DownloadMetadata::Mtls(MtlsDownloadMetadata::V1(MtlsDownloadMetadataV1::new(
            server_kp.public(),
            url.clone(),
            meta.clone(),
        )));
        let client = url_client.get_client(&dm).await.unwrap();

        let reader = Arc::new(ObjectHttpReader::new(url, client));

        let concurrency = Arc::new(Semaphore::new(2));
        let downloader = ObjectDownloader::new(concurrency, MIN_PART_SIZE, 40).unwrap();

        // Should finish instantly (skip)
        downloader
            .download(
                reader,
                Arc::clone(&dest_store) as Arc<dyn ObjectStore>,
                path.clone(),
                meta,
            )
            .await
            .unwrap();

        let got = dest_store
            .get(&path.path())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(got.to_vec(), data);

        manager.stop().await;
    }

    // --------------------------------------------------------------------- //
    //  Test: checksum mismatch → verification error
    // --------------------------------------------------------------------- //
    #[tokio::test]
    async fn download_checksum_mismatch_fails_via_http() {
        let mut rng = rand::thread_rng();
        let client_kp = NetworkKeyPair::generate(&mut rng);
        let server_kp = NetworkKeyPair::generate(&mut rng);

        let source_store: Arc<InMemory> = Arc::new(InMemory::new());
        let (mut manager, address) = spawn_server(
            source_store.clone(),
            server_kp.clone(),
            client_kp.public().clone(),
        )
        .await;

        let base = to_host_port_str(&address).unwrap();
        let base = format!("https://{base}");
        let chunk = MIN_PART_SIZE as usize;
        let data = vec![4u8; chunk];

        let mut h = DefaultHash::new();
        h.update(&data);
        let _real = Checksum::new_from_hash(h.finalize().into());

        // Wrong checksum on purpose
        let wrong = Checksum::new_from_hash(DefaultHash::digest(b"wrong").into());
        let meta = Metadata::V1(MetadataV1::new(wrong, data.len()));
        let path = ObjectPath::Probes(3, wrong);

        source_store.put(&path.path(), data.into()).await.unwrap();

        let url = signed_url(&base, &path, &server_kp, Duration::from_secs(30));

        let http_params = Arc::new(HttpParameters::default());
        let url_client = ObjectHttpClient::new(client_kp.clone(), http_params).unwrap();
        let dm = DownloadMetadata::Mtls(MtlsDownloadMetadata::V1(MtlsDownloadMetadataV1::new(
            server_kp.public(),
            url.clone(),
            meta.clone(),
        )));

        let client = url_client.get_client(&dm).await.unwrap();

        let reader = Arc::new(ObjectHttpReader::new(url, client));

        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let concurrency = Arc::new(Semaphore::new(2));
        let downloader = ObjectDownloader::new(concurrency, MIN_PART_SIZE, 40).unwrap();

        let err = downloader
            .download(reader, dest_store.clone(), path, meta)
            .await
            .unwrap_err();

        assert!(matches!(err, ObjectError::VerificationError(_)));

        manager.stop().await;
    }
}
