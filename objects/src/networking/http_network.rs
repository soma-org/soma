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
use quick_cache::sync::Cache;
use reqwest::Client;
use shared::{
    crypto::keys::{PeerKeyPair, PeerPublicKey},
    metadata::{Metadata, MetadataAPI},
};
use soma_http::ServerHandle;
use soma_tls::{create_rustls_client_config, AllowPublicKeys, TlsConnectionInfo};
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
        timeout: Duration,
    ) -> ObjectResult<()>
    where
        W: AsyncWrite + Unpin + Send,
    {
        let address = to_host_port_str(address).map_err(|e| {
            ObjectError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        let address = format!("https://{address}/{}", metadata.checksum());
        let url = Url::from_str(&address).map_err(|e| ObjectError::UrlParseError(e.to_string()))?;

        let mut response = self
            .get_client(peer)
            .await?
            .get(url.clone())
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?;

        if !response.status().is_success() {
            return Err(ObjectError::NetworkRequest(format!(
                "http status: {}",
                response.status().as_u16()
            )));
        }
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?
        {
            writer
                .write_all(&chunk)
                .await
                .map_err(|e| ObjectError::WriteError(e.to_string()))?;
        }

        writer
            .flush()
            .await
            .map_err(|e| ObjectError::WriteError(e.to_string()))?;
        Ok(())
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
        tls_info: axum::Extension<TlsConnectionInfo>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::new(path).map_err(|_| StatusCode::BAD_REQUEST)?;
        let pk = tls_info.public_key().unwrap();
        // instead handle this as unauthorized!
        let peer = PeerPublicKey::new(pk.clone());
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
    use fastcrypto::{
        ed25519::{Ed25519KeyPair, Ed25519PublicKey},
        traits::KeyPair,
    };
    use soma_tls::{
        create_rustls_client_config, AllowPublicKeys, ClientCertVerifier, SelfSignedCertificate,
        TlsAcceptor, TlsConnectionInfo,
    };
    use std::{collections::BTreeSet, time::Duration};
    use tower_http::compression::CompressionLayer;

    fn create_reqwest_client(
        target_public_key: Ed25519PublicKey,
        client_keypair: Ed25519KeyPair,
        server_name: &str,
    ) -> reqwest::Result<reqwest::Client> {
        // Use your existing create_rustls_client_config
        let tls_config = create_rustls_client_config(
            target_public_key,
            server_name.to_string(),
            Some(client_keypair.private()),
        );
        reqwest::Client::builder()
            .use_preconfigured_tls(tls_config) // Use the rustls ClientConfig directly
            .http2_prior_knowledge() // Equivalent to https_only + enable_http2
            .timeout(Duration::from_secs(60)) // General timeout (adjust as needed)
            .http2_keep_alive_interval(Duration::from_secs(60)) // Match hyper settings
            .http2_keep_alive_timeout(Duration::from_secs(60)) // Match hyper settings
            .build()
    }

    #[tokio::test]
    async fn axum_mtls() {
        const SERVER_NAME: &str = "test_server";
        use fastcrypto::ed25519::Ed25519KeyPair;
        use fastcrypto::traits::KeyPair;
        let mut rng = rand::thread_rng();
        let client_keypair = Ed25519KeyPair::generate(&mut rng);
        let client_public_key = client_keypair.copy().public().to_owned();
        let server_keypair = Ed25519KeyPair::generate(&mut rng);
        let server_public_key = server_keypair.copy().public().to_owned();
        let server_certificate = SelfSignedCertificate::new(server_keypair.private(), SERVER_NAME);

        let allowlist = AllowPublicKeys::new(BTreeSet::new());
        let tls_config = ClientCertVerifier::new(allowlist.clone(), SERVER_NAME.to_string())
            .rustls_server_config(
                vec![server_certificate.rustls_certificate()],
                server_certificate.rustls_private_key(),
            )
            .unwrap();

        async fn handler(tls_info: axum::Extension<TlsConnectionInfo>) -> String {
            let pk = tls_info.public_key().unwrap().to_string();
            println!("received message from: {pk}");
            pk
        }

        let app = axum::Router::new()
            .route("/", axum::routing::get(handler))
            .layer(CompressionLayer::new().zstd(true));
        let listener = std::net::TcpListener::bind("localhost:0").unwrap();
        let server_address = listener.local_addr().unwrap();
        let acceptor = TlsAcceptor::new(tls_config);
        let _server = tokio::spawn(async move {
            axum_server::Server::from_tcp(listener)
                .acceptor(acceptor)
                .serve(app.into_make_service())
                .await
                .unwrap()
        });

        let server_url = format!("https://localhost:{}", server_address.port());

        let client =
            create_reqwest_client(server_public_key.to_owned(), client_keypair, SERVER_NAME)
                .unwrap();

        let res = client
            .get(&server_url)
            .header("Accept-Encoding", "zstd") // Request zstd compression
            .send()
            .await
            .unwrap_err();
        println!("{:?}", res);

        allowlist.update(BTreeSet::from([client_public_key.clone()]));

        let res = client
            .get(&server_url)
            .header("Accept-Encoding", "zstd") // Request zstd compression
            .send()
            .await
            .unwrap();
        println!("{:?}", res);
        let body_str = res.text().await.unwrap(); // Automatically decompresses zstd
        println!("Public key from response: {}", body_str);
        assert_eq!(body_str, client_public_key.to_string());
    }
}
