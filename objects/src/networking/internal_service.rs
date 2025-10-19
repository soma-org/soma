use crate::{
    networking::{ClientPool, ObjectClient, ObjectService, ObjectServiceManager},
    storage::ObjectStorage,
};
use async_trait::async_trait;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use reqwest::Client;
use soma_http::ServerHandle;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{info, warn};
use types::{
    checksum::Checksum,
    committee::{AuthorityIndex, Epoch},
    consensus::block::Round,
    crypto::NetworkKeyPair,
    error::{ObjectError, ObjectResult},
    metadata::{DownloadableMetadata, ObjectPath},
    multiaddr::Multiaddr,
    p2p::to_socket_addr,
    parameters::HttpParameters,
    shard::Shard,
    shard_crypto::digest::Digest,
};

pub struct InternalClientPool {
    client: Client,
    parameters: Arc<HttpParameters>,
    own_key: NetworkKeyPair,
}

impl InternalClientPool {
    pub(crate) fn new(
        own_key: NetworkKeyPair,
        parameters: Arc<HttpParameters>,
    ) -> ObjectResult<Self> {
        let buffer_size = parameters.connection_buffer_size;

        let client = reqwest::Client::builder()
            .http2_initial_connection_window_size(Some(buffer_size as u32))
            .http2_initial_stream_window_size(Some(buffer_size as u32 / 2))
            .http2_keep_alive_while_idle(true)
            .http2_keep_alive_interval(parameters.keepalive_interval)
            .http2_keep_alive_timeout(parameters.keepalive_interval)
            .connect_timeout(parameters.connect_timeout)
            .user_agent("SOMA (Object Client)")
            .build()
            .map_err(|e| ObjectError::ReqwestError(e.to_string()))?;

        Ok(Self {
            client,
            parameters,
            own_key,
        })
    }
}

#[async_trait]
impl ClientPool for InternalClientPool {
    async fn get_client(
        &self,
        downloadable_metadata: &DownloadableMetadata,
    ) -> ObjectResult<Client> {
        Ok(self.client.clone())
    }

    fn calculate_timeout(&self, num_bytes: u64) -> Duration {
        let nanos = self
            .parameters
            .nanoseconds_per_byte
            .saturating_mul(num_bytes);
        self.parameters.connect_timeout + Duration::from_nanos(nanos)
    }
}

pub struct InternalObjectServiceManager {
    client: Arc<ObjectClient<InternalClientPool>>,
    parameters: Arc<HttpParameters>,
    own_key: NetworkKeyPair,
    server: Option<ServerHandle>,
}

impl InternalObjectServiceManager {
    pub fn new(own_key: NetworkKeyPair, parameters: Arc<HttpParameters>) -> ObjectResult<Self> {
        Ok(Self {
            client: Arc::new(ObjectClient::new(InternalClientPool::new(
                own_key.clone(),
                parameters.clone(),
            )?)?),
            parameters,
            own_key,
            server: None,
        })
    }
}

impl<S: ObjectStorage + Clone> ObjectServiceManager<S> for InternalObjectServiceManager {
    type ClientPool = InternalClientPool;
    fn client(&self) -> Arc<ObjectClient<Self::ClientPool>> {
        self.client.clone()
    }
    async fn start(&mut self, address: &Multiaddr, service: ObjectService<S>) {
        let own_address = if address.is_localhost_ip() {
            address.clone()
        } else {
            address.with_zero_ip()
        };

        let own_address = to_socket_addr(&own_address).unwrap();

        let deadline = Instant::now() + Duration::from_secs(20);
        let server = loop {
            match soma_http::Builder::new().serve(
                own_address,
                InternalObjectService::new(service.clone()).router(),
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

impl Drop for InternalObjectServiceManager {
    fn drop(&mut self) {
        if let Some(server) = self.server.as_ref() {
            server.trigger_shutdown();
        }
    }
}

#[derive(Clone)]
struct InternalObjectService<S: ObjectStorage> {
    service: ObjectService<S>,
}

impl<S: ObjectStorage + Clone> InternalObjectService<S> {
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
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Embeddings(epoch, shard, checksum);
        Self::download(service, path).await
    }

    pub async fn probes(
        Path((epoch, checksum)): Path<(Epoch, Checksum)>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Probes(epoch, checksum);
        Self::download(service, path).await
    }

    pub async fn inputs(
        Path((epoch, checksum)): Path<(Epoch, Checksum)>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Inputs(epoch, checksum);
        Self::download(service, path).await
    }

    pub async fn blocks(
        Path((epoch, round, authority_index, checksum)): Path<(
            Epoch,
            Round,
            AuthorityIndex,
            Checksum,
        )>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Blocks(epoch, round, authority_index, checksum);
        Self::download(service, path).await
    }

    pub async fn download(
        service: ObjectService<S>,
        path: ObjectPath,
    ) -> Result<impl IntoResponse, StatusCode> {
        service
            .handle_download_object(&path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)
    }
}
