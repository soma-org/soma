use crate::networking::{DownloadService, ObjectServiceManager};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use object_store::ObjectStore;
use soma_http::ServerHandle;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{info, warn};
use types::{
    checksum::Checksum, committee::Epoch, crypto::NetworkKeyPair, error::ObjectResult,
    metadata::ObjectPath, multiaddr::Multiaddr, parameters::HttpParameters, shard::Shard,
    shard_crypto::digest::Digest, sync::to_socket_addr,
};

pub struct InternalObjectServiceManager {
    parameters: Arc<HttpParameters>,
    own_key: NetworkKeyPair,
    server: Option<ServerHandle>,
}

impl InternalObjectServiceManager {
    pub fn new(own_key: NetworkKeyPair, parameters: Arc<HttpParameters>) -> ObjectResult<Self> {
        Ok(Self {
            parameters,
            own_key,
            server: None,
        })
    }
}

impl<S: ObjectStore> ObjectServiceManager<S> for InternalObjectServiceManager {
    async fn start(&mut self, address: &Multiaddr, service: DownloadService<S>) {
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

struct InternalObjectService<S: ObjectStore> {
    service: DownloadService<S>,
}

impl<S: ObjectStore> Clone for InternalObjectService<S> {
    fn clone(&self) -> Self {
        Self {
            service: self.service.clone(),
        }
    }
}

impl<S: ObjectStore> InternalObjectService<S> {
    const fn new(service: DownloadService<S>) -> Self {
        Self { service }
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
        Path((epoch, shard, checksum)): Path<(Epoch, Digest<Shard>, Checksum)>,
        State(Self { service }): State<Self>,
    ) -> Result<impl IntoResponse, StatusCode> {
        let path = ObjectPath::Inputs(epoch, shard, checksum);
        Self::download(service, path).await
    }

    pub async fn download(
        service: DownloadService<S>,
        path: ObjectPath,
    ) -> Result<impl IntoResponse, StatusCode> {
        service
            .handle_download_object(&path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)
    }
}
