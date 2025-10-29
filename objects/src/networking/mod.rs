pub mod download;
pub mod downloader;
pub mod external_service;
pub mod internal_service;
pub mod tus;

use axum::{
    body::Body,
    http::{header, Response, StatusCode},
    response::IntoResponse,
};
use object_store::ObjectStore;
use quick_cache::sync::Cache;
use reqwest::Client;
use soma_tls::create_rustls_client_config;
use std::sync::Arc;
use types::{
    crypto::{NetworkKeyPair, NetworkPublicKey},
    error::ObjectError,
    metadata::{DownloadMetadata, MtlsDownloadMetadataAPI, ObjectPath, SignedParams},
    multiaddr::Multiaddr,
    parameters::HttpParameters,
    shard_networking::CERTIFICATE_NAME,
};

use types::error::ObjectResult;

use crate::networking::download::Download;

/// Cloud providers typically require a minimum multipart part size except for the last part
const MIN_PART_SIZE: u64 = 5 * 1024 * 1024;
/// Cloud providers typically have a max multipart part size
const MAX_PART_SIZE: u64 = 5 * 1024 * 1024 * 1024;

pub struct DownloadClient<S: ObjectStore> {
    client: Client,
    mtls_clients: Arc<Cache<NetworkPublicKey, Client>>,
    own_key: NetworkKeyPair,
    parameters: Arc<HttpParameters>,
    storage: Arc<S>,
    max_size: u64,
}

impl<S: ObjectStore> DownloadClient<S> {
    pub fn new(
        storage: Arc<S>,
        own_key: NetworkKeyPair,
        parameters: Arc<HttpParameters>,
        max_size: u64,
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
            mtls_clients: Arc::new(Cache::new(parameters.client_pool_capacity)),
            own_key,
            parameters,
            storage,
            max_size,
        })
    }

    pub async fn download(
        &self,
        path: ObjectPath,
        download_metadata: &DownloadMetadata,
    ) -> ObjectResult<()> {
        let client = match download_metadata {
            DownloadMetadata::Default(dm) => self.client.clone(),
            DownloadMetadata::Mtls(dm) => {
                let peer = dm.peer();
                if let Some(client) = self.mtls_clients.get(peer) {
                    client
                } else {
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

                    self.mtls_clients.insert(peer.clone(), client.clone());
                    client
                }
            }
        };
        Download::new(
            client,
            download_metadata.url().clone(),
            self.storage.clone(),
            path,
            download_metadata.metadata().clone(),
            self.parameters.nanoseconds_per_byte,
            self.max_size,
        )
        .await
    }
}

impl<S: ObjectStore> Clone for DownloadClient<S> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            mtls_clients: self.mtls_clients.clone(),
            parameters: self.parameters.clone(),
            own_key: self.own_key.clone(),
            storage: self.storage.clone(),
            max_size: self.max_size,
        }
    }
}

pub struct DownloadService<S: ObjectStore> {
    storage: Arc<S>,
    own_key: NetworkPublicKey,
}

impl<S: ObjectStore> Clone for DownloadService<S> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            own_key: self.own_key.clone(),
        }
    }
}

impl<S: ObjectStore> DownloadService<S> {
    pub fn new(storage: Arc<S>, own_key: NetworkPublicKey) -> Self {
        Self { storage, own_key }
    }
    pub(crate) async fn handle_download_object(
        &self,
        path: &ObjectPath,
    ) -> Result<impl IntoResponse, StatusCode> {
        let reader = self
            .storage
            .get(&path.path())
            .await
            .map_err(|_| StatusCode::NOT_FOUND)?;

        // Wrap the stream in StreamBody
        let body = Body::from_stream(reader.into_stream());

        // Build the response
        let response = Response::builder()
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(body)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        Ok(response)
    }
    pub(crate) fn verify_signed_params(&self, params: &SignedParams) -> ObjectResult<()> {
        params
            .verify(&self.own_key)
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))
    }
}

pub trait ObjectServiceManager<S>: Send + Sync + Sized
where
    S: ObjectStore,
{
    // fn client(&self) -> Arc<DownloadClient<S>>;
    /// Starts the network services
    async fn start(&mut self, address: &Multiaddr, download_service: DownloadService<S>);
    /// Stops the network services
    async fn stop(&mut self);
}
