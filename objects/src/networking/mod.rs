pub mod downloader;
pub mod external_service;
pub mod internal_service;
pub mod tus;

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header, Response, StatusCode},
    response::IntoResponse,
};
use fastcrypto::hash::HashFunction;
use reqwest::Client;
use std::{sync::Arc, time::Duration};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio_util::io::ReaderStream;
use tracing::warn;
use types::{
    checksum::Checksum,
    crypto::{DefaultHash as DefaultHashFunction, NetworkPublicKey},
    error::ObjectError,
    metadata::{DownloadableMetadataAPI, MetadataAPI, ObjectPath, SignedParams},
    multiaddr::Multiaddr,
};

use crate::storage::ObjectStorage;
use types::error::ObjectResult;
use types::metadata::DownloadableMetadata;

#[async_trait]
pub trait ClientPool: Send + Sync + 'static {
    async fn get_client(
        &self,
        downloadable_metadata: &DownloadableMetadata,
    ) -> ObjectResult<Client>;
    fn calculate_timeout(&self, num_bytes: u64) -> Duration;
}

pub struct ObjectClient<C: ClientPool> {
    client_pool: C,
}

impl<C: ClientPool> ObjectClient<C> {
    pub fn new(client_pool: C) -> ObjectResult<Self> {
        Ok(Self { client_pool })
    }
    async fn download_object<W>(
        &self,
        writer: &mut W,
        downloadable_metadata: &DownloadableMetadata,
    ) -> ObjectResult<()>
    where
        W: AsyncWrite + Unpin + Send,
    {
        let url = downloadable_metadata
            .url()
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?;
        let timeout = self
            .client_pool
            .calculate_timeout(downloadable_metadata.metadata().size());

        let mut response = self
            .client_pool
            .get_client(&downloadable_metadata)
            .await?
            .get(url.clone())
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?;

        warn!("response: {:?}", response);

        if !response.status().is_success() {
            return Err(ObjectError::NetworkRequest(format!(
                "http status: {}",
                response.status().as_u16()
            )));
        }
        let mut hasher = DefaultHashFunction::new();
        let mut total_size = 0_u64;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?
        {
            hasher.update(&chunk);
            total_size += chunk.len() as u64;
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

        if total_size != downloadable_metadata.metadata().size() {
            return Err(ObjectError::VerificationError(format!(
                "Size mismatch: expected {}, got {}",
                downloadable_metadata.metadata().size(),
                total_size
            )));
        }

        // Verify checksum
        if checksum != downloadable_metadata.metadata().checksum() {
            return Err(ObjectError::VerificationError(format!(
                "Checksum mismatch: expected {}, got {}",
                downloadable_metadata.metadata().checksum(),
                checksum
            )));
        }
        return Ok(());
    }
}

#[derive(Clone)]
pub struct ObjectService<S: ObjectStorage> {
    storage: Arc<S>,
    own_key: NetworkPublicKey,
}

impl<S: ObjectStorage> ObjectService<S> {
    pub fn new(storage: Arc<S>, own_key: NetworkPublicKey) -> Self {
        Self { storage, own_key }
    }
    pub(crate) async fn handle_download_object(
        &self,
        path: &ObjectPath,
    ) -> Result<impl IntoResponse, StatusCode> {
        let reader = self
            .storage
            .stream_object(path)
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
    pub(crate) fn verify_signed_params(&self, params: &SignedParams) -> ObjectResult<()> {
        params
            .verify(&self.own_key)
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))
    }
}

pub trait ObjectServiceManager<S>: Send + Sync + Sized
where
    S: ObjectStorage,
{
    type ClientPool: ClientPool;

    fn client(&self) -> Arc<ObjectClient<Self::ClientPool>>;
    /// Starts the network services
    async fn start(&mut self, address: &Multiaddr, service: ObjectService<S>);
    /// Stops the network services
    async fn stop(&mut self);
}
