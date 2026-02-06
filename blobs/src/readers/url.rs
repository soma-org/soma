use async_trait::async_trait;
use bytes::Bytes;
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use std::{ops::Range, sync::Arc, time::Duration};
use tracing::info;
use types::{
    error::{BlobError, BlobResult},
    metadata::{Manifest, ManifestAPI},
    parameters::HttpParameters,
};
use url::Url;

use crate::readers::BlobReader;

#[derive(Clone)]
pub struct BlobClient {
    client: ClientWithMiddleware,
}

impl BlobClient {
    pub fn new(parameters: Arc<HttpParameters>) -> BlobResult<Self> {
        let buffer_size = parameters.connection_buffer_size;
        let retry_policy = ExponentialBackoff::builder().build_with_max_retries(3);
        let client = reqwest::Client::builder()
            .http2_initial_connection_window_size(Some(buffer_size as u32))
            .http2_initial_stream_window_size(Some(buffer_size as u32 / 2))
            .http2_keep_alive_while_idle(true)
            .http2_keep_alive_interval(parameters.keepalive_interval)
            .http2_keep_alive_timeout(parameters.keepalive_interval)
            .connect_timeout(parameters.connect_timeout)
            .user_agent("SOMA (Blob Client)")
            .build()
            .map_err(|e| BlobError::ReqwestError(e.to_string()))?;

        let client = ClientBuilder::new(client)
            .with(RetryTransientMiddleware::new_with_policy(retry_policy))
            .build();

        Ok(Self { client })
    }

    pub async fn get_reader(&self, manifest: &Manifest) -> BlobResult<BlobHttpReader> {
        let client = self.client.clone();
        Ok(BlobHttpReader { url: manifest.url().clone(), client })
    }
}

pub struct BlobHttpReader {
    url: Url,
    client: ClientWithMiddleware,
}

impl BlobHttpReader {
    pub fn new(url: Url, client: ClientWithMiddleware) -> Self {
        Self { url, client }
    }
}

#[async_trait]
impl BlobReader for BlobHttpReader {
    async fn get_full(&self, timeout: Duration) -> BlobResult<Bytes> {
        info!("get_full_called");
        let req = self.client.get(self.url.clone()).timeout(timeout);
        let response = req.send().await.map_err(|e| BlobError::NetworkRequest(e.to_string()))?;

        info!("{:?}", response);
        if !response.status().is_success() {
            return Err(BlobError::NetworkRequest("status threw error".to_string()));
        }

        info!("response non error?");
        let bytes = response.bytes().await.map_err(|e| BlobError::NetworkRequest(e.to_string()))?;
        Ok(bytes)
    }
    async fn get_range(&self, range: Range<usize>, timeout: Duration) -> BlobResult<Bytes> {
        let range_header = format!("bytes={}-{}", range.start, range.end - 1);
        let req = self
            .client
            .get(self.url.clone())
            .header(reqwest::header::RANGE, range_header)
            .timeout(timeout);

        let response = req.send().await.map_err(|e| BlobError::NetworkRequest(format!("{e:?}")))?;
        info!("{:?}", response);
        if !response.status().is_success() {
            return Err(BlobError::NetworkRequest("status threw error".to_string()));
        }
        let bytes = response.bytes().await.map_err(|e| BlobError::NetworkRequest(e.to_string()))?;
        Ok(bytes)
    }
}
