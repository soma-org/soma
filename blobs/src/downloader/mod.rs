use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use object_store::ObjectStore;
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use tokio::sync::Semaphore;
use types::{
    error::{BlobError, BlobResult},
    metadata::{Manifest, ManifestAPI},
    parameters::HttpParameters,
};
use url::Url;

use crate::engine::BlobEngine;
use crate::{BlobPath, engine::BlobReader};

#[async_trait]
pub trait BlobDownloader: Send + Sync + 'static {
    async fn download(&self, manifest: &Manifest, blob_path: BlobPath) -> BlobResult<()>;
}

pub struct HttpBlobDownloader {
    client: ClientWithMiddleware,
    dest: Arc<dyn ObjectStore>,
    engine: BlobEngine,
}

impl HttpBlobDownloader {
    pub fn new(
        parameters: &HttpParameters,
        dest: Arc<dyn ObjectStore>,
        semaphore: Arc<Semaphore>,
        chunk_size: u64,
        ns_per_byte: u16,
    ) -> BlobResult<Self> {
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
        let engine = BlobEngine::new(semaphore, chunk_size, ns_per_byte)?;
        Ok(Self { client, dest, engine })
    }
}

#[async_trait]
impl BlobDownloader for HttpBlobDownloader {
    async fn download(&self, manifest: &Manifest, blob_path: BlobPath) -> BlobResult<()> {
        let reader =
            Arc::new(HttpReader { url: manifest.url().clone(), client: self.client.clone() });
        self.engine
            .download(reader, self.dest.clone(), blob_path, manifest.metadata().clone())
            .await
    }
}

pub struct HttpReader {
    url: Url,
    client: ClientWithMiddleware,
}

impl HttpReader {
    pub fn new(url: Url, client: ClientWithMiddleware) -> Self {
        Self { url, client }
    }
}

#[async_trait]
impl BlobReader for HttpReader {
    async fn get_full(&self, timeout: Duration) -> BlobResult<Bytes> {
        let req = self.client.get(self.url.clone()).timeout(timeout);
        let response = req.send().await.map_err(|e| BlobError::NetworkRequest(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BlobError::HttpStatus {
                status: response.status().as_u16(),
                url: self.url.to_string(),
            });
        }

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
        let status = response.status();
        if status != reqwest::StatusCode::PARTIAL_CONTENT && status != reqwest::StatusCode::OK {
            return Err(BlobError::HttpStatus {
                status: status.as_u16(),
                url: self.url.to_string(),
            });
        }
        let bytes = response.bytes().await.map_err(|e| BlobError::NetworkRequest(e.to_string()))?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    fn test_client() -> ClientWithMiddleware {
        ClientBuilder::new(reqwest::Client::new()).build()
    }

    #[tokio::test]
    async fn get_full_returns_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/blob"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"hello blob"))
            .mount(&server)
            .await;

        let url = Url::parse(&format!("{}/blob", server.uri())).unwrap();
        let reader = HttpReader::new(url, test_client());
        let result = reader.get_full(Duration::from_secs(5)).await.unwrap();
        assert_eq!(result.as_ref(), b"hello blob");
    }

    #[tokio::test]
    async fn get_full_returns_error_on_404() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/blob"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let url = Url::parse(&format!("{}/blob", server.uri())).unwrap();
        let reader = HttpReader::new(url.clone(), test_client());
        let err = reader.get_full(Duration::from_secs(5)).await.unwrap_err();
        match err {
            BlobError::HttpStatus { status, url: u } => {
                assert_eq!(status, 404);
                assert_eq!(u, url.to_string());
            }
            other => panic!("expected HttpStatus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn get_range_sends_range_header() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/blob"))
            .and(header("range", "bytes=10-49"))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(b"partial content"))
            .mount(&server)
            .await;

        let url = Url::parse(&format!("{}/blob", server.uri())).unwrap();
        let reader = HttpReader::new(url, test_client());
        let result = reader.get_range(10..50, Duration::from_secs(5)).await.unwrap();
        assert_eq!(result.as_ref(), b"partial content");
    }

    #[tokio::test]
    async fn get_range_returns_error_on_500() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/blob"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let url = Url::parse(&format!("{}/blob", server.uri())).unwrap();
        let reader = HttpReader::new(url, test_client());
        let err = reader.get_range(0..10, Duration::from_secs(5)).await.unwrap_err();
        assert!(matches!(err, BlobError::HttpStatus { status: 500, .. }));
    }

    #[tokio::test]
    async fn get_full_connection_refused() {
        let url = Url::parse("http://127.0.0.1:1/blob").unwrap();
        let reader = HttpReader::new(url, test_client());
        let err = reader.get_full(Duration::from_secs(2)).await.unwrap_err();
        assert!(matches!(err, BlobError::NetworkRequest(_)));
    }
}
