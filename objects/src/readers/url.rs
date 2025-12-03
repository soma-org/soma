use async_trait::async_trait;
use bytes::Bytes;
use quick_cache::sync::Cache;
use reqwest::Client;
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use soma_tls::create_rustls_client_config;
use std::{ops::Range, sync::Arc, time::Duration};
use tracing::info;
use types::{
    crypto::{NetworkKeyPair, NetworkPublicKey},
    error::{ObjectError, ObjectResult},
    metadata::{DownloadMetadata, MtlsDownloadMetadataAPI},
    parameters::HttpParameters,
    shard_networking::CERTIFICATE_NAME,
};
use url::Url;

use crate::readers::ObjectReader;

#[derive(Clone)]
pub struct ObjectHttpClient {
    client: ClientWithMiddleware,
    mtls_clients: Arc<Cache<NetworkPublicKey, ClientWithMiddleware>>,
    own_key: NetworkKeyPair,
    parameters: Arc<HttpParameters>,
}

impl ObjectHttpClient {
    pub fn new(own_key: NetworkKeyPair, parameters: Arc<HttpParameters>) -> ObjectResult<Self> {
        let buffer_size = parameters.connection_buffer_size;
        let retry_policy = ExponentialBackoff::builder().build_with_max_retries(3);
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

        let client = ClientBuilder::new(client)
            .with(RetryTransientMiddleware::new_with_policy(retry_policy))
            .build();

        Ok(Self {
            client,
            mtls_clients: Arc::new(Cache::new(parameters.client_pool_capacity)),
            own_key,
            parameters,
        })
    }

    pub async fn get_client(
        &self,
        download_metadata: &DownloadMetadata,
    ) -> ObjectResult<ClientWithMiddleware> {
        Ok(match download_metadata {
            DownloadMetadata::Default(_dm) => self.client.clone(),
            DownloadMetadata::Mtls(dm) => {
                let peer = dm.peer();
                if let Some(client) = self.mtls_clients.get(peer) {
                    client
                } else {
                    let buffer_size = self.parameters.connection_buffer_size;
                    let retry_policy = ExponentialBackoff::builder().build_with_max_retries(3);

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
                    let client = ClientBuilder::new(client)
                        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
                        .build();

                    self.mtls_clients.insert(peer.clone(), client.clone());
                    client
                }
            }
        })
    }

    pub async fn get_reader(
        &self,
        download_metadata: &DownloadMetadata,
    ) -> ObjectResult<ObjectHttpReader> {
        let client = self.get_client(download_metadata).await?;
        Ok(ObjectHttpReader {
            url: download_metadata.url().clone(),
            client,
        })
    }
}

pub struct ObjectHttpReader {
    url: Url,
    client: ClientWithMiddleware,
}

impl ObjectHttpReader {
    pub fn new(url: Url, client: ClientWithMiddleware) -> Self {
        Self { url, client }
    }
}

#[async_trait]
impl ObjectReader for ObjectHttpReader {
    async fn get_full(&self, timeout: Duration) -> ObjectResult<Bytes> {
        info!("get_full_called");
        let req = self.client.get(self.url.clone()).timeout(timeout);
        let response = req
            .send()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?;

        info!("{:?}", response);
        if !response.status().is_success() {
            return Err(ObjectError::NetworkRequest(
                "status threw error".to_string(),
            ));
        }

        info!("response non error?");
        let bytes = response
            .bytes()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?;
        Ok(bytes)
    }
    async fn get_range(&self, range: Range<usize>, timeout: Duration) -> ObjectResult<Bytes> {
        let range_header = format!("bytes={}-{}", range.start, range.end - 1);
        let req = self
            .client
            .get(self.url.clone())
            .header(reqwest::header::RANGE, range_header)
            .timeout(timeout);

        println!("{:?}", req);
        let response = req
            .send()
            .await
            .map_err(|e| ObjectError::NetworkRequest(format!("{e:?}")))?;
        info!("{:?}", response);
        if !response.status().is_success() {
            return Err(ObjectError::NetworkRequest(
                "status threw error".to_string(),
            ));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?;
        Ok(bytes)
    }
}
