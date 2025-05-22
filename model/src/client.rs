use crate::{ByteRange, ModelInput, ModelOutput, ModelOutputV1};
use async_trait::async_trait;
use ndarray::array;
use reqwest::Client;
use shared::error::{ModelError, ModelResult};
use soma_network::multiaddr::{to_host_port_str, Multiaddr};
use std::{str::FromStr, time::Duration};
use url::Url;

#[async_trait]
pub trait ModelClient: Send + Sync + Sized + 'static {
    async fn call(&self, model_input: ModelInput, timeout: Duration) -> ModelResult<ModelOutput>;
}

pub struct MockModelClient {}

#[async_trait]
impl ModelClient for MockModelClient {
    async fn call(&self, model_input: ModelInput, timeout: Duration) -> ModelResult<ModelOutput> {
        Ok(ModelOutput::V1(ModelOutputV1 {
            embeddings: array![
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            byte_ranges: vec![
                ByteRange { start: 0, end: 0 },
                ByteRange { start: 0, end: 0 },
                ByteRange { start: 0, end: 0 },
                ByteRange { start: 0, end: 0 },
                ByteRange { start: 0, end: 0 },
                ByteRange { start: 0, end: 0 },
                ByteRange { start: 0, end: 0 },
            ],
        }))
    }
}

pub(crate) struct HttpModelClient {
    url: Url,
    client: Client,
}

impl HttpModelClient {
    pub(crate) fn new(address: Multiaddr) -> ModelResult<Self> {
        let address = to_host_port_str(&address).map_err(|e| {
            ModelError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        let address = format!("http://{address}");
        let url = Url::from_str(&address).map_err(|e| ModelError::UrlParseError(e.to_string()))?;
        let client = Client::new();
        Ok(Self { url, client })
    }
}

#[async_trait]
impl ModelClient for HttpModelClient {
    async fn call(&self, model_input: ModelInput, timeout: Duration) -> ModelResult<ModelOutput> {
        let response = self
            .client
            .post(self.url.clone())
            .timeout(timeout)
            .json(&model_input)
            .send()
            .await
            .map_err(|e| ModelError::NetworkRequestError(e.to_string()))?;

        let model_output = response
            .json::<ModelOutput>()
            .await
            .map_err(|e| ModelError::DeserializeError(e.to_string()))?;

        Ok(model_output)
    }
}
