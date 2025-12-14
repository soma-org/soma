use crate::inference::module::{
    ModuleClient, ModuleInput, ModuleInputAPI, ModuleOutput, ModuleOutputV1,
};
use async_trait::async_trait;
use object_store::local::LocalFileSystem;
use reqwest::{Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use types::{
    error::{InferenceError, InferenceResult},
    shard_crypto::keys::EncoderPublicKey,
};
use url::Url;

pub struct JsonModuleClient {
    url: Url,
    client: Client,
    storage: Arc<LocalFileSystem>,
    own_encoder: EncoderPublicKey,
}

impl JsonModuleClient {
    pub fn new(
        url: Url,
        storage: Arc<LocalFileSystem>,
        own_encoder: EncoderPublicKey,
    ) -> InferenceResult<Self> {
        Ok(Self {
            url,
            client: ClientBuilder::new()
                .pool_idle_timeout(None)
                .build()
                .map_err(InferenceError::ReqwestError)?,
            storage,
            own_encoder,
        })
    }
}

#[derive(Serialize)]
struct JsonCallRequest {
    input_path: String,
    output_path: String,
}

#[derive(Deserialize)]
struct JsonCallResponse {
    probe_encoder: Option<EncoderPublicKey>,
}

#[async_trait]
impl ModuleClient<LocalFileSystem> for JsonModuleClient {
    async fn call(&self, input: ModuleInput, timeout: Duration) -> InferenceResult<ModuleOutput> {
        let input_filepath = self
            .storage
            .path_to_filesystem(&input.input_path().path())
            .map_err(InferenceError::ObjectStoreError)?
            .to_string_lossy()
            .into_owned();

        let output_filepath = self
            .storage
            .path_to_filesystem(&input.output_path().path())
            .map_err(InferenceError::ObjectStoreError)?
            .to_string_lossy()
            .into_owned();

        let request_data = JsonCallRequest {
            input_path: input_filepath,
            output_path: output_filepath,
        };
        let response = self
            .client
            .post(self.url.clone())
            .timeout(timeout)
            .json(&request_data)
            .send()
            .await
            .map_err(InferenceError::ReqwestError)?;

        let response_data: JsonCallResponse = response
            .json()
            .await
            .map_err(InferenceError::ReqwestError)?;

        if let Some(probe_encoder) = response_data.probe_encoder {
            return Ok(ModuleOutput::V1(ModuleOutputV1::new(probe_encoder)));
        }

        return Ok(ModuleOutput::V1(ModuleOutputV1::new(
            self.own_encoder.clone(),
        )));
    }
}
