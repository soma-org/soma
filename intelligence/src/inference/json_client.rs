use async_trait::async_trait;
use reqwest::{Client, ClientBuilder, Url};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use types::{
    checksum::Checksum,
    error::{IntelligenceError, IntelligenceResult},
    evaluation::{ProbeSet, ProbeSetV1, ProbeWeightV1},
    metadata::{Metadata, MetadataAPI, MetadataV1, ObjectPath},
    shard_crypto::keys::EncoderPublicKey,
};

use crate::inference::{
    InferenceClient, InferenceInput, InferenceInputAPI, InferenceOutput, InferenceOutputV1,
};

pub struct JSONClient {
    object_server_url: Url,
    webhook_url: Url,
    client: Client,
}

impl JSONClient {
    pub fn new(object_server_url: Url, webhook_url: Url) -> IntelligenceResult<Self> {
        Ok(Self {
            object_server_url,
            webhook_url,
            client: ClientBuilder::new()
                .pool_idle_timeout(None)
                .build()
                .map_err(IntelligenceError::ReqwestError)?,
        })
    }
}

#[derive(Serialize)]
struct JsonCallRequest {
    download_url: String,
    upload_url: String,
    checksum: Checksum,
    size: u64,
    epoch: u64,
}

#[derive(Deserialize)]
struct JsonCallResponse {
    checksum: Checksum,
    size: u64,
    probe_set: Vec<JsonProbeWeight>,
}

#[derive(Deserialize)]
struct JsonProbeWeight {
    encoder: EncoderPublicKey,
    weight: u64,
}

#[async_trait]
impl InferenceClient for JSONClient {
    async fn call(
        &self,
        input: InferenceInput,
        timeout: Duration,
    ) -> IntelligenceResult<InferenceOutput> {
        let full_url = self
            .webhook_url
            .join("/call")
            .map_err(|e| IntelligenceError::ParseError(e.to_string()))?;

        let download_url = self
            .object_server_url
            .join(&format!("/{}", input.metadata().checksum()))
            .map_err(|e| IntelligenceError::ParseError(e.to_string()))?
            .to_string();

        let upload_url = self
            .object_server_url
            .join("/upload")
            .map_err(|e| IntelligenceError::ParseError(e.to_string()))?
            .to_string();

        let request_data = JsonCallRequest {
            download_url,
            upload_url,
            checksum: input.metadata().checksum(),
            size: input.metadata().size(),
            epoch: input.epoch(),
        };
        let response = self
            .client
            .post(full_url)
            .timeout(timeout)
            .json(&request_data)
            .send()
            .await
            .map_err(IntelligenceError::ReqwestError)?;

        let response_data: JsonCallResponse = response
            .json()
            .await
            .map_err(IntelligenceError::ReqwestError)?;

        let probe_weights: Vec<ProbeWeightV1> = response_data
            .probe_set
            .into_iter()
            .map(|pw| ProbeWeightV1::new(pw.encoder, pw.weight))
            .collect();

        Ok(InferenceOutput::V1(InferenceOutputV1::new(
            Metadata::V1(MetadataV1::new(
                ObjectPath::Tmp(input.epoch(), response_data.checksum),
                response_data.size,
            )),
            ProbeSet::V1(ProbeSetV1::new(probe_weights)),
        )))
    }
}
