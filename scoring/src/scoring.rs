use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, bail};
use burn::tensor::TensorData;
use runtime::{ManifestCompetitionInput, ModelConfig, RuntimeAPI, build_runtime};
use types::config::node_config::DeviceConfig;

use crate::types::{ScoreRequest, ScoreResponse, manifest_from_input};

pub struct ScoringEngine {
    runtime: Arc<dyn RuntimeAPI>,
}

impl ScoringEngine {
    pub fn new(data_dir: &Path, model_config: ModelConfig, device: &DeviceConfig) -> Result<Self> {
        let runtime = build_runtime(device, data_dir, model_config)?;
        Ok(Self { runtime })
    }

    pub async fn score(&self, request: ScoreRequest) -> Result<ScoreResponse> {
        if request.model_manifests.is_empty() {
            bail!("At least one model manifest is required");
        }
        if request.target_embedding.is_empty() {
            bail!("Target embedding must not be empty");
        }

        let data_manifest =
            manifest_from_input(&request.data_url, &request.data_checksum, request.data_size)?;

        let model_manifests = request
            .model_manifests
            .iter()
            .map(|m| manifest_from_input(&m.url, &m.checksum, m.size))
            .collect::<Result<Vec<_>>>()?;

        let model_keys = request
            .model_manifests
            .iter()
            .map(|m| {
                m.decryption_key.as_ref().map(|hex_key| {
                    let stripped = hex_key.strip_prefix("0x").unwrap_or(hex_key);
                    let bytes = hex::decode(stripped)
                        .expect("invalid hex in decryption_key");
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&bytes);
                    arr
                })
            })
            .collect::<Vec<_>>();

        let target = TensorData::new(
            request.target_embedding.clone(),
            [request.target_embedding.len()],
        );

        let input =
            ManifestCompetitionInput::new(data_manifest, model_manifests, target, request.seed)
                .with_model_keys(model_keys);

        let output = self
            .runtime
            .manifest_competition(input)
            .await
            .map_err(|e| anyhow::anyhow!("Runtime competition error: {e}"))?;

        let loss_score = output
            .loss_score()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert loss_score: {e:?}"))?;
        let embedding = output
            .embedding()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert embedding: {e:?}"))?;
        let distance = output
            .distance()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert distance: {e:?}"))?;

        Ok(ScoreResponse {
            winner: output.winner(),
            loss_score,
            embedding,
            distance,
        })
    }
}
