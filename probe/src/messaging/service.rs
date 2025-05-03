use crate::{
    error::{ProbeError, ProbeResult},
    EmbeddingAPI, ProbeInput, ProbeInputAPI, ProbeOutput, ProbeOutputV1, ScoreV1,
};

use super::ProbeService;
use async_trait::async_trait;
use bytes::Bytes;
use shared::crypto::keys::EncoderPublicKey;

pub struct MockProbeService {}

impl MockProbeService {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl ProbeService for MockProbeService {
    async fn handle_probe(&self, probe_input_bytes: Bytes) -> ProbeResult<Bytes> {
        let probe_input: ProbeInput =
            bcs::from_bytes(&probe_input_bytes).map_err(ProbeError::MalformedType)?;

        let mut encoders: Vec<EncoderPublicKey> = probe_input
            .embeddings()
            .iter()
            .map(|e| e.encoder().clone())
            .collect();

        encoders.sort();

        let scores = encoders
            .iter()
            .enumerate()
            .map(|(i, e)| ScoreV1::new(e.to_owned(), i as u8))
            .collect();

        let score_set = ProbeOutput::V1(ProbeOutputV1::new(scores));

        let score_set_bytes =
            Bytes::from(bcs::to_bytes(&score_set).map_err(ProbeError::MalformedType)?);
        Ok(score_set_bytes)
    }
}
