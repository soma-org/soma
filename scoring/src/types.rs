use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use types::checksum::Checksum;
use types::metadata::{Manifest, ManifestV1, Metadata, MetadataV1};
use url::Url;

#[derive(Debug, Deserialize, Serialize)]
pub struct ManifestInput {
    pub url: String,
    pub checksum: String,
    pub size: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ScoreRequest {
    pub data_url: String,
    pub data_checksum: String,
    pub data_size: usize,
    pub model_manifests: Vec<ManifestInput>,
    pub target_embedding: Vec<f32>,
    pub seed: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ScoreResponse {
    pub winner: usize,
    pub loss_score: Vec<f32>,
    pub embedding: Vec<f32>,
    pub distance: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

pub fn parse_hex_checksum(hex_str: &str) -> Result<Checksum> {
    let stripped = hex_str.strip_prefix("0x").unwrap_or(hex_str);
    let bytes = hex::decode(stripped)?;
    ensure!(bytes.len() == 32, "Checksum must be 32 bytes, got {}", bytes.len());
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(Checksum::new_from_hash(arr))
}

pub fn manifest_from_input(url: &str, checksum_hex: &str, size: usize) -> Result<Manifest> {
    let parsed_url = Url::parse(url)?;
    let checksum = parse_hex_checksum(checksum_hex)?;
    let metadata = Metadata::V1(MetadataV1::new(checksum, size));
    Ok(Manifest::V1(ManifestV1::new(parsed_url, metadata)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_request_round_trip() {
        let request = ScoreRequest {
            data_url: "https://example.com/data.bin".to_string(),
            data_checksum: "0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .to_string(),
            data_size: 1024,
            model_manifests: vec![ManifestInput {
                url: "https://example.com/weights.safetensors".to_string(),
                checksum:
                    "0xfedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
                        .to_string(),
                size: 5242880,
            }],
            target_embedding: vec![0.1, 0.2, 0.3],
            seed: 42,
        };
        let json = serde_json::to_string(&request).expect("serialize");
        let deserialized: ScoreRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.seed, 42);
        assert_eq!(deserialized.model_manifests.len(), 1);
        assert_eq!(deserialized.target_embedding.len(), 3);
    }

    #[test]
    fn score_response_serialization() {
        let response = ScoreResponse {
            winner: 0,
            loss_score: vec![0.123],
            embedding: vec![0.1, 0.2],
            distance: vec![0.456],
        };
        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains("\"winner\":0"));
        assert!(json.contains("\"distance\""));
    }

    #[test]
    fn parse_hex_checksum_valid() {
        let hex = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        assert!(parse_hex_checksum(hex).is_ok());
    }

    #[test]
    fn parse_hex_checksum_with_0x_prefix() {
        let hex = "0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        assert!(parse_hex_checksum(hex).is_ok());
    }

    #[test]
    fn parse_hex_checksum_wrong_length() {
        assert!(parse_hex_checksum("abcdef").is_err());
    }

    #[test]
    fn parse_hex_checksum_invalid_hex() {
        let hex = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";
        assert!(parse_hex_checksum(hex).is_err());
    }

    #[test]
    fn manifest_from_input_valid() {
        let result = manifest_from_input(
            "https://example.com/data.bin",
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            1024,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn manifest_from_input_bad_url() {
        let result = manifest_from_input(
            "not a url",
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            1024,
        );
        assert!(result.is_err());
    }
}
