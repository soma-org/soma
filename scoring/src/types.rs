// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use types::checksum::Checksum;
use types::metadata::{Manifest, ManifestV1, Metadata, MetadataV1};
use url::Url;

/// Serde module for `Vec<f32>` that serializes each f32 as its u32 bit
/// representation, making the type compatible with BCS (which does not
/// natively support floating-point types).
mod vec_f32_as_u32_bits {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(vals: &[f32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bits: Vec<u32> = vals.iter().map(|f| f.to_bits()).collect();
        bits.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bits = Vec::<u32>::deserialize(deserializer)?;
        Ok(bits.into_iter().map(f32::from_bits).collect())
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ManifestInput {
    pub url: String,
    pub checksum: String,
    pub size: usize,
    /// Optional Base58-encoded AES-256 decryption key (32 bytes).
    #[serde(default)]
    pub decryption_key: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ScoreRequest {
    pub data_url: String,
    pub data_checksum: String,
    pub data_size: usize,
    pub model_manifests: Vec<ManifestInput>,
    #[serde(with = "vec_f32_as_u32_bits")]
    pub target_embedding: Vec<f32>,
    pub seed: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ScoreResponse {
    pub winner: usize,
    #[serde(with = "vec_f32_as_u32_bits")]
    pub loss_score: Vec<f32>,
    #[serde(with = "vec_f32_as_u32_bits")]
    pub embedding: Vec<f32>,
    #[serde(with = "vec_f32_as_u32_bits")]
    pub distance: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HealthRequest {}

#[derive(Debug, Deserialize, Serialize)]
pub struct HealthResponse {
    pub ok: bool,
}

pub fn parse_base58_checksum(base58_str: &str) -> Result<Checksum> {
    use fastcrypto::encoding::{Base58, Encoding};
    let bytes = Base58::decode(base58_str)
        .map_err(|e| anyhow::anyhow!("Invalid base58 checksum: {}", e))?;
    ensure!(bytes.len() == 32, "Checksum must be 32 bytes, got {}", bytes.len());
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(Checksum::new_from_hash(arr))
}

pub fn manifest_from_input(url: &str, checksum_base58: &str, size: usize) -> Result<Manifest> {
    let parsed_url = Url::parse(url)?;
    let checksum = parse_base58_checksum(checksum_base58)?;
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
                checksum: "0xfedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
                    .to_string(),
                size: 5242880,
                decryption_key: None,
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
    fn parse_base58_checksum_valid() {
        // Base58-encoded 32 bytes
        use fastcrypto::encoding::{Base58, Encoding};
        let base58 = Base58::encode([0xab; 32]);
        assert!(parse_base58_checksum(&base58).is_ok());
    }

    #[test]
    fn parse_base58_checksum_wrong_length() {
        // Too short
        use fastcrypto::encoding::{Base58, Encoding};
        let base58 = Base58::encode([0xab; 16]);
        assert!(parse_base58_checksum(&base58).is_err());
    }

    #[test]
    fn parse_base58_checksum_invalid() {
        // Invalid base58 characters (0, O, I, l are not in base58)
        assert!(parse_base58_checksum("0OIl").is_err());
    }

    #[test]
    fn manifest_from_input_valid() {
        use fastcrypto::encoding::{Base58, Encoding};
        let checksum_base58 = Base58::encode([0xab; 32]);
        let result = manifest_from_input("https://example.com/data.bin", &checksum_base58, 1024);
        assert!(result.is_ok());
    }

    #[test]
    fn manifest_from_input_bad_url() {
        use fastcrypto::encoding::{Base58, Encoding};
        let checksum_base58 = Base58::encode([0xab; 32]);
        let result = manifest_from_input("not a url", &checksum_base58, 1024);
        assert!(result.is_err());
    }
}
