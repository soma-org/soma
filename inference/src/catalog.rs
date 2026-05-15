// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `ModelCard` mirrors the on-chain `Manifest` content schema (plus pricing
//! and a `SomaInfo` pointer back to the provider). Providers publish their
//! offerings via `/v1/models`; the proxy aggregates across providers.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    #[serde(default)]
    pub canonical_slug: Option<String>,
    #[serde(default)]
    pub hugging_face_id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub context_length: u64,
    pub architecture: Architecture,
    pub pricing: Pricing,
    pub top_provider: TopProvider,
    #[serde(default)]
    pub supported_parameters: Vec<String>,
    #[serde(default)]
    pub default_parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub expiration_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soma: Option<SomaInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Architecture {
    #[serde(default)]
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub output_modalities: Vec<String>,
    #[serde(default)]
    pub tokenizer: String,
    #[serde(default)]
    pub instruct_type: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pricing {
    #[serde(default = "zero_price")]
    pub prompt: String,
    #[serde(default = "zero_price")]
    pub completion: String,
    #[serde(default = "zero_price")]
    pub request: String,
    #[serde(default = "zero_price")]
    pub image: String,
    #[serde(default = "zero_price")]
    pub input_cache_read: String,
    #[serde(default = "zero_price")]
    pub input_cache_write: String,
}

fn zero_price() -> String {
    "0".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopProvider {
    #[serde(default)]
    pub context_length: u64,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub is_moderated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SomaInfo {
    pub provider_address: String,
    pub endpoint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelCard>,
}
