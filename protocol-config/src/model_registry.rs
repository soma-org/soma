// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Protocol-governed model registry.
//!
//! The set of canonical model identities is part of the protocol itself,
//! not a per-instance config knob: every validator running the same
//! binary at the same protocol version agrees on the *exact* set of
//! `ModelRegistryEntry` rows. Adding or deprecating a model is a
//! protocol upgrade — a new `ProtocolVersion` with a fresh
//! `entries_for_version(...)` arm — so the change is gated by the same
//! validator-stake quorum that gates any other consensus rule.
//!
//! ## What lives here vs. where else
//!
//! - **Model identity sheet** (this module): canonical slug, capabilities
//!   (modalities, tokenizer family, context limits). Model-intrinsic.
//!   Slow-changing.
//! - **Per-(provider, model) offering** (`types::offering`): prices, SLA
//!   bounds, active flag. Provider-controlled, mutable per-tx.
//! - **Per-channel snapshot** (`types::channel`): the offering's prices
//!   and SLA frozen at open time. Pinned for the channel's lifetime so
//!   settlement math is deterministic.
//!
//! Splitting on those three axes keeps each layer's update cadence
//! matched to the kind of change it represents (binary upgrade, tx,
//! one-shot at open).

use serde::{Deserialize, Serialize};

use crate::ProtocolVersion;

/// Family of tokenizer used by a model. Buyer's proxy uses this to pre-count
/// input tokens (worst-case sizing of `cumulative_amount` before sending the
/// request) without having to host every tokenizer for every model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerFamily {
    /// Claude/Anthropic — `claude_tokenizer` heuristic ≈ tiktoken `cl100k_base`.
    Claude,
    /// GPT-3.5/4/5 family — tiktoken `cl100k_base` / `o200k_base`.
    Gpt,
    /// Gemini — SentencePiece, ≈ chars/4 fallback if tokenizer unavailable.
    Gemini,
    /// Llama 3/4 — tiktoken-compatible `cl100k`-derivative.
    Llama,
    /// Mistral — SentencePiece, same fallback as Gemini.
    Mistral,
    /// Qwen — tiktoken-compatible.
    Qwen,
    /// DeepSeek — tiktoken-compatible.
    Deepseek,
    /// Grok / xAI — tiktoken-compatible.
    Grok,
    /// Unknown / not yet classified. Proxy falls back to chars/4 estimation.
    Unknown,
}

/// Bitset of supported input/output modalities. Powers of two so a model
/// supporting "text + image" is `Text as u16 | Image as u16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Modality {
    Text = 1 << 0,
    Image = 1 << 1,
    Audio = 1 << 2,
    Video = 1 << 3,
    File = 1 << 4,
    Embedding = 1 << 5,
}

impl Modality {
    pub const fn bit(self) -> u16 {
        self as u16
    }
}

/// Bitset of supported features (tools, JSON mode, reasoning, etc.).
/// Mirrors OpenRouter's `supported_features` list — providers register
/// offerings against models declared with a superset of the features
/// they intend to advertise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ModelFeature {
    Tools = 1 << 0,
    JsonMode = 1 << 1,
    StructuredOutputs = 1 << 2,
    Logprobs = 1 << 3,
    WebSearch = 1 << 4,
    Reasoning = 1 << 5,
    Caching = 1 << 6,
    Vision = 1 << 7,
}

impl ModelFeature {
    pub const fn bit(self) -> u32 {
        self as u32
    }
}

/// One canonical model identity. Per-version constant.
///
/// `model_id` is the wire-format identifier used in `OpenChannel.model_id`
/// and in OpenAI-compatible HTTP `model` fields. It MUST match the
/// `id` field a downstream OpenRouter-style backend would return from
/// `/v1/models` for the same model, so a provider can pass requests
/// straight through without re-mapping.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    pub model_id: String,
    pub display_name: String,
    pub tokenizer: TokenizerFamily,
    pub total_context_tokens: u32,
    pub max_output_tokens: u32,
    /// Bitset of `Modality`. e.g. `Modality::Text.bit() | Modality::Image.bit()`.
    pub input_modalities: u16,
    pub output_modalities: u16,
    /// Bitset of `ModelFeature`.
    pub supported_features: u32,
    /// Set to `false` to deprecate a model without removing the entry —
    /// open channels keep working but no new offerings may register.
    pub active: bool,
}

/// Protocol-versioned model registry. Constructed via
/// [`ModelRegistry::for_version`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub entries: Vec<ModelRegistryEntry>,
}

impl ModelRegistry {
    /// Registry contents at a given protocol version. Adding or
    /// deactivating a model requires a new version arm here + a
    /// `MAX_PROTOCOL_VERSION` bump.
    pub fn for_version(version: ProtocolVersion) -> Self {
        Self { entries: entries_for_version(version) }
    }

    /// Find an active entry by model_id.
    pub fn get(&self, model_id: &str) -> Option<&ModelRegistryEntry> {
        self.entries.iter().find(|e| e.model_id == model_id && e.active)
    }

    /// Find any entry by model_id (active or deprecated).
    pub fn get_any(&self, model_id: &str) -> Option<&ModelRegistryEntry> {
        self.entries.iter().find(|e| e.model_id == model_id)
    }

    /// Iterate all active entries.
    pub fn active(&self) -> impl Iterator<Item = &ModelRegistryEntry> {
        self.entries.iter().filter(|e| e.active)
    }
}

/// Inline helpers so the seed data below stays compact and readable.
fn modalities(items: &[Modality]) -> u16 {
    items.iter().fold(0, |acc, m| acc | m.bit())
}

fn features(items: &[ModelFeature]) -> u32 {
    items.iter().fold(0, |acc, f| acc | f.bit())
}

/// Constructor for a Gemma-family entry. Pins the SentencePiece tokenizer
/// (proxy falls back to chars/4) and the text+image+video input modalities
/// the multimodal Gemma 4 weights support.
fn gemma(model_id: &str, name: &str, ctx: u32, max_out: u32, features: u32) -> ModelRegistryEntry {
    ModelRegistryEntry {
        model_id: model_id.to_string(),
        display_name: name.to_string(),
        tokenizer: TokenizerFamily::Gemini,
        total_context_tokens: ctx,
        max_output_tokens: max_out,
        input_modalities: Modality::Text.bit() | Modality::Image.bit() | Modality::Video.bit(),
        output_modalities: Modality::Text.bit(),
        supported_features: features | ModelFeature::Vision.bit(),
        active: true,
    }
}

/// Compute the canonical entries for a given protocol version.
///
/// Soma launches with a single locally-servable model: every provider
/// runs the same `google/gemma-4-31b-it` weights under llama.cpp, so the
/// network is symmetric — any participant can be both a consumer and a
/// provider from day one without procuring upstream API keys. Additional
/// models land via new protocol-version arms here.
fn entries_for_version(version: ProtocolVersion) -> Vec<ModelRegistryEntry> {
    if version.as_u64() == 0 {
        return Vec::new();
    }

    let common_chat =
        features(&[ModelFeature::Tools, ModelFeature::JsonMode, ModelFeature::StructuredOutputs]);

    // The sole launch model. Context/output limits match the published
    // Gemma 4 31B IT card; `model_id` doubles as the HuggingFace repo
    // path (`huggingface.co/google/gemma-4-31b-it`) the provider's
    // llama.cpp downloader resolves the GGUF from.
    let mut entries =
        vec![gemma("google/gemma-4-31b-it", "Gemma 4 31B IT", 262_144, 16_384, common_chat)];

    // Future protocol versions extend by pushing new entries or
    // flipping `active`. Keep the function deterministic — no env
    // reads, no time-of-day branches.
    entries.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    entries.dedup_by(|a, b| a.model_id == b.model_id);

    let _ = version; // reserved for future per-version diffs.
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MAX_PROTOCOL_VERSION, MIN_PROTOCOL_VERSION};

    #[test]
    fn registry_for_min_version_is_gemma_only() {
        let r = ModelRegistry::for_version(ProtocolVersion::new(MIN_PROTOCOL_VERSION));
        assert_eq!(r.entries.len(), 1, "launch registry holds exactly one model");
        let gemma = r.get("google/gemma-4-31b-it").expect("gemma is the launch model");
        assert_eq!(gemma.tokenizer, TokenizerFamily::Gemini);
        assert_eq!(gemma.display_name, "Gemma 4 31B IT");
        // Models that used to seed the OpenRouter-era registry are gone.
        assert!(r.get("anthropic/claude-sonnet-4.6").is_none());
        assert!(r.get("openai/gpt-5.5").is_none());
    }

    #[test]
    fn registry_is_deterministic_across_versions() {
        for v in MIN_PROTOCOL_VERSION..=MAX_PROTOCOL_VERSION {
            let a = ModelRegistry::for_version(ProtocolVersion::new(v));
            let b = ModelRegistry::for_version(ProtocolVersion::new(v));
            assert_eq!(a, b, "registry at v{} must be deterministic", v);
        }
    }

    #[test]
    fn registry_entries_sorted_and_unique() {
        let r = ModelRegistry::for_version(ProtocolVersion::new(MIN_PROTOCOL_VERSION));
        for w in r.entries.windows(2) {
            assert!(w[0].model_id < w[1].model_id, "must be sorted by model_id");
        }
    }

    #[test]
    fn get_filters_inactive() {
        let r = ModelRegistry::for_version(ProtocolVersion::new(MIN_PROTOCOL_VERSION));
        for e in &r.entries {
            // All seeded as active.
            assert!(e.active, "{} must be seeded active", e.model_id);
        }
    }

    #[test]
    fn modality_bitset_combines() {
        let combined = modalities(&[Modality::Text, Modality::Image]);
        assert_eq!(combined, Modality::Text.bit() | Modality::Image.bit());
    }
}
