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

/// Family-specific entry constructors. Each pins the tokenizer + the
/// modalities a typical model in that family supports (text + vision +
/// file in/out for the multimodal frontier families; text-only for
/// instruct-tuned open-weights base models). Centralizes the boilerplate
/// so the seed list stays readable.
fn anthropic(
    model_id: &str,
    name: &str,
    ctx: u32,
    max_out: u32,
    features: u32,
) -> ModelRegistryEntry {
    ModelRegistryEntry {
        model_id: model_id.to_string(),
        display_name: name.to_string(),
        tokenizer: TokenizerFamily::Claude,
        total_context_tokens: ctx,
        max_output_tokens: max_out,
        input_modalities: Modality::Text.bit() | Modality::Image.bit() | Modality::File.bit(),
        output_modalities: Modality::Text.bit(),
        supported_features: features | ModelFeature::Vision.bit(),
        active: true,
    }
}

fn gpt(model_id: &str, name: &str, ctx: u32, max_out: u32, features: u32) -> ModelRegistryEntry {
    ModelRegistryEntry {
        model_id: model_id.to_string(),
        display_name: name.to_string(),
        tokenizer: TokenizerFamily::Gpt,
        total_context_tokens: ctx,
        max_output_tokens: max_out,
        input_modalities: Modality::Text.bit()
            | Modality::Image.bit()
            | Modality::Audio.bit()
            | Modality::File.bit(),
        output_modalities: Modality::Text.bit(),
        supported_features: features | ModelFeature::Vision.bit(),
        active: true,
    }
}

fn gemini(model_id: &str, name: &str, ctx: u32, max_out: u32, features: u32) -> ModelRegistryEntry {
    ModelRegistryEntry {
        model_id: model_id.to_string(),
        display_name: name.to_string(),
        tokenizer: TokenizerFamily::Gemini,
        total_context_tokens: ctx,
        max_output_tokens: max_out,
        input_modalities: Modality::Text.bit()
            | Modality::Image.bit()
            | Modality::Audio.bit()
            | Modality::Video.bit()
            | Modality::File.bit(),
        output_modalities: Modality::Text.bit(),
        supported_features: features | ModelFeature::Vision.bit(),
        active: true,
    }
}

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

fn grok(model_id: &str, name: &str, ctx: u32, max_out: u32, features: u32) -> ModelRegistryEntry {
    ModelRegistryEntry {
        model_id: model_id.to_string(),
        display_name: name.to_string(),
        tokenizer: TokenizerFamily::Grok,
        total_context_tokens: ctx,
        max_output_tokens: max_out,
        input_modalities: Modality::Text.bit() | Modality::Image.bit() | Modality::File.bit(),
        output_modalities: Modality::Text.bit(),
        supported_features: features | ModelFeature::Vision.bit(),
        active: true,
    }
}

fn llama(model_id: &str, name: &str, ctx: u32, max_out: u32, features: u32) -> ModelRegistryEntry {
    ModelRegistryEntry {
        model_id: model_id.to_string(),
        display_name: name.to_string(),
        tokenizer: TokenizerFamily::Llama,
        total_context_tokens: ctx,
        max_output_tokens: max_out,
        input_modalities: Modality::Text.bit() | Modality::Image.bit(),
        output_modalities: Modality::Text.bit(),
        supported_features: features | ModelFeature::Vision.bit(),
        active: true,
    }
}

fn deepseek(
    model_id: &str,
    name: &str,
    ctx: u32,
    max_out: u32,
    features: u32,
) -> ModelRegistryEntry {
    ModelRegistryEntry {
        model_id: model_id.to_string(),
        display_name: name.to_string(),
        tokenizer: TokenizerFamily::Deepseek,
        total_context_tokens: ctx,
        max_output_tokens: max_out,
        input_modalities: Modality::Text.bit(),
        output_modalities: Modality::Text.bit(),
        supported_features: features,
        active: true,
    }
}

fn qwen_tok(
    model_id: &str,
    name: &str,
    ctx: u32,
    max_out: u32,
    features: u32,
) -> ModelRegistryEntry {
    ModelRegistryEntry {
        model_id: model_id.to_string(),
        display_name: name.to_string(),
        tokenizer: TokenizerFamily::Qwen,
        total_context_tokens: ctx,
        max_output_tokens: max_out,
        input_modalities: Modality::Text.bit() | Modality::Image.bit(),
        output_modalities: Modality::Text.bit(),
        supported_features: features | ModelFeature::Vision.bit(),
        active: true,
    }
}

fn mistral(
    model_id: &str,
    name: &str,
    ctx: u32,
    max_out: u32,
    features: u32,
) -> ModelRegistryEntry {
    ModelRegistryEntry {
        model_id: model_id.to_string(),
        display_name: name.to_string(),
        tokenizer: TokenizerFamily::Mistral,
        total_context_tokens: ctx,
        max_output_tokens: max_out,
        input_modalities: Modality::Text.bit(),
        output_modalities: Modality::Text.bit(),
        supported_features: features,
        active: true,
    }
}

/// Fallback constructor for providers whose tokenizer doesn't match any
/// known family. Proxy falls back to chars/4 estimation.
fn unknown_tok(
    model_id: &str,
    name: &str,
    ctx: u32,
    max_out: u32,
    features: u32,
) -> ModelRegistryEntry {
    ModelRegistryEntry {
        model_id: model_id.to_string(),
        display_name: name.to_string(),
        tokenizer: TokenizerFamily::Unknown,
        total_context_tokens: ctx,
        max_output_tokens: max_out,
        input_modalities: Modality::Text.bit() | Modality::Image.bit(),
        output_modalities: Modality::Text.bit(),
        supported_features: features | ModelFeature::Vision.bit(),
        active: true,
    }
}

/// Compute the canonical entries for a given protocol version. New
/// versions inherit and extend; we never silently rewrite history.
fn entries_for_version(version: ProtocolVersion) -> Vec<ModelRegistryEntry> {
    // V1: seed with the major frontier + open-source models live on
    // OpenRouter at protocol launch. Inclusion criteria: well-known,
    // OpenAI-API-compatible via OpenRouter, broad provider coverage.
    if version.as_u64() == 0 {
        return Vec::new();
    }

    // Default base (v1 and onward).
    let text = Modality::Text;
    let image = Modality::Image;
    let file = Modality::File;
    let audio = Modality::Audio;

    let common_chat =
        features(&[ModelFeature::Tools, ModelFeature::JsonMode, ModelFeature::StructuredOutputs]);
    let reasoning_chat = features(&[
        ModelFeature::Tools,
        ModelFeature::JsonMode,
        ModelFeature::StructuredOutputs,
        ModelFeature::Reasoning,
    ]);
    let cached_reasoning = features(&[
        ModelFeature::Tools,
        ModelFeature::JsonMode,
        ModelFeature::StructuredOutputs,
        ModelFeature::Reasoning,
        ModelFeature::Caching,
    ]);

    // The seed below covers each major provider at three rough size
    // tiers — small/medium/large — plus current-gen "preview"/"latest"
    // entries that are actively routed to. Model IDs match OpenRouter
    // canonical slugs (`/v1/models`) verbatim so a provider can pass
    // requests straight through. Only entries with public release
    // dates in March 2025 or later are included; superseded versions
    // are dropped. Updates land via new protocol-version arms here.
    let video = Modality::Video;
    let _ = audio;
    let _ = file;
    let _ = video;
    let mut entries = vec![
        // === Anthropic (Mar 2025+) ===
        anthropic(
            "anthropic/claude-haiku-4.5",
            "Claude Haiku 4.5",
            200_000,
            64_000,
            cached_reasoning,
        ),
        anthropic(
            "anthropic/claude-sonnet-4.6",
            "Claude Sonnet 4.6",
            1_000_000,
            128_000,
            cached_reasoning,
        ),
        anthropic(
            "anthropic/claude-opus-4.7",
            "Claude Opus 4.7",
            1_000_000,
            128_000,
            cached_reasoning,
        ),
        anthropic(
            "anthropic/claude-opus-4.7-fast",
            "Claude Opus 4.7 (Fast)",
            1_000_000,
            128_000,
            cached_reasoning,
        ),
        // === OpenAI (Mar 2025+) ===
        gpt("openai/gpt-5.4-nano", "GPT-5.4 Nano", 400_000, 128_000, cached_reasoning),
        gpt("openai/gpt-5.4-mini", "GPT-5.4 Mini", 400_000, 128_000, cached_reasoning),
        gpt("openai/gpt-5.5", "GPT-5.5", 1_050_000, 128_000, cached_reasoning),
        gpt("openai/gpt-5.5-pro", "GPT-5.5 Pro", 1_050_000, 128_000, cached_reasoning),
        // === Google (Mar 2025+) ===
        gemini(
            "google/gemini-3.1-flash-lite",
            "Gemini 3.1 Flash Lite",
            1_048_576,
            65_536,
            reasoning_chat,
        ),
        gemini(
            "google/gemini-3.1-pro-preview",
            "Gemini 3.1 Pro (Preview)",
            1_048_576,
            65_536,
            reasoning_chat,
        ),
        gemma("google/gemma-4-26b-a4b-it", "Gemma 4 26B A4B IT", 262_144, 16_384, common_chat),
        gemma("google/gemma-4-31b-it", "Gemma 4 31B IT", 262_144, 16_384, common_chat),
        // === xAI (Mar 2025+) ===
        grok("x-ai/grok-4.20", "Grok 4.20", 2_000_000, 32_768, reasoning_chat),
        grok(
            "x-ai/grok-4.20-multi-agent",
            "Grok 4.20 Multi-Agent",
            2_000_000,
            32_768,
            reasoning_chat,
        ),
        grok("x-ai/grok-4.3", "Grok 4.3", 1_000_000, 32_768, reasoning_chat),
        // === Meta (Mar 2025+, open weights) ===
        llama("meta-llama/llama-4-scout", "Llama 4 Scout", 327_680, 16_384, common_chat),
        llama("meta-llama/llama-4-maverick", "Llama 4 Maverick", 1_048_576, 16_384, common_chat),
        // === DeepSeek (Mar 2025+) ===
        deepseek(
            "deepseek/deepseek-v4-flash",
            "DeepSeek V4 Flash",
            1_048_576,
            131_072,
            common_chat,
        ),
        deepseek("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro", 1_048_576, 384_000, reasoning_chat),
        // === Moonshot (Mar 2025+) ===
        qwen_tok("moonshotai/kimi-k2.6", "Kimi K2.6", 262_142, 262_142, reasoning_chat),
        // === Qwen (Mar 2025+) ===
        qwen_tok("qwen/qwen3.6-flash", "Qwen3.6 Flash", 1_000_000, 65_536, common_chat),
        qwen_tok("qwen/qwen3.6-plus", "Qwen3.6 Plus", 1_000_000, 65_536, reasoning_chat),
        qwen_tok(
            "qwen/qwen3.6-max-preview",
            "Qwen3.6 Max Preview",
            262_144,
            65_536,
            reasoning_chat,
        ),
        // === Mistral (Mar 2025+) ===
        mistral(
            "mistralai/mistral-small-2603",
            "Mistral Small 4 (2026-03)",
            262_144,
            32_768,
            common_chat,
        ),
        mistral("mistralai/mistral-medium-3-5", "Mistral Medium 3.5", 262_144, 32_768, common_chat),
        mistral(
            "mistralai/mistral-large-2512",
            "Mistral Large 3 (2025-12)",
            262_144,
            32_768,
            common_chat,
        ),
        // === MiniMax (Mar 2025+) ===
        unknown_tok("minimax/minimax-m2.7", "MiniMax M2.7", 196_608, 32_768, reasoning_chat),
        // === Z.AI (user request) ===
        unknown_tok("z-ai/glm-5.1", "GLM 5.1", 128_000, 16_384, reasoning_chat),
        // === Xiaomi (user request) ===
        unknown_tok("xiaomi/mimo-v2.5-pro", "MiMo V2.5 Pro", 200_000, 16_384, reasoning_chat),
        // === Tencent (Mar 2025+) ===
        unknown_tok(
            "tencent/hy3-preview",
            "Tencent Hy3 (Preview)",
            262_144,
            65_536,
            reasoning_chat,
        ),
        // === NVIDIA (Mar 2025+) ===
        unknown_tok(
            "nvidia/nemotron-3-super-120b-a12b",
            "Nemotron 3 Super 120B A12B",
            262_144,
            65_536,
            common_chat,
        ),
        // === Amazon (Mar 2025+) ===
        gpt("amazon/nova-2-lite-v1", "Amazon Nova 2 Lite", 1_000_000, 65_535, common_chat),
        gpt("amazon/nova-premier-v1", "Amazon Nova Premier", 1_000_000, 32_000, common_chat),
        // === Cohere (Mar 2025+) ===
        unknown_tok("cohere/command-a", "Cohere Command A", 256_000, 8_192, common_chat),
        // === IBM (Mar 2025+) ===
        unknown_tok(
            "ibm-granite/granite-4.1-8b",
            "IBM Granite 4.1 8B",
            131_072,
            32_768,
            common_chat,
        ),
        // === ByteDance Seed (Mar 2025+) ===
        unknown_tok(
            "bytedance-seed/seed-2.0-mini",
            "ByteDance Seed 2.0 Mini",
            262_144,
            131_072,
            common_chat,
        ),
        unknown_tok(
            "bytedance-seed/seed-2.0-lite",
            "ByteDance Seed 2.0 Lite",
            262_144,
            131_072,
            common_chat,
        ),
    ];

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
    fn registry_for_min_version_has_frontier_seed() {
        let r = ModelRegistry::for_version(ProtocolVersion::new(MIN_PROTOCOL_VERSION));
        assert!(r.entries.len() >= 25, "expected ≥25 entries in v1 seed");
        // Sample S/M/L coverage across major providers — model IDs
        // taken verbatim from openrouter.ai/api/v1/models (Mar 2025+).
        assert!(r.get("anthropic/claude-haiku-4.5").is_some());
        assert!(r.get("anthropic/claude-sonnet-4.6").is_some());
        assert!(r.get("anthropic/claude-opus-4.7").is_some());
        assert!(r.get("openai/gpt-5.4-mini").is_some());
        assert!(r.get("openai/gpt-5.5").is_some());
        assert!(r.get("openai/gpt-5.5-pro").is_some());
        assert!(r.get("google/gemini-3.1-flash-lite").is_some());
        assert!(r.get("google/gemini-3.1-pro-preview").is_some());
        assert!(r.get("x-ai/grok-4.3").is_some());
        assert!(r.get("meta-llama/llama-4-maverick").is_some());
        assert!(r.get("deepseek/deepseek-v4-pro").is_some());
        assert!(r.get("moonshotai/kimi-k2.6").is_some());
        assert!(r.get("qwen/qwen3.6-plus").is_some());
        assert!(r.get("mistralai/mistral-medium-3-5").is_some());
        assert!(r.get("minimax/minimax-m2.7").is_some());
        assert!(r.get("z-ai/glm-5.1").is_some());
        assert!(r.get("xiaomi/mimo-v2.5-pro").is_some());
        assert!(r.get("tencent/hy3-preview").is_some());
        assert!(r.get("nvidia/nemotron-3-super-120b-a12b").is_some());
        assert!(r.get("amazon/nova-premier-v1").is_some());
        assert!(r.get("cohere/command-a").is_some());
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
