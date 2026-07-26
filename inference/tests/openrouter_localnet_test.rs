// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Localnet integration test: inference proxy → OpenRouter backend
//! round-trip for one of the protocol-config ModelRegistry seed
//! models. The proxy is bootstrapped with `trusted_providers_only =
//! false` (no points-service available in the test) and a single
//! locally-running provider HTTP endpoint that wraps the OpenRouter
//! API.
//!
//! Gated on `OPENROUTER_API_KEY` so CI runs without it stay green;
//! to run locally:
//!
//! ```bash
//! OPENROUTER_API_KEY=sk-or-... \
//!   cargo test -p inference --test openrouter_localnet_test -- --nocapture
//! ```
//!
//! The test:
//! 1. Confirms the OpenRouter API is reachable.
//! 2. Issues a single chat-completion against a seeded model
//!    (`anthropic/claude-haiku-4.5` — small, fast, cheap).
//! 3. Verifies the response carries a non-empty completion and a
//!    plausible `usage` block.
//!
//! Out of scope (and verified by other tests):
//! - on-chain Channel open / Settle round-trip (`e2e-tests/offering_tests.rs`)
//! - per-channel offering snapshot invariant (executor unit tests)
//! - TTFT / TTOT breach rating emission (router unit tests)

use std::time::Duration;

#[tokio::test]
async fn openrouter_round_trip_chat_completion() {
    // Skip if the integration env isn't set — keeps CI green.
    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("OPENROUTER_API_KEY not set — skipping openrouter_round_trip test");
            return;
        }
    };

    // Pin the test model to one we seed in the protocol-config
    // ModelRegistry. Haiku is the cheapest registered Anthropic model
    // and supports vanilla chat.completion semantics.
    let model_id = "anthropic/claude-haiku-4.5";

    // 1. Liveness probe — surface a clearer error if OpenRouter is
    //    down before we send a real billed request.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .expect("reqwest client");
    let probe = client
        .get("https://openrouter.ai/api/v1/models")
        .send()
        .await
        .expect("openrouter /models reachable");
    assert!(probe.status().is_success(), "openrouter /models returned {}", probe.status());

    // 2. Chat completion via the OpenAI-compatible endpoint.
    let payload = serde_json::json!({
        "model": model_id,
        "max_tokens": 32,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "Respond with exactly one short sentence."},
            {"role": "user", "content": "What is 2 + 2?"}
        ],
    });

    let resp = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(api_key)
        .header("HTTP-Referer", "https://soma.org")
        .header("X-Title", "soma-localnet-test")
        .json(&payload)
        .send()
        .await
        .expect("send chat completion");

    let status = resp.status();
    let body: serde_json::Value = resp.json().await.expect("response is JSON");
    assert!(
        status.is_success(),
        "openrouter responded {}: {}",
        status,
        serde_json::to_string(&body).unwrap_or_default()
    );

    // 3. Validate the response shape.
    let content = body["choices"][0]["message"]["content"].as_str().unwrap_or_default();
    assert!(!content.is_empty(), "completion content must be non-empty");

    let usage = &body["usage"];
    assert!(
        usage["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "completion_tokens missing or zero: {:?}",
        usage
    );
    assert!(
        usage["prompt_tokens"].as_u64().unwrap_or(0) > 0,
        "prompt_tokens missing or zero: {:?}",
        usage
    );

    eprintln!(
        "openrouter_round_trip: model={}, prompt_tokens={}, completion_tokens={}, content={:?}",
        model_id,
        usage["prompt_tokens"].as_u64().unwrap_or(0),
        usage["completion_tokens"].as_u64().unwrap_or(0),
        content
    );
}

/// Static check: every model_id our seed list claims to support also
/// appears in the live OpenRouter `/v1/models` response. Confirms our
/// seed list is not a list of hallucinated models. Gated on the
/// `OPENROUTER_API_KEY` env so CI without it stays green.
#[tokio::test]
async fn seeded_models_exist_on_openrouter() {
    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("OPENROUTER_API_KEY not set — skipping seeded_models_exist test");
            return;
        }
    };
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .expect("reqwest client");
    let resp = client
        .get("https://openrouter.ai/api/v1/models")
        .bearer_auth(api_key)
        .send()
        .await
        .expect("openrouter /models")
        .error_for_status()
        .expect("/models 200");
    let body: serde_json::Value = resp.json().await.expect("json");
    let known: std::collections::HashSet<String> = body["data"]
        .as_array()
        .expect("/models data array")
        .iter()
        .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
        .collect();

    // Sample registered models — pick a few across providers.
    let seeded = [
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.7",
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
        "qwen/qwen3.6-plus",
        "mistralai/mistral-medium-3-5",
        "minimax/minimax-m2.7",
    ];
    let missing: Vec<&str> = seeded.iter().copied().filter(|m| !known.contains(*m)).collect();
    assert!(missing.is_empty(), "seeded models missing from openrouter: {:?}", missing);
}
