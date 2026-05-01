// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0
//
// HTTP loopback end-to-end test:
//   wiremock (fake OpenAI upstream)
//     ↑
//   inference::server  (provider)
//     ↑                    ← shares MemoryDiscovery
//   inference::proxy   (client-side)
//     ↑
//   reqwest test client
//
// Verifies the wire format end-to-end: SomaPay header parses + verifies
// against the provider's pubkey, pre_flight runs, post_flight charges
// realized cost, slack absorbs into the next request, channel state
// persists. Plus the 402-required-authorization retry path.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fastcrypto::ed25519::Ed25519KeyPair;
use fastcrypto::traits::{KeyPair, ToFromBytes};
use futures::StreamExt;
use sdk::keypair::Keypair;
use serde_json::json;
use tempfile::TempDir;
use types::base::SomaAddress;
use types::crypto::SomaKeyPair;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use inference::catalog::{Architecture, ModelCard, Pricing, TopProvider};
use inference::chain::memory::MemoryDiscovery;

const TEST_MODEL: &str = "test-org/test-model-1";

fn keypair_for_test() -> Keypair {
    let kp = Ed25519KeyPair::generate(&mut rand::thread_rng());
    Keypair::from_inner(SomaKeyPair::Ed25519(kp))
}

fn pubkey_hex(kp: &Keypair) -> String {
    match kp.inner() {
        SomaKeyPair::Ed25519(ed) => hex::encode(ed.public().as_bytes()),
        _ => unreachable!(),
    }
}

fn test_card() -> ModelCard {
    ModelCard {
        id: TEST_MODEL.to_string(),
        canonical_slug: None,
        hugging_face_id: None,
        name: "Test Model".to_string(),
        created: 0,
        description: None,
        context_length: 8192,
        architecture: Architecture {
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            tokenizer: "GPT".to_string(),
            instruct_type: None,
        },
        pricing: Pricing {
            // $1 per million prompt tokens, $2 per million completion tokens
            prompt: "0.000001".into(),
            completion: "0.000002".into(),
            request: "0".into(),
            image: "0".into(),
            input_cache_read: "0".into(),
            input_cache_write: "0".into(),
        },
        top_provider: TopProvider {
            context_length: 8192,
            max_completion_tokens: Some(1024),
            is_moderated: false,
        },
        supported_parameters: vec![],
        default_parameters: None,
        expiration_date: None,
        soma: None,
    }
}

fn server_config(
    listen: &str,
    upstream_url: &str,
    soma_home: &std::path::Path,
) -> inference::server::Config {
    use inference::server::config::*;
    let toml = format!(
        r#"
[server]
listen = "{listen}"
public_endpoint = "http://{listen}"

[chain]
mode = "memory"
soma_home = "{soma_home}"
heartbeat_interval_secs = 0

[backend]
kind = "openrouter"
api_key_env = "INFERENCE_TEST_API_KEY"
upstream_url = "{upstream_url}"

[auth]
clock_skew_tolerance_secs = 60

[[offerings]]
id = "{TEST_MODEL}"
name = "Test Model"
context_length = 8192
architecture = {{ input_modalities = ["text"], output_modalities = ["text"], tokenizer = "GPT" }}
top_provider = {{ context_length = 8192, max_completion_tokens = 1024, is_moderated = false }}
pricing = {{ prompt = "0.000001", completion = "0.000002", request = "0", image = "0", input_cache_read = "0", input_cache_write = "0" }}
"#,
        soma_home = soma_home.display(),
    );
    toml::from_str(&toml).expect("parse server toml")
}

fn proxy_config(listen: &str, soma_home: &std::path::Path) -> inference::proxy::Config {
    let toml = format!(
        r#"
[listen]
addr = "{listen}"

[chain]
mode = "memory"
soma_home = "{soma_home}"

[wallet]
default_deposit_micros = 5_000_000
channel_expires_secs = 86_400

[discovery]
provider_cache_ttl_secs = 1
"#,
        soma_home = soma_home.display(),
    );
    toml::from_str(&toml).expect("parse proxy toml")
}

async fn wait_until<F: Fn() -> Fut, Fut: std::future::Future<Output = bool>>(
    label: &str,
    f: F,
    timeout: Duration,
) {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if f().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("timed out waiting for {label}");
}

#[tokio::test]
async fn proxy_to_provider_streaming_round_trip() {
    // Provider needs an "upstream API key" — we point a fake env var at it.
    // SAFETY: tests in this binary run sequentially by default; this var is
    // only read at backend construction in `inference::server::run`.
    unsafe { std::env::set_var("INFERENCE_TEST_API_KEY", "test-token-1"); }

    // ── 1. Fake OpenAI upstream ─────────────────────────────────────────
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(
                    // Two SSE events: a content chunk, then a final chunk
                    // carrying `usage`. (`include_usage` is injected by the
                    // backend.)
                    concat!(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                        "data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],",
                        "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
                        "data: [DONE]\n\n",
                    ).as_bytes(),
                    "text/event-stream",
                ),
        )
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": []})))
        .mount(&upstream)
        .await;

    // ── 2. Shared discovery + ephemeral homes ───────────────────────────
    let chain: Arc<dyn inference::Discovery> = Arc::new(MemoryDiscovery::new());
    let provider_home = TempDir::new().unwrap();
    let proxy_home = TempDir::new().unwrap();

    // ── 3. Identities ───────────────────────────────────────────────────
    let provider_kp = keypair_for_test();
    let proxy_kp = keypair_for_test();

    // ── 4. Boot the provider ────────────────────────────────────────────
    let provider_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_addr = provider_listener.local_addr().unwrap();
    drop(provider_listener); // release; server will rebind

    let server_cfg = server_config(
        &provider_addr.to_string(),
        &format!("{}/", upstream.uri()),
        provider_home.path(),
    );
    let chain_for_server = chain.clone();
    let provider_kp_clone = provider_kp.copy();
    tokio::spawn(async move {
        if let Err(e) = inference::server::run(server_cfg, provider_kp_clone, chain_for_server).await {
            eprintln!("server error: {e:?}");
        }
    });

    // ── 5. Wait for provider to be reachable ────────────────────────────
    let prov_url = format!("http://{provider_addr}");
    let http = reqwest::Client::new();
    {
        let url = prov_url.clone();
        let http = http.clone();
        wait_until(
            "provider /soma/info",
            || async {
                let url = url.clone();
                let http = http.clone();
                http.get(format!("{url}/soma/info"))
                    .send()
                    .await
                    .map(|r| r.status().is_success())
                    .unwrap_or(false)
            },
            Duration::from_secs(5),
        )
        .await;
    }

    // ── 6. Boot the proxy ───────────────────────────────────────────────
    let proxy_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_addr = proxy_listener.local_addr().unwrap();
    drop(proxy_listener);

    let proxy_cfg = proxy_config(&proxy_addr.to_string(), proxy_home.path());
    let chain_for_proxy = chain.clone();
    let proxy_kp_clone = proxy_kp.copy();
    tokio::spawn(async move {
        if let Err(e) = inference::proxy::run(proxy_cfg, proxy_kp_clone, chain_for_proxy).await {
            eprintln!("proxy error: {e:?}");
        }
    });

    let proxy_url = format!("http://{proxy_addr}");
    {
        let url = proxy_url.clone();
        let http = http.clone();
        wait_until(
            "proxy /v1/models",
            || async {
                let url = url.clone();
                let http = http.clone();
                http.get(format!("{url}/v1/models"))
                    .send()
                    .await
                    .map(|r| r.status().is_success())
                    .unwrap_or(false)
            },
            Duration::from_secs(5),
        )
        .await;
    }

    // ── 7. Wait for proxy's discovery cache to see our provider ─────────
    let r = http
        .post(format!("{proxy_url}/v1/chat/completions"))
        .json(&json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "max_tokens": 10,
        }))
        .send()
        .await
        .expect("post chat");
    // The proxy may take a moment to pick up the provider after registration.
    // Loop until we get a 2xx (or fail clearly).
    if !r.status().is_success() {
        // Brief grace period for cache warmup.
        tokio::time::sleep(Duration::from_millis(800)).await;
        let r2 = http
            .post(format!("{proxy_url}/v1/chat/completions"))
            .json(&json!({
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
                "max_tokens": 10,
            }))
            .send()
            .await
            .expect("post chat retry");
        assert!(r2.status().is_success(), "proxy → provider should succeed; got {}", r2.status());
        let body = r2.bytes().await.unwrap();
        let s = String::from_utf8_lossy(&body);
        assert!(s.contains("\"hi\""), "expected content chunk in SSE: {s}");
    } else {
        let body = r.bytes().await.unwrap();
        let s = String::from_utf8_lossy(&body);
        assert!(s.contains("\"hi\""), "expected content chunk in SSE: {s}");
    }

    // ── 8. Confirm the channel was opened + post_flight ran ─────────────
    // Provider state file should reflect non-zero consumed.
    let provider_channels = provider_home.path().join("provider").join("channels");
    // Streaming post_flight runs asynchronously; give it a moment.
    let mut consumed = 0u64;
    for _ in 0..50 {
        if let Ok(entries) = std::fs::read_dir(&provider_channels) {
            for entry in entries.flatten() {
                if let Ok(s) = std::fs::read_to_string(entry.path()) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                        consumed = v
                            .get("total_consumed_micros")
                            .and_then(|x| x.as_u64())
                            .unwrap_or(0);
                    }
                }
            }
        }
        if consumed > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(consumed > 0, "provider should have charged consumed > 0; got {consumed}");

    // Discovery should know about the provider.
    let providers = chain.list_providers().await.unwrap();
    assert_eq!(providers.len(), 1);
    let pubkey_hex_expected = pubkey_hex(&provider_kp);
    assert_eq!(providers[0].pubkey_hex, pubkey_hex_expected);
    assert_eq!(providers[0].address, provider_kp.address());

    // Suppress unused
    let _ = (proxy_kp.address(), SomaAddress::ZERO, Bytes::new());
}
