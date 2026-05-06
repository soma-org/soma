// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Non-msim integration test: real `TestCluster` chain + the inference
// `proxy` and `server` binaries running in-process, with a `wiremock`
// upstream standing in for the LLM backend.
//
// The shape mirrors what `examples/localnet/up.sh` does manually,
// but in-process so it can assert on the on-chain channel state.
//
//   wiremock (canned OpenAI upstream)
//     ↑
//   inference::server  (provider; uses ChainChannelSurface)
//     ↑                    ← shared LocalDiscovery (provider registry)
//   inference::proxy   (client)
//     ↑
//   reqwest test client
//
// Run with:
//   PYO3_PYTHON=python3 cargo test -p inference --test e2e_localnet -- --ignored --nocapture
//
// Marked `#[ignore]` because booting a TestCluster takes ~5s and
// every CI run probably doesn't need it.

#![cfg(not(msim))]

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use sdk::wallet_context::WalletContext;
use tempfile::TempDir;
use test_cluster::TestClusterBuilder;
use types::base::SomaAddress;
use types::config::SOMA_CLIENT_CONFIG;
use types::object::CoinType;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use inference::chain::chain::ChainChannelSurface;
use inference::chain::local::LocalDiscovery;
use inference::chain::{ChannelSurface, ProviderRecord, ProviderRegistry};
use inference::channel::{PaymentChannel as _, RunningTab};
use inference::server::ledger::Ledger;

const TEST_MODEL: &str = "test-org/test-model";

fn wallet_for_path(path: &std::path::Path) -> WalletContext {
    WalletContext::new(path).expect("WalletContext from cluster's client.yaml")
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn proxy_provider_full_stack_against_real_chain() {

    // --- 1. Boot the chain ---------------------------------------------------
    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let provider_addr = addrs[1];
    let wallet_conf_path = test_cluster.swarm.dir().join(SOMA_CLIENT_CONFIG);

    // --- 2. wiremock upstream returning a deterministic OpenAI body ----------
    let upstream = MockServer::start().await;
    let canned_body = json!({
        "id": "test-id",
        "object": "chat.completion",
        "created": 0,
        "model": TEST_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
    });
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_body.clone()))
        .mount(&upstream)
        .await;

    // --- 3. Provider config (in-memory ledger + LocalDiscovery for registry) -
    let ledger_dir = TempDir::new().unwrap();
    let registry_dir = TempDir::new().unwrap();
    let registry: Arc<dyn ProviderRegistry> = Arc::new(
        LocalDiscovery::new(registry_dir.path().join("registry")).expect("local discovery"),
    );

    let provider_wallet = Arc::new(wallet_for_path(&wallet_conf_path));
    let provider_chain: Arc<dyn ChannelSurface> =
        Arc::new(ChainChannelSurface::new(provider_wallet.clone(), provider_addr));
    let provider_channel = Arc::new(RunningTab::for_provider(60));
    let ledger = Ledger::new(ledger_dir.path().to_path_buf());

    // --- 4. Build a single ModelCard the catalog will return -----------------
    let card = inference::catalog::ModelCard {
        id: TEST_MODEL.to_string(),
        name: TEST_MODEL.to_string(),
        canonical_slug: None,
        hugging_face_id: Some(TEST_MODEL.to_string()),
        created: 0,
        description: None,
        context_length: 4096,
        architecture: inference::catalog::Architecture {
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            tokenizer: "cl100k_base".into(),
            instruct_type: None,
        },
        top_provider: inference::catalog::TopProvider {
            context_length: 4096,
            max_completion_tokens: Some(1024),
            is_moderated: false,
        },
        supported_parameters: vec!["max_tokens".into()],
        default_parameters: None,
        expiration_date: None,
        pricing: inference::catalog::Pricing {
            prompt: "0.0000002".into(),
            completion: "0.0000004".into(),
            request: "0".into(),
            image: "0".into(),
            input_cache_read: "0".into(),
            input_cache_write: "0".into(),
        },
        soma: None,
    };

    // --- 5. Register the provider in the local discovery so the proxy
    //        can find it. (One PR away from being on-chain.) ----------------
    let provider_port = pick_free_port();
    let provider_endpoint = format!("http://127.0.0.1:{provider_port}");
    registry
        .register_provider(ProviderRecord {
            address: provider_addr,
            pubkey_hex: String::new(),
            endpoint: provider_endpoint.clone(),
            last_heartbeat_ms: 0,
        })
        .await
        .unwrap();

    // --- 6. Boot the provider ------------------------------------------------
    // SAFETY: tests are single-threaded WRT env vars at this point (we
    // haven't spawned any task that reads env). The provider only
    // reads `OPENROUTER_API_KEY` once during `OpenRouterBackend::new`.
    unsafe { std::env::set_var("OPENROUTER_API_KEY", "test-key"); }
    let prov_cfg = inference::server::Config {
        server: inference::server::config::Server {
            listen: format!("127.0.0.1:{provider_port}"),
            public_endpoint: provider_endpoint.clone(),
        },
        backend: inference::server::config::Backend {
            kind: "openrouter".into(),
            api_key_env: Some("OPENROUTER_API_KEY".into()),
            upstream_url: Some(upstream.uri()),
            endpoint_name: None,
        },
        auth: Default::default(),
        offerings: vec![card.clone()],
    };
    let prov_handle = tokio::spawn({
        let registry = registry.clone();
        let ledger_path = ledger_dir.path().to_path_buf();
        let provider_wallet = provider_wallet.clone();
        async move {
            inference::server::run(
                prov_cfg,
                provider_wallet,
                provider_addr,
                registry,
                ledger_path,
                0, // no heartbeats
            )
            .await
            .ok();
        }
    });

    // Wait for /health.
    wait_for_url(&format!("{provider_endpoint}/health")).await;

    // --- 7. Boot the proxy ---------------------------------------------------
    let proxy_port = pick_free_port();
    let proxy_wallet = Arc::new(wallet_for_path(&wallet_conf_path));
    let proxy_cfg = inference::proxy::Config {
        listen_addr: format!("127.0.0.1:{proxy_port}"),
        default_deposit_micros: 1_000_000,
        provider_cache_ttl_secs: 60,
    };
    let proxy_soma_home = TempDir::new().unwrap();
    let proxy_handle = tokio::spawn({
        let registry = registry.clone();
        let proxy_soma_home_path = proxy_soma_home.path().to_path_buf();
        let proxy_wallet = proxy_wallet.clone();
        async move {
            inference::proxy::run(proxy_cfg, proxy_wallet, payer, registry, proxy_soma_home_path)
                .await
                .ok();
        }
    });
    wait_for_url(&format!("http://127.0.0.1:{proxy_port}/v1/models")).await;

    // --- 8. Drive a chat completion through the stack ------------------------
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{proxy_port}/v1/chat/completions"))
        .json(&json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200, "chat must succeed");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["choices"][0]["message"]["content"].as_str(),
        Some("pong"),
    );

    // --- 9. Assert the proxy lazily opened a channel on-chain ---------------
    let provider_addr_no_prefix = provider_addr.to_string();
    let pointer_path = proxy_soma_home
        .path()
        .join("client/channels-by-provider")
        .join(format!("{}.txt", provider_addr_no_prefix.trim_start_matches("0x")));
    let channel_id_str = std::fs::read_to_string(&pointer_path)
        .unwrap_or_else(|e| panic!("expected proxy pointer at {pointer_path:?}: {e}"))
        .trim()
        .to_string();
    let channel_id: types::object::ObjectID = channel_id_str.parse().unwrap();
    let chan_before = provider_chain.get(channel_id).await.unwrap();
    assert_eq!(chan_before.payer, payer);
    assert_eq!(chan_before.payee, provider_addr);
    assert_eq!(chan_before.deposit, 1_000_000);
    assert_eq!(
        chan_before.settled_amount, 0,
        "no Settle has been submitted yet"
    );

    // --- 10. Trigger Settle directly via the SDK (mirrors what the
    //         provider's SIGTERM hook would do). Pulls the latest sig
    //         out of the in-memory ledger we wired into the provider. -------
    let provider_state = ledger
        .slot(&channel_id)
        .await
        .expect("provider ledger has a slot for the channel after one request");
    let final_pair = {
        let g = provider_state.lock().await;
        provider_channel
            .final_settlement(&g.state)
            .expect("provider holds an on-chain sig after one request")
    };
    let (voucher, sig) = final_pair;
    sdk::channel::settle(&provider_wallet, provider_addr, voucher, sig)
        .await
        .expect("provider settle on-chain");

    let chan_after = provider_chain.get(channel_id).await.unwrap();
    assert!(
        chan_after.settled_amount > 0,
        "settled_amount must increase after Settle (got {})",
        chan_after.settled_amount,
    );
    assert_eq!(
        chan_after.deposit + chan_after.settled_amount,
        chan_before.deposit,
        "deposit + settled must conserve the original deposit",
    );

    // --- cleanup -------------------------------------------------------------
    prov_handle.abort();
    proxy_handle.abort();
}

fn pick_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

async fn wait_for_url(url: &str) {
    for _ in 0..200 {
        if reqwest::get(url).await.map(|r| r.status().is_success()).unwrap_or(false) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("timed out waiting for {url}");
}
