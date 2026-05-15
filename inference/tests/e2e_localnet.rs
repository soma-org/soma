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
use inference::chain::memory::MemoryDiscovery;
use inference::chain::{ChannelSurface, ProviderRecord, ProviderRegistry};
use inference::channel::{PaymentChannel as _, RunningTab};
use inference::server::ledger::Ledger;
use types::effects::TransactionEffectsAPI as _;

const TEST_MODEL: &str = "anthropic/claude-sonnet-4.6";

/// Submit a `RegisterOffering` for `payee` against `TEST_MODEL` so the
/// OpenChannel path can find an offering to snapshot. Idempotent: if
/// the offering already exists on chain we exit cleanly.
async fn register_test_offering(
    test_cluster: &test_cluster::TestCluster,
    payee: SomaAddress,
) {
    use types::offering::Offering;
    let offering_id = Offering::derive_id(payee, TEST_MODEL);
    if test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_object_store().get_object(&offering_id).is_some())
    {
        return;
    }
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        payee,
        types::transaction::TransactionKind::RegisterOffering(
            types::transaction::RegisterOfferingArgs {
                model_id: TEST_MODEL.to_string(),
                prompt_micros_per_1k: 200,
                completion_micros_per_1k: 400,
                cache_read_micros_per_1k: 0,
                cache_write_micros_per_1k: 0,
                request_micros: 0,
                ttft_bound_ms: 60_000,
                ttot_bound_ms: 10_000,
            },
        ),
    );
    let r = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(
        r.effects.status().is_ok(),
        "RegisterOffering: {:?}",
        r.effects.status()
    );
}

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

    // --- 3. Provider config (in-memory ledger + MemoryDiscovery for registry) -
    let ledger_dir = TempDir::new().unwrap();
    let registry: Arc<dyn ProviderRegistry> = Arc::new(MemoryDiscovery::new());

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
        })
        .await
        .unwrap();

    // Register an on-chain offering for the provider against TEST_MODEL —
    // OpenChannel's executor reads it to snapshot prices + SLA bounds onto
    // the channel.
    register_test_offering(&test_cluster, provider_addr).await;

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
        auto_settle: Default::default(),
    };
    let prov_handle = tokio::spawn({
        let ledger_path = ledger_dir.path().to_path_buf();
        let provider_wallet = provider_wallet.clone();
        async move {
            inference::server::run(
                prov_cfg,
                provider_wallet,
                provider_addr,
                ledger_path,
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
        routing: Default::default(),
        trusted_providers_only: false,
        trusted_providers_url: None,
        trusted_providers_refresh_secs: 600,
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
    // The provider task owns its own `Ledger` instance pointing at the same
    // dir — in-memory state isn't shared, but the disk-persisted slots are.
    // Read the slot from disk to find the channel id + held signature.
    let provider_channels_dir = ledger_dir.path().join("provider").join("channels");
    let state = read_first_provider_slot(&provider_channels_dir)
        .expect("provider should have persisted at least one channel ledger entry");
    let channel_id = state.channel_id;
    let chan_before = provider_chain.get(channel_id).await.unwrap();
    assert_eq!(chan_before.payer(), payer);
    assert_eq!(chan_before.payee(), provider_addr);
    assert_eq!(chan_before.deposit(), 1_000_000);
    assert_eq!(
        chan_before.settled_amount(), 0,
        "no Settle has been submitted yet"
    );

    // --- 10. Trigger Settle directly via the SDK using the persisted
    //         provider state. Mirrors what the provider's SIGTERM hook
    //         or auto-settle ticker would do. ----------------------------
    let (voucher, sig) = provider_channel
        .final_settlement(&state)
        .expect("provider holds an on-chain sig after one request");
    sdk::channel::settle(&provider_wallet, provider_addr, voucher, sig)
        .await
        .expect("provider settle on-chain");

    let chan_after = provider_chain.get(channel_id).await.unwrap();
    assert!(
        chan_after.settled_amount() > 0,
        "settled_amount must increase after Settle (got {})",
        chan_after.settled_amount(),
    );
    assert_eq!(
        chan_after.deposit() + chan_after.settled_amount(),
        chan_before.deposit(),
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

/// Read the first persisted provider slot off disk. The provider's
/// `Ledger` writes one JSON file per `(channel_id)` slot into
/// `<soma_home>/provider/channels/`. Tests don't share the provider's
/// in-memory `Ledger` (different `Arc`), so reading from disk is the
/// canonical way to assert against provider-side state.
fn read_first_provider_slot(
    dir: &std::path::Path,
) -> Option<inference::channel::running_tab::TabProviderState> {
    let entries = std::fs::read_dir(dir).ok()?;
    for e in entries.flatten() {
        if e.path().extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let bytes = std::fs::read(e.path()).ok()?;
        if let Ok(s) = serde_json::from_slice::<
            inference::channel::running_tab::TabProviderState,
        >(&bytes)
        {
            return Some(s);
        }
    }
    None
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

/// Stateless-proxy cold-start. Verifies that after the proxy crashes
/// with non-zero unsettled cumulative still on the provider's ledger,
/// a freshly-booted proxy (empty in-memory state) can resume the
/// channel without issuing a non-monotonic voucher.
///
/// Sequence:
///   1. Boot provider + proxy v1.
///   2. Send N chat requests. Provider's ledger holds cumulative = K > 0.
///   3. Abort proxy v1 (no SIGTERM-settle path — ledger entry stays).
///   4. Boot proxy v2 (fresh `ClientStore`, same wallet/registry).
///      Proxy v2 has zero in-memory state; the channel id is gone from
///      its `pointer` map.
///   5. Send 1 more chat request. The router's lazy-open path either
///      reuses the existing channel (after seeding `pointer`) or
///      opens a new one. Either way, the resulting cumulative in the
///      provider's ledger must be strictly greater than K.
///
/// Caveat: this scenario exercises the cold-start *codepath* (in-memory
/// store empty → next request succeeds), not the specific recovery
/// path where the proxy *knows* about an existing channel. The latter
/// hits in production via the indexer — exercising it here would
/// require booting an indexer too, which we cover in
/// `indexer-e2e-tests/inference_pipeline_e2e.rs`.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn stateless_proxy_cold_start_resumes_safely() {
    // --- 1. Boot the chain -------------------------------------------------
    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let provider_addr = addrs[1];
    let wallet_conf_path = test_cluster.swarm.dir().join(SOMA_CLIENT_CONFIG);

    // --- 2. Wiremock upstream (deterministic OpenAI body) ------------------
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
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_body))
        .mount(&upstream)
        .await;

    // --- 3. Shared registry + provider state ------------------------------
    let ledger_dir = TempDir::new().unwrap();
    let registry: Arc<dyn ProviderRegistry> = Arc::new(MemoryDiscovery::new());
    let provider_wallet = Arc::new(wallet_for_path(&wallet_conf_path));
    let provider_chain: Arc<dyn ChannelSurface> =
        Arc::new(ChainChannelSurface::new(provider_wallet.clone(), provider_addr));
    let ledger = Ledger::new(ledger_dir.path().to_path_buf());

    // --- 4. Card --------------------------------------------------------
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

    // --- 5. Provider seed in registry + boot --------------------------------
    let provider_port = pick_free_port();
    let provider_endpoint = format!("http://127.0.0.1:{provider_port}");
    registry
        .register_provider(ProviderRecord {
            address: provider_addr,
            pubkey_hex: String::new(),
            endpoint: provider_endpoint.clone(),
        })
        .await
        .unwrap();
    register_test_offering(&test_cluster, provider_addr).await;

    // SAFETY: tests are single-threaded WRT env vars at this point.
    unsafe { std::env::set_var("OPENROUTER_API_KEY", "test-key"); }
    let prov_cfg = inference::server::Config {
        server: inference::server::config::Server {
            listen: format!("127.0.0.1:{provider_port}"),
            public_endpoint: provider_endpoint.clone(),
        },
        backend: inference::server::config::Backend {
            kind: "openrouter".to_string(),
            api_key_env: Some("OPENROUTER_API_KEY".to_string()),
            upstream_url: Some(upstream.uri()),
            endpoint_name: None,
        },
        auth: Default::default(),
        offerings: vec![card.clone()],
        auto_settle: Default::default(),
    };
    let prov_handle = tokio::spawn({
        let ledger_path = ledger_dir.path().to_path_buf();
        let provider_wallet = provider_wallet.clone();
        async move {
            inference::server::run(
                prov_cfg,
                provider_wallet,
                provider_addr,
                ledger_path,
            )
            .await
            .ok();
        }
    });
    wait_for_url(&format!("{provider_endpoint}/health")).await;

    // --- 6. Proxy v1: send 2 requests --------------------------------------
    let proxy_v1_port = pick_free_port();
    let proxy_wallet = Arc::new(wallet_for_path(&wallet_conf_path));
    let proxy_v1_cfg = inference::proxy::Config {
        listen_addr: format!("127.0.0.1:{proxy_v1_port}"),
        default_deposit_micros: 1_000_000,
        provider_cache_ttl_secs: 60,
        routing: Default::default(),
        trusted_providers_only: false,
        trusted_providers_url: None,
        trusted_providers_refresh_secs: 600,
    };
    let proxy_v1_home = TempDir::new().unwrap();
    let proxy_v1_handle = tokio::spawn({
        let registry = registry.clone();
        let proxy_home = proxy_v1_home.path().to_path_buf();
        let proxy_wallet = proxy_wallet.clone();
        async move {
            inference::proxy::run(
                proxy_v1_cfg,
                proxy_wallet,
                payer,
                registry,
                proxy_home,
            )
            .await
            .ok();
        }
    });
    wait_for_url(&format!("http://127.0.0.1:{proxy_v1_port}/v1/models")).await;

    let client = reqwest::Client::new();
    for i in 0..2 {
        let resp = client
            .post(format!(
                "http://127.0.0.1:{proxy_v1_port}/v1/chat/completions"
            ))
            .json(&json!({
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": format!("ping {i}")}],
                "max_tokens": 4,
                "stream": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200, "chat #{i} via proxy v1 must succeed");
    }

    // The provider task owns its own `Ledger` instance, so the test's
    // `ledger` variable doesn't see the provider's in-memory cache.
    // Read the persisted ledger files directly off disk instead — they
    // hold the same `TabProviderState` we'd get from `snapshot()`.
    let provider_channels_dir = ledger_dir.path().join("provider").join("channels");
    let entries: Vec<_> = std::fs::read_dir(&provider_channels_dir)
        .expect("provider channels dir exists after 2 requests")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    assert!(
        !entries.is_empty(),
        "provider should have persisted at least one channel ledger entry"
    );
    let state_v1: inference::channel::running_tab::TabProviderState =
        serde_json::from_slice(&std::fs::read(entries[0].path()).unwrap())
            .expect("ledger json parses");
    let channel_id = state_v1.channel_id;
    let cumulative_after_v1 = state_v1.cumulative_authorized_micros;
    assert!(
        cumulative_after_v1 > 0,
        "v1 should have authorized at least one request: cumulative={cumulative_after_v1}"
    );

    // Verify on-chain channel still has settled_amount = 0 (no SIGTERM
    // settle has run yet — the provider's signature is in the ledger).
    let chan_before = provider_chain.get(channel_id).await.unwrap();
    assert_eq!(
        chan_before.settled_amount(), 0,
        "no settle yet — proxy hasn't shut down, so the held voucher is still off-chain"
    );

    // --- 7. Kill proxy v1 — simulates a crash with held voucher ----------
    proxy_v1_handle.abort();
    // Give axum a moment to release the listen socket.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // --- 8. Proxy v2: fresh in-memory ClientStore --------------------------
    let proxy_v2_port = pick_free_port();
    let proxy_v2_cfg = inference::proxy::Config {
        listen_addr: format!("127.0.0.1:{proxy_v2_port}"),
        default_deposit_micros: 1_000_000,
        provider_cache_ttl_secs: 60,
        routing: Default::default(),
        trusted_providers_only: false,
        trusted_providers_url: None,
        trusted_providers_refresh_secs: 600,
    };
    let proxy_v2_home = TempDir::new().unwrap();
    let proxy_v2_handle = tokio::spawn({
        let registry = registry.clone();
        let proxy_home = proxy_v2_home.path().to_path_buf();
        let proxy_wallet = proxy_wallet.clone();
        async move {
            inference::proxy::run(
                proxy_v2_cfg,
                proxy_wallet,
                payer,
                registry,
                proxy_home,
            )
            .await
            .ok();
        }
    });
    wait_for_url(&format!("http://127.0.0.1:{proxy_v2_port}/v1/models")).await;

    // --- 9. One more chat — must succeed (no non-monotonic rejection) ----
    let resp = client
        .post(format!(
            "http://127.0.0.1:{proxy_v2_port}/v1/chat/completions"
        ))
        .json(&json!({
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "after-crash"}],
            "max_tokens": 4,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "post-cold-start chat must succeed via proxy v2"
    );

    // --- 10. Provider's ledger must reflect a successful post-crash auth.
    // Two acceptable shapes:
    //   (a) proxy v2 reused the old channel — its cumulative grew.
    //   (b) proxy v2 opened a fresh channel — there are now ≥2 ledger
    //       entries, the new one with cumulative > 0.
    // Either path proves the provider didn't reject a non-monotonic
    // voucher, which is the contract under test.
    let entries_v2: Vec<_> = std::fs::read_dir(&provider_channels_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    let states_v2: Vec<inference::channel::running_tab::TabProviderState> = entries_v2
        .iter()
        .map(|e| {
            serde_json::from_slice(&std::fs::read(e.path()).unwrap()).expect("parse")
        })
        .collect();

    let total_consumed_v2: u64 =
        states_v2.iter().map(|s| s.total_consumed_micros).sum();
    let total_consumed_v1 = state_v1.total_consumed_micros;
    assert!(
        total_consumed_v2 > total_consumed_v1,
        "total consumed across channels must grow after proxy v2's request: \
         v1={total_consumed_v1}, v2={total_consumed_v2} — \
         the provider must have served the v2 request without a non-monotonic rejection"
    );

    // Sanity: if v2 reused the same channel, that channel's cumulative
    // must have grown strictly. If v2 opened a new channel, there
    // should be ≥2 ledger entries.
    let reused = states_v2.iter().any(|s| s.channel_id == channel_id);
    let same_channel_grew = states_v2
        .iter()
        .find(|s| s.channel_id == channel_id)
        .map(|s| s.cumulative_authorized_micros > cumulative_after_v1)
        .unwrap_or(false);
    if reused {
        assert!(
            same_channel_grew || states_v2.len() > 1,
            "if v2 saw the old channel, its cumulative must grow OR a new channel must have opened"
        );
    } else {
        assert!(
            states_v2.len() >= 1,
            "v2 should have at least one ledger entry"
        );
    }

    // --- cleanup -----------------------------------------------------------
    prov_handle.abort();
    proxy_v2_handle.abort();
}
