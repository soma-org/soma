// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Live-OpenRouter variant of `e2e_localnet`: same in-process proxy +
// provider + real chain, but the provider's `Backend` points at the
// real OpenRouter `/v1/chat/completions` endpoint instead of a wiremock
// canned response. Validates the full vertical against a real upstream
// LLM:
//
//   real OpenRouter (`https://openrouter.ai/api/v1/chat/completions`)
//     ↑
//   inference::server  (provider; OpenRouter backend; on-chain channel)
//     ↑
//   inference::proxy   (client; signs vouchers; runs on real chain)
//     ↑
//   reqwest test client
//
// Gated on `OPENROUTER_API_KEY` so CI without credentials stays green.
//
// Run with:
//   OPENROUTER_API_KEY=$(grep OPENROUTER_API_KEY ../deep-research/.env | cut -d= -f2-) \
//     PYO3_PYTHON=python3 cargo test -p inference --test openrouter_chain_e2e -- --ignored --nocapture

#![cfg(not(msim))]

use std::sync::Arc;
use std::time::Duration;

use sdk::wallet_context::WalletContext;
use serde_json::json;
use tempfile::TempDir;
use test_cluster::TestClusterBuilder;
use types::base::SomaAddress;
use types::config::SOMA_CLIENT_CONFIG;
use types::effects::TransactionEffectsAPI as _;
use types::object::CoinType;

use inference::chain::chain::ChainChannelSurface;
use inference::chain::memory::MemoryDiscovery;
use inference::chain::{ChannelSurface, ProviderRecord, ProviderRegistry};
use inference::channel::{PaymentChannel as _, RunningTab};

const TEST_MODEL: &str = "google/gemma-4-31b-it";

fn wallet_for_path(path: &std::path::Path) -> WalletContext {
    WalletContext::new(path).expect("WalletContext from cluster's client.yaml")
}

/// Submit a `RegisterOffering` for `payee` against `TEST_MODEL` so the
/// OpenChannel path can find an offering to snapshot.
async fn register_test_offering(test_cluster: &test_cluster::TestCluster, payee: SomaAddress) {
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
                // Mirror OpenRouter's haiku-4.5 pricing approximately —
                // exact numbers don't matter for the round-trip, only
                // that they're non-zero so the `worst_case` math
                // produces a positive worst_case_micros.
                prompt_micros_per_1k: 1_000,
                completion_micros_per_1k: 5_000,
                cache_read_micros_per_1k: 100,
                cache_write_micros_per_1k: 1_000,
                request_micros: 0,
                ttft_bound_ms: 60_000,
                ttot_bound_ms: 10_000,
            },
        ),
    );
    let r = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(r.effects.status().is_ok(), "RegisterOffering: {:?}", r.effects.status());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live OpenRouter — set OPENROUTER_API_KEY before running"]
async fn openrouter_full_stack_round_trip() {
    let _api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("OPENROUTER_API_KEY not set — skipping openrouter_full_stack_round_trip");
            return;
        }
    };

    // 1. Real chain.
    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let payer = addrs[0];
    let provider_addr = addrs[1];
    let wallet_conf_path = test_cluster.swarm.dir().join(SOMA_CLIENT_CONFIG);

    // 2. In-memory provider discovery so the proxy can find the
    //    provider without standing up an indexer.
    let registry: Arc<dyn ProviderRegistry> = Arc::new(MemoryDiscovery::new());

    // 3. Register the on-chain offering before booting the provider —
    //    OpenChannel snapshots it onto the channel.
    register_test_offering(&test_cluster, provider_addr).await;

    // 4. Provider — points at the *real* OpenRouter upstream.
    let ledger_dir = TempDir::new().unwrap();
    let provider_wallet = Arc::new(wallet_for_path(&wallet_conf_path));
    let provider_chain: Arc<dyn ChannelSurface> =
        Arc::new(ChainChannelSurface::new(provider_wallet.clone(), provider_addr));
    let provider_channel_for_settle = Arc::new(RunningTab::for_provider(60));

    let card = inference::catalog::ModelCard {
        id: TEST_MODEL.to_string(),
        name: TEST_MODEL.to_string(),
        canonical_slug: None,
        hugging_face_id: Some(TEST_MODEL.to_string()),
        created: 0,
        description: None,
        context_length: 200_000,
        architecture: inference::catalog::Architecture {
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            tokenizer: "Claude".into(),
            instruct_type: None,
        },
        top_provider: inference::catalog::TopProvider {
            context_length: 200_000,
            max_completion_tokens: Some(1024),
            is_moderated: false,
        },
        supported_parameters: vec!["max_tokens".into(), "temperature".into()],
        default_parameters: None,
        expiration_date: None,
        ttft_bound_ms: Some(60_000),
        ttot_bound_ms: Some(10_000),
        pricing: inference::catalog::Pricing {
            // 0.000001 USD/token = $1/1M tokens. Matches the chain
            // snapshot's 1000 micros / 1k tokens above.
            prompt: "0.000001".into(),
            completion: "0.000005".into(),
            request: "0".into(),
            image: "0".into(),
            input_cache_read: "0.0000001".into(),
            input_cache_write: "0.000001".into(),
        },
        soma: None,
    };

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

    let prov_cfg = inference::server::Config {
        server: inference::server::config::Server {
            listen: format!("127.0.0.1:{provider_port}"),
            public_endpoint: provider_endpoint.clone(),
        },
        backend: inference::server::config::Backend {
            kind: "openrouter".into(),
            api_key_env: Some("OPENROUTER_API_KEY".into()),
            upstream_url: None, // default https://openrouter.ai/api/v1
            endpoint_name: None,
            slots: None,
        },
        auth: Default::default(),
        offerings: vec![card.clone()],
        auto_settle: Default::default(),
        autoprice: Default::default(),
    };
    let prov_handle = tokio::spawn({
        let ledger_path = ledger_dir.path().to_path_buf();
        let provider_wallet = provider_wallet.clone();
        async move {
            inference::server::run(prov_cfg, provider_wallet, provider_addr, ledger_path)
                .await
                .ok();
        }
    });
    wait_for_url(&format!("{provider_endpoint}/health")).await;

    // 5. Proxy.
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
        prewarm_model: None,
        // See e2e_localnet.rs comment.
        probe_liveness: false,
        liveness_refresh_secs: 30,
        liveness_timeout_ms: 1_500,
        local_token: None,
        low_balance_threshold_micros: 1_000_000,
        wallet_address: None,
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

    // 6. Live chat completion through the full stack.
    let client = reqwest::Client::builder().timeout(Duration::from_secs(120)).build().unwrap();
    let resp = client
        .post(format!("http://127.0.0.1:{proxy_port}/v1/chat/completions"))
        .json(&json!({
            "model": TEST_MODEL,
            "max_tokens": 32,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "Respond with exactly one short sentence."},
                {"role": "user",   "content": "What is 2 + 2?"}
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body_text = resp.text().await.unwrap();
    assert!(status.is_success(), "chat must succeed via proxy: {status}: {body_text}",);
    let body: serde_json::Value = serde_json::from_str(&body_text).unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap_or_default();
    assert!(!content.is_empty(), "non-empty completion: {body}");
    let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    assert!(prompt_tokens > 0, "prompt_tokens missing");
    assert!(completion_tokens > 0, "completion_tokens missing");
    eprintln!(
        "openrouter_chain_e2e: prompt={prompt_tokens} completion={completion_tokens} content={content:?}"
    );

    // 7. Provider's persisted ledger holds the channel + signed voucher.
    //    (The provider task owns its own in-memory Ledger — read off disk.)
    let provider_channels_dir = ledger_dir.path().join("provider").join("channels");
    let state = read_first_provider_slot(&provider_channels_dir)
        .expect("provider should have persisted at least one channel ledger entry");
    let channel_id = state.channel_id;

    let chan_before = provider_chain.get(channel_id).await.unwrap();
    assert_eq!(chan_before.payer(), payer);
    assert_eq!(chan_before.payee(), provider_addr);
    assert_eq!(chan_before.model_id(), TEST_MODEL);
    assert_eq!(
        chan_before.settled_amount(),
        0,
        "no Settle has run yet — provider holds voucher off-chain"
    );

    // The snapshotted on-chain prices match the offering.
    assert_eq!(chan_before.prompt_micros_per_1k(), 1_000);
    assert_eq!(chan_before.completion_micros_per_1k(), 5_000);

    // The persisted ledger advanced cumulative_*_tokens from the OpenRouter
    // response.
    assert!(
        state.cumulative_prompt_tokens >= prompt_tokens,
        "provider didn't bump cumulative_prompt_tokens: have {}, expected ≥{}",
        state.cumulative_prompt_tokens,
        prompt_tokens,
    );
    assert!(
        state.cumulative_completion_tokens >= completion_tokens,
        "provider didn't bump cumulative_completion_tokens: have {}, expected ≥{}",
        state.cumulative_completion_tokens,
        completion_tokens,
    );

    // 8. Driver-style Settle: rebuild voucher from the snapshotted signed
    //    values, hit the real chain. Mirrors what the provider's
    //    auto-settle ticker or SIGTERM hook would do.
    let (voucher, sig) = provider_channel_for_settle
        .final_settlement(&state)
        .expect("provider holds an on-chain sig + signed snapshot");
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
        "deposit + settled must conserve the original deposit"
    );

    prov_handle.abort();
    proxy_handle.abort();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn read_first_provider_slot(
    dir: &std::path::Path,
) -> Option<inference::channel::running_tab::TabProviderState> {
    let entries = std::fs::read_dir(dir).ok()?;
    for e in entries.flatten() {
        if e.path().extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let bytes = std::fs::read(e.path()).ok()?;
        if let Ok(s) =
            serde_json::from_slice::<inference::channel::running_tab::TabProviderState>(&bytes)
        {
            return Some(s);
        }
    }
    None
}
