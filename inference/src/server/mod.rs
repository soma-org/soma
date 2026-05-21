// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Provider-side server: fronts an OpenAI-compatible upstream
//! ([`backend::Backend`]) behind a [`crate::channel::PaymentChannel`]-guarded
//! HTTP path.

pub mod auth;
pub mod backend;
pub mod config;
pub mod handler;
pub mod ledger;

use std::path::PathBuf;
use std::sync::Arc;

use ::types::base::SomaAddress;
use anyhow::Context as _;
use sdk::wallet_context::WalletContext;

pub use config::Config;

use crate::chain::ChannelSurface;
use crate::chain::chain::ChainChannelSurface;
use crate::channel::RunningTab;

/// Run the provider until shutdown (SIGTERM / SIGINT).
///
/// On graceful shutdown, every channel with a stored on-chain
/// signature is settled — the provider claims its earned share
/// before exiting.
pub async fn run(
    cfg: Config,
    wallet: Arc<WalletContext>,
    address: SomaAddress,
    soma_home: PathBuf,
) -> anyhow::Result<()> {
    tracing::info!(address = %address, "loaded provider identity");

    let backend: Arc<dyn backend::Backend> = match cfg.backend.kind.as_str() {
        "openrouter" => backend::openrouter::OpenRouterBackend::new(&cfg)?,
        "vast" => backend::vast::VastBackend::new(&cfg)?,
        other => anyhow::bail!("unsupported backend kind: {other}"),
    };

    let mut catalog: Vec<crate::catalog::ModelCard> =
        backend.list_models().await.context("backend list_models on boot")?.data;
    if !cfg.offerings.is_empty() {
        catalog = cfg.offerings.clone();
    }

    let chain: Arc<dyn ChannelSurface> =
        Arc::new(ChainChannelSurface::new(wallet.clone(), address));

    let channel = Arc::new(RunningTab::for_provider(cfg.auth.clock_skew_tolerance_secs));
    let ledger = ledger::Ledger::new(soma_home);

    // On-chain registry + offerings, in parallel. Both go through the
    // finality-only path (no `wait_for_checkpoint`) — the boot sequence
    // has no read-after-write dependency on the indexer here, the
    // heartbeat re-stamps the provider record later, and the proxy
    // picks up offerings on its own discovery refresh cadence. Each
    // call independently logs a warning on failure; one bad offering
    // doesn't sink the others.
    let provider_fut = {
        let wallet = wallet.clone();
        let endpoint = cfg.server.public_endpoint.clone();
        async move {
            if let Err(e) =
                sdk::provider::register_or_update_no_wait(&wallet, address, endpoint).await
            {
                tracing::warn!(err = %e, "on-chain provider register/update failed (continuing)");
            }
        }
    };
    let offering_futs = cfg.offerings.iter().map(|card| {
        let wallet = wallet.clone();
        let model_id = card.id.clone();
        let prices = crate::pricing::offering_prices_from_card(card);
        async move {
            match sdk::offering::register_or_update_no_wait(
                &wallet,
                address,
                model_id.clone(),
                prices,
            )
            .await
            {
                Ok(()) => tracing::info!(model_id = %model_id, "offering registered on-chain"),
                Err(e) => tracing::warn!(
                    model_id = %model_id,
                    err = %e,
                    "on-chain offering register/update failed (continuing)",
                ),
            }
        }
    });
    let _ = futures::future::join(provider_fut, futures::future::join_all(offering_futs)).await;

    // Heartbeat. Periodically re-submits `UpdateProvider` so the
    // on-chain `Provider.registered_at_ms` keeps advancing while the
    // server is alive — the boot registration above stamps it once;
    // discovery needs an ongoing signal to tell a live provider from
    // one that vanished without unregistering.
    spawn_heartbeat(
        wallet.clone(),
        address,
        cfg.server.public_endpoint.clone(),
        HEARTBEAT_INTERVAL_SECS,
    );

    // Background settle ticker. Insurance against provider crashes —
    // SIGTERM hook only fires on graceful shutdown, so without this
    // any unsettled drift between ticks is earnings the provider
    // loses on an OOM / k8s restart.
    spawn_settle_ticker(
        chain.clone(),
        ledger.clone(),
        channel.clone(),
        cfg.auto_settle.interval_secs,
    );

    let state = Arc::new(handler::ProviderState {
        chain: chain.clone(),
        backend: backend.clone(),
        channel: channel.clone(),
        ledger: ledger.clone(),
        catalog,
    });

    let app = handler::build_router(state);
    let listener = tokio::net::TcpListener::bind(&cfg.server.listen)
        .await
        .with_context(|| format!("bind {}", cfg.server.listen))?;
    tracing::info!(addr = %cfg.server.listen, "inference server listening");

    let serve = axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(shutdown_signal(chain.clone(), ledger.clone(), channel.clone()));
    serve.await?;
    Ok(())
}

/// How often the provider re-stamps its on-chain heartbeat.
const HEARTBEAT_INTERVAL_SECS: u64 = 300;

/// Periodically re-submits `UpdateProvider` so the on-chain
/// `Provider.registered_at_ms` keeps advancing while the server is
/// alive. Discovery can then treat a provider whose heartbeat has gone
/// stale (no beat for several intervals) as gone — a crash-safe signal
/// the boot-time registration alone can't give, since it stamps the
/// field once and never again.
///
/// Uses `register_or_update` rather than `update` so a heartbeat also
/// heals a boot registration that failed transiently. Best-effort: a
/// failed beat is logged, not fatal; the next tick retries.
fn spawn_heartbeat(
    wallet: Arc<WalletContext>,
    address: SomaAddress,
    endpoint: String,
    interval_secs: u64,
) {
    if interval_secs == 0 {
        tracing::info!("provider heartbeat disabled (interval_secs = 0)");
        return;
    }
    tracing::info!(interval_secs, "provider heartbeat armed");
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        // Skip the immediate first tick — boot registration just ran.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // Finality-only — like boot, the heartbeat has no read-
            // after-write dependency on the indexer and shouldn't be
            // paying the up-to-30s checkpoint-inclusion wait.
            match sdk::provider::register_or_update_no_wait(&wallet, address, endpoint.clone())
                .await
            {
                Ok(()) => tracing::debug!("provider heartbeat sent"),
                Err(e) => tracing::warn!(err = %e, "provider heartbeat failed (will retry)"),
            }
        }
    });
}

/// Background ticker that settles every channel with new progress
/// since the last submitted `Settle`. Runs on a fixed interval; the
/// SIGTERM hook still drains everything on graceful shutdown, but
/// this caps the worst-case "earnings lost on a crash" to roughly
/// the inter-tick interval.
///
/// Skips channels with no progress (`cumulative_authorized ==
/// last_settled_at_amount`) so a quiet provider doesn't burn gas
/// re-submitting the same voucher every tick. On a successful
/// `Settle`, advances `last_settled_at_amount` so the next tick can
/// tell whether to skip again.
fn spawn_settle_ticker(
    chain: Arc<dyn ChannelSurface>,
    ledger: ledger::Ledger,
    channel: Arc<RunningTab>,
    interval_secs: u64,
) {
    if interval_secs == 0 {
        tracing::info!("auto-settle ticker disabled (interval_secs = 0)");
        return;
    }
    tracing::info!(interval_secs, "auto-settle ticker armed");
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        // Skip the immediate first tick — server boot already touches
        // the chain, no need to pile on.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            run_settle_pass(chain.as_ref(), &ledger, channel.as_ref()).await;
        }
    });
}

/// One pass over the ledger: settle every slot with new progress.
/// Pulled out of the ticker so the SIGTERM hook (which already does
/// the same walk-and-settle) can share the per-slot logic if we
/// want to consolidate later.
async fn run_settle_pass(
    chain: &dyn ChannelSurface,
    ledger: &ledger::Ledger,
    channel: &RunningTab,
) {
    use crate::channel::PaymentChannel as _;
    let snapshot = ledger.snapshot().await;
    for (id, state) in snapshot {
        // Skip channels with no new progress since the last settle.
        if state.cumulative_authorized_micros <= state.last_settled_at_amount {
            continue;
        }
        let pair = channel.final_settlement(&state);
        let Some((voucher, sig)) = pair else {
            continue;
        };
        let new_settled = state.cumulative_authorized_micros;
        match chain.settle(voucher, sig).await {
            Ok(()) => {
                tracing::info!(
                    channel = %id,
                    cumulative = new_settled,
                    "auto-settle: submitted",
                );
                // Advance the high-watermark + persist so the next
                // tick can tell whether there's been progress and
                // a crash-then-restart doesn't double-settle.
                if let Some(slot) = ledger.slot(&id).await {
                    let mut g = slot.lock().await;
                    if new_settled > g.state.last_settled_at_amount {
                        g.state.last_settled_at_amount = new_settled;
                        if let Err(e) = ledger.persist(&mut g).await {
                            tracing::warn!(
                                channel = %id,
                                err = %e,
                                "auto-settle: persisting last_settled_at_amount failed",
                            );
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!(channel = %id, err = %e, "auto-settle: settle failed");
            }
        }
    }
}

/// Wait for SIGTERM/SIGINT, then walk the ledger and settle every
/// channel that has a stored on-chain signature. We use the SDK's
/// `settle` helper so the on-chain side is identical to a manual
/// `soma channel settle` call.
async fn shutdown_signal(
    chain: Arc<dyn ChannelSurface>,
    ledger: ledger::Ledger,
    channel: Arc<RunningTab>,
) {
    use crate::channel::PaymentChannel as _;
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
    tracing::info!("shutdown signal received; settling open channels…");

    let snapshot = ledger.snapshot().await;
    for (id, state) in snapshot {
        // Same guard as `run_settle_pass`: if the auto-settle ticker
        // already drained the channel, replaying the same cumulative
        // would trip `ChannelVoucherNotMonotonic` (the executor rejects
        // equal-or-lower vouchers). Skip cleanly instead of logging a
        // false-failure warn.
        if state.cumulative_authorized_micros <= state.last_settled_at_amount {
            continue;
        }
        let pair = channel.final_settlement(&state);
        let Some((voucher, sig)) = pair else {
            tracing::info!(channel = %id, "no signature held; skipping settle");
            continue;
        };
        match chain.settle(voucher, sig).await {
            Ok(()) => {
                tracing::info!(channel = %id, cumulative = state.cumulative_authorized_micros, "settled on shutdown")
            }
            Err(e) => tracing::warn!(channel = %id, err = %e, "settle on shutdown failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use ::types::channel::{Channel, Voucher};
    use ::types::crypto::GenericSignature;
    use ::types::object::{CoinType, ObjectID};
    use async_trait::async_trait;
    use fastcrypto::traits::ToFromBytes;

    use crate::chain::{ChainError, ChannelSurface};
    use crate::channel::running_tab::TabProviderState;

    use super::*;

    /// Counts every `settle` call so the test can verify which
    /// channels the ticker actually submitted for. Other methods
    /// panic — this fake is only built for `run_settle_pass`.
    struct CountingChain {
        settles: StdMutex<Vec<(ObjectID, u64)>>,
    }

    #[async_trait]
    impl ChannelSurface for CountingChain {
        async fn open(
            &self,
            _payee: SomaAddress,
            _coin_type: CoinType,
            _deposit_amount: u64,
            _model_id: String,
        ) -> Result<(ObjectID, Option<Channel>), ChainError> {
            unimplemented!()
        }
        async fn get(&self, _id: ObjectID) -> Result<Channel, ChainError> {
            unimplemented!()
        }
        async fn settle(&self, voucher: Voucher, _sig: GenericSignature) -> Result<(), ChainError> {
            self.settles.lock().unwrap().push((voucher.channel_id(), voucher.cumulative_amount()));
            Ok(())
        }
        async fn top_up(
            &self,
            _id: ObjectID,
            _coin_type: CoinType,
            _amount: u64,
        ) -> Result<(), ChainError> {
            unimplemented!()
        }
        async fn request_close(&self, _id: ObjectID) -> Result<(), ChainError> {
            unimplemented!()
        }
        async fn withdraw_after_timeout(&self, _id: ObjectID) -> Result<(), ChainError> {
            unimplemented!()
        }
        fn signer_address(&self) -> SomaAddress {
            SomaAddress::ZERO
        }
    }

    /// Bogus signature shaped like a valid Ed25519 GenericSignature.
    fn dummy_sig() -> GenericSignature {
        let mut bytes = vec![0u8; 97];
        bytes[0] = 0; // ED25519 flag
        GenericSignature::from_bytes(&bytes).expect("ed25519 sig parses")
    }

    fn ed25519_addr_for_sig(sig: &GenericSignature) -> SomaAddress {
        // Auth signer that the dummy_sig "verifies" against — only
        // used by the ledger init path; the ticker doesn't verify.
        let _ = sig;
        SomaAddress::random()
    }

    /// `run_settle_pass` skips slots whose
    /// `cumulative_authorized_micros == last_settled_at_amount`
    /// (no progress) and submits a Settle for slots with progress.
    /// The persisted high-watermark advances on success so a second
    /// pass with no further activity submits nothing.
    #[tokio::test]
    async fn settle_pass_skips_no_progress_and_advances_watermark() {
        let tmp = tempfile::tempdir().unwrap();
        let ledger = ledger::Ledger::new(tmp.path().to_path_buf());
        let chain = Arc::new(CountingChain { settles: StdMutex::new(Vec::new()) });

        // Two channels: A has progress (50 → 75), B doesn't (90 == 90).
        let chan_a = ObjectID::random();
        let chan_b = ObjectID::random();
        let sig_a = dummy_sig();
        let sig_b = dummy_sig();

        let channel = Arc::new(RunningTab::for_provider(60));

        // Seed the ledger directly via a stand-in channel — we pre-set
        // last_settled_at_amount and cumulative_authorized_micros to
        // exercise the progress-filter logic.
        let chan_a_obj = Channel::new_for_testing(
            SomaAddress::random(),
            ed25519_addr_for_sig(&sig_a),
            ed25519_addr_for_sig(&sig_a),
            CoinType::Usdc,
            1_000,
        );
        let slot_a = ledger.init_slot(chan_a, &chan_a_obj).await.unwrap();
        {
            let mut g = slot_a.lock().await;
            g.state.cumulative_authorized_micros = 75;
            g.state.last_signed_cumulative_amount = 75;
            g.state.last_settled_at_amount = 50;
            g.state.last_onchain_sig = Some(sig_a.clone());
        }

        let chan_b_obj = Channel::new_for_testing(
            SomaAddress::random(),
            ed25519_addr_for_sig(&sig_b),
            ed25519_addr_for_sig(&sig_b),
            CoinType::Usdc,
            1_000,
        );
        let slot_b = ledger.init_slot(chan_b, &chan_b_obj).await.unwrap();
        {
            let mut g = slot_b.lock().await;
            g.state.cumulative_authorized_micros = 90;
            g.state.last_signed_cumulative_amount = 90;
            g.state.last_settled_at_amount = 90;
            g.state.last_onchain_sig = Some(sig_b.clone());
        }

        // First pass: should settle A only.
        run_settle_pass(chain.as_ref(), &ledger, channel.as_ref()).await;
        {
            let calls = chain.settles.lock().unwrap();
            assert_eq!(calls.len(), 1, "exactly one channel had progress");
            assert_eq!(calls[0], (chan_a, 75));
        }
        // Watermark advanced on A.
        assert_eq!(slot_a.lock().await.state.last_settled_at_amount, 75);
        // B unchanged.
        assert_eq!(slot_b.lock().await.state.last_settled_at_amount, 90);

        // Second pass with no further activity: no calls.
        run_settle_pass(chain.as_ref(), &ledger, channel.as_ref()).await;
        assert_eq!(
            chain.settles.lock().unwrap().len(),
            1,
            "no progress since last pass — must not re-submit",
        );

        // Bump A again; the next pass settles only A.
        {
            let mut g = slot_a.lock().await;
            g.state.cumulative_authorized_micros = 200;
            g.state.last_signed_cumulative_amount = 200;
        }
        run_settle_pass(chain.as_ref(), &ledger, channel.as_ref()).await;
        let calls = chain.settles.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1], (chan_a, 200));
    }
}
