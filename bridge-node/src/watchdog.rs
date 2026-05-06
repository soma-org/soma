//! Bridge conservation-invariant watchdog.
//!
//! Sui's `BridgeWatchDog` continuously monitors the Eth-locked vault balance
//! and the Sui-side total supply, alerting on divergence. Soma's watchdog
//! does the same for the single USDC route, plus auto-pauses the bridge if
//! a sustained violation is detected.
//!
//! ## The invariant
//!
//! At any time:
//!
//! ```text
//! Σ(USDC supply on Soma) ≤ Σ(USDC locked in Eth SomaBridge contract) + tolerance_in_flight
//! ```
//!
//! `tolerance_in_flight` allows for transient mismatches when a deposit
//! has been observed on Eth but the Soma-side mint hasn't been settled yet
//! (or vice versa). The watchdog's job is to flag *sustained* violations —
//! not single-poll race conditions.
//!
//! ## Auto-pause flow
//!
//! After `auto_pause_failure_threshold` consecutive polls report a real
//! violation:
//!
//! 1. Construct a `BridgeAction::EmergencyPause` with the current
//!    expected nonce (read off-chain from Soma's BridgeState).
//! 2. Hand the action to the action executor via the signing queue —
//!    the executor's `PeerBroadcastAggregator` fans out to peer
//!    bridge nodes via HTTP, collects a quorum cert, and submits the
//!    pause tx on Soma.
//!
//! Soma's pause threshold is intentionally low (450 BPS = 4.5%) so a
//! single watchdog flagging a real issue can usually get enough
//! validators to sign without majority cooperation.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use tracing::{error, info, warn};

use crate::action_executor::{BridgeActionExecutionWrapper, submit_to_executor};
use crate::error::BridgeResult;
use crate::eth_client::EthClient;
use crate::types::BridgeAction;

/// Configuration for the watchdog.
#[derive(Debug, Clone)]
pub struct WatchdogConfig {
    /// Eth-side USDC contract address. Used to read the bridge contract's
    /// locked balance via `balanceOf(bridge_contract)`.
    pub usdc_contract_address: String,
    /// Eth-side `SomaBridge` proxy address. The watchdog checks this
    /// contract's USDC balance — the "locked pool".
    pub eth_bridge_contract_address: String,
    /// How often to re-poll both chains.
    pub poll_interval: Duration,
    /// How many consecutive violation polls before the watchdog auto-pauses.
    /// One isolated mismatch can be a deposit-in-flight race; persistent
    /// mismatch is a real bridge state divergence.
    pub failure_threshold: u32,
    /// Tolerated USDC delta in microdollars (USDC has 6 decimals). A value
    /// of `0` requires exact equality; production should set this to cover
    /// realistic in-flight transfer volume — e.g. 1 USDC = 1_000_000.
    pub in_flight_tolerance_micro: u128,
}

/// Reader function returning the current Soma-side USDC supply, in
/// microdollars. Decoupled so the watchdog can be tested without spinning
/// up a full Soma node — production wiring reads from the live USDC
/// accumulator total via the Soma RPC.
pub type SomaSupplyReader =
    Arc<dyn Fn() -> futures_pin::PinnedFuture<BridgeResult<u128>> + Send + Sync>;

// Tiny pinned-future alias to avoid pulling a new crate. Use Pin<Box<...>>
// directly to keep the trait object 'static.
pub mod futures_pin {
    use std::future::Future;
    use std::pin::Pin;
    pub type PinnedFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
}

/// The watchdog actor. Construct with `BridgeWatchdog::new`, then call
/// `start` to spawn its background loop.
pub struct BridgeWatchdog {
    config: WatchdogConfig,
    eth_client: Arc<EthClient>,
    soma_supply: SomaSupplyReader,
    /// Sender on the action executor's signing queue. When the
    /// watchdog auto-pauses, it pushes an `EmergencyPause` action here
    /// and the executor fans it out via the peer-broadcast aggregator.
    signing_tx: mpsc::Sender<BridgeActionExecutionWrapper>,
    /// Next expected EmergencyOp seq num, fetched off-chain from the Soma
    /// bridge state. Closure so the watchdog can re-read on each violation.
    expected_pause_nonce: Arc<dyn Fn() -> u64 + Send + Sync>,
    /// Soma chain id for sanity-checking against a future multi-chain
    /// deployment; today only `SomaCustom`/`SomaTestnet`/`SomaMainnet`
    /// are used. Kept as a guardrail rather than load-bearing logic.
    _soma_chain_id: u64,
    /// Live committee — currently kept for future per-action-threshold
    /// computation; the executor does the threshold math now.
    _committee: Arc<RwLock<types::bridge::BridgeCommittee>>,
}

impl BridgeWatchdog {
    pub fn new(
        config: WatchdogConfig,
        eth_client: Arc<EthClient>,
        soma_supply: SomaSupplyReader,
        signing_tx: mpsc::Sender<BridgeActionExecutionWrapper>,
        expected_pause_nonce: Arc<dyn Fn() -> u64 + Send + Sync>,
        soma_chain_id: u64,
        committee: Arc<RwLock<types::bridge::BridgeCommittee>>,
    ) -> Self {
        Self {
            config,
            eth_client,
            soma_supply,
            signing_tx,
            expected_pause_nonce,
            _soma_chain_id: soma_chain_id,
            _committee: committee,
        }
    }

    pub fn start(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(self.run())
    }

    async fn run(self) {
        let mut consecutive_violations: u32 = 0;
        let mut auto_pause_emitted = false;
        let mut timer = interval(self.config.poll_interval);

        info!(
            poll_ms = self.config.poll_interval.as_millis() as u64,
            threshold = self.config.failure_threshold,
            "Bridge watchdog started"
        );

        loop {
            timer.tick().await;
            match self.poll_once().await {
                Ok(true) => {
                    // Healthy.
                    if consecutive_violations > 0 {
                        info!("Conservation invariant restored (was at {consecutive_violations} violations)");
                    }
                    consecutive_violations = 0;
                    auto_pause_emitted = false;
                }
                Ok(false) => {
                    consecutive_violations =
                        consecutive_violations.saturating_add(1);
                    warn!(
                        consecutive_violations,
                        threshold = self.config.failure_threshold,
                        "Bridge conservation invariant violated"
                    );
                    if consecutive_violations >= self.config.failure_threshold
                        && !auto_pause_emitted
                    {
                        if let Err(e) = self.emit_auto_pause().await {
                            error!("Failed to emit auto-pause sig: {e}");
                        } else {
                            auto_pause_emitted = true;
                            error!(
                                "AUTO-PAUSE: bridge conservation violated for {} consecutive polls — emergency pause sig posted",
                                consecutive_violations
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!("Watchdog poll error: {e}");
                    // Don't count poll *errors* (RPC down) as violations —
                    // we'd false-pause on routine network blips. The
                    // earlier `eth_client` rotation logic handles RPC health.
                }
            }
        }
    }

    /// Returns Ok(true) iff `soma_supply ≤ eth_locked + tolerance`.
    async fn poll_once(&self) -> BridgeResult<bool> {
        let eth_locked = self
            .eth_client
            .get_erc20_balance(
                &self.config.usdc_contract_address,
                &self.config.eth_bridge_contract_address,
            )
            .await?;
        let soma_supply = (self.soma_supply)().await?;
        info!(
            eth_locked,
            soma_supply, "Watchdog: Eth locked vs Soma supply"
        );
        let limit = eth_locked.saturating_add(self.config.in_flight_tolerance_micro);
        Ok(soma_supply <= limit)
    }

    /// Hand an `EmergencyPause` action to the executor's signing
    /// queue. The executor's loop will fan out to peers via the
    /// HTTP `/sign/emergency_button/{nonce}/0` endpoint (each peer
    /// re-verifies the action against its own pre-approved governance
    /// whitelist before signing) and submit the resulting cert on
    /// chain.
    ///
    /// Because the pause action is just queued here, this method
    /// returns as soon as the queue accepts it — actual on-chain
    /// landing happens in the executor's pipeline. Watchdog logs the
    /// fact that it tried; subsequent visibility comes from the
    /// executor's own structured logs.
    async fn emit_auto_pause(&self) -> BridgeResult<()> {
        let nonce = (self.expected_pause_nonce)();
        let pause_action = BridgeAction::EmergencyPause { nonce };
        submit_to_executor(&self.signing_tx, pause_action)
            .await
            .map_err(|e| {
                crate::error::BridgeError::Internal(format!(
                    "watchdog: failed to enqueue pause action: {e}"
                ))
            })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::atomic::AtomicBool;
    use crate::action_executor::BridgeActionExecutionWrapper;

    /// Construct a watchdog with `soma_supply` always returning `Ok(u128_value)`,
    /// `eth_locked` returning `Ok(u128_value)` via a fake EthClient
    /// (NOTE: EthClient::new_for_test gives us a non-functional one;
    /// for these tests we synthesize one with a wiremock-served
    /// `eth_call` for balanceOf, but that's heavy. Instead, we cover
    /// the *enqueue* contract: when emit_auto_pause is called, the
    /// signing queue gets the expected action shape. The poll-loop
    /// integration with EthClient is exercised in #62's e2e tests.).
    #[tokio::test]
    async fn auto_pause_emits_expected_action_shape() {
        let (signing_tx, mut signing_rx) =
            mpsc::channel::<BridgeActionExecutionWrapper>(8);
        let nonce_counter = Arc::new(AtomicU64::new(42));
        let nc = nonce_counter.clone();
        let expected_nonce: Arc<dyn Fn() -> u64 + Send + Sync> =
            Arc::new(move || nc.load(Ordering::SeqCst));
        let soma_supply: SomaSupplyReader =
            Arc::new(|| Box::pin(async { Ok::<u128, _>(0u128) }));

        let config = WatchdogConfig {
            usdc_contract_address: "0x0".to_string(),
            eth_bridge_contract_address: "0x0".to_string(),
            poll_interval: Duration::from_millis(50),
            failure_threshold: 1,
            in_flight_tolerance_micro: 0,
        };
        let eth = Arc::new(crate::eth_client::EthClient::new_for_test(
            "0x0".to_string(),
        ));
        let committee = Arc::new(RwLock::new(types::bridge::BridgeCommittee::empty()));
        let watchdog = BridgeWatchdog::new(
            config,
            eth,
            soma_supply,
            signing_tx,
            expected_nonce,
            0,
            committee,
        );

        // Directly invoke emit_auto_pause (private — call via a small
        // public test hook? We don't have one. Instead, exercise the
        // contract via the channel: push a fake call by spawning the
        // watchdog and observing.). Simpler: assert the action a
        // hand-built emit_auto_pause clone would send, by reading
        // the queue.
        watchdog
            .emit_auto_pause()
            .await
            .expect("emit_auto_pause should enqueue");
        let wrap = signing_rx.recv().await.expect("queued action");
        match wrap.0 {
            BridgeAction::EmergencyPause { nonce } => assert_eq!(nonce, 42),
            other => panic!("expected EmergencyPause, got {other:?}"),
        }
        assert_eq!(wrap.1, 0, "attempt counter starts at 0");
    }

    /// The watchdog's run loop must not crash on RPC errors. With a
    /// stub EthClient (no real provider) every poll fails on the Eth
    /// side, but the loop must keep ticking. Mirrors the runtime
    /// behavior we want: routine network blips don't take the
    /// watchdog down (and don't false-pause via violation counting —
    /// see `poll_once` returning `Err`, not `Ok(false)`).
    #[tokio::test]
    async fn run_loop_survives_rpc_errors() {
        let (signing_tx, _signing_rx) =
            mpsc::channel::<BridgeActionExecutionWrapper>(8);
        let _ = AtomicBool::new(false); // placeholder for the imports
        let soma_supply: SomaSupplyReader =
            Arc::new(|| Box::pin(async { Ok::<u128, _>(0u128) }));
        let config = WatchdogConfig {
            usdc_contract_address: "0x0".to_string(),
            eth_bridge_contract_address: "0x0".to_string(),
            poll_interval: Duration::from_millis(20),
            failure_threshold: 100,
            in_flight_tolerance_micro: 0,
        };
        // EthClient::new_for_test points at a non-listening port, so
        // every `get_erc20_balance` call fails — exactly the "RPC
        // blip" scenario.
        let eth = Arc::new(crate::eth_client::EthClient::new_for_test(
            "0x0".to_string(),
        ));
        let committee = Arc::new(RwLock::new(types::bridge::BridgeCommittee::empty()));
        let watchdog = BridgeWatchdog::new(
            config,
            eth,
            soma_supply,
            signing_tx,
            Arc::new(|| 0),
            0,
            committee,
        );
        let handle = watchdog.start();
        // Give the loop a handful of ticks; with poll_interval=20ms,
        // 200ms is ~10 ticks.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !handle.is_finished(),
            "watchdog must survive sustained RPC errors without exiting"
        );
        handle.abort();
    }
}
