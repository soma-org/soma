//! Bridge watchdog — multi-observable monitoring + conservation invariant.
//!
//! Soma's bridge watchdog mirrors Sui's `BridgeWatchDog` design: a
//! registry of independent `Observable`s, each spawned on its own
//! timer task. Observables report bridge state along independent
//! axes so on-call can triage *what kind* of divergence is happening
//! without rummaging through logs.
//!
//! ## Observables
//!
//! - **`EthVaultBalance`** — `usdc.balanceOf(vault)` on Eth. Catches
//!   vault drain.
//! - **`SomaUsdcSupply`** — `BridgeState.total_usdc_supply` on Soma.
//!   Catches counterfeit minting.
//! - **`EthBridgeStatus`** — `SomaBridge.paused()` on Eth. Flags
//!   unexpected pause/unpause from anyone's emergency action.
//! - **`SomaBridgeStatus`** — `BridgeState.paused` on Soma. Same for
//!   Soma side; alerts if the two halves disagree.
//! - **`ConservationInvariant`** — the aggregate check. Reads both
//!   readings, asserts `soma_supply ≤ eth_locked + tolerance`, and
//!   *auto-pauses* on sustained violation. The other observables are
//!   alert-only; only this one triggers an `EmergencyPause` action.
//!
//! ## Closures, not concrete clients
//!
//! The Soma observables take `Arc<dyn Fn() -> Future<...>>` closures
//! rather than a concrete `SomaBridgeClient<C>` reference. `SomaBridgeClient`
//! is generic over its inner trait, so threading it through observables
//! would either force a generic on each observable or commit them to
//! the production type. The closure decouples them from the client's
//! type parameter and lets tests stub readings cheaply.
//!
//! ## Auto-pause flow (conservation observable only)
//!
//! After `failure_threshold` consecutive violation polls:
//! 1. Construct `BridgeAction::EmergencyPause` with the current
//!    expected EmergencyOp seq num (read off-chain).
//! 2. Hand it to the action executor's signing queue. The executor's
//!    `PeerBroadcastAggregator` fans out via HTTP, collects a quorum
//!    cert (450 BPS = 4.5% is enough to pause), and submits on Soma.
//!
//! Sui *only* alerts; it does not auto-pause. Soma's low pause
//! threshold makes auto-pause useful — a single watchdog flagging a
//! real issue can usually get enough validators to sign without
//! majority cooperation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{error, info, warn};

use crate::action_executor::{BridgeActionExecutionWrapper, submit_to_executor};
use crate::error::BridgeResult;
use crate::eth_client::EthClient;
use crate::types::BridgeAction;

// ---------------------------------------------------------------------------
// Closure types for Soma-side reads
// ---------------------------------------------------------------------------

pub mod futures_pin {
    use std::future::Future;
    use std::pin::Pin;
    pub type PinnedFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
}

/// `BridgeState.total_usdc_supply` reader — closure indirection so
/// observables don't have to know about the concrete `SomaBridgeClient<C>`
/// generic.
pub type SomaSupplyReader =
    Arc<dyn Fn() -> futures_pin::PinnedFuture<BridgeResult<u64>> + Send + Sync>;

/// `BridgeState.paused` reader — same indirection rationale.
pub type SomaPausedReader =
    Arc<dyn Fn() -> futures_pin::PinnedFuture<BridgeResult<bool>> + Send + Sync>;

// ---------------------------------------------------------------------------
// Observable trait + registry
// ---------------------------------------------------------------------------

/// One axis of bridge health, polled on its own timer. Implementations
/// log + alert through their own observe step; they do NOT share state
/// (each owns the clients/handles it needs). Mirrors Sui's
/// `sui_bridge_watchdog::Observable` trait.
#[async_trait]
pub trait Observable: Send + Sync {
    /// Short label used in log lines so operators can correlate this
    /// observable's output across runs.
    fn name(&self) -> &str;

    /// Polling cadence. Each observable picks the rhythm appropriate
    /// to what it watches — a vault balance read is cheap; a Soma RPC
    /// hit is more expensive.
    fn interval(&self) -> Duration;

    /// Do one observation pass. Should NOT return Result — failures
    /// log warnings and let the next tick try again. A failed RPC is
    /// not a violation, it's a missing data point.
    async fn observe_and_report(&self);
}

/// Registry of observables; spawns one tokio task per observable on
/// its own timer. Construct with `new()` and chain `.with(...)` calls,
/// then call `start()` to spawn the tasks.
pub struct BridgeWatchdog {
    observables: Vec<Box<dyn Observable>>,
}

impl Default for BridgeWatchdog {
    fn default() -> Self {
        Self::new()
    }
}

impl BridgeWatchdog {
    pub fn new() -> Self {
        Self { observables: Vec::new() }
    }

    pub fn with(mut self, obs: Box<dyn Observable>) -> Self {
        self.observables.push(obs);
        self
    }

    /// Spawn one tokio task per observable. Returns the handles so the
    /// caller can keep them alive (and abort on shutdown). Mirrors
    /// Sui's `BridgeWatchDog::run`.
    pub fn start(self) -> Vec<tokio::task::JoinHandle<()>> {
        info!(
            count = self.observables.len(),
            "BridgeWatchdog: spawning observables"
        );
        self.observables
            .into_iter()
            .map(|obs| {
                tokio::spawn(async move {
                    let mut timer = interval(obs.interval());
                    // Skip missed ticks so a stalled observable doesn't burst
                    // out a backlog of catch-up polls when it recovers.
                    timer.set_missed_tick_behavior(
                        tokio::time::MissedTickBehavior::Skip,
                    );
                    let name = obs.name().to_string();
                    info!(observable = %name, "Observable task spawned");
                    loop {
                        timer.tick().await;
                        obs.observe_and_report().await;
                    }
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Observable: EthVaultBalance
// ---------------------------------------------------------------------------

/// Reads the USDC balance of the Eth-side `SomaBridge` vault every
/// `interval`. Logs every change; warns when the balance drops below
/// `zero_alert_threshold_micro` (catches drain events).
pub struct EthVaultBalanceObservable {
    pub eth_client: Arc<EthClient>,
    pub usdc_contract_address: String,
    pub eth_bridge_contract_address: String,
    pub interval: Duration,
    last_value: AtomicU64,
    /// Threshold below which a balance reading triggers a warn-log.
    /// Set to 1 USDC = 1_000_000 micro by default.
    pub zero_alert_threshold_micro: u128,
}

impl EthVaultBalanceObservable {
    pub fn new(
        eth_client: Arc<EthClient>,
        usdc_contract_address: String,
        eth_bridge_contract_address: String,
        interval: Duration,
    ) -> Self {
        Self {
            eth_client,
            usdc_contract_address,
            eth_bridge_contract_address,
            interval,
            last_value: AtomicU64::new(u64::MAX),
            zero_alert_threshold_micro: 1_000_000,
        }
    }
}

#[async_trait]
impl Observable for EthVaultBalanceObservable {
    fn name(&self) -> &str {
        "EthVaultBalance"
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    async fn observe_and_report(&self) {
        match self
            .eth_client
            .get_erc20_balance(
                &self.usdc_contract_address,
                &self.eth_bridge_contract_address,
            )
            .await
        {
            Ok(balance) => {
                let balance_u64 = balance.min(u64::MAX as u128) as u64;
                let prev = self.last_value.swap(balance_u64, Ordering::SeqCst);
                if prev == u64::MAX {
                    info!(balance, "EthVaultBalance: initial reading");
                } else if prev != balance_u64 {
                    info!(prev, current = balance, "EthVaultBalance: changed");
                }
                if balance < self.zero_alert_threshold_micro {
                    warn!(
                        balance,
                        threshold = self.zero_alert_threshold_micro,
                        "EthVaultBalance: below alert threshold"
                    );
                }
            }
            Err(e) => {
                warn!(error = %e, "EthVaultBalance: read failed");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Observable: SomaUsdcSupply
// ---------------------------------------------------------------------------

/// Reads `BridgeState.total_usdc_supply` from Soma every `interval`.
/// Logs each change.
pub struct SomaUsdcSupplyObservable {
    pub reader: SomaSupplyReader,
    pub interval: Duration,
    last_value: AtomicU64,
}

impl SomaUsdcSupplyObservable {
    pub fn new(reader: SomaSupplyReader, interval: Duration) -> Self {
        Self { reader, interval, last_value: AtomicU64::new(u64::MAX) }
    }
}

#[async_trait]
impl Observable for SomaUsdcSupplyObservable {
    fn name(&self) -> &str {
        "SomaUsdcSupply"
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    async fn observe_and_report(&self) {
        match (self.reader)().await {
            Ok(supply) => {
                let prev = self.last_value.swap(supply, Ordering::SeqCst);
                if prev == u64::MAX {
                    info!(supply, "SomaUsdcSupply: initial reading");
                } else if prev != supply {
                    info!(prev, current = supply, "SomaUsdcSupply: changed");
                }
            }
            Err(e) => warn!(error = %e, "SomaUsdcSupply: read failed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Observable: EthBridgeStatus
// ---------------------------------------------------------------------------

/// Reads `paused()` on the Eth `SomaBridge` contract. Logs only on
/// transitions — pause state should rarely change. A flip to paused
/// without a corresponding Soma-side pause indicates committee action
/// on Eth only (or attacker action) and should be investigated.
pub struct EthBridgeStatusObservable {
    pub eth_client: Arc<EthClient>,
    pub eth_bridge_contract_address: String,
    pub interval: Duration,
    /// `0` = unknown; `1` = paused; `2` = unpaused.
    last_state: AtomicU32,
}

impl EthBridgeStatusObservable {
    pub fn new(
        eth_client: Arc<EthClient>,
        eth_bridge_contract_address: String,
        interval: Duration,
    ) -> Self {
        Self {
            eth_client,
            eth_bridge_contract_address,
            interval,
            last_state: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl Observable for EthBridgeStatusObservable {
    fn name(&self) -> &str {
        "EthBridgeStatus"
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    async fn observe_and_report(&self) {
        match self
            .eth_client
            .get_bridge_paused(&self.eth_bridge_contract_address)
            .await
        {
            Ok(paused) => {
                let now = if paused { 1 } else { 2 };
                let prev = self.last_state.swap(now, Ordering::SeqCst);
                match (prev, paused) {
                    (0, true) => {
                        warn!("EthBridgeStatus: bridge is PAUSED (initial reading)");
                    }
                    (0, false) => info!("EthBridgeStatus: bridge is unpaused"),
                    (1, false) => info!("EthBridgeStatus: bridge transitioned PAUSED → unpaused"),
                    (2, true) => {
                        warn!("EthBridgeStatus: bridge transitioned unpaused → PAUSED");
                    }
                    _ => {} // no transition
                }
            }
            Err(e) => warn!(error = %e, "EthBridgeStatus: read failed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Observable: SomaBridgeStatus
// ---------------------------------------------------------------------------

/// Reads `BridgeState.paused` on Soma. Together with `EthBridgeStatus`,
/// the two let on-call notice when the two halves of the bridge are
/// out of sync (one paused, one not).
pub struct SomaBridgeStatusObservable {
    pub reader: SomaPausedReader,
    pub interval: Duration,
    last_state: AtomicU32,
}

impl SomaBridgeStatusObservable {
    pub fn new(reader: SomaPausedReader, interval: Duration) -> Self {
        Self { reader, interval, last_state: AtomicU32::new(0) }
    }
}

#[async_trait]
impl Observable for SomaBridgeStatusObservable {
    fn name(&self) -> &str {
        "SomaBridgeStatus"
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    async fn observe_and_report(&self) {
        match (self.reader)().await {
            Ok(paused) => {
                let now = if paused { 1 } else { 2 };
                let prev = self.last_state.swap(now, Ordering::SeqCst);
                match (prev, paused) {
                    (0, true) => warn!("SomaBridgeStatus: bridge is PAUSED (initial reading)"),
                    (0, false) => info!("SomaBridgeStatus: bridge is unpaused"),
                    (1, false) => info!("SomaBridgeStatus: bridge transitioned PAUSED → unpaused"),
                    (2, true) => warn!("SomaBridgeStatus: bridge transitioned unpaused → PAUSED"),
                    _ => {}
                }
            }
            Err(e) => warn!(error = %e, "SomaBridgeStatus: read failed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Observable: ConservationInvariant (the auto-pause owner)
// ---------------------------------------------------------------------------

/// The aggregate conservation check: reads both eth_locked and
/// soma_supply each poll, asserts `soma_supply ≤ eth_locked + tolerance`,
/// counts consecutive violations, and fires an `EmergencyPause` action
/// after `failure_threshold` violations.
///
/// This is the ONLY observable that takes corrective action. The
/// status observables alert only — they let humans triage; the
/// conservation observable presses the big red button.
pub struct ConservationInvariantObservable {
    pub eth_client: Arc<EthClient>,
    pub soma_supply: SomaSupplyReader,
    pub usdc_contract_address: String,
    pub eth_bridge_contract_address: String,
    pub interval: Duration,
    pub failure_threshold: u32,
    /// Tolerated USDC delta in microdollars. Covers transient
    /// in-flight transfer volume. Zero requires exact equality.
    pub in_flight_tolerance_micro: u128,
    /// Signing queue handle — auto-pause posts an EmergencyPause
    /// action here when the violation threshold trips.
    pub signing_tx: mpsc::Sender<BridgeActionExecutionWrapper>,
    /// Closure returning the next expected EmergencyOp seq num.
    /// Re-read on each violation so we don't burn a stale nonce.
    pub expected_pause_nonce: Arc<dyn Fn() -> u64 + Send + Sync>,
    consecutive_violations: AtomicU32,
    auto_pause_emitted: AtomicBool,
}

impl ConservationInvariantObservable {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eth_client: Arc<EthClient>,
        soma_supply: SomaSupplyReader,
        usdc_contract_address: String,
        eth_bridge_contract_address: String,
        interval: Duration,
        failure_threshold: u32,
        in_flight_tolerance_micro: u128,
        signing_tx: mpsc::Sender<BridgeActionExecutionWrapper>,
        expected_pause_nonce: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Self {
        Self {
            eth_client,
            soma_supply,
            usdc_contract_address,
            eth_bridge_contract_address,
            interval,
            failure_threshold,
            in_flight_tolerance_micro,
            signing_tx,
            expected_pause_nonce,
            consecutive_violations: AtomicU32::new(0),
            auto_pause_emitted: AtomicBool::new(false),
        }
    }

    /// One conservation check pass — paired vault + supply read, then
    /// compare. Returns the outcome so tests can verify directly
    /// without spawning the timer task.
    async fn check_once(&self) -> BridgeResult<bool> {
        let eth_locked = self
            .eth_client
            .get_erc20_balance(
                &self.usdc_contract_address,
                &self.eth_bridge_contract_address,
            )
            .await?;
        let soma_supply = (self.soma_supply)().await? as u128;
        info!(
            eth_locked,
            soma_supply, "ConservationInvariant: paired reading"
        );
        let limit = eth_locked.saturating_add(self.in_flight_tolerance_micro);
        Ok(soma_supply <= limit)
    }

    /// Hand an `EmergencyPause` action to the executor's signing queue.
    /// The executor fans out via the peer-broadcast aggregator and
    /// submits the resulting cert on Soma.
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

#[async_trait]
impl Observable for ConservationInvariantObservable {
    fn name(&self) -> &str {
        "ConservationInvariant"
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    async fn observe_and_report(&self) {
        match self.check_once().await {
            Ok(true) => {
                let prev = self.consecutive_violations.swap(0, Ordering::SeqCst);
                if prev > 0 {
                    info!(
                        was = prev,
                        "ConservationInvariant: restored to healthy"
                    );
                }
                self.auto_pause_emitted.store(false, Ordering::SeqCst);
            }
            Ok(false) => {
                let count = self
                    .consecutive_violations
                    .fetch_add(1, Ordering::SeqCst)
                    .saturating_add(1);
                warn!(
                    consecutive_violations = count,
                    threshold = self.failure_threshold,
                    "ConservationInvariant: violated"
                );
                if count >= self.failure_threshold
                    && !self.auto_pause_emitted.load(Ordering::SeqCst)
                {
                    match self.emit_auto_pause().await {
                        Ok(()) => {
                            self.auto_pause_emitted.store(true, Ordering::SeqCst);
                            error!(
                                consecutive_violations = count,
                                "AUTO-PAUSE: bridge conservation violated — emergency pause sig posted"
                            );
                        }
                        Err(e) => {
                            error!(error = %e, "Failed to emit auto-pause sig");
                        }
                    }
                }
            }
            Err(e) => {
                // RPC failures don't count as violations — we'd
                // false-pause on routine network blips.
                warn!(error = %e, "ConservationInvariant: poll error (not counted)");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn supply_reader(value: u64) -> SomaSupplyReader {
        Arc::new(move || Box::pin(async move { Ok(value) }))
    }

    /// `emit_auto_pause` queues an `EmergencyPause` with the current
    /// expected nonce.
    #[tokio::test]
    async fn conservation_emit_auto_pause_queues_emergency_pause() {
        let (signing_tx, mut signing_rx) =
            mpsc::channel::<BridgeActionExecutionWrapper>(8);
        let nonce_fn: Arc<dyn Fn() -> u64 + Send + Sync> = Arc::new(|| 42);
        let eth = Arc::new(EthClient::new_for_test("0x0".to_string()));

        let obs = ConservationInvariantObservable::new(
            eth,
            supply_reader(0),
            "0x0".to_string(),
            "0x0".to_string(),
            Duration::from_millis(20),
            1,
            0,
            signing_tx,
            nonce_fn,
        );
        obs.emit_auto_pause().await.expect("emit");
        let wrap = signing_rx.recv().await.expect("queued");
        match wrap.0 {
            BridgeAction::EmergencyPause { nonce } => assert_eq!(nonce, 42),
            other => panic!("expected EmergencyPause, got {other:?}"),
        }
        assert_eq!(wrap.1, 0, "attempt counter starts at 0");
    }

    /// The watchdog's spawned tasks must survive sustained RPC errors
    /// (each poll calls `EthClient` against a non-listening port).
    /// Errors are NOT counted as violations, so auto-pause never
    /// fires — the loop just keeps trying.
    #[tokio::test]
    async fn conservation_survives_rpc_errors() {
        let (signing_tx, _signing_rx) =
            mpsc::channel::<BridgeActionExecutionWrapper>(8);
        let nonce_fn: Arc<dyn Fn() -> u64 + Send + Sync> = Arc::new(|| 0);
        let eth = Arc::new(EthClient::new_for_test("0x0".to_string()));

        let watchdog = BridgeWatchdog::new().with(Box::new(
            ConservationInvariantObservable::new(
                eth,
                supply_reader(0),
                "0x0".to_string(),
                "0x0".to_string(),
                Duration::from_millis(20),
                100,
                0,
                signing_tx,
                nonce_fn,
            ),
        ));
        let handles = watchdog.start();
        tokio::time::sleep(Duration::from_millis(200)).await;
        for h in &handles {
            assert!(!h.is_finished(), "observable task exited unexpectedly");
        }
        for h in handles {
            h.abort();
        }
    }

    /// `check_once` returns Err when the Eth RPC is unreachable; the
    /// observe loop treats it as a missing data point (not a violation).
    #[tokio::test]
    async fn conservation_err_not_counted_as_violation() {
        let (signing_tx, _signing_rx) =
            mpsc::channel::<BridgeActionExecutionWrapper>(8);
        let nonce_fn: Arc<dyn Fn() -> u64 + Send + Sync> = Arc::new(|| 0);
        let eth = Arc::new(EthClient::new_for_test("0x0".to_string()));

        let obs = ConservationInvariantObservable::new(
            eth,
            supply_reader(0),
            "0x0".to_string(),
            "0x0".to_string(),
            Duration::from_millis(20),
            1,
            0,
            signing_tx,
            nonce_fn,
        );
        assert!(obs.check_once().await.is_err());
        for _ in 0..3 {
            obs.observe_and_report().await;
        }
        assert_eq!(obs.consecutive_violations.load(Ordering::SeqCst), 0);
    }

    /// Build a `BridgeWatchdog` with all 5 observables and confirm
    /// `start()` returns 5 join handles. Smoke-test for the registry
    /// shape — observables can be mixed and spawned without panic.
    #[tokio::test]
    async fn registry_spawns_one_task_per_observable() {
        let (signing_tx, _rx) =
            mpsc::channel::<BridgeActionExecutionWrapper>(8);
        let nonce_fn: Arc<dyn Fn() -> u64 + Send + Sync> = Arc::new(|| 0);
        let eth = Arc::new(EthClient::new_for_test("0x0".to_string()));
        let paused_reader: SomaPausedReader =
            Arc::new(|| Box::pin(async { Ok(false) }));

        let watchdog = BridgeWatchdog::new()
            .with(Box::new(EthVaultBalanceObservable::new(
                Arc::clone(&eth),
                "0x0".into(),
                "0x0".into(),
                Duration::from_millis(50),
            )))
            .with(Box::new(SomaUsdcSupplyObservable::new(
                supply_reader(0),
                Duration::from_millis(50),
            )))
            .with(Box::new(EthBridgeStatusObservable::new(
                Arc::clone(&eth),
                "0x0".into(),
                Duration::from_millis(50),
            )))
            .with(Box::new(SomaBridgeStatusObservable::new(
                paused_reader,
                Duration::from_millis(50),
            )))
            .with(Box::new(ConservationInvariantObservable::new(
                eth,
                supply_reader(0),
                "0x0".into(),
                "0x0".into(),
                Duration::from_millis(50),
                100,
                0,
                signing_tx,
                nonce_fn,
            )));
        let handles = watchdog.start();
        assert_eq!(handles.len(), 5);
        tokio::time::sleep(Duration::from_millis(120)).await;
        for h in &handles {
            assert!(!h.is_finished(), "an observable task exited");
        }
        for h in handles {
            h.abort();
        }
    }
}
