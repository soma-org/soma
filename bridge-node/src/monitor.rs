//! Bridge state monitor.
//!
//! Mirrors Sui's `sui-bridge/src/monitor.rs`. The monitor watches the
//! Soma chain for bridge state changes and publishes them to in-memory
//! channels that other subsystems subscribe to:
//!
//! ```text
//!     BridgeMonitor
//!       ├─ poll loop (every poll_interval)
//!       │    ├─ get_bridge_summary → bridge.paused
//!       │    └─ get_bridge_committee
//!       │
//!       ├─ outputs:
//!       │    ├─ bridge_paused: tokio::sync::watch::Sender<bool>
//!       │    │     • action executor borrow()s this in lieu of polling
//!       │    │       the RPC on every retry round
//!       │    │
//!       │    └─ committee: Arc<ArcSwap<BridgeCommittee>>
//!       │          • LocalAggregator + BridgeServer load() this on each
//!       │            sig request to get the current member set
//!       │
//!       └─ logs structured events on every change
//! ```
//!
//! ## Why polling instead of event subscription
//!
//! Sui's monitor consumes from a `SuiBridgeEvent` stream and reacts to
//! `EmergencyOpEvent`, `BlocklistValidatorEvent`,
//! `CommitteeMemberUrlUpdateEvent`. Soma doesn't emit explicit on-chain
//! events for these state changes today — the on-chain executor mutates
//! `bridge_state.paused` and `bridge_state.bridge_committee` silently.
//! Adding event emission is on the roadmap but for now polling the
//! consolidated `bridge_state` is the path of least scope creep, with
//! an acceptable cost: the monitor's poll cadence (5s default) bounds
//! how stale reads can be.
//!
//! Once Soma adds explicit bridge events the polling fallback can be
//! demoted to a startup health check, and the steady-state path will
//! be subscription-based — same shape Sui uses today.

use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use types::bridge::BridgeCommittee;

use crate::soma_client::{SomaBridgeClient, SomaBridgeClientInner};

/// Default cadence at which the monitor polls bridge state. Bounded
/// upper limit on observation staleness when the chain doesn't emit
/// explicit events. Smaller = lower-staleness, more RPC load.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Outputs published by the monitor. Constructed once at startup;
/// the monitor task owns the senders and downstream subsystems hold
/// the receivers / [`Arc`] handles.
pub struct BridgeMonitorChannels {
    /// Current bridge pause state. `true` = paused; the executor
    /// halts new submissions while paused.
    pub bridge_paused_rx: watch::Receiver<bool>,
    /// Live snapshot of the on-chain committee. `ArcSwap` lets readers
    /// `load()` without taking a lock — important for the sig-aggregator
    /// which reads on every action.
    pub committee: Arc<ArcSwap<BridgeCommittee>>,
}

/// The monitor task. Build with [`Self::new`], then call [`Self::run`]
/// to spawn the poll loop.
pub struct BridgeMonitor<C: SomaBridgeClientInner> {
    client: Arc<SomaBridgeClient<C>>,
    paused_tx: watch::Sender<bool>,
    committee: Arc<ArcSwap<BridgeCommittee>>,
    poll_interval: Duration,
}

impl<C: SomaBridgeClientInner + 'static> BridgeMonitor<C> {
    /// Construct a monitor + return the channels its readers will
    /// subscribe to. The caller seeds the initial pause state and
    /// committee from a fresh RPC fetch (or from the genesis snapshot
    /// during startup).
    pub fn new(
        client: Arc<SomaBridgeClient<C>>,
        initial_paused: bool,
        initial_committee: BridgeCommittee,
        poll_interval: Duration,
    ) -> (Self, BridgeMonitorChannels) {
        let (paused_tx, paused_rx) = watch::channel(initial_paused);
        let committee = Arc::new(ArcSwap::from_pointee(initial_committee));
        let channels = BridgeMonitorChannels {
            bridge_paused_rx: paused_rx,
            committee: Arc::clone(&committee),
        };
        let monitor = Self { client, paused_tx, committee, poll_interval };
        (monitor, channels)
    }

    /// Spawn the poll loop. The returned `JoinHandle` runs forever;
    /// dropping it without aborting is a structured shutdown of the
    /// task. Caller is responsible for keeping at least one
    /// `BridgeMonitorChannels` clone alive (otherwise downstream
    /// readers will see a closed channel).
    pub fn run(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            info!(poll_ms = self.poll_interval.as_millis() as u64, "BridgeMonitor started",);
            let mut timer = tokio::time::interval(self.poll_interval);
            // First tick fires immediately; that's the desired warm-up
            // (we want the first observation to be fresh, not
            // wait-one-interval-then-fresh).
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                timer.tick().await;
                self.refresh_paused().await;
                self.refresh_committee().await;
            }
        })
    }

    /// One round of pause-state polling. Logs + publishes only on
    /// change; steady-state observations are silent.
    async fn refresh_paused(&self) {
        match self.client.is_bridge_paused().await {
            Ok(now_paused) => {
                let prev = *self.paused_tx.borrow();
                if prev != now_paused {
                    info!(paused = now_paused, "bridge pause state changed");
                    // `send` returns Err only when all receivers have
                    // dropped — that means the bridge node is
                    // shutting down, so we stay silent rather than
                    // panic.
                    let _ = self.paused_tx.send(now_paused);
                } else {
                    debug!(paused = now_paused, "bridge pause state unchanged");
                }
            }
            Err(e) => {
                warn!(error = %e, "is_bridge_paused failed; will retry next tick");
            }
        }
    }

    /// One round of committee polling. Stores the new committee under
    /// `ArcSwap` and logs the diff (`now.members.len()` vs `prev.len()`).
    /// Storing on every tick is fine — `ArcSwap::store` is cheap and
    /// equality-checking the full `BridgeCommittee` is more expensive
    /// than swapping it.
    async fn refresh_committee(&self) {
        match self.client.get_bridge_committee().await {
            Ok(new_committee) => {
                let prev = self.committee.load();
                let prev_len = prev.members.len();
                let new_len = new_committee.members.len();
                let prev_blocklisted = prev.members.values().filter(|m| m.is_blocklisted).count();
                let new_blocklisted =
                    new_committee.members.values().filter(|m| m.is_blocklisted).count();
                if prev.members != new_committee.members {
                    info!(
                        prev_len,
                        new_len, prev_blocklisted, new_blocklisted, "bridge committee changed",
                    );
                    self.committee.store(Arc::new(new_committee));
                } else {
                    debug!(
                        len = new_len,
                        blocklisted = new_blocklisted,
                        "bridge committee unchanged",
                    );
                }
            }
            Err(e) => {
                warn!(error = %e, "get_bridge_committee failed; will retry next tick");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soma_client::tests::MockSomaClient;
    use fastcrypto::secp256k1::Secp256k1KeyPair;
    use fastcrypto::traits::KeyPair;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::sync::atomic::Ordering;
    use types::base::SomaAddress;
    use types::bridge::{BridgeChainId, BridgeCommittee, BridgeMember, BridgePubkey};

    fn fake_committee_with_n(n: usize) -> BridgeCommittee {
        let mut rng = StdRng::from_seed([42; 32]);
        let mut c = BridgeCommittee::empty();
        for _ in 0..n {
            let kp = Secp256k1KeyPair::generate(&mut rng);
            let pk = BridgePubkey::from_keypair(&kp);
            c.members.insert(
                pk,
                BridgeMember {
                    soma_address: SomaAddress::random(),
                    voting_power: 10_000 / n as u64,
                    http_url: String::new(),
                    is_blocklisted: false,
                },
            );
        }
        c
    }

    /// Pause state must propagate through the watch channel within a
    /// poll cycle. Verifies the basic feedback loop.
    #[tokio::test]
    async fn test_pause_state_propagates() {
        let mock = MockSomaClient::new();
        let client = Arc::new(SomaBridgeClient::new(mock, BridgeChainId::SomaCustom));
        let poll = Duration::from_millis(20);
        let (monitor, mut channels) =
            BridgeMonitor::new(client.clone(), false, BridgeCommittee::empty(), poll);
        let _h = monitor.run();

        // Flip pause on the underlying mock.
        client.inner_for_test().paused.store(true, Ordering::SeqCst);

        let started = std::time::Instant::now();
        loop {
            if *channels.bridge_paused_rx.borrow_and_update() {
                break;
            }
            if started.elapsed() > Duration::from_secs(2) {
                panic!("paused state never propagated");
            }
            channels.bridge_paused_rx.changed().await.expect("watch channel closed");
        }
    }

    /// Committee changes must be reflected in the `ArcSwap` snapshot.
    #[tokio::test]
    async fn test_committee_change_propagates() {
        let mock = MockSomaClient::new();
        let initial = fake_committee_with_n(3);
        *mock.committee.lock().unwrap() = initial.clone();
        let client = Arc::new(SomaBridgeClient::new(mock, BridgeChainId::SomaCustom));

        let poll = Duration::from_millis(20);
        let (monitor, channels) = BridgeMonitor::new(client.clone(), false, initial.clone(), poll);
        let _h = monitor.run();

        // After construction the ArcSwap holds the seed committee.
        assert_eq!(channels.committee.load().members.len(), 3);

        // Mutate the underlying mock to a new committee.
        let new_committee = fake_committee_with_n(5);
        *client.inner_for_test().committee.lock().unwrap() = new_committee.clone();

        let started = std::time::Instant::now();
        loop {
            let snap = channels.committee.load();
            if snap.members.len() == 5 {
                break;
            }
            if started.elapsed() > Duration::from_secs(2) {
                panic!("committee change never propagated; snapshot len = {}", snap.members.len());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Idempotent observation: when nothing changes upstream, the
    /// monitor doesn't churn the watch channel. Callers using
    /// `changed()` only wake up on real changes.
    #[tokio::test]
    async fn test_no_change_no_wake() {
        let mock = MockSomaClient::new();
        let client = Arc::new(SomaBridgeClient::new(mock, BridgeChainId::SomaCustom));
        let poll = Duration::from_millis(20);
        let (monitor, mut channels) =
            BridgeMonitor::new(client, false, BridgeCommittee::empty(), poll);
        let _h = monitor.run();

        // First observation matches initial value (no change → no
        // wake). `changed()` should time out.
        let timed_out =
            tokio::time::timeout(Duration::from_millis(150), channels.bridge_paused_rx.changed())
                .await
                .is_err();
        assert!(timed_out, "watch channel woke despite no upstream change");
    }
}
