//! Eth-side outbound relayer.
//!
//! Soma→Eth USDC withdrawal flow:
//!
//! ```text
//!  user calls bridge.withdraw on Soma
//!         │
//!         ▼
//!  Soma creates PendingWithdrawal{ nonce, … verified_signatures: None }
//!         │
//!         ▼
//!  soma_syncer notices it → executor signs + collects cert via peer broadcast
//!         │
//!         ▼
//!  executor submits BridgeAttachWithdrawalSignatures tx
//!         │
//!         ▼
//!  Soma now has PendingWithdrawal{ … verified_signatures: Some(cert) }
//!         │
//!         ▼
//!  *** this module ***
//!         │
//!         ▼
//!  poll for completed PendingWithdrawal → build Eth tx with cert sigs
//!  → sign with the relayer's Eth wallet → eth_sendRawTransaction
//!         │
//!         ▼
//!  SomaBridge.transferAttestedBridgedTokens(...) on Eth releases USDC
//! ```
//!
//! ## Sui parity
//!
//! Sui's bridge node intentionally does *not* drive Eth-side
//! submission. Their model: anyone with the on-chain cert can submit
//! it; Mysten Labs runs a separate relayer process for this in
//! production. Soma's design is to bundle it into bridge-node for
//! operational simplicity — one binary, one config, one set of WAL
//! recovery semantics. The on-chain contract still allows anyone to
//! submit, so a third party with the cert can land it independently.
//!
//! ## Status: skeleton
//!
//! The polling loop, idempotency-by-nonce tracking, and submission
//! plumbing are in place. The actual EVM tx construction
//! (`build_release_calldata`) is parameterized over an [`EthTxBuilder`]
//! trait; the production implementation needs to be filled in once
//! the Soma Eth-side contract's ABI is final. See the trait docs
//! below for what the impl must produce.

use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;
use tracing::{debug, info, instrument, warn};
use types::bridge::WithdrawalCertificate;

use crate::error::{BridgeError, BridgeResult};
use crate::soma_client::{SomaBridgeClient, SomaBridgeClientInner};

/// Default cadence for the relayer's polling loop. Polls Soma for
/// completed `PendingWithdrawal` objects; the actual submission cost
/// is the Eth-side `eth_sendRawTransaction`. 10s gives a low-latency
/// withdrawal experience without thrashing the Soma RPC.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(10);

/// How the relayer turns an on-chain cert into the bytes that go on
/// the wire to Ethereum. Implementations encode the Soma Eth-side
/// bridge contract's calldata, build the Eth tx envelope, sign it
/// with the operator's Eth wallet, and return the raw tx ready for
/// `eth_sendRawTransaction`.
///
/// Factored as a trait so:
/// - the polling loop can be tested without a real Eth wallet or ABI
///   (the stub returns canned bytes);
/// - the Soma Eth-side contracts at [`../bridge/evm/`](../bridge/evm/)
///   supply their own implementation once the integration layer is
///   wired up (alloy / ethers / hand-rolled ABI encoding);
/// - operators running different bridge contract versions can swap
///   without recompiling the polling loop.
#[async_trait::async_trait]
pub trait EthTxBuilder: Send + Sync {
    /// Return `Some(raw_tx_bytes)` if the relayer should submit this
    /// withdrawal; `None` if the relayer should skip (e.g. the cert
    /// is malformed, the contract is paused, etc.). On `Err` the
    /// relayer logs + retries on the next poll.
    async fn build_release_tx(
        &self,
        withdrawal: &OutboundWithdrawal,
    ) -> BridgeResult<Option<Vec<u8>>>;
}

/// What the polling loop hands to the [`EthTxBuilder`]. Carries the
/// canonical action bytes (matches Sui's `BridgeMessage` wire format)
/// so the builder doesn't have to re-derive them from the
/// `PendingWithdrawal` struct.
#[derive(Debug, Clone)]
pub struct OutboundWithdrawal {
    pub nonce: u64,
    pub recipient_eth_address: [u8; 20],
    pub amount: u64,
    pub created_at_ms: u64,
    /// The canonical signed-message bytes (what each committee
    /// signature ecrecovers against). The builder hashes these for
    /// signature verification on the EVM side.
    pub message_bytes: Vec<u8>,
    /// Labeled-pubkey signature envelope as produced by the
    /// PeerBroadcastAggregator. Order is BTreeMap-canonical — the
    /// Eth contract's sig-counting logic must be order-insensitive
    /// (Sui's `CommitteeUpgradeable` is; the Soma Eth contract
    /// will mirror that).
    pub certificate: WithdrawalCertificate,
}

/// How the relayer remembers which withdrawals it already pushed to
/// Eth. Without this, every poll would re-submit every completed
/// withdrawal — wasteful and gas-burning. The cheap version is an
/// in-memory `BTreeSet<u64>` of submitted nonces; production should
/// persist this set in the WAL so a restart doesn't double-submit.
///
/// Today the implementation lives in [`InMemoryRelayedTracker`];
/// follow-on work pushes it into [`crate::storage`] for restart-safe
/// dedup.
pub trait RelayedTracker: Send + Sync {
    /// `true` if this nonce has already been submitted to Eth.
    fn is_relayed(&self, nonce: u64) -> bool;
    /// Mark a nonce as submitted. Idempotent.
    fn mark_relayed(&self, nonce: u64);
}

pub struct InMemoryRelayedTracker {
    seen: std::sync::Mutex<std::collections::BTreeSet<u64>>,
}

impl InMemoryRelayedTracker {
    pub fn new() -> Self {
        Self { seen: std::sync::Mutex::new(std::collections::BTreeSet::new()) }
    }
}

impl Default for InMemoryRelayedTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl RelayedTracker for InMemoryRelayedTracker {
    fn is_relayed(&self, nonce: u64) -> bool {
        self.seen.lock().unwrap().contains(&nonce)
    }
    fn mark_relayed(&self, nonce: u64) {
        self.seen.lock().unwrap().insert(nonce);
    }
}

/// The outbound relayer. Polls Soma for completed PendingWithdrawals,
/// hands each to the [`EthTxBuilder`], and (on next iteration) submits
/// the resulting raw tx to Ethereum.
///
/// Tx submission via `eth_sendRawTransaction` is **not yet wired
/// here**: that requires plumbing a writeable EthClient method (the
/// current `rpc_call<T>` is read-shaped). It's the natural next
/// chunk to land. For now the relayer logs what it would have
/// submitted — operators can sanity-check the polling + ABI logic
/// without burning gas while the contract ABI finalizes.
pub struct OutboundRelayer<C: SomaBridgeClientInner> {
    soma_client: Arc<SomaBridgeClient<C>>,
    tx_builder: Arc<dyn EthTxBuilder>,
    tracker: Arc<dyn RelayedTracker>,
    poll_interval: Duration,
    /// Upper bound on withdrawal nonces to scan per poll. Without a
    /// cap, a node started after a long offline period would scan
    /// from 0 to current nonce on every tick. The relayer reads the
    /// chain's `next_withdrawal_nonce` once and walks backward up to
    /// this many nonces; deeper history requires WAL replay.
    scan_window: u64,
}

impl<C: SomaBridgeClientInner + 'static> OutboundRelayer<C> {
    pub fn new(
        soma_client: Arc<SomaBridgeClient<C>>,
        tx_builder: Arc<dyn EthTxBuilder>,
        tracker: Arc<dyn RelayedTracker>,
        poll_interval: Duration,
        scan_window: u64,
    ) -> Self {
        Self {
            soma_client,
            tx_builder,
            tracker,
            poll_interval,
            scan_window,
        }
    }

    pub fn start(self) -> JoinHandle<()> {
        tokio::spawn(self.run())
    }

    async fn run(self) {
        info!(
            poll_ms = self.poll_interval.as_millis() as u64,
            scan_window = self.scan_window,
            "OutboundRelayer started",
        );
        let mut timer = tokio::time::interval(self.poll_interval);
        timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            timer.tick().await;
            if let Err(e) = self.scan_once().await {
                warn!(error = %e, "outbound relayer scan failed; will retry");
            }
        }
    }

    /// One pass over the recent withdrawal range. Walks backward from
    /// the current high-water nonce and submits any cert-attached
    /// withdrawal not in the tracker.
    #[instrument(level = "info", skip_all)]
    async fn scan_once(&self) -> BridgeResult<()> {
        // We don't have a "next_withdrawal_nonce" reader yet; for the
        // skeleton, walk a small fixed range. Production replaces the
        // upper bound with a chain-state read (BridgeState
        // .next_withdrawal_nonce).
        let upper = self.scan_window;
        for nonce in 0..upper {
            if self.tracker.is_relayed(nonce) {
                continue;
            }
            let Some(pw) = self.soma_client.get_pending_withdrawal(nonce).await? else {
                continue;
            };
            let Some(cert) = pw.verified_signatures.clone() else {
                debug!(nonce, "withdrawal not yet cert-attached; skipping");
                continue;
            };

            // Compose the action bytes the cert was signed over.
            let action = crate::types::BridgeAction::Withdrawal {
                nonce: pw.nonce,
                sender: pw.sender,
                target_chain: types::bridge::BridgeChainId::EthCustom,
                recipient_eth_address: pw.recipient_eth_address,
                token_type: types::bridge::USDC_TOKEN_TYPE,
                amount: pw.amount,
                timestamp_ms: pw.created_at_ms,
            };
            let message_bytes = action.to_message_bytes();
            let outbound = OutboundWithdrawal {
                nonce: pw.nonce,
                recipient_eth_address: pw.recipient_eth_address,
                amount: pw.amount,
                created_at_ms: pw.created_at_ms,
                message_bytes,
                certificate: cert,
            };
            match self.tx_builder.build_release_tx(&outbound).await {
                Ok(Some(raw_tx)) => {
                    // TODO(soma#bridge-eth-submit): submit
                    // `raw_tx` via `eth_sendRawTransaction` on the
                    // EthClient. Today we log + mark relayed so the
                    // polling loop is exercised end-to-end against
                    // mock contracts; the actual EVM submission
                    // lands once the Soma Eth-side bridge contract
                    // repo finalizes its ABI (mirrors how
                    // the in-tree `bridge/evm/` Foundry project — the
                    // Rust side can land tests before the contract ABI
                    // integration is wired).
                    info!(
                        nonce = outbound.nonce,
                        raw_tx_len = raw_tx.len(),
                        sig_count = outbound.certificate.signatures.len(),
                        "would submit Eth-side release tx (pending Eth submission wiring)"
                    );
                    self.tracker.mark_relayed(outbound.nonce);
                }
                Ok(None) => {
                    debug!(nonce = outbound.nonce, "builder returned None; skipping");
                }
                Err(e) => {
                    warn!(
                        nonce = outbound.nonce,
                        error = %e,
                        "build_release_tx failed; will retry next scan"
                    );
                }
            }
        }
        Ok(())
    }
}

/// Placeholder `EthTxBuilder` for the skeleton wiring. Returns a
/// deterministic stub byte vector so tests can assert that the
/// polling loop calls into a builder for each completed withdrawal.
/// Production swaps this for an implementation that ABI-encodes
/// `transferAttestedBridgedTokens` (or whatever the Soma Eth contract
/// names its release function) and signs the tx with the operator's
/// wallet.
pub struct StubEthTxBuilder;

#[async_trait::async_trait]
impl EthTxBuilder for StubEthTxBuilder {
    async fn build_release_tx(
        &self,
        withdrawal: &OutboundWithdrawal,
    ) -> BridgeResult<Option<Vec<u8>>> {
        if withdrawal.certificate.signatures.is_empty() {
            return Err(BridgeError::Internal(
                "outbound relayer received cert with no signatures".to_string(),
            ));
        }
        // Stub: identifiable bytes so log inspection during dev is
        // useful. Format: `b"STUB" || nonce_be(8)`.
        let mut bytes = b"STUB".to_vec();
        bytes.extend_from_slice(&withdrawal.nonce.to_be_bytes());
        Ok(Some(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soma_client::SomaBridgeClient;
    use crate::soma_client::tests::MockSomaClient;
    use types::base::SomaAddress;
    use types::bridge::{BridgeChainId, PendingWithdrawal, WithdrawalCertificate};
    use types::object::ObjectID;

    fn pw(nonce: u64, cert_attached: bool) -> PendingWithdrawal {
        let cert = if cert_attached {
            Some(WithdrawalCertificate {
                signatures: {
                    let mut m = std::collections::BTreeMap::new();
                    m.insert(
                        types::bridge::BridgePubkey::from_bytes(&[
                            // Compressed secp256k1 pubkey for the
                            // generator point — a valid curve point so
                            // BridgePubkey::from_bytes doesn't error.
                            0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55,
                            0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07, 0x02, 0x9b, 0xfc,
                            0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16,
                            0xf8, 0x17, 0x98,
                        ])
                        .expect("generator point is on the curve"),
                        types::bridge::BridgeSignature::from_bytes(&[0xAA; 65]).unwrap(),
                    );
                    m
                },
                attached_at_epoch: 0,
            })
        } else {
            None
        };
        PendingWithdrawal {
            id: ObjectID::new([0u8; 32]),
            nonce,
            sender: SomaAddress::from([0u8; 32]),
            recipient_eth_address: [0x42; 20],
            amount: 1_000_000,
            created_at_ms: 0,
            verified_signatures: cert,
        }
    }

    fn build_client_with_withdrawals(
        pws: Vec<PendingWithdrawal>,
    ) -> Arc<SomaBridgeClient<MockSomaClient>> {
        let mock = MockSomaClient::new();
        for w in &pws {
            mock.pending_withdrawals
                .lock()
                .unwrap()
                .insert(w.nonce, w.clone());
        }
        Arc::new(SomaBridgeClient::new(mock, BridgeChainId::SomaCustom))
    }

    /// The relayer must only act on withdrawals that have a cert
    /// attached. A bare PendingWithdrawal (still being signed) is
    /// skipped.
    #[tokio::test]
    async fn scan_skips_unsigned_withdrawals() {
        let client = build_client_with_withdrawals(vec![pw(0, false), pw(1, true)]);
        let tracker = Arc::new(InMemoryRelayedTracker::new());
        let builder = Arc::new(StubEthTxBuilder);
        let relayer = OutboundRelayer::new(
            client,
            builder,
            tracker.clone(),
            Duration::from_millis(10),
            5,
        );
        relayer.scan_once().await.unwrap();
        assert!(!tracker.is_relayed(0), "unsigned must not be marked relayed");
        assert!(tracker.is_relayed(1), "cert-attached must be marked relayed");
    }

    /// Idempotency: a second scan after the first must not re-submit
    /// an already-relayed nonce.
    #[tokio::test]
    async fn scan_is_idempotent_across_runs() {
        let client = build_client_with_withdrawals(vec![pw(2, true)]);
        let tracker = Arc::new(InMemoryRelayedTracker::new());

        // Count builder invocations.
        struct CountingBuilder {
            calls: std::sync::atomic::AtomicUsize,
        }
        #[async_trait::async_trait]
        impl EthTxBuilder for CountingBuilder {
            async fn build_release_tx(
                &self,
                w: &OutboundWithdrawal,
            ) -> BridgeResult<Option<Vec<u8>>> {
                self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(Some(vec![0xAA; (8 + w.nonce as usize).min(64)]))
            }
        }
        let builder = Arc::new(CountingBuilder {
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let relayer = OutboundRelayer::new(
            client.clone(),
            builder.clone(),
            tracker.clone(),
            Duration::from_millis(10),
            5,
        );
        relayer.scan_once().await.unwrap();
        relayer.scan_once().await.unwrap();
        relayer.scan_once().await.unwrap();
        assert_eq!(
            builder.calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "builder must be called exactly once per nonce"
        );
    }

    /// Builder errors don't poison the tracker — the same nonce must
    /// be retried on the next scan.
    #[tokio::test]
    async fn builder_error_leaves_nonce_for_retry() {
        let client = build_client_with_withdrawals(vec![pw(3, true)]);
        let tracker = Arc::new(InMemoryRelayedTracker::new());

        struct FlakyBuilder {
            // Returns Err on first call, Ok(Some(...)) thereafter.
            n: std::sync::atomic::AtomicUsize,
        }
        #[async_trait::async_trait]
        impl EthTxBuilder for FlakyBuilder {
            async fn build_release_tx(
                &self,
                _: &OutboundWithdrawal,
            ) -> BridgeResult<Option<Vec<u8>>> {
                let n = self.n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    Err(BridgeError::Internal("flake".to_string()))
                } else {
                    Ok(Some(vec![0x01]))
                }
            }
        }
        let builder = Arc::new(FlakyBuilder {
            n: std::sync::atomic::AtomicUsize::new(0),
        });
        let relayer = OutboundRelayer::new(
            client,
            builder,
            tracker.clone(),
            Duration::from_millis(10),
            5,
        );
        relayer.scan_once().await.unwrap();
        assert!(!tracker.is_relayed(3));
        relayer.scan_once().await.unwrap();
        assert!(tracker.is_relayed(3));
    }
}
