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

use alloy::primitives::TxHash;
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

/// What the relayer does with each cert-attached withdrawal it
/// discovers. The production impl ([`crate::eth_submitter::EthSubmitter`])
/// builds the Eth-side ABI calldata, wraps it in an EIP-1559 envelope,
/// signs with the operator wallet, and submits via
/// `eth_sendRawTransaction`. Tests use mocks that record the call
/// without burning gas.
///
/// `Ok(Some(TxHash))` — submitted; mark relayed
/// `Ok(None)` — skipped intentionally (e.g. malformed cert)
/// `Err(_)` — transient; the relayer logs + retries next scan
#[async_trait::async_trait]
pub trait WithdrawalSubmitter: Send + Sync {
    async fn submit(
        &self,
        withdrawal: &OutboundWithdrawal,
    ) -> BridgeResult<Option<TxHash>>;
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

/// The outbound relayer. Polls Soma for completed PendingWithdrawals
/// and hands each one to the [`WithdrawalSubmitter`], which in
/// production calls `eth_sendRawTransaction` on the operator's Eth
/// RPC.
pub struct OutboundRelayer<C: SomaBridgeClientInner> {
    soma_client: Arc<SomaBridgeClient<C>>,
    submitter: Arc<dyn WithdrawalSubmitter>,
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
        submitter: Arc<dyn WithdrawalSubmitter>,
        tracker: Arc<dyn RelayedTracker>,
        poll_interval: Duration,
        scan_window: u64,
    ) -> Self {
        Self {
            soma_client,
            submitter,
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
            match self.submitter.submit(&outbound).await {
                Ok(Some(tx_hash)) => {
                    info!(
                        nonce = outbound.nonce,
                        sig_count = outbound.certificate.signatures.len(),
                        ?tx_hash,
                        "Eth-side release tx submitted"
                    );
                    self.tracker.mark_relayed(outbound.nonce);
                }
                Ok(None) => {
                    debug!(nonce = outbound.nonce, "submitter returned None; skipping");
                }
                Err(e) => {
                    warn!(
                        nonce = outbound.nonce,
                        error = %e,
                        "submit failed; will retry next scan"
                    );
                }
            }
        }
        Ok(())
    }
}

/// `WithdrawalSubmitter` impl on the real Eth submitter.
#[async_trait::async_trait]
impl WithdrawalSubmitter for crate::eth_submitter::EthSubmitter {
    async fn submit(
        &self,
        withdrawal: &OutboundWithdrawal,
    ) -> BridgeResult<Option<TxHash>> {
        if withdrawal.certificate.signatures.is_empty() {
            return Err(BridgeError::Internal(
                "outbound relayer received cert with no signatures".to_string(),
            ));
        }
        let tx_hash = self.submit_withdrawal(withdrawal).await?;
        Ok(Some(tx_hash))
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

    /// Sentinel tx hash for stub submitters — distinct from anything
    /// a real submitter would produce so test output makes the source
    /// obvious.
    fn stub_tx_hash(nonce: u64) -> alloy::primitives::TxHash {
        let mut b = [0u8; 32];
        b[24..32].copy_from_slice(&nonce.to_be_bytes());
        b[0] = 0xFA; // "fake"
        b[1] = 0xCE;
        alloy::primitives::TxHash::from(b)
    }

    /// Mock submitter that always succeeds; records each call.
    struct CapturingSubmitter {
        calls: std::sync::atomic::AtomicUsize,
        last_nonce: std::sync::Mutex<Option<u64>>,
    }
    #[async_trait::async_trait]
    impl WithdrawalSubmitter for CapturingSubmitter {
        async fn submit(
            &self,
            w: &OutboundWithdrawal,
        ) -> BridgeResult<Option<alloy::primitives::TxHash>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            *self.last_nonce.lock().unwrap() = Some(w.nonce);
            Ok(Some(stub_tx_hash(w.nonce)))
        }
    }
    fn capturing() -> Arc<CapturingSubmitter> {
        Arc::new(CapturingSubmitter {
            calls: std::sync::atomic::AtomicUsize::new(0),
            last_nonce: std::sync::Mutex::new(None),
        })
    }

    /// The relayer must only act on withdrawals that have a cert
    /// attached. A bare PendingWithdrawal (still being signed) is
    /// skipped.
    #[tokio::test]
    async fn scan_skips_unsigned_withdrawals() {
        let client = build_client_with_withdrawals(vec![pw(0, false), pw(1, true)]);
        let tracker = Arc::new(InMemoryRelayedTracker::new());
        let submitter = capturing();
        let relayer = OutboundRelayer::new(
            client,
            submitter.clone(),
            tracker.clone(),
            Duration::from_millis(10),
            5,
        );
        relayer.scan_once().await.unwrap();
        assert!(!tracker.is_relayed(0), "unsigned must not be marked relayed");
        assert!(tracker.is_relayed(1), "cert-attached must be marked relayed");
        assert_eq!(*submitter.last_nonce.lock().unwrap(), Some(1));
    }

    /// Idempotency: a second scan after the first must not re-submit
    /// an already-relayed nonce.
    #[tokio::test]
    async fn scan_is_idempotent_across_runs() {
        let client = build_client_with_withdrawals(vec![pw(2, true)]);
        let tracker = Arc::new(InMemoryRelayedTracker::new());
        let submitter = capturing();
        let relayer = OutboundRelayer::new(
            client.clone(),
            submitter.clone(),
            tracker.clone(),
            Duration::from_millis(10),
            5,
        );
        relayer.scan_once().await.unwrap();
        relayer.scan_once().await.unwrap();
        relayer.scan_once().await.unwrap();
        assert_eq!(
            submitter.calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "submitter must be called exactly once per nonce"
        );
    }

    /// Submitter errors don't poison the tracker — the same nonce
    /// must be retried on the next scan, and only marked relayed once
    /// a real submission succeeds.
    #[tokio::test]
    async fn submitter_error_leaves_nonce_for_retry() {
        let client = build_client_with_withdrawals(vec![pw(3, true)]);
        let tracker = Arc::new(InMemoryRelayedTracker::new());

        struct FlakySubmitter {
            n: std::sync::atomic::AtomicUsize,
        }
        #[async_trait::async_trait]
        impl WithdrawalSubmitter for FlakySubmitter {
            async fn submit(
                &self,
                _w: &OutboundWithdrawal,
            ) -> BridgeResult<Option<alloy::primitives::TxHash>> {
                let n = self.n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    Err(BridgeError::Internal("flake".to_string()))
                } else {
                    Ok(Some(stub_tx_hash(0)))
                }
            }
        }
        let submitter = Arc::new(FlakySubmitter {
            n: std::sync::atomic::AtomicUsize::new(0),
        });
        let relayer = OutboundRelayer::new(
            client,
            submitter,
            tracker.clone(),
            Duration::from_millis(10),
            5,
        );
        relayer.scan_once().await.unwrap();
        assert!(!tracker.is_relayed(3));
        relayer.scan_once().await.unwrap();
        assert!(tracker.is_relayed(3));
    }

    /// `Ok(None)` from the submitter (intentional skip) leaves the
    /// nonce in the unrelayed set — different from `Err` (transient
    /// retry) and different from `Ok(Some(_))` (success).
    #[tokio::test]
    async fn submitter_returning_none_does_not_mark_relayed() {
        let client = build_client_with_withdrawals(vec![pw(4, true)]);
        let tracker = Arc::new(InMemoryRelayedTracker::new());

        struct NoneSubmitter;
        #[async_trait::async_trait]
        impl WithdrawalSubmitter for NoneSubmitter {
            async fn submit(
                &self,
                _: &OutboundWithdrawal,
            ) -> BridgeResult<Option<alloy::primitives::TxHash>> {
                Ok(None)
            }
        }
        let relayer = OutboundRelayer::new(
            client,
            Arc::new(NoneSubmitter),
            tracker.clone(),
            Duration::from_millis(10),
            5,
        );
        relayer.scan_once().await.unwrap();
        assert!(!tracker.is_relayed(4));
    }
}
