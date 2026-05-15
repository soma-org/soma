//! Soma-side RPC client for the bridge node.
//!
//! Mirrors Sui's `SuiClient<C>` (see `sui-bridge/src/sui_client.rs`):
//! a thin facade over the chain RPC that exposes only what the bridge
//! orchestrator and action executor need. The `SomaBridgeClientInner`
//! trait isolates the chain-RPC surface so executor logic can be unit-
//! tested against a mock.
//!
//! The methods here cover three needs of the bridge spine:
//!   1. **Read-side** — query bridge state (paused?, committee, deposit
//!      replay set, pending withdrawal records).
//!   2. **Identity** — chain id (for metrics labels) + relayer's USDC
//!      balance (for fee budgeting).
//!   3. **Write-side** — submit an already-signed Soma `Transaction`
//!      and return the resulting `TransactionEffects`.
//!
//! All methods are intentionally `&self` so a single `Arc<SomaBridgeClient<_>>`
//! can be shared across the signing-aggregation loop and the on-chain
//! execution loop without contention. Underlying tonic channels are
//! cheaply cloneable, so the inner impl clones per-call.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{Mutex, OnceCell};
use tracing::{error, info, warn};
use types::base::SomaAddress;
use types::bridge::{
    BridgeChainId, BridgeCommittee, BridgeMessageType, PendingWithdrawal, derive_bridge_record_id,
};
use types::effects::TransactionEffects;
use types::object::{CoinType, ObjectType};
use types::transaction::Transaction;

use crate::error::{BridgeError, BridgeResult};
use crate::retry::retry_with_backoff;

/// Status of an outbound bridge withdrawal as observed on the Soma chain.
///
/// Mirrors Sui's `BridgeActionStatus`. The "Claimed on Eth" state Sui
/// distinguishes is unobservable from the Soma side — once the cert is
/// attached, the relayer has done their job; whether Eth has released
/// USDC is a separate question answered by the eth-side syncer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeActionStatus {
    /// No `PendingWithdrawal` object at this nonce yet (the withdraw tx
    /// hasn't landed, or never will). Executor should wait or fail
    /// terminally depending on context.
    NotFound,
    /// `PendingWithdrawal` exists; `verified_signatures = None`.
    /// Executor should proceed to sig collection.
    Pending,
    /// `PendingWithdrawal.verified_signatures = Some(_)`. Cert is on
    /// chain; executor removes the action from the WAL and skips.
    CertAttached,
}

/// How long a single retry round inside an `_until_success` method runs
/// before logging and looping again. Mirrors sui-bridge's 30s budget.
const UNTIL_SUCCESS_ATTEMPT_BUDGET: Duration = Duration::from_secs(30);

/// Sleep before re-entering the retry round after `_until_success`'s
/// inner attempt budget is exhausted. Keeps the log volume reasonable
/// when the RPC is hard-down.
const UNTIL_SUCCESS_BETWEEN_ROUNDS: Duration = Duration::from_secs(5);

/// Read-only and write-side operations the bridge spine performs against
/// its local Soma fullnode. Implementors:
///   - [`SomaBridgeRpcClient`] — production, talks to a Soma RPC endpoint.
///   - `SomaBridgeMockClient` (test-only, deferred) — controllable from
///     unit tests for the executor.
#[async_trait]
pub trait SomaBridgeClientInner: Send + Sync + 'static {
    async fn is_bridge_paused(&self) -> BridgeResult<bool>;

    /// Total USDC currently minted on Soma (raw 6-decimal units),
    /// read from `BridgeState.total_usdc_supply`. The conservation-
    /// invariant watchdog reads this to compare against the Eth-side
    /// vault balance — divergence signals custody breach or counterfeit.
    async fn get_total_usdc_supply(&self) -> BridgeResult<u64>;

    /// Next withdrawal nonce the chain would assign — i.e. one past
    /// the highest withdrawal nonce that has been created. The
    /// outbound relayer reads this to bound its scan window from
    /// "look at every PendingWithdrawal from 0 to N" without having
    /// to guess a fixed upper bound.
    async fn get_next_withdrawal_nonce(&self) -> BridgeResult<u64>;

    /// Next expected per-message-type sequence number for `msg_type`,
    /// read off `BridgeState.system_message_seq_nums[msg_type]`.
    /// Returns 0 if the key is absent (the seq map starts empty and
    /// gets populated on first use). The watchdog reads this for
    /// EmergencyOp before firing an auto-pause cert — without it,
    /// the auto-pause would burn nonce 0 every time and collide with
    /// any manual EmergencyPause that already used it.
    async fn get_next_system_message_seq(
        &self,
        msg_type: types::bridge::BridgeMessageType,
    ) -> BridgeResult<u64>;

    async fn get_bridge_committee(&self) -> BridgeResult<BridgeCommittee>;

    /// Membership query against `BridgeState.processed_deposit_nonces`.
    /// Returns `true` iff `nonce` has already been recorded as a processed
    /// inbound deposit. Used by the executor to short-circuit re-submission
    /// of an action another relayer already landed.
    async fn is_deposit_processed(&self, nonce: u64) -> BridgeResult<bool>;

    /// Fetch the on-chain [`PendingWithdrawal`] object for `nonce`, if it
    /// exists. The object id is derived deterministically — anyone can
    /// compute it offline. Returns `None` if not present (no on-chain
    /// withdrawal at that nonce yet).
    async fn get_pending_withdrawal(&self, nonce: u64) -> BridgeResult<Option<PendingWithdrawal>>;

    /// Returns the chain identifier (used as a metric label).
    async fn get_chain_identifier(&self) -> BridgeResult<String>;

    /// Read the relayer's USDC accumulator balance — bridge action
    /// submissions are paid out of this. Returned in microdollars.
    async fn get_usdc_balance(&self, address: SomaAddress) -> BridgeResult<u64>;

    /// Submit a fully-signed `Transaction` and return its effects once
    /// the validator network produces them. Errors propagate the
    /// underlying RPC status; the caller distinguishes "didn't reach
    /// validators" (transient, retry) from "executed and failed"
    /// (terminal, manual intervention) by inspecting `effects.status()`.
    async fn execute_transaction(
        &self,
        transaction: &Transaction,
    ) -> BridgeResult<TransactionEffects>;
}

/// Generic bridge client. Parameterized over [`SomaBridgeClientInner`]
/// so the executor can be unit-tested against an in-memory mock.
pub struct SomaBridgeClient<C: SomaBridgeClientInner> {
    inner: C,
    /// Cached `BridgeChainId` of the configured Soma chain (e.g.
    /// [`BridgeChainId::SomaCustom`] for dev). Used to derive
    /// deterministic record ids and label metrics without re-querying
    /// the chain on every call.
    soma_chain_id: BridgeChainId,
    /// Cached chain identifier string (RPC `GetServiceInfo.chain_id`).
    /// Invariant for the lifetime of the client; populated on first
    /// access. Mirrors Sui's `OnceCell<ObjectArg>` for the bridge
    /// object — invariants get cached, mutables get re-read.
    chain_identifier: OnceCell<String>,
}

/// Production type alias — what the rest of the bridge node holds.
pub type ProductionSomaBridgeClient = SomaBridgeClient<SomaBridgeRpcClient>;

impl<C: SomaBridgeClientInner> SomaBridgeClient<C> {
    pub fn new(inner: C, soma_chain_id: BridgeChainId) -> Self {
        Self { inner, soma_chain_id, chain_identifier: OnceCell::new() }
    }

    pub fn soma_chain_id(&self) -> BridgeChainId {
        self.soma_chain_id
    }

    /// Sanity-check the RPC connection at startup. Mirrors Sui's
    /// `SuiClient::describe`: fetches chain identifier (which doubles
    /// as a connectivity probe) and logs it. Call this once after
    /// constructing the client; failure means the RPC URL is wrong or
    /// the fullnode is unreachable.
    pub async fn describe(&self) -> BridgeResult<()> {
        let chain_id = self.cached_chain_identifier().await?;
        info!(
            chain_id,
            soma_chain_id = ?self.soma_chain_id,
            "SomaBridgeClient connected"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Single-attempt pass-throughs. Use these when one error is fine to
    // surface to the caller (e.g. validation paths). Use the
    // `_until_success` variants below when called from the executor's
    // hot loop, where giving up on a transient RPC blip would mean a
    // user-visible bridge stall.
    // -----------------------------------------------------------------------

    pub async fn is_bridge_paused(&self) -> BridgeResult<bool> {
        self.inner.is_bridge_paused().await
    }

    pub async fn get_total_usdc_supply(&self) -> BridgeResult<u64> {
        self.inner.get_total_usdc_supply().await
    }

    pub async fn get_next_withdrawal_nonce(&self) -> BridgeResult<u64> {
        self.inner.get_next_withdrawal_nonce().await
    }

    pub async fn get_next_system_message_seq(
        &self,
        msg_type: types::bridge::BridgeMessageType,
    ) -> BridgeResult<u64> {
        self.inner.get_next_system_message_seq(msg_type).await
    }

    pub async fn get_bridge_committee(&self) -> BridgeResult<BridgeCommittee> {
        self.inner.get_bridge_committee().await
    }

    pub async fn is_deposit_processed(&self, nonce: u64) -> BridgeResult<bool> {
        self.inner.is_deposit_processed(nonce).await
    }

    pub async fn get_pending_withdrawal(
        &self,
        nonce: u64,
    ) -> BridgeResult<Option<PendingWithdrawal>> {
        self.inner.get_pending_withdrawal(nonce).await
    }

    /// Three-state withdrawal status mirroring Sui's `BridgeActionStatus`.
    /// Lets the executor pattern-match on lifecycle without re-checking
    /// `Option::is_some` + `verified_signatures.is_some()` at every call
    /// site.
    pub async fn get_withdrawal_status(&self, nonce: u64) -> BridgeResult<BridgeActionStatus> {
        match self.inner.get_pending_withdrawal(nonce).await? {
            None => Ok(BridgeActionStatus::NotFound),
            Some(pw) if pw.verified_signatures.is_some() => Ok(BridgeActionStatus::CertAttached),
            Some(_) => Ok(BridgeActionStatus::Pending),
        }
    }

    pub async fn get_chain_identifier(&self) -> BridgeResult<String> {
        self.inner.get_chain_identifier().await
    }

    /// Returns the cached chain identifier, fetching from the RPC on
    /// first access. The chain identifier is invariant for the lifetime
    /// of the connection, so this avoids a per-call RPC roundtrip on
    /// every metric label / log line.
    pub async fn cached_chain_identifier(&self) -> BridgeResult<&str> {
        self.chain_identifier
            .get_or_try_init(|| async { self.inner.get_chain_identifier().await })
            .await
            .map(|s| s.as_str())
    }

    pub async fn get_usdc_balance(&self, address: SomaAddress) -> BridgeResult<u64> {
        self.inner.get_usdc_balance(address).await
    }

    pub async fn execute_transaction(
        &self,
        transaction: &Transaction,
    ) -> BridgeResult<TransactionEffects> {
        self.inner.execute_transaction(transaction).await
    }

    /// Test-only accessor for the wrapped inner client. Lets unit +
    /// integration tests reach in and mutate mock state (flip pause
    /// atomics, install fake withdrawals, etc.). Integration tests in
    /// `tests/` are a separate crate and don't see `cfg(test)`-gated
    /// items, so this is gated by name (`_for_test`) rather than
    /// `cfg(test)`. Production code should not call this — it's a
    /// future `test-utils` feature in spirit.
    pub fn inner_for_test(&self) -> &C {
        &self.inner
    }

    // -----------------------------------------------------------------------
    // Until-success variants for executor hot paths. Each loops until
    // the call succeeds or the bridge node is shut down. Mirrors Sui's
    // `_until_success` methods; per-attempt budget is 30s (with
    // exponential backoff inside `retry_with_backoff`), and a 5s gap
    // between rounds keeps the log volume bounded when the RPC is
    // hard-down. The executor's "manual intervention required" ceiling
    // is enforced at the executor level (max attempts), not here.
    // -----------------------------------------------------------------------

    pub async fn is_bridge_paused_until_success(&self) -> bool {
        loop {
            match retry_with_backoff("is_bridge_paused", UNTIL_SUCCESS_ATTEMPT_BUDGET, || {
                self.inner.is_bridge_paused()
            })
            .await
            {
                Ok(v) => return v,
                Err(e) => {
                    error!(error = %e, "is_bridge_paused failed; retrying");
                    tokio::time::sleep(UNTIL_SUCCESS_BETWEEN_ROUNDS).await;
                }
            }
        }
    }

    pub async fn is_deposit_processed_until_success(&self, nonce: u64) -> bool {
        loop {
            match retry_with_backoff("is_deposit_processed", UNTIL_SUCCESS_ATTEMPT_BUDGET, || {
                self.inner.is_deposit_processed(nonce)
            })
            .await
            {
                Ok(v) => return v,
                Err(e) => {
                    error!(nonce, error = %e, "is_deposit_processed failed; retrying");
                    tokio::time::sleep(UNTIL_SUCCESS_BETWEEN_ROUNDS).await;
                }
            }
        }
    }

    pub async fn get_withdrawal_status_until_success(&self, nonce: u64) -> BridgeActionStatus {
        loop {
            match retry_with_backoff("get_withdrawal_status", UNTIL_SUCCESS_ATTEMPT_BUDGET, || {
                self.get_withdrawal_status(nonce)
            })
            .await
            {
                Ok(v) => return v,
                Err(e) => {
                    error!(nonce, error = %e, "get_withdrawal_status failed; retrying");
                    tokio::time::sleep(UNTIL_SUCCESS_BETWEEN_ROUNDS).await;
                }
            }
        }
    }

    pub async fn get_bridge_committee_until_success(&self) -> BridgeCommittee {
        loop {
            match retry_with_backoff("get_bridge_committee", UNTIL_SUCCESS_ATTEMPT_BUDGET, || {
                self.inner.get_bridge_committee()
            })
            .await
            {
                Ok(v) => return v,
                Err(e) => {
                    error!(error = %e, "get_bridge_committee failed; retrying");
                    tokio::time::sleep(UNTIL_SUCCESS_BETWEEN_ROUNDS).await;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Production implementation: wraps the gRPC client from the `rpc` crate.
// ---------------------------------------------------------------------------

/// Production `SomaBridgeClientInner` backed by the gRPC RPC client.
///
/// The underlying [`rpc::api::client::Client`] is held behind a
/// [`Mutex`] so the `&self` trait surface can drive a single tonic
/// channel without forcing callers to manage interior mutability.
/// The mutex is uncontended in the common case (one RPC per executor
/// step) and the underlying tonic transport doesn't benefit from
/// parallel calls on the same channel.
pub struct SomaBridgeRpcClient {
    client: Mutex<rpc::api::client::Client>,
}

impl SomaBridgeRpcClient {
    pub fn new(rpc_url: &str) -> BridgeResult<Self> {
        let client = rpc::api::client::Client::new(rpc_url.to_string())
            .map_err(|e| BridgeError::Internal(format!("invalid Soma RPC url: {e}")))?;
        Ok(Self { client: Mutex::new(client) })
    }

    pub fn from_client(client: rpc::api::client::Client) -> Self {
        Self { client: Mutex::new(client) }
    }
}

#[async_trait]
impl SomaBridgeClientInner for SomaBridgeRpcClient {
    async fn is_bridge_paused(&self) -> BridgeResult<bool> {
        let mut c = self.client.lock().await;
        let state = c
            .get_latest_system_state()
            .await
            .map_err(|s| BridgeError::Internal(format!("get_latest_system_state: {s}")))?;
        Ok(state.bridge_state().paused)
    }

    async fn get_total_usdc_supply(&self) -> BridgeResult<u64> {
        let mut c = self.client.lock().await;
        let state = c
            .get_latest_system_state()
            .await
            .map_err(|s| BridgeError::Internal(format!("get_latest_system_state: {s}")))?;
        Ok(state.bridge_state().total_usdc_supply)
    }

    async fn get_next_withdrawal_nonce(&self) -> BridgeResult<u64> {
        let mut c = self.client.lock().await;
        let state = c
            .get_latest_system_state()
            .await
            .map_err(|s| BridgeError::Internal(format!("get_latest_system_state: {s}")))?;
        Ok(state.bridge_state().next_withdrawal_nonce)
    }

    async fn get_next_system_message_seq(
        &self,
        msg_type: types::bridge::BridgeMessageType,
    ) -> BridgeResult<u64> {
        let mut c = self.client.lock().await;
        let state = c
            .get_latest_system_state()
            .await
            .map_err(|s| BridgeError::Internal(format!("get_latest_system_state: {s}")))?;
        // Absent key = 0 (the seq map starts empty; first message of
        // a type expects seq=0). Mirrors the on-chain executor's read
        // semantics in authority/src/execution/bridge.rs.
        Ok(state.bridge_state().system_message_seq_nums.get(&msg_type).copied().unwrap_or(0))
    }

    async fn get_bridge_committee(&self) -> BridgeResult<BridgeCommittee> {
        let mut c = self.client.lock().await;
        let state = c
            .get_latest_system_state()
            .await
            .map_err(|s| BridgeError::Internal(format!("get_latest_system_state: {s}")))?;
        Ok(state.bridge_state().bridge_committee.clone())
    }

    async fn is_deposit_processed(&self, nonce: u64) -> BridgeResult<bool> {
        let mut c = self.client.lock().await;
        let state = c
            .get_latest_system_state()
            .await
            .map_err(|s| BridgeError::Internal(format!("get_latest_system_state: {s}")))?;
        Ok(state.bridge_state().is_deposit_nonce_processed(nonce))
    }

    async fn get_pending_withdrawal(&self, nonce: u64) -> BridgeResult<Option<PendingWithdrawal>> {
        let id = derive_bridge_record_id(
            // Soma-side outbound — the source chain in the record id is Soma.
            // (Eth-side counterpart for inbound deposits would use Eth's id.)
            // Defaults match Stage 6 dev config; for prod the chain id comes
            // from the wrapper's `soma_chain_id()`.
            types::bridge::SOMA_BRIDGE_CHAIN_ID,
            BridgeMessageType::UsdcWithdraw,
            nonce,
        );
        let mut c = self.client.lock().await;
        match c.get_object(id).await {
            Ok(obj) => {
                let pending = obj
                    .deserialize_contents::<PendingWithdrawal>(ObjectType::PendingWithdrawal)
                    .ok_or_else(|| {
                        BridgeError::Internal(format!(
                            "object {id} is not a PendingWithdrawal",
                        ))
                    })?;
                Ok(Some(pending))
            }
            // `rpc` uses an older tonic than `bridge-node`; compare via the
            // numeric code so the two `tonic::Code` enums don't clash.
            Err(s) if s.code() as i32 == 5 /* NotFound */ => Ok(None),
            Err(s) => Err(BridgeError::Internal(format!("get_object({id}): {s}"))),
        }
    }

    async fn get_chain_identifier(&self) -> BridgeResult<String> {
        let mut c = self.client.lock().await;
        c.get_chain_identifier()
            .await
            .map_err(|s| BridgeError::Internal(format!("get_chain_identifier: {s}")))
    }

    async fn get_usdc_balance(&self, address: SomaAddress) -> BridgeResult<u64> {
        let mut c = self.client.lock().await;
        c.get_balance_by_coin_type(&address, CoinType::Usdc)
            .await
            .map_err(|s| BridgeError::Internal(format!("get_balance: {s}")))
    }

    async fn execute_transaction(
        &self,
        transaction: &Transaction,
    ) -> BridgeResult<TransactionEffects> {
        let mut c = self.client.lock().await;
        let resp = c
            .execute_transaction(transaction)
            .await
            .map_err(|s| BridgeError::Internal(format!("execute_transaction: {s}")))?;
        Ok(resp.effects)
    }
}

// ---------------------------------------------------------------------------
// Convenience: the `Arc` form is what the executor and orchestrator pass
// around. Provide a constructor so callers don't need to import `std::sync`.
// ---------------------------------------------------------------------------

impl SomaBridgeClient<SomaBridgeRpcClient> {
    /// Construct a production client, sanity-check the RPC via
    /// `describe()`, and wrap in an `Arc` for sharing across the
    /// signing-aggregation loop and the on-chain execution loop.
    /// Returns an error if the RPC is unreachable.
    pub async fn new_rpc(rpc_url: &str, soma_chain_id: BridgeChainId) -> BridgeResult<Arc<Self>> {
        let inner = SomaBridgeRpcClient::new(rpc_url)?;
        let client = Self::new(inner, soma_chain_id);
        client.describe().await?;
        Ok(Arc::new(client))
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A minimal in-memory mock for unit-testing executor logic. Backed
    /// by atomic counters so tests can flip pause state, advance the
    /// deposit watermark, etc.
    pub struct MockSomaClient {
        pub paused: std::sync::atomic::AtomicBool,
        pub total_usdc_supply: AtomicU64,
        pub system_message_seq_nums:
            std::sync::Mutex<std::collections::BTreeMap<types::bridge::BridgeMessageType, u64>>,
        pub deposit_nonces_seen: std::sync::Mutex<std::collections::BTreeSet<u64>>,
        pub committee: std::sync::Mutex<BridgeCommittee>,
        /// `nonce -> PendingWithdrawal`. Tests insert here to drive
        /// `get_withdrawal_status` through Pending/CertAttached states.
        pub pending_withdrawals:
            std::sync::Mutex<std::collections::BTreeMap<u64, PendingWithdrawal>>,
        pub balance: AtomicU64,
        pub chain_id: String,
        /// Counts how many times the inner `get_chain_identifier` has
        /// been called — used to assert the `cached_chain_identifier`
        /// only fetches once.
        pub chain_id_call_count: AtomicU64,
        pub last_submitted_digest: std::sync::Mutex<Option<types::digests::TransactionDigest>>,
    }

    impl MockSomaClient {
        pub fn new() -> Self {
            Self {
                paused: false.into(),
                total_usdc_supply: 0.into(),
                system_message_seq_nums: Default::default(),
                deposit_nonces_seen: Default::default(),
                committee: std::sync::Mutex::new(BridgeCommittee::empty()),
                pending_withdrawals: Default::default(),
                balance: 1_000_000_000.into(),
                chain_id: "soma-mock".to_string(),
                chain_id_call_count: 0.into(),
                last_submitted_digest: Default::default(),
            }
        }

        /// Test helper: install a `PendingWithdrawal` at `nonce`. The
        /// `cert_attached` flag toggles between the `Pending` and
        /// `CertAttached` lifecycle states.
        pub fn install_withdrawal(&self, nonce: u64, cert_attached: bool) {
            use types::bridge::WithdrawalCertificate;
            use types::object::ObjectID;
            let cert = if cert_attached {
                Some(WithdrawalCertificate { signatures: Default::default(), attached_at_epoch: 0 })
            } else {
                None
            };
            let pw = PendingWithdrawal {
                id: ObjectID::new([0u8; 32]),
                nonce,
                sender: SomaAddress::from([0u8; 32]),
                recipient_eth_address: [0u8; 20],
                amount: 1,
                created_at_ms: 0,
                target_chain: types::bridge::BridgeChainId::EthCustom,
                verified_signatures: cert,
            };
            self.pending_withdrawals.lock().unwrap().insert(nonce, pw);
        }
    }

    #[async_trait]
    impl SomaBridgeClientInner for MockSomaClient {
        async fn is_bridge_paused(&self) -> BridgeResult<bool> {
            Ok(self.paused.load(Ordering::SeqCst))
        }

        async fn get_total_usdc_supply(&self) -> BridgeResult<u64> {
            Ok(self.total_usdc_supply.load(Ordering::SeqCst))
        }

        async fn get_next_withdrawal_nonce(&self) -> BridgeResult<u64> {
            // Mock uses the highest installed withdrawal nonce + 1 as
            // a plausible "next nonce" so the relayer's scan window
            // covers everything the test installed.
            let pending = self.pending_withdrawals.lock().unwrap();
            Ok(pending.keys().copied().max().map(|n| n + 1).unwrap_or(0))
        }

        async fn get_next_system_message_seq(
            &self,
            msg_type: types::bridge::BridgeMessageType,
        ) -> BridgeResult<u64> {
            Ok(self.system_message_seq_nums.lock().unwrap().get(&msg_type).copied().unwrap_or(0))
        }

        async fn get_bridge_committee(&self) -> BridgeResult<BridgeCommittee> {
            Ok(self.committee.lock().unwrap().clone())
        }

        async fn is_deposit_processed(&self, nonce: u64) -> BridgeResult<bool> {
            Ok(self.deposit_nonces_seen.lock().unwrap().contains(&nonce))
        }

        async fn get_pending_withdrawal(
            &self,
            nonce: u64,
        ) -> BridgeResult<Option<PendingWithdrawal>> {
            Ok(self.pending_withdrawals.lock().unwrap().get(&nonce).cloned())
        }

        async fn get_chain_identifier(&self) -> BridgeResult<String> {
            self.chain_id_call_count.fetch_add(1, Ordering::SeqCst);
            Ok(self.chain_id.clone())
        }

        async fn get_usdc_balance(&self, _addr: SomaAddress) -> BridgeResult<u64> {
            Ok(self.balance.load(Ordering::SeqCst))
        }

        async fn execute_transaction(
            &self,
            transaction: &Transaction,
        ) -> BridgeResult<TransactionEffects> {
            // Record submission so tests can assert it happened, but we
            // can't synthesize realistic effects without a chain — return
            // an error so tests opt-in to richer mocking when needed.
            *self.last_submitted_digest.lock().unwrap() = Some(*transaction.digest());
            Err(BridgeError::Internal(
                "MockSomaClient::execute_transaction is a stub; override for richer tests"
                    .to_string(),
            ))
        }
    }

    #[tokio::test]
    async fn test_pause_reads_through() {
        let mock = MockSomaClient::new();
        mock.paused.store(true, Ordering::SeqCst);
        let client = SomaBridgeClient::new(mock, BridgeChainId::SomaCustom);
        assert!(client.is_bridge_paused().await.unwrap());
    }

    #[tokio::test]
    async fn test_deposit_nonce_seen() {
        let mock = MockSomaClient::new();
        mock.deposit_nonces_seen.lock().unwrap().insert(42);
        let client = SomaBridgeClient::new(mock, BridgeChainId::SomaCustom);
        assert!(client.is_deposit_processed(42).await.unwrap());
        assert!(!client.is_deposit_processed(43).await.unwrap());
    }

    #[tokio::test]
    async fn test_committee_pass_through() {
        let mock = MockSomaClient::new();
        let committee = BridgeCommittee::empty();
        *mock.committee.lock().unwrap() = committee.clone();
        let client = SomaBridgeClient::new(mock, BridgeChainId::SomaCustom);
        let got = client.get_bridge_committee().await.unwrap();
        assert_eq!(got.threshold_unpause, committee.threshold_unpause);
    }

    #[tokio::test]
    async fn test_chain_id_pass_through() {
        let mock = MockSomaClient::new();
        let client = SomaBridgeClient::new(mock, BridgeChainId::SomaCustom);
        assert_eq!(client.get_chain_identifier().await.unwrap(), "soma-mock");
        assert_eq!(client.soma_chain_id(), BridgeChainId::SomaCustom);
    }

    /// `cached_chain_identifier` must hit the inner RPC at most once
    /// regardless of how many callers race for it. Mirrors Sui's
    /// `OnceCell<ObjectArg>` invariant for the bridge object arg.
    #[tokio::test]
    async fn test_cached_chain_identifier_calls_inner_once() {
        let mock = MockSomaClient::new();
        let client = SomaBridgeClient::new(mock, BridgeChainId::SomaCustom);
        for _ in 0..10 {
            let id = client.cached_chain_identifier().await.unwrap();
            assert_eq!(id, "soma-mock");
        }
        // Inner RPC should have been touched exactly once.
        let count = client.inner.chain_id_call_count.load(Ordering::SeqCst);
        assert_eq!(count, 1, "cached_chain_identifier must dedupe inner calls");
    }

    /// Withdrawal lifecycle: NotFound → Pending → CertAttached.
    /// Mirrors Sui's `BridgeActionStatus` flow (Pending → Approved → Claimed).
    #[tokio::test]
    async fn test_withdrawal_status_lifecycle() {
        let mock = MockSomaClient::new();
        let client = SomaBridgeClient::new(mock, BridgeChainId::SomaCustom);

        // No object → NotFound.
        assert_eq!(client.get_withdrawal_status(0).await.unwrap(), BridgeActionStatus::NotFound,);

        // Object exists, no cert → Pending.
        client.inner.install_withdrawal(0, false);
        assert_eq!(client.get_withdrawal_status(0).await.unwrap(), BridgeActionStatus::Pending,);

        // Cert attached → CertAttached.
        client.inner.install_withdrawal(0, true);
        assert_eq!(
            client.get_withdrawal_status(0).await.unwrap(),
            BridgeActionStatus::CertAttached,
        );

        // Different nonces are independent.
        assert_eq!(client.get_withdrawal_status(99).await.unwrap(), BridgeActionStatus::NotFound,);
    }

    /// `describe()` should round-trip the chain identifier through the
    /// RPC. Used as the production startup health check.
    #[tokio::test]
    async fn test_describe_succeeds_when_inner_works() {
        let mock = MockSomaClient::new();
        let client = SomaBridgeClient::new(mock, BridgeChainId::SomaCustom);
        client.describe().await.expect("describe must succeed");
        // Side effect: warmed the cache, so subsequent reads are free.
        let _ = client.cached_chain_identifier().await.unwrap();
        assert_eq!(client.inner.chain_id_call_count.load(Ordering::SeqCst), 1);
    }
}
