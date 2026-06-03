// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Pre-execution funds-sufficiency scheduling for balance-mode withdrawals.
//!
//! ## Why this exists
//!
//! Soma's reservation pre-pass (consensus_handler::run_reservation_prepass) is
//! the sole cross-commit double-spend gate, but it seeds balances by reading
//! the perpetual store directly (`WritebackCache::get_balance`). Settlement —
//! the only writer of balances — runs asynchronously and per-checkpoint, with
//! no barrier before the next commit's pre-pass. So the pre-pass can read a
//! stale (pre-settlement) balance, and because settlement-persist timing
//! differs per validator, admission decisions can diverge across validators →
//! a state fork. (See `settlement_scheduler.rs` "Differences from Sui": Soma
//! dropped Sui's post-settlement barrier and has no funds-withdraw scheduler.)
//!
//! This module is the Soma port of Sui's `FundsWithdrawScheduler` (design doc:
//! `crates/sui-core/src/accumulators/design_docs/address_funds_scheduling.md`).
//! It decides, deterministically and *before* execution, whether each
//! withdrawing transaction's declared maximum withdrawal can be covered.
//!
//! ## This file: the NAIVE core
//!
//! Per the Sui doc, the simplest correct model is "wait until the version is
//! fully settled, then check balances from storage." This file implements the
//! version-independent *essence* of that model as pure, deterministic logic
//! that any scheduler (naive or eager) must get right:
//!
//!   * [`VersionGate`] — given a request's settlement version and the current
//!     settled version, decide whether to check now, skip (already settled),
//!     or wait (not yet settled).
//!   * [`check_withdraws_at_settled_version`] — given the *exact* settled
//!     balances and the consensus-ordered transactions, return a per-tx
//!     [`ScheduleResult`], deducting each tx's declared maximum so later txs in
//!     the same batch see the reduced balance.
//!
//! Determinism rests on (1) consensus ordering of the txs, (2) deducting the
//! declared MAX (not the actual, unknown-until-execution amount), and (3)
//! deciding only against a settled (final) version. The wiring that feeds this
//! core from the execution scheduler + a settlement hook (the eager
//! optimization, in-memory reservation tracking, and the commit→checkpoint
//! version mapping) is integrated separately — this module is intentionally
//! pure so it can be exhaustively unit-tested in isolation.

use std::collections::BTreeMap;

use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::object::CoinType;

/// Soma's settlement "version": a globally-monotonic value a withdrawal is
/// evaluated against. It plays the role of Sui's accumulator `SequenceNumber`.
///
/// Soma's per-commit `checkpoint_height` equals the consensus round, which
/// **resets per epoch** (a fresh Mysticeti instance starts each epoch at round
/// 0). A bare height is therefore NOT monotonic across epochs — a stale
/// prior-epoch height would make a fresh epoch's (low-numbered) settlements
/// look already-applied and be ignored, re-opening the deposit fork at every
/// epoch boundary. So the version packs the epoch into the high bits above the
/// height, making it strictly increasing across `(epoch, round)`. See
/// [`pack_version`].
pub type SettlementVersion = u64;

/// Bits reserved for the per-epoch height (consensus round) in a packed
/// [`SettlementVersion`]; the epoch occupies the bits above. 40 bits ≈ 1.1e12
/// rounds/epoch (epochs are ~hours at sub-second rounds → ~1e6 rounds), leaving
/// 24 bits ≈ 16.7M epochs. Both bounds are unreachable in practice.
pub const VERSION_HEIGHT_BITS: u32 = 40;

/// Pack `(epoch, height)` into a globally-monotonic [`SettlementVersion`].
/// Monotonic because the epoch dominates the ordering and the height increases
/// within an epoch. Used by the settlement hook (writes the packed version into
/// the balance store), the seed read, and the consensus handler (request
/// version). See [`SettlementVersion`].
pub fn pack_version(epoch: u64, height: u64) -> SettlementVersion {
    debug_assert!(
        height < (1u64 << VERSION_HEIGHT_BITS),
        "consensus round {height} overflows the {VERSION_HEIGHT_BITS}-bit height field"
    );
    debug_assert!(
        epoch < (1u64 << (64 - VERSION_HEIGHT_BITS)),
        "epoch {epoch} overflows the packed-version epoch field"
    );
    (epoch << VERSION_HEIGHT_BITS) | (height & ((1u64 << VERSION_HEIGHT_BITS) - 1))
}

/// An accumulator account key — the same `(owner, coin_type)` pair the balance
/// store is keyed by.
pub type AccountKey = (SomaAddress, CoinType);

/// The maximum withdrawal a single transaction may perform from each account,
/// declared in the transaction data (an upper bound; execution may withdraw
/// less, never more). Mirrors Sui's `TxFundsWithdraw`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxFundsWithdraw {
    pub tx_digest: TransactionDigest,
    /// Per-account declared maximum withdrawal (already aggregated per account
    /// for this tx — at most one entry per `AccountKey`).
    pub reservations: BTreeMap<AccountKey, u64>,
}

/// Per-transaction scheduling outcome. Mirrors Sui's `ScheduleResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleResult {
    /// Every account can cover this tx's declared maximum. The tx may execute.
    SufficientFunds,
    /// Some account cannot cover the declared maximum at a *settled* version
    /// (the balance is final). The tx must fail early without executing.
    InsufficientFunds,
    /// The requested version has already been settled past — another path
    /// (e.g. the checkpoint executor) advanced storage. No action needed.
    SkipSchedule,
}

/// Whether a withdrawal batch at `request_version` can be decided against the
/// current `settled_version`. Mirrors the version gate in Sui's
/// `schedule_withdraws` (§3.1 of the design doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionGate {
    /// `request_version == settled_version`: storage is final for this
    /// version. Read balances and decide each tx now.
    CheckNow,
    /// `request_version < settled_version`: storage has already moved past
    /// this batch (settled by another path). Return `SkipSchedule` for all.
    Skip,
    /// `request_version > settled_version`: settlement hasn't reached this
    /// version yet. Block until it does, then re-evaluate.
    Wait,
}

impl VersionGate {
    /// Classify a withdrawal batch's `request_version` against the current
    /// `settled_version`. This is the per-validator-deterministic gate: every
    /// validator agrees on the settled version boundary, so they all
    /// CheckNow / Skip / Wait identically.
    pub fn classify(
        request_version: SettlementVersion,
        settled_version: SettlementVersion,
    ) -> Self {
        use std::cmp::Ordering::*;
        match request_version.cmp(&settled_version) {
            Equal => VersionGate::CheckNow,
            Less => VersionGate::Skip,
            Greater => VersionGate::Wait,
        }
    }
}

/// Check a consensus-ordered batch of withdrawals against the EXACT settled
/// balances for their version. Returns one [`ScheduleResult`] per input tx, in
/// the same order.
///
/// This is the heart of the naive scheduler (Sui doc §2): with final balances
/// in hand, walk the transactions in consensus order, deducting each tx's
/// declared maximum from a local balance copy so later txs see the reduced
/// balance. A tx is `SufficientFunds` iff *every* account it withdraws from can
/// cover its declared maximum given the deductions of all earlier-ordered
/// successful txs; otherwise `InsufficientFunds`.
///
/// `settled_balances` must contain an entry for every account referenced by
/// the txs (missing ⇒ treated as 0). Determinism: pure function of the ordered
/// txs and the (consensus-determined) settled balances.
pub fn check_withdraws_at_settled_version(
    settled_balances: &BTreeMap<AccountKey, u64>,
    txs: &[TxFundsWithdraw],
) -> Vec<ScheduleResult> {
    // Local, mutable copy of the balances we draw down as we approve txs.
    let mut available: BTreeMap<AccountKey, u64> = settled_balances.clone();
    let mut results = Vec::with_capacity(txs.len());

    for tx in txs {
        // A tx succeeds only if EVERY account can cover its declared max.
        // Check all first (no partial deduction) so a failing account leaves
        // the available balances untouched for this tx.
        let covered = tx
            .reservations
            .iter()
            .all(|(acct, &amount)| available.get(acct).copied().unwrap_or(0) >= amount);

        if covered {
            for (acct, &amount) in &tx.reservations {
                let bal = available.entry(*acct).or_insert(0);
                // `covered` guarantees `*bal >= amount`; saturating for safety.
                *bal = bal.saturating_sub(amount);
            }
            results.push(ScheduleResult::SufficientFunds);
        } else {
            // Insufficient at a settled (final) version — deterministic fail.
            // No deduction; this tx withdraws nothing.
            results.push(ScheduleResult::InsufficientFunds);
        }
    }

    results
}

// ===========================================================================
// Eager scheduler: per-account state machine
// ===========================================================================
//
// The naive core above must WAIT for a version to settle before deciding. The
// eager scheduler (Sui doc §3) maintains in-memory per-account state so it can
// approve withdrawals immediately against a conservative lower bound, without
// waiting — and only WAITS (Pending) for the genuinely-uncertain case (a
// withdrawal that doesn't fit and whose version isn't settled, where a future
// settlement might deposit funds). Crucially, this is what makes the design
// deterministic despite not-yet-settled deposits: validators never speculate
// on unsettled deposits, they wait for the version, and all agree on when a
// version is settled.
//
// This section implements the per-account state machine (Sui doc §3.3–§3.5,
// §4). The cross-account coordination for multi-account transactions and the
// async channel wrapper are layered on top at integration time; this core is
// kept synchronous and pure so it is exhaustively unit-testable in isolation.

use std::collections::VecDeque;

/// The per-account outcome of a (drained) withdrawal request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountOutcome {
    /// This account reserved the request's declared maximum.
    Reserved(TransactionDigest),
    /// This account cannot cover the request at a settled (final) version.
    Insufficient(TransactionDigest),
}

#[derive(Debug, Clone)]
struct PendingReq {
    id: TransactionDigest,
    version: SettlementVersion,
    amount: u64,
}

/// Per-account in-memory state (Sui doc §3.3). Tracks the known balance, the
/// version it corresponds to, the running total of reserved (approved but
/// not-yet-settled) withdrawals tagged by request version, and a FIFO queue of
/// withdrawals that could not yet be decided.
#[derive(Debug, Clone)]
pub struct AccountState {
    /// Most recent balance known to be correct (Sui's `last_updated_balance`).
    balance: u64,
    /// The version `balance` corresponds to (Sui's `last_updated_version`).
    settled_version: SettlementVersion,
    /// Approved-but-unsettled reservations, tagged by the request's version,
    /// in reserve order (Sui's `reserved_funds`).
    reserved: VecDeque<(SettlementVersion, u64)>,
    /// Running sum of `reserved` for O(1) sufficiency checks.
    total_reserved: u64,
    /// FIFO queue of undecided withdrawals (Sui's `pending_reservations`).
    pending: VecDeque<PendingReq>,
}

impl AccountState {
    /// Seed from the balance read at `version` (the first time the account is
    /// seen). Mirrors Sui's lazy init from storage.
    pub fn new(balance: u64, version: SettlementVersion) -> Self {
        Self {
            balance,
            settled_version: version,
            reserved: VecDeque::new(),
            total_reserved: 0,
            pending: VecDeque::new(),
        }
    }

    /// Submit a withdrawal of declared maximum `amount` at `version`. Appends
    /// to the FIFO queue and drains as far as possible, returning the per-tx
    /// outcomes that became decided (in order). A tx that doesn't fit and whose
    /// version isn't settled yields NO outcome (it stays pending — "wait").
    pub fn reserve(
        &mut self,
        version: SettlementVersion,
        amount: u64,
        id: TransactionDigest,
    ) -> Vec<AccountOutcome> {
        self.pending.push_back(PendingReq { id, version, amount });
        self.drain()
    }

    /// Apply a settlement that advances this account to `new_settled_version`
    /// with net balance change `delta` (deposits +, actual withdrawals −).
    /// Releases reservations for now-settled versions, advances the balance,
    /// and drains the pending queue (which may now resolve). Idempotent: a
    /// settlement at or below the current settled version is ignored (Sui §4).
    pub fn settle(
        &mut self,
        new_settled_version: SettlementVersion,
        delta: i128,
    ) -> Vec<AccountOutcome> {
        if new_settled_version <= self.settled_version {
            return Vec::new();
        }
        // Apply the actual net delta (clamped at 0 — balances never go
        // negative; settlement is the authoritative source).
        let updated = (self.balance as i128).saturating_add(delta);
        self.balance = updated.max(0) as u64;

        // Release reservations tagged at any version now settled (< the new
        // settled version): the persisted balance now reflects their actual
        // (possibly smaller) withdrawal, so the reserved MAX must stop being
        // held back. Settlement normally advances one version at a time, so
        // this releases just the freshly-settled version's tags.
        let mut kept = VecDeque::with_capacity(self.reserved.len());
        while let Some((v, amt)) = self.reserved.pop_front() {
            if v < new_settled_version {
                self.total_reserved = self.total_reserved.saturating_sub(amt);
            } else {
                kept.push_back((v, amt));
            }
        }
        self.reserved = kept;
        self.settled_version = new_settled_version;

        self.drain()
    }

    /// True when the account has no reserved funds and nothing pending — the
    /// caller may garbage-collect it (Sui §4 step 3).
    pub fn is_empty(&self) -> bool {
        self.total_reserved == 0 && self.pending.is_empty() && self.reserved.is_empty()
    }

    /// Process the pending FIFO from the front. Each front request is either
    /// Reserved (fits the conservative bound → reserve & continue),
    /// Insufficient (doesn't fit AND its version is settled → final fail &
    /// continue), or Pending (doesn't fit and version not settled → STOP:
    /// head-of-line blocking preserves determinism, Sui §3.5).
    fn drain(&mut self) -> Vec<AccountOutcome> {
        let mut out = Vec::new();
        while let Some(front) = self.pending.front() {
            if self.total_reserved.saturating_add(front.amount) <= self.balance {
                let req = self.pending.pop_front().expect("front exists");
                self.total_reserved = self.total_reserved.saturating_add(req.amount);
                self.reserved.push_back((req.version, req.amount));
                out.push(AccountOutcome::Reserved(req.id));
            } else if front.version <= self.settled_version {
                // Balance is final for this version and still short → doomed.
                let req = self.pending.pop_front().expect("front exists");
                out.push(AccountOutcome::Insufficient(req.id));
            } else {
                // Uncertain: a later settlement may deposit funds. Wait, and
                // block everything behind it (FIFO) for determinism.
                break;
            }
        }
        out
    }
}

// ===========================================================================
// Eager scheduler: multi-account coordinator
// ===========================================================================
//
// A single transaction may withdraw from several accounts; it is `Sufficient`
// only if EVERY account can reserve its portion, `Insufficient` if ANY account
// is deterministically short, and `Pending` while any account is still waiting
// and none has failed (Sui doc §5.3). This coordinator owns the per-account
// `AccountState` map and a per-tx progress record, routes each account's
// drained outcomes back to the owning tx, and emits a per-tx result the moment
// it becomes decided — either immediately at `schedule_batch` time or later
// when `settle` drains a previously-blocked account.
//
// Determinism (Sui doc §6): accounts are visited in sorted (`BTreeMap`) order,
// txs are processed in consensus order, per-account queues are FIFO, and a
// decision is only finalized against a settled version. This is the complete
// scheduler *logic*; the async channel wrapper and the hot-path wiring
// (routing withdrawals from the execution scheduler, the settlement hook, and
// Soma's commit→checkpoint version mapping) are layered on at integration.

/// The decided outcome for a whole transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxScheduleResult {
    /// All accounts reserved — the tx may execute.
    Sufficient,
    /// Some account is short at a settled version — the tx must fail early.
    Insufficient,
    /// Not yet decided — waiting on settlement for at least one account.
    Pending,
    /// The version was already settled past — no scheduling needed.
    Skip,
}

#[derive(Debug)]
struct TxProgress {
    /// Accounts that have not yet reserved this tx's portion.
    remaining: usize,
    /// Set once an account reports the tx Insufficient.
    failed: bool,
    /// Set once a final (Sufficient/Insufficient) result has been emitted.
    decided: bool,
}

/// The eager funds-withdraw scheduler (single-threaded core). Tracks per-account
/// state and per-tx progress; deterministic given the consensus order of
/// `schedule_batch` / `settle` calls.
#[derive(Debug)]
pub struct EagerFundsScheduler {
    accounts: BTreeMap<AccountKey, AccountState>,
    txs: BTreeMap<TransactionDigest, TxProgress>,
    settled_version: SettlementVersion,
}

impl EagerFundsScheduler {
    pub fn new(settled_version: SettlementVersion) -> Self {
        Self { accounts: BTreeMap::new(), txs: BTreeMap::new(), settled_version }
    }

    /// Schedule a consensus-ordered batch of withdrawals declared against
    /// `version`. `seed` supplies an account's `(balance, version)` the first
    /// time it is referenced — integration reads this from the store via
    /// `get_balance_with_version`. The seed *version* matters: it tells the
    /// scheduler which settlement state the balance already reflects, so a
    /// later [`Self::settle`] at or below that version is correctly ignored
    /// (no double-apply). Seeding at the scheduler's `settled_version` instead
    /// would double-count any settlement the store already absorbed but the
    /// scheduler hasn't — the exact deposit-then-withdraw fork.
    ///
    /// Returns one `(tx_digest, result)` per input tx, in order. `Pending`
    /// results are finalized later by [`Self::settle`].
    pub fn schedule_batch(
        &mut self,
        version: SettlementVersion,
        txs: &[TxFundsWithdraw],
        seed: impl Fn(AccountKey) -> (u64, SettlementVersion),
    ) -> Vec<(TransactionDigest, TxScheduleResult)> {
        // Version gate (Sui §3.1): a batch older than the settled version was
        // already carried forward by some other path → skip all.
        if version < self.settled_version {
            return txs.iter().map(|t| (t.tx_digest, TxScheduleResult::Skip)).collect();
        }

        // The scheduler's global settled version. An account's seed version is
        // raised to at least this: if the store's per-account version is BEHIND
        // global, then no settlement in (store_version, global] touched this
        // account (otherwise the store would reflect it), so its balance is
        // unchanged through `global` and we may treat it as settled there.
        // Seeding at the raw (behind) store version would strand the account —
        // the catch-up `settle` for those versions was already processed before
        // the account was first seen, so it would never advance and a short
        // withdrawal would wait Pending forever.
        let global = self.settled_version;

        let mut results = Vec::with_capacity(txs.len());
        for tx in txs {
            // Initialize progress BEFORE submitting to any account, so an
            // immediate Reserved correctly decrements from the full count.
            self.txs.insert(
                tx.tx_digest,
                TxProgress { remaining: tx.reservations.len(), failed: false, decided: false },
            );

            // Submit the tx's portion to every account it touches, in sorted
            // order (BTreeMap), seeding untracked accounts from the store's
            // (balance, version). Submit to ALL accounts even if one fails —
            // succeeding accounts keep their reservation (Sui §5.3) and
            // accounting stays consistent across validators.
            for (&account, &amount) in &tx.reservations {
                let state = self.accounts.entry(account).or_insert_with(|| {
                    let (bal, ver) = seed(account);
                    AccountState::new(bal, ver.max(global))
                });
                let outcomes = state.reserve(version, amount, tx.tx_digest);
                self.apply_outcomes(outcomes);
            }

            results.push((tx.tx_digest, self.current_result(&tx.tx_digest)));
        }
        results
    }

    /// Apply a settlement advancing the scheduler to `new_version` with the
    /// given per-account net deltas. Drains every affected account's pending
    /// queue and returns the `(tx_digest, result)` pairs that became newly
    /// decided (`Sufficient`/`Insufficient`). Idempotent below `new_version`.
    pub fn settle(
        &mut self,
        new_version: SettlementVersion,
        deltas: &BTreeMap<AccountKey, i128>,
    ) -> Vec<(TransactionDigest, TxScheduleResult)> {
        if new_version <= self.settled_version {
            return Vec::new();
        }
        self.settled_version = new_version;

        // Settle the accounts named in the deltas, plus any tracked account
        // that hasn't yet reached the new version (so reservations release and
        // pending queues drain). Visit in sorted order for determinism.
        let keys: std::collections::BTreeSet<AccountKey> =
            self.accounts.keys().copied().chain(deltas.keys().copied()).collect();

        let mut decided = Vec::new();
        for account in keys {
            let delta = deltas.get(&account).copied().unwrap_or(0);
            // Only accounts we are tracking have pending/reserved state. An
            // untracked account with a delta has no scheduling consequence.
            if let Some(state) = self.accounts.get_mut(&account) {
                let outcomes = state.settle(new_version, delta);
                decided.extend(self.drain_outcomes_collecting(outcomes));
            }
        }

        // Garbage-collect accounts with nothing reserved or pending.
        self.accounts.retain(|_, s| !s.is_empty());
        decided
    }

    /// Route per-account outcomes into per-tx progress (no return — used during
    /// `schedule_batch` where the batch loop reads results afterward).
    fn apply_outcomes(&mut self, outcomes: Vec<AccountOutcome>) {
        for o in outcomes {
            self.record_outcome(o);
        }
    }

    /// Route per-account outcomes and collect any newly-decided tx results.
    fn drain_outcomes_collecting(
        &mut self,
        outcomes: Vec<AccountOutcome>,
    ) -> Vec<(TransactionDigest, TxScheduleResult)> {
        let mut decided = Vec::new();
        for o in outcomes {
            if let Some(result) = self.record_outcome(o) {
                decided.push(result);
            }
        }
        decided
    }

    /// Update a tx's progress from one account outcome. Returns a
    /// `(digest, result)` iff this outcome finalized the tx.
    fn record_outcome(
        &mut self,
        outcome: AccountOutcome,
    ) -> Option<(TransactionDigest, TxScheduleResult)> {
        match outcome {
            AccountOutcome::Reserved(id) => {
                let p = self.txs.get_mut(&id)?;
                if p.decided {
                    return None;
                }
                p.remaining = p.remaining.saturating_sub(1);
                if p.remaining == 0 && !p.failed {
                    p.decided = true;
                    return Some((id, TxScheduleResult::Sufficient));
                }
                None
            }
            AccountOutcome::Insufficient(id) => {
                let p = self.txs.get_mut(&id)?;
                if p.decided {
                    return None;
                }
                p.failed = true;
                p.decided = true;
                Some((id, TxScheduleResult::Insufficient))
            }
        }
    }

    /// The current result for a tx right after it was submitted to all its
    /// accounts in `schedule_batch`.
    fn current_result(&self, id: &TransactionDigest) -> TxScheduleResult {
        match self.txs.get(id) {
            Some(p) if p.failed => TxScheduleResult::Insufficient,
            Some(p) if p.remaining == 0 => TxScheduleResult::Sufficient,
            Some(_) => TxScheduleResult::Pending,
            None => TxScheduleResult::Pending,
        }
    }
}

// ===========================================================================
// Async channel wrapper (Sui's `FundsWithdrawScheduler`)
// ===========================================================================
//
// The `EagerFundsScheduler` above is the synchronous, single-threaded decision
// core. The hot path needs an async front-end: the execution scheduler submits
// consensus-ordered withdrawal batches and awaits a per-tx verdict, while the
// settlement path fires settlements that resolve previously-`Pending` txs. This
// wrapper mirrors Sui's `FundsWithdrawScheduler` (the channel-based public
// struct), with one Soma simplification: because our `EagerFundsScheduler`
// returns decided results directly (rather than holding its own oneshots), a
// SINGLE background task drains a unified event channel and owns both the inner
// scheduler and the map of not-yet-resolved `Pending` senders. Single-consumer
// = the "sequential processing" determinism guarantee (Sui doc §6) for free.

use std::sync::Arc;

use futures::stream::FuturesUnordered;
use tokio::sync::{mpsc, oneshot};

/// Seeds the scheduler with an account's settled `(balance, version)` the first
/// time it is referenced. Integration reads this from the balance store via
/// `get_balance_with_version` (the `(balance, settled_height)` CF).
pub trait AccountFundsRead: Send + Sync {
    fn balance_and_version(&self, account: &AccountKey) -> (u64, SettlementVersion);
}

/// Per-transaction verdict delivered to the caller. Mirrors Sui's
/// `ScheduleStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleStatus {
    /// Execute normally — the declared withdrawal fits.
    SufficientFunds,
    /// Fail early with `InsufficientFundsForWithdraw` (do not run the body).
    InsufficientFunds,
    /// The request's version was already settled past; no action needed
    /// (another path carried it forward).
    SkipSchedule,
}

/// A consensus-ordered batch of withdrawals sharing one request version (Sui's
/// `WithdrawReservations`).
#[derive(Debug, Clone)]
pub struct WithdrawReservations {
    /// The version the batch reads against (in Soma: the prior commit's
    /// checkpoint height — the settled baseline this commit builds on).
    pub version: SettlementVersion,
    pub withdraws: Vec<TxFundsWithdraw>,
}

/// A settlement advancing the scheduler to `next_version` with the actual
/// per-account net deltas (Sui's `FundsSettlement`). Deposits are positive,
/// actual withdrawals negative.
#[derive(Debug, Clone)]
pub struct FundsSettlement {
    pub next_version: SettlementVersion,
    pub funds_changes: BTreeMap<AccountKey, i128>,
}

type StatusSender = oneshot::Sender<(TransactionDigest, ScheduleStatus)>;

enum SchedulerEvent {
    Withdraw {
        reservations: WithdrawReservations,
        senders: BTreeMap<TransactionDigest, StatusSender>,
    },
    Settle(FundsSettlement),
}

/// The async, channel-based funds-withdraw scheduler. Cheap to clone the
/// sender; the inner state lives in one background task. Dropping the last
/// handle closes the channel and the task exits (epoch teardown).
#[derive(Clone)]
pub struct FundsWithdrawScheduler {
    event_sender: mpsc::UnboundedSender<SchedulerEvent>,
}

impl FundsWithdrawScheduler {
    /// Spawn the background task with a fresh `EagerFundsScheduler` seeded at
    /// `starting_version` (the epoch-start settled height).
    pub fn new(funds_read: Arc<dyn AccountFundsRead>, starting_version: SettlementVersion) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let inner = EagerFundsScheduler::new(starting_version);
        tokio::spawn(Self::run(inner, funds_read, event_receiver));
        Self { event_sender }
    }

    /// Submit a consensus-ordered batch. Returns one oneshot receiver per tx,
    /// each resolving to that tx's `(digest, ScheduleStatus)` — immediately for
    /// decided txs, or later (when settlement resolves them) for `Pending` ones.
    pub fn schedule_withdraws(
        &self,
        reservations: WithdrawReservations,
    ) -> FuturesUnordered<oneshot::Receiver<(TransactionDigest, ScheduleStatus)>> {
        let mut senders = BTreeMap::new();
        let receivers = FuturesUnordered::new();
        for tx in &reservations.withdraws {
            let (s, r) = oneshot::channel();
            senders.insert(tx.tx_digest, s);
            receivers.push(r);
        }
        if let Err(e) = self.event_sender.send(SchedulerEvent::Withdraw { reservations, senders }) {
            tracing::error!("funds withdraw scheduler closed, dropping batch: {e}");
        }
        receivers
    }

    /// Fire-and-forget a settlement. Resolves any `Pending` txs the new version
    /// makes decidable, releases settled reservations, and applies actual
    /// deltas. Mirrors Sui's `settle_funds`.
    pub fn settle_funds(&self, settlement: FundsSettlement) {
        if let Err(e) = self.event_sender.send(SchedulerEvent::Settle(settlement)) {
            tracing::error!("funds withdraw scheduler closed, dropping settlement: {e}");
        }
    }

    async fn run(
        mut inner: EagerFundsScheduler,
        funds_read: Arc<dyn AccountFundsRead>,
        mut rx: mpsc::UnboundedReceiver<SchedulerEvent>,
    ) {
        // Senders for txs the core returned `Pending`; resolved on a later settle.
        let mut pending: BTreeMap<TransactionDigest, StatusSender> = BTreeMap::new();
        while let Some(event) = rx.recv().await {
            match event {
                SchedulerEvent::Withdraw { reservations, mut senders } => {
                    let results = inner.schedule_batch(
                        reservations.version,
                        &reservations.withdraws,
                        |acct| funds_read.balance_and_version(&acct),
                    );
                    for (digest, result) in results {
                        let Some(sender) = senders.remove(&digest) else { continue };
                        match result {
                            TxScheduleResult::Sufficient => {
                                let _ = sender.send((digest, ScheduleStatus::SufficientFunds));
                            }
                            TxScheduleResult::Insufficient => {
                                let _ = sender.send((digest, ScheduleStatus::InsufficientFunds));
                            }
                            TxScheduleResult::Skip => {
                                let _ = sender.send((digest, ScheduleStatus::SkipSchedule));
                            }
                            // Undecided — hold the sender until settlement resolves it.
                            TxScheduleResult::Pending => {
                                pending.insert(digest, sender);
                            }
                        }
                    }
                }
                SchedulerEvent::Settle(settlement) => {
                    let decided = inner.settle(settlement.next_version, &settlement.funds_changes);
                    for (digest, result) in decided {
                        let Some(sender) = pending.remove(&digest) else { continue };
                        let status = match result {
                            TxScheduleResult::Sufficient => ScheduleStatus::SufficientFunds,
                            TxScheduleResult::Insufficient => ScheduleStatus::InsufficientFunds,
                            TxScheduleResult::Skip => ScheduleStatus::SkipSchedule,
                            // settle() only ever finalizes Sufficient/Insufficient.
                            TxScheduleResult::Pending => continue,
                        };
                        let _ = sender.send((digest, status));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn acct(seed: u8) -> AccountKey {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        (SomaAddress::from_bytes(bytes).unwrap(), CoinType::Usdc)
    }

    fn tx(digest_seed: u8, reservations: &[(AccountKey, u64)]) -> TxFundsWithdraw {
        TxFundsWithdraw {
            tx_digest: TransactionDigest::new([digest_seed; 32]),
            reservations: reservations.iter().copied().collect(),
        }
    }

    // ----- VersionGate -----

    #[test]
    fn version_gate_classifies_check_skip_wait() {
        assert_eq!(VersionGate::classify(5, 5), VersionGate::CheckNow);
        assert_eq!(VersionGate::classify(4, 5), VersionGate::Skip); // already settled past
        assert_eq!(VersionGate::classify(6, 5), VersionGate::Wait); // not settled yet
    }

    // ----- check_withdraws_at_settled_version -----

    /// Sui doc §2 worked example: order matters. Account A balance 1000;
    /// TX1 max 400, TX2 max 300, TX3 max 500 — all in consensus order.
    /// TX1, TX2 succeed (deducting 400 then 300 → 300 left); TX3 (500) fails.
    #[test]
    fn ordering_determines_outcomes() {
        let a = acct(1);
        let balances = BTreeMap::from([(a, 1000u64)]);
        let txs = vec![tx(1, &[(a, 400)]), tx(2, &[(a, 300)]), tx(3, &[(a, 500)])];
        let results = check_withdraws_at_settled_version(&balances, &txs);
        assert_eq!(
            results,
            vec![
                ScheduleResult::SufficientFunds,
                ScheduleResult::SufficientFunds,
                ScheduleResult::InsufficientFunds,
            ]
        );
    }

    /// Reordering changes the outcome (TX3 before TX2): TX3 (500) succeeds
    /// (1000→500), TX2 (300) succeeds (500→200), TX1 (400) fails (200<400).
    /// Confirms the result is order-sensitive — hence consensus order is what
    /// makes it deterministic across validators.
    #[test]
    fn reordering_changes_who_succeeds() {
        let a = acct(1);
        let balances = BTreeMap::from([(a, 1000u64)]);
        let txs = vec![tx(3, &[(a, 500)]), tx(2, &[(a, 300)]), tx(1, &[(a, 400)])];
        let results = check_withdraws_at_settled_version(&balances, &txs);
        assert_eq!(
            results,
            vec![
                ScheduleResult::SufficientFunds,
                ScheduleResult::SufficientFunds,
                ScheduleResult::InsufficientFunds,
            ]
        );
    }

    /// The cross-commit double-spend this whole module exists to stop, at the
    /// logic level: one account, balance 100, two 80-withdrawals. The first is
    /// approved (100→20), the second is rejected (20<80) — they can't BOTH
    /// withdraw 80. (The bug today is that the second reads a stale 100 from a
    /// not-yet-settled store and is wrongly approved.)
    #[test]
    fn two_withdrawals_cannot_both_overspend_one_balance() {
        let a = acct(1);
        let balances = BTreeMap::from([(a, 100u64)]);
        let txs = vec![tx(1, &[(a, 80)]), tx(2, &[(a, 80)])];
        let results = check_withdraws_at_settled_version(&balances, &txs);
        assert_eq!(
            results,
            vec![ScheduleResult::SufficientFunds, ScheduleResult::InsufficientFunds]
        );
    }

    /// Multi-account tx fails as a whole if ANY account is short, and a failed
    /// tx deducts nothing (Sui doc §5.3). TX1 needs A:100 (ok) and B:600
    /// (B only has 500) → fail; A's 100 must NOT have been deducted, so TX2
    /// (A:900) still succeeds against the full 1000.
    #[test]
    fn multi_account_failure_does_not_partially_deduct() {
        let a = acct(1);
        let b = acct(2);
        let balances = BTreeMap::from([(a, 1000u64), (b, 500u64)]);
        let txs = vec![tx(1, &[(a, 100), (b, 600)]), tx(2, &[(a, 900)])];
        let results = check_withdraws_at_settled_version(&balances, &txs);
        assert_eq!(
            results,
            vec![ScheduleResult::InsufficientFunds, ScheduleResult::SufficientFunds]
        );
    }

    /// A multi-account tx that fits on every account succeeds and deducts each.
    #[test]
    fn multi_account_success_deducts_all() {
        let a = acct(1);
        let b = acct(2);
        let balances = BTreeMap::from([(a, 1000u64), (b, 500u64)]);
        let txs = vec![tx(1, &[(a, 100), (b, 200)]), tx(2, &[(b, 350)])];
        let results = check_withdraws_at_settled_version(&balances, &txs);
        // TX1 deducts B→300; TX2 needs 350 from B (300 left) → insufficient.
        assert_eq!(
            results,
            vec![ScheduleResult::SufficientFunds, ScheduleResult::InsufficientFunds]
        );
    }

    /// Missing account entry is treated as zero balance.
    #[test]
    fn missing_account_is_zero() {
        let a = acct(1);
        let balances = BTreeMap::new();
        let txs = vec![tx(1, &[(a, 1)])];
        assert_eq!(
            check_withdraws_at_settled_version(&balances, &txs),
            vec![ScheduleResult::InsufficientFunds]
        );
    }

    fn digest(seed: u8) -> TransactionDigest {
        TransactionDigest::new([seed; 32])
    }

    // ----- AccountState (eager per-account state machine) -----

    /// Fast path (Sui §3.4 outcome 1): a withdrawal that fits the conservative
    /// bound is reserved immediately, no waiting.
    #[test]
    fn account_reserves_immediately_when_it_fits() {
        let mut s = AccountState::new(1000, 5);
        assert_eq!(s.reserve(5, 400, digest(1)), vec![AccountOutcome::Reserved(digest(1))]);
        // Second reservation stacks against the running reserved total.
        assert_eq!(s.reserve(5, 300, digest(2)), vec![AccountOutcome::Reserved(digest(2))]);
        // 400 + 300 + 400 = 1100 > 1000, and version 5 is settled → final fail.
        assert_eq!(s.reserve(5, 400, digest(3)), vec![AccountOutcome::Insufficient(digest(3))]);
    }

    /// Sui doc §5.2: a deposit unblocks a pending withdrawal. Balance 100 at
    /// v5; TX1 wants 200 at v6 → can't reserve, v6 not settled → pending (no
    /// outcome yet). Settlement v5→v6 deposits +150 → balance 250 → TX1
    /// drains to Reserved.
    #[test]
    fn account_pending_withdrawal_unblocked_by_deposit() {
        let mut s = AccountState::new(100, 5);
        // 0 + 200 > 100, version 6 > settled 5 → uncertain → wait (no outcome).
        assert_eq!(s.reserve(6, 200, digest(1)), vec![]);
        // Settlement v5→v6 brings a +150 deposit.
        assert_eq!(s.settle(6, 150), vec![AccountOutcome::Reserved(digest(1))]);
        assert_eq!(s.balance, 250);
    }

    /// Sui doc §5.2 negative: if no deposit arrives and the version settles,
    /// the pending withdrawal becomes a deterministic Insufficient.
    #[test]
    fn account_pending_withdrawal_fails_when_version_settles_short() {
        let mut s = AccountState::new(100, 5);
        assert_eq!(s.reserve(6, 200, digest(1)), vec![]);
        // Settlement v5→v6 with no helpful deposit (delta 0) → 200 > 100, and
        // now version 6 is settled → deterministic Insufficient.
        assert_eq!(s.settle(6, 0), vec![AccountOutcome::Insufficient(digest(1))]);
    }

    /// Sui doc §5.4: queued withdrawals across versions with FIFO head-of-line
    /// blocking. Balance 500 at v5. TX1(400@v5) reserves. TX2(200@v6) can't
    /// (600>500), v6 unsettled → pending. TX3(50@v6) WOULD fit (450≤500) but
    /// is blocked behind TX2 (FIFO). Settlement v5→v6 delta −400 (TX1's actual
    /// withdrawal): balance 100, reserved released → TX2 needs 200>100 at
    /// settled v6 → Insufficient; TX3 needs 50≤100 → Reserved.
    #[test]
    fn account_fifo_blocking_and_settlement_drain() {
        let mut s = AccountState::new(500, 5);
        assert_eq!(s.reserve(5, 400, digest(1)), vec![AccountOutcome::Reserved(digest(1))]);
        assert_eq!(s.reserve(6, 200, digest(2)), vec![]); // pending
        assert_eq!(s.reserve(6, 50, digest(3)), vec![]); // blocked behind TX2 despite fitting
        // Settlement v5→v6: TX1 actually withdrew 400.
        let drained = s.settle(6, -400);
        assert_eq!(
            drained,
            vec![AccountOutcome::Insufficient(digest(2)), AccountOutcome::Reserved(digest(3))]
        );
        assert_eq!(s.balance, 100);
        assert!(!s.is_empty()); // TX3 still reserved
    }

    /// Settlement idempotency (Sui §4): a re-delivered settlement at or below
    /// the current settled version is ignored (no double-application).
    #[test]
    fn account_settlement_is_idempotent() {
        let mut s = AccountState::new(100, 5);
        assert_eq!(s.settle(6, -40), vec![]);
        assert_eq!(s.balance, 60);
        // Replay of v6 (and an older v5) must not re-apply the delta.
        assert_eq!(s.settle(6, -40), vec![]);
        assert_eq!(s.settle(5, -40), vec![]);
        assert_eq!(s.balance, 60);
    }

    /// Over-reservation is corrected by settlement (Sui §3.2 / §5.1): reserve
    /// the MAX (400) but actual withdrawal is smaller (300); after settlement
    /// the freed 100 is available again.
    #[test]
    fn account_over_reservation_freed_by_actual_delta() {
        let mut s = AccountState::new(1000, 5);
        s.reserve(5, 400, digest(1)); // reserve max 400 → effective available 600
        s.reserve(5, 300, digest(2)); // reserve max 300 → effective available 300
        // A 400 withdrawal at v6 doesn't fit (700+400 > 1000), wait.
        assert_eq!(s.reserve(6, 400, digest(3)), vec![]);
        // Settlement v5→v6: actual total withdrawn was only 500 (not 700).
        // balance 1000-500=500; reserved released → TX3 needs 400 ≤ 500 → Reserved.
        assert_eq!(s.settle(6, -500), vec![AccountOutcome::Reserved(digest(3))]);
    }

    // ----- EagerFundsScheduler (multi-account coordinator) -----

    /// Sui doc §5.1: basic eager scheduling across two accounts, no waiting.
    /// Alice 1000, Bob 500. TX1 max {A:400, B:200}; TX2 max {A:300}. Both
    /// approved immediately.
    #[test]
    fn coordinator_basic_multi_account_success() {
        let a = acct(1);
        let b = acct(2);
        let seed = |k: AccountKey| if k == a { (1000, 5) } else { (500, 5) };
        let mut sched = EagerFundsScheduler::new(5);

        let txs = vec![tx(1, &[(a, 400), (b, 200)]), tx(2, &[(a, 300)])];
        let res = sched.schedule_batch(5, &txs, seed);
        assert_eq!(
            res,
            vec![
                (digest(1), TxScheduleResult::Sufficient),
                (digest(2), TxScheduleResult::Sufficient),
            ]
        );
    }

    /// A multi-account tx fails as a whole if ANY account is short, and a
    /// later single-account tx still sees the failed tx's *succeeding* account
    /// reservation held (Sui §5.3). Alice 1000, Bob 100. TX1{A:200,B:300}:
    /// A reserves 200, B is short (300>100) at settled v5 → TX1 Insufficient.
    /// TX2{A:900}: A already has 200 reserved → 200+900>1000 → Insufficient.
    #[test]
    fn coordinator_multi_account_failure_holds_succeeding_reservation() {
        let a = acct(1);
        let b = acct(2);
        let seed = |k: AccountKey| if k == a { (1000, 5) } else { (100, 5) };
        let mut sched = EagerFundsScheduler::new(5);

        let txs = vec![tx(1, &[(a, 200), (b, 300)]), tx(2, &[(a, 900)])];
        let res = sched.schedule_batch(5, &txs, seed);
        assert_eq!(
            res,
            vec![
                (digest(1), TxScheduleResult::Insufficient),
                (digest(2), TxScheduleResult::Insufficient),
            ]
        );
    }

    /// Cross-version pending resolution: a tx that can't be decided now stays
    /// Pending, then is finalized by `settle`. Alice 100 at v5. TX1{A:200}@v6
    /// → Pending (200>100, v6 not settled). settle v5→v6 deposits +150 →
    /// TX1 becomes Sufficient.
    #[test]
    fn coordinator_pending_then_settled_sufficient() {
        let a = acct(1);
        let seed = |_k: AccountKey| (100, 5);
        let mut sched = EagerFundsScheduler::new(5);

        let res = sched.schedule_batch(6, &[tx(1, &[(a, 200)])], seed);
        assert_eq!(res, vec![(digest(1), TxScheduleResult::Pending)]);

        let decided = sched.settle(6, &BTreeMap::from([(a, 150i128)]));
        assert_eq!(decided, vec![(digest(1), TxScheduleResult::Sufficient)]);
    }

    /// Same as above but no deposit arrives → settling the version makes the
    /// pending tx a deterministic Insufficient.
    #[test]
    fn coordinator_pending_then_settled_insufficient() {
        let a = acct(1);
        let seed = |_k: AccountKey| (100, 5);
        let mut sched = EagerFundsScheduler::new(5);

        assert_eq!(
            sched.schedule_batch(6, &[tx(1, &[(a, 200)])], seed),
            vec![(digest(1), TxScheduleResult::Pending)]
        );
        let decided = sched.settle(6, &BTreeMap::from([(a, 0i128)]));
        assert_eq!(decided, vec![(digest(1), TxScheduleResult::Insufficient)]);
    }

    /// Version gate: a batch older than the settled version is skipped.
    #[test]
    fn coordinator_skips_already_settled_version() {
        let a = acct(1);
        let seed = |_k: AccountKey| (1000, 10);
        let mut sched = EagerFundsScheduler::new(10);
        let res = sched.schedule_batch(7, &[tx(1, &[(a, 100)])], seed);
        assert_eq!(res, vec![(digest(1), TxScheduleResult::Skip)]);
    }

    /// The cross-commit double-spend, end to end through the coordinator: an
    /// 80 withdrawal lands at v5 (settled, approved), a second 80 lands at v6
    /// (not yet settled). The second can't fit (80 reserved + 80 > 100) and
    /// waits; when v5→v6 settles with the actual −80, balance is 20 → the
    /// second 80 is finalized Insufficient. They can never both spend.
    #[test]
    fn coordinator_blocks_cross_commit_double_spend() {
        let a = acct(1);
        let seed = |_k: AccountKey| (100, 5);
        let mut sched = EagerFundsScheduler::new(5);

        // Commit at v5: first 80 — approved.
        assert_eq!(
            sched.schedule_batch(5, &[tx(1, &[(a, 80)])], seed),
            vec![(digest(1), TxScheduleResult::Sufficient)]
        );
        // Commit at v6 (before v5 settles): second 80 — can't fit (80+80>100),
        // v6 unsettled → Pending (NOT wrongly approved against a stale 100).
        assert_eq!(
            sched.schedule_batch(6, &[tx(2, &[(a, 80)])], seed),
            vec![(digest(2), TxScheduleResult::Pending)]
        );
        // Settlement v5→v6 applies the first tx's actual −80 → balance 20.
        let decided = sched.settle(6, &BTreeMap::from([(a, -80i128)]));
        assert_eq!(decided, vec![(digest(2), TxScheduleResult::Insufficient)]);
    }

    /// Cross-validator determinism for the DEPOSIT-then-withdraw case (residual
    /// 1). The two validators differ only in settlement-persist timing, which
    /// shows up as a different seed `(balance, version)` and a different
    /// schedule/settle interleaving — exactly the production divergence. The
    /// final decision must be identical.
    ///
    /// Setup: account A starts at 100 @ v5. A deposit of +150 settles at v6.
    /// A withdrawal of 200 from A arrives in the commit at version 6.
    ///
    /// Validator LATE has not persisted v6 when it seeds A → (100, v5); the
    /// withdrawal can't fit and v6 is unsettled → Pending; settle(6,+150) then
    /// makes it Sufficient.
    ///
    /// Validator EARLY has already persisted v6 when it first seeds A → seed
    /// (250, v6); the withdrawal fits immediately → Sufficient, and the later
    /// settle(6) is ignored (idempotent, 6 ≤ 6). No double-count.
    #[test]
    fn interleaving_invariance_deposit_then_withdraw() {
        let a = acct(1);

        // Validator LATE: seed reflects pre-deposit state (v5).
        let mut late = EagerFundsScheduler::new(5);
        let r_late = late.schedule_batch(6, &[tx(1, &[(a, 200)])], |_k| (100, 5));
        assert_eq!(r_late, vec![(digest(1), TxScheduleResult::Pending)]);
        let d_late = late.settle(6, &BTreeMap::from([(a, 150i128)]));
        assert_eq!(d_late, vec![(digest(1), TxScheduleResult::Sufficient)]);

        // Validator EARLY: seed already reflects the v6 deposit.
        let mut early = EagerFundsScheduler::new(5);
        let r_early = early.schedule_batch(6, &[tx(1, &[(a, 200)])], |_k| (250, 6));
        assert_eq!(r_early, vec![(digest(1), TxScheduleResult::Sufficient)]);
        // The settle for v6 still arrives but must be a no-op (already absorbed
        // into the seed) — crucially NOT a second +150.
        let d_early = early.settle(6, &BTreeMap::from([(a, 150i128)]));
        assert_eq!(d_early, vec![]);

        // Both validators reach the SAME final decision for tx1: Sufficient.
        // (LATE via Pending→settle, EARLY via immediate.)
    }

    /// Cross-validator determinism for the pure WITHDRAWAL-overspend case. Two
    /// 80-withdrawals against a 100 balance in consecutive commits; whatever the
    /// seed/settle timing, exactly one succeeds and one fails on every validator.
    #[test]
    fn interleaving_invariance_withdrawal_overspend() {
        let a = acct(1);

        // Validator LATE: v5 not settled when the v6 tx is seen.
        let mut late = EagerFundsScheduler::new(5);
        assert_eq!(
            late.schedule_batch(5, &[tx(1, &[(a, 80)])], |_k| (100, 5)),
            vec![(digest(1), TxScheduleResult::Sufficient)]
        );
        assert_eq!(
            late.schedule_batch(6, &[tx(2, &[(a, 80)])], |_k| (100, 5)),
            vec![(digest(2), TxScheduleResult::Pending)]
        );
        assert_eq!(
            late.settle(6, &BTreeMap::from([(a, -80i128)])),
            vec![(digest(2), TxScheduleResult::Insufficient)]
        );

        // Validator EARLY: v5's settlement (−80 → balance 20) already persisted
        // before the v6 tx is seen, so A is seeded (20, v6). The first tx is
        // Skip (its version v5 < settled 6 in EARLY's view is NOT the case here
        // — EARLY still scheduled tx1 at v5 first). Model EARLY as: tx1 at v5
        // Sufficient, then settles, THEN tx2 seen with the post-settlement view.
        let mut early = EagerFundsScheduler::new(5);
        assert_eq!(
            early.schedule_batch(5, &[tx(1, &[(a, 80)])], |_k| (100, 5)),
            vec![(digest(1), TxScheduleResult::Sufficient)]
        );
        // v5 settles first on EARLY (actual −80 → 20), advancing to v6.
        assert_eq!(early.settle(6, &BTreeMap::from([(a, -80i128)])), vec![]);
        // Now tx2 at v6 is seen against the settled state: 80 > 20, v6 settled
        // → deterministic Insufficient.
        assert_eq!(
            early.schedule_batch(6, &[tx(2, &[(a, 80)])], |_k| (20, 6)),
            vec![(digest(2), TxScheduleResult::Insufficient)]
        );

        // Both: tx1 Sufficient, tx2 Insufficient — identical outcome.
    }

    /// Determinism: identical inputs → identical outputs across repeated runs.
    #[test]
    fn deterministic_across_runs() {
        let a = acct(1);
        let b = acct(2);
        let balances = BTreeMap::from([(a, 1000u64), (b, 500u64)]);
        let txs = vec![tx(1, &[(a, 400), (b, 200)]), tx(2, &[(a, 700)]), tx(3, &[(b, 400)])];
        let first = check_withdraws_at_settled_version(&balances, &txs);
        for _ in 0..50 {
            assert_eq!(check_withdraws_at_settled_version(&balances, &txs), first);
        }
    }

    // ----- seed-version-behind-global fix -----

    /// An account whose store version is BEHIND the scheduler's global settled
    /// version must be seeded at the global version, not the (stale) store
    /// version — otherwise the catch-up settle was already processed and the
    /// account would wait Pending forever. Here the scheduler is already at
    /// global v8; an untracked account seeds at store (50, v3) but is treated as
    /// settled at v8, so a short withdrawal at v8 is a deterministic Insufficient
    /// (not a stuck Pending).
    #[test]
    fn seed_raised_to_global_settled_version() {
        let a = acct(1);
        let mut sched = EagerFundsScheduler::new(8);
        // store says (50, v3): nothing settled in (3, 8] touched this account.
        let res = sched.schedule_batch(8, &[tx(1, &[(a, 100)])], |_k| (50, 3));
        // 100 > 50 and v8 is the settled global → final Insufficient, NOT Pending.
        assert_eq!(res, vec![(digest(1), TxScheduleResult::Insufficient)]);
    }

    // ----- async FundsWithdrawScheduler wrapper -----

    struct MapFundsRead(BTreeMap<AccountKey, (u64, SettlementVersion)>);
    impl AccountFundsRead for MapFundsRead {
        fn balance_and_version(&self, account: &AccountKey) -> (u64, SettlementVersion) {
            self.0.get(account).copied().unwrap_or((0, 0))
        }
    }

    async fn recv_all(
        mut receivers: FuturesUnordered<oneshot::Receiver<(TransactionDigest, ScheduleStatus)>>,
    ) -> BTreeMap<TransactionDigest, ScheduleStatus> {
        use futures::StreamExt;
        let mut out = BTreeMap::new();
        while let Some(r) = receivers.next().await {
            let (d, s) = r.unwrap();
            out.insert(d, s);
        }
        out
    }

    #[tokio::test]
    async fn wrapper_immediate_sufficient_and_insufficient() {
        let a = acct(1);
        let read = Arc::new(MapFundsRead(BTreeMap::from([(a, (100u64, 5u64))])));
        let sched = FundsWithdrawScheduler::new(read, 5);

        // Batch at settled version 5: tx1 (80) fits, tx2 (80) does not (80
        // reserved + 80 > 100) and v5 is settled → Insufficient.
        let receivers = sched.schedule_withdraws(WithdrawReservations {
            version: 5,
            withdraws: vec![tx(1, &[(a, 80)]), tx(2, &[(a, 80)])],
        });
        let out = recv_all(receivers).await;
        assert_eq!(out[&digest(1)], ScheduleStatus::SufficientFunds);
        assert_eq!(out[&digest(2)], ScheduleStatus::InsufficientFunds);
    }

    #[tokio::test]
    async fn wrapper_pending_resolved_by_settlement_deposit() {
        let a = acct(1);
        let read = Arc::new(MapFundsRead(BTreeMap::from([(a, (100u64, 5u64))])));
        let sched = FundsWithdrawScheduler::new(read, 5);

        // Withdrawal of 200 at unsettled v6 → Pending (held, not yet delivered).
        let receivers = sched.schedule_withdraws(WithdrawReservations {
            version: 6,
            withdraws: vec![tx(1, &[(a, 200)])],
        });
        // A deposit of +150 settles v6 → balance 250 → tx1 becomes Sufficient.
        sched.settle_funds(FundsSettlement {
            next_version: 6,
            funds_changes: BTreeMap::from([(a, 150i128)]),
        });
        let out = recv_all(receivers).await;
        assert_eq!(out[&digest(1)], ScheduleStatus::SufficientFunds);
    }

    #[tokio::test]
    async fn wrapper_pending_resolved_insufficient_when_no_deposit() {
        let a = acct(1);
        let read = Arc::new(MapFundsRead(BTreeMap::from([(a, (100u64, 5u64))])));
        let sched = FundsWithdrawScheduler::new(read, 5);

        let receivers = sched.schedule_withdraws(WithdrawReservations {
            version: 6,
            withdraws: vec![tx(1, &[(a, 200)])],
        });
        // Settle v6 with no deposit → balance final at 100 < 200 → Insufficient.
        sched.settle_funds(FundsSettlement { next_version: 6, funds_changes: BTreeMap::new() });
        let out = recv_all(receivers).await;
        assert_eq!(out[&digest(1)], ScheduleStatus::InsufficientFunds);
    }
}
