// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! # Account-balance accumulator types
//!
//! Soma uses an account-balance model for fungible tokens (USDC, SOMA)
//! instead of separate `Coin<T>` objects. Each `(SomaAddress, CoinType)`
//! pair has a balance tracked in a validator-side table; user
//! transactions don't directly mutate it. Instead they emit
//! [`BalanceEvent`]s during execution and the validator aggregates these
//! at consensus-commit time into a settlement transaction that applies
//! the deltas atomically.
//!
//! See SIP-58 (Sui Address Balances) for the design pattern this is
//! adapted from. Soma's variant differs by:
//!
//! - Storing balances in a separate RocksDB column family (not as
//!   dynamic fields on a root object) — direct O(1) lookup.
//! - Removing fastpath entirely; settlement always runs at consensus
//!   commit.
//! - No backward-compat layer for `Coin<T>` objects (Soma's pre-mainnet
//!   migration; coins are deleted in Stage 13).
//!
//! ## Lifecycle of a balance change
//!
//! 1. Tx declares a [`WithdrawalReservation`] for funds it will spend.
//!    Multiple reservations per tx are allowed (e.g., multi-currency).
//! 2. Scheduler sums all reservations against `(owner, coin_type)` for
//!    the current commit and verifies `sum ≤ current_balance`. Drops
//!    underfunded txs deterministically before execution.
//! 3. Executor receives the reservations and emits [`BalanceEvent`]s
//!    (`Withdraw` for the consumed amounts, `Deposit` for credits to
//!    recipients). Events are collected on the per-tx
//!    `TransactionEffects` rather than mutating live state.
//! 4. At consensus-commit construction, validators aggregate all events
//!    via [`aggregate_events`] and produce a single settlement
//!    transaction whose effect is to apply the net deltas to the
//!    balance table.
//!
//! ## Why events, not direct mutation
//!
//! Withdrawals from a single balance from many parallel txs would
//! otherwise serialize through that balance as a shared resource.
//! Emitting events instead lets the txs run in parallel — the math is
//! commutative within a commit, so the order of event emission doesn't
//! matter, only the sum at settlement time.
//!
//! See `aggregate_events` for the merge function and the unit tests at
//! the bottom of this file for invariants.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::object::CoinType;

/// A single balance change emitted by a transaction during execution.
///
/// Events are NOT applied to the balance table directly. They are
/// collected on `TransactionEffects::balance_changes` (added in a later
/// stage) and aggregated at consensus-commit construction into a
/// settlement transaction.
///
/// Both variants carry the absolute `amount` (always non-negative).
/// Use [`Self::signed_delta`] to get the signed contribution to the
/// owner's balance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BalanceEvent {
    /// `owner`'s `coin_type` balance increases by `amount`.
    Deposit { owner: SomaAddress, coin_type: CoinType, amount: u64 },
    /// `owner`'s `coin_type` balance decreases by `amount`.
    /// The withdrawal must have been pre-authorized by a
    /// [`WithdrawalReservation`] on the originating transaction; the
    /// scheduler ensures sufficient balance before execution.
    Withdraw { owner: SomaAddress, coin_type: CoinType, amount: u64 },
}

impl BalanceEvent {
    pub const fn deposit(owner: SomaAddress, coin_type: CoinType, amount: u64) -> Self {
        Self::Deposit { owner, coin_type, amount }
    }

    pub const fn withdraw(owner: SomaAddress, coin_type: CoinType, amount: u64) -> Self {
        Self::Withdraw { owner, coin_type, amount }
    }

    pub fn owner(&self) -> SomaAddress {
        match self {
            Self::Deposit { owner, .. } | Self::Withdraw { owner, .. } => *owner,
        }
    }

    pub fn coin_type(&self) -> CoinType {
        match self {
            Self::Deposit { coin_type, .. } | Self::Withdraw { coin_type, .. } => *coin_type,
        }
    }

    pub fn amount(&self) -> u64 {
        match self {
            Self::Deposit { amount, .. } | Self::Withdraw { amount, .. } => *amount,
        }
    }

    /// Signed delta this event contributes to its owner's balance.
    ///
    /// `i128` is wide enough that aggregating any realistic mix of
    /// `u64` deposits and withdrawals cannot overflow during summation
    /// (would require ~2^63 events with maximum-value amounts).
    pub fn signed_delta(&self) -> i128 {
        match self {
            Self::Deposit { amount, .. } => *amount as i128,
            Self::Withdraw { amount, .. } => -(*amount as i128),
        }
    }

    /// Key used for aggregation: `(owner, coin_type)`. Two events with
    /// the same key combine into a single delta.
    pub fn aggregation_key(&self) -> (SomaAddress, CoinType) {
        (self.owner(), self.coin_type())
    }
}

/// A pre-execution declaration that a transaction will withdraw at most
/// `amount` of `coin_type` from `owner`'s balance. Attached to the
/// transaction at construction; the scheduler verifies the reservation
/// against the current balance before execution.
///
/// Multiple reservations per tx are allowed and may target different
/// owners or coin types (cross-currency atomicity). The scheduler
/// requires *all* reservations on a tx to be satisfiable; if any one
/// fails the entire tx is dropped from the commit.
///
/// ## Upper-bound semantics (important for executor authors)
///
/// `amount` is the **maximum** the executor may withdraw, not a
/// commitment to withdraw exactly that. The executor is free to emit
/// a [`BalanceEvent::Withdraw`] with any value `≤ amount` (or no
/// withdraw event at all). Anything not consumed automatically returns
/// to the owner's available balance after the commit.
///
/// This differs from Sui's `Withdrawal<T>` hot-potato capability,
/// which Move enforces must be redeemed within the tx. Soma's
/// executor is in-tree Rust code, so the type system is not load-
/// bearing for safety here. Instead, executor authors must uphold
/// **the conservation invariant**: for non-system, non-minting txs,
/// `sum(emitted Deposits) ≤ sum(emitted Withdraws ≤ reserved amount)`
/// per `(owner, coin_type)`. Mints (bridge deposits, staking rewards,
/// genesis) may emit a Deposit without a matching Withdraw. Stage 2
/// will add per-tx execution-time validation of this invariant; for
/// now it is a documented contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WithdrawalReservation {
    pub owner: SomaAddress,
    pub coin_type: CoinType,
    pub amount: u64,
}

impl WithdrawalReservation {
    pub const fn new(owner: SomaAddress, coin_type: CoinType, amount: u64) -> Self {
        Self { owner, coin_type, amount }
    }

    /// Convenience: the `(owner, coin_type)` pair the scheduler keys
    /// reservations on for its underflow check.
    pub fn aggregation_key(&self) -> (SomaAddress, CoinType) {
        (self.owner, self.coin_type)
    }
}

/// Aggregate a slice of `BalanceEvent`s into a deterministic map of net
/// signed deltas keyed by `(owner, coin_type)`.
///
/// Order-independent: any permutation of `events` produces the same
/// output. This is the property that lets parallel-executing txs emit
/// events freely; only the sum matters at settlement time.
///
/// Returns `i128` values to permit later checks against the underlying
/// `u64` balances (e.g., applying a delta of `-50` to a current
/// balance of `30` is invalid; settlement must catch this — see
/// [`apply_delta_to_balance`]).
///
/// Uses `BTreeMap` (not `HashMap`) so iteration order over the result
/// is deterministic, which matters for cross-validator state-hash
/// consistency.
pub fn aggregate_events(events: &[BalanceEvent]) -> BTreeMap<(SomaAddress, CoinType), i128> {
    let mut map: BTreeMap<(SomaAddress, CoinType), i128> = BTreeMap::new();
    for ev in events {
        let entry = map.entry(ev.aggregation_key()).or_insert(0);
        *entry += ev.signed_delta();
    }
    map
}

/// Convert an aggregated delta map back to a sorted, zero-filtered list
/// of `BalanceEvent`s — the form the per-commit settlement transaction
/// carries on chain.
///
/// - **Net positive** delta → `Deposit` of magnitude.
/// - **Net negative** delta → `Withdraw` of magnitude.
/// - **Zero** delta → dropped entirely (no on-chain entry).
///
/// Magnitudes are converted from `i128` to `u64` via saturating cast.
/// Total supply fits well under `u64::MAX`, so a saturated value
/// indicates a settlement-pipeline bug; we return an error so callers
/// can surface it loudly rather than silently truncating.
///
/// Output ordering follows `BTreeMap` iteration order over
/// `(owner, coin_type)`, which makes the resulting `Vec` deterministic
/// across validators — critical for matching settlement-tx digests.
pub fn aggregated_events_to_settlement_changes(
    aggregated: BTreeMap<(SomaAddress, CoinType), i128>,
) -> Result<Vec<BalanceEvent>, AggregationError> {
    let mut out = Vec::with_capacity(aggregated.len());
    for ((owner, coin_type), delta) in aggregated {
        if delta == 0 {
            continue;
        }
        let magnitude_u128 = delta.unsigned_abs();
        if magnitude_u128 > u64::MAX as u128 {
            return Err(AggregationError::DeltaOverflow { owner, coin_type, delta });
        }
        let magnitude = magnitude_u128 as u64;
        let event = if delta > 0 {
            BalanceEvent::deposit(owner, coin_type, magnitude)
        } else {
            BalanceEvent::withdraw(owner, coin_type, magnitude)
        };
        out.push(event);
    }
    Ok(out)
}

/// Errors that can arise while compacting events into a settlement form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationError {
    /// A net delta exceeded `u64::MAX` in magnitude. Should never
    /// happen in practice — total supply is far below this — so it
    /// indicates a pipeline bug or memory corruption.
    DeltaOverflow { owner: SomaAddress, coin_type: CoinType, delta: i128 },
}

/// Outcome of the scheduler's reservation pre-pass for a single
/// transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationDecision {
    /// All of the tx's reservations fit within the running tentative
    /// balances — proceed to execute.
    Accept,
    /// At least one reservation underflowed; the tx is dropped from
    /// the commit deterministically. The first failing key is
    /// recorded for diagnostics.
    Drop { reason: ReservationFailure },
}

/// Why a reservation pre-pass dropped a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationFailure {
    /// `available` running balance was less than `requested` for the
    /// `(owner, coin_type)` pair. `available` reflects the balance
    /// **after** accumulating all earlier-accepted txs in the same
    /// commit; the underlying chain-state balance may be higher.
    InsufficientBalance {
        owner: SomaAddress,
        coin_type: CoinType,
        requested: u64,
        available: u64,
    },
    /// Two or more reservations on the *same* tx targeting the same
    /// `(owner, coin_type)` summed to more than `u64::MAX`. Pure
    /// pathological-input defense; will never happen with realistic
    /// reservations (total supply ≪ u64::MAX).
    IntraTxOverflow { owner: SomaAddress, coin_type: CoinType },
}

/// Check a commit's transaction reservations in order against an
/// initial balance oracle, returning a per-tx decision.
///
/// This is the parallel-safety primitive of the accumulator system.
/// Without it, two transactions from the same address could each
/// pass execution and the second would underflow at settlement —
/// breaking the no-account-locking-but-still-safe property that
/// SIP-58 derives its parallelism from. Sui's scheduler runs an
/// equivalent pre-pass before dispatching txs to executors.
///
/// ## Determinism
///
/// All validators must agree on which txs are dropped, so this
/// function is **strictly deterministic** in its inputs:
///
/// 1. `txs` is the consensus-ordered tx list — caller is responsible
///    for the canonical ordering.
/// 2. `balance_at` is consulted **once per `(owner, coin_type)` key**
///    encountered, then cached internally. Callers may freely
///    implement it as a DB read; cross-validator agreement is
///    guaranteed because every validator reads the same pre-commit
///    state.
/// 3. The intra-tx reservation list is processed in slice order. If
///    two reservations on the same tx target the same key, they sum.
///
/// ## Upper-bound semantics
///
/// A reservation declares the *maximum* a tx will withdraw. The
/// scheduler treats it as the actual debit during the pre-pass —
/// reserving the worst case. If the executor ends up withdrawing
/// less, the unconsumed reservation simply isn't reflected as a
/// `BalanceEvent::Withdraw` and the unused amount stays in the
/// account at settlement. This is conservative-correct: never
/// over-credits, may slightly under-utilize parallelism.
///
/// ## Returns
///
/// `Vec<ReservationDecision>` of the same length as `txs`. Entry `i`
/// is the decision for `txs[i]`. Accepted txs have their reservations
/// debited from the running tentative balances; dropped txs leave
/// the tentatives untouched.
pub fn check_reservations<F>(
    txs: &[&[WithdrawalReservation]],
    mut balance_at: F,
) -> Vec<ReservationDecision>
where
    F: FnMut(SomaAddress, CoinType) -> u64,
{
    // Tentative running balances per (owner, coin_type) for the
    // commit. Populated lazily from `balance_at` on first access.
    let mut running: BTreeMap<(SomaAddress, CoinType), u64> = BTreeMap::new();
    let mut decisions = Vec::with_capacity(txs.len());

    'tx: for reservations in txs {
        if reservations.is_empty() {
            decisions.push(ReservationDecision::Accept);
            continue;
        }

        // Sum reservations within this tx by key first. Multi-key
        // txs (cross-currency, multi-account) are common; multi-
        // reservation-same-key on a single tx is rare but legal
        // (e.g., gas + value both debiting USDC from sender).
        let mut tx_total: BTreeMap<(SomaAddress, CoinType), u64> = BTreeMap::new();
        for r in *reservations {
            let entry = tx_total.entry(r.aggregation_key()).or_insert(0);
            match entry.checked_add(r.amount) {
                Some(v) => *entry = v,
                None => {
                    decisions.push(ReservationDecision::Drop {
                        reason: ReservationFailure::IntraTxOverflow {
                            owner: r.owner,
                            coin_type: r.coin_type,
                        },
                    });
                    continue 'tx;
                }
            }
        }

        // Verify each requested key against the running tentative.
        // Collect the deltas we'd apply on success — we only mutate
        // `running` after every key is verified, so a partially-
        // failing tx doesn't poison earlier-accepted balances.
        let mut to_apply: Vec<((SomaAddress, CoinType), u64)> =
            Vec::with_capacity(tx_total.len());
        for (key, requested) in &tx_total {
            let available = *running
                .entry(*key)
                .or_insert_with(|| balance_at(key.0, key.1));
            if available < *requested {
                decisions.push(ReservationDecision::Drop {
                    reason: ReservationFailure::InsufficientBalance {
                        owner: key.0,
                        coin_type: key.1,
                        requested: *requested,
                        available,
                    },
                });
                continue 'tx;
            }
            to_apply.push((*key, *requested));
        }

        // All keys cleared — debit the running tentatives.
        for (key, requested) in to_apply {
            *running.get_mut(&key).expect("key was populated above") -= requested;
        }
        decisions.push(ReservationDecision::Accept);
    }

    decisions
}

/// Result of applying a signed delta to a `u64` balance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalanceUpdate {
    /// Delta applied successfully; new balance returned.
    Ok(u64),
    /// Delta would have made the balance negative; the original balance
    /// is preserved.
    Underflow { current: u64, delta: i128 },
    /// Delta would overflow a `u64`; the original balance is preserved.
    Overflow { current: u64, delta: i128 },
}

/// Apply a signed delta to a `u64` balance, returning the new balance
/// or a structured error on under/overflow.
///
/// At settlement time the scheduler should already have prevented any
/// underflow via the reservation system, so an `Underflow` result here
/// indicates a bug or a settlement-pipeline inconsistency. Tests rely
/// on this returning a discriminated error rather than panicking.
pub fn apply_delta_to_balance(current: u64, delta: i128) -> BalanceUpdate {
    if delta >= 0 {
        let positive = delta as u128;
        match (current as u128).checked_add(positive) {
            Some(sum) if sum <= u64::MAX as u128 => BalanceUpdate::Ok(sum as u64),
            _ => BalanceUpdate::Overflow { current, delta },
        }
    } else {
        // delta < 0; check that |delta| <= current
        let to_subtract = delta.unsigned_abs(); // i128::unsigned_abs -> u128
        if to_subtract > current as u128 {
            BalanceUpdate::Underflow { current, delta }
        } else {
            BalanceUpdate::Ok(current - to_subtract as u64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(seed: u8) -> SomaAddress {
        SomaAddress::new([seed; 32])
    }

    // ---------------------------------------------------------------
    // BalanceEvent
    // ---------------------------------------------------------------

    #[test]
    fn deposit_signed_delta_is_positive() {
        let ev = BalanceEvent::deposit(addr(1), CoinType::Usdc, 100);
        assert_eq!(ev.signed_delta(), 100i128);
        assert_eq!(ev.amount(), 100);
        assert_eq!(ev.coin_type(), CoinType::Usdc);
    }

    #[test]
    fn withdraw_signed_delta_is_negative() {
        let ev = BalanceEvent::withdraw(addr(1), CoinType::Soma, 30);
        assert_eq!(ev.signed_delta(), -30i128);
        assert_eq!(ev.amount(), 30);
        assert_eq!(ev.coin_type(), CoinType::Soma);
    }

    #[test]
    fn aggregation_key_combines_owner_and_type() {
        let a = addr(1);
        let dep = BalanceEvent::deposit(a, CoinType::Usdc, 1);
        let wit = BalanceEvent::withdraw(a, CoinType::Usdc, 1);
        let other_owner = BalanceEvent::deposit(addr(2), CoinType::Usdc, 1);
        let other_type = BalanceEvent::deposit(a, CoinType::Soma, 1);

        assert_eq!(dep.aggregation_key(), wit.aggregation_key());
        assert_ne!(dep.aggregation_key(), other_owner.aggregation_key());
        assert_ne!(dep.aggregation_key(), other_type.aggregation_key());
    }

    #[test]
    fn balance_event_bcs_round_trips() {
        let ev = BalanceEvent::deposit(addr(7), CoinType::Usdc, 12345);
        let bytes = bcs::to_bytes(&ev).expect("serializes");
        let decoded: BalanceEvent = bcs::from_bytes(&bytes).expect("deserializes");
        assert_eq!(ev, decoded);
    }

    // ---------------------------------------------------------------
    // WithdrawalReservation
    // ---------------------------------------------------------------

    #[test]
    fn reservation_round_trips_via_bcs() {
        let r = WithdrawalReservation::new(addr(3), CoinType::Soma, 999_999);
        let bytes = bcs::to_bytes(&r).expect("serializes");
        let decoded: WithdrawalReservation = bcs::from_bytes(&bytes).expect("deserializes");
        assert_eq!(r, decoded);
    }

    #[test]
    fn reservation_aggregation_key_matches_event() {
        let owner = addr(5);
        let r = WithdrawalReservation::new(owner, CoinType::Usdc, 50);
        let ev = BalanceEvent::withdraw(owner, CoinType::Usdc, 50);
        assert_eq!(r.aggregation_key(), ev.aggregation_key());
    }

    // ---------------------------------------------------------------
    // aggregate_events
    // ---------------------------------------------------------------

    /// Empty input → empty map. Trivial but covers a real edge case
    /// (commits with no balance-touching txs still go through
    /// aggregation).
    #[test]
    fn aggregate_empty_returns_empty() {
        let agg = aggregate_events(&[]);
        assert!(agg.is_empty());
    }

    #[test]
    fn aggregate_sums_per_owner_and_type() {
        let alice = addr(1);
        let bob = addr(2);
        let events = [
            BalanceEvent::deposit(alice, CoinType::Usdc, 100),
            BalanceEvent::withdraw(alice, CoinType::Usdc, 30),
            BalanceEvent::deposit(bob, CoinType::Usdc, 50),
            BalanceEvent::deposit(alice, CoinType::Soma, 200),
            BalanceEvent::withdraw(alice, CoinType::Soma, 75),
        ];
        let agg = aggregate_events(&events);

        assert_eq!(agg.len(), 3);
        assert_eq!(agg.get(&(alice, CoinType::Usdc)), Some(&70i128));
        assert_eq!(agg.get(&(bob, CoinType::Usdc)), Some(&50i128));
        assert_eq!(agg.get(&(alice, CoinType::Soma)), Some(&125i128));
    }

    /// Order-independence is the key parallelism property: any
    /// permutation of events must produce the same aggregated output.
    /// This is what lets validators emit events from concurrent
    /// parallel transactions without worrying about ordering.
    #[test]
    fn aggregate_is_order_independent() {
        let alice = addr(1);
        let events_in_order = vec![
            BalanceEvent::deposit(alice, CoinType::Usdc, 100),
            BalanceEvent::withdraw(alice, CoinType::Usdc, 40),
            BalanceEvent::deposit(alice, CoinType::Usdc, 50),
            BalanceEvent::withdraw(alice, CoinType::Usdc, 10),
        ];
        let events_reversed: Vec<_> = events_in_order.iter().copied().rev().collect();
        let mut events_shuffled = events_in_order.clone();
        events_shuffled.swap(0, 3);
        events_shuffled.swap(1, 2);

        let agg_in = aggregate_events(&events_in_order);
        let agg_rev = aggregate_events(&events_reversed);
        let agg_shuf = aggregate_events(&events_shuffled);
        assert_eq!(agg_in, agg_rev);
        assert_eq!(agg_in, agg_shuf);
        assert_eq!(agg_in.get(&(alice, CoinType::Usdc)), Some(&100i128));
    }

    /// A withdrawal with no matching deposit yields a negative delta —
    /// settlement is responsible for rejecting these via the scheduler's
    /// reservation check, but `aggregate_events` itself stays
    /// order-independent and reports the negative.
    #[test]
    fn aggregate_can_produce_negative_delta() {
        let alice = addr(1);
        let events = [BalanceEvent::withdraw(alice, CoinType::Usdc, 50)];
        let agg = aggregate_events(&events);
        assert_eq!(agg.get(&(alice, CoinType::Usdc)), Some(&-50i128));
    }

    /// Cross-currency events on the same owner produce separate map
    /// entries — no accidental commingling.
    #[test]
    fn aggregate_keeps_currencies_separate() {
        let alice = addr(1);
        let events = [
            BalanceEvent::deposit(alice, CoinType::Usdc, 100),
            BalanceEvent::deposit(alice, CoinType::Soma, 100),
        ];
        let agg = aggregate_events(&events);
        assert_eq!(agg.len(), 2);
        assert_eq!(agg.get(&(alice, CoinType::Usdc)), Some(&100i128));
        assert_eq!(agg.get(&(alice, CoinType::Soma)), Some(&100i128));
    }

    /// Net-zero combinations should still appear in the result with
    /// delta 0 (rather than being absent) — leaving them in lets
    /// settlement record "this address was touched" for indexing.
    #[test]
    fn aggregate_keeps_net_zero_entries() {
        let alice = addr(1);
        let events = [
            BalanceEvent::deposit(alice, CoinType::Usdc, 100),
            BalanceEvent::withdraw(alice, CoinType::Usdc, 100),
        ];
        let agg = aggregate_events(&events);
        assert_eq!(agg.get(&(alice, CoinType::Usdc)), Some(&0i128));
    }

    /// Splitting events into two batches and aggregating each, then
    /// merging the two maps, must produce the same result as
    /// aggregating them all together. This is the parallelism /
    /// associativity property the settlement pipeline relies on
    /// (e.g., per-tx aggregation collected on `TransactionEffects`,
    /// then commit-level aggregation merging across txs). Without this
    /// property, splitting work across executors would change the
    /// outcome.
    #[test]
    fn aggregate_is_associative_across_partitions() {
        let alice = addr(1);
        let bob = addr(2);
        let all_events = vec![
            BalanceEvent::deposit(alice, CoinType::Usdc, 100),
            BalanceEvent::withdraw(alice, CoinType::Usdc, 30),
            BalanceEvent::deposit(bob, CoinType::Usdc, 50),
            BalanceEvent::deposit(alice, CoinType::Soma, 200),
        ];

        let single_pass = aggregate_events(&all_events);

        // Split into two arbitrary batches and aggregate independently
        let (batch1, batch2) = all_events.split_at(2);
        let agg1 = aggregate_events(batch1);
        let agg2 = aggregate_events(batch2);

        // Merge the two partial aggregates by summing entries
        let mut merged = agg1.clone();
        for (k, v) in agg2 {
            *merged.entry(k).or_insert(0) += v;
        }

        assert_eq!(single_pass, merged);
    }

    /// Idempotence under empty merge: aggregating, then merging with an
    /// empty aggregate, should be a no-op.
    #[test]
    fn aggregate_idempotent_under_empty_merge() {
        let alice = addr(1);
        let events = [BalanceEvent::deposit(alice, CoinType::Usdc, 42)];
        let agg = aggregate_events(&events);
        let empty = aggregate_events(&[]);

        let mut merged = agg.clone();
        for (k, v) in empty {
            *merged.entry(k).or_insert(0) += v;
        }
        assert_eq!(agg, merged);
    }

    // ---------------------------------------------------------------
    // apply_delta_to_balance
    // ---------------------------------------------------------------

    #[test]
    fn apply_positive_delta_increases_balance() {
        match apply_delta_to_balance(100, 50) {
            BalanceUpdate::Ok(b) => assert_eq!(b, 150),
            other => panic!("expected Ok(150), got {:?}", other),
        }
    }

    #[test]
    fn apply_negative_delta_decreases_balance() {
        match apply_delta_to_balance(100, -30) {
            BalanceUpdate::Ok(b) => assert_eq!(b, 70),
            other => panic!("expected Ok(70), got {:?}", other),
        }
    }

    #[test]
    fn apply_zero_delta_is_noop() {
        match apply_delta_to_balance(100, 0) {
            BalanceUpdate::Ok(b) => assert_eq!(b, 100),
            other => panic!("expected Ok(100), got {:?}", other),
        }
    }

    /// Withdrawing more than the balance is rejected, leaving the
    /// original balance untouched. This shouldn't happen in normal
    /// operation (the scheduler filters such txs) but is the
    /// defense-in-depth for settlement.
    #[test]
    fn apply_underflow_preserves_balance() {
        match apply_delta_to_balance(50, -100) {
            BalanceUpdate::Underflow { current, delta } => {
                assert_eq!(current, 50);
                assert_eq!(delta, -100i128);
            }
            other => panic!("expected Underflow, got {:?}", other),
        }
    }

    /// Edge: withdrawing exactly the balance is fine and yields zero.
    #[test]
    fn apply_exact_drain_yields_zero() {
        match apply_delta_to_balance(100, -100) {
            BalanceUpdate::Ok(b) => assert_eq!(b, 0),
            other => panic!("expected Ok(0), got {:?}", other),
        }
    }

    /// Adding `i128::MAX` (or anything that doesn't fit u64) to a
    /// non-zero balance is reported as overflow rather than panicking.
    #[test]
    fn apply_overflow_preserves_balance() {
        let huge = i128::MAX;
        match apply_delta_to_balance(1, huge) {
            BalanceUpdate::Overflow { current, delta } => {
                assert_eq!(current, 1);
                assert_eq!(delta, huge);
            }
            other => panic!("expected Overflow, got {:?}", other),
        }
    }

    /// Going from u64::MAX to overflow by +1.
    #[test]
    fn apply_max_plus_one_overflows() {
        match apply_delta_to_balance(u64::MAX, 1) {
            BalanceUpdate::Overflow { .. } => (),
            other => panic!("expected Overflow, got {:?}", other),
        }
    }

    /// Boundary: balance == u64::MAX with delta 0 stays MAX.
    #[test]
    fn apply_zero_delta_at_max_is_noop() {
        match apply_delta_to_balance(u64::MAX, 0) {
            BalanceUpdate::Ok(b) => assert_eq!(b, u64::MAX),
            other => panic!("expected Ok(MAX), got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // aggregated_events_to_settlement_changes (Stage 3)
    // ---------------------------------------------------------------

    /// The end-to-end happy path: a mixed bag of events aggregates,
    /// drops zero-net entries, and emerges as a sorted Vec of
    /// (signed) BalanceEvents.
    #[test]
    fn settlement_changes_round_trip_from_events() {
        let alice = addr(1);
        let bob = addr(2);
        let events = [
            BalanceEvent::deposit(alice, CoinType::Usdc, 100),
            BalanceEvent::withdraw(alice, CoinType::Usdc, 30),
            BalanceEvent::deposit(bob, CoinType::Usdc, 50),
            // alice's SOMA: net zero — must be dropped.
            BalanceEvent::deposit(alice, CoinType::Soma, 200),
            BalanceEvent::withdraw(alice, CoinType::Soma, 200),
        ];

        let agg = aggregate_events(&events);
        let changes =
            aggregated_events_to_settlement_changes(agg).expect("magnitudes fit in u64");

        assert_eq!(changes.len(), 2, "net-zero entries must be dropped");

        // BTreeMap iteration order: (owner, coin_type). Both entries
        // are CoinType::Usdc; ordering is by SomaAddress raw bytes,
        // which our test's `addr(1)` < `addr(2)`.
        match &changes[0] {
            BalanceEvent::Deposit { owner, coin_type, amount } => {
                assert_eq!(*owner, alice);
                assert_eq!(*coin_type, CoinType::Usdc);
                assert_eq!(*amount, 70);
            }
            _ => panic!("expected Deposit for alice/USDC, got {:?}", changes[0]),
        }
        match &changes[1] {
            BalanceEvent::Deposit { owner, coin_type, amount } => {
                assert_eq!(*owner, bob);
                assert_eq!(*coin_type, CoinType::Usdc);
                assert_eq!(*amount, 50);
            }
            _ => panic!("expected Deposit for bob/USDC, got {:?}", changes[1]),
        }
    }

    /// A net-negative aggregate must surface as `Withdraw`. This is the
    /// path the consensus handler takes for accounts that spent more
    /// than they received in a commit (typical for senders of payments).
    #[test]
    fn settlement_changes_emit_withdraw_for_net_negative() {
        let alice = addr(1);
        let events = [
            BalanceEvent::deposit(alice, CoinType::Usdc, 30),
            BalanceEvent::withdraw(alice, CoinType::Usdc, 100),
        ];
        let changes =
            aggregated_events_to_settlement_changes(aggregate_events(&events)).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0], BalanceEvent::withdraw(alice, CoinType::Usdc, 70));
    }

    /// Empty input → empty Vec. Settlement with no changes is a valid
    /// no-op transaction that we still want to inject every commit
    /// for ordering predictability.
    #[test]
    fn settlement_changes_empty_input_returns_empty() {
        let changes =
            aggregated_events_to_settlement_changes(aggregate_events(&[])).unwrap();
        assert!(changes.is_empty());
    }

    /// Magnitudes exceeding u64::MAX surface as a structured error.
    /// In practice this can't happen — total supply is far under
    /// u64::MAX — but a corrupted aggregator must not silently truncate.
    #[test]
    fn settlement_changes_reject_oversized_magnitude() {
        let alice = addr(1);
        let mut agg = BTreeMap::new();
        // i128 magnitude > u64::MAX
        agg.insert((alice, CoinType::Usdc), (u64::MAX as i128) + 1);
        let err = aggregated_events_to_settlement_changes(agg).unwrap_err();
        match err {
            AggregationError::DeltaOverflow { owner, coin_type, .. } => {
                assert_eq!(owner, alice);
                assert_eq!(coin_type, CoinType::Usdc);
            }
        }
    }

    /// Output ordering must be deterministic. Aggregation runs on
    /// every validator independently; if order varies, settlement-tx
    /// digests diverge and consensus stalls.
    #[test]
    fn settlement_changes_order_is_deterministic() {
        let alice = addr(1);
        let bob = addr(9);
        let events_a = [
            BalanceEvent::deposit(alice, CoinType::Usdc, 1),
            BalanceEvent::deposit(bob, CoinType::Usdc, 1),
        ];
        let events_b = [
            BalanceEvent::deposit(bob, CoinType::Usdc, 1),
            BalanceEvent::deposit(alice, CoinType::Usdc, 1),
        ];
        let changes_a =
            aggregated_events_to_settlement_changes(aggregate_events(&events_a)).unwrap();
        let changes_b =
            aggregated_events_to_settlement_changes(aggregate_events(&events_b)).unwrap();
        assert_eq!(
            changes_a, changes_b,
            "two permutations of the same events must yield identical settlement payloads"
        );
    }

    // ---------------------------------------------------------------
    // check_reservations (Stage 4 — scheduler pre-pass)
    // ---------------------------------------------------------------

    /// Helper: build a constant-balance oracle from a slice.
    fn oracle(
        entries: &[(SomaAddress, CoinType, u64)],
    ) -> impl FnMut(SomaAddress, CoinType) -> u64 + '_ {
        move |o, c| {
            entries
                .iter()
                .find(|(eo, ec, _)| *eo == o && *ec == c)
                .map(|(_, _, v)| *v)
                .unwrap_or(0)
        }
    }

    /// A tx with no reservations is always accepted. This is the
    /// universal opt-out — every existing tx kind today returns []
    /// from `reservations()`, so the pre-pass is a strict no-op until
    /// Stage 6 starts populating them.
    #[test]
    fn reservations_empty_tx_always_accepts() {
        let txs: &[&[WithdrawalReservation]] = &[&[], &[], &[]];
        let decisions = check_reservations(txs, oracle(&[]));
        assert_eq!(
            decisions,
            vec![
                ReservationDecision::Accept,
                ReservationDecision::Accept,
                ReservationDecision::Accept,
            ]
        );
    }

    /// Single tx, single reservation, sufficient balance: accept.
    #[test]
    fn reservations_single_tx_sufficient() {
        let alice = addr(1);
        let r = WithdrawalReservation::new(alice, CoinType::Usdc, 50);
        let decisions = check_reservations(
            &[&[r]],
            oracle(&[(alice, CoinType::Usdc, 100)]),
        );
        assert_eq!(decisions, vec![ReservationDecision::Accept]);
    }

    /// Single tx, single reservation, insufficient balance: drop.
    #[test]
    fn reservations_single_tx_insufficient_balance_drops() {
        let alice = addr(1);
        let r = WithdrawalReservation::new(alice, CoinType::Usdc, 100);
        let decisions = check_reservations(
            &[&[r]],
            oracle(&[(alice, CoinType::Usdc, 50)]),
        );
        assert_eq!(
            decisions,
            vec![ReservationDecision::Drop {
                reason: ReservationFailure::InsufficientBalance {
                    owner: alice,
                    coin_type: CoinType::Usdc,
                    requested: 100,
                    available: 50,
                }
            }]
        );
    }

    /// Two txs from the same address, second underfunded due to first's
    /// debit. This is the *core* invariant the pre-pass exists to
    /// protect: without running tentative balances, both would pass
    /// and settlement would underflow.
    #[test]
    fn reservations_running_tentative_drops_second_when_overspent() {
        let alice = addr(1);
        let r1 = WithdrawalReservation::new(alice, CoinType::Usdc, 60);
        let r2 = WithdrawalReservation::new(alice, CoinType::Usdc, 60);
        let decisions = check_reservations(
            &[&[r1], &[r2]],
            oracle(&[(alice, CoinType::Usdc, 100)]),
        );
        assert_eq!(decisions[0], ReservationDecision::Accept);
        match decisions[1] {
            ReservationDecision::Drop {
                reason: ReservationFailure::InsufficientBalance { available, requested, .. },
            } => {
                // After accepting tx 0, alice's running USDC is 100 - 60 = 40.
                assert_eq!(available, 40);
                assert_eq!(requested, 60);
            }
            other => panic!("expected drop with InsufficientBalance, got {:?}", other),
        }
    }

    /// Order matters: if the (otherwise identical) txs swap, the second
    /// is dropped instead of the first. Determinism is the property —
    /// every validator must see the same drop. This is also why the
    /// caller is responsible for producing the consensus-canonical tx
    /// order.
    #[test]
    fn reservations_drop_decision_is_position_dependent() {
        let alice = addr(1);
        let r_big = WithdrawalReservation::new(alice, CoinType::Usdc, 80);
        let r_small = WithdrawalReservation::new(alice, CoinType::Usdc, 30);

        // Big first: big accepts (80/100), small drops (20 < 30).
        let d1 = check_reservations(
            &[&[r_big], &[r_small]],
            oracle(&[(alice, CoinType::Usdc, 100)]),
        );
        assert!(matches!(d1[0], ReservationDecision::Accept));
        assert!(matches!(d1[1], ReservationDecision::Drop { .. }));

        // Small first: both fit (30 + 80 ≤ 110, but available is 100;
        // small accepts leaving 70, big drops because 70 < 80).
        let d2 = check_reservations(
            &[&[r_small], &[r_big]],
            oracle(&[(alice, CoinType::Usdc, 100)]),
        );
        assert!(matches!(d2[0], ReservationDecision::Accept));
        assert!(matches!(d2[1], ReservationDecision::Drop { .. }));
    }

    /// Cross-currency txs: failing on one currency shouldn't affect
    /// the other accumulator. Tx fails atomically (all-or-nothing).
    #[test]
    fn reservations_cross_currency_atomicity() {
        let alice = addr(1);
        // Tx wants 50 USDC AND 200 SOMA, but alice only has 100 SOMA.
        let r_usdc = WithdrawalReservation::new(alice, CoinType::Usdc, 50);
        let r_soma = WithdrawalReservation::new(alice, CoinType::Soma, 200);
        let decisions = check_reservations(
            &[&[r_usdc, r_soma]],
            oracle(&[
                (alice, CoinType::Usdc, 1_000_000),
                (alice, CoinType::Soma, 100),
            ]),
        );
        // Tx must drop because SOMA is insufficient. USDC running balance
        // must NOT have been debited.
        assert!(matches!(
            decisions[0],
            ReservationDecision::Drop {
                reason: ReservationFailure::InsufficientBalance {
                    coin_type: CoinType::Soma,
                    ..
                }
            }
        ));

        // Verify the no-poison property by following up with a USDC-only
        // tx that needs the full 1M balance — it should still pass
        // because the prior tx didn't debit USDC.
        let r_usdc_full =
            WithdrawalReservation::new(alice, CoinType::Usdc, 1_000_000);
        let decisions = check_reservations(
            &[&[r_usdc, r_soma], &[r_usdc_full]],
            oracle(&[
                (alice, CoinType::Usdc, 1_000_000),
                (alice, CoinType::Soma, 100),
            ]),
        );
        assert!(matches!(decisions[0], ReservationDecision::Drop { .. }));
        assert!(matches!(decisions[1], ReservationDecision::Accept));
    }

    /// Independent accounts: bob's drop must not affect alice's accept.
    /// This is the parallelism property — accumulator scheduling is
    /// per-key, not per-tx-batch.
    #[test]
    fn reservations_independent_accounts() {
        let alice = addr(1);
        let bob = addr(2);
        let r_alice = WithdrawalReservation::new(alice, CoinType::Usdc, 50);
        let r_bob = WithdrawalReservation::new(bob, CoinType::Usdc, 200);
        let decisions = check_reservations(
            &[&[r_alice], &[r_bob]],
            oracle(&[
                (alice, CoinType::Usdc, 100),
                (bob, CoinType::Usdc, 100),
            ]),
        );
        assert_eq!(decisions[0], ReservationDecision::Accept);
        assert!(matches!(decisions[1], ReservationDecision::Drop { .. }));
    }

    /// Multiple reservations on the same tx for the same key sum
    /// together. Realistic example post-Stage 6: gas + value both
    /// debit USDC from the sender. Combined demand must fit.
    #[test]
    fn reservations_intra_tx_same_key_sums() {
        let alice = addr(1);
        let r_gas = WithdrawalReservation::new(alice, CoinType::Usdc, 30);
        let r_value = WithdrawalReservation::new(alice, CoinType::Usdc, 60);
        // 30 + 60 = 90 ≤ 100 → accept.
        let d_ok = check_reservations(
            &[&[r_gas, r_value]],
            oracle(&[(alice, CoinType::Usdc, 100)]),
        );
        assert_eq!(d_ok, vec![ReservationDecision::Accept]);

        // Same reservations against 80 → drop with requested = 90.
        let d_drop = check_reservations(
            &[&[r_gas, r_value]],
            oracle(&[(alice, CoinType::Usdc, 80)]),
        );
        match d_drop[0] {
            ReservationDecision::Drop {
                reason: ReservationFailure::InsufficientBalance { requested, available, .. },
            } => {
                assert_eq!(requested, 90);
                assert_eq!(available, 80);
            }
            other => panic!("expected drop, got {:?}", other),
        }
    }

    /// Intra-tx u64 overflow: two reservations that each near `u64::MAX`
    /// summing past it must be caught explicitly, not silently truncate.
    /// Not reachable with realistic balances (total supply ≪ u64::MAX)
    /// but defense-in-depth against constructed/malicious tx inputs.
    #[test]
    fn reservations_intra_tx_overflow_drops_with_specific_reason() {
        let alice = addr(1);
        let r1 = WithdrawalReservation::new(alice, CoinType::Usdc, u64::MAX);
        let r2 = WithdrawalReservation::new(alice, CoinType::Usdc, 1);
        let decisions = check_reservations(
            &[&[r1, r2]],
            oracle(&[(alice, CoinType::Usdc, u64::MAX)]),
        );
        assert_eq!(
            decisions[0],
            ReservationDecision::Drop {
                reason: ReservationFailure::IntraTxOverflow {
                    owner: alice,
                    coin_type: CoinType::Usdc
                }
            }
        );
    }

    /// Underfunded-then-funded tx pattern: a drop in the middle must
    /// not poison later txs' running balances. The dropped tx's
    /// reservation is NOT debited.
    #[test]
    fn reservations_drop_does_not_debit_running_balance() {
        let alice = addr(1);
        let r1 = WithdrawalReservation::new(alice, CoinType::Usdc, 30);
        let r2_too_big = WithdrawalReservation::new(alice, CoinType::Usdc, 1000);
        let r3 = WithdrawalReservation::new(alice, CoinType::Usdc, 40);
        let decisions = check_reservations(
            &[&[r1], &[r2_too_big], &[r3]],
            oracle(&[(alice, CoinType::Usdc, 100)]),
        );
        // After tx 0: 70. tx 1 drops (1000 > 70, no debit). tx 2: 40 ≤ 70 → accept.
        assert_eq!(decisions[0], ReservationDecision::Accept);
        assert!(matches!(decisions[1], ReservationDecision::Drop { .. }));
        assert_eq!(decisions[2], ReservationDecision::Accept);
    }

    /// Permutation independence for non-conflicting txs: as long as no
    /// account is touched by more than one tx, ordering doesn't change
    /// outcomes. This is the parallelism guarantee in code form.
    #[test]
    fn reservations_independent_txs_are_order_invariant() {
        let alice = addr(1);
        let bob = addr(2);
        let charlie = addr(3);
        let txs_a: &[&[WithdrawalReservation]] = &[
            &[WithdrawalReservation::new(alice, CoinType::Usdc, 10)],
            &[WithdrawalReservation::new(bob, CoinType::Usdc, 20)],
            &[WithdrawalReservation::new(charlie, CoinType::Usdc, 30)],
        ];
        let txs_b: &[&[WithdrawalReservation]] = &[
            &[WithdrawalReservation::new(charlie, CoinType::Usdc, 30)],
            &[WithdrawalReservation::new(alice, CoinType::Usdc, 10)],
            &[WithdrawalReservation::new(bob, CoinType::Usdc, 20)],
        ];
        let oracle_data = [
            (alice, CoinType::Usdc, 100u64),
            (bob, CoinType::Usdc, 100),
            (charlie, CoinType::Usdc, 100),
        ];
        let da = check_reservations(txs_a, oracle(&oracle_data));
        let db = check_reservations(txs_b, oracle(&oracle_data));
        assert!(da.iter().all(|d| matches!(d, ReservationDecision::Accept)));
        assert!(db.iter().all(|d| matches!(d, ReservationDecision::Accept)));
    }

    /// Zero-amount reservation is a degenerate but valid input — it
    /// reserves nothing, must always pass even at zero balance.
    #[test]
    fn reservations_zero_amount_always_accepts() {
        let alice = addr(1);
        let r = WithdrawalReservation::new(alice, CoinType::Usdc, 0);
        let decisions = check_reservations(&[&[r]], oracle(&[]));
        assert_eq!(decisions, vec![ReservationDecision::Accept]);
    }

    /// The oracle must be called *at most once per key*. Repeated
    /// queries for the same key would amplify DB load on every commit;
    /// guarantee internal caching.
    #[test]
    fn reservations_oracle_called_at_most_once_per_key() {
        use std::cell::RefCell;
        use std::collections::HashMap;

        let alice = addr(1);
        let bob = addr(2);
        let calls: RefCell<HashMap<(SomaAddress, CoinType), usize>> =
            RefCell::new(HashMap::new());

        // Three txs all touching alice's USDC, plus one for bob.
        let txs: Vec<Vec<WithdrawalReservation>> = vec![
            vec![WithdrawalReservation::new(alice, CoinType::Usdc, 10)],
            vec![WithdrawalReservation::new(alice, CoinType::Usdc, 10)],
            vec![WithdrawalReservation::new(alice, CoinType::Usdc, 10)],
            vec![WithdrawalReservation::new(bob, CoinType::Usdc, 5)],
        ];
        let tx_refs: Vec<&[WithdrawalReservation]> =
            txs.iter().map(|v| v.as_slice()).collect();

        check_reservations(&tx_refs, |o, c| {
            *calls.borrow_mut().entry((o, c)).or_insert(0) += 1;
            if o == alice && c == CoinType::Usdc {
                100
            } else if o == bob && c == CoinType::Usdc {
                100
            } else {
                0
            }
        });

        let calls = calls.into_inner();
        assert_eq!(calls.get(&(alice, CoinType::Usdc)), Some(&1));
        assert_eq!(calls.get(&(bob, CoinType::Usdc)), Some(&1));
    }
}
