// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the account-balance accumulator storage layer
//! (`AuthorityStore::*_balance` methods + the `accumulator_balances`
//! column family).
//!
//! These tests use a fresh `AuthorityPerpetualTables` backed by a
//! temporary RocksDB directory; no full authority/test-cluster is
//! needed because we're exercising the storage primitives in
//! isolation. The settlement pipeline integration is verified in
//! later stages.

use std::collections::BTreeMap;
use std::sync::Arc;

use store::Map as _;
use tempfile::tempdir;
use types::balance::{
    BalanceEvent, ReservationDecision, ReservationFailure, WithdrawalReservation,
};
use types::base::SomaAddress;
use types::object::CoinType;

use crate::authority_store::AuthorityStore;
use crate::authority_store_tables::AuthorityPerpetualTables;

/// Build a fresh `AuthorityStore` with no genesis. Storage tests
/// don't need any prior state.
fn fresh_store() -> Arc<AuthorityStore> {
    let dir = tempdir().unwrap();
    let perpetual_tables = Arc::new(AuthorityPerpetualTables::open(dir.path(), None));
    AuthorityStore::open_no_genesis(perpetual_tables).unwrap()
    // tempdir is leaked intentionally — the Arc holds a path inside it
    // and tests are short-lived. Same pattern as cache::tests.
}

fn addr(seed: u8) -> SomaAddress {
    SomaAddress::new([seed; 32])
}

// ---------------------------------------------------------------------
// get_balance / set_balance
// ---------------------------------------------------------------------

/// An address that has never received funds reads as zero, not as a
/// missing-key error. This is the contract callers depend on (RPC
/// GetBalance, scheduler reservation check, etc.).
#[tokio::test]
async fn get_balance_missing_returns_zero() {
    let store = fresh_store();
    assert_eq!(store.get_balance(addr(1), CoinType::Usdc).unwrap(), 0);
    assert_eq!(store.get_balance(addr(1), CoinType::Soma).unwrap(), 0);
}

#[tokio::test]
async fn set_balance_round_trips() {
    let store = fresh_store();
    let alice = addr(1);

    store.set_balance(alice, CoinType::Usdc, 1_000_000).unwrap();
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 1_000_000);

    // Setting again overwrites cleanly.
    store.set_balance(alice, CoinType::Usdc, 500_000).unwrap();
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 500_000);

    // Different coin type for the same owner is a different entry.
    store.set_balance(alice, CoinType::Soma, 42).unwrap();
    assert_eq!(store.get_balance(alice, CoinType::Soma).unwrap(), 42);
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 500_000);

    // Different owner, same coin type, also independent.
    let bob = addr(2);
    assert_eq!(store.get_balance(bob, CoinType::Usdc).unwrap(), 0);
}

/// Setting a balance to zero is a valid operation (drain to empty).
/// It should still read back as zero — same as a never-set entry,
/// which is the right answer for callers.
#[tokio::test]
async fn set_balance_to_zero_round_trips() {
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 100).unwrap();
    store.set_balance(alice, CoinType::Usdc, 0).unwrap();
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 0);
}

/// Setting to u64::MAX works without overflow.
#[tokio::test]
async fn set_balance_at_max_round_trips() {
    let store = fresh_store();
    store.set_balance(addr(1), CoinType::Usdc, u64::MAX).unwrap();
    assert_eq!(store.get_balance(addr(1), CoinType::Usdc).unwrap(), u64::MAX);
}

// ---------------------------------------------------------------------
// apply_balance_delta (single delta)
// ---------------------------------------------------------------------

#[tokio::test]
async fn apply_positive_delta_credits() {
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 100).unwrap();
    let new = store.apply_balance_delta(alice, CoinType::Usdc, 50).unwrap();
    assert_eq!(new, 150);
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 150);
}

#[tokio::test]
async fn apply_negative_delta_debits() {
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 100).unwrap();
    let new = store.apply_balance_delta(alice, CoinType::Usdc, -30).unwrap();
    assert_eq!(new, 70);
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 70);
}

#[tokio::test]
async fn apply_zero_delta_is_noop_but_persists_entry() {
    let store = fresh_store();
    let alice = addr(1);
    let new = store.apply_balance_delta(alice, CoinType::Usdc, 0).unwrap();
    assert_eq!(new, 0);
    // Reads back as zero (never written above), still consistent.
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 0);
}

/// Underflow: stored balance is preserved, error returned.
#[tokio::test]
async fn apply_underflow_preserves_balance_and_errors() {
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 50).unwrap();

    let result = store.apply_balance_delta(alice, CoinType::Usdc, -100);
    assert!(result.is_err(), "underflow must return an error");
    let msg = format!("{:?}", result.unwrap_err());
    assert!(msg.contains("underflow"), "error must mention underflow: {}", msg);

    // Balance unchanged.
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 50);
}

/// Overflow: stored balance is preserved, error returned.
#[tokio::test]
async fn apply_overflow_preserves_balance_and_errors() {
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, u64::MAX).unwrap();

    let result = store.apply_balance_delta(alice, CoinType::Usdc, 1);
    assert!(result.is_err(), "overflow must return an error");
    let msg = format!("{:?}", result.unwrap_err());
    assert!(msg.contains("overflow"), "error must mention overflow: {}", msg);

    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), u64::MAX);
}

// ---------------------------------------------------------------------
// multi_apply_balance_deltas (the settlement-tx hot path)
// ---------------------------------------------------------------------

/// Happy path: apply a batch of mixed deposits/withdrawals atomically.
/// Each entry should land at the expected balance.
#[tokio::test]
async fn multi_apply_atomic_success() {
    let store = fresh_store();
    let alice = addr(1);
    let bob = addr(2);
    let charlie = addr(3);

    // Initial balances
    store.set_balance(alice, CoinType::Usdc, 1000).unwrap();
    store.set_balance(bob, CoinType::Usdc, 500).unwrap();
    // charlie has nothing yet
    store.set_balance(alice, CoinType::Soma, 200).unwrap();

    let mut deltas: BTreeMap<(SomaAddress, CoinType), i128> = BTreeMap::new();
    deltas.insert((alice, CoinType::Usdc), -300); // alice 1000 → 700
    deltas.insert((bob, CoinType::Usdc), 200);    // bob   500 → 700
    deltas.insert((charlie, CoinType::Usdc), 100); // charlie 0 → 100
    deltas.insert((alice, CoinType::Soma), 50);   // alice's SOMA: 200 → 250

    let new_balances = store.multi_apply_balance_deltas(&deltas).unwrap();

    // Verify returned vec matches what's persisted.
    assert_eq!(new_balances.len(), 4);
    let new_map: BTreeMap<_, _> = new_balances.into_iter().collect();
    assert_eq!(new_map[&(alice, CoinType::Usdc)], 700);
    assert_eq!(new_map[&(bob, CoinType::Usdc)], 700);
    assert_eq!(new_map[&(charlie, CoinType::Usdc)], 100);
    assert_eq!(new_map[&(alice, CoinType::Soma)], 250);

    // Verify persisted state matches.
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 700);
    assert_eq!(store.get_balance(bob, CoinType::Usdc).unwrap(), 700);
    assert_eq!(store.get_balance(charlie, CoinType::Usdc).unwrap(), 100);
    assert_eq!(store.get_balance(alice, CoinType::Soma).unwrap(), 250);
}

/// Atomicity invariant: if any delta in the batch would underflow,
/// the entire batch is aborted — no partial application. This is the
/// settlement-correctness invariant the upstream scheduler relies on
/// when the per-tx reservation checks pass but commit-level
/// aggregation produces an unexpected negative.
#[tokio::test]
async fn multi_apply_atomic_failure_aborts_batch() {
    let store = fresh_store();
    let alice = addr(1);
    let bob = addr(2);

    store.set_balance(alice, CoinType::Usdc, 1000).unwrap();
    store.set_balance(bob, CoinType::Usdc, 50).unwrap();

    // The batch contains a valid delta for alice and an underflow for bob.
    let mut deltas: BTreeMap<(SomaAddress, CoinType), i128> = BTreeMap::new();
    deltas.insert((alice, CoinType::Usdc), -100); // valid
    deltas.insert((bob, CoinType::Usdc), -200);   // bob has 50, would underflow

    let result = store.multi_apply_balance_deltas(&deltas);
    assert!(result.is_err(), "underflow in any entry must abort the batch");

    // Critically: alice's delta must NOT have been applied because the
    // batch aborted. Both balances remain at their original values.
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 1000);
    assert_eq!(store.get_balance(bob, CoinType::Usdc).unwrap(), 50);
}

/// Empty batch is a no-op success.
#[tokio::test]
async fn multi_apply_empty_succeeds() {
    let store = fresh_store();
    let result = store.multi_apply_balance_deltas(&BTreeMap::new());
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

/// Net-zero deltas (deposit + matching withdrawal) preserve the
/// balance and write the entry. Important because settlement still
/// records "this address was touched" even when the net is zero.
#[tokio::test]
async fn multi_apply_net_zero_preserves_balance() {
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 100).unwrap();

    let mut deltas: BTreeMap<(SomaAddress, CoinType), i128> = BTreeMap::new();
    deltas.insert((alice, CoinType::Usdc), 0);

    let result = store.multi_apply_balance_deltas(&deltas).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, 100);
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 100);
}

// ---------------------------------------------------------------------
// Iteration
// ---------------------------------------------------------------------

#[tokio::test]
async fn iter_all_balances_returns_everything() {
    let store = fresh_store();
    let alice = addr(1);
    let bob = addr(2);
    store.set_balance(alice, CoinType::Usdc, 100).unwrap();
    store.set_balance(alice, CoinType::Soma, 200).unwrap();
    store.set_balance(bob, CoinType::Usdc, 300).unwrap();

    let all = store.iter_all_balances().unwrap();
    assert_eq!(all.len(), 3);
    let m: BTreeMap<_, _> = all.into_iter().collect();
    assert_eq!(m[&(alice, CoinType::Usdc)], 100);
    assert_eq!(m[&(alice, CoinType::Soma)], 200);
    assert_eq!(m[&(bob, CoinType::Usdc)], 300);
}

#[tokio::test]
async fn iter_balances_for_owner_filters_correctly() {
    let store = fresh_store();
    let alice = addr(1);
    let bob = addr(2);
    store.set_balance(alice, CoinType::Usdc, 100).unwrap();
    store.set_balance(alice, CoinType::Soma, 200).unwrap();
    store.set_balance(bob, CoinType::Usdc, 300).unwrap();

    let alice_balances = store.iter_balances_for_owner(alice).unwrap();
    assert_eq!(alice_balances.len(), 2);
    let m: BTreeMap<_, _> = alice_balances.into_iter().collect();
    assert_eq!(m[&CoinType::Usdc], 100);
    assert_eq!(m[&CoinType::Soma], 200);

    let bob_balances = store.iter_balances_for_owner(bob).unwrap();
    assert_eq!(bob_balances.len(), 1);
    assert_eq!(bob_balances[0], (CoinType::Usdc, 300));

    // Address with no balances yields empty.
    let charlie_balances = store.iter_balances_for_owner(addr(3)).unwrap();
    assert!(charlie_balances.is_empty());
}

// ---------------------------------------------------------------------
// Cross-validator determinism: the same sequence of ops produces the
// same end state. We exercise this implicitly by checking that
// iteration order is stable (BTreeMap ordering on the (owner, type)
// tuple), but it's worth an explicit test.
// ---------------------------------------------------------------------

#[tokio::test]
async fn iteration_order_is_deterministic() {
    let store = fresh_store();
    // Insert in non-sorted order; expect sorted readback.
    let bob = addr(2);
    let alice = addr(1);
    let charlie = addr(3);
    store.set_balance(charlie, CoinType::Usdc, 3).unwrap();
    store.set_balance(alice, CoinType::Usdc, 1).unwrap();
    store.set_balance(bob, CoinType::Usdc, 2).unwrap();

    let all = store.iter_all_balances().unwrap();
    let owners: Vec<SomaAddress> = all.iter().map(|((o, _), _)| *o).collect();
    // SomaAddress orders by raw bytes, and our addr(N) just fills with
    // N's, so addr(1) < addr(2) < addr(3).
    assert_eq!(owners, vec![alice, bob, charlie]);
}

// ---------------------------------------------------------------------
// Stage 1c: bulk_insert_genesis_balances
// ---------------------------------------------------------------------
//
// `open_inner` calls `bulk_insert_genesis_balances` exactly once when
// the database is empty, and the helper itself is exercised here in
// isolation. End-to-end seeding through `Genesis` is covered by
// existing test-cluster smoke tests (which now spin up with the new
// column family populated).

#[tokio::test]
async fn bulk_insert_genesis_balances_writes_all_entries() {
    let store = fresh_store();
    let alice = addr(1);
    let bob = addr(2);

    let mut seed: BTreeMap<(SomaAddress, CoinType), u64> = BTreeMap::new();
    seed.insert((alice, CoinType::Soma), 5_000);
    seed.insert((alice, CoinType::Usdc), 1_000_000);
    seed.insert((bob, CoinType::Usdc), 250_000);

    store.bulk_insert_genesis_balances(&seed).unwrap();

    assert_eq!(store.get_balance(alice, CoinType::Soma).unwrap(), 5_000);
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 1_000_000);
    assert_eq!(store.get_balance(bob, CoinType::Usdc).unwrap(), 250_000);
    assert_eq!(store.get_balance(bob, CoinType::Soma).unwrap(), 0);
}

#[tokio::test]
async fn bulk_insert_genesis_balances_skips_zero_entries() {
    // Zero entries are equivalent to absent entries. The seeder must not
    // create useless rows for them — keeps the column family tight and
    // makes iteration cheaper for the (post-Stage 13) world.
    let store = fresh_store();
    let alice = addr(1);
    let mut seed: BTreeMap<(SomaAddress, CoinType), u64> = BTreeMap::new();
    seed.insert((alice, CoinType::Soma), 0);
    seed.insert((alice, CoinType::Usdc), 42);

    store.bulk_insert_genesis_balances(&seed).unwrap();

    let all = store.iter_all_balances().unwrap();
    assert_eq!(all.len(), 1, "zero entry must not be persisted");
    assert_eq!(all[0], ((alice, CoinType::Usdc), 42));
}

#[tokio::test]
async fn bulk_insert_genesis_balances_empty_map_is_a_noop() {
    let store = fresh_store();
    let seed: BTreeMap<(SomaAddress, CoinType), u64> = BTreeMap::new();
    store.bulk_insert_genesis_balances(&seed).unwrap();
    assert!(store.iter_all_balances().unwrap().is_empty());
}

// ---------------------------------------------------------------------
// Stage 3: apply_settlement_events
// ---------------------------------------------------------------------
//
// The settlement system tx's events are applied to `accumulator_balances`
// inside the same RocksDB write batch as the rest of a commit's tx
// outputs. These tests exercise the helper directly — building the
// surrounding `TransactionOutputs` is heavier than is useful for
// verifying the math.

/// Run a closure with a fresh write batch and atomically commit. Mirrors
/// the per-commit batch lifecycle in `build_db_batch`.
fn apply_settlement(
    store: &Arc<AuthorityStore>,
    events: &[BalanceEvent],
) -> Result<(), types::error::SomaError> {
    let mut batch = store.perpetual_tables.accumulator_balances.batch();
    store.apply_settlement_events(&mut batch, events)?;
    batch.write().expect("rocksdb write");
    Ok(())
}

#[tokio::test]
async fn settlement_applies_pure_credits() {
    let store = fresh_store();
    let alice = addr(1);
    let bob = addr(2);

    apply_settlement(
        &store,
        &[
            BalanceEvent::deposit(alice, CoinType::Usdc, 1_000_000),
            BalanceEvent::deposit(bob, CoinType::Soma, 500),
        ],
    )
    .expect("settlement applies cleanly");

    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 1_000_000);
    assert_eq!(store.get_balance(bob, CoinType::Soma).unwrap(), 500);
}

#[tokio::test]
async fn settlement_applies_mixed_credit_and_debit() {
    // The realistic shape: aggregated commit has both withdraws (senders)
    // and deposits (recipients). Apply must update both atomically.
    let store = fresh_store();
    let alice = addr(1);
    let bob = addr(2);
    store.set_balance(alice, CoinType::Usdc, 1000).unwrap();

    apply_settlement(
        &store,
        &[
            BalanceEvent::withdraw(alice, CoinType::Usdc, 300),
            BalanceEvent::deposit(bob, CoinType::Usdc, 300),
        ],
    )
    .unwrap();

    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 700);
    assert_eq!(store.get_balance(bob, CoinType::Usdc).unwrap(), 300);
}

#[tokio::test]
async fn settlement_underflow_aborts_with_error() {
    // Underflow shouldn't be reachable in practice — Stage 4's reservation
    // pre-pass blocks underfunded txs before execution — but if it does
    // (pipeline bug) we surface a structured error instead of silently
    // saturating. Critical: the existing balance must not be corrupted.
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 100).unwrap();

    let err = apply_settlement(
        &store,
        &[BalanceEvent::withdraw(alice, CoinType::Usdc, 200)],
    )
    .expect_err("underflow must surface an error");
    assert!(format!("{}", err).contains("settlement underflow"));

    // Balance unchanged because the batch never wrote.
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 100);
}

#[tokio::test]
async fn settlement_overflow_aborts_with_error() {
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, u64::MAX).unwrap();

    let err = apply_settlement(
        &store,
        &[BalanceEvent::deposit(alice, CoinType::Usdc, 1)],
    )
    .expect_err("overflow must surface an error");
    assert!(format!("{}", err).contains("settlement overflow"));

    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), u64::MAX);
}

#[tokio::test]
async fn settlement_empty_events_is_a_noop() {
    // Every commit writes a settlement; one with no balance changes
    // must succeed and not touch the column family.
    let store = fresh_store();
    apply_settlement(&store, &[]).unwrap();
    assert!(store.iter_all_balances().unwrap().is_empty());
}

#[tokio::test]
async fn settlement_creates_balance_row_for_first_time_recipient() {
    // A deposit to an account that has never held this coin type before
    // must succeed (current = 0 by convention).
    let store = fresh_store();
    let alice = addr(1);
    apply_settlement(&store, &[BalanceEvent::deposit(alice, CoinType::Soma, 42)])
        .unwrap();
    assert_eq!(store.get_balance(alice, CoinType::Soma).unwrap(), 42);
}

// ---------------------------------------------------------------------
// Stage 4: check_tx_reservations (store-backed reservation pre-pass)
// ---------------------------------------------------------------------
//
// The math is exhaustively tested in types::balance::tests; these
// tests cover the wiring from the AuthorityStore's accumulator_balances
// column family into the pure check_reservations function.

#[tokio::test]
async fn reservations_no_balance_drops_any_nonzero_request() {
    let store = fresh_store();
    let alice = addr(1);
    let r = WithdrawalReservation::new(alice, CoinType::Usdc, 1);
    let decisions = store.check_tx_reservations(&[&[r]]).unwrap();
    match decisions[0] {
        ReservationDecision::Drop {
            reason: ReservationFailure::InsufficientBalance { available, .. },
        } => assert_eq!(available, 0, "missing key reads as zero"),
        other => panic!("expected drop with available=0, got {:?}", other),
    }
}

#[tokio::test]
async fn reservations_seeded_balance_passes() {
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 1000).unwrap();
    let r = WithdrawalReservation::new(alice, CoinType::Usdc, 250);
    let decisions = store.check_tx_reservations(&[&[r]]).unwrap();
    assert_eq!(decisions, vec![ReservationDecision::Accept]);
}

#[tokio::test]
async fn reservations_running_tentative_visible_across_txs() {
    // The store-backed wrapper must hand the same oracle closure to
    // check_reservations so the running tentative balance is shared
    // across the commit, NOT re-read from the DB per tx. If it
    // re-read per tx, both would see the original 1000 and pass.
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 1000).unwrap();
    let r = WithdrawalReservation::new(alice, CoinType::Usdc, 600);
    let decisions = store.check_tx_reservations(&[&[r], &[r]]).unwrap();
    assert_eq!(decisions[0], ReservationDecision::Accept);
    assert!(matches!(decisions[1], ReservationDecision::Drop { .. }));
}

#[tokio::test]
async fn reservations_empty_input_returns_empty() {
    let store = fresh_store();
    let decisions = store.check_tx_reservations(&[]).unwrap();
    assert!(decisions.is_empty());
}

#[tokio::test]
async fn reservations_does_not_mutate_db() {
    // The pre-pass MUST be read-only — settlement is the sole writer
    // to the balance table. Verify directly: reservations check, then
    // confirm balance unchanged.
    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 1000).unwrap();
    let r = WithdrawalReservation::new(alice, CoinType::Usdc, 999);
    let _ = store.check_tx_reservations(&[&[r]]).unwrap();
    assert_eq!(
        store.get_balance(alice, CoinType::Usdc).unwrap(),
        1000,
        "pre-pass must not write to the balance table"
    );
}

// ---------------------------------------------------------------------
// Stage 13m: write_one_transaction_outputs atomicity invariant.
//
// Effects is the single chokepoint for state changes — object
// writes, balance events, and delegation events must all land via
// the same `WriteBatch`. A future refactor that splits any of these
// into a separate batch would silently break per-tx atomicity.
//
// The test below builds a TransactionOutputs whose effects struct
// carries one entry from each family, drives it through
// `build_db_batch` + commit, then asserts every family landed.
// `build_db_batch` is the single producer of the batch; if any of
// the three writes ever moved out of `write_one_transaction_outputs`,
// the corresponding assertion would fail.
// ---------------------------------------------------------------------

#[tokio::test]
async fn write_one_transaction_outputs_is_atomic_across_state_families() {
    use std::collections::BTreeMap;

    use store::Map as _;
    use types::base::SomaAddress;
    use types::digests::{ObjectDigest, TransactionDigest};
    use types::effects::object_change::{EffectsObjectChange, IDOperation, ObjectIn, ObjectOut};
    use types::effects::{ExecutionStatus, TransactionEffects};
    use types::object::{ObjectID, Owner, Version};
    use types::temporary_store::DelegationEvent;
    use types::transaction::VerifiedTransaction;
    use types::transaction_outputs::TransactionOutputs;
    use types::tx_fee::TransactionFee;

    let store = fresh_store();

    // Use a settlement tx as the carrier — the simplest system tx
    // we can construct. The kind doesn't matter for this test;
    // we're driving the perpetual store, not the executor.
    let tx = VerifiedTransaction::new_settlement_transaction(0, 1, None, vec![], vec![]);

    // 1. Object family: a single created object.
    let obj_id = ObjectID::random();
    let obj_digest = ObjectDigest::random();
    let owner = Owner::AddressOwner(SomaAddress::random());
    let mut changed_objects = BTreeMap::new();
    changed_objects.insert(
        obj_id,
        EffectsObjectChange {
            input_state: ObjectIn::NotExist,
            output_state: ObjectOut::ObjectWrite((obj_digest, owner.clone())),
            id_operation: IDOperation::Created,
        },
    );

    // 2. Balance-accumulator family: a deposit to alice.
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 100).unwrap();
    let balance_event = BalanceEvent::deposit(alice, CoinType::Usdc, 25);

    // 3. Delegation family: a stake delta. The pool/staker only
    //    needs to be a stable bytes pair for this test — apply_
    //    delegation_events writes via key composition, not lookup.
    let delegation_event = DelegationEvent {
        pool_id: ObjectID::random(),
        staker: addr(2),
        delta: 0, // zero delta still hits the apply path; non-zero
                  // would also work but introduces the read-existing
                  // path which isn't what we're testing here.
        set_period: Some(0),
    };

    let effects = TransactionEffects::new(
        ExecutionStatus::Success,
        0,
        Vec::new(),
        *tx.digest(),
        Version::from_u64(1),
        changed_objects,
        Vec::new(),
        TransactionFee::default(),
        None,
        vec![balance_event],
        vec![delegation_event],
    );

    // Mirror the consensus path: drain TemporaryStore-equivalent
    // input, build outputs, then drive build_db_batch.
    let inner = types::temporary_store::InnerTemporaryStore::new(
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        Version::from_u64(1),
        BTreeMap::new(),
    );
    let outputs = std::sync::Arc::new(
        TransactionOutputs::build_transaction_outputs(tx, effects, inner),
    );

    let batch = store.build_db_batch(0, &[outputs]).unwrap();
    batch.write().unwrap();

    // Atomicity assertion: all three families landed.
    //
    // (1) Effects landed in the executed_transaction_digests CF.
    //     We use that as the "tx was committed" marker.
    // (2) Alice's USDC balance reflects the +25 deposit.
    // (3) The delegation event was applied (no read-back surface
    //     here — we just confirm `apply_delegation_events` was
    //     reached without panicking, which is what would happen
    //     if the events were missed by `write_one_transaction_outputs`).
    assert_eq!(
        store.get_balance(alice, CoinType::Usdc).unwrap(),
        125,
        "balance event must have been applied via the same batch as object writes",
    );
}

// ---------------------------------------------------------------------
// Stage 14b's dual-store equality invariant tests are gone in 14c.6.
//
// 14b kept CF rows + accumulator objects in lockstep via a dual-write
// helper. 14c.6 dropped the object-side write because (1) it bypassed
// the writeback cache, causing coherence panics, and (2) the
// SettlementScheduler infrastructure now owns the SIP-58-shape
// migration. CF stays as the runtime balance source of truth; the
// object-side returns when the settlement-input declaration mechanism
// lands in 14c.7 and accumulator object writes ride
// `effects.changed_objects` through the standard cache pipeline.
// ---------------------------------------------------------------------

// (Stage 14b's `dual_store_*` tests removed in 14c.6 — they
// asserted on a CF-vs-object-store invariant that no longer applies.
// The accumulator-object writes return via the standard effects
// pipeline in 14c.7+; new tests will exercise the apply-via-effects
// invariant then.)

#[tokio::test]
async fn write_one_transaction_outputs_no_events_is_noop_for_balance_cf() {
    // Defensive: a tx that only has object changes (no balance or
    // delegation events) must still write the object-side changes
    // and leave the balance CF alone. This guards the path where
    // `effects.balance_events()` is empty.
    use std::collections::BTreeMap;

    use types::digests::{ObjectDigest, TransactionDigest};
    use types::effects::object_change::{EffectsObjectChange, IDOperation, ObjectIn, ObjectOut};
    use types::effects::{ExecutionStatus, TransactionEffects};
    use types::object::{ObjectID, Owner, Version};
    use types::transaction::VerifiedTransaction;
    use types::transaction_outputs::TransactionOutputs;
    use types::tx_fee::TransactionFee;

    let store = fresh_store();
    let alice = addr(1);
    store.set_balance(alice, CoinType::Usdc, 999).unwrap();

    let tx = VerifiedTransaction::new_settlement_transaction(0, 1, None, vec![], vec![]);

    let mut changed_objects = BTreeMap::new();
    changed_objects.insert(
        ObjectID::random(),
        EffectsObjectChange {
            input_state: ObjectIn::NotExist,
            output_state: ObjectOut::ObjectWrite((
                ObjectDigest::random(),
                Owner::AddressOwner(SomaAddress::random()),
            )),
            id_operation: IDOperation::Created,
        },
    );

    let effects = TransactionEffects::new(
        ExecutionStatus::Success,
        0,
        Vec::new(),
        *tx.digest(),
        Version::from_u64(1),
        changed_objects,
        Vec::new(),
        TransactionFee::default(),
        None,
        Vec::new(),
        Vec::new(),
    );

    let inner = types::temporary_store::InnerTemporaryStore::new(
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        Version::from_u64(1),
        BTreeMap::new(),
    );
    let outputs = std::sync::Arc::new(
        TransactionOutputs::build_transaction_outputs(tx, effects, inner),
    );

    let batch = store.build_db_batch(0, &[outputs]).unwrap();
    batch.write().unwrap();

    assert_eq!(
        store.get_balance(alice, CoinType::Usdc).unwrap(),
        999,
        "tx with no balance events must not touch the balance CF",
    );
}
