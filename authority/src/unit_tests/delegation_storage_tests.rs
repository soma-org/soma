// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the F1-shaped delegation-balance storage layer
//! (`AuthorityStore::*_delegation` methods + the `delegations` column
//! family).
//!
//! Stage 9d-C1 reshapes the table to one row per (pool, staker)
//! holding a `Delegation { principal, last_collected_period }`. These
//! tests exercise the storage primitives in isolation so the executor
//! changes in C2/C3 can rely on them being correct.

use std::sync::Arc;

use tempfile::tempdir;
use types::base::SomaAddress;
use types::object::ObjectID;
use types::system_state::staking::Delegation;

use crate::authority_store::AuthorityStore;
use crate::authority_store_tables::AuthorityPerpetualTables;

fn fresh_store() -> Arc<AuthorityStore> {
    let dir = tempdir().unwrap();
    let perpetual_tables = Arc::new(AuthorityPerpetualTables::open(dir.path(), None));
    AuthorityStore::open_no_genesis(perpetual_tables).unwrap()
}

fn pool(seed: u8) -> ObjectID {
    ObjectID::new([seed; 32])
}

fn addr(seed: u8) -> SomaAddress {
    SomaAddress::new([seed; 32])
}

fn delegation(principal: u64, period: u64) -> Delegation {
    Delegation::new(principal, period)
}

// ---------------------------------------------------------------------
// get_delegation / set_delegation
// ---------------------------------------------------------------------

/// A delegation row that has never been written reads as the default
/// (zero principal, period 0), not as a missing-key error. Mirrors
/// `get_balance` semantics.
#[tokio::test]
async fn get_delegation_missing_returns_default() {
    let store = fresh_store();
    assert_eq!(store.get_delegation(pool(1), addr(1)).unwrap(), Delegation::default());
    assert_eq!(store.get_delegation(pool(1), addr(2)).unwrap().principal, 0);
}

#[tokio::test]
async fn set_delegation_round_trips() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.set_delegation(p, alice, delegation(1_000_000, 7)).unwrap();
    let got = store.get_delegation(p, alice).unwrap();
    assert_eq!(got.principal, 1_000_000);
    assert_eq!(got.last_collected_period, 7);
}

/// Setting a delegation row whose principal is zero deletes the row
/// entirely. Important because `get_delegation` returns the default
/// for missing entries — keeping a zero-principal row would still
/// appear in `iter_delegations_for_staker` scans and waste storage.
/// Withdrawing a stake fully should leave no trace.
#[tokio::test]
async fn set_delegation_zero_principal_deletes_row() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.set_delegation(p, alice, delegation(500, 3)).unwrap();
    assert_eq!(store.iter_delegations_for_staker(alice).unwrap().len(), 1);

    store.set_delegation(p, alice, delegation(0, 3)).unwrap();
    assert_eq!(store.get_delegation(p, alice).unwrap().principal, 0);
    assert!(
        store.iter_delegations_for_staker(alice).unwrap().is_empty(),
        "zero-principal row must be deleted, not kept",
    );
}

/// F1 schema: ONE row per (pool, staker). Repeat sets to the same key
/// overwrite — they do NOT create separate rows by activation epoch.
#[tokio::test]
async fn one_row_per_pool_staker_pair() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.set_delegation(p, alice, delegation(100, 5)).unwrap();
    store.set_delegation(p, alice, delegation(200, 6)).unwrap();
    store.set_delegation(p, alice, delegation(300, 7)).unwrap();

    let got = store.get_delegation(p, alice).unwrap();
    assert_eq!(got.principal, 300, "last set wins");
    assert_eq!(got.last_collected_period, 7);

    let listed = store.iter_delegations_for_staker(alice).unwrap();
    assert_eq!(listed.len(), 1, "F1 schema: one row per (pool, staker)");
}

// ---------------------------------------------------------------------
// apply_delegation_delta
// ---------------------------------------------------------------------

#[tokio::test]
async fn apply_delegation_delta_increments() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    let after = store.apply_delegation_delta(p, alice, 1_000, None).unwrap();
    assert_eq!(after, 1_000);

    let after = store.apply_delegation_delta(p, alice, 500, None).unwrap();
    assert_eq!(after, 1_500);

    // last_collected_period stays at 0 when no set_period override.
    assert_eq!(store.get_delegation(p, alice).unwrap().last_collected_period, 0);
}

#[tokio::test]
async fn apply_delegation_delta_advances_period_when_requested() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.apply_delegation_delta(p, alice, 1_000, None).unwrap();
    assert_eq!(store.get_delegation(p, alice).unwrap().last_collected_period, 0);

    // F1 fold during AddStake: principal += 500 AND advance period to 5.
    store.apply_delegation_delta(p, alice, 500, Some(5)).unwrap();
    let got = store.get_delegation(p, alice).unwrap();
    assert_eq!(got.principal, 1_500);
    assert_eq!(got.last_collected_period, 5);
}

/// Underflow is a hard error — withdrawing more than was staked must
/// not silently zero the row.
#[tokio::test]
async fn apply_delegation_delta_underflow_errors() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.set_delegation(p, alice, delegation(500, 0)).unwrap();
    let err = store
        .apply_delegation_delta(p, alice, -1_000, None)
        .expect_err("underflow must error");
    assert!(format!("{:?}", err).contains("underflow"), "got: {:?}", err);
    // Row unchanged after the failed apply.
    assert_eq!(store.get_delegation(p, alice).unwrap().principal, 500);
}

// ---------------------------------------------------------------------
// sum_delegations_for_pool / iter_delegations_for_staker
// ---------------------------------------------------------------------

#[tokio::test]
async fn sum_delegations_for_pool_aggregates_across_stakers() {
    let store = fresh_store();
    let p1 = pool(1);
    let p2 = pool(2);
    let alice = addr(10);
    let bob = addr(11);
    let carol = addr(12);

    store.set_delegation(p1, alice, delegation(300, 0)).unwrap();
    store.set_delegation(p1, bob, delegation(300, 0)).unwrap();
    store.set_delegation(p1, carol, delegation(400, 0)).unwrap();
    // A delegation in a different pool — must not contribute.
    store.set_delegation(p2, alice, delegation(9_999, 0)).unwrap();

    assert_eq!(store.sum_delegations_for_pool(p1).unwrap(), 1_000);
    assert_eq!(store.sum_delegations_for_pool(p2).unwrap(), 9_999);
    // Empty pool sums to zero.
    assert_eq!(store.sum_delegations_for_pool(pool(99)).unwrap(), 0);
}

/// Aggregate primitive used by future "total stake" RPC endpoints.
/// Sums every row owned by a staker across pools; missing staker
/// returns 0 (consistent with `get_balance` / `get_delegation`).
#[tokio::test]
async fn total_delegated_principal_for_staker_aggregates_across_pools() {
    let store = fresh_store();
    let p1 = pool(1);
    let p2 = pool(2);
    let alice = addr(10);
    let bob = addr(11);

    store.set_delegation(p1, alice, delegation(300, 0)).unwrap();
    store.set_delegation(p2, alice, delegation(300, 0)).unwrap();
    store.set_delegation(p1, bob, delegation(9_999, 0)).unwrap(); // unrelated staker

    assert_eq!(store.total_delegated_principal_for_staker(alice).unwrap(), 600);
    assert_eq!(store.total_delegated_principal_for_staker(bob).unwrap(), 9_999);
    assert_eq!(store.total_delegated_principal_for_staker(addr(99)).unwrap(), 0);
}

/// Saturation is caught and reported as an error rather than silently
/// truncating. Real-world total supply is far below u64::MAX; an
/// overflow here indicates corruption upstream and we want callers to
/// see the failure loudly.
#[tokio::test]
async fn total_delegated_principal_for_staker_overflow_errors() {
    let store = fresh_store();
    let alice = addr(10);
    store.set_delegation(pool(1), alice, delegation(u64::MAX, 0)).unwrap();
    store.set_delegation(pool(2), alice, delegation(1, 0)).unwrap();

    let err = store
        .total_delegated_principal_for_staker(alice)
        .expect_err("total overflow must error");
    assert!(format!("{:?}", err).contains("overflow"), "got: {:?}", err);
}

/// Symmetric to `iter_delegations_for_staker`: list every staker
/// who's delegated into a given pool. The pool-mismatched row is
/// filtered out.
#[tokio::test]
async fn iter_delegators_for_pool_filters_by_pool() {
    let store = fresh_store();
    let p1 = pool(1);
    let p2 = pool(2);
    let alice = addr(10);
    let bob = addr(11);

    store.set_delegation(p1, alice, delegation(300, 1)).unwrap();
    store.set_delegation(p1, bob, delegation(300, 2)).unwrap();
    // A row in p2 — must not contribute to p1's listing.
    store.set_delegation(p2, alice, delegation(9_999, 3)).unwrap();

    let mut listed = store.iter_delegators_for_pool(p1).unwrap();
    listed.sort_by_key(|(addr, _)| *addr);
    assert_eq!(listed.len(), 2);
    assert_eq!(listed[0].0, alice);
    assert_eq!(listed[0].1.principal, 300);
    assert_eq!(listed[1].0, bob);
    assert_eq!(listed[1].1.principal, 300);

    let listed = store.iter_delegators_for_pool(p2).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].0, alice);
    assert_eq!(listed[0].1.principal, 9_999);

    // Nonexistent pool → empty listing.
    assert!(store.iter_delegators_for_pool(pool(99)).unwrap().is_empty());
}

#[tokio::test]
async fn iter_delegations_for_staker_filters_by_address() {
    let store = fresh_store();
    let p1 = pool(1);
    let p2 = pool(2);
    let alice = addr(10);
    let bob = addr(11);

    store.set_delegation(p1, alice, delegation(100, 5)).unwrap();
    store.set_delegation(p2, alice, delegation(200, 6)).unwrap();
    store.set_delegation(p1, bob, delegation(300, 7)).unwrap();

    let mut listed = store.iter_delegations_for_staker(alice).unwrap();
    listed.sort_by_key(|(pool, _)| *pool);
    assert_eq!(listed.len(), 2);
    assert_eq!(listed[0].0, p1);
    assert_eq!(listed[0].1.principal, 100);
    assert_eq!(listed[1].0, p2);
    assert_eq!(listed[1].1.principal, 200);

    let listed = store.iter_delegations_for_staker(bob).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].0, p1);
    assert_eq!(listed[0].1.principal, 300);

    // Nonexistent staker → empty list.
    assert!(store.iter_delegations_for_staker(addr(99)).unwrap().is_empty());
}

/// Stage 9 invariant: balances and delegations live in separate
/// column families. A staker's USDC balance must be unaffected by
/// their delegations and vice versa.
#[tokio::test]
async fn delegations_and_balances_are_independent() {
    use types::object::CoinType;
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.set_delegation(p, alice, delegation(1_000, 0)).unwrap();
    assert_eq!(store.get_balance(alice, CoinType::Soma).unwrap(), 0);
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 0);

    store.set_balance(alice, CoinType::Soma, 7_777).unwrap();
    assert_eq!(store.get_delegation(p, alice).unwrap().principal, 1_000);
}
