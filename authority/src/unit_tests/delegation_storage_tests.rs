// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the delegation-balance storage layer
//! (`AuthorityStore::*_delegation` methods + the `delegations` column
//! family).
//!
//! Stage 9a is additive: the table exists with accessors, but no
//! production execution path writes to it yet. These tests exercise
//! the storage primitives in isolation so that Stage 9b (which will
//! route AddStake/WithdrawStake through this table) can rely on them
//! being correct.

use std::sync::Arc;

use tempfile::tempdir;
use types::base::SomaAddress;
use types::object::ObjectID;

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

// ---------------------------------------------------------------------
// get_delegation / set_delegation
// ---------------------------------------------------------------------

/// A delegation row that has never been written reads as zero, not as
/// a missing-key error. Mirrors `get_balance` semantics — the rest of
/// the staking layer will rely on this contract once Stage 9b lands.
#[tokio::test]
async fn get_delegation_missing_returns_zero() {
    let store = fresh_store();
    assert_eq!(store.get_delegation(pool(1), addr(1), 0).unwrap(), 0);
    assert_eq!(store.get_delegation(pool(1), addr(2), 5).unwrap(), 0);
}

#[tokio::test]
async fn set_delegation_round_trips() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.set_delegation(p, alice, 7, 1_000_000).unwrap();
    assert_eq!(store.get_delegation(p, alice, 7).unwrap(), 1_000_000);
}

/// Setting a delegation row to zero deletes the row entirely. Important
/// because `get_delegation` returns 0 for missing entries — keeping a
/// zero-valued row would still appear in `iter_delegations_for_staker`
/// scans and waste storage. Withdrawing a stake fully should leave no
/// trace.
#[tokio::test]
async fn set_delegation_zero_deletes_row() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.set_delegation(p, alice, 7, 500).unwrap();
    assert_eq!(store.iter_delegations_for_staker(alice).unwrap().len(), 1);

    store.set_delegation(p, alice, 7, 0).unwrap();
    assert_eq!(store.get_delegation(p, alice, 7).unwrap(), 0);
    assert!(
        store.iter_delegations_for_staker(alice).unwrap().is_empty(),
        "zero-valued row must be deleted, not kept",
    );
}

/// The triplet key separates rows by activation epoch — same staker
/// into the same pool but in different epochs locks in different
/// exchange rates and so must be tracked as distinct rows.
#[tokio::test]
async fn delegations_for_same_pool_and_staker_distinguish_by_epoch() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.set_delegation(p, alice, 5, 100).unwrap();
    store.set_delegation(p, alice, 6, 200).unwrap();
    store.set_delegation(p, alice, 7, 300).unwrap();

    assert_eq!(store.get_delegation(p, alice, 5).unwrap(), 100);
    assert_eq!(store.get_delegation(p, alice, 6).unwrap(), 200);
    assert_eq!(store.get_delegation(p, alice, 7).unwrap(), 300);

    let listed = store.iter_delegations_for_staker(alice).unwrap();
    assert_eq!(listed.len(), 3, "all three epochs surface independently");
}

// ---------------------------------------------------------------------
// apply_delegation_delta
// ---------------------------------------------------------------------

#[tokio::test]
async fn apply_delegation_delta_increments() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    let after = store.apply_delegation_delta(p, alice, 7, 1_000).unwrap();
    assert_eq!(after, 1_000);

    let after = store.apply_delegation_delta(p, alice, 7, 500).unwrap();
    assert_eq!(after, 1_500);
}

/// Underflow is a hard error — withdrawing more than was staked must
/// not silently zero the row.
#[tokio::test]
async fn apply_delegation_delta_underflow_errors() {
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.set_delegation(p, alice, 7, 500).unwrap();
    let err = store
        .apply_delegation_delta(p, alice, 7, -1_000)
        .expect_err("underflow must error");
    assert!(format!("{:?}", err).contains("underflow"), "got: {:?}", err);
    // Row unchanged after the failed apply.
    assert_eq!(store.get_delegation(p, alice, 7).unwrap(), 500);
}

// ---------------------------------------------------------------------
// sum_delegations_for_pool / iter_delegations_for_staker
// ---------------------------------------------------------------------

#[tokio::test]
async fn sum_delegations_for_pool_aggregates_across_stakers_and_epochs() {
    let store = fresh_store();
    let p1 = pool(1);
    let p2 = pool(2);
    let alice = addr(10);
    let bob = addr(11);
    let carol = addr(12);

    store.set_delegation(p1, alice, 5, 100).unwrap();
    store.set_delegation(p1, alice, 6, 200).unwrap(); // same staker, diff epoch
    store.set_delegation(p1, bob, 5, 300).unwrap();
    store.set_delegation(p1, carol, 7, 400).unwrap();
    // A delegation in a different pool — must not contribute.
    store.set_delegation(p2, alice, 5, 9_999).unwrap();

    assert_eq!(store.sum_delegations_for_pool(p1).unwrap(), 1_000);
    assert_eq!(store.sum_delegations_for_pool(p2).unwrap(), 9_999);
    // Empty pool sums to zero.
    assert_eq!(store.sum_delegations_for_pool(pool(99)).unwrap(), 0);
}

/// Aggregate primitive used by future "total stake" RPC endpoints.
/// Sums every row owned by a staker across pools and epochs; missing
/// staker returns 0 (consistent with `get_balance` / `get_delegation`).
#[tokio::test]
async fn total_delegated_principal_for_staker_aggregates_across_pools_and_epochs() {
    let store = fresh_store();
    let p1 = pool(1);
    let p2 = pool(2);
    let alice = addr(10);
    let bob = addr(11);

    store.set_delegation(p1, alice, 5, 100).unwrap();
    store.set_delegation(p1, alice, 6, 200).unwrap(); // same pool, diff epoch
    store.set_delegation(p2, alice, 5, 300).unwrap(); // diff pool
    store.set_delegation(p1, bob, 5, 9_999).unwrap(); // unrelated staker

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
    store.set_delegation(pool(1), alice, 5, u64::MAX).unwrap();
    store.set_delegation(pool(2), alice, 5, 1).unwrap();

    let err = store
        .total_delegated_principal_for_staker(alice)
        .expect_err("total overflow must error");
    assert!(format!("{:?}", err).contains("overflow"), "got: {:?}", err);
}

/// Symmetric to `iter_delegations_for_staker`: list every staker
/// who's delegated into a given pool. Different stakers, different
/// activation epochs, all surface; the pool-mismatched row is filtered
/// out.
#[tokio::test]
async fn iter_delegators_for_pool_filters_by_pool() {
    let store = fresh_store();
    let p1 = pool(1);
    let p2 = pool(2);
    let alice = addr(10);
    let bob = addr(11);

    store.set_delegation(p1, alice, 5, 100).unwrap();
    store.set_delegation(p1, alice, 6, 200).unwrap(); // same staker, different epoch
    store.set_delegation(p1, bob, 5, 300).unwrap();
    // A row in p2 — must not contribute to p1's listing.
    store.set_delegation(p2, alice, 5, 9_999).unwrap();

    let mut listed = store.iter_delegators_for_pool(p1).unwrap();
    listed.sort();
    assert_eq!(
        listed,
        vec![(alice, 5, 100), (alice, 6, 200), (bob, 5, 300)],
        "every p1 delegation surfaces; p2 row excluded"
    );

    let listed = store.iter_delegators_for_pool(p2).unwrap();
    assert_eq!(listed, vec![(alice, 5, 9_999)]);

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

    store.set_delegation(p1, alice, 5, 100).unwrap();
    store.set_delegation(p2, alice, 6, 200).unwrap();
    store.set_delegation(p1, bob, 5, 300).unwrap();

    let mut listed = store.iter_delegations_for_staker(alice).unwrap();
    listed.sort();
    assert_eq!(listed, vec![(p1, 5, 100), (p2, 6, 200)]);

    let listed = store.iter_delegations_for_staker(bob).unwrap();
    assert_eq!(listed, vec![(p1, 5, 300)]);

    // Nonexistent staker → empty list.
    assert!(store.iter_delegations_for_staker(addr(99)).unwrap().is_empty());
}

/// Stage 9 invariant: balances and delegations live in separate
/// column families. A staker's USDC balance must be unaffected by
/// their delegations and vice versa. (Especially load-bearing once
/// Stage 9b starts emitting Withdraw events for the SOMA principal —
/// we want to be sure nothing leaks across the two tables.)
#[tokio::test]
async fn delegations_and_balances_are_independent() {
    use types::object::CoinType;
    let store = fresh_store();
    let p = pool(1);
    let alice = addr(2);

    store.set_delegation(p, alice, 5, 1_000).unwrap();
    assert_eq!(store.get_balance(alice, CoinType::Soma).unwrap(), 0);
    assert_eq!(store.get_balance(alice, CoinType::Usdc).unwrap(), 0);

    store.set_balance(alice, CoinType::Soma, 7_777).unwrap();
    assert_eq!(store.get_delegation(p, alice, 5).unwrap(), 1_000);
}
