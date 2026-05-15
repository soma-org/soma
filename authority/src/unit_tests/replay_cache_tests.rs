// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the executed-transaction-digest replay-protection
//! cache (Stage 5.5b: storage layer).
//!
//! These tests exercise the perpetual store's `(epoch, digest)`
//! column family and the `was_transaction_executed_in_last_epoch`
//! lookup directly. End-to-end replay rejection through the
//! transaction-validation path is verified in Stage 5.5c.

use std::sync::Arc;

use store::Map as _;
use tempfile::tempdir;
use types::digests::TransactionDigest;

use crate::authority_store::AuthorityStore;
use crate::authority_store_tables::AuthorityPerpetualTables;

fn fresh_store() -> Arc<AuthorityStore> {
    let dir = tempdir().unwrap();
    let perpetual_tables = Arc::new(AuthorityPerpetualTables::open(dir.path(), None));
    AuthorityStore::open_no_genesis(perpetual_tables).unwrap()
}

fn digest(seed: u8) -> TransactionDigest {
    TransactionDigest::new([seed; 32])
}

/// Direct insert via the column family — mirrors what
/// `write_one_transaction_outputs` does atomically with effects.
fn insert(store: &Arc<AuthorityStore>, epoch: u64, digest: TransactionDigest) {
    let mut batch = store.perpetual_tables.executed_transaction_digests.batch();
    batch
        .insert_batch(
            &store.perpetual_tables.executed_transaction_digests,
            [((epoch, digest), ())],
        )
        .unwrap();
    batch.write().unwrap();
}

/// At epoch 0 there is no previous epoch — must always return false,
/// even if a digest somehow ended up in the cache. This avoids an
/// `epoch - 1` underflow on a fresh chain.
#[tokio::test]
async fn epoch_zero_always_returns_false() {
    let store = fresh_store();
    let d = digest(1);
    // Even if we (somehow) wrote a digest at epoch 0, the lookup at
    // epoch 0 must short-circuit to false.
    insert(&store, 0, d);
    assert!(!store.was_transaction_executed_in_last_epoch(&d, 0).unwrap());
}

/// A digest written at epoch N is visible from epoch N+1's lookup
/// (which probes prev-epoch). This is the core replay-rejection signal.
#[tokio::test]
async fn prev_epoch_digest_is_found() {
    let store = fresh_store();
    let d = digest(1);
    insert(&store, 5, d);
    assert!(store.was_transaction_executed_in_last_epoch(&d, 6).unwrap());
}

/// A digest written at epoch N is NOT visible from epoch N's own
/// lookup (current-epoch hits go through a different code path —
/// `is_tx_already_executed` — which represents re-votes, not replays).
#[tokio::test]
async fn same_epoch_lookup_is_negative() {
    let store = fresh_store();
    let d = digest(1);
    insert(&store, 5, d);
    // Asking "was this executed in the last epoch?" at epoch 5 is
    // really asking about epoch 4, which is empty.
    assert!(!store.was_transaction_executed_in_last_epoch(&d, 5).unwrap());
}

/// A digest written at epoch N is NOT visible from epoch N+2's
/// lookup. This is the windowing property that bounds the cache:
/// once a tx's validity window has expired (max_epoch ≤ current - 1),
/// the structural validity check rejects it before this lookup
/// matters; once the previous epoch advances past it, prune drops it.
#[tokio::test]
async fn two_epochs_old_digest_is_not_found() {
    let store = fresh_store();
    let d = digest(1);
    insert(&store, 5, d);
    assert!(!store.was_transaction_executed_in_last_epoch(&d, 7).unwrap());
}

/// A digest only present in the *current* epoch's prefix is not
/// visible to the prev-epoch lookup. Defends against accidentally
/// indexing the wrong column family.
#[tokio::test]
async fn current_epoch_only_does_not_leak_to_prev_lookup() {
    let store = fresh_store();
    let d = digest(1);
    insert(&store, 6, d);
    assert!(!store.was_transaction_executed_in_last_epoch(&d, 6).unwrap());
}

/// Different digests at the same epoch don't false-positive each other.
#[tokio::test]
async fn distinct_digests_do_not_collide() {
    let store = fresh_store();
    let a = digest(1);
    let b = digest(2);
    insert(&store, 5, a);
    assert!(store.was_transaction_executed_in_last_epoch(&a, 6).unwrap());
    assert!(!store.was_transaction_executed_in_last_epoch(&b, 6).unwrap());
}

/// Same digest at multiple epochs (legitimately re-submitted with a
/// fresh validity window after the original expired): both entries
/// coexist because the key is `(epoch, digest)`. Lookup against the
/// most recent prev-epoch sees the recent one.
#[tokio::test]
async fn same_digest_across_epochs_does_not_collide() {
    let store = fresh_store();
    let d = digest(1);
    insert(&store, 5, d);
    insert(&store, 8, d);
    // From epoch 9, prev = 8, hit.
    assert!(store.was_transaction_executed_in_last_epoch(&d, 9).unwrap());
    // From epoch 6, prev = 5, also hit.
    assert!(store.was_transaction_executed_in_last_epoch(&d, 6).unwrap());
    // From epoch 7, prev = 6, miss — neither write was at epoch 6.
    assert!(!store.was_transaction_executed_in_last_epoch(&d, 7).unwrap());
}

/// Lookup on an empty cache always returns false.
#[tokio::test]
async fn empty_cache_lookup_is_false() {
    let store = fresh_store();
    assert!(!store.was_transaction_executed_in_last_epoch(&digest(7), 1).unwrap());
    assert!(!store.was_transaction_executed_in_last_epoch(&digest(7), 100).unwrap());
}

// ---------------------------------------------------------------------
// Stage 5.5d: prune_executed_transaction_digests
// ---------------------------------------------------------------------

/// Helper: count all entries in the column family. Range-iter so we
/// don't need to know specific keys.
fn count_all(store: &Arc<AuthorityStore>) -> usize {
    store.perpetual_tables.executed_transaction_digests.safe_iter().count()
}

#[tokio::test]
async fn prune_at_epoch_zero_is_noop() {
    // At epoch 0 there's no previous epoch — nothing to retain or
    // drop. Pruning must be a no-op even if entries exist.
    let store = fresh_store();
    insert(&store, 0, digest(1));
    store.prune_executed_transaction_digests(0).unwrap();
    assert_eq!(count_all(&store), 1);
}

#[tokio::test]
async fn prune_at_epoch_one_is_noop() {
    // At epoch 1, prev = 0 — we want to retain epoch 0 + epoch 1, so
    // there's nothing to drop.
    let store = fresh_store();
    insert(&store, 0, digest(1));
    insert(&store, 1, digest(2));
    store.prune_executed_transaction_digests(1).unwrap();
    assert_eq!(count_all(&store), 2);
}

#[tokio::test]
async fn prune_keeps_current_and_previous_epoch() {
    let store = fresh_store();
    insert(&store, 3, digest(1)); // should be dropped (older than prev)
    insert(&store, 4, digest(2)); // retained (prev epoch)
    insert(&store, 5, digest(3)); // retained (current epoch)

    store.prune_executed_transaction_digests(5).unwrap();

    assert_eq!(count_all(&store), 2, "exactly 2 entries should remain");
    // Verify the right ones survived.
    assert!(!store.was_transaction_executed_in_last_epoch(&digest(1), 4).unwrap());
    assert!(store.was_transaction_executed_in_last_epoch(&digest(2), 5).unwrap());
    // Current-epoch entry isn't visible via the prev-epoch lookup, but
    // it's still in the cache:
    assert!(
        store
            .perpetual_tables
            .executed_transaction_digests
            .contains_key(&(5, digest(3)))
            .unwrap()
    );
}

#[tokio::test]
async fn prune_drops_multiple_old_epochs() {
    let store = fresh_store();
    for epoch in 0..10 {
        insert(&store, epoch, digest(epoch as u8));
    }
    // Prune at epoch 10: retain 9 + 10. Everything from 0..=8 dropped.
    store.prune_executed_transaction_digests(10).unwrap();
    assert_eq!(count_all(&store), 1, "only epoch 9 entry survives (no epoch 10 was inserted)");
    assert!(
        store
            .perpetual_tables
            .executed_transaction_digests
            .contains_key(&(9, digest(9)))
            .unwrap()
    );
}

#[tokio::test]
async fn prune_is_idempotent() {
    // Running prune twice in a row must not change state. Defends
    // against epoch-boundary races where the pruner is invoked multiple
    // times for the same epoch.
    let store = fresh_store();
    insert(&store, 3, digest(1));
    insert(&store, 4, digest(2));
    insert(&store, 5, digest(3));

    store.prune_executed_transaction_digests(5).unwrap();
    let count_after_first = count_all(&store);
    store.prune_executed_transaction_digests(5).unwrap();
    assert_eq!(count_all(&store), count_after_first);
}

#[tokio::test]
async fn prune_does_not_affect_other_column_families() {
    // The pruner uses an epoch-prefixed range delete on a single
    // column family. Confirm it doesn't accidentally clobber adjacent
    // tables.
    let store = fresh_store();
    let alice = types::base::SomaAddress::new([1u8; 32]);
    store.set_balance(alice, types::object::CoinType::Usdc, 12345).unwrap();
    insert(&store, 3, digest(1));

    store.prune_executed_transaction_digests(5).unwrap();

    assert_eq!(store.get_balance(alice, types::object::CoinType::Usdc).unwrap(), 12345);
}
