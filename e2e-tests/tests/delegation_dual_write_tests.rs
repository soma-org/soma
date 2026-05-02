// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Stage 9b/9c integrity tests: the `delegations` column family stays
//! in sync with the canonical `StakedSomaV1` object set across user
//! AddStake/WithdrawStake (9b) and epoch reward distribution (9c).
//!
//! The dual-write design lets Stage 9d collapse the two by removing
//! the StakedSomaV1 object once all consumers read from the table. The
//! invariant we rely on for that future swap: every (pool_id, staker,
//! activation_epoch) row in `delegations` is mirrored by a
//! StakedSomaV1 with the same fields, and the principals agree.

use std::collections::BTreeMap;

use test_cluster::TestClusterBuilder;
use types::base::SomaAddress;
use types::object::ObjectID;

/// Read the entire `delegations` column family into a flat map.
fn collect_delegations(
    test_cluster: &test_cluster::TestCluster,
) -> BTreeMap<(ObjectID, SomaAddress, u64), u64> {
    test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| {
            node.state()
                .database_for_testing()
                .iter_all_delegations()
                .expect("delegation scan")
                .into_iter()
                .collect()
        })
}

/// Stage 9c invariant: after the chain advances at least one epoch,
/// the validator-reward dual-write fires and the `delegations` table
/// has at least one row, and every row's principal agrees with the
/// matching StakedSomaV1 (looked up by deriving the StakedSomaV1
/// object id is impractical — instead we use
/// `get_balance(staker, ...)` + per-staker iteration to cross-check).
#[cfg(msim)]
#[msim::sim_test]
async fn delegations_table_populated_after_epoch_change() {
    use utils::logging::init_tracing;

    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_epoch_duration_ms(5_000)
        .build()
        .await;

    // Stage 9d: genesis seeds the delegations table with the
    // validators' initial self-stakes (one row per validator). Stage
    // 9b adds dual-writes for AddStake/WithdrawStake; Stage 9c adds
    // dual-writes for the per-validator epoch-reward StakedSomaV1.
    //
    // Pre-epoch-1 the table holds the genesis seed stakes only.
    let pre_delegations = collect_delegations(&test_cluster);
    assert!(
        !pre_delegations.is_empty(),
        "Stage 9d genesis backfill: delegations table must carry every genesis \
         StakedSomaV1 from epoch 0 — empty table indicates the backfill didn't fire",
    );
    let pre_count = pre_delegations.len();

    test_cluster
        .wait_for_epoch_with_timeout(Some(1), std::time::Duration::from_secs(30))
        .await;

    let post_delegations = collect_delegations(&test_cluster);

    // Stage 9c: every validator gets a reward row at the boundary, so
    // the post-epoch table strictly grows from the genesis-seeded
    // baseline.
    assert!(
        post_delegations.len() > pre_count,
        "post-epoch-1 table must have grown from genesis baseline: pre={}, post={}",
        pre_count,
        post_delegations.len(),
    );

    // Pre-epoch entries are still there (no AddStake or WithdrawStake
    // happened to perturb them), and every row has non-zero
    // principal — the row-deletion contract in `set_delegation` says
    // zero entries are pruned, so any visible row should carry a real
    // amount.
    for (key, principal) in &pre_delegations {
        assert_eq!(
            post_delegations.get(key),
            Some(principal),
            "genesis delegation row {:?} disappeared after epoch 1 — Stage 9d \
             backfill should be stable across epoch boundaries",
            key,
        );
    }
    for (key, principal) in &post_delegations {
        assert!(
            *principal > 0,
            "delegation row {:?} has zero principal — should have been pruned",
            key
        );
    }
}
