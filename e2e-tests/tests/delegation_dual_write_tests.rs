// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Stage 9d-C1 integrity tests: the F1-shaped `delegations` column
//! family stays in sync with the canonical `StakedSomaV1` object set
//! across user AddStake/WithdrawStake and epoch reward distribution.
//!
//! Schema is ONE row per (pool_id, staker). Repeat stakes from the
//! same staker into the same validator collapse into a single row;
//! per-epoch validator commission credits accumulate into that one
//! row rather than spawning new rows.

use std::collections::BTreeMap;

use test_cluster::TestClusterBuilder;
use types::base::SomaAddress;
use types::object::ObjectID;
use types::system_state::staking::Delegation;

/// Read the entire `delegations` column family into a flat map.
fn collect_delegations(
    test_cluster: &test_cluster::TestCluster,
) -> BTreeMap<(ObjectID, SomaAddress), Delegation> {
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

/// Stage 9d-C1 invariant: after the chain advances at least one
/// epoch, the validator-reward dual-write fires and grows the
/// principal of each validator's existing self-stake row. Row count
/// stays the same because F1 schema collapses repeat stakes into one
/// row per (pool, staker).
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

    // Genesis seeds the delegations table with the validators'
    // initial self-stakes (one row per validator).
    let pre_delegations = collect_delegations(&test_cluster);
    assert!(
        !pre_delegations.is_empty(),
        "genesis backfill: delegations table must carry every genesis \
         self-stake from epoch 0 — empty table indicates the backfill didn't fire",
    );
    let pre_count = pre_delegations.len();
    let pre_total: u128 =
        pre_delegations.values().map(|d| d.principal as u128).sum();

    test_cluster
        .wait_for_epoch_with_timeout(Some(1), std::time::Duration::from_secs(30))
        .await;

    let post_delegations = collect_delegations(&test_cluster);
    let post_total: u128 =
        post_delegations.values().map(|d| d.principal as u128).sum();

    // F1 schema: validator commission credits accumulate into the
    // existing self-stake row rather than spawning new rows. So the
    // count is stable, but the total principal grows.
    assert_eq!(
        post_delegations.len(),
        pre_count,
        "post-epoch row count must match: F1 schema collapses repeat stakes \
         into one row per (pool, staker). pre={}, post={}",
        pre_count,
        post_delegations.len(),
    );
    assert!(
        post_total > pre_total,
        "post-epoch total principal must exceed genesis baseline (validator \
         commission credits): pre={}, post={}",
        pre_total,
        post_total,
    );

    for (key, post) in &post_delegations {
        assert!(
            post.principal > 0,
            "delegation row {:?} has zero principal — should have been pruned",
            key
        );
        let pre = pre_delegations
            .get(key)
            .expect("post-epoch row must mirror a genesis row by key");
        assert!(
            post.principal >= pre.principal,
            "validator {:?} commission credit must not shrink principal: pre={} post={}",
            key,
            pre.principal,
            post.principal,
        );
    }
}
