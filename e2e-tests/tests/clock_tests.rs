// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the on-chain Clock object (`0x6`).
//!
//! These tests run a real `TestCluster` under the msim deterministic
//! simulator and verify that the Clock object's `timestamp_ms` is updated
//! by the consensus commit prologue every commit, with all validators
//! agreeing on the value.

use std::time::Duration;

use test_cluster::TestClusterBuilder;
use tokio::time::sleep;
use tracing::info;
use types::CLOCK_OBJECT_ID;
use types::effects::TransactionEffectsAPI as _;
use types::transaction::{TransactionData, TransactionKind};
use utils::logging::init_tracing;

/// Read the Clock timestamp from the fullnode's authority state. Sync —
/// uses the underlying ObjectStore directly so we can call it from
/// inside `with(...)` closures.
fn read_clock_ts_from_fullnode(test_cluster: &test_cluster::TestCluster) -> u64 {
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let obj = node
            .state()
            .get_object_store()
            .get_object(&CLOCK_OBJECT_ID)
            .expect("Clock object must exist");
        obj.clock_timestamp_ms()
    })
}

/// Build a transfer transaction the test cluster can execute. Used
/// purely to drive consensus commits — every commit triggers a prologue,
/// which is what we're observing. Mirrors the pattern used in
/// `basic_checkpoints_integration_test`.
async fn drive_one_commit(test_cluster: &test_cluster::TestCluster) {
    let addrs = test_cluster.wallet.get_addresses();
    let sender = addrs[0];
    let recipient = addrs[1];
    let gas =
        test_cluster.wallet.get_one_gas_object_owned_by_address(sender).await.unwrap().unwrap();
    let tx_data = TransactionData::new(
        TransactionKind::Transfer {
            coins: vec![gas],
            amounts: Some(1000).map(|a| vec![a]),
            recipients: vec![recipient],
        },
        sender,
        vec![gas],
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(
        response.effects.status().is_ok(),
        "driver tx must succeed; status={:?}",
        response.effects.status()
    );
}

/// The Clock advances under real consensus traffic. Each user
/// transaction we submit triggers a consensus commit, which runs a
/// ConsensusCommitPrologueV1 that mutates the Clock.
#[cfg(msim)]
#[msim::sim_test]
async fn test_clock_advances_under_consensus() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    // Genesis Clock starts at 0.
    let initial = read_clock_ts_from_fullnode(&test_cluster);
    assert_eq!(initial, 0, "genesis Clock timestamp must be 0");

    // Drive a commit — this causes the prologue to fire and the Clock
    // to be mutated with the consensus-agreed timestamp.
    drive_one_commit(&test_cluster).await;

    let after_first = read_clock_ts_from_fullnode(&test_cluster);
    assert!(
        after_first > 0,
        "Clock must have advanced past 0 after a consensus commit (got {})",
        after_first,
    );

    // Drive another commit; Clock must advance further (or at least
    // stay the same — strictly greater is the common case but a same-ms
    // commit is theoretically allowed).
    drive_one_commit(&test_cluster).await;

    let after_second = read_clock_ts_from_fullnode(&test_cluster);
    assert!(
        after_second >= after_first,
        "Clock must not regress: was {}, now {}",
        after_first,
        after_second,
    );

    info!(
        initial = initial,
        first = after_first,
        second = after_second,
        "Clock advanced as expected under consensus"
    );
}

/// All validators must agree on the Clock state. This is the
/// Byzantine-agreement property: `commit_timestamp_ms` comes from
/// consensus output, so honest validators executing the same prologue
/// produce identical Clock state.
#[cfg(msim)]
#[msim::sim_test]
async fn test_clock_state_agrees_across_validators() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    // Drive a few commits so every validator has had a chance to
    // process at least one prologue.
    for _ in 0..3 {
        drive_one_commit(&test_cluster).await;
    }

    // Give state a moment to settle across validators.
    sleep(Duration::from_secs(1)).await;

    // Snapshot every validator's view of the Clock alongside the
    // version. A mismatch in version is a clearer signal of execution
    // divergence than a mismatch in timestamp alone.
    let snapshots: Vec<(u64, types::object::Version)> = test_cluster
        .swarm
        .validator_node_handles()
        .into_iter()
        .map(|handle| {
            handle.with(|node| {
                let obj = node
                    .state()
                    .get_object_store()
                    .get_object(&CLOCK_OBJECT_ID)
                    .expect("Clock must exist on validator");
                (obj.clock_timestamp_ms(), obj.version())
            })
        })
        .collect();

    assert!(snapshots.len() >= 4, "expected at least 4 validators in the cluster");
    let min_ts = snapshots.iter().map(|(t, _)| *t).min().unwrap();
    assert!(min_ts > 0, "every validator's Clock must be past 0 after commits run");

    // Strongest property: any two validators that are at the same Clock
    // version MUST be at the same timestamp. Same version means they
    // executed the same prologue and the Clock state must be byte-
    // identical. A divergence here is an execution-correctness bug.
    let mut by_version: std::collections::BTreeMap<types::object::Version, u64> =
        std::collections::BTreeMap::new();
    for (ts, v) in &snapshots {
        if let Some(prev) = by_version.insert(*v, *ts) {
            assert_eq!(
                prev, *ts,
                "two validators at Clock version {:?} disagree on timestamp ({} vs {})",
                v, prev, ts
            );
        }
    }

    info!(snapshots = ?snapshots, "all validators agree on Clock state per version");
}

/// The Clock continues to advance across an epoch boundary. The
/// ChangeEpoch path doesn't reset it, and the new epoch's prologues
/// keep mutating it.
#[cfg(msim)]
#[msim::sim_test]
async fn test_clock_survives_epoch_change() {
    init_tracing();

    let epoch_duration_ms = 10_000;
    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_epoch_duration_ms(epoch_duration_ms)
        .build()
        .await;

    // Drive some commits in epoch 0.
    drive_one_commit(&test_cluster).await;
    drive_one_commit(&test_cluster).await;

    let pre_epoch = read_clock_ts_from_fullnode(&test_cluster);
    assert!(pre_epoch > 0, "Clock must have advanced in epoch 0");

    // Cross an epoch boundary.
    test_cluster.wait_for_epoch(Some(1)).await;

    // Drive more commits in the new epoch so the prologue runs there.
    drive_one_commit(&test_cluster).await;
    drive_one_commit(&test_cluster).await;

    let post_epoch = read_clock_ts_from_fullnode(&test_cluster);
    assert!(
        post_epoch > pre_epoch,
        "Clock must continue advancing past the epoch boundary: pre={} post={}",
        pre_epoch,
        post_epoch,
    );

    info!(pre_epoch = pre_epoch, post_epoch = post_epoch, "Clock survived epoch change");
}
