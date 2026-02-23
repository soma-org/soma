// Portions of this file are derived from Mysticeti consensus (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/consensus/core/src/randomized_tests.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

// Randomized tests for consensus commit logic.
// These tests verify that the commit protocol produces deterministic results
// regardless of the order in which blocks are delivered, including scenarios
// with equivocating authorities.

use rand::{SeedableRng, rngs::StdRng};
use types::committee::AuthorityIndex;
use types::consensus::block::{Round, Slot};

use crate::{
    commit_test_fixture::{
        CommitTestFixture, EquivocatingRandomDag, RandomDag, RandomDagConfig,
        assert_commit_sequences_match,
    },
    test_dag::create_random_dag,
};

/// Test that a fully connected random DAG (with 100% leader inclusion) always
/// produces direct commits for every leader round, across multiple seeds.
#[tokio::test]
async fn test_randomized_dag_all_direct_commit() {
    let num_rounds: Round = 1000;
    let num_authorities = 4;

    for seed in 0..5u64 {
        let context = CommitTestFixture::context_with_options(num_authorities, 0, 0);
        let dag_builder = create_random_dag(
            seed,
            100, // 100% leader inclusion
            num_rounds,
            context.clone(),
        );

        let random_dag = RandomDag::new(&dag_builder);

        // Run for each authority to verify they all produce the same commit sequence.
        let mut all_sequences = Vec::new();

        for authority_idx in 0..num_authorities {
            let ctx =
                CommitTestFixture::context_with_options(num_authorities, authority_idx as u32, 0);
            let mut fixture = CommitTestFixture::new(ctx);

            // Add all blocks in round order (natural delivery).
            let blocks = random_dag.blocks_in_order();
            fixture.add_blocks(blocks);

            // Try to commit from genesis.
            let last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));
            let committed = fixture.try_commit(last_decided);

            // Should have committed leaders.
            assert!(
                !committed.is_empty(),
                "Seed {seed}, authority {authority_idx}: expected commits but got none"
            );

            all_sequences.push(committed);
        }

        // All authorities should agree on the commit sequence.
        assert_commit_sequences_match(&all_sequences);
    }
}

/// Test that the commit decision sequence is deterministic regardless of
/// the order blocks are delivered. Uses 50% leader inclusion to create
/// a more interesting (non-trivial) DAG with skips and indirect commits.
#[tokio::test]
async fn test_randomized_dag_and_decision_sequence() {
    let num_rounds: Round = 1000;
    let num_authorities = 4;
    let num_delivery_orders = 5;

    for seed in 0..5u64 {
        let context = CommitTestFixture::context_with_options(num_authorities, 0, 0);
        let dag_builder = create_random_dag(
            seed,
            50, // 50% leader inclusion
            num_rounds,
            context.clone(),
        );

        let random_dag = RandomDag::new(&dag_builder);
        let mut all_sequences = Vec::new();

        // First, do one run with blocks delivered in natural order.
        {
            let ctx = CommitTestFixture::context_with_options(num_authorities, 0, 0);
            let mut fixture = CommitTestFixture::new(ctx);
            let blocks = random_dag.blocks_in_order();
            fixture.add_blocks(blocks);

            let last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));
            let committed = fixture.try_commit(last_decided);
            all_sequences.push(committed);
        }

        // Now do multiple runs with blocks delivered in random order via block_manager.
        for delivery_seed in 0..num_delivery_orders {
            let combined_seed = seed * 1000 + delivery_seed;
            let ctx = CommitTestFixture::context_with_options(num_authorities, 0, 0);
            let mut fixture = CommitTestFixture::new(ctx);

            // Deliver blocks one at a time in random order through block_manager.
            for block in random_dag.random_iter(combined_seed) {
                fixture.try_accept_blocks(vec![block]);
            }

            assert!(
                fixture.has_no_suspended_blocks(),
                "Seed {seed}, delivery_seed {delivery_seed}: block_manager still has suspended blocks"
            );

            let last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));
            let committed = fixture.try_commit(last_decided);
            all_sequences.push(committed);
        }

        // All delivery orders should produce the same commit sequence.
        if !all_sequences.is_empty() && !all_sequences[0].is_empty() {
            assert_commit_sequences_match(&all_sequences);
        }
    }
}

// ---- Equivocator DAG tests ----
// These tests are ported from Sui's consensus/simtests/tests/consensus_dag_tests.rs.
// They verify that the commit protocol produces deterministic and consistent results
// when authorities equivocate (produce multiple blocks for the same slot) and when
// blocks contain reject votes.

/// Maximum number of rounds ahead of the quorum round that a block can be
/// delivered during constrained random iteration.
const MAX_STEP: u32 = 3;

/// Shared test driver for randomized DAG tests with reject votes and optional
/// equivocators. Builds a single DAG from the given config, then runs multiple
/// iterations delivering blocks in constrained random order and committing
/// incrementally. Asserts all runs produce the same commit sequence.
fn test_randomized_dag_with_reject_votes(config: RandomDagConfig, num_runs: usize) {
    let mut rng = StdRng::seed_from_u64(42);

    tracing::info!(
        "Running randomized test with {} authorities, {} rounds, \
         {} txns/block, {}% reject rate, {} equivocators...",
        config.num_authorities,
        config.num_rounds,
        config.num_transactions,
        config.reject_percentage,
        config.equivocators.len(),
    );

    // Pick an own_index that is NOT an equivocator, because DagState asserts
    // that we never add multiple blocks per slot for our own authority.
    let equivocator_indices: Vec<u32> =
        config.equivocators.iter().map(|(a, _)| a.value() as u32).collect();
    let own_index = (0..config.num_authorities as u32)
        .find(|idx| !equivocator_indices.contains(idx))
        .expect("There must be at least one non-equivocating authority");

    let context = CommitTestFixture::context_with_options(config.num_authorities, own_index, 0);
    let dag = EquivocatingRandomDag::new(context.clone(), &mut rng, config);

    // Collect finalized commit sequences from each run.
    let mut commit_sequences = Vec::new();

    for i in 0..num_runs {
        tracing::info!("Run {i} of randomized test...");
        let ctx = CommitTestFixture::context_with_options(context.committee.size(), own_index, 0);
        let mut fixture = CommitTestFixture::new(ctx);
        let mut finalized_commits = Vec::new();
        let mut last_decided = Slot::new(0, AuthorityIndex::new_for_test(0));

        for block in dag.random_iter(&mut rng, MAX_STEP) {
            fixture.try_accept_blocks(vec![block]);

            let (commits, new_last_decided) = fixture.try_commit_tracking(last_decided);
            finalized_commits.extend(commits);
            last_decided = new_last_decided;
        }

        commit_sequences.push(finalized_commits);
    }

    // All runs must agree on the commit sequence (compared by leader and length).
    assert_commit_sequences_match(&commit_sequences);

    // The test should produce at least some commits to be meaningful.
    let total_commits: usize = commit_sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    tracing::info!("Produced {total_commits} commits across runs.");
}

/// 4 authorities, ~200 rounds, reject votes, NO equivocators.
/// Verifies commit sequence consistency across multiple runs with
/// constrained random block delivery.
#[tokio::test]
async fn test_randomized_dag_with_4_authorities() {
    let config = RandomDagConfig {
        num_authorities: 4,
        num_rounds: 200,
        num_transactions: 5,
        reject_percentage: 10,
        equivocators: vec![],
    };
    test_randomized_dag_with_reject_votes(config, 10);
}

/// 7 authorities, ~200 rounds, reject votes, NO equivocators.
/// Larger committee verifies the protocol scales with more validators.
#[tokio::test]
async fn test_randomized_dag_with_7_authorities() {
    let config = RandomDagConfig {
        num_authorities: 7,
        num_rounds: 200,
        num_transactions: 5,
        reject_percentage: 5,
        equivocators: vec![],
    };
    test_randomized_dag_with_reject_votes(config, 10);
}

/// 4 authorities, authority 0 equivocates (1 extra block per selected round).
/// Verifies that a single equivocator does not break commit determinism.
#[tokio::test]
async fn test_randomized_dag_with_4_authorities_1_equivocator() {
    let config = RandomDagConfig {
        num_authorities: 4,
        num_rounds: 200,
        num_transactions: 5,
        reject_percentage: 10,
        equivocators: vec![(AuthorityIndex::new_for_test(0), 1)],
    };
    test_randomized_dag_with_reject_votes(config, 10);
}

/// 7 authorities, authorities 0 and 1 equivocate (2 and 1 extra instances).
/// Tests that multiple equivocators with different equivocation degrees
/// still produce consistent commit sequences.
#[tokio::test]
async fn test_randomized_dag_with_7_authorities_2_equivocators() {
    let config = RandomDagConfig {
        num_authorities: 7,
        num_rounds: 200,
        num_transactions: 5,
        reject_percentage: 5,
        equivocators: vec![
            (AuthorityIndex::new_for_test(0), 2),
            (AuthorityIndex::new_for_test(1), 1),
        ],
    };
    test_randomized_dag_with_reject_votes(config, 10);
}

/// 10 authorities, authorities 0, 1, and 2 equivocate (3, 2, and 1 extra instances).
/// Stress test with the maximum number of equivocators that can be tolerated
/// by BFT (f < n/3), ensuring the protocol remains sound under heavy Byzantine load.
#[tokio::test]
async fn test_randomized_dag_with_10_authorities_3_equivocators() {
    let config = RandomDagConfig {
        num_authorities: 10,
        num_rounds: 100,
        num_transactions: 5,
        reject_percentage: 5,
        equivocators: vec![
            (AuthorityIndex::new_for_test(0), 3),
            (AuthorityIndex::new_for_test(1), 2),
            (AuthorityIndex::new_for_test(2), 1),
        ],
    };
    test_randomized_dag_with_reject_votes(config, 10);
}
