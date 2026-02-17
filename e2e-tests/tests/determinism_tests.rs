//! Determinism verification E2E tests.
//!
//! These tests verify that the Soma network produces deterministic, consistent
//! state across all validators. This is fundamental for blockchain correctness —
//! if two validators process the same transactions and produce different states,
//! consensus breaks.
//!
//! Tests:
//! 1. test_epoch_state_commitments_all_validators_agree — All nodes produce identical ECMH digests
//! 2. test_checkpoint_digests_consistent_across_validators — No checkpoint-level forks
//! 3. test_transaction_effects_deterministic_across_validators — Same input yields same effects
//! 4. test_state_root_consistency_across_epochs — Multi-epoch state accumulation correctness
//! 5. test_deterministic_execution_with_check_determinism — Bit-for-bit reproducible via msim

use std::time::Duration;
use test_cluster::TestClusterBuilder;
use tokio::time::sleep;
use tracing::info;
use types::{
    checkpoints::CheckpointCommitment,
    effects::TransactionEffectsAPI,
    envelope::Message,
    transaction::{TransactionData, TransactionKind},
};
use utils::logging::init_tracing;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Execute an AddStake transaction from the first wallet address to the first validator.
/// Returns the transaction digest.
async fn execute_stake_tx(
    test_cluster: &test_cluster::TestCluster,
) -> types::digests::TransactionDigest {
    let sender = test_cluster.get_addresses()[0];
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .unwrap()
            .validators
            .validators[0]
            .metadata
            .soma_address
    });

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("Should have a gas object");

    let tx_data = TransactionData::new(
        TransactionKind::AddStake {
            address: validator_address,
            coin_ref: gas,
            amount: Some(1_000_000),
        },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok(), "Stake tx should succeed");
    *response.effects.transaction_digest()
}

/// Wait until a transaction is included in a checkpoint on all validators.
async fn wait_for_tx_in_checkpoint(
    test_cluster: &test_cluster::TestCluster,
    digest: &types::digests::TransactionDigest,
) {
    for _ in 0..600 {
        let all_included = test_cluster
            .swarm
            .validator_node_handles()
            .into_iter()
            .all(|handle| {
                handle.with(|node| {
                    node.state()
                        .epoch_store_for_testing()
                        .is_transaction_executed_in_checkpoint(digest)
                        .unwrap()
                })
            });
        if all_included {
            return;
        }
        sleep(Duration::from_millis(100)).await;
    }
    panic!("Transaction not included in checkpoint on all validators within 60s");
}

// ---------------------------------------------------------------------------
// Test 1: Epoch state commitments agree across all validators
// ---------------------------------------------------------------------------

/// After an epoch transition, all validators and the fullnode must agree on the
/// ECMHLiveObjectSetDigest (the state root commitment). This proves that every
/// node accumulated the exact same live object set through the epoch.
#[cfg(msim)]
#[msim::sim_test]
async fn test_epoch_state_commitments_all_validators_agree() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .build()
        .await;

    // Execute several transactions to mutate state within epoch 0.
    for _ in 0..3 {
        execute_stake_tx(&test_cluster).await;
    }

    // Trigger epoch transition so epoch 0's final checkpoint gets EndOfEpochData.
    test_cluster.trigger_reconfiguration().await;

    // Give nodes time to finalize epoch 0 commitments.
    sleep(Duration::from_secs(2)).await;

    // Collect epoch 0 commitments from every validator.
    let validator_commitments: Vec<Vec<CheckpointCommitment>> = test_cluster
        .swarm
        .validator_node_handles()
        .into_iter()
        .map(|handle| {
            handle.with(|node| {
                node.state()
                    .get_epoch_state_commitments(0)
                    .expect("Should read commitments")
                    .expect("Epoch 0 commitments must exist after reconfig")
            })
        })
        .collect();

    // Also get from fullnode.
    let fullnode_commitments = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_epoch_state_commitments(0)
            .expect("Should read commitments")
            .expect("Fullnode should have epoch 0 commitments")
    });

    // Every commitment vector must be non-empty (at least the ECMH digest).
    assert!(!fullnode_commitments.is_empty(), "Commitments must include ECMH digest");

    // Extract the ECMH digest specifically.
    let extract_ecmh = |commitments: &[CheckpointCommitment]| {
        commitments
            .iter()
            .find_map(|c| match c {
                CheckpointCommitment::ECMHLiveObjectSetDigest(d) => Some(d.clone()),
                _ => None,
            })
            .expect("ECMH digest must be present in epoch commitments")
    };

    let fullnode_ecmh = extract_ecmh(&fullnode_commitments);
    info!("Fullnode ECMH: {:?}", fullnode_ecmh);

    // All validators must produce the same ECMH digest as the fullnode.
    for (i, commitments) in validator_commitments.iter().enumerate() {
        let validator_ecmh = extract_ecmh(commitments);
        assert_eq!(
            validator_ecmh, fullnode_ecmh,
            "Validator {} ECMH digest diverges from fullnode",
            i,
        );
    }

    info!(
        "All {} validators + fullnode agree on epoch 0 state root",
        validator_commitments.len()
    );
}

// ---------------------------------------------------------------------------
// Test 2: Checkpoint digests consistent across validators
// ---------------------------------------------------------------------------

/// Every validator must produce identical checkpoint digests for each sequence
/// number. If any validator has a different digest, it means a checkpoint fork
/// occurred — a critical safety violation.
#[cfg(msim)]
#[msim::sim_test]
async fn test_checkpoint_digests_consistent_across_validators() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .build()
        .await;

    // Execute transactions to create checkpoints.
    let digest = execute_stake_tx(&test_cluster).await;
    wait_for_tx_in_checkpoint(&test_cluster, &digest).await;

    // Let a few more checkpoints finalize.
    sleep(Duration::from_secs(2)).await;

    // Find the minimum highest-executed checkpoint across all validators.
    let min_highest_seq = test_cluster
        .swarm
        .validator_node_handles()
        .into_iter()
        .map(|handle| {
            handle.with(|node| {
                node.state()
                    .get_checkpoint_store()
                    .get_highest_executed_checkpoint_seq_number()
                    .expect("Should read highest checkpoint")
                    .expect("Should have at least one checkpoint")
            })
        })
        .min()
        .expect("Must have validators");

    assert!(min_highest_seq > 0, "Should have produced multiple checkpoints");
    info!("Checking checkpoint digests up to sequence {}", min_highest_seq);

    // For every checkpoint up to min_highest_seq, verify all validators agree.
    for seq in 0..=min_highest_seq {
        let digests: Vec<_> = test_cluster
            .swarm
            .validator_node_handles()
            .into_iter()
            .map(|handle| {
                handle.with(|node| {
                    let cp = node
                        .state()
                        .get_checkpoint_store()
                        .get_checkpoint_by_sequence_number(seq)
                        .expect("DB error")
                        .unwrap_or_else(|| panic!("Validator missing checkpoint {}", seq));
                    *cp.digest()
                })
            })
            .collect();

        // All digests should match the first one.
        let first = &digests[0];
        for (i, d) in digests.iter().enumerate().skip(1) {
            assert_eq!(
                d, first,
                "Checkpoint {} fork: validator {} digest {:?} != validator 0 digest {:?}",
                seq, i, d, first,
            );
        }
    }

    info!(
        "All validators agree on checkpoint digests for sequences 0..={}",
        min_highest_seq
    );
}

// ---------------------------------------------------------------------------
// Test 3: Transaction effects deterministic across validators
// ---------------------------------------------------------------------------

/// Execute multiple transactions and verify every validator computed identical
/// TransactionEffectsDigest for each one. Different effects digests would mean
/// non-deterministic execution.
#[cfg(msim)]
#[msim::sim_test]
async fn test_transaction_effects_deterministic_across_validators() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .build()
        .await;

    // Execute several transactions and collect their digests.
    let mut tx_digests = Vec::new();
    for _ in 0..5 {
        let digest = execute_stake_tx(&test_cluster).await;
        tx_digests.push(digest);
    }

    // Wait for all transactions to be checkpointed on all validators.
    for digest in &tx_digests {
        wait_for_tx_in_checkpoint(&test_cluster, digest).await;
    }

    // For each transaction, verify all validators have identical effects digests.
    for tx_digest in &tx_digests {
        let effects_digests: Vec<_> = test_cluster
            .swarm
            .validator_node_handles()
            .into_iter()
            .map(|handle| {
                handle.with(|node| {
                    let effects = node
                        .state()
                        .get_transaction_cache_reader()
                        .get_executed_effects(tx_digest)
                        .unwrap_or_else(|| {
                            panic!("Validator missing effects for tx {}", tx_digest)
                        });
                    effects.digest()
                })
            })
            .collect();

        let first = &effects_digests[0];
        for (i, d) in effects_digests.iter().enumerate().skip(1) {
            assert_eq!(
                d, first,
                "Tx {} effects divergence: validator {} digest {:?} != validator 0 digest {:?}",
                tx_digest, i, d, first,
            );
        }
    }

    info!(
        "All validators agree on effects digests for {} transactions",
        tx_digests.len()
    );
}

// ---------------------------------------------------------------------------
// Test 4: State root consistency across multiple epochs
// ---------------------------------------------------------------------------

/// Run through 3 epoch transitions, executing transactions in each epoch.
/// Verify: (a) epoch commitments exist for every epoch, (b) all validators
/// agree, and (c) consecutive epoch state roots differ (state was mutated).
#[cfg(msim)]
#[msim::sim_test]
async fn test_state_root_consistency_across_epochs() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .build()
        .await;

    let num_epochs = 3u64;
    let mut epoch_ecmh_digests = Vec::new();

    for epoch in 0..num_epochs {
        // Execute transactions within this epoch.
        for _ in 0..2 {
            execute_stake_tx(&test_cluster).await;
        }

        // Trigger epoch transition.
        test_cluster.trigger_reconfiguration().await;

        // Give nodes time to finalize.
        sleep(Duration::from_secs(1)).await;

        // Verify all validators agree on this epoch's commitments.
        let validator_ecmhs: Vec<_> = test_cluster
            .swarm
            .validator_node_handles()
            .into_iter()
            .map(|handle| {
                handle.with(|node| {
                    let commitments = node
                        .state()
                        .get_epoch_state_commitments(epoch)
                        .expect("Should read commitments")
                        .expect("Epoch commitments must exist");
                    commitments
                        .iter()
                        .find_map(|c| match c {
                            CheckpointCommitment::ECMHLiveObjectSetDigest(d) => Some(d.clone()),
                            _ => None,
                        })
                        .expect("ECMH digest must be present")
                })
            })
            .collect();

        // All validators agree.
        let first = &validator_ecmhs[0];
        for (i, d) in validator_ecmhs.iter().enumerate().skip(1) {
            assert_eq!(d, first, "Epoch {} ECMH divergence at validator {}", epoch, i);
        }

        epoch_ecmh_digests.push(first.clone());
        info!("Epoch {} state root verified across all validators", epoch);
    }

    // Consecutive epoch digests should differ (transactions mutated state).
    for i in 1..epoch_ecmh_digests.len() {
        assert_ne!(
            epoch_ecmh_digests[i],
            epoch_ecmh_digests[i - 1],
            "Epoch {} and {} have identical state roots despite state mutations",
            i - 1,
            i,
        );
    }

    info!(
        "State root consistency verified across {} epochs with distinct digests",
        num_epochs
    );
}

// ---------------------------------------------------------------------------
// Test 5: Full deterministic replay via msim check_determinism
// ---------------------------------------------------------------------------

/// Use msim's `check_determinism` attribute to run the entire test twice and
/// verify bit-for-bit identical execution paths (PRNG logs match). This covers
/// consensus scheduling, transaction execution, state accumulation, and epoch
/// transition logic.
#[cfg(msim)]
#[msim::sim_test(check_determinism)]
async fn test_deterministic_execution_with_check_determinism() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_epoch_duration_ms(5000)
        .build()
        .await;

    // Execute transactions to create meaningful state.
    let sender = test_cluster.get_addresses()[0];
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .unwrap()
            .validators
            .validators[0]
            .metadata
            .soma_address
    });

    for _ in 0..3 {
        let gas = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(sender)
            .await
            .unwrap()
            .expect("Should have gas");

        let tx_data = TransactionData::new(
            TransactionKind::AddStake {
                address: validator_address,
                coin_ref: gas,
                amount: Some(1_000_000),
            },
            sender,
            vec![gas],
        );

        let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
        assert!(response.effects.status().is_ok());
    }

    // Wait for epoch transition (passive via timer).
    test_cluster.wait_for_epoch(Some(1)).await;

    // Read epoch 0 commitments to anchor the state in the PRNG log.
    // The check_determinism attribute will verify this produces identical
    // results on both runs.
    let commitments = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_epoch_state_commitments(0)
            .expect("Should read commitments")
            .expect("Epoch 0 commitments must exist")
    });
    assert!(!commitments.is_empty());

    info!("Deterministic execution verified via check_determinism");
}
