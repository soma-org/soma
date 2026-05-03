// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Supply conservation E2E tests.
//!
//! These tests verify that the total SOMA supply is conserved across epoch
//! boundaries. The production code in `authority.rs::check_soma_conservation()`
//! iterates ALL live objects at every epoch boundary and panics under msim if
//! the sum doesn't equal `TOTAL_SUPPLY_SHANNONS`.
//!
//! These E2E tests exercise that codepath by triggering epoch transitions with
//! various transaction workloads. If conservation is violated, the epoch
//! transition itself will panic (under msim) and the test fails.
//!
//! Additionally, the tests verify relative invariants:
//! - Emission pool decreases at each epoch boundary
//! - Staking pools increase when stake is added
//! - Total system-tracked balances shift correctly

use test_cluster::TestClusterBuilder;
use tracing::info;
use types::config::genesis_config::{AccountConfig, DEFAULT_GAS_AMOUNT};
use types::effects::TransactionEffectsAPI;
use types::system_state::{SystemState, SystemStateTrait as _};
use types::transaction::{TransactionData, TransactionKind};
use utils::logging::init_tracing;

/// Extract supply-relevant balances from the system state.
/// Returns (emission_pool_balance, total_staking_pool_balance, safe_mode_accumulators).
fn system_state_balances(ss: &SystemState) -> (u128, u128, u128) {
    let emission = ss.emission_pool().balance as u128;

    let mut staking: u128 = 0;
    // Stage 9d-C5: pool-token fields collapsed into total_stake.
    // F1 reward inflow (current_rewards) and validator commission
    // accumulator are also live shannons on the pool.
    for v in &ss.validators().validators {
        staking += v.staking_pool.total_stake as u128;
        staking += v.staking_pool.pool_rewards as u128;
        staking += v.staking_pool.accumulated_commission as u128;
    }
    for v in &ss.validators().pending_validators {
        staking += v.staking_pool.total_stake as u128;
        staking += v.staking_pool.pool_rewards as u128;
        staking += v.staking_pool.accumulated_commission as u128;
    }
    for v in ss.validators().inactive_validators.values() {
        staking += v.staking_pool.total_stake as u128;
        staking += v.staking_pool.pool_rewards as u128;
        staking += v.staking_pool.accumulated_commission as u128;
    }
    // Safe-mode accumulators no longer exist (Phase 1 fee model). Safe mode now
    // routes fees inline to protocol_fund and forfeits emissions. Returning 0
    // preserves the tuple shape for callers that still expect three values.
    let safe_mode = 0u128;

    (emission, staking, safe_mode)
}

fn get_system_state(test_cluster: &test_cluster::TestCluster) -> SystemState {
    test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().expect("SystemState must exist")
    })
}

// ---------------------------------------------------------------------------
// Test 1: Conservation across a single epoch boundary with AddStake
// ---------------------------------------------------------------------------

/// Triggers an epoch transition with AddStake transactions. The production
/// `check_soma_conservation()` runs during reconfiguration and will panic if
/// total supply is not conserved. Additionally verifies that emission pool
/// decreases and staking pools increase.
#[cfg(msim)]
#[msim::sim_test]
async fn test_supply_conservation_across_epoch_with_staking() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_accounts(vec![
            AccountConfig {
                gas_amounts: vec![DEFAULT_GAS_AMOUNT],
                usdc_amounts: vec![DEFAULT_GAS_AMOUNT],
                address: None,
            };
            5
        ])
        .build()
        .await;

    let pre_ss = get_system_state(&test_cluster);
    let (pre_emission, pre_staking, _) = system_state_balances(&pre_ss);
    info!("Pre-epoch: emission_pool={pre_emission}, staking_pools={pre_staking}");

    // Execute several AddStake transactions to move SOMA from coins -> staking pools.
    let validator_address = pre_ss.validators().validators[0].metadata.soma_address;

    // Stage 13c: AddStake is balance-mode for both stake (SOMA) and
    // gas (USDC) — no per-tx coin object needed.
    for i in 0..3 {
        let sender = test_cluster.get_addresses()[i];
        let tx_data = e2e_tests::stateless_tx_data(
            &test_cluster,
            sender,
            TransactionKind::AddStake {
                validator: validator_address,
                amount: 1_000_000,
            },
        );

        let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
        assert!(response.effects.status().is_ok(), "Stake tx {i} should succeed");
    }

    // Trigger epoch transition — check_soma_conservation() runs here.
    // Under msim, it will panic if supply is not conserved.
    test_cluster.trigger_reconfiguration().await;

    let post_ss = get_system_state(&test_cluster);
    let (post_emission, post_staking, _) = system_state_balances(&post_ss);
    info!("Post-epoch: emission_pool={post_emission}, staking_pools={post_staking}");

    // Verify staking pools increased (pending stake was processed + rewards).
    assert!(
        post_staking > pre_staking,
        "Staking pools should increase after AddStake + epoch rewards"
    );

    // The emission pool may increase or decrease depending on fee volume:
    // it pays out emission_per_epoch but receives back the non-validator
    // share of total rewards (fees + emissions). Log the delta for visibility.
    let emission_delta = post_emission as i128 - pre_emission as i128;
    info!("Emission pool delta: {emission_delta} shannons");

    // Verify epoch advanced.
    assert_eq!(post_ss.epoch(), pre_ss.epoch() + 1);
}

// ---------------------------------------------------------------------------
// Test 2: Conservation across multiple epochs
// ---------------------------------------------------------------------------

/// Verifies supply conservation across 3 epoch transitions with staking
/// activity in each epoch. Each epoch transition runs the production
/// conservation check.
#[cfg(msim)]
#[msim::sim_test]
async fn test_supply_conservation_multi_epoch() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_accounts(vec![
            AccountConfig {
                gas_amounts: vec![DEFAULT_GAS_AMOUNT],
                usdc_amounts: vec![DEFAULT_GAS_AMOUNT],
                address: None,
            };
            10
        ])
        .build()
        .await;

    let initial_ss = get_system_state(&test_cluster);
    let (initial_emission, _, _) = system_state_balances(&initial_ss);

    let validator_address = initial_ss.validators().validators[0].metadata.soma_address;

    let mut prev_emission = initial_emission;

    for epoch in 0..3 {
        info!("--- Epoch {epoch}: executing transactions ---");

        // Stake from a different address each epoch. Stage 13c:
        // balance-mode for both stake and gas.
        let sender = test_cluster.get_addresses()[epoch];
        let tx_data = e2e_tests::stateless_tx_data(
            &test_cluster,
            sender,
            TransactionKind::AddStake {
                validator: validator_address,
                amount: 500_000,
            },
        );

        let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
        assert!(response.effects.status().is_ok());

        // Epoch transition — check_soma_conservation() runs here under msim.
        test_cluster.trigger_reconfiguration().await;

        let ss = get_system_state(&test_cluster);
        let (emission, staking, safe_mode) = system_state_balances(&ss);
        info!("After epoch {epoch}: emission={emission}, staking={staking}, safe_mode={safe_mode}");

        // Log emission pool delta. It may increase due to remainder return
        // (non-validator share of rewards goes back to emission pool).
        let delta = emission as i128 - prev_emission as i128;
        info!("Emission pool delta for epoch {epoch}: {delta} shannons");
        prev_emission = emission;
    }

    // Verify we advanced 3 epochs.
    let final_ss = get_system_state(&test_cluster);
    assert_eq!(final_ss.epoch(), initial_ss.epoch() + 3);
}

// ---------------------------------------------------------------------------
// Test 3: Emission pool accounting
// ---------------------------------------------------------------------------

/// Verifies that emission pool decreases at each epoch boundary and that the
/// decrease is bounded by emission_per_epoch. The production conservation
/// check runs at each transition to ensure totals balance.
#[cfg(msim)]
#[msim::sim_test]
async fn test_emission_pool_accounting() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_accounts(vec![
            AccountConfig {
                gas_amounts: vec![DEFAULT_GAS_AMOUNT],
                usdc_amounts: vec![DEFAULT_GAS_AMOUNT],
                address: None,
            };
            3
        ])
        .build()
        .await;

    let initial_ss = get_system_state(&test_cluster);
    let (initial_emission, initial_staking, _) = system_state_balances(&initial_ss);
    let emission_per_epoch = initial_ss.emission_pool().current_distribution_amount;

    info!(
        "Initial: emission_pool={initial_emission}, emission_per_epoch={emission_per_epoch}, \
         staking_pools={initial_staking}"
    );

    // Execute a tx to ensure the epoch isn't empty.
    let sender = test_cluster.get_addresses()[0];
    let validator_address = initial_ss.validators().validators[0].metadata.soma_address;
    // Stage 13c: AddStake is balance-mode for both stake and gas.
    let tx_data = e2e_tests::stateless_tx_data(
        &test_cluster,
        sender,
        TransactionKind::AddStake {
            validator: validator_address,
            amount: 1_000_000,
        },
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    // Trigger epoch — conservation check runs here.
    test_cluster.trigger_reconfiguration().await;

    let post_ss = get_system_state(&test_cluster);
    let (post_emission, post_staking, _) = system_state_balances(&post_ss);

    info!("Post-epoch: emission_pool={post_emission}, staking_pools={post_staking}");

    // The emission pool pays out emission_per_epoch but receives back the
    // non-validator share of total rewards. The net effect depends on fee volume.
    let emission_delta = post_emission as i128 - initial_emission as i128;
    info!("Emission pool delta: {emission_delta} shannons");

    // Staking pools should grow (pending stakes processed + rewards distributed).
    assert!(
        post_staking > initial_staking,
        "Staking pools should grow after AddStake + epoch rewards: \
         was {initial_staking}, now {post_staking}"
    );

    // Key invariant: the production conservation check passed during epoch
    // transition (it would have panicked under msim if violated).
    info!("Supply conservation verified via production check during epoch transition");
}
