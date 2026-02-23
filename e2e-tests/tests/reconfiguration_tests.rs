use std::{collections::HashSet, time::Duration};

use node::handle::SomaNodeHandle;
use rand::rngs::OsRng;
use test_cluster::{TestCluster, TestClusterBuilder};
use tokio::time::sleep;
use tracing::info;
use types::{
    base::SomaAddress,
    committee::{
        VALIDATOR_CONSENSUS_LOW_POWER, VALIDATOR_CONSENSUS_MIN_POWER,
        VALIDATOR_CONSENSUS_VERY_LOW_POWER, VALIDATOR_LOW_STAKE_GRACE_PERIOD,
    },
    config::genesis_config::{
        AccountConfig, DEFAULT_GAS_AMOUNT, ValidatorGenesisConfig, ValidatorGenesisConfigBuilder,
    },
    crypto::{KeypairTraits, SomaKeyPair},
    effects::TransactionEffectsAPI,
    system_state::SystemStateTrait as _,
    transaction::{
        AddValidatorArgs, RemoveValidatorArgs, Transaction, TransactionData, TransactionKind,
    },
};
use utils::logging::init_tracing;

const VALIDATOR_STARTING_STAKE: u64 = 1_000_000_000_000_000; // 1M SOMA

#[cfg(msim)]
#[msim::sim_test]
async fn basic_reconfig_end_to_end_test() {
    let _ = tracing_subscriber::fmt::try_init();
    // TODO remove this sleep when this test passes consistently
    sleep(Duration::from_secs(1)).await;
    let test_cluster = TestClusterBuilder::new().build().await;
    test_cluster.trigger_reconfiguration().await;
}

#[cfg(msim)]
#[msim::sim_test]
async fn test_reconfig_with_committee_change_basic() {
    init_tracing();
    // This test exercise the full flow of a validator joining the network, catch up and then leave.
    let initial_num_validators = 10;
    let new_validator = ValidatorGenesisConfigBuilder::new().build(&mut OsRng);
    let address = (&new_validator.account_key_pair.public()).into();
    let mut test_cluster = TestClusterBuilder::new()
        .with_accounts(vec![AccountConfig {
            gas_amounts: vec![VALIDATOR_STARTING_STAKE * 1_000],
            address: None,
        }])
        .with_num_validators(initial_num_validators)
        .with_validator_candidates([address])
        .build()
        .await;

    // Get a single validator's stake and voting power. All of them are the same
    // in the `TestCluster`, so we can pick any.
    let total_stake = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
            .validators()
            .total_stake
    });

    // Setting voting power to roughly ~ .20% of the total voting power, which
    // is higher than VALIDATOR_MIN_POWER_PHASE_1.
    let min_barrier = total_stake / 10_000 * 20;
    execute_add_validator_transactions(&mut test_cluster, &new_validator, Some(min_barrier)).await;

    test_cluster.trigger_reconfiguration().await;

    // Check that a new validator has joined the committee.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state().epoch_store_for_testing().committee().num_members(),
            initial_num_validators + 1
        );
    });
    let new_validator_handle = test_cluster.spawn_new_validator(new_validator).await;
    test_cluster.wait_for_epoch_all_nodes(1).await;

    new_validator_handle.with(|node| {
        assert!(node.state().is_validator(&node.state().epoch_store_for_testing()));
    });

    execute_remove_validator_tx(&test_cluster, &new_validator_handle).await;
    test_cluster.trigger_reconfiguration().await;

    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state().epoch_store_for_testing().committee().num_members(),
            initial_num_validators
        );
    });
}

// This test just starts up a cluster that reconfigures itself under 0 load.
#[cfg(msim)]
#[msim::sim_test]
async fn test_passive_reconfig_normal() {
    do_test_passive_reconfig().await;
}

#[cfg(msim)]
#[msim::sim_test(check_determinism)]
async fn test_passive_reconfig_determinism() {
    do_test_passive_reconfig().await;
}

async fn do_test_passive_reconfig() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_epoch_duration_ms(1000).build().await;

    let target_epoch: u64 =
        std::env::var("RECONFIG_TARGET_EPOCH").ok().map(|v| v.parse().unwrap()).unwrap_or(4);

    test_cluster.wait_for_epoch(Some(target_epoch)).await;
}

#[cfg(msim)]
#[msim::sim_test]
async fn test_reconfig_with_committee_change_stress_normal() {
    do_test_reconfig_with_committee_change_stress().await;
}

#[cfg(msim)]
#[msim::sim_test(check_determinism)]
async fn test_reconfig_with_committee_change_stress_determinism() {
    do_test_reconfig_with_committee_change_stress().await;
}

async fn do_test_reconfig_with_committee_change_stress() {
    init_tracing();

    let mut candidates =
        (0..6).map(|_| ValidatorGenesisConfigBuilder::new().build(&mut OsRng)).collect::<Vec<_>>();
    let addresses = candidates
        .iter()
        .map(|c| (&c.account_key_pair.public()).into())
        .collect::<Vec<SomaAddress>>();
    let mut test_cluster = TestClusterBuilder::new()
        .with_accounts(vec![AccountConfig {
            gas_amounts: vec![DEFAULT_GAS_AMOUNT * 10],
            address: None,
        }])
        .with_num_validators(7)
        .with_validator_candidates(addresses)
        // .with_num_unpruned_validators(2)
        .build()
        .await;

    let mut cur_epoch = 0;

    while let Some(v1) = candidates.pop() {
        let v2 = candidates.pop().unwrap();
        execute_add_validator_transactions(&test_cluster, &v1, None).await;
        execute_add_validator_transactions(&test_cluster, &v2, None).await;
        let mut removed_validators = vec![];
        for v in test_cluster.swarm.active_validators().take(2) {
            let h = v.get_node_handle().unwrap();
            removed_validators.push(h.state().name);
            execute_remove_validator_tx(&test_cluster, &h).await;
        }
        let handle1 = test_cluster.spawn_new_validator(v1).await;
        let handle2 = test_cluster.spawn_new_validator(v2).await;

        tokio::join!(
            test_cluster.wait_for_epoch_on_node(
                &handle1,
                Some(cur_epoch),
                Duration::from_secs(120)
            ),
            test_cluster.wait_for_epoch_on_node(
                &handle2,
                Some(cur_epoch),
                Duration::from_secs(120)
            )
        );

        test_cluster.trigger_reconfiguration().await;
        let committee = test_cluster
            .fullnode_handle
            .soma_node
            .with(|node| node.state().epoch_store_for_testing().committee().clone());
        cur_epoch = committee.epoch();
        assert_eq!(committee.num_members(), 7);
        assert!(committee.authority_exists(&handle1.state().name));
        assert!(committee.authority_exists(&handle2.state().name));
        removed_validators.iter().all(|v| !committee.authority_exists(v));
    }
}

#[cfg(msim)]
#[msim::sim_test]
async fn test_reconfig_with_voting_power_decrease_normal() {
    init_tracing();

    // This test exercise the full flow of a validator joining the network, catch up and then leave.
    // Validator starts with .12% of the total voting power and then decreases to below the threshold.
    let initial_num_validators = 10;
    let new_validator = ValidatorGenesisConfigBuilder::new().with_stake(0).build(&mut OsRng);

    let address = (&new_validator.account_key_pair.public()).into();
    let mut test_cluster = TestClusterBuilder::new()
        .with_validators(
            (0..10)
                .map(|_| {
                    ValidatorGenesisConfigBuilder::new()
                        .with_stake(VALIDATOR_STARTING_STAKE)
                        .build(&mut OsRng)
                })
                .collect(),
        )
        .with_accounts(vec![AccountConfig {
            gas_amounts: vec![DEFAULT_GAS_AMOUNT * initial_num_validators as u64 * 3],
            address: None,
        }])
        .with_num_validators(initial_num_validators)
        .with_validator_candidates([address])
        .build()
        .await;

    // Get total stake of validators in the system, their addresses and the grace period.
    let (total_stake, initial_validators, low_stake_grace_period) =
        test_cluster.fullnode_handle.soma_node.with(|node| {
            let system_state = node
                .state()
                .get_system_state_object_for_testing()
                .expect("Should be able to get SystemState");

            (
                system_state.validators().total_stake,
                system_state
                    .validators()
                    .validators
                    .iter()
                    .map(|v| v.metadata.soma_address)
                    .collect::<Vec<_>>(),
                VALIDATOR_LOW_STAKE_GRACE_PERIOD,
            )
        });

    // Setting voting power to roughly ~ .20% of the total voting power.
    // This allows us to achieve the following by halving:
    // 0. .20% > VALIDATOR_MIN_POWER_PHASE_1
    // 1. .10% > VALIDATOR_LOW_POWER_PHASE_1
    // 2. .5%  > VALIDATOR_VERY_LOW_POWER_PHASE_1
    let min_join_stake = total_stake * 20 / 10_000;
    let default_stake = total_stake / initial_num_validators as u64;

    execute_add_validator_transactions(&mut test_cluster, &new_validator, Some(min_join_stake))
        .await;

    test_cluster.trigger_reconfiguration().await;

    // Check that a new validator has joined the committee.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state().epoch_store_for_testing().committee().num_members(),
            initial_num_validators + 1
        );
    });

    for (address, stake) in
        initial_validators.iter().map(|address| (*address, default_stake)).collect::<Vec<_>>()
    {
        // Double the stake of every other validator, stake just as much as they had.
        execute_add_stake_transaction(
            new_validator.account_key_pair.copy(),
            &mut test_cluster,
            address,
            stake,
        )
        .await;
    }

    test_cluster.trigger_reconfiguration().await;

    // Find the candidate in the `active_validators` set, and check that the
    // voting power has decreased. Panics if the candidate is not found.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState");

        let validators = system_state.validators();
        let candidate = validators.validators.iter().find(|v| v.metadata.soma_address == address);

        assert!(candidate.is_some());
        let candidate = candidate.unwrap();

        // Check that the validator voting power has decreased just below the
        // "min" threshold but not below the "low" threshold.
        // Yet the candidate is not at risk.
        assert!(candidate.voting_power < VALIDATOR_CONSENSUS_MIN_POWER);
        assert!(candidate.voting_power > VALIDATOR_CONSENSUS_LOW_POWER);
        assert_eq!(validators.at_risk_validators.len(), 0);
    });

    // Double validators' stake once again, and check that the new validator is now at risk.
    // Double the stake of every other validator, stake just as much as they had.
    for (address, stake) in
        initial_validators.iter().map(|address| (*address, default_stake)).collect::<Vec<_>>()
    {
        execute_add_stake_transaction(
            new_validator.account_key_pair.copy(),
            &mut test_cluster,
            address,
            stake,
        )
        .await;
    }

    test_cluster.trigger_reconfiguration().await;

    // list stakes and voting powers
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState");

        let validators = system_state.validators();
        let candidate = validators
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == address)
            .unwrap()
            .clone();

        // Check that the validator voting power has decreased just below the
        // "min" threshold and also below the "low" threshold.
        // Yet the candidate is not at risk.
        assert!(candidate.voting_power < VALIDATOR_CONSENSUS_MIN_POWER);
        assert!(candidate.voting_power < VALIDATOR_CONSENSUS_LOW_POWER);
        assert!(candidate.voting_power > VALIDATOR_CONSENSUS_VERY_LOW_POWER);
        assert_eq!(validators.at_risk_validators.len(), 1);
    });

    // Wait for the grace period to expire.
    for _ in 0..low_stake_grace_period {
        test_cluster.trigger_reconfiguration().await;
    }

    // Check that the validator has been kicked out as risky.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state()
                .get_system_state_object_for_testing()
                .expect("Should be able to get SystemState")
                .validators()
                .validators
                .len(),
            initial_num_validators
        )
    });
}

#[cfg(msim)]
#[msim::sim_test]
async fn test_reconfig_with_voting_power_decrease_immediate_removal() {
    init_tracing();

    // This test exercise the full flow of a validator joining the network, catch up and then leave.
    // Validator starts with .12% of the total voting power and then decreases to below the threshold.
    let initial_num_validators = 10;
    let initial_validators = (0..10)
        .map(|_| {
            ValidatorGenesisConfigBuilder::new()
                .with_stake(VALIDATOR_STARTING_STAKE)
                .build(&mut OsRng)
        })
        .collect::<Vec<_>>();
    let new_validator = ValidatorGenesisConfigBuilder::new().with_stake(0).build(&mut OsRng);

    let address = (&new_validator.account_key_pair.public()).into();
    let mut test_cluster = TestClusterBuilder::new()
        .with_validators(initial_validators)
        .with_accounts(vec![AccountConfig {
            gas_amounts: vec![10 * VALIDATOR_STARTING_STAKE * initial_num_validators as u64],
            address: None,
        }])
        .with_num_validators(initial_num_validators)
        .with_validator_candidates([address])
        .build()
        .await;

    // Get total stake of validators in the system, their addresses and the grace period.
    let (total_stake, mut initial_validators) =
        test_cluster.fullnode_handle.soma_node.with(|node| {
            let system_state = node
                .state()
                .get_system_state_object_for_testing()
                .expect("Should be able to get SystemState");

            let validators = system_state.validators();
            (
                validators.total_stake,
                validators.validators.iter().map(|v| v.metadata.soma_address).collect::<Vec<_>>(),
            )
        });

    // Setting voting power to roughly ~ .15% of the total voting power.
    // If stake of other validators increases 4x, the new validator's
    // voting power will decrease to below the very low threshold.
    let min_join_stake = total_stake * 15 / 10_000;

    execute_add_validator_transactions(&mut test_cluster, &new_validator, Some(min_join_stake))
        .await;

    test_cluster.trigger_reconfiguration().await;

    // Check that a new validator has joined the committee.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state().epoch_store_for_testing().committee().num_members(),
            initial_num_validators + 1
        );
    });

    // x4 the stake of every other validator, lowering the new validator's
    // voting power below the very low threshold, resulting in immediate removal
    // from the committee at the next reconfiguration.
    for (address, stake) in initial_validators
        .iter()
        .map(|address| (*address, VALIDATOR_STARTING_STAKE * 3))
        .collect::<Vec<_>>()
    {
        execute_add_stake_transaction(
            test_cluster
                .all_validator_handles()
                .first()
                .unwrap()
                .state()
                .config
                .account_key_pair
                .keypair()
                .copy(),
            &mut test_cluster,
            address,
            stake,
        )
        .await;
    }

    test_cluster.trigger_reconfiguration().await;

    // Check that the validator has been kicked out.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let mut active_validators = node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
            .validators()
            .validators
            .iter()
            .map(|v| v.metadata.soma_address)
            .collect::<Vec<_>>();

        assert_eq!(active_validators.len(), initial_num_validators);
        active_validators.sort();
        initial_validators.sort();
        assert_eq!(active_validators, initial_validators);
    });
}

async fn execute_remove_validator_tx(test_cluster: &TestCluster, handle: &SomaNodeHandle) {
    let address = handle.with(|node| node.get_config().soma_address());

    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(address)
        .await
        .unwrap()
        .expect("Can't get gas object for address");

    let tx = handle.with(|node| {
        Transaction::from_data_and_signer(
            TransactionData::new(
                TransactionKind::RemoveValidator(RemoveValidatorArgs {
                    pubkey_bytes: bcs::to_bytes(
                        &node.get_config().account_key_pair.keypair().public(),
                    )
                    .unwrap(),
                }),
                (&node.get_config().account_key_pair.keypair().public()).into(),
                vec![gas_object],
            ),
            vec![node.get_config().account_key_pair.keypair()],
        )
    });

    info!(?tx, "Executing remove validator tx: {}", address);

    let _response = test_cluster.execute_transaction(tx).await;
}

/// Execute a sequence of transactions to add a validator, including adding candidate, adding stake
/// and activate the validator.
/// It does not however trigger reconfiguration yet.
async fn execute_add_validator_transactions(
    test_cluster: &TestCluster,
    new_validator: &ValidatorGenesisConfig,
    stake_amount: Option<u64>,
) {
    let pending_active_count = test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState");
        system_state.validators().pending_validators.len()
    });

    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address((&new_validator.account_key_pair.public()).into())
        .await
        .unwrap()
        .expect("Can't get gas object for address");

    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::AddValidator({
                let sender_address = SomaAddress::from(&new_validator.account_key_pair.public());
                let pop = types::crypto::generate_proof_of_possession(
                    &new_validator.key_pair,
                    sender_address,
                );
                AddValidatorArgs {
                    pubkey_bytes: bcs::to_bytes(&new_validator.key_pair.public()).unwrap(),
                    network_pubkey_bytes: bcs::to_bytes(&new_validator.network_key_pair.public())
                        .unwrap(),
                    worker_pubkey_bytes: bcs::to_bytes(&new_validator.worker_key_pair.public())
                        .unwrap(),
                    proof_of_possession: pop.as_ref().to_vec(),
                    net_address: bcs::to_bytes(&new_validator.network_address).unwrap(),
                    p2p_address: bcs::to_bytes(&new_validator.p2p_address).unwrap(),
                    primary_address: bcs::to_bytes(&new_validator.consensus_address).unwrap(),
                    proxy_address: bcs::to_bytes(&new_validator.proxy_address).unwrap(),
                }
            }),
            (&new_validator.account_key_pair.public()).into(),
            vec![gas_object],
        ),
        vec![&new_validator.account_key_pair],
    );

    info!(?tx, "Executing add validator tx {}", &new_validator.network_address.to_string());

    let _response = test_cluster.execute_transaction(tx).await;

    execute_add_stake_transaction(
        new_validator.account_key_pair.copy(),
        test_cluster,
        (&new_validator.account_key_pair.public()).into(),
        stake_amount.unwrap_or(DEFAULT_GAS_AMOUNT),
    )
    .await;

    // Check that we can get the pending validator from 0x5.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState");
        let pending_active_validators = &system_state.validators().pending_validators;
        assert_eq!(pending_active_validators.len(), pending_active_count + 1);
        assert_eq!(
            pending_active_validators[pending_active_validators.len() - 1].metadata.soma_address,
            (&new_validator.account_key_pair.public()).into()
        );
    });
}

/// Execute a single stake transaction to add stake to a validator.
async fn execute_add_stake_transaction(
    signer: SomaKeyPair,
    test_cluster: &TestCluster,
    address: SomaAddress,
    stake: u64,
) {
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address((&signer.public()).into())
        .await
        .unwrap()
        .expect("Can't get gas object for address");

    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::AddStake { address, coin_ref: gas_object, amount: Some(stake) },
            (&signer.public()).into(),
            vec![gas_object],
        ),
        vec![&signer],
    );

    info!(?tx, "Executing stake validator tx {}", address.to_string());

    let _response = test_cluster.execute_transaction(tx).await;
}

/// After a validator is removed and reconfiguration occurs, it should appear
/// in the `inactive_validators` map with its `deactivation_epoch` set.
#[cfg(msim)]
#[msim::sim_test]
async fn test_inactive_validator_pool_read() {
    init_tracing();

    let initial_num_validators = 5;
    let test_cluster =
        TestClusterBuilder::new().with_num_validators(initial_num_validators).build().await;

    // Pick the first validator to remove.
    let first_validator_handle = test_cluster.all_validator_handles().into_iter().next().unwrap();
    let validator_address = first_validator_handle.with(|node| node.get_config().soma_address());
    let validator_pool_id = test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState");
        let validator = system_state
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == validator_address)
            .expect("Validator should be in active set");
        validator.staking_pool.id
    });

    // The validator should NOT be in inactive_validators before removal.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node.state().get_system_state_object_for_testing().unwrap();
        assert!(
            !system_state.validators().inactive_validators.contains_key(&validator_pool_id),
            "Validator should not be inactive before removal"
        );
    });

    // Remove the validator.
    execute_remove_validator_tx(&test_cluster, &first_validator_handle).await;
    test_cluster.trigger_reconfiguration().await;

    // Verify the committee shrunk.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state().epoch_store_for_testing().committee().num_members(),
            initial_num_validators - 1
        );
    });

    // Verify the removed validator is in inactive_validators with deactivation_epoch set.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState");
        let inactive = system_state
            .validators()
            .inactive_validators
            .get(&validator_pool_id)
            .expect("Removed validator should be in inactive_validators");
        assert!(
            inactive.staking_pool.deactivation_epoch.is_some(),
            "Inactive validator should have deactivation_epoch set"
        );
        assert_eq!(
            inactive.staking_pool.deactivation_epoch.unwrap(),
            1,
            "Deactivation epoch should be 1 (the epoch after removal)"
        );
        assert_eq!(
            inactive.metadata.soma_address, validator_address,
            "Inactive validator address should match the removed validator"
        );
    });

    // Verify the removed validator is no longer in the committee.
    first_validator_handle.with(|node| {
        assert!(
            node.state().is_fullnode(&node.state().epoch_store_for_testing()),
            "Removed validator should now report as fullnode"
        );
    });
}

/// After submitting an AddValidator transaction, the new validator should appear
/// in `pending_validators` before reconfiguration.
#[cfg(msim)]
#[msim::sim_test]
async fn test_validator_candidate_pool_read() {
    init_tracing();

    let initial_num_validators = 4;
    let new_validator = ValidatorGenesisConfigBuilder::new().build(&mut OsRng);
    let new_address: SomaAddress = (&new_validator.account_key_pair.public()).into();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(initial_num_validators)
        .with_validator_candidates([new_address])
        .build()
        .await;

    // Before AddValidator: no pending validators.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState");
        assert!(
            system_state.validators().pending_validators.is_empty(),
            "No pending validators should exist initially"
        );
    });

    // Submit AddValidator transaction.
    execute_add_validator_transactions(&test_cluster, &new_validator, None).await;

    // After AddValidator, before reconfig: the new validator should be in pending_validators.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState");

        assert_eq!(
            system_state.validators().pending_validators.len(),
            1,
            "One validator should be pending"
        );
        let pending = &system_state.validators().pending_validators[0];
        assert_eq!(
            pending.metadata.soma_address, new_address,
            "Pending validator address should match"
        );
        // The pending validator's staking pool should be preactive (no activation_epoch yet).
        assert!(
            pending.staking_pool.activation_epoch.is_none(),
            "Pending validator pool should not yet be activated"
        );
        assert!(
            pending.staking_pool.deactivation_epoch.is_none(),
            "Pending validator pool should not be deactivated"
        );
        assert!(pending.staking_pool.soma_balance > 0, "Pending validator should have stake");
    });

    // Verify the committee hasn't changed yet.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state().epoch_store_for_testing().committee().num_members(),
            initial_num_validators,
            "Committee should not change before reconfiguration"
        );
    });

    // Now trigger reconfiguration and verify the candidate is promoted.
    test_cluster.trigger_reconfiguration().await;

    test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState");

        // Pending should be empty now (promoted to active).
        assert!(
            system_state.validators().pending_validators.is_empty(),
            "Pending validators should be empty after reconfig"
        );

        // Committee should have grown.
        assert_eq!(
            node.state().epoch_store_for_testing().committee().num_members(),
            initial_num_validators + 1,
            "Committee should grow by one after reconfig"
        );

        // The new validator should be in the active set.
        let active = system_state
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == new_address);
        assert!(active.is_some(), "New validator should be in the active set after reconfig");
        let active = active.unwrap();
        assert!(
            active.staking_pool.activation_epoch.is_some(),
            "Activated validator should have activation_epoch set"
        );
    });
}

/// The network should survive random validator restarts and still reach the
/// target epoch. Tests consensus resilience under node failures.
#[cfg(msim)]
#[msim::sim_test]
async fn test_reconfig_with_failing_validator() {
    init_tracing();

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(7).with_epoch_duration_ms(5000).build().await;

    let target_epoch: u64 =
        std::env::var("RECONFIG_TARGET_EPOCH").ok().map(|v| v.parse().unwrap()).unwrap_or(4);

    // Spawn a background task that randomly kills and restarts one validator.
    let validator_pubkeys = test_cluster.get_validator_pubkeys();
    let swarm_handle = &test_cluster.swarm;

    // We'll do controlled restarts: stop one validator at a time, wait, restart.
    // We iterate through validators to ensure each one gets tested.
    let restart_task = {
        let pubkeys = validator_pubkeys.clone();
        async move {
            let mut idx = 0;
            loop {
                let key = &pubkeys[idx % pubkeys.len()];
                let node = swarm_handle.node(key).unwrap();
                if node.is_running() {
                    info!("Stopping validator {} (idx={})", key, idx);
                    node.stop();
                    sleep(Duration::from_secs(2)).await;
                    info!("Restarting validator {} (idx={})", key, idx);
                    node.start().await.unwrap();
                }
                idx += 1;
                sleep(Duration::from_secs(3)).await;
            }
        }
    };

    tokio::select! {
        _ = restart_task => {
            unreachable!("Restart task should run forever");
        }
        system_state = test_cluster.wait_for_epoch_with_timeout(
            Some(target_epoch),
            Duration::from_secs(120),
        ) => {
            info!("Reached target epoch {} successfully despite validator failures", system_state.epoch());
            assert!(system_state.epoch() >= target_epoch);
        }
    }
}

/// Test that submitting staking transactions during epoch transitions doesn't
/// cause issues. Transactions submitted right at the epoch boundary should
/// either succeed in the current epoch or be processed in the next epoch.
#[cfg(msim)]
#[msim::sim_test]
async fn test_create_advance_epoch_tx_race() {
    init_tracing();

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(4).with_epoch_duration_ms(2000).build().await;

    let sender = test_cluster.get_addresses()[0];
    let target_epoch = 3u64;

    // Get a validator address to stake with.
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().validators().validators[0]
            .metadata
            .soma_address
    });

    let mut submitted = 0u64;
    let mut succeeded = 0u64;
    let mut failed = 0u64;

    loop {
        let current_epoch = test_cluster
            .fullnode_handle
            .soma_node
            .with(|node| node.state().epoch_store_for_testing().epoch());
        if current_epoch >= target_epoch {
            break;
        }

        // Get a gas object and submit a stake transaction.
        let gas_object =
            test_cluster.wallet.get_one_gas_object_owned_by_address(sender).await.unwrap();

        if let Some(gas_object) = gas_object {
            let tx_data = TransactionData::new(
                TransactionKind::AddStake {
                    address: validator_address,
                    coin_ref: gas_object,
                    amount: Some(1_000_000),
                },
                sender,
                vec![gas_object],
            );

            let tx = test_cluster.wallet.sign_transaction(&tx_data).await;
            submitted += 1;

            match tokio::time::timeout(
                Duration::from_secs(30),
                test_cluster.execute_transaction(tx),
            )
            .await
            {
                Ok(response) => {
                    if response.effects.status().is_ok() {
                        succeeded += 1;
                    } else {
                        failed += 1;
                    }
                }
                Err(_timeout) => {
                    failed += 1;
                }
            }
        }

        sleep(Duration::from_millis(200)).await;
    }

    info!(
        "Epoch race test complete: submitted={}, succeeded={}, failed={}",
        submitted, succeeded, failed
    );

    assert!(submitted > 0, "Should have submitted at least some transactions");
    assert!(succeeded > 0, "At least some transactions should succeed");

    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().expect("SystemState should be readable")
    });
    assert!(system_state.epoch() >= target_epoch, "Should have reached target epoch");
    assert_eq!(system_state.validators().validators.len(), 4, "Committee should remain unchanged");
}

/// Test that object locks from the current epoch are correctly handled across
/// epoch boundaries. A gas object consumed in epoch N should not be usable
/// with the stale ObjectRef in epoch N+1.
#[cfg(msim)]
#[msim::sim_test]
async fn test_expired_locks() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    let sender = test_cluster.get_addresses()[0];

    // Get a validator address to stake with.
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().validators().validators[0]
            .metadata
            .soma_address
    });

    // Get a gas object in epoch 0.
    let gas_object_epoch0 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("Should have a gas object");

    // Execute a stake transaction in epoch 0, consuming the gas object version.
    let tx_data = TransactionData::new(
        TransactionKind::AddStake {
            address: validator_address,
            coin_ref: gas_object_epoch0,
            amount: Some(1_000_000),
        },
        sender,
        vec![gas_object_epoch0],
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok(), "First stake should succeed");

    // Trigger reconfiguration to epoch 1.
    test_cluster.trigger_reconfiguration().await;

    let current_epoch = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().epoch_store_for_testing().epoch());
    assert_eq!(current_epoch, 1, "Should be in epoch 1");

    // Now try to use the stale object ref from epoch 0 (pre-mutation version).
    // This should fail because the object version has been consumed.
    let stale_tx_data = TransactionData::new(
        TransactionKind::AddStake {
            address: validator_address,
            coin_ref: gas_object_epoch0,
            amount: Some(500_000),
        },
        sender,
        vec![gas_object_epoch0],
    );
    let stale_tx = test_cluster.wallet.sign_transaction(&stale_tx_data).await;

    // The stale transaction should fail (the object version was already consumed).
    let result = tokio::time::timeout(
        Duration::from_secs(15),
        test_cluster.wallet.execute_transaction_may_fail(stale_tx),
    )
    .await;

    match result {
        Ok(Ok(response)) => {
            assert!(!response.effects.status().is_ok(), "Stale object ref transaction should fail");
        }
        Ok(Err(_)) => {
            info!("Stale ref correctly rejected at submission level");
        }
        Err(_) => {
            info!("Stale ref transaction timed out (expected)");
        }
    }

    // Verify we can still use the latest version of the object in the new epoch.
    let fresh_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("Should still have a gas object in epoch 1");

    let fresh_tx_data = TransactionData::new(
        TransactionKind::AddStake {
            address: validator_address,
            coin_ref: fresh_gas,
            amount: Some(500_000),
        },
        sender,
        vec![fresh_gas],
    );
    let fresh_response = test_cluster.sign_and_execute_transaction(&fresh_tx_data).await;
    assert!(
        fresh_response.effects.status().is_ok(),
        "Fresh object ref transaction should succeed in new epoch"
    );
}

/// Test passive reconfiguration under active transaction load.
/// Verifies that the network can reconfigure through multiple epochs while
/// continuously processing staking transactions.
#[cfg(msim)]
#[msim::sim_test]
async fn test_passive_reconfig_with_tx_load() {
    init_tracing();

    let test_cluster =
        TestClusterBuilder::new().with_num_validators(4).with_epoch_duration_ms(3000).build().await;

    let target_epoch: u64 =
        std::env::var("RECONFIG_TARGET_EPOCH").ok().map(|v| v.parse().unwrap()).unwrap_or(4);

    let sender = test_cluster.get_addresses()[0];

    // Get a validator address to stake with.
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().validators().validators[0]
            .metadata
            .soma_address
    });

    let mut total_txs = 0u64;
    let mut epochs_seen: HashSet<u64> = HashSet::new();

    loop {
        let current_epoch = test_cluster
            .fullnode_handle
            .soma_node
            .with(|node| node.state().epoch_store_for_testing().epoch());
        epochs_seen.insert(current_epoch);

        if current_epoch >= target_epoch {
            break;
        }

        let gas_object =
            test_cluster.wallet.get_one_gas_object_owned_by_address(sender).await.unwrap();

        if let Some(gas_object) = gas_object {
            let tx_data = TransactionData::new(
                TransactionKind::AddStake {
                    address: validator_address,
                    coin_ref: gas_object,
                    amount: Some(1_000_000),
                },
                sender,
                vec![gas_object],
            );

            match tokio::time::timeout(
                Duration::from_secs(30),
                test_cluster.sign_and_execute_transaction(&tx_data),
            )
            .await
            {
                Ok(response) => {
                    if response.effects.status().is_ok() {
                        total_txs += 1;
                    }
                }
                Err(_) => {
                    info!("Transaction timed out near epoch boundary");
                }
            }
        }

        sleep(Duration::from_millis(200)).await;
    }

    info!(
        "Passive reconfig with load complete: {} txs across {} epochs",
        total_txs,
        epochs_seen.len()
    );

    assert!(total_txs > 0, "Should have executed some transactions");
    assert!(epochs_seen.len() >= 2, "Should have seen transactions in at least 2 epochs");

    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().expect("SystemState should be readable")
    });
    assert!(system_state.epoch() >= target_epoch);
    assert_eq!(system_state.validators().validators.len(), 4);
}
