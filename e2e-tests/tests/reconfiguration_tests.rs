use std::{collections::HashSet, time::Duration};

use futures::future::join_all;
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
    system_state::SystemStateTrait,
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
            .validators
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
            node.state()
                .epoch_store_for_testing()
                .committee()
                .num_members(),
            initial_num_validators + 1
        );
    });
    let new_validator_handle = test_cluster.spawn_new_validator(new_validator).await;
    test_cluster.wait_for_epoch_all_nodes(1).await;

    new_validator_handle.with(|node| {
        assert!(
            node.state()
                .is_validator(&node.state().epoch_store_for_testing())
        );
    });

    execute_remove_validator_tx(&test_cluster, &new_validator_handle).await;
    test_cluster.trigger_reconfiguration().await;

    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state()
                .epoch_store_for_testing()
                .committee()
                .num_members(),
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

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(1000)
        .build()
        .await;

    let target_epoch: u64 = std::env::var("RECONFIG_TARGET_EPOCH")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(4);

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

    let mut candidates = (0..6)
        .map(|_| ValidatorGenesisConfigBuilder::new().build(&mut OsRng))
        .collect::<Vec<_>>();
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
        removed_validators
            .iter()
            .all(|v| !committee.authority_exists(v));
    }
}

#[cfg(msim)]
#[msim::sim_test]
async fn test_reconfig_with_voting_power_decrease_normal() {
    init_tracing();

    // This test exercise the full flow of a validator joining the network, catch up and then leave.
    // Validator starts with .12% of the total voting power and then decreases to below the threshold.
    let initial_num_validators = 10;
    let new_validator = ValidatorGenesisConfigBuilder::new()
        .with_stake(0)
        .build(&mut OsRng);

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
                system_state.validators.total_stake,
                system_state
                    .validators
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
            node.state()
                .epoch_store_for_testing()
                .committee()
                .num_members(),
            initial_num_validators + 1
        );
    });

    for (address, stake) in initial_validators
        .iter()
        .map(|address| (*address, default_stake))
        .collect::<Vec<_>>()
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
            .expect("Should be able to get SystemState")
            .validators;

        let candidate = system_state
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == address);

        assert!(candidate.is_some());
        let candidate = candidate.unwrap();

        // Check that the validator voting power has decreased just below the
        // "min" threshold but not below the "low" threshold.
        // Yet the candidate is not at risk.
        assert!(candidate.voting_power < VALIDATOR_CONSENSUS_MIN_POWER);
        assert!(candidate.voting_power > VALIDATOR_CONSENSUS_LOW_POWER);
        assert_eq!(system_state.at_risk_validators.len(), 0);
    });

    // Double validators' stake once again, and check that the new validator is now at risk.
    // Double the stake of every other validator, stake just as much as they had.
    for (address, stake) in initial_validators
        .iter()
        .map(|address| (*address, default_stake))
        .collect::<Vec<_>>()
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
            .expect("Should be able to get SystemState")
            .validators;

        let candidate = system_state
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
        assert_eq!(system_state.at_risk_validators.len(), 1);
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
                .validators
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
    let new_validator = ValidatorGenesisConfigBuilder::new()
        .with_stake(0)
        .build(&mut OsRng);

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
                .expect("Should be able to get SystemState")
                .validators;

            (
                system_state.total_stake,
                system_state
                    .validators
                    .iter()
                    .map(|v| v.metadata.soma_address)
                    .collect::<Vec<_>>(),
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
            node.state()
                .epoch_store_for_testing()
                .committee()
                .num_members(),
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
            .validators
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
        system_state.validators.pending_validators.len()
    });

    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address((&new_validator.account_key_pair.public()).into())
        .await
        .unwrap()
        .expect("Can't get gas object for address");

    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::AddValidator(AddValidatorArgs {
                pubkey_bytes: bcs::to_bytes(&new_validator.key_pair.public()).unwrap(),
                network_pubkey_bytes: bcs::to_bytes(&new_validator.network_key_pair.public())
                    .unwrap(),
                worker_pubkey_bytes: bcs::to_bytes(&new_validator.worker_key_pair.public())
                    .unwrap(),
                net_address: bcs::to_bytes(&new_validator.network_address).unwrap(),
                p2p_address: bcs::to_bytes(&new_validator.p2p_address).unwrap(),
                primary_address: bcs::to_bytes(&new_validator.consensus_address).unwrap(),
            }),
            (&new_validator.account_key_pair.public()).into(),
            vec![gas_object],
        ),
        vec![&new_validator.account_key_pair],
    );

    info!(
        ?tx,
        "Executing add validator tx {}",
        &new_validator.network_address.to_string()
    );

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
        let pending_active_validators = system_state.validators.pending_validators;
        assert_eq!(pending_active_validators.len(), pending_active_count + 1);
        assert_eq!(
            pending_active_validators[pending_active_validators.len() - 1]
                .metadata
                .soma_address,
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
            TransactionKind::AddStake {
                address: address,
                coin_ref: gas_object,
                amount: Some(stake),
            },
            (&signer.public()).into(),
            vec![gas_object],
        ),
        vec![&signer],
    );

    info!(?tx, "Executing stake validator tx {}", address.to_string());

    let _response = test_cluster.execute_transaction(tx).await;
}

// TODO: async fn test_inactive_validator_pool_read()
// TODO: async fn test_validator_candidate_pool_read()
// TODO: async fn test_reconfig_with_failing_validator(
// TODO: async fn test_create_advance_epoch_tx_race()
// TODO: async fn test_expired_locks()
// TODO: async fn do_test_passive_reconfig()
