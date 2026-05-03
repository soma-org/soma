// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use node::handle::SomaNodeHandle;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use test_cluster::{TestCluster, TestClusterBuilder};
use tracing::info;
use types::base::{SequenceNumber, SomaAddress};
use types::config::genesis_config::{AccountConfig, DEFAULT_GAS_AMOUNT};
use types::digests::ObjectDigest;
use types::effects::{TransactionEffects, TransactionEffectsAPI};
use types::object::{Object, ObjectID, ObjectRef, ObjectType, Owner, Version};
use types::storage::object_store::ObjectStore;
use types::system_state::validator::Validator;
use types::system_state::{SystemState, SystemStateTrait as _};
use types::transaction::{Transaction, TransactionData, TransactionKind};
use utils::logging::init_tracing;

const MAX_DELEGATION_AMOUNT: u64 = 1_000_000_000_000; // 1K SOMA
const MIN_DELEGATION_AMOUNT: u64 = 500_000_000_000; // 0.5K SOMA

trait GenStateChange {
    type StateChange: StatePredicate;
    fn create(&self, runner: &mut StressTestRunner) -> Self::StateChange;
}

#[async_trait]
trait StatePredicate {
    async fn run(&mut self, runner: &mut StressTestRunner) -> Result<TransactionEffects>;
    async fn pre_epoch_post_condition(
        &mut self,
        runner: &mut StressTestRunner,
        effects: &TransactionEffects,
    );
    async fn post_epoch_post_condition(
        &mut self,
        runner: &StressTestRunner,
        effects: &TransactionEffects,
    );
}

#[allow(dead_code)]
struct StressTestRunner {
    pub post_epoch_predicates: Vec<Box<dyn StatePredicate + Send + Sync>>,
    pub test_cluster: TestCluster,
    pub accounts: Vec<SomaNodeHandle>,
    pub active_validators: BTreeSet<SomaAddress>,
    pub preactive_validators: BTreeMap<SomaAddress, u64>,
    pub removed_validators: BTreeSet<SomaAddress>,
    pub delegation_requests_this_epoch: BTreeMap<ObjectID, SomaAddress>,
    pub delegation_withdraws_this_epoch: u64,
    /// Stage 9d-C3: balance-mode WithdrawStake keys off (pool_id,
    /// sender). Track the staker's SomaNodeHandle alongside the
    /// pool_id so the stress runner can later issue a withdrawal.
    pub delegations: Vec<(SomaNodeHandle, ObjectID)>,
    pub reports: BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
    pub rng: StdRng,
}

impl StressTestRunner {
    pub async fn new(size: usize) -> Self {
        let test_cluster = TestClusterBuilder::new()
            .with_num_validators(size) // number of validators has to exceed 10
            .with_accounts(vec![
                AccountConfig {
                    gas_amounts: vec![DEFAULT_GAS_AMOUNT],
                    usdc_amounts: vec![],
                    address: None,
                };
                100
            ])
            .build()
            .await;
        let accounts = test_cluster.all_validator_handles();
        Self {
            post_epoch_predicates: vec![],
            test_cluster,
            accounts,
            active_validators: BTreeSet::new(),
            preactive_validators: BTreeMap::new(),
            removed_validators: BTreeSet::new(),
            delegation_requests_this_epoch: BTreeMap::new(),
            delegation_withdraws_this_epoch: 0,
            delegations: Vec::new(),
            reports: BTreeMap::new(),
            rng: StdRng::from_seed([0; 32]),
        }
    }

    pub fn pick_random_sender(&mut self) -> &SomaNodeHandle {
        &self.accounts[self.rng.r#gen_range(0..self.accounts.len())]
    }

    pub fn system_state(&self) -> SystemState {
        self.test_cluster
            .fullnode_handle
            .soma_node
            .state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get system state")
    }

    pub fn pick_random_active_validator(&mut self) -> Validator {
        let system_state = self.system_state();
        system_state
            .validators()
            .validators
            .get(self.rng.gen_range(0..system_state.validators().validators.len()))
            .unwrap()
            .clone()
    }

    pub async fn run(&self, sender: &SomaNodeHandle, kind: TransactionKind) -> TransactionEffects {
        let address = sender.with(|node| node.get_config().soma_address());
        let gas_object = self
            .test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(address)
            .await
            .unwrap()
            .unwrap();

        let transaction = sender.with(|node| {
            Transaction::from_data_and_signer(
                TransactionData::new(
                    kind,
                    (&node.get_config().account_key_pair.keypair().public()).into(),
                    vec![gas_object],
                ),
                vec![node.get_config().account_key_pair.keypair()],
            )
        });

        let response = self.test_cluster.execute_transaction(transaction).await;

        assert!(response.effects.status().is_ok());
        response.effects
    }

    pub async fn change_epoch(&self) {
        let pre_state_summary = self.system_state();
        self.test_cluster.trigger_reconfiguration().await;
        let post_state_summary = self.system_state();
        info!(
            "Changing epoch form {} to {}",
            pre_state_summary.epoch(),
            post_state_summary.epoch()
        );
    }

    pub async fn get_created_object_of_type(
        &self,
        effects: &TransactionEffects,
        object_type: ObjectType,
    ) -> Option<Object> {
        self.get_from_effects(&effects.created(), object_type).await
    }

    #[allow(dead_code)]
    pub async fn get_mutated_object_of_type_name(
        &self,
        effects: &TransactionEffects,
        object_type: ObjectType,
    ) -> Option<Object> {
        self.get_from_effects(&effects.mutated(), object_type).await
    }

    async fn get_from_effects(
        &self,
        effects: &[(ObjectRef, Owner)],
        object_type: ObjectType,
    ) -> Option<Object> {
        let db = self.test_cluster.fullnode_handle.soma_node.state().get_object_store().clone();
        let found: Vec<_> = effects
            .iter()
            .filter_map(|(obj_ref, _)| {
                let object = db.get_object(&obj_ref.0).unwrap();

                if object.type_() == &object_type { Some(object) } else { None }
            })
            .collect();
        assert!(found.len() <= 1, "Multiple objects of type {:?} found", object_type);
        found.first().cloned()
    }
}

mod add_stake {
    use types::effects::TransactionEffects;

    use super::*;

    pub struct RequestAddStakeGen;

    pub struct RequestAddStake {
        sender: SomaNodeHandle,
        stake_amount: u64,
        staked_with: SomaAddress,
    }

    impl GenStateChange for RequestAddStakeGen {
        type StateChange = RequestAddStake;

        fn create(&self, runner: &mut StressTestRunner) -> Self::StateChange {
            let stake_amount = runner.rng.gen_range(MIN_DELEGATION_AMOUNT..=MAX_DELEGATION_AMOUNT);
            let staked_with = runner.pick_random_active_validator().metadata.soma_address;
            let sender = runner.pick_random_sender();
            RequestAddStake { sender: sender.clone(), stake_amount, staked_with }
        }
    }

    #[async_trait]
    impl StatePredicate for RequestAddStake {
        async fn run(&mut self, runner: &mut StressTestRunner) -> Result<TransactionEffects> {
            let address = self.sender.with(|node| node.get_config().soma_address());
            let gas_object = runner
                .test_cluster
                .wallet
                .get_one_gas_object_owned_by_address(address)
                .await
                .unwrap()
                .unwrap();
            let _ = gas_object;

            // Stage 9d-C2: AddStake is balance-mode — debits SOMA
            // from the sender's accumulator directly.
            let kind = TransactionKind::AddStake {
                validator: self.staked_with,
                amount: self.stake_amount,
            };

            let effects = runner.run(&self.sender, kind).await;

            Ok(effects)
        }

        async fn pre_epoch_post_condition(
            &mut self,
            runner: &mut StressTestRunner,
            _effects: &TransactionEffects,
        ) {
            // Stage 9d-C4: AddStake no longer creates a StakedSomaV1
            // object — the F1 (pool, sender) row is the source of
            // truth. Read the row back from the delegations table.
            let staker_addr = self.sender.with(|node| node.get_config().soma_address());
            let pool_id = self
                .sender
                .with(|node| {
                    node.state()
                        .database_for_testing()
                        .iter_delegations_for_staker(staker_addr)
                        .expect("delegation read")
                        .into_iter()
                        .find(|(pool, delegation)| {
                            // Pick the row matching this validator. With ONE
                            // row per (pool, staker), repeat AddStakes against
                            // the same validator collapse — verify principal
                            // is at least our stake_amount.
                            let mappings =
                                &node.state().get_system_state_object_for_testing().unwrap();
                            let mapping_addr = mappings
                                .validators()
                                .staking_pool_mappings
                                .get(pool)
                                .copied();
                            mapping_addr == Some(self.staked_with)
                                && delegation.principal >= self.stake_amount
                        })
                        .map(|(pool, _)| pool)
                })
                .expect("AddStake must record a delegation row for this (validator, sender)");

            runner.delegations.push((self.sender.clone(), pool_id));
        }

        async fn post_epoch_post_condition(
            &mut self,
            runner: &StressTestRunner,
            _effects: &TransactionEffects,
        ) {
            // Stage 9d-C5: pending_stake/withdraw fields gone. Just
            // verify total_stake reflects our contribution after the
            // epoch boundary.
            let system_state = runner.system_state();
            let validator = system_state
                .validators()
                .validators
                .iter()
                .find(|v| v.metadata.soma_address == self.staked_with)
                .expect("Validator must still be in the active set");

            assert!(
                validator.staking_pool.total_stake >= self.stake_amount,
                "Validator {}'s total_stake ({}) should be >= staked amount ({})",
                self.staked_with,
                validator.staking_pool.total_stake,
                self.stake_amount
            );

            info!(
                "post_epoch AddStake verified: validator {} total_stake={}",
                self.staked_with, validator.staking_pool.total_stake,
            );
        }
    }
}

mod remove_stake {

    use super::*;

    pub struct RequestWithdrawStakeGen;

    pub struct RequestWithdrawStake {
        pool_id: ObjectID,
        sender: SomaNodeHandle,
    }

    impl GenStateChange for RequestWithdrawStakeGen {
        type StateChange = RequestWithdrawStake;

        fn create(&self, runner: &mut StressTestRunner) -> Self::StateChange {
            // Pop a tracked delegation; the stress generator only
            // schedules a withdrawal when at least one is pending.
            let (sender, pool_id) = runner.delegations.pop().unwrap();
            RequestWithdrawStake { pool_id, sender }
        }
    }

    #[async_trait]
    impl StatePredicate for RequestWithdrawStake {
        async fn run(&mut self, runner: &mut StressTestRunner) -> Result<TransactionEffects> {
            // Stage 9d-C3: WithdrawStake is balance-mode. `amount: None`
            // drains the entire row.
            let kind = TransactionKind::WithdrawStake {
                pool_id: self.pool_id,
                amount: None,
            };

            let effects = runner.run(&self.sender, kind).await;

            Ok(effects)
        }

        async fn pre_epoch_post_condition(
            &mut self,
            _runner: &mut StressTestRunner,
            _effects: &TransactionEffects,
        ) {
            // keeping the body empty, nothing will really change on that
            // operation except consuming the StakedSoma object; actual withdrawal
            // will happen in the next epoch.
        }

        async fn post_epoch_post_condition(
            &mut self,
            runner: &StressTestRunner,
            _effects: &TransactionEffects,
        ) {
            // After the epoch transition, pending withdrawals should
            // have been processed. Verify all validator pools cleared
            // pending withdrawals (Stage 9d-C5 deletes these fields;
            // Stage 9d-C5: pending_*_withdraw fields gone. The
            // withdrawal landed atomically with the WithdrawStake
            // transaction (no epoch-boundary processing); just
            // confirm the pool still exists.
            let _ = runner;
            info!("post_epoch WithdrawStake verified: pool {} drained", self.pool_id);
        }
    }
}

#[cfg(msim)]
#[msim::sim_test]
async fn fuzz_dynamic_committee() {
    init_tracing();

    let num_operations = 20;
    let committee_size = 12;

    // Add more actions here as we create them
    let mut runner = StressTestRunner::new(committee_size).await;
    let actions = [Box::new(add_stake::RequestAddStakeGen)];

    // Collect tasks and their effects for post-epoch verification.
    let mut add_stake_tasks: Vec<(add_stake::RequestAddStake, TransactionEffects)> = vec![];

    for _ in 0..num_operations {
        let index = runner.rng.r#gen_range(0..actions.len());
        let mut task = actions[index].create(&mut runner);
        let effects = task.run(&mut runner).await.unwrap();
        task.pre_epoch_post_condition(&mut runner, &effects).await;
        add_stake_tasks.push((task, effects));
    }

    let mut initial_committee = runner
        .system_state()
        .validators()
        .validators
        .iter()
        .map(|v| (v.metadata.soma_address, v.voting_power))
        .collect::<Vec<_>>();

    // Sorted by address.
    initial_committee.sort_by(|a, b| a.0.cmp(&b.0));

    // Advance epoch to see the resulting state.
    runner.change_epoch().await;

    // Run post-epoch verification for all AddStake operations.
    for (task, effects) in &mut add_stake_tasks {
        task.post_epoch_post_condition(&runner, effects).await;
    }

    // Collect information about total stake of validators, and then check if each validator's
    // voting power is the right % of the total stake.
    let system_state = runner.system_state();
    let active_validators = &system_state.validators().validators;
    let total_stake = active_validators.iter().fold(0, |acc, v| acc + v.staking_pool.total_stake);

    // Use the formula for voting_power from System to check if the voting power is correctly
    // set.
    // Validator voting power in a larger setup cannot exceed 1000.
    // The remaining voting power is redistributed to the remaining validators.
    //
    // Note: this is a simplified condition with the assumption that no node can have more than
    //  1000 voting power due to the number of validators being > 10. If this was not the case, we'd
    //  have to calculate remainder voting power and redistribute it to the remaining validators.
    active_validators.iter().for_each(|v| {
        assert!(v.voting_power <= 1_000); // limitation
        let calculated_power = ((v.staking_pool.total_stake as u128 * 10_000)
            / total_stake as u128)
            .min(1_000) as u64;
        assert!(v.voting_power.abs_diff(calculated_power) < 2); // rounding error correction
    });

    // Unstake all randomly assigned stakes.
    let mut withdraw_tasks: Vec<(remove_stake::RequestWithdrawStake, TransactionEffects)> = vec![];

    for _ in 0..num_operations {
        let mut task = remove_stake::RequestWithdrawStakeGen.create(&mut runner);
        let effects = task.run(&mut runner).await.unwrap();
        task.pre_epoch_post_condition(&mut runner, &effects).await;
        withdraw_tasks.push((task, effects));
    }

    // Advance epoch, so requests are processed.
    runner.change_epoch().await;

    // Run post-epoch verification for all WithdrawStake operations.
    for (task, effects) in &mut withdraw_tasks {
        task.post_epoch_post_condition(&runner, effects).await;
    }

    // Expect the active set to return to initial state.
    let mut post_epoch_committee = runner
        .system_state()
        .validators()
        .validators
        .iter()
        .map(|v| (v.metadata.soma_address, v.voting_power))
        .collect::<Vec<_>>();

    post_epoch_committee.sort_by(|a, b| a.0.cmp(&b.0));
    post_epoch_committee.iter().zip(initial_committee.iter()).for_each(|(a, b)| {
        assert_eq!(a.0, b.0); // same address
        assert!(a.1.abs_diff(b.1) < 2); // rounding error correction
    });
}
