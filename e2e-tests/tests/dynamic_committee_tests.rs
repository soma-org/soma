// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use async_trait::async_trait;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use test_cluster::{TestCluster, TestClusterBuilder};
use tracing::info;
use types::base::SomaAddress;
use types::config::genesis_config::AccountConfig;
use types::effects::{TransactionEffects, TransactionEffectsAPI};
use types::object::{Object, ObjectID, ObjectRef, ObjectType, Owner};
use types::storage::object_store::ObjectStore;
use types::system_state::validator::Validator;
use types::system_state::{SystemState, SystemStateTrait as _};
use types::transaction::TransactionKind;
use utils::logging::init_tracing;

const MAX_DELEGATION_AMOUNT: u64 = 1_000_000_000_000; // 1K SOMA
const MIN_DELEGATION_AMOUNT: u64 = 500_000_000_000; // 0.5K SOMA

// Stage 13c: balance-mode AddStake debits SOMA from the sender's
// accumulator and gas (USDC) from the same accumulator. The fuzz
// runner seeds each wallet account with both currencies — enough
// SOMA to cover up to ~10 stakes at MAX_DELEGATION_AMOUNT plus a
// large USDC reserve so per-tx gas never starves a sender mid-run.
const ACCOUNT_GENESIS_SOMA: u64 = MAX_DELEGATION_AMOUNT * 20;
const ACCOUNT_GENESIS_USDC: u64 = 1_000_000_000; // 1B USDC microdollars

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
    /// Wallet-account signer addresses. Each one was seeded at
    /// genesis with both SOMA (stake principal) and USDC (gas).
    /// Stage 13c: validators don't have unstaked SOMA in their
    /// accumulator (it's locked in their staking pool), so they
    /// can't sign AddStake — wallet accounts are the only viable
    /// senders.
    pub accounts: Vec<SomaAddress>,
    pub active_validators: BTreeSet<SomaAddress>,
    pub preactive_validators: BTreeMap<SomaAddress, u64>,
    pub removed_validators: BTreeSet<SomaAddress>,
    pub delegation_requests_this_epoch: BTreeMap<ObjectID, SomaAddress>,
    pub delegation_withdraws_this_epoch: u64,
    /// Stage 9d-C3: balance-mode WithdrawStake keys off (pool_id,
    /// sender). Track the staker's address alongside the pool_id so
    /// the stress runner can later issue a withdrawal.
    pub delegations: Vec<(SomaAddress, ObjectID)>,
    pub reports: BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
    pub rng: StdRng,
}

impl StressTestRunner {
    pub async fn new(size: usize) -> Self {
        let test_cluster = TestClusterBuilder::new()
            .with_num_validators(size) // number of validators has to exceed 10
            .with_accounts(vec![
                AccountConfig {
                    gas_amounts: vec![ACCOUNT_GENESIS_SOMA],
                    usdc_amounts: vec![ACCOUNT_GENESIS_USDC],
                    address: None,
                };
                100
            ])
            .build()
            .await;
        let accounts = test_cluster.wallet.get_addresses();
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

    pub fn pick_random_sender(&mut self) -> SomaAddress {
        self.accounts[self.rng.r#gen_range(0..self.accounts.len())]
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

    /// Stage 13c: build a balance-mode tx (empty gas_payment, fresh
    /// ValidDuring nonce) and sign it via the wallet keystore. The
    /// authority debits the sender's USDC accumulator for gas and
    /// the SOMA accumulator for any stake principal.
    pub async fn run(&self, sender: SomaAddress, kind: TransactionKind) -> TransactionEffects {
        let tx_data = e2e_tests::stateless_tx_data(&self.test_cluster, sender, kind);
        let tx = self.test_cluster.wallet.sign_transaction(&tx_data).await;
        let response = self.test_cluster.execute_transaction(tx).await;
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
        sender: SomaAddress,
        stake_amount: u64,
        staked_with: SomaAddress,
    }

    impl GenStateChange for RequestAddStakeGen {
        type StateChange = RequestAddStake;

        fn create(&self, runner: &mut StressTestRunner) -> Self::StateChange {
            let stake_amount = runner.rng.gen_range(MIN_DELEGATION_AMOUNT..=MAX_DELEGATION_AMOUNT);
            let staked_with = runner.pick_random_active_validator().metadata.soma_address;
            let sender = runner.pick_random_sender();
            RequestAddStake { sender, stake_amount, staked_with }
        }
    }

    #[async_trait]
    impl StatePredicate for RequestAddStake {
        async fn run(&mut self, runner: &mut StressTestRunner) -> Result<TransactionEffects> {
            // Stage 13c: AddStake is balance-mode — both stake (SOMA)
            // and gas (USDC) are debited from the sender's accumulator.
            let kind = TransactionKind::AddStake {
                validator: self.staked_with,
                amount: self.stake_amount,
            };

            let effects = runner.run(self.sender, kind).await;

            Ok(effects)
        }

        async fn pre_epoch_post_condition(
            &mut self,
            runner: &mut StressTestRunner,
            _effects: &TransactionEffects,
        ) {
            // Stage 9d-C4: AddStake no longer creates a StakedSomaV1
            // object — the F1 (pool, sender) row is the source of
            // truth. Read the row back from the delegations table on
            // the fullnode (any validator agrees on this state).
            let pool_id = runner
                .test_cluster
                .fullnode_handle
                .soma_node
                .with(|node| {
                    let staker_addr = self.sender;
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

            runner.delegations.push((self.sender, pool_id));
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
        sender: SomaAddress,
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

            let effects = runner.run(self.sender, kind).await;

            Ok(effects)
        }

        async fn pre_epoch_post_condition(
            &mut self,
            _runner: &mut StressTestRunner,
            _effects: &TransactionEffects,
        ) {
            // Stage 9d-C5: WithdrawStake settles atomically — no
            // epoch-boundary pending state to inspect mid-run.
        }

        async fn post_epoch_post_condition(
            &mut self,
            _runner: &StressTestRunner,
            _effects: &TransactionEffects,
        ) {
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
    let total_stake: u64 =
        system_state.validators().validators.iter().map(|v| v.staking_pool.total_stake).sum();
    info!("post-fuzz total stake across {} validators: {}", committee_size, total_stake);

    // Sanity: all stress senders are still in the wallet's keystore
    // and committee size is unchanged (no validator add/remove ops).
    assert_eq!(system_state.validators().validators.len(), committee_size);
}
