// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Helpers for SystemState-level staking tests. Per-staker reward
// math is exercised at the pool level in `auto_compound_pool_tests`
// (using `StakingPool::deposit_staker_rewards` /
// `StakingPool::pending_compound` directly) — at the SystemState
// layer we only verify aggregate behavior (active_stake, voting
// power, validator removal lifecycle).

use std::collections::BTreeMap;
use std::str::FromStr;

use fastcrypto::traits::KeyPair;
use protocol_config::ProtocolVersion;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::base::SomaAddress;
use crate::bridge::{BridgeCommittee, MarketplaceParameters};
use crate::config::genesis_config::SHANNONS_PER_SOMA;
use crate::crypto::{self, AuthorityKeyPair, NetworkKeyPair};
use crate::multiaddr::Multiaddr;
use crate::object::ObjectID;
use crate::system_state::validator::{Validator, ValidatorRewardCredit};
use crate::system_state::{SystemState, SystemStateTrait};

/// Create a test validator with specified address, stake, and a unique seed
/// for key generation.
pub fn create_validator_for_testing_with_seed(
    addr: SomaAddress,
    init_stake_amount: u64,
    seed: [u8; 32],
    port_base: u16,
) -> Validator {
    let mut rng = StdRng::from_seed(seed);

    let protocol_keypair = AuthorityKeyPair::generate(&mut rng);
    let network_keypair = NetworkKeyPair::generate(&mut rng);
    let worker_keypair = NetworkKeyPair::generate(&mut rng);

    let pop = crypto::generate_proof_of_possession(&protocol_keypair, addr);

    let net_address = Multiaddr::from_str(&format!("/ip4/127.0.0.1/tcp/{}", port_base)).unwrap();
    let p2p_address =
        Multiaddr::from_str(&format!("/ip4/127.0.0.1/tcp/{}", port_base + 1)).unwrap();
    let primary_address =
        Multiaddr::from_str(&format!("/ip4/127.0.0.1/tcp/{}", port_base + 2)).unwrap();

    let mut validator = Validator::new(
        addr,
        protocol_keypair.public().to_owned(),
        network_keypair.public(),
        worker_keypair.public(),
        pop,
        net_address,
        p2p_address,
        primary_address,
        0, // voting_power, set later
        0, // commission_rate
        ObjectID::random(),
    );

    // Auto-compound: seed the pool's active_stake directly (preactive
    // pools accept stake without going through the pending bucket).
    validator.staking_pool.active_stake = init_stake_amount;

    validator
}

/// Create a test validator with specified address and stake amount.
pub fn create_validator_for_testing(addr: SomaAddress, init_stake_amount: u64) -> Validator {
    let addr_bytes: &[u8] = addr.as_ref();
    let mut seed = [0u8; 32];
    let copy_len = std::cmp::min(addr_bytes.len(), 32);
    seed[..copy_len].copy_from_slice(&addr_bytes[..copy_len]);
    let port_base = 10000u16.wrapping_add(u16::from_le_bytes([seed[0], seed[1]]));
    create_validator_for_testing_with_seed(addr, init_stake_amount, seed, port_base)
}

/// Create validators with the given per-validator stake amounts (SOMA, not shannons).
pub fn create_validators_with_stakes(stakes: Vec<u64>) -> Vec<Validator> {
    stakes
        .into_iter()
        .map(|stake| {
            let addr = SomaAddress::random();
            create_validator_for_testing(addr, stake * SHANNONS_PER_SOMA)
        })
        .collect()
}

/// Create a test system state with specified validators and subsidy parameters.
pub fn create_test_system_state(
    validators: Vec<Validator>,
    supply_amount: u64,
    emission_per_epoch: u64,
) -> SystemState {
    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        ProtocolVersion::MAX,
        protocol_config::Chain::default(),
    );
    let epoch_start_timestamp_ms = 1000;
    let stake_subsidy_fund = supply_amount * SHANNONS_PER_SOMA;

    SystemState::create(
        validators,
        ProtocolVersion::MAX.as_u64(),
        epoch_start_timestamp_ms,
        &protocol_config,
        stake_subsidy_fund,
        emission_per_epoch * SHANNONS_PER_SOMA,
        10,
        1000,
        None,
        MarketplaceParameters::default(),
        BridgeCommittee::empty(),
    )
}

/// No-op tracker kept for backwards compatibility with the existing
/// SystemState-level tests. Validator-commission credits are
/// inspected by reading the post-advance system state directly.
#[derive(Clone, Default)]
pub struct ValidatorRewards;

impl ValidatorRewards {
    pub fn new(_validators: &[Validator]) -> Self {
        Self
    }

    pub fn add_commission_rewards(
        &mut self,
        _epoch: u64,
        _rewards: BTreeMap<SomaAddress, ValidatorRewardCredit>,
    ) {
    }
}

/// Drive an epoch with the given SOMA reward budget injected into
/// the emission pool, run advance_epoch with zero fees, drop the
/// returned credit map (test reads pool state directly).
pub fn advance_epoch_with_reward_amounts(
    system_state: &mut SystemState,
    reward_amount: u64,
    _validator_stakes: &mut ValidatorRewards,
) {
    let _ = advance_epoch_returning_credits(system_state, reward_amount);
}

/// Same as [`advance_epoch_with_reward_amounts`] but returns the
/// commission-credit map so tests can assert on per-validator
/// commission directly.
pub fn advance_epoch_returning_credits(
    system_state: &mut SystemState,
    reward_amount: u64,
) -> BTreeMap<SomaAddress, ValidatorRewardCredit> {
    let next_epoch = system_state.epoch() + 1;
    let new_timestamp =
        system_state.epoch_start_timestamp_ms() + system_state.parameters().epoch_duration_ms;

    let reward_shannons = reward_amount * SHANNONS_PER_SOMA;
    match system_state {
        SystemState::V1(v1) => {
            v1.emission_pool.current_distribution_amount = reward_shannons;
            if v1.emission_pool.balance < reward_shannons {
                v1.emission_pool.balance = reward_shannons;
            }
        }
    }

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        ProtocolVersion::MAX,
        protocol_config::Chain::default(),
    );

    system_state
        .advance_epoch(next_epoch, &protocol_config, 0, new_timestamp, vec![0; 32])
        .expect("Failed to advance epoch")
}
