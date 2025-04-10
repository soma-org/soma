use crate::{
    base::SomaAddress,
    committee::TOTAL_VOTING_POWER,
    crypto::{self, AuthorityKeyPair, NetworkKeyPair, NetworkPublicKey, ProtocolKeyPair},
    effects::ExecutionFailureStatus,
    error::ExecutionResult,
    multiaddr::Multiaddr,
    system_state::{
        staking::{PoolTokenExchangeRate, StakedSoma, StakingPool},
        subsidy::StakeSubsidy,
        validator::{Validator, ValidatorSet},
        PublicKey, SystemParameters, SystemState,
    },
};
use fastcrypto::{
    bls12381,
    ed25519::Ed25519PublicKey,
    traits::{KeyPair, ToFromBytes},
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::str::FromStr;

// Constants for testing
pub const SHANNONS_PER_SOMA: u64 = 1_000_000_000;

/// Create a test validator with specified address and stake amount
pub fn create_validator_for_testing(addr: SomaAddress, init_stake_amount: u64) -> Validator {
    let mut rng = StdRng::from_seed([0; 32]);

    // Create protocol public key (BLS)
    let protocol_keypair = AuthorityKeyPair::generate(&mut rng);

    // Create network public key (ED25519)
    let network_keypair = NetworkKeyPair::generate(&mut rng);

    // Create worker public key (ED25519)
    let worker_keypair = NetworkKeyPair::generate(&mut rng);

    // Create multiaddresses
    let net_address = Multiaddr::from_str("/ip4/127.0.0.1/tcp/8080").unwrap();
    let p2p_address = Multiaddr::from_str("/ip4/127.0.0.1/tcp/8081").unwrap();
    let primary_address = Multiaddr::from_str("/ip4/127.0.0.1/tcp/8082").unwrap();

    // Commission rate (10% in basis points)
    let commission_rate = 1000;

    // Create validator
    let mut validator = Validator::new(
        addr,
        protocol_keypair.public().to_owned(),
        network_keypair.public(),
        worker_keypair.public(),
        net_address,
        p2p_address,
        primary_address,
        0, // Initial voting power is 0, will be set later
        commission_rate,
    );

    // Initialize staking pool with stake
    validator.staking_pool.soma_balance = init_stake_amount;
    validator.staking_pool.pool_token_balance = init_stake_amount;

    validator
}

/// Create validators with specified stake amounts
pub fn create_validators_with_stakes(stakes: Vec<u64>) -> Vec<Validator> {
    let mut validators = Vec::new();

    for (i, &stake) in stakes.iter().enumerate() {
        let addr = SomaAddress::random(); // Generate a random address
        let validator = create_validator_for_testing(addr, stake * SHANNONS_PER_SOMA);
        validators.push(validator);
    }

    validators
}

/// Create a test system state with specified validators and subsidy parameters
pub fn create_test_system_state(
    validators: Vec<Validator>,
    sui_supply_amount: u64,
    stake_subsidy_initial_amount: u64,
    stake_subsidy_period_length: u64,
    stake_subsidy_decrease_rate: u16,
) -> SystemState {
    // System parameters
    let parameters = SystemParameters {
        epoch_duration_ms: 42, // Doesn't matter what number we put here for tests
        min_validator_joining_stake: 1 * SHANNONS_PER_SOMA,
        validator_low_stake_threshold: 1 * SHANNONS_PER_SOMA,
        validator_very_low_stake_threshold: 0,
        validator_low_stake_grace_period: 7,
    };

    // Create system state
    let epoch_start_timestamp_ms = 1000;
    let stake_subsidy_fund = sui_supply_amount * SHANNONS_PER_SOMA;

    SystemState::create(
        validators,
        epoch_start_timestamp_ms,
        parameters,
        stake_subsidy_fund,
        stake_subsidy_initial_amount * SHANNONS_PER_SOMA,
        stake_subsidy_period_length,
        stake_subsidy_decrease_rate,
    )
}

/// Setup a system state with specified validator addresses
pub fn set_up_system_state(addrs: Vec<SomaAddress>) -> SystemState {
    let mut validators = Vec::new();

    for addr in addrs {
        validators.push(create_validator_for_testing(addr, 100 * SHANNONS_PER_SOMA));
    }

    create_test_system_state(validators, 1000, 0, 10, 500)
}

/// Advance epoch with rewards
pub fn advance_epoch_with_rewards(
    system_state: &mut SystemState,
    storage_charge: u64,
    computation_charge: u64,
) -> ExecutionResult<()> {
    // Calculate next epoch
    let next_epoch = system_state.epoch + 1;

    // Calculate new timestamp (ensuring it's at least epoch_duration_ms later)
    let new_timestamp =
        system_state.epoch_start_timestamp_ms + system_state.parameters.epoch_duration_ms;

    // Advance the epoch
    system_state.advance_epoch(next_epoch, new_timestamp)
}

/// Request to add stake to a validator
pub fn stake_with(
    system_state: &mut SystemState,
    staker: SomaAddress,
    validator: SomaAddress,
    amount: u64,
) -> ExecutionResult<StakedSoma> {
    system_state.request_add_stake(staker, validator, amount * SHANNONS_PER_SOMA)
}

/// Request to withdraw stake
pub fn unstake(system_state: &mut SystemState, staked_soma: StakedSoma) -> ExecutionResult<u64> {
    system_state.request_withdraw_stake(staked_soma)
}

/// Request to add a validator
pub fn add_validator(
    system_state: &mut SystemState,
    validator_address: SomaAddress,
    pubkey_bytes: Vec<u8>,
    network_pubkey_bytes: Vec<u8>,
    worker_pubkey_bytes: Vec<u8>,
    net_address: Vec<u8>,
    p2p_address: Vec<u8>,
    primary_address: Vec<u8>,
) -> ExecutionResult {
    system_state.request_add_validator(
        validator_address,
        pubkey_bytes,
        network_pubkey_bytes,
        worker_pubkey_bytes,
        net_address,
        p2p_address,
        primary_address,
    )
}

/// Request to remove a validator
pub fn remove_validator(
    system_state: &mut SystemState,
    validator_address: SomaAddress,
    pubkey_bytes: Vec<u8>,
) -> ExecutionResult {
    system_state.request_remove_validator(validator_address, pubkey_bytes)
}

/// Calculate the total stake of a validator including rewards
pub fn validator_stake_amount(
    system_state: &SystemState,
    validator_address: SomaAddress,
) -> Option<u64> {
    for validator in &system_state.validators.active_validators {
        if validator.metadata.soma_address == validator_address {
            return Some(validator.staking_pool.soma_balance);
        }
    }
    None
}

/// Calculate the self-stake plus rewards for a validator
pub fn stake_plus_current_rewards_for_validator(
    system_state: &SystemState,
    validator_address: SomaAddress,
) -> Option<u64> {
    for validator in &system_state.validators.active_validators {
        if validator.metadata.soma_address == validator_address {
            return Some(validator.staking_pool.soma_balance);
        }
    }
    None
}

/// Assert that validators have the expected self-stake amounts
pub fn assert_validator_self_stake_amounts(
    system_state: &SystemState,
    validator_addrs: Vec<SomaAddress>,
    stake_amounts: Vec<u64>,
) {
    assert_eq!(
        validator_addrs.len(),
        stake_amounts.len(),
        "Address and amount arrays must have the same length"
    );

    for (i, addr) in validator_addrs.iter().enumerate() {
        let expected_amount = stake_amounts[i] * SHANNONS_PER_SOMA;
        let actual_amount = stake_plus_current_rewards_for_validator(system_state, *addr)
            .expect("Validator not found");

        assert_eq!(
            actual_amount, expected_amount,
            "Validator {} expected stake {} but got {}",
            addr, expected_amount, actual_amount
        );
    }
}

/// Assert that validators have the expected total stake amounts
pub fn assert_validator_total_stake_amounts(
    system_state: &SystemState,
    validator_addrs: Vec<SomaAddress>,
    stake_amounts: Vec<u64>,
) {
    assert_eq!(
        validator_addrs.len(),
        stake_amounts.len(),
        "Address and amount arrays must have the same length"
    );

    for (i, addr) in validator_addrs.iter().enumerate() {
        let expected_amount = stake_amounts[i] * SHANNONS_PER_SOMA;
        let actual_amount =
            validator_stake_amount(system_state, *addr).expect("Validator not found");

        assert_eq!(
            actual_amount, expected_amount,
            "Validator {} expected stake {} but got {}",
            addr, expected_amount, actual_amount
        );
    }
}

/// Distribute rewards and advance epoch (similar to Move's distribute_rewards_and_advance_epoch)
pub fn distribute_rewards_and_advance_epoch(
    system_state: &mut SystemState,
    reward_amount: u64,
) -> u64 {
    // Calculate next epoch
    let next_epoch = system_state.epoch + 1;

    // Calculate new timestamp (ensuring it's at least epoch_duration_ms later)
    let new_timestamp =
        system_state.epoch_start_timestamp_ms + system_state.parameters.epoch_duration_ms;

    // Advance the epoch
    system_state
        .advance_epoch(next_epoch, new_timestamp)
        .unwrap();

    // Return new epoch number
    system_state.epoch
}
