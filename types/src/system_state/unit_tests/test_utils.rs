use crate::{
    base::SomaAddress,
    committee::TOTAL_VOTING_POWER,
    config::genesis_config::SHANNONS_PER_SOMA,
    crypto::{self, AuthorityKeyPair, NetworkKeyPair, NetworkPublicKey, ProtocolKeyPair},
    effects::ExecutionFailureStatus,
    encoder_validator,
    error::ExecutionResult,
    multiaddr::Multiaddr,
    object::ObjectID,
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
use shared::crypto::keys::EncoderKeyPair;
use std::{
    collections::{BTreeMap, HashMap},
    str::FromStr,
};
use tracing_subscriber::fmt::init;

use super::encoder::Encoder;

#[cfg(test)]
#[derive(Clone)]
pub struct ValidatorRewards {
    // Initial stake amounts for each validator
    initial_stakes: BTreeMap<SomaAddress, u64>,

    // Commission rewards for validators per epoch
    // First key is validator address, second key is epoch, value is StakedSoma
    commission_rewards: BTreeMap<SomaAddress, BTreeMap<u64, StakedSoma>>,
}

#[cfg(test)]
impl ValidatorRewards {
    /// Create a new ValidatorRewards tracker
    pub fn new(validators: &[Validator]) -> Self {
        let mut initial_stakes = BTreeMap::new();
        for validator in validators {
            initial_stakes.insert(
                validator.metadata.soma_address,
                validator.staking_pool.soma_balance,
            );
        }

        Self {
            initial_stakes,
            commission_rewards: BTreeMap::new(),
        }
    }

    /// Get the initial stake for a validator
    pub fn get_initial_stake(&self, validator_addr: SomaAddress) -> u64 {
        *self.initial_stakes.get(&validator_addr).unwrap_or(&0)
    }

    /// Add commission rewards for an epoch
    pub fn add_commission_rewards(
        &mut self,
        epoch: u64,
        rewards: HashMap<SomaAddress, StakedSoma>,
    ) {
        for (addr, staked_soma) in rewards {
            self.commission_rewards
                .entry(addr)
                .or_insert_with(BTreeMap::new)
                .insert(epoch, staked_soma);
        }
    }

    /// Calculate a validator's self-stake including initial stake rewards and commission rewards
    pub fn calculate_self_stake_with_rewards(
        &self,
        validator: &Validator,
        current_epoch: u64,
    ) -> u64 {
        let addr = validator.metadata.soma_address;
        let initial_stake = self.get_initial_stake(addr);

        // Calculate rewards for initial stake
        let mut self_stake = validator.calculate_rewards(
            initial_stake,
            0, // Initial stake is active from epoch 0
            current_epoch,
        );

        // Add commission rewards
        if let Some(rewards_by_epoch) = self.commission_rewards.get(&addr) {
            for (&reward_epoch, staked_soma) in rewards_by_epoch {
                if reward_epoch <= current_epoch {
                    // Calculate rewards for this commission reward
                    let reward_with_growth = validator.staking_pool.calculate_rewards(
                        staked_soma.principal,
                        staked_soma.stake_activation_epoch,
                        current_epoch,
                    );
                    self_stake += reward_with_growth;
                }
            }
        }

        self_stake
    }
}

// Helper function to request to add stake to a validator
pub fn stake_with(
    system_state: &mut SystemState,
    staker: SomaAddress,
    validator: SomaAddress,
    amount: u64,
) -> StakedSoma {
    system_state
        .request_add_stake(staker, validator, amount * SHANNONS_PER_SOMA)
        .expect("Failed to add stake")
}

// Helper function to request to withdraw stake
pub fn unstake(system_state: &mut SystemState, staked_soma: StakedSoma) -> u64 {
    system_state
        .request_withdraw_stake(staked_soma)
        .expect("Failed to withdraw stake")
}

// Helper function to distribute rewards and advance epoch
pub fn advance_epoch_with_reward_amounts(
    system_state: &mut SystemState,
    reward_amount: u64,
    validator_stakes: &mut ValidatorRewards,
) {
    // Calculate next epoch
    let next_epoch = system_state.epoch + 1;

    // Calculate new timestamp (ensuring it's at least epoch_duration_ms later)
    let new_timestamp =
        system_state.epoch_start_timestamp_ms + system_state.parameters.epoch_duration_ms;

    // Advance the epoch
    let rewards = system_state
        .advance_epoch(
            next_epoch,
            reward_amount * SHANNONS_PER_SOMA,
            new_timestamp,
            1000,
        )
        .expect("Failed to advance epoch");

    validator_stakes.add_commission_rewards(next_epoch, rewards);
}

// Helper function to advance epoch with reward amounts and slashing rates
pub fn advance_epoch_with_reward_amounts_and_slashing_rates(
    system_state: &mut SystemState,
    reward_amount: u64,
    reward_slashing_rate: u64,
    validator_stakes: &mut ValidatorRewards,
) {
    // Calculate next epoch
    let next_epoch = system_state.epoch + 1;

    // Calculate new timestamp (ensuring it's at least epoch_duration_ms later)
    let new_timestamp =
        system_state.epoch_start_timestamp_ms + system_state.parameters.epoch_duration_ms;

    // Advance the epoch
    let rewards = system_state
        .advance_epoch(
            next_epoch,
            reward_amount * SHANNONS_PER_SOMA,
            new_timestamp,
            reward_slashing_rate,
        )
        .expect("Failed to advance epoch");

    validator_stakes.add_commission_rewards(next_epoch, rewards);
}

// Helper to assert validator total stake amounts
pub fn assert_validator_total_stake_amounts(
    system_state: &SystemState,
    validator_addrs: Vec<SomaAddress>,
    expected_amounts: Vec<u64>,
) {
    assert_eq!(
        validator_addrs.len(),
        expected_amounts.len(),
        "Address and amount arrays must be the same length"
    );

    for (i, addr) in validator_addrs.iter().enumerate() {
        let validator = system_state
            .validators
            .consensus_validators
            .iter()
            .find(|v| v.metadata.soma_address == *addr)
            .expect("Validator not found");

        let actual_amount = validator.staking_pool.soma_balance;
        let expected_amount = expected_amounts[i];

        assert_eq!(
            actual_amount, expected_amount,
            "Validator {} expected stake {}, but got {}",
            addr, expected_amount, actual_amount
        );
    }
}

// Helper to assert validator self stake amounts (principal + rewards)
pub fn assert_validator_self_stake_amounts(
    system_state: &SystemState,
    validator_addrs: Vec<SomaAddress>,
    expected_amounts: Vec<u64>,
    validator_stakes: &ValidatorRewards,
) {
    assert_eq!(
        validator_addrs.len(),
        expected_amounts.len(),
        "Address and amount arrays must be the same length"
    );

    for (i, addr) in validator_addrs.iter().enumerate() {
        let validator = system_state
            .validators
            .consensus_validators
            .iter()
            .find(|v| v.metadata.soma_address == *addr)
            .expect("Validator not found");

        // Calculate self-stake with rewards

        let self_stake_with_rewards =
            validator_stakes.calculate_self_stake_with_rewards(validator, system_state.epoch);

        let expected_amount = expected_amounts[i];

        assert_eq!(
            self_stake_with_rewards, expected_amount,
            "Validator {} expected self-stake {}, but got {}",
            addr, expected_amount, self_stake_with_rewards
        );
    }
}

// Helper to assert validator non-self stake amounts (delegated stake + rewards)
pub fn assert_validator_non_self_stake_amounts(
    system_state: &SystemState,
    validator_addrs: Vec<SomaAddress>,
    expected_amounts: Vec<u64>,
    validator_stakes: &ValidatorRewards,
) {
    assert_eq!(
        validator_addrs.len(),
        expected_amounts.len(),
        "Address and amount arrays must be the same length"
    );

    for (i, addr) in validator_addrs.iter().enumerate() {
        let validator = system_state
            .validators
            .consensus_validators
            .iter()
            .find(|v| v.metadata.soma_address == *addr)
            .expect("Validator not found");

        // Get self-stake
        let self_stake_with_rewards =
            validator_stakes.calculate_self_stake_with_rewards(validator, system_state.epoch);

        // Non-self stake is total stake minus self stake with rewards
        let non_self_stake = validator.staking_pool.soma_balance - self_stake_with_rewards;

        let expected_amount = expected_amounts[i];

        assert_eq!(
            non_self_stake, expected_amount,
            "Validator {} expected non-self stake {}, but got {}",
            addr, expected_amount, non_self_stake
        );
    }
}

// Helper to get total SUI balance for a staker (this would be simpler in a real test environment)
pub fn total_soma_balance(
    staker_withdrawals: &BTreeMap<SomaAddress, u64>,
    staker: SomaAddress,
) -> u64 {
    *staker_withdrawals.get(&staker).unwrap_or(&0)
}

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
    let encoder_validator_address = Multiaddr::from_str("/ip4/127.0.0.1/tcp/8083").unwrap();

    // Create validator
    let mut validator = Validator::new(
        addr,
        protocol_keypair.public().to_owned(),
        network_keypair.public(),
        worker_keypair.public(),
        net_address,
        p2p_address,
        primary_address,
        encoder_validator_address,
        0, // Initial voting power is 0, will be set later
        0,
        ObjectID::random(),
    );

    // Initialize staking pool with stake
    validator.next_epoch_stake = init_stake_amount;
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

pub fn create_encoder_for_testing(addr: SomaAddress, init_stake_amount: u64) -> Encoder {
    let mut rng = StdRng::from_seed([0; 32]);

    let encoder_keypair = EncoderKeyPair::generate(&mut rng);

    // Create network public key (ED25519)
    let network_keypair = NetworkKeyPair::generate(&mut rng);

    // Create multiaddress
    let external_net_address = Multiaddr::from_str("/ip4/127.0.0.1/tcp/8080").unwrap();
    let object_server_address = Multiaddr::from_str("/ip4/127.0.0.1/tcp/8081").unwrap();
    let internal_net_address = Multiaddr::from_str("/ip4/127.0.0.1/tcp/8082").unwrap();

    // Create encoder
    let mut encoder = Encoder::new(
        addr,
        encoder_keypair.public(),
        network_keypair.public(),
        internal_net_address,
        external_net_address,
        object_server_address,
        0, // Initial voting power is 0, will be set later
        0,
        1_000,
        ObjectID::random(),
    );

    // Initialize staking pool with stake
    encoder.next_epoch_stake = init_stake_amount;
    encoder.staking_pool.soma_balance = init_stake_amount;
    encoder.staking_pool.pool_token_balance = init_stake_amount;

    encoder
}

/// Helper function to add an encoder candidate
pub fn add_encoder(system_state: &mut SystemState, address: SomaAddress) -> Encoder {
    let encoder = create_encoder_for_testing(address, 0);

    // Add the encoder to pending active encoders
    system_state
        .encoders
        .request_add_encoder(encoder.clone())
        .expect("Failed to add encoder candidate");

    encoder
}

/// Helper function to request to add stake to an encoder
pub fn stake_with_encoder(
    system_state: &mut SystemState,
    staker: SomaAddress,
    encoder: SomaAddress,
    amount: u64,
) -> StakedSoma {
    system_state
        .request_add_stake_to_encoder(staker, encoder, amount * SHANNONS_PER_SOMA)
        .expect("Failed to add stake to encoder")
}

/// Calculate the total stake of an encoder including rewards
pub fn encoder_stake_amount(
    system_state: &SystemState,
    encoder_address: SomaAddress,
) -> Option<u64> {
    for encoder in &system_state.encoders.active_encoders {
        if encoder.metadata.soma_address == encoder_address {
            return Some(encoder.staking_pool.soma_balance);
        }
    }
    None
}

/// Helper to assert encoder total stake amounts
pub fn assert_encoder_total_stake_amounts(
    system_state: &SystemState,
    encoder_addrs: Vec<SomaAddress>,
    expected_amounts: Vec<u64>,
) {
    assert_eq!(
        encoder_addrs.len(),
        expected_amounts.len(),
        "Address and amount arrays must be the same length"
    );

    for (i, addr) in encoder_addrs.iter().enumerate() {
        let encoder = system_state
            .encoders
            .active_encoders
            .iter()
            .find(|v| v.metadata.soma_address == *addr)
            .expect("Encoder not found");

        let actual_amount = encoder.staking_pool.soma_balance;
        let expected_amount = expected_amounts[i];

        assert_eq!(
            actual_amount, expected_amount,
            "Encoder {} expected stake {}, but got {}",
            addr, expected_amount, actual_amount
        );
    }
}

/// Create a test system state with specified validators and subsidy parameters
pub fn create_test_system_state(
    validators: Vec<Validator>,
    encoders: Vec<Encoder>,
    supply_amount: u64,
    stake_subsidy_initial_amount: u64,
    stake_subsidy_period_length: u64,
    stake_subsidy_decrease_rate: u16,
) -> SystemState {
    // System parameters
    let parameters = SystemParameters {
        epoch_duration_ms: 42, // Doesn't matter what number we put here for tests
        vdf_iterations: 1,
    };

    // Create system state
    let epoch_start_timestamp_ms = 1000;
    let stake_subsidy_fund = supply_amount * SHANNONS_PER_SOMA;

    SystemState::create(
        validators,
        vec![],
        encoders,
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

    create_test_system_state(validators, vec![], 1000, 0, 10, 500)
}

/// Advance epoch with rewards
pub fn advance_epoch_with_rewards(
    system_state: &mut SystemState,
    reward_amount: u64,
) -> ExecutionResult<HashMap<SomaAddress, StakedSoma>> {
    // Calculate next epoch
    let next_epoch = system_state.epoch + 1;

    // Calculate new timestamp (ensuring it's at least epoch_duration_ms later)
    let new_timestamp =
        system_state.epoch_start_timestamp_ms + system_state.parameters.epoch_duration_ms;

    // Advance the epoch
    system_state.advance_epoch(next_epoch, reward_amount, new_timestamp, 1000)
}

// Helper to add a validator candidate
pub fn add_validator(system_state: &mut SystemState, address: SomaAddress) -> Validator {
    let validator = create_validator_for_testing(address, 0);

    // Add the validator to pending active validators
    system_state
        .validators
        .request_add_validator(validator.clone())
        .expect("Failed to add validator candidate");

    validator
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
    for validator in &system_state.validators.consensus_validators {
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
    for validator in &system_state.validators.consensus_validators {
        if validator.metadata.soma_address == validator_address {
            return Some(validator.staking_pool.soma_balance);
        }
    }
    None
}
