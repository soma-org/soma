// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::str::FromStr;

use fastcrypto::bls12381;
use fastcrypto::ed25519::Ed25519PublicKey;
use fastcrypto::hash::HashFunction as _;
use fastcrypto::traits::{KeyPair, ToFromBytes};
use protocol_config::ProtocolVersion;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tracing_subscriber::fmt::init;
use url::Url;

use crate::base::SomaAddress;
use crate::checksum::Checksum;
use crate::committee::TOTAL_VOTING_POWER;
use crate::config::genesis_config::SHANNONS_PER_SOMA;
use crate::crypto::{
    self, AuthorityKeyPair, DecryptionKey, DefaultHash, NetworkKeyPair, NetworkPublicKey,
    ProtocolKeyPair,
};
use crate::digests::{ModelWeightsCommitment, ModelWeightsUrlCommitment};
use crate::effects::ExecutionFailureStatus;
use crate::error::ExecutionResult;
use crate::metadata::{Manifest, ManifestV1, Metadata, MetadataV1};
use crate::model::{ModelId, ModelWeightsManifest};
use crate::multiaddr::Multiaddr;
use crate::object::ObjectID;
use crate::system_state::emission::EmissionPool;
use crate::system_state::staking::{PoolTokenExchangeRate, StakedSomaV1, StakingPool};
use crate::system_state::validator::{Validator, ValidatorSet};
use crate::system_state::{PublicKey, SystemParameters, SystemState, SystemStateTrait};
use crate::tensor::SomaTensor;

#[cfg(test)]
#[derive(Clone)]
pub struct ValidatorRewards {
    // Initial stake amounts for each validator
    initial_stakes: BTreeMap<SomaAddress, u64>,

    // Commission rewards for validators per epoch
    // First key is validator address, second key is epoch, value is StakedSomaV1
    commission_rewards: BTreeMap<SomaAddress, BTreeMap<u64, StakedSomaV1>>,
}

#[cfg(test)]
impl ValidatorRewards {
    /// Create a new ValidatorRewards tracker
    pub fn new(validators: &[Validator]) -> Self {
        let mut initial_stakes = BTreeMap::new();
        for validator in validators {
            initial_stakes
                .insert(validator.metadata.soma_address, validator.staking_pool.soma_balance);
        }

        Self { initial_stakes, commission_rewards: BTreeMap::new() }
    }

    /// Get the initial stake for a validator
    pub fn get_initial_stake(&self, validator_addr: SomaAddress) -> u64 {
        *self.initial_stakes.get(&validator_addr).unwrap_or(&0)
    }

    /// Add commission rewards for an epoch
    pub fn add_commission_rewards(
        &mut self,
        epoch: u64,
        rewards: BTreeMap<SomaAddress, StakedSomaV1>,
    ) {
        for (addr, staked_soma) in rewards {
            self.commission_rewards.entry(addr).or_default().insert(epoch, staked_soma);
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
) -> StakedSomaV1 {
    system_state
        .request_add_stake(staker, validator, amount * SHANNONS_PER_SOMA)
        .expect("Failed to add stake")
}

// Helper function to request to withdraw stake
pub fn unstake(system_state: &mut SystemState, staked_soma: StakedSomaV1) -> u64 {
    system_state.request_withdraw_stake(staked_soma).expect("Failed to withdraw stake")
}

// Helper function to distribute rewards and advance epoch.
// `reward_amount` is in SOMA and represents the total transaction fees for the epoch.
// Tests set validator_reward_allocation_bps=10000 so 100% of fees go to validators.
pub fn advance_epoch_with_reward_amounts(
    system_state: &mut SystemState,
    reward_amount: u64,
    validator_stakes: &mut ValidatorRewards,
) {
    // Calculate next epoch
    let next_epoch = system_state.epoch() + 1;

    // Calculate new timestamp (ensuring it's at least epoch_duration_ms later)
    let new_timestamp =
        system_state.epoch_start_timestamp_ms() + system_state.parameters().epoch_duration_ms;

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        ProtocolVersion::MAX,
        protocol_config::Chain::default(),
    );

    // Advance the epoch
    let rewards = system_state
        .advance_epoch(
            next_epoch,
            &protocol_config,
            reward_amount * SHANNONS_PER_SOMA,
            new_timestamp,
            vec![0; 32],
        )
        .expect("Failed to advance epoch");

    validator_stakes.add_commission_rewards(next_epoch, rewards);
}

// Helper function to advance epoch with reward amounts and slashing rates.
// `reward_amount` is in SOMA and represents the total transaction fees for the epoch.
pub fn advance_epoch_with_reward_amounts_and_slashing_rates(
    system_state: &mut SystemState,
    reward_amount: u64,
    reward_slashing_rate: u64,
    validator_stakes: &mut ValidatorRewards,
) {
    // Calculate next epoch
    let next_epoch = system_state.epoch() + 1;

    // Calculate new timestamp (ensuring it's at least epoch_duration_ms later)
    let new_timestamp =
        system_state.epoch_start_timestamp_ms() + system_state.parameters().epoch_duration_ms;

    let mut protocol_config = protocol_config::ProtocolConfig::get_for_version(
        ProtocolVersion::MAX,
        protocol_config::Chain::default(),
    );

    // Override the slashing rate with the test-specified value
    protocol_config.set_reward_slashing_rate_bps_for_testing(reward_slashing_rate);

    // Advance the epoch
    let rewards = system_state
        .advance_epoch(
            next_epoch,
            &protocol_config,
            reward_amount * SHANNONS_PER_SOMA,
            new_timestamp,
            vec![0; 32],
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
            .validators()
            .validators
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
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == *addr)
            .expect("Validator not found");

        // Calculate self-stake with rewards

        let self_stake_with_rewards =
            validator_stakes.calculate_self_stake_with_rewards(validator, system_state.epoch());

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
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == *addr)
            .expect("Validator not found");

        // Get self-stake
        let self_stake_with_rewards =
            validator_stakes.calculate_self_stake_with_rewards(validator, system_state.epoch());

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

// Helper to get total SOMA balance for a staker (this would be simpler in a real test environment)
pub fn total_soma_balance(
    staker_withdrawals: &BTreeMap<SomaAddress, u64>,
    staker: SomaAddress,
) -> u64 {
    *staker_withdrawals.get(&staker).unwrap_or(&0)
}

/// Create a test validator with specified address, stake, and a unique seed for key generation.
/// The seed ensures each validator in a set gets distinct keys and addresses.
pub fn create_validator_for_testing_with_seed(
    addr: SomaAddress,
    init_stake_amount: u64,
    seed: [u8; 32],
    port_base: u16,
) -> Validator {
    let mut rng = StdRng::from_seed(seed);

    // Create protocol public key (BLS)
    let protocol_keypair = AuthorityKeyPair::generate(&mut rng);

    // Create network public key (ED25519)
    let network_keypair = NetworkKeyPair::generate(&mut rng);

    // Create worker public key (ED25519)
    let worker_keypair = NetworkKeyPair::generate(&mut rng);

    // Generate proof of possession
    let pop = crypto::generate_proof_of_possession(&protocol_keypair, addr);

    // Create unique multiaddresses using port_base
    let net_address = Multiaddr::from_str(&format!("/ip4/127.0.0.1/tcp/{}", port_base)).unwrap();
    let p2p_address =
        Multiaddr::from_str(&format!("/ip4/127.0.0.1/tcp/{}", port_base + 1)).unwrap();
    let primary_address =
        Multiaddr::from_str(&format!("/ip4/127.0.0.1/tcp/{}", port_base + 2)).unwrap();
    let proxy_address =
        Multiaddr::from_str(&format!("/ip4/127.0.0.1/tcp/{}/http", port_base + 3)).unwrap();

    // Create validator
    let mut validator = Validator::new(
        addr,
        protocol_keypair.public().to_owned(),
        network_keypair.public(),
        worker_keypair.public(),
        pop,
        net_address,
        p2p_address,
        primary_address,
        proxy_address,
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

/// Create a test validator with specified address and stake amount.
/// Uses a deterministic seed derived from the address for unique keys.
pub fn create_validator_for_testing(addr: SomaAddress, init_stake_amount: u64) -> Validator {
    // Derive a unique seed from the address bytes so each address gets unique keys
    let addr_bytes: &[u8] = addr.as_ref();
    let mut seed = [0u8; 32];
    let copy_len = std::cmp::min(addr_bytes.len(), 32);
    seed[..copy_len].copy_from_slice(&addr_bytes[..copy_len]);
    // Use first 2 bytes for port base to avoid collisions (port range 10000+)
    let port_base = 10000u16.wrapping_add(u16::from_le_bytes([seed[0], seed[1]]));
    create_validator_for_testing_with_seed(addr, init_stake_amount, seed, port_base)
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
    supply_amount: u64,
    emission_per_epoch: u64,
) -> SystemState {
    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        ProtocolVersion::MAX,
        protocol_config::Chain::default(),
    );
    // Create system state
    let epoch_start_timestamp_ms = 1000;
    let stake_subsidy_fund = supply_amount * SHANNONS_PER_SOMA;

    let mut state = SystemState::create(
        validators,
        ProtocolVersion::MAX.as_u64(),
        epoch_start_timestamp_ms,
        &protocol_config,
        stake_subsidy_fund,
        emission_per_epoch * SHANNONS_PER_SOMA,
        None,
    );
    // Set validator_reward_allocation_bps to 100% so all fees go to validators.
    // This decouples validator/delegation/reward tests from the data submission reward split.
    state.parameters_mut().validator_reward_allocation_bps = 10000; // 100%
    state
}

/// Setup a system state with specified validator addresses
pub fn set_up_system_state(addrs: Vec<SomaAddress>) -> SystemState {
    let mut validators = Vec::new();

    for addr in addrs {
        validators.push(create_validator_for_testing(addr, 100 * SHANNONS_PER_SOMA));
    }

    create_test_system_state(validators, 1000, 0)
}

/// Advance epoch with rewards.
/// `reward_amount` is in shannons and represents the total transaction fees for the epoch.
/// Tests set validator_reward_allocation_bps=10000 so 100% of fees go to validators.
#[allow(clippy::result_large_err)]
pub fn advance_epoch_with_rewards(
    system_state: &mut SystemState,
    reward_amount: u64,
) -> ExecutionResult<BTreeMap<SomaAddress, StakedSomaV1>> {
    // Calculate next epoch
    let next_epoch = system_state.epoch() + 1;

    // Calculate new timestamp (ensuring it's at least epoch_duration_ms later)
    let new_timestamp =
        system_state.epoch_start_timestamp_ms() + system_state.parameters().epoch_duration_ms;

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        ProtocolVersion::MAX,
        protocol_config::Chain::default(),
    );

    // Advance the epoch
    system_state.advance_epoch(
        next_epoch,
        &protocol_config,
        reward_amount,
        new_timestamp,
        vec![0; 32],
    )
}

// Helper to add a validator candidate
pub fn add_validator(system_state: &mut SystemState, address: SomaAddress) -> Validator {
    let validator = create_validator_for_testing(address, 0);

    // Add the validator to pending active validators
    system_state
        .validators_mut()
        .request_add_validator(validator.clone())
        .expect("Failed to add validator candidate");

    validator
}

/// Request to remove a validator
#[allow(clippy::result_large_err)]
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
    for validator in &system_state.validators().validators {
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
    for validator in &system_state.validators().validators {
        if validator.metadata.soma_address == validator_address {
            return Some(validator.staking_pool.soma_balance);
        }
    }
    None
}

// ===== Model test helpers =====

/// Build a `ModelWeightsUrlCommitment` that matches the hash of the given URL string.
pub fn url_commitment_for(url_str: &str) -> ModelWeightsUrlCommitment {
    let mut hasher = DefaultHash::default();
    hasher.update(url_str.as_bytes());
    let hash = hasher.finalize();
    let bytes: [u8; 32] = hash.as_ref().try_into().unwrap();
    ModelWeightsUrlCommitment::new(bytes)
}

/// Build a dummy `ModelWeightsManifest` whose URL matches the given commitment.
pub fn make_weights_manifest(url_str: &str) -> ModelWeightsManifest {
    let url = Url::parse(url_str).expect("Invalid URL in test helper");
    let metadata = Metadata::V1(MetadataV1::new(Checksum::new_from_hash([1u8; 32]), 1024));
    let manifest = Manifest::V1(ManifestV1::new(url, metadata));
    ModelWeightsManifest { manifest, decryption_key: DecryptionKey::new([0xAA; 32]) }
}

/// Commit a model into `pending_models`. Returns the StakedSomaV1 receipt.
/// Uses a deterministic test URL derived from model_id to generate matching commitments.
pub fn commit_model(
    system_state: &mut SystemState,
    owner: SomaAddress,
    model_id: ModelId,
    stake_amount: u64,
) -> StakedSomaV1 {
    commit_model_with_commission(system_state, owner, model_id, stake_amount, 0)
}

/// Commit a model with a specified commission rate.
pub fn commit_model_with_commission(
    system_state: &mut SystemState,
    owner: SomaAddress,
    model_id: ModelId,
    stake_amount: u64,
    commission_rate: u64,
) -> StakedSomaV1 {
    let url_str = format!("https://example.com/models/{}", model_id);
    let url_commitment = url_commitment_for(&url_str);
    let weights_commitment = ModelWeightsCommitment::new([0xBB; 32]);
    let staking_pool_id = ObjectID::random();

    system_state
        .request_commit_model(
            owner,
            model_id,
            url_commitment,
            weights_commitment,
            system_state.parameters().model_architecture_version,
            stake_amount,
            commission_rate,
            staking_pool_id,
        )
        .expect("Failed to commit model")
}

/// Reveal a previously committed model (moves pending -> active).
/// Must be called in `commit_epoch + 1`.
/// Uses a default 10-dimensional embedding.
pub fn reveal_model(system_state: &mut SystemState, owner: SomaAddress, model_id: &ModelId) {
    reveal_model_with_dim(system_state, owner, model_id, 10);
}

/// Reveal a previously committed model with a specific embedding dimension.
/// Must be called in `commit_epoch + 1`.
pub fn reveal_model_with_dim(
    system_state: &mut SystemState,
    owner: SomaAddress,
    model_id: &ModelId,
    embedding_dim: usize,
) {
    let url_str = format!("https://example.com/models/{}", model_id);
    let manifest = make_weights_manifest(&url_str);
    // Create a deterministic test embedding based on model_id
    let embedding = make_test_embedding(model_id, embedding_dim);

    system_state
        .request_reveal_model(owner, model_id, manifest, embedding)
        .expect("Failed to reveal model");
}

/// Create a deterministic test embedding based on model_id
fn make_test_embedding(model_id: &ModelId, dim: usize) -> SomaTensor {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    model_id.hash(&mut hasher);
    let seed = hasher.finish();

    // Generate deterministic values based on the model_id
    let values: Vec<f32> = (0..dim)
        .map(|i| {
            // Use a simple deterministic function based on seed and index
            let x = ((seed.wrapping_add(i as u64)).wrapping_mul(2654435761) % 1000) as f32;
            (x / 1000.0) - 0.5 // Normalize to [-0.5, 0.5]
        })
        .collect();

    SomaTensor::new(values, vec![dim])
}

/// Commit a model update for an active model.
/// Uses a deterministic "update" URL derived from model_id.
pub fn commit_model_update(system_state: &mut SystemState, owner: SomaAddress, model_id: &ModelId) {
    let url_str = format!("https://example.com/models/{}/update", model_id);
    let url_commitment = url_commitment_for(&url_str);
    let weights_commitment = ModelWeightsCommitment::new([0xCC; 32]);

    system_state
        .request_commit_model_update(owner, model_id, url_commitment, weights_commitment)
        .expect("Failed to commit model update");
}

/// Reveal a pending model update.
/// Uses a default 10-dimensional embedding.
pub fn reveal_model_update(system_state: &mut SystemState, owner: SomaAddress, model_id: &ModelId) {
    reveal_model_update_with_dim(system_state, owner, model_id, 10);
}

/// Reveal a pending model update with a specific embedding dimension.
pub fn reveal_model_update_with_dim(
    system_state: &mut SystemState,
    owner: SomaAddress,
    model_id: &ModelId,
    embedding_dim: usize,
) {
    let url_str = format!("https://example.com/models/{}/update", model_id);
    let manifest = make_weights_manifest(&url_str);
    // Create a slightly different deterministic embedding for the update
    let mut embedding = make_test_embedding(model_id, embedding_dim);
    // Modify embedding slightly for the update version
    let values = embedding.to_vec();
    let updated_values: Vec<f32> = values.iter().map(|v| v + 0.1).collect();
    let embedding = SomaTensor::new(updated_values, vec![embedding_dim]);

    system_state
        .request_reveal_model_update(owner, model_id, manifest, embedding)
        .expect("Failed to reveal model update");
}

/// Stake to a model (any sender). Amount is in SOMA (not shannons).
pub fn stake_with_model(
    system_state: &mut SystemState,
    model_id: &ModelId,
    amount: u64,
) -> StakedSomaV1 {
    system_state
        .request_add_stake_to_model(model_id, amount * SHANNONS_PER_SOMA)
        .expect("Failed to stake with model")
}
