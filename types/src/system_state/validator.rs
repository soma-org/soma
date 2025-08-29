use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    str::FromStr,
};

use fastcrypto::{ed25519::Ed25519PublicKey, traits::ToFromBytes};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::{
    base::SomaAddress,
    committee::{
        MAX_VOTING_POWER, QUORUM_THRESHOLD, TOTAL_VOTING_POWER, VALIDATOR_CONSENSUS_LOW_POWER,
        VALIDATOR_CONSENSUS_MIN_POWER, VALIDATOR_CONSENSUS_VERY_LOW_POWER,
        VALIDATOR_NETWORKING_MIN_POWER,
    },
    crypto::{self, NetworkPublicKey},
    effects::ExecutionFailureStatus,
    error::ExecutionResult,
    multiaddr::Multiaddr,
    object::ObjectID,
    transaction::UpdateValidatorMetadataArgs,
};

use super::{
    staking::{PoolTokenExchangeRate, StakedSoma, StakingPool},
    PublicKey,
};

/// # ValidatorMetadata
///
/// Contains the identifying information and network addresses for a validator.
///
/// ## Purpose
/// Stores all the necessary information to identify a validator and communicate
/// with it over the network. This includes cryptographic keys for different purposes
/// and network addresses for different services.
///
/// ## Lifecycle
/// ValidatorMetadata is created when a validator joins the network and is updated
/// when a validator changes its keys or network addresses. The next_epoch_* fields
/// allow for smooth transitions when validators update their information.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize, Serialize, Hash)]
pub struct ValidatorMetadata {
    /// The Soma blockchain address of the validator
    pub soma_address: SomaAddress,

    /// The BLS public key used for consensus protocol operations
    pub protocol_pubkey: PublicKey,

    /// The network public key used for network identity and authentication
    pub network_pubkey: crate::crypto::NetworkPublicKey,

    /// The worker public key used for worker operations
    pub worker_pubkey: crate::crypto::NetworkPublicKey,

    /// The network address for general network communication
    pub net_address: Multiaddr,

    /// The p2p address for peer-to-peer communication
    pub p2p_address: Multiaddr,

    /// The primary address for validator services
    pub primary_address: Multiaddr,

    pub encoder_validator_address: Multiaddr,

    /// Optional new protocol public key for the next epoch
    pub next_epoch_protocol_pubkey: Option<PublicKey>,

    /// Optional new network public key for the next epoch
    pub next_epoch_network_pubkey: Option<crate::crypto::NetworkPublicKey>,

    /// Optional new network address for the next epoch
    pub next_epoch_net_address: Option<Multiaddr>,

    /// Optional new p2p address for the next epoch
    pub next_epoch_p2p_address: Option<Multiaddr>,

    /// Optional new primary address for the next epoch
    pub next_epoch_primary_address: Option<Multiaddr>,

    pub next_epoch_worker_pubkey: Option<crate::crypto::NetworkPublicKey>,

    pub next_epoch_encoder_validator_address: Option<Multiaddr>,
}

/// # Validator
///
/// Represents a validator in the Soma blockchain with its metadata and voting power.
///
/// ## Purpose
/// Combines validator metadata with voting power to represent a validator's
/// complete state in the system. The voting power determines the validator's
/// influence in the consensus protocol.
///
/// ## Usage
/// Validators are stored in the ValidatorSet and are used to form the committee
/// for consensus operations. They are added, removed, and updated during epoch
/// transitions.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct Validator {
    /// The validator's metadata including keys and network addresses
    pub metadata: ValidatorMetadata,

    /// The validator's voting power in the consensus protocol
    pub voting_power: u64,

    pub staking_pool: StakingPool,

    pub commission_rate: u64,

    pub next_epoch_stake: u64,

    pub next_epoch_commission_rate: u64,
}

impl Validator {
    /// # Create a new validator
    ///
    /// Creates a new validator with the specified metadata and voting power.
    ///
    /// ## Arguments
    /// * `soma_address` - The Soma blockchain address of the validator
    /// * `protocol_pubkey` - The BLS public key for consensus operations
    /// * `network_pubkey` - The network public key for network identity
    /// * `worker_pubkey` - The worker public key for worker operations
    /// * `net_address` - The network address for general communication
    /// * `p2p_address` - The p2p address for peer-to-peer communication
    /// * `primary_address` - The primary address for validator services
    /// * `voting_power` - The validator's voting power
    ///
    /// ## Returns
    /// A new Validator instance with the specified metadata and voting power
    pub fn new(
        soma_address: SomaAddress,
        protocol_pubkey: PublicKey,
        network_pubkey: crypto::NetworkPublicKey,
        worker_pubkey: crypto::NetworkPublicKey,
        net_address: Multiaddr,
        p2p_address: Multiaddr,
        primary_address: Multiaddr,
        encoder_validator_address: Multiaddr,
        voting_power: u64,
        commission_rate: u64,
        staking_pool_id: ObjectID,
    ) -> Self {
        Self {
            metadata: ValidatorMetadata {
                soma_address,
                protocol_pubkey,
                network_pubkey,
                worker_pubkey,
                net_address,
                p2p_address,
                primary_address,
                encoder_validator_address,
                next_epoch_protocol_pubkey: None,
                next_epoch_network_pubkey: None,
                next_epoch_net_address: None,
                next_epoch_p2p_address: None,
                next_epoch_primary_address: None,
                next_epoch_worker_pubkey: None,
                next_epoch_encoder_validator_address: None,
            },
            voting_power,
            commission_rate,
            next_epoch_stake: 0,
            next_epoch_commission_rate: commission_rate,
            staking_pool: StakingPool::new(staking_pool_id),
        }
    }

    /// Request to add stake to the validator
    pub fn request_add_stake(
        &mut self,
        stake: u64,
        staker_address: SomaAddress,
        current_epoch: u64,
    ) -> StakedSoma {
        assert!(stake > 0, "Stake amount must be positive");

        // Calculate activation epoch (typically next epoch)
        let stake_activation_epoch = current_epoch + 1;

        // Add stake to the staking pool
        let staked_soma = self
            .staking_pool
            .request_add_stake(stake, stake_activation_epoch);

        // If pool is preactive, process stake immediately
        if self.staking_pool.is_preactive() {
            self.staking_pool.process_pending_stake();
        }

        // Update next epoch stake
        self.next_epoch_stake += stake;

        staked_soma
    }

    pub fn request_add_stake_at_genesis(
        &mut self,
        stake: u64,
        staker_address: SomaAddress,
        current_epoch: u64,
    ) -> StakedSoma {
        assert!(current_epoch == 0, "Must be called during genesis");
        assert!(stake > 0, "Stake amount must be positive");

        // Add stake to the staking pool
        let staked_soma = self.staking_pool.request_add_stake(stake, 0);

        self.staking_pool.process_pending_stake();
        self.next_epoch_stake += stake;

        staked_soma
    }

    /// Request to withdraw stake from the validator
    pub fn request_withdraw_stake(&mut self, staked_soma: StakedSoma, current_epoch: u64) -> u64 {
        // Process withdrawal in staking pool
        let withdrawn_amount = self
            .staking_pool
            .request_withdraw_stake(staked_soma, current_epoch);

        // Update next epoch stake
        self.next_epoch_stake -= withdrawn_amount;

        withdrawn_amount
    }

    /// Add rewards to staking pool for delegation rewards
    pub fn deposit_staker_rewards(&mut self, amount: u64) {
        if amount == 0 {
            return;
        }

        // Credit rewards to staking pool for auto-compounding
        self.next_epoch_stake += amount;
        self.staking_pool.deposit_rewards(amount);
    }

    /// Set commission rate for the next epoch
    pub fn request_set_commission_rate(&mut self, new_rate: u64) -> Result<(), String> {
        const MAX_COMMISSION_RATE: u64 = 10000; // 100% in basis points

        if new_rate > MAX_COMMISSION_RATE {
            return Err("Commission rate too high".to_string());
        }

        self.next_epoch_commission_rate = new_rate;
        Ok(())
    }

    /// Activate this validator and its staking pool
    pub fn activate(&mut self, activation_epoch: u64) {
        // Add initial exchange rate to staking pool
        self.staking_pool.exchange_rates.insert(
            activation_epoch,
            PoolTokenExchangeRate {
                soma_amount: self.staking_pool.soma_balance,
                pool_token_amount: self.staking_pool.pool_token_balance,
            },
        );

        // Ensure pool is preactive
        assert!(self.staking_pool.is_preactive(), "Pool is already active");
        assert!(
            !self.staking_pool.is_inactive(),
            "Cannot activate inactive pool"
        );

        // Set activation epoch
        self.staking_pool.activation_epoch = Some(activation_epoch);
    }

    /// Deactivate this validator and its staking pool
    pub fn deactivate(&mut self, deactivation_epoch: u64) {
        // Cannot deactivate already inactive pool
        assert!(!self.staking_pool.is_inactive(), "Pool already inactive");

        // Set deactivation epoch
        self.staking_pool.deactivation_epoch = Some(deactivation_epoch);
    }

    fn adjust_commission_rate(&mut self) {
        self.commission_rate = self.next_epoch_commission_rate;
    }

    pub fn stage_next_epoch_metadata(
        &mut self,
        args: &UpdateValidatorMetadataArgs,
    ) -> ExecutionResult<()> {
        // Process Network Address
        if let Some(ref addr_bytes) = args.next_epoch_network_address {
            let addr_str: String = bcs::from_bytes(addr_bytes).map_err(|_| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Failed to BCS deserialize network address string"),
                }
            })?;
            let multiaddr = Multiaddr::from_str(&addr_str).map_err(|e| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Invalid network multiaddr format: {}", e),
                }
            })?;
            // TODO: Additional validation? Check for duplicates against other *staged* values?
            self.metadata.next_epoch_net_address = Some(multiaddr);
        }

        // Process P2P Address
        if let Some(ref addr_bytes) = args.next_epoch_p2p_address {
            let addr_str: String = bcs::from_bytes(addr_bytes).map_err(|_| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Failed to BCS deserialize p2p address string"),
                }
            })?;
            let multiaddr = Multiaddr::from_str(&addr_str).map_err(|e| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Invalid p2p multiaddr format: {}", e),
                }
            })?;
            self.metadata.next_epoch_p2p_address = Some(multiaddr);
        }

        // Process Primary Address
        if let Some(ref addr_bytes) = args.next_epoch_primary_address {
            let addr_str: String = bcs::from_bytes(addr_bytes).map_err(|_| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Failed to BCS deserialize primary address string"),
                }
            })?;
            let multiaddr = Multiaddr::from_str(&addr_str).map_err(|e| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Invalid primary multiaddr format: {}", e),
                }
            })?;
            self.metadata.next_epoch_primary_address = Some(multiaddr);
        }

        // Process Protocol Public Key
        if let Some(ref key_bytes) = args.next_epoch_protocol_pubkey {
            let pubkey = PublicKey::from_bytes(key_bytes).map_err(|e| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Invalid protocol public key format: {}", e),
                }
            })?;
            // TODO: Proof of possession validation if needed
            self.metadata.next_epoch_protocol_pubkey = Some(pubkey);
            // Handle PoP if necessary:
            // if let Some(pop_bytes) = &args.next_epoch_proof_of_possession {
            //     verify_pop(&pubkey, pop_bytes)?;
            //     self.metadata.next_epoch_proof_of_possession = Some(pop_bytes.clone());
            // } else if self.metadata.next_epoch_protocol_pubkey.is_some() {
            //     // If protocol key changes, PoP might be mandatory
            //     return Err(ExecutionFailureStatus::MissingProofOfPossession);
            // }
        }

        // Process Worker Public Key
        if let Some(ref key_bytes) = args.next_epoch_worker_pubkey {
            let pubkey = Ed25519PublicKey::from_bytes(key_bytes).map_err(|e| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Invalid worker public key format: {}", e),
                }
            })?;
            let network_pubkey = NetworkPublicKey::new(pubkey);
            self.metadata.next_epoch_worker_pubkey = Some(network_pubkey);
        }

        // Process Network Public Key
        if let Some(ref key_bytes) = args.next_epoch_network_pubkey {
            let pubkey = Ed25519PublicKey::from_bytes(key_bytes).map_err(|e| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Invalid network public key format: {}", e),
                }
            })?;
            let network_pubkey = NetworkPublicKey::new(pubkey);
            self.metadata.next_epoch_network_pubkey = Some(network_pubkey);
        }

        // TODO: Add duplicate checks here? E.g., ensure staged network key != staged worker key?

        Ok(())
    }

    /// Apply staged metadata changes. Called during epoch transition.
    fn effectuate_staged_metadata(&mut self) {
        if let Some(addr) = self.metadata.next_epoch_net_address.take() {
            self.metadata.net_address = addr;
        }
        if let Some(addr) = self.metadata.next_epoch_p2p_address.take() {
            self.metadata.p2p_address = addr;
        }
        if let Some(addr) = self.metadata.next_epoch_primary_address.take() {
            self.metadata.primary_address = addr;
        }

        if let Some(key) = self.metadata.next_epoch_protocol_pubkey.take() {
            self.metadata.protocol_pubkey = key;
            // Apply staged PoP if it exists
            // if let Some(pop) = self.metadata.next_epoch_proof_of_possession.take() {
            //     self.metadata.proof_of_possession = pop;
            // }
        }
        if let Some(key) = self.metadata.next_epoch_network_pubkey.take() {
            self.metadata.network_pubkey = key;
        }
        if let Some(key) = self.metadata.next_epoch_worker_pubkey.take() {
            self.metadata.worker_pubkey = key;
        }
    }
}

#[cfg(test)]
impl Validator {
    /// Get the pool token exchange rate at a specific epoch
    pub fn pool_token_exchange_rate_at_epoch(&self, epoch: u64) -> PoolTokenExchangeRate {
        self.staking_pool.pool_token_exchange_rate_at_epoch(epoch)
    }

    /// Calculate rewards for this validator's initial stake (self-stake)
    pub fn calculate_rewards(
        &self,
        initial_stake: u64,
        stake_activation_epoch: u64,
        current_epoch: u64,
    ) -> u64 {
        self.staking_pool
            .calculate_rewards(initial_stake, stake_activation_epoch, current_epoch)
    }
}

/// # ValidatorSet
///
/// Manages the set of validators in the Soma blockchain.
///
/// ## Purpose
/// Maintains the current set of active validators, pending validators, and
/// validators scheduled for removal. It also tracks the total stake in the system
/// and provides operations for adding and removing validators.
///
/// ## Lifecycle
/// The ValidatorSet is updated during epoch transitions, when pending changes
/// are applied to the active validator set. Validators can be added to the
/// pending set during an epoch and will become active in the next epoch.
///
/// ## Thread Safety
/// This struct is not thread-safe on its own and should be protected by
/// appropriate synchronization mechanisms when accessed concurrently.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct ValidatorSet {
    /// The total stake across all active validators
    pub total_stake: u64,

    /// Validators participating in consensus (formerly "active_validators")
    pub consensus_validators: Vec<Validator>,

    /// Validators that only participate in networking (not consensus)
    pub networking_validators: Vec<Validator>,

    /// Validators that will be added in the next epoch
    pub pending_validators: Vec<Validator>,

    /// Indices of validators that will be removed in the next epoch
    /// Format: (is_consensus, index) - true for consensus, false for networking
    pub pending_removals: Vec<(bool, usize)>,

    pub staking_pool_mappings: BTreeMap<ObjectID, SomaAddress>,

    pub inactive_validators: BTreeMap<ObjectID, Validator>,

    pub at_risk_validators: BTreeMap<SomaAddress, u64>,
}

impl ValidatorSet {
    /// # Create a new validator set
    ///
    /// Creates a new validator set with the specified initial active validators.
    ///
    /// ## Arguments
    /// * `init_active_validators` - The initial set of active validators
    ///
    /// ## Returns
    /// A new ValidatorSet instance with the specified active validators and
    /// calculated total stake
    pub fn new(
        consensus_validators: Vec<Validator>,
        networking_validators: Vec<Validator>,
    ) -> Self {
        // Calculate total stake from both sets
        let consensus_stake: u64 = consensus_validators
            .iter()
            .map(|v| v.staking_pool.soma_balance)
            .sum();

        let networking_stake: u64 = networking_validators
            .iter()
            .map(|v| v.staking_pool.soma_balance)
            .sum();

        let total_stake = consensus_stake + networking_stake;

        let mut staking_pool_mappings = BTreeMap::new();

        // Initialize staking pool mappings for consensus validators
        for validator in &consensus_validators {
            staking_pool_mappings
                .insert(validator.staking_pool.id, validator.metadata.soma_address);
        }

        // Initialize staking pool mappings for networking validators
        for validator in &networking_validators {
            staking_pool_mappings
                .insert(validator.staking_pool.id, validator.metadata.soma_address);
        }

        let mut validator_set = Self {
            total_stake,
            consensus_validators,
            networking_validators,
            pending_validators: Vec::new(),
            pending_removals: Vec::new(),
            staking_pool_mappings,
            inactive_validators: BTreeMap::new(),
            at_risk_validators: BTreeMap::new(),
        };

        // Set initial voting power
        validator_set.set_voting_power();

        validator_set
    }

    /// Request to add a validator (either new or previously removed)
    pub fn request_add_validator(
        &mut self,
        validator: Validator,
    ) -> Result<(), ExecutionFailureStatus> {
        let address = validator.metadata.soma_address;

        // Check for an existing validator with the same address
        if self.find_validator_with_pending_mut(address).is_some() {
            return Err(ExecutionFailureStatus::DuplicateValidator);
        }

        // TODO: Check for duplicate information with other validators
        // if self.is_duplicate_validator(&validator) {
        //     return Err(ExecutionFailureStatus::DuplicateValidator);
        // }

        // Add the staking pool mapping
        let new_pool_id = validator.staking_pool.id;
        self.staking_pool_mappings.insert(new_pool_id, address);

        // Add to pending validators
        self.pending_validators.push(validator);

        Ok(())
    }

    /// # Request to remove a validator
    ///
    /// Requests to remove a validator from the active set in the next epoch.
    ///
    /// ## Behavior
    /// The validator's index is added to the pending_removals list and the
    /// validator will be removed from the active set in the next epoch when
    /// advance_epoch() is called.
    ///
    /// ## Arguments
    /// * `address` - The address of the validator to remove
    ///
    /// ## Returns
    /// Ok(()) if the validator was successfully marked for removal,
    /// or an error if the validator is not active or already marked for removal
    ///
    /// ## Errors
    /// Returns SomaError::NotAValidator if the validator is not in the active set
    /// Returns SomaError::ValidatorAlreadyRemoved if the validator is already marked for removal
    pub fn request_remove_validator(&mut self, address: SomaAddress) -> ExecutionResult {
        // Check consensus validators first
        if let Some((i, _)) = self
            .consensus_validators
            .iter()
            .find_position(|v| address == v.metadata.soma_address)
        {
            if self
                .pending_removals
                .iter()
                .any(|(is_consensus, idx)| *is_consensus && *idx == i)
            {
                return Err(ExecutionFailureStatus::ValidatorAlreadyRemoved);
            }
            self.pending_removals.push((true, i));
            return Ok(());
        }

        // Check networking validators
        if let Some((i, _)) = self
            .networking_validators
            .iter()
            .find_position(|v| address == v.metadata.soma_address)
        {
            if self
                .pending_removals
                .iter()
                .any(|(is_consensus, idx)| !*is_consensus && *idx == i)
            {
                return Err(ExecutionFailureStatus::ValidatorAlreadyRemoved);
            }
            self.pending_removals.push((false, i));
            return Ok(());
        }

        Err(ExecutionFailureStatus::NotAValidator)
    }

    /// # Advance to the next epoch
    ///
    /// Processes pending validator changes and advances the validator set to the next epoch.
    ///
    /// ## Behavior
    /// This method:
    /// 1. Processes pending validator additions by moving them to the active set
    /// 2. Processes pending validator removals by removing them from the active set
    /// 3. Updates the total stake based on the new active validator set
    ///
    /// Note: This method does not yet implement validator rewards, slashing,
    /// or automatic removal of low-stake validators.
    /// Advance the validator set to the next epoch
    pub fn advance_epoch(
        &mut self,
        new_epoch: u64,
        total_rewards: &mut u64,
        reward_slashing_rate: u64,
        validator_report_records: &mut BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        validator_low_stake_grace_period: u64,
    ) -> HashMap<SomaAddress, StakedSoma> {
        let validator_rewards = self.calculate_and_distribute_rewards(
            total_rewards,
            reward_slashing_rate,
            validator_report_records,
            new_epoch,
        );

        self.adjust_commission_rates();

        // Process pending stakes and withdrawals for active validators
        self.process_active_validator_stakes(new_epoch);

        // Process pending stakes and withdrawals for pending validators
        // This ensures pending validators accumulate stake during epoch changes
        self.process_pending_validator_stakes(new_epoch);

        self.process_pending_removals(validator_report_records, new_epoch);

        // 5. Process validator transitions (REPLACES update_validator_positions_and_calculate_total_stake)
        self.process_validator_transitions(
            new_epoch,
            validator_report_records,
            validator_low_stake_grace_period,
        );

        // 6. Update total stake after all transitions
        self.total_stake = self.calculate_total_stake();

        self.effectuate_staged_metadata();

        // Finally readjust voting power after all validator changes
        self.set_voting_power();

        return validator_rewards;
    }

    /// Effectuate pending next epoch metadata changes for all active validators.
    fn effectuate_staged_metadata(&mut self) {
        for validator in &mut self.consensus_validators {
            validator.effectuate_staged_metadata();
            // TODO: Add validation after effectuation if needed, e.g., check for duplicates based on new metadata.
            // self.assert_no_active_duplicates(validator); // Example if needed
        }

        for validator in &mut self.networking_validators {
            validator.effectuate_staged_metadata();
        }

        // Optional: Check for duplicates *after* all changes are applied
        self.check_for_duplicate_metadata_post_effectuation();
    }

    /// Helper function to check for duplicate metadata after effectuation (optional)
    fn check_for_duplicate_metadata_post_effectuation(&self) {
        let mut seen_protocol_keys = HashSet::new();
        let mut seen_network_keys = HashSet::new();
        let mut seen_worker_keys = HashSet::new();
        let mut seen_net_addrs = HashSet::new();
        let mut seen_p2p_addrs = HashSet::new();
        // Add others as needed (primary, worker addr)

        let all_validators = self
            .consensus_validators
            .iter()
            .chain(self.networking_validators.iter());

        for validator in all_validators {
            let meta = &validator.metadata;
            if !seen_protocol_keys.insert(meta.protocol_pubkey.as_bytes()) {
                error!(
                    "Duplicate protocol key found after effectuation: {:?}",
                    meta.protocol_pubkey
                );
                // Potentially panic or handle error, though epoch change is usually non-recoverable
            }
            if !seen_network_keys.insert(meta.network_pubkey.to_bytes()) {
                error!(
                    "Duplicate network key found after effectuation: {:?}",
                    meta.network_pubkey
                );
            }
            if !seen_worker_keys.insert(meta.worker_pubkey.to_bytes()) {
                error!(
                    "Duplicate worker key found after effectuation: {:?}",
                    meta.worker_pubkey
                );
            }
            if !seen_net_addrs.insert(meta.net_address.to_string()) {
                // Use string representation for Multiaddr comparison
                error!(
                    "Duplicate network address found after effectuation: {}",
                    meta.net_address
                );
            }
            if !seen_p2p_addrs.insert(meta.p2p_address.to_string()) {
                error!(
                    "Duplicate P2P address found after effectuation: {}",
                    meta.p2p_address
                );
            }
            // Add checks for primary_address, worker_address if they exist and need uniqueness
        }
    }

    /// Check if address is specifically a consensus validator
    pub fn is_consensus_validator(&self, address: SomaAddress) -> bool {
        self.consensus_validators
            .iter()
            .any(|v| v.metadata.soma_address == address)
    }

    /// Check if address is specifically a networking-only validator  
    pub fn is_networking_validator(&self, address: SomaAddress) -> bool {
        self.networking_validators
            .iter()
            .any(|v| v.metadata.soma_address == address)
    }

    /// Get all validators eligible for networking (consensus + networking-only)
    pub fn get_all_networking_validators(&self) -> impl Iterator<Item = &Validator> {
        self.consensus_validators
            .iter()
            .chain(self.networking_validators.iter())
    }

    /// Find a validator by address (including pending validators)
    pub fn find_validator_with_pending_mut(
        &mut self,
        validator_address: SomaAddress,
    ) -> Option<&mut Validator> {
        // First check consensus validators
        for validator in &mut self.consensus_validators {
            if validator.metadata.soma_address == validator_address {
                return Some(validator);
            }
        }

        // Then check networking-only validators
        for validator in &mut self.networking_validators {
            if validator.metadata.soma_address == validator_address {
                return Some(validator);
            }
        }

        // Then check pending validators
        for validator in &mut self.pending_validators {
            if validator.metadata.soma_address == validator_address {
                return Some(validator);
            }
        }

        None
    }

    /// Find a validator by address
    pub fn find_validator_mut(&mut self, validator_address: SomaAddress) -> Option<&mut Validator> {
        // First check consensus validators
        for validator in &mut self.consensus_validators {
            if validator.metadata.soma_address == validator_address {
                return Some(validator);
            }
        }

        // Then check networking validators
        for validator in &mut self.networking_validators {
            if validator.metadata.soma_address == validator_address {
                return Some(validator);
            }
        }

        None
    }

    /// Find a validator by address (immutable version)
    pub fn find_validator(&self, validator_address: SomaAddress) -> Option<&Validator> {
        // Check consensus validators
        for validator in &self.consensus_validators {
            if validator.metadata.soma_address == validator_address {
                return Some(validator);
            }
        }

        // Check networking validators
        for validator in &self.networking_validators {
            if validator.metadata.soma_address == validator_address {
                return Some(validator);
            }
        }

        None
    }

    /// Check if an address is an active validator
    pub fn is_active_validator(&self, address: SomaAddress) -> bool {
        self.find_validator(address).is_some()
    }

    /// Check if an address belongs to a pending active validator
    pub fn is_pending_validator(&self, address: SomaAddress) -> bool {
        self.pending_validators
            .iter()
            .any(|v| v.metadata.soma_address == address)
    }

    /// Calculate unadjusted reward distribution without slashing
    pub fn compute_unadjusted_reward_distribution(
        &self,
        total_voting_power: u64,
        total_rewards: u64,
    ) -> Vec<u64> {
        let mut reward_amounts = Vec::with_capacity(self.consensus_validators.len());

        for validator in &self.consensus_validators {
            // Calculate proportional to voting power
            let voting_power = validator.voting_power as u128;
            let reward = voting_power * (total_rewards as u128) / (total_voting_power as u128);

            reward_amounts.push(reward as u64);
        }

        reward_amounts
    }

    /// Identify validators that should be slashed based on reports
    pub fn compute_slashed_validators(
        &self,
        validator_report_records: &BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
    ) -> Vec<SomaAddress> {
        let mut slashed_validators = Vec::new();
        let quorum_threshold = QUORUM_THRESHOLD;

        for (validator_address, reporters) in validator_report_records {
            // Make sure validator is active
            if self.find_validator(*validator_address).is_some() {
                // Calculate total voting power of reporters
                let reporter_votes =
                    self.sum_voting_power_by_addresses(&reporters.iter().cloned().collect());

                // If threshold reached, validator should be slashed
                if reporter_votes >= quorum_threshold {
                    slashed_validators.push(*validator_address);
                }
            }
        }

        slashed_validators
    }

    /// Sum voting power of a list of validators
    pub fn sum_voting_power_by_addresses(&self, addresses: &Vec<SomaAddress>) -> u64 {
        let mut sum = 0;

        for &address in addresses {
            if let Some(validator) = self.find_validator(address) {
                sum += validator.voting_power;
            }
        }

        sum
    }

    /// Find validator indices in consensus validators list
    pub fn get_consensus_validator_indices(&self, addresses: &[SomaAddress]) -> Vec<usize> {
        let mut indices = Vec::new();

        for &address in addresses {
            for (i, validator) in self.consensus_validators.iter().enumerate() {
                if validator.metadata.soma_address == address {
                    indices.push(i);
                    break;
                }
            }
        }

        indices
    }

    fn calculate_and_distribute_rewards(
        &mut self,
        total_rewards: &mut u64,
        reward_slashing_rate: u64,
        validator_report_records: &BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        new_epoch: u64,
    ) -> HashMap<SomaAddress, StakedSoma> {
        // Only consensus validators participate in rewards
        let total_voting_power: u64 = self
            .consensus_validators
            .iter()
            .map(|v| v.voting_power)
            .sum();

        if total_voting_power == 0 {
            return HashMap::new();
        }

        // Calculate rewards distribution (existing logic)
        let unadjusted_rewards =
            self.compute_unadjusted_reward_distribution(total_voting_power, *total_rewards);
        let slashed_validators = self.compute_slashed_validators(validator_report_records);
        let total_slashed_voting_power = self.sum_voting_power_by_addresses(&slashed_validators);
        let slashed_indices = self.get_consensus_validator_indices(&slashed_validators);
        let (total_adjustment, individual_adjustments) = self.compute_reward_adjustments(
            slashed_indices,
            reward_slashing_rate,
            &unadjusted_rewards,
        );
        let adjusted_rewards = self.compute_adjusted_reward_distribution(
            total_voting_power,
            total_slashed_voting_power,
            unadjusted_rewards,
            total_adjustment,
            &individual_adjustments,
        );

        self.distribute_rewards(&adjusted_rewards, total_rewards, new_epoch)
    }

    /// Calculate reward adjustments for slashed validators
    pub fn compute_reward_adjustments(
        &self,
        slashed_indices: Vec<usize>,
        reward_slashing_rate: u64,
        unadjusted_rewards: &[u64],
    ) -> (u64, BTreeMap<usize, u64>) {
        let mut total_adjustment = 0;
        let mut individual_adjustments = BTreeMap::new();

        for &index in &slashed_indices {
            let unadjusted_reward = unadjusted_rewards[index];
            let adjustment = (unadjusted_reward as u128) * (reward_slashing_rate as u128) / 10000;

            individual_adjustments.insert(index, adjustment as u64);
            total_adjustment += adjustment as u64;
        }

        (total_adjustment, individual_adjustments)
    }

    /// Calculate adjusted rewards after applying slashing
    pub fn compute_adjusted_reward_distribution(
        &self,
        total_voting_power: u64,
        total_slashed_voting_power: u64,
        unadjusted_rewards: Vec<u64>,
        total_adjustment: u64,
        individual_adjustments: &BTreeMap<usize, u64>,
    ) -> Vec<u64> {
        let total_unslashed_voting_power = total_voting_power - total_slashed_voting_power;
        let mut adjusted_rewards = Vec::with_capacity(self.consensus_validators.len());

        for i in 0..self.consensus_validators.len() {
            let validator = &self.consensus_validators[i];
            let unadjusted_reward = unadjusted_rewards[i];

            let adjusted_reward = if individual_adjustments.contains_key(&i) {
                // Slashed validator - subtract adjustment
                let adjustment = individual_adjustments[&i];
                unadjusted_reward.saturating_sub(adjustment)
            } else {
                // Unslashed validator - gets bonus based on voting power
                let bonus = (total_adjustment as u128) * (validator.voting_power as u128)
                    / (total_unslashed_voting_power as u128);
                unadjusted_reward + (bonus as u64)
            };

            adjusted_rewards.push(adjusted_reward);
        }

        adjusted_rewards
    }

    /// Distribute rewards to validators and their stakers
    pub fn distribute_rewards(
        &mut self,
        adjusted_rewards: &[u64],
        total_rewards: &mut u64,
        new_epoch: u64,
    ) -> HashMap<SomaAddress, StakedSoma> {
        assert!(!self.consensus_validators.is_empty(), "Validator set empty");

        let mut rewards = HashMap::new();
        let mut distributed_total = 0;

        for (i, validator) in self.consensus_validators.iter_mut().enumerate() {
            let reward_amount = adjusted_rewards[i];

            // Validator commission
            let commission_amount =
                (reward_amount as u128) * (validator.commission_rate as u128) / 10000;

            // Split rewards between validator and stakers
            let validator_commission = commission_amount as u64;
            let staker_reward = reward_amount - validator_commission;

            // Apply rewards
            if validator_commission > 0 {
                let reward = validator.request_add_stake(
                    validator_commission,
                    validator.metadata.soma_address,
                    new_epoch - 1,
                );
                rewards.insert(validator.metadata.soma_address, reward);
            }

            validator.deposit_staker_rewards(staker_reward);

            distributed_total += reward_amount;
        }

        // Deduct distributed rewards from total
        *total_rewards = total_rewards.saturating_sub(distributed_total);
        return rewards;
    }

    /// Calculate total stake across all validators
    pub fn calculate_total_stake(&self) -> u64 {
        let consensus_stake: u64 = self
            .consensus_validators
            .iter()
            .map(|v| v.staking_pool.soma_balance)
            .sum();

        let networking_stake: u64 = self
            .networking_validators
            .iter()
            .map(|v| v.staking_pool.soma_balance)
            .sum();

        consensus_stake + networking_stake
    }

    /// Calculate total stake INCLUDING pending (for threshold calculations)
    pub fn calculate_total_stake_with_pending(&self) -> u64 {
        let active_stake = self.calculate_total_stake();

        let pending_stake: u64 = self
            .pending_validators
            .iter()
            .map(|v| v.staking_pool.soma_balance)
            .sum();

        active_stake + pending_stake
    }

    /// Process the pending stake changes for each validator.
    fn adjust_commission_rates(&mut self) {
        self.consensus_validators
            .iter_mut()
            .for_each(|validator| validator.adjust_commission_rate());

        self.networking_validators
            .iter_mut()
            .for_each(|validator| validator.adjust_commission_rate());
    }

    /// Update validator voting power based on stake
    pub fn set_voting_power(&mut self) {
        let total_stake = self.calculate_total_stake();
        self.total_stake = total_stake;
        if total_stake == 0 {
            return;
        }

        // Combine all validators for voting power calculation
        let mut all_validators: Vec<&mut Validator> = Vec::new();
        all_validators.extend(self.consensus_validators.iter_mut());
        all_validators.extend(self.networking_validators.iter_mut());

        // Calculate dynamic threshold
        // Ensure threshold is at least high enough to distribute all power
        let validator_count = all_validators.len();
        let min_threshold = if validator_count > 0 {
            (TOTAL_VOTING_POWER + validator_count as u64 - 1) / validator_count as u64
        // divide_and_round_up
        } else {
            TOTAL_VOTING_POWER
        };
        let threshold = std::cmp::min(
            TOTAL_VOTING_POWER,
            std::cmp::max(MAX_VOTING_POWER, min_threshold),
        );

        // Sort validators by stake in descending order for consistent processing
        all_validators.sort_by(|a, b| {
            b.staking_pool
                .soma_balance
                .cmp(&a.staking_pool.soma_balance)
        });

        // First pass: calculate capped voting power based on stake
        let mut total_power = 0;
        for validator in &mut all_validators {
            let stake_fraction = (validator.staking_pool.soma_balance as u128)
                * (TOTAL_VOTING_POWER as u128)
                / (total_stake as u128);
            validator.voting_power = std::cmp::min(stake_fraction as u64, threshold);
            total_power += validator.voting_power;
        }

        // Second pass: distribute remaining power proportionally
        let mut remaining_power = TOTAL_VOTING_POWER.saturating_sub(total_power);
        let mut i = 0;
        while i < all_validators.len() && remaining_power > 0 {
            // Calculate planned distribution (evenly among remaining validators)
            let validators_left = all_validators.len() - i;
            let planned = (remaining_power + validators_left as u64 - 1) / validators_left as u64; // divide_and_round_up

            // Target power capped by threshold
            let validator = &mut all_validators[i];
            let target = std::cmp::min(threshold, validator.voting_power + planned);

            // Actual power to distribute to this validator
            let actual = std::cmp::min(remaining_power, target - validator.voting_power);
            validator.voting_power += actual;

            // Update remaining power
            remaining_power -= actual;

            i += 1;
        }

        // Verify all power was distributed
        assert!(
            remaining_power == 0,
            "Failed to distribute all voting power"
        );

        // Verify relative power ordering respects stake ordering
        self.verify_voting_power_ordering();
    }

    fn verify_voting_power_ordering(&self) {
        // Combine all validators for verification
        let all_validators: Vec<&Validator> = self
            .consensus_validators
            .iter()
            .chain(self.networking_validators.iter())
            .collect();

        for i in 0..all_validators.len() {
            for j in i + 1..all_validators.len() {
                let stake_i = all_validators[i].staking_pool.soma_balance;
                let stake_j = all_validators[j].staking_pool.soma_balance;
                let power_i = all_validators[i].voting_power;
                let power_j = all_validators[j].voting_power;

                if stake_i > stake_j {
                    assert!(
                        power_i >= power_j,
                        "Voting power order mismatch with stake order"
                    );
                }
                if stake_i < stake_j {
                    assert!(
                        power_i <= power_j,
                        "Voting power order mismatch with stake order"
                    );
                }
            }
        }
    }

    fn process_validator_transitions(
        &mut self,
        new_epoch: u64,
        validator_report_records: &mut BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        grace_period: u64,
    ) {
        let total_stake_for_thresholds = self.calculate_total_stake_with_pending();

        // === PROCESS CONSENSUS VALIDATORS ===
        let mut i = self.consensus_validators.len();
        while i > 0 {
            i -= 1;
            let validator = &self.consensus_validators[i];
            let validator_address = validator.metadata.soma_address;
            let voting_power = derive_raw_voting_power(
                validator.staking_pool.soma_balance,
                total_stake_for_thresholds,
            );

            if voting_power >= VALIDATOR_CONSENSUS_LOW_POWER {
                // Safe - remove from at-risk
                self.at_risk_validators.remove(&validator_address);
            } else if voting_power >= VALIDATOR_CONSENSUS_VERY_LOW_POWER {
                // At risk - track grace period
                let period = self
                    .at_risk_validators
                    .get(&validator_address)
                    .unwrap_or(&0)
                    + 1;
                self.at_risk_validators.insert(validator_address, period);

                if period > grace_period {
                    // Grace period exceeded - demote to networking
                    let validator = self.consensus_validators.swap_remove(i);
                    if voting_power >= VALIDATOR_NETWORKING_MIN_POWER {
                        info!(
                            "Demoting {} to networking after grace period",
                            validator_address
                        );
                        self.networking_validators.push(validator);
                    } else {
                        self.process_validator_departure(
                            validator,
                            validator_report_records,
                            new_epoch,
                            false,
                        );
                    }
                }
            } else {
                // Below critical threshold - immediate action
                let validator = self.consensus_validators.swap_remove(i);
                if voting_power >= VALIDATOR_NETWORKING_MIN_POWER {
                    // Demote to networking
                    info!(
                        "Immediately demoting {} to networking (power: {})",
                        validator.metadata.soma_address, voting_power
                    );
                    self.networking_validators.push(validator);
                } else {
                    // Remove entirely
                    self.process_validator_departure(
                        validator,
                        validator_report_records,
                        new_epoch,
                        false,
                    );
                }
            }
        }

        // === PROCESS NETWORKING VALIDATORS ===
        let mut i = self.networking_validators.len();
        while i > 0 {
            i -= 1;
            let validator = &self.networking_validators[i];
            let voting_power = derive_raw_voting_power(
                validator.staking_pool.soma_balance,
                total_stake_for_thresholds,
            );

            if voting_power >= VALIDATOR_CONSENSUS_MIN_POWER {
                // Promote to consensus
                let validator = self.networking_validators.swap_remove(i);
                let address = validator.metadata.soma_address;
                self.consensus_validators.push(validator);
                self.at_risk_validators.remove(&address);
                info!(
                    "Promoting {} to consensus (power: {})",
                    address, voting_power
                );
            } else if voting_power < VALIDATOR_NETWORKING_MIN_POWER {
                // Remove from networking
                let validator = self.networking_validators.swap_remove(i);
                self.process_validator_departure(
                    validator,
                    validator_report_records,
                    new_epoch,
                    false,
                );
            }
        }

        // === PROCESS PENDING VALIDATORS ===
        let mut i = 0;
        while i < self.pending_validators.len() {
            let validator = &mut self.pending_validators[i];
            let voting_power = derive_raw_voting_power(
                validator.staking_pool.soma_balance,
                total_stake_for_thresholds,
            );

            if voting_power >= VALIDATOR_CONSENSUS_MIN_POWER {
                // Join as consensus validator
                validator.activate(new_epoch);
                let validator = self.pending_validators.remove(i);
                info!(
                    "New consensus validator {} (power: {})",
                    validator.metadata.soma_address, voting_power
                );
                self.consensus_validators.push(validator);
            } else if voting_power >= VALIDATOR_NETWORKING_MIN_POWER {
                // Join as networking validator
                validator.activate(new_epoch);
                let validator = self.pending_validators.remove(i);
                info!(
                    "New networking validator {} (power: {})",
                    validator.metadata.soma_address, voting_power
                );
                self.networking_validators.push(validator);
            } else {
                // Stay pending
                i += 1;
            }
        }
    }

    /// Handle validator departure while maintaining staking pool
    pub fn process_validator_departure(
        &mut self,
        validator: Validator,
        validator_report_records: &mut BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        new_epoch: u64,
        is_voluntary: bool,
    ) {
        let validator_address = validator.metadata.soma_address;
        let pool_id = validator.staking_pool.id;

        // Remove from mappings
        self.at_risk_validators.remove(&validator_address);

        // Update total stake
        // self.total_stake -= validator.staking_pool.soma_balance;

        // Clean up report records
        self.clean_report_records(validator_report_records, validator_address);

        // Deactivate the validator
        let mut inactive_validator = validator;
        inactive_validator.deactivate(new_epoch);

        // Move to inactive validators, preserving staking pool
        self.inactive_validators.insert(pool_id, inactive_validator);
    }

    /// Remove validator from report records
    fn clean_report_records(
        &self,
        validator_report_records: &mut BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        leaving_validator: SomaAddress,
    ) {
        // Remove records about this validator
        validator_report_records.remove(&leaving_validator);

        // Remove reports submitted by this validator
        for reporters in validator_report_records.values_mut() {
            reporters.remove(&leaving_validator);
        }

        // Remove empty entries
        validator_report_records.retain(|_, reporters| !reporters.is_empty());
    }

    /// Process pending stakes and withdrawals for all validators
    fn process_active_validator_stakes(&mut self, new_epoch: u64) {
        // Process consensus validators
        for validator in &mut self.consensus_validators {
            validator.staking_pool.process_pending_stake_withdraw();
            validator.staking_pool.process_pending_stake();
            validator.staking_pool.exchange_rates.insert(
                new_epoch,
                PoolTokenExchangeRate {
                    soma_amount: validator.staking_pool.soma_balance,
                    pool_token_amount: validator.staking_pool.pool_token_balance,
                },
            );
        }

        // Process networking validators
        for validator in &mut self.networking_validators {
            validator.staking_pool.process_pending_stake_withdraw();
            validator.staking_pool.process_pending_stake();
            validator.staking_pool.exchange_rates.insert(
                new_epoch,
                PoolTokenExchangeRate {
                    soma_amount: validator.staking_pool.soma_balance,
                    pool_token_amount: validator.staking_pool.pool_token_balance,
                },
            );
        }
    }

    /// Process pending stakes for pending validators
    fn process_pending_validator_stakes(&mut self, new_epoch: u64) {
        for validator in &mut self.pending_validators {
            // Process pending stakes and withdrawals
            validator.staking_pool.process_pending_stake_withdraw();
            validator.staking_pool.process_pending_stake();

            // Note: We don't update exchange rates for pending validators
            // since they don't have active pools yet
        }
    }

    /// Process validators that have requested removal
    pub fn process_pending_removals(
        &mut self,
        validator_report_records: &mut BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        new_epoch: u64,
    ) {
        let mut consensus_removals: Vec<usize> = Vec::new();
        let mut networking_removals: Vec<usize> = Vec::new();

        for (is_consensus, index) in &self.pending_removals {
            if *is_consensus {
                consensus_removals.push(*index);
            } else {
                networking_removals.push(*index);
            }
        }

        // Sort both lists in descending order using existing function
        sort_removal_list(&mut consensus_removals);
        sort_removal_list(&mut networking_removals);

        // Process consensus removals first
        for index in consensus_removals.into_iter().rev() {
            if index < self.consensus_validators.len() {
                let validator = self.consensus_validators.remove(index);
                self.process_validator_departure(
                    validator,
                    validator_report_records,
                    new_epoch,
                    true, // voluntary removal
                );
            }
        }

        // Then process networking removals
        for index in networking_removals.into_iter().rev() {
            if index < self.networking_validators.len() {
                let validator = self.networking_validators.remove(index);
                self.process_validator_departure(
                    validator,
                    validator_report_records,
                    new_epoch,
                    true, // voluntary removal
                );
            }
        }

        // Clear pending removals
        self.pending_removals.clear();
    }
}

fn sort_removal_list(withdraw_list: &mut Vec<usize>) {
    let length = withdraw_list.len();
    let mut i = 1;
    while i < length {
        let cur = withdraw_list[i];
        let mut j = i;
        while j > 0 {
            j = j - 1;
            if withdraw_list[j] > cur {
                withdraw_list.swap(j, j + 1);
            } else {
                break;
            }
        }
        i = i + 1;
    }
}

/// Helper function to derive raw voting power from stake amount
fn derive_raw_voting_power(stake: u64, total_stake: u64) -> u64 {
    if total_stake == 0 {
        return 0;
    }
    ((stake as u128) * (TOTAL_VOTING_POWER as u128) / (total_stake as u128)) as u64
}
