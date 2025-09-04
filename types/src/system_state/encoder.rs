use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet},
    str::FromStr,
};

use crate::shard_crypto::keys::EncoderPublicKey;
use fastcrypto::{ed25519::Ed25519PublicKey, traits::ToFromBytes};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use crate::{
    base::SomaAddress,
    committee::{
        ENCODER_LOW_POWER, ENCODER_MIN_POWER, ENCODER_VERY_LOW_POWER, MAX_VOTING_POWER,
        QUORUM_THRESHOLD, TOTAL_VOTING_POWER,
    },
    crypto::{self, NetworkPublicKey},
    effects::ExecutionFailureStatus,
    error::ExecutionResult,
    multiaddr::Multiaddr,
    object::ObjectID,
    transaction::UpdateEncoderMetadataArgs,
};

use super::staking::{PoolTokenExchangeRate, StakedSoma, StakingPool};

/// # EncoderMetadata
///
/// Contains the identifying information and network address for an encoder.
///
/// ## Purpose
/// Stores all the necessary information to identify an encoder and communicate
/// with it over the network. This includes a cryptographic key and network address.
///
/// ## Lifecycle
/// EncoderMetadata is created when an encoder joins the network and is updated
/// when an encoder changes its key or network address. The next_epoch_* fields
/// allow for smooth transitions when encoders update their information.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize, Serialize, Hash)]
pub struct EncoderMetadata {
    /// The Soma blockchain address of the encoder
    pub soma_address: SomaAddress,

    pub encoder_pubkey: EncoderPublicKey,

    /// The network public key used for network identity and authentication
    pub network_pubkey: crate::crypto::NetworkPublicKey,

    pub internal_network_address: Multiaddr,
    pub external_network_address: Multiaddr,

    pub object_server_address: Multiaddr,

    /// Optional new network public key for the next epoch
    pub next_epoch_network_pubkey: Option<crate::crypto::NetworkPublicKey>,

    /// Optional new network address for the next epoch
    pub next_epoch_internal_network_address: Option<Multiaddr>,
    pub next_epoch_external_network_address: Option<Multiaddr>,

    /// Optional new object server address for the next epoch
    pub next_epoch_object_server_address: Option<Multiaddr>,
}

/// # Encoder
///
/// Represents an encoder in the Soma blockchain with its metadata and voting power.
///
/// ## Purpose
/// Combines encoder metadata with voting power to represent an encoder's
/// complete state in the system. The voting power determines the encoder's
/// influence in the encoder committee.
///
/// ## Usage
/// Encoders are stored in the EncoderSet and are used to form the committee
/// for encoding operations. They are added, removed, and updated during epoch
/// transitions.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct Encoder {
    /// The encoder's metadata including key and network address
    pub metadata: EncoderMetadata,

    /// The encoder's voting power in the encoding committee
    pub voting_power: u64,

    pub staking_pool: StakingPool,

    pub commission_rate: u64,

    pub next_epoch_stake: u64,

    pub next_epoch_commission_rate: u64,

    pub byte_price: u64,

    pub next_epoch_byte_price: u64,
}

impl Encoder {
    /// # Create a new encoder
    ///
    /// Creates a new encoder with the specified metadata and voting power.
    ///
    /// ## Arguments
    /// * `soma_address` - The Soma blockchain address of the encoder
    /// * `network_pubkey` - The network public key for network identity
    /// * `net_address` - The network address for general communication
    /// * `voting_power` - The encoder's voting power
    /// * `commission_rate` - The encoder's initial commission rate
    /// * `staking_pool_id` - The ID for the encoder's staking pool
    ///
    /// ## Returns
    /// A new Encoder instance with the specified metadata and voting power
    pub fn new(
        soma_address: SomaAddress,
        encoder_pubkey: EncoderPublicKey,
        network_pubkey: crypto::NetworkPublicKey,
        internal_network_address: Multiaddr,
        external_network_address: Multiaddr,
        object_server_address: Multiaddr,
        voting_power: u64,
        commission_rate: u64,
        byte_price: u64,
        staking_pool_id: ObjectID,
    ) -> Self {
        Self {
            metadata: EncoderMetadata {
                soma_address,
                encoder_pubkey,
                network_pubkey,
                internal_network_address,
                external_network_address,
                object_server_address,
                next_epoch_network_pubkey: None,
                next_epoch_external_network_address: None,
                next_epoch_internal_network_address: None,
                next_epoch_object_server_address: None,
            },
            voting_power,
            commission_rate,
            next_epoch_stake: 0,
            next_epoch_commission_rate: commission_rate,
            staking_pool: StakingPool::new(staking_pool_id),
            byte_price,
            next_epoch_byte_price: byte_price,
        }
    }

    /// Request to add stake to the encoder
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

    /// Request to withdraw stake from the encoder
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

    pub fn request_set_byte_price(&mut self, new_price: u64) -> Result<(), String> {
        const MIN_BYTE_PRICE: u64 = 1;
        const MAX_BYTE_PRICE: u64 = 1_000_000; // TODO: Set appropriate max

        if new_price < MIN_BYTE_PRICE || new_price > MAX_BYTE_PRICE {
            return Err(format!(
                "Byte price must be between {} and {}",
                MIN_BYTE_PRICE, MAX_BYTE_PRICE
            ));
        }

        self.next_epoch_byte_price = new_price;
        Ok(())
    }

    /// Activate this encoder and its staking pool
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

    /// Deactivate this encoder and its staking pool
    pub fn deactivate(&mut self, deactivation_epoch: u64) {
        // Cannot deactivate already inactive pool
        assert!(!self.staking_pool.is_inactive(), "Pool already inactive");

        // Set deactivation epoch
        self.staking_pool.deactivation_epoch = Some(deactivation_epoch);
    }

    fn adjust_commission_rate_and_byte_price(&mut self) {
        self.commission_rate = self.next_epoch_commission_rate;
        self.byte_price = self.next_epoch_byte_price;
    }

    pub fn stage_next_epoch_metadata(
        &mut self,
        args: &UpdateEncoderMetadataArgs,
    ) -> ExecutionResult<()> {
        // Process Network Address
        if let Some(ref addr_bytes) = args.next_epoch_external_network_address {
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
            self.metadata.next_epoch_external_network_address = Some(multiaddr);
        }

        if let Some(ref addr_bytes) = args.next_epoch_internal_network_address {
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
            self.metadata.next_epoch_internal_network_address = Some(multiaddr);
        }

        if let Some(ref addr_bytes) = args.next_epoch_object_server_address {
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
            self.metadata.next_epoch_object_server_address = Some(multiaddr);
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

        Ok(())
    }

    /// Apply staged metadata changes. Called during epoch transition.
    fn effectuate_staged_metadata(&mut self) {
        if let Some(addr) = self.metadata.next_epoch_internal_network_address.take() {
            self.metadata.internal_network_address = addr;
        }
        if let Some(addr) = self.metadata.next_epoch_external_network_address.take() {
            self.metadata.external_network_address = addr;
        }
        if let Some(addr) = self.metadata.next_epoch_object_server_address.take() {
            self.metadata.object_server_address = addr;
        }
        if let Some(key) = self.metadata.next_epoch_network_pubkey.take() {
            self.metadata.network_pubkey = key;
        }
    }
}

#[cfg(test)]
impl Encoder {
    /// Get the pool token exchange rate at a specific epoch
    pub fn pool_token_exchange_rate_at_epoch(&self, epoch: u64) -> PoolTokenExchangeRate {
        self.staking_pool.pool_token_exchange_rate_at_epoch(epoch)
    }

    /// Calculate rewards for this encoder's initial stake (self-stake)
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

/// # EncoderSet
///
/// Manages the set of encoders in the Soma blockchain.
///
/// ## Purpose
/// Maintains the current set of active encoders, pending encoders, and
/// encoders scheduled for removal. It also tracks the total stake in the system
/// and provides operations for adding and removing encoders.
///
/// ## Lifecycle
/// The EncoderSet is updated during epoch transitions, when pending changes
/// are applied to the active encoder set. Encoders can be added to the
/// pending set during an epoch and will become active in the next epoch.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct EncoderSet {
    /// The total stake across all active encoders
    pub total_stake: u64,

    /// The currently active encoders participating in encoding
    pub active_encoders: Vec<Encoder>,

    /// Encoders that will be added to the active set in the next epoch
    pub pending_active_encoders: Vec<Encoder>,

    /// Active encoders that will be removed in the next epoch
    pub pending_removals: Vec<usize>,

    pub staking_pool_mappings: BTreeMap<ObjectID, SomaAddress>,

    pub inactive_encoders: BTreeMap<ObjectID, Encoder>,

    pub at_risk_encoders: BTreeMap<SomaAddress, u64>,

    pub reference_byte_price: u64,
}

impl EncoderSet {
    /// # Create a new encoder set
    ///
    /// Creates a new encoder set with the specified initial active encoders.
    ///
    /// ## Arguments
    /// * `init_active_encoders` - The initial set of active encoders
    ///
    /// ## Returns
    /// A new EncoderSet instance with the specified active encoders and
    /// calculated total stake
    pub fn new(init_active_encoders: Vec<Encoder>) -> Self {
        let total_stake = init_active_encoders
            .iter()
            .map(|v| v.staking_pool.soma_balance)
            .sum();

        let mut staking_pool_mappings = BTreeMap::new();

        // Initialize staking pool mappings
        for encoder in &init_active_encoders {
            staking_pool_mappings.insert(encoder.staking_pool.id, encoder.metadata.soma_address);
        }

        let mut encoder_set = Self {
            total_stake,
            active_encoders: init_active_encoders,
            pending_active_encoders: Vec::new(),
            pending_removals: Vec::new(),
            staking_pool_mappings,
            inactive_encoders: BTreeMap::new(),
            at_risk_encoders: BTreeMap::new(),
            reference_byte_price: 1,
        };

        // Set initial voting power and reference byte price
        encoder_set.set_voting_power();

        encoder_set.calculate_reference_byte_price();

        encoder_set
    }

    /// Calculate the reference byte price using a weighted approach based on voting power
    /// The reference byte price is the price at which ≥2/3 of the voting power agrees
    /// (i.e., the price is greater than or equal to what 2/3 of validators want, weighted by stake)
    fn calculate_reference_byte_price(&mut self) {
        if self.active_encoders.is_empty() {
            self.reference_byte_price = 1; // Default if no encoders
            return;
        }

        // Get total voting power
        let total_voting_power: u64 = self
            .active_encoders
            .iter()
            .map(|encoder| encoder.voting_power)
            .sum();

        if total_voting_power == 0 {
            self.reference_byte_price = 1; // Default if no voting power
            return;
        }

        // Calculate the quorum threshold (2/3 of total voting power)
        // This is the amount of voting power needed to reach consensus
        let quorum_threshold = (total_voting_power * 2 + 2) / 3; // Ceiling division for 2/3

        // Create entries sorted by byte price (descending)
        let mut entries: Vec<_> = self
            .active_encoders
            .iter()
            .map(|encoder| (encoder.byte_price, encoder.voting_power))
            .collect();

        // Sort by byte price in descending order
        entries.sort_by(|a, b| b.0.cmp(&a.0));

        // Find the highest price where accumulated voting power >= quorum threshold
        let mut accumulated_power = 0;
        let mut result = 1; // Default to minimum byte price

        for (byte_price, voting_power) in entries {
            accumulated_power += voting_power;
            // The moment we reach the threshold, we have our price
            if accumulated_power >= quorum_threshold {
                result = byte_price;
                break;
            }
        }

        self.reference_byte_price = result;
    }

    /// Request to add an encoder (either new or previously removed)
    pub fn request_add_encoder(&mut self, encoder: Encoder) -> Result<(), ExecutionFailureStatus> {
        let address = encoder.metadata.soma_address;

        // Check for an existing encoder with the same address
        if self.find_encoder_with_pending_mut(address).is_some() {
            return Err(ExecutionFailureStatus::DuplicateEncoder);
        }

        // Add the staking pool mapping
        let new_pool_id = encoder.staking_pool.id;
        self.staking_pool_mappings.insert(new_pool_id, address);

        info!(
            "ADDED PENDING ACTIVE ENCODER: {:?}",
            encoder.metadata.external_network_address
        );

        // Add to pending encoders
        self.pending_active_encoders.push(encoder);

        Ok(())
    }

    /// # Request to remove an encoder
    ///
    /// Requests to remove an encoder from the active set in the next epoch.
    ///
    /// ## Behavior
    /// The encoder's index is added to the pending_removals list and the
    /// encoder will be removed from the active set in the next epoch when
    /// advance_epoch() is called.
    pub fn request_remove_encoder(&mut self, address: SomaAddress) -> ExecutionResult {
        let encoder = self
            .active_encoders
            .iter()
            .find_position(|v| address == v.metadata.soma_address);

        if let Some((i, _)) = encoder {
            if self.pending_removals.contains(&i) {
                return Err(ExecutionFailureStatus::EncoderAlreadyRemoved);
            }
            self.pending_removals.push(i);
        } else {
            return Err(ExecutionFailureStatus::NotAnEncoder);
        }
        Ok(())
    }

    /// # Advance to the next epoch
    ///
    /// Processes pending encoder changes and advances the encoder set to the next epoch.
    pub fn advance_epoch(
        &mut self,
        new_epoch: u64,
        total_rewards: &mut u64,
        reward_slashing_rate: u64,
        encoder_report_records: &mut BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        encoder_low_stake_grace_period: u64,
    ) -> HashMap<SomaAddress, StakedSoma> {
        // Calculate total voting power
        let total_voting_power = self.active_encoders.iter().map(|v| v.voting_power).sum();

        // Compute basic reward distribution
        let unadjusted_reward_amounts =
            self.compute_unadjusted_reward_distribution(total_voting_power, *total_rewards);

        // Identify encoders to be slashed
        let slashed_encoders = self.compute_slashed_encoders(encoder_report_records);

        // Calculate total voting power of slashed encoders
        let total_slashed_encoder_voting_power =
            self.sum_voting_power_by_addresses(&slashed_encoders);

        // Get indices of slashed encoders
        let slashed_indices = self.get_encoder_indices(&slashed_encoders);

        // Compute reward adjustments for slashed encoders
        let (total_adjustment, individual_adjustments) = self.compute_reward_adjustments(
            slashed_indices,
            reward_slashing_rate,
            &unadjusted_reward_amounts,
        );

        // Calculate adjusted rewards
        let adjusted_reward_amounts = self.compute_adjusted_reward_distribution(
            total_voting_power,
            total_slashed_encoder_voting_power,
            unadjusted_reward_amounts,
            total_adjustment,
            &individual_adjustments,
        );

        // Distribute rewards to encoders
        let encoder_rewards =
            self.distribute_rewards(&adjusted_reward_amounts, total_rewards, new_epoch);

        self.adjust_commission_rates();

        // Process pending stakes and withdrawals for active encoders
        self.process_active_encoder_stakes(new_epoch);

        // Process pending stakes and withdrawals for pending encoders
        // This ensures pending encoders accumulate stake during epoch changes
        self.process_pending_encoder_stakes(new_epoch);

        self.process_pending_removals(encoder_report_records, new_epoch);

        // Process encoders with low voting power and get new total stake
        let new_total_stake = self.update_encoder_positions_and_calculate_total_stake(
            encoder_low_stake_grace_period,
            encoder_report_records,
            new_epoch,
        );
        self.total_stake = new_total_stake;

        // Now process pending encoders AFTER processing low voting power encoders
        self.process_pending_encoders(new_epoch);

        self.effectuate_staged_metadata();

        // Finally readjust voting power after all encoder changes
        self.set_voting_power();

        // Recalculate reference byte price for the new epoch
        self.calculate_reference_byte_price();

        return encoder_rewards;
    }

    /// Effectuate pending next epoch metadata changes for all active encoders.
    fn effectuate_staged_metadata(&mut self) {
        for encoder in &mut self.active_encoders {
            encoder.effectuate_staged_metadata();
        }
    }

    /// Find an encoder by address (including pending encoders)
    pub fn find_encoder_with_pending_mut(
        &mut self,
        encoder_address: SomaAddress,
    ) -> Option<&mut Encoder> {
        // First check active encoders
        for encoder in &mut self.active_encoders {
            if encoder.metadata.soma_address == encoder_address {
                return Some(encoder);
            }
        }

        // Then check pending encoders
        for encoder in &mut self.pending_active_encoders {
            if encoder.metadata.soma_address == encoder_address {
                return Some(encoder);
            }
        }

        None
    }

    /// Find an encoder by address
    pub fn find_encoder_mut(&mut self, encoder_address: SomaAddress) -> Option<&mut Encoder> {
        for encoder in &mut self.active_encoders {
            if encoder.metadata.soma_address == encoder_address {
                return Some(encoder);
            }
        }
        None
    }

    /// Find an encoder by address (immutable version)
    pub fn find_encoder(&self, encoder_address: SomaAddress) -> Option<&Encoder> {
        for encoder in &self.active_encoders {
            if encoder.metadata.soma_address == encoder_address {
                return Some(encoder);
            }
        }
        None
    }

    /// Check if an address is an active encoder
    pub fn is_active_encoder(&self, address: SomaAddress) -> bool {
        self.find_encoder(address).is_some()
    }

    /// Check if an address belongs to a pending active encoder
    pub fn is_pending_encoder(&self, address: SomaAddress) -> bool {
        self.pending_active_encoders
            .iter()
            .any(|v| v.metadata.soma_address == address)
    }

    /// Calculate unadjusted reward distribution without slashing
    pub fn compute_unadjusted_reward_distribution(
        &self,
        total_voting_power: u64,
        total_rewards: u64,
    ) -> Vec<u64> {
        let mut reward_amounts = Vec::with_capacity(self.active_encoders.len());

        for encoder in &self.active_encoders {
            // Calculate proportional to voting power
            let voting_power = encoder.voting_power as u128;
            let reward = voting_power * (total_rewards as u128) / (total_voting_power as u128);

            reward_amounts.push(reward as u64);
        }

        reward_amounts
    }

    /// Identify encoders that should be slashed based on reports
    pub fn compute_slashed_encoders(
        &self,
        encoder_report_records: &BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
    ) -> Vec<SomaAddress> {
        let mut slashed_encoders = Vec::new();
        let quorum_threshold = QUORUM_THRESHOLD;

        for (encoder_address, reporters) in encoder_report_records {
            // Make sure encoder is active
            if self.find_encoder(*encoder_address).is_some() {
                // Calculate total voting power of reporters
                let reporter_votes =
                    self.sum_voting_power_by_addresses(&reporters.iter().cloned().collect());

                // If threshold reached, encoder should be slashed
                if reporter_votes >= quorum_threshold {
                    slashed_encoders.push(*encoder_address);
                }
            }
        }

        slashed_encoders
    }

    /// Sum voting power of a list of encoders and validators
    pub fn sum_voting_power_by_addresses(&self, addresses: &Vec<SomaAddress>) -> u64 {
        let mut sum = 0;

        for &address in addresses {
            if let Some(encoder) = self.find_encoder(address) {
                sum += encoder.voting_power;
            }
            // Note: In a real implementation, we might also need to check for validator voting power
            // if validators can report encoders
        }

        sum
    }

    /// Find encoder indices in active_encoders list
    pub fn get_encoder_indices(&self, addresses: &[SomaAddress]) -> Vec<usize> {
        let mut indices = Vec::new();

        for &address in addresses {
            for (i, encoder) in self.active_encoders.iter().enumerate() {
                if encoder.metadata.soma_address == address {
                    indices.push(i);
                    break;
                }
            }
        }

        indices
    }

    /// Calculate reward adjustments for slashed encoders
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
        let mut adjusted_rewards = Vec::with_capacity(self.active_encoders.len());

        for i in 0..self.active_encoders.len() {
            let encoder = &self.active_encoders[i];
            let unadjusted_reward = unadjusted_rewards[i];

            let adjusted_reward = if individual_adjustments.contains_key(&i) {
                // Slashed encoder - subtract adjustment
                let adjustment = individual_adjustments[&i];
                unadjusted_reward.saturating_sub(adjustment)
            } else {
                // Unslashed encoder - gets bonus based on voting power
                let bonus = (total_adjustment as u128) * (encoder.voting_power as u128)
                    / (total_unslashed_voting_power as u128);
                unadjusted_reward + (bonus as u64)
            };

            adjusted_rewards.push(adjusted_reward);
        }

        adjusted_rewards
    }

    /// Distribute rewards to encoders and their stakers
    pub fn distribute_rewards(
        &mut self,
        adjusted_rewards: &[u64],
        total_rewards: &mut u64,
        new_epoch: u64,
    ) -> HashMap<SomaAddress, StakedSoma> {
        if self.active_encoders.is_empty() {
            return HashMap::new();
        }

        let mut rewards = HashMap::new();
        let mut distributed_total = 0;

        for (i, encoder) in self.active_encoders.iter_mut().enumerate() {
            let reward_amount = adjusted_rewards[i];

            // Encoder commission
            let commission_amount =
                (reward_amount as u128) * (encoder.commission_rate as u128) / 10000;

            // Split rewards between encoder and stakers
            let encoder_commission = commission_amount as u64;
            let staker_reward = reward_amount - encoder_commission;

            // Apply rewards
            if encoder_commission > 0 {
                let reward = encoder.request_add_stake(
                    encoder_commission,
                    encoder.metadata.soma_address,
                    new_epoch - 1,
                );
                rewards.insert(encoder.metadata.soma_address, reward);
            }

            encoder.deposit_staker_rewards(staker_reward);

            distributed_total += reward_amount;
        }

        // Deduct distributed rewards from total
        *total_rewards = total_rewards.saturating_sub(distributed_total);
        return rewards;
    }

    /// Process the pending stake changes for each encoder.
    fn adjust_commission_rates(&mut self) {
        self.active_encoders
            .iter_mut()
            .for_each(|encoder| encoder.adjust_commission_rate_and_byte_price());
    }

    /// Update encoder voting power based on stake
    pub fn set_voting_power(&mut self) {
        let total_stake = self.calculate_total_stake();
        self.total_stake = total_stake;
        if total_stake == 0 {
            return;
        }

        // Calculate dynamic threshold
        // Ensure threshold is at least high enough to distribute all power
        let encoder_count = self.active_encoders.len();
        let min_threshold = if encoder_count > 0 {
            (TOTAL_VOTING_POWER + encoder_count as u64 - 1) / encoder_count as u64
        } else {
            TOTAL_VOTING_POWER
        };
        let threshold = std::cmp::min(
            TOTAL_VOTING_POWER,
            std::cmp::max(MAX_VOTING_POWER, min_threshold),
        );

        // Sort encoders by stake in descending order for consistent processing
        self.active_encoders.sort_by(|a, b| {
            b.staking_pool
                .soma_balance
                .cmp(&a.staking_pool.soma_balance)
        });

        // First pass: calculate capped voting power based on stake
        let mut total_power = 0;
        for encoder in &mut self.active_encoders {
            let stake_fraction = (encoder.staking_pool.soma_balance as u128)
                * (TOTAL_VOTING_POWER as u128)
                / (total_stake as u128);
            encoder.voting_power = std::cmp::min(stake_fraction as u64, threshold);
            total_power += encoder.voting_power;
        }

        // Second pass: distribute remaining power proportionally
        let mut remaining_power = TOTAL_VOTING_POWER.saturating_sub(total_power);
        let mut i = 0;
        while i < self.active_encoders.len() && remaining_power > 0 {
            // Calculate planned distribution (evenly among remaining encoders)
            let encoders_left = self.active_encoders.len() - i;
            let planned = (remaining_power + encoders_left as u64 - 1) / encoders_left as u64; // divide_and_round_up

            // Target power capped by threshold
            let encoder = &mut self.active_encoders[i];
            let target = std::cmp::min(threshold, encoder.voting_power + planned);

            // Actual power to distribute to this encoder
            let actual = std::cmp::min(remaining_power, target - encoder.voting_power);
            encoder.voting_power += actual;

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

    // Add this helper function to verify relative power ordering
    fn verify_voting_power_ordering(&self) {
        for i in 0..self.active_encoders.len() {
            for j in i + 1..self.active_encoders.len() {
                let stake_i = self.active_encoders[i].staking_pool.soma_balance;
                let stake_j = self.active_encoders[j].staking_pool.soma_balance;
                let power_i = self.active_encoders[i].voting_power;
                let power_j = self.active_encoders[j].voting_power;

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

    // Calculate total stake across all encoders
    pub fn calculate_total_stake(&self) -> u64 {
        self.active_encoders
            .iter()
            .map(|v| v.staking_pool.soma_balance)
            .sum()
    }

    /// Process pending stakes and withdrawals for all encoders
    fn process_active_encoder_stakes(&mut self, new_epoch: u64) {
        for encoder in &mut self.active_encoders {
            // Process pending stakes and withdrawals
            encoder.staking_pool.process_pending_stake_withdraw();
            encoder.staking_pool.process_pending_stake();

            // Update exchange rate for new epoch
            encoder.staking_pool.exchange_rates.insert(
                new_epoch,
                PoolTokenExchangeRate {
                    soma_amount: encoder.staking_pool.soma_balance,
                    pool_token_amount: encoder.staking_pool.pool_token_balance,
                },
            );
        }
    }

    /// Process pending stakes for pending encoders
    fn process_pending_encoder_stakes(&mut self, new_epoch: u64) {
        for encoder in &mut self.pending_active_encoders {
            // Process pending stakes and withdrawals
            encoder.staking_pool.process_pending_stake_withdraw();
            encoder.staking_pool.process_pending_stake();
        }
    }

    /// Process encoders with voting power below thresholds
    pub fn update_encoder_positions_and_calculate_total_stake(
        &mut self,
        low_stake_grace_period: u64,
        encoder_report_records: &mut BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        new_epoch: u64,
    ) -> u64 {
        // Calculate total stake including pending encoders
        let pending_total_stake: u64 = self
            .pending_active_encoders
            .iter()
            .map(|v| v.staking_pool.soma_balance)
            .sum();
        let initial_total_stake = self.calculate_total_stake() + pending_total_stake;

        // Process active encoders for removal if below thresholds
        let mut total_removed_stake = 0;
        let mut i = self.active_encoders.len();

        while i > 0 {
            i -= 1;
            let encoder = &self.active_encoders[i];
            let encoder_address = encoder.metadata.soma_address;
            let encoder_stake = encoder.staking_pool.soma_balance;

            // Calculate voting power as a proportion of total stake
            let voting_power = derive_raw_voting_power(encoder_stake, initial_total_stake);

            if voting_power >= ENCODER_LOW_POWER {
                // Encoder is safe - remove from at-risk if present
                self.at_risk_encoders.remove(&encoder_address);
            } else if voting_power >= ENCODER_VERY_LOW_POWER {
                // Below threshold but above critical - increment grace period
                let new_low_stake_period =
                    if let Some(&period) = self.at_risk_encoders.get(&encoder_address) {
                        // Already at risk, increment period
                        let new_period = period + 1;
                        self.at_risk_encoders.insert(encoder_address, new_period);
                        new_period
                    } else {
                        // New at-risk encoder
                        self.at_risk_encoders.insert(encoder_address, 1);
                        1
                    };

                // If grace period exceeded, remove encoder
                if new_low_stake_period > low_stake_grace_period {
                    let encoder = self.active_encoders.swap_remove(i);
                    let removed_stake = encoder.staking_pool.soma_balance;
                    self.process_encoder_departure(
                        encoder,
                        encoder_report_records,
                        new_epoch,
                        false, // not voluntary departure
                    );
                    total_removed_stake += removed_stake;
                }
            } else {
                // Voting power critically low - remove immediately
                let encoder = self.active_encoders.swap_remove(i);
                let removed_stake = encoder.staking_pool.soma_balance;
                self.process_encoder_departure(
                    encoder,
                    encoder_report_records,
                    new_epoch,
                    false, // not voluntary departure
                );
                total_removed_stake += removed_stake;
            }
        }

        // Return new total stake (initial minus removed)
        initial_total_stake - total_removed_stake
    }

    /// Handle encoder departure while maintaining staking pool
    pub fn process_encoder_departure(
        &mut self,
        encoder: Encoder,
        encoder_report_records: &mut BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        new_epoch: u64,
        is_voluntary: bool,
    ) {
        let encoder_address = encoder.metadata.soma_address;
        let pool_id = encoder.staking_pool.id;

        // Remove from mappings
        self.at_risk_encoders.remove(&encoder_address);

        // Update total stake
        self.total_stake -= encoder.staking_pool.soma_balance;

        // Clean up report records
        self.clean_report_records(encoder_report_records, encoder_address);

        // Deactivate the encoder
        let mut inactive_encoder = encoder;
        inactive_encoder.deactivate(new_epoch);

        // Move to inactive encoders, preserving staking pool
        self.inactive_encoders.insert(pool_id, inactive_encoder);
    }

    /// Remove encoder from report records
    fn clean_report_records(
        &self,
        encoder_report_records: &mut BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        leaving_encoder: SomaAddress,
    ) {
        // Remove records about this encoder
        encoder_report_records.remove(&leaving_encoder);

        // Remove reports submitted by this encoder
        for reporters in encoder_report_records.values_mut() {
            reporters.remove(&leaving_encoder);
        }

        // Remove empty entries
        encoder_report_records.retain(|_, reporters| !reporters.is_empty());
    }

    /// Process encoders during epoch advancement
    pub fn process_pending_encoders(&mut self, new_epoch: u64) {
        // Calculate total stake for voting power calculations
        let total_stake = self.calculate_total_stake();

        let mut i = 0;
        while i < self.pending_active_encoders.len() {
            let encoder = &mut self.pending_active_encoders[i];
            let encoder_stake = encoder.staking_pool.soma_balance;

            // Calculate voting power
            let voting_power = derive_raw_voting_power(encoder_stake, total_stake);

            // Check if encoder meets minimum voting power requirement
            if voting_power >= ENCODER_MIN_POWER {
                // Activate encoder's staking pool
                encoder.activate(new_epoch);
                info!(
                    "Encoder activated!: {:?}, {}",
                    encoder.metadata.external_network_address, voting_power
                );

                // Move to active encoders
                let encoder = self.pending_active_encoders.remove(i);
                self.active_encoders.push(encoder);
            } else {
                warn!(
                    "VOTING POWER NOT ENOUGH: {:?}, {}, {}",
                    encoder.metadata.external_network_address, voting_power, ENCODER_MIN_POWER
                );
                // Keep in pending and check the next one
                i += 1;
            }
        }
    }

    /// Process encoders that have requested removal
    pub fn process_pending_removals(
        &mut self,
        encoder_report_records: &mut BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,
        new_epoch: u64,
    ) {
        // Sort removal list in descending order to avoid index shifting issues
        sort_removal_list(&mut self.pending_removals);

        // Process each removal
        while let Some(index) = self.pending_removals.pop() {
            let encoder = self.active_encoders.remove(index);

            // Process as voluntary departure
            self.process_encoder_departure(encoder, encoder_report_records, new_epoch, true);
        }
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
