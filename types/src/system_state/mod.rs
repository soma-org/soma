use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    str::FromStr,
};

use crate::{
    checksum::Checksum,
    crypto::DefaultHash,
};
use emission::EmissionPool;
use epoch_start::{EpochStartSystemState, EpochStartValidatorInfo};
use fastcrypto::{
    bls12381::{self, min_sig::BLS12381PublicKey},
    ed25519::Ed25519PublicKey,
    hash::HashFunction as _,
    traits::ToFromBytes,
};
use protocol_config::{ProtocolConfig, SystemParameters};
use serde::{Deserialize, Serialize};
use staking::StakedSoma;
use tracing::{error, info};
use url::Url;
use validator::{Validator, ValidatorSet};

use crate::{
    SYSTEM_STATE_OBJECT_ID,
    base::{AuthorityName, SomaAddress},
    committee::{
        Authority, Committee, CommitteeWithNetworkMetadata, EpochId, NetworkMetadata,
        VALIDATOR_LOW_STAKE_GRACE_PERIOD, VotingPower,
    },
    config::genesis_config::{SHANNONS_PER_SOMA, TokenDistributionSchedule},
    crypto::{self, NetworkPublicKey, ProtocolPublicKey},
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError, SomaResult},
    multiaddr::Multiaddr,
    object::ObjectID,
    parameters,
    peer_id::PeerId,
    transaction::{UpdateEncoderMetadataArgs, UpdateValidatorMetadataArgs},
};
use crate::{
    crypto::{AuthorityPublicKey, SomaKeyPair, SomaPublicKey},
    storage::object_store::ObjectStore,
};

pub mod emission;
pub mod epoch_start;
pub mod staking;
pub mod validator;

#[cfg(test)]
#[path = "unit_tests/delegation_tests.rs"]
mod delegation_tests;
#[cfg(test)]
#[path = "unit_tests/rewards_distribution_tests.rs"]
mod rewards_distribution_tests;
#[cfg(test)]
#[path = "unit_tests/test_utils.rs"]
pub mod test_utils;

/// Fee parameters for transaction execution
/// Derived from SystemParameters at epoch start
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
pub struct FeeParameters {
    pub base_fee: u64,
    pub write_object_fee: u64,
    pub value_fee_bps: u64,
}

impl FeeParameters {
    pub fn from_system_parameters(params: &SystemParameters) -> Self {
        Self {
            base_fee: params.base_fee,
            write_object_fee: params.write_object_fee,
            value_fee_bps: params.value_fee_bps,
        }
    }

    /// Calculate value fee for a given amount
    pub fn calculate_value_fee(&self, amount: u64) -> u64 {
        (amount * self.value_fee_bps) / BPS_DENOMINATOR
    }

    /// Calculate operation fee for N object writes
    pub fn calculate_operation_fee(&self, num_objects: u64) -> u64 {
        num_objects * self.write_object_fee
    }
}

/// The public key type used for validator protocol keys
///
/// This is a BLS12-381 public key used for validator signatures in the consensus protocol.
pub type PublicKey = bls12381::min_sig::BLS12381PublicKey;

const BPS_DENOMINATOR: u64 = 10000;

pub trait SystemStateTrait {
    /// Get the current epoch number
    fn epoch(&self) -> u64;

    /// Get the timestamp when the current epoch started (in milliseconds)
    fn epoch_start_timestamp_ms(&self) -> u64;

    /// Get the duration of an epoch (in milliseconds)
    fn epoch_duration_ms(&self) -> u64;

    /// Get the committee for the current epoch, including network metadata
    fn get_current_epoch_committee(&self) -> CommitteeWithNetworkMetadata;

    /// Convert this system state to an epoch start system state
    fn into_epoch_start_state(self) -> EpochStartSystemState;

    fn protocol_version(&self) -> u64;
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
pub struct SystemState {
    /// The current epoch number
    pub epoch: u64,

    pub protocol_version: u64,
    // pub system_state_version: u64,
    /// The current validator set
    pub validators: ValidatorSet,

    /// System-wide configuration parameters
    pub parameters: SystemParameters,

    /// The timestamp when the current epoch started (in milliseconds)
    pub epoch_start_timestamp_ms: u64,

    pub validator_report_records: BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,

    pub emission_pool: EmissionPool,

    /// Map of epoch -> reward amount per target for that epoch
    pub target_rewards_per_epoch: BTreeMap<EpochId, u64>,

    /// Count of targets created per epoch (for reward calculation)
    pub targets_created_per_epoch: BTreeMap<EpochId, u64>,

    /// Epoch seeds for deterministic randomness
    pub epoch_seeds: BTreeMap<EpochId, Vec<u8>>,
}

impl SystemState {
    pub fn create(
        validators: Vec<Validator>,
        protocol_version: u64,
        epoch_start_timestamp_ms: u64,
        protocol_config: &ProtocolConfig,
        emission_fund: u64,
        emission_per_epoch: u64,
    ) -> Self {
        // Create Emission Pool
        let emission_pool = EmissionPool::new(emission_fund, emission_per_epoch);
        let parameters = protocol_config.build_system_parameters(None);
        let mut validators = ValidatorSet::new(validators);

        for validator in &mut validators.validators {
            validator.activate(0);
        }

        let mut system_state = Self {
            epoch: 0,
            validators,
            protocol_version,
            parameters,
            epoch_start_timestamp_ms,
            validator_report_records: BTreeMap::new(),
            emission_pool,
            target_rewards_per_epoch: BTreeMap::new(),
            targets_created_per_epoch: BTreeMap::new(),
            epoch_seeds: BTreeMap::new(),
        };

        system_state
    }

    pub fn request_add_validator(
        &mut self,
        signer: SomaAddress,
        pubkey_bytes: Vec<u8>,
        network_pubkey_bytes: Vec<u8>,
        worker_pubkey_bytes: Vec<u8>,
        net_address: Vec<u8>,
        p2p_address: Vec<u8>,
        primary_address: Vec<u8>,
        staking_pool_id: ObjectID,
    ) -> ExecutionResult {
        let validator = Validator::new(
            signer,
            PublicKey::from_bytes(&pubkey_bytes).unwrap(),
            crypto::NetworkPublicKey::new(
                Ed25519PublicKey::from_bytes(&network_pubkey_bytes).unwrap(),
            ),
            crypto::NetworkPublicKey::new(
                Ed25519PublicKey::from_bytes(&worker_pubkey_bytes).unwrap(),
            ),
            Multiaddr::from_str(bcs::from_bytes(&net_address).unwrap()).unwrap(),
            Multiaddr::from_str(bcs::from_bytes(&p2p_address).unwrap()).unwrap(),
            Multiaddr::from_str(bcs::from_bytes(&primary_address).unwrap()).unwrap(),
            0,
            10,
            staking_pool_id,
        );

        // Request to add validator to the validator set
        self.validators
            .request_add_validator(validator)
            .map_err(|e| e) // Pass through error
    }

    pub fn request_remove_validator(
        &mut self,
        signer: SomaAddress,
        pubkey_bytes: Vec<u8>,
    ) -> ExecutionResult {
        self.validators.request_remove_validator(signer)
    }

    pub fn request_update_validator_metadata(
        &mut self,
        signer: SomaAddress,
        args: &UpdateValidatorMetadataArgs,
    ) -> ExecutionResult<()> {
        let validator = self
            .validators
            .find_validator_mut(signer)
            // Ensure only active validators can stage changes for the next epoch
            .ok_or(ExecutionFailureStatus::NotAValidator)?;

        // Delegate the processing of optional fields to the validator
        validator.stage_next_epoch_metadata(args)
    }

    /// Request to add stake to a validator
    pub fn request_add_stake(
        &mut self,
        signer: SomaAddress,
        address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<StakedSoma> {
        // Try to find the validator in active or pending validators
        let validator = self.validators.find_validator_with_pending_mut(address);

        if let Some(validator) = validator {
            if amount == 0 {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: "Stake amount cannot be 0!".to_string(),
                });
            }
            // Found in active or pending validators
            let staked_soma = validator.request_add_stake(amount, signer, self.epoch);

            // Update staking pool mappings
            self.validators
                .staking_pool_mappings
                .insert(staked_soma.pool_id, address);

            Ok(staked_soma)
        } else {
            Err(ExecutionFailureStatus::ValidatorNotFound)
        }
    }

    pub fn request_add_stake_at_genesis(
        &mut self,
        signer: SomaAddress,
        address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<StakedSoma> {
        let validator = self.validators.find_validator_with_pending_mut(address);

        if let Some(validator) = validator {
            if amount == 0 {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: "Stake amount cannot be 0!".to_string(),
                });
            }
            // Found in active or pending validators
            let staked_soma = validator.request_add_stake_at_genesis(amount, signer, self.epoch);

            // Update staking pool mappings
            self.validators
                .staking_pool_mappings
                .insert(staked_soma.pool_id, address);

            Ok(staked_soma)
        } else {
            Err(ExecutionFailureStatus::ValidatorNotFound)
        }
    }

    /// Request to withdraw stake
    pub fn request_withdraw_stake(&mut self, staked_soma: StakedSoma) -> ExecutionResult<u64> {
        let pool_id = staked_soma.pool_id;

        // First check validator pools (active, pending, inactive)
        if let Some(validator_address) =
            self.validators.staking_pool_mappings.get(&pool_id).cloned()
        {
            if let Some(validator) = self
                .validators
                .find_validator_with_pending_mut(validator_address)
            {
                let withdrawn_amount = validator.request_withdraw_stake(staked_soma, self.epoch);
                return Ok(withdrawn_amount);
            }

            if let Some(inactive_validator) = self.validators.inactive_validators.get_mut(&pool_id)
            {
                let withdrawn_amount =
                    inactive_validator.request_withdraw_stake(staked_soma, self.epoch);
                return Ok(withdrawn_amount);
            }
        }

        // No pool found with this ID
        Err(ExecutionFailureStatus::StakingPoolNotFound)
    }

    /// Report a validator for misbehavior
    pub fn report_validator(
        &mut self,
        reporter: SomaAddress,
        reportee: SomaAddress,
    ) -> ExecutionResult {
        // Verify reporter is a validator
        if !self.validators.is_active_validator(reporter) {
            return Err(ExecutionFailureStatus::NotAValidator);
        }

        // Verify reportee is a validator
        if !self.validators.is_active_validator(reportee) {
            return Err(ExecutionFailureStatus::NotAValidator);
        }

        // Cannot report yourself
        if reporter == reportee {
            return Err(ExecutionFailureStatus::CannotReportOneself);
        }

        // Add report to records
        self.validator_report_records
            .entry(reportee)
            .or_insert_with(BTreeSet::new)
            .insert(reporter);

        Ok(())
    }

    /// Undo a validator report
    pub fn undo_report_validator(
        &mut self,
        reporter: SomaAddress,
        reportee: SomaAddress,
    ) -> ExecutionResult {
        // Verify reporter is a validator
        if !self.validators.is_active_validator(reporter) {
            return Err(ExecutionFailureStatus::NotAValidator);
        }

        // Check if report exists
        let reports = self
            .validator_report_records
            .get_mut(&reportee)
            .ok_or(ExecutionFailureStatus::ReportRecordNotFound)?;

        // Remove the report
        if !reports.remove(&reporter) {
            return Err(ExecutionFailureStatus::ReportRecordNotFound);
        }

        // Clean up empty report sets
        if reports.is_empty() {
            self.validator_report_records.remove(&reportee);
        }

        Ok(())
    }

    /// Set validator commission rate
    pub fn request_set_commission_rate(
        &mut self,
        signer: SomaAddress,
        new_rate: u64,
    ) -> Result<(), ExecutionFailureStatus> {
        // Find validator by address
        let validator = self
            .validators
            .find_validator_mut(signer)
            .ok_or(ExecutionFailureStatus::NotAValidator)?;

        // Set commission rate
        validator
            .request_set_commission_rate(new_rate)
            .map_err(|e| ExecutionFailureStatus::SomaError(SomaError::from(e)))?;

        Ok(())
    }

    pub fn advance_epoch(
        &mut self,
        new_epoch: u64,
        next_protocol_config: &ProtocolConfig,
        epoch_total_transaction_fees: u64,
        epoch_start_timestamp_ms: u64,
        epoch_randomness: Vec<u8>,
    ) -> ExecutionResult<BTreeMap<SomaAddress, StakedSoma>> {
        // 1. Verify we're advancing to the correct epoch
        if new_epoch != self.epoch + 1 {
            return Err(ExecutionFailureStatus::AdvancedToWrongEpoch);
        }

        let prev_epoch = self.epoch;
        let prev_epoch_start_timestamp = self.epoch_start_timestamp_ms;
        self.epoch_start_timestamp_ms = epoch_start_timestamp_ms;

        let next_protocol_version = next_protocol_config.version.as_u64();

        // Check if protocol version is changing
        if next_protocol_version != self.protocol_version {
            info!(
                "Protocol upgrade: {} -> {}",
                self.protocol_version, next_protocol_version
            );

            // Update parameters from new protocol config
            // Preserve current value_fee_bps since it's dynamically adjusted
            self.parameters =
                next_protocol_config.build_system_parameters(Some(self.parameters.value_fee_bps));
        }

        // Get reward_slashing_rate from protocol config
        let reward_slashing_rate = next_protocol_config.reward_slashing_rate_bps();

        // 2. Calculate total rewards (emissions + fees)
        let mut total_rewards = epoch_total_transaction_fees;
        if epoch_start_timestamp_ms
            >= prev_epoch_start_timestamp + self.parameters.epoch_duration_ms
        {
            total_rewards += self.emission_pool.advance_epoch();
        }
        // 3. Generate and store epoch seed for deterministic randomness
        let epoch_seed = self.generate_epoch_seed(new_epoch, &epoch_randomness);
        self.set_epoch_seed(new_epoch, epoch_seed);

        // Adjust fees for next epoch BEFORE processing rewards
        self.adjust_value_fee(epoch_total_transaction_fees);

        // 6. Increment epoch
        self.epoch = new_epoch;

        // 7. Allocate rewards: targets vs validators
        let target_allocation =
            (total_rewards * self.parameters.target_reward_allocation_bps) / BPS_DENOMINATOR;
        let validator_allocation = total_rewards - target_allocation;

        // 8. Calculate and store target rewards for the PREVIOUS epoch
        // Targets created in prev_epoch are valid in new_epoch, claimable in new_epoch + 1
        self.calculate_target_rewards(prev_epoch, target_allocation);

        // 9. Process validator rewards (minimal - just for consensus participation)
        let mut validator_reward_pool = validator_allocation;
        let validator_rewards = self.validators.advance_epoch(
            new_epoch,
            &mut validator_reward_pool,
            reward_slashing_rate, // 50% of tallied rewards get slashed and redistributed to other validators
            &mut self.validator_report_records,
            VALIDATOR_LOW_STAKE_GRACE_PERIOD,
        );

        self.protocol_version = next_protocol_version;

        // 12. Return remainder to emission pool
        if validator_reward_pool > 0 {
            self.emission_pool.balance += validator_reward_pool;
        }

        Ok(validator_rewards)
    }

    fn generate_epoch_seed(&self, epoch: EpochId, state_hash_digest: &[u8]) -> Vec<u8> {
        let mut hasher = DefaultHash::default();
        hasher.update(&epoch.to_le_bytes());
        hasher.update(state_hash_digest);

        // Chain with previous seed for continuity
        if epoch > 0 {
            if let Some(prev_seed) = self.epoch_seeds.get(&(epoch - 1)) {
                hasher.update(prev_seed);
            }
        }

        hasher.finalize().to_vec()
    }

    /// Return funds to the emissions pool (used when system targets have no valid winner)
    pub fn return_to_emissions_pool(&mut self, amount: u64) {
        self.emission_pool.balance += amount;
    }

    /// Increment the target count for a given epoch
    pub fn increment_target_count(&mut self, epoch: EpochId) {
        *self.targets_created_per_epoch.entry(epoch).or_insert(0) += 1;
    }

    /// Get the epoch seed for deterministic randomness
    pub fn get_epoch_seed(&self, epoch: EpochId) -> Option<Vec<u8>> {
        self.epoch_seeds.get(&epoch).cloned()
    }

    /// Set the epoch seed (called during epoch transitions)
    pub fn set_epoch_seed(&mut self, epoch: EpochId, seed: Vec<u8>) {
        self.epoch_seeds.insert(epoch, seed);
    }

    /// Get the reward amount per target for a given epoch
    /// This should be calculated based on emissions allocated for that epoch
    /// divided by the number of targets created
    pub fn get_target_reward(&self, epoch: EpochId) -> Option<u64> {
        // First check if we have a pre-calculated reward
        if let Some(&reward) = self.target_rewards_per_epoch.get(&epoch) {
            return Some(reward);
        }

        // If not pre-calculated, we can't determine it retroactively
        // (this shouldn't happen if calculate_target_rewards is called at epoch end)
        None
    }

    /// Calculate and store target rewards for an epoch
    /// Should be called during epoch transition after all targets for that epoch are known
    pub fn calculate_target_rewards(&mut self, epoch: EpochId, total_target_emissions: u64) {
        let target_count = self
            .targets_created_per_epoch
            .get(&epoch)
            .copied()
            .unwrap_or(0);

        if target_count == 0 {
            // No targets created, return emissions to pool
            self.emission_pool.balance += total_target_emissions;
            info!(
                "No targets in epoch {}, returning {} to emission pool",
                epoch, total_target_emissions
            );
            return;
        }

        let reward_per_target = total_target_emissions / target_count;
        self.target_rewards_per_epoch
            .insert(epoch, reward_per_target);

        // Handle remainder - return to emissions pool
        let remainder = total_target_emissions % target_count;
        if remainder > 0 {
            self.emission_pool.balance += remainder;
        }

        info!(
            "Epoch {} target rewards: {} per target ({} targets, {} total)",
            epoch, reward_per_target, target_count, total_target_emissions
        );
    }

    /// Adjust value fee based on actual vs target fee collection
    /// Called during advance_epoch
    fn adjust_value_fee(&mut self, actual_fees_collected: u64) {
        let target = self.parameters.target_epoch_fee_collection;
        let current_bps = self.parameters.value_fee_bps;
        let adjustment_rate = self.parameters.fee_adjustment_rate_bps;

        // Avoid division by zero
        if target == 0 {
            return;
        }

        // Calculate ratio: actual / target (scaled by BPS_DENOMINATOR)
        // ratio > 10000 means over target (network busy)
        // ratio < 10000 means under target (network quiet)
        let ratio = (actual_fees_collected.saturating_mul(BPS_DENOMINATOR))
            .checked_div(target)
            .unwrap_or(BPS_DENOMINATOR);

        let new_bps = if ratio > BPS_DENOMINATOR {
            // Over target - increase fees to reduce demand
            // Scale increase by how much we exceeded
            let excess_ratio = ratio - BPS_DENOMINATOR;
            let increase = std::cmp::min(
                (current_bps * excess_ratio) / BPS_DENOMINATOR,
                (current_bps * adjustment_rate) / BPS_DENOMINATOR,
            );
            std::cmp::min(
                current_bps.saturating_add(increase),
                self.parameters.max_value_fee_bps,
            )
        } else {
            // Under target - decrease fees to encourage activity
            let deficit_ratio = BPS_DENOMINATOR - ratio;
            let decrease = std::cmp::min(
                (current_bps * deficit_ratio) / BPS_DENOMINATOR,
                (current_bps * adjustment_rate) / BPS_DENOMINATOR,
            );
            std::cmp::max(
                current_bps.saturating_sub(decrease),
                self.parameters.min_value_fee_bps,
            )
        };

        if new_bps != current_bps {
            info!(
                "Fee adjustment: {} -> {} bps (collected {} vs target {})",
                current_bps, new_bps, actual_fees_collected, target
            );
        }

        self.parameters.value_fee_bps = new_bps;
    }

    /// Get current fee parameters for transaction execution
    pub fn fee_parameters(&self) -> FeeParameters {
        FeeParameters::from_system_parameters(&self.parameters)
    }
}

impl SystemStateTrait for SystemState {
    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn protocol_version(&self) -> u64 {
        self.protocol_version
    }

    fn epoch_start_timestamp_ms(&self) -> u64 {
        self.epoch_start_timestamp_ms
    }

    fn epoch_duration_ms(&self) -> u64 {
        self.parameters.epoch_duration_ms
    }

    fn get_current_epoch_committee(&self) -> CommitteeWithNetworkMetadata {
        let validators = self
            .validators
            .validators
            .iter()
            .map(|validator| {
                let verified_metadata = validator.metadata.clone();
                let name = (&verified_metadata.protocol_pubkey).into();
                (
                    name,
                    (
                        validator.voting_power,
                        NetworkMetadata {
                            consensus_address: verified_metadata.p2p_address.clone(),
                            network_address: verified_metadata.net_address.clone(),
                            primary_address: verified_metadata.primary_address.clone(),
                            protocol_key: ProtocolPublicKey::new(
                                verified_metadata.worker_pubkey.into_inner(),
                            ),
                            network_key: verified_metadata.network_pubkey,
                            authority_key: verified_metadata.protocol_pubkey,
                            hostname: verified_metadata.net_address.to_string(),
                        },
                    ),
                )
            })
            .collect();
        CommitteeWithNetworkMetadata::new(self.epoch, validators)
    }

    fn into_epoch_start_state(self) -> EpochStartSystemState {
        EpochStartSystemState {
            epoch: self.epoch,
            protocol_version: self.protocol_version,
            epoch_start_timestamp_ms: self.epoch_start_timestamp_ms,
            epoch_duration_ms: self.parameters.epoch_duration_ms,
            active_validators: self
                .validators
                .validators
                .iter()
                .map(|validator| {
                    let metadata = validator.metadata.clone();
                    EpochStartValidatorInfo {
                        soma_address: metadata.soma_address,
                        protocol_pubkey: metadata.protocol_pubkey.clone(),
                        network_pubkey: metadata.network_pubkey.clone(),
                        worker_pubkey: metadata.worker_pubkey.clone(),
                        net_address: metadata.net_address.clone(),
                        p2p_address: metadata.p2p_address.clone(),
                        primary_address: metadata.primary_address.clone(),
                        voting_power: validator.voting_power,
                        hostname: metadata.net_address.to_string(),
                    }
                })
                .collect(),

            fee_parameters: self.fee_parameters(),
        }
    }
}

pub fn get_system_state(object_store: &dyn ObjectStore) -> Result<SystemState, SomaError> {
    let object = object_store
        .get_object(&SYSTEM_STATE_OBJECT_ID)
        // Don't panic here on None because object_store is a generic store.
        .ok_or_else(|| {
            SomaError::SystemStateReadError("SystemState object not found".to_owned())
        })?;

    let result = bcs::from_bytes::<SystemState>(object.as_inner().data.contents())
        .map_err(|err| SomaError::SystemStateReadError(err.to_string()))?;
    Ok(result)
}
