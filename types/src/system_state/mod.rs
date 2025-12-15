use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    str::FromStr,
};

use crate::{
    checksum::Checksum,
    crypto::DefaultHash,
    metadata::{
        DefaultDownloadMetadata, DefaultDownloadMetadataV1, DownloadMetadata, Metadata, MetadataV1,
        ObjectPath,
    },
    shard_crypto::{digest::Digest, keys::EncoderPublicKey},
};
use crate::{encoder_committee::CountUnit, shard::Shard};
use emission::EmissionPool;
use encoder::{Encoder, EncoderSet};
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
    base::{AuthorityName, SomaAddress},
    committee::{
        Authority, Committee, CommitteeWithNetworkMetadata, EpochId, NetworkMetadata,
        NetworkingCommittee, VotingPower, ENCODER_LOW_STAKE_GRACE_PERIOD,
        VALIDATOR_LOW_STAKE_GRACE_PERIOD,
    },
    config::genesis_config::{TokenDistributionSchedule, SHANNONS_PER_SOMA},
    crypto::{self, NetworkPublicKey, ProtocolPublicKey},
    effects::ExecutionFailureStatus,
    encoder_committee::{EncoderCommittee, EncoderNetworkMetadata},
    error::{ExecutionResult, SomaError, SomaResult},
    multiaddr::Multiaddr,
    object::ObjectID,
    parameters,
    peer_id::PeerId,
    transaction::{UpdateEncoderMetadataArgs, UpdateValidatorMetadataArgs},
    SYSTEM_STATE_OBJECT_ID,
};
use crate::{
    crypto::{AuthorityPublicKey, SomaKeyPair, SomaPublicKey},
    storage::object_store::ObjectStore,
};

pub mod emission;
pub mod encoder;
pub mod epoch_start;
pub mod shard;
pub mod staking;
pub mod validator;

#[cfg(test)]
#[path = "unit_tests/delegation_tests.rs"]
mod delegation_tests;
#[cfg(test)]
#[path = "unit_tests/encoder_staking.rs"]
mod encoder_staking_tests;
#[cfg(test)]
#[path = "unit_tests/networking_validator_tests.rs"]
mod networking_validator_tests;
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

    /// Get the encoder committee for the current epoch
    fn get_current_epoch_encoder_committee(&self) -> EncoderCommittee;

    /// Get the networking committee for the current epoch
    fn get_current_epoch_networking_committee(&self) -> NetworkingCommittee;

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

    /// The reference byte price for the current epoch
    pub reference_byte_price: u64,

    pub validator_report_records: BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,

    pub emission_pool: EmissionPool,

    pub encoders: EncoderSet,
    pub encoder_report_records: BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,

    /// Cached committees: [previous_epoch, current_epoch]
    /// Index 0: Previous epoch committees
    /// Index 1: Current epoch committees
    pub committees: [Option<Committees>; 2],

    /// Map of epoch -> reward amount per target for that epoch
    pub target_rewards_per_epoch: BTreeMap<EpochId, u64>,

    /// Count of targets created per epoch (for reward calculation)
    pub targets_created_per_epoch: BTreeMap<EpochId, u64>,

    /// Epoch seeds for deterministic randomness
    pub epoch_seeds: BTreeMap<EpochId, Vec<u8>>,
}

impl SystemState {
    pub fn create(
        consensus_validators: Vec<Validator>,
        networking_validators: Vec<Validator>,
        encoders: Vec<Encoder>,
        protocol_version: u64,
        epoch_start_timestamp_ms: u64,
        protocol_config: &ProtocolConfig,
        emission_fund: u64,
        emission_per_epoch: u64,
    ) -> Self {
        // Create Emission Pool
        let emission_pool = EmissionPool::new(emission_fund, emission_per_epoch);
        let parameters = protocol_config.build_system_parameters(None);
        let mut validators = ValidatorSet::new(consensus_validators, networking_validators);
        let mut encoders = EncoderSet::new(encoders);

        for validator in &mut validators.consensus_validators {
            validator.activate(0);
        }

        for validator in &mut validators.networking_validators {
            validator.activate(0);
        }

        for encoder in &mut encoders.active_encoders {
            encoder.activate(0);
        }

        // Derive initial reference byte price
        let reference_byte_price = encoders.derive_reference_byte_price();

        let mut system_state = Self {
            epoch: 0,
            validators,
            protocol_version,
            encoders,
            parameters,
            epoch_start_timestamp_ms,
            reference_byte_price,
            validator_report_records: BTreeMap::new(),
            encoder_report_records: BTreeMap::new(),
            emission_pool,
            committees: [None, None],
            target_rewards_per_epoch: BTreeMap::new(),
            targets_created_per_epoch: BTreeMap::new(),
            epoch_seeds: BTreeMap::new(),
        };

        // Initialize current epoch committees
        let current_committees =
            system_state.build_committees_for_epoch(0, protocol_config.vdf_iterations());
        system_state.committees[1] = Some(current_committees);

        system_state
    }

    /// Build committees for a specific epoch using current validator and encoder sets
    pub fn build_committees_for_epoch(&self, epoch: u64, vdf_iterations: u64) -> Committees {
        // Create snapshots of the current validator and encoder sets
        Committees::new(
            epoch,
            self.validators.clone(),
            self.encoders.clone(),
            vdf_iterations,
        )
    }

    /// Get the current epoch committees
    pub fn current_committees(&self) -> Result<&Committees, SomaError> {
        self.committees[1].as_ref().ok_or_else(|| {
            SomaError::SystemStateReadError("Current committees not initialized".to_string())
        })
    }

    /// Get the previous epoch committees
    pub fn previous_committees(&self) -> Result<&Committees, SomaError> {
        self.committees[0].as_ref().ok_or_else(|| {
            SomaError::SystemStateReadError("Previous committees not available".to_string())
        })
    }

    /// Get committees for a specific epoch
    /// Only supports current and previous epoch
    pub fn committees(&self, epoch: u64) -> Result<&Committees, SomaError> {
        if epoch == self.epoch {
            return self.current_committees();
        } else if epoch == self.epoch.saturating_sub(1) && epoch < self.epoch {
            return self.previous_committees();
        }
        Err(SomaError::SystemStateReadError(format!(
            "Committees for epoch {} not available. Current epoch: {}",
            epoch, self.epoch
        )))
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
        encoder_validator_address: Vec<u8>,
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
            Multiaddr::from_str(bcs::from_bytes(&encoder_validator_address).unwrap()).unwrap(),
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
        // TODO: make this work for validators and encoders
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

    pub fn request_add_encoder_stake_at_genesis(
        &mut self,
        signer: SomaAddress,
        address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<StakedSoma> {
        let encoder = self.encoders.find_encoder_with_pending_mut(address);

        if let Some(encoder) = encoder {
            if amount == 0 {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: "Stake amount cannot be 0!".to_string(),
                });
            }
            // Found in active or pending validators
            let staked_soma = encoder.request_add_stake_at_genesis(amount, signer, self.epoch);

            // Update staking pool mappings
            self.encoders
                .staking_pool_mappings
                .insert(staked_soma.pool_id, address);

            Ok(staked_soma)
        } else {
            Err(ExecutionFailureStatus::EncoderNotFound)
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

        // Then check encoder pools (active, pending, inactive)
        if let Some(encoder_address) = self.encoders.staking_pool_mappings.get(&pool_id).cloned() {
            if let Some(encoder) = self.encoders.find_encoder_with_pending_mut(encoder_address) {
                let withdrawn_amount = encoder.request_withdraw_stake(staked_soma, self.epoch);
                return Ok(withdrawn_amount);
            }

            if let Some(inactive_encoder) = self.encoders.inactive_encoders.get_mut(&pool_id) {
                let withdrawn_amount =
                    inactive_encoder.request_withdraw_stake(staked_soma, self.epoch);
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

    /// Request to add an encoder to the active set in the next epoch
    pub fn request_add_encoder(
        &mut self,
        signer: SomaAddress,
        encoder_pubkey_bytes: Vec<u8>,
        network_pubkey_bytes: Vec<u8>,
        internal_net_address: Vec<u8>,
        external_net_address: Vec<u8>,
        object_server_address: Vec<u8>,
        staking_pool_id: ObjectID,
    ) -> ExecutionResult {
        let encoder = Encoder::new(
            signer,
            EncoderPublicKey::new(BLS12381PublicKey::from_bytes(&encoder_pubkey_bytes).unwrap()),
            crypto::NetworkPublicKey::new(
                Ed25519PublicKey::from_bytes(&network_pubkey_bytes).unwrap(),
            ),
            Multiaddr::from_str(bcs::from_bytes(&internal_net_address).unwrap()).unwrap(),
            Multiaddr::from_str(bcs::from_bytes(&external_net_address).unwrap()).unwrap(),
            Multiaddr::from_str(bcs::from_bytes(&object_server_address).unwrap()).unwrap(),
            0,     // Initial voting power
            10,    // Default commission rate (0.1%)
            1_000, // TODO: Default Shannons per byte
            staking_pool_id,
        );

        // Request to add encoder to the encoder set
        self.encoders.request_add_encoder(encoder).map_err(|e| e) // Pass through error
    }

    /// Request to remove an encoder
    pub fn request_remove_encoder(&mut self, signer: SomaAddress) -> ExecutionResult {
        self.encoders.request_remove_encoder(signer)
    }

    /// Request to update encoder metadata
    pub fn request_update_encoder_metadata(
        &mut self,
        signer: SomaAddress,
        args: &UpdateEncoderMetadataArgs,
    ) -> ExecutionResult<()> {
        let encoder = self
            .encoders
            .find_encoder_mut(signer)
            .ok_or(ExecutionFailureStatus::NotAnEncoder)?;

        // Delegate the processing of optional fields to the encoder
        encoder.stage_next_epoch_metadata(args)
    }

    /// Request to add stake to an encoder
    pub fn request_add_stake_to_encoder(
        &mut self,
        signer: SomaAddress,
        address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<StakedSoma> {
        // Try to find the encoder in active or pending encoders
        let encoder = self.encoders.find_encoder_with_pending_mut(address);

        if let Some(encoder) = encoder {
            if amount == 0 {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: "Stake amount cannot be 0!".to_string(),
                });
            }
            // Found in active or pending encoders
            let staked_soma = encoder.request_add_stake(amount, signer, self.epoch);

            // Update staking pool mappings
            self.encoders
                .staking_pool_mappings
                .insert(staked_soma.pool_id, address);

            Ok(staked_soma)
        } else {
            Err(ExecutionFailureStatus::EncoderNotFound)
        }
    }

    /// Request to add stake to an encoder at genesis
    pub fn request_add_stake_to_encoder_at_genesis(
        &mut self,
        signer: SomaAddress,
        address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<StakedSoma> {
        let encoder = self.encoders.find_encoder_with_pending_mut(address);

        if let Some(encoder) = encoder {
            if amount == 0 {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: "Stake amount cannot be 0!".to_string(),
                });
            }
            // Found in active or pending encoders
            let staked_soma = encoder.request_add_stake_at_genesis(amount, signer, self.epoch);

            // Update staking pool mappings
            self.encoders
                .staking_pool_mappings
                .insert(staked_soma.pool_id, address);

            Ok(staked_soma)
        } else {
            Err(ExecutionFailureStatus::EncoderNotFound)
        }
    }

    /// Report an encoder for misbehavior
    pub fn report_encoder(
        &mut self,
        reporter: SomaAddress,
        reportee: SomaAddress,
    ) -> ExecutionResult {
        let is_encoder = self.encoders.is_active_encoder(reporter);

        if !is_encoder {
            return Err(ExecutionFailureStatus::NotAnEncoder);
        }

        // Verify reportee is an encoder
        if !self.encoders.is_active_encoder(reportee) {
            return Err(ExecutionFailureStatus::NotAnEncoder);
        }

        // Cannot report yourself
        if reporter == reportee {
            return Err(ExecutionFailureStatus::CannotReportOneself);
        }

        // Add report to records
        self.encoder_report_records
            .entry(reportee)
            .or_insert_with(BTreeSet::new)
            .insert(reporter);

        Ok(())
    }

    /// Undo an encoder report
    pub fn undo_report_encoder(
        &mut self,
        reporter: SomaAddress,
        reportee: SomaAddress,
    ) -> ExecutionResult {
        // Verify reporter is a encoder
        let is_encoder = self.encoders.is_active_encoder(reporter);

        if !is_encoder {
            return Err(ExecutionFailureStatus::NotAnEncoder);
        }

        // Check if report exists
        let reports = self
            .encoder_report_records
            .get_mut(&reportee)
            .ok_or(ExecutionFailureStatus::ReportRecordNotFound)?;

        // Remove the report
        if !reports.remove(&reporter) {
            return Err(ExecutionFailureStatus::ReportRecordNotFound);
        }

        // Clean up empty report sets
        if reports.is_empty() {
            self.encoder_report_records.remove(&reportee);
        }

        Ok(())
    }

    /// Set encoder commission rate
    pub fn request_set_encoder_commission_rate(
        &mut self,
        signer: SomaAddress,
        new_rate: u64,
    ) -> Result<(), ExecutionFailureStatus> {
        // Find encoder by address
        let encoder = self
            .encoders
            .find_encoder_mut(signer)
            .ok_or(ExecutionFailureStatus::NotAnEncoder)?;

        // Set commission rate
        encoder
            .request_set_commission_rate(new_rate)
            .map_err(|e| ExecutionFailureStatus::SomaError(SomaError::from(e)))?;

        Ok(())
    }

    pub fn request_set_encoder_byte_price(
        &mut self,
        signer: SomaAddress,
        new_price: u64,
    ) -> Result<(), ExecutionFailureStatus> {
        // Find encoder by address
        let encoder = self
            .encoders
            .find_encoder_mut(signer)
            .ok_or(ExecutionFailureStatus::NotAnEncoder)?;

        // Set byte price
        encoder
            .request_set_byte_price(new_price)
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
    ) -> ExecutionResult<(
        BTreeMap<SomaAddress, StakedSoma>,
        BTreeMap<SomaAddress, StakedSoma>,
    )> {
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

        // 4. Cache current committees as previous before any changes
        self.committees[0] = self.committees[1].take();

        // Adjust fees for next epoch BEFORE processing rewards
        self.adjust_value_fee(epoch_total_transaction_fees);

        // 5. Process tally-based encoder slashing and redistribution (BEFORE epoch increment)
        let slash_rewards = self.encoders.process_and_redistribute_tally_slashing(
            &mut self.encoder_report_records,
            new_epoch,
            self.parameters.encoder_tally_slash_rate_bps,
        );

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

        // 10. Process encoder epoch transition (no direct rewards - they earn via shards)
        self.encoders.advance_epoch(
            new_epoch,
            &mut self.encoder_report_records,
            ENCODER_LOW_STAKE_GRACE_PERIOD,
        );

        // 11. Derive reference byte price for new epoch
        self.reference_byte_price = self.encoders.derive_reference_byte_price();
        self.protocol_version = next_protocol_version;

        // 12. Build new committees with new epoch's vdf_iterations
        let new_committees =
            self.build_committees_for_epoch(new_epoch, next_protocol_config.vdf_iterations());
        self.committees[1] = Some(new_committees);

        // 13. Return remainder to emission pool
        if validator_reward_pool > 0 {
            self.emission_pool.balance += validator_reward_pool;
        }

        Ok((validator_rewards, slash_rewards))
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

    /// Get an encoder's SomaAddress from their public key
    /// Searches both active and inactive encoders
    pub fn get_encoder_address(&self, encoder_pubkey: &EncoderPublicKey) -> Option<SomaAddress> {
        // Search active encoders
        for encoder in &self.encoders.active_encoders {
            if &encoder.metadata.encoder_pubkey == encoder_pubkey {
                return Some(encoder.metadata.soma_address);
            }
        }

        // Search pending encoders
        for encoder in &self.encoders.pending_active_encoders {
            if &encoder.metadata.encoder_pubkey == encoder_pubkey {
                return Some(encoder.metadata.soma_address);
            }
        }

        // Search inactive encoders (they may have won before being removed)
        for encoder in self.encoders.inactive_encoders.values() {
            if &encoder.metadata.encoder_pubkey == encoder_pubkey {
                return Some(encoder.metadata.soma_address);
            }
        }

        None
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
    /// Check if an encoder has been "tallied" (slashed/removed from the committee)
    /// relative to a shard's lifecycle.
    ///
    /// An encoder is considered tallied if they were removed from the active committee
    /// in either shard.created_epoch + 1 or shard.created_epoch + 2.
    ///
    /// This is used to invalidate wins by encoders who were later found to be malicious.
    pub fn is_encoder_tallied(
        &self,
        encoder_pubkey: &EncoderPublicKey,
        shard_created_epoch: EpochId,
    ) -> bool {
        let report_epoch = shard_created_epoch + 1;
        let claim_epoch = shard_created_epoch + 2;

        // Check if encoder was tallied in either relevant epoch
        self.was_encoder_tallied_in_epoch(encoder_pubkey, report_epoch)
            || self.was_encoder_tallied_in_epoch(encoder_pubkey, claim_epoch)
    }

    /// Check if an encoder was tallied (removed/slashed) in a specific epoch.
    ///
    /// Returns true if:
    /// - The encoder was in the previous epoch's committee, AND
    /// - The encoder is NOT in this epoch's committee
    ///
    /// This indicates they were removed during the transition to this epoch.
    fn was_encoder_tallied_in_epoch(
        &self,
        encoder_pubkey: &EncoderPublicKey,
        epoch: EpochId,
    ) -> bool {
        // Can't be tallied in epoch 0
        if epoch == 0 {
            return false;
        }

        // We need both the previous and current epoch's committee to determine
        // if an encoder was removed during a transition
        let previous_epoch = epoch - 1;

        // Get previous epoch committee
        let prev_committees = match self.committees(previous_epoch) {
            Ok(c) => c,
            Err(_) => {
                // If we can't access previous committee, we can't determine tallying
                // This might happen for very old epochs - conservatively return false
                return false;
            }
        };

        // Get current epoch committee
        let curr_committees = match self.committees(epoch) {
            Ok(c) => c,
            Err(_) => {
                // If checking current epoch and committees not yet built,
                // check against current active encoders directly
                if epoch == self.epoch {
                    let was_in_previous = prev_committees
                        .encoder_set
                        .active_encoders
                        .iter()
                        .any(|e| &e.metadata.encoder_pubkey == encoder_pubkey);
                    let is_in_current = self
                        .encoders
                        .active_encoders
                        .iter()
                        .any(|e| &e.metadata.encoder_pubkey == encoder_pubkey);
                    return was_in_previous && !is_in_current;
                }
                return false;
            }
        };

        // Encoder was tallied if they were in previous but not in current
        let was_in_previous = prev_committees
            .encoder_set
            .active_encoders
            .iter()
            .any(|e| &e.metadata.encoder_pubkey == encoder_pubkey);
        let is_in_current = curr_committees
            .encoder_set
            .active_encoders
            .iter()
            .any(|e| &e.metadata.encoder_pubkey == encoder_pubkey);

        was_in_previous && !is_in_current
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
        // Use cached committee if available, otherwise build on-demand
        if let Ok(committees) = self.current_committees() {
            committees.build_validator_committee()
        } else {
            // Fallback: build directly from current state
            let validators = self
                .validators
                .consensus_validators
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
                                encoder_validator_address: verified_metadata
                                    .encoder_validator_address
                                    .clone(),
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
    }

    fn get_current_epoch_encoder_committee(&self) -> EncoderCommittee {
        // Use cached committee if available, otherwise build on-demand
        if let Ok(committees) = self.current_committees() {
            committees.build_encoder_committee()
        } else {
            // Fallback: build directly from current state
            let encoders: Vec<_> = self
                .encoders
                .active_encoders
                .iter()
                .map(|encoder| crate::encoder_committee::Encoder {
                    voting_power: encoder.voting_power,
                    encoder_key: encoder.metadata.encoder_pubkey.clone(),
                    // TODO: actually correctly set the probes
                    probe: DownloadMetadata::Default(DefaultDownloadMetadata::V1(
                        DefaultDownloadMetadataV1::new(
                            Url::from_str("https://example.com").unwrap(),
                            Metadata::V1(MetadataV1::new(Checksum::default(), 0)),
                        ),
                    )),
                })
                .collect();

            let network_metadata = self
                .encoders
                .active_encoders
                .iter()
                .map(|encoder| {
                    let metadata = &encoder.metadata;
                    let name = metadata.encoder_pubkey.clone();
                    (
                        name,
                        EncoderNetworkMetadata {
                            external_network_address: metadata.external_network_address.clone(),
                            internal_network_address: metadata.internal_network_address.clone(),
                            network_key: metadata.network_pubkey.clone(),
                            hostname: metadata.external_network_address.to_string(),
                            object_server_address: metadata.object_server_address.clone(),
                        },
                    )
                })
                .collect();

            // TODO: Calculate shard size based on number of encoders or protocol config
            let encoder_count = encoders.len() as CountUnit;
            let shard_size = std::cmp::min(encoder_count, std::cmp::max(3, encoder_count / 2));

            // TODO: Calculate quorum threshold (typically 2/3 rounded up)
            let quorum_threshold = (shard_size * 2 + 2) / 3;

            EncoderCommittee::new(
                self.epoch,
                encoders,
                shard_size,
                quorum_threshold,
                network_metadata,
            )
        }
    }

    fn get_current_epoch_networking_committee(&self) -> NetworkingCommittee {
        if let Ok(committees) = self.current_committees() {
            committees.build_networking_committee()
        } else {
            // Fallback: build directly from current state
            let members = self
                .validators
                .get_all_networking_validators()
                .map(|validator| {
                    let metadata = &validator.metadata;
                    let name = (&metadata.protocol_pubkey).into();
                    (
                        name,
                        NetworkMetadata {
                            consensus_address: metadata.p2p_address.clone(),
                            network_address: metadata.net_address.clone(),
                            primary_address: metadata.primary_address.clone(),
                            encoder_validator_address: metadata.encoder_validator_address.clone(),
                            protocol_key: ProtocolPublicKey::new(
                                metadata.worker_pubkey.clone().into_inner(),
                            ),
                            network_key: metadata.network_pubkey.clone(),
                            authority_key: metadata.protocol_pubkey.clone(),
                            hostname: metadata.net_address.to_string(),
                        },
                    )
                })
                .collect();

            NetworkingCommittee::new(self.epoch, members)
        }
    }

    fn into_epoch_start_state(self) -> EpochStartSystemState {
        EpochStartSystemState {
            epoch: self.epoch,
            protocol_version: self.protocol_version,
            epoch_start_timestamp_ms: self.epoch_start_timestamp_ms,
            epoch_duration_ms: self.parameters.epoch_duration_ms,
            active_validators: self
                .validators
                .consensus_validators
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
                        encoder_validator_address: metadata.encoder_validator_address.clone(),
                        voting_power: validator.voting_power,
                        hostname: metadata.net_address.to_string(),
                    }
                })
                .collect(),
            reference_byte_price: self.reference_byte_price,
            fee_parameters: self.fee_parameters(),
        }
    }
}

/// # Get the system state from storage
///
/// Retrieves the system state object from the object store.
///
/// ## Behavior
/// This function:
/// 1. Retrieves the system state object from the object store using the SYSTEM_STATE_OBJECT_ID
/// 2. Deserializes the object data into a SystemState struct
///
/// ## Arguments
/// * `object_store` - The object store to retrieve the system state from
///
/// ## Returns
/// The deserialized SystemState if successful, or an error if the system state
/// object could not be found or deserialized
///
/// ## Errors
/// Returns SomaError::SystemStateReadError if the system state object could not
/// be found or deserialized
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

/// # Committees
///
/// Combined committee information for both validators and encoders for a specific epoch.
///
/// ## Purpose
/// Stores both validator and encoder set snapshots for efficient lookup
/// and to maintain historical committee data across epoch transitions.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct Committees {
    /// The epoch these committees represent
    pub epoch: u64,
    /// Snapshot of the validator set for this epoch
    pub validator_set: ValidatorSet,
    /// Snapshot of the encoder set for this epoch
    pub encoder_set: EncoderSet,
    /// VDF iterations for this epoch
    pub vdf_iterations: u64,
}

impl Committees {
    /// Create new committees for the given epoch
    pub fn new(
        epoch: u64,
        validator_set: ValidatorSet,
        encoder_set: EncoderSet,
        vdf_iterations: u64,
    ) -> Self {
        Self {
            epoch,
            validator_set,
            encoder_set,
            vdf_iterations,
        }
    }

    /// Get the validator set
    pub fn validators(&self) -> &ValidatorSet {
        &self.validator_set
    }

    /// Get the encoder set
    pub fn encoders(&self) -> &EncoderSet {
        &self.encoder_set
    }

    /// Build validator committee with network metadata from the stored validator set
    pub fn build_validator_committee(&self) -> CommitteeWithNetworkMetadata {
        let validators = self
            .validator_set
            .consensus_validators
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
                            encoder_validator_address: verified_metadata
                                .encoder_validator_address
                                .clone(),
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

    /// Build encoder committee from the stored encoder set
    pub fn build_encoder_committee(&self) -> EncoderCommittee {
        let encoders: Vec<_> = self
            .encoder_set
            .active_encoders
            .iter()
            .map(|encoder| crate::encoder_committee::Encoder {
                voting_power: encoder.voting_power,
                encoder_key: encoder.metadata.encoder_pubkey.clone(),
                // TODO: correctly handle the probe
                probe: DownloadMetadata::Default(DefaultDownloadMetadata::V1(
                    DefaultDownloadMetadataV1::new(
                        Url::from_str("https://example.com").unwrap(),
                        Metadata::V1(MetadataV1::new(Checksum::default(), 0)),
                    ),
                )),
            })
            .collect();

        let network_metadata = self
            .encoder_set
            .active_encoders
            .iter()
            .map(|encoder| {
                let metadata = &encoder.metadata;
                let name = metadata.encoder_pubkey.clone();
                (
                    name,
                    EncoderNetworkMetadata {
                        internal_network_address: metadata.internal_network_address.clone(),
                        external_network_address: metadata.external_network_address.clone(),
                        network_key: metadata.network_pubkey.clone(),
                        hostname: metadata.external_network_address.to_string(),
                        object_server_address: metadata.object_server_address.clone(),
                    },
                )
            })
            .collect();

        // TODO: Calculate shard size based on number of encoders
        let encoder_count = encoders.len() as CountUnit;
        let shard_size = std::cmp::min(encoder_count, std::cmp::max(3, encoder_count / 2));

        // TODO: Calculate quorum threshold (typically 2/3 rounded up)
        let quorum_threshold = (shard_size * 2 + 2) / 3;

        EncoderCommittee::new(
            self.epoch,
            encoders,
            shard_size,
            quorum_threshold,
            network_metadata,
        )
    }

    pub fn build_networking_committee(&self) -> NetworkingCommittee {
        let members = self
            .validator_set
            .get_all_networking_validators()
            .map(|validator| {
                let metadata = &validator.metadata;
                let name = (&metadata.protocol_pubkey).into();
                (
                    name,
                    NetworkMetadata {
                        consensus_address: metadata.p2p_address.clone(),
                        network_address: metadata.net_address.clone(),
                        primary_address: metadata.primary_address.clone(),
                        encoder_validator_address: metadata.encoder_validator_address.clone(),
                        protocol_key: ProtocolPublicKey::new(
                            metadata.worker_pubkey.clone().into_inner(),
                        ),
                        network_key: metadata.network_pubkey.clone(),
                        authority_key: metadata.protocol_pubkey.clone(),
                        hostname: metadata.net_address.to_string(),
                    },
                )
            })
            .collect();

        NetworkingCommittee::new(self.epoch, members)
    }
}
