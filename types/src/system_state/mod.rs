//! # System State
//!
//! ## Overview
//! This module defines the system state of the Soma blockchain, including epoch management,
//! validator set management, and system parameters. The system state is a critical component
//! that maintains the global state of the blockchain across epochs.
//!
//! ## Responsibilities
//! - Manage the validator set (active validators, pending validators, etc.)
//! - Track epoch transitions and reconfiguration
//! - Store system-wide parameters and configuration
//! - Provide committee information for consensus
//! - Support validator addition and removal operations
//!
//! ## Component Relationships
//! - Used by Authority module to determine the current validator set
//! - Provides committee information to Consensus module
//! - Interacts with storage layer to persist system state
//! - Referenced during transaction validation and execution
//!
//! ## Key Workflows
//! 1. Epoch advancement: Processes pending validator changes and updates epoch information
//! 2. Validator management: Handles addition and removal of validators
//! 3. Committee formation: Creates committee structures for consensus operations
//!
//! ## Design Patterns
//! - Trait-based interfaces (SystemStateTrait, EpochStartSystemStateTrait) for abstraction
//! - Immutable epoch state snapshots for consistent reference
//! - Clear separation between current state and epoch transition logic

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    str::FromStr,
};

use epoch_start::{EpochStartSystemState, EpochStartValidatorInfo};
use fastcrypto::{bls12381, ed25519::Ed25519PublicKey, traits::ToFromBytes};
use serde::{Deserialize, Serialize};
use staking::StakedSoma;
use subsidy::StakeSubsidy;
use tracing::error;
use validator::{Validator, ValidatorSet};

use crate::{
    base::{AuthorityName, SomaAddress},
    committee::{
        Authority, Committee, CommitteeWithNetworkMetadata, EpochId, NetworkMetadata, VotingPower,
    },
    config::genesis_config::{TokenDistributionSchedule, SHANNONS_PER_SOMA},
    crypto::{self, NetworkPublicKey, ProtocolPublicKey},
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError, SomaResult},
    multiaddr::Multiaddr,
    parameters,
    peer_id::PeerId,
    SYSTEM_STATE_OBJECT_ID,
};
use crate::{
    crypto::{AuthorityPublicKey, SomaKeyPair, SomaPublicKey},
    storage::object_store::ObjectStore,
};

pub mod epoch_start;
pub mod staking;
pub mod subsidy;
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

/// The public key type used for validator protocol keys
///
/// This is a BLS12-381 public key used for validator signatures in the consensus protocol.
pub type PublicKey = bls12381::min_sig::BLS12381PublicKey;

/// # SystemParameters
///
/// System-wide configuration parameters that govern the behavior of the Soma blockchain.
///
/// ## Purpose
/// Defines operational parameters for the blockchain, including epoch duration,
/// validator requirements, and stake thresholds. These parameters control the
/// validator lifecycle and epoch management process.
///
/// ## Usage
/// These parameters are stored as part of the SystemState and are used during
/// epoch transitions and validator management operations.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct SystemParameters {
    /// The duration of an epoch, in milliseconds.
    pub epoch_duration_ms: u64,

    /// Lower-bound on the amount of stake required to become a validator.
    pub min_validator_joining_stake: u64,

    /// Validators with stake amount below `validator_low_stake_threshold` are considered to
    /// have low stake and will be escorted out of the validator set after being below this
    /// threshold for more than `validator_low_stake_grace_period` number of epochs.
    pub validator_low_stake_threshold: u64,

    /// Validators with stake below `validator_very_low_stake_threshold` will be removed
    /// immediately at epoch change, no grace period.
    pub validator_very_low_stake_threshold: u64,

    /// A validator can have stake below `validator_low_stake_threshold`
    /// for this many epochs before being kicked out.
    pub validator_low_stake_grace_period: u64,
}

impl Default for SystemParameters {
    /// Creates a default set of system parameters
    ///
    /// Note: These default values are primarily for testing and development.
    /// Production deployments should configure these parameters explicitly.
    // TODO: make this configurable
    fn default() -> Self {
        Self {
            epoch_duration_ms: 1000 * 60, // TODO: 1000 * 60 * 60 * 24, // 1 day
            min_validator_joining_stake: 30_000_000 * SHANNONS_PER_SOMA,
            validator_low_stake_threshold: 20_000_000 * SHANNONS_PER_SOMA,
            validator_very_low_stake_threshold: 15_000_000 * SHANNONS_PER_SOMA,
            validator_low_stake_grace_period: 7,
        }
    }
}

/// # SystemStateTrait
///
/// A trait defining the interface for accessing system state information.
///
/// ## Purpose
/// Provides a common interface for accessing epoch information and committee
/// data regardless of the underlying system state implementation. This allows
/// for different system state implementations to be used interchangeably.
///
/// ## Usage
/// This trait is implemented by SystemState and can be used by components
/// that need to access system state information without depending on the
/// specific SystemState implementation.
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
}

/// # SystemState
///
/// The global system state of the Soma blockchain.
///
/// ## Purpose
/// Represents the current state of the blockchain system, including the
/// current epoch, validator set, and system parameters. This is the primary
/// data structure for managing the blockchain's global state.
///
/// ## Lifecycle
/// The SystemState is created at genesis and updated during epoch transitions.
/// It is stored as a special object in the object store and can be accessed
/// using the SYSTEM_STATE_OBJECT_ID.
///
/// ## Thread Safety
/// This struct is not thread-safe on its own and should be protected by
/// appropriate synchronization mechanisms when accessed concurrently.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct SystemState {
    /// The current epoch number
    pub epoch: u64,
    // pub protocol_version: u64,
    // pub system_state_version: u64,
    /// The current validator set
    pub validators: ValidatorSet,

    /// System-wide configuration parameters
    pub parameters: SystemParameters,

    /// The timestamp when the current epoch started (in milliseconds)
    pub epoch_start_timestamp_ms: u64,

    pub validator_report_records: BTreeMap<SomaAddress, BTreeSet<SomaAddress>>,

    stake_subsidy: StakeSubsidy,
}

impl SystemState {
    /// # Create a new system state
    ///
    /// Creates a new system state with the specified validators, timestamp, and parameters.
    ///
    /// ## Arguments
    /// * `validators` - The initial set of validators
    /// * `epoch_start_timestamp_ms` - The timestamp when the epoch starts (in milliseconds)
    /// * `parameters` - The system parameters
    ///
    /// ## Returns
    /// A new SystemState instance with epoch 0 and the specified validators, parameters, and timestamp
    pub fn create(
        validators: Vec<Validator>,
        epoch_start_timestamp_ms: u64,
        parameters: SystemParameters,
        stake_subsidy_fund: u64,
        initial_distribution_amount: u64,
        stake_subsidy_period_length: u64,
        stake_subsidy_decrease_rate: u16,
    ) -> Self {
        // Create stake subsidy
        let stake_subsidy = StakeSubsidy::new(
            stake_subsidy_fund,
            initial_distribution_amount,
            stake_subsidy_period_length,
            stake_subsidy_decrease_rate,
        );

        let mut validators = ValidatorSet::new(validators);

        for validator in &mut validators.active_validators {
            validator.activate(0);
        }

        Self {
            epoch: 0,
            validators,
            parameters,
            epoch_start_timestamp_ms,
            validator_report_records: BTreeMap::new(),
            stake_subsidy,
        }
    }

    /// # Request to add a validator
    ///
    /// Requests to add a validator to the active set in the next epoch.
    ///
    /// ## Behavior
    /// Creates a new validator from the provided information and adds it to the
    /// pending_active_validators list in the validator set.
    ///
    /// ## Arguments
    /// * `signer` - The Soma address of the validator
    /// * `pubkey_bytes` - The BLS public key bytes for consensus operations
    /// * `network_pubkey_bytes` - The network public key bytes for network identity
    /// * `worker_pubkey_bytes` - The worker public key bytes for worker operations
    /// * `net_address` - The network address bytes for general communication
    /// * `p2p_address` - The p2p address bytes for peer-to-peer communication
    /// * `primary_address` - The primary address bytes for validator services
    ///
    /// ## Returns
    /// Ok(()) if the validator was successfully added to the pending set,
    /// or an error if the validator is already active or pending
    ///
    /// ## Errors
    /// Returns SomaError::DuplicateValidator if the validator is already
    /// in the active or pending set
    pub fn request_add_validator(
        &mut self,
        signer: SomaAddress,
        pubkey_bytes: Vec<u8>,
        network_pubkey_bytes: Vec<u8>,
        worker_pubkey_bytes: Vec<u8>,
        net_address: Vec<u8>,
        p2p_address: Vec<u8>,
        primary_address: Vec<u8>,
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
        );

        // Request to add validator to the validator set
        self.validators
            .request_add_validator(validator)
            .map_err(|e| e) // Pass through error
    }

    /// # Request to remove a validator
    ///
    /// Requests to remove a validator from the active set in the next epoch.
    ///
    /// ## Behavior
    /// Adds the validator's index to the pending_removals list in the validator set.
    ///
    /// ## Arguments
    /// * `signer` - The Soma address of the validator to remove
    /// * `pubkey_bytes` - The BLS public key bytes of the validator (unused)
    ///
    /// ## Returns
    /// Ok(()) if the validator was successfully marked for removal,
    /// or an error if the validator is not active or already marked for removal
    ///
    /// ## Errors
    /// Returns SomaError::NotAValidator if the validator is not in the active set
    /// Returns SomaError::ValidatorAlreadyRemoved if the validator is already marked for removal
    pub fn request_remove_validator(
        &mut self,
        signer: SomaAddress,
        pubkey_bytes: Vec<u8>,
    ) -> ExecutionResult {
        self.validators.request_remove_validator(signer)
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

    /// Request to withdraw stake
    pub fn request_withdraw_stake(&mut self, staked_soma: StakedSoma) -> ExecutionResult<u64> {
        let pool_id = staked_soma.pool_id;

        // TODO: make this work for validators and encoders
        // First check inactive validators since that's the most common case for withdrawal
        // from pools that don't match active validators
        if let Some(inactive_validator) = self.validators.inactive_validators.get_mut(&pool_id) {
            // Process withdrawal from inactive validator

            let withdrawn_amount =
                inactive_validator.request_withdraw_stake(staked_soma, self.epoch);

            return Ok(withdrawn_amount);
        }

        // Check active and pending validators via staking pool mappings
        if let Some(&validator_address) = self.validators.staking_pool_mappings.get(&pool_id) {
            // Find in active or pending validators
            if let Some(validator) = self
                .validators
                .find_validator_with_pending_mut(validator_address)
            {
                // Process withdrawal
                let withdrawn_amount = validator.request_withdraw_stake(staked_soma, self.epoch);
                return Ok(withdrawn_amount);
            }
        }

        // No validator found with this pool ID
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

    /// # Advance to the next epoch
    ///
    /// Processes pending validator changes and advances the system state to the next epoch.
    ///
    /// ## Behavior
    /// This method:
    /// 1. Updates the epoch start timestamp
    /// 2. Increments the epoch number
    /// 3. Calls advance_epoch() on the validator set to process pending validator changes
    ///
    /// ## Arguments
    /// * `new_epoch` - The expected new epoch number (must be current epoch + 1)
    /// * `epoch_start_timestamp_ms` - The timestamp when the new epoch starts (in milliseconds)
    ///
    /// ## Returns
    /// Ok(()) if the epoch was successfully advanced,
    /// or an error if the new epoch number is invalid
    ///
    /// ## Errors
    /// Returns SomaError::AdvancedToWrongEpoch if the new epoch number is not
    /// the current epoch + 1
    /// Advance the system to the next epoch
    pub fn advance_epoch(
        &mut self,
        new_epoch: u64,
        epoch_total_transaction_fees: u64,
        epoch_start_timestamp_ms: u64,
        reward_slashing_rate: u64,
    ) -> ExecutionResult<HashMap<SomaAddress, StakedSoma>> {
        // Verify we're advancing to the correct epoch
        if new_epoch != self.epoch + 1 {
            return Err(ExecutionFailureStatus::AdvancedToWrongEpoch);
        }

        // Save previous epoch timestamp
        let prev_epoch_start_timestamp = self.epoch_start_timestamp_ms;
        self.epoch_start_timestamp_ms = epoch_start_timestamp_ms;

        // Calculate stake subsidy if appropriate
        let mut total_rewards = epoch_total_transaction_fees;
        if self.epoch >= 0 // TODO: self.parameters.stake_subsidy_start_epoch
            && epoch_start_timestamp_ms
                >= prev_epoch_start_timestamp + self.parameters.epoch_duration_ms
        {
            // Add stake subsidy to rewards
            let stake_subsidy = self.stake_subsidy.advance_epoch();
            total_rewards += stake_subsidy;
        }

        // Actually increment the epoch number
        self.epoch = new_epoch;

        // Process validator set epoch advancement
        let rewards = self.validators.advance_epoch(
            new_epoch,
            &mut total_rewards,
            reward_slashing_rate,
            &mut self.validator_report_records,
            self.parameters.validator_low_stake_threshold,
            self.parameters.validator_very_low_stake_threshold,
            self.parameters.validator_low_stake_grace_period,
        );

        // Now process pending validators AFTER processing rewards
        // Only validators that meet minimum stake requirements will be activated
        self.validators
            .process_pending_validators(new_epoch, self.parameters.min_validator_joining_stake);

        // Finally readjust voting power after new validators are added
        // Recalculate voting power
        self.validators.set_voting_power();

        Ok(rewards)
    }
}

impl SystemStateTrait for SystemState {
    fn epoch(&self) -> u64 {
        self.epoch
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
            .active_validators
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
                            // Use net_address as hostname if no explicit hostname is available
                            hostname: verified_metadata.net_address.to_string(),
                        },
                    ),
                )
            })
            .collect();
        CommitteeWithNetworkMetadata::new(self.epoch, validators)
    }

    // fn get_pending_active_validators<S: ObjectStore + ?Sized>(
    //     &self,
    //     object_store: &S,
    // ) -> Result<Vec<SuiValidatorSummary>, SuiError> {
    //     let table_id = self.validators.pending_active_validators.contents.id;
    //     let table_size = self.validators.pending_active_validators.contents.size;
    //     let validators: Vec<Validator> =
    //         get_validators_from_table_vec(&object_store, table_id, table_size)?;
    //     Ok(validators
    //         .into_iter()
    //         .map(|v| v.into_sui_validator_summary())
    //         .collect())
    // }

    fn into_epoch_start_state(self) -> EpochStartSystemState {
        EpochStartSystemState {
            epoch: self.epoch,
            epoch_start_timestamp_ms: self.epoch_start_timestamp_ms,
            epoch_duration_ms: self.parameters.epoch_duration_ms,
            active_validators: self
                .validators
                .active_validators
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
        .get_object(&SYSTEM_STATE_OBJECT_ID)?
        // Don't panic here on None because object_store is a generic store.
        .ok_or_else(|| {
            SomaError::SystemStateReadError("SystemState object not found".to_owned())
        })?;

    let result = bcs::from_bytes::<SystemState>(object.as_inner().data.contents())
        .map_err(|err| SomaError::SystemStateReadError(err.to_string()))?;
    Ok(result)
}
