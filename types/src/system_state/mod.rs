// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::str::FromStr;

use emission::EmissionPool;
use enum_dispatch::enum_dispatch;
use epoch_start::{EpochStartSystemState, EpochStartValidatorInfoV1};
use fastcrypto::bls12381::min_sig::BLS12381PublicKey;
use fastcrypto::bls12381::{self};
use fastcrypto::ed25519::Ed25519PublicKey;
use fastcrypto::hash::HashFunction as _;
use fastcrypto::traits::ToFromBytes;
use model_registry::ModelRegistry;
use protocol_config::{ProtocolConfig, SomaTensor, SystemParameters};
use serde::{Deserialize, Serialize};
use staking::{StakedSomaV1, StakingPool};
use target_state::TargetState;
use tracing::{error, info};
use url::Url;
use validator::{Validator, ValidatorSet};

use crate::base::{AuthorityName, SomaAddress};
use crate::checksum::Checksum;
use crate::committee::{
    Authority, Committee, CommitteeWithNetworkMetadata, EpochId, NetworkMetadata,
    VALIDATOR_LOW_STAKE_GRACE_PERIOD, VotingPower,
};
use crate::config::genesis_config::{SHANNONS_PER_SOMA, TokenDistributionSchedule};
use crate::crypto::{
    self, AuthorityPublicKey, DecryptionKey, DefaultHash, NetworkPublicKey, ProtocolPublicKey,
    SomaKeyPair, SomaPublicKey,
};
use crate::digests::{DecryptionKeyCommitment, EmbeddingCommitment, ModelWeightsCommitment};
use crate::effects::ExecutionFailureStatus;
use crate::error::{ExecutionResult, SomaError, SomaResult};
use crate::metadata::Manifest;
use crate::model::{ArchitectureVersion, ModelId, ModelV1, PendingModelUpdate};
use crate::multiaddr::Multiaddr;
use crate::object::ObjectID;
use crate::peer_id::PeerId;
use crate::storage::object_store::ObjectStore;
use crate::transaction::UpdateValidatorMetadataArgs;
use crate::{SYSTEM_STATE_OBJECT_ID, parameters};

pub mod emission;
pub mod epoch_start;
pub mod model_registry;
pub mod staking;
pub mod target_state;
pub mod validator;

#[cfg(test)]
#[path = "unit_tests/delegation_tests.rs"]
mod delegation_tests;
#[cfg(test)]
#[path = "unit_tests/model_tests.rs"]
mod model_tests;
#[cfg(test)]
#[path = "unit_tests/rewards_distribution_tests.rs"]
mod rewards_distribution_tests;
#[cfg(test)]
#[path = "unit_tests/submission_tests.rs"]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod submission_tests;
#[cfg(test)]
#[path = "unit_tests/target_tests.rs"]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod target_tests;
#[cfg(test)]
#[path = "unit_tests/test_utils.rs"]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub mod test_utils;
#[cfg(test)]
#[path = "unit_tests/validator_pop_tests.rs"]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod validator_pop_tests;

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

#[enum_dispatch]
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

/// Versioned wrapper for SystemState.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[enum_dispatch(SystemStateTrait)]
pub enum SystemState {
    V1(SystemStateV1),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SystemStateV1 {
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

    /// Registry of all models (active, pending, inactive) in the data submission system
    pub model_registry: ModelRegistry,

    pub emission_pool: EmissionPool,

    /// Lightweight coordination state for target generation and difficulty
    pub target_state: TargetState,

    /// Whether the system is in safe mode due to a failed epoch transition.
    /// Set to true when advance_epoch() fails; reset to false on next successful advance.
    pub safe_mode: bool,

    /// Transaction fees accumulated during safe mode epochs, waiting to be distributed.
    pub safe_mode_accumulated_fees: u64,

    /// Emission rewards accumulated during safe mode epochs, waiting to be distributed.
    pub safe_mode_accumulated_emissions: u64,
}

impl SystemStateV1 {
    pub fn create(
        validators: Vec<Validator>,
        protocol_version: u64,
        epoch_start_timestamp_ms: u64,
        protocol_config: &ProtocolConfig,
        emission_fund: u64,
        emission_per_epoch: u64,
        epoch_duration_ms_override: Option<u64>,
    ) -> Self {
        // Create Emission Pool
        let emission_pool = EmissionPool::new(emission_fund, emission_per_epoch);
        let mut parameters = protocol_config.build_system_parameters(None);
        if let Some(epoch_duration_ms) = epoch_duration_ms_override {
            parameters.epoch_duration_ms = epoch_duration_ms;
        }
        let mut validators = ValidatorSet::new(validators);

        for validator in &mut validators.validators {
            validator.activate(0);
        }

        // Initialize target state with initial thresholds from parameters
        let target_state = TargetState::new(parameters.target_initial_distance_threshold.clone());

        Self {
            epoch: 0,
            validators,
            protocol_version,
            parameters,
            epoch_start_timestamp_ms,
            validator_report_records: BTreeMap::new(),
            model_registry: ModelRegistry::new(),
            emission_pool,
            target_state,
            safe_mode: false,
            safe_mode_accumulated_fees: 0,
            safe_mode_accumulated_emissions: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::result_large_err)]
    pub fn request_add_validator(
        &mut self,
        signer: SomaAddress,
        pubkey_bytes: Vec<u8>,
        network_pubkey_bytes: Vec<u8>,
        worker_pubkey_bytes: Vec<u8>,
        proof_of_possession_bytes: Vec<u8>,
        net_address: Vec<u8>,
        p2p_address: Vec<u8>,
        primary_address: Vec<u8>,
        proxy_address: Vec<u8>,
        staking_pool_id: ObjectID,
    ) -> ExecutionResult {
        let protocol_pubkey = PublicKey::from_bytes(&pubkey_bytes).map_err(|e| {
            ExecutionFailureStatus::InvalidArguments {
                reason: format!("Invalid protocol public key: {}", e),
            }
        })?;

        let network_pubkey = crypto::NetworkPublicKey::new(
            Ed25519PublicKey::from_bytes(&network_pubkey_bytes).map_err(|e| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Invalid network public key: {}", e),
                }
            })?,
        );

        let worker_pubkey = crypto::NetworkPublicKey::new(
            Ed25519PublicKey::from_bytes(&worker_pubkey_bytes).map_err(|e| {
                ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Invalid worker public key: {}", e),
                }
            })?,
        );

        // Parse and verify proof of possession
        let pop =
            crypto::AuthoritySignature::from_bytes(&proof_of_possession_bytes).map_err(|e| {
                ExecutionFailureStatus::InvalidProofOfPossession {
                    reason: format!("Invalid PoP signature bytes: {}", e),
                }
            })?;
        crypto::verify_proof_of_possession(&pop, &protocol_pubkey, signer).map_err(|e| {
            ExecutionFailureStatus::InvalidProofOfPossession {
                reason: format!("PoP verification failed: {}", e),
            }
        })?;

        let parse_address = |bytes: &[u8],
                             field: &str|
         -> Result<Multiaddr, ExecutionFailureStatus> {
            let addr_str: String =
                bcs::from_bytes(bytes).map_err(|_| ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Failed to BCS deserialize {} string", field),
                })?;
            Multiaddr::from_str(&addr_str).map_err(|e| ExecutionFailureStatus::InvalidArguments {
                reason: format!("Invalid {} multiaddr format: {}", field, e),
            })
        };

        let net_addr = parse_address(&net_address, "network address")?;
        let p2p_addr = parse_address(&p2p_address, "p2p address")?;
        let primary_addr = parse_address(&primary_address, "primary address")?;
        let proxy_addr = parse_address(&proxy_address, "proxy address")?;

        let validator = Validator::new(
            signer,
            protocol_pubkey,
            network_pubkey,
            worker_pubkey,
            pop,
            net_addr,
            p2p_addr,
            primary_addr,
            proxy_addr,
            0,
            10,
            staking_pool_id,
        );

        // Request to add validator to the validator set
        self.validators.request_add_validator(validator)
    }

    #[allow(clippy::result_large_err)]
    pub fn request_remove_validator(
        &mut self,
        signer: SomaAddress,
        pubkey_bytes: Vec<u8>,
    ) -> ExecutionResult {
        self.validators.request_remove_validator(signer)
    }

    #[allow(clippy::result_large_err)]
    pub fn request_update_validator_metadata(
        &mut self,
        signer: SomaAddress,
        args: &UpdateValidatorMetadataArgs,
    ) -> ExecutionResult<()> {
        // Snapshot all validators for cross-validator duplicate checks.
        // We need this before taking a mutable borrow on the target validator.
        let all_validators = self.validators.validators.clone();

        let validator = self
            .validators
            .find_validator_mut(signer)
            // Ensure only active validators can stage changes for the next epoch
            .ok_or(ExecutionFailureStatus::NotAValidator)?;

        // Delegate the processing of optional fields to the validator
        validator.stage_next_epoch_metadata(args, &all_validators)
    }

    /// Request to add stake to a validator
    #[allow(clippy::result_large_err)]
    pub fn request_add_stake(
        &mut self,
        signer: SomaAddress,
        address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<StakedSomaV1> {
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
            self.validators.staking_pool_mappings.insert(staked_soma.pool_id, address);

            Ok(staked_soma)
        } else {
            Err(ExecutionFailureStatus::ValidatorNotFound)
        }
    }

    #[allow(clippy::result_large_err)]
    pub fn request_add_stake_at_genesis(
        &mut self,
        signer: SomaAddress,
        address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<StakedSomaV1> {
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
            self.validators.staking_pool_mappings.insert(staked_soma.pool_id, address);

            Ok(staked_soma)
        } else {
            Err(ExecutionFailureStatus::ValidatorNotFound)
        }
    }

    /// Add a model directly at genesis, bypassing commit-reveal.
    /// The model is created as active with `activation_epoch = Some(0)`.
    /// Mirrors how validators are created directly in `SystemState::create()`.
    ///
    /// The staking pool starts empty — initial stake is added via
    /// `request_add_stake_to_model_at_genesis` through token allocations
    /// (same pattern as validator staking at genesis).
    #[allow(clippy::too_many_arguments)]
    pub fn add_model_at_genesis(
        &mut self,
        model_id: ModelId,
        owner: SomaAddress,
        manifest: Manifest,
        decryption_key: DecryptionKey,
        weights_commitment: ModelWeightsCommitment,
        architecture_version: ArchitectureVersion,
        embedding_commitment: EmbeddingCommitment,
        decryption_key_commitment: DecryptionKeyCommitment,
        embedding: SomaTensor,
        commission_rate: u64,
    ) {
        assert!(self.epoch == 0, "Must be called during genesis");
        assert!(commission_rate <= BPS_DENOMINATOR, "Commission rate exceeds max");

        let mut staking_pool = StakingPool::new(ObjectID::random());
        // Activate the pool at epoch 0 (same as validator.activate(0))
        staking_pool.activation_epoch = Some(0);
        staking_pool
            .exchange_rates
            .insert(0, staking::PoolTokenExchangeRate { soma_amount: 0, pool_token_amount: 0 });

        let model = ModelV1 {
            owner,
            architecture_version,
            manifest,
            weights_commitment,
            embedding_commitment,
            decryption_key_commitment,
            commit_epoch: 0,
            decryption_key: Some(decryption_key),
            embedding: Some(embedding),
            staking_pool,
            commission_rate,
            next_epoch_commission_rate: commission_rate,
            pending_update: None,
        };

        self.model_registry.staking_pool_mappings.insert(model.staking_pool.id, model_id);
        self.model_registry.active_models.insert(model_id, model);
    }

    /// Add stake to a genesis model. Mirrors `request_add_stake_at_genesis` for validators.
    /// Immediately processes pending stake (no epoch delay).
    #[allow(clippy::result_large_err)]
    pub fn request_add_stake_to_model_at_genesis(
        &mut self,
        model_id: &ModelId,
        amount: u64,
    ) -> ExecutionResult<StakedSomaV1> {
        assert!(self.epoch == 0, "Must be called during genesis");

        if amount == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Stake amount cannot be 0!".to_string(),
            });
        }

        let model = self
            .model_registry
            .active_models
            .get_mut(model_id)
            .ok_or(ExecutionFailureStatus::ModelNotFound)?;

        let staked_soma = model.staking_pool.request_add_stake(amount, 0);
        model.staking_pool.process_pending_stake();
        self.model_registry.total_model_stake += amount;

        Ok(staked_soma)
    }

    /// Request to withdraw stake from a validator or model staking pool.
    /// Uses `StakedSomaV1.pool_id` to route to the correct pool via staking_pool_mappings.
    #[allow(clippy::result_large_err)]
    pub fn request_withdraw_stake(&mut self, staked_soma: StakedSomaV1) -> ExecutionResult<u64> {
        let pool_id = staked_soma.pool_id;

        // First check validator pools (active, pending, inactive)
        if let Some(validator_address) =
            self.validators.staking_pool_mappings.get(&pool_id).cloned()
        {
            if let Some(validator) =
                self.validators.find_validator_with_pending_mut(validator_address)
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

        // Then check model pools (active, pending, inactive)
        if let Some(model_id) = self.model_registry.staking_pool_mappings.get(&pool_id).cloned() {
            // Check active models
            if let Some(model) = self.model_registry.active_models.get_mut(&model_id) {
                let withdrawn_amount =
                    model.staking_pool.request_withdraw_stake(staked_soma, self.epoch);
                self.model_registry.total_model_stake =
                    self.model_registry.total_model_stake.saturating_sub(withdrawn_amount);
                return Ok(withdrawn_amount);
            }

            // Check pending models
            if let Some(model) = self.model_registry.pending_models.get_mut(&model_id) {
                let withdrawn_amount =
                    model.staking_pool.request_withdraw_stake(staked_soma, self.epoch);
                return Ok(withdrawn_amount);
            }

            // Check inactive models
            if let Some(model) = self.model_registry.inactive_models.get_mut(&model_id) {
                let withdrawn_amount =
                    model.staking_pool.request_withdraw_stake(staked_soma, self.epoch);
                return Ok(withdrawn_amount);
            }
        }

        // No pool found with this ID
        Err(ExecutionFailureStatus::StakingPoolNotFound)
    }

    /// Report a validator for misbehavior
    #[allow(clippy::result_large_err)]
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
        self.validator_report_records.entry(reportee).or_default().insert(reporter);

        Ok(())
    }

    /// Undo a validator report
    #[allow(clippy::result_large_err)]
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
    #[allow(clippy::result_large_err)]
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

    // -----------------------------------------------------------------------
    // Model Registry methods
    // -----------------------------------------------------------------------

    /// Find a model by ID in active or pending registries (immutable).
    pub fn find_model(&self, model_id: &ModelId) -> Option<&ModelV1> {
        self.model_registry
            .active_models
            .get(model_id)
            .or_else(|| self.model_registry.pending_models.get(model_id))
    }

    /// Find a model by ID in active or pending registries (mutable).
    pub fn find_model_mut(&mut self, model_id: &ModelId) -> Option<&mut ModelV1> {
        if self.model_registry.active_models.contains_key(model_id) {
            return self.model_registry.active_models.get_mut(model_id);
        }
        self.model_registry.pending_models.get_mut(model_id)
    }

    /// Commit a new model (Phase 1 of commit-reveal).
    /// Creates a pending model with a new StakingPool. Returns the StakedSomaV1 receipt.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::result_large_err)]
    pub fn request_commit_model(
        &mut self,
        owner: SomaAddress,
        model_id: ModelId,
        manifest: Manifest,
        weights_commitment: ModelWeightsCommitment,
        architecture_version: ArchitectureVersion,
        embedding_commitment: EmbeddingCommitment,
        decryption_key_commitment: DecryptionKeyCommitment,
        stake_amount: u64,
        commission_rate: u64,
        staking_pool_id: ObjectID,
    ) -> ExecutionResult<StakedSomaV1> {
        if commission_rate > BPS_DENOMINATOR {
            return Err(ExecutionFailureStatus::ModelCommissionRateTooHigh);
        }

        let mut staking_pool = StakingPool::new(staking_pool_id);
        let stake_activation_epoch = self.epoch + 1;
        let staked_soma = staking_pool.request_add_stake(stake_amount, stake_activation_epoch);
        // Pre-active pool: process stake immediately so soma_balance is set
        staking_pool.process_pending_stake();

        let model = ModelV1 {
            owner,
            architecture_version,
            manifest,
            weights_commitment,
            embedding_commitment,
            decryption_key_commitment,
            commit_epoch: self.epoch,
            decryption_key: None,
            embedding: None,
            staking_pool,
            commission_rate,
            next_epoch_commission_rate: commission_rate,
            pending_update: None,
        };

        self.model_registry.pending_models.insert(model_id, model);
        self.model_registry.staking_pool_mappings.insert(staking_pool_id, model_id);

        Ok(staked_soma)
    }

    /// Reveal a pending model (Phase 2 of commit-reveal).
    /// Moves model from pending to active. Provides the decryption key and full
    /// embedding (verified against the embedding_commitment from commit).
    #[allow(clippy::result_large_err)]
    pub fn request_reveal_model(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
        decryption_key: DecryptionKey,
        embedding: SomaTensor,
    ) -> ExecutionResult {
        let model = self
            .model_registry
            .pending_models
            .get(model_id)
            .ok_or(ExecutionFailureStatus::ModelNotPending)?;

        if model.owner != signer {
            return Err(ExecutionFailureStatus::NotModelOwner);
        }
        if self.epoch != model.commit_epoch + 1 {
            return Err(ExecutionFailureStatus::ModelRevealEpochMismatch);
        }

        // Verify decryption key commitment: hash(key_bytes) must match
        let dk_hash = {
            let mut hasher = DefaultHash::default();
            hasher.update(decryption_key.as_ref());
            hasher.finalize()
        };
        let expected_dk: [u8; 32] = model.decryption_key_commitment.into();
        if dk_hash.as_ref() != expected_dk {
            return Err(ExecutionFailureStatus::ModelDecryptionKeyCommitmentMismatch);
        }

        // Verify embedding commitment: hash(bcs(embedding)) must match
        let embedding_bytes = bcs::to_bytes(&embedding).expect("BCS serialization cannot fail");
        let embedding_hash = {
            let mut hasher = DefaultHash::default();
            hasher.update(&embedding_bytes);
            hasher.finalize()
        };
        let expected: [u8; 32] = model.embedding_commitment.into();
        if embedding_hash.as_ref() != expected {
            return Err(ExecutionFailureStatus::ModelEmbeddingCommitmentMismatch);
        }

        // Move from pending to active
        let mut model = self.model_registry.pending_models.remove(model_id).unwrap();
        model.decryption_key = Some(decryption_key);
        model.embedding = Some(embedding);
        model.staking_pool.activation_epoch = Some(self.epoch);

        // Set initial exchange rate at activation
        model.staking_pool.exchange_rates.insert(
            self.epoch,
            staking::PoolTokenExchangeRate {
                soma_amount: model.staking_pool.soma_balance,
                pool_token_amount: model.staking_pool.pool_token_balance,
            },
        );

        self.model_registry.total_model_stake += model.staking_pool.soma_balance;
        self.model_registry.active_models.insert(*model_id, model);

        Ok(())
    }

    /// Commit an update to an active model's weights.
    #[allow(clippy::result_large_err)]
    pub fn request_commit_model_update(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
        manifest: Manifest,
        weights_commitment: ModelWeightsCommitment,
        embedding_commitment: EmbeddingCommitment,
        decryption_key_commitment: DecryptionKeyCommitment,
    ) -> ExecutionResult {
        let model = self
            .model_registry
            .active_models
            .get_mut(model_id)
            .ok_or(ExecutionFailureStatus::ModelNotActive)?;

        if model.owner != signer {
            return Err(ExecutionFailureStatus::NotModelOwner);
        }

        // Overwrite any existing pending update for this epoch
        model.pending_update = Some(PendingModelUpdate {
            manifest,
            weights_commitment,
            embedding_commitment,
            decryption_key_commitment,
            commit_epoch: self.epoch,
        });

        Ok(())
    }

    /// Reveal a pending model update.
    /// Provides the decryption key and full embedding (verified against commitment).
    #[allow(clippy::result_large_err)]
    pub fn request_reveal_model_update(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
        decryption_key: DecryptionKey,
        embedding: SomaTensor,
    ) -> ExecutionResult {
        let model = self
            .model_registry
            .active_models
            .get_mut(model_id)
            .ok_or(ExecutionFailureStatus::ModelNotActive)?;

        if model.owner != signer {
            return Err(ExecutionFailureStatus::NotModelOwner);
        }

        let pending =
            model.pending_update.as_ref().ok_or(ExecutionFailureStatus::ModelNoPendingUpdate)?;

        if self.epoch != pending.commit_epoch + 1 {
            return Err(ExecutionFailureStatus::ModelRevealEpochMismatch);
        }

        // Verify decryption key commitment
        let dk_hash = {
            let mut hasher = DefaultHash::default();
            hasher.update(decryption_key.as_ref());
            hasher.finalize()
        };
        let expected_dk: [u8; 32] = pending.decryption_key_commitment.into();
        if dk_hash.as_ref() != expected_dk {
            return Err(ExecutionFailureStatus::ModelDecryptionKeyCommitmentMismatch);
        }

        // Verify embedding commitment
        let embedding_bytes = bcs::to_bytes(&embedding).expect("BCS serialization cannot fail");
        let embedding_hash = {
            let mut hasher = DefaultHash::default();
            hasher.update(&embedding_bytes);
            hasher.finalize()
        };
        let expected: [u8; 32] = pending.embedding_commitment.into();
        if embedding_hash.as_ref() != expected {
            return Err(ExecutionFailureStatus::ModelEmbeddingCommitmentMismatch);
        }

        // Apply the update
        let pending = model.pending_update.take().unwrap();
        model.manifest = pending.manifest;
        model.weights_commitment = pending.weights_commitment;
        model.embedding_commitment = pending.embedding_commitment;
        model.decryption_key_commitment = pending.decryption_key_commitment;
        model.embedding = Some(embedding);
        model.decryption_key = Some(decryption_key);

        Ok(())
    }

    /// Add stake to a model (any sender).
    #[allow(clippy::result_large_err)]
    pub fn request_add_stake_to_model(
        &mut self,
        model_id: &ModelId,
        amount: u64,
    ) -> ExecutionResult<StakedSomaV1> {
        if amount == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Stake amount cannot be 0!".to_string(),
            });
        }

        let current_epoch = self.epoch;

        // Look up the model directly in active or pending registries
        let (is_active, model) =
            if let Some(m) = self.model_registry.active_models.get_mut(model_id) {
                (true, m)
            } else if let Some(m) = self.model_registry.pending_models.get_mut(model_id) {
                (false, m)
            } else {
                return Err(ExecutionFailureStatus::ModelNotFound);
            };

        if model.is_inactive() {
            return Err(ExecutionFailureStatus::ModelAlreadyInactive);
        }

        let stake_activation_epoch = current_epoch + 1;
        let staked_soma = model.staking_pool.request_add_stake(amount, stake_activation_epoch);

        // If pool is preactive, process stake immediately
        if model.staking_pool.is_preactive() {
            model.staking_pool.process_pending_stake();
        }

        let pool_id = staked_soma.pool_id;

        // Update total model stake if model is active
        if is_active {
            self.model_registry.total_model_stake += amount;
        }

        // Ensure staking pool mapping exists
        self.model_registry.staking_pool_mappings.insert(pool_id, *model_id);

        Ok(staked_soma)
    }

    /// Set model commission rate (staged for next epoch).
    #[allow(clippy::result_large_err)]
    pub fn request_set_model_commission_rate(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
        new_rate: u64,
    ) -> ExecutionResult {
        if new_rate > BPS_DENOMINATOR {
            return Err(ExecutionFailureStatus::ModelCommissionRateTooHigh);
        }

        let model = self
            .model_registry
            .active_models
            .get_mut(model_id)
            .ok_or(ExecutionFailureStatus::ModelNotActive)?;

        if model.owner != signer {
            return Err(ExecutionFailureStatus::NotModelOwner);
        }

        model.next_epoch_commission_rate = new_rate;
        Ok(())
    }

    /// Deactivate a model (owner voluntary withdrawal).
    #[allow(clippy::result_large_err)]
    pub fn request_deactivate_model(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
    ) -> ExecutionResult {
        let model = self
            .model_registry
            .active_models
            .get(model_id)
            .ok_or(ExecutionFailureStatus::ModelNotActive)?;

        if model.owner != signer {
            return Err(ExecutionFailureStatus::NotModelOwner);
        }

        let mut model = self.model_registry.active_models.remove(model_id).unwrap();

        self.model_registry.total_model_stake =
            self.model_registry.total_model_stake.saturating_sub(model.staking_pool.soma_balance);

        model.staking_pool.deactivation_epoch = Some(self.epoch);
        self.model_registry.inactive_models.insert(*model_id, model);

        // Clean up report records for this model
        self.model_registry.model_report_records.remove(model_id);

        Ok(())
    }

    /// Report a model for unavailability (sender must be active validator).
    #[allow(clippy::result_large_err)]
    pub fn report_model(&mut self, reporter: SomaAddress, model_id: &ModelId) -> ExecutionResult {
        if !self.validators.is_active_validator(reporter) {
            return Err(ExecutionFailureStatus::NotAValidator);
        }

        if !self.model_registry.active_models.contains_key(model_id) {
            return Err(ExecutionFailureStatus::ModelNotActive);
        }

        self.model_registry.model_report_records.entry(*model_id).or_default().insert(reporter);

        Ok(())
    }

    /// Undo a model report (sender must be an active validator and in the report set).
    #[allow(clippy::result_large_err)]
    pub fn undo_report_model(
        &mut self,
        reporter: SomaAddress,
        model_id: &ModelId,
    ) -> ExecutionResult {
        // Validate reporter is an active validator
        if !self.validators.is_active_validator(reporter) {
            return Err(ExecutionFailureStatus::NotAValidator);
        }

        // Validate model is still active (can only undo reports on active models)
        if !self.model_registry.active_models.contains_key(model_id) {
            return Err(ExecutionFailureStatus::ModelNotActive);
        }

        let reports = self
            .model_registry
            .model_report_records
            .get_mut(model_id)
            .ok_or(ExecutionFailureStatus::ReportRecordNotFound)?;

        if !reports.remove(&reporter) {
            return Err(ExecutionFailureStatus::ReportRecordNotFound);
        }

        if reports.is_empty() {
            self.model_registry.model_report_records.remove(model_id);
        }

        Ok(())
    }

    /// Process model registry at epoch boundary.
    ///
    /// Called from `advance_epoch` after validator processing. Performs:
    /// 1. Report processing: 2f+1 quorum → slash at `model_tally_slash_rate_bps`, move to inactive
    /// 2. Pending reveal timeout: slash unrevealed models at `model_reveal_slash_rate_bps`, move to inactive
    /// 3. Pending update cancellation: clear unrevealed updates (no slash)
    /// 4. Commission rate adjustment: `commission_rate = next_epoch_commission_rate`
    /// 5. Staking pool processing: `process_pending_stakes_and_withdraws(new_epoch)`
    fn advance_epoch_models(&mut self, new_epoch: u64) {
        let prev_epoch = new_epoch - 1;
        let tally_slash_rate = self.parameters.model_tally_slash_rate_bps;
        let reveal_slash_rate = self.parameters.model_reveal_slash_rate_bps;

        // --- Step 1: Process model report records (2f+1 quorum slash) ---
        // Mirrors validator report processing: compute_slashed_validators pattern.
        let quorum_threshold = crate::committee::QUORUM_THRESHOLD;
        let mut slashed_model_ids: Vec<ModelId> = Vec::new();

        for (model_id, reporters) in &self.model_registry.model_report_records {
            if self.model_registry.active_models.contains_key(model_id) {
                let reporter_votes = self
                    .validators
                    .sum_voting_power_by_addresses(&reporters.iter().cloned().collect());
                if reporter_votes >= quorum_threshold {
                    slashed_model_ids.push(*model_id);
                }
            }
        }

        for model_id in &slashed_model_ids {
            if let Some(mut model) = self.model_registry.active_models.remove(model_id) {
                // Save original balance before slashing (for accurate total_model_stake deduction)
                let original_balance = model.staking_pool.soma_balance;

                // Slash stake: reduce soma_balance by tally_slash_rate_bps
                let slash_amount = (original_balance as u128 * tally_slash_rate as u128
                    / BPS_DENOMINATOR as u128) as u64;
                model.staking_pool.soma_balance = original_balance.saturating_sub(slash_amount);

                // Remove the model's full original stake from the total
                self.model_registry.total_model_stake =
                    self.model_registry.total_model_stake.saturating_sub(original_balance);

                model.staking_pool.deactivation_epoch = Some(new_epoch);
                self.model_registry.inactive_models.insert(*model_id, model);

                info!(
                    "Model {:?} slashed (tally quorum): {} shannons at {}bps",
                    model_id, slash_amount, tally_slash_rate
                );
            }
        }

        // Clear all report records (mirrors validator pattern)
        self.model_registry.model_report_records.clear();

        // --- Step 2: Process pending reveal timeouts ---
        // Models committed in epoch N must be revealed by the end of epoch N+1.
        // At the boundary entering new_epoch, any model with commit_epoch < prev_epoch
        // has missed its reveal window.
        let mut unrevealed_ids: Vec<ModelId> = Vec::new();
        for (model_id, model) in &self.model_registry.pending_models {
            // The reveal must happen during (commit_epoch + 1). We are now transitioning
            // to new_epoch, so the previous epoch (prev_epoch) just ended. If the model
            // was committed in an epoch <= prev_epoch - 1, the reveal window has passed.
            if model.commit_epoch < prev_epoch {
                unrevealed_ids.push(*model_id);
            }
        }

        for model_id in &unrevealed_ids {
            if let Some(mut model) = self.model_registry.pending_models.remove(model_id) {
                // Slash stake at reveal_slash_rate_bps
                let slash_amount = (model.staking_pool.soma_balance as u128
                    * reveal_slash_rate as u128
                    / BPS_DENOMINATOR as u128) as u64;
                model.staking_pool.soma_balance =
                    model.staking_pool.soma_balance.saturating_sub(slash_amount);

                model.staking_pool.deactivation_epoch = Some(new_epoch);
                self.model_registry.inactive_models.insert(*model_id, model);

                info!(
                    "Model {:?} slashed (unrevealed): {} shannons at {}bps",
                    model_id, slash_amount, reveal_slash_rate
                );
            }
        }

        // --- Step 3: Process pending update cancellations ---
        // Updates committed in epoch N must be revealed by the end of epoch N+1.
        // Cancel any expired pending updates (no slash).
        for model in self.model_registry.active_models.values_mut() {
            if let Some(pending) = &model.pending_update {
                if pending.commit_epoch < prev_epoch {
                    info!(
                        "Model pending update cancelled (unrevealed, committed epoch {})",
                        pending.commit_epoch
                    );
                    model.pending_update = None;
                }
            }
        }

        // --- Step 4: Adjust commission rates ---
        for model in self.model_registry.active_models.values_mut() {
            model.commission_rate = model.next_epoch_commission_rate;
        }

        // --- Step 5: Process model staking pools ---
        for model in self.model_registry.active_models.values_mut() {
            model.staking_pool.process_pending_stakes_and_withdraws(new_epoch);
        }

        // Also process pending model pools (they may have accumulated stake)
        for model in self.model_registry.pending_models.values_mut() {
            model.staking_pool.process_pending_stake_withdraw();
            model.staking_pool.process_pending_stake();
        }

        // Recompute total_model_stake from active models
        self.model_registry.total_model_stake =
            self.model_registry.active_models.values().map(|m| m.staking_pool.soma_balance).sum();
    }

    #[allow(clippy::result_large_err)]
    pub fn advance_epoch(
        &mut self,
        new_epoch: u64,
        next_protocol_config: &ProtocolConfig,
        epoch_total_transaction_fees: u64,
        epoch_start_timestamp_ms: u64,
        epoch_randomness: Vec<u8>,
    ) -> ExecutionResult<BTreeMap<SomaAddress, StakedSomaV1>> {
        // 1. Verify we're advancing to the correct epoch
        if new_epoch != self.epoch + 1 {
            return Err(ExecutionFailureStatus::AdvancedToWrongEpoch);
        }

        // Drain safe mode accumulators if recovering from safe mode
        let mut safe_mode_extra_rewards = 0u64;
        if self.safe_mode {
            info!(
                "Recovering from safe mode — draining accumulated rewards: fees={}, emissions={}",
                self.safe_mode_accumulated_fees, self.safe_mode_accumulated_emissions
            );
            safe_mode_extra_rewards = self
                .safe_mode_accumulated_fees
                .saturating_add(self.safe_mode_accumulated_emissions);
            self.safe_mode_accumulated_fees = 0;
            self.safe_mode_accumulated_emissions = 0;
            self.safe_mode = false;
        }

        let prev_epoch = self.epoch;
        let prev_epoch_start_timestamp = self.epoch_start_timestamp_ms;
        self.epoch_start_timestamp_ms = epoch_start_timestamp_ms;

        let next_protocol_version = next_protocol_config.version.as_u64();

        // Check if protocol version is changing
        if next_protocol_version != self.protocol_version {
            info!("Protocol upgrade: {} -> {}", self.protocol_version, next_protocol_version);

            // Update parameters from new protocol config
            // Preserve current value_fee_bps since it's dynamically adjusted
            self.parameters =
                next_protocol_config.build_system_parameters(Some(self.parameters.value_fee_bps));
        }

        // Get reward_slashing_rate from protocol config
        let reward_slashing_rate = next_protocol_config.reward_slashing_rate_bps();

        // 2. Calculate total rewards (emissions + fees + safe mode accumulators)
        let mut total_rewards =
            epoch_total_transaction_fees.saturating_add(safe_mode_extra_rewards);
        if epoch_start_timestamp_ms
            >= prev_epoch_start_timestamp + self.parameters.epoch_duration_ms
        {
            total_rewards = total_rewards.saturating_add(self.emission_pool.advance_epoch());
        }
        // Adjust fees for next epoch BEFORE processing rewards
        self.adjust_value_fee(epoch_total_transaction_fees);

        // 3. Increment epoch
        self.epoch = new_epoch;

        // 4. Allocate rewards: validators get their share, remainder funds target pool
        // Note: Target rewards are pre-allocated at target creation time from emission pool,
        // so we only allocate validator rewards here.
        let validator_allocation_bps = self.parameters.validator_reward_allocation_bps;
        // Use u128 intermediate to avoid overflow
        let validator_allocation = (total_rewards as u128 * validator_allocation_bps as u128
            / BPS_DENOMINATOR as u128) as u64;
        let remainder = total_rewards - validator_allocation;
        // Target rewards are pre-allocated from the emission pool at target creation time,
        // so the non-validator portion of epoch rewards is returned to the emission pool
        // to maintain supply conservation.
        self.emission_pool.balance = self.emission_pool.balance.saturating_add(remainder);

        // 5. Advance target state for new epoch (difficulty adjustment + reward calculation)
        self.advance_epoch_targets();

        // 9. Process validator rewards (minimal - just for consensus participation)
        let mut validator_reward_pool = validator_allocation;
        let validator_rewards = self.validators.advance_epoch(
            new_epoch,
            &mut validator_reward_pool,
            reward_slashing_rate, // 50% of tallied rewards get slashed and redistributed to other validators
            &mut self.validator_report_records,
            VALIDATOR_LOW_STAKE_GRACE_PERIOD,
        );

        // 10. Process model registry epoch boundary logic
        self.advance_epoch_models(new_epoch);

        self.protocol_version = next_protocol_version;

        // 12. Return remainder to emission pool
        if validator_reward_pool > 0 {
            self.emission_pool.balance =
                self.emission_pool.balance.saturating_add(validator_reward_pool);
        }

        Ok(validator_rewards)
    }

    /// Minimal epoch transition when normal advance_epoch fails.
    /// This is the safe mode fallback — it cannot fail because it performs
    /// no complex math, no loops, and no external calls.
    ///
    /// It only:
    /// - Sets `safe_mode = true`
    /// - Increments epoch + timestamp
    /// - Accumulates fees and emissions into holding fields
    ///
    /// Everything else (validator rewards, model processing, target generation,
    /// difficulty adjustment) is skipped. The committee, parameters, and all
    /// registries remain frozen until a successful `advance_epoch()` recovers.
    pub fn advance_epoch_safe_mode(
        &mut self,
        new_epoch: u64,
        epoch_total_transaction_fees: u64,
        epoch_start_timestamp_ms: u64,
    ) {
        self.safe_mode = true;
        self.epoch = new_epoch;
        self.epoch_start_timestamp_ms = epoch_start_timestamp_ms;

        // Accumulate fees — will be distributed on recovery
        self.safe_mode_accumulated_fees =
            self.safe_mode_accumulated_fees.saturating_add(epoch_total_transaction_fees);

        // Accumulate emissions if the emission pool has balance
        let emission =
            std::cmp::min(self.emission_pool.emission_per_epoch, self.emission_pool.balance);
        self.emission_pool.balance = self.emission_pool.balance.saturating_sub(emission);
        self.safe_mode_accumulated_emissions =
            self.safe_mode_accumulated_emissions.saturating_add(emission);

        // No validator rewards, no model processing, no target generation,
        // no difficulty adjustment, no staking pool processing.
        // The committee, parameters, and all registries remain frozen.

        info!(
            "Safe mode activated for epoch {}. Accumulated fees: {}, accumulated emissions: {}",
            new_epoch, self.safe_mode_accumulated_fees, self.safe_mode_accumulated_emissions
        );
    }

    /// Return funds to the emissions pool (used when targets expire unfilled)
    pub fn return_to_emissions_pool(&mut self, amount: u64) {
        self.emission_pool.balance += amount;
    }

    /// Calculate the reward per target for the upcoming epoch.
    /// Based on target_reward_allocation_bps of epoch emissions divided by estimated targets.
    ///
    /// Uses target_initial_targets_per_epoch as the estimate for number of targets.
    pub fn calculate_reward_per_target(&self) -> u64 {
        let epoch_emissions = self.emission_pool.emission_per_epoch;
        let target_allocation_bps = self.parameters.target_reward_allocation_bps;
        // Use u128 intermediate to avoid overflow when epoch_emissions is large
        let target_emissions = (epoch_emissions as u128 * target_allocation_bps as u128
            / BPS_DENOMINATOR as u128) as u64;

        // Use initial targets per epoch as the estimate
        let estimated_targets = self.parameters.target_initial_targets_per_epoch.max(1);

        target_emissions / estimated_targets
    }

    /// Adjust difficulty thresholds based on hits per epoch.
    /// Called during epoch transition in advance_epoch_targets().
    ///
    /// Compares the EMA of absolute hit counts against target_hits_per_epoch:
    /// If ema_hits > target: too many hits → decrease thresholds (harder)
    /// If ema_hits < target: too few hits → increase thresholds (easier)
    ///
    /// Uses an EMA of hits per epoch for smoother adjustments.
    pub fn adjust_difficulty(&mut self) {
        let target_hits = self.parameters.target_hits_per_epoch;
        let decay_bps = self.parameters.target_hits_ema_decay_bps;

        // Update the EMA with this epoch's hit count
        let ema_hits = self.target_state.update_hits_ema(decay_bps);

        // Skip adjustment if still in bootstrap mode (EMA is 0)
        if ema_hits == 0 {
            info!("Difficulty adjustment skipped: bootstrap mode (no hit data yet)");
            return;
        }

        let adjustment_rate = self.parameters.target_difficulty_adjustment_rate_bps;
        let min_distance = self.parameters.target_min_distance_threshold.as_scalar();
        let max_distance = self.parameters.target_max_distance_threshold.as_scalar();

        // Calculate adjustment factor based on EMA vs target
        // If ema > target, we're too easy → decrease thresholds (harder)
        // If ema < target, we're too hard → increase thresholds (easier)
        // If ema == target, no adjustment needed
        let adjustment_factor: f32 = if ema_hits > target_hits {
            // Too easy - make harder (decrease thresholds)
            // factor < 1.0
            (BPS_DENOMINATOR - adjustment_rate).min(BPS_DENOMINATOR) as f32 / BPS_DENOMINATOR as f32
        } else if ema_hits < target_hits {
            // Too hard - make easier (increase thresholds)
            // factor > 1.0
            (BPS_DENOMINATOR + adjustment_rate) as f32 / BPS_DENOMINATOR as f32
        } else {
            1.0 // Exact match, no adjustment
        };

        // Apply adjustment to distance threshold
        let current_distance = self.target_state.distance_threshold.as_scalar();
        let new_distance = (current_distance * adjustment_factor).clamp(min_distance, max_distance);
        self.target_state.distance_threshold = SomaTensor::scalar(new_distance);

        info!(
            "Difficulty adjusted: distance={} (ema_hits={}, target_hits={}, hits_this_epoch={})",
            self.target_state.distance_threshold,
            ema_hits,
            target_hits,
            self.target_state.hits_this_epoch,
        );
    }

    /// Called at epoch boundary to update target state for the new epoch.
    ///
    /// 1. Adjust difficulty based on hits from the previous epoch
    /// 2. Reset epoch counters for the new epoch
    /// 3. Calculate reward_per_target for the new epoch
    ///
    /// Note: Actual target objects are separate shared objects.
    /// This only updates the coordination state in SystemState.
    pub fn advance_epoch_targets(&mut self) {
        // 1. Adjust difficulty thresholds based on last epoch's hits
        self.adjust_difficulty();

        // 2. Reset epoch counters for the new epoch
        self.target_state.reset_epoch_counters();

        // 3. Calculate and set reward_per_target for new epoch
        self.target_state.reward_per_target = self.calculate_reward_per_target();

        info!(
            "Target state advanced: reward_per_target={}, distance_threshold={}",
            self.target_state.reward_per_target, self.target_state.distance_threshold
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
            std::cmp::min(current_bps.saturating_add(increase), self.parameters.max_value_fee_bps)
        } else {
            // Under target - decrease fees to encourage activity
            let deficit_ratio = BPS_DENOMINATOR - ratio;
            let decrease = std::cmp::min(
                (current_bps * deficit_ratio) / BPS_DENOMINATOR,
                (current_bps * adjustment_rate) / BPS_DENOMINATOR,
            );
            std::cmp::max(current_bps.saturating_sub(decrease), self.parameters.min_value_fee_bps)
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

impl SystemStateTrait for SystemStateV1 {
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
        EpochStartSystemState::V1(epoch_start::EpochStartSystemStateV1 {
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
                    EpochStartValidatorInfoV1 {
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
        })
    }
}

#[allow(clippy::too_many_arguments, clippy::result_large_err)]
impl SystemState {
    // --- Constructor ---

    pub fn create(
        validators: Vec<Validator>,
        protocol_version: u64,
        epoch_start_timestamp_ms: u64,
        protocol_config: &ProtocolConfig,
        emission_fund: u64,
        emission_per_epoch: u64,
        epoch_duration_ms_override: Option<u64>,
    ) -> Self {
        Self::V1(SystemStateV1::create(
            validators,
            protocol_version,
            epoch_start_timestamp_ms,
            protocol_config,
            emission_fund,
            emission_per_epoch,
            epoch_duration_ms_override,
        ))
    }

    /// Backward-compatible deserialization: try versioned format first, fall back to raw V1.
    pub fn deserialize(contents: &[u8]) -> Result<Self, bcs::Error> {
        bcs::from_bytes::<Self>(contents)
            .or_else(|_| bcs::from_bytes::<SystemStateV1>(contents).map(Self::V1))
    }

    // --- Field accessors ---

    pub fn parameters(&self) -> &SystemParameters {
        match self {
            Self::V1(v1) => &v1.parameters,
        }
    }
    pub fn parameters_mut(&mut self) -> &mut SystemParameters {
        match self {
            Self::V1(v1) => &mut v1.parameters,
        }
    }
    pub fn validators(&self) -> &ValidatorSet {
        match self {
            Self::V1(v1) => &v1.validators,
        }
    }
    pub fn validators_mut(&mut self) -> &mut ValidatorSet {
        match self {
            Self::V1(v1) => &mut v1.validators,
        }
    }
    pub fn model_registry(&self) -> &ModelRegistry {
        match self {
            Self::V1(v1) => &v1.model_registry,
        }
    }
    pub fn model_registry_mut(&mut self) -> &mut ModelRegistry {
        match self {
            Self::V1(v1) => &mut v1.model_registry,
        }
    }
    pub fn target_state(&self) -> &TargetState {
        match self {
            Self::V1(v1) => &v1.target_state,
        }
    }
    pub fn target_state_mut(&mut self) -> &mut TargetState {
        match self {
            Self::V1(v1) => &mut v1.target_state,
        }
    }
    pub fn emission_pool(&self) -> &EmissionPool {
        match self {
            Self::V1(v1) => &v1.emission_pool,
        }
    }
    pub fn emission_pool_mut(&mut self) -> &mut EmissionPool {
        match self {
            Self::V1(v1) => &mut v1.emission_pool,
        }
    }
    pub fn safe_mode(&self) -> bool {
        match self {
            Self::V1(v1) => v1.safe_mode,
        }
    }
    pub fn safe_mode_accumulated_fees(&self) -> u64 {
        match self {
            Self::V1(v1) => v1.safe_mode_accumulated_fees,
        }
    }
    pub fn safe_mode_accumulated_emissions(&self) -> u64 {
        match self {
            Self::V1(v1) => v1.safe_mode_accumulated_emissions,
        }
    }
    pub fn validator_report_records(&self) -> &BTreeMap<SomaAddress, BTreeSet<SomaAddress>> {
        match self {
            Self::V1(v1) => &v1.validator_report_records,
        }
    }

    // --- Forwarding for all public non-trait methods ---

    pub fn request_add_validator(
        &mut self,
        signer: SomaAddress,
        pubkey_bytes: Vec<u8>,
        network_pubkey_bytes: Vec<u8>,
        worker_pubkey_bytes: Vec<u8>,
        proof_of_possession_bytes: Vec<u8>,
        net_address: Vec<u8>,
        p2p_address: Vec<u8>,
        primary_address: Vec<u8>,
        proxy_address: Vec<u8>,
        staking_pool_id: ObjectID,
    ) -> ExecutionResult {
        match self {
            Self::V1(v1) => v1.request_add_validator(
                signer,
                pubkey_bytes,
                network_pubkey_bytes,
                worker_pubkey_bytes,
                proof_of_possession_bytes,
                net_address,
                p2p_address,
                primary_address,
                proxy_address,
                staking_pool_id,
            ),
        }
    }

    pub fn request_remove_validator(
        &mut self,
        signer: SomaAddress,
        pubkey_bytes: Vec<u8>,
    ) -> ExecutionResult {
        match self {
            Self::V1(v1) => v1.request_remove_validator(signer, pubkey_bytes),
        }
    }

    pub fn request_update_validator_metadata(
        &mut self,
        signer: SomaAddress,
        args: &UpdateValidatorMetadataArgs,
    ) -> ExecutionResult<()> {
        match self {
            Self::V1(v1) => v1.request_update_validator_metadata(signer, args),
        }
    }

    pub fn request_add_stake(
        &mut self,
        signer: SomaAddress,
        address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<StakedSomaV1> {
        match self {
            Self::V1(v1) => v1.request_add_stake(signer, address, amount),
        }
    }

    pub fn request_add_stake_at_genesis(
        &mut self,
        signer: SomaAddress,
        address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<StakedSomaV1> {
        match self {
            Self::V1(v1) => v1.request_add_stake_at_genesis(signer, address, amount),
        }
    }

    pub fn add_model_at_genesis(
        &mut self,
        model_id: ModelId,
        owner: SomaAddress,
        manifest: Manifest,
        decryption_key: DecryptionKey,
        weights_commitment: ModelWeightsCommitment,
        architecture_version: ArchitectureVersion,
        embedding_commitment: EmbeddingCommitment,
        decryption_key_commitment: DecryptionKeyCommitment,
        embedding: SomaTensor,
        commission_rate: u64,
    ) {
        match self {
            Self::V1(v1) => v1.add_model_at_genesis(
                model_id,
                owner,
                manifest,
                decryption_key,
                weights_commitment,
                architecture_version,
                embedding_commitment,
                decryption_key_commitment,
                embedding,
                commission_rate,
            ),
        }
    }

    pub fn request_add_stake_to_model_at_genesis(
        &mut self,
        model_id: &ModelId,
        amount: u64,
    ) -> ExecutionResult<StakedSomaV1> {
        match self {
            Self::V1(v1) => v1.request_add_stake_to_model_at_genesis(model_id, amount),
        }
    }

    pub fn request_withdraw_stake(&mut self, staked_soma: StakedSomaV1) -> ExecutionResult<u64> {
        match self {
            Self::V1(v1) => v1.request_withdraw_stake(staked_soma),
        }
    }

    pub fn report_validator(
        &mut self,
        reporter: SomaAddress,
        reportee: SomaAddress,
    ) -> ExecutionResult {
        match self {
            Self::V1(v1) => v1.report_validator(reporter, reportee),
        }
    }

    pub fn undo_report_validator(
        &mut self,
        reporter: SomaAddress,
        reportee: SomaAddress,
    ) -> ExecutionResult {
        match self {
            Self::V1(v1) => v1.undo_report_validator(reporter, reportee),
        }
    }

    pub fn request_set_commission_rate(
        &mut self,
        signer: SomaAddress,
        new_rate: u64,
    ) -> Result<(), ExecutionFailureStatus> {
        match self {
            Self::V1(v1) => v1.request_set_commission_rate(signer, new_rate),
        }
    }

    pub fn find_model(&self, model_id: &ModelId) -> Option<&ModelV1> {
        match self {
            Self::V1(v1) => v1.find_model(model_id),
        }
    }

    pub fn find_model_mut(&mut self, model_id: &ModelId) -> Option<&mut ModelV1> {
        match self {
            Self::V1(v1) => v1.find_model_mut(model_id),
        }
    }

    pub fn request_commit_model(
        &mut self,
        owner: SomaAddress,
        model_id: ModelId,
        manifest: Manifest,
        weights_commitment: ModelWeightsCommitment,
        architecture_version: ArchitectureVersion,
        embedding_commitment: EmbeddingCommitment,
        decryption_key_commitment: DecryptionKeyCommitment,
        stake_amount: u64,
        commission_rate: u64,
        staking_pool_id: ObjectID,
    ) -> ExecutionResult<StakedSomaV1> {
        match self {
            Self::V1(v1) => v1.request_commit_model(
                owner,
                model_id,
                manifest,
                weights_commitment,
                architecture_version,
                embedding_commitment,
                decryption_key_commitment,
                stake_amount,
                commission_rate,
                staking_pool_id,
            ),
        }
    }

    pub fn request_reveal_model(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
        decryption_key: DecryptionKey,
        embedding: SomaTensor,
    ) -> ExecutionResult {
        match self {
            Self::V1(v1) => {
                v1.request_reveal_model(signer, model_id, decryption_key, embedding)
            }
        }
    }

    pub fn request_commit_model_update(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
        manifest: Manifest,
        weights_commitment: ModelWeightsCommitment,
        embedding_commitment: EmbeddingCommitment,
        decryption_key_commitment: DecryptionKeyCommitment,
    ) -> ExecutionResult {
        match self {
            Self::V1(v1) => v1.request_commit_model_update(
                signer,
                model_id,
                manifest,
                weights_commitment,
                embedding_commitment,
                decryption_key_commitment,
            ),
        }
    }

    pub fn request_reveal_model_update(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
        decryption_key: DecryptionKey,
        embedding: SomaTensor,
    ) -> ExecutionResult {
        match self {
            Self::V1(v1) => {
                v1.request_reveal_model_update(signer, model_id, decryption_key, embedding)
            }
        }
    }

    pub fn request_add_stake_to_model(
        &mut self,
        model_id: &ModelId,
        amount: u64,
    ) -> ExecutionResult<StakedSomaV1> {
        match self {
            Self::V1(v1) => v1.request_add_stake_to_model(model_id, amount),
        }
    }

    pub fn request_set_model_commission_rate(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
        new_rate: u64,
    ) -> ExecutionResult {
        match self {
            Self::V1(v1) => v1.request_set_model_commission_rate(signer, model_id, new_rate),
        }
    }

    pub fn request_deactivate_model(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
    ) -> ExecutionResult {
        match self {
            Self::V1(v1) => v1.request_deactivate_model(signer, model_id),
        }
    }

    pub fn report_model(&mut self, reporter: SomaAddress, model_id: &ModelId) -> ExecutionResult {
        match self {
            Self::V1(v1) => v1.report_model(reporter, model_id),
        }
    }

    pub fn undo_report_model(
        &mut self,
        reporter: SomaAddress,
        model_id: &ModelId,
    ) -> ExecutionResult {
        match self {
            Self::V1(v1) => v1.undo_report_model(reporter, model_id),
        }
    }

    pub fn advance_epoch(
        &mut self,
        new_epoch: u64,
        next_protocol_config: &ProtocolConfig,
        epoch_total_transaction_fees: u64,
        epoch_start_timestamp_ms: u64,
        epoch_randomness: Vec<u8>,
    ) -> ExecutionResult<BTreeMap<SomaAddress, StakedSomaV1>> {
        match self {
            Self::V1(v1) => v1.advance_epoch(
                new_epoch,
                next_protocol_config,
                epoch_total_transaction_fees,
                epoch_start_timestamp_ms,
                epoch_randomness,
            ),
        }
    }

    pub fn advance_epoch_safe_mode(
        &mut self,
        new_epoch: u64,
        epoch_total_transaction_fees: u64,
        epoch_start_timestamp_ms: u64,
    ) {
        match self {
            Self::V1(v1) => v1.advance_epoch_safe_mode(
                new_epoch,
                epoch_total_transaction_fees,
                epoch_start_timestamp_ms,
            ),
        }
    }

    pub fn return_to_emissions_pool(&mut self, amount: u64) {
        match self {
            Self::V1(v1) => v1.return_to_emissions_pool(amount),
        }
    }

    pub fn calculate_reward_per_target(&self) -> u64 {
        match self {
            Self::V1(v1) => v1.calculate_reward_per_target(),
        }
    }

    pub fn adjust_difficulty(&mut self) {
        match self {
            Self::V1(v1) => v1.adjust_difficulty(),
        }
    }

    pub fn advance_epoch_targets(&mut self) {
        match self {
            Self::V1(v1) => v1.advance_epoch_targets(),
        }
    }

    pub fn fee_parameters(&self) -> FeeParameters {
        match self {
            Self::V1(v1) => v1.fee_parameters(),
        }
    }
}

pub fn get_system_state(object_store: &dyn ObjectStore) -> Result<SystemState, SomaError> {
    let object = object_store.get_object(&SYSTEM_STATE_OBJECT_ID).ok_or_else(|| {
        SomaError::SystemStateReadError("SystemState object not found".to_owned())
    })?;

    SystemState::deserialize(object.as_inner().data.contents())
        .map_err(|err| SomaError::SystemStateReadError(err.to_string()))
}
