use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    str::FromStr,
};

use crate::{checksum::Checksum, crypto::DefaultHash, metadata::ManifestAPI as _};
use emission::EmissionPool;
use epoch_start::{EpochStartSystemState, EpochStartValidatorInfo};
use fastcrypto::{
    bls12381::{self, min_sig::BLS12381PublicKey},
    ed25519::Ed25519PublicKey,
    hash::HashFunction as _,
    traits::ToFromBytes,
};
use model_registry::ModelRegistry;
use protocol_config::{ProtocolConfig, SomaTensor, SystemParameters};
use serde::{Deserialize, Serialize};
use staking::{StakedSoma, StakingPool};
use target_state::TargetState;
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
    crypto::{self, DecryptionKey, NetworkPublicKey, ProtocolPublicKey},
    digests::{ModelWeightsCommitment, ModelWeightsUrlCommitment},
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError, SomaResult},
    model::{ArchitectureVersion, Model, ModelId, ModelWeightsManifest, PendingModelUpdate},
    multiaddr::Multiaddr,
    object::ObjectID,
    parameters,
    peer_id::PeerId,
    transaction::UpdateValidatorMetadataArgs,
};
use crate::{
    crypto::{AuthorityPublicKey, SomaKeyPair, SomaPublicKey},
    storage::object_store::ObjectStore,
};

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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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

    /// Registry of all models (active, pending, inactive) in the mining system
    pub model_registry: ModelRegistry,

    pub emission_pool: EmissionPool,

    /// Lightweight coordination state for target generation and difficulty
    pub target_state: TargetState,
}

impl SystemState {
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
        let target_state = TargetState::new(
            parameters.target_initial_distance_threshold.clone(),
        );

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
        net_address: Vec<u8>,
        p2p_address: Vec<u8>,
        primary_address: Vec<u8>,
        proxy_address: Vec<u8>,
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
            Multiaddr::from_str(bcs::from_bytes(&proxy_address).unwrap()).unwrap(),
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
        let validator = self
            .validators
            .find_validator_mut(signer)
            // Ensure only active validators can stage changes for the next epoch
            .ok_or(ExecutionFailureStatus::NotAValidator)?;

        // Delegate the processing of optional fields to the validator
        validator.stage_next_epoch_metadata(args)
    }

    /// Request to add stake to a validator
    #[allow(clippy::result_large_err)]
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
        weights_manifest: ModelWeightsManifest,
        weights_url_commitment: ModelWeightsUrlCommitment,
        weights_commitment: ModelWeightsCommitment,
        architecture_version: ArchitectureVersion,
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

        let model = Model {
            owner,
            architecture_version,
            weights_url_commitment,
            weights_commitment,
            commit_epoch: 0,
            weights_manifest: Some(weights_manifest),
            embedding: None, // Genesis models start without embeddings; can be set via update
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
    ) -> ExecutionResult<StakedSoma> {
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
    /// Uses `StakedSoma.pool_id` to route to the correct pool via staking_pool_mappings.
    #[allow(clippy::result_large_err)]
    pub fn request_withdraw_stake(&mut self, staked_soma: StakedSoma) -> ExecutionResult<u64> {
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
        self.validator_report_records
            .entry(reportee)
            .or_default()
            .insert(reporter);

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
    pub fn find_model(&self, model_id: &ModelId) -> Option<&Model> {
        self.model_registry
            .active_models
            .get(model_id)
            .or_else(|| self.model_registry.pending_models.get(model_id))
    }

    /// Find a model by ID in active or pending registries (mutable).
    pub fn find_model_mut(&mut self, model_id: &ModelId) -> Option<&mut Model> {
        if self.model_registry.active_models.contains_key(model_id) {
            return self.model_registry.active_models.get_mut(model_id);
        }
        self.model_registry.pending_models.get_mut(model_id)
    }

    /// Commit a new model (Phase 1 of commit-reveal).
    /// Creates a pending model with a new StakingPool. Returns the StakedSoma receipt.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::result_large_err)]
    pub fn request_commit_model(
        &mut self,
        owner: SomaAddress,
        model_id: ModelId,
        weights_url_commitment: ModelWeightsUrlCommitment,
        weights_commitment: ModelWeightsCommitment,
        architecture_version: ArchitectureVersion,
        stake_amount: u64,
        commission_rate: u64,
        staking_pool_id: ObjectID,
    ) -> ExecutionResult<StakedSoma> {
        if commission_rate > BPS_DENOMINATOR {
            return Err(ExecutionFailureStatus::ModelCommissionRateTooHigh);
        }

        let mut staking_pool = StakingPool::new(staking_pool_id);
        let stake_activation_epoch = self.epoch + 1;
        let staked_soma = staking_pool.request_add_stake(stake_amount, stake_activation_epoch);
        // Pre-active pool: process stake immediately so soma_balance is set
        staking_pool.process_pending_stake();

        let model = Model {
            owner,
            architecture_version,
            weights_url_commitment,
            weights_commitment,
            commit_epoch: self.epoch,
            weights_manifest: None,
            embedding: None, // Set during reveal via request_reveal_model
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
    /// Moves model from pending to active, setting its embedding for KNN selection.
    #[allow(clippy::result_large_err)]
    pub fn request_reveal_model(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
        weights_manifest: ModelWeightsManifest,
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

        // Verify URL commitment: hash(url) must match weights_url_commitment
        let url_bytes = weights_manifest.manifest.url().as_str().as_bytes();
        let url_hash = {
            let mut hasher = DefaultHash::default();
            hasher.update(url_bytes);
            hasher.finalize()
        };
        let expected: [u8; 32] = model.weights_url_commitment.into();
        if url_hash.as_ref() != expected {
            return Err(ExecutionFailureStatus::ModelWeightsUrlMismatch);
        }

        // Move from pending to active
        let mut model = self.model_registry.pending_models.remove(model_id).unwrap();
        model.weights_manifest = Some(weights_manifest);
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
        weights_url_commitment: ModelWeightsUrlCommitment,
        weights_commitment: ModelWeightsCommitment,
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
            weights_url_commitment,
            weights_commitment,
            commit_epoch: self.epoch,
        });

        Ok(())
    }

    /// Reveal a pending model update.
    /// Also updates the model's embedding for KNN selection.
    #[allow(clippy::result_large_err)]
    pub fn request_reveal_model_update(
        &mut self,
        signer: SomaAddress,
        model_id: &ModelId,
        weights_manifest: ModelWeightsManifest,
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

        // Verify URL commitment
        let url_bytes = weights_manifest.manifest.url().as_str().as_bytes();
        let url_hash = {
            let mut hasher = DefaultHash::default();
            hasher.update(url_bytes);
            hasher.finalize()
        };
        let expected: [u8; 32] = pending.weights_url_commitment.into();
        if url_hash.as_ref() != expected {
            return Err(ExecutionFailureStatus::ModelWeightsUrlMismatch);
        }

        // Apply the update
        let pending = model.pending_update.take().unwrap();
        model.weights_url_commitment = pending.weights_url_commitment;
        model.weights_commitment = pending.weights_commitment;
        model.weights_manifest = Some(weights_manifest);
        model.embedding = Some(embedding);

        Ok(())
    }

    /// Add stake to a model (any sender).
    #[allow(clippy::result_large_err)]
    pub fn request_add_stake_to_model(
        &mut self,
        model_id: &ModelId,
        amount: u64,
    ) -> ExecutionResult<StakedSoma> {
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

        self.model_registry
            .model_report_records
            .entry(*model_id)
            .or_default()
            .insert(reporter);

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
                // Slash stake: reduce soma_balance by tally_slash_rate_bps
                let slash_amount = (model.staking_pool.soma_balance as u128
                    * tally_slash_rate as u128
                    / BPS_DENOMINATOR as u128) as u64;
                model.staking_pool.soma_balance =
                    model.staking_pool.soma_balance.saturating_sub(slash_amount);

                self.model_registry.total_model_stake = self
                    .model_registry
                    .total_model_stake
                    .saturating_sub(model.staking_pool.soma_balance + slash_amount);

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
            if let Some(pending) = &model.pending_update
                && pending.commit_epoch < prev_epoch
            {
                info!(
                    "Model pending update cancelled (unrevealed, committed epoch {})",
                    pending.commit_epoch
                );
                model.pending_update = None;
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
            info!("Protocol upgrade: {} -> {}", self.protocol_version, next_protocol_version);

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
        // Adjust fees for next epoch BEFORE processing rewards
        self.adjust_value_fee(epoch_total_transaction_fees);

        // 3. Increment epoch
        self.epoch = new_epoch;

        // 4. Allocate rewards: validators get their share, remainder funds target pool
        // Note: Target rewards are pre-allocated at target creation time from emission pool,
        // so we only allocate validator rewards here.
        let validator_allocation_bps = self.parameters.validator_reward_allocation_bps;
        let validator_allocation = (total_rewards * validator_allocation_bps) / BPS_DENOMINATOR;
        let _remainder = total_rewards - validator_allocation;
        // The remainder could be added back to emission_pool if desired, but for now
        // target rewards are funded directly from emission_pool at target creation time.

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
            self.emission_pool.balance += validator_reward_pool;
        }

        Ok(validator_rewards)
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
        let target_emissions = (epoch_emissions * target_allocation_bps) / BPS_DENOMINATOR;

        // Use initial targets per epoch as the estimate
        let estimated_targets = self.parameters.target_initial_targets_per_epoch.max(1);

        target_emissions / estimated_targets
    }

    /// Adjust difficulty thresholds based on hit rate for the epoch.
    /// Called during epoch transition in advance_epoch_targets().
    ///
    /// Hit rate = hits_this_epoch / targets_generated_this_epoch
    /// If hit_rate_ema > target_rate: make harder (decrease thresholds)
    /// If hit_rate_ema < target_rate: make easier (increase thresholds)
    ///
    /// Uses an EMA of hit rate across epochs for smoother adjustments.
    pub fn adjust_difficulty(&mut self) {
        let target_hit_rate_bps = self.parameters.target_hit_rate_target_bps;
        let decay_bps = self.parameters.target_hit_rate_ema_decay_bps;

        // Update the EMA with this epoch's hit rate
        let ema_bps = self.target_state.update_hit_rate_ema(decay_bps);

        // Skip adjustment if still in bootstrap mode (EMA is 0)
        if ema_bps == 0 {
            info!("Difficulty adjustment skipped: bootstrap mode (no hit data yet)");
            return;
        }

        let adjustment_rate = self.parameters.target_difficulty_adjustment_rate_bps;
        let min_distance = self.parameters.target_min_distance_threshold.as_scalar();
        let max_distance = self.parameters.target_max_distance_threshold.as_scalar();

        // Calculate adjustment factor based on EMA
        // If ema > target_rate, we're too easy → decrease thresholds (harder)
        // If ema < target_rate, we're too hard → increase thresholds (easier)
        let adjustment_factor: f32 = if ema_bps > target_hit_rate_bps {
            // Too easy - make harder (decrease thresholds)
            // factor < 1.0
            (BPS_DENOMINATOR - adjustment_rate).min(BPS_DENOMINATOR) as f32
                / BPS_DENOMINATOR as f32
        } else {
            // Too hard - make easier (increase thresholds)
            // factor > 1.0
            (BPS_DENOMINATOR + adjustment_rate) as f32 / BPS_DENOMINATOR as f32
        };

        // Apply adjustment to distance threshold
        let current_distance = self.target_state.distance_threshold.as_scalar();
        let new_distance = (current_distance * adjustment_factor).clamp(min_distance, max_distance);
        self.target_state.distance_threshold = SomaTensor::scalar(new_distance);

        info!(
            "Difficulty adjusted: distance={} (ema={}bps, target={}bps, hits={}, targets={})",
            self.target_state.distance_threshold,
            ema_bps,
            target_hit_rate_bps,
            self.target_state.hits_this_epoch,
            self.target_state.targets_generated_this_epoch
        );
    }

    /// Called at epoch boundary to update target state for the new epoch.
    ///
    /// 1. Adjust difficulty based on hit rate from the previous epoch
    /// 2. Reset epoch counters for the new epoch
    /// 3. Calculate reward_per_target for the new epoch
    ///
    /// Note: Actual target objects are separate shared objects.
    /// This only updates the coordination state in SystemState.
    pub fn advance_epoch_targets(&mut self) {
        // 1. Adjust difficulty thresholds based on last epoch's hit rate
        self.adjust_difficulty();

        // 2. Reset epoch counters for the new epoch
        self.target_state.reset_epoch_counters();

        // 3. Calculate and set reward_per_target for new epoch
        self.target_state.reward_per_target = self.calculate_reward_per_target();

        info!(
            "Target state advanced: reward_per_target={}, distance_threshold={}",
            self.target_state.reward_per_target,
            self.target_state.distance_threshold
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
