// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::str::FromStr;

use crate::bridge::{BridgeCommittee, BridgeState, MarketplaceParameters};
use emission::EmissionPool;
use enum_dispatch::enum_dispatch;
use epoch_start::{EpochStartSystemState, EpochStartValidatorInfoV1};
use fastcrypto::bls12381::min_sig::BLS12381PublicKey;
use fastcrypto::bls12381::{self};
use fastcrypto::ed25519::Ed25519PublicKey;
use fastcrypto::hash::{Blake2b256, HashFunction as _};
use fastcrypto::traits::ToFromBytes;
use protocol_config::{ProtocolConfig, SystemParameters};
use serde::{Deserialize, Serialize};
use staking::StakingPool;
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
    self, AuthorityPublicKey, DefaultHash, NetworkPublicKey, ProtocolPublicKey,
    SomaKeyPair, SomaPublicKey,
};
use crate::effects::ExecutionFailureStatus;
use crate::error::{ExecutionResult, SomaError, SomaResult};
use crate::multiaddr::Multiaddr;
use crate::object::ObjectID;
use crate::peer_id::PeerId;
use crate::storage::object_store::ObjectStore;
use crate::transaction::UpdateValidatorMetadataArgs;
use crate::{SYSTEM_STATE_OBJECT_ID, parameters};

pub mod emission;
pub mod epoch_start;
pub mod staking;
pub mod validator;

#[cfg(test)]
#[path = "unit_tests/delegation_tests.rs"]
mod delegation_tests;
#[cfg(test)]
#[path = "unit_tests/f1_pool_tests.rs"]
mod f1_pool_tests;
#[cfg(test)]
#[path = "unit_tests/rewards_distribution_tests.rs"]
mod rewards_distribution_tests;
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
    /// Per-unit fee in USDC microdollars. Tx fee = `unit_fee * executor.fee_units(...)`.
    /// All fees on Soma are paid in USDC.
    pub unit_fee: u64,
}

impl FeeParameters {
    pub fn from_system_parameters(params: &SystemParameters) -> Self {
        Self { unit_fee: params.unit_fee }
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

    pub emission_pool: EmissionPool,

    /// Marketplace configuration parameters
    pub marketplace_params: MarketplaceParameters,

    /// Accumulated USDC microdollars collected from transaction fees.
    /// Grows monotonically until withdrawn via WithdrawProtocolFund (future).
    /// All user-paid gas fees route here; eventually used to buy back and burn SOMA.
    pub protocol_fund_balance: u64,

    /// Bridge state for USDC bridge between Ethereum and Soma
    pub bridge_state: BridgeState,

    /// Whether the system is in safe mode due to a failed epoch transition.
    /// Set to true when advance_epoch() fails; cleared on next successful advance.
    /// During safe mode: emissions are forfeited (schedule pauses), fees still
    /// route to protocol_fund inline.
    pub safe_mode: bool,
}

impl SystemStateV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        validators: Vec<Validator>,
        protocol_version: u64,
        epoch_start_timestamp_ms: u64,
        protocol_config: &ProtocolConfig,
        emission_fund: u64,
        emission_initial_distribution_amount: u64,
        emission_period_length: u64,
        emission_decrease_rate: u16,
        epoch_duration_ms_override: Option<u64>,
        marketplace_params: MarketplaceParameters,
        bridge_committee: BridgeCommittee,
    ) -> Self {
        let emission_pool = EmissionPool::new(
            emission_fund,
            emission_initial_distribution_amount,
            emission_period_length,
            emission_decrease_rate,
        );
        let mut parameters = protocol_config.build_system_parameters();
        if let Some(epoch_duration_ms) = epoch_duration_ms_override {
            parameters.epoch_duration_ms = epoch_duration_ms;
        }
        let mut validators = ValidatorSet::new(validators);

        for validator in &mut validators.validators {
            validator.activate(0);
        }

        Self {
            epoch: 0,
            validators,
            protocol_version,
            parameters,
            epoch_start_timestamp_ms,
            validator_report_records: BTreeMap::new(),
            emission_pool,
            marketplace_params,
            protocol_fund_balance: 0,
            bridge_state: BridgeState::new(bridge_committee),
            safe_mode: false,
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

    /// Stage 9d-C5: bump the validator's pool total_stake by `amount`.
    /// Returns the pool_id so callers can emit the matching
    /// DelegationEvent. The (pool, staker) row update happens in the
    /// executor — this method only mutates the pool aggregate.
    #[allow(clippy::result_large_err)]
    pub fn add_stake_to_validator(
        &mut self,
        validator_address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<ObjectID> {
        if amount == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Stake amount cannot be 0!".to_string(),
            });
        }
        let validator = self
            .validators
            .find_validator_with_pending_mut(validator_address)
            .ok_or(ExecutionFailureStatus::ValidatorNotFound)?;
        let pool_id = validator.staking_pool.id;
        validator.add_stake_principal(amount);
        self.validators.staking_pool_mappings.insert(pool_id, validator_address);
        Ok(pool_id)
    }

    /// Stage 9d-C5: at-genesis variant — pool starts preactive but
    /// the genesis builder activates it before sealing the genesis
    /// state. Behaviorally identical to `add_stake_to_validator` at
    /// runtime since `add_stake_principal` doesn't gate on
    /// activation.
    #[allow(clippy::result_large_err)]
    pub fn add_stake_to_validator_at_genesis(
        &mut self,
        validator_address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<ObjectID> {
        self.add_stake_to_validator(validator_address, amount)
    }

    /// Stage 9d-C5: drop `amount` of principal from the validator's
    /// pool total_stake. Returns the pool_id so callers can emit the
    /// matching DelegationEvent.
    #[allow(clippy::result_large_err)]
    pub fn remove_stake_from_validator(
        &mut self,
        pool_id: ObjectID,
        amount: u64,
    ) -> ExecutionResult<()> {
        let validator_address = self
            .validators
            .staking_pool_mappings
            .get(&pool_id)
            .copied()
            .ok_or(ExecutionFailureStatus::StakingPoolNotFound)?;

        if let Some(validator) =
            self.validators.find_validator_with_pending_mut(validator_address)
        {
            validator.remove_stake_principal(amount);
            return Ok(());
        }
        if let Some(inactive_validator) =
            self.validators.inactive_validators.get_mut(&pool_id)
        {
            inactive_validator.remove_stake_principal(amount);
            return Ok(());
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


    #[allow(clippy::result_large_err)]
    pub fn advance_epoch(
        &mut self,
        new_epoch: u64,
        next_protocol_config: &ProtocolConfig,
        epoch_total_transaction_fees: u64,
        epoch_start_timestamp_ms: u64,
        _epoch_randomness: Vec<u8>,
    ) -> ExecutionResult<BTreeMap<SomaAddress, validator::ValidatorRewardCredit>> {
        // 1. Verify we're advancing to the correct epoch
        if new_epoch != self.epoch + 1 {
            return Err(ExecutionFailureStatus::AdvancedToWrongEpoch);
        }

        // Clear safe mode flag if recovering. Nothing to drain — fees were routed
        // to protocol_fund inline during safe mode, and emissions for those epochs
        // were forfeited (schedule paused).
        if self.safe_mode {
            info!("Recovering from safe mode at epoch {}", new_epoch);
            self.safe_mode = false;
        }

        let prev_epoch_start_timestamp = self.epoch_start_timestamp_ms;
        self.epoch_start_timestamp_ms = epoch_start_timestamp_ms;

        let next_protocol_version = next_protocol_config.version.as_u64();

        // Check if protocol version is changing
        if next_protocol_version != self.protocol_version {
            info!("Protocol upgrade: {} -> {}", self.protocol_version, next_protocol_version);
            self.parameters = next_protocol_config.build_system_parameters();
        }

        // Get reward_slashing_rate from protocol config
        let reward_slashing_rate = next_protocol_config.reward_slashing_rate_bps();

        // 2. Route this epoch's USDC fees to the protocol fund.
        // Validators are paid from SOMA emissions only; fees fund future buybacks.
        self.protocol_fund_balance =
            self.protocol_fund_balance.saturating_add(epoch_total_transaction_fees);

        // 3. Calculate validator rewards — pure SOMA emissions.
        let mut total_rewards = 0u64;
        if epoch_start_timestamp_ms
            >= prev_epoch_start_timestamp + self.parameters.epoch_duration_ms
        {
            total_rewards = self.emission_pool.advance_epoch();
        }

        // 4. Increment epoch and update protocol version
        self.epoch = new_epoch;
        self.protocol_version = next_protocol_version;

        // 5. Process validator rewards (SOMA emissions only).
        let mut validator_reward_pool = total_rewards;
        let validator_rewards = self.validators.advance_epoch(
            new_epoch,
            &mut validator_reward_pool,
            reward_slashing_rate,
            &mut self.validator_report_records,
            VALIDATOR_LOW_STAKE_GRACE_PERIOD,
        );

        // Return any undistributed remainder to emission pool
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
    /// - Bumps epoch, timestamp, and protocol_version (so a fix can land via upgrade)
    /// - Routes this epoch's fees to `protocol_fund_balance` (saturating_add, can't fail)
    ///
    /// Emissions are forfeited — `emission_pool` is untouched, the schedule pauses.
    /// Validators get nothing for safe-mode epochs but fees keep flowing to the fund.
    /// All registries and the committee remain frozen until `advance_epoch()` recovers.
    pub fn advance_epoch_safe_mode(
        &mut self,
        new_epoch: u64,
        next_protocol_version: u64,
        epoch_total_transaction_fees: u64,
        epoch_start_timestamp_ms: u64,
    ) {
        self.safe_mode = true;
        self.epoch = new_epoch;
        self.protocol_version = next_protocol_version;
        self.epoch_start_timestamp_ms = epoch_start_timestamp_ms;

        // Fees → fund directly. saturating_add can't fail.
        self.protocol_fund_balance =
            self.protocol_fund_balance.saturating_add(epoch_total_transaction_fees);

        // Emissions: forfeited. emission_pool is untouched (matches Sui's
        // stake_subsidy behavior during safe mode).

        info!(
            "Safe mode activated for epoch {} (protocol v{}). Fees routed to fund: {}",
            new_epoch, next_protocol_version, epoch_total_transaction_fees
        );
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

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        validators: Vec<Validator>,
        protocol_version: u64,
        epoch_start_timestamp_ms: u64,
        protocol_config: &ProtocolConfig,
        emission_fund: u64,
        emission_initial_distribution_amount: u64,
        emission_period_length: u64,
        emission_decrease_rate: u16,
        epoch_duration_ms_override: Option<u64>,
        marketplace_params: MarketplaceParameters,
        bridge_committee: BridgeCommittee,
    ) -> Self {
        Self::V1(SystemStateV1::create(
            validators,
            protocol_version,
            epoch_start_timestamp_ms,
            protocol_config,
            emission_fund,
            emission_initial_distribution_amount,
            emission_period_length,
            emission_decrease_rate,
            epoch_duration_ms_override,
            marketplace_params,
            bridge_committee,
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
    #[cfg(test)]
    pub fn set_protocol_version_for_testing(&mut self, version: u64) {
        match self {
            Self::V1(v1) => v1.protocol_version = version,
        }
    }
    pub fn safe_mode(&self) -> bool {
        match self {
            Self::V1(v1) => v1.safe_mode,
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

    pub fn add_stake_to_validator(
        &mut self,
        validator_address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<ObjectID> {
        match self {
            Self::V1(v1) => v1.add_stake_to_validator(validator_address, amount),
        }
    }

    pub fn add_stake_to_validator_at_genesis(
        &mut self,
        validator_address: SomaAddress,
        amount: u64,
    ) -> ExecutionResult<ObjectID> {
        match self {
            Self::V1(v1) => v1.add_stake_to_validator_at_genesis(validator_address, amount),
        }
    }

    pub fn remove_stake_from_validator(
        &mut self,
        pool_id: ObjectID,
        amount: u64,
    ) -> ExecutionResult<()> {
        match self {
            Self::V1(v1) => v1.remove_stake_from_validator(pool_id, amount),
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

    pub fn advance_epoch(
        &mut self,
        new_epoch: u64,
        next_protocol_config: &ProtocolConfig,
        epoch_total_transaction_fees: u64,
        epoch_start_timestamp_ms: u64,
        epoch_randomness: Vec<u8>,
    ) -> ExecutionResult<BTreeMap<SomaAddress, validator::ValidatorRewardCredit>> {
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
        next_protocol_version: u64,
        epoch_total_transaction_fees: u64,
        epoch_start_timestamp_ms: u64,
    ) {
        match self {
            Self::V1(v1) => v1.advance_epoch_safe_mode(
                new_epoch,
                next_protocol_version,
                epoch_total_transaction_fees,
                epoch_start_timestamp_ms,
            ),
        }
    }


    pub fn fee_parameters(&self) -> FeeParameters {
        match self {
            Self::V1(v1) => v1.fee_parameters(),
        }
    }

    // --- Marketplace accessors ---

    pub fn marketplace_params(&self) -> &MarketplaceParameters {
        match self {
            Self::V1(v1) => &v1.marketplace_params,
        }
    }

    pub fn protocol_fund_balance(&self) -> u64 {
        match self {
            Self::V1(v1) => v1.protocol_fund_balance,
        }
    }

    pub fn add_protocol_fund_balance(&mut self, amount: u64) -> ExecutionResult<()> {
        match self {
            Self::V1(v1) => {
                v1.protocol_fund_balance = v1
                    .protocol_fund_balance
                    .checked_add(amount)
                    .ok_or(ExecutionFailureStatus::ArithmeticOverflow)?;
                Ok(())
            }
        }
    }

    // --- Bridge accessors ---

    pub fn bridge_state(&self) -> &BridgeState {
        match self {
            Self::V1(v1) => &v1.bridge_state,
        }
    }

    pub fn bridge_state_mut(&mut self) -> &mut BridgeState {
        match self {
            Self::V1(v1) => &mut v1.bridge_state,
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
