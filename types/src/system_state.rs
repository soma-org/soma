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
    collections::{BTreeMap, HashMap, HashSet},
    str::FromStr,
};

use fastcrypto::{bls12381, ed25519::Ed25519PublicKey, traits::ToFromBytes};
use serde::{Deserialize, Serialize};
use tracing::error;

use crate::{
    base::{AuthorityName, SomaAddress},
    committee::{
        Authority, Committee, CommitteeWithNetworkMetadata, EpochId, NetworkMetadata, VotingPower,
    },
    crypto::{self, NetworkPublicKey, ProtocolPublicKey},
    error::{SomaError, SomaResult},
    multiaddr::Multiaddr,
    parameters,
    peer_id::PeerId,
    SYSTEM_STATE_OBJECT_ID,
};
use crate::{
    crypto::{AuthorityPublicKey, SomaKeyPair, SomaPublicKey},
    storage::object_store::ObjectStore,
};

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

    /// Minimum number of active validators at any moment.
    pub min_validator_count: u64,

    /// Maximum number of active validators at any moment.
    /// We do not allow the number of validators in any epoch to go above this.
    pub max_validator_count: u64,

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
            min_validator_count: 0,
            max_validator_count: 0,
            min_validator_joining_stake: 0,
            validator_low_stake_threshold: 0,
            validator_very_low_stake_threshold: 0,
            validator_low_stake_grace_period: 0,
        }
    }
}

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
        voting_power: u64,
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
                next_epoch_protocol_pubkey: None,
                next_epoch_network_pubkey: None,
                next_epoch_net_address: None,
                next_epoch_p2p_address: None,
                next_epoch_primary_address: None,
            },
            voting_power,
        }
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

    /// The currently active validators participating in consensus
    pub active_validators: Vec<Validator>,

    /// Validators that will be added to the active set in the next epoch
    pub pending_active_validators: Vec<Validator>,

    /// Active validators that will be removed in the next epoch
    pub pending_removals: Vec<Validator>,

    // pub inactive_validators: Vec<Validator>,
    /// Validators that are candidates for joining the active set
    pub validator_candidates: Vec<Validator>,
    // pub at_risk_validators: HashMap<Validator, u64>,
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
    pub fn new(init_active_validators: Vec<Validator>) -> Self {
        let total_stake = init_active_validators.iter().map(|v| v.voting_power).sum();
        Self {
            total_stake,
            active_validators: init_active_validators,
            pending_active_validators: Vec::new(),
            pending_removals: Vec::new(),
            // inactive_validators: Vec::new(),
            validator_candidates: Vec::new(),
            // at_risk_validators: HashMap::new(),
        }
    }

    /// # Request to add a validator
    ///
    /// Requests to add a validator to the active set in the next epoch.
    ///
    /// ## Behavior
    /// The validator is added to the pending_active_validators list and will
    /// become active in the next epoch when advance_epoch() is called.
    ///
    /// ## Arguments
    /// * `validator` - The validator to add
    ///
    /// ## Returns
    /// Ok(()) if the validator was successfully added to the pending set,
    /// or an error if the validator is already active or pending
    ///
    /// ## Errors
    /// Returns SomaError::DuplicateValidator if the validator is already
    /// in the active or pending set
    pub fn request_add_validator(&mut self, validator: Validator) -> SomaResult {
        // assert!(
        //     self.validator_candidates.contains(validator_address),
        //     ENotValidatorCandidate
        // );
        // let wrapper = self.validator_candidates.remove(validator_address);
        // let validator = wrapper.destroy();
        // assert!(validator.is_preactive(), EValidatorNotCandidate);
        // assert!(validator.total_stake_amount() >= min_joining_stake_amount, EMinJoiningStakeNotReached);

        if self.active_validators.contains(&validator)
            || self.pending_active_validators.contains(&validator)
        {
            return Err(SomaError::DuplicateValidator);
        }

        self.pending_active_validators.push(validator);
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
    pub fn request_remove_validator(&mut self, address: SomaAddress) -> SomaResult {
        let validator = self
            .active_validators
            .iter()
            .find(|v| address == v.metadata.soma_address);

        if let Some(v) = validator {
            if self.pending_removals.contains(&v) {
                return Err(SomaError::ValidatorAlreadyRemoved);
            }
            self.pending_removals.push(v.clone());
        } else {
            return Err(SomaError::NotAValidator);
        }
        Ok(())
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
    pub fn advance_epoch(&mut self) {
        // TODO: compute and distribute validator rewards and slashing

        // TODO: process pending stakes and withdrawals

        while let Some(validator) = self.pending_removals.pop() {
            let validator_index = self
                .active_validators
                .iter()
                .position(|v| validator.metadata.soma_address == v.metadata.soma_address)
                .map(|i| i as u64)
                .expect("Cannot remove validator that is not in active validators");
            self.active_validators.remove(validator_index as usize);
        }

        while let Some(validator) = self.pending_active_validators.pop() {
            self.active_validators.push(validator);
        }

        // TODO: kick low validators out

        // total stake is updated
        self.total_stake = self.active_validators.iter().map(|v| v.voting_power).sum();
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

    // fn get_pending_active_validators<S: ObjectStore + ?Sized>(
    //     &self,
    //     object_store: &S,
    // ) -> Result<Vec<SuiValidatorSummary>, SuiError>;

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
    ) -> Self {
        Self {
            epoch: 0,
            validators: ValidatorSet::new(validators),
            parameters,
            epoch_start_timestamp_ms,
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
    ) -> SomaResult {
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
        );
        self.validators.request_add_validator(validator)
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
    ) -> SomaResult {
        self.validators.request_remove_validator(signer)
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
    pub fn advance_epoch(&mut self, new_epoch: u64, epoch_start_timestamp_ms: u64) -> SomaResult {
        self.epoch_start_timestamp_ms = epoch_start_timestamp_ms;

        // Sanity check to make sure we are advancing to the right epoch.
        if new_epoch == self.epoch {
            return Err(SomaError::AdvancedToWrongEpoch);
        }

        self.epoch += 1;

        self.validators.advance_epoch();

        Ok(())
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

/// # EpochStartSystemState
///
/// A snapshot of the system state at the beginning of an epoch.
///
/// ## Purpose
/// Provides an immutable view of the system state at the start of an epoch,
/// including the active validators and epoch information. This is used by
/// components that need a consistent view of the system state throughout
/// an epoch, even as the mutable system state may change.
///
/// ## Usage
/// This struct is created from a SystemState at the beginning of an epoch
/// and is used by various components to access epoch-specific information
/// without needing to access the full SystemState.
#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct EpochStartSystemState {
    /// The epoch number
    epoch: EpochId,

    /// The timestamp when the epoch started (in milliseconds)
    epoch_start_timestamp_ms: u64,

    /// The duration of the epoch (in milliseconds)
    epoch_duration_ms: u64,

    /// The active validators at the start of the epoch
    active_validators: Vec<EpochStartValidatorInfo>,
}

impl EpochStartSystemState {
    /// # Create a new epoch start system state
    ///
    /// Creates a new epoch start system state with the specified epoch information
    /// and active validators.
    ///
    /// ## Arguments
    /// * `epoch` - The epoch number
    /// * `epoch_start_timestamp_ms` - The timestamp when the epoch started (in milliseconds)
    /// * `epoch_duration_ms` - The duration of the epoch (in milliseconds)
    /// * `active_validators` - The active validators at the start of the epoch
    ///
    /// ## Returns
    /// A new EpochStartSystemState instance with the specified epoch information
    /// and active validators
    pub fn new(
        epoch: EpochId,
        epoch_start_timestamp_ms: u64,
        epoch_duration_ms: u64,
        active_validators: Vec<EpochStartValidatorInfo>,
    ) -> Self {
        Self {
            epoch,
            epoch_start_timestamp_ms,
            epoch_duration_ms,
            active_validators,
        }
    }

    /// # Create a new epoch start system state for testing
    ///
    /// Creates a new epoch start system state with default values for testing.
    ///
    /// ## Returns
    /// A new EpochStartSystemState instance with epoch 0, timestamp 0,
    /// duration 1000ms, and no active validators
    pub fn new_for_testing() -> Self {
        Self::new_for_testing_with_epoch(0)
    }

    /// # Create a new epoch start system state for testing with a specific epoch
    ///
    /// Creates a new epoch start system state with default values for testing,
    /// but with the specified epoch number.
    ///
    /// ## Arguments
    /// * `epoch` - The epoch number to use
    ///
    /// ## Returns
    /// A new EpochStartSystemState instance with the specified epoch, timestamp 0,
    /// duration 1000ms, and no active validators
    pub fn new_for_testing_with_epoch(epoch: EpochId) -> Self {
        Self {
            epoch,
            epoch_start_timestamp_ms: 0,
            epoch_duration_ms: 1000,
            active_validators: vec![],
        }
    }
}

impl EpochStartSystemStateTrait for EpochStartSystemState {
    fn epoch(&self) -> EpochId {
        self.epoch
    }

    fn epoch_start_timestamp_ms(&self) -> u64 {
        self.epoch_start_timestamp_ms
    }

    fn epoch_duration_ms(&self) -> u64 {
        self.epoch_duration_ms
    }

    fn get_validator_addresses(&self) -> Vec<SomaAddress> {
        self.active_validators
            .iter()
            .map(|validator| validator.soma_address)
            .collect()
    }

    fn get_committee_with_network_metadata(&self) -> CommitteeWithNetworkMetadata {
        let validators = self
            .active_validators
            .iter()
            .map(|validator| {
                (
                    validator.authority_name(),
                    (
                        validator.voting_power,
                        NetworkMetadata {
                            consensus_address: validator.p2p_address.clone(),
                            network_address: validator.net_address.clone(),
                            primary_address: validator.primary_address.clone(),
                            protocol_key: ProtocolPublicKey::new(
                                validator.worker_pubkey.clone().into_inner(),
                            ),
                            network_key: validator.network_pubkey.clone(),
                            authority_key: validator.protocol_pubkey.clone(),
                            hostname: validator.hostname.clone(),
                        },
                    ),
                )
            })
            .collect();

        CommitteeWithNetworkMetadata::new(self.epoch, validators)
    }

    fn get_committee(&self) -> Committee {
        let voting_rights: BTreeMap<_, _> = self
            .active_validators
            .iter()
            .map(|v| (v.authority_name(), v.voting_power))
            .collect();

        let authorities: BTreeMap<_, _> = self
            .active_validators
            .iter()
            .map(|v| {
                (
                    v.authority_name(),
                    Authority {
                        stake: v.voting_power,
                        address: v.primary_address.clone(),
                        hostname: v.hostname.clone(),
                        protocol_key: ProtocolPublicKey::new(v.worker_pubkey.clone().into_inner()),
                        network_key: v.network_pubkey.clone(),
                        authority_key: v.protocol_pubkey.clone(),
                    },
                )
            })
            .collect();

        Committee::new(self.epoch, voting_rights, authorities)
    }

    fn get_authority_names_to_peer_ids(&self) -> HashMap<AuthorityName, PeerId> {
        self.active_validators
            .iter()
            .map(|validator| {
                let name = validator.authority_name();
                let peer_id = PeerId(validator.network_pubkey.to_bytes());

                (name, peer_id)
            })
            .collect()
    }

    fn get_authority_names_to_hostnames(&self) -> HashMap<AuthorityName, String> {
        self.active_validators
            .iter()
            .map(|validator| {
                let name = validator.authority_name();
                let hostname = validator.hostname.clone();

                (name, hostname)
            })
            .collect()
    }
}

/// # EpochStartSystemStateTrait
///
/// A trait defining the interface for accessing epoch start system state information.
///
/// ## Purpose
/// Provides a common interface for accessing epoch information and committee
/// data from an epoch start system state. This allows components to work with
/// the immutable epoch state without depending on the specific implementation.
///
/// ## Usage
/// This trait is implemented by EpochStartSystemState and can be used by components
/// that need to access epoch-specific information without depending on the
/// full system state.
pub trait EpochStartSystemStateTrait {
    /// Get the epoch number
    fn epoch(&self) -> EpochId;

    /// Get the timestamp when the epoch started (in milliseconds)
    fn epoch_start_timestamp_ms(&self) -> u64;

    /// Get the duration of the epoch (in milliseconds)
    fn epoch_duration_ms(&self) -> u64;

    /// Get the addresses of all active validators in the epoch
    fn get_validator_addresses(&self) -> Vec<SomaAddress>;

    /// Get the committee for the epoch
    fn get_committee(&self) -> Committee;

    /// Get the committee for the epoch, including network metadata
    fn get_committee_with_network_metadata(&self) -> CommitteeWithNetworkMetadata;

    /// Get a mapping from authority names to peer IDs
    fn get_authority_names_to_peer_ids(&self) -> HashMap<AuthorityName, PeerId>;

    /// Get a mapping from authority names to hostnames
    fn get_authority_names_to_hostnames(&self) -> HashMap<AuthorityName, String>;
}

/// # EpochStartValidatorInfo
///
/// Information about a validator at the start of an epoch.
///
/// ## Purpose
/// Stores the essential information about a validator that is needed during
/// an epoch, including its cryptographic keys, network addresses, and voting power.
/// This is a snapshot of the validator's state at the beginning of the epoch.
///
/// ## Usage
/// This struct is stored in the EpochStartSystemState and is used to form
/// the committee for the epoch and to provide information about validators
/// to various components.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct EpochStartValidatorInfo {
    /// The Soma blockchain address of the validator
    pub soma_address: SomaAddress,

    /// The BLS public key used for consensus protocol operations
    pub protocol_pubkey: PublicKey,

    /// The worker public key used for worker operations
    pub worker_pubkey: crypto::NetworkPublicKey,

    /// The network public key used for network identity and authentication
    pub network_pubkey: crypto::NetworkPublicKey,

    /// The network address for general network communication
    pub net_address: Multiaddr,

    /// The p2p address for peer-to-peer communication
    pub p2p_address: Multiaddr,

    /// The primary address for validator services
    pub primary_address: Multiaddr,

    /// The validator's voting power in the consensus protocol
    pub voting_power: VotingPower,

    /// The hostname of the validator
    pub hostname: String,
}

impl EpochStartValidatorInfo {
    /// # Get the authority name
    ///
    /// Derives the authority name from the validator's protocol public key.
    ///
    /// ## Returns
    /// The authority name derived from the protocol public key
    pub fn authority_name(&self) -> AuthorityName {
        (&self.protocol_pubkey).into()
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
