use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::{
    base::{AuthorityName, SomaAddress},
    committee::{
        Authority, Committee, CommitteeWithNetworkMetadata, EpochId, NetworkMetadata, VotingPower,
    },
    crypto::{self, ProtocolPublicKey},
    multiaddr::Multiaddr,
    peer_id::PeerId,
};

use super::PublicKey;

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
    pub epoch: EpochId,

    /// The timestamp when the epoch started (in milliseconds)
    pub epoch_start_timestamp_ms: u64,

    /// The duration of the epoch (in milliseconds)
    pub epoch_duration_ms: u64,

    /// The active validators at the start of the epoch
    pub active_validators: Vec<EpochStartValidatorInfo>,
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
