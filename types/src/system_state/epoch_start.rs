use std::collections::{BTreeMap, HashMap};

use protocol_config::{Chain, ProtocolVersion};
use serde::{Deserialize, Serialize};

use crate::{
    base::{AuthorityName, SomaAddress},
    committee::{
        Authority, Committee, CommitteeWithNetworkMetadata, EpochId, NetworkMetadata, VotingPower,
    },
    crypto::{self, ProtocolPublicKey},
    multiaddr::Multiaddr,
    peer_id::PeerId,
    system_state::{FeeParameters, SystemParameters},
};

use super::PublicKey;

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

    pub protocol_version: u64,

    pub fee_parameters: FeeParameters,
}

impl EpochStartSystemState {
    pub fn new(
        epoch: EpochId,
        protocol_version: u64,
        epoch_start_timestamp_ms: u64,
        epoch_duration_ms: u64,
        active_validators: Vec<EpochStartValidatorInfo>,
        fee_parameters: FeeParameters,
    ) -> Self {
        Self {
            epoch,
            protocol_version,
            epoch_start_timestamp_ms,
            epoch_duration_ms,
            active_validators,
            fee_parameters,
        }
    }

    pub fn new_for_testing() -> Self {
        Self::new_for_testing_with_epoch(0)
    }

    pub fn new_for_testing_with_epoch(epoch: EpochId) -> Self {
        let protocol_config =
            protocol_config::ProtocolConfig::get_for_version(ProtocolVersion::MAX, Chain::Mainnet);
        Self {
            epoch,
            protocol_version: ProtocolVersion::MAX.as_u64(),
            epoch_start_timestamp_ms: 0,
            epoch_duration_ms: 1000,
            active_validators: vec![],
            fee_parameters: FeeParameters::from_system_parameters(
                &protocol_config.build_system_parameters(None),
            ),
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

    fn protocol_version(&self) -> ProtocolVersion {
        ProtocolVersion::new(self.protocol_version)
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
                        address: v.primary_address.clone(), // TODO: review the naming to clear up these address names
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

    fn protocol_version(&self) -> ProtocolVersion;
}

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
    pub fn authority_name(&self) -> AuthorityName {
        (&self.protocol_pubkey).into()
    }
}
