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

pub type PublicKey = bls12381::min_sig::BLS12381PublicKey;

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
    // TODO: make this configurable
    fn default() -> Self {
        Self {
            epoch_duration_ms: 999, //1000 * 60, // TODO: 1000 * 60 * 60 * 24, // 1 day
            min_validator_count: 0,
            max_validator_count: 0,
            min_validator_joining_stake: 0,
            validator_low_stake_threshold: 0,
            validator_very_low_stake_threshold: 0,
            validator_low_stake_grace_period: 0,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Deserialize, Serialize, Hash)]
pub struct ValidatorMetadata {
    pub soma_address: SomaAddress,
    pub protocol_pubkey: PublicKey,
    pub network_pubkey: crate::crypto::NetworkPublicKey,
    pub worker_pubkey: crate::crypto::NetworkPublicKey,
    pub net_address: Multiaddr,
    pub p2p_address: Multiaddr,
    pub primary_address: Multiaddr,
    pub next_epoch_protocol_pubkey: Option<PublicKey>,
    pub next_epoch_network_pubkey: Option<crate::crypto::NetworkPublicKey>,
    pub next_epoch_net_address: Option<Multiaddr>,
    pub next_epoch_p2p_address: Option<Multiaddr>,
    pub next_epoch_primary_address: Option<Multiaddr>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct Validator {
    pub metadata: ValidatorMetadata,
    pub voting_power: u64,
}

impl Validator {
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

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct ValidatorSet {
    pub total_stake: u64,
    pub active_validators: Vec<Validator>,
    pub pending_active_validators: Vec<Validator>,
    pub pending_removals: Vec<u64>,
    // pub inactive_validators: Vec<Validator>,
    pub validator_candidates: Vec<Validator>,
    // pub at_risk_validators: HashMap<Validator, u64>,
}

impl ValidatorSet {
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

    pub fn request_remove_validator(&mut self, address: SomaAddress) -> SomaResult {
        let validator_index = self
            .active_validators
            .iter()
            .position(|v| address == v.metadata.soma_address)
            .map(|i| i as u64);
        if let Some(index) = validator_index {
            if self.pending_removals.contains(&(index as u64)) {
                return Err(SomaError::ValidatorAlreadyRemoved);
            }
            self.pending_removals.push(index as u64);
        } else {
            return Err(SomaError::NotAValidator);
        }
        Ok(())
    }

    pub fn advance_epoch(&mut self) {
        // TODO: compute and distribute validator rewards and slashing

        // TODO: process pending stakes and withdrawals

        while let Some(validator) = self.pending_active_validators.pop() {
            self.active_validators.push(validator);
        }

        while let Some(index) = self.pending_removals.pop() {
            self.active_validators.remove(index as usize);
        }

        // TODO: kick low validators out

        // total stake is updated
        self.total_stake = self.active_validators.iter().map(|v| v.voting_power).sum();
    }
}

pub trait SystemStateTrait {
    fn epoch(&self) -> u64;
    fn epoch_start_timestamp_ms(&self) -> u64;
    fn epoch_duration_ms(&self) -> u64;
    fn get_current_epoch_committee(&self) -> CommitteeWithNetworkMetadata;
    // fn get_pending_active_validators<S: ObjectStore + ?Sized>(
    //     &self,
    //     object_store: &S,
    // ) -> Result<Vec<SuiValidatorSummary>, SuiError>;
    fn into_epoch_start_state(self) -> EpochStartSystemState;
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct SystemState {
    pub epoch: u64,
    // pub protocol_version: u64,
    // pub system_state_version: u64,
    pub validators: ValidatorSet,
    pub parameters: SystemParameters,
    pub epoch_start_timestamp_ms: u64,
}

impl SystemState {
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

    pub fn request_remove_validator(
        &mut self,
        signer: SomaAddress,
        pubkey_bytes: Vec<u8>,
    ) -> SomaResult {
        self.validators.request_remove_validator(signer)
    }

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

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct EpochStartSystemState {
    epoch: EpochId,
    epoch_start_timestamp_ms: u64,
    epoch_duration_ms: u64,
    active_validators: Vec<EpochStartValidatorInfo>,
}

impl EpochStartSystemState {
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

    pub fn new_for_testing() -> Self {
        Self::new_for_testing_with_epoch(0)
    }

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

pub trait EpochStartSystemStateTrait {
    fn epoch(&self) -> EpochId;
    fn epoch_start_timestamp_ms(&self) -> u64;
    fn epoch_duration_ms(&self) -> u64;
    fn get_validator_addresses(&self) -> Vec<SomaAddress>;
    fn get_committee(&self) -> Committee;
    fn get_committee_with_network_metadata(&self) -> CommitteeWithNetworkMetadata;
    fn get_authority_names_to_peer_ids(&self) -> HashMap<AuthorityName, PeerId>;
    fn get_authority_names_to_hostnames(&self) -> HashMap<AuthorityName, String>;
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct EpochStartValidatorInfo {
    pub soma_address: SomaAddress,
    pub protocol_pubkey: PublicKey,
    pub worker_pubkey: crypto::NetworkPublicKey,
    pub network_pubkey: crypto::NetworkPublicKey,
    pub net_address: Multiaddr,
    pub p2p_address: Multiaddr,
    pub primary_address: Multiaddr,
    pub voting_power: VotingPower,
    pub hostname: String,
}

impl EpochStartValidatorInfo {
    pub fn authority_name(&self) -> AuthorityName {
        (&self.protocol_pubkey).into()
    }
}

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
