use genesis_config::ValidatorGenesisConfig;
use std::num::NonZeroUsize;
use types::crypto::SomaKeyPair;

pub mod genesis_config;
pub mod local_ip_utils;
pub mod network_config;
pub mod node_config_builder;

pub enum CommitteeConfig {
    Size(NonZeroUsize),
    Validators(Vec<ValidatorGenesisConfig>),
    AccountKeys(Vec<SomaKeyPair>),
    /// Indicates that a committee should be deterministically generated, using the provided rng
    /// as a source of randomness as well as generating deterministic network port information.
    Deterministic((NonZeroUsize, Option<Vec<SomaKeyPair>>)),
}
