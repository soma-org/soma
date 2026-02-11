use std::{
    cell::RefCell,
    collections::BTreeSet,
    sync::atomic::{AtomicBool, Ordering},
};

use clap::*;
use fastcrypto::encoding::{Base58, Encoding, Hex};
use protocol_config_macros::{
    ProtocolConfigAccessors, ProtocolConfigFeatureFlagsGetters, ProtocolConfigOverride,
};
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use tracing::{info, warn};

mod tensor;
pub use tensor::{BcsF32, SomaTensor};

/// The minimum and maximum protocol versions supported by this build.
pub const MIN_PROTOCOL_VERSION: u64 = 1;
pub const MAX_PROTOCOL_VERSION: u64 = 1;

#[derive(Copy, Clone, Debug, Hash, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProtocolVersion(u64);

impl ProtocolVersion {
    // The minimum and maximum protocol version supported by this binary. Counterintuitively, this constant may
    // change over time as support for old protocol versions is removed from the source. This
    // ensures that when a new network (such as a testnet) is created, its genesis committee will
    // use a protocol version that is actually supported by the binary.
    pub const MIN: Self = Self(MIN_PROTOCOL_VERSION);

    pub const MAX: Self = Self(MAX_PROTOCOL_VERSION);

    #[cfg(not(msim))]
    pub const MAX_ALLOWED: Self = Self::MAX;

    // We create one additional "fake" version in simulator builds so that we can test upgrades.
    #[cfg(msim)]
    pub const MAX_ALLOWED: Self = Self(MAX_PROTOCOL_VERSION + 1);

    pub fn new(v: u64) -> Self {
        Self(v)
    }

    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    // For serde deserialization - we don't define a Default impl because there isn't a single
    // universally appropriate default value.
    pub fn max() -> Self {
        Self::MAX
    }

    pub fn prev(self) -> Self {
        Self(self.0.checked_sub(1).unwrap())
    }
}

impl From<u64> for ProtocolVersion {
    fn from(v: u64) -> Self {
        Self::new(v)
    }
}

impl std::ops::Sub<u64> for ProtocolVersion {
    type Output = Self;
    fn sub(self, rhs: u64) -> Self::Output {
        Self::new(self.0 - rhs)
    }
}

impl std::ops::Add<u64> for ProtocolVersion {
    type Output = Self;
    fn add(self, rhs: u64) -> Self::Output {
        Self::new(self.0 + rhs)
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Copy, PartialOrd, Ord, Eq)]
pub enum Chain {
    Mainnet,
    Testnet,
    #[default]
    Unknown,
}

impl Chain {
    pub fn as_str(self) -> &'static str {
        match self {
            Chain::Mainnet => "mainnet",
            Chain::Testnet => "testnet",
            Chain::Unknown => "unknown",
        }
    }
}

pub struct Error(pub String);

fn is_false(b: &bool) -> bool {
    !b
}

fn is_empty(b: &BTreeSet<String>) -> bool {
    b.is_empty()
}

fn is_zero(val: &u64) -> bool {
    *val == 0
}

/// Records on/off feature flags that may vary at each protocol version.
#[derive(Default, Clone, Serialize, Deserialize, Debug, ProtocolConfigFeatureFlagsGetters)]
struct FeatureFlags {}

/// Constants that change the behavior of the protocol.
///
/// The value of each constant here must be fixed for a given protocol version. To change the value
/// of a constant, advance the protocol version, and add support for it in `get_for_version` under
/// the new version number.
/// (below).
///
/// To add a new field to this struct, use the following procedure:
/// - Advance the protocol version.
/// - Add the field as a private `Option<T>` to the struct.
/// - Initialize the field to `None` in prior protocol versions.
/// - Initialize the field to `Some(val)` for your new protocol version.
/// - Add a public getter that simply unwraps the field.
/// - Two public getters of the form `field(&self) -> field_type`
///     and `field_as_option(&self) -> Option<field_type>` will be automatically generated for you.
/// Example for a field: `new_constant: Option<u64>`
/// ```rust,ignore
///      pub fn new_constant(&self) -> u64 {
///         self.new_constant.expect(Self::CONSTANT_ERR_MSG)
///     }
///      pub fn new_constant_as_option(&self) -> Option<u64> {
///         self.new_constant.expect(Self::CONSTANT_ERR_MSG)
///     }
/// ```
/// With `pub fn new_constant(&self) -> u64`, if the constant is accessed in a protocol version
/// in which it is not defined, the validator will crash. (Crashing is necessary because
/// this type of error would almost always result in forking if not prevented here).
/// If you don't want the validator to crash, you can use the
/// `pub fn new_constant_as_option(&self) -> Option<u64>` getter, which will
/// return `None` if the field is not defined at that version.
/// - If you want a customized getter, you can add a method in the impl.
#[skip_serializing_none]
#[derive(Clone, Serialize, Debug, ProtocolConfigAccessors, ProtocolConfigOverride)]
pub struct ProtocolConfig {
    pub version: ProtocolVersion,

    feature_flags: FeatureFlags,

    /// Minimum interval of commit timestamps between consecutive checkpoints.
    min_checkpoint_interval_ms: Option<u64>,

    /// The maximum serialised transaction size (in bytes) accepted by consensus. That should be bigger than the
    /// `max_tx_size_bytes` with some additional headroom.
    consensus_max_transaction_size_bytes: Option<u64>,
    /// The maximum size of transactions included in a consensus block.
    consensus_max_transactions_in_block_bytes: Option<u64>,
    /// The maximum number of transactions included in a consensus block.
    consensus_max_num_transactions_in_block: Option<u64>,

    /// Configures the garbage collection depth for consensus. When is unset or `0` then the garbage collection
    /// is disabled.
    consensus_gc_depth: Option<u32>,

    /// The number of commits to consider when computing a deterministic commit rate.
    consensus_commit_rate_estimation_window_size: Option<u32>,

    // Dictates the threshold (percentage of stake) that is used to calculate the "bad" nodes to be
    // swapped when creating the consensus schedule. The values should be of the range [0 - 33]. Anything
    // above 33 (f) will not be allowed.
    consensus_bad_nodes_stake_threshold: Option<u64>,

    mysticeti_num_leaders_per_round: Option<usize>,

    // === Core Protocol ===
    /// Max number of transactions per checkpoint.
    /// Note that this is a protocol constant and not a config as validators must have this set to
    /// the same value, otherwise they *will* fork.
    max_transactions_per_checkpoint: Option<u64>,

    /// Max size of a checkpoint in bytes.
    /// Note that this is a protocol constant and not a config as validators must have this set to
    /// the same value, otherwise they *will* fork.
    max_checkpoint_size_bytes: Option<u64>,

    /// A protocol upgrade always requires 2f+1 stake to agree. We support a buffer of additional
    /// stake (as a fraction of f, expressed in basis points) that is required before an upgrade
    /// can happen automatically. 10000bps would indicate that complete unanimity is required (all
    /// 3f+1 must vote), while 0bps would indicate that 2f+1 is sufficient.
    buffer_stake_for_protocol_upgrade_bps: Option<u64>,

    epoch_duration_ms: Option<u64>,

    // === Reward/Emission Parameters ===
    /// Percentage of epoch rewards allocated to validators (bps, e.g. 7000 = 70%)
    validator_reward_allocation_bps: Option<u64>,
    reward_slashing_rate_bps: Option<u64>,

    // === Model Parameters ===
    /// Minimum stake required to commit a new model (in shannons)
    model_min_stake: Option<u64>,
    /// Required architecture version for new model commits
    model_architecture_version: Option<u64>,
    /// Slash rate (bps) for models that fail to reveal after commit (e.g. 5000 = 50%)
    model_reveal_slash_rate_bps: Option<u64>,
    /// Slash rate (bps) for models removed via tally reports (e.g. 9500 = 95%)
    model_tally_slash_rate_bps: Option<u64>,

    // === Fee Parameters ===
    target_epoch_fee_collection: Option<u64>,
    base_fee: Option<u64>,
    write_object_fee: Option<u64>,
    initial_value_fee_bps: Option<u64>,
    min_value_fee_bps: Option<u64>,
    max_value_fee_bps: Option<u64>,
    fee_adjustment_rate_bps: Option<u64>,

    // === Target/Mining Parameters ===
    /// Number of models assigned to each target (uniformly random selection)
    target_models_per_target: Option<u64>,
    /// Dimension of target embedding vectors
    target_embedding_dim: Option<u64>,
    /// Initial distance threshold for new targets (f32 cosine distance, as BcsF32)
    target_initial_distance_threshold: Option<BcsF32>,
    /// Percentage of epoch emissions allocated to targets (bps, e.g. 8000 = 80%)
    target_reward_allocation_bps: Option<u64>,
    /// Target hit rate (bps, e.g. 8000 = 80% of targets should be filled)
    /// Difficulty adjusts toward this rate. Higher rate = make easier, lower = harder.
    target_hit_rate_target_bps: Option<u64>,
    /// Decay factor for hit rate EMA (bps, e.g. 9000 = 90% decay means 10% weight on new data)
    /// Higher decay = smoother/slower adjustments; lower = more responsive.
    target_hit_rate_ema_decay_bps: Option<u64>,
    /// Difficulty adjustment rate per epoch (bps, e.g. 500 = 5% max change)
    target_difficulty_adjustment_rate_bps: Option<u64>,
    /// Maximum distance threshold (cap for difficulty decrease, as BcsF32)
    target_max_distance_threshold: Option<BcsF32>,
    /// Minimum distance threshold (floor for difficulty increase, as BcsF32)
    target_min_distance_threshold: Option<BcsF32>,
    /// Number of targets to issue at genesis and at each epoch start
    target_initial_targets_per_epoch: Option<u64>,

    // === Reward Distribution Parameters ===
    /// Percentage of target reward allocated to the miner (bps, e.g. 5000 = 50%)
    target_miner_reward_share_bps: Option<u64>,
    /// Percentage of target reward allocated to the model owner (bps, e.g. 3000 = 30%)
    target_model_reward_share_bps: Option<u64>,
    /// Percentage of target reward given to the claimer as incentive (bps, e.g. 100 = 1%)
    target_claimer_incentive_bps: Option<u64>,

    // === Submission Parameters ===
    /// Bond per byte of submitted data (in shannons)
    /// Bond is held on the Target and returned to miner on successful claim,
    /// or forfeited to emission pool on successful challenge.
    submission_bond_per_byte: Option<u64>,

    // === Challenge Parameters ===
    /// Bond per byte for challengers (in shannons)
    /// Challenger must post challenger_bond_per_byte * data_size as bond.
    /// If challenger wins: miner's bond is slashed, challenger gets reward.
    /// If challenger loses: challenger's bond is slashed.
    challenger_bond_per_byte: Option<u64>,

    // === Data Size Limits ===
    /// Maximum data size allowed for submissions (in bytes).
    /// Submissions exceeding this size will be rejected.
    /// This also affects audit timeouts which are calculated based on data size.
    max_submission_data_size: Option<u64>,
}

// Instantiations for each protocol version.
impl ProtocolConfig {
    /// Get the value ProtocolConfig that are in effect during the given protocol version.
    pub fn get_for_version(version: ProtocolVersion, chain: Chain) -> Self {
        // ProtocolVersion can be deserialized so we need to check it here as well.
        assert!(
            version >= ProtocolVersion::MIN,
            "Network protocol version is {:?}, but the minimum supported version by the binary is {:?}. Please upgrade the binary.",
            version,
            ProtocolVersion::MIN.0,
        );
        assert!(
            version <= ProtocolVersion::MAX_ALLOWED,
            "Network protocol version is {:?}, but the maximum supported version by the binary is {:?}. Please upgrade the binary.",
            version,
            ProtocolVersion::MAX_ALLOWED.0,
        );

        let mut ret = Self::get_for_version_impl(version, chain);
        ret.version = version;

        ret
    }

    fn get_for_version_impl(version: ProtocolVersion, chain: Chain) -> Self {
        // TODO: tweak this for msim runs after adding more protocol versions
        // #[cfg(msim)]
        // {
        //     // populate the fake simulator version # with a different base tx cost.
        //     if version == ProtocolVersion::MAX_ALLOWED {
        //         let mut config = Self::get_for_version_impl(version - 1, Chain::Unknown);
        //         config.base_tx_cost_fixed = Some(config.base_tx_cost_fixed() + 1000);
        //         return config;
        //     }
        // }

        // IMPORTANT: Never modify the value of any constant for a pre-existing protocol version.
        // To change the values here you must create a new protocol version with the new values!
        let mut cfg = Self {
            // will be overwritten before being returned
            version,

            // All flags are disabled in V1
            feature_flags: Default::default(),

            min_checkpoint_interval_ms: Some(200),

            consensus_max_transaction_size_bytes: Some(256 * 1024), // 256KB

            // Assume 1KB per transaction and 500 transactions per block.
            consensus_max_transactions_in_block_bytes: Some(512 * 1024),
            // Assume 20_000 TPS * 5% max stake per validator / (minimum) 4 blocks per round = 250 transactions per block maximum
            // Using a higher limit that is 512, to account for bursty traffic and system transactions.
            consensus_max_num_transactions_in_block: Some(512),

            // Assuming a round rate of max 15/sec, then using a gc depth of 60 allow blocks within a window of ~4 seconds
            // to be included before be considered garbage collected.
            consensus_gc_depth: Some(60),

            consensus_commit_rate_estimation_window_size: Some(10),

            consensus_bad_nodes_stake_threshold: Some(30),

            mysticeti_num_leaders_per_round: Some(1),

            max_transactions_per_checkpoint: Some(20_000),
            max_checkpoint_size_bytes: Some(30 * 1024 * 1024),

            buffer_stake_for_protocol_upgrade_bps: Some(5000),

            epoch_duration_ms: Some(24 * 60 * 60), // 1 day

            // Reward parameters
            validator_reward_allocation_bps: Some(7000), // 70% of validator rewards
            reward_slashing_rate_bps: Some(5000),        // 50%

            // Model parameters
            model_min_stake: Some(1_000_000_000), // 1 SOMA (in shannons)
            model_architecture_version: Some(1),
            model_reveal_slash_rate_bps: Some(5000), // 50%
            model_tally_slash_rate_bps: Some(9500),  // 95%

            // Fee parameters
            target_epoch_fee_collection: Some(1_000_000_000),
            base_fee: Some(1000),
            write_object_fee: Some(300),
            initial_value_fee_bps: Some(10),     // 0.1%
            min_value_fee_bps: Some(1),          // 0.01%
            max_value_fee_bps: Some(100),        // 1%
            fee_adjustment_rate_bps: Some(1250), // 12.5%

            // Target/Mining parameters
            target_models_per_target: Some(3), // 3 models per target
            target_embedding_dim: Some(768),   // Standard transformer embedding dim
            target_initial_distance_threshold: Some(BcsF32(0.5)), // Cosine distance (0-2 range)
            target_reward_allocation_bps: Some(8000), // 80% of emissions to targets
            target_hit_rate_target_bps: Some(8000), // 80% target hit rate
            target_hit_rate_ema_decay_bps: Some(9000), // 90% decay (10% weight on new data)
            target_difficulty_adjustment_rate_bps: Some(500), // 5% max adjustment per epoch
            target_max_distance_threshold: Some(BcsF32(1.0)), // Max distance (easiest)
            target_min_distance_threshold: Some(BcsF32(0.1)), // Min distance (hardest)
            target_initial_targets_per_epoch: Some(20), // 20 targets at genesis and each epoch start

            // Reward distribution parameters
            target_miner_reward_share_bps: Some(5000), // 50% to miner
            target_model_reward_share_bps: Some(3000), // 30% to model owner
            target_claimer_incentive_bps: Some(100),   // 1% to claimer as incentive

            // Submission parameters
            submission_bond_per_byte: Some(10), // 10 shannons per byte

            // Challenge parameters
            challenger_bond_per_byte: Some(5), // 5 shannons per byte (half of submission bond)

            // Data size limits
            max_submission_data_size: Some(1024 * 1024 * 1024), // 1 GiB max data size

                                                                // When adding a new constant, set it to None in the earliest version, like this:
                                                                // new_constant: None,
        };
        if version.0 >= 2 {
            panic!("unsupported version {:?}", version);
        }

        // Simtest specific overrides.
        if cfg!(msim) {
            // Trigger GC more often.
            cfg.consensus_gc_depth = Some(5);

            cfg.epoch_duration_ms = Some(1000 * 60);
        }

        cfg
    }

    /// Get the value ProtocolConfig that are in effect during the given protocol version.
    /// Or none if the version is not supported.
    pub fn get_for_version_if_supported(version: ProtocolVersion, chain: Chain) -> Option<Self> {
        if version.0 >= ProtocolVersion::MIN.0 && version.0 <= ProtocolVersion::MAX_ALLOWED.0 {
            let mut ret = Self::get_for_version_impl(version, chain);
            ret.version = version;
            Some(ret)
        } else {
            None
        }
    }
}

impl ProtocolConfig {
    pub fn max_transaction_size_bytes(&self) -> u64 {
        self.consensus_max_transaction_size_bytes.unwrap_or(256 * 1024)
    }

    pub fn max_transactions_in_block_bytes(&self) -> u64 {
        if cfg!(msim) {
            256 * 1024
        } else {
            self.consensus_max_transactions_in_block_bytes.unwrap_or(512 * 1024)
        }
    }

    pub fn max_num_transactions_in_block(&self) -> u64 {
        if cfg!(msim) { 8 } else { self.consensus_max_num_transactions_in_block.unwrap_or(512) }
    }

    pub fn gc_depth(&self) -> u32 {
        self.consensus_gc_depth.unwrap_or(0)
    }

    pub fn get_consensus_commit_rate_estimation_window_size(&self) -> u32 {
        self.consensus_commit_rate_estimation_window_size.unwrap_or(0)
    }

    pub fn consensus_num_requested_prior_commits_at_startup(&self) -> u32 {
        // Currently there is only one parameter driving this value. If there are multiple
        // things computed from prior consensus commits, this function must return the max
        // of all of them.
        self.get_consensus_commit_rate_estimation_window_size()
    }

    /// Build SystemParameters from protocol config.
    /// Note: value_fee_bps is dynamic and may differ from initial_value_fee_bps
    /// after fee adjustments. Pass the current value when updating.
    pub fn build_system_parameters(&self, current_value_fee_bps: Option<u64>) -> SystemParameters {
        SystemParameters {
            epoch_duration_ms: self.epoch_duration_ms(),
            validator_reward_allocation_bps: self.validator_reward_allocation_bps(),
            model_min_stake: self.model_min_stake(),
            model_architecture_version: self.model_architecture_version(),
            model_reveal_slash_rate_bps: self.model_reveal_slash_rate_bps(),
            model_tally_slash_rate_bps: self.model_tally_slash_rate_bps(),
            target_epoch_fee_collection: self.target_epoch_fee_collection(),
            base_fee: self.base_fee(),
            write_object_fee: self.write_object_fee(),
            // Use current value if provided (preserves fee adjustments), otherwise use initial
            value_fee_bps: current_value_fee_bps.unwrap_or_else(|| self.initial_value_fee_bps()),
            min_value_fee_bps: self.min_value_fee_bps(),
            max_value_fee_bps: self.max_value_fee_bps(),
            fee_adjustment_rate_bps: self.fee_adjustment_rate_bps(),
            // Target/Mining parameters
            target_models_per_target: self.target_models_per_target(),
            target_embedding_dim: self.target_embedding_dim(),
            target_initial_distance_threshold: SomaTensor::scalar(
                self.target_initial_distance_threshold().value(),
            ),
            target_reward_allocation_bps: self.target_reward_allocation_bps(),
            target_hit_rate_target_bps: self.target_hit_rate_target_bps(),
            target_hit_rate_ema_decay_bps: self.target_hit_rate_ema_decay_bps(),
            target_difficulty_adjustment_rate_bps: self.target_difficulty_adjustment_rate_bps(),
            target_max_distance_threshold: SomaTensor::scalar(
                self.target_max_distance_threshold().value(),
            ),
            target_min_distance_threshold: SomaTensor::scalar(
                self.target_min_distance_threshold().value(),
            ),
            target_initial_targets_per_epoch: self.target_initial_targets_per_epoch(),
            target_miner_reward_share_bps: self.target_miner_reward_share_bps(),
            target_model_reward_share_bps: self.target_model_reward_share_bps(),
            target_claimer_incentive_bps: self.target_claimer_incentive_bps(),
            submission_bond_per_byte: self.submission_bond_per_byte(),
            // Challenge parameters
            challenger_bond_per_byte: self.challenger_bond_per_byte(),
            // Data size limits
            max_submission_data_size: self.max_submission_data_size(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SystemParameters {
    /// The duration of an epoch, in milliseconds.
    pub epoch_duration_ms: u64,

    /// Percentage of epoch rewards allocated to validators (bps, e.g. 7000 = 70%)
    pub validator_reward_allocation_bps: u64,

    // === Model Parameters ===
    /// Minimum stake required to commit a new model (in shannons)
    pub model_min_stake: u64,
    /// Required architecture version for new model commits
    pub model_architecture_version: u64,
    /// Slash rate for models that fail to reveal after commit (bps, e.g. 5000 = 50%)
    pub model_reveal_slash_rate_bps: u64,
    /// Slash rate for models removed via tally reports (bps, e.g. 9500 = 95%)
    pub model_tally_slash_rate_bps: u64,

    // === Fee Parameters ===
    /// Target fee collection per epoch (network adjusts fees to hit this)
    pub target_epoch_fee_collection: u64,

    /// Base fee per transaction (in shannons)
    pub base_fee: u64,

    /// Fee per object write (in shannons)
    pub write_object_fee: u64,

    /// Current value fee rate in basis points (e.g., 10 = 0.1%)
    pub value_fee_bps: u64,

    /// Minimum value fee rate (floor)
    pub min_value_fee_bps: u64,

    /// Maximum value fee rate (ceiling)
    pub max_value_fee_bps: u64,

    /// Max adjustment per epoch in basis points (e.g., 1250 = 12.5% max change)
    pub fee_adjustment_rate_bps: u64,

    // === Target/Mining Parameters ===
    /// Number of models assigned to each target
    pub target_models_per_target: u64,
    /// Dimension of target embedding vectors
    pub target_embedding_dim: u64,
    /// Initial distance threshold for new targets (cosine distance as scalar SomaTensor)
    pub target_initial_distance_threshold: SomaTensor,
    /// Percentage of epoch emissions allocated to targets (bps)
    pub target_reward_allocation_bps: u64,
    /// Target hit rate (bps, e.g. 8000 = 80% of targets should be filled)
    /// Difficulty adjusts toward this rate.
    pub target_hit_rate_target_bps: u64,
    /// Decay factor for hit rate EMA (bps, e.g. 9000 = 90% decay)
    pub target_hit_rate_ema_decay_bps: u64,
    /// Difficulty adjustment rate per epoch (bps)
    pub target_difficulty_adjustment_rate_bps: u64,
    /// Maximum distance threshold (cap, as scalar SomaTensor)
    pub target_max_distance_threshold: SomaTensor,
    /// Minimum distance threshold (floor, as scalar SomaTensor)
    pub target_min_distance_threshold: SomaTensor,
    /// Number of targets to issue at genesis and at each epoch start
    pub target_initial_targets_per_epoch: u64,

    // === Reward Distribution Parameters ===
    /// Percentage of target reward allocated to the miner (bps, e.g. 5000 = 50%)
    pub target_miner_reward_share_bps: u64,
    /// Percentage of target reward allocated to the model owner (bps, e.g. 3000 = 30%)
    pub target_model_reward_share_bps: u64,
    /// Percentage of target reward given to the claimer as incentive (bps, e.g. 100 = 1%)
    pub target_claimer_incentive_bps: u64,

    // === Submission Parameters ===
    /// Bond per byte of submitted data (in shannons)
    /// Bond is held on the Target and returned to miner on successful claim,
    /// or forfeited to emission pool on successful challenge.
    pub submission_bond_per_byte: u64,

    // === Challenge Parameters ===
    /// Bond per byte for challengers (in shannons)
    /// Challenger must post challenger_bond_per_byte * data_size as bond.
    /// If challenger wins: miner's bond is slashed, challenger gets reward.
    /// If challenger loses: challenger's bond is slashed.
    pub challenger_bond_per_byte: u64,

    // === Data Size Limits ===
    /// Maximum data size allowed for submissions (in bytes).
    /// Submissions exceeding this size will be rejected.
    /// This also affects audit timeouts which are calculated based on data size.
    pub max_submission_data_size: u64,
}
