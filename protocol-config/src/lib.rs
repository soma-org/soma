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

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Copy, PartialOrd, Ord, Eq)]
pub enum Chain {
    Mainnet,
    Testnet,
    Unknown,
}

impl Default for Chain {
    fn default() -> Self {
        Self::Unknown
    }
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
            // When adding a new constant, set it to None in the earliest version, like this:
            // new_constant: None,

            //   cfg.consensus_max_transaction_size_bytes = Some(256 * 1024); // 256KB
            //    // Assume 1KB per transaction and 500 transactions per block.
            //         cfg.consensus_max_transactions_in_block_bytes = Some(512 * 1024);
            //         // Assume 20_000 TPS * 5% max stake per validator / (minimum) 4 blocks per round = 250 transactions per block maximum
            //         // Using a higher limit that is 512, to account for bursty traffic and system transactions.
            //         cfg.consensus_max_num_transactions_in_block = Some(512);
            // max_tx_size_bytes: Some(128 * 1024),
            //     cfg.consensus_voting_rounds = Some(40);

            //      cfg.consensus_gc_depth = Some(60);
            //     cfg.consensus_bad_nodes_stake_threshold = Some(30)
            //     cfg.max_transactions_per_checkpoint = Some(20_000);

            // reward_slashing_rate: Some(5000),
            // max_checkpoint_size_bytes: Some(30 * 1024 * 1024),
        };
        for cur in 2..=version.0 {
            match cur {
                1 => unreachable!(),
                // Use this template when making changes:
                //
                //     // modify an existing constant.
                //     move_binary_format_version: Some(7),
                //
                //     // Add a new constant (which is set to None in prior versions).
                //     new_constant: Some(new_value),
                //
                //     // Remove a constant (ensure that it is never accessed during this version).
                //     max_move_object_size: None,
                _ => panic!("unsupported version {:?}", version),
            }
        }

        // Simtest specific overrides.
        if cfg!(msim) {
            // Trigger GC more often.
            cfg.consensus_gc_depth = Some(5);
        }

        cfg
    }
}

impl ProtocolConfig {
    pub fn max_transaction_size_bytes(&self) -> u64 {
        self.consensus_max_transaction_size_bytes
            .unwrap_or(256 * 1024)
    }

    pub fn max_transactions_in_block_bytes(&self) -> u64 {
        if cfg!(msim) {
            256 * 1024
        } else {
            self.consensus_max_transactions_in_block_bytes
                .unwrap_or(512 * 1024)
        }
    }

    pub fn max_num_transactions_in_block(&self) -> u64 {
        if cfg!(msim) {
            8
        } else {
            self.consensus_max_num_transactions_in_block.unwrap_or(512)
        }
    }

    pub fn gc_depth(&self) -> u32 {
        self.consensus_gc_depth.unwrap_or(0)
    }

    pub fn get_consensus_commit_rate_estimation_window_size(&self) -> u32 {
        self.consensus_commit_rate_estimation_window_size
            .unwrap_or(0)
    }

    pub fn consensus_num_requested_prior_commits_at_startup(&self) -> u32 {
        // Currently there is only one parameter driving this value. If there are multiple
        // things computed from prior consensus commits, this function must return the max
        // of all of them.
        let window_size = self.get_consensus_commit_rate_estimation_window_size();

        window_size
    }
}
