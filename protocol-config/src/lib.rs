// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use protocol_config_macros::{
    ProtocolConfigAccessors, ProtocolConfigFeatureFlagsGetters, ProtocolConfigOverride,
};
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

/// The minimum and maximum protocol versions supported by this build.
pub const MIN_PROTOCOL_VERSION: u64 = 1;
/// V7 (Stage 14c.1): `ObjectOut::AccumulatorWriteV1` variant added
/// for SIP-58-style per-tx accumulator delta records. Effects digest
/// format changes — pre-mainnet network-wide flip, no runtime gate.
pub const MAX_PROTOCOL_VERSION: u64 = 7;

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
    // Mainnet,
    Testnet,
    #[default]
    Unknown,
}

impl Chain {
    pub fn as_str(self) -> &'static str {
        match self {
            // Chain::Mainnet => "mainnet",
            Chain::Testnet => "testnet",
            Chain::Unknown => "localnet",
        }
    }
}

pub struct Error(pub String);

#[allow(dead_code)]
fn is_false(b: &bool) -> bool {
    !b
}

#[allow(dead_code)]
fn is_empty(b: &BTreeSet<String>) -> bool {
    b.is_empty()
}

#[allow(dead_code)]
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
    reward_slashing_rate_bps: Option<u64>,

    // === Fee Parameters ===
    /// Per-unit fee in USDC microdollars (1 USDC = 1_000_000 µUSDC).
    /// The fee for a transaction is `unit_fee * executor.fee_units(...)`.
    /// Each executor decides how many units its operation costs based on op shape.
    /// All fees on Soma are paid in USDC; the gas coin must be Coin(Usdc).
    unit_fee: Option<u64>,

    // === Execution Versioning ===
    /// Execution version controls which code paths are used when executing transactions.
    /// Bumped independently of protocol_version when execution logic changes.
    /// This ensures state sync determinism: a node replaying old epochs uses the
    /// execution version from that epoch, even if its binary supports newer versions.
    execution_version: Option<u64>,

    // === Payment-Channel Parameters ===
    /// How long after `RequestClose` the payer must wait before they
    /// can call `WithdrawAfterTimeout` to reclaim the deposit. Gives
    /// the payee a window to submit any final `Settle` before the
    /// channel deletes. Defaults to 10 minutes.
    channel_grace_period_ms: Option<u64>,
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
        #[cfg(msim)]
        {
            // Populate the fake simulator version with a slightly different config
            // so protocol version upgrade tests can verify the version actually changed.
            if version == ProtocolVersion::MAX_ALLOWED {
                let mut config =
                    Self::get_for_version_impl(ProtocolVersion::new(version.as_u64() - 1), chain);
                config.unit_fee = Some(config.unit_fee.unwrap_or(1000) + 1000);
                return config;
            }
        }

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

            epoch_duration_ms: Some(24 * 60 * 60 * 1000), // 1 day (ms)

            // Reward parameters
            reward_slashing_rate_bps: Some(5000), // 50%

            // Fee parameters: 1000 µUSDC = $0.001 per unit. A simple Transfer
            // (1 coin + 1 recipient = 2 units) costs ~$0.002.
            unit_fee: Some(1000),

            // Execution versioning
            execution_version: Some(0), // Initial execution version

            // Payment-channel grace period: 10 minutes. Time the payee
            // has after `RequestClose` to submit any final voucher via
            // `Settle`/`Close` before the payer can `WithdrawAfterTimeout`.
            channel_grace_period_ms: Some(10 * 60 * 1000),

            // When adding a new constant, set it to None in the earliest version, like this:
            // new_constant: None,
        };
        if version.0 >= 2 {
            // V2: Bump execution_version to fix permanent object lock on
            // failed transactions. `ensure_active_inputs_mutated()` is now
            // called so that all mutable owned inputs get their version
            // bumped even when execution fails, releasing epoch-store locks.
            cfg.execution_version = Some(1);
        }
        if version.0 >= 3 {
            cfg.execution_version = Some(2);
        }
        // V6 (Stage 13m): TransactionEffects gains `balance_events` and
        // `delegation_events` so indexers can attribute per-tx changes
        // to the balance accumulator and F1 delegation rows without
        // re-executing transactions. Effects digest format changes;
        // since Soma is pre-mainnet there is no runtime gate — the
        // entire network upgrades atomically.
        //
        // V7 (Stage 14c.1): `ObjectOut::AccumulatorWriteV1` variant —
        // per-tx accumulator delta records ride effects.changed_objects
        // (Sui SIP-58 style). The variant is unused at this version
        // (Stage 14c.2+ migrates executors to emit it); the bump
        // exists to gate the wire-format change.
        if version.0 >= 9 {
            panic!("unsupported version {:?}", version);
        }

        // Simtest specific overrides.
        if cfg!(msim) {
            // Trigger GC more often.
            cfg.consensus_gc_depth = Some(5);

            cfg.epoch_duration_ms = Some(1000 * 60);

            // Set buffer stake to 0 so that protocol upgrades require only a 2/3 quorum
            // (the standard BFT threshold), not 2/3 + 50% buffer. This makes 3/4 validators
            // sufficient for upgrades in tests.
            cfg.buffer_stake_for_protocol_upgrade_bps = Some(0);

            // Shorten the channel grace period so e2e tests don't have
            // to advance the simulated clock by 10 minutes to verify
            // the timed-close path. 5 seconds is plenty under msim,
            // where Clock advances on every consensus commit.
            cfg.channel_grace_period_ms = Some(5_000);
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
    pub fn build_system_parameters(&self) -> SystemParameters {
        SystemParameters {
            epoch_duration_ms: self.epoch_duration_ms(),
            unit_fee: self.unit_fee(),
            channel_grace_period_ms: self.channel_grace_period_ms(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SystemParameters {
    /// The duration of an epoch, in milliseconds.
    pub epoch_duration_ms: u64,

    /// Per-unit fee. Tx fee = `unit_fee * executor.fee_units(...)`.
    pub unit_fee: u64,

    /// Payment-channel forced-close grace period (ms). After
    /// `RequestClose`, the payer must wait this long before
    /// `WithdrawAfterTimeout` succeeds.
    pub channel_grace_period_ms: u64,
}

// =============================================================================
// Snapshot tests — prevent accidental protocol config changes that cause forks
// =============================================================================
//
// These tests use `insta` to capture YAML snapshots of every ProtocolConfig for
// every (Chain, ProtocolVersion) pair.  If any field value changes, the test
// fails until the developer explicitly runs `cargo insta review` to accept the
// new snapshot.
//
// IMPORTANT: Never update snapshots for existing protocol versions.
//            Only add new snapshots for new protocol versions.
//
// Excluded from msim builds because the simulator intentionally mutates config
// values (gc_depth, epoch_duration_ms, buffer_stake_for_protocol_upgrade_bps)
// and adds a fake protocol version, which would cause snapshot mismatches.
#[cfg(all(test, not(msim)))]
mod snapshot_tests {
    use insta::assert_yaml_snapshot;

    use super::*;

    #[test]
    fn protocol_config_version_snapshots() {
        println!("\n============================================================================");
        println!("!                                                                          !");
        println!("! IMPORTANT: never update snapshots from this test. only add new versions! !");
        println!("!                                                                          !");
        println!("============================================================================\n");
        for chain in &[Chain::Unknown, /* Chain::Mainnet, */ Chain::Testnet] {
            // Chain::Unknown uses empty prefix for backward compat with pre-chain snapshots.
            let chain_str = match chain {
                Chain::Unknown => String::new(),
                _ => format!("{:?}_", chain),
            };
            for v in MIN_PROTOCOL_VERSION..=MAX_PROTOCOL_VERSION {
                let version = ProtocolVersion::new(v);
                assert_yaml_snapshot!(
                    format!("{}version_{}", chain_str, version.as_u64()),
                    ProtocolConfig::get_for_version(version, *chain)
                );
            }
        }
    }

    /// Verify that every field in ProtocolConfig is set (not None) for the latest version.
    /// A None field in the latest version likely means a new field was added but not initialized.
    #[test]
    fn all_fields_set_in_latest_version() {
        for chain in &[Chain::Unknown, /* Chain::Mainnet, */ Chain::Testnet] {
            let config = ProtocolConfig::get_for_version(ProtocolVersion::MAX, *chain);
            let attr_map = config.attr_map();
            for (name, value) in &attr_map {
                assert!(
                    value.is_some(),
                    "Field '{}' is None in latest protocol version {} for chain {:?}. \
                     All fields must be initialized in the latest version.",
                    name,
                    MAX_PROTOCOL_VERSION,
                    chain,
                );
            }
        }
    }

    /// Verify version field is set correctly.
    #[test]
    fn version_field_matches() {
        for v in MIN_PROTOCOL_VERSION..=MAX_PROTOCOL_VERSION {
            let version = ProtocolVersion::new(v);
            let config = ProtocolConfig::get_for_version(version, Chain::Unknown);
            assert_eq!(config.version, version);
        }
    }

    /// Verify that feature_map returns a consistent set of flags for each version.
    #[test]
    fn feature_flags_snapshot() {
        for chain in &[Chain::Unknown, /* Chain::Mainnet, */ Chain::Testnet] {
            let chain_str = match chain {
                Chain::Unknown => String::new(),
                _ => format!("{:?}_", chain),
            };
            for v in MIN_PROTOCOL_VERSION..=MAX_PROTOCOL_VERSION {
                let version = ProtocolVersion::new(v);
                let config = ProtocolConfig::get_for_version(version, *chain);
                assert_yaml_snapshot!(
                    format!("{}feature_flags_version_{}", chain_str, version.as_u64()),
                    config.feature_map()
                );
            }
        }
    }

    /// Verify that attr_map returns a consistent set of attributes for each version.
    #[test]
    fn attr_map_snapshot() {
        for chain in &[Chain::Unknown, /* Chain::Mainnet, */ Chain::Testnet] {
            let chain_str = match chain {
                Chain::Unknown => String::new(),
                _ => format!("{:?}_", chain),
            };
            for v in MIN_PROTOCOL_VERSION..=MAX_PROTOCOL_VERSION {
                let version = ProtocolVersion::new(v);
                let config = ProtocolConfig::get_for_version(version, *chain);
                assert_yaml_snapshot!(
                    format!("{}attr_map_version_{}", chain_str, version.as_u64()),
                    config.attr_map()
                );
            }
        }
    }
}
