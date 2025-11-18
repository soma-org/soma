use serde::{Deserialize, Serialize};

/// The minimum and maximum protocol versions supported by this build.
const MIN_PROTOCOL_VERSION: u64 = 1;
const MAX_PROTOCOL_VERSION: u64 = 1;

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

#[derive(Clone, Serialize, Debug, Default)]
pub struct ProtocolConfig {
    /// Minimum interval of commit timestamps between consecutive checkpoints.
    min_checkpoint_interval_ms: Option<u64>,
}
