// adapted heavily from code from mysten labs

use eyre::eyre;
use fastcrypto::encoding::decode_bytes_hex;
use serde::{Deserialize, Serialize};
use serde_repr::Deserialize_repr;
use serde_repr::Serialize_repr;
use std::str::FromStr;

/// len(Version, Scope) == 2
const INTENT_PREFIX_LENGTH: usize = 2;

/// The version is to distinguish between signing different versions of the struct
/// or enum. Serialized output between two different versions of the same struct/enum
/// might accidentally (or maliciously on purpose) match.
#[derive(Serialize_repr, Deserialize_repr, Default, Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
enum ScopeVersion {
    #[default]
    V0 = 0,
}

impl TryFrom<u8> for ScopeVersion {
    type Error = eyre::Report;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        bcs::from_bytes(&[value]).map_err(|_| eyre!("Invalid ScopeVersion"))
    }
}

/// This enums specifies the scope. Two different scopes should
/// never collide, so no signature provided for one scope can be used for
/// another, even when the serialized data itself may be the same.
#[derive(Serialize_repr, Deserialize_repr, Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Scope {
    TransactionData = 0,         // Used for a user signature on a transaction data.
    TransactionEffects = 1,      // Used for an authority signature on transaction effects.
    CheckpointSummary = 2,       // Used for an authority signature on a checkpoint summary.
    PersonalMessage = 3,         // Used for a user signature on a personal message.
    SenderSignedTransaction = 4, // Used for an authority signature on a user signed transaction.
    ProofOfPossession = 5, // Used as a signature representing an authority's proof of possession of its authority protocol key.
    HeaderDigest = 6,      // Used for narwhal authority signature on header digest.
    BridgeEventUnused = 7, // for bridge purposes but it's currently not included in messages.
    ConsensusBlockHeader = 8, // Used for consensus authority signature on block's digest
    ShardInput = 9,
    ShardCommit = 10,
    ShardReveal = 11,
    ShardEndorsement = 12,
}

impl TryFrom<u8> for Scope {
    type Error = eyre::Report;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        bcs::from_bytes(&[value]).map_err(|_| eyre!("Invalid Scope"))
    }
}

/// A versioned scope is a compact struct serves as the domain separator for a message that a signature commits to.
/// It consists of two parts: [enum Scope] (what the type of the message is), and [enum ScopeVersion] (what version of scope is being used).
/// It is used to construct [struct ScopedMessage] that what a signature commits to.
///
/// The serialization of an VersionedScope is a 2-byte array where each field is represented by a byte.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Clone, Hash)]
pub(crate) struct VersionedScope {
    scope: Scope,
    scope_version: ScopeVersion,
}

impl VersionedScope {
    /// converts versioned scope into two byte array
    fn to_bytes(&self) -> [u8; INTENT_PREFIX_LENGTH] {
        [self.scope as u8, self.scope_version as u8]
    }

    /// converts bytes into versioned scope using a try_into
    fn from_bytes(bytes: &[u8]) -> Result<Self, eyre::Report> {
        if bytes.len() != INTENT_PREFIX_LENGTH {
            return Err(eyre!("Invalid Intent"));
        }
        Ok(Self {
            scope: bytes[0].try_into()?,
            scope_version: bytes[1].try_into()?,
        })
    }
}

impl FromStr for VersionedScope {
    type Err = eyre::Report;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes: Vec<u8> = decode_bytes_hex(s).map_err(|_| eyre!("Invalid Intent"))?;
        Self::from_bytes(bytes.as_slice())
    }
}

impl VersionedScope {
    /// creates a new versioned scope
    pub(crate) const fn new(scope: Scope) -> Self {
        Self {
            scope_version: ScopeVersion::V0,
            scope,
        }
    }
}

/// Scoped Message is a wrapper around a message with its intent. The message can
/// be any type that implements [trait Serialize]. *ALL* signatures must commit
/// to the scoped message, not the message itself. This guarantees any scoped
/// message signed in the system cannot collide with another since they are domain
/// separated by scope.
///
/// The serialization of an ScopedMessage is compact: it only appends two bytes
/// to the message itself.
#[derive(Debug, PartialEq, Eq, Serialize, Clone, Hash, Deserialize)]
pub struct ScopedMessage<T: Serialize> {
    /// the version and scope of a message
    versioned_scope: VersionedScope,
    ///the actual message value
    value: T,
}

impl<T: Serialize> ScopedMessage<T> {
    /// creates a new scoped message for the given generic message
    pub const fn new(scope: Scope, value: T) -> Self {
        Self {
            versioned_scope: VersionedScope::new(scope),
            value,
        }
    }
}
