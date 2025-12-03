//! # Base Type Definitions
//!
//! ## Overview
//! This module defines the fundamental types and data structures used throughout the
//! Soma blockchain system. It serves as a foundation layer for all components.
//!
//! ## Responsibilities
//! - Define primitive types for blockchain operations (timestamps, rounds, sequence numbers)
//! - Provide address representation and manipulation (SomaAddress)
//! - Define object identity types and references (ObjectID, FullObjectID)
//! - Implement common utility types and traits for the system
//!
//! ## Component Relationships
//! - Used by virtually all other modules in the system
//! - Relies on crypto module for cryptographic primitives
//! - Interacts with object system through ObjectID types
//!
//! ## Key Workflows
//! 1. Address derivation from public keys for transaction verification
//! 2. Object identification and referencing across the system
//! 3. Type conversion between various blockchain identifiers
//!
//! ## Design Patterns
//! - Type aliases for semantic clarity (TimestampMs, Round, etc.)
//! - Strong typing for blockchain addresses with validation
//! - Comprehensive serialization/deserialization support for network operations

use crate::crypto::{DefaultHash, GenericSignature, PublicKey, SomaPublicKey, SomaSignature};
use crate::digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest};
use crate::effects::{TransactionEffects, TransactionEffectsAPI as _};
use crate::error::SomaResult;
use crate::object::{ObjectID, Version};
use crate::serde::Readable;
use crate::transaction::{Transaction, VerifiedTransaction};
use crate::{crypto::AuthorityPublicKeyBytes, error::SomaError};
use anyhow::anyhow;
use fastcrypto::encoding::{decode_bytes_hex, Encoding, Hex};
use fastcrypto::hash::HashFunction;
use hex::FromHex;
use rand::rngs::OsRng;
use rand::Rng;
use schemars::JsonSchema;
use serde::Deserializer;
use serde::{ser::SerializeSeq, Deserialize, Serialize, Serializer};
use serde_with::{serde_as, DeserializeAs, SerializeAs};
use std::fmt;
use std::str::FromStr;

/// Timestamp in milliseconds.
///
/// Used throughout the system for time-related operations and event ordering.
pub type TimestampMs = u64;

/// The round number in the consensus protocol.
///
/// Rounds are used to track the progress of consensus and identify
/// when certain events occurred in relation to the consensus process.
pub type Round = u64;

/// Identifier for an authority/validator in the network.
///
/// This is based on the public key of the authority and serves
/// as a unique identifier for validators throughout the system.
pub type AuthorityName = AuthorityPublicKeyBytes;

/// A global sequence number assigned to every CommittedSubDag.
///
/// These numbers provide a total ordering of committed operations
/// in the blockchain and are used for synchronization and state tracking.
pub type SequenceNumber = u64;

/// A trait that provides concise representations of potentially large or complex types.
///
/// This trait allows for debug-friendly representations of types that might otherwise
/// be too verbose or contain too much information for practical debugging.
///
/// ## Examples
///
/// ```
/// # trait ConciseableName<'a> {
/// #     type ConciseTypeRef: std::fmt::Debug;
/// #     type ConciseType: std::fmt::Debug;
/// #     fn concise(&'a self) -> Self::ConciseTypeRef;
/// #     fn concise_owned(&self) -> Self::ConciseType;
/// # }
/// # struct LargeStruct { data: Vec<u8>, id: u64 }
/// # impl<'a> ConciseableName<'a> for LargeStruct {
/// #     type ConciseTypeRef = &'a u64;
/// #     type ConciseType = u64;
/// #     fn concise(&'a self) -> Self::ConciseTypeRef { &self.id }
/// #     fn concise_owned(&self) -> Self::ConciseType { self.id }
/// # }
/// # let large_struct = LargeStruct { data: vec![0; 1000], id: 42 };
/// // Instead of printing the entire large structure:
/// // println!("{:?}", large_struct);
///
/// // Just print a concise representation:
/// // println!("{:?}", large_struct.concise());
/// ```
pub trait ConciseableName<'a> {
    /// Reference type returned by the concise method
    type ConciseTypeRef: std::fmt::Debug;

    /// Owned type returned by the concise_owned method
    type ConciseType: std::fmt::Debug;

    /// Returns a reference to a concise representation of the object
    fn concise(&'a self) -> Self::ConciseTypeRef;

    /// Returns an owned concise representation of the object
    fn concise_owned(&self) -> Self::ConciseType;
}

/// A wrapper around a single element that acts like a vector of size 1.
///
/// This container ensures protocol messages that currently require exactly one element
/// have this constraint enforced at the serialization/deserialization level,
/// while allowing for potential future expansion to multiple elements without
/// changing the types.
///
/// ## Thread Safety
/// This struct does not implement any internal synchronization. When shared across
/// threads, external synchronization should be used if mutation is required.
///
/// ## Examples
///
/// ```
/// # use serde::{Serialize, Deserialize};
/// # #[derive(Debug, Deserialize, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
/// # #[serde(try_from = "Vec<T>")]
/// # struct SizeOneVec<T> { e: T }
/// # impl<T> SizeOneVec<T> {
/// #     pub fn new(e: T) -> Self { Self { e } }
/// #     pub fn element(&self) -> &T { &self.e }
/// #     pub fn into_inner(self) -> T { self.e }
/// # }
/// # let value = 42;
/// // Create a new SizeOneVec containing a single integer
/// let vec = SizeOneVec::new(value);
///
/// // Access the contained element
/// assert_eq!(*vec.element(), 42);
///
/// // Extract the inner value
/// let inner = vec.into_inner();
/// assert_eq!(inner, 42);
/// ```
#[derive(Debug, Deserialize, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[serde(try_from = "Vec<T>")]
pub struct SizeOneVec<T> {
    /// The single element contained in this collection
    e: T,
}

impl<T> SizeOneVec<T> {
    /// Creates a new SizeOneVec containing a single element.
    ///
    /// # Arguments
    /// * `e` - The element to store in the collection.
    ///
    /// # Returns
    /// A new SizeOneVec instance containing the provided element.
    pub fn new(e: T) -> Self {
        Self { e }
    }

    /// Returns a reference to the contained element.
    ///
    /// # Returns
    /// A reference to the single element in this collection.
    pub fn element(&self) -> &T {
        &self.e
    }

    /// Returns a mutable reference to the contained element.
    ///
    /// # Returns
    /// A mutable reference to the single element in this collection.
    pub fn element_mut(&mut self) -> &mut T {
        &mut self.e
    }

    /// Consumes the SizeOneVec and returns the contained element.
    ///
    /// # Returns
    /// The element that was contained in this collection.
    pub fn into_inner(self) -> T {
        self.e
    }

    /// Returns an iterator that yields a reference to the single element.
    ///
    /// # Returns
    /// An iterator that produces exactly one element - a reference to the
    /// contained value.
    pub fn iter(&self) -> std::iter::Once<&T> {
        std::iter::once(&self.e)
    }
}

impl<T> Serialize for SizeOneVec<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(1))?;
        seq.serialize_element(&self.e)?;
        seq.end()
    }
}

impl<T> TryFrom<Vec<T>> for SizeOneVec<T> {
    type Error = anyhow::Error;

    fn try_from(mut v: Vec<T>) -> Result<Self, Self::Error> {
        if v.len() != 1 {
            Err(anyhow!("Expected a vec of size 1"))
        } else {
            Ok(SizeOneVec {
                e: v.pop().unwrap(),
            })
        }
    }
}

/// The length of a Soma address in bytes.
pub const SOMA_ADDRESS_LENGTH: usize = 32;

/// Represents an address in the Soma blockchain.
///
/// A SomaAddress is a 32-byte identifier that can represent various entities
/// in the blockchain, such as:
/// - User accounts
/// - Smart contracts
/// - On-chain resources
///
/// Addresses can be derived from public keys (for user accounts),
/// object IDs (for certain system objects), or generated randomly
/// (for testing or initial allocation).
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
///
/// ## Examples
///
/// ```
/// # use serde::{Serialize, Deserialize};
/// # use schemars::JsonSchema;
/// # use serde_with::serde_as;
/// # const SOMA_ADDRESS_LENGTH: usize = 32;
/// # #[serde_as]
/// # #[derive(Eq, Default, PartialEq, Copy, Clone)]
/// # pub struct SomaAddress([u8; SOMA_ADDRESS_LENGTH]);
/// # impl SomaAddress {
/// #     pub fn random() -> Self { Self([0; SOMA_ADDRESS_LENGTH]) }
/// #     pub const ZERO: Self = Self([0; SOMA_ADDRESS_LENGTH]);
/// # }
/// // Create a zero address
/// let zero_addr = SomaAddress::ZERO;
///
/// // Generate a random address (useful for testing)
/// let random_addr = SomaAddress::random();
///
/// // Typically addresses would be derived from public keys
/// // or parsed from hex strings in actual usage
/// ```
#[serde_as]
#[derive(Eq, Default, PartialEq, Ord, PartialOrd, Copy, Clone, Hash, JsonSchema)]
pub struct SomaAddress(
    /// The raw bytes of the address
    #[schemars(with = "Hex")]
    #[serde_as(as = "Readable<Hex, _>")]
    [u8; SOMA_ADDRESS_LENGTH],
);

impl SomaAddress {
    /// The byte length of a SomaAddress
    pub const LENGTH: usize = SOMA_ADDRESS_LENGTH;

    /// A constant representing the zero address (all bytes set to 0)
    pub const ZERO: Self = Self([0u8; SOMA_ADDRESS_LENGTH]);

    /// Creates a new SomaAddress from a byte array.
    ///
    /// # Arguments
    /// * `address` - A 32-byte array containing the address data
    ///
    /// # Returns
    /// A new SomaAddress instance
    pub const fn new(address: [u8; Self::LENGTH]) -> Self {
        Self(address)
    }

    /// Converts the address to a vector of bytes.
    ///
    /// # Returns
    /// A vector containing a copy of the address bytes
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.to_vec()
    }

    /// Generates a random SomaAddress using the system's secure random number generator.
    ///
    /// This is primarily useful for testing purposes.
    ///
    /// # Returns
    /// A randomly generated SomaAddress
    pub fn random() -> Self {
        let mut rng = OsRng;
        let buf: [u8; Self::LENGTH] = rng.gen();
        Self(buf)
    }

    /// Generates a random SomaAddress using the provided random number generator.
    ///
    /// # Arguments
    /// * `rng` - A cryptographically secure random number generator
    ///
    /// # Returns
    /// A randomly generated SomaAddress
    pub fn generate<R: rand::RngCore + rand::CryptoRng>(mut rng: R) -> Self {
        let buf: [u8; SOMA_ADDRESS_LENGTH] = rng.gen();
        Self(buf)
    }

    /// Returns the underlying byte array of the SomaAddress.
    ///
    /// # Returns
    /// The raw 32-byte array that makes up this address
    pub fn to_inner(self) -> [u8; SOMA_ADDRESS_LENGTH] {
        self.0
    }

    /// Parses a SomaAddress from a byte array or buffer.
    ///
    /// # Arguments
    /// * `bytes` - A byte array or buffer that can be converted to a reference to bytes
    ///
    /// # Returns
    /// A Result containing either the parsed SomaAddress or an error if the input
    /// doesn't have exactly 32 bytes
    ///
    /// # Errors
    /// Returns SomaError::InvalidAddress if the provided bytes aren't exactly 32 bytes
    ///
    /// # Examples
    ///
    /// ```
    /// # struct SomaAddress([u8; 32]);
    /// # impl SomaAddress {
    /// #     pub fn from_bytes<T: AsRef<[u8]>>(bytes: T) -> Result<Self, ()> {
    /// #         let bytes_ref = bytes.as_ref();
    /// #         if bytes_ref.len() != 32 {
    /// #             return Err(());
    /// #         }
    /// #         let mut arr = [0u8; 32];
    /// #         arr.copy_from_slice(bytes_ref);
    /// #         Ok(Self(arr))
    /// #     }
    /// # }
    /// # let some_bytes = [1u8; 32];
    /// // Parse from a byte array
    /// let addr = SomaAddress::from_bytes([1u8; 32]).unwrap();
    ///
    /// // Parse from a vector
    /// let vec_bytes = vec![1u8; 32];
    /// let addr = SomaAddress::from_bytes(&vec_bytes).unwrap();
    /// ```
    pub fn from_bytes<T: AsRef<[u8]>>(bytes: T) -> Result<Self, AccountAddressParseError> {
        <[u8; Self::LENGTH]>::try_from(bytes.as_ref())
            .map_err(|_| AccountAddressParseError)
            .map(Self)
    }

    pub fn from_hex_literal(literal: &str) -> Result<Self, AccountAddressParseError> {
        if !literal.starts_with("0x") {
            return Err(AccountAddressParseError);
        }

        let hex_len = literal.len() - 2;

        // If the string is too short, pad it
        if hex_len < Self::LENGTH * 2 {
            let mut hex_str = String::with_capacity(Self::LENGTH * 2);
            for _ in 0..Self::LENGTH * 2 - hex_len {
                hex_str.push('0');
            }
            hex_str.push_str(&literal[2..]);
            SomaAddress::from_hex(hex_str)
        } else {
            SomaAddress::from_hex(&literal[2..])
        }
    }

    /// Return a canonical string representation of the address
    /// Addresses are hex-encoded lowercase values of length ADDRESS_LENGTH (16, 20, or 32 depending on the Move platform)
    /// e.g., 0000000000000000000000000000000a, *not* 0x0000000000000000000000000000000a, 0xa, or 0xA
    pub fn to_canonical_string(&self) -> String {
        hex::encode(self.0)
    }

    pub fn short_str_lossless(&self) -> String {
        let hex_str = hex::encode(self.0).trim_start_matches('0').to_string();
        if hex_str.is_empty() {
            "0".to_string()
        } else {
            hex_str
        }
    }

    pub fn to_hex_literal(&self) -> String {
        format!("0x{}", self.short_str_lossless())
    }

    pub fn from_hex<T: AsRef<[u8]>>(hex: T) -> Result<Self, AccountAddressParseError> {
        <[u8; Self::LENGTH]>::from_hex(hex)
            .map_err(|_| AccountAddressParseError)
            .map(Self)
    }

    pub fn to_hex(&self) -> String {
        format!("{:x}", self)
    }
}

impl fmt::LowerHex for SomaAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "0x")?;
        }

        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }

        Ok(())
    }
}

impl fmt::UpperHex for SomaAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "0x")?;
        }

        for byte in &self.0 {
            write!(f, "{:02X}", byte)?;
        }

        Ok(())
    }
}

impl<T: SomaPublicKey> From<&T> for SomaAddress {
    fn from(pk: &T) -> Self {
        let mut hasher = DefaultHash::default();
        hasher.update([T::SIGNATURE_SCHEME.flag()]);
        hasher.update(pk);
        let g_arr = hasher.finalize();
        SomaAddress(g_arr.digest)
    }
}

impl From<&PublicKey> for SomaAddress {
    fn from(pk: &PublicKey) -> Self {
        let mut hasher = DefaultHash::default();
        hasher.update([pk.flag()]);
        hasher.update(pk);
        let g_arr = hasher.finalize();
        SomaAddress(g_arr.digest)
    }
}

impl TryFrom<&GenericSignature> for SomaAddress {
    type Error = SomaError;
    /// Derive a SomaAddress from a serialized signature in Soma [GenericSignature].
    fn try_from(sig: &GenericSignature) -> SomaResult<Self> {
        match sig {
            GenericSignature::Signature(sig) => {
                let scheme = sig.scheme();
                let pub_key_bytes = sig.public_key_bytes();
                let pub_key = PublicKey::try_from_bytes(scheme, pub_key_bytes).map_err(|_| {
                    SomaError::InvalidSignature {
                        error: "Cannot parse pubkey".to_string(),
                    }
                })?;
                Ok(SomaAddress::from(&pub_key))
            }
        }
    }
}

impl From<ObjectID> for SomaAddress {
    fn from(object_id: ObjectID) -> SomaAddress {
        Self(object_id.into_bytes())
    }
}

impl AsRef<[u8]> for SomaAddress {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl std::ops::Deref for SomaAddress {
    type Target = [u8; Self::LENGTH];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for SomaAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:x}", self)
    }
}

impl fmt::Debug for SomaAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}", self)
    }
}

impl From<[u8; SomaAddress::LENGTH]> for SomaAddress {
    fn from(bytes: [u8; SomaAddress::LENGTH]) -> Self {
        Self::new(bytes)
    }
}

/// Hex serde for AccountAddress
pub(crate) struct HexAccountAddress;

impl SerializeAs<SomaAddress> for HexAccountAddress {
    fn serialize_as<S>(value: &SomaAddress, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Hex::serialize_as(value, serializer)
    }
}

impl<'de> DeserializeAs<'de, SomaAddress> for HexAccountAddress {
    fn deserialize_as<D>(deserializer: D) -> Result<SomaAddress, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.starts_with("0x") {
            SomaAddress::from_hex_literal(&s).map_err(<D::Error as serde::de::Error>::custom)
        } else {
            SomaAddress::from_hex(&s).map_err(<D::Error as serde::de::Error>::custom)
        }
    }
}

impl TryFrom<&[u8]> for SomaAddress {
    type Error = AccountAddressParseError;

    /// Tries to convert the provided byte array into Address.
    fn try_from(bytes: &[u8]) -> Result<SomaAddress, AccountAddressParseError> {
        Self::from_bytes(bytes)
    }
}

impl TryFrom<Vec<u8>> for SomaAddress {
    type Error = AccountAddressParseError;

    /// Tries to convert the provided byte buffer into Address.
    fn try_from(bytes: Vec<u8>) -> Result<SomaAddress, AccountAddressParseError> {
        Self::from_bytes(bytes)
    }
}

impl From<SomaAddress> for Vec<u8> {
    fn from(addr: SomaAddress) -> Vec<u8> {
        addr.0.to_vec()
    }
}

impl From<&SomaAddress> for Vec<u8> {
    fn from(addr: &SomaAddress) -> Vec<u8> {
        addr.0.to_vec()
    }
}

impl From<SomaAddress> for [u8; SomaAddress::LENGTH] {
    fn from(addr: SomaAddress) -> Self {
        addr.0
    }
}

impl From<&SomaAddress> for [u8; SomaAddress::LENGTH] {
    fn from(addr: &SomaAddress) -> Self {
        addr.0
    }
}

impl From<&SomaAddress> for String {
    fn from(addr: &SomaAddress) -> String {
        ::hex::encode(addr.as_ref())
    }
}

impl TryFrom<String> for SomaAddress {
    type Error = AccountAddressParseError;

    fn try_from(s: String) -> Result<SomaAddress, AccountAddressParseError> {
        Self::from_hex(s)
    }
}

impl FromStr for SomaAddress {
    type Err = AccountAddressParseError;

    fn from_str(s: &str) -> Result<Self, AccountAddressParseError> {
        // Accept 0xADDRESS or ADDRESS
        if let Ok(address) = SomaAddress::from_hex_literal(s) {
            Ok(address)
        } else {
            Self::from_hex(s)
        }
    }
}

impl<'de> Deserialize<'de> for SomaAddress {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = <String>::deserialize(deserializer)?;
            SomaAddress::from_str(&s).map_err(<D::Error as serde::de::Error>::custom)
        } else {
            // In order to preserve the Serde data model and help analysis tools,
            // make sure to wrap our value in a container with the same name
            // as the original type.
            #[derive(::serde::Deserialize)]
            #[serde(rename = "AccountAddress")]
            struct Value([u8; SomaAddress::LENGTH]);

            let value = Value::deserialize(deserializer)?;
            Ok(SomaAddress::new(value.0))
        }
    }
}

impl Serialize for SomaAddress {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            self.to_hex().serialize(serializer)
        } else {
            // See comment in deserialize.
            serializer.serialize_newtype_struct("AccountAddress", &self.0)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AccountAddressParseError;

impl fmt::Display for AccountAddressParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Unable to parse AccountAddress (must be hex string of length {})",
            SomaAddress::LENGTH
        )
    }
}

impl std::error::Error for AccountAddressParseError {}

/// Generate a fake SomaAddress with repeated one byte.
pub fn dbg_addr(name: u8) -> SomaAddress {
    let addr = [name; SOMA_ADDRESS_LENGTH];
    SomaAddress(addr)
}

/// Represents a complete object identifier that can distinguish between different
/// processing paths for objects.
///
/// FullObjectID can represent either:
/// - A simple object ID (Fastpath) for non-shared objects
/// - A composite key (Consensus) for shared objects that includes both the
///   object ID and a starting version
///
/// This distinction is important for determining how objects are processed
/// in the system, particularly for transaction ordering and execution.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[serde_as]
#[derive(Debug, Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FullObjectID {
    /// Simple object identifier for non-shared objects
    Fastpath(ObjectID),

    /// Combined object ID and starting version for shared objects
    Consensus(ConsensusObjectSequenceKey),
}

impl FullObjectID {
    /// Creates a new FullObjectID from an object ID and optional start version.
    ///
    /// If a start version is provided, creates a Consensus variant, otherwise creates
    /// a Fastpath variant.
    ///
    /// # Arguments
    /// * `object_id` - The unique identifier of the object
    /// * `start_version` - Optional starting version for shared objects
    ///
    /// # Returns
    /// A FullObjectID instance representing either a Fastpath or Consensus object
    ///
    /// # Examples
    ///
    /// ```
    /// # struct ObjectID([u8; 32]);
    /// # type Version = u64;
    /// # enum FullObjectID {
    /// #    Fastpath(ObjectID),
    /// #    Consensus((ObjectID, Version)),
    /// # }
    /// # impl FullObjectID {
    /// #    fn new(object_id: ObjectID, start_version: Option<Version>) -> Self {
    /// #        if let Some(start_version) = start_version {
    /// #            Self::Consensus((object_id, start_version))
    /// #        } else {
    /// #            Self::Fastpath(object_id)
    /// #        }
    /// #    }
    /// # }
    /// # let object_id = ObjectID([0; 32]);
    /// // Create a Fastpath object ID (non-shared object)
    /// let fastpath_id = FullObjectID::new(object_id, None);
    ///
    /// // Create a Consensus object ID (shared object)
    /// let consensus_id = FullObjectID::new(object_id, Some(1));
    /// ```
    pub fn new(object_id: ObjectID, start_version: Option<Version>) -> Self {
        if let Some(start_version) = start_version {
            Self::Consensus((object_id, start_version))
        } else {
            Self::Fastpath(object_id)
        }
    }

    /// Extracts the base object ID from this FullObjectID.
    ///
    /// This method returns just the object ID component regardless of whether
    /// this is a Fastpath or Consensus variant.
    ///
    /// # Returns
    /// The ObjectID component of this FullObjectID
    pub fn id(&self) -> ObjectID {
        match &self {
            FullObjectID::Fastpath(object_id) => *object_id,
            FullObjectID::Consensus(consensus_object_sequence_key) => {
                consensus_object_sequence_key.0
            }
        }
    }
}

/// A complete reference to an object, including its identity, version, and digest.
///
/// This type combines:
/// - A FullObjectID (either Fastpath or Consensus)
/// - A specific Version of the object
/// - An ObjectDigest that represents the content hash of the object
///
/// FullObjectRef is used throughout the system to uniquely identify a specific
/// version of an object with its content hash for verification.
pub type FullObjectRef = (FullObjectID, Version, ObjectDigest);

/// Represents a distinct stream of object versions for a Shared or ConsensusV2 object.
///
/// This key consists of:
/// - The object's unique ID
/// - A starting version that identifies the sequence of updates
///
/// For shared objects or objects using the ConsensusV2 processing path, this
/// key helps track and order the sequence of operations on the object.
pub type ConsensusObjectSequenceKey = (ObjectID, Version);

#[derive(
    Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Hash, Serialize, Deserialize, JsonSchema, Debug,
)]
pub struct ExecutionDigests {
    pub transaction: TransactionDigest,
    pub effects: TransactionEffectsDigest,
}

impl ExecutionDigests {
    pub fn new(transaction: TransactionDigest, effects: TransactionEffectsDigest) -> Self {
        Self {
            transaction,
            effects,
        }
    }

    pub fn random() -> Self {
        Self {
            transaction: TransactionDigest::random(),
            effects: TransactionEffectsDigest::random(),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct ExecutionData {
    pub transaction: Transaction,
    pub effects: TransactionEffects,
}

impl ExecutionData {
    pub fn new(transaction: Transaction, effects: TransactionEffects) -> ExecutionData {
        debug_assert_eq!(transaction.digest(), effects.transaction_digest());
        Self {
            transaction,
            effects,
        }
    }

    pub fn digests(&self) -> ExecutionDigests {
        self.effects.execution_digests()
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct VerifiedExecutionData {
    pub transaction: VerifiedTransaction,
    pub effects: TransactionEffects,
}

impl VerifiedExecutionData {
    pub fn new(transaction: VerifiedTransaction, effects: TransactionEffects) -> Self {
        debug_assert_eq!(transaction.digest(), effects.transaction_digest());
        Self {
            transaction,
            effects,
        }
    }

    pub fn new_unchecked(data: ExecutionData) -> Self {
        Self {
            transaction: VerifiedTransaction::new_unchecked(data.transaction),
            effects: data.effects,
        }
    }

    pub fn into_inner(self) -> ExecutionData {
        ExecutionData {
            transaction: self.transaction.into_inner(),
            effects: self.effects,
        }
    }

    pub fn digests(&self) -> ExecutionDigests {
        self.effects.execution_digests()
    }
}
