// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::cmp::max;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;
use std::sync::Arc;

use anyhow::anyhow;
use fastcrypto::encoding::{Encoding, Hex, decode_bytes_hex};
use fastcrypto::hash::HashFunction;
use fastcrypto::traits::AllowedRng;
use rand::Rng;
use rand::rngs::OsRng;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::{Bytes, DeserializeAs, SerializeAs, serde_as};

use crate::base::{
    FullObjectID, FullObjectRef, HexAccountAddress, SOMA_ADDRESS_LENGTH, SomaAddress,
};
use crate::committee::EpochId;
use crate::crypto::{DefaultHash, default_hash};
use crate::digests::{ObjectDigest, TransactionDigest};
use crate::error::{SomaError, SomaResult};
use crate::serde::Readable;
use crate::system_state::staking::StakedSomaV1;
use crate::target::TargetV1;

/// The starting version for all newly created objects
pub const OBJECT_START_VERSION: Version = Version::from_u64(1);
pub const GAS_VALUE_FOR_TESTING: u64 = 300_000_000_000_000;

#[derive(
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Copy,
    Clone,
    Hash,
    Default,
    Debug,
    Serialize,
    Deserialize,
    JsonSchema
)]
pub struct Version(u64);

impl Version {
    /// Minimum possible version (0)
    pub const MIN: Version = Version(u64::MIN);

    /// Maximum valid version for normal operations
    pub const MAX: Version = Version(0x7fff_ffff_ffff_ffff);

    /// Special value indicating a read operation was cancelled
    pub const CANCELLED_READ: Version = Version(Version::MAX.value() + 1);

    /// Special value indicating system congestion
    pub const CONGESTED: Version = Version(Version::MAX.value() + 2);

    /// Creates a new Version with value 0
    pub const fn new() -> Self {
        Version(0)
    }

    /// Returns the underlying u64 value
    pub const fn value(&self) -> u64 {
        self.0
    }

    /// Creates a Version from a u64 value
    pub const fn from_u64(u: u64) -> Self {
        Version(u)
    }

    /// Increments the version by 1, panics if already at maximum
    pub fn increment(&mut self) {
        assert_ne!(self.0, u64::MAX);
        self.0 += 1;
    }

    /// Increments the version to a specific next version
    /// Debug asserts that the next version is greater than the current
    pub fn increment_to(&mut self, next: Version) {
        debug_assert!(*self < next, "Not an increment: {:?} to {:?}", self, next);
        *self = next;
    }

    /// Decrements the version by 1, panics if already at 0
    pub fn decrement(&mut self) {
        assert_ne!(self.0, 0);
        self.0 -= 1;
    }

    /// Decrements the version to a specific previous version
    /// Debug asserts that the previous version is less than the current
    pub fn decrement_to(&mut self, prev: Version) {
        debug_assert!(prev < *self, "Not a decrement: {:?} to {:?}", self, prev);
        *self = prev;
    }

    /// Returns the version one before the current, or None if at 0
    pub fn one_before(&self) -> Option<Version> {
        if self.0 == 0 { None } else { Some(Version(self.0 - 1)) }
    }

    /// Returns the next version (current + 1)
    pub fn next(&self) -> Version {
        Version(self.0 + 1)
    }

    /// Implements Lamport timestamp logic to determine the next version
    /// based on the maximum of input versions plus 1
    pub fn lamport_increment(inputs: impl IntoIterator<Item = Version>) -> Version {
        let max_input = inputs.into_iter().fold(Version::new(), max);

        // TODO: Ensure this never overflows.
        // Option 1: Freeze the object when sequence number reaches MAX.
        // Option 2: Reject tx with MAX sequence number.
        assert_ne!(max_input.0, u64::MAX);

        Version(max_input.0 + 1)
    }

    /// Checks if this version represents a cancelled operation
    pub fn is_cancelled(&self) -> bool {
        self == &Version::CANCELLED_READ || self == &Version::CONGESTED
    }

    /// Checks if this is a valid version for normal operations
    pub fn is_valid(&self) -> bool {
        self < &Version::MAX
    }
}

impl From<Version> for u64 {
    fn from(val: Version) -> Self {
        val.0
    }
}

impl From<u64> for Version {
    fn from(value: u64) -> Self {
        Version(value)
    }
}

/// A tuple of (Version, ObjectDigest) used to uniquely identify an object at a specific version
pub type VersionDigest = (Version, ObjectDigest);

/// A tuple of (ObjectID, Version, ObjectDigest) that uniquely identifies an object
/// at a specific version with its content digest, used for verification
pub type ObjectRef = (ObjectID, Version, ObjectDigest);

/// # ObjectInner
///
/// The core implementation of an object in the SOMA blockchain.
///
/// ## Purpose
/// ObjectInner contains the actual data, ownership information, and transaction
/// history of an object. It represents the complete state of an object at a
/// specific version.
///
/// ## Lifecycle
/// Objects are created by transactions, can be modified by subsequent transactions
/// if mutable, and maintain a reference to their previous transaction for provenance.
///
/// ## Thread Safety
/// ObjectInner is typically wrapped in Arc for thread-safe sharing.
#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash)]
#[serde(rename = "Object")]
pub struct ObjectInner {
    /// The core data of the object, including its ID, type, version, and contents
    pub data: ObjectData,

    /// The ownership model that determines who can use or modify this object
    pub owner: Owner,

    /// The digest of the transaction that created or last modified this object
    pub previous_transaction: TransactionDigest,
}

impl ObjectInner {
    /// Computes the object reference (ID, version, digest) for this object
    pub fn compute_object_reference(&self) -> ObjectRef {
        (self.id(), self.version(), self.digest())
    }

    /// Computes the content digest of this object
    pub fn digest(&self) -> ObjectDigest {
        ObjectDigest::new(default_hash(self))
    }

    /// Returns the object's ID
    pub fn id(&self) -> ObjectID {
        self.data.id()
    }

    /// Returns the object's current version
    pub fn version(&self) -> Version {
        self.data.version()
    }

    /// Returns the object's type
    pub fn type_(&self) -> &ObjectType {
        self.data.object_type()
    }

    /// Returns both the owner and object ID if the object has a single owner
    pub fn get_owner_and_id(&self) -> Option<(Owner, ObjectID)> {
        Some((self.owner.clone(), self.id()))
    }

    /// Checks if the object is immutable
    pub fn is_immutable(&self) -> bool {
        self.owner.is_immutable()
    }

    /// Checks if the object is owned by an address
    pub fn is_address_owned(&self) -> bool {
        self.owner.is_address_owned()
    }

    /// Checks if the object is shared
    pub fn is_shared(&self) -> bool {
        self.owner.is_shared()
    }

    /// Returns the single owner address if applicable
    pub fn get_single_owner(&self) -> Option<SomaAddress> {
        self.owner.get_owner_address().ok()
    }

    /// Returns the full object ID, which includes consensus information for shared objects
    pub fn full_id(&self) -> FullObjectID {
        let id = self.id();
        if let Some(start_version) = self.owner.start_version() {
            FullObjectID::Consensus((id, start_version))
        } else {
            FullObjectID::Fastpath(id)
        }
    }

    /// Computes the full object reference including consensus information
    pub fn compute_full_object_reference(&self) -> FullObjectRef {
        (self.full_id(), self.version(), self.digest())
    }
}

/// # Object
///
/// A thread-safe wrapper around ObjectInner using Arc for efficient sharing.
///
/// ## Purpose
/// Object provides a reference-counted wrapper around ObjectInner, allowing
/// efficient sharing of object data across components while maintaining
/// immutability for thread safety.
///
/// ## Usage Patterns
/// - Created during transaction execution
/// - Shared across components that need read access
/// - Cloned efficiently due to Arc wrapper
/// - Can be converted to mutable when needed via DerefMut
///
/// ## Thread Safety
/// Object uses Arc for thread-safe sharing. Mutation requires exclusive
/// access through Arc::make_mut.
#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash)]
#[serde(from = "ObjectInner")]
pub struct Object(Arc<ObjectInner>);

impl From<ObjectInner> for Object {
    fn from(inner: ObjectInner) -> Self {
        Self(Arc::new(inner))
    }
}

impl Object {
    /// Attempts to unwrap the Arc to get the inner value
    /// If there are other references, clones the inner value
    pub fn into_inner(self) -> ObjectInner {
        match Arc::try_unwrap(self.0) {
            Ok(inner) => inner,
            Err(inner_arc) => (*inner_arc).clone(),
        }
    }

    /// Returns a reference to the inner ObjectInner
    pub fn as_inner(&self) -> &ObjectInner {
        &self.0
    }

    /// Creates a new Object with the specified data, owner, and transaction digest
    pub fn new(data: ObjectData, owner: Owner, previous_transaction: TransactionDigest) -> Self {
        ObjectInner { data, owner, previous_transaction }.into()
    }

    /// Returns a reference to the object's owner
    pub fn owner(&self) -> &Owner {
        &self.0.owner
    }

    /// Create a new coin with the specified balance
    pub fn new_coin(
        id: ObjectID,
        balance: u64,
        owner: Owner,
        previous_transaction: TransactionDigest,
    ) -> Self {
        let data = ObjectData::new_with_id(
            id,
            ObjectType::Coin,
            Version::MIN,
            bcs::to_bytes(&balance).unwrap(),
        );
        Self::new(data, owner, previous_transaction)
    }

    /// Extract the coin balance if this is a coin object
    pub fn as_coin(&self) -> Option<u64> {
        if *self.data.object_type() == ObjectType::Coin {
            bcs::from_bytes(self.data.contents()).ok()
        } else {
            None
        }
    }

    /// Update the balance of a coin object
    pub fn update_coin_balance(&mut self, new_balance: u64) {
        self.data.update_contents(bcs::to_bytes(&new_balance).unwrap());
    }

    pub fn with_id_owner_for_testing(id: ObjectID, owner: SomaAddress) -> Self {
        // For testing, we provide sufficient gas by default.
        Self::with_id_owner_coin_for_testing(id, owner, GAS_VALUE_FOR_TESTING)
    }

    pub fn with_id_owner_version_for_testing(id: ObjectID, version: Version, owner: Owner) -> Self {
        let data = ObjectData::new_with_id(
            id,
            ObjectType::Coin,
            version,
            bcs::to_bytes(&GAS_VALUE_FOR_TESTING).unwrap(),
        );
        Self::new(data, owner, TransactionDigest::genesis_marker())
    }

    pub fn with_id_owner_coin_for_testing(id: ObjectID, owner: SomaAddress, balance: u64) -> Self {
        let data = ObjectData::new_with_id(
            id,
            ObjectType::Coin,
            Version::MIN,
            bcs::to_bytes(&balance).unwrap(),
        );
        Self::new(data, Owner::AddressOwner(owner), TransactionDigest::genesis_marker())
    }

    /// Create a new Object containing a StakedSoma
    pub fn new_staked_soma_object(
        id: ObjectID,
        staked_soma: StakedSomaV1,
        owner: Owner,
        previous_transaction: TransactionDigest,
    ) -> Object {
        // Serialize StakedSoma to bytes
        let staked_soma_bytes = bcs::to_bytes(&staked_soma).unwrap();

        // Create ObjectData
        let data = ObjectData::new_with_id(
            id,
            ObjectType::StakedSoma, // Assuming you've added this to your ObjectType enum
            Version::MIN,           // Start with minimum version
            staked_soma_bytes,
        );

        // Create and return the Object
        Object::new(data, owner, previous_transaction)
    }

    /// Extract StakedSoma from an Object
    pub fn as_staked_soma(&self) -> Option<StakedSomaV1> {
        if *self.data.object_type() == ObjectType::StakedSoma {
            bcs::from_bytes(self.data.contents()).ok()
        } else {
            None
        }
    }

    /// Create a new Object containing a Target.
    ///
    /// Targets are shared objects with `initial_shared_version` set to `OBJECT_START_VERSION`.
    /// Using OBJECT_START_VERSION (1) instead of Version::new() (0) ensures TemporaryStore
    /// won't replace it with the lamport timestamp, giving targets a predictable
    /// initial_shared_version that matches TARGET_OBJECT_SHARED_VERSION regardless of
    /// when the target is created (genesis or epoch change).
    pub fn new_target_object(
        id: ObjectID,
        target: TargetV1,
        previous_transaction: TransactionDigest,
    ) -> Object {
        // Serialize Target to bytes
        let target_bytes = bcs::to_bytes(&target).unwrap();

        // Create ObjectData - use Version::MIN, TemporaryStore assigns lamport version
        let data = ObjectData::new_with_id(id, ObjectType::Target, Version::MIN, target_bytes);

        // Targets are shared objects - use OBJECT_START_VERSION (1) directly
        // so TemporaryStore won't update it (it only updates Version::new() = 0)
        let owner = Owner::Shared { initial_shared_version: OBJECT_START_VERSION };

        Object::new(data, owner, previous_transaction)
    }

    /// Extract Target from an Object
    pub fn as_target(&self) -> Option<TargetV1> {
        if *self.data.object_type() == ObjectType::Target {
            bcs::from_bytes(self.data.contents()).ok()
        } else {
            None
        }
    }
}

impl std::ops::Deref for Object {
    type Target = ObjectInner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Object {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.0)
    }
}

impl From<&Object> for ObjectType {
    fn from(o: &Object) -> Self {
        o.data.object_type().clone()
    }
}

/// # ObjectData
///
/// Contains the core data of an object, including its type, version, and contents.
///
/// ## Purpose
/// ObjectData encapsulates the actual data stored in an object, including its
/// serialized contents, type information, and version. The first bytes of the
/// contents always contain the object's ID.
///
/// ## Lifecycle
/// ObjectData is created when an object is created and updated when the object
/// is modified. The version is incremented with each modification.
///
/// ## Thread Safety
/// ObjectData is typically accessed through Object which provides thread-safety.
#[serde_as]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct ObjectData {
    /// The type of the object, which determines its behavior and structure
    object_type: ObjectType,

    /// The version of the object, incremented with each modification
    /// This acts as a timestamp, not a sequentially increasing version
    version: Version,

    /// BCS serialized object contents, with the first bytes containing the object ID
    #[serde_as(as = "Bytes")]
    contents: Vec<u8>,
}

impl ObjectData {
    /// Creates a new ObjectData with a specified ID
    /// The ID is prepended to the contents for efficient access
    pub fn new_with_id(
        id: ObjectID,
        object_type: ObjectType,
        version: Version,
        contents: Vec<u8>,
    ) -> Self {
        let mut data = Vec::with_capacity(ObjectID::LENGTH + contents.len());
        data.extend_from_slice(&id.into_bytes());
        data.extend_from_slice(&contents);

        Self { object_type, version, contents: data }
    }

    /// Increments the version to a specific next version
    pub fn increment_version_to(&mut self, next: Version) {
        self.version.increment_to(next);
    }

    /// Decrements the version to a specific previous version
    pub fn decrement_version_to(&mut self, prev: Version) {
        self.version.decrement_to(prev);
    }

    /// Sets the version to a specific value
    pub fn set_version_to(&mut self, version: Version) {
        self.version = version;
    }

    /// Returns the raw contents without the ID bytes
    pub fn contents(&self) -> &[u8] {
        &self.contents[ObjectID::LENGTH..]
    }

    /// Returns the object type
    pub fn object_type(&self) -> &ObjectType {
        &self.object_type
    }

    /// Deserializes the contents into a Rust type
    pub fn to_rust<'de, T: Deserialize<'de>>(&'de self) -> Option<T> {
        bcs::from_bytes(self.contents()).ok()
    }

    /// Extracts the object ID from the contents
    pub fn id(&self) -> ObjectID {
        Self::id_opt(&self.contents).unwrap()
    }

    /// Attempts to extract an object ID from a byte slice
    pub fn id_opt(contents: &[u8]) -> Result<ObjectID, ObjectIDParseError> {
        if ObjectID::LENGTH > contents.len() {
            return Err(ObjectIDParseError::TryFromSliceError);
        }
        ObjectID::try_from(&contents[0..ObjectID::LENGTH])
    }

    /// Returns the object's version
    pub fn version(&self) -> Version {
        self.version
    }

    /// Updates the contents of this object but preserves the ID and does not increment version
    pub fn update_contents(&mut self, new_contents: Vec<u8>) {
        let id_bytes: Vec<u8> = self.contents[0..ObjectID::LENGTH].to_vec();
        let mut updated_contents = Vec::with_capacity(ObjectID::LENGTH + new_contents.len());
        updated_contents.extend_from_slice(&id_bytes);
        updated_contents.extend_from_slice(&new_contents);

        self.update_contents_with_limit(updated_contents);
    }

    /// Internal helper to update contents while ensuring ID is preserved
    fn update_contents_with_limit(&mut self, new_contents: Vec<u8>) {
        let old_id = self.id();
        self.contents = new_contents;

        // Update should not modify ID
        debug_assert_eq!(self.id(), old_id);
    }
}

/// # ObjectType
///
/// Defines the type of an object, which determines its behavior and structure.
///
/// ## Purpose
/// ObjectType categorizes objects based on their role in the system, allowing
/// for type-specific handling and validation.
///
/// ## Usage
/// Different object types may have different validation rules, execution
/// behaviors, and storage requirements.
#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash, PartialOrd, Ord)]
pub enum ObjectType {
    /// Represents the global system state object
    SystemState,
    /// Represents an owned SOMA Token object
    Coin,
    /// Represents an owned Staked SOMA object
    StakedSoma,
    /// Represents a data submission target object
    Target,
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectType::SystemState => write!(f, "SystemState"),
            ObjectType::Coin => write!(f, "Coin"),
            ObjectType::StakedSoma => write!(f, "StakedSoma"),
            ObjectType::Target => write!(f, "Target"),
        }
    }
}

impl FromStr for ObjectType {
    type Err = String; // Or use a custom error type

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "SystemState" => Ok(ObjectType::SystemState),
            "Coin" => Ok(ObjectType::Coin),
            "StakedSoma" => Ok(ObjectType::StakedSoma),
            "Target" => Ok(ObjectType::Target),
            _ => Err(format!("Unknown ObjectType: {}", s)),
        }
    }
}

/// # ObjectID
///
/// A unique identifier for objects in the SOMA blockchain.
///
/// ## Purpose
/// ObjectID provides a globally unique identifier for objects in the system,
/// based on the SomaAddress type. It includes methods for creating, parsing,
/// and manipulating object IDs.
///
/// ## Usage Patterns
/// - Created deterministically from transaction digests
/// - Used as keys in storage systems
/// - Part of object references for verification
///
/// ## Thread Safety
/// ObjectID is Copy and can be safely shared across threads.
#[serde_as]
#[derive(Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema)]
pub struct ObjectID(
    #[schemars(with = "Hex")]
    #[serde_as(as = "Readable<HexAccountAddress, _>")]
    SomaAddress,
);

impl ObjectID {
    /// The number of bytes in an object ID
    pub const LENGTH: usize = SOMA_ADDRESS_LENGTH;

    /// Hex address: 0x0
    pub const ZERO: Self = Self::new([0u8; Self::LENGTH]);

    /// Maximum possible object ID (all bytes set to 0xff)
    pub const MAX: Self = Self::new([0xff; Self::LENGTH]);

    /// Creates a new ObjectID from a byte array
    pub const fn new(obj_id: [u8; Self::LENGTH]) -> Self {
        Self(SomaAddress::new(obj_id))
    }

    /// Creates an ObjectID from a SomaAddress (const fn variant)
    pub const fn from_address(addr: SomaAddress) -> Self {
        Self(addr)
    }

    /// Returns a random ObjectID
    pub fn random() -> Self {
        let mut rng = OsRng;
        let buf: [u8; Self::LENGTH] = rng.r#gen();
        ObjectID::new(buf)
    }

    /// Returns a random ObjectID using the provided random number generator
    pub fn random_from_rng<R>(rng: &mut R) -> Self
    where
        R: AllowedRng,
    {
        let buf: [u8; Self::LENGTH] = rng.r#gen();
        ObjectID::new(buf)
    }

    /// Returns the underlying bytes as a vector
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.to_vec()
    }

    /// Creates an ObjectID from a byte array or buffer
    pub fn from_bytes<T: AsRef<[u8]>>(bytes: T) -> Result<Self, ObjectIDParseError> {
        <[u8; Self::LENGTH]>::try_from(bytes.as_ref())
            .map_err(|_| ObjectIDParseError::TryFromSliceError)
            .map(ObjectID::new)
    }

    /// Returns the underlying bytes array
    pub fn into_bytes(self) -> [u8; Self::LENGTH] {
        self.0.to_inner()
    }

    /// Creates an ObjectID with padding 0s before a single byte
    pub const fn from_single_byte(byte: u8) -> ObjectID {
        let mut bytes = [0u8; Self::LENGTH];
        bytes[Self::LENGTH - 1] = byte;
        ObjectID::new(bytes)
    }

    /// Converts from a hex string with 0x prefix to ObjectID
    /// Pads with 0s if the string is too short
    pub fn from_hex_literal(literal: &str) -> Result<Self, ObjectIDParseError> {
        if !literal.starts_with("0x") {
            return Err(ObjectIDParseError::HexLiteralPrefixMissing);
        }

        let hex_len = literal.len() - 2;

        // If the string is too short, pad it
        if hex_len < Self::LENGTH * 2 {
            let mut hex_str = String::with_capacity(Self::LENGTH * 2);
            for _ in 0..Self::LENGTH * 2 - hex_len {
                hex_str.push('0');
            }
            hex_str.push_str(&literal[2..]);
            Self::from_str(&hex_str)
        } else {
            Self::from_str(&literal[2..])
        }
    }

    /// Creates an ObjectID deterministically from a transaction digest and creation number
    /// Used to ensure globally unique object IDs within a transaction
    pub fn derive_id(digest: TransactionDigest, creation_num: u64) -> Self {
        let mut hasher = DefaultHash::default();
        // hasher.update([HashingIntentScope::RegularObjectId as u8]);
        hasher.update(digest);
        hasher.update(creation_num.to_le_bytes());
        let hash = hasher.finalize();

        // truncate into an ObjectID.
        // OK to access slice because digest should never be shorter than ObjectID::LENGTH.
        ObjectID::try_from(&hash.as_ref()[0..ObjectID::LENGTH]).unwrap()
    }

    /// Increments the ObjectID by a specified number of steps
    /// Treats the ObjectID as a big-endian number
    pub fn advance(&self, step: usize) -> Result<ObjectID, anyhow::Error> {
        let mut curr_vec = self.to_vec();
        let mut step_copy = step;

        let mut carry = 0;
        for idx in (0..Self::LENGTH).rev() {
            if step_copy == 0 {
                // Nothing else to do
                break;
            }
            // Extract the relevant part
            let g = (step_copy % 0x100) as u16;
            // Shift to next group
            step_copy >>= 8;
            let mut val = curr_vec[idx] as u16;
            (carry, val) = ((val + carry + g) / 0x100, (val + carry + g) % 0x100);
            curr_vec[idx] = val as u8;
        }

        if carry > 0 {
            return Err(anyhow!("Increment will cause overflow"));
        }
        ObjectID::try_from(curr_vec).map_err(|w| w.into())
    }

    /// Increments the ObjectID by one
    /// Treats the ObjectID as a big-endian number
    pub fn next_increment(&self) -> Result<ObjectID, anyhow::Error> {
        let mut prev_val = self.to_vec();
        let mx = [0xFF; Self::LENGTH];

        if prev_val == mx {
            return Err(anyhow!("Increment will cause overflow"));
        }

        // This logic increments the integer representation of an ObjectID u8 array
        for idx in (0..Self::LENGTH).rev() {
            if prev_val[idx] == 0xFF {
                prev_val[idx] = 0;
            } else {
                prev_val[idx] += 1;
                break;
            };
        }
        ObjectID::try_from(prev_val.clone()).map_err(|w| w.into())
    }

    /// Creates a range of sequential object IDs starting from an offset
    pub fn in_range(offset: ObjectID, count: u64) -> Result<Vec<ObjectID>, anyhow::Error> {
        let mut ret = Vec::new();
        let mut prev = offset;
        for o in 0..count {
            if o != 0 {
                prev = prev.next_increment()?;
            }
            ret.push(prev);
        }
        Ok(ret)
    }

    /// Returns the full hex string with 0x prefix without removing trailing 0s
    pub fn to_hex_uncompressed(&self) -> String {
        format!("{self}")
    }
}

impl From<SomaAddress> for ObjectID {
    fn from(address: SomaAddress) -> ObjectID {
        ObjectID(address)
    }
}

impl fmt::Display for ObjectID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "0x{}", Hex::encode(self.0))
    }
}

impl fmt::Debug for ObjectID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "0x{}", Hex::encode(self.0))
    }
}

impl AsRef<[u8]> for ObjectID {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl TryFrom<&[u8]> for ObjectID {
    type Error = ObjectIDParseError;

    /// Tries to convert a byte slice into an ObjectID
    fn try_from(bytes: &[u8]) -> Result<ObjectID, ObjectIDParseError> {
        Self::from_bytes(bytes)
    }
}

impl TryFrom<Vec<u8>> for ObjectID {
    type Error = ObjectIDParseError;

    /// Tries to convert a byte vector into an ObjectID
    fn try_from(bytes: Vec<u8>) -> Result<ObjectID, ObjectIDParseError> {
        Self::from_bytes(bytes)
    }
}

impl std::ops::Deref for ObjectID {
    type Target = SomaAddress;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Error type for ObjectID parsing operations
#[derive(PartialEq, Eq, Clone, Debug, thiserror::Error)]
pub enum ObjectIDParseError {
    #[error("ObjectID hex literal must start with 0x")]
    HexLiteralPrefixMissing,

    #[error("Could not convert from bytes slice")]
    TryFromSliceError,
}

impl FromStr for ObjectID {
    type Err = ObjectIDParseError;

    /// Parses an ObjectID from a hex string with or without 0x prefix
    fn from_str(s: &str) -> Result<Self, ObjectIDParseError> {
        decode_bytes_hex(s).or_else(|_| Self::from_hex_literal(s))
    }
}

/// # ObjectInfo
///
/// A lightweight representation of an object's metadata without its contents.
///
/// ## Purpose
/// ObjectInfo provides a compact view of an object's key properties for
/// efficient storage, transmission, and indexing without carrying the
/// full object contents.
///
/// ## Usage
/// Used in APIs, indexes, and other contexts where the full object
/// contents are not needed.
#[derive(Clone, Serialize, Deserialize, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub struct ObjectInfo {
    /// The unique identifier of the object
    pub object_id: ObjectID,

    /// The current version of the object
    pub version: Version,

    /// The content digest for verification
    pub digest: ObjectDigest,

    /// The type of the object
    pub object_type: ObjectType,

    /// The ownership model of the object
    pub owner: Owner,

    /// The digest of the transaction that created or last modified this object
    pub previous_transaction: TransactionDigest,
}

impl ObjectInfo {
    /// Creates an ObjectInfo from an object reference and the object itself
    pub fn new(oref: &ObjectRef, o: &Object) -> Self {
        let (object_id, version, digest) = *oref;
        Self {
            object_id,
            version,
            digest,
            object_type: o.into(),
            owner: o.owner.clone(),
            previous_transaction: o.previous_transaction,
        }
    }

    /// Creates an ObjectInfo directly from an Object
    pub fn from_object(object: &Object) -> Self {
        Self {
            object_id: object.id(),
            version: object.version(),
            digest: object.digest(),
            object_type: object.into(),
            owner: object.owner.clone(),
            previous_transaction: object.previous_transaction,
        }
    }
}

/// # LiveObject
///
/// Represents an object that is currently active in the system.
///
/// ## Purpose
/// LiveObject is an enum that can represent different types of live objects
/// in the system. Currently, it only has one variant (Normal), but the
/// enum structure allows for future extension to other object types.
///
/// ## Usage
/// Used to represent objects that are currently part of the active state.
#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash)]
pub enum LiveObject {
    /// A normal object with standard behavior
    Normal(Object),
}

impl LiveObject {
    /// Returns the object ID
    pub fn object_id(&self) -> ObjectID {
        match self {
            LiveObject::Normal(obj) => obj.id(),
        }
    }

    /// Returns the object version
    pub fn version(&self) -> Version {
        match self {
            LiveObject::Normal(obj) => obj.version(),
        }
    }

    /// Returns the object reference (ID, version, digest)
    pub fn object_reference(&self) -> ObjectRef {
        match self {
            LiveObject::Normal(obj) => obj.compute_object_reference(),
        }
    }

    /// Converts to a normal object if possible
    pub fn to_normal(self) -> Option<Object> {
        match self {
            LiveObject::Normal(object) => Some(object),
        }
    }
}

/// # Owner
///
/// Defines the ownership model for an object, determining who can access and modify it.
///
/// ## Purpose
/// Owner represents different ownership models in the system, including:
/// - Address ownership (owned by a specific account)
/// - Shared ownership (accessible by anyone)
/// - Immutable objects (no ownership, cannot be modified)
///
/// ## Usage Patterns
/// - Determines access control for objects
/// - Affects transaction validation rules
/// - Influences consensus requirements (shared objects require consensus)
///
/// ## Thread Safety
/// Owner is Clone and can be safely shared across threads.
#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash, Ord, PartialOrd)]
pub enum Owner {
    /// Object is exclusively owned by a single address, and is mutable.
    AddressOwner(SomaAddress),
    /// Object is shared, can be used by any address, and is mutable.
    Shared {
        /// The version at which the object became shared
        initial_shared_version: Version,
    },
    /// Object is immutable, and hence ownership doesn't matter.
    Immutable,
}

impl Owner {
    // NOTE: only return address of AddressOwner, otherwise return error,
    pub fn get_address_owner_address(&self) -> SomaResult<SomaAddress> {
        match self {
            Self::AddressOwner(address) => Ok(*address),
            Self::Shared { .. } | Self::Immutable => Err(SomaError::UnexpectedOwnerType),
        }
    }

    // NOTE: this function will return address of AddressOwner
    pub fn get_owner_address(&self) -> SomaResult<SomaAddress> {
        match self {
            Self::AddressOwner(address) => Ok(*address),
            Self::Shared { .. } | Self::Immutable => Err(SomaError::UnexpectedOwnerType),
        }
    }

    // Returns initial_shared_version for Shared objects, and start_version for ConsensusV2 objects.
    pub fn start_version(&self) -> Option<Version> {
        match self {
            Self::Shared { initial_shared_version } => Some(*initial_shared_version),
            Self::Immutable | Self::AddressOwner(_) => None,
        }
    }

    pub fn is_immutable(&self) -> bool {
        matches!(self, Owner::Immutable)
    }

    pub fn is_address_owned(&self) -> bool {
        matches!(self, Owner::AddressOwner(_))
    }

    pub fn is_shared(&self) -> bool {
        matches!(self, Owner::Shared { .. })
    }
}

impl Display for Owner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AddressOwner(address) => {
                write!(f, "Account Address ( {} )", address)
            }

            Self::Immutable => {
                write!(f, "Immutable")
            }
            Self::Shared { initial_shared_version } => {
                write!(f, "Shared( {} )", initial_shared_version.value())
            }
        }
    }
}
