use crate::{
    base::{SomaAddress, SOMA_ADDRESS_LENGTH},
    crypto::{default_hash, DefaultHash},
    digests::{ObjectDigest, TransactionDigest},
};
use anyhow::anyhow;
use fastcrypto::{
    encoding::{decode_bytes_hex, Encoding, Hex},
    hash::HashFunction,
    traits::AllowedRng,
};
use rand::Rng;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, Bytes};
use std::cmp::max;
use std::{fmt, str::FromStr, sync::Arc};

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
    JsonSchema,
)]
pub struct Version(u64);

impl Version {
    pub const MIN: Version = Version(u64::MIN);
    pub const MAX: Version = Version(0x7fff_ffff_ffff_ffff);

    pub const fn new() -> Self {
        Version(0)
    }

    pub const fn value(&self) -> u64 {
        self.0
    }

    pub const fn from_u64(u: u64) -> Self {
        Version(u)
    }

    pub fn increment(&mut self) {
        assert_ne!(self.0, u64::MAX);
        self.0 += 1;
    }

    pub fn increment_to(&mut self, next: Version) {
        debug_assert!(*self < next, "Not an increment: {:?} to {:?}", self, next);
        *self = next;
    }

    pub fn decrement(&mut self) {
        assert_ne!(self.0, 0);
        self.0 -= 1;
    }

    pub fn decrement_to(&mut self, prev: Version) {
        debug_assert!(prev < *self, "Not a decrement: {:?} to {:?}", self, prev);
        *self = prev;
    }

    pub fn one_before(&self) -> Option<Version> {
        if self.0 == 0 {
            None
        } else {
            Some(Version(self.0 - 1))
        }
    }

    pub fn next(&self) -> Version {
        Version(self.0 + 1)
    }

    pub fn lamport_increment(inputs: impl IntoIterator<Item = Version>) -> Version {
        let max_input = inputs.into_iter().fold(Version::new(), max);

        // TODO: Ensure this never overflows.
        // Option 1: Freeze the object when sequence number reaches MAX.
        // Option 2: Reject tx with MAX sequence number.
        assert_ne!(max_input.0, u64::MAX);

        Version(max_input.0 + 1)
    }
}

pub type VersionDigest = (Version, ObjectDigest);
pub type ObjectRef = (ObjectID, Version, ObjectDigest);

#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash)]
#[serde(rename = "Object")]
pub struct ObjectInner {
    /// The meat of the object
    pub data: ObjectData,
    /// The digest of the transaction that created or last mutated this object
    pub previous_transaction: TransactionDigest,
}

impl ObjectInner {
    pub fn compute_object_reference(&self) -> ObjectRef {
        (self.id(), self.version(), self.digest())
    }

    pub fn digest(&self) -> ObjectDigest {
        ObjectDigest::new(default_hash(self))
    }

    pub fn id(&self) -> ObjectID {
        self.data.id()
    }

    pub fn version(&self) -> Version {
        self.data.version()
    }

    pub fn type_(&self) -> &ObjectType {
        self.data.object_type()
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash)]
#[serde(from = "ObjectInner")]
pub struct Object(Arc<ObjectInner>);

impl From<ObjectInner> for Object {
    fn from(inner: ObjectInner) -> Self {
        Self(Arc::new(inner))
    }
}

impl Object {
    pub fn into_inner(self) -> ObjectInner {
        match Arc::try_unwrap(self.0) {
            Ok(inner) => inner,
            Err(inner_arc) => (*inner_arc).clone(),
        }
    }

    pub fn as_inner(&self) -> &ObjectInner {
        &self.0
    }

    /// Create a new object
    pub fn new(data: ObjectData, previous_transaction: TransactionDigest) -> Self {
        ObjectInner {
            data,
            previous_transaction,
        }
        .into()
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

#[serde_as]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct ObjectData {
    // Immutable type
    object_type: ObjectType,
    /// Number that increases each time a tx takes this object as a mutable input
    /// This is a timestamp, not a sequentially increasing version
    version: Version,
    /// BCS serialized object
    #[serde_as(as = "Bytes")]
    contents: Vec<u8>,
}

impl ObjectData {
    /// Create a new ObjectData with specified ID
    pub fn new_with_id(
        id: ObjectID,
        object_type: ObjectType,
        version: Version,
        contents: Vec<u8>,
    ) -> Self {
        let mut data = Vec::with_capacity(ObjectID::LENGTH + contents.len());
        data.extend_from_slice(&id.into_bytes());
        data.extend_from_slice(&contents);

        Self {
            object_type,
            version,
            contents: data,
        }
    }

    pub fn increment_version_to(&mut self, next: Version) {
        self.version.increment_to(next);
    }

    pub fn decrement_version_to(&mut self, prev: Version) {
        self.version.decrement_to(prev);
    }

    /// Get the raw contents without the ID bytes
    pub fn contents(&self) -> &[u8] {
        &self.contents[ObjectID::LENGTH..]
    }

    pub fn object_type(&self) -> &ObjectType {
        &self.object_type
    }

    pub fn to_rust<'de, T: Deserialize<'de>>(&'de self) -> Option<T> {
        bcs::from_bytes(self.contents()).ok()
    }

    pub fn id(&self) -> ObjectID {
        Self::id_opt(&self.contents).unwrap()
    }

    pub fn id_opt(contents: &[u8]) -> Result<ObjectID, ObjectIDParseError> {
        if ObjectID::LENGTH > contents.len() {
            return Err(ObjectIDParseError::TryFromSliceError);
        }
        ObjectID::try_from(&contents[0..ObjectID::LENGTH])
    }

    pub fn version(&self) -> Version {
        self.version
    }

    /// Update the contents of this object but does not increment its version
    /// Update the contents of this object but preserve the ID
    pub fn update_contents(&mut self, new_contents: Vec<u8>) {
        let id_bytes: Vec<u8> = self.contents[0..ObjectID::LENGTH].to_vec();
        let mut updated_contents = Vec::with_capacity(ObjectID::LENGTH + new_contents.len());
        updated_contents.extend_from_slice(&id_bytes);
        updated_contents.extend_from_slice(&new_contents);

        self.update_contents_with_limit(updated_contents);
    }

    fn update_contents_with_limit(&mut self, new_contents: Vec<u8>) {
        let old_id = self.id();
        self.contents = new_contents;

        // Update should not modify ID
        debug_assert_eq!(self.id(), old_id);
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Deserialize, Serialize, Hash, PartialOrd, Ord)]
pub enum ObjectType {
    // System State
    SystemState,
}

#[serde_as]
#[derive(Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema)]
pub struct ObjectID(SomaAddress);

impl ObjectID {
    /// The number of bytes in an address.
    pub const LENGTH: usize = SOMA_ADDRESS_LENGTH;
    /// Hex address: 0x0
    pub const ZERO: Self = Self::new([0u8; Self::LENGTH]);
    pub const MAX: Self = Self::new([0xff; Self::LENGTH]);
    /// Create a new ObjectID
    pub const fn new(obj_id: [u8; Self::LENGTH]) -> Self {
        Self(SomaAddress::new(obj_id))
    }

    /// Const fn variant of `<ObjectID as From<SomaAddress>>::from`
    pub const fn from_address(addr: SomaAddress) -> Self {
        Self(addr)
    }

    /// Return a random ObjectID.
    pub fn random() -> Self {
        Self::from(SomaAddress::random())
    }

    /// Return a random ObjectID from a given RNG.
    pub fn random_from_rng<R>(rng: &mut R) -> Self
    where
        R: AllowedRng,
    {
        let buf: [u8; Self::LENGTH] = rng.gen();
        ObjectID::new(buf)
    }

    /// Return the underlying bytes buffer of the ObjectID.
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.to_vec()
    }

    /// Parse the ObjectID from byte array or buffer.
    pub fn from_bytes<T: AsRef<[u8]>>(bytes: T) -> Result<Self, ObjectIDParseError> {
        <[u8; Self::LENGTH]>::try_from(bytes.as_ref())
            .map_err(|_| ObjectIDParseError::TryFromSliceError)
            .map(ObjectID::new)
    }

    /// Return the underlying bytes array of the ObjectID.
    pub fn into_bytes(self) -> [u8; Self::LENGTH] {
        self.0.to_inner()
    }

    /// Make an ObjectID with padding 0s before the single byte.
    pub const fn from_single_byte(byte: u8) -> ObjectID {
        let mut bytes = [0u8; Self::LENGTH];
        bytes[Self::LENGTH - 1] = byte;
        ObjectID::new(bytes)
    }

    /// Convert from hex string to ObjectID where the string is prefixed with 0x
    /// Padding 0s if the string is too short.
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

    /// Create an ObjectID from `TransactionDigest` and `creation_num`.
    /// Caller is responsible for ensuring that `creation_num` is fresh
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

    /// Incremenent the ObjectID by usize IDs, assuming the ObjectID hex is a number represented as an array of bytes
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

    /// Increment the ObjectID by one, assuming the ObjectID hex is a number represented as an array of bytes
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

    /// Create `count` object IDs starting with one at `offset`
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

    /// Return the full hex string with 0x prefix without removing trailing 0s. Prefer this
    /// over [fn to_hex_literal] if the string needs to be fully preserved.
    pub fn to_hex_uncompressed(&self) -> String {
        format!("{self}")
    }
}

impl From<SomaAddress> for ObjectID {
    fn from(address: SomaAddress) -> ObjectID {
        address.into()
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

    /// Tries to convert the provided byte array into ObjectID.
    fn try_from(bytes: &[u8]) -> Result<ObjectID, ObjectIDParseError> {
        Self::from_bytes(bytes)
    }
}

impl TryFrom<Vec<u8>> for ObjectID {
    type Error = ObjectIDParseError;

    /// Tries to convert the provided byte buffer into ObjectID.
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

#[derive(PartialEq, Eq, Clone, Debug, thiserror::Error)]
pub enum ObjectIDParseError {
    #[error("ObjectID hex literal must start with 0x")]
    HexLiteralPrefixMissing,

    #[error("Could not convert from bytes slice")]
    TryFromSliceError,
}

impl FromStr for ObjectID {
    type Err = ObjectIDParseError;

    /// Parse ObjectID from hex string with or without 0x prefix, pad with 0s if needed.
    fn from_str(s: &str) -> Result<Self, ObjectIDParseError> {
        decode_bytes_hex(s).or_else(|_| Self::from_hex_literal(s))
    }
}

#[derive(Clone, Serialize, Deserialize, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub struct ObjectInfo {
    pub object_id: ObjectID,
    pub version: Version,
    pub digest: ObjectDigest,
    pub object_type: ObjectType,
    pub previous_transaction: TransactionDigest,
}

impl ObjectInfo {
    pub fn new(oref: &ObjectRef, o: &Object) -> Self {
        let (object_id, version, digest) = *oref;
        Self {
            object_id,
            version,
            digest,
            object_type: o.into(),
            previous_transaction: o.previous_transaction,
        }
    }

    pub fn from_object(object: &Object) -> Self {
        Self {
            object_id: object.id(),
            version: object.version(),
            digest: object.digest(),
            object_type: object.into(),
            previous_transaction: object.previous_transaction,
        }
    }
}
