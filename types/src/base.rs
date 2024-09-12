use std::fmt;

use crate::crypto::{DefaultHash, GenericSignature, PublicKey, SomaPublicKey, SomaSignature};
use crate::error::SomaResult;
use crate::serde::Readable;
use crate::{crypto::AuthorityPublicKeyBytes, error::SomaError};
use anyhow::anyhow;
use fastcrypto::encoding::{Encoding, Hex};
use fastcrypto::hash::HashFunction;
use rand::Rng;
use schemars::JsonSchema;
use serde::{ser::SerializeSeq, Deserialize, Serialize, Serializer};
use serde_with::serde_as;

pub type TimestampMs = u64;

/// The round number.
pub type Round = u64;

pub type AuthorityName = AuthorityPublicKeyBytes;

/// A global sequence number assigned to every CommittedSubDag.
pub type SequenceNumber = u64;

pub trait ConciseableName<'a> {
    type ConciseTypeRef: std::fmt::Debug;
    type ConciseType: std::fmt::Debug;

    fn concise(&'a self) -> Self::ConciseTypeRef;
    fn concise_owned(&self) -> Self::ConciseType;
}

// SizeOneVec is a wrapper around Vec<T> that enforces the size of the vec to be 1.
// This seems pointless, but it allows us to have fields in protocol messages that are
// current enforced to be of size 1, but might later allow other sizes, and to have
// that constraint enforced in the serialization/deserialization layer, instead of
// requiring manual input validation.
#[derive(Debug, Deserialize, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[serde(try_from = "Vec<T>")]
pub struct SizeOneVec<T> {
    e: T,
}

impl<T> SizeOneVec<T> {
    pub fn new(e: T) -> Self {
        Self { e }
    }

    pub fn element(&self) -> &T {
        &self.e
    }

    pub fn element_mut(&mut self) -> &mut T {
        &mut self.e
    }

    pub fn into_inner(self) -> T {
        self.e
    }

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

pub const SOMA_ADDRESS_LENGTH: usize = 32;

#[serde_as]
#[derive(
    Eq, Default, PartialEq, Ord, PartialOrd, Copy, Clone, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct SomaAddress(
    #[schemars(with = "Hex")]
    #[serde_as(as = "Readable<Hex, _>")]
    [u8; SOMA_ADDRESS_LENGTH],
);

impl SomaAddress {
    pub const ZERO: Self = Self([0u8; SOMA_ADDRESS_LENGTH]);

    /// Convert the address to a byte buffer.
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.to_vec()
    }

    pub fn generate<R: rand::RngCore + rand::CryptoRng>(mut rng: R) -> Self {
        let buf: [u8; SOMA_ADDRESS_LENGTH] = rng.gen();
        Self(buf)
    }

    /// Return the underlying byte array of a SuiAddress.
    pub fn to_inner(self) -> [u8; SOMA_ADDRESS_LENGTH] {
        self.0
    }

    /// Parse a SuiAddress from a byte array or buffer.
    pub fn from_bytes<T: AsRef<[u8]>>(bytes: T) -> Result<Self, SomaError> {
        <[u8; SOMA_ADDRESS_LENGTH]>::try_from(bytes.as_ref())
            .map_err(|_| SomaError::InvalidAddress)
            .map(SomaAddress)
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

impl fmt::Display for SomaAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}", Hex::encode(self.0))
    }
}

impl fmt::Debug for SomaAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "0x{}", Hex::encode(self.0))
    }
}

impl TryFrom<&GenericSignature> for SomaAddress {
    type Error = SomaError;
    /// Derive a SuiAddress from a serialized signature in Sui [GenericSignature].
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
