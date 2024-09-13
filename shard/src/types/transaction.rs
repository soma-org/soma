use crate::{
    crypto::{address::Address, keys::ProtocolKeySignature, DefaultHashFunction, DIGEST_LENGTH},
    error::{ShardError, ShardResult},
};
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    hash::{Hash, Hasher},
    str::FromStr,
    sync::Arc,
};

use crate::types::authority_committee::Epoch;

use super::{scope::ScopedMessage, shard::ShardSecretDigest};

/// TransactionData contains the details of a transaction. TransactionData is versioned
/// at the top-level given that it is sent over the network.
#[enum_dispatch(TransactionDataAPI)]
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum TransactionData {
    V1(TransactionDataV1),
    // When new variants are introduced, add them here, e.g.:
    // V2(TransactionDataV2),
}

/// Sets an optional transaction expiration for a given epoch.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize)]
pub enum TransactionExpiration {
    /// The transaction has no expiration
    None,
    /// Validators wont sign a transaction unless the expiration Epoch
    /// is greater than or equal to the current epoch
    Epoch(Epoch),
}

/// Version 1 of `TransactionData`
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct TransactionDataV1 {
    pub kind: TransactionKind,
    pub sender: Address,
    // pub gas_data: GasData,
    pub expiration: TransactionExpiration,
}

/// Supported Transaction Kinds
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum TransactionKind {
    /// ShardTransaction pays for a shard to process data
    ShardTransaction(ShardTransaction),
    // add more here ...
}

#[enum_dispatch]
pub trait TransactionDataAPI {
    fn sender(&self) -> Address;
    fn kind(&self) -> &TransactionKind;
    // fn gas_data(&self) -> &GasData;
    fn expiration(&self) -> &TransactionExpiration;
    // Add other common methods here
}

impl TransactionDataAPI for TransactionDataV1 {
    fn sender(&self) -> Address {
        self.sender
    }

    fn kind(&self) -> &TransactionKind {
        &self.kind
    }

    // fn gas_data(&self) -> &GasData {
    //     &self.gas_data
    // }

    fn expiration(&self) -> &TransactionExpiration {
        &self.expiration
    }

    // Implement other methods...
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct ShardTransaction {
    shard_secret_digest: ShardSecretDigest,
}

impl TransactionData {
    pub fn new_v1(
        kind: TransactionKind,
        sender: Address,
        // gas_data: GasData,
        expiration: TransactionExpiration,
    ) -> Self {
        TransactionData::V1(TransactionDataV1 {
            kind,
            sender,
            // gas_data,
            expiration,
        })
    }

    // Add other constructor methods as needed
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SignedTransaction {
    pub scoped_message: ScopedMessage<TransactionData>,
    /// A list of signatures signed by all transaction participants.
    /// 1. non participant signature must not be present.
    /// 2. signature order does not matter.
    pub tx_signatures: Vec<ProtocolKeySignature>,
}

impl SignedTransaction {
    fn serialize(&self) -> ShardResult<Bytes> {
        let bytes = bcs::to_bytes(self).map_err(ShardError::SerializationFailure)?;
        Ok(bytes.into())
    }
    fn digest(&self) -> ShardResult<TransactionDigest> {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(self.serialize()?);
        Ok(TransactionDigest(hasher.finalize().into()))
    }
}
/// Digest of a `Transaction` which covers the `Transaction` in Bytes format.
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TransactionDigest([u8; DIGEST_LENGTH]);

impl TransactionDigest {
    /// Lexicographic min & max digest.
    pub const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    pub const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}

impl Hash for TransactionDigest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<TransactionDigest> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: TransactionDigest) -> Self {
        Digest::new(hd.0)
    }
}

impl fmt::Display for TransactionDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
                .get(0..4)
                .ok_or(fmt::Error)?
        )
    }
}

impl fmt::Debug for TransactionDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for TransactionDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// `TransactionRef` uniquely identifies a `VerifiedTransaction` via `digest`. It also contains the slot
/// info (round and author) so it can be used in logic such as aggregating stakes for a round.
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TransactionRef {
    pub digest: TransactionDigest,
}

impl TransactionRef {
    pub const MIN: Self = Self {
        digest: TransactionDigest::MIN,
    };

    pub const MAX: Self = Self {
        digest: TransactionDigest::MAX,
    };

    pub fn new(digest: TransactionDigest) -> Self {
        Self { digest }
    }
}

// TODO: re-evaluate formats for production debugging.
impl fmt::Display for TransactionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "T{}", self.digest)
    }
}

impl fmt::Debug for TransactionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "T{}", self.digest)
    }
}

impl Hash for TransactionRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.digest.0[..8]);
    }
}

#[derive(Clone)]
pub struct VerifiedTransaction {
    block: Arc<SignedTransaction>,
    digest: TransactionDigest,
    serialized: Bytes,
}

impl VerifiedTransaction {
    pub(crate) fn new(
        signed_transaction: SignedTransaction,
        digest: TransactionDigest,
        serialized: Bytes,
    ) -> Self {
        Self {
            block: Arc::new(signed_transaction),
            digest,
            serialized,
        }
    }
}
