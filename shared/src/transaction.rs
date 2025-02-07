use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    hash::{Hash, Hasher},
};

use crate::{
    crypto::{address::Address, keys::ProtocolKeySignature},
    metadata::MetadataCommitment,
};

use super::{digest::Digest, scope::ScopedMessage};

type Epoch = u64;

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
    metadata_commitment_digest: Digest<MetadataCommitment>,
    // TODO: value is just a placeholder this code was written
    // pre account/balances
    value: u64,
}

impl ShardTransaction {
    pub fn new(metadata_commitment_digest: Digest<MetadataCommitment>, value: u64) -> Self {
        Self {
            metadata_commitment_digest,
            value,
        }
    }
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

/// `TransactionRef` uniquely identifies a `VerifiedTransaction` via `digest`. It also contains the slot
/// info (round and author) so it can be used in logic such as aggregating stakes for a round.
#[derive(Clone, Serialize, Deserialize)]
pub struct TransactionRef {
    pub digest: Digest<SignedTransaction>,
}

impl TransactionRef {
    pub const MIN: Self = Self {
        digest: Digest::MIN,
    };

    pub const MAX: Self = Self {
        digest: Digest::MAX,
    };

    pub fn new(digest: Digest<SignedTransaction>) -> Self {
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
        state.write(&self.digest.as_ref()[..8]);
    }
}
