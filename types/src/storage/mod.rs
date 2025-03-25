//! # Storage Module
//!
//! ## Overview
//! This module defines the core storage abstractions and types used throughout the Soma blockchain.
//! It provides interfaces and data structures for storing and retrieving blockchain objects,
//! transactions, and other state information.
//!
//! ## Responsibilities
//! - Define storage interfaces for different types of blockchain data
//! - Provide key types and structures for object storage and retrieval
//! - Support different storage access patterns (read, write, versioned)
//! - Handle object lifecycle states (active, deleted, wrapped)
//! - Manage consensus object storage and versioning
//!
//! ## Component Relationships
//! - Used by the Authority module to persist and retrieve blockchain state
//! - Provides storage abstractions for transaction processing
//! - Interfaces with the underlying database implementation
//! - Supports the object model defined in the object module
//!
//! ## Key Workflows
//! 1. Object storage and retrieval with versioning
//! 2. Transaction input and output object management
//! 3. Consensus object handling with special sequencing requirements
//! 4. Object tombstone management for deleted objects
//!
//! ## Design Patterns
//! - Trait-based interfaces for storage operations
//! - Type-safe key structures for database access
//! - Enum-based state representation for object lifecycle
//! - Separation of read and write operations

use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::{
    base::{ConsensusObjectSequenceKey, FullObjectID, FullObjectRef},
    digests::TransactionDigest,
    error::SomaResult,
    object::{Object, ObjectID, ObjectRef, Version},
    transaction::SenderSignedData,
};

pub mod committee_store;
pub mod consensus;
pub mod object_store;
pub mod read_store;
pub mod storage_error;
pub mod write_store;

/// # InputKey
///
/// Represents a key for looking up potential inputs to a transaction.
///
/// ## Purpose
/// Provides a standardized way to reference objects that may be used as inputs
/// to transactions, with versioning information to ensure the correct object
/// version is used.
///
/// ## Usage
/// Used during transaction validation and execution to look up and verify
/// the existence and state of input objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum InputKey {
    /// A versioned object reference, including both the object ID and its version
    VersionedObject { id: FullObjectID, version: Version },
}

impl InputKey {
    pub fn id(&self) -> FullObjectID {
        match self {
            InputKey::VersionedObject { id, .. } => *id,
        }
    }

    pub fn version(&self) -> Option<Version> {
        match self {
            InputKey::VersionedObject { version, .. } => Some(*version),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        match self {
            InputKey::VersionedObject { version, .. } => version.is_cancelled(),
        }
    }
}

impl From<&Object> for InputKey {
    fn from(obj: &Object) -> Self {
        InputKey::VersionedObject {
            id: obj.full_id(),
            version: obj.version(),
        }
    }
}

/// # WriteKind
///
/// Indicates how an object was written to storage during a transaction.
///
/// ## Purpose
/// Tracks the origin and modification type of objects written to storage,
/// which is important for correctly processing transaction effects and
/// maintaining object history.
///
/// ## Usage
/// Used in transaction effects to indicate how objects were modified,
/// which affects how they are processed by the storage layer.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum WriteKind {
    /// The object was in storage already but has been modified
    Mutate,

    /// The object was created in this transaction
    Create,

    /// The object was previously wrapped in another object, but has been restored to storage
    Unwrap,
}

/// # MarkerValue
///
/// Represents different states that can be marked for an object in storage.
///
/// ## Purpose
/// Tracks special states of objects that affect their availability for future
/// transactions, such as being received, deleted, or consumed.
///
/// ## Usage
/// Used by the storage layer to maintain object state and prevent double-spending
/// or use of deleted objects.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum MarkerValue {
    /// An object was received at the given version in the transaction and is no longer able
    /// to be received at that version in subsequent transactions
    Received,

    /// An owned object was deleted at the given version, and is no longer able to be
    /// accessed or used in subsequent transactions
    OwnedDeleted,

    /// A shared object was deleted by the transaction and is no longer able to be accessed or
    /// used in subsequent transactions
    /// Includes the digest of the transaction that deleted it
    SharedDeleted(TransactionDigest),
}

/// # ObjectKey
///
/// The primary key type for object storage, combining an object ID and version.
///
/// ## Purpose
/// Provides a unique identifier for objects in storage that includes both the
/// object ID and its version, allowing for versioned storage and retrieval.
///
/// ## Usage
/// Used as the primary key in object storage tables and for referencing
/// specific versions of objects throughout the system.
#[serde_as]
#[derive(Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash, Serialize, Deserialize, Debug)]
pub struct ObjectKey(pub ObjectID, pub Version);

impl ObjectKey {
    pub const ZERO: ObjectKey = ObjectKey(ObjectID::ZERO, Version::MIN);

    pub fn max_for_id(id: &ObjectID) -> Self {
        Self(*id, Version::MAX)
    }

    pub fn min_for_id(id: &ObjectID) -> Self {
        Self(*id, Version::MIN)
    }
}

impl From<ObjectRef> for ObjectKey {
    fn from(object_ref: ObjectRef) -> Self {
        ObjectKey::from(&object_ref)
    }
}

impl From<&ObjectRef> for ObjectKey {
    fn from(object_ref: &ObjectRef) -> Self {
        Self(object_ref.0, object_ref.1)
    }
}

/// # ObjectOrTombstone
///
/// Represents either a full object or a tombstone reference for a deleted object.
///
/// ## Purpose
/// Allows the storage system to handle both active objects and references to
/// deleted objects (tombstones) in a unified way, which is important for
/// maintaining object history and preventing object resurrection.
///
/// ## Usage
/// Used when retrieving objects from storage, where the result might be
/// either a full object or just a reference to a deleted object.
#[derive(Clone)]
pub enum ObjectOrTombstone {
    /// A complete object with all its data
    Object(Object),

    /// A reference to a deleted object (tombstone)
    Tombstone(ObjectRef),
}

impl ObjectOrTombstone {
    pub fn as_objref(&self) -> ObjectRef {
        match self {
            ObjectOrTombstone::Object(obj) => obj.compute_object_reference(),
            ObjectOrTombstone::Tombstone(obref) => *obref,
        }
    }
}

impl From<Object> for ObjectOrTombstone {
    fn from(object: Object) -> Self {
        ObjectOrTombstone::Object(object)
    }
}

/// # ConsensusObjectKey
///
/// A key type for consensus objects that includes sequence information.
///
/// ## Purpose
/// Provides a unique identifier for consensus objects that includes both
/// the sequence key and version, allowing for proper ordering and retrieval.
///
/// ## Usage
/// Used for storing and retrieving consensus objects that require special
/// sequencing and versioning.
#[serde_as]
#[derive(Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash, Serialize, Deserialize, Debug)]
pub struct ConsensusObjectKey(pub ConsensusObjectSequenceKey, pub Version);

/// # FullObjectKey
///
/// Represents a unique object at a specific version, handling both fastpath and consensus objects.
///
/// ## Purpose
/// Provides a unified key type that can reference both regular (fastpath) objects
/// and consensus objects, which have different storage requirements and versioning.
///
/// ## Usage
/// Used as a comprehensive key type for object storage and retrieval that can
/// handle all object types in the system.
///
/// ## Variants
/// - Fastpath: Regular objects with simple ID and version
/// - Consensus: Consensus objects that include sequence information
#[serde_as]
#[derive(Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash, Serialize, Deserialize, Debug)]
pub enum FullObjectKey {
    /// Regular object key with ID and version
    Fastpath(ObjectKey),

    /// Consensus object key with sequence information and version
    Consensus(ConsensusObjectKey),
}

impl FullObjectKey {
    pub fn max_for_id(id: &FullObjectID) -> Self {
        match id {
            FullObjectID::Fastpath(object_id) => Self::Fastpath(ObjectKey::max_for_id(object_id)),
            FullObjectID::Consensus(consensus_object_sequence_key) => Self::Consensus(
                ConsensusObjectKey(*consensus_object_sequence_key, Version::MAX),
            ),
        }
    }

    pub fn min_for_id(id: &FullObjectID) -> Self {
        match id {
            FullObjectID::Fastpath(object_id) => Self::Fastpath(ObjectKey::min_for_id(object_id)),
            FullObjectID::Consensus(consensus_object_sequence_key) => Self::Consensus(
                ConsensusObjectKey(*consensus_object_sequence_key, Version::MIN),
            ),
        }
    }

    pub fn new(object_id: FullObjectID, version: Version) -> Self {
        match object_id {
            FullObjectID::Fastpath(object_id) => Self::Fastpath(ObjectKey(object_id, version)),
            FullObjectID::Consensus(consensus_object_sequence_key) => {
                Self::Consensus(ConsensusObjectKey(consensus_object_sequence_key, version))
            }
        }
    }

    pub fn id(&self) -> FullObjectID {
        match self {
            FullObjectKey::Fastpath(object_key) => FullObjectID::Fastpath(object_key.0),
            FullObjectKey::Consensus(consensus_object_key) => {
                FullObjectID::Consensus(consensus_object_key.0)
            }
        }
    }

    pub fn version(&self) -> Version {
        match self {
            FullObjectKey::Fastpath(object_key) => object_key.1,
            FullObjectKey::Consensus(consensus_object_key) => consensus_object_key.1,
        }
    }

    // Returns the equivalent ObjectKey for this FullObjectKey, discarding any initial
    // shared version information, if present.
    // TODO: Delete this function once marker table migration is complete.
    pub fn into_object_key(self) -> ObjectKey {
        match self {
            FullObjectKey::Fastpath(object_key) => object_key,
            FullObjectKey::Consensus(consensus_object_key) => {
                ObjectKey(consensus_object_key.0 .0, consensus_object_key.1)
            }
        }
    }
}

impl From<FullObjectRef> for FullObjectKey {
    fn from(object_ref: FullObjectRef) -> Self {
        FullObjectKey::from(&object_ref)
    }
}

impl From<&FullObjectRef> for FullObjectKey {
    fn from(object_ref: &FullObjectRef) -> Self {
        FullObjectKey::new(object_ref.0, object_ref.1)
    }
}

/// # transaction_non_shared_input_object_keys
///
/// Fetches the ObjectKeys for non-shared input objects in a transaction.
///
/// ## Purpose
/// Extracts keys for owned and immutable objects used as inputs in a transaction,
/// which is useful for transaction validation and execution.
///
/// ## Arguments
/// * `tx` - The sender-signed transaction data to extract input object keys from
///
/// ## Returns
/// A Result containing a vector of ObjectKeys for non-shared input objects
///
/// ## Behavior
/// Includes owned and immutable objects as well as gas objects, but excludes
/// move packages and shared objects.
pub fn transaction_non_shared_input_object_keys(
    tx: &SenderSignedData,
) -> SomaResult<Vec<ObjectKey>> {
    use crate::transaction::InputObjectKind as I;
    Ok(tx
        .intent_message()
        .value
        .input_objects()?
        .into_iter()
        .filter_map(|object| match object {
            I::SharedObject { .. } => None,
            I::ImmOrOwnedObject(obj) => Some(obj.into()),
        })
        .collect())
}

/// # transaction_receiving_object_keys
///
/// Extracts the ObjectKeys for objects being received in a transaction.
///
/// ## Purpose
/// Identifies objects that are being received by the transaction, which is
/// important for tracking object transfers and preventing double-spending.
///
/// ## Arguments
/// * `tx` - The sender-signed transaction data to extract receiving object keys from
///
/// ## Returns
/// A vector of ObjectKeys for objects being received in the transaction
///
/// ## Usage
/// Used during transaction processing to mark objects as received and
/// prevent them from being received again in other transactions.
pub fn transaction_receiving_object_keys(tx: &SenderSignedData) -> Vec<ObjectKey> {
    tx.intent_message()
        .value
        .receiving_objects()
        .into_iter()
        .map(|oref| oref.into())
        .collect()
}
