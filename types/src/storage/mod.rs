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

/// A potential input to a transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum InputKey {
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

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum WriteKind {
    /// The object was in storage already but has been modified
    Mutate,
    /// The object was created in this transaction
    Create,
    /// The object was previously wrapped in another object, but has been restored to storage
    Unwrap,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum MarkerValue {
    /// An object was received at the given version in the transaction and is no longer able
    /// to be received at that version in subequent transactions.
    Received,
    /// An owned object was deleted  at the given version, and is no longer able to be
    /// accessed or used in subsequent transactions.
    OwnedDeleted,
    /// A shared object was deleted by the transaction and is no longer able to be accessed or
    /// used in subsequent transactions.
    SharedDeleted(TransactionDigest),
}

// The primary key type for object storage.
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

#[derive(Clone)]
pub enum ObjectOrTombstone {
    Object(Object),
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

#[serde_as]
#[derive(Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash, Serialize, Deserialize, Debug)]
pub struct ConsensusObjectKey(pub ConsensusObjectSequenceKey, pub Version);

/// FullObjectKey represents a unique object a specific version. For fastpath objects, this
/// is the same as ObjectKey. For consensus objects, this includes the start version, which
/// may change if an object is transferred out of and back into consensus.
#[serde_as]
#[derive(Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Hash, Serialize, Deserialize, Debug)]
pub enum FullObjectKey {
    Fastpath(ObjectKey),
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

/// Fetch the `ObjectKey`s (IDs and versions) for non-shared input objects.  Includes owned,
/// and immutable objects as well as the gas objects, but not move packages or shared objects.
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

pub fn transaction_receiving_object_keys(tx: &SenderSignedData) -> Vec<ObjectKey> {
    tx.intent_message()
        .value
        .receiving_objects()
        .into_iter()
        .map(|oref| oref.into())
        .collect()
}
