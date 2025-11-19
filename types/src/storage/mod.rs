use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use tracing::info;

use crate::{
    base::{ConsensusObjectSequenceKey, FullObjectID, FullObjectRef},
    digests::TransactionDigest,
    effects::{TransactionEffects, TransactionEffectsAPI},
    envelope::Message,
    error::SomaResult,
    object::{Object, ObjectID, ObjectRef, Version},
    storage::object_store::ObjectStore,
    transaction::{SenderSignedData, TransactionData},
};
use storage_error::Error as StorageError;

pub mod committee_store;
pub mod consensus;
pub mod object_store;
pub mod read_store;
pub mod shared_in_memory_store;
pub mod storage_error;
pub mod write_store;

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
    /// A fastpath object was deleted, wrapped, or transferred to consensus at the given
    /// version, and is no longer able to be accessed or used in subsequent transactions via
    /// fastpath unless/until it is returned to fastpath.
    OwnedDeleted,
    /// A shared object was deleted or removed from consensus by the transaction and is no longer
    /// able to be accessed or used in subsequent transactions with the same initial shared version.
    SharedDeleted(TransactionDigest),
}

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
pub fn get_transaction_input_objects(
    object_store: &dyn ObjectStore,
    effects: &TransactionEffects,
) -> Result<Vec<Object>, StorageError> {
    let input_object_keys = effects
        .modified_at_versions()
        .into_iter()
        .map(|(object_id, version)| ObjectKey(object_id, version))
        .collect::<Vec<_>>();

    let input_objects = object_store
        .multi_get_objects_by_key(&input_object_keys)
        .into_iter()
        .enumerate()
        .map(|(idx, maybe_object)| {
            maybe_object.ok_or_else(|| {
                StorageError::custom(format!(
                    "missing input object key {:?} from tx {} effects {}",
                    input_object_keys[idx],
                    effects.transaction_digest(),
                    effects.digest()
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(input_objects)
}

pub fn get_transaction_output_objects(
    object_store: &dyn ObjectStore,
    effects: &TransactionEffects,
) -> Result<Vec<Object>, StorageError> {
    let output_object_keys = effects
        .all_changed_objects()
        .into_iter()
        .map(|(object_ref, _owner, _kind)| ObjectKey::from(object_ref))
        .collect::<Vec<_>>();

    let output_objects = object_store
        .multi_get_objects_by_key(&output_object_keys)
        .into_iter()
        .enumerate()
        .map(|(idx, maybe_object)| {
            maybe_object.ok_or_else(|| {
                StorageError::custom(format!(
                    "missing output object key {:?} from tx {} effects {}",
                    output_object_keys[idx],
                    effects.transaction_digest(),
                    effects.digest()
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(output_objects)
}

// Returns an iterator over the ObjectKey's of objects read or written by this transaction
pub fn get_transaction_object_set(
    transaction: &TransactionData,
    effects: &TransactionEffects,
    unchanged_loaded_runtime_objects: &[ObjectKey],
) -> BTreeSet<ObjectKey> {
    // enumerate the full set of input objects in order to properly capture immutable objects that
    // may not appear in the effects.
    //
    // This excludes packages
    let input_objects = transaction
        .input_objects()
        .expect("txn was executed and must have valid input objects")
        .into_iter()
        .filter_map(|input| {
            input
                .version()
                .map(|version| ObjectKey(input.object_id(), version))
        });

    // The full set of output/written objects as well as any of their initial versions
    let modified_set = effects
        .object_changes()
        .into_iter()
        .flat_map(|change| {
            [
                change
                    .input_version
                    .map(|version| ObjectKey(change.id, version)),
                change
                    .output_version
                    .map(|version| ObjectKey(change.id, version)),
            ]
        })
        .flatten();

    // The set of unchanged consensus objects
    let unchanged_consensus =
        effects
            .unchanged_shared_objects()
            .into_iter()
            .flat_map(|unchanged| {
                if let crate::effects::UnchangedSharedKind::ReadOnlyRoot((version, _)) = unchanged.1
                {
                    Some(ObjectKey(unchanged.0, version))
                } else {
                    None
                }
            });

    input_objects
        .chain(modified_set)
        .chain(unchanged_consensus)
        .chain(unchanged_loaded_runtime_objects.iter().copied())
        .collect()
}
