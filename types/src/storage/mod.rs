use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::object::{Object, ObjectID, ObjectRef, Version};

pub mod consensus;
pub mod object_store;
pub mod read_store;
pub mod storage_error;
pub mod write_store;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum WriteKind {
    /// The object was in storage already but has been modified
    Mutate,
    /// The object was created in this transaction
    Create,
    /// The object was previously wrapped in another object, but has been restored to storage
    Unwrap,
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
