use serde::{Deserialize, Serialize};

use crate::types::Address;

pub type Version = u64;

/// Reference to an object
///
/// Contains sufficient information to uniquely identify a specific object.
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// object-ref = address u64 digest
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectReference {
    /// The object id of this object.
    object_id: Address,
    /// The version of this object.
    version: Version,
    /// The digest of this object.
    digest: Digest,
}

impl ObjectReference {
    /// Creates a new object reference from the object's id, version, and digest.
    pub fn new(object_id: Address, version: Version, digest: Digest) -> Self {
        Self {
            object_id,
            version,
            digest,
        }
    }

    /// Returns a reference to the object id that this ObjectReference is referring to.
    pub fn object_id(&self) -> &Address {
        &self.object_id
    }

    /// Returns the version of the object that this ObjectReference is referring to.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Returns the digest of the object that this ObjectReference is referring to.
    pub fn digest(&self) -> &Digest {
        &self.digest
    }

    /// Returns a 3-tuple containing the object id, version, and digest.
    pub fn into_parts(self) -> (Address, Version, Digest) {
        let Self {
            object_id,
            version,
            digest,
        } = self;

        (object_id, version, digest)
    }
}

/// Enum of different types of ownership for an object.
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// owner = owner-address / owner-object / owner-shared / owner-immutable
///
/// owner-address   = %x00 address
/// owner-object    = %x01 address
/// owner-shared    = %x02 u64
/// owner-immutable = %x03
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
#[non_exhaustive]
pub enum Owner {
    /// Object is exclusively owned by a single address, and is mutable.
    Address(Address),
    /// Object is shared, can be used by any address, and is mutable.
    Shared(
        /// The version at which the object became shared
        Version,
    ),
    /// Object is immutable, and hence ownership doesn't matter.
    Immutable,
}

/// Object data, either a package or struct
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// object-data = object-data-struct / object-data-package
///
/// object-data-struct  = %x00 object-move-struct
/// object-data-package = %x01 object-move-package
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
//TODO think about hiding this type and not exposing it
pub enum ObjectData {
    /// An object whose governing logic lives in a published Move module
    Struct(MoveStruct),
    /// Map from each module name to raw serialized Move module bytes
    Package(MovePackage),
    // ... Sui "native" types go here
}

/// Type of a Sui object
#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum ObjectType {
    /// Move package containing one or more bytecode modules
    Package,
    /// A Move struct of the given type
    Struct(StructTag),
}

/// An object on the sui blockchain
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// object = object-data owner digest u64
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Object {
    /// The meat of the object
    pub(crate) data: ObjectData,

    /// The owner that unlocks this object
    owner: Owner,

    /// The digest of the transaction that created or last mutated this object
    previous_transaction: Digest,
}

impl Object {
    /// Build an object
    pub fn new(data: ObjectData, owner: Owner, previous_transaction: Digest) -> Self {
        Self {
            data,
            owner,
            previous_transaction,
        }
    }

    /// Return this object's id
    pub fn object_id(&self) -> Address {
        match &self.data {
            ObjectData::Struct(struct_) => id_opt(&struct_.contents).unwrap(),
            ObjectData::Package(package) => package.id,
        }
    }

    /// Return this object's version
    pub fn version(&self) -> Version {
        match &self.data {
            ObjectData::Struct(struct_) => struct_.version,
            ObjectData::Package(package) => package.version,
        }
    }

    /// Return this object's type
    pub fn object_type(&self) -> ObjectType {
        match &self.data {
            ObjectData::Struct(struct_) => ObjectType::Struct(struct_.type_.clone()),
            ObjectData::Package(_) => ObjectType::Package,
        }
    }

    /// Try to interpret this object as a move struct
    pub fn as_struct(&self) -> Option<&MoveStruct> {
        match &self.data {
            ObjectData::Struct(struct_) => Some(struct_),
            _ => None,
        }
    }

    /// Return this object's owner
    pub fn owner(&self) -> &Owner {
        &self.owner
    }

    /// Return this object's data
    pub fn data(&self) -> &ObjectData {
        &self.data
    }

    /// Return the digest of the transaction that last modified this object
    pub fn previous_transaction(&self) -> Digest {
        self.previous_transaction
    }

    /// Return the storage rebate locked in this object
    ///
    /// Storage rebates are credited to the gas coin used in a transaction that deletes this
    /// object.
    pub fn storage_rebate(&self) -> u64 {
        self.storage_rebate
    }
}

fn id_opt(contents: &[u8]) -> Option<Address> {
    if Address::LENGTH > contents.len() {
        return None;
    }

    Some(Address::from_bytes(&contents[..Address::LENGTH]).unwrap())
}

/// An object part of the initial chain state
///
/// `GenesisObject`'s are included as a part of genesis, the initial checkpoint/transaction, that
/// initializes the state of the blockchain.
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// genesis-object = object-data owner
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenesisObject {
    data: ObjectData,
    owner: Owner,
}

impl GenesisObject {
    pub fn new(data: ObjectData, owner: Owner) -> Self {
        Self { data, owner }
    }

    pub fn object_id(&self) -> Address {
        match &self.data {
            ObjectData::Struct(struct_) => id_opt(&struct_.contents).unwrap(),
            ObjectData::Package(package) => package.id,
        }
    }

    pub fn version(&self) -> Version {
        match &self.data {
            ObjectData::Struct(struct_) => struct_.version,
            ObjectData::Package(package) => package.version,
        }
    }

    pub fn object_type(&self) -> ObjectType {
        match &self.data {
            ObjectData::Struct(struct_) => ObjectType::Struct(struct_.type_.clone()),
            ObjectData::Package(_) => ObjectType::Package,
        }
    }

    pub fn owner(&self) -> &Owner {
        &self.owner
    }

    pub fn data(&self) -> &ObjectData {
        &self.data
    }
}
