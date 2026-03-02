// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::types::Address;
use crate::types::digest::Digest;

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
        Self { object_id, version, digest }
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
        let Self { object_id, version, digest } = self;

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
    /// The object's unique identifier
    pub object_id: Address,
    /// The object's version
    pub version: Version,
    /// The object's type
    pub object_type: ObjectType,
    /// The object's owner
    pub owner: Owner,
    /// The digest of the transaction that created or last modified this object
    pub previous_transaction: Digest,
    /// The raw contents (without the object ID prefix)
    pub contents: Vec<u8>,
}

impl Object {
    /// Build an object
    pub fn new(
        object_id: Address,
        version: Version,
        object_type: ObjectType,
        owner: Owner,
        previous_transaction: Digest,
        contents: Vec<u8>,
    ) -> Self {
        Self { object_id, version, object_type, owner, previous_transaction, contents }
    }

    /// Return this object's id
    pub fn object_id(&self) -> Address {
        self.object_id
    }

    /// Return this object's version
    pub fn version(&self) -> Version {
        self.version
    }

    /// Return this object's owner
    pub fn owner(&self) -> &Owner {
        &self.owner
    }

    /// Return the digest of the transaction that last modified this object
    pub fn previous_transaction(&self) -> Digest {
        self.previous_transaction
    }
}

/// Type of an object
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum ObjectType {
    SystemState,
    Coin,
    StakedSoma,
    Target,
}
