// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::types::digest::Digest;
use crate::types::{Address, EpochId, ExecutionStatus, Owner, TransactionFee, Version};

#[derive(PartialEq, Clone, Debug, Deserialize, Serialize)]
pub struct TransactionEffects {
    /// The status of the execution
    pub status: ExecutionStatus,

    /// The epoch when this transaction was executed
    pub epoch: EpochId,

    /// The transaction fee
    pub fee: TransactionFee,

    /// The transaction digest
    pub transaction_digest: Digest,

    /// The updated gas object reference, as an index into the `changed_objects` vector.
    /// Having a dedicated field for convenient access.
    /// System transaction that don't require gas will leave this as None.
    pub gas_object_index: Option<u32>,

    /// The set of transaction digests this transaction depends on
    pub dependencies: Vec<Digest>,

    /// The version number of all the written Move objects
    pub lamport_version: Version,

    /// Objects whose state are changed in the object store
    pub changed_objects: Vec<ChangedObject>,

    /// Consensus objects that are not mutated in this transaction
    pub unchanged_shared_objects: Vec<UnchangedSharedObject>,
}

/// Input/output state of an object that was changed during execution
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// changed-object = address object-in object-out id-operation
/// ```
#[derive(Eq, PartialEq, Clone, Debug, Deserialize, Serialize)]
pub struct ChangedObject {
    /// Id of the object
    pub object_id: Address,

    /// State of the object in the store prior to this transaction.
    pub input_state: ObjectIn,

    /// State of the object in the store after this transaction.
    pub output_state: ObjectOut,

    /// Whether this object ID is created or deleted in this transaction.
    /// This information isn't required by the protocol but is useful for providing more detailed
    /// semantics on object changes.
    pub id_operation: IdOperation,
}

/// A Consensus object that wasn't changed during execution
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// unchanged-consensus-object = address unchanged-consensus-object-kind
/// ```
#[derive(Eq, PartialEq, Clone, Debug, Deserialize, Serialize)]
pub struct UnchangedSharedObject {
    pub object_id: Address,
    pub kind: UnchangedSharedKind,
}

/// Type of unchanged consensus object
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// unchanged-consensus-object-kind =  read-only-root
///                                 =/ mutate-deleted
///                                 =/ read-deleted
///                                 =/ canceled
///                                 =/ per-epoch-config
///
/// read-only-root                           = %x00 u64 digest
/// mutate-deleted                           = %x01 u64
/// read-deleted                             = %x02 u64
/// canceled                                 = %x03 u64
/// per-epoch-config                         = %x04
/// per-epoch-config-with-sequence-number    = %x05 u64
/// ```
#[derive(Eq, PartialEq, Clone, Debug, Deserialize, Serialize)]
#[non_exhaustive]
pub enum UnchangedSharedKind {
    /// Read-only consensus objects from the input. We don't really need ObjectDigest
    /// for protocol correctness, but it will make it easier to verify untrusted read.
    ReadOnlyRoot { version: Version, digest: Digest },

    /// Deleted consensus objects that appear mutably/owned in the input.
    MutateDeleted { version: Version },

    /// Deleted consensus objects that appear as read-only in the input.
    ReadDeleted { version: Version },

    /// Consensus objects in canceled transaction. The sequence number embed cancellation reason.
    Canceled { version: Version },
}

/// State of an object prior to execution
///
/// If an object exists (at root-level) in the store prior to this transaction,
/// it should be Exist, otherwise it's NonExist, e.g. wrapped objects should be
/// NonExist.
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// object-in = object-in-not-exist / object-in-exist
///
/// object-in-not-exist = %x00
/// object-in-exist     = %x01 u64 digest owner
/// ```
#[derive(Eq, PartialEq, Clone, Debug, Deserialize, Serialize)]
#[non_exhaustive]
pub enum ObjectIn {
    NotExist,

    /// The old version, digest and owner.
    Exist {
        version: Version,
        digest: Digest,
        owner: Owner,
    },
}

/// State of an object after execution
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// object-out  =  object-out-not-exist
///             =/ object-out-object-write
///             =/ object-out-package-write
///
///
/// object-out-not-exist        = %x00
/// object-out-object-write     = %x01 digest owner
/// object-out-package-write    = %x02 version digest
/// ```
#[derive(Eq, PartialEq, Clone, Debug, Deserialize, Serialize)]
#[non_exhaustive]
pub enum ObjectOut {
    /// Same definition as in ObjectIn.
    NotExist,

    /// Any written object, including all of mutated, created, unwrapped today.
    ObjectWrite { digest: Digest, owner: Owner },
}

/// Defines what happened to an ObjectId during execution
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// id-operation =  id-operation-none
///              =/ id-operation-created
///              =/ id-operation-deleted
///
/// id-operation-none       = %x00
/// id-operation-created    = %x01
/// id-operation-deleted    = %x02
/// ```
#[derive(Eq, PartialEq, Copy, Clone, Debug, Deserialize, Serialize)]
#[non_exhaustive]
pub enum IdOperation {
    None,
    Created,
    Deleted,
}

impl TransactionEffects {
    /// The status of the execution
    pub fn status(&self) -> &ExecutionStatus {
        &self.status
    }

    /// The epoch when this transaction was executed.
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// The gas used in this transaction.
    pub fn fee(&self) -> &TransactionFee {
        &self.fee
    }
}
