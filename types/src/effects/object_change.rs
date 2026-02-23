// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

//! # Object Change Module
//!
//! ## Overview
//! This module defines the structures that represent how objects change during transaction execution.
//! It tracks the state of objects before and after a transaction, as well as operations on object IDs.
//!
//! ## Responsibilities
//! - Track object state before and after transaction execution
//! - Represent object creation, modification, and deletion operations
//! - Provide a structured way to represent object changes in transaction effects
//!
//! ## Component Relationships
//! - Used by the TransactionEffects structure to record object changes
//! - Consumed by the storage layer to apply changes to the object store
//! - Used by clients to understand how objects were affected by a transaction

use serde::{Deserialize, Serialize};

use crate::{
    digests::ObjectDigest,
    object::{Object, Owner, VersionDigest},
};

/// # IDOperation
///
/// Represents operations that can be performed on object IDs during transaction execution.
///
/// ## Purpose
/// Tracks whether an object ID was created, deleted, or unchanged during a transaction.
/// This is important for understanding the lifecycle of objects in the system.
#[derive(Eq, PartialEq, Copy, Clone, Debug, Serialize, Deserialize)]
pub enum IDOperation {
    /// No change to the object ID (object may still be modified)
    None,
    /// Object ID was created in this transaction
    Created,
    /// Object ID was deleted in this transaction
    Deleted,
}

/// # EffectsObjectChange
///
/// Represents the complete change to an object during transaction execution,
/// including its state before and after the transaction, and any ID operations.
///
/// ## Purpose
/// Provides a comprehensive record of how an object changed during a transaction,
/// which is essential for understanding transaction effects and maintaining the object store.
///
/// ## Lifecycle
/// Created during transaction execution to track changes to objects, then included
/// in the TransactionEffects to communicate these changes to other components.
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct EffectsObjectChange {
    // input_state and output_state are the core fields that's required by
    // the protocol as it tells how an object changes on-chain.
    /// State of the object in the store prior to this transaction
    pub input_state: ObjectIn,

    /// State of the object in the store after this transaction
    pub output_state: ObjectOut,

    /// Whether this object ID is created or deleted in this transaction
    pub id_operation: IDOperation,
}

impl EffectsObjectChange {
    /// # Create a new EffectsObjectChange
    ///
    /// Creates a new EffectsObjectChange instance that represents the change to an object
    /// during transaction execution.
    ///
    /// ## Arguments
    /// * `modified_at` - The version, digest, and owner of the object before the transaction, if it existed
    /// * `written` - The object after the transaction, if it exists
    /// * `id_created` - Whether the object ID was created in this transaction
    /// * `id_deleted` - Whether the object ID was deleted in this transaction
    ///
    /// ## Returns
    /// A new EffectsObjectChange instance representing the change to the object
    pub fn new(
        modified_at: Option<(VersionDigest, Owner)>,
        written: Option<&Object>,
        id_created: bool,
        id_deleted: bool,
    ) -> Self {
        debug_assert!(
            !id_created || !id_deleted,
            "Object ID can't be created and deleted at the same time."
        );
        Self {
            input_state: modified_at.map_or(ObjectIn::NotExist, ObjectIn::Exist),
            output_state: written.map_or(ObjectOut::NotExist, |o| {
                ObjectOut::ObjectWrite((o.digest(), o.owner.clone()))
            }),
            id_operation: if id_created {
                IDOperation::Created
            } else if id_deleted {
                IDOperation::Deleted
            } else {
                IDOperation::None
            },
        }
    }
}

/// # ObjectIn
///
/// Represents the state of an object before a transaction is executed.
///
/// ## Purpose
/// Tracks whether an object existed before a transaction and, if it did,
/// its version, digest, and owner. This is essential for understanding
/// the starting state of objects in transaction effects.
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum ObjectIn {
    /// Object did not exist in the store before this transaction
    NotExist,

    /// Object existed in the store before this transaction
    /// Contains the version, digest, and owner of the object
    Exist((VersionDigest, Owner)),
}

/// # ObjectOut
///
/// Represents the state of an object after a transaction is executed.
///
/// ## Purpose
/// Tracks whether an object exists after a transaction and, if it does,
/// its digest and owner. This is essential for understanding the final
/// state of objects in transaction effects.
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum ObjectOut {
    /// Object does not exist in the store after this transaction
    /// (it was deleted or wrapped)
    NotExist,

    /// Object exists in the store after this transaction
    /// Contains the digest and owner of the object
    /// This includes all mutated, created, and unwrapped objects
    ObjectWrite((ObjectDigest, Owner)),
}
