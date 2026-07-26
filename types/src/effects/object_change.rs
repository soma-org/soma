// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
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

use crate::base::SomaAddress;
use crate::digests::ObjectDigest;
use crate::object::{CoinType, Object, ObjectID, Owner, VersionDigest};

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

    /// Stage 14c: per-tx accumulator delta record. Mirrors Sui SIP-58's
    /// `AccumulatorWriteV1`. Multiple transactions in a single
    /// consensus commit can emit deltas to the same accumulator in
    /// parallel without write-conflict on its underlying object —
    /// settlement aggregates them serially per commit.
    ///
    /// `address` is the canonical accumulator-object ID, derived
    /// deterministically by `BalanceAccumulator::derive_id` (or its
    /// delegation counterpart). The accumulator object itself is
    /// mutated by the per-commit `Settlement` system tx after the
    /// per-tx deltas are aggregated.
    AccumulatorWriteV1(AccumulatorWriteV1),
}

/// Stage 14c: a single per-tx delta record emitted by an executor
/// against a specific accumulator. `operation` says merge (deposit)
/// or split (withdraw); `value` carries the magnitude.
///
/// The structure intentionally mirrors Sui's so consumers (indexers,
/// RPC, light clients) that already know the SIP-58 shape can recognize
/// it. Soma differs from Sui in that there is no `EventDigest` value
/// variant yet — Soma's accumulator universe is balance-only at the
/// moment.
#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash, Serialize, Deserialize)]
pub struct AccumulatorWriteV1 {
    /// Identifies the accumulator being written to. For balance
    /// accumulators this is `BalanceAccumulator::derive_id(owner,
    /// coin_type)`. The (owner, coin_type) natural key is recovered
    /// from the matching accumulator-object payload at apply time.
    pub address: AccumulatorAddress,
    pub operation: AccumulatorOperation,
    pub value: AccumulatorValue,
}

/// Stage 14c: full natural-key address of an accumulator entry.
/// Carries enough information for indexers to attribute deltas
/// without fetching the accumulator object — the `(owner, type)`
/// tuple plus the canonical ObjectID derived from them.
///
/// Soma's first accumulator family is balance — `(SomaAddress,
/// CoinType)` keys. The variant is BCS-stable so adding more
/// families (e.g., a future delegation accumulator family that
/// participates in user-tx effects) is non-breaking.
#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash, Serialize, Deserialize)]
pub enum AccumulatorAddress {
    Balance { owner: SomaAddress, coin_type: CoinType, id: ObjectID },
}

impl AccumulatorAddress {
    /// Construct a `Balance` address with the deterministic ObjectID
    /// derivation baked in. Prefer this over hand-building the variant
    /// so the ID stays consistent with the lookup-side.
    pub fn balance(owner: SomaAddress, coin_type: CoinType) -> Self {
        let id = crate::accumulator::BalanceAccumulator::derive_id(owner, coin_type);
        Self::Balance { owner, coin_type, id }
    }

    /// The canonical ObjectID for this accumulator. Indexers and the
    /// settlement executor use this to look up the accumulator object.
    pub fn object_id(&self) -> ObjectID {
        match self {
            Self::Balance { id, .. } => *id,
        }
    }
}

/// Stage 14c: operation an executor performs on an accumulator.
/// `Merge` is a deposit (account gains value); `Split` is a withdraw
/// (account loses value). Same naming as Sui's SIP-58.
#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash, Serialize, Deserialize)]
pub enum AccumulatorOperation {
    Merge,
    Split,
}

/// Stage 14c: payload for an accumulator delta. `U64` covers the only
/// in-use family today (balance amounts are u64). The variant kept
/// open so future accumulator families can carry richer payloads
/// (Sui has `Integer` for balances and `IntegerTuple` for vectorized
/// quantities; we only need the first today).
#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash, Serialize, Deserialize)]
pub enum AccumulatorValue {
    U64(u64),
}

impl AccumulatorValue {
    /// Magnitude as u64. Convenience for the common case.
    pub fn as_u64(&self) -> u64 {
        match self {
            Self::U64(v) => *v,
        }
    }
}

impl AccumulatorWriteV1 {
    /// Signed delta this write contributes to the targeted accumulator's
    /// balance. Merge → positive, Split → negative. `i128` is wide
    /// enough that aggregating any realistic mix of u64 deltas cannot
    /// overflow during summation.
    pub fn signed_delta(&self) -> i128 {
        let mag = self.value.as_u64() as i128;
        match self.operation {
            AccumulatorOperation::Merge => mag,
            AccumulatorOperation::Split => -mag,
        }
    }
}
