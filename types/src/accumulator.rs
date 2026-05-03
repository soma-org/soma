// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! # Accumulator objects (Stage 14a)
//!
//! Soma's account-balance and F1 delegation state are modeled as
//! Sui-style objects rather than separate column-family rows. Each
//! `(owner, coin_type)` balance and each `(pool_id, staker)` delegation
//! row is a single Object owned by [`Owner::Accumulator`], with a
//! deterministically-derived [`ObjectID`].
//!
//! ## Why objects
//!
//! Modeling these state families as objects collapses the "two
//! storage models, two write paths, two effects representations"
//! discipline problem we hit pre-13m: every state change rides
//! `TransactionEffects.changed_objects`, the global state hash and
//! snapshot integrity come for free, and indexer integration is a
//! single walker. Sui SIP-58 took the same architectural choice; this
//! module is the Soma analogue, scoped to the two state families that
//! were CF-resident.
//!
//! ## Why deterministic IDs
//!
//! Every validator must agree on an accumulator object's ID without
//! consulting an auxiliary table. We derive the ID from the natural
//! key (`(owner, coin_type)` or `(pool_id, staker)`) plus a
//! domain-separation tag, hashed with Blake2b-256 and truncated to
//! 32 bytes — same width as a regular [`ObjectID`].
//!
//! ## Why a new owner variant
//!
//! Accumulator objects must not be transferable, deletable, or
//! mutated by user transactions. [`Owner::Accumulator`] flags this
//! explicitly so:
//!
//!   - `TransferObjects` rejects them up-front,
//!   - the transaction-checks layer treats them as system-managed
//!     (neither `AddressOwner`-typed nor `Shared`-sequenced),
//!   - and the effects layer can rely on a kernel invariant that
//!     mutations only originate from privileged executors (Settlement
//!     for `Balance`, AddStake/WithdrawStake/ChangeEpoch for
//!     `Delegation`).

use fastcrypto::hash::{Blake2b256, HashFunction as _};
use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::object::{CoinType, ObjectID};

/// Domain separation tag for `BalanceAccumulator` ObjectID derivation.
/// Versioned so a future schema change can yield a fresh ID space.
const BALANCE_ACCUMULATOR_DOMAIN_TAG: &[u8] = b"soma/balance-accumulator/v1";

/// Domain separation tag for `DelegationAccumulator` ObjectID
/// derivation.
const DELEGATION_ACCUMULATOR_DOMAIN_TAG: &[u8] = b"soma/delegation-accumulator/v1";

/// Stage 14a: per-(owner, coin_type) account-balance accumulator
/// payload. Stored as the BCS-serialized `contents` of an [`Object`]
/// of type [`ObjectType::BalanceAccumulator`], owned by
/// `Owner::Accumulator { kind: Balance }`.
///
/// The `owner` and `coin_type` fields duplicate what's already encoded
/// in the ObjectID (`derive_id` is bijective from `(owner,
/// coin_type)`); they're persisted in the contents as well so any
/// indexer or RPC consumer that's holding an Object can recover the
/// natural key without re-deriving from the ID.
///
/// `balance` is `u64` — same domain as the pre-14a CF rows.
///
/// [`Object`]: crate::object::Object
/// [`ObjectType::BalanceAccumulator`]: crate::object::ObjectType::BalanceAccumulator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BalanceAccumulator {
    pub owner: SomaAddress,
    pub coin_type: CoinType,
    pub balance: u64,
}

impl BalanceAccumulator {
    /// Construct a fresh accumulator at the given balance.
    pub const fn new(owner: SomaAddress, coin_type: CoinType, balance: u64) -> Self {
        Self { owner, coin_type, balance }
    }

    /// Derive the canonical [`ObjectID`] for the `(owner, coin_type)`
    /// pair. Deterministic across validators; same inputs always
    /// produce the same ID.
    ///
    /// Uses Blake2b-256 over the domain tag, owner bytes, and a
    /// length-prefixed coin-type discriminator. The discriminator is
    /// the BCS encoding of the [`CoinType`] enum, which is a single
    /// byte today but BCS-stable if more variants are added later.
    pub fn derive_id(owner: SomaAddress, coin_type: CoinType) -> ObjectID {
        let mut hasher = Blake2b256::default();
        hasher.update((BALANCE_ACCUMULATOR_DOMAIN_TAG.len() as u32).to_le_bytes());
        hasher.update(BALANCE_ACCUMULATOR_DOMAIN_TAG);
        hasher.update(owner.to_vec());
        let coin_bytes =
            bcs::to_bytes(&coin_type).expect("BCS serialization of CoinType is infallible");
        hasher.update((coin_bytes.len() as u32).to_le_bytes());
        hasher.update(coin_bytes);
        let digest: [u8; 32] = hasher.finalize().into();
        ObjectID::new(digest)
    }
}

/// Stage 14a: per-(pool_id, staker) F1 delegation row payload.
/// Stored as the BCS contents of an [`Object`] of type
/// [`ObjectType::DelegationAccumulator`], owned by
/// `Owner::Accumulator { kind: Delegation }`.
///
/// The shape mirrors the pre-14a `delegations` CF row schema so the
/// migration is value-preserving.
///
/// [`Object`]: crate::object::Object
/// [`ObjectType::DelegationAccumulator`]: crate::object::ObjectType::DelegationAccumulator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DelegationAccumulator {
    pub pool_id: ObjectID,
    pub staker: SomaAddress,
    pub principal: u64,
    pub last_collected_period: u64,
}

impl DelegationAccumulator {
    pub const fn new(
        pool_id: ObjectID,
        staker: SomaAddress,
        principal: u64,
        last_collected_period: u64,
    ) -> Self {
        Self { pool_id, staker, principal, last_collected_period }
    }

    /// Derive the canonical [`ObjectID`] for the `(pool_id, staker)`
    /// pair. Same determinism contract as
    /// [`BalanceAccumulator::derive_id`].
    pub fn derive_id(pool_id: ObjectID, staker: SomaAddress) -> ObjectID {
        let mut hasher = Blake2b256::default();
        hasher.update((DELEGATION_ACCUMULATOR_DOMAIN_TAG.len() as u32).to_le_bytes());
        hasher.update(DELEGATION_ACCUMULATOR_DOMAIN_TAG);
        hasher.update(pool_id.into_bytes());
        hasher.update(staker.to_vec());
        let digest: [u8; 32] = hasher.finalize().into();
        ObjectID::new(digest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(seed: u8) -> SomaAddress {
        SomaAddress::new([seed; 32])
    }

    fn pool(seed: u8) -> ObjectID {
        ObjectID::new([seed; 32])
    }

    // -----------------------------------------------------------------
    // BalanceAccumulator
    // -----------------------------------------------------------------

    #[test]
    fn balance_id_is_deterministic() {
        // Same inputs must always yield the same ID across validators.
        let a = BalanceAccumulator::derive_id(addr(1), CoinType::Usdc);
        let b = BalanceAccumulator::derive_id(addr(1), CoinType::Usdc);
        assert_eq!(a, b);
    }

    #[test]
    fn balance_id_differs_per_owner() {
        let alice = BalanceAccumulator::derive_id(addr(1), CoinType::Usdc);
        let bob = BalanceAccumulator::derive_id(addr(2), CoinType::Usdc);
        assert_ne!(alice, bob);
    }

    #[test]
    fn balance_id_differs_per_coin_type() {
        // SOMA and USDC for the same owner must collide-never.
        let soma = BalanceAccumulator::derive_id(addr(1), CoinType::Soma);
        let usdc = BalanceAccumulator::derive_id(addr(1), CoinType::Usdc);
        assert_ne!(soma, usdc);
    }

    #[test]
    fn balance_bcs_roundtrip() {
        let acc = BalanceAccumulator::new(addr(1), CoinType::Usdc, 1_000_000);
        let bytes = bcs::to_bytes(&acc).unwrap();
        let round: BalanceAccumulator = bcs::from_bytes(&bytes).unwrap();
        assert_eq!(acc, round);
    }

    // -----------------------------------------------------------------
    // DelegationAccumulator
    // -----------------------------------------------------------------

    #[test]
    fn delegation_id_is_deterministic() {
        let a = DelegationAccumulator::derive_id(pool(7), addr(1));
        let b = DelegationAccumulator::derive_id(pool(7), addr(1));
        assert_eq!(a, b);
    }

    #[test]
    fn delegation_id_differs_per_pool() {
        let pool_a = DelegationAccumulator::derive_id(pool(7), addr(1));
        let pool_b = DelegationAccumulator::derive_id(pool(8), addr(1));
        assert_ne!(pool_a, pool_b);
    }

    #[test]
    fn delegation_id_differs_per_staker() {
        let alice = DelegationAccumulator::derive_id(pool(7), addr(1));
        let bob = DelegationAccumulator::derive_id(pool(7), addr(2));
        assert_ne!(alice, bob);
    }

    #[test]
    fn delegation_bcs_roundtrip() {
        let acc = DelegationAccumulator::new(pool(7), addr(1), 500, 42);
        let bytes = bcs::to_bytes(&acc).unwrap();
        let round: DelegationAccumulator = bcs::from_bytes(&bytes).unwrap();
        assert_eq!(acc, round);
    }

    // -----------------------------------------------------------------
    // Cross-family domain separation
    // -----------------------------------------------------------------

    #[test]
    fn balance_and_delegation_id_spaces_are_disjoint() {
        // Even with structurally similar key bytes, the domain tag
        // forces the two ID spaces to never collide. This matters
        // because the object store is a single keyspace — a balance
        // and a delegation that hashed to the same ObjectID would
        // be a fatal collision.
        let bal = BalanceAccumulator::derive_id(addr(1), CoinType::Soma);
        // Use the same seed for pool_id as for the address; the
        // domain tag must still keep them apart.
        let del = DelegationAccumulator::derive_id(pool(1), addr(1));
        assert_ne!(bal, del);
    }
}
