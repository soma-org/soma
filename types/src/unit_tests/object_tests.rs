// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::str::FromStr;

use crate::base::SomaAddress;
use crate::digests::TransactionDigest;
use crate::object::*;

/// Helper to create a simple ObjectData for testing.
fn make_test_object_data(id: ObjectID) -> ObjectData {
    ObjectData::new_with_id(id, ObjectType::Coin(CoinType::Soma), Version::from_u64(1), vec![1, 2, 3, 4])
}

/// Helper to create a full Object for testing.
fn make_test_object() -> Object {
    let id = ObjectID::random();
    let data = make_test_object_data(id);
    let owner = Owner::AddressOwner(SomaAddress::default());
    let prev_tx = TransactionDigest::genesis_marker();
    Object::new(data, owner, prev_tx)
}

#[test]
fn test_object_bcs_roundtrip() {
    let obj = make_test_object();
    let bytes = bcs::to_bytes(&obj).expect("BCS serialize should succeed");
    let deserialized: Object = bcs::from_bytes(&bytes).expect("BCS deserialize should succeed");
    assert_eq!(obj, deserialized);
}

#[test]
fn test_object_digest_determinism() {
    let obj = make_test_object();
    let digest1 = obj.digest();
    let digest2 = obj.digest();
    assert_eq!(digest1, digest2, "Same object should always produce the same digest");
}

#[test]
fn test_object_id_derive_deterministic() {
    let digest = TransactionDigest::new([42u8; 32]);
    let creation_num: u64 = 7;

    let id1 = ObjectID::derive_id(digest, creation_num);
    let id2 = ObjectID::derive_id(digest, creation_num);
    assert_eq!(id1, id2, "derive_id with same inputs must be deterministic");

    // Different creation_num should yield different ID
    let id3 = ObjectID::derive_id(digest, creation_num + 1);
    assert_ne!(id1, id3, "Different creation_num should yield different IDs");

    // Different digest should yield different ID
    let other_digest = TransactionDigest::new([99u8; 32]);
    let id4 = ObjectID::derive_id(other_digest, creation_num);
    assert_ne!(id1, id4, "Different digest should yield different IDs");
}

// Stage 13k: test_coin_object / test_coin_balance_update deleted.
// They tested Object::new_coin / as_coin / update_coin_balance —
// all removed in Stage 13k now that production never constructs
// or reads Coin objects.

#[test]
fn test_object_type_variants() {
    let variants = [
        (ObjectType::SystemState, "SystemState"),
        (ObjectType::Coin(CoinType::Soma), "Coin(SOMA)"),
        (ObjectType::StakedSoma, "StakedSoma"),
    ];

    for (variant, expected_str) in &variants {
        // Test Display
        let display = format!("{}", variant);
        assert_eq!(&display, expected_str, "Display for {:?} should be {}", variant, expected_str);

        // Test FromStr roundtrip
        let parsed = ObjectType::from_str(expected_str)
            .unwrap_or_else(|e| panic!("FromStr should succeed for {}: {}", expected_str, e));
        assert_eq!(&parsed, variant, "FromStr roundtrip failed for {}", expected_str);
    }

    // Also test that an unknown string fails
    let result = ObjectType::from_str("UnknownType");
    assert!(result.is_err(), "FromStr should fail for unknown type");
}

#[test]
fn test_owner_address_owned() {
    let addr = SomaAddress::default();
    let owner = Owner::AddressOwner(addr);

    assert!(owner.is_address_owned());
    assert!(!owner.is_shared());
    assert!(!owner.is_immutable());

    let retrieved = owner.get_owner_address().expect("Should return Ok for AddressOwner");
    assert_eq!(retrieved, addr);
}

#[test]
fn test_owner_shared() {
    let version = Version::from_u64(5);
    let owner = Owner::Shared { initial_shared_version: version };

    assert!(owner.is_shared());
    assert!(!owner.is_address_owned());
    assert!(!owner.is_immutable());

    // get_owner_address should fail for Shared
    assert!(owner.get_owner_address().is_err());

    // start_version should return the initial_shared_version
    assert_eq!(owner.start_version(), Some(version));
}

#[test]
fn test_owner_immutable() {
    let owner = Owner::Immutable;

    assert!(owner.is_immutable());
    assert!(!owner.is_address_owned());
    assert!(!owner.is_shared());

    // get_owner_address should fail for Immutable
    assert!(owner.get_owner_address().is_err());
    assert_eq!(owner.start_version(), None);
}

#[test]
fn test_owner_accumulator_balance() {
    // Stage 14a: Owner::Accumulator { Balance } is system-managed —
    // not transferable, not address-owned, not Shared (no consensus
    // sequencing on it, version follows the mutating system tx).
    let owner = Owner::Accumulator { kind: AccumulatorKind::Balance };

    assert!(owner.is_accumulator());
    assert_eq!(owner.accumulator_kind(), Some(AccumulatorKind::Balance));
    assert!(!owner.is_address_owned());
    assert!(!owner.is_shared());
    assert!(!owner.is_immutable());

    assert!(owner.get_owner_address().is_err());
    assert!(owner.get_address_owner_address().is_err());
    assert_eq!(owner.start_version(), None);
}

#[test]
fn test_owner_accumulator_delegation() {
    let owner = Owner::Accumulator { kind: AccumulatorKind::Delegation };
    assert!(owner.is_accumulator());
    assert_eq!(owner.accumulator_kind(), Some(AccumulatorKind::Delegation));
    assert!(!owner.is_address_owned());
    assert!(!owner.is_shared());
    assert!(!owner.is_immutable());
}

#[test]
fn test_non_accumulator_owners_report_no_kind() {
    assert_eq!(Owner::Immutable.accumulator_kind(), None);
    assert_eq!(Owner::AddressOwner(SomaAddress::default()).accumulator_kind(), None);
    assert_eq!(
        Owner::Shared { initial_shared_version: Version::from_u64(1) }.accumulator_kind(),
        None,
    );
}

#[test]
fn test_owner_accumulator_bcs_roundtrip() {
    // Both kinds must serialize/deserialize stably so the on-disk
    // form of accumulator objects is consistent across validators.
    for kind in [AccumulatorKind::Balance, AccumulatorKind::Delegation] {
        let owner = Owner::Accumulator { kind };
        let bytes = bcs::to_bytes(&owner).expect("BCS serialize Owner::Accumulator");
        let round: Owner = bcs::from_bytes(&bytes).expect("BCS deserialize Owner::Accumulator");
        assert_eq!(owner, round, "BCS roundtrip must preserve the variant exactly");
    }
}

#[test]
fn test_owner_display_shape_matches_variants() {
    // The CLI/log surface formats Owner::Accumulator distinctly from
    // the other variants so log scrapers can recognize them.
    assert!(format!("{}", Owner::Immutable).contains("Immutable"));
    assert!(
        format!("{}", Owner::Accumulator { kind: AccumulatorKind::Balance })
            .contains("Accumulator(Balance)"),
    );
    assert!(
        format!("{}", Owner::Accumulator { kind: AccumulatorKind::Delegation })
            .contains("Accumulator(Delegation)"),
    );
}

#[test]
fn test_balance_accumulator_object_roundtrip() {
    // Stage 14a: constructing a BalanceAccumulator object and reading
    // it back must preserve the owner/coin_type/balance contents and
    // anchor the ObjectID at the deterministic derivation.
    use crate::accumulator::BalanceAccumulator;

    let owner = SomaAddress::new([7u8; 32]);
    let acc = BalanceAccumulator::new(owner, CoinType::Usdc, 1_000_000);
    let obj = Object::new_balance_accumulator(acc, TransactionDigest::genesis_marker());

    // ObjectID matches the deterministic derivation.
    assert_eq!(obj.id(), BalanceAccumulator::derive_id(owner, CoinType::Usdc));

    // Owner is the system-managed Accumulator(Balance) flavor.
    assert!(obj.owner().is_accumulator());
    assert_eq!(obj.owner().accumulator_kind(), Some(AccumulatorKind::Balance));

    // Type is the dedicated variant.
    assert_eq!(*obj.type_(), ObjectType::BalanceAccumulator);

    // Contents round-trip through BCS to the original payload.
    let read_back = obj.as_balance_accumulator().expect("must deserialize");
    assert_eq!(read_back, acc);

    // The wrong-type accessor must yield None — guards against
    // accidentally reading a BalanceAccumulator as a DelegationAccumulator.
    assert!(obj.as_delegation_accumulator().is_none());
}

#[test]
fn test_delegation_accumulator_object_roundtrip() {
    use crate::accumulator::DelegationAccumulator;

    let pool_id = ObjectID::new([3u8; 32]);
    let staker = SomaAddress::new([5u8; 32]);
    let acc = DelegationAccumulator::new(pool_id, staker, 500, 7);
    let obj = Object::new_delegation_accumulator(acc, TransactionDigest::genesis_marker());

    assert_eq!(obj.id(), DelegationAccumulator::derive_id(pool_id, staker));
    assert!(obj.owner().is_accumulator());
    assert_eq!(obj.owner().accumulator_kind(), Some(AccumulatorKind::Delegation));
    assert_eq!(*obj.type_(), ObjectType::DelegationAccumulator);

    let read_back = obj.as_delegation_accumulator().expect("must deserialize");
    assert_eq!(read_back, acc);
    assert!(obj.as_balance_accumulator().is_none());
}

#[test]
fn test_set_balance_accumulator_preserves_id_and_owner() {
    // Stage 14a: in-place mutation via `set_balance_accumulator`
    // must preserve the ObjectID (so versioning lines up across
    // mutations) and leave the owner/type alone.
    use crate::accumulator::BalanceAccumulator;

    let owner = SomaAddress::new([1u8; 32]);
    let initial = BalanceAccumulator::new(owner, CoinType::Soma, 100);
    let mut obj = Object::new_balance_accumulator(initial, TransactionDigest::genesis_marker());
    let id_before = obj.id();
    let owner_before = obj.owner().clone();
    let type_before = obj.type_().clone();

    let updated = BalanceAccumulator::new(owner, CoinType::Soma, 250);
    obj.set_balance_accumulator(&updated);

    assert_eq!(obj.id(), id_before, "in-place mutation must preserve ID");
    assert_eq!(*obj.owner(), owner_before);
    assert_eq!(*obj.type_(), type_before);
    assert_eq!(obj.as_balance_accumulator(), Some(updated));
}

#[test]
fn test_object_type_accumulator_fromstr_roundtrip() {
    assert_eq!(
        ObjectType::from_str("BalanceAccumulator").unwrap(),
        ObjectType::BalanceAccumulator,
    );
    assert_eq!(
        ObjectType::from_str("DelegationAccumulator").unwrap(),
        ObjectType::DelegationAccumulator,
    );
    assert_eq!(format!("{}", ObjectType::BalanceAccumulator), "BalanceAccumulator");
    assert_eq!(format!("{}", ObjectType::DelegationAccumulator), "DelegationAccumulator");
}

#[test]
fn test_version_ordering() {
    assert!(Version::MIN < Version::MAX, "MIN should be less than MAX");
    assert!(Version::MAX < Version::CANCELLED_READ, "MAX should be less than CANCELLED_READ");

    // Verify specific values
    assert_eq!(Version::MIN.value(), 0);
    assert_eq!(Version::MAX.value(), 0x7fff_ffff_ffff_ffff);
    assert_eq!(Version::CANCELLED_READ.value(), Version::MAX.value() + 1);

    // Test increment
    let mut v = Version::from_u64(10);
    v.increment();
    assert_eq!(v.value(), 11);

    // Test next
    let v2 = Version::from_u64(20);
    assert_eq!(v2.next().value(), 21);

    // Test is_cancelled
    assert!(Version::CANCELLED_READ.is_cancelled());
    assert!(Version::CONGESTED.is_cancelled());
    assert!(!Version::MAX.is_cancelled());
    assert!(!Version::MIN.is_cancelled());
}
