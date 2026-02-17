use crate::base::SomaAddress;
use crate::digests::TransactionDigest;
use crate::object::*;
use std::str::FromStr;

/// Helper to create a simple ObjectData for testing.
fn make_test_object_data(id: ObjectID) -> ObjectData {
    ObjectData::new_with_id(id, ObjectType::Coin, Version::from_u64(1), vec![1, 2, 3, 4])
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

#[test]
fn test_coin_object() {
    let id = ObjectID::random();
    let balance: u64 = 1_000_000;
    let owner = Owner::AddressOwner(SomaAddress::default());
    let prev_tx = TransactionDigest::genesis_marker();

    let coin = Object::new_coin(id, balance, owner, prev_tx);

    // Verify it roundtrips
    let extracted = coin.as_coin();
    assert!(extracted.is_some(), "as_coin() should return Some for a coin object");
    assert_eq!(extracted.unwrap(), balance, "Coin balance should roundtrip correctly");
    assert_eq!(*coin.type_(), ObjectType::Coin);
    assert_eq!(coin.id(), id);
}

#[test]
fn test_coin_balance_update() {
    let id = ObjectID::random();
    let initial_balance: u64 = 500;
    let owner = Owner::AddressOwner(SomaAddress::default());
    let prev_tx = TransactionDigest::genesis_marker();

    let mut coin = Object::new_coin(id, initial_balance, owner, prev_tx);
    assert_eq!(coin.as_coin().unwrap(), initial_balance);

    let new_balance: u64 = 999;
    coin.update_coin_balance(new_balance);
    assert_eq!(
        coin.as_coin().unwrap(),
        new_balance,
        "Balance should be updated after update_coin_balance"
    );

    // ID should remain unchanged after balance update
    assert_eq!(coin.id(), id, "Object ID should not change after balance update");
}

#[test]
fn test_object_type_variants() {
    let variants = [
        (ObjectType::SystemState, "SystemState"),
        (ObjectType::Coin, "Coin"),
        (ObjectType::StakedSoma, "StakedSoma"),
        (ObjectType::Target, "Target"),
        (ObjectType::Submission, "Submission"),
        (ObjectType::Challenge, "Challenge"),
    ];

    for (variant, expected_str) in &variants {
        // Test Display
        let display = format!("{}", variant);
        assert_eq!(&display, expected_str, "Display for {:?} should be {}", variant, expected_str);

        // Test FromStr roundtrip
        let parsed = ObjectType::from_str(expected_str)
            .unwrap_or_else(|e| panic!("FromStr should succeed for {}: {}", expected_str, e));
        assert_eq!(
            &parsed, variant,
            "FromStr roundtrip failed for {}",
            expected_str
        );
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
fn test_version_ordering() {
    assert!(Version::MIN < Version::MAX, "MIN should be less than MAX");
    assert!(
        Version::MAX < Version::CANCELLED_READ,
        "MAX should be less than CANCELLED_READ"
    );

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
