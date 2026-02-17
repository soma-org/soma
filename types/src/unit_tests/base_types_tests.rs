use crate::base::*;
use crate::crypto::{DefaultHash, SignatureScheme, get_key_pair};
use crate::digests::*;
use crate::object::*;
use fastcrypto::ed25519::Ed25519KeyPair;
use fastcrypto::encoding::{Base58, Encoding};
use fastcrypto::hash::HashFunction;
use fastcrypto::traits::KeyPair as _;
use std::str::FromStr;

// ---------------------------------------------------------------------------
// SomaAddress tests
// ---------------------------------------------------------------------------

#[test]
fn test_soma_address_zero_constant() {
    let zero = SomaAddress::ZERO;
    assert_eq!(zero.to_vec(), vec![0u8; SOMA_ADDRESS_LENGTH]);
    assert_eq!(zero, SomaAddress::new([0u8; SOMA_ADDRESS_LENGTH]));
}

#[test]
fn test_soma_address_random_is_unique() {
    let a = SomaAddress::random();
    let b = SomaAddress::random();
    assert_ne!(a, b, "Two random addresses should almost certainly differ");
}

#[test]
fn test_soma_address_from_bytes_roundtrip() {
    let bytes = [0xABu8; SOMA_ADDRESS_LENGTH];
    let addr = SomaAddress::from_bytes(bytes).unwrap();
    assert_eq!(addr.to_inner(), bytes);
    assert_eq!(addr.to_vec(), bytes.to_vec());
}

#[test]
fn test_soma_address_from_bytes_wrong_length() {
    let short = vec![0u8; 16];
    assert!(SomaAddress::from_bytes(&short).is_err());

    let long = vec![0u8; 64];
    assert!(SomaAddress::from_bytes(&long).is_err());
}

#[test]
fn test_soma_address_from_hex_literal() {
    // Padded short literal
    let addr = SomaAddress::from_hex_literal("0x1").unwrap();
    let mut expected = [0u8; SOMA_ADDRESS_LENGTH];
    expected[SOMA_ADDRESS_LENGTH - 1] = 1;
    assert_eq!(addr, SomaAddress::new(expected));

    // Full length literal
    let full_hex = format!("0x{}", "ab".repeat(SOMA_ADDRESS_LENGTH));
    let addr2 = SomaAddress::from_hex_literal(&full_hex).unwrap();
    assert_eq!(addr2, SomaAddress::new([0xAB; SOMA_ADDRESS_LENGTH]));

    // Missing prefix
    assert!(SomaAddress::from_hex_literal("1234").is_err());
}

#[test]
fn test_soma_address_to_hex_and_display() {
    let bytes = [0u8; SOMA_ADDRESS_LENGTH];
    let addr = SomaAddress::new(bytes);
    let hex_str = addr.to_hex();
    // Display and to_hex both use lowercase hex
    let display_str = format!("{}", addr);
    assert_eq!(hex_str, display_str);
    assert_eq!(hex_str, "0".repeat(SOMA_ADDRESS_LENGTH * 2));
}

#[test]
fn test_soma_address_debug_formatting() {
    let addr = SomaAddress::ZERO;
    let debug_str = format!("{:?}", addr);
    // Debug uses the same lowercase hex representation
    assert_eq!(debug_str, "0".repeat(SOMA_ADDRESS_LENGTH * 2));
}

#[test]
fn test_soma_address_bcs_roundtrip() {
    let addr = SomaAddress::random();

    // BCS non-human-readable roundtrip
    let bcs_bytes = bcs::to_bytes(&addr).unwrap();
    let decoded: SomaAddress = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(addr, decoded);
}

#[test]
fn test_soma_address_from_str() {
    let addr = SomaAddress::random();
    let hex_str = addr.to_hex();
    let parsed = SomaAddress::from_str(&hex_str).unwrap();
    assert_eq!(addr, parsed);

    // Also test with 0x prefix
    let hex_literal = format!("0x{}", hex_str);
    let parsed2 = SomaAddress::from_str(&hex_literal).unwrap();
    assert_eq!(addr, parsed2);
}

#[test]
fn test_soma_address_short_str_lossless() {
    // Zero address should display as "0"
    assert_eq!(SomaAddress::ZERO.short_str_lossless(), "0");

    // Non-zero address should strip leading zeros
    let mut bytes = [0u8; SOMA_ADDRESS_LENGTH];
    bytes[SOMA_ADDRESS_LENGTH - 1] = 0xFF;
    let addr = SomaAddress::new(bytes);
    assert_eq!(addr.short_str_lossless(), "ff");

    // to_hex_literal should prepend 0x
    assert_eq!(addr.to_hex_literal(), "0xff");
}

#[test]
fn test_soma_address_from_public_key_derivation() {
    let (_addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let pk = kp.public().clone();

    // Derive address from public key manually
    let mut hasher = DefaultHash::default();
    hasher.update([SignatureScheme::ED25519.flag()]);
    hasher.update(&pk);
    let expected = SomaAddress::new(hasher.finalize().digest);

    // The get_key_pair function should return the same derived address
    assert_eq!(_addr, expected);
}

// ---------------------------------------------------------------------------
// ObjectID tests
// ---------------------------------------------------------------------------

#[test]
fn test_object_id_zero_and_max_constants() {
    assert_eq!(ObjectID::ZERO, ObjectID::new([0u8; ObjectID::LENGTH]));
    assert_eq!(ObjectID::MAX, ObjectID::new([0xFF; ObjectID::LENGTH]));
    assert!(ObjectID::ZERO < ObjectID::MAX);
}

#[test]
fn test_object_id_random() {
    let a = ObjectID::random();
    let b = ObjectID::random();
    assert_ne!(a, b);
}

#[test]
fn test_object_id_from_bytes() {
    let bytes = [42u8; ObjectID::LENGTH];
    let id = ObjectID::from_bytes(bytes).unwrap();
    assert_eq!(id.into_bytes(), bytes);

    // Wrong length
    assert!(ObjectID::from_bytes(&[0u8; 16]).is_err());
}

#[test]
fn test_object_id_from_hex_literal() {
    let id = ObjectID::from_hex_literal("0x0").unwrap();
    assert_eq!(id, ObjectID::ZERO);

    let id2 = ObjectID::from_hex_literal("0x1").unwrap();
    assert_eq!(id2, ObjectID::from_single_byte(1));

    // Missing prefix should fail
    assert!(ObjectID::from_hex_literal("1234").is_err());
}

#[test]
fn test_object_id_from_single_byte() {
    let id = ObjectID::from_single_byte(0x42);
    let bytes = id.into_bytes();
    // All leading bytes should be 0
    for i in 0..ObjectID::LENGTH - 1 {
        assert_eq!(bytes[i], 0);
    }
    assert_eq!(bytes[ObjectID::LENGTH - 1], 0x42);
}

#[test]
fn test_object_id_derive_id_determinism() {
    let digest = TransactionDigest::random();
    let id1 = ObjectID::derive_id(digest, 0);
    let id2 = ObjectID::derive_id(digest, 0);
    assert_eq!(id1, id2, "derive_id must be deterministic for same inputs");

    // Different creation_num should yield different IDs
    let id3 = ObjectID::derive_id(digest, 1);
    assert_ne!(id1, id3);

    // Different digest should yield different IDs
    let digest2 = TransactionDigest::random();
    let id4 = ObjectID::derive_id(digest2, 0);
    assert_ne!(id1, id4);
}

#[test]
fn test_object_id_advance() {
    let id = ObjectID::from_single_byte(0);
    let advanced = id.advance(1).unwrap();
    assert_eq!(advanced, ObjectID::from_single_byte(1));

    let advanced2 = id.advance(255).unwrap();
    assert_eq!(advanced2, ObjectID::from_single_byte(255));

    // Overflow should return error
    let max = ObjectID::MAX;
    assert!(max.advance(1).is_err());
}

#[test]
fn test_object_id_display_formatting() {
    let id = ObjectID::from_single_byte(0xAB);
    let display = format!("{}", id);
    // Display includes 0x prefix and full hex
    assert!(display.starts_with("0x"));
    assert!(display.ends_with("ab"));
    assert_eq!(display.len(), 2 + ObjectID::LENGTH * 2); // "0x" + hex chars
}

#[test]
fn test_object_id_bcs_roundtrip() {
    let id = ObjectID::random();
    let bcs_bytes = bcs::to_bytes(&id).unwrap();
    let decoded: ObjectID = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(id, decoded);
}

// ---------------------------------------------------------------------------
// Version tests
// ---------------------------------------------------------------------------

#[test]
fn test_version_min_max_special_constants() {
    assert_eq!(Version::MIN.value(), u64::MIN);
    assert_eq!(Version::MAX.value(), 0x7fff_ffff_ffff_ffff);
    assert!(Version::MIN < Version::MAX);
    assert!(Version::CANCELLED_READ.value() > Version::MAX.value());
    assert!(Version::CONGESTED.value() > Version::MAX.value());
    assert_ne!(Version::CANCELLED_READ, Version::CONGESTED);
}

#[test]
fn test_version_increment_and_decrement() {
    let mut v = Version::from_u64(5);
    v.increment();
    assert_eq!(v.value(), 6);

    v.decrement();
    assert_eq!(v.value(), 5);
}

#[test]
#[should_panic]
fn test_version_decrement_at_zero_panics() {
    let mut v = Version::from_u64(0);
    v.decrement();
}

#[test]
fn test_version_lamport_increment() {
    let inputs = vec![Version::from_u64(3), Version::from_u64(7), Version::from_u64(5)];
    let result = Version::lamport_increment(inputs);
    assert_eq!(result.value(), 8); // max(3,7,5) + 1

    // Empty inputs should return Version(1) (max of nothing = Version::new() = 0, then +1)
    let empty: Vec<Version> = vec![];
    let result2 = Version::lamport_increment(empty);
    assert_eq!(result2.value(), 1);
}

#[test]
fn test_version_is_valid_and_is_cancelled() {
    assert!(Version::from_u64(0).is_valid());
    assert!(Version::from_u64(100).is_valid());
    assert!(!Version::MAX.is_valid()); // MAX itself is not valid (< MAX is valid)
    assert!(!Version::CANCELLED_READ.is_valid());
    assert!(!Version::CONGESTED.is_valid());

    assert!(Version::CANCELLED_READ.is_cancelled());
    assert!(Version::CONGESTED.is_cancelled());
    assert!(!Version::from_u64(0).is_cancelled());
    assert!(!Version::MAX.is_cancelled());
}

#[test]
fn test_version_next_and_one_before() {
    let v = Version::from_u64(10);
    assert_eq!(v.next().value(), 11);

    assert_eq!(v.one_before(), Some(Version::from_u64(9)));
    assert_eq!(Version::from_u64(0).one_before(), None);
}

// ---------------------------------------------------------------------------
// SizeOneVec tests
// ---------------------------------------------------------------------------

#[test]
fn test_size_one_vec_new_element_into_inner() {
    let sov = SizeOneVec::new(42u32);
    assert_eq!(*sov.element(), 42);
    assert_eq!(sov.into_inner(), 42);
}

#[test]
fn test_size_one_vec_bcs_serialization() {
    // SizeOneVec serializes as a Vec (sequence)
    let sov = SizeOneVec::new(99u32);
    let bcs_bytes = bcs::to_bytes(&sov).unwrap();

    // It should deserialize as a Vec<u32> containing exactly one element
    let as_vec: Vec<u32> = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(as_vec, vec![99u32]);

    // And the Vec should deserialize back to SizeOneVec
    let roundtrip: SizeOneVec<u32> = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(*roundtrip.element(), 99);
}

#[test]
fn test_size_one_vec_try_from_vec() {
    // Exactly 1 element succeeds
    let v = vec![7u32];
    let sov = SizeOneVec::try_from(v).unwrap();
    assert_eq!(*sov.element(), 7);

    // 0 elements fails
    let empty: Vec<u32> = vec![];
    assert!(SizeOneVec::try_from(empty).is_err());

    // 2+ elements fails
    let two = vec![1u32, 2];
    assert!(SizeOneVec::try_from(two).is_err());

    let three = vec![1u32, 2, 3];
    assert!(SizeOneVec::try_from(three).is_err());
}

// ---------------------------------------------------------------------------
// Owner enum tests
// ---------------------------------------------------------------------------

#[test]
fn test_owner_address_owner_bcs_roundtrip() {
    let addr = SomaAddress::random();
    let owner = Owner::AddressOwner(addr);
    let bcs_bytes = bcs::to_bytes(&owner).unwrap();
    let decoded: Owner = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(owner, decoded);
    assert!(decoded.is_address_owned());
    assert!(!decoded.is_shared());
    assert!(!decoded.is_immutable());
}

#[test]
fn test_owner_shared_bcs_roundtrip() {
    let owner = Owner::Shared { initial_shared_version: Version::from_u64(5) };
    let bcs_bytes = bcs::to_bytes(&owner).unwrap();
    let decoded: Owner = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(owner, decoded);
    assert!(decoded.is_shared());
    assert_eq!(decoded.start_version(), Some(Version::from_u64(5)));
}

#[test]
fn test_owner_immutable_bcs_roundtrip() {
    let owner = Owner::Immutable;
    let bcs_bytes = bcs::to_bytes(&owner).unwrap();
    let decoded: Owner = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(owner, decoded);
    assert!(decoded.is_immutable());
    assert!(decoded.get_owner_address().is_err());
}

// ---------------------------------------------------------------------------
// ExecutionDigests tests
// ---------------------------------------------------------------------------

#[test]
fn test_execution_digests_new() {
    let tx = TransactionDigest::random();
    let fx = TransactionEffectsDigest::random();
    let ed = ExecutionDigests::new(tx, fx);
    assert_eq!(ed.transaction, tx);
    assert_eq!(ed.effects, fx);
}

#[test]
fn test_execution_digests_random() {
    let a = ExecutionDigests::random();
    let b = ExecutionDigests::random();
    // Both fields should differ (with overwhelming probability)
    assert_ne!(a.transaction, b.transaction);
    assert_ne!(a.effects, b.effects);
}

// ---------------------------------------------------------------------------
// FullObjectID tests
// ---------------------------------------------------------------------------

#[test]
fn test_full_object_id_fastpath() {
    let oid = ObjectID::random();
    let full_id = FullObjectID::new(oid, None);
    assert!(matches!(full_id, FullObjectID::Fastpath(_)));
    assert_eq!(full_id.id(), oid);
}

#[test]
fn test_full_object_id_consensus() {
    let oid = ObjectID::random();
    let version = Version::from_u64(42);
    let full_id = FullObjectID::new(oid, Some(version));
    assert!(matches!(full_id, FullObjectID::Consensus(_)));
    assert_eq!(full_id.id(), oid);
}

#[test]
fn test_full_object_id_fastpath_vs_consensus_different() {
    let oid = ObjectID::random();
    let fp = FullObjectID::new(oid, None);
    let cs = FullObjectID::new(oid, Some(Version::from_u64(1)));
    assert_ne!(fp, cs, "Fastpath and Consensus with same ObjectID should differ");
    // But both extract the same base ObjectID
    assert_eq!(fp.id(), cs.id());
}

// ---------------------------------------------------------------------------
// Digest / TransactionDigest tests
// ---------------------------------------------------------------------------

#[test]
fn test_digest_zero() {
    assert_eq!(Digest::ZERO.into_inner(), [0u8; 32]);
    assert_eq!(TransactionDigest::ZERO, TransactionDigest::new([0u8; 32]));
}

#[test]
fn test_transaction_digest_random() {
    let a = TransactionDigest::random();
    let b = TransactionDigest::random();
    assert_ne!(a, b);
}

#[test]
fn test_transaction_digest_base58_encoding_roundtrip() {
    let digest = TransactionDigest::random();
    let b58 = digest.base58_encode();

    // Decode back
    let decoded_bytes = Base58::decode(&b58).unwrap();
    assert_eq!(decoded_bytes.len(), 32);
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&decoded_bytes);
    let recovered = TransactionDigest::new(arr);
    assert_eq!(digest, recovered);
}

#[test]
fn test_transaction_digest_from_str_base58() {
    let digest = TransactionDigest::random();
    let b58 = digest.base58_encode();
    let parsed = TransactionDigest::from_str(&b58).unwrap();
    assert_eq!(digest, parsed);
}

#[test]
fn test_transaction_digest_bcs_roundtrip() {
    let digest = TransactionDigest::random();
    let bcs_bytes = bcs::to_bytes(&digest).unwrap();
    let decoded: TransactionDigest = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(digest, decoded);
}

#[test]
fn test_transaction_effects_digest_bcs_roundtrip() {
    let digest = TransactionEffectsDigest::random();
    let bcs_bytes = bcs::to_bytes(&digest).unwrap();
    let decoded: TransactionEffectsDigest = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(digest, decoded);
}

#[test]
fn test_object_digest_bcs_roundtrip() {
    let digest = ObjectDigest::random();
    let bcs_bytes = bcs::to_bytes(&digest).unwrap();
    let decoded: ObjectDigest = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(digest, decoded);
}

#[test]
fn test_checkpoint_digest_bcs_roundtrip() {
    let digest = CheckpointDigest::random();
    let bcs_bytes = bcs::to_bytes(&digest).unwrap();
    let decoded: CheckpointDigest = bcs::from_bytes(&bcs_bytes).unwrap();
    assert_eq!(digest, decoded);
}

// ---------------------------------------------------------------------------
// Additional edge-case tests
// ---------------------------------------------------------------------------

#[test]
fn test_soma_address_from_object_id_conversion() {
    let oid = ObjectID::random();
    let addr: SomaAddress = oid.into();
    assert_eq!(addr.to_vec(), oid.to_vec());
}

#[test]
fn test_dbg_addr() {
    let addr = dbg_addr(0x42);
    assert_eq!(addr, SomaAddress::new([0x42; SOMA_ADDRESS_LENGTH]));
}

#[test]
fn test_object_id_from_str_hex() {
    let id = ObjectID::random();
    let hex_str = format!("{}", id); // "0x..."
    let parsed = ObjectID::from_str(&hex_str[2..]).unwrap(); // strip 0x for from_str
    assert_eq!(id, parsed);
}

#[test]
fn test_object_id_next_increment() {
    let id = ObjectID::from_single_byte(0);
    let next = id.next_increment().unwrap();
    assert_eq!(next, ObjectID::from_single_byte(1));

    // MAX should overflow
    assert!(ObjectID::MAX.next_increment().is_err());
}

#[test]
fn test_object_id_in_range() {
    let start = ObjectID::from_single_byte(10);
    let range = ObjectID::in_range(start, 3).unwrap();
    assert_eq!(range.len(), 3);
    assert_eq!(range[0], ObjectID::from_single_byte(10));
    assert_eq!(range[1], ObjectID::from_single_byte(11));
    assert_eq!(range[2], ObjectID::from_single_byte(12));
}

#[test]
fn test_version_u64_conversions() {
    let v = Version::from_u64(999);
    let u: u64 = v.into();
    assert_eq!(u, 999);

    let v2: Version = 123u64.into();
    assert_eq!(v2.value(), 123);
}

#[test]
fn test_digest_next_lexicographical() {
    let d = Digest::new([0u8; 32]);
    let next = d.next_lexicographical().unwrap();
    let mut expected = [0u8; 32];
    expected[31] = 1;
    assert_eq!(next.into_inner(), expected);

    // All 0xFF should return None (no next)
    let max = Digest::new([0xFF; 32]);
    assert!(max.next_lexicographical().is_none());
}

#[test]
fn test_size_one_vec_iter() {
    let sov = SizeOneVec::new(String::from("hello"));
    let items: Vec<&String> = sov.iter().collect();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0], "hello");
}

#[test]
fn test_size_one_vec_element_mut() {
    let mut sov = SizeOneVec::new(10u32);
    *sov.element_mut() = 20;
    assert_eq!(*sov.element(), 20);
}
