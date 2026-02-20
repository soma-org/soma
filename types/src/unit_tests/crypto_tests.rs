use crate::base::SomaAddress;
use crate::crypto::*;
use crate::intent::*;
use fastcrypto::ed25519::Ed25519KeyPair;
use fastcrypto::traits::{EncodeDecodeBase64, KeyPair, ToFromBytes};
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn test_keypair_generation() {
    // Generate an Ed25519 keypair via get_key_pair and verify public key derivation.
    let (addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();

    // The address should be derived from the public key.
    let expected_addr = SomaAddress::from(kp.public());
    assert_eq!(addr, expected_addr);

    // Public key bytes should have correct length (32 for Ed25519).
    assert_eq!(kp.public().as_ref().len(), 32);
}

#[test]
fn test_keypair_sign_verify() {
    let (_addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let soma_kp = SomaKeyPair::Ed25519(kp);

    let msg = IntentMessage::new(
        Intent::soma_transaction(),
        PersonalMessage { message: b"test message".to_vec() },
    );

    // Sign with correct key and verify succeeds.
    let sig = Signature::new_secure(&msg, &soma_kp);

    let author = SomaAddress::from(&soma_kp.public());
    assert!(sig.verify_secure(&msg, author, soma_kp.public().scheme()).is_ok());

    // Verify with a different key should fail.
    let (_addr2, kp2): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let soma_kp2 = SomaKeyPair::Ed25519(kp2);
    let wrong_author = SomaAddress::from(&soma_kp2.public());
    assert!(sig
        .verify_secure(&msg, wrong_author, SignatureScheme::ED25519)
        .is_err());
}

#[test]
fn test_keypair_serde_roundtrip() {
    let mut rng = StdRng::from_seed([42u8; 32]);
    let (_addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair_from_rng(&mut rng);
    let soma_kp = SomaKeyPair::Ed25519(kp);

    // Base64 encode/decode roundtrip (SomaKeyPair serializes as Base64).
    let encoded = soma_kp.encode_base64();
    let decoded = SomaKeyPair::decode_base64(&encoded).unwrap();
    assert_eq!(soma_kp, decoded);

    // Also verify the bytes roundtrip.
    let bytes = soma_kp.to_bytes();
    let from_bytes = SomaKeyPair::from_bytes(&bytes).unwrap();
    assert_eq!(soma_kp, from_bytes);
}

#[test]
fn test_public_key_serde_roundtrip() {
    let mut rng = StdRng::from_seed([7u8; 32]);
    let (_addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair_from_rng(&mut rng);
    let soma_kp = SomaKeyPair::Ed25519(kp);
    let pk = soma_kp.public();

    // Base64 encode/decode roundtrip for PublicKey.
    let encoded = pk.encode_base64();
    let decoded = PublicKey::decode_base64(&encoded).unwrap();
    assert_eq!(pk, decoded);

    // Verify the flag is correct.
    assert_eq!(pk.flag(), SignatureScheme::ED25519.flag());
}

#[test]
fn test_authority_public_key_bytes() {
    let mut rng = StdRng::from_seed([99u8; 32]);
    let (_addr, authority_kp): (SomaAddress, AuthorityKeyPair) = get_key_pair_from_rng(&mut rng);

    // Convert AuthorityPublicKey -> AuthorityPublicKeyBytes.
    let pk = authority_kp.public().clone();
    let pk_bytes = AuthorityPublicKeyBytes::from(&pk);

    // Roundtrip: AuthorityPublicKeyBytes -> AuthorityPublicKey.
    let pk_recovered = AuthorityPublicKey::try_from(pk_bytes).unwrap();
    assert_eq!(pk, pk_recovered);

    // Bytes representation should have correct length.
    let expected_len = pk.as_ref().len();
    assert_eq!(pk_bytes.as_ref().len(), expected_len);
}

#[test]
fn test_authority_sign_info() {
    let mut rng = StdRng::from_seed([11u8; 32]);
    let (_addr, authority_kp): (SomaAddress, AuthorityKeyPair) = get_key_pair_from_rng(&mut rng);

    let authority_name = AuthorityPublicKeyBytes::from(authority_kp.public());
    let epoch: u64 = 5;

    let data = PersonalMessage { message: b"authority message".to_vec() };
    let intent = Intent::soma_app(IntentScope::TransactionEffects);

    let sign_info =
        AuthoritySignInfo::new(epoch, &data, intent, authority_name, &authority_kp);

    // Verify fields.
    assert_eq!(sign_info.epoch, epoch);
    assert_eq!(sign_info.authority, authority_name);
}

#[test]
fn test_authority_keypair_sign_verify_via_trait() {
    // Test BLS authority key pair generation, signing, and verification
    // using the SomaAuthoritySignature trait.
    let mut rng = StdRng::from_seed([55u8; 32]);
    let (_addr, authority_kp): (SomaAddress, AuthorityKeyPair) = get_key_pair_from_rng(&mut rng);

    let authority_name = AuthorityPublicKeyBytes::from(authority_kp.public());
    let epoch: u64 = 1;

    let data = PersonalMessage { message: b"proof data".to_vec() };
    let intent = Intent::soma_transaction();
    let intent_msg = IntentMessage::new(intent, &data);

    // Sign using the SomaAuthoritySignature trait.
    let sig = AuthoritySignature::new_secure(&intent_msg, &epoch, &authority_kp);

    // Verify using the SomaAuthoritySignature trait.
    let result = sig.verify_secure(&intent_msg, epoch, authority_name);
    assert!(result.is_ok());

    // Verify with wrong key should fail.
    let (_addr2, authority_kp2): (SomaAddress, AuthorityKeyPair) =
        get_key_pair_from_rng(&mut rng);
    let wrong_name = AuthorityPublicKeyBytes::from(authority_kp2.public());
    let result2 = sig.verify_secure(&intent_msg, epoch, wrong_name);
    assert!(result2.is_err());
}

#[test]
fn test_address_from_keypair() {
    let mut rng = StdRng::from_seed([13u8; 32]);
    let (addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair_from_rng(&mut rng);
    let soma_kp = SomaKeyPair::Ed25519(kp);

    // Address derived from the PublicKey enum should match.
    let pk = soma_kp.public();
    let addr_from_pk = SomaAddress::from(&pk);
    assert_eq!(addr, addr_from_pk);

    // Address should not be zero for a randomly generated key.
    assert_ne!(addr, SomaAddress::ZERO);

    // Same keypair should always produce the same address.
    let addr_again = SomaAddress::from(&soma_kp.public());
    assert_eq!(addr_from_pk, addr_again);
}

#[test]
fn test_signature_scheme_flags() {
    // Each SignatureScheme should have a unique flag byte.
    let ed25519_flag = SignatureScheme::ED25519.flag();
    let bls_flag = SignatureScheme::BLS12381.flag();
    let multisig_flag = SignatureScheme::MultiSig.flag();

    assert_eq!(ed25519_flag, 0x00);
    assert_eq!(bls_flag, 0x01);
    assert_eq!(multisig_flag, 0x02);

    // All flags are unique.
    assert_ne!(ed25519_flag, bls_flag);
    assert_ne!(ed25519_flag, multisig_flag);
    assert_ne!(bls_flag, multisig_flag);

    // Roundtrip from flag byte back to scheme.
    assert_eq!(
        SignatureScheme::from_flag_byte(&ed25519_flag).unwrap(),
        SignatureScheme::ED25519
    );
    assert_eq!(
        SignatureScheme::from_flag_byte(&bls_flag).unwrap(),
        SignatureScheme::BLS12381
    );
    assert_eq!(
        SignatureScheme::from_flag_byte(&multisig_flag).unwrap(),
        SignatureScheme::MultiSig
    );

    // Invalid flag byte should error.
    assert!(SignatureScheme::from_flag_byte(&0xFF).is_err());
}

#[test]
fn test_soma_keypair_base64_roundtrip() {
    let mut rng = StdRng::from_seed([77u8; 32]);
    let (_addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair_from_rng(&mut rng);
    let soma_kp = SomaKeyPair::Ed25519(kp);

    // Base64 encode/decode roundtrip.
    let encoded = soma_kp.encode_base64();
    let decoded = SomaKeyPair::decode_base64(&encoded).unwrap();
    assert_eq!(soma_kp, decoded);
}

#[test]
fn test_soma_keypair_bech32_roundtrip() {
    let mut rng = StdRng::from_seed([88u8; 32]);
    let (_addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair_from_rng(&mut rng);
    let soma_kp = SomaKeyPair::Ed25519(kp);

    // Bech32 encode/decode roundtrip.
    let encoded = soma_kp.encode().unwrap();
    assert!(encoded.starts_with(SOMA_PRIV_KEY_PREFIX));
    let decoded = SomaKeyPair::decode(&encoded).unwrap();
    assert_eq!(soma_kp, decoded);
}

// =============================================================================
// Proof of Possession tests
// =============================================================================

#[test]
fn test_proof_of_possession_roundtrip() {
    // Generate a PoP and verify it succeeds with the correct key and address.
    let mut rng = StdRng::from_seed([0; 32]);
    let (address, authority_kp): (SomaAddress, AuthorityKeyPair) =
        get_key_pair_from_rng(&mut rng);

    let pop = generate_proof_of_possession(&authority_kp, address);
    assert!(
        verify_proof_of_possession(&pop, authority_kp.public(), address).is_ok(),
        "PoP should verify with correct key and address"
    );
}

#[test]
fn test_proof_of_possession_wrong_address() {
    // PoP generated for one address should NOT verify with a different address.
    let mut rng = StdRng::from_seed([1; 32]);
    let (address, authority_kp): (SomaAddress, AuthorityKeyPair) =
        get_key_pair_from_rng(&mut rng);
    let (wrong_address, _): (SomaAddress, AuthorityKeyPair) = get_key_pair_from_rng(&mut rng);
    assert_ne!(address, wrong_address);

    let pop = generate_proof_of_possession(&authority_kp, address);
    assert!(
        verify_proof_of_possession(&pop, authority_kp.public(), wrong_address).is_err(),
        "PoP should fail when verified with wrong address"
    );
}

#[test]
fn test_proof_of_possession_wrong_key() {
    // PoP generated with one key should NOT verify against a different public key.
    let mut rng = StdRng::from_seed([2; 32]);
    let (address, authority_kp): (SomaAddress, AuthorityKeyPair) =
        get_key_pair_from_rng(&mut rng);
    let (_, wrong_kp): (SomaAddress, AuthorityKeyPair) = get_key_pair_from_rng(&mut rng);

    let pop = generate_proof_of_possession(&authority_kp, address);
    assert!(
        verify_proof_of_possession(&pop, wrong_kp.public(), address).is_err(),
        "PoP should fail when verified with wrong public key"
    );
}

#[test]
fn test_proof_of_possession_deterministic() {
    // Same key + same address should produce the same PoP.
    let mut rng = StdRng::from_seed([3; 32]);
    let (address, authority_kp): (SomaAddress, AuthorityKeyPair) =
        get_key_pair_from_rng(&mut rng);

    let pop1 = generate_proof_of_possession(&authority_kp, address);
    let pop2 = generate_proof_of_possession(&authority_kp, address);
    assert_eq!(pop1, pop2, "PoP should be deterministic for same inputs");
}

#[test]
fn test_proof_of_possession_bytes_roundtrip() {
    // Verify PoP survives serialization to/from bytes (as would happen in transactions).
    let mut rng = StdRng::from_seed([4; 32]);
    let (address, authority_kp): (SomaAddress, AuthorityKeyPair) =
        get_key_pair_from_rng(&mut rng);

    let pop = generate_proof_of_possession(&authority_kp, address);
    let pop_bytes = pop.as_ref().to_vec();

    // Reconstruct from bytes and verify
    let pop_restored = AuthoritySignature::from_bytes(&pop_bytes)
        .expect("Should parse PoP from valid bytes");
    assert!(
        verify_proof_of_possession(&pop_restored, authority_kp.public(), address).is_ok(),
        "PoP should verify after bytes roundtrip"
    );
}

#[test]
fn test_proof_of_possession_invalid_bytes() {
    // Garbage bytes should fail to parse as a valid PoP signature.
    let address = SomaAddress::random();
    let mut rng = StdRng::from_seed([5; 32]);
    let (_, authority_kp): (SomaAddress, AuthorityKeyPair) = get_key_pair_from_rng(&mut rng);

    let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let result = AuthoritySignature::from_bytes(&garbage);
    assert!(result.is_err(), "Garbage bytes should not parse as PoP");

    // Empty bytes should also fail
    let empty = vec![];
    let result = AuthoritySignature::from_bytes(&empty);
    assert!(result.is_err(), "Empty bytes should not parse as PoP");
}
