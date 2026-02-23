// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use crate::base::SomaAddress;
use crate::crypto::{
    AuthenticatorTrait, GenericSignature, Signature, SomaKeyPair, get_key_pair_from_rng,
};
use crate::intent::{Intent, IntentMessage, PersonalMessage};
use crate::multisig::{MultiSig, MultiSigPublicKey};
use rand::{SeedableRng, rngs::StdRng};

/// Helper: generate N Ed25519 key pairs from a deterministic seed.
fn ed25519_keys(n: usize) -> Vec<SomaKeyPair> {
    let mut rng = StdRng::from_seed([42; 32]);
    (0..n).map(|_| SomaKeyPair::Ed25519(get_key_pair_from_rng(&mut rng).1)).collect()
}

/// End-to-end: create MultiSigPublicKey from 3 Ed25519 keys (weights 1,1,1,
/// threshold 2), sign a message with 2 of them, combine into MultiSig, and
/// verify that authentication succeeds.
#[test]
fn test_multisig_verify_ed25519() {
    let keys = ed25519_keys(3);
    let pks: Vec<_> = keys.iter().map(|k| k.public()).collect();

    let multisig_pk = MultiSigPublicKey::new(pks, vec![1, 1, 1], 2).expect("valid multisig pk");

    let multisig_address = SomaAddress::from(&multisig_pk);

    let msg = IntentMessage::new(
        Intent::soma_transaction(),
        PersonalMessage { message: b"test_multisig_verify_ed25519".to_vec() },
    );

    // Sign with key 0 and key 2 (weight sum = 2, meets threshold 2)
    let sig0: GenericSignature = Signature::new_secure(&msg, &keys[0]).into();
    let sig2: GenericSignature = Signature::new_secure(&msg, &keys[2]).into();

    let multisig =
        MultiSig::combine(vec![sig0, sig2], multisig_pk).expect("combine should succeed");

    // Verification should succeed
    let result = multisig.verify_claims(&msg, multisig_address);
    assert!(result.is_ok(), "MultiSig verification should succeed: {:?}", result.err());
}

/// Sign with keys whose combined weight is below the threshold. Verification
/// must fail with an insufficient-weight error.
#[test]
fn test_multisig_insufficient_weight() {
    let keys = ed25519_keys(3);
    let pks: Vec<_> = keys.iter().map(|k| k.public()).collect();

    // Weights: [1, 1, 1], threshold: 3  =>  any 2 keys only sum to 2 < 3
    let multisig_pk = MultiSigPublicKey::new(pks, vec![1, 1, 1], 3).expect("valid multisig pk");

    let multisig_address = SomaAddress::from(&multisig_pk);

    let msg = IntentMessage::new(
        Intent::soma_transaction(),
        PersonalMessage { message: b"test_multisig_insufficient_weight".to_vec() },
    );

    // Sign with only key 0 and key 1 (weight sum = 2 < threshold 3)
    let sig0: GenericSignature = Signature::new_secure(&msg, &keys[0]).into();
    let sig1: GenericSignature = Signature::new_secure(&msg, &keys[1]).into();

    let multisig =
        MultiSig::combine(vec![sig0, sig1], multisig_pk).expect("combine should succeed");

    let result = multisig.verify_claims(&msg, multisig_address);
    assert!(result.is_err(), "Verification must fail when combined weight < threshold");

    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("Insufficient weight"),
        "Error should mention insufficient weight, got: {}",
        err_msg
    );
}

/// Sign with exactly the threshold weight (not more). Verification should
/// succeed since the check is `weight_sum >= threshold`.
#[test]
fn test_multisig_exact_threshold() {
    let keys = ed25519_keys(3);
    let pks: Vec<_> = keys.iter().map(|k| k.public()).collect();

    // Weights: [2, 3, 5], threshold: 5
    // key[0] alone has weight 2 < 5, keys[0]+[1] = 5 == threshold, exactly enough
    let multisig_pk = MultiSigPublicKey::new(pks, vec![2, 3, 5], 5).expect("valid multisig pk");

    let multisig_address = SomaAddress::from(&multisig_pk);

    let msg = IntentMessage::new(
        Intent::soma_transaction(),
        PersonalMessage { message: b"test_multisig_exact_threshold".to_vec() },
    );

    // Sign with key 0 (weight=2) and key 1 (weight=3) => total = 5 == threshold
    let sig0: GenericSignature = Signature::new_secure(&msg, &keys[0]).into();
    let sig1: GenericSignature = Signature::new_secure(&msg, &keys[1]).into();

    let multisig =
        MultiSig::combine(vec![sig0, sig1], multisig_pk).expect("combine should succeed");

    let result = multisig.verify_claims(&msg, multisig_address);
    assert!(
        result.is_ok(),
        "MultiSig verification should succeed when weight == threshold: {:?}",
        result.err()
    );
}

/// The same set of public keys and threshold must always derive the same
/// SomaAddress, regardless of how many times the derivation is performed.
#[test]
fn test_multisig_address_deterministic() {
    let keys = ed25519_keys(3);
    let pks: Vec<_> = keys.iter().map(|k| k.public()).collect();

    let multisig_pk_a =
        MultiSigPublicKey::new(pks.clone(), vec![1, 2, 3], 3).expect("valid multisig pk");
    let multisig_pk_b =
        MultiSigPublicKey::new(pks.clone(), vec![1, 2, 3], 3).expect("valid multisig pk");

    let addr_a = SomaAddress::from(&multisig_pk_a);
    let addr_b = SomaAddress::from(&multisig_pk_b);

    assert_eq!(addr_a, addr_b, "Same keys + threshold must yield the same address");

    // A different threshold should produce a different address
    let multisig_pk_c =
        MultiSigPublicKey::new(pks.clone(), vec![1, 2, 3], 4).expect("valid multisig pk");
    let addr_c = SomaAddress::from(&multisig_pk_c);
    assert_ne!(addr_a, addr_c, "Different thresholds should yield different addresses");

    // Different weights should produce a different address
    let multisig_pk_d = MultiSigPublicKey::new(pks, vec![2, 2, 3], 3).expect("valid multisig pk");
    let addr_d = SomaAddress::from(&multisig_pk_d);
    assert_ne!(addr_a, addr_d, "Different weights should yield different addresses");
}
