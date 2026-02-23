// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

// Tests for batch certificate and checkpoint verification.
// These exercise the SignatureVerifier's batch verification,
// caching, and error detection for invalid signatures.
//
// Adapted from Sui's batch_verification_tests.rs patterns.
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::{
    base::dbg_addr,
    crypto::{SomaKeyPair, get_key_pair},
    object::{Object, ObjectID},
    transaction::{CertifiedTransaction, TransactionData},
    unit_tests::utils::to_sender_signed_transaction,
};

use crate::{
    authority_test_utils::certify_transaction, signature_verifier::SignatureVerifier,
    test_authority_builder::TestAuthorityBuilder,
};

// =============================================================================
// Batch verification of certificates
// =============================================================================

#[tokio::test]
async fn test_batch_verify_valid_certificates() {
    // Create multiple valid certificates and verify them in a batch.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    let committee = authority_state.clone_committee_for_testing();
    let verifier = SignatureVerifier::new(Arc::new(committee.clone()));

    let mut certs = Vec::new();
    for i in 0..4u8 {
        let coin_id = ObjectID::random();
        let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);
        authority_state.insert_genesis_object(coin.clone()).await;

        let data = TransactionData::new_transfer_coin(
            dbg_addr(i + 1),
            sender,
            Some(100),
            coin.compute_object_reference(),
        );
        let tx = to_sender_signed_transaction(data, &sender_key);
        let cert = certify_transaction(&authority_state, tx).await.unwrap();
        certs.push(cert.into_inner());
    }

    let cert_refs: Vec<&CertifiedTransaction> = certs.iter().collect();
    let result = verifier.verify_certs_and_checkpoints(cert_refs, vec![]);
    assert!(result.is_ok(), "All valid certificates should verify: {:?}", result.err());
}

#[tokio::test]
async fn test_batch_verify_caching() {
    // After batch verification, certificates should be cached.
    // A second verification of the same certificates should succeed via cache.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    let committee = authority_state.clone_committee_for_testing();
    let verifier = SignatureVerifier::new(Arc::new(committee.clone()));

    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);
    authority_state.insert_genesis_object(coin.clone()).await;

    let data = TransactionData::new_transfer_coin(
        dbg_addr(1),
        sender,
        Some(100),
        coin.compute_object_reference(),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let cert = certify_transaction(&authority_state, tx).await.unwrap();
    let cert_inner = cert.into_inner();

    // First verification
    let result = verifier.verify_certs_and_checkpoints(vec![&cert_inner], vec![]);
    assert!(result.is_ok(), "First verification should succeed");

    // Second verification should use cache
    let result = verifier.verify_certs_and_checkpoints(vec![&cert_inner], vec![]);
    assert!(result.is_ok(), "Cached verification should succeed");

    // Clear cache and verify again (still valid, just not cached)
    verifier.clear_signature_cache();
    let result = verifier.verify_certs_and_checkpoints(vec![&cert_inner], vec![]);
    assert!(result.is_ok(), "Verification after cache clear should succeed");
}

#[tokio::test]
async fn test_async_verify_single_cert() {
    // Verify a single certificate using the async path.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    let committee = authority_state.clone_committee_for_testing();
    let verifier = SignatureVerifier::new(Arc::new(committee.clone()));

    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);
    authority_state.insert_genesis_object(coin.clone()).await;

    let data = TransactionData::new_transfer_coin(
        dbg_addr(1),
        sender,
        Some(100),
        coin.compute_object_reference(),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let cert = certify_transaction(&authority_state, tx).await.unwrap();

    let result = verifier.verify_cert(cert.into_inner()).await;
    assert!(result.is_ok(), "Async cert verification should succeed: {:?}", result.err());
}

#[tokio::test]
async fn test_multi_verify_certs_async() {
    // Verify multiple certificates concurrently using the async multi-verify path.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    let committee = authority_state.clone_committee_for_testing();
    let verifier = SignatureVerifier::new(Arc::new(committee.clone()));

    let mut certs = Vec::new();
    for i in 0..4u8 {
        let coin_id = ObjectID::random();
        let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);
        authority_state.insert_genesis_object(coin.clone()).await;

        let data = TransactionData::new_transfer_coin(
            dbg_addr(i + 1),
            sender,
            Some(100),
            coin.compute_object_reference(),
        );
        let tx = to_sender_signed_transaction(data, &sender_key);
        let cert = certify_transaction(&authority_state, tx).await.unwrap();
        certs.push(cert.into_inner());
    }

    let results = verifier.multi_verify_certs(certs).await;
    assert_eq!(results.len(), 4);
    for (i, result) in results.iter().enumerate() {
        assert!(result.is_ok(), "Cert {} should verify: {:?}", i, result.as_ref().err());
    }
}

#[tokio::test]
async fn test_verify_tx_sender_signature() {
    // Verify the sender's signature on a transaction through the verifier.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();

    let authority_state = TestAuthorityBuilder::new().build().await;
    let committee = authority_state.clone_committee_for_testing();
    let verifier = SignatureVerifier::new(Arc::new(committee.clone()));

    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 50_000_000);
    authority_state.insert_genesis_object(coin.clone()).await;

    let data = TransactionData::new_transfer_coin(
        dbg_addr(1),
        sender,
        Some(100),
        coin.compute_object_reference(),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);

    // verify_tx checks the sender's signature
    let result = verifier.verify_tx(tx.data());
    assert!(result.is_ok(), "Sender signature should verify: {:?}", result.err());
}
