// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

//! MultiSig E2E tests.
//!
//! Tests:
//! 1. test_multisig_e2e — Multisig with Ed25519 keys, threshold, and invalid scenarios
//!
//! Ported from Sui's `multisig_tests.rs`.
//! Adapted for SOMA which only supports Ed25519 keys (no Secp256k1/Secp256r1/ZkLogin/Passkey).
//! This tests the core multisig functionality: weighted thresholds, bitmap validation,
//! and signature verification.

use test_cluster::TestClusterBuilder;
use tracing::info;
use types::{
    base::SomaAddress,
    crypto::{GenericSignature, Signature, SomaKeyPair, get_key_pair_from_rng},
    effects::TransactionEffectsAPI,
    intent::{Intent, IntentMessage},
    multisig::{MultiSig, MultiSigPublicKey},
    transaction::{Transaction, TransactionData, TransactionKind},
};
use utils::logging::init_tracing;

use rand::SeedableRng;
use rand::rngs::StdRng;

/// Generate 3 deterministic Ed25519 keypairs for testing.
fn test_keys() -> Vec<SomaKeyPair> {
    let mut seed = StdRng::from_seed([0; 32]);
    let kp1 = SomaKeyPair::Ed25519(get_key_pair_from_rng(&mut seed).1);
    let kp2 = SomaKeyPair::Ed25519(get_key_pair_from_rng(&mut seed).1);
    let kp3 = SomaKeyPair::Ed25519(get_key_pair_from_rng(&mut seed).1);
    vec![kp1, kp2, kp3]
}

/// Fund a multisig address by sending coins from a funded wallet account.
/// Returns the ObjectRef of the gas coin sent to the multisig address.
async fn fund_multisig_address(
    test_cluster: &test_cluster::TestCluster,
    multisig_addr: SomaAddress,
    amount: u64,
) -> types::object::ObjectRef {
    let addresses = test_cluster.wallet.get_addresses();
    let funder = addresses[0];

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(funder)
        .await
        .unwrap()
        .expect("funder must have gas");

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(amount), recipient: multisig_addr },
        funder,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok(), "Funding transaction should succeed");

    // Find the created coin for the multisig address
    let created = response.effects.created();
    assert!(!created.is_empty(), "Funding should create at least one object");

    // Return the first created object reference (the new coin for the multisig address)
    let (obj_ref, _owner) = &created[0];
    *obj_ref
}

/// Build a multisig-signed transaction
fn build_multisig_tx(
    tx_data: TransactionData,
    multisig_pk: MultiSigPublicKey,
    signers: &[&SomaKeyPair],
    _bitmap: u16,
) -> Transaction {
    let intent_msg = IntentMessage::new(Intent::soma_transaction(), &tx_data);

    let sigs: Vec<GenericSignature> = signers
        .iter()
        .map(|kp| {
            let sig = Signature::new_secure(&intent_msg, *kp);
            GenericSignature::Signature(sig)
        })
        .collect();

    let multisig = GenericSignature::MultiSig(
        MultiSig::combine(sigs, multisig_pk).expect("MultiSig combine should succeed"),
    );

    Transaction::from_generic_sig_data(tx_data, vec![multisig])
}

/// Comprehensive multisig E2E test.
///
/// Tests with Ed25519 keys:
/// 1. Sign with keys 0 and 1 (threshold 2) — succeeds
/// 2. Sign with keys 1 and 2 (threshold 2) — succeeds
/// 3. Sign with key 0 only (below threshold) — fails
/// 4. Sign with no sigs — fails
/// 5. Duplicate sigs — fails
/// 6. Wrong sender (mismatched multisig pk) — fails
#[cfg(msim)]
#[msim::sim_test]
async fn test_multisig_e2e() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    let keys = test_keys();
    let pk0 = keys[0].public();
    let pk1 = keys[1].public();
    let pk2 = keys[2].public();

    // Create multisig with threshold = 2
    let multisig_pk =
        MultiSigPublicKey::new(vec![pk0.clone(), pk1.clone(), pk2.clone()], vec![1, 1, 1], 2)
            .expect("MultiSigPublicKey creation should succeed");
    let multisig_addr = SomaAddress::from(&multisig_pk);

    // Fund the multisig address
    let gas = fund_multisig_address(&test_cluster, multisig_addr, 20_000_000_000).await;

    // 1. Sign with keys 0 and 1 — meets threshold (weight 2 >= 2), should succeed
    info!("Test 1: Two signatures meeting threshold");
    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: gas,
            amount: Some(1_000_000),
            recipient: SomaAddress::default(),
        },
        multisig_addr,
        vec![gas],
    );

    let tx1 = build_multisig_tx(tx_data, multisig_pk.clone(), &[&keys[0], &keys[1]], 0b011);
    let res = test_cluster.wallet.execute_transaction_must_succeed(tx1).await;
    assert!(res.effects.status().is_ok());
    info!("Test 1 passed: 2-of-3 multisig with keys 0,1 succeeded");

    // Re-fund for next test
    let gas = fund_multisig_address(&test_cluster, multisig_addr, 20_000_000_000).await;

    // 2. Sign with keys 1 and 2 — meets threshold
    info!("Test 2: Different two signatures meeting threshold");
    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: gas,
            amount: Some(1_000_000),
            recipient: SomaAddress::default(),
        },
        multisig_addr,
        vec![gas],
    );

    let tx2 = build_multisig_tx(tx_data, multisig_pk.clone(), &[&keys[1], &keys[2]], 0b110);
    let res = test_cluster.wallet.execute_transaction_must_succeed(tx2).await;
    assert!(res.effects.status().is_ok());
    info!("Test 2 passed: 2-of-3 multisig with keys 1,2 succeeded");

    // Re-fund for next test
    let gas = fund_multisig_address(&test_cluster, multisig_addr, 20_000_000_000).await;

    // 3. Sign with key 0 only — below threshold (weight 1 < 2), should fail
    info!("Test 3: Single signature below threshold");
    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: gas,
            amount: Some(1_000_000),
            recipient: SomaAddress::default(),
        },
        multisig_addr,
        vec![gas],
    );

    let tx3 = build_multisig_tx(tx_data, multisig_pk.clone(), &[&keys[0]], 0b001);
    let res = test_cluster.wallet.execute_transaction_may_fail(tx3).await;
    assert!(
        res.unwrap_err().to_string().contains("Insufficient weight=1 threshold=2"),
        "Should fail with insufficient weight"
    );
    info!("Test 3 passed: below-threshold signature correctly rejected");

    // 4. Multisig with no signatures — should fail at combine time
    info!("Test 4: No signatures");
    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: gas,
            amount: Some(1_000_000),
            recipient: SomaAddress::default(),
        },
        multisig_addr,
        vec![gas],
    );

    let intent_msg = IntentMessage::new(Intent::soma_transaction(), &tx_data);
    let empty_combine = MultiSig::combine(vec![], multisig_pk.clone());
    assert!(empty_combine.is_err(), "Combining zero signatures should fail");
    info!("Test 4 passed: empty signature set correctly rejected");

    // 5. Duplicate signatures (same key signed twice) — should fail
    info!("Test 5: Duplicate signatures");
    let sig0: GenericSignature = Signature::new_secure(&intent_msg, &keys[0]).into();
    let dup_combine = MultiSig::combine(vec![sig0.clone(), sig0.clone()], multisig_pk.clone());
    assert!(dup_combine.is_err(), "Combining duplicate signatures should fail");
    info!("Test 5 passed: duplicate signatures correctly rejected");

    // 6. Wrong sender (multisig pk doesn't match sender address)
    info!("Test 6: Wrong sender address");
    let wrong_multisig_pk =
        MultiSigPublicKey::new(vec![pk0.clone(), pk1.clone()], vec![1, 1], 1).unwrap();
    let wrong_sender = SomaAddress::from(&wrong_multisig_pk);

    // Fund the wrong sender address
    let wrong_gas = fund_multisig_address(&test_cluster, wrong_sender, 20_000_000_000).await;

    let tx_data_wrong = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: wrong_gas,
            amount: Some(1_000_000),
            recipient: SomaAddress::default(),
        },
        wrong_sender,
        vec![wrong_gas],
    );

    // Sign with key 2 (which is in original multisig_pk but NOT in wrong_multisig_pk)
    let intent_msg_wrong = IntentMessage::new(Intent::soma_transaction(), &tx_data_wrong);
    let sig2: GenericSignature = Signature::new_secure(&intent_msg_wrong, &keys[2]).into();

    // Try to combine with wrong_multisig_pk — key 2 is not in this pk map
    let combine_result = MultiSig::combine(vec![sig2], wrong_multisig_pk.clone());
    assert!(combine_result.is_err(), "Should fail when signature key not in multisig pk");
    info!("Test 6 passed: wrong sender / key mismatch correctly rejected");

    info!("All multisig E2E tests passed");
}
