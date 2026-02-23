// Tests for the AuthorityServer / ValidatorService layer.
// These exercise the server's methods directly (not via gRPC transport),
// verifying that the server correctly delegates to AuthorityState.
//
// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-core/src/unit_tests/server_tests.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use fastcrypto::ed25519::Ed25519KeyPair;
use types::{
    base::dbg_addr,
    crypto::{SomaKeyPair, get_key_pair},
    effects::{ExecutionStatus, TransactionEffectsAPI},
    envelope::Message as _,
    messages_grpc::{ObjectInfoRequest, TransactionInfoRequest},
    object::{Object, ObjectID},
    transaction::TransactionData,
    unit_tests::utils::to_sender_signed_transaction,
};

use crate::{
    authority_server::AuthorityServer, authority_test_utils::send_and_confirm_transaction,
    test_authority_builder::TestAuthorityBuilder,
};

// =============================================================================
// handle_transaction tests
// =============================================================================

#[tokio::test]
async fn test_handle_transaction_basic() {
    // Submit a TransferCoin via the ValidatorService's handle_transaction_for_benchmarking
    // method which wraps handle_transaction internally.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 10_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    let server = AuthorityServer::new_for_test(authority_state.clone());
    let validator_service = crate::authority_server::ValidatorService::new(
        server.state.clone(),
        std::sync::Arc::new(crate::consensus_adapter::ConsensusAdapter::new(
            std::sync::Arc::new(crate::mysticeti_adapter::LazyMysticetiClient::new()),
            crate::checkpoints::CheckpointStore::new_for_tests(),
            server.state.name,
            100_000,
            100_000,
            None,
            None,
            server.state.epoch_store_for_testing().protocol_config().clone(),
        )),
    );

    let data = TransactionData::new_transfer_coin(
        recipient,
        sender,
        Some(1000),
        coin.compute_object_reference(),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);

    // handle_transaction_for_benchmarking returns a tonic::Response<HandleTransactionResponse>
    let response = validator_service.handle_transaction_for_benchmarking(tx).await;

    assert!(response.is_ok(), "handle_transaction should succeed: {:?}", response.err());
    let handle_response = response.unwrap().into_inner();
    // The response should contain a Signed status (authority signed the tx)
    assert!(
        matches!(handle_response.status, types::messages_grpc::TransactionStatus::Signed(_)),
        "Response should contain a Signed status"
    );
}

// =============================================================================
// handle_object_info_request tests
// =============================================================================

#[tokio::test]
async fn test_handle_object_info_request() {
    // Insert an object, then request its info via the authority state's
    // handle_object_info_request method (the same path the server delegates to).
    let (sender, _sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let object_id = ObjectID::random();
    let obj = Object::with_id_owner_for_testing(object_id, sender);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(obj.clone()).await;

    let request = ObjectInfoRequest { object_id };
    let response = authority_state.handle_object_info_request(request).await;

    assert!(response.is_ok(), "Object info request should succeed: {:?}", response.err());
    let info = response.unwrap();
    assert_eq!(info.object.id(), object_id, "Returned object should have the requested ID");
    assert_eq!(
        info.object.owner,
        types::object::Owner::AddressOwner(sender),
        "Returned object should be owned by the expected sender"
    );
}

#[tokio::test]
async fn test_handle_object_info_request_not_found() {
    // Requesting info for a non-existent object should return an error.
    let authority_state = TestAuthorityBuilder::new().build().await;
    let nonexistent_id = ObjectID::random();

    let request = ObjectInfoRequest { object_id: nonexistent_id };
    let response = authority_state.handle_object_info_request(request).await;

    assert!(response.is_err(), "Should fail for non-existent object");
}

// =============================================================================
// handle_transaction_info_request tests
// =============================================================================

#[tokio::test]
async fn test_handle_transaction_info_request() {
    // Execute a transaction, then request its info by digest.
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(1);
    let coin_id = ObjectID::random();
    let coin = Object::with_id_owner_coin_for_testing(coin_id, sender, 10_000_000);

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(coin.clone()).await;

    let data = TransactionData::new_transfer_coin(
        recipient,
        sender,
        Some(1000),
        coin.compute_object_reference(),
    );
    let tx = to_sender_signed_transaction(data, &sender_key);
    let tx_digest = *tx.digest();

    let (_cert, effects) = send_and_confirm_transaction(&authority_state, tx).await.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // Now request transaction info
    let request = TransactionInfoRequest { transaction_digest: tx_digest };
    let response = authority_state.handle_transaction_info_request(request).await;

    assert!(response.is_ok(), "Transaction info request should succeed: {:?}", response.err());
    let info = response.unwrap();

    // The returned transaction data should match the original digest
    assert_eq!(
        info.transaction.digest(),
        tx_digest,
        "Returned transaction should match the executed one"
    );

    // Status should be Executed (since we already executed and committed it)
    assert!(
        matches!(info.status, types::messages_grpc::TransactionStatus::Executed(_, _)),
        "Status should be Executed after transaction has been committed"
    );
}

#[tokio::test]
async fn test_handle_transaction_info_request_not_found() {
    // Requesting info for a non-existent transaction should return an error.
    let authority_state = TestAuthorityBuilder::new().build().await;

    let request =
        TransactionInfoRequest { transaction_digest: types::digests::TransactionDigest::random() };
    let response = authority_state.handle_transaction_info_request(request).await;

    assert!(response.is_err(), "Should fail for non-existent transaction digest");
}
