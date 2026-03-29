// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::ask::{Ask, AskStatus};
use types::base::{SomaAddress, dbg_addr};
use types::bid::{Bid, BidStatus};
use types::crypto::{SomaKeyPair, get_key_pair};
use types::digests::{ResponseDigest, TaskDigest, TransactionDigest};
use types::effects::{
    ExecutionFailureStatus, ExecutionStatus, SignedTransactionEffects, TransactionEffectsAPI,
};
use types::error::SomaError;
use types::object::{CoinType, Object, ObjectID, ObjectRef, ObjectType, Owner};
use types::settlement::{SellerRating, Settlement};
use types::transaction::{
    AcceptBidArgs, CreateAskArgs, CreateBidArgs, TransactionData, TransactionKind,
};
use types::unit_tests::utils::to_sender_signed_transaction;
use types::vault::SellerVault;

use crate::authority::AuthorityState;
use crate::authority_test_utils::send_and_confirm_transaction_;
use crate::test_authority_builder::TestAuthorityBuilder;

// =============================================================================
// Helpers
// =============================================================================

struct TransactionResult {
    authority_state: Arc<AuthorityState>,
    txn_result: Result<SignedTransactionEffects, SomaError>,
}

fn make_task_digest() -> TaskDigest {
    TaskDigest::new(types::digests::Digest::random().into())
}

fn make_response_digest() -> ResponseDigest {
    ResponseDigest::new(types::digests::Digest::random().into())
}

fn make_usdc_coin(id: ObjectID, owner: SomaAddress, balance: u64) -> Object {
    Object::new_coin(
        id,
        CoinType::Usdc,
        balance,
        Owner::AddressOwner(owner),
        TransactionDigest::genesis_marker(),
    )
}

async fn execute_marketplace_tx(
    objects: Vec<Object>,
    kind: TransactionKind,
    sender: SomaAddress,
    sender_key: &SomaKeyPair,
    gas: Object,
) -> TransactionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;
    for obj in objects {
        authority_state.insert_genesis_object(obj).await;
    }

    let data = TransactionData::new(kind, sender, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, sender_key);
    let txn_result = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .map(|(_, effects)| effects);

    TransactionResult { authority_state, txn_result }
}

/// Helper to run a CreateAsk and return (authority_state, ask_id, effects).
async fn create_ask_helper(
    buyer: SomaAddress,
    buyer_key: &SomaKeyPair,
) -> (Arc<AuthorityState>, ObjectID, SignedTransactionEffects) {
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let gas_ref = gas.compute_object_reference();

    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::CreateAsk(CreateAskArgs {
        task_digest: make_task_digest(),
        max_price_per_bid: 1_000_000,
        num_bids_wanted: 1,
        timeout_ms: 60_000, // 1 minute
    });
    let data = TransactionData::new(kind, buyer, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();

    let effects_data = effects.clone().into_data();
    assert_eq!(*effects_data.status(), ExecutionStatus::Success);
    let ask_id = effects_data.created()[0].0 .0;

    (authority_state, ask_id, effects)
}

// =============================================================================
// CreateAsk tests
// =============================================================================

#[tokio::test]
async fn test_create_ask_success() {
    let (buyer, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);

    let res = execute_marketplace_tx(
        vec![],
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: make_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 3,
            timeout_ms: 60_000,
        }),
        buyer,
        &SomaKeyPair::Ed25519(key),
        gas,
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);
    // Should create one Ask object
    assert_eq!(effects.created().len(), 1);

    let ask_id = effects.created()[0].0 .0;
    let ask_obj = res.authority_state.get_object(&ask_id).await.unwrap();
    let ask: Ask = ask_obj.deserialize_contents(ObjectType::Ask).unwrap();
    assert_eq!(ask.buyer, buyer);
    assert_eq!(ask.max_price_per_bid, 1_000_000);
    assert_eq!(ask.num_bids_wanted, 3);
    assert_eq!(ask.status, AskStatus::Open);
    assert_eq!(ask.accepted_bid_count, 0);
}

#[tokio::test]
async fn test_create_ask_zero_price_rejected() {
    let (buyer, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);

    let res = execute_marketplace_tx(
        vec![],
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: make_task_digest(),
            max_price_per_bid: 0,
            num_bids_wanted: 1,
            timeout_ms: 60_000,
        }),
        buyer,
        &SomaKeyPair::Ed25519(key),
        gas,
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(matches!(
        effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InvalidArguments { .. } }
    ));
}

#[tokio::test]
async fn test_create_ask_zero_bids_rejected() {
    let (buyer, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);

    let res = execute_marketplace_tx(
        vec![],
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: make_task_digest(),
            max_price_per_bid: 1_000,
            num_bids_wanted: 0,
            timeout_ms: 60_000,
        }),
        buyer,
        &SomaKeyPair::Ed25519(key),
        gas,
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(matches!(
        effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InvalidArguments { .. } }
    ));
}

#[tokio::test]
async fn test_create_ask_timeout_too_short_rejected() {
    let (buyer, key): (_, Ed25519KeyPair) = get_key_pair();
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);

    // min_ask_timeout_ms default is 10_000 (10 seconds)
    let res = execute_marketplace_tx(
        vec![],
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: make_task_digest(),
            max_price_per_bid: 1_000,
            num_bids_wanted: 1,
            timeout_ms: 1, // way too short
        }),
        buyer,
        &SomaKeyPair::Ed25519(key),
        gas,
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert!(matches!(
        effects.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InvalidArguments { .. } }
    ));
}

// =============================================================================
// CancelAsk tests
// =============================================================================

#[tokio::test]
async fn test_cancel_ask_success() {
    let (buyer, key): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(key);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    // Now cancel it
    let gas2 = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let gas2_ref = gas2.compute_object_reference();
    authority_state.insert_genesis_object(gas2).await;

    let kind = TransactionKind::CancelAsk { ask_id };
    let data = TransactionData::new(kind, buyer, vec![gas2_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();

    let effects_data = effects.into_data();
    assert_eq!(*effects_data.status(), ExecutionStatus::Success);

    // Verify ask is now cancelled
    let ask_obj = authority_state.get_object(&ask_id).await.unwrap();
    let ask: Ask = ask_obj.deserialize_contents(ObjectType::Ask).unwrap();
    assert_eq!(ask.status, AskStatus::Cancelled);
}

#[tokio::test]
async fn test_cancel_ask_wrong_sender_rejected() {
    let (buyer, key): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(key);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    // Try to cancel as a different sender
    let (other, other_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let other_key = SomaKeyPair::Ed25519(other_key_ed);
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), other, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::CancelAsk { ask_id };
    let data = TransactionData::new(kind, other, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &other_key);
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            let effects_data = effects.into_data();
            assert!(matches!(
                effects_data.status(),
                ExecutionStatus::Failure {
                    error: ExecutionFailureStatus::InvalidOwnership { .. }
                }
            ));
        }
        Err(_) => {
            // Also acceptable if pre-certification check rejects
        }
    }
}

// =============================================================================
// CreateBid tests
// =============================================================================

#[tokio::test]
async fn test_create_bid_success() {
    let (buyer, key): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(key);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    // Seller creates a bid
    let (seller, seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller_key = SomaKeyPair::Ed25519(seller_key_ed);
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::CreateBid(CreateBidArgs {
        ask_id,
        price: 500_000,
        response_digest: make_response_digest(),
    });
    let data = TransactionData::new(kind, seller, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();

    let effects_data = effects.into_data();
    assert_eq!(*effects_data.status(), ExecutionStatus::Success);
    assert_eq!(effects_data.created().len(), 1);

    let bid_id = effects_data.created()[0].0 .0;
    let bid_obj = authority_state.get_object(&bid_id).await.unwrap();
    let bid: Bid = bid_obj.deserialize_contents(ObjectType::Bid).unwrap();
    assert_eq!(bid.seller, seller);
    assert_eq!(bid.ask_id, ask_id);
    assert_eq!(bid.price, 500_000);
    assert_eq!(bid.status, BidStatus::Pending);
}

#[tokio::test]
async fn test_create_bid_price_too_high_rejected() {
    let (buyer, key): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(key);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    let (seller, seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller_key = SomaKeyPair::Ed25519(seller_key_ed);
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    // max_price_per_bid is 1_000_000, bid 2_000_000
    let kind = TransactionKind::CreateBid(CreateBidArgs {
        ask_id,
        price: 2_000_000,
        response_digest: make_response_digest(),
    });
    let data = TransactionData::new(kind, seller, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();

    let effects_data = effects.into_data();
    assert!(matches!(
        effects_data.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::BidPriceTooHigh }
    ));
}

#[tokio::test]
async fn test_create_bid_seller_is_buyer_rejected() {
    let (buyer, key): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(key);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    // Buyer tries to bid on own ask
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::CreateBid(CreateBidArgs {
        ask_id,
        price: 500_000,
        response_digest: make_response_digest(),
    });
    let data = TransactionData::new(kind, buyer, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();

    let effects_data = effects.into_data();
    assert!(matches!(
        effects_data.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::SellerCannotBidOnOwnAsk }
    ));
}

#[tokio::test]
async fn test_create_bid_zero_price_rejected() {
    let (buyer, key): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(key);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    let (seller, seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller_key = SomaKeyPair::Ed25519(seller_key_ed);
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::CreateBid(CreateBidArgs {
        ask_id,
        price: 0,
        response_digest: make_response_digest(),
    });
    let data = TransactionData::new(kind, seller, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();

    let effects_data = effects.into_data();
    assert!(matches!(
        effects_data.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InvalidArguments { .. } }
    ));
}

// =============================================================================
// AcceptBid tests
// =============================================================================

/// Full happy path: create ask → create bid → accept bid → verify settlement
#[tokio::test]
async fn test_accept_bid_success() {
    let (buyer, buyer_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(buyer_key_ed);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    // Seller creates bid
    let (seller, seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller_key = SomaKeyPair::Ed25519(seller_key_ed);
    let seller_gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller, 100_000_000);
    let seller_gas_ref = seller_gas.compute_object_reference();
    authority_state.insert_genesis_object(seller_gas).await;

    let bid_price = 500_000u64;
    let kind = TransactionKind::CreateBid(CreateBidArgs {
        ask_id,
        price: bid_price,
        response_digest: make_response_digest(),
    });
    let data = TransactionData::new(kind, seller, vec![seller_gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller_key);
    let (_, bid_effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let bid_effects_data = bid_effects.into_data();
    assert_eq!(*bid_effects_data.status(), ExecutionStatus::Success);
    let bid_id = bid_effects_data.created()[0].0 .0;

    // Buyer creates USDC coin and accepts bid
    let usdc_id = ObjectID::random();
    let usdc_coin = make_usdc_coin(usdc_id, buyer, 10_000_000);
    let usdc_ref = usdc_coin.compute_object_reference();
    authority_state.insert_genesis_object(usdc_coin).await;

    let buyer_gas2 = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let buyer_gas2_ref = buyer_gas2.compute_object_reference();
    authority_state.insert_genesis_object(buyer_gas2).await;

    let kind = TransactionKind::AcceptBid(AcceptBidArgs {
        ask_id,
        bid_id,
        payment_coin: usdc_ref,
    });
    let data = TransactionData::new(kind, buyer, vec![buyer_gas2_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert_eq!(*effects_data.status(), ExecutionStatus::Success);

    // Should create 2 objects: SellerVault + Settlement
    assert_eq!(effects_data.created().len(), 2);

    // Verify bid is now Accepted
    let bid_obj = authority_state.get_object(&bid_id).await.unwrap();
    let bid: Bid = bid_obj.deserialize_contents(ObjectType::Bid).unwrap();
    assert_eq!(bid.status, BidStatus::Accepted);

    // Verify ask is now Filled (num_bids_wanted=1, accepted_bid_count=1)
    let ask_obj = authority_state.get_object(&ask_id).await.unwrap();
    let ask: Ask = ask_obj.deserialize_contents(ObjectType::Ask).unwrap();
    assert_eq!(ask.status, AskStatus::Filled);
    assert_eq!(ask.accepted_bid_count, 1);

    // Verify USDC coin balance reduced by bid.price
    let usdc_after = authority_state.get_object(&usdc_id).await.unwrap();
    assert_eq!(usdc_after.as_coin().unwrap(), 10_000_000 - bid_price);

    // Find and verify settlement
    let mut settlement_found = false;
    let mut vault_found = false;
    for (oref, owner) in effects_data.created() {
        let obj = authority_state.get_object(&oref.0).await.unwrap();
        if let Some(settlement) = obj.deserialize_contents::<Settlement>(ObjectType::Settlement) {
            settlement_found = true;
            assert_eq!(settlement.buyer, buyer);
            assert_eq!(settlement.seller, seller);
            assert_eq!(settlement.seller_rating, SellerRating::Positive);
        }
        if let Some(vault) = obj.deserialize_contents::<SellerVault>(ObjectType::SellerVault) {
            vault_found = true;
            assert_eq!(vault.owner, seller);
            // Vault balance = bid_price - marketplace_fee
            // marketplace_fee_bps default = 250 (2.5%)
            let expected_fee = bid_price * 250 / 10_000;
            assert_eq!(vault.balance, bid_price - expected_fee);
        }
    }
    assert!(settlement_found, "Settlement should be created");
    assert!(vault_found, "SellerVault should be created");
}

#[tokio::test]
async fn test_accept_bid_wrong_coin_type_rejected() {
    let (buyer, buyer_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(buyer_key_ed);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    // Seller creates bid
    let (seller, seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller_key = SomaKeyPair::Ed25519(seller_key_ed);
    let seller_gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller, 100_000_000);
    let seller_gas_ref = seller_gas.compute_object_reference();
    authority_state.insert_genesis_object(seller_gas).await;

    let kind = TransactionKind::CreateBid(CreateBidArgs {
        ask_id,
        price: 500_000,
        response_digest: make_response_digest(),
    });
    let data = TransactionData::new(kind, seller, vec![seller_gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller_key);
    let (_, bid_effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let bid_id = bid_effects.into_data().created()[0].0 .0;

    // Try to pay with SOMA coin instead of USDC
    let soma_coin_id = ObjectID::random();
    let soma_coin = Object::with_id_owner_coin_for_testing(soma_coin_id, buyer, 10_000_000);
    let soma_ref = soma_coin.compute_object_reference();
    authority_state.insert_genesis_object(soma_coin).await;

    let buyer_gas2 = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let buyer_gas2_ref = buyer_gas2.compute_object_reference();
    authority_state.insert_genesis_object(buyer_gas2).await;

    let kind = TransactionKind::AcceptBid(AcceptBidArgs {
        ask_id,
        bid_id,
        payment_coin: soma_ref,
    });
    let data = TransactionData::new(kind, buyer, vec![buyer_gas2_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert!(matches!(
        effects_data.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::WrongCoinTypeForPayment }
    ));
}

#[tokio::test]
async fn test_accept_bid_insufficient_balance_rejected() {
    let (buyer, buyer_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(buyer_key_ed);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    // Seller bids 500_000
    let (seller, seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller_key = SomaKeyPair::Ed25519(seller_key_ed);
    let seller_gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller, 100_000_000);
    let seller_gas_ref = seller_gas.compute_object_reference();
    authority_state.insert_genesis_object(seller_gas).await;

    let kind = TransactionKind::CreateBid(CreateBidArgs {
        ask_id,
        price: 500_000,
        response_digest: make_response_digest(),
    });
    let data = TransactionData::new(kind, seller, vec![seller_gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller_key);
    let (_, bid_effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let bid_id = bid_effects.into_data().created()[0].0 .0;

    // USDC coin with only 100 (not enough for bid.price of 500_000)
    let usdc_id = ObjectID::random();
    let usdc_coin = make_usdc_coin(usdc_id, buyer, 100);
    let usdc_ref = usdc_coin.compute_object_reference();
    authority_state.insert_genesis_object(usdc_coin).await;

    let buyer_gas2 = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let buyer_gas2_ref = buyer_gas2.compute_object_reference();
    authority_state.insert_genesis_object(buyer_gas2).await;

    let kind = TransactionKind::AcceptBid(AcceptBidArgs {
        ask_id,
        bid_id,
        payment_coin: usdc_ref,
    });
    let data = TransactionData::new(kind, buyer, vec![buyer_gas2_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert!(matches!(
        effects_data.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientCoinBalance }
    ));
}

#[tokio::test]
async fn test_accept_bid_over_accept_rejected() {
    // Create ask with num_bids_wanted=1, accept one bid, then try to accept another
    let (buyer, buyer_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(buyer_key_ed);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    // Two sellers create bids
    let (seller1, seller1_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller1_key = SomaKeyPair::Ed25519(seller1_key_ed);
    let (seller2, seller2_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller2_key = SomaKeyPair::Ed25519(seller2_key_ed);

    // Seller 1 bids
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller1, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;
    let kind = TransactionKind::CreateBid(CreateBidArgs {
        ask_id,
        price: 500_000,
        response_digest: make_response_digest(),
    });
    let data = TransactionData::new(kind, seller1, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller1_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true).await.unwrap();
    let bid1_id = effects.into_data().created()[0].0 .0;

    // Seller 2 bids
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller2, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;
    let kind = TransactionKind::CreateBid(CreateBidArgs {
        ask_id,
        price: 400_000,
        response_digest: make_response_digest(),
    });
    let data = TransactionData::new(kind, seller2, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller2_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true).await.unwrap();
    let bid2_id = effects.into_data().created()[0].0 .0;

    // Accept bid 1
    let usdc = make_usdc_coin(ObjectID::random(), buyer, 10_000_000);
    let usdc_ref = usdc.compute_object_reference();
    authority_state.insert_genesis_object(usdc).await;
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::AcceptBid(AcceptBidArgs {
        ask_id,
        bid_id: bid1_id,
        payment_coin: usdc_ref,
    });
    let data = TransactionData::new(kind, buyer, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true).await.unwrap();
    assert_eq!(*effects.into_data().status(), ExecutionStatus::Success);

    // Try to accept bid 2 — ask is already Filled (num_bids_wanted=1)
    let usdc2 = make_usdc_coin(ObjectID::random(), buyer, 10_000_000);
    let usdc2_ref = usdc2.compute_object_reference();
    authority_state.insert_genesis_object(usdc2).await;
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::AcceptBid(AcceptBidArgs {
        ask_id,
        bid_id: bid2_id,
        payment_coin: usdc2_ref,
    });
    let data = TransactionData::new(kind, buyer, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true).await.unwrap();
    let effects_data = effects.into_data();
    assert!(matches!(
        effects_data.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::AskAlreadyFilled }
            | ExecutionStatus::Failure { error: ExecutionFailureStatus::AskNotOpen }
    ));
}

// =============================================================================
// RateSeller tests
// =============================================================================

/// Helper: run the full ask→bid→accept flow, return (authority_state, settlement_id, buyer, buyer_key)
async fn create_settlement_helper() -> (Arc<AuthorityState>, ObjectID, SomaAddress, SomaKeyPair) {
    let (buyer, buyer_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let buyer_key = SomaKeyPair::Ed25519(buyer_key_ed);
    let (authority_state, ask_id, _) = create_ask_helper(buyer, &buyer_key).await;

    // Seller bids
    let (seller, seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller_key = SomaKeyPair::Ed25519(seller_key_ed);
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;
    let kind = TransactionKind::CreateBid(CreateBidArgs {
        ask_id,
        price: 500_000,
        response_digest: make_response_digest(),
    });
    let data = TransactionData::new(kind, seller, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true).await.unwrap();
    let bid_id = effects.into_data().created()[0].0 .0;

    // Buyer accepts
    let usdc = make_usdc_coin(ObjectID::random(), buyer, 10_000_000);
    let usdc_ref = usdc.compute_object_reference();
    authority_state.insert_genesis_object(usdc).await;
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::AcceptBid(AcceptBidArgs {
        ask_id,
        bid_id,
        payment_coin: usdc_ref,
    });
    let data = TransactionData::new(kind, buyer, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true).await.unwrap();
    let effects_data = effects.into_data();
    assert_eq!(*effects_data.status(), ExecutionStatus::Success);

    // Find the settlement
    let mut settlement_id = None;
    for (oref, _owner) in effects_data.created() {
        let obj = authority_state.get_object(&oref.0).await.unwrap();
        if obj.deserialize_contents::<Settlement>(ObjectType::Settlement).is_some() {
            settlement_id = Some(oref.0);
        }
    }

    (authority_state, settlement_id.unwrap(), buyer, buyer_key)
}

#[tokio::test]
async fn test_rate_seller_negative_success() {
    let (authority_state, settlement_id, buyer, buyer_key) = create_settlement_helper().await;

    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::RateSeller { settlement_id };
    let data = TransactionData::new(kind, buyer, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert_eq!(*effects_data.status(), ExecutionStatus::Success);

    // Verify settlement is now Negative
    let settlement_obj = authority_state.get_object(&settlement_id).await.unwrap();
    let settlement: Settlement = settlement_obj
        .deserialize_contents(ObjectType::Settlement)
        .unwrap();
    assert_eq!(settlement.seller_rating, SellerRating::Negative);
}

#[tokio::test]
async fn test_rate_seller_double_rate_rejected() {
    let (authority_state, settlement_id, buyer, buyer_key) = create_settlement_helper().await;

    // First rating
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;
    let kind = TransactionKind::RateSeller { settlement_id };
    let data = TransactionData::new(kind, buyer, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    assert_eq!(*effects.into_data().status(), ExecutionStatus::Success);

    // Second rating — should fail
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), buyer, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;
    let kind = TransactionKind::RateSeller { settlement_id };
    let data = TransactionData::new(kind, buyer, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &buyer_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert!(matches!(
        effects_data.status(),
        ExecutionStatus::Failure {
            error: ExecutionFailureStatus::SettlementAlreadyRatedNegative
        }
    ));
}

#[tokio::test]
async fn test_rate_seller_wrong_sender_rejected() {
    let (authority_state, settlement_id, _buyer, _buyer_key) = create_settlement_helper().await;

    // Random person tries to rate
    let (other, other_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let other_key = SomaKeyPair::Ed25519(other_key_ed);
    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), other, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::RateSeller { settlement_id };
    let data = TransactionData::new(kind, other, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &other_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert!(matches!(
        effects_data.status(),
        ExecutionStatus::Failure {
            error: ExecutionFailureStatus::InvalidOwnership { .. }
        }
    ));
}

// =============================================================================
// WithdrawFromVault tests
// =============================================================================

#[tokio::test]
async fn test_withdraw_from_vault_full_balance() {
    let (authority_state, settlement_id, buyer, buyer_key) = create_settlement_helper().await;

    // Find the seller vault from the settlement's created objects
    // We need to find the vault created during AcceptBid
    // Since create_settlement_helper doesn't return the vault directly,
    // we need to search for it. The seller is stored in the settlement.
    let settlement_obj = authority_state.get_object(&settlement_id).await.unwrap();
    let settlement: Settlement = settlement_obj
        .deserialize_contents(ObjectType::Settlement)
        .unwrap();
    let seller = settlement.seller;

    // We need the seller's key to withdraw. Since create_settlement_helper
    // doesn't return it, we'll create a fresh test with everything we need.
    // For simplicity, test WithdrawFromVault by directly inserting a vault.
    let (seller_addr, seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller_key = SomaKeyPair::Ed25519(seller_key_ed);

    let authority_state = TestAuthorityBuilder::new().build().await;

    let vault_id = ObjectID::random();
    let vault = SellerVault {
        id: vault_id,
        owner: seller_addr,
        balance: 1_000_000,
    };
    let vault_obj = Object::new_marketplace_object(
        vault_id,
        ObjectType::SellerVault,
        &vault,
        Owner::AddressOwner(seller_addr),
        TransactionDigest::genesis_marker(),
    );
    let vault_ref = vault_obj.compute_object_reference();
    authority_state.insert_genesis_object(vault_obj).await;

    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller_addr, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::WithdrawFromVault {
        vault: vault_ref,
        amount: None, // withdraw all
        recipient_coin: None,
    };
    let data = TransactionData::new(kind, seller_addr, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert_eq!(*effects_data.status(), ExecutionStatus::Success);

    // Vault should be deleted (balance drained)
    let deleted_ids: Vec<ObjectID> = effects_data.deleted().iter().map(|d| d.0).collect();
    assert!(deleted_ids.contains(&vault_id), "Vault should be deleted when fully drained");

    // A new USDC coin should be created for the seller
    assert_eq!(effects_data.created().len(), 1);
    let created_id = effects_data.created()[0].0 .0;
    let coin_obj = authority_state.get_object(&created_id).await.unwrap();
    assert_eq!(coin_obj.as_coin().unwrap(), 1_000_000);
    assert_eq!(coin_obj.coin_type(), Some(CoinType::Usdc));
}

#[tokio::test]
async fn test_withdraw_from_vault_partial() {
    let (seller_addr, seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller_key = SomaKeyPair::Ed25519(seller_key_ed);

    let authority_state = TestAuthorityBuilder::new().build().await;

    let vault_id = ObjectID::random();
    let vault = SellerVault {
        id: vault_id,
        owner: seller_addr,
        balance: 1_000_000,
    };
    let vault_obj = Object::new_marketplace_object(
        vault_id,
        ObjectType::SellerVault,
        &vault,
        Owner::AddressOwner(seller_addr),
        TransactionDigest::genesis_marker(),
    );
    let vault_ref = vault_obj.compute_object_reference();
    authority_state.insert_genesis_object(vault_obj).await;

    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller_addr, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::WithdrawFromVault {
        vault: vault_ref,
        amount: Some(300_000),
        recipient_coin: None,
    };
    let data = TransactionData::new(kind, seller_addr, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert_eq!(*effects_data.status(), ExecutionStatus::Success);

    // Vault should still exist with reduced balance
    let vault_after = authority_state.get_object(&vault_id).await.unwrap();
    let vault_data: SellerVault = vault_after.deserialize_contents(ObjectType::SellerVault).unwrap();
    assert_eq!(vault_data.balance, 700_000);

    // New USDC coin created for seller
    assert_eq!(effects_data.created().len(), 1);
    let created_id = effects_data.created()[0].0 .0;
    let coin_obj = authority_state.get_object(&created_id).await.unwrap();
    assert_eq!(coin_obj.as_coin().unwrap(), 300_000);
}

#[tokio::test]
async fn test_withdraw_from_vault_wrong_owner_rejected() {
    let (seller_addr, _seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let (other, other_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let other_key = SomaKeyPair::Ed25519(other_key_ed);

    let authority_state = TestAuthorityBuilder::new().build().await;

    let vault_id = ObjectID::random();
    let vault = SellerVault {
        id: vault_id,
        owner: seller_addr,
        balance: 1_000_000,
    };
    let vault_obj = Object::new_marketplace_object(
        vault_id,
        ObjectType::SellerVault,
        &vault,
        Owner::AddressOwner(seller_addr),
        TransactionDigest::genesis_marker(),
    );
    let vault_ref = vault_obj.compute_object_reference();
    authority_state.insert_genesis_object(vault_obj).await;

    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), other, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::WithdrawFromVault {
        vault: vault_ref,
        amount: None,
        recipient_coin: None,
    };
    let data = TransactionData::new(kind, other, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &other_key);

    // This will likely fail at ownership check during input loading
    let result = send_and_confirm_transaction_(&authority_state, None, tx, true).await;
    match result {
        Ok((_, effects)) => {
            let effects_data = effects.into_data();
            assert!(!effects_data.status().is_ok(), "Should fail: wrong owner");
        }
        Err(_) => {
            // Expected: ownership check fails before execution
        }
    }
}

#[tokio::test]
async fn test_withdraw_from_vault_insufficient_balance_rejected() {
    let (seller_addr, seller_key_ed): (_, Ed25519KeyPair) = get_key_pair();
    let seller_key = SomaKeyPair::Ed25519(seller_key_ed);

    let authority_state = TestAuthorityBuilder::new().build().await;

    let vault_id = ObjectID::random();
    let vault = SellerVault {
        id: vault_id,
        owner: seller_addr,
        balance: 1_000,
    };
    let vault_obj = Object::new_marketplace_object(
        vault_id,
        ObjectType::SellerVault,
        &vault,
        Owner::AddressOwner(seller_addr),
        TransactionDigest::genesis_marker(),
    );
    let vault_ref = vault_obj.compute_object_reference();
    authority_state.insert_genesis_object(vault_obj).await;

    let gas = Object::with_id_owner_coin_for_testing(ObjectID::random(), seller_addr, 100_000_000);
    let gas_ref = gas.compute_object_reference();
    authority_state.insert_genesis_object(gas).await;

    let kind = TransactionKind::WithdrawFromVault {
        vault: vault_ref,
        amount: Some(999_999), // way more than balance
        recipient_coin: None,
    };
    let data = TransactionData::new(kind, seller_addr, vec![gas_ref]);
    let tx = to_sender_signed_transaction(data, &seller_key);
    let (_, effects) = send_and_confirm_transaction_(&authority_state, None, tx, true)
        .await
        .unwrap();
    let effects_data = effects.into_data();
    assert!(matches!(
        effects_data.status(),
        ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientVaultBalance }
    ));
}
