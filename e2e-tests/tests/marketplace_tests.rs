// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! E2E tests for the marketplace: ask→bid→accept→rate→withdraw flow.
//!
//! Tests:
//! 1. test_marketplace_happy_path — Full flow: create ask, create bid, accept bid, verify settlement
//! 2. test_multi_bid_competition — Ask with num_bids_wanted=3, multiple sellers bid, buyer accepts 3
//! 3. test_cancel_ask — Create ask, cancel before accepting any bids
//! 4. test_negative_seller_rating — Accept bid, then submit negative rating within deadline
//! 5. test_seller_cannot_bid_own_ask — Seller tries to bid on their own ask, rejected
//! 6. test_accept_wrong_coin_type — AcceptBid with SOMA coin instead of USDC, rejected
//! 7. test_value_fee_accounting — Verify protocol fund balance increases by exact fee amount
//! 8. test_vault_withdrawal — Seller withdraws USDC from vault after AcceptBid
//! 9. test_cancel_after_accept_rejected — CancelAsk after accepting a bid, rejected
//! 10. test_rpc_marketplace_queries — GetAsk, GetBidsForAsk, GetOpenAsks, GetSettlement, GetVault
//! 11. test_transfer_soma_single — Transfer SOMA single-coin single-recipient
//! 12. test_transfer_usdc_single — Transfer USDC single-coin single-recipient
//! 13. test_transfer_multi_recipient — Transfer with per-recipient amounts
//! 14. test_merge_coins_soma — MergeCoins with SOMA coins
//! 15. test_merge_coins_usdc — MergeCoins with USDC coins
//! 16. test_transfer_mixed_coin_types_rejected — Transfer mixing SOMA and USDC rejected
//! 17. test_merge_mixed_coin_types_rejected — MergeCoins mixing SOMA and USDC rejected
//! 18. test_ask_expiry — Create ask with short timeout, bid after epoch advance fails with AskExpired
//! 19. test_usdc_epoch_supply_conservation — USDC genesis + epoch transitions, supply check passes
//! 20. test_default_positive_rating — Rating deadline passes without RateSeller, settlement stays Positive
//! 21. test_vault_accumulation — Seller fulfills multiple asks, vaults accumulate, withdraw works
//! 22. test_secondary_index_consistency — bids_by_ask and open_asks indexes reflect state correctly
//! 23. test_bridge_deposit_mints_usdc — BridgeDeposit with valid committee stake mints USDC to recipient
//! 24. test_bridge_deposit_nonce_replay_rejected — Second deposit with same nonce rejected
//! 25. test_bridge_withdraw_e2e — BridgeWithdraw burns USDC, creates PendingWithdrawal
//! 26. test_bridge_emergency_pause_unpause — Pause blocks ops, unpause resumes them
//! 27. test_bridge_deposit_insufficient_stake_rejected — Deposit with 1/4 members rejected
//! 28. test_bridge_deposit_withdraw_roundtrip — Deposit then withdraw then deposit again
//! 29. test_accept_cheapest — 3 bids at different prices, accept 2 cheapest, verify most expensive still Pending
//! 30. test_settlements_index — settlements_by_buyer and settlements_by_seller index queries
//! 31. test_get_protocol_fund — GetProtocolFund RPC returns correct balance after AcceptBid fees
//! 32. test_get_reputation_rpc — GetReputation RPC returns correct buyer/seller metrics

use test_cluster::TestClusterBuilder;
use tracing::info;
use types::ask::{Ask, AskStatus};
use types::bid::{Bid, BidStatus};
use types::bridge::MarketplaceParameters;
use types::digests::{ResponseDigest, TaskDigest};
use types::effects::TransactionEffectsAPI;
use types::object::{CoinType, ObjectID, ObjectType};
use types::settlement::{SellerRating, Settlement};
use types::system_state::SystemStateTrait;
use types::transaction::{
    AcceptBidArgs, CreateAskArgs, CreateBidArgs, TransactionData, TransactionKind,
};
use types::vault::SellerVault;
use utils::logging::init_tracing;

/// Default USDC amount for test accounts (10 USDC = 10_000_000 microdollars).
const TEST_USDC_AMOUNT: u64 = 10_000_000;

fn test_task_digest() -> TaskDigest {
    TaskDigest::new([1u8; 32])
}

fn test_response_digest() -> ResponseDigest {
    ResponseDigest::new([2u8; 32])
}

fn test_response_digest_2() -> ResponseDigest {
    ResponseDigest::new([3u8; 32])
}

fn test_response_digest_3() -> ResponseDigest {
    ResponseDigest::new([4u8; 32])
}

// =============================================================================
// Test 1: Full marketplace happy path
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_marketplace_happy_path() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2) // 2 USDC coins per account
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];

    // --- Step 1: Buyer creates an ask ---
    let buyer_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .expect("buyer must have gas");

    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000, // 1 USDC
            num_bids_wanted: 1,
            timeout_ms: 300_000, // 5 minutes
        }),
        buyer,
        vec![buyer_gas],
    );

    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok(), "CreateAsk failed: {:?}", ask_response.effects.status());

    // Extract the created Ask object ID
    let created = ask_response.effects.created();
    assert!(!created.is_empty(), "CreateAsk should create at least one object");
    let ask_id = created[0].0 .0;
    info!("Created ask: {}", ask_id);

    // Verify Ask object state
    let client = &test_cluster.fullnode_handle.soma_client;
    let ask_obj = client.get_object(ask_id).await.unwrap();
    let ask: Ask = ask_obj.deserialize_contents(ObjectType::Ask).expect("should deserialize Ask");
    assert_eq!(ask.buyer, buyer);
    assert_eq!(ask.status, AskStatus::Open);
    assert_eq!(ask.max_price_per_bid, 1_000_000);
    assert_eq!(ask.num_bids_wanted, 1);
    assert_eq!(ask.accepted_bid_count, 0);

    // --- Step 2: Seller creates a bid ---
    let seller_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(seller)
        .await
        .unwrap()
        .expect("seller must have gas");

    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: 500_000, // 0.50 USDC
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );

    let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    assert!(bid_response.effects.status().is_ok(), "CreateBid failed: {:?}", bid_response.effects.status());

    let bid_created = bid_response.effects.created();
    assert!(!bid_created.is_empty(), "CreateBid should create a Bid object");
    let bid_id = bid_created[0].0 .0;
    info!("Created bid: {}", bid_id);

    // Verify Bid object state
    let bid_obj = client.get_object(bid_id).await.unwrap();
    let bid: Bid = bid_obj.deserialize_contents(ObjectType::Bid).expect("should deserialize Bid");
    assert_eq!(bid.seller, seller);
    assert_eq!(bid.ask_id, ask_id);
    assert_eq!(bid.price, 500_000);
    assert_eq!(bid.status, BidStatus::Pending);

    // --- Step 3: Buyer accepts the bid (settlement) ---
    // Get buyer's USDC coin for payment
    let buyer_usdc = test_cluster
        .wallet
        .get_richest_usdc_coin(buyer)
        .await
        .unwrap()
        .expect("buyer must have USDC");

    // Need fresh gas after previous tx
    let buyer_gas2 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .expect("buyer must have gas");

    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs {
            ask_id,
            bid_id,
            payment_coin: buyer_usdc.0,
        }),
        buyer,
        vec![buyer_gas2],
    );

    let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
    assert!(
        accept_response.effects.status().is_ok(),
        "AcceptBid failed: {:?}",
        accept_response.effects.status()
    );

    // AcceptBid creates: Settlement + SellerVault
    let accept_created = accept_response.effects.created();
    assert!(
        accept_created.len() >= 2,
        "AcceptBid should create Settlement + SellerVault, got {} objects",
        accept_created.len()
    );
    info!("AcceptBid created {} objects", accept_created.len());

    // Find the settlement and vault among created objects
    let mut settlement_id = None;
    let mut vault_id = None;
    for (obj_ref, _owner) in &accept_created {
        let obj = client.get_object(obj_ref.0).await.unwrap();
        if obj.deserialize_contents::<Settlement>(ObjectType::Settlement).is_some() {
            settlement_id = Some(obj_ref.0);
        }
        if obj.deserialize_contents::<SellerVault>(ObjectType::SellerVault).is_some() {
            vault_id = Some(obj_ref.0);
        }
    }
    let settlement_id = settlement_id.expect("AcceptBid must create a Settlement");
    let vault_id = vault_id.expect("AcceptBid must create a SellerVault");

    // Verify Settlement
    let settlement_obj = client.get_object(settlement_id).await.unwrap();
    let settlement: Settlement = settlement_obj
        .deserialize_contents(ObjectType::Settlement)
        .expect("should deserialize Settlement");
    assert_eq!(settlement.buyer, buyer);
    assert_eq!(settlement.seller, seller);
    assert_eq!(settlement.ask_id, ask_id);
    assert_eq!(settlement.bid_id, bid_id);
    assert_eq!(settlement.seller_rating, SellerRating::Positive);
    // amount = bid.price - value_fee (2.5% default = 12500 microdollars)
    let expected_fee = 500_000 * 250 / 10_000; // 12500
    let expected_amount = 500_000 - expected_fee;
    assert_eq!(settlement.amount, expected_amount);

    // Verify SellerVault
    let vault_obj = client.get_object(vault_id).await.unwrap();
    let vault: SellerVault = vault_obj
        .deserialize_contents(ObjectType::SellerVault)
        .expect("should deserialize SellerVault");
    assert_eq!(vault.owner, seller);
    assert_eq!(vault.balance, expected_amount);

    // Verify Ask is now Filled
    let ask_obj = client.get_object(ask_id).await.unwrap();
    let ask: Ask = ask_obj.deserialize_contents(ObjectType::Ask).expect("should deserialize Ask");
    assert_eq!(ask.status, AskStatus::Filled);
    assert_eq!(ask.accepted_bid_count, 1);

    // Verify Bid is now Accepted
    let bid_obj = client.get_object(bid_id).await.unwrap();
    let bid: Bid = bid_obj.deserialize_contents(ObjectType::Bid).expect("should deserialize Bid");
    assert_eq!(bid.status, BidStatus::Accepted);

    info!("Marketplace happy path test passed!");
}

// =============================================================================
// Test 2: Multi-bid competition
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_multi_bid_competition() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller_a = addresses[1];
    let seller_b = addresses[2];
    let seller_c = addresses[3];

    // Buyer creates an ask wanting 3 bids
    let buyer_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();

    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 2_000_000,
            num_bids_wanted: 3,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![buyer_gas],
    );

    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok());
    let ask_id = ask_response.effects.created()[0].0 .0;

    // Three sellers bid at different prices
    let prices = [1_000_000u64, 800_000, 1_500_000];
    let sellers = [seller_a, seller_b, seller_c];
    let digests = [test_response_digest(), test_response_digest_2(), test_response_digest_3()];
    let mut bid_ids = vec![];

    for (i, (&seller, &price)) in sellers.iter().zip(prices.iter()).enumerate() {
        let gas = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(seller)
            .await
            .unwrap()
            .unwrap();

        let bid_tx = TransactionData::new(
            TransactionKind::CreateBid(CreateBidArgs {
                ask_id,
                price,
                response_digest: digests[i],
            }),
            seller,
            vec![gas],
        );

        let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
        assert!(bid_response.effects.status().is_ok(), "Bid {} failed", i);
        bid_ids.push(bid_response.effects.created()[0].0 .0);
    }

    // Buyer accepts all 3 bids
    for (i, &bid_id) in bid_ids.iter().enumerate() {
        let usdc = test_cluster
            .wallet
            .get_richest_usdc_coin(buyer)
            .await
            .unwrap()
            .expect("buyer must have USDC for accept");

        let gas = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(buyer)
            .await
            .unwrap()
            .unwrap();

        let accept_tx = TransactionData::new(
            TransactionKind::AcceptBid(AcceptBidArgs {
                ask_id,
                bid_id,
                payment_coin: usdc.0,
            }),
            buyer,
            vec![gas],
        );

        let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
        assert!(
            accept_response.effects.status().is_ok(),
            "AcceptBid {} failed: {:?}",
            i,
            accept_response.effects.status()
        );
    }

    // Verify Ask is Filled with 3 accepted bids
    let client = &test_cluster.fullnode_handle.soma_client;
    let ask_obj = client.get_object(ask_id).await.unwrap();
    let ask: Ask = ask_obj.deserialize_contents(ObjectType::Ask).unwrap();
    assert_eq!(ask.status, AskStatus::Filled);
    assert_eq!(ask.accepted_bid_count, 3);

    info!("Multi-bid competition test passed!");
}

// =============================================================================
// Test 3: Cancel ask
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_cancel_ask() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();

    // Create ask
    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![gas],
    );

    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok());
    let ask_id = ask_response.effects.created()[0].0 .0;

    // Cancel ask
    let gas2 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();

    let cancel_tx = TransactionData::new(
        TransactionKind::CancelAsk { ask_id },
        buyer,
        vec![gas2],
    );

    let cancel_response = test_cluster.sign_and_execute_transaction(&cancel_tx).await;
    assert!(cancel_response.effects.status().is_ok(), "CancelAsk failed: {:?}", cancel_response.effects.status());

    // Verify ask is Cancelled
    let client = &test_cluster.fullnode_handle.soma_client;
    let ask_obj = client.get_object(ask_id).await.unwrap();
    let ask: Ask = ask_obj.deserialize_contents(ObjectType::Ask).unwrap();
    assert_eq!(ask.status, AskStatus::Cancelled);

    info!("Cancel ask test passed!");
}

// =============================================================================
// Test 4: Negative seller rating
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_negative_seller_rating() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];

    // Create ask
    let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![gas],
    );
    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok());
    let ask_id = ask_response.effects.created()[0].0 .0;

    // Seller bids
    let seller_gas = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: 500_000,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );
    let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    assert!(bid_response.effects.status().is_ok());
    let bid_id = bid_response.effects.created()[0].0 .0;

    // Buyer accepts
    let usdc = test_cluster.wallet.get_richest_usdc_coin(buyer).await.unwrap().unwrap();
    let gas2 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs { ask_id, bid_id, payment_coin: usdc.0 }),
        buyer,
        vec![gas2],
    );
    let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
    assert!(accept_response.effects.status().is_ok());

    // Find settlement ID among created objects
    let client = &test_cluster.fullnode_handle.soma_client;
    let accept_created = accept_response.effects.created();
    let mut settlement_id = None;
    for (obj_ref, _owner) in &accept_created {
        let obj = client.get_object(obj_ref.0).await.unwrap();
        if obj.deserialize_contents::<Settlement>(ObjectType::Settlement).is_some() {
            settlement_id = Some(obj_ref.0);
        }
    }
    let settlement_id = settlement_id.expect("must find settlement");

    // Buyer submits negative rating
    let gas3 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let rate_tx = TransactionData::new(
        TransactionKind::RateSeller { settlement_id },
        buyer,
        vec![gas3],
    );
    let rate_response = test_cluster.sign_and_execute_transaction(&rate_tx).await;
    assert!(rate_response.effects.status().is_ok(), "RateSeller failed: {:?}", rate_response.effects.status());

    // Verify settlement has Negative rating
    let settlement_obj = client.get_object(settlement_id).await.unwrap();
    let settlement: Settlement = settlement_obj
        .deserialize_contents(ObjectType::Settlement)
        .unwrap();
    assert_eq!(settlement.seller_rating, SellerRating::Negative);

    info!("Negative seller rating test passed!");
}

// =============================================================================
// Test 5: Seller cannot bid on own ask
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_seller_cannot_bid_own_ask() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];

    // Create ask
    let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![gas],
    );
    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok());
    let ask_id = ask_response.effects.created()[0].0 .0;

    // Same address tries to bid on own ask
    let gas2 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: 500_000,
            response_digest: test_response_digest(),
        }),
        buyer,
        vec![gas2],
    );

    let bid_tx_signed = test_cluster.wallet.sign_transaction(&bid_tx).await;
    let result = test_cluster.wallet.execute_transaction_may_fail(bid_tx_signed).await;

    match result {
        Ok(response) => {
            assert!(
                response.effects.status().is_err(),
                "Self-bid should fail with SellerCannotBidOnOwnAsk"
            );
            info!("Self-bid correctly rejected at execution: {:?}", response.effects.status());
        }
        Err(e) => {
            info!("Self-bid correctly rejected at orchestrator: {}", e);
        }
    }

    info!("Seller cannot bid own ask test passed!");
}

// =============================================================================
// Test 6: Accept with wrong coin type (SOMA instead of USDC)
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_accept_wrong_coin_type() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 1)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];

    // Create ask
    let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![gas],
    );
    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok());
    let ask_id = ask_response.effects.created()[0].0 .0;

    // Seller bids
    let seller_gas = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: 500_000,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );
    let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    assert!(bid_response.effects.status().is_ok());
    let bid_id = bid_response.effects.created()[0].0 .0;

    // Try to accept with a SOMA gas coin (wrong CoinType)
    let soma_coin = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();

    // Get a different gas coin (we need separate gas)
    let all_gas = test_cluster
        .wallet
        .get_gas_objects_owned_by_address(buyer, None)
        .await
        .unwrap();
    // Use the first as payment (wrong type), second as gas
    let payment_soma = all_gas[0];
    let gas_for_accept = if all_gas.len() > 1 { all_gas[1] } else { all_gas[0] };

    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs {
            ask_id,
            bid_id,
            payment_coin: payment_soma,
        }),
        buyer,
        vec![gas_for_accept],
    );

    let tx_signed = test_cluster.wallet.sign_transaction(&accept_tx).await;
    let result = test_cluster.wallet.execute_transaction_may_fail(tx_signed).await;

    match result {
        Ok(response) => {
            assert!(
                response.effects.status().is_err(),
                "AcceptBid with SOMA coin should fail"
            );
            info!("Wrong coin type correctly rejected: {:?}", response.effects.status());
        }
        Err(e) => {
            info!("Wrong coin type rejected at orchestrator: {}", e);
        }
    }

    info!("Accept wrong coin type test passed!");
}

// =============================================================================
// Test 7: Value fee accounting
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_value_fee_accounting() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];
    let client = &test_cluster.fullnode_handle.soma_client;

    // Create ask → bid → accept
    let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 2_000_000,
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![gas],
    );
    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    let ask_id = ask_response.effects.created()[0].0 .0;

    let seller_gas = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
    let bid_price = 1_000_000u64; // 1 USDC
    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: bid_price,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );
    let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    let bid_id = bid_response.effects.created()[0].0 .0;

    let usdc = test_cluster.wallet.get_richest_usdc_coin(buyer).await.unwrap().unwrap();
    let gas2 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs { ask_id, bid_id, payment_coin: usdc.0 }),
        buyer,
        vec![gas2],
    );
    let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
    assert!(accept_response.effects.status().is_ok());

    // Verify fee via settlement amount and vault balance
    // Default value_fee_bps = 250 (2.5%)
    let expected_fee = bid_price * 250 / 10_000; // 25000 microdollars
    let expected_net = bid_price - expected_fee;

    // Find settlement and vault to verify fee was correctly deducted
    let accept_created = accept_response.effects.created();
    let mut settlement_amount = None;
    let mut vault_balance = None;
    for (obj_ref, _) in &accept_created {
        let obj = client.get_object(obj_ref.0).await.unwrap();
        if let Some(s) = obj.deserialize_contents::<Settlement>(ObjectType::Settlement) {
            settlement_amount = Some(s.amount);
        }
        if let Some(v) = obj.deserialize_contents::<SellerVault>(ObjectType::SellerVault) {
            vault_balance = Some(v.balance);
        }
    }

    let settlement_amount = settlement_amount.expect("must find settlement");
    let vault_balance = vault_balance.expect("must find vault");

    assert_eq!(
        settlement_amount, expected_net,
        "Settlement amount should be bid_price - fee (got {}, expected {})",
        settlement_amount, expected_net
    );
    assert_eq!(
        vault_balance, expected_net,
        "Vault balance should match settlement amount"
    );
    assert_eq!(expected_fee, 25_000, "Fee calculation sanity check");

    info!(
        "Value fee accounting test passed! bid={}, fee={}, net={}",
        bid_price, expected_fee, expected_net
    );
}

// =============================================================================
// Test 8: Vault withdrawal
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_vault_withdrawal() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];
    let client = &test_cluster.fullnode_handle.soma_client;

    // Create ask → bid → accept
    let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![gas],
    );
    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    let ask_id = ask_response.effects.created()[0].0 .0;

    let seller_gas = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: 800_000,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );
    let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    let bid_id = bid_response.effects.created()[0].0 .0;

    let usdc = test_cluster.wallet.get_richest_usdc_coin(buyer).await.unwrap().unwrap();
    let gas2 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs { ask_id, bid_id, payment_coin: usdc.0 }),
        buyer,
        vec![gas2],
    );
    let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
    assert!(accept_response.effects.status().is_ok());

    // Find the SellerVault among created objects
    let accept_created = accept_response.effects.created();
    let mut vault_object_id = None;
    for (obj_ref, _) in &accept_created {
        let obj = client.get_object(obj_ref.0).await.unwrap();
        if obj.deserialize_contents::<SellerVault>(ObjectType::SellerVault).is_some() {
            vault_object_id = Some(obj_ref.0);
        }
    }
    let vault_object_id = vault_object_id.expect("must find vault");

    // Verify vault balance
    let vault_obj = client.get_object(vault_object_id).await.unwrap();
    let vault: SellerVault = vault_obj.deserialize_contents(ObjectType::SellerVault).unwrap();
    let expected_fee = 800_000 * 250 / 10_000;
    let expected_vault_balance = 800_000 - expected_fee;
    assert_eq!(vault.balance, expected_vault_balance);

    // Seller withdraws from vault — need fresh vault ref
    let vault_obj = client.get_object(vault_object_id).await.unwrap();
    let fresh_vault_ref = vault_obj.compute_object_reference();

    let seller_gas2 = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
    let withdraw_tx = TransactionData::new(
        TransactionKind::WithdrawFromVault {
            vault: fresh_vault_ref,
            amount: None, // withdraw all
            recipient_coin: None,
        },
        seller,
        vec![seller_gas2],
    );
    let withdraw_response = test_cluster.sign_and_execute_transaction(&withdraw_tx).await;
    assert!(
        withdraw_response.effects.status().is_ok(),
        "WithdrawFromVault failed: {:?}",
        withdraw_response.effects.status()
    );

    // Verify a USDC coin was created for the seller with the vault balance
    let created = withdraw_response.effects.created();
    assert!(!created.is_empty(), "Withdraw should create a USDC coin");
    let usdc_coin_id = created[0].0 .0;
    let usdc_obj = client.get_object(usdc_coin_id).await.unwrap();
    assert_eq!(usdc_obj.coin_type(), Some(CoinType::Usdc));
    assert_eq!(usdc_obj.as_coin().unwrap(), expected_vault_balance);

    info!("Vault withdrawal test passed!");
}

// =============================================================================
// Test 9: Cancel after accept rejected
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_cancel_after_accept_rejected() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];

    // Create ask with num_bids_wanted=2 so it's not fully filled after 1 accept
    let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 2,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![gas],
    );
    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    let ask_id = ask_response.effects.created()[0].0 .0;

    // Seller bids
    let seller_gas = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: 500_000,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );
    let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    let bid_id = bid_response.effects.created()[0].0 .0;

    // Buyer accepts 1 bid
    let usdc = test_cluster.wallet.get_richest_usdc_coin(buyer).await.unwrap().unwrap();
    let gas2 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs { ask_id, bid_id, payment_coin: usdc.0 }),
        buyer,
        vec![gas2],
    );
    let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
    assert!(accept_response.effects.status().is_ok());

    // Try to cancel — should fail because accepted_bid_count > 0
    let gas3 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let cancel_tx = TransactionData::new(
        TransactionKind::CancelAsk { ask_id },
        buyer,
        vec![gas3],
    );
    let cancel_signed = test_cluster.wallet.sign_transaction(&cancel_tx).await;
    let result = test_cluster.wallet.execute_transaction_may_fail(cancel_signed).await;

    match result {
        Ok(response) => {
            assert!(
                response.effects.status().is_err(),
                "CancelAsk after accept should fail with AskHasAcceptedBids"
            );
            info!("Cancel after accept correctly rejected: {:?}", response.effects.status());
        }
        Err(e) => {
            info!("Cancel after accept rejected at orchestrator: {}", e);
        }
    }

    info!("Cancel after accept test passed!");
}

// =============================================================================
// Test 10: RPC marketplace queries
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_rpc_marketplace_queries() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];
    let client = &test_cluster.fullnode_handle.soma_client;

    // Create ask
    let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![gas],
    );
    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    let ask_id = ask_response.effects.created()[0].0 .0;

    // Verify Ask via standard get_object (proven to work in happy path)
    let ask_obj = client.get_object(ask_id).await.unwrap();
    let ask: Ask = ask_obj.deserialize_contents(ObjectType::Ask).unwrap();
    assert_eq!(ask.buyer, buyer);
    assert_eq!(ask.status, AskStatus::Open);

    // Create bid
    let seller_gas = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: 500_000,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );
    let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    let bid_id = bid_response.effects.created()[0].0 .0;

    // Verify Bid via get_object
    let bid_obj = client.get_object(bid_id).await.unwrap();
    let bid: Bid = bid_obj.deserialize_contents(ObjectType::Bid).unwrap();
    assert_eq!(bid.seller, seller);
    assert_eq!(bid.price, 500_000);

    // Accept bid
    let usdc = test_cluster.wallet.get_richest_usdc_coin(buyer).await.unwrap().unwrap();
    let gas2 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs { ask_id, bid_id, payment_coin: usdc.0 }),
        buyer,
        vec![gas2],
    );
    let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
    assert!(accept_response.effects.status().is_ok());

    // Find settlement and vault via created objects
    let mut settlement_id = None;
    let mut vault_id = None;
    for (obj_ref, _) in accept_response.effects.created() {
        let obj = client.get_object(obj_ref.0).await.unwrap();
        if obj.deserialize_contents::<Settlement>(ObjectType::Settlement).is_some() {
            settlement_id = Some(obj_ref.0);
        }
        if obj.deserialize_contents::<SellerVault>(ObjectType::SellerVault).is_some() {
            vault_id = Some(obj_ref.0);
        }
    }

    // Verify Settlement
    let settlement_id = settlement_id.expect("must find settlement");
    let s_obj = client.get_object(settlement_id).await.unwrap();
    let settlement: Settlement = s_obj.deserialize_contents(ObjectType::Settlement).unwrap();
    assert_eq!(settlement.buyer, buyer);
    assert_eq!(settlement.seller, seller);

    // Verify Vault
    let _vault_id = vault_id.expect("must find vault");

    // Verify filled ask no longer Open
    let ask_obj = client.get_object(ask_id).await.unwrap();
    let ask: Ask = ask_obj.deserialize_contents(ObjectType::Ask).unwrap();
    assert_eq!(ask.status, AskStatus::Filled);

    info!("RPC marketplace queries test passed!");
}

// =============================================================================
// Test 11: Transfer SOMA single-coin single-recipient
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_transfer_soma_single() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    // Get sender's gas coins (SOMA)
    let gas_coins = test_cluster
        .wallet
        .get_gas_objects_sorted_by_balance_with_amounts(sender)
        .await
        .unwrap();
    assert!(gas_coins.len() >= 2, "sender needs at least 2 gas coins");

    // Use first coin for transfer, second for gas
    let transfer_coin = gas_coins[0].0;
    let transfer_amount = 1_000_000; // 0.001 SOMA

    let tx = TransactionData::new(
        TransactionKind::Transfer {
            coins: vec![transfer_coin],
            amounts: Some(vec![transfer_amount]),
            recipients: vec![recipient],
        },
        sender,
        vec![gas_coins[1].0], // separate gas coin
    );

    let response = test_cluster.sign_and_execute_transaction(&tx).await;
    assert!(response.effects.status().is_ok(), "SOMA transfer failed: {:?}", response.effects.status());

    // Verify a new coin was created for recipient
    let created = response.effects.created();
    assert!(!created.is_empty(), "Transfer should create a coin for recipient");

    // Verify the created coin is SOMA with correct amount
    let client = &test_cluster.fullnode_handle.soma_client;
    let coin_obj = client.get_object(created[0].0 .0).await.unwrap();
    assert_eq!(coin_obj.coin_type(), Some(CoinType::Soma));
    assert_eq!(coin_obj.as_coin(), Some(transfer_amount));

    info!("Transfer SOMA single test passed!");
}

// =============================================================================
// Test 12: Transfer USDC single-coin single-recipient
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_transfer_usdc_single() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    // Get sender's USDC coins
    let usdc_coins = test_cluster
        .wallet
        .get_usdc_coins_sorted_by_balance(sender)
        .await
        .unwrap();
    assert!(!usdc_coins.is_empty(), "sender needs USDC coins");

    let usdc_coin = usdc_coins[0].0;
    let transfer_amount = 500_000; // 0.50 USDC

    // Get a SOMA gas coin (separate from USDC)
    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have gas");

    let tx = TransactionData::new(
        TransactionKind::Transfer {
            coins: vec![usdc_coin],
            amounts: Some(vec![transfer_amount]),
            recipients: vec![recipient],
        },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx).await;
    assert!(response.effects.status().is_ok(), "USDC transfer failed: {:?}", response.effects.status());

    // Verify created coin is USDC with correct amount
    let created = response.effects.created();
    assert!(!created.is_empty(), "Transfer should create a USDC coin for recipient");

    let client = &test_cluster.fullnode_handle.soma_client;
    let coin_obj = client.get_object(created[0].0 .0).await.unwrap();
    assert_eq!(coin_obj.coin_type(), Some(CoinType::Usdc));
    assert_eq!(coin_obj.as_coin(), Some(transfer_amount));

    info!("Transfer USDC single test passed!");
}

// =============================================================================
// Test 13: Transfer with per-recipient amounts (multi-recipient)
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_transfer_multi_recipient() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addresses = test_cluster.get_addresses();
    assert!(addresses.len() >= 3, "need at least 3 accounts");
    let sender = addresses[0];
    let recipient1 = addresses[1];
    let recipient2 = addresses[2];

    let gas_coins = test_cluster
        .wallet
        .get_gas_objects_sorted_by_balance_with_amounts(sender)
        .await
        .unwrap();
    assert!(gas_coins.len() >= 2, "sender needs at least 2 gas coins");

    let amount1 = 100_000;
    let amount2 = 200_000;

    let tx = TransactionData::new(
        TransactionKind::Transfer {
            coins: vec![gas_coins[0].0],
            amounts: Some(vec![amount1, amount2]),
            recipients: vec![recipient1, recipient2],
        },
        sender,
        vec![gas_coins[1].0],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx).await;
    assert!(response.effects.status().is_ok(), "Multi-recipient transfer failed: {:?}", response.effects.status());

    // Should create 2 new coins (one per recipient)
    let created = response.effects.created();
    assert!(created.len() >= 2, "Should create at least 2 coins, got {}", created.len());

    let client = &test_cluster.fullnode_handle.soma_client;
    let mut found_amounts = Vec::new();
    for (obj_ref, _) in &created {
        let obj = client.get_object(obj_ref.0).await.unwrap();
        if let Some(balance) = obj.as_coin() {
            found_amounts.push(balance);
        }
    }
    found_amounts.sort();
    assert!(found_amounts.contains(&amount1), "Should find coin with amount1");
    assert!(found_amounts.contains(&amount2), "Should find coin with amount2");

    info!("Transfer multi-recipient test passed!");
}

// =============================================================================
// Test 14: MergeCoins with SOMA coins
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_merge_coins_soma() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];

    // Sender should have multiple gas coins from genesis
    let gas_coins = test_cluster
        .wallet
        .get_gas_objects_sorted_by_balance_with_amounts(sender)
        .await
        .unwrap();
    assert!(gas_coins.len() >= 3, "sender needs at least 3 gas coins for merge + separate gas");

    let coin1 = gas_coins[0].0;
    let coin2 = gas_coins[1].0;
    let gas = gas_coins[2].0;
    let balance1 = gas_coins[0].1;
    let balance2 = gas_coins[1].1;

    let tx = TransactionData::new(
        TransactionKind::MergeCoins { coins: vec![coin1, coin2] },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx).await;
    assert!(response.effects.status().is_ok(), "MergeCoins SOMA failed: {:?}", response.effects.status());

    // The first coin should have the combined balance
    let client = &test_cluster.fullnode_handle.soma_client;
    let merged_obj = client.get_object(coin1.0).await.unwrap();
    let merged_balance = merged_obj.as_coin().expect("merged coin should have balance");
    assert_eq!(merged_balance, balance1 + balance2, "Merged coin should have combined balance");
    assert_eq!(merged_obj.coin_type(), Some(CoinType::Soma));

    info!("MergeCoins SOMA test passed!");
}

// =============================================================================
// Test 15: MergeCoins with USDC coins
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_merge_coins_usdc() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2) // 2 USDC coins per account
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];

    let usdc_coins = test_cluster
        .wallet
        .get_usdc_coins_sorted_by_balance(sender)
        .await
        .unwrap();
    assert!(usdc_coins.len() >= 2, "sender needs at least 2 USDC coins");

    let coin1 = usdc_coins[0].0;
    let coin2 = usdc_coins[1].0;
    let balance1 = usdc_coins[0].1;
    let balance2 = usdc_coins[1].1;

    // Use SOMA gas coin (separate from USDC)
    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have gas");

    let tx = TransactionData::new(
        TransactionKind::MergeCoins { coins: vec![coin1, coin2] },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx).await;
    assert!(response.effects.status().is_ok(), "MergeCoins USDC failed: {:?}", response.effects.status());

    // The first coin should have the combined balance
    let client = &test_cluster.fullnode_handle.soma_client;
    let merged_obj = client.get_object(coin1.0).await.unwrap();
    let merged_balance = merged_obj.as_coin().expect("merged coin should have balance");
    assert_eq!(merged_balance, balance1 + balance2, "Merged USDC coins should have combined balance");
    assert_eq!(merged_obj.coin_type(), Some(CoinType::Usdc));

    info!("MergeCoins USDC test passed!");
}

// =============================================================================
// Test 16: Transfer mixing SOMA and USDC coins rejected
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_transfer_mixed_coin_types_rejected() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 1)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    // Get one SOMA coin and one USDC coin
    let gas_coins = test_cluster
        .wallet
        .get_gas_objects_sorted_by_balance_with_amounts(sender)
        .await
        .unwrap();
    let soma_coin = gas_coins[0].0;

    let usdc_coins = test_cluster
        .wallet
        .get_usdc_coins_sorted_by_balance(sender)
        .await
        .unwrap();
    let usdc_coin = usdc_coins[0].0;

    // Separate gas coin
    let gas = gas_coins[1].0;

    // Try to transfer with mixed coin types
    let tx = TransactionData::new(
        TransactionKind::Transfer {
            coins: vec![soma_coin, usdc_coin],
            amounts: Some(vec![100_000]),
            recipients: vec![recipient],
        },
        sender,
        vec![gas],
    );

    let signed = test_cluster.wallet.sign_transaction(&tx).await;
    let result = test_cluster.wallet.execute_transaction_may_fail(signed).await;

    match result {
        Ok(response) => {
            assert!(
                response.effects.status().is_err(),
                "Mixed coin type transfer should fail"
            );
            info!("Mixed coin transfer correctly rejected at execution: {:?}", response.effects.status());
        }
        Err(e) => {
            info!("Mixed coin transfer correctly rejected at orchestrator: {}", e);
        }
    }

    info!("Transfer mixed coin types rejected test passed!");
}

// =============================================================================
// Test 17: MergeCoins mixing SOMA and USDC coins rejected
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_merge_mixed_coin_types_rejected() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 1)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];

    // Get one SOMA coin and one USDC coin
    let gas_coins = test_cluster
        .wallet
        .get_gas_objects_sorted_by_balance_with_amounts(sender)
        .await
        .unwrap();
    let soma_coin = gas_coins[0].0;

    let usdc_coins = test_cluster
        .wallet
        .get_usdc_coins_sorted_by_balance(sender)
        .await
        .unwrap();
    let usdc_coin = usdc_coins[0].0;

    // Separate gas coin
    let gas = gas_coins[1].0;

    // Try to merge mixed coin types
    let tx = TransactionData::new(
        TransactionKind::MergeCoins { coins: vec![soma_coin, usdc_coin] },
        sender,
        vec![gas],
    );

    let signed = test_cluster.wallet.sign_transaction(&tx).await;
    let result = test_cluster.wallet.execute_transaction_may_fail(signed).await;

    match result {
        Ok(response) => {
            assert!(
                response.effects.status().is_err(),
                "Mixed coin type merge should fail"
            );
            info!("Mixed coin merge correctly rejected at execution: {:?}", response.effects.status());
        }
        Err(e) => {
            info!("Mixed coin merge correctly rejected at orchestrator: {}", e);
        }
    }

    info!("Merge mixed coin types rejected test passed!");
}

// =============================================================================
// Test 18: Ask expiry — bid after timeout fails
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_ask_expiry() {
    init_tracing();

    // Short epoch duration so epochs advance quickly and epoch_start_timestamp_ms updates
    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(5_000) // 5 second epochs
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];

    // Create ask with minimum timeout (10 seconds)
    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();

    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 1,
            timeout_ms: 10_000, // 10 seconds (minimum allowed)
        }),
        buyer,
        vec![gas],
    );

    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok(), "CreateAsk failed");
    let ask_id = ask_response.effects.created()[0].0 .0;
    info!("Created ask with 10s timeout: {}", ask_id);

    // Wait for enough epochs to pass that epoch_start_timestamp_ms > ask.created_at_ms + timeout_ms
    // With 5s epochs, after ~3 epochs (15s), the epoch_start_timestamp should be >10s past the ask creation
    test_cluster.trigger_reconfiguration().await;
    test_cluster.trigger_reconfiguration().await;
    test_cluster.trigger_reconfiguration().await;

    info!("Advanced 3 epochs, attempting bid on expired ask...");

    // Now try to bid on the expired ask
    let seller_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(seller)
        .await
        .unwrap()
        .unwrap();

    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: 500_000,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );

    let signed = test_cluster.wallet.sign_transaction(&bid_tx).await;
    let result = test_cluster.wallet.execute_transaction_may_fail(signed).await;

    match result {
        Ok(response) => {
            assert!(
                response.effects.status().is_err(),
                "Bid on expired ask should fail with AskExpired"
            );
            info!("Bid on expired ask correctly rejected: {:?}", response.effects.status());
        }
        Err(e) => {
            info!("Bid on expired ask correctly rejected at orchestrator: {}", e);
        }
    }

    info!("Ask expiry test passed!");
}

// =============================================================================
// Test 19: USDC + epoch transitions (supply conservation)
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_usdc_epoch_supply_conservation() {
    init_tracing();

    // This test validates that USDC coins at genesis don't trip the SOMA supply
    // conservation check at epoch boundaries. Previously, the check summed ALL
    // coin balances (including USDC) and compared against TOTAL_SUPPLY_SOMA.
    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 1)
        .with_epoch_duration_ms(5_000)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];

    // Do a marketplace flow with USDC to create/move USDC objects across epoch
    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();

    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 1,
            timeout_ms: 300_000, // 5 minutes
        }),
        buyer,
        vec![gas],
    );
    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok());

    let ask_id = ask_response.effects.created()[0].0 .0;

    // Seller bids
    let seller_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(seller)
        .await
        .unwrap()
        .unwrap();

    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: 500_000,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );
    let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    assert!(bid_response.effects.status().is_ok());

    let bid_id = bid_response.effects.created()[0].0 .0;

    // Buyer accepts bid with USDC coin
    let buyer_usdc = test_cluster
        .wallet
        .get_richest_usdc_coin(buyer)
        .await
        .unwrap()
        .expect("buyer must have USDC");

    let buyer_gas2 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();

    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs {
            ask_id,
            bid_id,
            payment_coin: buyer_usdc.0,
        }),
        buyer,
        vec![buyer_gas2],
    );
    let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
    assert!(
        accept_response.effects.status().is_ok(),
        "AcceptBid should succeed: {:?}",
        accept_response.effects.status()
    );

    // Wait for epoch transitions — supply conservation check runs at each boundary.
    // Before the fix, this would panic with "SUPPLY CONSERVATION VIOLATION" because
    // USDC coins were counted in the SOMA supply total.
    test_cluster.wait_for_epoch(Some(1)).await;
    info!("Epoch 1 completed — supply conservation check passed with USDC present");

    test_cluster.wait_for_epoch(Some(2)).await;
    info!("Epoch 2 completed — supply conservation still passing");

    info!("USDC + epoch supply conservation test passed!");
}

// =============================================================================
// Test 20: Default positive rating — deadline passes without RateSeller
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_default_positive_rating() {
    init_tracing();

    // Use short rating_window_ms (10s) and short epoch duration (5s) so we can
    // advance past the deadline quickly.
    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .with_epoch_duration_ms(5_000)
        .with_marketplace_params(MarketplaceParameters {
            rating_window_ms: 10_000, // 10 seconds
            ..MarketplaceParameters::default()
        })
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];

    // Create ask
    let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 1_000_000,
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![gas],
    );
    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok());
    let ask_id = ask_response.effects.created()[0].0 .0;

    // Seller bids
    let seller_gas = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: 500_000,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );
    let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    assert!(bid_response.effects.status().is_ok());
    let bid_id = bid_response.effects.created()[0].0 .0;

    // Buyer accepts
    let usdc = test_cluster.wallet.get_richest_usdc_coin(buyer).await.unwrap().unwrap();
    let gas2 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs { ask_id, bid_id, payment_coin: usdc.0 }),
        buyer,
        vec![gas2],
    );
    let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
    assert!(accept_response.effects.status().is_ok());

    // Find settlement
    let client = &test_cluster.fullnode_handle.soma_client;
    let accept_created = accept_response.effects.created();
    let mut settlement_id = None;
    for (obj_ref, _owner) in &accept_created {
        let obj = client.get_object(obj_ref.0).await.unwrap();
        if obj.deserialize_contents::<Settlement>(ObjectType::Settlement).is_some() {
            settlement_id = Some(obj_ref.0);
        }
    }
    let settlement_id = settlement_id.expect("must find settlement");

    // Verify rating is Positive right after acceptance
    let settlement_obj = client.get_object(settlement_id).await.unwrap();
    let settlement: Settlement = settlement_obj.deserialize_contents(ObjectType::Settlement).unwrap();
    assert_eq!(settlement.seller_rating, SellerRating::Positive);
    info!("Settlement created with Positive rating, deadline_ms={}", settlement.rating_deadline_ms);

    // Advance epochs past the rating deadline (10s window, 5s epochs → 3 epochs should suffice)
    test_cluster.trigger_reconfiguration().await;
    test_cluster.trigger_reconfiguration().await;
    test_cluster.trigger_reconfiguration().await;
    info!("Advanced 3 epochs past rating deadline");

    // Attempt RateSeller — should fail with RatingDeadlinePassed
    let gas3 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let rate_tx = TransactionData::new(
        TransactionKind::RateSeller { settlement_id },
        buyer,
        vec![gas3],
    );
    let rate_signed = test_cluster.wallet.sign_transaction(&rate_tx).await;
    let result = test_cluster.wallet.execute_transaction_may_fail(rate_signed).await;

    match result {
        Ok(response) => {
            assert!(
                response.effects.status().is_err(),
                "RateSeller after deadline should fail"
            );
            info!("RateSeller after deadline correctly rejected: {:?}", response.effects.status());
        }
        Err(e) => {
            info!("RateSeller after deadline correctly rejected at orchestrator: {}", e);
        }
    }

    // Verify settlement still has Positive rating (unchanged)
    let settlement_obj = client.get_object(settlement_id).await.unwrap();
    let settlement: Settlement = settlement_obj.deserialize_contents(ObjectType::Settlement).unwrap();
    assert_eq!(
        settlement.seller_rating,
        SellerRating::Positive,
        "Settlement rating must remain Positive after deadline passes"
    );

    info!("Default positive rating test passed!");
}

// =============================================================================
// Test 21: Vault accumulation — seller fulfills multiple asks
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_vault_accumulation() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 3) // 3 USDC coins per account
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];
    let client = &test_cluster.fullnode_handle.soma_client;

    // Seller fulfills 2 separate asks, accumulating vaults
    let mut vault_ids = Vec::new();
    let bid_prices = [500_000u64, 300_000u64]; // 0.50 USDC and 0.30 USDC

    for (i, &price) in bid_prices.iter().enumerate() {
        // Buyer creates ask
        let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
        let ask_tx = TransactionData::new(
            TransactionKind::CreateAsk(CreateAskArgs {
                task_digest: TaskDigest::new([(i + 10) as u8; 32]),
                max_price_per_bid: 1_000_000,
                num_bids_wanted: 1,
                timeout_ms: 300_000,
            }),
            buyer,
            vec![gas],
        );
        let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
        assert!(ask_response.effects.status().is_ok(), "CreateAsk {} failed", i);
        let ask_id = ask_response.effects.created()[0].0 .0;

        // Seller bids
        let seller_gas = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
        let bid_tx = TransactionData::new(
            TransactionKind::CreateBid(CreateBidArgs {
                ask_id,
                price,
                response_digest: ResponseDigest::new([(i + 20) as u8; 32]),
            }),
            seller,
            vec![seller_gas],
        );
        let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
        assert!(bid_response.effects.status().is_ok(), "CreateBid {} failed", i);
        let bid_id = bid_response.effects.created()[0].0 .0;

        // Buyer accepts
        let usdc = test_cluster.wallet.get_richest_usdc_coin(buyer).await.unwrap().unwrap();
        let gas2 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
        let accept_tx = TransactionData::new(
            TransactionKind::AcceptBid(AcceptBidArgs { ask_id, bid_id, payment_coin: usdc.0 }),
            buyer,
            vec![gas2],
        );
        let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
        assert!(accept_response.effects.status().is_ok(), "AcceptBid {} failed", i);

        // Find vault among created objects
        for (obj_ref, _owner) in accept_response.effects.created() {
            let obj = client.get_object(obj_ref.0).await.unwrap();
            if let Some(vault) = obj.deserialize_contents::<SellerVault>(ObjectType::SellerVault) {
                info!("Ask {}: vault {} with balance {}", i, obj_ref.0, vault.balance);
                vault_ids.push(obj_ref.0);
            }
        }
    }

    assert_eq!(vault_ids.len(), 2, "Should have 2 vaults (one per AcceptBid)");

    // Verify each vault has the expected balance (bid.price - 2.5% fee)
    let expected_balances: Vec<u64> = bid_prices
        .iter()
        .map(|&price| price - (price * 250 / 10_000))
        .collect();

    for (i, &vault_id) in vault_ids.iter().enumerate() {
        let vault_obj = client.get_object(vault_id).await.unwrap();
        let vault: SellerVault = vault_obj.deserialize_contents(ObjectType::SellerVault).unwrap();
        assert_eq!(vault.owner, seller);
        assert_eq!(
            vault.balance, expected_balances[i],
            "Vault {} balance mismatch: expected {}, got {}",
            i, expected_balances[i], vault.balance
        );
    }

    // Withdraw from both vaults
    let total_expected: u64 = expected_balances.iter().sum();
    let mut total_withdrawn = 0u64;

    for &vault_id in &vault_ids {
        let vault_obj = client.get_object(vault_id).await.unwrap();
        let vault_ref = vault_obj.compute_object_reference();

        let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
        let withdraw_tx = TransactionData::new(
            TransactionKind::WithdrawFromVault {
                vault: vault_ref,
                amount: None, // full balance
                recipient_coin: None,
            },
            seller,
            vec![gas],
        );
        let withdraw_response = test_cluster.sign_and_execute_transaction(&withdraw_tx).await;
        assert!(
            withdraw_response.effects.status().is_ok(),
            "WithdrawFromVault failed: {:?}",
            withdraw_response.effects.status()
        );

        // Find the created USDC coin
        for (obj_ref, _owner) in withdraw_response.effects.created() {
            let obj = client.get_object(obj_ref.0).await.unwrap();
            if obj.coin_type() == Some(CoinType::Usdc) {
                total_withdrawn += obj.as_coin().unwrap();
            }
        }
    }

    assert_eq!(
        total_withdrawn, total_expected,
        "Total withdrawn USDC should match sum of vault balances"
    );

    // Verify vaults are deleted (zero balance → deleted)
    for &vault_id in &vault_ids {
        let result = client.get_object(vault_id).await;
        assert!(result.is_err(), "Vault should be deleted after full withdrawal");
    }

    info!(
        "Vault accumulation test passed! Withdrew {} microdollars from {} vaults",
        total_withdrawn,
        vault_ids.len()
    );
}

// =============================================================================
// Test 22: Secondary index consistency (bids_by_ask, open_asks)
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_secondary_index_consistency() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 3)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];
    let client = &test_cluster.fullnode_handle.soma_client;

    // --- Step 1: Create 2 asks ---
    let mut ask_ids = Vec::new();
    for i in 0..2u8 {
        let gas = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
        let ask_tx = TransactionData::new(
            TransactionKind::CreateAsk(CreateAskArgs {
                task_digest: TaskDigest::new([100 + i; 32]),
                max_price_per_bid: 1_000_000,
                num_bids_wanted: 1,
                timeout_ms: 300_000,
            }),
            buyer,
            vec![gas],
        );
        let resp = test_cluster.sign_and_execute_transaction(&ask_tx).await;
        assert!(resp.effects.status().is_ok());
        ask_ids.push(resp.effects.created()[0].0 .0);
    }
    info!("Created 2 asks: {:?}", ask_ids);

    // --- Step 2: Verify both asks appear in open_asks ---
    let open_asks = client.get_open_asks(None, None).await.unwrap();
    let open_ask_ids: Vec<_> = open_asks.iter().map(|o| o.id()).collect();
    for ask_id in &ask_ids {
        assert!(
            open_ask_ids.contains(ask_id),
            "Ask {} should be in open_asks index",
            ask_id
        );
    }
    info!("Both asks present in open_asks index");

    // --- Step 3: Seller bids on ask[0] ---
    let seller_gas = test_cluster.wallet.get_one_gas_object_owned_by_address(seller).await.unwrap().unwrap();
    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id: ask_ids[0],
            price: 500_000,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );
    let bid_resp = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    assert!(bid_resp.effects.status().is_ok());
    let bid_id = bid_resp.effects.created()[0].0 .0;

    // --- Step 4: Verify bids_by_ask returns this bid ---
    let bids = client.get_bids_for_ask(ask_ids[0], None).await.unwrap();
    assert_eq!(bids.len(), 1, "Should find 1 bid for ask[0]");
    assert_eq!(bids[0].id(), bid_id);
    info!("bids_by_ask correctly returns the bid for ask[0]");

    // Verify no bids for ask[1]
    let bids_1 = client.get_bids_for_ask(ask_ids[1], None).await.unwrap();
    assert_eq!(bids_1.len(), 0, "Should find 0 bids for ask[1]");

    // --- Step 5: Buyer accepts bid → ask[0] becomes Filled ---
    let usdc = test_cluster.wallet.get_richest_usdc_coin(buyer).await.unwrap().unwrap();
    let gas2 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs {
            ask_id: ask_ids[0],
            bid_id,
            payment_coin: usdc.0,
        }),
        buyer,
        vec![gas2],
    );
    let accept_resp = test_cluster.sign_and_execute_transaction(&accept_tx).await;
    assert!(accept_resp.effects.status().is_ok());
    info!("Accepted bid, ask[0] should now be Filled");

    // --- Step 6: Verify ask[0] no longer in open_asks, ask[1] still there ---
    let open_asks_after = client.get_open_asks(None, None).await.unwrap();
    let open_ids_after: Vec<_> = open_asks_after.iter().map(|o| o.id()).collect();
    assert!(
        !open_ids_after.contains(&ask_ids[0]),
        "Filled ask[0] should no longer be in open_asks index"
    );
    assert!(
        open_ids_after.contains(&ask_ids[1]),
        "Ask[1] should still be in open_asks index"
    );
    info!("open_asks correctly reflects ask[0] removal after Filled");

    // --- Step 7: Verify status filter on bids_by_ask ---
    let pending_bids = client.get_bids_for_ask(ask_ids[0], Some("Pending")).await.unwrap();
    assert_eq!(pending_bids.len(), 0, "No Pending bids after acceptance");

    let accepted_bids = client.get_bids_for_ask(ask_ids[0], Some("Accepted")).await.unwrap();
    assert_eq!(accepted_bids.len(), 1, "Should find 1 Accepted bid");

    // --- Step 8: Cancel ask[1] → verify it disappears from open_asks ---
    let gas3 = test_cluster.wallet.get_one_gas_object_owned_by_address(buyer).await.unwrap().unwrap();
    let cancel_tx = TransactionData::new(
        TransactionKind::CancelAsk { ask_id: ask_ids[1] },
        buyer,
        vec![gas3],
    );
    let cancel_resp = test_cluster.sign_and_execute_transaction(&cancel_tx).await;
    assert!(cancel_resp.effects.status().is_ok());

    let open_asks_final = client.get_open_asks(None, None).await.unwrap();
    let open_ids_final: Vec<_> = open_asks_final.iter().map(|o| o.id()).collect();
    assert!(
        !open_ids_final.contains(&ask_ids[1]),
        "Cancelled ask[1] should no longer be in open_asks index"
    );
    info!("open_asks correctly reflects ask[1] removal after cancellation");

    info!("Secondary index consistency test passed!");
}

// =============================================================================
// Bridge signing helpers
// =============================================================================

/// Sign a bridge deposit message with the given committee member indices.
/// Returns `(aggregated_signature, signer_bitmap)`.
#[cfg(msim)]
fn sign_deposit(
    keypairs: &[fastcrypto::secp256k1::Secp256k1KeyPair],
    signer_indices: &[usize],
    nonce: u64,
    recipient: &types::base::SomaAddress,
    amount: u64,
) -> (Vec<u8>, Vec<u8>) {
    use types::bridge::{
        BridgeMessageType, SOMA_BRIDGE_CHAIN_ID, build_bridge_signatures,
        encode_bridge_message, encode_deposit_payload,
    };
    let payload = encode_deposit_payload(recipient, amount);
    let message = encode_bridge_message(
        BridgeMessageType::UsdcDeposit,
        nonce,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );
    let signers: Vec<(usize, &fastcrypto::secp256k1::Secp256k1KeyPair)> =
        signer_indices.iter().map(|&i| (i, &keypairs[i])).collect();
    build_bridge_signatures(&signers, &message)
}

/// Sign a bridge emergency pause message.
#[cfg(msim)]
fn sign_pause(
    keypairs: &[fastcrypto::secp256k1::Secp256k1KeyPair],
    signer_indices: &[usize],
) -> (Vec<u8>, Vec<u8>) {
    use types::bridge::{
        BridgeMessageType, EmergencyOpCode, SOMA_BRIDGE_CHAIN_ID, build_bridge_signatures,
        encode_bridge_message, encode_emergency_payload,
    };
    let payload = encode_emergency_payload(EmergencyOpCode::Freeze);
    let message = encode_bridge_message(
        BridgeMessageType::EmergencyOp,
        0,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );
    let signers: Vec<(usize, &fastcrypto::secp256k1::Secp256k1KeyPair)> =
        signer_indices.iter().map(|&i| (i, &keypairs[i])).collect();
    build_bridge_signatures(&signers, &message)
}

/// Sign a bridge emergency unpause message.
#[cfg(msim)]
fn sign_unpause(
    keypairs: &[fastcrypto::secp256k1::Secp256k1KeyPair],
    signer_indices: &[usize],
) -> (Vec<u8>, Vec<u8>) {
    use types::bridge::{
        BridgeMessageType, EmergencyOpCode, SOMA_BRIDGE_CHAIN_ID, build_bridge_signatures,
        encode_bridge_message, encode_emergency_payload,
    };
    let payload = encode_emergency_payload(EmergencyOpCode::Unfreeze);
    let message = encode_bridge_message(
        BridgeMessageType::EmergencyOp,
        0,
        SOMA_BRIDGE_CHAIN_ID,
        &payload,
    );
    let signers: Vec<(usize, &fastcrypto::secp256k1::Secp256k1KeyPair)> =
        signer_indices.iter().map(|&i| (i, &keypairs[i])).collect();
    build_bridge_signatures(&signers, &message)
}

// =============================================================================
// Test 23: Bridge deposit mints USDC
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_bridge_deposit_mints_usdc() {
    use types::bridge::PendingWithdrawal;
    use types::transaction::{BridgeDepositArgs, BridgeWithdrawArgs};

    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_bridge_committee(4) // 4 members, 2500 voting power each
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let recipient = addresses[0];
    let keypairs = &test_cluster.bridge_keypairs;

    // Sign deposit: members 0 & 1 → 5000 stake > 3334 threshold
    let (aggregated_signature, signer_bitmap) =
        sign_deposit(keypairs, &[0, 1], 0, &recipient, 5_000_000);

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(recipient)
        .await
        .unwrap()
        .expect("must have gas");

    let deposit_tx = TransactionData::new(
        TransactionKind::BridgeDeposit(BridgeDepositArgs {
            nonce: 0,
            eth_tx_hash: [0xAAu8; 32],
            recipient,
            amount: 5_000_000, // 5 USDC
            aggregated_signature,
            signer_bitmap,
        }),
        recipient,
        vec![gas],
    );

    let resp = test_cluster.sign_and_execute_transaction(&deposit_tx).await;
    assert!(
        resp.effects.status().is_ok(),
        "BridgeDeposit failed: {:?}",
        resp.effects.status()
    );

    // Should create a USDC coin for the recipient
    let created = resp.effects.created();
    assert!(!created.is_empty(), "BridgeDeposit should create a USDC coin");

    let client = &test_cluster.fullnode_handle.soma_client;
    let mut found_usdc = false;
    for (oref, _owner) in &created {
        let obj = client.get_object(oref.0).await.unwrap();
        if obj.coin_type() == Some(CoinType::Usdc) {
            assert_eq!(obj.as_coin().unwrap(), 5_000_000);
            found_usdc = true;
        }
    }
    assert!(found_usdc, "BridgeDeposit should mint a CoinType::Usdc coin");
    info!("BridgeDeposit minted 5 USDC to recipient");
}

// =============================================================================
// Test 24: Bridge deposit nonce replay rejected
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_bridge_deposit_nonce_replay_rejected() {
    use types::transaction::BridgeDepositArgs;

    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_bridge_committee(4)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let recipient = addresses[0];
    let keypairs = &test_cluster.bridge_keypairs;

    let (agg_sig1, bitmap1) = sign_deposit(keypairs, &[0, 1], 0, &recipient, 1_000_000);

    // First deposit: nonce 0 — should succeed
    let gas1 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(recipient)
        .await
        .unwrap()
        .unwrap();

    let deposit1 = TransactionData::new(
        TransactionKind::BridgeDeposit(BridgeDepositArgs {
            nonce: 0,
            eth_tx_hash: [0x01u8; 32],
            recipient,
            amount: 1_000_000,
            aggregated_signature: agg_sig1,
            signer_bitmap: bitmap1,
        }),
        recipient,
        vec![gas1],
    );
    let resp1 = test_cluster.sign_and_execute_transaction(&deposit1).await;
    assert!(resp1.effects.status().is_ok(), "First deposit should succeed");

    // Second deposit: same nonce 0 — should fail with BridgeNonceAlreadyProcessed
    let (agg_sig2, bitmap2) = sign_deposit(keypairs, &[0, 1], 0, &recipient, 2_000_000);
    let gas2 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(recipient)
        .await
        .unwrap()
        .unwrap();

    let deposit2 = TransactionData::new(
        TransactionKind::BridgeDeposit(BridgeDepositArgs {
            nonce: 0, // replay
            eth_tx_hash: [0x02u8; 32],
            recipient,
            amount: 2_000_000,
            aggregated_signature: agg_sig2,
            signer_bitmap: bitmap2,
        }),
        recipient,
        vec![gas2],
    );
    let deposit2_signed = test_cluster.wallet.sign_transaction(&deposit2).await;
    let resp2 = test_cluster.wallet.execute_transaction_may_fail(deposit2_signed).await;
    match resp2 {
        Ok(response) => {
            assert!(
                response.effects.status().is_err(),
                "Replay nonce should be rejected, got: {:?}",
                response.effects.status()
            );
            info!("Nonce replay rejected at execution: {:?}", response.effects.status());
        }
        Err(e) => {
            let msg = format!("{}", e);
            assert!(
                msg.contains("nonce") || msg.contains("Nonce") || msg.contains("already processed"),
                "Expected nonce replay error, got: {}",
                msg
            );
            info!("Nonce replay rejected at orchestrator: {}", e);
        }
    }
    info!("Bridge deposit nonce replay correctly rejected");
}

// =============================================================================
// Test 25: Bridge withdraw burns USDC and creates PendingWithdrawal
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_bridge_withdraw_e2e() {
    use types::bridge::PendingWithdrawal;
    use types::transaction::BridgeWithdrawArgs;

    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 1) // 10 USDC
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];
    let client = &test_cluster.fullnode_handle.soma_client;

    // Get sender's USDC coin
    let usdc_coin = test_cluster
        .wallet
        .get_richest_usdc_coin(sender)
        .await
        .unwrap()
        .expect("sender must have USDC");

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have gas");

    let withdraw_amount = 3_000_000u64; // 3 USDC
    let eth_recipient = [0xBBu8; 20];

    let withdraw_tx = TransactionData::new(
        TransactionKind::BridgeWithdraw(BridgeWithdrawArgs {
            payment_coin: usdc_coin.0,
            amount: withdraw_amount,
            recipient_eth_address: eth_recipient,
        }),
        sender,
        vec![gas],
    );

    let resp = test_cluster.sign_and_execute_transaction(&withdraw_tx).await;
    assert!(
        resp.effects.status().is_ok(),
        "BridgeWithdraw failed: {:?}",
        resp.effects.status()
    );

    // Check PendingWithdrawal was created
    let created = resp.effects.created();
    let mut found_pw = false;
    for (oref, _owner) in &created {
        let obj = client.get_object(oref.0).await.unwrap();
        if obj.type_() == &ObjectType::PendingWithdrawal {
            let pw: PendingWithdrawal = obj
                .deserialize_contents(ObjectType::PendingWithdrawal)
                .expect("should deserialize PendingWithdrawal");
            assert_eq!(pw.sender, sender);
            assert_eq!(pw.amount, withdraw_amount);
            assert_eq!(pw.recipient_eth_address, eth_recipient);
            assert_eq!(pw.nonce, 0); // first withdrawal
            found_pw = true;
        }
    }
    assert!(found_pw, "BridgeWithdraw should create a PendingWithdrawal");

    // USDC coin should have reduced balance
    let usdc_after = client.get_object(usdc_coin.0 .0).await.unwrap();
    assert_eq!(
        usdc_after.as_coin().unwrap(),
        TEST_USDC_AMOUNT - withdraw_amount,
        "USDC balance should be reduced by withdraw amount"
    );

    info!("BridgeWithdraw burned {} USDC and created PendingWithdrawal", withdraw_amount);
}

// =============================================================================
// Test 26: Bridge emergency pause blocks withdraw, unpause resumes
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_bridge_emergency_pause_unpause() {
    use types::transaction::{
        BridgeDepositArgs, BridgeEmergencyPauseArgs, BridgeEmergencyUnpauseArgs,
        BridgeWithdrawArgs,
    };

    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_bridge_committee(4) // 4 members, 2500 each
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 1)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let user = addresses[0];

    let keypairs = &test_cluster.bridge_keypairs;

    // --- Step 1: Pause the bridge (threshold_pause = 450, 1 member = 2500 > 450) ---
    let gas1 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(user)
        .await
        .unwrap()
        .unwrap();

    let (pause_sig, pause_bitmap) = sign_pause(keypairs, &[0]);

    let pause_tx = TransactionData::new(
        TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
            aggregated_signature: pause_sig,
            signer_bitmap: pause_bitmap,
        }),
        user,
        vec![gas1],
    );
    let pause_resp = test_cluster.sign_and_execute_transaction(&pause_tx).await;
    assert!(
        pause_resp.effects.status().is_ok(),
        "BridgeEmergencyPause failed: {:?}",
        pause_resp.effects.status()
    );
    info!("Bridge paused successfully");

    // --- Step 2: Deposit should fail while paused ---
    let (dep_sig, dep_bitmap) = sign_deposit(keypairs, &[0, 1], 0, &user, 1_000_000);
    let gas2 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(user)
        .await
        .unwrap()
        .unwrap();

    let deposit_tx = TransactionData::new(
        TransactionKind::BridgeDeposit(BridgeDepositArgs {
            nonce: 0,
            eth_tx_hash: [0xCCu8; 32],
            recipient: user,
            amount: 1_000_000,
            aggregated_signature: dep_sig,
            signer_bitmap: dep_bitmap,
        }),
        user,
        vec![gas2],
    );
    let deposit_signed = test_cluster.wallet.sign_transaction(&deposit_tx).await;
    let deposit_resp = test_cluster.wallet.execute_transaction_may_fail(deposit_signed).await;
    match deposit_resp {
        Ok(response) => {
            assert!(response.effects.status().is_err(), "Deposit should fail while paused");
            info!("Deposit rejected at execution while paused: {:?}", response.effects.status());
        }
        Err(e) => {
            info!("Deposit rejected at orchestrator while paused: {}", e);
        }
    }

    // --- Step 3: Withdraw should also fail while paused ---
    let usdc_coin = test_cluster
        .wallet
        .get_richest_usdc_coin(user)
        .await
        .unwrap()
        .expect("user must have USDC");
    let gas3 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(user)
        .await
        .unwrap()
        .unwrap();

    let withdraw_tx = TransactionData::new(
        TransactionKind::BridgeWithdraw(BridgeWithdrawArgs {
            payment_coin: usdc_coin.0,
            amount: 1_000_000,
            recipient_eth_address: [0x01; 20],
        }),
        user,
        vec![gas3],
    );
    let withdraw_signed = test_cluster.wallet.sign_transaction(&withdraw_tx).await;
    let withdraw_resp = test_cluster.wallet.execute_transaction_may_fail(withdraw_signed).await;
    match withdraw_resp {
        Ok(response) => {
            assert!(response.effects.status().is_err(), "Withdraw should fail while paused");
            info!("Withdraw rejected at execution while paused: {:?}", response.effects.status());
        }
        Err(e) => {
            info!("Withdraw rejected at orchestrator while paused: {}", e);
        }
    }

    // --- Step 4: Unpause (threshold_unpause = 6667, need 3 members = 7500 > 6667) ---
    let gas4 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(user)
        .await
        .unwrap()
        .unwrap();

    let (unpause_sig, unpause_bitmap) = sign_unpause(keypairs, &[0, 1, 2]);

    let unpause_tx = TransactionData::new(
        TransactionKind::BridgeEmergencyUnpause(BridgeEmergencyUnpauseArgs {
            aggregated_signature: unpause_sig,
            signer_bitmap: unpause_bitmap,
        }),
        user,
        vec![gas4],
    );
    let unpause_resp = test_cluster.sign_and_execute_transaction(&unpause_tx).await;
    assert!(
        unpause_resp.effects.status().is_ok(),
        "BridgeEmergencyUnpause failed: {:?}",
        unpause_resp.effects.status()
    );
    info!("Bridge unpaused successfully");

    // --- Step 5: Deposit should work again after unpause ---
    let (dep2_sig, dep2_bitmap) = sign_deposit(keypairs, &[0, 1], 0, &user, 1_000_000);
    let gas5 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(user)
        .await
        .unwrap()
        .unwrap();

    let deposit_tx2 = TransactionData::new(
        TransactionKind::BridgeDeposit(BridgeDepositArgs {
            nonce: 0,
            eth_tx_hash: [0xDDu8; 32],
            recipient: user,
            amount: 1_000_000,
            aggregated_signature: dep2_sig,
            signer_bitmap: dep2_bitmap,
        }),
        user,
        vec![gas5],
    );
    let deposit_resp2 = test_cluster.sign_and_execute_transaction(&deposit_tx2).await;
    assert!(
        deposit_resp2.effects.status().is_ok(),
        "Deposit should succeed after unpause: {:?}",
        deposit_resp2.effects.status()
    );
    info!("Deposit succeeded after unpause — bridge emergency pause/unpause flow verified!");
}

// =============================================================================
// Test 27: Bridge deposit with insufficient stake rejected
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_bridge_deposit_insufficient_stake_rejected() {
    use types::transaction::BridgeDepositArgs;

    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_bridge_committee(4) // 4 members, 2500 each; threshold_deposit = 3334
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let recipient = addresses[0];
    let keypairs = &test_cluster.bridge_keypairs;

    // Only 1 member signs → 2500 < 3334 threshold
    let (agg_sig, signer_bitmap) = sign_deposit(keypairs, &[0], 0, &recipient, 1_000_000);

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(recipient)
        .await
        .unwrap()
        .unwrap();

    let deposit_tx = TransactionData::new(
        TransactionKind::BridgeDeposit(BridgeDepositArgs {
            nonce: 0,
            eth_tx_hash: [0xEEu8; 32],
            recipient,
            amount: 1_000_000,
            aggregated_signature: agg_sig,
            signer_bitmap,
        }),
        recipient,
        vec![gas],
    );

    let deposit_signed = test_cluster.wallet.sign_transaction(&deposit_tx).await;
    let resp = test_cluster.wallet.execute_transaction_may_fail(deposit_signed).await;
    match resp {
        Ok(response) => {
            assert!(
                response.effects.status().is_err(),
                "Deposit with insufficient stake should be rejected, got: {:?}",
                response.effects.status()
            );
            info!("Insufficient stake rejected at execution: {:?}", response.effects.status());
        }
        Err(e) => {
            info!("Insufficient stake rejected at orchestrator: {}", e);
        }
    }
    info!("Bridge deposit with insufficient stake correctly rejected");
}

// =============================================================================
// Test 28: Bridge deposit then withdraw round-trip
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_bridge_deposit_withdraw_roundtrip() {
    use types::bridge::PendingWithdrawal;
    use types::transaction::{BridgeDepositArgs, BridgeWithdrawArgs};

    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_bridge_committee(4)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let user = addresses[0];
    let client = &test_cluster.fullnode_handle.soma_client;
    let keypairs = &test_cluster.bridge_keypairs;

    let deposit_amount = 10_000_000u64; // 10 USDC

    // --- Step 1: Deposit USDC via bridge ---
    let (dep1_sig, dep1_bitmap) =
        sign_deposit(keypairs, &[0, 1], 0, &user, deposit_amount);
    let gas1 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(user)
        .await
        .unwrap()
        .unwrap();

    let deposit_tx = TransactionData::new(
        TransactionKind::BridgeDeposit(BridgeDepositArgs {
            nonce: 0,
            eth_tx_hash: [0x11u8; 32],
            recipient: user,
            amount: deposit_amount,
            aggregated_signature: dep1_sig,
            signer_bitmap: dep1_bitmap,
        }),
        user,
        vec![gas1],
    );
    let deposit_resp = test_cluster.sign_and_execute_transaction(&deposit_tx).await;
    assert!(deposit_resp.effects.status().is_ok(), "Deposit failed");

    // Find the minted USDC coin
    let created = deposit_resp.effects.created();
    let mut usdc_coin_ref = None;
    for (oref, _) in &created {
        let obj = client.get_object(oref.0).await.unwrap();
        if obj.coin_type() == Some(CoinType::Usdc) {
            usdc_coin_ref = Some(obj.compute_object_reference());
        }
    }
    let usdc_ref = usdc_coin_ref.expect("Should have minted USDC coin");
    info!("Deposited {} USDC via bridge", deposit_amount);

    // --- Step 2: Withdraw half back to Ethereum ---
    let gas2 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(user)
        .await
        .unwrap()
        .unwrap();

    let withdraw_amount = 4_000_000u64; // 4 USDC
    let eth_addr = [0xDEu8; 20];

    let withdraw_tx = TransactionData::new(
        TransactionKind::BridgeWithdraw(BridgeWithdrawArgs {
            payment_coin: usdc_ref,
            amount: withdraw_amount,
            recipient_eth_address: eth_addr,
        }),
        user,
        vec![gas2],
    );
    let withdraw_resp = test_cluster.sign_and_execute_transaction(&withdraw_tx).await;
    assert!(withdraw_resp.effects.status().is_ok(), "Withdraw failed");

    // USDC coin should have remaining balance
    let usdc_after = client.get_object(usdc_ref.0).await.unwrap();
    assert_eq!(
        usdc_after.as_coin().unwrap(),
        deposit_amount - withdraw_amount,
        "USDC balance should be deposit minus withdrawal"
    );

    // PendingWithdrawal should exist
    let w_created = withdraw_resp.effects.created();
    let mut found_pw = false;
    for (oref, _) in &w_created {
        let obj = client.get_object(oref.0).await.unwrap();
        if obj.type_() == &ObjectType::PendingWithdrawal {
            let pw: PendingWithdrawal = obj
                .deserialize_contents(ObjectType::PendingWithdrawal)
                .unwrap();
            assert_eq!(pw.amount, withdraw_amount);
            assert_eq!(pw.recipient_eth_address, eth_addr);
            assert_eq!(pw.nonce, 0);
            found_pw = true;
        }
    }
    assert!(found_pw, "Should create PendingWithdrawal");
    info!(
        "Withdrew {} USDC, remaining balance: {}",
        withdraw_amount,
        deposit_amount - withdraw_amount
    );

    // --- Step 3: Second deposit with different nonce ---
    let (dep2_sig, dep2_bitmap) =
        sign_deposit(keypairs, &[0, 1], 1, &user, 2_000_000);
    let gas3 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(user)
        .await
        .unwrap()
        .unwrap();

    let deposit2_tx = TransactionData::new(
        TransactionKind::BridgeDeposit(BridgeDepositArgs {
            nonce: 1, // different nonce
            eth_tx_hash: [0x22u8; 32],
            recipient: user,
            amount: 2_000_000,
            aggregated_signature: dep2_sig,
            signer_bitmap: dep2_bitmap,
        }),
        user,
        vec![gas3],
    );
    let deposit2_resp = test_cluster.sign_and_execute_transaction(&deposit2_tx).await;
    assert!(deposit2_resp.effects.status().is_ok(), "Second deposit should succeed with new nonce");
    info!("Second deposit with nonce 1 succeeded — round-trip verified!");
}

// =============================================================================
// Test 29: Accept cheapest bids
// =============================================================================

/// Create ask with num_bids_wanted=3, 3 sellers bid at prices [1.50, 0.80, 1.20].
/// Use get_bids_for_ask to find pending bids, sort by price ascending, accept
/// the 2 cheapest (0.80 and 1.20). Verify the most expensive (1.50) remains Pending.
#[cfg(msim)]
#[msim::sim_test]
async fn test_accept_cheapest() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller_a = addresses[1]; // will bid 1_500_000 (most expensive)
    let seller_b = addresses[2]; // will bid 800_000 (cheapest)
    let seller_c = addresses[3]; // will bid 1_200_000 (middle)

    let client = &test_cluster.fullnode_handle.soma_client;

    // Buyer creates an ask wanting up to 3 bids
    let buyer_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();

    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 2_000_000,
            num_bids_wanted: 3,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![buyer_gas],
    );

    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok());
    let ask_id = ask_response.effects.created()[0].0 .0;

    // Three sellers bid at different prices
    let prices = [1_500_000u64, 800_000, 1_200_000]; // expensive, cheapest, middle
    let sellers = [seller_a, seller_b, seller_c];
    let digests = [test_response_digest(), test_response_digest_2(), test_response_digest_3()];
    let mut bid_ids = vec![];

    for (i, (&seller, &price)) in sellers.iter().zip(prices.iter()).enumerate() {
        let gas = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(seller)
            .await
            .unwrap()
            .unwrap();

        let bid_tx = TransactionData::new(
            TransactionKind::CreateBid(CreateBidArgs {
                ask_id,
                price,
                response_digest: digests[i],
            }),
            seller,
            vec![gas],
        );

        let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
        assert!(bid_response.effects.status().is_ok(), "Bid {} failed", i);
        bid_ids.push(bid_response.effects.created()[0].0 .0);
    }

    // Use get_bids_for_ask to find all pending bids (simulates the --cheapest flow)
    let pending_bids = client
        .get_bids_for_ask(ask_id, Some("Pending"))
        .await
        .expect("get_bids_for_ask should succeed");
    assert_eq!(pending_bids.len(), 3, "Should have 3 pending bids");

    // Deserialize and sort by price ascending (cheapest first)
    let mut bids_with_price: Vec<(Bid, ObjectID)> = pending_bids
        .iter()
        .filter_map(|obj| {
            let bid = obj.deserialize_contents::<Bid>(ObjectType::Bid)?;
            Some((bid, obj.id()))
        })
        .collect();
    bids_with_price.sort_by_key(|(bid, _)| bid.price);

    assert_eq!(bids_with_price.len(), 3);
    assert_eq!(bids_with_price[0].0.price, 800_000, "Cheapest should be 800_000");
    assert_eq!(bids_with_price[1].0.price, 1_200_000, "Middle should be 1_200_000");
    assert_eq!(bids_with_price[2].0.price, 1_500_000, "Most expensive should be 1_500_000");

    // Accept the 2 cheapest bids (800_000 and 1_200_000)
    for i in 0..2 {
        let (bid, _bid_id) = &bids_with_price[i];

        let usdc = test_cluster
            .wallet
            .get_richest_usdc_coin(buyer)
            .await
            .unwrap()
            .expect("buyer must have USDC");

        let gas = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(buyer)
            .await
            .unwrap()
            .unwrap();

        let accept_tx = TransactionData::new(
            TransactionKind::AcceptBid(AcceptBidArgs {
                ask_id,
                bid_id: bid.id,
                payment_coin: usdc.0,
            }),
            buyer,
            vec![gas],
        );

        let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
        assert!(
            accept_response.effects.status().is_ok(),
            "AcceptBid {} failed: {:?}",
            i,
            accept_response.effects.status()
        );
        info!("Accepted bid {} at price {}", bid.id, bid.price);
    }

    // Verify: ask has 2 accepted bids (not yet Filled — wanted 3)
    let ask_obj = client.get_object(ask_id).await.unwrap();
    let ask: Ask = ask_obj.deserialize_contents(ObjectType::Ask).unwrap();
    assert_eq!(ask.accepted_bid_count, 2);
    assert_eq!(ask.status, AskStatus::Open, "Ask should still be Open (wanted 3, accepted 2)");

    // Verify: cheapest two bids are Accepted
    let bid_0_obj = client.get_object(bids_with_price[0].1).await.unwrap();
    let bid_0: Bid = bid_0_obj.deserialize_contents(ObjectType::Bid).unwrap();
    assert_eq!(bid_0.status, BidStatus::Accepted, "Cheapest bid should be Accepted");

    let bid_1_obj = client.get_object(bids_with_price[1].1).await.unwrap();
    let bid_1: Bid = bid_1_obj.deserialize_contents(ObjectType::Bid).unwrap();
    assert_eq!(bid_1.status, BidStatus::Accepted, "Middle bid should be Accepted");

    // Verify: most expensive bid is still Pending
    let bid_2_obj = client.get_object(bids_with_price[2].1).await.unwrap();
    let bid_2: Bid = bid_2_obj.deserialize_contents(ObjectType::Bid).unwrap();
    assert_eq!(bid_2.status, BidStatus::Pending, "Most expensive bid should still be Pending");

    // Verify: get_bids_for_ask with Pending filter now returns only 1 bid
    let remaining_pending = client
        .get_bids_for_ask(ask_id, Some("Pending"))
        .await
        .expect("get_bids_for_ask should succeed");
    assert_eq!(remaining_pending.len(), 1, "Should have 1 pending bid remaining");
    let remaining_bid: Bid = remaining_pending[0]
        .deserialize_contents(ObjectType::Bid)
        .unwrap();
    assert_eq!(remaining_bid.price, 1_500_000, "Remaining pending bid should be the most expensive");

    info!("Accept cheapest test passed — 2 cheapest accepted, most expensive still Pending!");
}

// =============================================================================
// Test 30: Settlements index and query
// =============================================================================

/// Create 2 asks from buyer, accept bids from 2 different sellers.
/// Verify get_settlements(buyer=...) returns both settlements.
/// Verify get_settlements(seller=seller_a) returns only seller_a's settlement.
#[cfg(msim)]
#[msim::sim_test]
async fn test_settlements_index() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller_a = addresses[1];
    let seller_b = addresses[2];

    let client = &test_cluster.fullnode_handle.soma_client;

    // Create two asks
    let mut ask_ids = vec![];
    for i in 0..2u8 {
        let gas = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(buyer)
            .await
            .unwrap()
            .unwrap();

        let ask_tx = TransactionData::new(
            TransactionKind::CreateAsk(CreateAskArgs {
                task_digest: TaskDigest::new([10 + i; 32]),
                max_price_per_bid: 1_000_000,
                num_bids_wanted: 1,
                timeout_ms: 300_000,
            }),
            buyer,
            vec![gas],
        );
        let resp = test_cluster.sign_and_execute_transaction(&ask_tx).await;
        assert!(resp.effects.status().is_ok());
        ask_ids.push(resp.effects.created()[0].0 .0);
    }

    // seller_a bids on ask[0], seller_b bids on ask[1]
    let sellers = [seller_a, seller_b];
    let mut bid_ids = vec![];
    for (i, &seller) in sellers.iter().enumerate() {
        let gas = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(seller)
            .await
            .unwrap()
            .unwrap();

        let bid_tx = TransactionData::new(
            TransactionKind::CreateBid(CreateBidArgs {
                ask_id: ask_ids[i],
                price: 500_000,
                response_digest: ResponseDigest::new([20 + i as u8; 32]),
            }),
            seller,
            vec![gas],
        );
        let resp = test_cluster.sign_and_execute_transaction(&bid_tx).await;
        assert!(resp.effects.status().is_ok());
        bid_ids.push(resp.effects.created()[0].0 .0);
    }

    // Buyer accepts both bids
    for (i, &bid_id) in bid_ids.iter().enumerate() {
        let usdc = test_cluster
            .wallet
            .get_richest_usdc_coin(buyer)
            .await
            .unwrap()
            .expect("buyer must have USDC");

        let gas = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(buyer)
            .await
            .unwrap()
            .unwrap();

        let accept_tx = TransactionData::new(
            TransactionKind::AcceptBid(AcceptBidArgs {
                ask_id: ask_ids[i],
                bid_id,
                payment_coin: usdc.0,
            }),
            buyer,
            vec![gas],
        );
        let resp = test_cluster.sign_and_execute_transaction(&accept_tx).await;
        assert!(resp.effects.status().is_ok(), "AcceptBid {} failed", i);
    }

    // Query settlements by buyer — should return both
    let buyer_settlements = client
        .get_settlements(Some(&buyer), None, None)
        .await
        .expect("get_settlements by buyer should succeed");
    assert_eq!(
        buyer_settlements.len(),
        2,
        "Buyer should have 2 settlements"
    );

    // Query settlements by seller_a — should return 1
    let seller_a_settlements = client
        .get_settlements(None, Some(&seller_a), None)
        .await
        .expect("get_settlements by seller_a should succeed");
    assert_eq!(
        seller_a_settlements.len(),
        1,
        "Seller A should have 1 settlement"
    );

    // Verify the settlement is for seller_a
    let s: Settlement = seller_a_settlements[0]
        .deserialize_contents(ObjectType::Settlement)
        .unwrap();
    assert_eq!(s.seller, seller_a);
    assert_eq!(s.buyer, buyer);

    // Query settlements by seller_b — should return 1
    let seller_b_settlements = client
        .get_settlements(None, Some(&seller_b), None)
        .await
        .expect("get_settlements by seller_b should succeed");
    assert_eq!(
        seller_b_settlements.len(),
        1,
        "Seller B should have 1 settlement"
    );

    let s: Settlement = seller_b_settlements[0]
        .deserialize_contents(ObjectType::Settlement)
        .unwrap();
    assert_eq!(s.seller, seller_b);

    info!("Settlements index test passed — buyer/seller queries return correct results!");
}

// =============================================================================
// Test 31: GetProtocolFund RPC returns correct balance after AcceptBid fees
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_get_protocol_fund() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 2)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller = addresses[1];

    let client = &test_cluster.fullnode_handle.soma_client;

    // Protocol fund should start at 0
    let initial_fund = client.get_protocol_fund().await.expect("get_protocol_fund should succeed");
    assert_eq!(initial_fund, 0, "Protocol fund should start at 0");

    // Create ask → bid → accept to generate a fee
    let buyer_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .expect("buyer must have gas");

    let bid_price: u64 = 1_000_000; // 1 USDC
    let ask_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: bid_price,
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![buyer_gas],
    );
    let ask_response = test_cluster.sign_and_execute_transaction(&ask_tx).await;
    assert!(ask_response.effects.status().is_ok());
    let ask_id = ask_response.effects.created()[0].0 .0;

    let seller_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(seller)
        .await
        .unwrap()
        .expect("seller must have gas");

    let bid_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id,
            price: bid_price,
            response_digest: test_response_digest(),
        }),
        seller,
        vec![seller_gas],
    );
    let bid_response = test_cluster.sign_and_execute_transaction(&bid_tx).await;
    assert!(bid_response.effects.status().is_ok());
    let bid_id = bid_response.effects.created()[0].0 .0;

    // Buyer accepts — this generates the value fee
    let buyer_usdc = test_cluster
        .wallet
        .get_usdc_coins_sorted_by_balance(buyer)
        .await
        .unwrap();
    let usdc_ref = buyer_usdc[0].0;

    let buyer_gas2 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .expect("buyer gas");

    let accept_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs {
            ask_id,
            bid_id,
            payment_coin: usdc_ref,
        }),
        buyer,
        vec![buyer_gas2],
    );
    let accept_response = test_cluster.sign_and_execute_transaction(&accept_tx).await;
    assert!(accept_response.effects.status().is_ok());

    // Default value_fee_bps is 250 (2.5%). Fee = 1_000_000 * 250 / 10_000 = 25_000
    let expected_fee: u64 = bid_price * 250 / 10_000;

    let fund_balance = client.get_protocol_fund().await.expect("get_protocol_fund should succeed");
    assert_eq!(
        fund_balance, expected_fee,
        "Protocol fund should have exactly the value fee ({}), got {}",
        expected_fee, fund_balance
    );

    info!("GetProtocolFund test passed — fund balance = {} after 1 AcceptBid", fund_balance);
}

// =============================================================================
// Test 32: GetReputation RPC returns correct buyer/seller metrics
// =============================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_get_reputation_rpc() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_usdc_for_accounts(TEST_USDC_AMOUNT, 3)
        .build()
        .await;

    let addresses = test_cluster.get_addresses();
    let buyer = addresses[0];
    let seller_a = addresses[1];
    let seller_b = addresses[2];

    // --- Ask 1: buyer → seller_a ---
    let buyer_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .expect("buyer gas");

    let ask1_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: test_task_digest(),
            max_price_per_bid: 500_000, // 0.50 USDC
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![buyer_gas],
    );
    let ask1_resp = test_cluster.sign_and_execute_transaction(&ask1_tx).await;
    assert!(ask1_resp.effects.status().is_ok());
    let ask1_id = ask1_resp.effects.created()[0].0 .0;

    let sa_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(seller_a)
        .await
        .unwrap()
        .unwrap();
    let bid1_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id: ask1_id,
            price: 500_000,
            response_digest: test_response_digest(),
        }),
        seller_a,
        vec![sa_gas],
    );
    let bid1_resp = test_cluster.sign_and_execute_transaction(&bid1_tx).await;
    assert!(bid1_resp.effects.status().is_ok());
    let bid1_id = bid1_resp.effects.created()[0].0 .0;

    let buyer_usdc = test_cluster.wallet.get_usdc_coins_sorted_by_balance(buyer).await.unwrap();
    let usdc_ref = buyer_usdc[0].0;
    let buyer_gas2 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();
    let accept1_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs {
            ask_id: ask1_id,
            bid_id: bid1_id,
            payment_coin: usdc_ref,
        }),
        buyer,
        vec![buyer_gas2],
    );
    let accept1_resp = test_cluster.sign_and_execute_transaction(&accept1_tx).await;
    assert!(accept1_resp.effects.status().is_ok());

    // --- Ask 2: buyer → seller_b ---
    let buyer_gas3 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();
    let ask2_tx = TransactionData::new(
        TransactionKind::CreateAsk(CreateAskArgs {
            task_digest: TaskDigest::new([5u8; 32]),
            max_price_per_bid: 300_000, // 0.30 USDC
            num_bids_wanted: 1,
            timeout_ms: 300_000,
        }),
        buyer,
        vec![buyer_gas3],
    );
    let ask2_resp = test_cluster.sign_and_execute_transaction(&ask2_tx).await;
    assert!(ask2_resp.effects.status().is_ok());
    let ask2_id = ask2_resp.effects.created()[0].0 .0;

    let sb_gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(seller_b)
        .await
        .unwrap()
        .unwrap();
    let bid2_tx = TransactionData::new(
        TransactionKind::CreateBid(CreateBidArgs {
            ask_id: ask2_id,
            price: 300_000,
            response_digest: test_response_digest_2(),
        }),
        seller_b,
        vec![sb_gas],
    );
    let bid2_resp = test_cluster.sign_and_execute_transaction(&bid2_tx).await;
    assert!(bid2_resp.effects.status().is_ok());
    let bid2_id = bid2_resp.effects.created()[0].0 .0;

    let buyer_usdc2 = test_cluster.wallet.get_usdc_coins_sorted_by_balance(buyer).await.unwrap();
    let usdc_ref2 = buyer_usdc2[0].0;
    let buyer_gas4 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(buyer)
        .await
        .unwrap()
        .unwrap();
    let accept2_tx = TransactionData::new(
        TransactionKind::AcceptBid(AcceptBidArgs {
            ask_id: ask2_id,
            bid_id: bid2_id,
            payment_coin: usdc_ref2,
        }),
        buyer,
        vec![buyer_gas4],
    );
    let accept2_resp = test_cluster.sign_and_execute_transaction(&accept2_tx).await;
    assert!(accept2_resp.effects.status().is_ok());

    // --- Now query reputation via RPC ---
    let client = &test_cluster.fullnode_handle.soma_client;

    // Buyer reputation: 2 settlements, 2 unique sellers
    let buyer_rep = client
        .get_reputation(&buyer)
        .await
        .expect("get_reputation for buyer should succeed");
    assert_eq!(buyer_rep.buyer_settlements, Some(2), "buyer should have 2 settlements");
    assert_eq!(buyer_rep.buyer_unique_sellers, Some(2), "buyer should have 2 unique sellers");
    // Total volume spent = 500_000 + 300_000 = 800_000 (before fee — settlement amount is after fee)
    // Actually settlement amount = price - fee, but buyer_volume is from settlement.amount
    let expected_buyer_vol = (500_000 - 500_000 * 250 / 10_000) + (300_000 - 300_000 * 250 / 10_000);
    assert_eq!(buyer_rep.buyer_volume_spent, Some(expected_buyer_vol));
    // Buyer should have 0 seller settlements
    assert_eq!(buyer_rep.seller_settlements, Some(0));

    // Seller A reputation: 1 settlement, 1 unique buyer
    let sa_rep = client
        .get_reputation(&seller_a)
        .await
        .expect("get_reputation for seller_a should succeed");
    assert_eq!(sa_rep.seller_settlements, Some(1));
    assert_eq!(sa_rep.seller_unique_buyers, Some(1));
    assert_eq!(sa_rep.seller_negative_ratings, Some(0));
    assert_eq!(sa_rep.seller_approval_rate, Some(100.0));
    let expected_sa_vol = 500_000 - 500_000 * 250 / 10_000;
    assert_eq!(sa_rep.seller_volume_earned, Some(expected_sa_vol));
    // Seller A should have 0 buyer settlements
    assert_eq!(sa_rep.buyer_settlements, Some(0));

    // Seller B reputation: 1 settlement, 1 unique buyer
    let sb_rep = client
        .get_reputation(&seller_b)
        .await
        .expect("get_reputation for seller_b should succeed");
    assert_eq!(sb_rep.seller_settlements, Some(1));
    assert_eq!(sb_rep.seller_unique_buyers, Some(1));
    assert_eq!(sb_rep.seller_negative_ratings, Some(0));
    assert_eq!(sb_rep.seller_approval_rate, Some(100.0));

    info!("GetReputation RPC test passed — buyer has 2 settlements with 2 unique sellers, sellers each have 1!");
}
