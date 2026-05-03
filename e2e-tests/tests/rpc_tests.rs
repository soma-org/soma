// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! RPC endpoint integration tests.
//!
//! Tests:
//! 1. test_get_service_info — Query chain_id and server_version
//! 2. test_get_object — Get a genesis gas object by ID
//! 3. test_get_object_with_version — Get object at specific version after mutation
//! 4. test_get_transaction — Execute transfer, query by digest
//! 5. test_get_checkpoint — Query checkpoint 0 and latest
//! 6. test_get_epoch — Query epoch 0 and latest
//! 7. test_get_balance_and_list_owned_objects — Query balance and list objects for addresses
//!
//! Ported from Sui's RPC v2 tests.
//! Skipped: ~36 Move-specific, coin registry, and ZkLogin tests.

use futures::StreamExt;
use test_cluster::TestClusterBuilder;
use tracing::info;
use types::effects::TransactionEffectsAPI;
use types::transaction::{TransactionData, TransactionKind};
use utils::logging::init_tracing;

/// Query chain_id and server_version from the service info endpoint.
/// Both should be non-empty strings.
#[cfg(msim)]
#[msim::sim_test]
async fn test_get_service_info() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let client = &test_cluster.fullnode_handle.soma_client;

    let chain_id = client.get_chain_identifier().await.unwrap();
    assert!(!chain_id.is_empty(), "chain_id should not be empty");

    let server_version = client.get_server_version().await.unwrap();
    assert!(!server_version.is_empty(), "server_version should not be empty");

    info!("chain_id={}, server_version={}", chain_id, server_version);
}

/// Fetch a genesis shared object (the SystemState) by its ObjectID
/// via the gRPC endpoint. Stage 13a: addresses no longer own Coin
/// objects, so we use the always-present SystemState instead of a
/// gas coin to exercise the get_object path.
#[cfg(msim)]
#[msim::sim_test]
async fn test_get_object() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let client = &test_cluster.fullnode_handle.soma_client;

    let obj = client.get_object(types::SYSTEM_STATE_OBJECT_ID).await.unwrap();
    assert_eq!(obj.id(), types::SYSTEM_STATE_OBJECT_ID, "Object ID should match");
    assert!(matches!(obj.owner, types::object::Owner::Shared { .. }), "SystemState is shared");

    info!("Got SystemState object at version {}", obj.version().value());
}

/// Use AddStake (which mutates the shared SystemState) to advance an
/// object version, then query the object at the new version via RPC.
/// Stage 13c: BalanceTransfer touches no per-object versions, so we
/// exercise `get_object_with_version` against SystemState instead.
#[cfg(msim)]
#[msim::sim_test]
async fn test_get_object_with_version() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let client = &test_cluster.fullnode_handle.soma_client;

    let sender = test_cluster.wallet.get_addresses()[0];
    let validator = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().validators().validators[0]
            .metadata
            .soma_address
    });
    let chain = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_chain_identifier());
    let current_epoch = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().epoch_store_for_testing().epoch());

    let tx_data = types::transaction::TransactionData::new_with_expiration(
        types::transaction::TransactionKind::AddStake { validator, amount: 1_000 },
        sender,
        Vec::new(),
        types::transaction::TransactionExpiration::ValidDuring {
            min_epoch: Some(current_epoch),
            max_epoch: Some(current_epoch + 1),
            chain,
            nonce: 0,
        },
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    let changed = response.effects.all_changed_objects();
    let (obj_ref, _, _) = changed
        .iter()
        .next()
        .expect("AddStake must mutate SystemState (a shared object)");
    let obj = client.get_object_with_version(obj_ref.0, obj_ref.1).await.unwrap();
    assert_eq!(obj.version(), obj_ref.1, "Version should match");

    info!("Got object {} at version {}", obj_ref.0, obj_ref.1.value());
}

/// Execute a transfer, then query the transaction by its digest.
/// Verify effects, checkpoint, and timestamp are present.
#[cfg(msim)]
#[msim::sim_test]
async fn test_get_transaction() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let client = &test_cluster.fullnode_handle.soma_client;

    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    let tx_data = e2e_tests::balance_transfer_data(
        &test_cluster,
        types::object::CoinType::Usdc,
        sender,
        vec![(recipient, 1000)],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());
    let digest = *response.effects.transaction_digest();

    let tx_result = client.get_transaction(digest).await.unwrap();
    assert_eq!(tx_result.digest, digest, "Digest should match");
    assert!(tx_result.effects.status().is_ok(), "Effects should show success");
    assert!(tx_result.checkpoint.is_some(), "Checkpoint should be present");
    assert!(tx_result.timestamp_ms.is_some(), "Timestamp should be present");

    info!("Transaction {} in checkpoint {}", digest, tx_result.checkpoint.unwrap());
}

/// Query checkpoint 0 (genesis) and the latest checkpoint.
/// Verify sequence numbers and epoch fields.
#[cfg(msim)]
#[msim::sim_test]
async fn test_get_checkpoint() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    // Execute a transaction to ensure at least one non-genesis checkpoint exists
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    let tx_data = e2e_tests::balance_transfer_data(
        &test_cluster,
        types::object::CoinType::Usdc,
        sender,
        vec![(recipient, 1000)],
    );
    test_cluster.sign_and_execute_transaction(&tx_data).await;

    let client = &test_cluster.fullnode_handle.soma_client;

    // Query checkpoint 0
    let ckpt0 = client.get_checkpoint_summary(0).await.unwrap();
    assert_eq!(*ckpt0.sequence_number(), 0u64, "Sequence number should be 0");
    assert_eq!(ckpt0.epoch(), 0, "Epoch should be 0 for genesis checkpoint");

    // Query latest checkpoint
    let latest = client.get_latest_checkpoint().await.unwrap();
    assert!(
        *latest.sequence_number() >= 1u64,
        "Latest checkpoint should be >= 1 after a transaction"
    );

    info!(
        "Checkpoint 0 epoch={}, latest checkpoint seq={}",
        ckpt0.epoch(),
        latest.sequence_number()
    );
}

/// Query epoch 0 and the latest epoch. Verify epoch and protocol_version fields.
#[cfg(msim)]
#[msim::sim_test]
async fn test_get_epoch() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let client = &test_cluster.fullnode_handle.soma_client;

    // Query epoch 0
    let epoch0 = client.get_epoch(Some(0)).await.unwrap();
    let epoch_info = epoch0.epoch.as_ref().expect("epoch info should be present");
    assert_eq!(epoch_info.epoch, Some(0), "Epoch should be 0");
    let protocol_config =
        epoch_info.protocol_config.as_ref().expect("protocol_config should be present");
    assert!(protocol_config.protocol_version.is_some(), "protocol_version should be present");

    // Query latest epoch
    let latest = client.get_epoch(None).await.unwrap();
    let latest_epoch_info = latest.epoch.as_ref().expect("latest epoch info should be present");
    assert!(latest_epoch_info.epoch.is_some(), "Latest epoch should have epoch number");

    info!(
        "Epoch 0 protocol_version={:?}, latest epoch={:?}",
        protocol_config.protocol_version, latest_epoch_info.epoch
    );
}

/// Query balance for a funded address (should be > 0) and a fresh address (should be 0).
/// Also list owned objects for the funded address (should be non-empty).
#[cfg(msim)]
#[msim::sim_test]
async fn test_get_balance_and_list_owned_objects() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let client = &test_cluster.fullnode_handle.soma_client;

    let addresses = test_cluster.wallet.get_addresses();
    let funded_address = addresses[0];

    // Stage 13c: balance lives in the per-(owner, coin_type) accumulator.
    // The default `get_balance` reads USDC; check both currencies.
    let usdc = client.get_balance(&funded_address).await.unwrap();
    assert!(usdc > 0, "Funded address should have USDC > 0, got {}", usdc);

    let soma = client
        .get_balance_by_coin_type(&funded_address, types::object::CoinType::Soma)
        .await
        .unwrap();
    assert!(soma > 0, "Funded address should have SOMA > 0, got {}", soma);

    // A fresh address has no accumulator row → balance 0.
    let fresh_address = types::base::SomaAddress::random();
    let fresh_usdc = client.get_balance(&fresh_address).await.unwrap();
    assert_eq!(fresh_usdc, 0, "Fresh address should have USDC balance 0");
    let fresh_soma = client
        .get_balance_by_coin_type(&fresh_address, types::object::CoinType::Soma)
        .await
        .unwrap();
    assert_eq!(fresh_soma, 0, "Fresh address should have SOMA balance 0");

    // Stage 13a stopped materializing Coin objects, so a funded
    // address owns no objects at genesis. ListOwnedObjects should
    // return an empty stream — that's the balance-mode contract.
    let mut request = rpc::proto::soma::ListOwnedObjectsRequest::default();
    request.owner = Some(funded_address.to_string());
    let objects: Vec<_> = client
        .list_owned_objects(request)
        .await
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(
        objects.is_empty(),
        "Stage 13a: funded address owns no Coin objects (balance-mode); got {} objects",
        objects.len(),
    );

    info!("Balance for {}: USDC={}, SOMA={}", funded_address, usdc, soma);
}

/// Stage 9d: `ListDelegations` RPC reads delegations directly from the
/// post-migration `delegations` column family (instead of scanning a
/// staker's owned StakedSomaV1 objects). After at least one epoch
/// boundary, the validator's self-stake address has at least one
/// delegation row, and `total_principal` matches the sum.
#[cfg(msim)]
#[msim::sim_test]
async fn test_list_delegations() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_num_validators(4)
        .with_epoch_duration_ms(5_000)
        .build()
        .await;
    let client = &test_cluster.fullnode_handle.soma_client;

    // Pick a validator address — it'll have at least its self-stake
    // row in the delegations table (Stage 9d genesis backfill).
    let validator_address = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().unwrap().validators().validators[0]
            .metadata
            .soma_address
    });

    let request = rpc::proto::soma::ListDelegationsRequest::default()
        .with_staker(validator_address.to_string());
    let response = client.list_delegations(request).await.unwrap();

    assert!(
        !response.delegations.is_empty(),
        "validator should have at least one delegation row at genesis (got 0)",
    );

    // Server-computed total_principal must equal the client-side sum.
    // Catches a class of bugs where the server forgets to populate the
    // total or computes it wrong.
    let client_total: u64 = response
        .delegations
        .iter()
        .map(|d| d.principal.unwrap_or(0))
        .sum();
    assert_eq!(
        response.total_principal.unwrap_or(0),
        client_total,
        "server total_principal must match client-side sum of delegation principals",
    );

    // A staker who never staked has zero delegations and zero total.
    let fresh_request = rpc::proto::soma::ListDelegationsRequest::default()
        .with_staker(types::base::SomaAddress::random().to_string());
    let fresh_response = client.list_delegations(fresh_request).await.unwrap();
    assert!(fresh_response.delegations.is_empty());
    assert_eq!(fresh_response.total_principal.unwrap_or(0), 0);

    info!(
        "validator {} has {} delegation rows totaling {} shannons",
        validator_address,
        response.delegations.len(),
        response.total_principal.unwrap_or(0),
    );
}
