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
use types::{
    effects::TransactionEffectsAPI,
    transaction::{TransactionData, TransactionKind},
};
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

/// Get a genesis gas object by its ObjectID via the gRPC endpoint.
#[cfg(msim)]
#[msim::sim_test]
async fn test_get_object() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let client = &test_cluster.fullnode_handle.soma_client;

    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have a gas object");

    let obj = client.get_object(gas.0).await.unwrap();
    assert_eq!(obj.id(), gas.0, "Object ID should match");
    assert_eq!(obj.owner.get_owner_address().unwrap(), sender, "Owner should be sender");

    info!("Got object {} owned by {}", obj.id(), sender);
}

/// Transfer a coin (which mutates the gas object), then query the object
/// at the new version. Verify version matches.
#[cfg(msim)]
#[msim::sim_test]
async fn test_get_object_with_version() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let client = &test_cluster.fullnode_handle.soma_client;

    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have a gas object");

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(1000), recipient },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    // The gas object was mutated — get its new version from effects
    let changed = response.effects.all_changed_objects();
    let (gas_ref, _, _) = changed
        .iter()
        .find(|(obj_ref, _, _)| obj_ref.0 == gas.0)
        .expect("Gas object should be in changed objects");

    let obj = client.get_object_with_version(gas_ref.0, gas_ref.1).await.unwrap();
    assert_eq!(obj.version(), gas_ref.1, "Version should match");

    info!("Got object at version {}", gas_ref.1.value());
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

    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("sender must have a gas object");

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(1000), recipient },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());
    let digest = *response.effects.transaction_digest();

    let tx_result = client.get_transaction(digest).await.unwrap();
    assert_eq!(tx_result.digest, digest, "Digest should match");
    assert!(tx_result.effects.status().is_ok(), "Effects should show success");
    assert!(tx_result.checkpoint.is_some(), "Checkpoint should be present");
    assert!(tx_result.timestamp_ms.is_some(), "Timestamp should be present");

    info!(
        "Transaction {} in checkpoint {}",
        digest,
        tx_result.checkpoint.unwrap()
    );
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
    let gas = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .unwrap();

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(1000), recipient },
        sender,
        vec![gas],
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
    let protocol_config = epoch_info.protocol_config.as_ref().expect("protocol_config should be present");
    assert!(
        protocol_config.protocol_version.is_some(),
        "protocol_version should be present"
    );

    // Query latest epoch
    let latest = client.get_epoch(None).await.unwrap();
    let latest_epoch_info = latest.epoch.as_ref().expect("latest epoch info should be present");
    assert!(latest_epoch_info.epoch.is_some(), "Latest epoch should have epoch number");

    info!(
        "Epoch 0 protocol_version={:?}, latest epoch={:?}",
        protocol_config.protocol_version,
        latest_epoch_info.epoch
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

    // Query balance for a funded address
    let balance = client.get_balance(&funded_address).await.unwrap();
    assert!(balance > 0, "Funded address should have balance > 0, got {}", balance);

    // Query balance for a fresh (unfunded) address
    let fresh_address = types::base::SomaAddress::random();
    let fresh_balance = client.get_balance(&fresh_address).await.unwrap();
    assert_eq!(fresh_balance, 0, "Fresh address should have balance 0");

    // List owned objects for funded address (no type filter — lists all types)
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

    assert!(!objects.is_empty(), "Funded address should own at least one object");

    // List with explicit Coin type filter
    let mut coin_request = rpc::proto::soma::ListOwnedObjectsRequest::default();
    coin_request.owner = Some(funded_address.to_string());
    coin_request.object_type = Some("Coin".to_string());
    let coins: Vec<_> = client
        .list_owned_objects(coin_request)
        .await
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert!(!coins.is_empty(), "Funded address should own at least one Coin");
    assert_eq!(objects.len(), coins.len(), "All objects for a funded address should be Coins");

    info!(
        "Balance for {} = {}, objects owned = {}",
        funded_address,
        balance,
        objects.len()
    );
}
