// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! gRPC integration tests for the archival KvRpcServer.
//!
//! All tests are `#[ignore]` and require the BigTable emulator (`cbtemulator` + `cbt`).
//! Run with:
//!   cargo test -p indexer-kvstore --test grpc_tests -- --ignored

mod emulator;

use std::sync::Arc;

use indexer_framework::pipeline::Processor;
use indexer_kvstore::kv_rpc::KvRpcServer;
use indexer_kvstore::{
    BigTableClient, CheckpointsByDigestPipeline, CheckpointsPipeline, EpochStartPipeline,
    ObjectsPipeline, TransactionsPipeline, Watermark,
};
use rpc::proto::soma::get_checkpoint_request::CheckpointId;
use rpc::proto::soma::ledger_service_client::LedgerServiceClient;
use rpc::proto::soma::{
    GetCheckpointRequest, GetEpochRequest, GetObjectRequest, GetServiceInfoRequest,
    GetTransactionRequest,
};
use types::base::SomaAddress;
use types::full_checkpoint_content::Checkpoint;
use types::test_checkpoint_data_builder::{
    TestCheckpointBuilder, default_test_system_state,
};

/// Build a genesis checkpoint with system state.
fn genesis_checkpoint() -> Checkpoint {
    TestCheckpointBuilder::new(0)
        .with_genesis_system_state(default_test_system_state())
        .build()
}

/// Build a transfer checkpoint at sequence_number=1.
fn transfer_checkpoint() -> Checkpoint {
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    TestCheckpointBuilder::new(1).add_transfer_coin(sender, recipient, 1000).build()
}

/// Seed the emulator with genesis + transfer checkpoints and set watermarks.
/// Returns (genesis_checkpoint, transfer_checkpoint).
async fn seed_data(client: &mut BigTableClient) -> (Checkpoint, Checkpoint) {
    let cp0 = genesis_checkpoint();
    let cp1 = transfer_checkpoint();

    // Write checkpoints
    for cp in [&cp0, &cp1] {
        let arc = Arc::new(cp.clone());
        let entries = CheckpointsPipeline.process(&arc).await.unwrap();
        client.write_entries(indexer_kvstore::tables::checkpoints::NAME, entries).await.unwrap();
        let entries = CheckpointsByDigestPipeline.process(&arc).await.unwrap();
        client
            .write_entries(indexer_kvstore::tables::checkpoints_by_digest::NAME, entries)
            .await
            .unwrap();
        let entries = TransactionsPipeline.process(&arc).await.unwrap();
        client.write_entries(indexer_kvstore::tables::transactions::NAME, entries).await.unwrap();
        let entries = ObjectsPipeline.process(&arc).await.unwrap();
        client.write_entries(indexer_kvstore::tables::objects::NAME, entries).await.unwrap();
    }

    // Write epoch start for genesis
    let entries = EpochStartPipeline.process(&Arc::new(cp0.clone())).await.unwrap();
    client.write_entries(indexer_kvstore::tables::epochs::NAME, entries).await.unwrap();

    // Set watermarks
    let wm = Watermark {
        epoch_hi_inclusive: 0,
        checkpoint_hi_inclusive: 1,
        tx_hi: 2,
        timestamp_ms_hi_inclusive: cp1.summary.timestamp_ms,
    };
    for name in indexer_kvstore::ALL_PIPELINE_NAMES {
        client.set_pipeline_watermark(name, &wm).await.unwrap();
    }

    (cp0, cp1)
}

/// Start a KvRpcServer on a random port and return a connected LedgerServiceClient.
async fn start_server(
    client: BigTableClient,
) -> LedgerServiceClient<rpc_tonic::transport::Channel> {
    let server = KvRpcServer::new(client).with_chain_id("test-chain".to_string());

    let router = server.into_router().await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let channel = rpc_tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    LedgerServiceClient::new(channel)
}

// ---------- GetServiceInfo ----------

#[tokio::test]
#[ignore]
async fn test_get_service_info() {
    let emu = emulator::BigTableEmulator::start().unwrap();
    emulator::create_tables(emu.host(), emulator::INSTANCE_ID).await.unwrap();
    let mut client = emulator::client(emu.host()).await.unwrap();

    let (_cp0, _cp1) = seed_data(&mut client).await;

    let server_client = emulator::client(emu.host()).await.unwrap();
    let mut grpc_client = start_server(server_client).await;

    let response =
        grpc_client.get_service_info(GetServiceInfoRequest::default()).await.unwrap().into_inner();

    assert_eq!(response.chain_id.as_deref(), Some("test-chain"));
    assert_eq!(response.checkpoint_height, Some(1));
    assert_eq!(response.epoch, Some(0));
}

// ---------- GetCheckpoint by sequence number ----------

#[tokio::test]
#[ignore]
async fn test_get_checkpoint_by_seq() {
    let emu = emulator::BigTableEmulator::start().unwrap();
    emulator::create_tables(emu.host(), emulator::INSTANCE_ID).await.unwrap();
    let mut client = emulator::client(emu.host()).await.unwrap();

    let (cp0, _cp1) = seed_data(&mut client).await;

    let server_client = emulator::client(emu.host()).await.unwrap();
    let mut grpc_client = start_server(server_client).await;

    let response = grpc_client
        .get_checkpoint({
            let mut r = GetCheckpointRequest::default();
            r.checkpoint_id = Some(CheckpointId::SequenceNumber(0));
            r
        })
        .await
        .unwrap()
        .into_inner();

    let checkpoint = response.checkpoint.expect("checkpoint should be present");
    assert_eq!(checkpoint.sequence_number, Some(0));
    assert_eq!(checkpoint.digest.as_deref(), Some(cp0.summary.digest().to_string().as_str()));
}

// ---------- GetCheckpoint by digest ----------

#[tokio::test]
#[ignore]
async fn test_get_checkpoint_by_digest() {
    let emu = emulator::BigTableEmulator::start().unwrap();
    emulator::create_tables(emu.host(), emulator::INSTANCE_ID).await.unwrap();
    let mut client = emulator::client(emu.host()).await.unwrap();

    let (cp0, _cp1) = seed_data(&mut client).await;

    let server_client = emulator::client(emu.host()).await.unwrap();
    let mut grpc_client = start_server(server_client).await;

    let response = grpc_client
        .get_checkpoint({
            let mut r = GetCheckpointRequest::default();
            r.checkpoint_id = Some(CheckpointId::Digest(cp0.summary.digest().to_string()));
            r
        })
        .await
        .unwrap()
        .into_inner();

    let checkpoint = response.checkpoint.expect("checkpoint should be present");
    assert_eq!(checkpoint.sequence_number, Some(0));
}

// ---------- GetTransaction ----------

#[tokio::test]
#[ignore]
async fn test_get_transaction() {
    let emu = emulator::BigTableEmulator::start().unwrap();
    emulator::create_tables(emu.host(), emulator::INSTANCE_ID).await.unwrap();
    let mut client = emulator::client(emu.host()).await.unwrap();

    let (_cp0, cp1) = seed_data(&mut client).await;

    let server_client = emulator::client(emu.host()).await.unwrap();
    let mut grpc_client = start_server(server_client).await;

    let tx_digest = cp1.contents.iter().next().unwrap().transaction;

    let mut request = GetTransactionRequest::default();
    request.digest = Some(tx_digest.to_string());

    let response = grpc_client.get_transaction(request).await.unwrap().into_inner();

    let transaction = response.transaction.expect("transaction should be present");
    assert_eq!(transaction.digest.as_deref(), Some(tx_digest.to_string().as_str()));
}

// ---------- GetObject ----------

#[tokio::test]
#[ignore]
async fn test_get_object() {
    let emu = emulator::BigTableEmulator::start().unwrap();
    emulator::create_tables(emu.host(), emulator::INSTANCE_ID).await.unwrap();
    let mut client = emulator::client(emu.host()).await.unwrap();

    let (_cp0, cp1) = seed_data(&mut client).await;

    let server_client = emulator::client(emu.host()).await.unwrap();
    let mut grpc_client = start_server(server_client).await;

    // Get an output object from the transfer checkpoint
    let first_tx = &cp1.transactions[0];
    let output_objects: Vec<_> = first_tx.output_objects(&cp1.object_set).collect();
    let obj = &output_objects[0];

    let mut request = GetObjectRequest::default();
    request.object_id = Some(obj.id().to_string());
    request.version = Some(obj.version().value());

    let response = grpc_client.get_object(request).await.unwrap().into_inner();

    let object = response.object.expect("object should be present");
    assert_eq!(object.object_id.as_deref(), Some(obj.id().to_string().as_str()));
}

// ---------- GetEpoch ----------

#[tokio::test]
#[ignore]
async fn test_get_epoch() {
    let emu = emulator::BigTableEmulator::start().unwrap();
    emulator::create_tables(emu.host(), emulator::INSTANCE_ID).await.unwrap();
    let mut client = emulator::client(emu.host()).await.unwrap();

    let (_cp0, _cp1) = seed_data(&mut client).await;

    let server_client = emulator::client(emu.host()).await.unwrap();
    let mut grpc_client = start_server(server_client).await;

    let mut request = GetEpochRequest::default();
    request.epoch = Some(0);

    let response = grpc_client.get_epoch(request).await.unwrap().into_inner();

    let epoch = response.epoch.expect("epoch should be present");
    assert_eq!(epoch.epoch, Some(0));
}

// ---------- Not-found errors ----------

#[tokio::test]
#[ignore]
async fn test_not_found_checkpoint() {
    let emu = emulator::BigTableEmulator::start().unwrap();
    emulator::create_tables(emu.host(), emulator::INSTANCE_ID).await.unwrap();
    let mut client = emulator::client(emu.host()).await.unwrap();
    seed_data(&mut client).await;

    let server_client = emulator::client(emu.host()).await.unwrap();
    let mut grpc_client = start_server(server_client).await;

    let err = grpc_client
        .get_checkpoint({
            let mut r = GetCheckpointRequest::default();
            r.checkpoint_id = Some(CheckpointId::SequenceNumber(9999));
            r
        })
        .await
        .unwrap_err();

    assert_eq!(err.code(), rpc_tonic::Code::NotFound);
}

#[tokio::test]
#[ignore]
async fn test_not_found_transaction() {
    let emu = emulator::BigTableEmulator::start().unwrap();
    emulator::create_tables(emu.host(), emulator::INSTANCE_ID).await.unwrap();
    let mut client = emulator::client(emu.host()).await.unwrap();
    seed_data(&mut client).await;

    let server_client = emulator::client(emu.host()).await.unwrap();
    let mut grpc_client = start_server(server_client).await;

    let fake_digest = types::digests::TransactionDigest::random();
    let mut request = GetTransactionRequest::default();
    request.digest = Some(fake_digest.to_string());

    let err = grpc_client.get_transaction(request).await.unwrap_err();

    assert_eq!(err.code(), rpc_tonic::Code::NotFound);
}
