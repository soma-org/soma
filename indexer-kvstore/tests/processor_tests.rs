// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for each pipeline's `Processor::process()` method.
//!
//! These tests do NOT require BigTable — they verify that the processor
//! correctly extracts entries from synthetic checkpoints built with
//! `TestCheckpointBuilder`.

use std::sync::Arc;

use indexer_framework::pipeline::Processor;
use indexer_kvstore::{
    CheckpointsByDigestPipeline, CheckpointsPipeline, EpochEndPipeline, EpochStartPipeline,
    ObjectsPipeline, TransactionsPipeline,
};
use types::base::SomaAddress;
use types::committee::Committee;
use types::test_checkpoint_data_builder::{
    TestCheckpointBuilder, default_test_system_state,
};

#[tokio::test]
async fn test_checkpoints_pipeline_process() {
    let checkpoint = TestCheckpointBuilder::new(1).build();
    let entries = CheckpointsPipeline.process(&Arc::new(checkpoint)).await.unwrap();
    assert_eq!(entries.len(), 1, "should produce exactly 1 checkpoint entry");
}

#[tokio::test]
async fn test_checkpoints_by_digest_process() {
    let checkpoint = TestCheckpointBuilder::new(1).build();
    let entries = CheckpointsByDigestPipeline.process(&Arc::new(checkpoint)).await.unwrap();
    assert_eq!(entries.len(), 1, "should produce exactly 1 checkpoint-by-digest entry");
}

#[tokio::test]
async fn test_transactions_process() {
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let checkpoint =
        TestCheckpointBuilder::new(1).add_transfer_coin(sender, recipient, 1000).build();
    let entries = TransactionsPipeline.process(&Arc::new(checkpoint)).await.unwrap();
    assert_eq!(entries.len(), 1, "should produce 1 transaction entry for 1 transfer");
}

#[tokio::test]
async fn test_objects_process() {
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let checkpoint =
        TestCheckpointBuilder::new(1).add_transfer_coin(sender, recipient, 1000).build();
    let entries = ObjectsPipeline.process(&Arc::new(checkpoint)).await.unwrap();
    // A transfer produces a gas object + a coin object = 2 output objects.
    assert!(entries.len() >= 2, "should produce entries for output objects, got {}", entries.len());
}

#[tokio::test]
async fn test_epoch_start_genesis() {
    let checkpoint = TestCheckpointBuilder::new(0)
        .with_genesis_system_state(default_test_system_state())
        .build();
    let entries = EpochStartPipeline.process(&Arc::new(checkpoint)).await.unwrap();
    assert_eq!(entries.len(), 1, "genesis checkpoint should produce 1 epoch-start entry");
}

#[tokio::test]
async fn test_epoch_start_non_epoch() {
    // A non-genesis, non-end-of-epoch checkpoint should produce no epoch-start entries.
    let checkpoint = TestCheckpointBuilder::new(5).build();
    let entries = EpochStartPipeline.process(&Arc::new(checkpoint)).await.unwrap();
    assert_eq!(entries.len(), 0, "non-epoch checkpoint should produce no epoch-start entries");
}

#[tokio::test]
async fn test_epoch_end_with_boundary() {
    let (next_committee, _) = Committee::new_simple_test_committee_of_size(4);
    let system_state = default_test_system_state();
    let checkpoint = TestCheckpointBuilder::new(10)
        .with_end_of_epoch(next_committee)
        .add_change_epoch(system_state)
        .build();
    let entries = EpochEndPipeline.process(&Arc::new(checkpoint)).await.unwrap();
    assert_eq!(entries.len(), 1, "end-of-epoch checkpoint should produce 1 epoch-end entry");
}

#[tokio::test]
async fn test_epoch_end_without_boundary() {
    let checkpoint = TestCheckpointBuilder::new(5).build();
    let entries = EpochEndPipeline.process(&Arc::new(checkpoint)).await.unwrap();
    assert_eq!(entries.len(), 0, "normal checkpoint should produce no epoch-end entries");
}

