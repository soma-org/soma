// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::base::ExecutionDigests;
use crate::checkpoints::{CheckpointContents, CheckpointSummary};
use crate::tx_fee::TransactionFee;

/// Helper to create a CheckpointContents with the given number of random execution digests.
fn make_checkpoint_contents(num_txns: usize) -> CheckpointContents {
    let digests: Vec<ExecutionDigests> =
        (0..num_txns).map(|_| ExecutionDigests::random()).collect();
    CheckpointContents::new_with_digests_only_for_tests(digests)
}

/// Helper to create a CheckpointSummary with reasonable defaults.
fn make_checkpoint_summary(
    epoch: u64,
    seq: u64,
    contents: &CheckpointContents,
) -> CheckpointSummary {
    CheckpointSummary::new(
        epoch,
        seq,
        /* network_total_transactions */ seq + contents.size() as u64,
        contents,
        /* previous_digest */ None,
        TransactionFee::default(),
        /* end_of_epoch_data */ None,
        /* timestamp_ms */ 1_000_000,
        /* checkpoint_commitments */ vec![],
    )
}

#[test]
fn test_checkpoint_summary_bcs_roundtrip() {
    let contents = make_checkpoint_contents(3);
    let summary = make_checkpoint_summary(0, 0, &contents);

    let bytes = bcs::to_bytes(&summary).expect("BCS serialize should succeed");
    let deserialized: CheckpointSummary =
        bcs::from_bytes(&bytes).expect("BCS deserialize should succeed");

    assert_eq!(summary.epoch, deserialized.epoch);
    assert_eq!(summary.sequence_number, deserialized.sequence_number);
    assert_eq!(summary.network_total_transactions, deserialized.network_total_transactions);
    assert_eq!(summary.content_digest, deserialized.content_digest);
    assert_eq!(summary.previous_digest, deserialized.previous_digest);
    assert_eq!(summary.timestamp_ms, deserialized.timestamp_ms);
}

#[test]
fn test_checkpoint_contents_bcs_roundtrip() {
    let contents = make_checkpoint_contents(5);
    let bytes = bcs::to_bytes(&contents).expect("BCS serialize should succeed");
    let deserialized: CheckpointContents =
        bcs::from_bytes(&bytes).expect("BCS deserialize should succeed");

    assert_eq!(contents.size(), deserialized.size());
    // Verify individual execution digests match
    let original: Vec<_> = contents.iter().cloned().collect();
    let restored: Vec<_> = deserialized.iter().cloned().collect();
    assert_eq!(original, restored);
}

#[test]
fn test_checkpoint_sequence_ordering() {
    let contents = make_checkpoint_contents(1);
    let s1 = make_checkpoint_summary(0, 1, &contents);
    let s2 = make_checkpoint_summary(0, 5, &contents);
    let s3 = make_checkpoint_summary(0, 10, &contents);

    assert!(s1.sequence_number < s2.sequence_number, "Sequence numbers should be orderable");
    assert!(s2.sequence_number < s3.sequence_number, "Sequence numbers should be orderable");
    assert_eq!(*s1.sequence_number(), 1);
    assert_eq!(*s2.sequence_number(), 5);
    assert_eq!(*s3.sequence_number(), 10);
}

#[test]
fn test_checkpoint_contents_digest() {
    let contents = make_checkpoint_contents(4);

    let digest1 = *contents.digest();
    let digest2 = *contents.digest();
    assert_eq!(digest1, digest2, "Contents digest should be deterministic across calls");

    // Also verify compute_digest matches the cached digest
    let computed = contents.compute_digest().expect("compute_digest should succeed");
    assert_eq!(digest1, computed, "Cached digest and computed digest should match");
}

#[test]
fn test_checkpoint_summary_fields() {
    let contents = make_checkpoint_contents(2);
    let epoch = 42u64;
    let seq = 100u64;
    let timestamp = 999_999u64;
    let network_total = seq + contents.size() as u64;

    let summary = CheckpointSummary::new(
        epoch,
        seq,
        network_total,
        &contents,
        None,
        TransactionFee::default(),
        None,
        timestamp,
        vec![],
    );

    assert_eq!(summary.epoch, epoch);
    assert_eq!(summary.sequence_number, seq);
    assert_eq!(summary.network_total_transactions, network_total);
    assert_eq!(summary.content_digest, *contents.digest());
    assert_eq!(summary.previous_digest, None);
    assert_eq!(summary.timestamp_ms, timestamp);
    assert!(summary.end_of_epoch_data.is_none());
    assert!(!summary.is_last_checkpoint_of_epoch());
}

#[test]
fn test_checkpoint_different_content_different_digest() {
    let contents_a = make_checkpoint_contents(3);
    let contents_b = make_checkpoint_contents(3);

    // Two separately generated random contents should produce different digests
    // (probability of collision is negligible)
    let digest_a = *contents_a.digest();
    let digest_b = *contents_b.digest();
    assert_ne!(digest_a, digest_b, "Different contents should produce different digests");

    // Also verify empty contents have a consistent digest
    let empty1 = make_checkpoint_contents(0);
    let empty2 = make_checkpoint_contents(0);
    let empty_digest1 = *empty1.digest();
    let empty_digest2 = *empty2.digest();
    assert_eq!(empty_digest1, empty_digest2, "Two empty contents should produce the same digest");
}
