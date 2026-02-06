//! Unit tests for submission types.
//!
//! Tests for:
//! - Submission construction and field access
//! - SubmissionManifest creation and size calculation
//! - DataCommitment type behavior
//! - Transaction type helpers for submissions

use crate::{
    base::SomaAddress,
    checksum::Checksum,
    crypto::DIGEST_LENGTH,
    digests::DataCommitment,
    metadata::{Manifest, ManifestV1, Metadata, MetadataV1},
    model::ModelId,
    object::ObjectID,
    submission::{Submission, SubmissionManifest},
    target::Embedding,
    transaction::{ClaimRewardsArgs, SubmitDataArgs, TransactionKind},
};
use ndarray::Array1;
use url::Url;

/// Helper to create a test SubmissionManifest
fn test_submission_manifest(size: usize) -> SubmissionManifest {
    let url = Url::parse("https://example.com/data/test.bin").unwrap();
    let metadata =
        Metadata::V1(MetadataV1::new(Checksum::new_from_hash([0u8; DIGEST_LENGTH]), size));
    let manifest = Manifest::V1(ManifestV1::new(url, metadata));
    SubmissionManifest::new(manifest)
}

/// Test SubmissionManifest creation and size accessor
#[test]
fn test_submission_manifest_creation() {
    let manifest = test_submission_manifest(2048);
    assert_eq!(manifest.size(), 2048);
}

/// Test SubmissionManifest with different sizes
#[test]
fn test_submission_manifest_various_sizes() {
    // Small file
    let small = test_submission_manifest(100);
    assert_eq!(small.size(), 100);

    // Large file (1 MB)
    let large = test_submission_manifest(1_000_000);
    assert_eq!(large.size(), 1_000_000);

    // Zero-size (edge case)
    let zero = test_submission_manifest(0);
    assert_eq!(zero.size(), 0);
}

/// Test Submission construction with all fields
#[test]
fn test_submission_new() {
    let miner = SomaAddress::random();
    let data_commitment = DataCommitment::random();
    let data_manifest = test_submission_manifest(1024);
    let model_id = ModelId::random();
    let embedding: Embedding = Array1::from_vec(vec![100, 200, 300, 400, 500]);

    let submission = Submission::new(
        miner,
        data_commitment,
        data_manifest.clone(),
        model_id,
        embedding.clone(),
        500,   // distance_score
        100,   // reconstruction_score
        10240, // bond_amount
        5,     // submit_epoch
    );

    assert_eq!(submission.miner, miner);
    assert_eq!(submission.data_commitment, data_commitment);
    assert_eq!(submission.model_id, model_id);
    assert_eq!(submission.distance_score, 500);
    assert_eq!(submission.reconstruction_score, 100);
    assert_eq!(submission.bond_amount, 10240);
    assert_eq!(submission.submit_epoch, 5);
    assert_eq!(submission.embedding, embedding);
}

/// Test Submission embedding storage
#[test]
fn test_submission_embedding() {
    let miner = SomaAddress::random();
    let data_commitment = DataCommitment::random();
    let data_manifest = test_submission_manifest(512);
    let model_id = ModelId::random();

    // Create a 768-dim embedding (typical for transformer models)
    let embedding: Embedding = Array1::from_vec((0..768).map(|i| i as i64 * 1000).collect());

    let submission = Submission::new(
        miner,
        data_commitment,
        data_manifest,
        model_id,
        embedding.clone(),
        1000,
        500,
        5120,
        1,
    );

    assert_eq!(submission.embedding.len(), 768);
    assert_eq!(submission.embedding[0], 0);
    assert_eq!(submission.embedding[767], 767 * 1000);
}

/// Test DataCommitment creation and comparison
#[test]
fn test_data_commitment_random() {
    let c1 = DataCommitment::random();
    let c2 = DataCommitment::random();

    // Two random commitments should be different
    assert_ne!(c1, c2);

    // Same commitment should equal itself
    assert_eq!(c1, c1);
}

/// Test DataCommitment from bytes
#[test]
fn test_data_commitment_from_bytes() {
    let bytes = [0xABu8; 32];
    let commitment = DataCommitment::new(bytes);

    // Should be able to get inner bytes back
    let inner: [u8; 32] = commitment.into_inner();
    assert_eq!(inner, bytes);
}

/// Test SubmitData transaction kind helper
#[test]
fn test_submit_data_transaction_kind() {
    let target_id = ObjectID::random();
    let model_id = ModelId::random();
    let data_commitment = DataCommitment::random();
    let data_manifest = test_submission_manifest(1024);
    let embedding: Embedding = Array1::zeros(768);
    let bond_coin =
        (ObjectID::random(), crate::object::Version::new(), crate::digests::ObjectDigest::random());

    let args = SubmitDataArgs {
        target_id,
        data_commitment,
        data_manifest,
        model_id,
        embedding,
        distance_score: 1000,
        reconstruction_score: 500,
        bond_coin,
    };

    let kind = TransactionKind::SubmitData(args);

    // Test is_submission_tx helper
    assert!(kind.is_submission_tx());
    assert!(kind.requires_system_state());
}

/// Test ClaimRewards transaction kind helper
#[test]
fn test_claim_rewards_transaction_kind() {
    let target_id = ObjectID::random();

    let args = ClaimRewardsArgs { target_id };
    let kind = TransactionKind::ClaimRewards(args);

    // Test is_submission_tx helper
    assert!(kind.is_submission_tx());
    assert!(kind.requires_system_state());
}

/// Test that non-submission transactions don't match is_submission_tx
#[test]
fn test_non_submission_transactions() {
    // TransferCoin is not a submission tx
    let transfer = TransactionKind::TransferCoin {
        coin: (
            ObjectID::random(),
            crate::object::Version::new(),
            crate::digests::ObjectDigest::random(),
        ),
        recipient: SomaAddress::random(),
        amount: Some(1000),
    };
    assert!(!transfer.is_submission_tx());

    // AddStake is not a submission tx
    let stake = TransactionKind::AddStake {
        address: SomaAddress::random(),
        coin_ref: (
            ObjectID::random(),
            crate::object::Version::new(),
            crate::digests::ObjectDigest::random(),
        ),
        amount: None,
    };
    assert!(!stake.is_submission_tx());
}

/// Test bond calculation from data size
#[test]
fn test_bond_scales_with_data_size() {
    // This tests the logic that bond = bond_per_byte * data_size
    // Using typical values from protocol config
    let bond_per_byte: u64 = 10; // 10 shannons per byte

    // Small data: 1KB
    let small_manifest = test_submission_manifest(1024);
    let small_bond = bond_per_byte * small_manifest.size() as u64;
    assert_eq!(small_bond, 10240); // 10 * 1024

    // Large data: 1MB
    let large_manifest = test_submission_manifest(1_000_000);
    let large_bond = bond_per_byte * large_manifest.size() as u64;
    assert_eq!(large_bond, 10_000_000); // 10 * 1M
}

/// Test submission scores are correctly stored
#[test]
fn test_submission_scores() {
    let miner = SomaAddress::random();
    let data_commitment = DataCommitment::random();
    let data_manifest = test_submission_manifest(512);
    let model_id = ModelId::random();
    let embedding: Embedding = Array1::zeros(10);

    // Test with various distance scores (can be negative)
    let submission_negative = Submission::new(
        miner,
        data_commitment,
        data_manifest.clone(),
        model_id,
        embedding.clone(),
        -500, // negative distance score
        100,
        5000,
        1,
    );
    assert_eq!(submission_negative.distance_score, -500);

    // Test with zero scores
    let submission_zero = Submission::new(
        miner,
        DataCommitment::random(),
        data_manifest.clone(),
        model_id,
        embedding.clone(),
        0,
        0,
        5000,
        1,
    );
    assert_eq!(submission_zero.distance_score, 0);
    assert_eq!(submission_zero.reconstruction_score, 0);

    // Test with large scores
    let submission_large = Submission::new(
        miner,
        DataCommitment::random(),
        data_manifest,
        model_id,
        embedding,
        i64::MAX / 2,
        u64::MAX / 2,
        5000,
        1,
    );
    assert_eq!(submission_large.distance_score, i64::MAX / 2);
    assert_eq!(submission_large.reconstruction_score, u64::MAX / 2);
}

/// Test submission serialization round-trip
#[test]
fn test_submission_serialization() {
    let miner = SomaAddress::random();
    let data_commitment = DataCommitment::random();
    let data_manifest = test_submission_manifest(1024);
    let model_id = ModelId::random();
    let embedding: Embedding = Array1::from_vec(vec![100, -200, 300, -400, 500]);

    let submission = Submission::new(
        miner,
        data_commitment,
        data_manifest,
        model_id,
        embedding,
        1000,
        500,
        10240,
        5,
    );

    // Serialize and deserialize
    let bytes = bcs::to_bytes(&submission).expect("Serialization should succeed");
    let deserialized: Submission = bcs::from_bytes(&bytes).expect("Deserialization should succeed");

    assert_eq!(submission.miner, deserialized.miner);
    assert_eq!(submission.data_commitment, deserialized.data_commitment);
    assert_eq!(submission.model_id, deserialized.model_id);
    assert_eq!(submission.distance_score, deserialized.distance_score);
    assert_eq!(submission.reconstruction_score, deserialized.reconstruction_score);
    assert_eq!(submission.bond_amount, deserialized.bond_amount);
    assert_eq!(submission.submit_epoch, deserialized.submit_epoch);
    assert_eq!(submission.embedding, deserialized.embedding);
}
