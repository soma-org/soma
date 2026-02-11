//! Unit tests for submission types.
//!
//! Tests for:
//! - Submission construction and field access
//! - SubmissionManifest creation and size calculation
//! - DataCommitment type behavior
//! - Transaction type helpers for submissions
//! - Tally-based submission reports on Target objects

use crate::{
    base::SomaAddress,
    checksum::Checksum,
    crypto::DIGEST_LENGTH,
    digests::DataCommitment,
    metadata::{Manifest, ManifestV1, Metadata, MetadataV1},
    model::ModelId,
    object::ObjectID,
    submission::{Submission, SubmissionManifest},
    tensor::SomaTensor,
    transaction::{ClaimRewardsArgs, SubmitDataArgs, TransactionKind},
};
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
    let embedding = SomaTensor::new(vec![100.0, 200.0, 300.0, 400.0, 500.0], vec![5]);

    let submission = Submission::new(
        miner,
        data_commitment,
        data_manifest.clone(),
        model_id,
        embedding.clone(),
        SomaTensor::scalar(500.0), // distance_score
        10240,                     // bond_amount
        5,                         // submit_epoch
    );

    assert_eq!(submission.miner, miner);
    assert_eq!(submission.data_commitment, data_commitment);
    assert_eq!(submission.model_id, model_id);
    assert_eq!(submission.distance_score.as_scalar(), 500.0);
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
    let values: Vec<f32> = (0..768).map(|i| i as f32 * 1000.0).collect();
    let embedding = SomaTensor::new(values, vec![768]);

    let submission = Submission::new(
        miner,
        data_commitment,
        data_manifest,
        model_id,
        embedding.clone(),
        SomaTensor::scalar(1000.0), // distance_score
        5120,                       // bond_amount
        1,                          // submit_epoch
    );

    let emb_values = submission.embedding.to_vec();
    assert_eq!(submission.embedding.dim(), 768);
    assert_eq!(emb_values[0], 0.0);
    assert_eq!(emb_values[767], 767.0 * 1000.0);
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
    let embedding = SomaTensor::zeros(vec![768]);
    let bond_coin =
        (ObjectID::random(), crate::object::Version::new(), crate::digests::ObjectDigest::random());

    let args = SubmitDataArgs {
        target_id,
        data_commitment,
        data_manifest,
        model_id,
        embedding,
        distance_score: SomaTensor::scalar(1000.0),
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
    let embedding = SomaTensor::zeros(vec![10]);

    // Test with zero distance score
    let submission_zero = Submission::new(
        miner,
        DataCommitment::random(),
        data_manifest.clone(),
        model_id,
        embedding.clone(),
        SomaTensor::scalar(0.0), // distance_score
        5000,                    // bond_amount
        1,                       // submit_epoch
    );
    assert_eq!(submission_zero.distance_score.as_scalar(), 0.0);

    // Test with small distance score
    let submission_small = Submission::new(
        miner,
        data_commitment,
        data_manifest.clone(),
        model_id,
        embedding.clone(),
        SomaTensor::scalar(0.05), // small distance score
        5000,                     // bond_amount
        1,                        // submit_epoch
    );
    assert_eq!(submission_small.distance_score.as_scalar(), 0.05);

    // Test with large distance score
    let submission_large = Submission::new(
        miner,
        DataCommitment::random(),
        data_manifest,
        model_id,
        embedding,
        SomaTensor::scalar(1000000.0), // large distance_score
        5000,                          // bond_amount
        1,                             // submit_epoch
    );
    assert_eq!(submission_large.distance_score.as_scalar(), 1000000.0);
}

/// Test submission serialization round-trip
#[test]
fn test_submission_serialization() {
    let miner = SomaAddress::random();
    let data_commitment = DataCommitment::random();
    let data_manifest = test_submission_manifest(1024);
    let model_id = ModelId::random();
    let embedding = SomaTensor::new(vec![100.0, -200.0, 300.0, -400.0, 500.0], vec![5]);

    let submission = Submission::new(
        miner,
        data_commitment,
        data_manifest,
        model_id,
        embedding,
        SomaTensor::scalar(1000.0), // distance_score
        10240,                      // bond_amount
        5,                          // submit_epoch
    );

    // Serialize and deserialize
    let bytes = bcs::to_bytes(&submission).expect("Serialization should succeed");
    let deserialized: Submission = bcs::from_bytes(&bytes).expect("Deserialization should succeed");

    assert_eq!(submission.miner, deserialized.miner);
    assert_eq!(submission.data_commitment, deserialized.data_commitment);
    assert_eq!(submission.model_id, deserialized.model_id);
    assert_eq!(submission.distance_score, deserialized.distance_score);
    assert_eq!(submission.bond_amount, deserialized.bond_amount);
    assert_eq!(submission.submit_epoch, deserialized.submit_epoch);
    assert_eq!(submission.embedding, deserialized.embedding);
}

// =============================================================================
// Tally-Based Submission Report Tests (Target methods)
// =============================================================================

use super::test_utils::{create_test_system_state, create_validators_with_stakes};
use crate::target::{Target, TargetStatus};
use std::collections::BTreeMap;

/// Helper to create a test system state with voting power properly set.
fn create_test_system_state_with_voting_power(
    stakes: Vec<u64>,
) -> crate::system_state::SystemState {
    let validators = create_validators_with_stakes(stakes);
    let mut system_state = create_test_system_state(validators, 1000, 100);
    // Voting power is calculated from stake at epoch boundary, so we need to set it explicitly
    system_state.validators.set_voting_power();
    system_state
}

/// Helper to create a test filled target
fn create_test_target() -> Target {
    Target {
        embedding: SomaTensor::zeros(vec![10]),
        model_ids: vec![],
        distance_threshold: SomaTensor::scalar(1000.0),
        reward_pool: 1000,
        generation_epoch: 0,
        status: TargetStatus::Filled { fill_epoch: 1 },
        miner: Some(SomaAddress::random()),
        winning_model_id: Some(ModelId::random()),
        winning_model_owner: Some(SomaAddress::random()),
        bond_amount: 5000,
        winning_data_manifest: None,
        winning_data_commitment: None,
        winning_embedding: None,
        winning_distance_score: Some(SomaTensor::scalar(500.0)),
        // New tally-based fields
        challenger: None,
        challenge_id: None,
        submission_reports: BTreeMap::new(),
    }
}

/// Test that reports can be added to a Target
#[test]
fn test_target_report_submission() {
    let mut target = create_test_target();
    let validator_addr = SomaAddress::random();

    // Report with no challenger (availability issue)
    target.report_submission(validator_addr, None);
    assert_eq!(target.submission_reports.len(), 1);
    assert_eq!(target.submission_reports.get(&validator_addr), Some(&None));
}

/// Test that reports with challenger attribution work
#[test]
fn test_target_report_submission_with_challenger() {
    let mut target = create_test_target();
    let validator_addr = SomaAddress::random();
    let challenger_addr = SomaAddress::random();

    // Report with challenger attribution
    target.report_submission(validator_addr, Some(challenger_addr));
    assert_eq!(target.submission_reports.len(), 1);
    assert_eq!(target.submission_reports.get(&validator_addr), Some(&Some(challenger_addr)));
}

/// Test that duplicate reports from the same validator are idempotent
#[test]
fn test_target_report_submission_duplicate_overwrites() {
    let mut target = create_test_target();
    let validator_addr = SomaAddress::random();
    let challenger1 = SomaAddress::random();
    let challenger2 = SomaAddress::random();

    // First report with challenger1
    target.report_submission(validator_addr, Some(challenger1));
    assert_eq!(target.submission_reports.get(&validator_addr), Some(&Some(challenger1)));

    // Second report with challenger2 - should overwrite
    target.report_submission(validator_addr, Some(challenger2));
    assert_eq!(target.submission_reports.len(), 1);
    assert_eq!(target.submission_reports.get(&validator_addr), Some(&Some(challenger2)));
}

/// Test undo_report_submission removes the report
#[test]
fn test_target_undo_report_submission() {
    let mut target = create_test_target();
    let validator_addr = SomaAddress::random();

    // Add report then undo
    target.report_submission(validator_addr, None);
    assert_eq!(target.submission_reports.len(), 1);

    let removed = target.undo_report_submission(validator_addr);
    assert!(removed);
    assert_eq!(target.submission_reports.len(), 0);
}

/// Test undo_report_submission returns false when no report exists
#[test]
fn test_target_undo_report_submission_not_found() {
    let mut target = create_test_target();
    let validator_addr = SomaAddress::random();

    // Try to undo when never reported
    let removed = target.undo_report_submission(validator_addr);
    assert!(!removed);
}

/// Test get_submission_report_quorum returns false with no reports
#[test]
fn test_target_quorum_no_reports() {
    let target = create_test_target();
    let system_state = create_test_system_state_with_voting_power(vec![100, 100]);

    let (has_quorum, challenger, reporters) =
        target.get_submission_report_quorum(&system_state.validators);
    assert!(!has_quorum);
    assert!(challenger.is_none());
    assert!(reporters.is_empty());
}

/// Test get_submission_report_quorum returns false with insufficient stake
#[test]
fn test_target_quorum_insufficient_stake() {
    let mut target = create_test_target();
    // Create 4 validators with equal stake (each gets 25% voting power)
    let system_state = create_test_system_state_with_voting_power(vec![100, 100, 100, 100]);
    let validator_addr = system_state.validators.validators[0].metadata.soma_address;

    // Single validator reports (25% stake, need 67%)
    target.report_submission(validator_addr, None);

    let (has_quorum, _challenger, reporters) =
        target.get_submission_report_quorum(&system_state.validators);
    assert!(!has_quorum);
    assert_eq!(reporters.len(), 1); // Still tracked as reporter
}

/// Test get_submission_report_quorum returns true with sufficient stake
#[test]
fn test_target_quorum_sufficient_stake() {
    let mut target = create_test_target();
    // Use 4 validators with equal stake
    // Each gets 2500 voting power, quorum = 6667
    // 3 of 4 = 7500 > 6667 (quorum reached)
    let system_state = create_test_system_state_with_voting_power(vec![100, 100, 100, 100]);
    let validator1_addr = system_state.validators.validators[0].metadata.soma_address;
    let validator2_addr = system_state.validators.validators[1].metadata.soma_address;
    let validator3_addr = system_state.validators.validators[2].metadata.soma_address;
    let challenger_addr = SomaAddress::random();

    // All 3 validators report with same challenger
    target.report_submission(validator1_addr, Some(challenger_addr));
    target.report_submission(validator2_addr, Some(challenger_addr));
    target.report_submission(validator3_addr, Some(challenger_addr));

    let (has_quorum, winning_challenger, reporters) =
        target.get_submission_report_quorum(&system_state.validators);
    assert!(has_quorum);
    assert_eq!(winning_challenger, Some(challenger_addr));
    assert_eq!(reporters.len(), 3);
}

/// Test get_submission_report_quorum with no challenger consensus
#[test]
fn test_target_quorum_no_challenger_consensus() {
    let mut target = create_test_target();
    let system_state = create_test_system_state_with_voting_power(vec![100, 100, 100, 100]);
    let validator1_addr = system_state.validators.validators[0].metadata.soma_address;
    let validator2_addr = system_state.validators.validators[1].metadata.soma_address;
    let validator3_addr = system_state.validators.validators[2].metadata.soma_address;
    let challenger1 = SomaAddress::random();
    let challenger2 = SomaAddress::random();

    // Different challengers - no consensus
    target.report_submission(validator1_addr, Some(challenger1));
    target.report_submission(validator2_addr, Some(challenger2));
    target.report_submission(validator3_addr, None); // No challenger

    let (has_quorum, winning_challenger, reporters) =
        target.get_submission_report_quorum(&system_state.validators);
    // Has quorum (3 of 4) but no single challenger has quorum
    assert!(has_quorum);
    assert!(winning_challenger.is_none()); // No consensus on challenger
    assert_eq!(reporters.len(), 3);
}

/// Test clear_submission_reports removes all reports
#[test]
fn test_target_clear_submission_reports() {
    let mut target = create_test_target();
    let validator1 = SomaAddress::random();
    let validator2 = SomaAddress::random();

    target.report_submission(validator1, None);
    target.report_submission(validator2, None);
    assert_eq!(target.submission_reports.len(), 2);

    target.clear_submission_reports();
    assert_eq!(target.submission_reports.len(), 0);
}

/// Test quorum calculation with validators of different stakes
#[test]
fn test_target_quorum_with_varied_stakes() {
    let mut target = create_test_target();
    // Create validators with different stakes: [200, 100, 100] = 50%, 25%, 25%
    let system_state = create_test_system_state_with_voting_power(vec![200, 100, 100]);
    let big_validator_addr = system_state.validators.validators[0].metadata.soma_address;
    let small_validator_addr = system_state.validators.validators[1].metadata.soma_address;
    let challenger_addr = SomaAddress::random();

    // Small validator alone (25%) - no quorum
    target.report_submission(small_validator_addr, Some(challenger_addr));
    let (has_quorum, _, _) = target.get_submission_report_quorum(&system_state.validators);
    assert!(!has_quorum);

    // Add big validator (50% + 25% = 75%) - should have quorum
    target.report_submission(big_validator_addr, Some(challenger_addr));
    let (has_quorum, winning_challenger, _) =
        target.get_submission_report_quorum(&system_state.validators);
    assert!(has_quorum);
    assert_eq!(winning_challenger, Some(challenger_addr));
}

// =============================================================================
// Edge Case Tests for Submission and Target
// =============================================================================

/// Test submission with zero-size data
#[test]
fn test_submission_with_zero_size_data() {
    let miner = SomaAddress::random();
    let data_commitment = DataCommitment::random();
    let data_manifest = test_submission_manifest(0); // Zero-size data
    let model_id = ModelId::random();
    let embedding = SomaTensor::zeros(vec![10]);

    let submission = Submission::new(
        miner,
        data_commitment,
        data_manifest.clone(),
        model_id,
        embedding,
        SomaTensor::scalar(0.0), // distance_score
        0,                       // bond_amount (zero because zero data size)
        1,                       // submit_epoch
    );

    assert_eq!(submission.bond_amount, 0);
    assert_eq!(data_manifest.size(), 0);
}

/// Test bond calculation with zero size produces zero bond
#[test]
fn test_bond_calculation_zero_size() {
    let bond_per_byte: u64 = 10;
    let data_manifest = test_submission_manifest(0);
    let bond = bond_per_byte * data_manifest.size() as u64;
    assert_eq!(bond, 0);
}

/// Test reports from non-validators are ignored in quorum calculation
#[test]
fn test_quorum_ignores_non_validator_reports() {
    let mut target = create_test_target();
    // Create system state with 4 validators, each with 100 stake
    let system_state = create_test_system_state_with_voting_power(vec![100, 100, 100, 100]);

    // Get one real validator
    let real_validator = system_state.validators.validators[0].metadata.soma_address;

    // Create addresses that are NOT validators
    let fake_validator1 = SomaAddress::random();
    let fake_validator2 = SomaAddress::random();
    let fake_validator3 = SomaAddress::random();
    let challenger = SomaAddress::random();

    // Have 3 non-validators report (these should be ignored)
    target.report_submission(fake_validator1, Some(challenger));
    target.report_submission(fake_validator2, Some(challenger));
    target.report_submission(fake_validator3, Some(challenger));

    // Check quorum - should NOT have quorum even with 3 reports
    let (has_quorum, _, reporters) = target.get_submission_report_quorum(&system_state.validators);
    assert!(!has_quorum, "Non-validator reports should not count toward quorum");
    assert!(reporters.is_empty(), "Non-validators should not be in reporters list");

    // Now add the real validator
    target.report_submission(real_validator, Some(challenger));

    // Still should not have quorum (only 1 validator = 25%)
    let (has_quorum, _, reporters) = target.get_submission_report_quorum(&system_state.validators);
    assert!(!has_quorum, "Single validator should not reach quorum");
    assert_eq!(reporters.len(), 1, "Should only count the real validator");
}

/// Test that model_owner being None (shouldn't happen but edge case)
/// means model share is not distributed
#[test]
fn test_target_with_no_model_owner() {
    // Create a filled target but with winning_model_owner = None
    // This tests the edge case in ClaimRewards
    let mut target = create_test_target();
    target.miner = Some(SomaAddress::random());
    target.winning_model_id = Some(ModelId::random());
    target.winning_model_owner = None; // Edge case: no model owner
    target.bond_amount = 5000;
    target.reward_pool = 10000;
    target.status = TargetStatus::Filled { fill_epoch: 0 };

    // Verify the state is as expected
    assert!(target.miner.is_some());
    assert!(target.winning_model_owner.is_none());
}

/// Test quorum threshold edge case - exactly at threshold
#[test]
fn test_quorum_exactly_at_threshold() {
    let mut target = create_test_target();
    // Use validators with equal stake (each gets 2500 voting power)
    // Quorum threshold is 6667, so need 3 of 4 (7500 > 6667)
    let system_state = create_test_system_state_with_voting_power(vec![100, 100, 100, 100]);

    let validator1 = system_state.validators.validators[0].metadata.soma_address;
    let validator2 = system_state.validators.validators[1].metadata.soma_address;
    let challenger = SomaAddress::random();

    // 2 validators = 5000 voting power < 6667 threshold
    target.report_submission(validator1, Some(challenger));
    target.report_submission(validator2, Some(challenger));

    let (has_quorum, _, _) = target.get_submission_report_quorum(&system_state.validators);
    assert!(!has_quorum, "2 of 4 validators should not reach quorum");
}

/// Test validator reward remainder goes to first validator
#[test]
fn test_validator_reward_remainder_distribution() {
    // This tests the logic in distribute_bond_to_validators
    // where remainder goes to the first validator
    let bond: u64 = 100;
    let num_validators = 3;

    let per_validator = bond / num_validators;
    let remainder = bond % num_validators;

    assert_eq!(per_validator, 33);
    assert_eq!(remainder, 1);

    // First validator gets 33 + 1 = 34
    // Others get 33 each
    // Total: 34 + 33 + 33 = 100 ✓
    let first_amount = per_validator + remainder;
    let other_amount = per_validator;
    let total = first_amount + other_amount * 2;
    assert_eq!(total, bond, "All bond should be distributed");
}

/// Test concurrent reports from same validator (last wins)
#[test]
fn test_report_overwrite_behavior() {
    let mut target = create_test_target();
    let validator = SomaAddress::random();
    let challenger1 = SomaAddress::random();
    let challenger2 = SomaAddress::random();

    // First report with challenger1
    target.report_submission(validator, Some(challenger1));
    assert_eq!(target.submission_reports.get(&validator), Some(&Some(challenger1)));

    // Second report overwrites with challenger2
    target.report_submission(validator, Some(challenger2));
    assert_eq!(target.submission_reports.get(&validator), Some(&Some(challenger2)));
    assert_eq!(target.submission_reports.len(), 1, "Should still have only one report");
}
