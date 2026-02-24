//! Challenge types for the SOMA data submission competition dispute resolution.
//!
//! Challenges enable anyone to dispute data submissions. Validators audit
//! the submission and submit report transactions. A 2f+1 quorum determines the outcome.
//!
//! Key design decisions:
//! - Challenges are shared objects (epoch-scoped, no registry needed)
//! - Challenge window is only during the fill epoch
//! - Fraud-only challenges (availability is handled via submission reports)
//! - Challenger bond = challenger_bond_per_byte * data_size (from target's winning submission)
//!
//! **Tally-Based Design**:
//! - Validators submit ReportChallenge transactions if they determine the challenger is wrong
//!   (i.e., the submission is actually valid)
//! - If 2f+1 validators report, the challenger loses and their bond goes to validators
//! - If no quorum is reached, challenger gets benefit of doubt and bond is returned
//!
//! **Simplified Design (v2)**:
//! - Removed ChallengeVerdict enum - reports simply indicate "challenger is wrong"
//! - If reported by 2f+1 validators, challenger loses; otherwise challenger wins (benefit of doubt)
//! - Submission reports (ReportSubmission/UndoReportSubmission) handle the submitter's bond

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    base::SomaAddress, committee::EpochId, digests::DataCommitment, model::ModelId,
    object::ObjectID, submission::SubmissionManifest, system_state::validator::ValidatorSet,
    target::TargetId, tensor::SomaTensor,
};

/// Unique identifier for a challenge (same as ObjectID).
pub type ChallengeId = ObjectID;

/// A challenge against a filled target's submission.
///
/// Challenges are shared objects that track the dispute state.
/// Created by InitiateChallenge, resolved by ClaimChallengeBond when quorum is reached.
///
/// **Self-contained for auditing**: Contains all necessary data for validators to audit
/// without needing to load the Target object. This includes the target's embedding,
/// model assignments, and the submitter's claimed submission data.
///
/// **Fraud-only**: All challenges are fraud challenges. Availability issues are handled
/// separately via submission reports (ReportSubmission/UndoReportSubmission).
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ChallengeV1 {
    /// Unique identifier for this challenge (same as the Object's ID).
    /// Set when the challenge object is created.
    pub id: ChallengeId,

    /// The target being challenged (for reference)
    pub target_id: TargetId,

    /// Address of the challenger
    pub challenger: SomaAddress,

    /// Bond locked by challenger (bond_per_byte * data_size, retrieved from target)
    pub challenger_bond: u64,

    /// Epoch when challenge was initiated (challenge expires at epoch end)
    pub challenge_epoch: EpochId,

    /// Current status
    pub status: ChallengeStatus,

    // =========================================================================
    // Audit data (copied from Target at challenge creation time)
    // =========================================================================
    /// All models assigned to the target (for multi-model fraud verification)
    pub model_ids: Vec<ModelId>,

    /// Target's embedding - used to compute distance from submitter's embedding
    pub target_embedding: SomaTensor,

    /// Target's distance threshold - submissions must be within this threshold
    pub distance_threshold: SomaTensor,

    /// The model ID the submitter claimed to use
    pub winning_model_id: ModelId,

    /// Manifest for the submitter's data (URL + checksum + size)
    pub winning_data_manifest: SubmissionManifest,

    /// Hash commitment of the submitter's raw data
    pub winning_data_commitment: DataCommitment,

    /// The embedding the submitter claimed to produce
    pub winning_embedding: SomaTensor,

    /// The distance score the submitter claimed
    pub winning_distance_score: SomaTensor,

    // =========================================================================
    // Tally-based challenge report fields
    // =========================================================================
    /// Validators who have reported that the challenger is wrong (submission is valid).
    /// If 2f+1 validators report, challenger loses their bond.
    /// Stored on Challenge object (not SystemState) for locality.
    /// Cleared when challenge bond is claimed.
    pub challenge_reports: BTreeSet<SomaAddress>,
}

impl ChallengeV1 {
    /// Creates a new pending fraud challenge with audit data from the target.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ChallengeId,
        target_id: TargetId,
        challenger: SomaAddress,
        challenger_bond: u64,
        challenge_epoch: EpochId,
        // Audit data from target
        model_ids: Vec<ModelId>,
        target_embedding: SomaTensor,
        distance_threshold: SomaTensor,
        winning_model_id: ModelId,
        winning_data_manifest: SubmissionManifest,
        winning_data_commitment: DataCommitment,
        winning_embedding: SomaTensor,
        winning_distance_score: SomaTensor,
    ) -> Self {
        Self {
            id,
            target_id,
            challenger,
            challenger_bond,
            challenge_epoch,
            status: ChallengeStatus::Pending,
            // Audit data
            model_ids,
            target_embedding,
            distance_threshold,
            winning_model_id,
            winning_data_manifest,
            winning_data_commitment,
            winning_embedding,
            winning_distance_score,
            // Tally-based challenge reports
            challenge_reports: BTreeSet::new(),
        }
    }

    /// Returns true if the challenge is still pending.
    pub fn is_pending(&self) -> bool {
        matches!(self.status, ChallengeStatus::Pending)
    }

    /// Returns true if the challenge has been resolved.
    pub fn is_resolved(&self) -> bool {
        matches!(self.status, ChallengeStatus::Resolved { .. })
    }

    /// Returns whether the challenger lost (if resolved).
    pub fn challenger_lost(&self) -> Option<bool> {
        match &self.status {
            ChallengeStatus::Resolved { challenger_lost } => Some(*challenger_lost),
            _ => None,
        }
    }

    // =========================================================================
    // Tally-based challenge report methods
    // =========================================================================

    /// Record a challenge report from a validator (indicating challenger is wrong).
    pub fn report_challenge(&mut self, reporter: SomaAddress) {
        self.challenge_reports.insert(reporter);
    }

    /// Remove a challenge report from a validator.
    /// Returns true if the report was found and removed.
    pub fn undo_report_challenge(&mut self, reporter: SomaAddress) -> bool {
        self.challenge_reports.remove(&reporter)
    }

    /// Check if 2f+1 validators have reported against the challenger.
    ///
    /// Returns (has_quorum, reporters) where:
    /// - has_quorum: true if 2f+1 stake has reported against challenger
    /// - reporters: list of validators who reported
    pub fn get_challenge_report_quorum(
        &self,
        validators: &ValidatorSet,
    ) -> (bool, Vec<SomaAddress>) {
        use crate::committee::QUORUM_THRESHOLD;

        // Collect valid reporters (validators who have reported)
        let reporters: Vec<SomaAddress> = self
            .challenge_reports
            .iter()
            .filter(|addr| validators.find_validator(**addr).is_some())
            .cloned()
            .collect();

        // Sum voting power of reporters
        let total_stake = validators.sum_voting_power_by_addresses(&reporters);

        // Use protocol quorum threshold (2f+1)
        (total_stake >= QUORUM_THRESHOLD, reporters)
    }

    /// Clear challenge reports (called when challenge bond is claimed).
    pub fn clear_challenge_reports(&mut self) {
        self.challenge_reports.clear();
    }
}

/// Status of a challenge in its lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub enum ChallengeStatus {
    /// Challenge initiated, awaiting validator audit
    Pending,

    /// Challenge resolved
    /// - challenger_lost = true: 2f+1 validators reported challenger was wrong, bond forfeited
    /// - challenger_lost = false: no quorum, challenger gets benefit of doubt, bond returned
    Resolved { challenger_lost: bool },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        checksum::Checksum,
        metadata::{Manifest, ManifestV1, Metadata, MetadataV1},
    };

    /// Helper to create a test challenge with dummy audit data.
    fn make_test_challenge(
        challenge_id: ChallengeId,
        target_id: TargetId,
        challenger: SomaAddress,
        challenger_bond: u64,
        challenge_epoch: EpochId,
    ) -> ChallengeV1 {
        let model_id = ObjectID::random();
        let dummy_embedding = SomaTensor::zeros(vec![10]);
        let dummy_checksum = Checksum::default();
        let dummy_url = url::Url::parse("https://example.com/data").unwrap();
        let dummy_metadata = Metadata::V1(MetadataV1::new(dummy_checksum, 1000));
        let dummy_manifest = Manifest::V1(ManifestV1::new(dummy_url, dummy_metadata));
        let dummy_submission_manifest = SubmissionManifest { manifest: dummy_manifest };
        let dummy_commitment = DataCommitment::new([0u8; 32]);

        ChallengeV1::new(
            challenge_id,
            target_id,
            challenger,
            challenger_bond,
            challenge_epoch,
            vec![model_id],
            dummy_embedding.clone(),
            SomaTensor::scalar(0.5), // distance threshold
            model_id,
            dummy_submission_manifest,
            dummy_commitment,
            dummy_embedding,
            SomaTensor::scalar(0.25), // winning distance score
        )
    }

    #[test]
    fn test_challenge_creation() {
        let challenge_id = ObjectID::random();
        let target_id = ObjectID::random();
        let challenger = SomaAddress::random();
        let challenge = make_test_challenge(challenge_id, target_id, challenger, 1000, 5);

        assert_eq!(challenge.id, challenge_id);
        assert_eq!(challenge.target_id, target_id);
        assert_eq!(challenge.challenger, challenger);
        assert_eq!(challenge.challenger_bond, 1000);
        assert_eq!(challenge.challenge_epoch, 5);
        assert!(challenge.is_pending());
        assert!(!challenge.is_resolved());
        assert!(challenge.challenge_reports.is_empty());
    }

    #[test]
    fn test_challenge_reports() {
        let challenge_id = ObjectID::random();
        let target_id = ObjectID::random();
        let challenger = SomaAddress::random();
        let mut challenge = make_test_challenge(challenge_id, target_id, challenger, 1000, 5);

        let reporter1 = SomaAddress::random();
        let reporter2 = SomaAddress::random();

        // Add reports
        challenge.report_challenge(reporter1);
        assert_eq!(challenge.challenge_reports.len(), 1);
        assert!(challenge.challenge_reports.contains(&reporter1));

        challenge.report_challenge(reporter2);
        assert_eq!(challenge.challenge_reports.len(), 2);

        // Duplicate report should not add
        challenge.report_challenge(reporter1);
        assert_eq!(challenge.challenge_reports.len(), 2);

        // Undo report
        assert!(challenge.undo_report_challenge(reporter1));
        assert_eq!(challenge.challenge_reports.len(), 1);
        assert!(!challenge.challenge_reports.contains(&reporter1));

        // Undo non-existent report returns false
        assert!(!challenge.undo_report_challenge(reporter1));

        // Clear reports
        challenge.clear_challenge_reports();
        assert!(challenge.challenge_reports.is_empty());
    }

    #[test]
    fn test_challenge_status_transitions() {
        let challenge_id = ObjectID::random();
        let target_id = ObjectID::random();
        let challenger = SomaAddress::random();
        let mut challenge = make_test_challenge(challenge_id, target_id, challenger, 1000, 5);

        assert!(challenge.is_pending());
        assert!(!challenge.is_resolved());
        assert!(challenge.challenger_lost().is_none());

        // Resolve with challenger losing (2f+1 reported against)
        challenge.status = ChallengeStatus::Resolved { challenger_lost: true };
        assert!(!challenge.is_pending());
        assert!(challenge.is_resolved());
        assert_eq!(challenge.challenger_lost(), Some(true));

        // Resolve with challenger winning (no quorum)
        challenge.status = ChallengeStatus::Resolved { challenger_lost: false };
        assert!(!challenge.is_pending());
        assert!(challenge.is_resolved());
        assert_eq!(challenge.challenger_lost(), Some(false));
    }
}
