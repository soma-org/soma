//! Target generation for the Soma data submission competition.
//!
//! Targets are shared objects that submitters compete to fill. Each target has an embedding
//! (center point) and a distance threshold. Submitters submit data that embeds within the
//! target's radius.
//!
//! Key design decisions:
//! - Targets are shared objects (not SystemState fields) for parallelism
//! - Multiple models per target (uniformly random selection)
//! - Epoch-scoped: all targets expire at epoch boundary
//! - Spawn-on-fill: filling a target spawns 1 replacement

use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom as _};
use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

#[cfg(feature = "ml")]
use crate::model_selection::{ModelSelectionData, select_models};
use crate::{
    base::SomaAddress,
    challenge::ChallengeId,
    committee::EpochId,
    crypto::DefaultHash,
    digests::{DataCommitment, TransactionDigest},
    effects::ExecutionFailureStatus,
    error::ExecutionResult,
    model::ModelId,
    object::ObjectID,
    submission::SubmissionManifest,
    system_state::model_registry::ModelRegistry,
    system_state::target_state::TargetState,
    system_state::validator::ValidatorSet,
    tensor::SomaTensor,
};
#[cfg(feature = "ml")]
use burn::backend::NdArray;
use fastcrypto::hash::HashFunction as _;

/// Type alias: targets are identified by their ObjectID.
pub type TargetId = ObjectID;

/// A target in the Soma data submission competition.
///
/// Targets are shared objects — not fields inside SystemState. This prevents
/// SystemState from becoming a contention bottleneck at high TPS, since each
/// target can be mutated independently by consensus.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct TargetV1 {
    /// Center embedding point submitters must get close to.
    /// Stored as SomaTensor (wraps Burn's TensorData) for f32 embeddings.
    pub embedding: SomaTensor,

    /// Multiple models assigned to this target (uniformly random selection).
    /// Submitters choose which model to use.
    pub model_ids: Vec<ModelId>,

    /// Distance threshold (cosine distance).
    /// Stored as scalar SomaTensor for consistency with CompetitionAPI.
    /// Submitter must report distance <= this value (lower is better).
    pub distance_threshold: SomaTensor,

    /// Pre-allocated reward amount (in shannons) for this target.
    /// Funded from emissions at target creation time.
    pub reward_pool: u64,

    /// Epoch in which this target was generated.
    /// Target expires at the end of this epoch (regardless of when spawned).
    pub generation_epoch: EpochId,

    /// Current status
    pub status: TargetStatus,

    /// The submitter who filled this target (set when status transitions to Filled).
    /// Used by ClaimRewards for reward distribution.
    pub submitter: Option<SomaAddress>,

    /// The model used by the submitter (set when status transitions to Filled).
    pub winning_model_id: Option<ModelId>,

    /// The owner of the winning model (set when status transitions to Filled).
    /// Captured at fill time so rewards can be distributed even if model becomes inactive.
    pub winning_model_owner: Option<SomaAddress>,

    /// Bond amount held for the submission (in shannons).
    /// Set when the target is filled. On successful claim (no challenge),
    /// the bond is returned to the submitter. On successful challenge,
    /// the bond is forfeited to the emission pool.
    pub bond_amount: u64,

    // =========================================================================
    // Fields for challenge audit (set when target is filled)
    // =========================================================================
    /// Manifest for the winning submission's data (URL + checksum + size).
    /// Used by challengers and auditing validators to download and verify.
    pub winning_data_manifest: Option<SubmissionManifest>,

    /// Commitment to the winning submission's raw data: hash(data_bytes).
    /// Used to verify data integrity during challenge audit.
    pub winning_data_commitment: Option<DataCommitment>,

    /// Embedding vector from the winning submission.
    /// Verified during challenge audit by re-running inference.
    pub winning_embedding: Option<SomaTensor>,

    /// Distance score from the winning submission (cosine distance).
    /// Stored as scalar SomaTensor for consistency with CompetitionAPI.
    /// Verified during ScoreFraud challenge audit.
    pub winning_distance_score: Option<SomaTensor>,

    // =========================================================================
    // Tally-based challenge fields (set when target is filled)
    // =========================================================================
    /// Challenger address (set by InitiateChallenge, first one wins).
    /// If set, ReportSubmission reports can attribute fraud to this challenger.
    pub challenger: Option<SomaAddress>,

    /// Challenge object ID (set by InitiateChallenge).
    /// Used to look up Challenge for ClaimChallengeBond.
    pub challenge_id: Option<ChallengeId>,

    /// Submission reports: reporter → optional challenger attribution.
    /// Stored on Target object (not SystemState) for locality.
    /// Cleared when target is claimed.
    pub submission_reports: BTreeMap<SomaAddress, Option<SomaAddress>>,
}

/// Status of a target in its lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub enum TargetStatus {
    /// Accepting submissions (during generation_epoch only)
    Open,
    /// A valid submission has been accepted; challenge window runs through fill_epoch + 1
    Filled { fill_epoch: EpochId },
    /// Rewards have been claimed (or target expired unfilled and was reclaimed)
    Claimed,
}

impl TargetV1 {
    /// Returns true if this target is accepting submissions.
    pub fn is_open(&self) -> bool {
        matches!(self.status, TargetStatus::Open)
    }

    /// Returns true if this target has been filled.
    pub fn is_filled(&self) -> bool {
        matches!(self.status, TargetStatus::Filled { .. })
    }

    /// Returns true if rewards have been claimed.
    pub fn is_claimed(&self) -> bool {
        matches!(self.status, TargetStatus::Claimed)
    }

    /// Returns the fill epoch if this target is filled.
    pub fn fill_epoch(&self) -> Option<EpochId> {
        match self.status {
            TargetStatus::Filled { fill_epoch } => Some(fill_epoch),
            _ => None,
        }
    }

    // =========================================================================
    // Tally-based submission report methods
    // =========================================================================

    /// Record a submission report from a validator.
    /// The optional challenger parameter allows attributing fraud to a specific challenger.
    pub fn report_submission(&mut self, reporter: SomaAddress, challenger: Option<SomaAddress>) {
        self.submission_reports.insert(reporter, challenger);
    }

    /// Remove a submission report.
    pub fn undo_report_submission(&mut self, reporter: SomaAddress) -> bool {
        self.submission_reports.remove(&reporter).is_some()
    }

    /// Get submission report quorum result.
    /// Returns (has_quorum, winning_challenger, reporting_validators).
    /// - has_quorum: true if total reporter stake >= 2f+1
    /// - winning_challenger: Some if >2/3 of reporting stake agrees on same challenger
    /// - reporting_validators: list of validators who reported (for bond distribution)
    pub fn get_submission_report_quorum(
        &self,
        validator_set: &ValidatorSet,
    ) -> (bool, Option<SomaAddress>, Vec<SomaAddress>) {
        if self.submission_reports.is_empty() {
            return (false, None, vec![]);
        }

        let quorum_threshold = crate::committee::QUORUM_THRESHOLD;

        // Calculate total reporter stake
        let mut total_stake = 0u64;
        let mut reporters = vec![];
        for reporter in self.submission_reports.keys() {
            if let Some(v) = validator_set.find_validator(*reporter) {
                total_stake += v.voting_power;
                reporters.push(*reporter);
            }
        }

        if total_stake < quorum_threshold {
            return (false, None, reporters);
        }

        // Count stake by challenger attribution
        let mut challenger_stakes: BTreeMap<Option<SomaAddress>, u64> = BTreeMap::new();
        for (reporter, challenger) in &self.submission_reports {
            if let Some(v) = validator_set.find_validator(*reporter) {
                *challenger_stakes.entry(*challenger).or_default() += v.voting_power;
            }
        }

        // Find if any challenger has quorum among reporting stake
        for (challenger, stake) in challenger_stakes {
            if stake >= quorum_threshold {
                return (true, challenger, reporters);
            }
        }

        // Has reporting quorum but no consensus on challenger
        (true, None, reporters)
    }

    /// Clear submission reports (called when target is claimed).
    pub fn clear_submission_reports(&mut self) {
        self.submission_reports.clear();
    }
}

#[cfg(feature = "ml")]
/// Generate a new target with the given parameters.
///
/// Uses stake-weighted KNN model selection: models with embeddings closer to the
/// target embedding are preferred, weighted by their normalized stake (voting power).
/// Falls back to uniform random selection if no models have embeddings.
///
/// # Arguments
/// * `seed` - Deterministic seed derived from transaction digest + creation number
/// * `model_registry` - Registry of active models for selection
/// * `target_state` - Current difficulty thresholds and reward per target
/// * `models_per_target` - Number of models to assign to this target
/// * `embedding_dim` - Dimension of the embedding vector
/// * `current_epoch` - The epoch in which this target is being created
///
/// # Errors
/// Returns `ExecutionFailureStatus::NoActiveModels` if no active models exist.
#[allow(clippy::result_large_err)]
pub fn generate_target(
    seed: u64,
    model_registry: &ModelRegistry,
    target_state: &TargetState,
    models_per_target: u64,
    embedding_dim: u64,
    current_epoch: EpochId,
) -> ExecutionResult<TargetV1> {
    // 1. Generate deterministic embedding for the target
    let embedding = deterministic_embedding(seed, embedding_dim);

    // 2. Select models via stake-weighted KNN (falls back to uniform if no embeddings)
    let model_ids =
        select_models_weighted_knn(seed, model_registry, &embedding, models_per_target)?;

    // 3. Create target with current difficulty thresholds and reward
    Ok(TargetV1 {
        embedding,
        model_ids,
        distance_threshold: target_state.distance_threshold.clone(),
        reward_pool: target_state.reward_per_target,
        generation_epoch: current_epoch,
        status: TargetStatus::Open,
        submitter: None,
        winning_model_id: None,
        winning_model_owner: None,
        bond_amount: 0, // Set when target is filled by a submission
        // Challenge audit fields (set when target is filled)
        winning_data_manifest: None,
        winning_data_commitment: None,
        winning_embedding: None,
        winning_distance_score: None,
        // Tally-based challenge fields (set when challenged)
        challenger: None,
        challenge_id: None,
        submission_reports: BTreeMap::new(),
    })
}

#[cfg(feature = "ml")]
/// Select models using stake-weighted KNN based on target embedding.
///
/// Models are scored by: `weighted_score = distance² / voting_power`
/// where voting_power is the model's normalized stake (sums to 1.0).
/// Lower scores are better (closer distance and/or higher stake).
///
/// Falls back to uniform random selection if no models have embeddings.
///
/// # Arguments
/// * `seed` - Random seed for fallback selection
/// * `model_registry` - Registry containing active models
/// * `target_embedding` - The target's embedding vector
/// * `count` - Number of models to select
///
/// # Errors
/// Returns `ExecutionFailureStatus::NoActiveModels` if no active models exist.
#[allow(clippy::result_large_err)]
pub fn select_models_weighted_knn(
    seed: u64,
    model_registry: &ModelRegistry,
    target_embedding: &SomaTensor,
    count: u64,
) -> ExecutionResult<Vec<ModelId>> {
    if model_registry.active_models.is_empty() {
        return Err(ExecutionFailureStatus::NoActiveModels);
    }

    // Collect models that have embeddings for weighted selection
    let models_with_embeddings: Vec<(ModelId, SomaTensor, u64)> = model_registry
        .active_models
        .iter()
        .filter_map(|(id, model)| {
            model.embedding.as_ref().map(|emb| (*id, emb.clone(), model.stake()))
        })
        .collect();

    // If no models have embeddings, fall back to uniform selection
    if models_with_embeddings.is_empty() {
        return select_models_uniform(seed, model_registry, count);
    }

    let count = count.min(models_with_embeddings.len() as u64) as usize;

    // Use NdArray backend for deterministic CPU computation
    type B = NdArray;
    let device: <B as burn::prelude::Backend>::Device = Default::default();

    // Prepare model data for selection
    let selection_data = ModelSelectionData::<B>::new(models_with_embeddings, &device);

    // Select top-k models by weighted score
    let matches = select_models(target_embedding, &selection_data, count);

    Ok(matches.into_iter().map(|m| m.model_id).collect())
}

/// Select models uniformly at random from the active model registry.
///
/// Uses Fisher-Yates partial shuffle with `StdRng` (same RNG as arrgen).
///
/// # Arguments
/// * `seed` - Random seed for reproducibility
/// * `model_registry` - Registry containing active models
/// * `count` - Number of models to select
///
/// # Errors
/// Returns `ExecutionFailureStatus::NoActiveModels` if no active models exist.
#[allow(clippy::result_large_err)]
pub fn select_models_uniform(
    seed: u64,
    model_registry: &ModelRegistry,
    count: u64,
) -> ExecutionResult<Vec<ModelId>> {
    let mut active: Vec<ModelId> = model_registry.active_models.keys().copied().collect();
    if active.is_empty() {
        return Err(ExecutionFailureStatus::NoActiveModels);
    }

    let count = count.min(active.len() as u64) as usize;

    // Fisher-Yates partial shuffle using StdRng
    let mut rng = StdRng::seed_from_u64(seed);
    // partial_shuffle returns (shuffled, remaining) tuple - we only need the shuffled portion
    let (shuffled, _remaining) = active.partial_shuffle(&mut rng, count);

    Ok(shuffled.to_vec())
}

/// Generate a deterministic embedding vector from a seed.
///
/// Uses standard normal distribution (mean=0, std_dev=1).
/// The implementation uses the same RNG as `arrgen` for consistency.
///
/// # Arguments
/// * `seed` - Random seed for reproducibility
/// * `dim` - Dimension of the embedding vector
pub fn deterministic_embedding(seed: u64, dim: u64) -> SomaTensor {
    // Generate standard normal distribution (mean=0, std_dev=1)
    // We use arrgen's normal_array which provides deterministic generation
    let f32_array = arrgen::normal_array(seed, &[dim as usize], 0.0, 1.0);

    // Convert from dynamic array to Vec<f32>
    let flat: Vec<f32> = f32_array.into_iter().collect();
    SomaTensor::new(flat, vec![dim as usize])
}

/// Construct a deterministic seed from transaction digest and creation number.
///
/// This ensures unique seeds for each target created within a transaction.
///
/// # Arguments
/// * `tx_digest` - The transaction digest
/// * `creation_num` - The creation counter within the transaction
pub fn make_target_seed(tx_digest: &TransactionDigest, creation_num: u64) -> u64 {
    let mut hasher = DefaultHash::default();
    // Explicitly convert to [u8; 32] to avoid ambiguity between AsRef<[u8]> and AsRef<[u8; 32]>
    let digest_bytes: &[u8; 32] = tx_digest.as_ref();
    hasher.update(digest_bytes);
    hasher.update(creation_num.to_le_bytes());
    let hash = hasher.finalize();
    // Use first 8 bytes as u64 seed
    u64::from_le_bytes(hash.as_ref()[..8].try_into().expect("hash is at least 8 bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_embedding() {
        let embedding1 = deterministic_embedding(42, 10);
        let embedding2 = deterministic_embedding(42, 10);

        // Same seed should produce same embedding
        assert_eq!(embedding1, embedding2);

        // Different seed should produce different embedding
        let embedding3 = deterministic_embedding(43, 10);
        assert_ne!(embedding1, embedding3);
    }

    #[test]
    fn test_embedding_dimension() {
        let embedding = deterministic_embedding(42, 768);
        assert_eq!(embedding.len(), 768);
    }

    #[test]
    fn test_make_target_seed_deterministic() {
        let digest = TransactionDigest::random();
        let seed1 = make_target_seed(&digest, 0);
        let seed2 = make_target_seed(&digest, 0);
        assert_eq!(seed1, seed2);

        // Different creation_num should give different seed
        let seed3 = make_target_seed(&digest, 1);
        assert_ne!(seed1, seed3);
    }

    #[test]
    fn test_target_status_methods() {
        let mut target = TargetV1 {
            embedding: SomaTensor::zeros(vec![10]),
            model_ids: vec![],
            distance_threshold: SomaTensor::scalar(0.5),
            reward_pool: 1000,
            generation_epoch: 0,
            status: TargetStatus::Open,
            submitter: None,
            winning_model_id: None,
            winning_model_owner: None,
            bond_amount: 0,
            winning_data_manifest: None,
            winning_data_commitment: None,
            winning_embedding: None,
            winning_distance_score: None,
            challenger: None,
            challenge_id: None,
            submission_reports: BTreeMap::new(),
        };

        assert!(target.is_open());
        assert!(!target.is_filled());
        assert!(!target.is_claimed());
        assert_eq!(target.fill_epoch(), None);

        target.status = TargetStatus::Filled { fill_epoch: 5 };
        assert!(!target.is_open());
        assert!(target.is_filled());
        assert!(!target.is_claimed());
        assert_eq!(target.fill_epoch(), Some(5));

        target.status = TargetStatus::Claimed;
        assert!(!target.is_open());
        assert!(!target.is_filled());
        assert!(target.is_claimed());
        assert_eq!(target.fill_epoch(), None);
    }
}
