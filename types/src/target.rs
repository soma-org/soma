//! Target generation for the Soma mining competition.
//!
//! Targets are shared objects that miners compete to fill. Each target has an embedding
//! (center point) and thresholds for distance and reconstruction score. Miners submit
//! data that embeds within the target's radius.
//!
//! Key design decisions:
//! - Targets are shared objects (not SystemState fields) for parallelism
//! - Multiple models per target (uniformly random selection)
//! - Epoch-scoped: all targets expire at epoch boundary
//! - Spawn-on-fill: filling a target spawns 1 replacement

use ndarray::Array1;
use rand::{rngs::StdRng, seq::SliceRandom as _, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    base::SomaAddress,
    committee::EpochId,
    crypto::DefaultHash,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::ExecutionResult,
    model::ModelId,
    object::ObjectID,
    system_state::model_registry::ModelRegistry,
    system_state::target_state::TargetState,
};
use fastcrypto::hash::HashFunction as _;

/// Type alias: targets are identified by their ObjectID.
pub type TargetId = ObjectID;

/// Scale factor for converting f32 normal values to fixed-point i64.
/// Using 10^6 gives 6 decimal places of precision.
pub const EMBEDDING_SCALE: i64 = 1_000_000;

/// Fixed-point embedding vector. Scale factor is a protocol constant.
/// i64 gives ~18 decimal digits — more than enough for f32-origin values
/// while allowing safe accumulation in dot products without overflow
/// for reasonable embedding dimensions (up to ~10k dims).
///
/// Using ndarray::Array1 for seamless NumPy interop in the Python SDK.
pub type Embedding = Array1<i64>;

/// A target in the Soma mining competition.
///
/// Targets are shared objects — not fields inside SystemState. This prevents
/// SystemState from becoming a contention bottleneck at high TPS, since each
/// target can be mutated independently by consensus.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Target {
    /// Center embedding point miners must get close to.
    pub embedding: Embedding,

    /// Multiple models assigned to this target (uniformly random selection).
    /// Miners choose which model to use. Models are scored by reconstruction quality.
    pub model_ids: Vec<ModelId>,

    /// Distance threshold (fixed-point, scale DISTANCE_SCALE).
    /// Submitter must report distance <= this value (lower is better).
    pub distance_threshold: i64,

    /// Reconstruction error threshold (MSE, fixed-point).
    /// Submitter must report reconstruction_score <= this value (lower is better).
    pub reconstruction_threshold: u64,

    /// Pre-allocated reward amount (in shannons) for this target.
    /// Funded from emissions at target creation time.
    pub reward_pool: u64,

    /// Epoch in which this target was generated.
    /// Target expires at the end of this epoch (regardless of when spawned).
    pub generation_epoch: EpochId,

    /// Current status
    pub status: TargetStatus,

    /// The miner who filled this target (set when status transitions to Filled).
    /// Used by ClaimRewards for reward distribution.
    pub miner: Option<SomaAddress>,

    /// The model used by the miner (set when status transitions to Filled).
    pub winning_model_id: Option<ModelId>,

    /// The owner of the winning model (set when status transitions to Filled).
    /// Captured at fill time so rewards can be distributed even if model becomes inactive.
    pub winning_model_owner: Option<SomaAddress>,

    /// Bond amount held for the submission (in shannons).
    /// Set when the target is filled. On successful claim (no challenge),
    /// the bond is returned to the miner. On successful challenge,
    /// the bond is forfeited to the emission pool.
    pub bond_amount: u64,
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

impl Target {
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
}

/// Generate a new target with the given parameters.
///
/// # Arguments
/// * `seed` - Deterministic seed derived from transaction digest + creation number
/// * `model_registry` - Registry of active models for random selection
/// * `target_state` - Current difficulty thresholds and reward per target
/// * `models_per_target` - Number of models to assign to this target
/// * `embedding_dim` - Dimension of the embedding vector
/// * `current_epoch` - The epoch in which this target is being created
///
/// # Errors
/// Returns `ExecutionFailureStatus::NoActiveModels` if no active models exist.
pub fn generate_target(
    seed: u64,
    model_registry: &ModelRegistry,
    target_state: &TargetState,
    models_per_target: u64,
    embedding_dim: u64,
    current_epoch: EpochId,
) -> ExecutionResult<Target> {
    // 1. Select models via uniform random sampling (no replacement)
    let model_ids = select_models_uniform(seed, model_registry, models_per_target)?;

    // 2. Generate deterministic embedding
    let embedding = deterministic_embedding(seed, embedding_dim);

    // 3. Create target with current difficulty thresholds and reward
    Ok(Target {
        embedding,
        model_ids,
        distance_threshold: target_state.distance_threshold,
        reconstruction_threshold: target_state.reconstruction_threshold,
        reward_pool: target_state.reward_per_target,
        generation_epoch: current_epoch,
        status: TargetStatus::Open,
        miner: None,
        winning_model_id: None,
        winning_model_owner: None,
        bond_amount: 0, // Set when target is filled by a submission
    })
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
/// Uses standard normal distribution (mean=0, std_dev=1) scaled to fixed-point i64.
/// The implementation uses the same RNG as `arrgen` for consistency.
///
/// # Arguments
/// * `seed` - Random seed for reproducibility
/// * `dim` - Dimension of the embedding vector
pub fn deterministic_embedding(seed: u64, dim: u64) -> Embedding {
    // Generate standard normal distribution (mean=0, std_dev=1)
    // We use arrgen's normal_array which provides deterministic generation
    let f32_array = arrgen::normal_array(seed, &[dim as usize], 0.0, 1.0);

    // Convert from dynamic to 1D array, then map to fixed-point i64
    // The f32_array is ArrayD<f32>, we need to convert to Array1<i64>
    let flat: Vec<f32> = f32_array.into_iter().collect();
    Array1::from_vec(
        flat.into_iter()
            .map(|v| {
                let scaled = f64::from(v) * EMBEDDING_SCALE as f64;
                // Clamp to i64 range to prevent overflow (though standard normal
                // values scaled by 10^6 should never approach these bounds)
                scaled.clamp(i64::MIN as f64, i64::MAX as f64) as i64
            })
            .collect(),
    )
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
    hasher.update(&creation_num.to_le_bytes());
    let hash = hasher.finalize();
    // Use first 8 bytes as u64 seed
    u64::from_le_bytes(
        hash.as_ref()[..8]
            .try_into()
            .expect("hash is at least 8 bytes"),
    )
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
        let mut target = Target {
            embedding: Array1::zeros(10),
            model_ids: vec![],
            distance_threshold: 1000,
            reconstruction_threshold: 1000,
            reward_pool: 1000,
            generation_epoch: 0,
            status: TargetStatus::Open,
            miner: None,
            winning_model_id: None,
            winning_model_owner: None,
            bond_amount: 0,
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
