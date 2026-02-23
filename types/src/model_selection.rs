//! Stake-weighted KNN model selection for target assignment.
//!
//! Models register with embeddings that describe their "expertise" in the shared embedding space.
//! When creating targets, we use stake-weighted KNN to select models whose embeddings are close
//! to the target embedding, with high-stake models receiving preference.
//!
//! **Scoring formula**: `weighted_score = distance² / voting_power`
//!
//! Where `voting_power` is the model's normalized stake (sums to 1.0 across all models),
//! calculated the same way as validator voting power. This encourages:
//! - Specialization: models that focus on specific regions of embedding space
//! - Reputation building: models with more stake get priority for nearby targets
//!
//! Uses Burn tensor operations with NdArray backend for deterministic CPU-based computation.

use burn::{
    backend::NdArray,
    prelude::Backend,
    tensor::{Tensor, TensorData},
};

use crate::{model::ModelId, tensor::SomaTensor};

/// Result of model selection - contains model ID, raw distance², and weighted score.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMatch {
    /// The model ID
    pub model_id: ModelId,
    /// Squared Euclidean distance to target
    pub distance_sq: f32,
    /// Stake-weighted score: distance² / voting_power
    pub weighted_score: f32,
}

/// Prepared model data for efficient selection across multiple targets.
///
/// Pre-computes normalized voting power and stacks embeddings into a 2D tensor.
/// Reuse this struct when selecting models for multiple targets in the same epoch.
pub struct ModelSelectionData<B: Backend> {
    /// Model IDs in order
    pub model_ids: Vec<ModelId>,
    /// Model embeddings stacked as [num_models, dim]
    pub embeddings: Tensor<B, 2>,
    /// Normalized voting power per model (sums to 1.0)
    pub voting_power: Tensor<B, 1>,
    /// Device for tensor operations
    pub device: B::Device,
}

impl<B: Backend> ModelSelectionData<B> {
    /// Create selection data from model information.
    ///
    /// # Arguments
    /// * `models` - List of (ModelId, embedding, stake) tuples
    /// * `device` - Burn device for tensor operations
    ///
    /// # Panics
    /// Panics if models is empty or embeddings have inconsistent dimensions.
    pub fn new(models: Vec<(ModelId, SomaTensor, u64)>, device: &B::Device) -> Self {
        assert!(!models.is_empty(), "Cannot create ModelSelectionData with no models");

        let model_ids: Vec<ModelId> = models.iter().map(|(id, _, _)| *id).collect();
        let dim = models[0].1.shape()[0];
        let num_models = models.len();

        // Flatten embeddings into 1D then reshape to [num_models, dim]
        let flat_embeddings: Vec<f32> = models
            .iter()
            .flat_map(|(_, emb, _)| {
                assert_eq!(
                    emb.shape(),
                    &[dim],
                    "All model embeddings must have the same dimension"
                );
                emb.to_vec()
            })
            .collect();

        let embeddings = Tensor::<B, 1>::from_floats(flat_embeddings.as_slice(), device)
            .reshape([num_models, dim]);

        // Calculate normalized voting power (like validator voting power, but sums to 1.0)
        let stakes: Vec<u64> = models.iter().map(|(_, _, stake)| *stake).collect();
        let voting_power = compute_normalized_voting_power::<B>(&stakes, device);

        Self { model_ids, embeddings, voting_power, device: device.clone() }
    }

    /// Number of models available for selection.
    pub fn num_models(&self) -> usize {
        self.model_ids.len()
    }

    /// Embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embeddings.dims()[1]
    }
}

/// Compute normalized voting power from stakes.
///
/// Uses the same proportional calculation as validator voting power,
/// but normalizes to sum to 1.0 instead of TOTAL_VOTING_POWER.
///
/// # Arguments
/// * `stakes` - Raw stake amounts per model
/// * `device` - Burn device for tensor operations
///
/// # Returns
/// Tensor of voting powers that sum to 1.0
fn compute_normalized_voting_power<B: Backend>(stakes: &[u64], device: &B::Device) -> Tensor<B, 1> {
    let total_stake: u64 = stakes.iter().sum();

    if total_stake == 0 {
        // Equal weight if no stake (shouldn't happen in practice)
        let equal_weight = 1.0 / stakes.len() as f32;
        let weights: Vec<f32> = vec![equal_weight; stakes.len()];
        return Tensor::<B, 1>::from_floats(weights.as_slice(), device);
    }

    // Convert to f64 for precision, then to f32
    let weights: Vec<f32> =
        stakes.iter().map(|&s| (s as f64 / total_stake as f64) as f32).collect();

    Tensor::<B, 1>::from_floats(weights.as_slice(), device)
}

/// Select top-k models based on stake-weighted distance to target.
///
/// # Algorithm
/// 1. Compute squared Euclidean distance from target to each model embedding
/// 2. Weight by inverse voting power: `weighted_score = dist_sq / voting_power`
/// 3. Select k models with lowest weighted scores
///
/// # Arguments
/// * `target` - Target embedding as SomaTensor
/// * `data` - Pre-computed model selection data
/// * `k` - Number of models to select
///
/// # Returns
/// Vec of ModelMatch sorted by weighted_score (ascending - best first)
pub fn select_models<B: Backend>(
    target: &SomaTensor,
    data: &ModelSelectionData<B>,
    k: usize,
) -> Vec<ModelMatch> {
    let num_models = data.num_models();
    let dim = data.embedding_dim();

    if num_models == 0 || k == 0 {
        return vec![];
    }

    let k = k.min(num_models);

    // Convert target to Burn tensor
    let target_vec = target.to_vec();
    assert_eq!(target_vec.len(), dim, "Target dimension must match model embeddings");

    let target_tensor =
        Tensor::<B, 1>::from_floats(target_vec.as_slice(), &data.device).unsqueeze::<2>();
    // Shape: [1, dim] -> broadcast to [num_models, dim]
    let target_expanded = target_tensor.expand([num_models, dim]);

    // Compute squared distances: ||target - embedding||²
    let diff = target_expanded - data.embeddings.clone();
    let dist_sq: Tensor<B, 1> = diff.clone().mul(diff).sum_dim(1).squeeze_dim(1);

    // Compute weighted scores: dist_sq / voting_power
    // Add epsilon to avoid division by zero (though voting_power should never be zero)
    let eps = Tensor::<B, 1>::from_floats([1e-10], &data.device).expand([num_models]);
    let voting_power_safe = data.voting_power.clone().add(eps);
    let weighted_scores = dist_sq.clone().div(voting_power_safe);

    // Find indices of k smallest weighted scores
    // Burn's argsort is ascending by default
    let sorted_indices = weighted_scores.clone().argsort(0);

    // Extract top-k indices and gather values
    let topk_indices: Vec<i64> =
        sorted_indices.slice(0..k).into_data().to_vec::<i64>().expect("indices should be i64");

    let dist_sq_data: Vec<f32> = dist_sq.into_data().to_vec::<f32>().expect("f32");
    let weighted_data: Vec<f32> = weighted_scores.into_data().to_vec::<f32>().expect("f32");

    // Build results
    topk_indices
        .into_iter()
        .map(|idx| {
            let idx = idx as usize;
            ModelMatch {
                model_id: data.model_ids[idx],
                distance_sq: dist_sq_data[idx],
                weighted_score: weighted_data[idx],
            }
        })
        .collect()
}

/// Select models for multiple targets in a batch.
///
/// More efficient than calling `select_models` repeatedly when processing
/// many targets with the same model set.
///
/// # Arguments
/// * `targets` - Target embeddings
/// * `data` - Pre-computed model selection data
/// * `k` - Number of models to select per target
///
/// # Returns
/// Vec of Vec<ModelMatch>, one per target
pub fn batch_select_models<B: Backend>(
    targets: &[SomaTensor],
    data: &ModelSelectionData<B>,
    k: usize,
) -> Vec<Vec<ModelMatch>> {
    if targets.is_empty() || data.num_models() == 0 || k == 0 {
        return vec![vec![]; targets.len()];
    }

    // For now, iterate over targets. Could be optimized to fully vectorize.
    targets.iter().map(|target| select_models(target, data, k)).collect()
}

/// Convenience function using NdArray backend (CPU, deterministic).
///
/// This is the primary entry point for model selection in the system.
pub fn select_models_weighted(
    target: &SomaTensor,
    models: Vec<(ModelId, SomaTensor, u64)>,
    k: usize,
) -> Vec<ModelMatch> {
    if models.is_empty() || k == 0 {
        return vec![];
    }

    type B = NdArray;
    let device: <B as Backend>::Device = Default::default();
    let data = ModelSelectionData::<B>::new(models, &device);
    select_models(target, &data, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectID;

    type TestBackend = NdArray;

    fn device() -> <TestBackend as Backend>::Device {
        Default::default()
    }

    fn make_model_id(n: u8) -> ModelId {
        ObjectID::new([n; 32])
    }

    #[test]
    fn test_select_models_basic() {
        let models = vec![
            (make_model_id(0), SomaTensor::new(vec![0.0, 1.0], vec![2]), 100),
            (make_model_id(1), SomaTensor::new(vec![1.0, 0.0], vec![2]), 100),
            (make_model_id(2), SomaTensor::new(vec![10.0, 10.0], vec![2]), 100),
        ];

        let target = SomaTensor::new(vec![0.5, 0.5], vec![2]);

        let results = select_models_weighted(&target, models, 2);

        assert_eq!(results.len(), 2);
        // Model 0 and 1 should be selected (closest to target)
        // Model 2 is far away
        let selected_ids: Vec<ModelId> = results.iter().map(|m| m.model_id).collect();
        assert!(selected_ids.contains(&make_model_id(0)));
        assert!(selected_ids.contains(&make_model_id(1)));
        assert!(!selected_ids.contains(&make_model_id(2)));
    }

    #[test]
    fn test_stake_weighting() {
        // Two models at same distance from target
        let models = vec![
            (make_model_id(0), SomaTensor::new(vec![1.0, 0.0], vec![2]), 100),
            (make_model_id(1), SomaTensor::new(vec![-1.0, 0.0], vec![2]), 1000), // 10x stake
        ];

        let target = SomaTensor::new(vec![0.0, 0.0], vec![2]);

        let results = select_models_weighted(&target, models, 2);

        assert_eq!(results.len(), 2);
        // Model 1 should be first (higher stake = higher voting power = lower weighted score)
        assert_eq!(results[0].model_id, make_model_id(1));
        assert_eq!(results[1].model_id, make_model_id(0));

        // Verify the math:
        // dist² = 1.0 for both
        // voting_power: model 0 = 100/1100 = 0.0909, model 1 = 1000/1100 = 0.909
        // weighted_score: model 0 = 1.0 / 0.0909 = 11.0, model 1 = 1.0 / 0.909 = 1.1
        assert!(results[0].weighted_score < results[1].weighted_score);
    }

    #[test]
    fn test_normalized_voting_power_sums_to_one() {
        let stakes = vec![100, 200, 700];
        let device: <TestBackend as Backend>::Device = Default::default();
        let vp = compute_normalized_voting_power::<TestBackend>(&stakes, &device);

        let sum: f32 = vp.into_data().to_vec::<f32>().expect("f32").iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Voting power should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_voting_power_proportional() {
        let stakes = vec![100, 200, 700];
        let device: <TestBackend as Backend>::Device = Default::default();
        let vp = compute_normalized_voting_power::<TestBackend>(&stakes, &device);
        let vp_vec: Vec<f32> = vp.into_data().to_vec::<f32>().expect("f32");

        // 10%, 20%, 70%
        assert!((vp_vec[0] - 0.1).abs() < 1e-6);
        assert!((vp_vec[1] - 0.2).abs() < 1e-6);
        assert!((vp_vec[2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_empty_models() {
        let target = SomaTensor::new(vec![0.0, 0.0], vec![2]);
        let results = select_models_weighted(&target, vec![], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_k_larger_than_models() {
        let models = vec![
            (make_model_id(0), SomaTensor::new(vec![0.0, 0.0], vec![2]), 100),
            (make_model_id(1), SomaTensor::new(vec![1.0, 1.0], vec![2]), 100),
        ];

        let target = SomaTensor::new(vec![0.0, 0.0], vec![2]);

        // Request k=10 but only 2 models
        let results = select_models_weighted(&target, models, 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_model_selection_data_reuse() {
        let device = device();
        let models = vec![
            (make_model_id(0), SomaTensor::new(vec![0.0, 0.0], vec![2]), 100),
            (make_model_id(1), SomaTensor::new(vec![1.0, 0.0], vec![2]), 100),
            (make_model_id(2), SomaTensor::new(vec![0.0, 1.0], vec![2]), 100),
        ];

        let data = ModelSelectionData::<TestBackend>::new(models, &device);

        // Select for target at origin - should prefer model 0
        let target1 = SomaTensor::new(vec![0.0, 0.0], vec![2]);
        let results1 = select_models(&target1, &data, 1);
        assert_eq!(results1[0].model_id, make_model_id(0));

        // Select for target at (1, 0) - should prefer model 1
        let target2 = SomaTensor::new(vec![1.0, 0.0], vec![2]);
        let results2 = select_models(&target2, &data, 1);
        assert_eq!(results2[0].model_id, make_model_id(1));

        // Select for target at (0, 1) - should prefer model 2
        let target3 = SomaTensor::new(vec![0.0, 1.0], vec![2]);
        let results3 = select_models(&target3, &data, 1);
        assert_eq!(results3[0].model_id, make_model_id(2));
    }

    #[test]
    fn test_batch_select() {
        let device = device();
        let models = vec![
            (make_model_id(0), SomaTensor::new(vec![0.0, 0.0], vec![2]), 100),
            (make_model_id(1), SomaTensor::new(vec![1.0, 1.0], vec![2]), 100),
        ];

        let data = ModelSelectionData::<TestBackend>::new(models, &device);

        let targets = vec![
            SomaTensor::new(vec![0.0, 0.0], vec![2]),
            SomaTensor::new(vec![1.0, 1.0], vec![2]),
        ];

        let results = batch_select_models(&targets, &data, 1);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].model_id, make_model_id(0)); // Target at origin prefers model 0
        assert_eq!(results[1][0].model_id, make_model_id(1)); // Target at (1,1) prefers model 1
    }

    #[test]
    fn test_distance_affects_selection() {
        // Model with 10x stake but further away should lose to closer model
        let models = vec![
            (make_model_id(0), SomaTensor::new(vec![0.1, 0.0], vec![2]), 100), // Close, low stake
            (make_model_id(1), SomaTensor::new(vec![10.0, 0.0], vec![2]), 1000), // Far, high stake
        ];

        let target = SomaTensor::new(vec![0.0, 0.0], vec![2]);
        let results = select_models_weighted(&target, models, 2);

        // Model 0: dist² = 0.01, voting_power = 100/1100 = 0.0909
        //          weighted = 0.01 / 0.0909 = 0.11
        // Model 1: dist² = 100, voting_power = 1000/1100 = 0.909
        //          weighted = 100 / 0.909 = 110
        // Model 0 should win despite lower stake because it's much closer
        assert_eq!(results[0].model_id, make_model_id(0));
    }
}
