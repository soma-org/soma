//! Target state management within SystemState.
//!
//! `TargetState` is a lightweight struct stored in SystemState that tracks:
//! - Current difficulty thresholds for new targets
//! - Per-epoch counters for hits and targets
//! - Hits-per-epoch EMA for difficulty adjustment (absolute count, not percentage)
//! - Reward per target for the current epoch
//!
//! The actual Target objects are shared objects stored separately from SystemState.
//! This design prevents SystemState from becoming a contention bottleneck.

use crate::tensor::SomaTensor;
use serde::{Deserialize, Serialize};

/// Lightweight coordination state for target generation.
///
/// This struct is stored inside `SystemState` but the actual `Target` objects
/// are separate shared objects. This separation prevents contention on SystemState
/// when multiple submissions target different targets.
///
/// Uses per-epoch counters and an EMA of absolute hit counts for difficulty adjustment.
/// This avoids needing consensus timestamps during execution, which would create
/// issues during checkpoint replay (fullnodes don't have consensus timestamps).
///
/// The `hits_ema` provides a smoothed view of hits per epoch (absolute count)
/// for stable difficulty adjustments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetState {
    /// Current distance threshold for new targets (cosine distance as scalar SomaTensor).
    /// Lower distance = closer to target center = harder.
    /// This threshold is dynamically adjusted based on hits_ema vs target_hits_per_epoch.
    pub distance_threshold: SomaTensor,

    /// Number of targets generated this epoch (initial + spawn-on-fill replacements).
    /// Reset to 0 at epoch boundary after difficulty adjustment.
    pub targets_generated_this_epoch: u64,

    /// Number of successful hits (filled targets) this epoch.
    /// Reset to 0 at epoch boundary after difficulty adjustment.
    pub hits_this_epoch: u64,

    /// Exponential Moving Average of hits per epoch (absolute count).
    /// Updated at each epoch boundary: EMA = decay * EMA + (1-decay) * hits_this_epoch.
    /// 0 indicates bootstrap mode (no data yet).
    pub hits_ema: u64,

    /// Reward per target for the current epoch (in shannons).
    /// Calculated at epoch boundary from target_allocation / estimated_targets.
    /// Used for both initial targets and spawn-on-fill replacements.
    pub reward_per_target: u64,
}

impl Default for TargetState {
    fn default() -> Self {
        Self {
            distance_threshold: SomaTensor::scalar(0.0),
            targets_generated_this_epoch: 0,
            hits_this_epoch: 0,
            hits_ema: 0,
            reward_per_target: 0,
        }
    }
}

impl TargetState {
    /// Create a new TargetState with initial thresholds from protocol config.
    ///
    /// Epoch counters start at 0 and are reset at each epoch boundary.
    /// `hits_ema` starts at 0 (bootstrap mode).
    /// `reward_per_target` is calculated separately after construction.
    pub fn new(initial_distance_threshold: SomaTensor) -> Self {
        Self {
            distance_threshold: initial_distance_threshold,
            targets_generated_this_epoch: 0,
            hits_this_epoch: 0,
            hits_ema: 0,
            reward_per_target: 0,
        }
    }

    /// Reset epoch counters at the start of a new epoch.
    /// Called after difficulty adjustment is performed.
    pub fn reset_epoch_counters(&mut self) {
        self.targets_generated_this_epoch = 0;
        self.hits_this_epoch = 0;
    }

    /// Record that a new target was generated.
    pub fn record_target_generated(&mut self) {
        self.targets_generated_this_epoch += 1;
    }

    /// Record a successful hit (target filled).
    pub fn record_hit(&mut self) {
        self.hits_this_epoch += 1;
    }

    /// Update the hits-per-epoch EMA at epoch boundary.
    /// `decay_bps` is the decay factor in basis points (e.g., 9000 = 0.9).
    /// Returns the new EMA value.
    ///
    /// EMA formula: new_ema = decay * old_ema + (1 - decay) * hits_this_epoch
    /// In bps arithmetic: new_ema = (decay_bps * old_ema + (10000 - decay_bps) * hits_this_epoch) / 10000
    ///
    /// If `hits_ema` is 0 (bootstrap), we initialize it to the current hit count.
    pub fn update_hits_ema(&mut self, decay_bps: u64) -> u64 {
        let current_hits = self.hits_this_epoch;
        if self.hits_ema == 0 {
            // Bootstrap: initialize to current count
            self.hits_ema = current_hits;
        } else {
            // EMA update: decay * old + (1-decay) * new
            let weight_old = decay_bps;
            let weight_new = 10000 - decay_bps;
            self.hits_ema = (weight_old * self.hits_ema + weight_new * current_hits) / 10000;
        }
        self.hits_ema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_state_default() {
        let state = TargetState::default();
        assert_eq!(state.distance_threshold.as_scalar(), 0.0);
        assert_eq!(state.targets_generated_this_epoch, 0);
        assert_eq!(state.hits_this_epoch, 0);
        assert_eq!(state.hits_ema, 0);
        assert_eq!(state.reward_per_target, 0);
    }

    #[test]
    fn test_target_state_new() {
        let state = TargetState::new(SomaTensor::scalar(0.5));
        assert_eq!(state.distance_threshold.as_scalar(), 0.5);
        assert_eq!(state.targets_generated_this_epoch, 0);
        assert_eq!(state.hits_this_epoch, 0);
        assert_eq!(state.hits_ema, 0);
        assert_eq!(state.reward_per_target, 0);
    }

    #[test]
    fn test_target_state_counters() {
        let mut state = TargetState::new(SomaTensor::scalar(0.5));

        // Record some targets and hits
        state.record_target_generated();
        state.record_target_generated();
        state.record_target_generated();
        state.record_hit();
        state.record_hit();

        assert_eq!(state.targets_generated_this_epoch, 3);
        assert_eq!(state.hits_this_epoch, 2);

        // Reset counters
        state.reset_epoch_counters();
        assert_eq!(state.targets_generated_this_epoch, 0);
        assert_eq!(state.hits_this_epoch, 0);
    }

    #[test]
    fn test_hits_ema_bootstrap() {
        let mut state = TargetState::new(SomaTensor::scalar(0.5));

        // Record 8 hits this epoch
        for _ in 0..8 {
            state.record_hit();
        }

        // Bootstrap: EMA starts at 0, should be set to current hit count
        assert_eq!(state.hits_ema, 0);
        let ema = state.update_hits_ema(9000); // 90% decay
        assert_eq!(ema, 8); // Initialized to 8 hits
        assert_eq!(state.hits_ema, 8);
    }

    #[test]
    fn test_hits_ema_update() {
        let mut state = TargetState::new(SomaTensor::scalar(0.5));
        state.hits_ema = 20; // Previous EMA: 20 hits/epoch

        // This epoch: 10 hits
        for _ in 0..10 {
            state.record_hit();
        }

        // EMA update with 90% decay
        // new_ema = (9000 * 20 + 1000 * 10) / 10000 = (180000 + 10000) / 10000 = 19
        let ema = state.update_hits_ema(9000);
        assert_eq!(ema, 19);
    }

    #[test]
    fn test_hits_ema_no_hits() {
        let mut state = TargetState::new(SomaTensor::scalar(0.5));
        state.hits_ema = 20; // Previous EMA: 20 hits/epoch

        // No hits this epoch
        // new_ema = (9000 * 20 + 1000 * 0) / 10000 = 180000 / 10000 = 18
        let ema = state.update_hits_ema(9000);
        assert_eq!(ema, 18);
    }

    #[test]
    fn test_hits_ema_bootstrap_zero_hits() {
        let mut state = TargetState::new(SomaTensor::scalar(0.5));

        // No hits, bootstrap mode: stays at 0
        assert_eq!(state.hits_ema, 0);
        let ema = state.update_hits_ema(9000);
        assert_eq!(ema, 0); // Still in bootstrap mode
    }
}
