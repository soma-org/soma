//! Target state management within SystemState.
//!
//! `TargetState` is a lightweight struct stored in SystemState that tracks:
//! - Current difficulty thresholds for new targets
//! - Per-epoch counters for hits and targets (used for difficulty adjustment)
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
/// Uses per-epoch counters for difficulty adjustment rather than time-based EMAs.
/// This avoids needing consensus timestamps during execution, which would create
/// issues during checkpoint replay (fullnodes don't have consensus timestamps).
///
/// The hit_rate_ema_bps provides a smoothed view of hit rate across epochs
/// for more stable difficulty adjustments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetState {
    /// Current distance threshold for new targets (cosine distance as scalar SomaTensor).
    /// Lower distance = closer to target center = better.
    /// This threshold is dynamically adjusted based on hit_rate_ema_bps.
    pub distance_threshold: SomaTensor,

    /// Number of targets generated this epoch (initial + spawn-on-fill replacements).
    /// Reset to 0 at epoch boundary after difficulty adjustment.
    pub targets_generated_this_epoch: u64,

    /// Number of successful hits (filled targets) this epoch.
    /// Reset to 0 at epoch boundary after difficulty adjustment.
    /// hit_rate = hits_this_epoch / targets_generated_this_epoch
    pub hits_this_epoch: u64,

    /// Exponential Moving Average of hit rate in basis points (0-10000).
    /// Updated at each epoch boundary: EMA = decay * EMA + (1-decay) * current_rate.
    /// 0 indicates bootstrap mode (no data yet).
    pub hit_rate_ema_bps: u64,

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
            hit_rate_ema_bps: 0,
            reward_per_target: 0,
        }
    }
}

impl TargetState {
    /// Create a new TargetState with initial thresholds from protocol config.
    ///
    /// Epoch counters start at 0 and are reset at each epoch boundary.
    /// `hit_rate_ema_bps` starts at 0 (bootstrap mode).
    /// `reward_per_target` is calculated separately after construction.
    pub fn new(initial_distance_threshold: SomaTensor) -> Self {
        Self {
            distance_threshold: initial_distance_threshold,
            targets_generated_this_epoch: 0,
            hits_this_epoch: 0,
            hit_rate_ema_bps: 0,
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

    /// Calculate the hit rate for this epoch in basis points (0-10000).
    /// Returns None if no targets were generated.
    pub fn hit_rate_bps(&self) -> Option<u64> {
        if self.targets_generated_this_epoch == 0 {
            None
        } else {
            Some((self.hits_this_epoch * 10000) / self.targets_generated_this_epoch)
        }
    }

    /// Update the hit rate EMA at epoch boundary.
    /// `decay_bps` is the decay factor in basis points (e.g., 9000 = 0.9).
    /// Returns the new EMA value.
    ///
    /// EMA formula: new_ema = decay * old_ema + (1 - decay) * current_rate
    /// In bps arithmetic: new_ema = (decay_bps * old_ema + (10000 - decay_bps) * current_rate) / 10000
    ///
    /// If `hit_rate_ema_bps` is 0 (bootstrap), we initialize it to the current rate.
    pub fn update_hit_rate_ema(&mut self, decay_bps: u64) -> u64 {
        if let Some(current_rate_bps) = self.hit_rate_bps() {
            if self.hit_rate_ema_bps == 0 {
                // Bootstrap: initialize to current rate
                self.hit_rate_ema_bps = current_rate_bps;
            } else {
                // EMA update: decay * old + (1-decay) * new
                let weight_old = decay_bps;
                let weight_new = 10000 - decay_bps;
                self.hit_rate_ema_bps =
                    (weight_old * self.hit_rate_ema_bps + weight_new * current_rate_bps) / 10000;
            }
        }
        // If no targets generated, keep the EMA unchanged
        self.hit_rate_ema_bps
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
        assert_eq!(state.hit_rate_ema_bps, 0);
        assert_eq!(state.reward_per_target, 0);
    }

    #[test]
    fn test_target_state_new() {
        let state = TargetState::new(SomaTensor::scalar(0.5));
        assert_eq!(state.distance_threshold.as_scalar(), 0.5);
        assert_eq!(state.targets_generated_this_epoch, 0);
        assert_eq!(state.hits_this_epoch, 0);
        assert_eq!(state.hit_rate_ema_bps, 0);
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

        // Check hit rate (2/3 = 6666 bps)
        let rate_bps = state.hit_rate_bps().unwrap();
        assert_eq!(rate_bps, 6666);

        // Reset counters
        state.reset_epoch_counters();
        assert_eq!(state.targets_generated_this_epoch, 0);
        assert_eq!(state.hits_this_epoch, 0);
        assert!(state.hit_rate_bps().is_none());
    }

    #[test]
    fn test_hit_rate_ema_bootstrap() {
        let mut state = TargetState::new(SomaTensor::scalar(0.5));

        // Record 8/10 = 80% hit rate
        for _ in 0..10 {
            state.record_target_generated();
        }
        for _ in 0..8 {
            state.record_hit();
        }

        // Bootstrap: EMA starts at 0, should be set to current rate
        assert_eq!(state.hit_rate_ema_bps, 0);
        let ema = state.update_hit_rate_ema(9000); // 90% decay
        assert_eq!(ema, 8000); // 80% = 8000 bps
        assert_eq!(state.hit_rate_ema_bps, 8000);
    }

    #[test]
    fn test_hit_rate_ema_update() {
        let mut state = TargetState::new(SomaTensor::scalar(0.5));
        state.hit_rate_ema_bps = 8000; // 80%

        // Epoch with 60% hit rate
        for _ in 0..10 {
            state.record_target_generated();
        }
        for _ in 0..6 {
            state.record_hit();
        }

        // EMA update with 90% decay
        // new_ema = (9000 * 8000 + 1000 * 6000) / 10000 = (72000000 + 6000000) / 10000 = 7800
        let ema = state.update_hit_rate_ema(9000);
        assert_eq!(ema, 7800);
    }

    #[test]
    fn test_hit_rate_ema_no_targets() {
        let mut state = TargetState::new(SomaTensor::scalar(0.5));
        state.hit_rate_ema_bps = 8000; // 80%

        // No targets generated this epoch
        // EMA should remain unchanged
        let ema = state.update_hit_rate_ema(9000);
        assert_eq!(ema, 8000);
    }
}
