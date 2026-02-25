// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::model::{ModelId, ModelV1};
use crate::object::ObjectID;

/// Registry of all models in the SOMA data submission system.
///
/// Tracks active, pending (committed but not yet revealed), and inactive models.
/// Mirrors the `ValidatorSet` pattern: models are keyed by `ModelId` (an `ObjectID`),
/// staking pools are mapped for stake routing, and report records follow the same
/// quorum-threshold logic as validator reports.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRegistry {
    /// Active models: revealed and eligible for target selection.
    pub active_models: BTreeMap<ModelId, ModelV1>,

    /// Pending models: committed, awaiting reveal in the next epoch.
    pub pending_models: BTreeMap<ModelId, ModelV1>,

    /// Maps staking pool ObjectID -> ModelId (for stake routing, mirrors ValidatorSet pattern).
    pub staking_pool_mappings: BTreeMap<ObjectID, ModelId>,

    /// Inactive models: pool kept alive for delegator withdrawals
    /// (mirrors ValidatorSet.inactive_validators).
    pub inactive_models: BTreeMap<ModelId, ModelV1>,

    /// Sum of all active model staking pool balances (cache for weighted selection).
    pub total_model_stake: u64,

    /// Report records: model_id -> set of reporter validator addresses.
    /// Mirrors validator_report_records pattern exactly.
    /// Cleared at epoch boundary after processing.
    pub model_report_records: BTreeMap<ModelId, BTreeSet<SomaAddress>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            active_models: BTreeMap::new(),
            pending_models: BTreeMap::new(),
            staking_pool_mappings: BTreeMap::new(),
            inactive_models: BTreeMap::new(),
            total_model_stake: 0,
            model_report_records: BTreeMap::new(),
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
