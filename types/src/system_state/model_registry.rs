// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::model::{Model, ModelId, ModelStateV1};
use crate::object::ObjectID;

/// Registry of all models in the SOMA data submission system.
///
/// Uses a single `models` map keyed by `ModelId`. The lifecycle state is
/// encoded in the `Model` enum (`Created`, `Pending`, `Active`, `Inactive`).
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRegistry {
    /// All models, keyed by ModelId. State is encoded in the Model enum.
    pub models: BTreeMap<ModelId, Model>,

    /// Maps staking pool ObjectID -> ModelId (for stake routing, mirrors ValidatorSet pattern).
    pub staking_pool_mappings: BTreeMap<ObjectID, ModelId>,

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
            models: BTreeMap::new(),
            staking_pool_mappings: BTreeMap::new(),
            total_model_stake: 0,
            model_report_records: BTreeMap::new(),
        }
    }

    /// Iterator over active models only.
    pub fn active_models(&self) -> impl Iterator<Item = (&ModelId, &crate::model::ActiveModel)> {
        self.models.iter().filter_map(|(id, model)| match model {
            Model::V1(ModelStateV1::Active(m)) => Some((id, m)),
            _ => None,
        })
    }

    /// Iterator over pending models only.
    pub fn pending_models(&self) -> impl Iterator<Item = (&ModelId, &crate::model::PendingModel)> {
        self.models.iter().filter_map(|(id, model)| match model {
            Model::V1(ModelStateV1::Pending(m)) => Some((id, m)),
            _ => None,
        })
    }

    /// Iterator over created models only.
    pub fn created_models(&self) -> impl Iterator<Item = (&ModelId, &crate::model::CreatedModel)> {
        self.models.iter().filter_map(|(id, model)| match model {
            Model::V1(ModelStateV1::Created(m)) => Some((id, m)),
            _ => None,
        })
    }

    /// Iterator over inactive models only.
    pub fn inactive_models(
        &self,
    ) -> impl Iterator<Item = (&ModelId, &crate::model::InactiveModel)> {
        self.models.iter().filter_map(|(id, model)| match model {
            Model::V1(ModelStateV1::Inactive(m)) => Some((id, m)),
            _ => None,
        })
    }

    /// Check if a model is active.
    pub fn is_active(&self, model_id: &ModelId) -> bool {
        self.models.get(model_id).map_or(false, |m| m.is_active())
    }

    /// Number of active models.
    pub fn active_model_count(&self) -> usize {
        self.active_models().count()
    }

    /// Check if there are any active models.
    pub fn has_active_models(&self) -> bool {
        self.models.values().any(|m| m.is_active())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
