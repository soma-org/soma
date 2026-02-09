//! Target proto conversions and merge implementations.

use super::*;
use crate::utils::{field::FieldMaskTree, merge::Merge};
use types::target::{Target as DomainTarget, TargetStatus};

//
// Target
//

impl Merge<&DomainTarget> for Target {
    fn merge(&mut self, source: &DomainTarget, mask: &FieldMaskTree) {
        // Note: id is set separately by the caller (from object metadata)
        // as the domain Target struct doesn't contain its own id

        if mask.contains(Self::EMBEDDING_FIELD.name) {
            self.embedding = source.embedding.iter().copied().collect();
        }

        if mask.contains(Self::MODEL_IDS_FIELD.name) {
            self.model_ids = source.model_ids.iter().map(|id| id.to_hex()).collect();
        }

        if mask.contains(Self::DISTANCE_THRESHOLD_FIELD.name) {
            self.distance_threshold = Some(source.distance_threshold);
        }

        if mask.contains(Self::REWARD_POOL_FIELD.name) {
            self.reward_pool = Some(source.reward_pool);
        }

        if mask.contains(Self::GENERATION_EPOCH_FIELD.name) {
            self.generation_epoch = Some(source.generation_epoch);
        }

        // Note: generation_timestamp_ms field removed from domain Target

        if mask.contains(Self::STATUS_FIELD.name) {
            self.status = Some(match &source.status {
                TargetStatus::Open => "open".to_string(),
                TargetStatus::Filled { .. } => "filled".to_string(),
                TargetStatus::Claimed => "claimed".to_string(),
            });
        }

        if mask.contains(Self::FILL_EPOCH_FIELD.name) {
            if let TargetStatus::Filled { fill_epoch } = &source.status {
                self.fill_epoch = Some(*fill_epoch);
            }
        }

        if mask.contains(Self::MINER_FIELD.name) {
            self.miner = source.miner.map(|addr| addr.to_string());
        }

        if mask.contains(Self::WINNING_MODEL_ID_FIELD.name) {
            self.winning_model_id = source.winning_model_id.map(|id| id.to_hex());
        }

        if mask.contains(Self::WINNING_MODEL_OWNER_FIELD.name) {
            self.winning_model_owner = source.winning_model_owner.map(|addr| addr.to_string());
        }

        if mask.contains(Self::BOND_AMOUNT_FIELD.name) {
            self.bond_amount = Some(source.bond_amount);
        }
    }
}

impl Merge<&Target> for Target {
    fn merge(&mut self, source: &Target, mask: &FieldMaskTree) {
        if mask.contains(Self::ID_FIELD.name) {
            self.id = source.id.clone();
        }

        if mask.contains(Self::EMBEDDING_FIELD.name) {
            self.embedding = source.embedding.clone();
        }

        if mask.contains(Self::MODEL_IDS_FIELD.name) {
            self.model_ids = source.model_ids.clone();
        }

        if mask.contains(Self::DISTANCE_THRESHOLD_FIELD.name) {
            self.distance_threshold = source.distance_threshold;
        }

        if mask.contains(Self::REWARD_POOL_FIELD.name) {
            self.reward_pool = source.reward_pool;
        }

        if mask.contains(Self::GENERATION_EPOCH_FIELD.name) {
            self.generation_epoch = source.generation_epoch;
        }

        // Note: generation_timestamp_ms field removed

        if mask.contains(Self::STATUS_FIELD.name) {
            self.status = source.status.clone();
        }

        if mask.contains(Self::FILL_EPOCH_FIELD.name) {
            self.fill_epoch = source.fill_epoch;
        }

        if mask.contains(Self::MINER_FIELD.name) {
            self.miner = source.miner.clone();
        }

        if mask.contains(Self::WINNING_MODEL_ID_FIELD.name) {
            self.winning_model_id = source.winning_model_id.clone();
        }

        if mask.contains(Self::WINNING_MODEL_OWNER_FIELD.name) {
            self.winning_model_owner = source.winning_model_owner.clone();
        }

        if mask.contains(Self::BOND_AMOUNT_FIELD.name) {
            self.bond_amount = source.bond_amount;
        }
    }
}

/// Convert a domain Target to a proto Target with all fields populated.
impl From<&DomainTarget> for Target {
    fn from(source: &DomainTarget) -> Self {
        Self::merge_from(source, &FieldMaskTree::new_wildcard())
    }
}

/// Helper to create a Target proto with object ID set.
pub fn target_to_proto_with_id(
    target_id: &types::object::ObjectID,
    target: &DomainTarget,
    mask: &FieldMaskTree,
) -> Target {
    let mut proto = Target::default();

    if mask.contains(Target::ID_FIELD.name) {
        proto.id = Some(target_id.to_hex());
    }

    proto.merge(target, mask);
    proto
}
