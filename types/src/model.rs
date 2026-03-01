// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::committee::EpochId;
use crate::crypto::DecryptionKey;
use crate::digests::{DecryptionKeyCommitment, EmbeddingCommitment, ModelWeightsCommitment};
use crate::metadata::Manifest;
use crate::object::ObjectID;
use crate::system_state::staking::StakingPool;
use crate::tensor::SomaTensor;

/// Version identifier for model architecture. Protocol config controls the current version.
pub type ArchitectureVersion = u64;

/// Type alias: models are identified by their ObjectID in the ModelRegistry maps.
pub type ModelId = ObjectID;

/// Download information for encrypted model weights.
/// Uses the existing Manifest infrastructure (URL + checksum + size).
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct ModelWeightsManifest {
    /// Existing type: URL + Metadata(checksum, size)
    pub manifest: Manifest,
    /// AES-256 decryption key for the encrypted weights
    pub decryption_key: DecryptionKey,
}

/// A pending update to an active model's weights (commit-reveal).
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct PendingModelUpdate {
    pub manifest: Manifest,
    pub weights_commitment: ModelWeightsCommitment,
    pub embedding_commitment: EmbeddingCommitment,
    pub decryption_key_commitment: DecryptionKeyCommitment,
    pub commit_epoch: EpochId,
}

/// A registered model in the SOMA data submission system.
///
/// Models go through a commit-reveal lifecycle:
/// - **Committed**: `decryption_key.is_none()` && `deactivation_epoch == None`
/// - **Active**: `decryption_key.is_some()` && `deactivation_epoch == None`
/// - **PendingUpdate**: Active with `pending_update.is_some()`
/// - **Inactive**: `staking_pool.deactivation_epoch.is_some()`
///
/// No explicit `status` enum -- derive from field state (same pattern as validators).
/// The `Model` struct does not store its own ID; it is identified by `ModelId`
/// (an `ObjectID`) as the key in `ModelRegistry` maps.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelV1 {
    /// Owner address (the account that committed/revealed this model)
    pub owner: SomaAddress,
    /// Architecture version (must match protocol config at commit time)
    pub architecture_version: ArchitectureVersion,

    // -- Commit state --
    /// Manifest for the encrypted weights (URL + checksum + size). Set at commit time.
    pub manifest: Manifest,
    /// Commitment to the decrypted model weights
    pub weights_commitment: ModelWeightsCommitment,
    /// Commitment to the model embedding (hash of BCS-serialized SomaTensor)
    pub embedding_commitment: EmbeddingCommitment,
    /// Commitment to the decryption key (hash of key bytes), verified at reveal
    pub decryption_key_commitment: DecryptionKeyCommitment,
    /// Epoch in which CommitModel was executed
    pub commit_epoch: EpochId,

    // -- Reveal state (None while Committed) --
    /// AES-256 decryption key, set when RevealModel is executed
    pub decryption_key: Option<DecryptionKey>,

    // -- Model embedding for stake-weighted KNN selection --
    /// The model's embedding vector in the shared embedding space.
    /// Used for stake-weighted KNN model selection: targets prefer nearby models
    /// weighted by normalized stake (voting power). This encourages specialization
    /// and reputation building within specific regions of the embedding space.
    /// Set during reveal (verified against embedding_commitment from commit).
    pub embedding: Option<SomaTensor>,

    // -- Staking (reuses existing StakingPool, identical to validators) --
    pub staking_pool: StakingPool,

    // -- Commission (mirrors validator commission) --
    /// Current epoch commission rate in basis points (max 10000)
    pub commission_rate: u64,
    /// Staged for next epoch
    pub next_epoch_commission_rate: u64,

    // -- Pending update (None if no update in flight) --
    pub pending_update: Option<PendingModelUpdate>,
}

impl ModelV1 {
    /// Returns true if this model is in the committed (pre-reveal) state.
    pub fn is_committed(&self) -> bool {
        self.decryption_key.is_none() && self.staking_pool.deactivation_epoch.is_none()
    }

    /// Returns true if this model has been revealed and is active.
    pub fn is_active(&self) -> bool {
        self.decryption_key.is_some() && self.staking_pool.deactivation_epoch.is_none()
    }

    /// Returns true if this model has been deactivated (slashed or owner-deactivated).
    pub fn is_inactive(&self) -> bool {
        self.staking_pool.deactivation_epoch.is_some()
    }

    /// Returns true if this model has a pending weight update.
    pub fn has_pending_update(&self) -> bool {
        self.pending_update.is_some()
    }

    /// Returns true if this model is eligible for stake-weighted selection.
    /// A model is selectable if it's active and has an embedding.
    pub fn is_selectable(&self) -> bool {
        self.is_active() && self.embedding.is_some()
    }

    /// Get the model's stake (soma_balance in the staking pool).
    pub fn stake(&self) -> u64 {
        self.staking_pool.soma_balance
    }
}
