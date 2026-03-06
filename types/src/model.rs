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

// ---------------------------------------------------------------------------
// Versioned Model enum
// ---------------------------------------------------------------------------

/// Versioned model envelope. Outer enum handles schema versioning.
/// To evolve the schema, add `V2(ModelStateV2)`, etc.
///
/// All serialization goes through this enum, so old nodes can deserialize
/// models created before a schema upgrade by matching the variant tag.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub enum Model {
    V1(ModelStateV1),
}

/// V1 model lifecycle state machine. Encodes states at the type level:
///
/// ```text
/// CreateModel ──> CommitModel ──> RevealModel
///   (Created)      (Pending)       (Active)
///
/// Active model update:
/// CommitModel ──> RevealModel
///   (sets pending_update on Active)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub enum ModelStateV1 {
    Created(CreatedModel),
    Pending(PendingModel),
    Active(ActiveModel),
    Inactive(InactiveModel),
}

/// A model that has been created but not yet committed.
/// Only economic setup (stake, commission) is done.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct CreatedModel {
    pub owner: SomaAddress,
    pub architecture_version: ArchitectureVersion,
    pub staking_pool: StakingPool,
    pub commission_rate: u64,
    pub next_epoch_commission_rate: u64,
    pub create_epoch: EpochId,
}

/// A model that has been committed but not yet revealed.
/// Cryptographic commitments are set; must reveal in commit_epoch + 1.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct PendingModel {
    pub owner: SomaAddress,
    pub architecture_version: ArchitectureVersion,
    pub staking_pool: StakingPool,
    pub commission_rate: u64,
    pub next_epoch_commission_rate: u64,
    pub manifest: Manifest,
    pub weights_commitment: ModelWeightsCommitment,
    pub embedding_commitment: EmbeddingCommitment,
    pub decryption_key_commitment: DecryptionKeyCommitment,
    pub commit_epoch: EpochId,
}

/// A fully revealed, active model eligible for target selection.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ActiveModel {
    pub owner: SomaAddress,
    pub architecture_version: ArchitectureVersion,
    pub staking_pool: StakingPool,
    pub commission_rate: u64,
    pub next_epoch_commission_rate: u64,
    pub manifest: Manifest,
    pub weights_commitment: ModelWeightsCommitment,
    pub embedding_commitment: EmbeddingCommitment,
    pub decryption_key_commitment: DecryptionKeyCommitment,
    /// AES-256 decryption key — always present on Active
    pub decryption_key: DecryptionKey,
    /// The model's embedding vector in the shared embedding space
    pub embedding: SomaTensor,
    /// Pending weight update (None if no update in flight)
    pub pending_update: Option<PendingModelUpdate>,
}

/// A deactivated model. Pool kept alive for delegator withdrawals.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct InactiveModel {
    pub owner: SomaAddress,
    pub architecture_version: ArchitectureVersion,
    pub staking_pool: StakingPool,
    pub commission_rate: u64,
    pub next_epoch_commission_rate: u64,
    pub manifest: Manifest,
    pub weights_commitment: ModelWeightsCommitment,
    pub embedding_commitment: EmbeddingCommitment,
    pub decryption_key_commitment: DecryptionKeyCommitment,
    pub decryption_key: DecryptionKey,
    pub embedding: SomaTensor,
}

// ---------------------------------------------------------------------------
// Model impl — delegates through V1 to the inner state
// ---------------------------------------------------------------------------

impl Model {
    /// Returns the owner address of this model regardless of state.
    pub fn owner(&self) -> SomaAddress {
        match self {
            Model::V1(s) => s.owner(),
        }
    }

    /// Get a reference to the staking pool.
    pub fn staking_pool(&self) -> &StakingPool {
        match self {
            Model::V1(s) => s.staking_pool(),
        }
    }

    /// Get a mutable reference to the staking pool.
    pub fn staking_pool_mut(&mut self) -> &mut StakingPool {
        match self {
            Model::V1(s) => s.staking_pool_mut(),
        }
    }

    /// Get the model's stake (soma_balance in the staking pool).
    pub fn stake(&self) -> u64 {
        self.staking_pool().soma_balance
    }

    /// Architecture version.
    pub fn architecture_version(&self) -> ArchitectureVersion {
        match self {
            Model::V1(s) => s.architecture_version(),
        }
    }

    /// Commission rate.
    pub fn commission_rate(&self) -> u64 {
        match self {
            Model::V1(s) => s.commission_rate(),
        }
    }

    /// Next epoch commission rate.
    pub fn next_epoch_commission_rate(&self) -> u64 {
        match self {
            Model::V1(s) => s.next_epoch_commission_rate(),
        }
    }

    /// Returns true if this model is in Created state.
    pub fn is_created(&self) -> bool {
        matches!(self, Model::V1(ModelStateV1::Created(_)))
    }

    /// Returns true if this model is in Pending (committed, awaiting reveal) state.
    pub fn is_pending(&self) -> bool {
        matches!(self, Model::V1(ModelStateV1::Pending(_)))
    }

    /// Returns true if this model is Active (revealed).
    pub fn is_active(&self) -> bool {
        matches!(self, Model::V1(ModelStateV1::Active(_)))
    }

    /// Returns true if this model is Inactive (deactivated/slashed).
    pub fn is_inactive(&self) -> bool {
        matches!(self, Model::V1(ModelStateV1::Inactive(_)))
    }

    /// Returns true if this model has a pending weight update.
    pub fn has_pending_update(&self) -> bool {
        match self {
            Model::V1(ModelStateV1::Active(m)) => m.pending_update.is_some(),
            _ => false,
        }
    }

    /// Returns true if this model is eligible for stake-weighted selection.
    pub fn is_selectable(&self) -> bool {
        self.is_active()
    }

    /// Get the commit epoch (for Pending models).
    pub fn commit_epoch(&self) -> Option<EpochId> {
        match self {
            Model::V1(ModelStateV1::Pending(m)) => Some(m.commit_epoch),
            _ => None,
        }
    }

    /// Get the create epoch (for Created models).
    pub fn create_epoch(&self) -> Option<EpochId> {
        match self {
            Model::V1(ModelStateV1::Created(m)) => Some(m.create_epoch),
            _ => None,
        }
    }

    /// Get the embedding (only available on Active and Inactive models).
    pub fn embedding(&self) -> Option<&SomaTensor> {
        match self {
            Model::V1(ModelStateV1::Active(m)) => Some(&m.embedding),
            Model::V1(ModelStateV1::Inactive(m)) => Some(&m.embedding),
            _ => None,
        }
    }

    /// Get the manifest (available on Pending, Active, Inactive).
    pub fn manifest(&self) -> Option<&Manifest> {
        match self {
            Model::V1(ModelStateV1::Pending(m)) => Some(&m.manifest),
            Model::V1(ModelStateV1::Active(m)) => Some(&m.manifest),
            Model::V1(ModelStateV1::Inactive(m)) => Some(&m.manifest),
            _ => None,
        }
    }

    /// Get the decryption key (only available on Active and Inactive models).
    pub fn decryption_key(&self) -> Option<&DecryptionKey> {
        match self {
            Model::V1(ModelStateV1::Active(m)) => Some(&m.decryption_key),
            Model::V1(ModelStateV1::Inactive(m)) => Some(&m.decryption_key),
            _ => None,
        }
    }

    /// Try to get a reference to the inner ActiveModel.
    pub fn as_active(&self) -> Option<&ActiveModel> {
        match self {
            Model::V1(ModelStateV1::Active(m)) => Some(m),
            _ => None,
        }
    }

    /// Try to get a mutable reference to the inner ActiveModel.
    pub fn as_active_mut(&mut self) -> Option<&mut ActiveModel> {
        match self {
            Model::V1(ModelStateV1::Active(m)) => Some(m),
            _ => None,
        }
    }

    /// Try to get a reference to the inner PendingModel.
    pub fn as_pending(&self) -> Option<&PendingModel> {
        match self {
            Model::V1(ModelStateV1::Pending(m)) => Some(m),
            _ => None,
        }
    }

    /// Try to get a reference to the inner CreatedModel.
    pub fn as_created(&self) -> Option<&CreatedModel> {
        match self {
            Model::V1(ModelStateV1::Created(m)) => Some(m),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelStateV1 impl — shared accessors across all states
// ---------------------------------------------------------------------------

impl ModelStateV1 {
    pub fn owner(&self) -> SomaAddress {
        match self {
            Self::Created(m) => m.owner,
            Self::Pending(m) => m.owner,
            Self::Active(m) => m.owner,
            Self::Inactive(m) => m.owner,
        }
    }

    pub fn staking_pool(&self) -> &StakingPool {
        match self {
            Self::Created(m) => &m.staking_pool,
            Self::Pending(m) => &m.staking_pool,
            Self::Active(m) => &m.staking_pool,
            Self::Inactive(m) => &m.staking_pool,
        }
    }

    pub fn staking_pool_mut(&mut self) -> &mut StakingPool {
        match self {
            Self::Created(m) => &mut m.staking_pool,
            Self::Pending(m) => &mut m.staking_pool,
            Self::Active(m) => &mut m.staking_pool,
            Self::Inactive(m) => &mut m.staking_pool,
        }
    }

    pub fn architecture_version(&self) -> ArchitectureVersion {
        match self {
            Self::Created(m) => m.architecture_version,
            Self::Pending(m) => m.architecture_version,
            Self::Active(m) => m.architecture_version,
            Self::Inactive(m) => m.architecture_version,
        }
    }

    pub fn commission_rate(&self) -> u64 {
        match self {
            Self::Created(m) => m.commission_rate,
            Self::Pending(m) => m.commission_rate,
            Self::Active(m) => m.commission_rate,
            Self::Inactive(m) => m.commission_rate,
        }
    }

    pub fn next_epoch_commission_rate(&self) -> u64 {
        match self {
            Self::Created(m) => m.next_epoch_commission_rate,
            Self::Pending(m) => m.next_epoch_commission_rate,
            Self::Active(m) => m.next_epoch_commission_rate,
            Self::Inactive(m) => m.next_epoch_commission_rate,
        }
    }
}
