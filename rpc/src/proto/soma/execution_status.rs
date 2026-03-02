// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::proto::TryFromProtoError;

impl From<crate::types::ExecutionStatus> for ExecutionStatus {
    fn from(value: crate::types::ExecutionStatus) -> Self {
        match value {
            crate::types::ExecutionStatus::Success => Self { success: Some(true), error: None },
            crate::types::ExecutionStatus::Failure { error } => {
                Self { success: Some(false), error: Some(error.into()) }
            }
        }
    }
}

impl From<crate::types::ExecutionError> for ExecutionError {
    fn from(value: crate::types::ExecutionError) -> Self {
        use execution_error::{ErrorDetails, ExecutionErrorKind};

        use crate::types::ExecutionError as E;

        let description = Some(format!("{:?}", value));

        let (kind, error_details) = match value {
            E::InsufficientGas => (ExecutionErrorKind::InsufficientGas, None),

            E::InvalidOwnership { object_id } => (
                ExecutionErrorKind::InvalidOwnership,
                Some(ErrorDetails::ObjectId(object_id.to_string())),
            ),

            E::ObjectNotFound { object_id } => (
                ExecutionErrorKind::ObjectNotFound,
                Some(ErrorDetails::ObjectId(object_id.to_string())),
            ),

            E::InvalidObjectType { object_id } => (
                ExecutionErrorKind::InvalidObjectType,
                Some(ErrorDetails::ObjectId(object_id.to_string())),
            ),

            E::InvalidTransactionType => (ExecutionErrorKind::InvalidTransactionType, None),

            E::InvalidArguments { reason } => {
                (ExecutionErrorKind::InvalidArguments, Some(ErrorDetails::OtherError(reason)))
            }

            E::DuplicateValidator => (ExecutionErrorKind::DuplicateValidator, None),
            E::DuplicateValidatorMetadata { field } => (
                ExecutionErrorKind::DuplicateValidatorMetadata,
                Some(ErrorDetails::OtherError(field)),
            ),
            E::MissingProofOfPossession => (ExecutionErrorKind::MissingProofOfPossession, None),
            E::InvalidProofOfPossession { reason } => (
                ExecutionErrorKind::InvalidProofOfPossession,
                Some(ErrorDetails::OtherError(reason)),
            ),
            E::NotAValidator => (ExecutionErrorKind::NotAValidator, None),
            E::ValidatorAlreadyRemoved => (ExecutionErrorKind::ValidatorAlreadyRemoved, None),
            E::AdvancedToWrongEpoch => (ExecutionErrorKind::AdvancedToWrongEpoch, None),

            E::ModelNotFound => (ExecutionErrorKind::ModelNotFound, None),
            E::NotModelOwner => (ExecutionErrorKind::NotModelOwner, None),
            E::ModelNotActive => (ExecutionErrorKind::ModelNotActive, None),
            E::ModelNotPending => (ExecutionErrorKind::ModelNotPending, None),
            E::ModelAlreadyInactive => (ExecutionErrorKind::ModelAlreadyInactive, None),
            E::ModelRevealEpochMismatch => (ExecutionErrorKind::ModelRevealEpochMismatch, None),
            E::ModelEmbeddingCommitmentMismatch => {
                (ExecutionErrorKind::ModelEmbeddingCommitmentMismatch, None)
            }
            E::ModelDecryptionKeyCommitmentMismatch => {
                (ExecutionErrorKind::ModelDecryptionKeyCommitmentMismatch, None)
            }
            E::ModelNoPendingUpdate => (ExecutionErrorKind::ModelNoPendingUpdate, None),
            E::ModelArchitectureVersionMismatch => {
                (ExecutionErrorKind::ModelArchitectureVersionMismatch, None)
            }
            E::ModelCommissionRateTooHigh => (ExecutionErrorKind::ModelCommissionRateTooHigh, None),
            E::ModelMinStakeNotMet => (ExecutionErrorKind::ModelMinStakeNotMet, None),

            // Target errors
            E::NoActiveModels => (ExecutionErrorKind::NoActiveModels, None),
            E::TargetNotFound => (ExecutionErrorKind::TargetNotFound, None),
            E::TargetNotOpen => (ExecutionErrorKind::TargetNotOpen, None),
            E::TargetExpired { .. } => (ExecutionErrorKind::TargetExpired, None),
            E::TargetNotFilled => (ExecutionErrorKind::TargetNotFilled, None),
            E::AuditWindowOpen { .. } => (ExecutionErrorKind::ChallengeWindowOpen, None),
            E::TargetAlreadyClaimed => (ExecutionErrorKind::TargetAlreadyClaimed, None),

            // Submission errors
            E::ModelNotInTarget { .. } => (ExecutionErrorKind::ModelNotInTarget, None),
            E::EmbeddingDimensionMismatch { .. } => {
                (ExecutionErrorKind::EmbeddingDimensionMismatch, None)
            }
            E::DistanceExceedsThreshold { .. } => {
                (ExecutionErrorKind::DistanceExceedsThreshold, None)
            }
            E::InsufficientBond { .. } => (ExecutionErrorKind::InsufficientBond, None),
            E::InsufficientEmissionBalance => {
                (ExecutionErrorKind::InsufficientEmissionBalance, None)
            }

            // Audit errors
            E::AuditWindowClosed { .. } => (ExecutionErrorKind::ChallengeWindowClosed, None),
            E::DataExceedsMaxSize { size, max_size } => (
                ExecutionErrorKind::DataExceedsMaxSize,
                Some(ErrorDetails::OtherError(format!("size={}, max_size={}", size, max_size))),
            ),

            E::InsufficientCoinBalance => (ExecutionErrorKind::InsufficientCoinBalance, None),
            E::CoinBalanceOverflow => (ExecutionErrorKind::CoinBalanceOverflow, None),

            E::ValidatorNotFound => (ExecutionErrorKind::ValidatorNotFound, None),
            E::StakingPoolNotFound => (ExecutionErrorKind::StakingPoolNotFound, None),
            E::CannotReportOneself => (ExecutionErrorKind::CannotReportOneself, None),
            E::ReportRecordNotFound => (ExecutionErrorKind::ReportRecordNotFound, None),

            E::InputObjectDeleted => (ExecutionErrorKind::InputObjectDeleted, None),
            E::CertificateDenied => (ExecutionErrorKind::CertificateDenied, None),
            E::SharedObjectCongestion => (ExecutionErrorKind::SharedObjectCongestion, None),

            E::OtherError(msg) => {
                (ExecutionErrorKind::OtherError, Some(ErrorDetails::OtherError(msg)))
            }
        };

        Self { description, kind: Some(kind.into()), error_details }
    }
}

// Conversions from protobuf to RPC types

impl TryFrom<&ExecutionStatus> for crate::types::ExecutionStatus {
    type Error = TryFromProtoError;

    fn try_from(value: &ExecutionStatus) -> Result<Self, Self::Error> {
        let success = value.success.ok_or_else(|| TryFromProtoError::missing("success"))?;

        match (success, &value.error) {
            (true, None) => Ok(Self::Success),
            (false, Some(error)) => Ok(Self::Failure { error: error.try_into()? }),
            (true, Some(_)) => Err(TryFromProtoError::invalid(
                "ExecutionStatus",
                "error present when success is true",
            )),
            (false, None) => Err(TryFromProtoError::invalid(
                "ExecutionStatus",
                "error missing when success is false",
            )),
        }
    }
}

impl TryFrom<&ExecutionError> for crate::types::ExecutionError {
    type Error = TryFromProtoError;

    fn try_from(value: &ExecutionError) -> Result<Self, Self::Error> {
        use execution_error::{ErrorDetails, ExecutionErrorKind as K};

        let kind = value.kind();

        match kind {
            K::Unknown => {
                let msg = if let Some(ErrorDetails::OtherError(msg)) = &value.error_details {
                    msg.clone()
                } else if let Some(desc) = &value.description {
                    desc.clone()
                } else {
                    "Unknown error".to_string()
                };
                Ok(Self::OtherError(msg))
            }

            K::InsufficientGas => Ok(Self::InsufficientGas),

            K::InvalidOwnership => {
                if let Some(ErrorDetails::ObjectId(object_id)) = &value.error_details {
                    Ok(Self::InvalidOwnership {
                        object_id: object_id
                            .parse()
                            .map_err(|e| TryFromProtoError::invalid("object_id", e))?,
                    })
                } else {
                    Err(TryFromProtoError::missing("object_id for InvalidOwnership"))
                }
            }

            K::ObjectNotFound => {
                if let Some(ErrorDetails::ObjectId(object_id)) = &value.error_details {
                    Ok(Self::ObjectNotFound {
                        object_id: object_id
                            .parse()
                            .map_err(|e| TryFromProtoError::invalid("object_id", e))?,
                    })
                } else {
                    Err(TryFromProtoError::missing("object_id for ObjectNotFound"))
                }
            }

            K::InvalidObjectType => {
                if let Some(ErrorDetails::ObjectId(object_id)) = &value.error_details {
                    Ok(Self::InvalidObjectType {
                        object_id: object_id
                            .parse()
                            .map_err(|e| TryFromProtoError::invalid("object_id", e))?,
                    })
                } else {
                    Err(TryFromProtoError::missing("object_id for InvalidObjectType"))
                }
            }

            K::InvalidTransactionType => Ok(Self::InvalidTransactionType),

            K::InvalidArguments => {
                let reason = if let Some(ErrorDetails::OtherError(reason)) = &value.error_details {
                    reason.clone()
                } else if let Some(desc) = &value.description {
                    desc.clone()
                } else {
                    "Invalid arguments".to_string()
                };
                Ok(Self::InvalidArguments { reason })
            }

            K::DuplicateValidator => Ok(Self::DuplicateValidator),
            K::DuplicateValidatorMetadata => {
                let field = if let Some(ErrorDetails::OtherError(f)) = &value.error_details {
                    f.clone()
                } else {
                    "unknown".to_string()
                };
                Ok(Self::DuplicateValidatorMetadata { field })
            }
            K::MissingProofOfPossession => Ok(Self::MissingProofOfPossession),
            K::InvalidProofOfPossession => {
                let reason = if let Some(ErrorDetails::OtherError(r)) = &value.error_details {
                    r.clone()
                } else {
                    "unknown".to_string()
                };
                Ok(Self::InvalidProofOfPossession { reason })
            }
            K::NotAValidator => Ok(Self::NotAValidator),
            K::ValidatorAlreadyRemoved => Ok(Self::ValidatorAlreadyRemoved),
            K::AdvancedToWrongEpoch => Ok(Self::AdvancedToWrongEpoch),

            K::ModelNotFound => Ok(Self::ModelNotFound),
            K::NotModelOwner => Ok(Self::NotModelOwner),
            K::ModelNotActive => Ok(Self::ModelNotActive),
            K::ModelNotPending => Ok(Self::ModelNotPending),
            K::ModelAlreadyInactive => Ok(Self::ModelAlreadyInactive),
            K::ModelRevealEpochMismatch => Ok(Self::ModelRevealEpochMismatch),
            K::ModelEmbeddingCommitmentMismatch => Ok(Self::ModelEmbeddingCommitmentMismatch),
            K::ModelDecryptionKeyCommitmentMismatch => {
                Ok(Self::ModelDecryptionKeyCommitmentMismatch)
            }
            K::ModelNoPendingUpdate => Ok(Self::ModelNoPendingUpdate),
            K::ModelArchitectureVersionMismatch => Ok(Self::ModelArchitectureVersionMismatch),
            K::ModelCommissionRateTooHigh => Ok(Self::ModelCommissionRateTooHigh),
            K::ModelMinStakeNotMet => Ok(Self::ModelMinStakeNotMet),

            // Target errors
            K::NoActiveModels => Ok(Self::NoActiveModels),
            K::TargetNotFound => Ok(Self::TargetNotFound),
            K::TargetNotOpen => Ok(Self::TargetNotOpen),
            K::TargetExpired => Ok(Self::TargetExpired { generation_epoch: 0, current_epoch: 0 }),
            K::TargetNotFilled => Ok(Self::TargetNotFilled),
            K::ChallengeWindowOpen => Ok(Self::AuditWindowOpen { fill_epoch: 0, current_epoch: 0 }),
            K::TargetAlreadyClaimed => Ok(Self::TargetAlreadyClaimed),

            // Submission errors
            K::ModelNotInTarget => Ok(Self::ModelNotInTarget {
                model_id: crate::types::Address::new([0u8; 32]),
                target_id: crate::types::Address::new([0u8; 32]),
            }),
            K::EmbeddingDimensionMismatch => {
                Ok(Self::EmbeddingDimensionMismatch { expected: 0, actual: 0 })
            }
            K::DistanceExceedsThreshold => {
                Ok(Self::DistanceExceedsThreshold { score: 0.0, threshold: 0.0 })
            }
            K::InsufficientBond => Ok(Self::InsufficientBond { required: 0, provided: 0 }),
            K::InsufficientEmissionBalance => Ok(Self::InsufficientEmissionBalance),

            // Audit errors
            K::ChallengeWindowClosed => {
                Ok(Self::AuditWindowClosed { fill_epoch: 0, current_epoch: 0 })
            }
            // Legacy challenge proto variants map to OtherError
            K::InsufficientChallengerBond
            | K::ChallengeNotFound
            | K::ChallengeNotPending
            | K::ChallengeExpired
            | K::InvalidChallengeResult
            | K::InvalidChallengeQuorum
            | K::ChallengeAlreadyExists => {
                let msg = value
                    .description
                    .clone()
                    .unwrap_or_else(|| "Legacy challenge error".to_string());
                Ok(Self::OtherError(msg))
            }
            K::DataExceedsMaxSize => Ok(Self::DataExceedsMaxSize { size: 0, max_size: 0 }),

            K::InsufficientCoinBalance => Ok(Self::InsufficientCoinBalance),
            K::CoinBalanceOverflow => Ok(Self::CoinBalanceOverflow),

            K::ValidatorNotFound => Ok(Self::ValidatorNotFound),
            K::StakingPoolNotFound => Ok(Self::StakingPoolNotFound),
            K::CannotReportOneself => Ok(Self::CannotReportOneself),
            K::ReportRecordNotFound => Ok(Self::ReportRecordNotFound),

            K::InputObjectDeleted => Ok(Self::InputObjectDeleted),
            K::CertificateDenied => Ok(Self::CertificateDenied),
            K::SharedObjectCongestion => Ok(Self::SharedObjectCongestion),

            K::OtherError => {
                let msg = if let Some(ErrorDetails::OtherError(msg)) = &value.error_details {
                    msg.clone()
                } else if let Some(desc) = &value.description {
                    desc.clone()
                } else {
                    "Other error".to_string()
                };
                Ok(Self::OtherError(msg))
            }
        }
    }
}
