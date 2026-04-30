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

            E::InvalidGasCoinType { object_id } => (
                ExecutionErrorKind::InvalidGasCoinType,
                Some(ErrorDetails::ObjectId(object_id.to_string())),
            ),

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
            E::TargetExpired { generation_epoch, current_epoch } => (
                ExecutionErrorKind::TargetExpired,
                Some(ErrorDetails::OtherError(format!(
                    "generation_epoch={}, current_epoch={}",
                    generation_epoch, current_epoch
                ))),
            ),
            E::TargetNotFilled => (ExecutionErrorKind::TargetNotFilled, None),
            E::AuditWindowOpen { fill_epoch, current_epoch } => (
                ExecutionErrorKind::ChallengeWindowOpen,
                Some(ErrorDetails::OtherError(format!(
                    "fill_epoch={}, current_epoch={}",
                    fill_epoch, current_epoch
                ))),
            ),
            E::TargetAlreadyClaimed => (ExecutionErrorKind::TargetAlreadyClaimed, None),

            // Submission errors
            E::ModelNotInTarget { model_id, target_id } => (
                ExecutionErrorKind::ModelNotInTarget,
                Some(ErrorDetails::OtherError(format!(
                    "model_id={}, target_id={}",
                    model_id, target_id
                ))),
            ),
            E::EmbeddingDimensionMismatch { expected, actual } => (
                ExecutionErrorKind::EmbeddingDimensionMismatch,
                Some(ErrorDetails::OtherError(format!("expected={}, actual={}", expected, actual))),
            ),
            E::InsufficientBond { required, provided } => (
                ExecutionErrorKind::InsufficientBond,
                Some(ErrorDetails::OtherError(format!(
                    "required={}, provided={}",
                    required, provided
                ))),
            ),
            E::InsufficientEmissionBalance => {
                (ExecutionErrorKind::InsufficientEmissionBalance, None)
            }

            // Audit errors
            E::AuditWindowClosed { fill_epoch, current_epoch } => (
                ExecutionErrorKind::ChallengeWindowClosed,
                Some(ErrorDetails::OtherError(format!(
                    "fill_epoch={}, current_epoch={}",
                    fill_epoch, current_epoch
                ))),
            ),
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

            E::ArithmeticOverflow => (
                ExecutionErrorKind::OtherError,
                Some(ErrorDetails::OtherError("Arithmetic overflow in execution".into())),
            ),

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

            K::InvalidGasCoinType => {
                if let Some(ErrorDetails::ObjectId(object_id)) = &value.error_details {
                    Ok(Self::InvalidGasCoinType {
                        object_id: object_id
                            .parse()
                            .map_err(|e| TryFromProtoError::invalid("object_id", e))?,
                    })
                } else {
                    Err(TryFromProtoError::missing("object_id for InvalidGasCoinType"))
                }
            }

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
            K::TargetExpired => {
                let (generation_epoch, current_epoch) =
                    parse_two_u64s(&value.error_details, "generation_epoch", "current_epoch");
                Ok(Self::TargetExpired { generation_epoch, current_epoch })
            }
            K::TargetNotFilled => Ok(Self::TargetNotFilled),
            K::ChallengeWindowOpen => {
                let (fill_epoch, current_epoch) =
                    parse_two_u64s(&value.error_details, "fill_epoch", "current_epoch");
                Ok(Self::AuditWindowOpen { fill_epoch, current_epoch })
            }
            K::TargetAlreadyClaimed => Ok(Self::TargetAlreadyClaimed),

            // Submission errors
            K::ModelNotInTarget => {
                let (model_id, target_id) = if let Some(ErrorDetails::OtherError(details)) =
                    &value.error_details
                {
                    let mut model_id = crate::types::Address::new([0u8; 32]);
                    let mut target_id = crate::types::Address::new([0u8; 32]);
                    for part in details.split(", ") {
                        if let Some(v) = part.strip_prefix("model_id=") {
                            if let Ok(addr) = v.parse() {
                                model_id = addr;
                            }
                        } else if let Some(v) = part.strip_prefix("target_id=") {
                            if let Ok(addr) = v.parse() {
                                target_id = addr;
                            }
                        }
                    }
                    (model_id, target_id)
                } else {
                    (crate::types::Address::new([0u8; 32]), crate::types::Address::new([0u8; 32]))
                };
                Ok(Self::ModelNotInTarget { model_id, target_id })
            }
            K::EmbeddingDimensionMismatch => {
                let (expected, actual) = parse_two_u64s(&value.error_details, "expected", "actual");
                Ok(Self::EmbeddingDimensionMismatch { expected, actual })
            }
            K::DistanceExceedsThreshold => {
                // Deprecated variant - map to OtherError
                let msg = value.description.clone().unwrap_or_else(|| "Distance exceeds threshold (deprecated)".to_string());
                Ok(Self::OtherError(msg))
            }
            K::InsufficientBond => {
                let (required, provided) =
                    parse_two_u64s(&value.error_details, "required", "provided");
                Ok(Self::InsufficientBond { required, provided })
            }
            K::InsufficientEmissionBalance => Ok(Self::InsufficientEmissionBalance),

            // Audit errors
            K::ChallengeWindowClosed => {
                let (fill_epoch, current_epoch) =
                    parse_two_u64s(&value.error_details, "fill_epoch", "current_epoch");
                Ok(Self::AuditWindowClosed { fill_epoch, current_epoch })
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
            K::DataExceedsMaxSize => {
                let (size, max_size) = parse_two_u64s(&value.error_details, "size", "max_size");
                Ok(Self::DataExceedsMaxSize { size, max_size })
            }

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

/// Parse two u64 values from an `ErrorDetails::OtherError` string of the form "key1=val1, key2=val2".
fn parse_two_u64s(
    details: &Option<execution_error::ErrorDetails>,
    key1: &str,
    key2: &str,
) -> (u64, u64) {
    if let Some(execution_error::ErrorDetails::OtherError(s)) = details {
        let mut v1 = 0u64;
        let mut v2 = 0u64;
        for part in s.split(", ") {
            if let Some(v) = part.strip_prefix(&format!("{}=", key1)) {
                v1 = v.parse().unwrap_or(0);
            } else if let Some(v) = part.strip_prefix(&format!("{}=", key2)) {
                v2 = v.parse().unwrap_or(0);
            }
        }
        (v1, v2)
    } else {
        (0, 0)
    }
}

/// Parse two f32 values from an `ErrorDetails::OtherError` string of the form "key1=val1, key2=val2".
fn parse_two_f32s(
    details: &Option<execution_error::ErrorDetails>,
    key1: &str,
    key2: &str,
) -> (f32, f32) {
    if let Some(execution_error::ErrorDetails::OtherError(s)) = details {
        let mut v1 = 0.0f32;
        let mut v2 = 0.0f32;
        for part in s.split(", ") {
            if let Some(v) = part.strip_prefix(&format!("{}=", key1)) {
                v1 = v.parse().unwrap_or(0.0);
            } else if let Some(v) = part.strip_prefix(&format!("{}=", key2)) {
                v2 = v.parse().unwrap_or(0.0);
            }
        }
        (v1, v2)
    } else {
        (0.0, 0.0)
    }
}
