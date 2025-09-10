use super::*;
use crate::proto::TryFromProtoError;

impl From<crate::types::ExecutionStatus> for ExecutionStatus {
    fn from(value: crate::types::ExecutionStatus) -> Self {
        match value {
            crate::types::ExecutionStatus::Success => Self {
                success: Some(true),
                error: None,
            },
            crate::types::ExecutionStatus::Failure { error } => Self {
                success: Some(false),
                error: Some(error.into()),
            },
        }
    }
}

impl From<crate::types::ExecutionError> for ExecutionError {
    fn from(value: crate::types::ExecutionError) -> Self {
        use crate::types::ExecutionError as E;
        use execution_error::{ErrorDetails, ExecutionErrorKind};

        let mut message = Self::default();

        // Set human-readable description
        message.description = Some(format!("{:?}", value));

        // Map to protobuf error kind and details
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

            E::InvalidArguments { reason } => (
                ExecutionErrorKind::InvalidArguments,
                Some(ErrorDetails::OtherError(reason)),
            ),

            E::DuplicateValidator => (ExecutionErrorKind::DuplicateValidator, None),
            E::NotAValidator => (ExecutionErrorKind::NotAValidator, None),
            E::ValidatorAlreadyRemoved => (ExecutionErrorKind::ValidatorAlreadyRemoved, None),
            E::AdvancedToWrongEpoch => (ExecutionErrorKind::AdvancedToWrongEpoch, None),

            E::DuplicateEncoder => (ExecutionErrorKind::DuplicateEncoder, None),
            E::NotAnEncoder => (ExecutionErrorKind::NotAnEncoder, None),
            E::EncoderAlreadyRemoved => (ExecutionErrorKind::EncoderAlreadyRemoved, None),

            E::InsufficientCoinBalance => (ExecutionErrorKind::InsufficientCoinBalance, None),
            E::CoinBalanceOverflow => (ExecutionErrorKind::CoinBalanceOverflow, None),

            E::ValidatorNotFound => (ExecutionErrorKind::ValidatorNotFound, None),
            E::EncoderNotFound => (ExecutionErrorKind::EncoderNotFound, None),
            E::StakingPoolNotFound => (ExecutionErrorKind::StakingPoolNotFound, None),
            E::CannotReportOneself => (ExecutionErrorKind::CannotReportOneself, None),
            E::ReportRecordNotFound => (ExecutionErrorKind::ReportRecordNotFound, None),

            E::OtherError(msg) => (
                ExecutionErrorKind::OtherError,
                Some(ErrorDetails::OtherError(msg)),
            ),
        };

        message.kind = Some(kind.into());
        message.error_details = error_details;
        message
    }
}

// Conversions from protobuf to RPC types

impl TryFrom<&ExecutionStatus> for crate::types::ExecutionStatus {
    type Error = TryFromProtoError;

    fn try_from(value: &ExecutionStatus) -> Result<Self, Self::Error> {
        let success = value
            .success
            .ok_or_else(|| TryFromProtoError::missing("success"))?;

        match (success, &value.error) {
            (true, None) => Ok(Self::Success),
            (false, Some(error)) => Ok(Self::Failure {
                error: error.try_into()?,
            }),
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
                    Err(TryFromProtoError::missing(
                        "object_id for InvalidObjectType",
                    ))
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
            K::NotAValidator => Ok(Self::NotAValidator),
            K::ValidatorAlreadyRemoved => Ok(Self::ValidatorAlreadyRemoved),
            K::AdvancedToWrongEpoch => Ok(Self::AdvancedToWrongEpoch),

            K::DuplicateEncoder => Ok(Self::DuplicateEncoder),
            K::NotAnEncoder => Ok(Self::NotAnEncoder),
            K::EncoderAlreadyRemoved => Ok(Self::EncoderAlreadyRemoved),

            K::InsufficientCoinBalance => Ok(Self::InsufficientCoinBalance),
            K::CoinBalanceOverflow => Ok(Self::CoinBalanceOverflow),

            K::ValidatorNotFound => Ok(Self::ValidatorNotFound),
            K::EncoderNotFound => Ok(Self::EncoderNotFound),
            K::StakingPoolNotFound => Ok(Self::StakingPoolNotFound),
            K::CannotReportOneself => Ok(Self::CannotReportOneself),
            K::ReportRecordNotFound => Ok(Self::ReportRecordNotFound),

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
