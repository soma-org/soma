use super::*;
use crate::proto::TryFromProtoError;
use tap::Pipe;

//
// ExecutionStatus
//

impl From<crate::types::ExecutionStatus> for ExecutionStatus {
    fn from(value: crate::types::ExecutionStatus) -> Self {
        match value {
            crate::types::ExecutionStatus::Success => Self {
                success: Some(true),
                error: None,
            },
            crate::types::ExecutionStatus::Failure { error, command } => {
                let mut error_message = ExecutionError::from(error);
                error_message.command = command;
                Self {
                    success: Some(false),
                    error: Some(error_message),
                }
            }
        }
    }
}

impl TryFrom<&ExecutionStatus> for crate::types::ExecutionStatus {
    type Error = TryFromProtoError;

    fn try_from(value: &ExecutionStatus) -> Result<Self, Self::Error> {
        let success = value
            .success
            .ok_or_else(|| TryFromProtoError::missing(ExecutionStatus::SUCCESS_FIELD))?;
        match (success, &value.error) {
            (true, None) => Self::Success,
            (false, Some(error)) => Self::Failure {
                error: crate::types::ExecutionError::try_from(error)
                    .map_err(|e| e.nested(ExecutionStatus::ERROR_FIELD))?,
                command: error.command,
            },
            (true, Some(_)) | (false, None) => {
                return Err(TryFromProtoError::invalid(
                    ExecutionStatus::ERROR_FIELD,
                    "invalid execution status",
                ));
            }
        }
        .pipe(Ok)
    }
}

//
// ExecutionError
//

impl From<crate::types::ExecutionError> for ExecutionError {
    fn from(value: crate::types::ExecutionError) -> Self {
        use crate::types::ExecutionError as E;
        use execution_error::ErrorDetails;
        use execution_error::ExecutionErrorKind;

        let mut message = Self::default();

        let kind = match value {
            E::InsufficientGas => ExecutionErrorKind::InsufficientGas,
            E::InvalidGasObject => ExecutionErrorKind::InvalidGasObject,
            E::InvariantViolation => ExecutionErrorKind::InvariantViolation,
            E::FeatureNotYetSupported => ExecutionErrorKind::FeatureNotYetSupported,
            E::ObjectTooBig {
                object_size,
                max_object_size,
            } => {
                message.error_details = Some(ErrorDetails::SizeError(SizeError {
                    size: Some(object_size),
                    max_size: Some(max_object_size),
                }));
                ExecutionErrorKind::ObjectTooBig
            }
            E::PackageTooBig {
                object_size,
                max_object_size,
            } => {
                message.error_details = Some(ErrorDetails::SizeError(SizeError {
                    size: Some(object_size),
                    max_size: Some(max_object_size),
                }));
                ExecutionErrorKind::PackageTooBig
            }
            E::CircularObjectOwnership { object } => {
                message.error_details = Some(ErrorDetails::ObjectId(object.to_string()));
                ExecutionErrorKind::CircularObjectOwnership
            }
            E::InsufficientCoinBalance => ExecutionErrorKind::InsufficientCoinBalance,
            E::CoinBalanceOverflow => ExecutionErrorKind::CoinBalanceOverflow,
            E::PublishErrorNonZeroAddress => ExecutionErrorKind::PublishErrorNonZeroAddress,
            E::SuiMoveVerificationError => ExecutionErrorKind::SuiMoveVerificationError,
            E::MovePrimitiveRuntimeError { location } => {
                message.error_details = location.map(|l| {
                    ErrorDetails::Abort(MoveAbort {
                        location: Some(l.into()),
                        ..Default::default()
                    })
                });
                ExecutionErrorKind::MovePrimitiveRuntimeError
            }
            E::MoveAbort { location, code } => {
                message.error_details = Some(ErrorDetails::Abort(MoveAbort {
                    abort_code: Some(code),
                    location: Some(location.into()),
                    clever_error: None,
                }));
                ExecutionErrorKind::MoveAbort
            }
            E::VmVerificationOrDeserializationError => {
                ExecutionErrorKind::VmVerificationOrDeserializationError
            }
            E::VmInvariantViolation => ExecutionErrorKind::VmInvariantViolation,
            E::FunctionNotFound => ExecutionErrorKind::FunctionNotFound,
            E::ArityMismatch => ExecutionErrorKind::ArityMismatch,
            E::TypeArityMismatch => ExecutionErrorKind::TypeArityMismatch,
            E::NonEntryFunctionInvoked => ExecutionErrorKind::NonEntryFunctionInvoked,
            E::CommandArgumentError { argument, kind } => {
                let mut command_argument_error = CommandArgumentError::from(kind);
                command_argument_error.argument = Some(argument.into());
                message.error_details =
                    Some(ErrorDetails::CommandArgumentError(command_argument_error));
                ExecutionErrorKind::CommandArgumentError
            }
            E::TypeArgumentError {
                type_argument,
                kind,
            } => {
                let type_argument_error = TypeArgumentError {
                    type_argument: Some(type_argument.into()),
                    kind: Some(type_argument_error::TypeArgumentErrorKind::from(kind).into()),
                };
                message.error_details = Some(ErrorDetails::TypeArgumentError(type_argument_error));
                ExecutionErrorKind::TypeArgumentError
            }
            E::UnusedValueWithoutDrop { result, subresult } => {
                message.error_details = Some(ErrorDetails::IndexError(IndexError {
                    index: Some(result.into()),
                    subresult: Some(subresult.into()),
                }));
                ExecutionErrorKind::UnusedValueWithoutDrop
            }
            E::InvalidPublicFunctionReturnType { index } => {
                message.error_details = Some(ErrorDetails::IndexError(IndexError {
                    index: Some(index.into()),
                    subresult: None,
                }));
                ExecutionErrorKind::InvalidPublicFunctionReturnType
            }
            E::InvalidTransferObject => ExecutionErrorKind::InvalidTransferObject,
            E::EffectsTooLarge {
                current_size,
                max_size,
            } => {
                message.error_details = Some(ErrorDetails::SizeError(SizeError {
                    size: Some(current_size),
                    max_size: Some(max_size),
                }));
                ExecutionErrorKind::EffectsTooLarge
            }
            E::PublishUpgradeMissingDependency => {
                ExecutionErrorKind::PublishUpgradeMissingDependency
            }
            E::PublishUpgradeDependencyDowngrade => {
                ExecutionErrorKind::PublishUpgradeDependencyDowngrade
            }
            E::PackageUpgradeError { kind } => {
                message.error_details = Some(ErrorDetails::PackageUpgradeError(kind.into()));
                ExecutionErrorKind::PackageUpgradeError
            }
            E::WrittenObjectsTooLarge {
                object_size,
                max_object_size,
            } => {
                message.error_details = Some(ErrorDetails::SizeError(SizeError {
                    size: Some(object_size),
                    max_size: Some(max_object_size),
                }));

                ExecutionErrorKind::WrittenObjectsTooLarge
            }
            E::CertificateDenied => ExecutionErrorKind::CertificateDenied,
            E::SuiMoveVerificationTimedout => ExecutionErrorKind::SuiMoveVerificationTimedout,
            E::ConsensusObjectOperationNotAllowed => {
                ExecutionErrorKind::ConsensusObjectOperationNotAllowed
            }
            E::InputObjectDeleted => ExecutionErrorKind::InputObjectDeleted,
            E::ExecutionCanceledDueToConsensusObjectCongestion { congested_objects } => {
                message.error_details = Some(ErrorDetails::CongestedObjects(CongestedObjects {
                    objects: congested_objects.iter().map(ToString::to_string).collect(),
                }));

                ExecutionErrorKind::ExecutionCanceledDueToConsensusObjectCongestion
            }
            E::AddressDeniedForCoin { address, coin_type } => {
                message.error_details = Some(ErrorDetails::CoinDenyListError(CoinDenyListError {
                    address: Some(address.to_string()),
                    coin_type: Some(coin_type),
                }));
                ExecutionErrorKind::AddressDeniedForCoin
            }
            E::CoinTypeGlobalPause { coin_type } => {
                message.error_details = Some(ErrorDetails::CoinDenyListError(CoinDenyListError {
                    address: None,
                    coin_type: Some(coin_type),
                }));
                ExecutionErrorKind::CoinTypeGlobalPause
            }
            E::ExecutionCanceledDueToRandomnessUnavailable => {
                ExecutionErrorKind::ExecutionCanceledDueToRandomnessUnavailable
            }
            E::MoveVectorElemTooBig {
                value_size,
                max_scaled_size,
            } => {
                message.error_details = Some(ErrorDetails::SizeError(SizeError {
                    size: Some(value_size),
                    max_size: Some(max_scaled_size),
                }));

                ExecutionErrorKind::MoveVectorElemTooBig
            }
            E::MoveRawValueTooBig {
                value_size,
                max_scaled_size,
            } => {
                message.error_details = Some(ErrorDetails::SizeError(SizeError {
                    size: Some(value_size),
                    max_size: Some(max_scaled_size),
                }));
                ExecutionErrorKind::MoveRawValueTooBig
            }
            E::InvalidLinkage => ExecutionErrorKind::InvalidLinkage,
            _ => ExecutionErrorKind::Unknown,
        };

        message.set_kind(kind);
        message
    }
}

impl TryFrom<&ExecutionError> for crate::types::ExecutionError {
    type Error = TryFromProtoError;

    fn try_from(value: &ExecutionError) -> Result<Self, Self::Error> {
        use execution_error::ErrorDetails;
        use execution_error::ExecutionErrorKind as K;

        match value.kind() {
            K::Unknown => return Err(TryFromProtoError::invalid(ExecutionError::KIND_FIELD, "unknown ExecutionErrorKind")),
            K::InsufficientGas => Self::InsufficientGas,
            K::InvalidGasObject => Self::InvalidGasObject,
            K::InvariantViolation => Self::InvariantViolation,
            K::FeatureNotYetSupported => Self::FeatureNotYetSupported,
            K::ObjectTooBig => {
                let Some(ErrorDetails::SizeError(SizeError { size, max_size })) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing(ExecutionError::SIZE_ERROR_FIELD));
                };
                Self::ObjectTooBig {
                    object_size: size.ok_or_else(|| TryFromProtoError::missing(SizeError::SIZE_FIELD))?,
                    max_object_size: max_size
                        .ok_or_else(|| TryFromProtoError::missing(SizeError::MAX_SIZE_FIELD))?,
                }
            }
            K::PackageTooBig => {
                let Some(ErrorDetails::SizeError(SizeError { size, max_size })) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing(ExecutionError::SIZE_ERROR_FIELD));
                };
                Self::PackageTooBig {
                    object_size: size.ok_or_else(|| TryFromProtoError::missing(SizeError::SIZE_FIELD))?,
                    max_object_size: max_size
                        .ok_or_else(|| TryFromProtoError::missing(SizeError::MAX_SIZE_FIELD))?,
                }
            }
            K::CircularObjectOwnership => {
                let Some(ErrorDetails::ObjectId(object_id)) = &value.error_details else {
                    return Err(TryFromProtoError::missing(ExecutionError::OBJECT_ID_FIELD));
                };
                Self::CircularObjectOwnership {
                    object: object_id.parse().map_err(|e| TryFromProtoError::invalid(ExecutionError::OBJECT_ID_FIELD, e))?,
                }
            }
            K::InsufficientCoinBalance => Self::InsufficientCoinBalance,
            K::CoinBalanceOverflow => Self::CoinBalanceOverflow,
            K::PublishErrorNonZeroAddress => Self::PublishErrorNonZeroAddress,
            K::SuiMoveVerificationError => Self::SuiMoveVerificationError,
            K::MovePrimitiveRuntimeError => {
                let location = if let Some(ErrorDetails::Abort(abort)) = &value.error_details {
                    abort.location.as_ref().map(TryInto::try_into).transpose()?
                } else {
                    None
                };
                Self::MovePrimitiveRuntimeError { location }
            }
            K::MoveAbort => {
                let Some(ErrorDetails::Abort(abort)) = &value.error_details else {
                    return Err(TryFromProtoError::missing("abort"));
                };
                Self::MoveAbort {
                    location: abort
                        .location
                        .as_ref()
                        .ok_or_else(|| TryFromProtoError::missing("location"))?
                        .try_into()?,
                    code: abort
                        .abort_code
                        .ok_or_else(|| TryFromProtoError::missing("abort_code"))?,
                }
            }
            K::VmVerificationOrDeserializationError => Self::VmVerificationOrDeserializationError,
            K::VmInvariantViolation => Self::VmInvariantViolation,
            K::FunctionNotFound => Self::FunctionNotFound,
            K::ArityMismatch => Self::ArityMismatch,
            K::TypeArityMismatch => Self::TypeArityMismatch,
            K::NonEntryFunctionInvoked => Self::NonEntryFunctionInvoked,
            K::CommandArgumentError => {
                let Some(ErrorDetails::CommandArgumentError(command_argument_error)) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing("command_argument_error"));
                };
                Self::CommandArgumentError {
                    argument: command_argument_error
                        .argument
                        .ok_or_else(|| TryFromProtoError::missing("argument"))?
                        .try_into()
                        .map_err(|e| TryFromProtoError::invalid(CommandArgumentError::ARGUMENT_FIELD, e))?,
                    kind: command_argument_error.try_into()?,
                }
            }
            K::TypeArgumentError => {
                let Some(ErrorDetails::TypeArgumentError(type_argument_error)) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing("type_argument_error"));
                };
                Self::TypeArgumentError {
                    type_argument: type_argument_error
                        .type_argument
                        .ok_or_else(|| TryFromProtoError::missing("type_argument"))?
                        .try_into()
                        .map_err(|e| TryFromProtoError::invalid(TypeArgumentError::TYPE_ARGUMENT_FIELD, e))?,
                    kind: type_argument_error.kind().try_into()?,
                }
            }
            K::UnusedValueWithoutDrop => {
                let Some(ErrorDetails::IndexError(IndexError { index, subresult })) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing("index_error"));
                };
                Self::UnusedValueWithoutDrop {
                    result: index
                        .ok_or_else(|| TryFromProtoError::missing("result"))?
                        .try_into()
                        .map_err(|e| TryFromProtoError::invalid(IndexError::INDEX_FIELD, e))?,
                    subresult: subresult
                        .ok_or_else(|| TryFromProtoError::missing("subresult"))?
                        .try_into()
                        .map_err(|e| TryFromProtoError::invalid(IndexError::SUBRESULT_FIELD, e))?,
                }
            }
            K::InvalidPublicFunctionReturnType => {
                let Some(ErrorDetails::IndexError(IndexError { index, .. })) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing("index_error"));
                };
                Self::InvalidPublicFunctionReturnType {
                    index: index
                        .ok_or_else(|| TryFromProtoError::missing("index"))?
                        .try_into()
                        .map_err(|e| TryFromProtoError::invalid(IndexError::INDEX_FIELD, e))?,
                }
            }
            K::InvalidTransferObject => Self::InvalidTransferObject,
            K::EffectsTooLarge => {
                let Some(ErrorDetails::SizeError(SizeError { size, max_size })) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing(ExecutionError::SIZE_ERROR_FIELD));
                };
                Self::EffectsTooLarge {
                    current_size: size.ok_or_else(|| TryFromProtoError::missing(SizeError::SIZE_FIELD))?,
                    max_size: max_size.ok_or_else(|| TryFromProtoError::missing(SizeError::MAX_SIZE_FIELD))?,
                }
            }
            K::PublishUpgradeMissingDependency => Self::PublishUpgradeMissingDependency,
            K::PublishUpgradeDependencyDowngrade => Self::PublishUpgradeDependencyDowngrade,
            K::PackageUpgradeError => {
                let Some(ErrorDetails::PackageUpgradeError(package_upgrade_error)) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing("package_upgrade_error"));
                };
                Self::PackageUpgradeError {
                    kind: package_upgrade_error.try_into()?,
                }
            }
            K::WrittenObjectsTooLarge => {
                let Some(ErrorDetails::SizeError(SizeError { size, max_size })) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing(ExecutionError::SIZE_ERROR_FIELD));
                };

                Self::WrittenObjectsTooLarge {
                    object_size: size.ok_or_else(|| TryFromProtoError::missing(SizeError::SIZE_FIELD))?,
                    max_object_size: max_size
                        .ok_or_else(|| TryFromProtoError::missing(SizeError::MAX_SIZE_FIELD))?,
                }
            }
            K::CertificateDenied => Self::CertificateDenied,
            K::SuiMoveVerificationTimedout => Self::SuiMoveVerificationTimedout,
            K::ConsensusObjectOperationNotAllowed => Self::ConsensusObjectOperationNotAllowed,
            K::InputObjectDeleted => Self::InputObjectDeleted,
            K::ExecutionCanceledDueToConsensusObjectCongestion => {
                let Some(ErrorDetails::CongestedObjects(CongestedObjects { objects })) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing("congested_objects"));
                };

                Self::ExecutionCanceledDueToConsensusObjectCongestion {
                    congested_objects: objects
                        .iter()
                        .map(|s| s.parse())
                        .collect::<Result<_, _>>()
                        .map_err(|e| TryFromProtoError::invalid(CongestedObjects::OBJECTS_FIELD, e))?,
                }
            }
            K::AddressDeniedForCoin => {
                let Some(ErrorDetails::CoinDenyListError(CoinDenyListError {
                    address,
                    coin_type,
                })) = &value.error_details
                else {
                    return Err(TryFromProtoError::missing("coin_deny_list_error"));
                };
                Self::AddressDeniedForCoin {
                    address: address
                        .as_ref()
                        .ok_or_else(|| TryFromProtoError::missing("address"))?
                        .parse()
                        .map_err(|e| TryFromProtoError::invalid(CoinDenyListError::ADDRESS_FIELD, e))?,
                    coin_type: coin_type
                        .as_ref()
                        .ok_or_else(|| TryFromProtoError::missing("coin_type"))?
                        .to_owned(),
                }
            }
            K::CoinTypeGlobalPause => {
                let Some(ErrorDetails::CoinDenyListError(CoinDenyListError {
                    coin_type,
                    ..
                })) = &value.error_details
                else {
                    return Err(TryFromProtoError::missing("coin_deny_list_error"));
                };
                Self::CoinTypeGlobalPause {
                    coin_type: coin_type
                        .as_ref()
                        .ok_or_else(|| TryFromProtoError::missing("coin_type"))?
                        .to_owned(),
                }
            }
            K::ExecutionCanceledDueToRandomnessUnavailable => {
                Self::ExecutionCanceledDueToRandomnessUnavailable
            }
            K::MoveVectorElemTooBig => {
                let Some(ErrorDetails::SizeError(SizeError { size, max_size })) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing(ExecutionError::SIZE_ERROR_FIELD));
                };

                Self::MoveVectorElemTooBig {
                    value_size: size.ok_or_else(|| TryFromProtoError::missing(SizeError::SIZE_FIELD))?,
                    max_scaled_size: max_size
                        .ok_or_else(|| TryFromProtoError::missing(SizeError::MAX_SIZE_FIELD))?,
                }
            }
            K::MoveRawValueTooBig => {
                let Some(ErrorDetails::SizeError(SizeError { size, max_size })) =
                    &value.error_details
                else {
                    return Err(TryFromProtoError::missing(ExecutionError::SIZE_ERROR_FIELD));
                };

                Self::MoveRawValueTooBig {
                    value_size: size.ok_or_else(|| TryFromProtoError::missing(SizeError::SIZE_FIELD))?,
                    max_scaled_size: max_size
                        .ok_or_else(|| TryFromProtoError::missing(SizeError::MAX_SIZE_FIELD))?,
                }
            }
            K::InvalidLinkage => Self::InvalidLinkage,
        }
        .pipe(Ok)
    }
}
