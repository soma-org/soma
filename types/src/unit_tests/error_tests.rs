// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::base::SomaAddress;
use crate::crypto::AuthorityPublicKeyBytes;
use crate::digests::TransactionDigest;
use crate::effects::ExecutionFailureStatus;
use crate::error::{ErrorCategory, SomaError};
use crate::object::{ObjectID, ObjectType};

/// Each SomaError variant we can easily construct has a non-empty Display impl.
#[test]
fn test_soma_error_display() {
    let variants: Vec<SomaError> = vec![
        SomaError::InvalidCommittee("bad committee".into()),
        SomaError::SignatureKeyGenError("keygen failed".into()),
        SomaError::InvalidPrivateKey,
        SomaError::KeyConversionError("conversion failed".into()),
        SomaError::InvalidSignature { error: "bad sig".into() },
        SomaError::InvalidAddress,
        SomaError::IncorrectSigner { error: "wrong signer".into() },
        SomaError::RpcError("rpc msg".into(), "code".into()),
        SomaError::ValidatorHaltedAtEpochEnd,
        SomaError::EpochEnded(5),
        SomaError::AdvanceEpochError { error: "advance failed".into() },
        SomaError::TimeoutError,
        SomaError::TransactionOrchestratorLocalExecutionError { error: "orchestrator fail".into() },
        SomaError::FailedToSubmitToConsensus("submit fail".into()),
        SomaError::GenericAuthorityError { error: "authority err".into() },
        SomaError::MissingCommitteeAtEpoch(10),
        SomaError::QuorumDriverCommunicationError { error: "quorum err".into() },
        SomaError::InvalidAuthenticator,
        SomaError::CertificateRequiresQuorum,
        SomaError::InvalidDigestLength { expected: 32, actual: 16 },
        SomaError::ErrorWhileProcessingCertificate { err: "cert err".into() },
        SomaError::FullNodeCantHandleCertificate,
        SomaError::InvalidTransactionDigest,
        SomaError::ObjectNotFound { object_id: ObjectID::ZERO, version: None },
        SomaError::Storage("storage err".into()),
        SomaError::SystemStateReadError("sys state err".into()),
        SomaError::NetworkConfig("net config".into()),
        SomaError::NetworkClientConnection("client conn".into()),
        SomaError::NetworkServerConnection("server conn".into()),
        SomaError::Consensus("consensus err".into()),
        SomaError::UnexpectedOwnerType,
        SomaError::DuplicateObjectRefInput,
        SomaError::SerializationFailure("serialization err".into()),
        SomaError::FailedVDF("vdf err".into()),
        SomaError::ShardSamplingError("shard err".into()),
        SomaError::InvalidFinalityProof("finality proof err".into()),
        SomaError::VerifiedCheckpointNotFound(42),
        SomaError::VerifiedCheckpointDigestNotFound("digest".into()),
        SomaError::LatestCheckpointSequenceNumberNotFound,
        SomaError::TransactionSerializationError { error: "ser err".into() },
        SomaError::TransactionDeserializationError { error: "de err".into() },
        SomaError::TransactionEffectsSerializationError { error: "eff ser err".into() },
        SomaError::TransactionEffectsDeserializationError { error: "eff de err".into() },
        SomaError::ObjectSerializationError { error: "obj ser err".into() },
        SomaError::ObjectDeserializationError { error: "obj de err".into() },
        SomaError::FileIOError("file io err".into()),
        SomaError::GrpcMessageSerializeError {
            type_info: "SomeType".into(),
            error: "grpc ser".into(),
        },
        SomaError::GrpcMessageDeserializeError {
            type_info: "SomeType".into(),
            error: "grpc de".into(),
        },
        SomaError::InvalidRequest("invalid req".into()),
        SomaError::TotalTransactionSizeTooLargeInBatch { size: 1000, limit: 500 },
        SomaError::UnsupportedFeatureError { error: "unsupported".into() },
        SomaError::TooManyRequests,
        SomaError::IncorrectUserSignature { error: "user sig err".into() },
        SomaError::GenesisTransactionNotFound,
        SomaError::Unknown("unknown err".into()),
        SomaError::TransactionNotFound { digest: TransactionDigest::ZERO },
        SomaError::TooManyTransactionsPendingConsensus,
        SomaError::ValidatorConsensusLagging { round: 10, last_committed_round: 5 },
        SomaError::ExecutionError("exec err".into()),
        SomaError::TxAlreadyFinalizedWithDifferentUserSigs,
        SomaError::TransactionDenied { error: "denied".into() },
        SomaError::SharedObjectStartingVersionMismatch,
        SomaError::NotSharedObjectError,
        SomaError::InvalidSequenceNumber,
        SomaError::NotOwnedObjectError,
        SomaError::MutableObjectUsedMoreThanOnce { object_id: ObjectID::ZERO },
        SomaError::ObjectInputArityViolation,
        SomaError::MutableParameterExpected { object_id: ObjectID::ZERO },
        SomaError::GasPaymentError("gas err".into()),
        SomaError::EncoderServiceUnavailable,
        SomaError::TransactionNotFinalized,
        SomaError::NotEmbedDataTransaction,
        SomaError::TransactionFailed("tx failed".into()),
        SomaError::VdfComputationFailed("vdf computation failed".into()),
    ];

    for variant in &variants {
        let display = format!("{}", variant);
        assert!(!display.is_empty(), "Display for {:?} should be non-empty", variant);
    }
}

/// All ExecutionFailureStatus variants we can easily construct display properly.
#[test]
fn test_execution_failure_status_variants() {
    let variants: Vec<ExecutionFailureStatus> = vec![
        ExecutionFailureStatus::InsufficientGas,
        ExecutionFailureStatus::InvalidOwnership {
            object_id: ObjectID::ZERO,
            expected_owner: SomaAddress::ZERO,
            actual_owner: None,
        },
        ExecutionFailureStatus::ObjectNotFound { object_id: ObjectID::ZERO },
        ExecutionFailureStatus::InvalidObjectType {
            object_id: ObjectID::ZERO,
            expected_type: ObjectType::Coin,
            actual_type: ObjectType::SystemState,
        },
        ExecutionFailureStatus::InvalidTransactionType,
        ExecutionFailureStatus::InvalidArguments { reason: "bad args".into() },
        ExecutionFailureStatus::DuplicateValidator,
        ExecutionFailureStatus::NotAValidator,
        ExecutionFailureStatus::ValidatorAlreadyRemoved,
        ExecutionFailureStatus::AdvancedToWrongEpoch,
        ExecutionFailureStatus::ModelNotFound,
        ExecutionFailureStatus::NotModelOwner,
        ExecutionFailureStatus::ModelNotActive,
        ExecutionFailureStatus::ModelNotPending,
        ExecutionFailureStatus::ModelAlreadyInactive,
        ExecutionFailureStatus::ModelRevealEpochMismatch,
        ExecutionFailureStatus::ModelEmbeddingCommitmentMismatch,
        ExecutionFailureStatus::ModelDecryptionKeyCommitmentMismatch,
        ExecutionFailureStatus::ModelNoPendingUpdate,
        ExecutionFailureStatus::ModelArchitectureVersionMismatch,
        ExecutionFailureStatus::ModelCommissionRateTooHigh,
        ExecutionFailureStatus::ModelMinStakeNotMet,
        ExecutionFailureStatus::NoActiveModels,
        ExecutionFailureStatus::TargetNotFound,
        ExecutionFailureStatus::TargetNotOpen,
        ExecutionFailureStatus::TargetExpired { generation_epoch: 1, current_epoch: 5 },
        ExecutionFailureStatus::TargetNotFilled,
        ExecutionFailureStatus::AuditWindowOpen { fill_epoch: 1, current_epoch: 2 },
        ExecutionFailureStatus::TargetAlreadyClaimed,
        ExecutionFailureStatus::ModelNotInTarget {
            model_id: ObjectID::ZERO,
            target_id: ObjectID::ZERO,
        },
        ExecutionFailureStatus::EmbeddingDimensionMismatch { expected: 512, actual: 256 },
        ExecutionFailureStatus::InsufficientBond { required: 1000, provided: 500 },
        ExecutionFailureStatus::DataExceedsMaxSize { size: 2000, max_size: 1000 },
        ExecutionFailureStatus::InsufficientEmissionBalance,
        ExecutionFailureStatus::AuditWindowClosed { fill_epoch: 1, current_epoch: 5 },
        ExecutionFailureStatus::InsufficientCoinBalance,
        ExecutionFailureStatus::CoinBalanceOverflow,
        ExecutionFailureStatus::ValidatorNotFound,
        ExecutionFailureStatus::StakingPoolNotFound,
        ExecutionFailureStatus::CannotReportOneself,
        ExecutionFailureStatus::ReportRecordNotFound,
        ExecutionFailureStatus::InputObjectDeleted,
        ExecutionFailureStatus::CertificateDenied,
        ExecutionFailureStatus::ExecutionCancelledDueToSharedObjectCongestion,
        ExecutionFailureStatus::SomaError(SomaError::TimeoutError),
    ];

    for variant in &variants {
        let display = format!("{}", variant);
        assert!(!display.is_empty(), "Display for {:?} should be non-empty", variant);
    }
}

/// Validate which SomaError variants are retryable vs permanent, and that
/// the categorization flag is set correctly.
#[test]
fn test_soma_error_is_retryable() {
    // Retryable errors (retryable=true, categorized=true)
    let retryable_errors: Vec<SomaError> = vec![
        SomaError::RpcError("msg".into(), "code".into()),
        SomaError::ValidatorHaltedAtEpochEnd,
        SomaError::MissingCommitteeAtEpoch(1),
        SomaError::WrongEpoch { expected_epoch: 1, actual_epoch: 2 },
        SomaError::EpochEnded(1),
        SomaError::ObjectNotFound { object_id: ObjectID::ZERO, version: None },
    ];

    for err in &retryable_errors {
        let (retryable, categorized) = err.is_retryable();
        assert!(retryable, "Expected {:?} to be retryable", err);
        assert!(categorized, "Expected {:?} to be categorized", err);
    }

    // Non-retryable errors (retryable=false, categorized=true)
    let non_retryable_errors: Vec<SomaError> = vec![
        SomaError::ExecutionError("exec err".into()),
        SomaError::ByzantineAuthoritySuspicion {
            authority: AuthorityPublicKeyBytes::ZERO,
            reason: "faulty".into(),
        },
        SomaError::TxAlreadyFinalizedWithDifferentUserSigs,
        SomaError::TooManyRequests,
    ];

    for err in &non_retryable_errors {
        let (retryable, categorized) = err.is_retryable();
        assert!(!retryable, "Expected {:?} to NOT be retryable", err);
        assert!(categorized, "Expected {:?} to be categorized", err);
    }

    // Uncategorized errors (retryable=false, categorized=false)
    let uncategorized_errors: Vec<SomaError> = vec![
        SomaError::InvalidCommittee("test".into()),
        SomaError::TimeoutError,
        SomaError::InvalidAddress,
        SomaError::Unknown("test".into()),
    ];

    for err in &uncategorized_errors {
        let (retryable, categorized) = err.is_retryable();
        assert!(!retryable, "Expected {:?} to NOT be retryable (uncategorized)", err);
        assert!(!categorized, "Expected {:?} to NOT be categorized", err);
    }
}

/// SomaError::categorize returns expected ErrorCategory for known variants.
#[test]
fn test_soma_error_categorize() {
    // Aborted category
    assert_eq!(
        SomaError::ObjectNotFound { object_id: ObjectID::ZERO, version: None }.categorize(),
        ErrorCategory::Aborted,
    );

    // InvalidTransaction category
    assert_eq!(
        SomaError::InvalidSignature { error: "bad".into() }.categorize(),
        ErrorCategory::InvalidTransaction,
    );
    assert_eq!(
        SomaError::SignerSignatureNumberMismatch { expected: 1, actual: 2 }.categorize(),
        ErrorCategory::InvalidTransaction,
    );

    // LockConflict category
    assert_eq!(
        SomaError::ObjectLockConflict {
            obj_ref: (ObjectID::ZERO, 0.into(), crate::digests::ObjectDigest::MIN),
            pending_transaction: TransactionDigest::ZERO,
        }
        .categorize(),
        ErrorCategory::LockConflict,
    );

    // Internal category
    assert_eq!(SomaError::Unknown("test".into()).categorize(), ErrorCategory::Internal,);
    assert_eq!(
        SomaError::UnsupportedFeatureError { error: "disabled".into() }.categorize(),
        ErrorCategory::Internal,
    );

    // Unavailable category
    assert_eq!(SomaError::TimeoutError.categorize(), ErrorCategory::Unavailable,);

    // Submission retriability based on category
    assert!(ErrorCategory::Aborted.is_submission_retriable());
    assert!(ErrorCategory::Unavailable.is_submission_retriable());
    assert!(!ErrorCategory::InvalidTransaction.is_submission_retriable());
    assert!(!ErrorCategory::LockConflict.is_submission_retriable());
    assert!(!ErrorCategory::Internal.is_submission_retriable());
}
