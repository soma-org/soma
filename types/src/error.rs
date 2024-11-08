use std::collections::BTreeMap;

use fastcrypto::error;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tonic::Status;

use crate::{
    base::AuthorityName,
    committee::{Committee, EpochId, VotingPower},
    digests::{ObjectDigest, TransactionDigest, TransactionEffectsDigest},
    effects::ExecutionFailureStatus,
    object::{ObjectID, ObjectRef, Version},
};
pub type SomaResult<T = ()> = Result<T, SomaError>;

/// Custom error type for Sui.
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize, Error, Hash)]
pub enum SomaError {
    #[error("Invalid committee composition")]
    InvalidCommittee(String),

    #[error("Key Conversion Error: {0}")]
    KeyConversionError(String),

    #[error("Signature is not valid: {}", error)]
    InvalidSignature { error: String },

    #[error("Invalid address")]
    InvalidAddress,

    #[error("Value was not signed by the correct sender: {}", error)]
    IncorrectSigner { error: String },

    #[error("Cannot add validator that is already active or pending")]
    DuplicateValidator,

    #[error("Cannot remove validator that is not active")]
    NotAValidator,

    #[error("Cannot remove validator that is already removed")]
    ValidatorAlreadyRemoved,

    #[error("Advanced to wrong epoch")]
    AdvancedToWrongEpoch,

    // These are errors that occur when an RPC fails and is simply the utf8 message sent in a
    // Tonic::Status
    #[error("{1} - {0}")]
    RpcError(String, String),

    // Epoch related errors.
    #[error("Validator temporarily stopped processing transactions due to epoch change")]
    ValidatorHaltedAtEpochEnd,
    #[error("Operations for epoch {0} have ended")]
    EpochEnded(EpochId),
    #[error("Error when advancing epoch: {:?}", error)]
    AdvanceEpochError { error: String },

    #[error("Operation timed out")]
    TimeoutError,

    #[error("Failed to execute transaction locally by Orchestrator: {error:?}")]
    TransactionOrchestratorLocalExecutionError { error: String },

    // Errors related to the authority-consensus interface.
    #[error("Failed to submit transaction to consensus: {0}")]
    FailedToSubmitToConsensus(String),

    #[error("Authority Error: {error:?}")]
    GenericAuthorityError { error: String },

    #[error("Missing committee information for epoch {0}")]
    MissingCommitteeAtEpoch(EpochId),

    #[error("Unable to communicate with the Quorum Driver channel: {:?}", error)]
    QuorumDriverCommunicationError { error: String },

    #[error(
        "Failed to verify Tx certificate with executed effects, error: {error:?}, validator: {validator_name:?}"
    )]
    FailedToVerifyTxCertWithExecutedEffects {
        validator_name: AuthorityName,
        error: String,
    },
    #[error("Too many authority errors were detected for {}: {:?}", action, errors)]
    TooManyIncorrectAuthorities {
        errors: Vec<(AuthorityName, SomaError)>,
        action: String,
    },

    // Certificate verification and execution
    #[error(
        "Signature or certificate from wrong epoch, expected {expected_epoch}, got {actual_epoch}"
    )]
    WrongEpoch {
        expected_epoch: EpochId,
        actual_epoch: EpochId,
    },

    #[error("Expect {expected} signer signatures but got {actual}")]
    SignerSignatureNumberMismatch { expected: usize, actual: usize },
    #[error("Required Signature from {expected} is absent {:?}", actual)]
    SignerSignatureAbsent {
        expected: String,
        actual: Vec<String>,
    },
    #[error("Value was not signed by a known authority. signer: {:?}, index: {:?}, committee: {committee}", signer, index)]
    UnknownSigner {
        signer: Option<String>,
        index: Option<u32>,
        committee: Box<Committee>,
    },

    #[error("Invalid authenticator")]
    InvalidAuthenticator,

    #[error("Signatures in a certificate must form a quorum")]
    CertificateRequiresQuorum,

    #[error(
        "Validator {:?} responded multiple signatures for the same message, conflicting: {:?}",
        signer,
        conflicting_sig
    )]
    StakeAggregatorRepeatedSigner {
        signer: AuthorityName,
        conflicting_sig: bool,
    },

    #[error("Validator {authority:?} is faulty in a Byzantine manner: {reason:?}")]
    ByzantineAuthoritySuspicion {
        authority: AuthorityName,
        reason: String,
    },

    #[error("Invalid digest length. Expected {expected}, got {actual}")]
    InvalidDigestLength { expected: usize, actual: usize },

    #[error(
        "Failed to get a quorum of signed effects when processing transaction: {effects_map:?}"
    )]
    QuorumFailedToGetEffectsQuorumWhenProcessingTransaction {
        effects_map: BTreeMap<TransactionEffectsDigest, (Vec<AuthorityName>, VotingPower)>,
    },

    #[error("Transaction certificate processing failed: {err}")]
    ErrorWhileProcessingCertificate { err: String },

    // Unsupported Operations on Fullnode
    #[error("Fullnode does not support handle_certificate")]
    FullNodeCantHandleCertificate,

    #[error("Invalid transaction digest.")]
    InvalidTransactionDigest,

    #[error(
        "Could not find the referenced object {:?} at version {:?}",
        object_id,
        version
    )]
    ObjectNotFound {
        object_id: ObjectID,
        version: Option<Version>,
    },
    #[error(
        "Object {obj_ref:?} already locked by a different transaction: {pending_transaction:?}"
    )]
    ObjectLockConflict {
        obj_ref: ObjectRef,
        pending_transaction: TransactionDigest,
    },
    #[error("Object {provided_obj_ref:?} is not available for consumption, its current version: {current_version:?}")]
    ObjectVersionUnavailableForConsumption {
        provided_obj_ref: ObjectRef,
        current_version: Version,
    },
    #[error(
        "Invalid Object digest for object {object_id:?}. Expected digest : {expected_digest:?}"
    )]
    InvalidObjectDigest {
        object_id: ObjectID,
        expected_digest: ObjectDigest,
    },

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Attempt to re-initialize a transaction lock for objects {:?}.", refs)]
    ObjectLockAlreadyInitialized { refs: Vec<ObjectRef> },

    #[error("Failed to read or deserialize system state related data structures on-chain: {0}")]
    SystemStateReadError(String),
}

impl From<Status> for SomaError {
    fn from(status: Status) -> Self {
        // if status.message() == "Too many requests" {
        //     return Self::TooManyRequests;
        // }

        let result = bcs::from_bytes::<SomaError>(status.details());
        if let Ok(sui_error) = result {
            sui_error
        } else {
            Self::RpcError(
                status.message().to_owned(),
                status.code().description().to_owned(),
            )
        }
    }
}

impl From<SomaError> for Status {
    fn from(error: SomaError) -> Self {
        let bytes = bcs::to_bytes(&error).unwrap();
        Status::with_details(tonic::Code::Internal, error.to_string(), bytes.into())
    }
}

impl From<String> for SomaError {
    fn from(error: String) -> Self {
        SomaError::GenericAuthorityError { error }
    }
}

impl From<&str> for SomaError {
    fn from(error: &str) -> Self {
        SomaError::GenericAuthorityError {
            error: error.to_string(),
        }
    }
}

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub type ExecutionErrorKind = ExecutionFailureStatus;

#[derive(Debug)]
pub struct ExecutionError {
    inner: Box<ExecutionErrorInner>,
}

#[derive(Debug)]
struct ExecutionErrorInner {
    kind: ExecutionErrorKind,
    source: Option<BoxError>,
}

impl ExecutionError {
    pub fn new(kind: ExecutionErrorKind, source: Option<BoxError>) -> Self {
        Self {
            inner: Box::new(ExecutionErrorInner { kind, source }),
        }
    }

    pub fn new_with_source<E: Into<BoxError>>(kind: ExecutionErrorKind, source: E) -> Self {
        Self::new(kind, Some(source.into()))
    }

    pub fn from_kind(kind: ExecutionErrorKind) -> Self {
        Self::new(kind, None)
    }

    pub fn kind(&self) -> &ExecutionErrorKind {
        &self.inner.kind
    }

    pub fn source(&self) -> &Option<BoxError> {
        &self.inner.source
    }

    pub fn to_execution_status(&self) -> ExecutionFailureStatus {
        self.kind().clone()
    }
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExecutionError: {:?}", self)
    }
}

impl std::error::Error for ExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.inner.source.as_ref().map(|e| &**e as _)
    }
}

impl From<ExecutionErrorKind> for ExecutionError {
    fn from(kind: ExecutionErrorKind) -> Self {
        Self::from_kind(kind)
    }
}

impl From<crate::storage::storage_error::Error> for SomaError {
    fn from(e: crate::storage::storage_error::Error) -> Self {
        Self::Storage(e.to_string())
    }
}
