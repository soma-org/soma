use fastcrypto::error;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tonic::Status;

use crate::{
    base::AuthorityName,
    committee::{Committee, EpochId},
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
