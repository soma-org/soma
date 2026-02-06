use std::collections::BTreeMap;
use std::time::Duration;

use crate::base::AuthorityName;
use crate::committee::EpochId;

use crate::{
    effects::SignedTransactionEffects,
    transaction::{CertifiedTransaction, SignedTransaction, Transaction},
};

use crate::checkpoints::CheckpointSequenceNumber;
use crate::committee::StakeUnit;
use crate::crypto::{AuthorityStrongQuorumSignInfo, ConciseAuthorityPublicKeyBytes};
use crate::digests::TransactionDigest;
use crate::effects::{
    CertifiedTransactionEffects, TransactionEffects, VerifiedCertifiedTransactionEffects,
};
use crate::error::{ErrorCategory, SomaError};
use crate::object::Object;
use crate::object::ObjectRef;
use crate::transaction::VerifiedTransaction;
use serde::{Deserialize, Serialize};
use strum::AsRefStr;
use thiserror::Error;

pub type QuorumDriverResult = Result<QuorumDriverResponse, QuorumDriverError>;

pub type QuorumDriverEffectsQueueResult =
    Result<(Transaction, QuorumDriverResponse), (TransactionDigest, QuorumDriverError)>;

pub const NON_RECOVERABLE_ERROR_MSG: &str =
    "Transaction has non recoverable errors from at least 1/3 of validators";

/// Client facing errors regarding transaction submission via Quorum Driver.
/// Every invariant needs detailed documents to instruct client handling.
#[derive(Eq, PartialEq, Clone, Debug, Error, Hash, AsRefStr)]
pub enum QuorumDriverError {
    #[error("QuorumDriver internal error: {0}.")]
    QuorumDriverInternalError(SomaError),
    #[error("Invalid user signature: {0}.")]
    InvalidUserSignature(SomaError),
    #[error(
        "Failed to sign transaction by a quorum of validators because of locked objects: {conflicting_txes:?}"
    )]
    ObjectsDoubleUsed {
        conflicting_txes: BTreeMap<TransactionDigest, (Vec<(AuthorityName, ObjectRef)>, StakeUnit)>,
    },
    #[error("Transaction timed out before reaching finality")]
    TimeoutBeforeFinality,
    #[error(
        "Transaction timed out before reaching finality. Last recorded retriable error: {last_error}"
    )]
    TimeoutBeforeFinalityWithErrors { last_error: String, attempts: u32, timeout: Duration },
    #[error(
        "Transaction failed to reach finality with transient error after {total_attempts} attempts."
    )]
    FailedWithTransientErrorAfterMaximumAttempts { total_attempts: u32 },
    #[error("{NON_RECOVERABLE_ERROR_MSG}: {errors:?}.")]
    NonRecoverableTransactionError { errors: GroupedErrors },

    #[error("Transaction is already finalized but with different user signatures")]
    TxAlreadyFinalizedWithDifferentUserSignatures,

    // Wrapped error from Transaction Driver.
    #[error("Transaction processing failed. Details: {details}")]
    TransactionFailed { category: ErrorCategory, details: String },

    #[error(
        "Transaction is already being processed in transaction orchestrator (most likely by quorum driver), wait for results"
    )]
    PendingExecutionInTransactionOrchestrator,
}

pub type GroupedErrors = Vec<(SomaError, StakeUnit, Vec<ConciseAuthorityPublicKeyBytes>)>;

#[derive(Serialize, Deserialize, Clone, Debug, schemars::JsonSchema)]
pub enum ExecuteTransactionRequestType {
    WaitForEffectsCert,
    WaitForLocalExecution,
}

#[derive(Debug)]
pub enum TransactionType {
    SingleWriter, // Txes that only use owned objects and/or immutable objects
    SharedObject, // Txes that use at least one shared object
}

/// Proof of finality of transaction effects.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum EffectsFinalityInfo {
    /// Effects are certified by a quorum of validators.
    Certified(AuthorityStrongQuorumSignInfo),

    /// Effects are included in a checkpoint.
    Checkpointed(EpochId, CheckpointSequenceNumber),

    /// A quorum of validators have acknowledged effects.
    QuorumExecuted(EpochId),
}

/// When requested to execute a transaction with WaitForLocalExecution,
/// TransactionOrchestrator attempts to execute this transaction locally
/// after it is finalized. This value represents whether the transaction
/// is confirmed to be executed on this node before the response returns.
pub type IsTransactionExecutedLocally = bool;

#[derive(Clone, Debug)]
pub struct QuorumDriverRequest {
    pub transaction: VerifiedTransaction,
}

#[derive(Debug, Clone)]
pub struct QuorumDriverResponse {
    pub effects_cert: VerifiedCertifiedTransactionEffects,

    // Input objects will only be populated in the happy path
    pub input_objects: Option<Vec<Object>>,
    // Output objects will only be populated in the happy path
    pub output_objects: Option<Vec<Object>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExecuteTransactionRequest {
    pub transaction: Transaction,

    pub include_input_objects: bool,
    pub include_output_objects: bool,
}

impl ExecuteTransactionRequest {
    pub fn new<T: Into<Transaction>>(transaction: T) -> Self {
        Self {
            transaction: transaction.into(),

            include_input_objects: false,
            include_output_objects: false,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExecuteTransactionResponse {
    pub effects: FinalizedEffects,

    // Input objects will only be populated in the happy path
    pub input_objects: Option<Vec<Object>>,
    // Output objects will only be populated in the happy path
    pub output_objects: Option<Vec<Object>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FinalizedEffects {
    pub effects: TransactionEffects,
    pub finality_info: EffectsFinalityInfo,
}

impl FinalizedEffects {
    pub fn new_from_effects_cert(effects_cert: CertifiedTransactionEffects) -> Self {
        let (data, sig) = effects_cert.into_data_and_sig();
        Self { effects: data, finality_info: EffectsFinalityInfo::Certified(sig) }
    }

    pub fn epoch(&self) -> EpochId {
        match &self.finality_info {
            EffectsFinalityInfo::Certified(cert) => cert.epoch,
            EffectsFinalityInfo::Checkpointed(epoch, _) => *epoch,
            EffectsFinalityInfo::QuorumExecuted(epoch) => *epoch,
        }
    }

    pub fn data(&self) -> &TransactionEffects {
        &self.effects
    }
}

/// This enum represents all possible states of a response returned from
/// the safe client. Note that [struct SignedTransaction] and
/// [struct SignedTransactionEffects] are represented as an Envelope
/// instead of an VerifiedEnvelope. This is because the verification is
/// now performed by the authority aggregator as an aggregated signature,
/// instead of in SafeClient.
#[derive(Clone, Debug)]
pub enum PlainTransactionInfoResponse {
    Signed(SignedTransaction),
    ExecutedWithCert(CertifiedTransaction, SignedTransactionEffects),
    ExecutedWithoutCert(Transaction, SignedTransactionEffects),
}

impl PlainTransactionInfoResponse {
    pub fn is_executed(&self) -> bool {
        match self {
            PlainTransactionInfoResponse::Signed(_) => false,
            PlainTransactionInfoResponse::ExecutedWithCert(_, _)
            | PlainTransactionInfoResponse::ExecutedWithoutCert(_, _) => true,
        }
    }
}
