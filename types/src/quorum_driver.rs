use std::collections::BTreeMap;

use crate::{
    base::AuthorityName,
    committee::{EpochId, VotingPower},
    crypto::{AuthorityStrongQuorumSignInfo, ConciseAuthorityPublicKeyBytes},
    digests::TransactionDigest,
    effects::{
        CertifiedTransactionEffects, SignedTransactionEffects, TransactionEffects,
        VerifiedCertifiedTransactionEffects,
    },
    error::SomaError,
    transaction::{CertifiedTransaction, SignedTransaction, Transaction, VerifiedTransaction},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExecuteTransactionRequest {
    pub transaction: Transaction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExecuteTransactionResponse {
    pub effects: FinalizedEffects,
}

#[derive(Serialize, Deserialize, Clone, Debug, schemars::JsonSchema)]
pub enum ExecuteTransactionRequestType {
    WaitForEffectsCert,
    WaitForLocalExecution,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum EffectsFinalityInfo {
    Certified(AuthorityStrongQuorumSignInfo),
    // Checkpointed(EpochId, CheckpointSequenceNumber),
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
}

pub type QuorumDriverResult = Result<QuorumDriverResponse, QuorumDriverError>;

pub type QuorumDriverEffectsQueueResult =
    Result<(Transaction, QuorumDriverResponse), (TransactionDigest, QuorumDriverError)>;

pub const NON_RECOVERABLE_ERROR_MSG: &str =
    "Transaction has non recoverable errors from at least 1/3 of validators";

/// Client facing errors regarding transaction submission via Quorum Driver.
/// Every invariant needs detailed documents to instruct client handling.
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize, Error, Hash)]
pub enum QuorumDriverError {
    #[error("QuorumDriver internal error: {0:?}.")]
    QuorumDriverInternalError(SomaError),
    #[error("Invalid user signature: {0:?}.")]
    InvalidUserSignature(SomaError),
    #[error(
        "Failed to sign transaction by a quorum of validators because of locked objects: {:?}, retried a conflicting transaction {:?}, success: {:?}",
        conflicting_txes,
        retried_tx,
        retried_tx_success
    )]
    ObjectsDoubleUsed {
        conflicting_txes: BTreeMap<TransactionDigest, (Vec<AuthorityName>, VotingPower)>,
        retried_tx: Option<TransactionDigest>,
        retried_tx_success: Option<bool>,
    },
    #[error("Transaction timed out before reaching finality")]
    TimeoutBeforeFinality,
    #[error("Transaction failed to reach finality with transient error after {total_attempts} attempts.")]
    FailedWithTransientErrorAfterMaximumAttempts { total_attempts: u32 },
    #[error("{NON_RECOVERABLE_ERROR_MSG}: {errors:?}.")]
    NonRecoverableTransactionError { errors: GroupedErrors },
    #[error("Transaction is not processed because {overloaded_stake} of validators by stake are overloaded with certificates pending execution.")]
    SystemOverload {
        overloaded_stake: VotingPower,
        errors: GroupedErrors,
    },
    #[error("Transaction is already finalized but with different user signatures")]
    TxAlreadyFinalizedWithDifferentUserSignatures,
    #[error("Transaction is not processed because {overload_stake} of validators are overloaded and asked client to retry after {retry_after_secs}.")]
    SystemOverloadRetryAfter {
        overload_stake: VotingPower,
        errors: GroupedErrors,
        retry_after_secs: u64,
    },
}

pub type GroupedErrors = Vec<(SomaError, VotingPower, Vec<ConciseAuthorityPublicKeyBytes>)>;

/// This enum represents all possible states of a response returned from
/// the client. Note that [struct SignedTransaction] is represented as an Envelope
/// instead of an VerifiedEnvelope. This is because the verification is
/// now performed by the authority aggregator as an aggregated signature,
/// instead of in the client.
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
            PlainTransactionInfoResponse::ExecutedWithCert(_, _) => true,
            PlainTransactionInfoResponse::ExecutedWithoutCert(_, _) => true,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FinalizedEffects {
    pub effects: TransactionEffects,
    pub finality_info: EffectsFinalityInfo,
}

impl FinalizedEffects {
    pub fn new_from_effects_cert(effects_cert: CertifiedTransactionEffects) -> Self {
        let (data, sig) = effects_cert.into_data_and_sig();
        Self {
            effects: data,
            finality_info: EffectsFinalityInfo::Certified(sig),
        }
    }

    pub fn epoch(&self) -> EpochId {
        match &self.finality_info {
            EffectsFinalityInfo::Certified(cert) => cert.epoch,
            // EffectsFinalityInfo::Checkpointed(epoch, _) => *epoch,
        }
    }
}
