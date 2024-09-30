use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    committee::{Committee, EpochId},
    crypto::{
        default_hash, AuthoritySignInfo, AuthoritySignInfoTrait, AuthorityStrongQuorumSignInfo,
        EmptySignInfo,
    },
    digests::{TransactionDigest, TransactionEffectsDigest},
    envelope::{Envelope, Message, TrustedEnvelope, VerifiedEnvelope},
    error::{SomaError, SomaResult},
    intent::{Intent, IntentScope},
};

/// The response from processing a transaction or a certified transaction
#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct TransactionEffects {
    /// The status of the execution
    status: ExecutionStatus,
    /// The epoch when this transaction was executed.
    executed_epoch: EpochId,
    /// The transaction digest
    transaction_digest: TransactionDigest,
}

impl TransactionEffectsAPI for TransactionEffects {
    fn status(&self) -> &ExecutionStatus {
        &self.status
    }

    fn into_status(self) -> ExecutionStatus {
        self.status
    }

    fn executed_epoch(&self) -> EpochId {
        self.executed_epoch
    }

    fn transaction_digest(&self) -> &TransactionDigest {
        &self.transaction_digest
    }
}

impl TransactionEffects {
    pub fn new(
        status: ExecutionStatus,
        executed_epoch: EpochId,
        transaction_digest: TransactionDigest,
    ) -> Self {
        Self {
            status,
            executed_epoch,
            transaction_digest,
        }
    }
}

impl Default for TransactionEffects {
    fn default() -> Self {
        Self {
            status: ExecutionStatus::Success,
            executed_epoch: 0,
            transaction_digest: TransactionDigest::default(),
        }
    }
}

impl Message for TransactionEffects {
    type DigestType = TransactionEffectsDigest;
    const SCOPE: IntentScope = IntentScope::TransactionData;

    fn digest(&self) -> Self::DigestType {
        TransactionEffectsDigest::new(default_hash(self))
    }
}

pub trait TransactionEffectsAPI {
    fn status(&self) -> &ExecutionStatus;
    fn into_status(self) -> ExecutionStatus;
    fn executed_epoch(&self) -> EpochId;
    fn transaction_digest(&self) -> &TransactionDigest;
}

pub type TransactionEffectsEnvelope<S> = Envelope<TransactionEffects, S>;
pub type UnsignedTransactionEffects = TransactionEffectsEnvelope<EmptySignInfo>;
pub type SignedTransactionEffects = TransactionEffectsEnvelope<AuthoritySignInfo>;
pub type CertifiedTransactionEffects = TransactionEffectsEnvelope<AuthorityStrongQuorumSignInfo>;

pub type TrustedSignedTransactionEffects = TrustedEnvelope<TransactionEffects, AuthoritySignInfo>;
pub type VerifiedTransactionEffectsEnvelope<S> = VerifiedEnvelope<TransactionEffects, S>;
pub type VerifiedSignedTransactionEffects = VerifiedTransactionEffectsEnvelope<AuthoritySignInfo>;
pub type VerifiedCertifiedTransactionEffects =
    VerifiedTransactionEffectsEnvelope<AuthorityStrongQuorumSignInfo>;

impl CertifiedTransactionEffects {
    pub fn verify_authority_signatures(&self, committee: &Committee) -> SomaResult {
        self.auth_sig()
            .verify_secure(self.data(), Intent::soma_transaction(), committee)
    }

    pub fn verify(self, committee: &Committee) -> SomaResult<VerifiedCertifiedTransactionEffects> {
        self.verify_authority_signatures(committee)?;
        Ok(VerifiedCertifiedTransactionEffects::new_from_verified(self))
    }
}

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Success,
    /// Gas used in the failed case, and the error.
    Failure {
        /// The error
        error: ExecutionFailureStatus,
    },
}

impl ExecutionStatus {
    pub fn new_failure(error: ExecutionFailureStatus) -> ExecutionStatus {
        ExecutionStatus::Failure { error }
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, ExecutionStatus::Success { .. })
    }

    pub fn is_err(&self) -> bool {
        matches!(self, ExecutionStatus::Failure { .. })
    }

    pub fn unwrap(&self) {
        match self {
            ExecutionStatus::Success => {}
            ExecutionStatus::Failure { .. } => {
                panic!("Unable to unwrap() on {:?}", self);
            }
        }
    }

    pub fn unwrap_err(self) -> ExecutionFailureStatus {
        match self {
            ExecutionStatus::Success { .. } => {
                panic!("Unable to unwrap() on {:?}", self);
            }
            ExecutionStatus::Failure { error } => error,
        }
    }
}

#[derive(Eq, PartialEq, Clone, Debug, Serialize, Deserialize, Error)]
pub enum ExecutionFailureStatus {
    //
    // General transaction errors
    //
    #[error("Insufficient Gas.")]
    InsufficientGas,

    //
    // Coin errors
    //
    #[error("Insufficient coin balance for operation.")]
    InsufficientCoinBalance,
    #[error("The coin balance overflows u64")]
    CoinBalanceOverflow,

    //
    // Post-execution errors
    //
    // Indicates the effects from the transaction are too large
    #[error(
        "Effects of size {current_size} bytes too large. \
    Limit is {max_size} bytes"
    )]
    EffectsTooLarge { current_size: u64, max_size: u64 },

    #[error("Certificate is cancelled because randomness could not be generated this epoch")]
    ExecutionCancelledDueToRandomnessUnavailable,

    #[error("Soma Error {0}")]
    SomaError(SomaError),
}
