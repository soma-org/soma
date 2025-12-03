use serde::{Deserialize, Serialize};

use crate::types::Address;

#[derive(Eq, PartialEq, Clone, Debug, Deserialize, Serialize)]
pub enum ExecutionStatus {
    /// The Transaction successfully executed.
    Success,

    /// The Transaction didn't execute successfully.
    ///
    /// Failed transactions are still committed to the blockchain but any intended effects are
    /// rolled back to prior to this transaction executing with the caveat that gas objects are
    /// still smashed and gas usage is still charged.
    Failure {
        /// The error encountered during execution.
        error: ExecutionError,
    },
}

#[derive(Eq, PartialEq, Clone, Debug, Deserialize, Serialize)]
#[non_exhaustive]
pub enum ExecutionError {
    // General transaction errors
    InsufficientGas,
    InvalidOwnership { object_id: Address },
    ObjectNotFound { object_id: Address },
    InvalidObjectType { object_id: Address },
    InvalidTransactionType,
    InvalidArguments { reason: String },

    // Validator errors
    DuplicateValidator,
    NotAValidator,
    ValidatorAlreadyRemoved,
    AdvancedToWrongEpoch,

    // Encoder errors
    DuplicateEncoder,
    NotAnEncoder,
    EncoderAlreadyRemoved,

    // Coin errors
    InsufficientCoinBalance,
    CoinBalanceOverflow,

    // Staking errors
    ValidatorNotFound,
    EncoderNotFound,
    StakingPoolNotFound,
    CannotReportOneself,
    ReportRecordNotFound,

    // Other errors,
    InputObjectDeleted,
    CertificateDenied,
    SharedObjectCongestion,

    // Generic error for cases not covered by specific variants
    OtherError(String),
}
