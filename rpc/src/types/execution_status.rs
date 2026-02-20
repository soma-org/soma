use serde::{Deserialize, Serialize};

use crate::types::Address;

#[derive(PartialEq, Clone, Debug, Deserialize, Serialize)]
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

#[derive(PartialEq, Clone, Debug, Deserialize, Serialize)]
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
    DuplicateValidatorMetadata { field: String },
    MissingProofOfPossession,
    InvalidProofOfPossession { reason: String },
    NotAValidator,
    ValidatorAlreadyRemoved,
    AdvancedToWrongEpoch,

    // Model errors
    ModelNotFound,
    NotModelOwner,
    ModelNotActive,
    ModelNotPending,
    ModelAlreadyInactive,
    ModelRevealEpochMismatch,
    ModelWeightsUrlMismatch,
    ModelNoPendingUpdate,
    ModelArchitectureVersionMismatch,
    ModelCommissionRateTooHigh,
    ModelMinStakeNotMet,

    // Target errors
    NoActiveModels,
    TargetNotFound,
    TargetNotOpen,
    TargetExpired { generation_epoch: u64, current_epoch: u64 },
    TargetNotFilled,
    ChallengeWindowOpen { fill_epoch: u64, current_epoch: u64 },
    TargetAlreadyClaimed,

    // Submission errors
    ModelNotInTarget { model_id: Address, target_id: Address },
    EmbeddingDimensionMismatch { expected: u64, actual: u64 },
    DistanceExceedsThreshold { score: f32, threshold: f32 },
    InsufficientBond { required: u64, provided: u64 },
    InsufficientEmissionBalance,

    // Challenge errors
    ChallengeWindowClosed { fill_epoch: u64, current_epoch: u64 },
    InsufficientChallengerBond { required: u64, provided: u64 },
    ChallengeNotFound { challenge_id: Address },
    ChallengeNotPending { challenge_id: Address },
    ChallengeExpired { challenge_epoch: u64, current_epoch: u64 },
    InvalidChallengeResult,
    InvalidChallengeQuorum,
    ChallengeAlreadyExists,

    // Data size errors
    DataExceedsMaxSize { size: u64, max_size: u64 },

    // Coin errors
    InsufficientCoinBalance,
    CoinBalanceOverflow,

    // Staking errors
    ValidatorNotFound,
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
