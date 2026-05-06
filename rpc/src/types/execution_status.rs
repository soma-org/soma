// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

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
    InvalidGasCoinType { object_id: Address },
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
    ModelEmbeddingCommitmentMismatch,
    ModelDecryptionKeyCommitmentMismatch,
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
    AuditWindowOpen { fill_epoch: u64, current_epoch: u64 },
    TargetAlreadyClaimed,

    // Submission errors
    ModelNotInTarget { model_id: Address, target_id: Address },
    EmbeddingDimensionMismatch { expected: u64, actual: u64 },
    InsufficientBond { required: u64, provided: u64 },
    InsufficientEmissionBalance,

    // Audit errors
    AuditWindowClosed { fill_epoch: u64, current_epoch: u64 },

    // Data size errors
    DataExceedsMaxSize { size: u64, max_size: u64 },

    // Coin errors
    InsufficientCoinBalance,
    CoinBalanceOverflow,

    // Arithmetic errors
    ArithmeticOverflow,

    // Staking errors
    ValidatorNotFound,
    StakingPoolNotFound,
    CannotReportOneself,
    ReportRecordNotFound,

    // Other errors,
    InputObjectDeleted,
    CertificateDenied,
    SharedObjectCongestion,

    // Payment-channel errors
    ChannelCallerNotPayee { expected: Address, actual: Address },
    ChannelCallerNotPayer { expected: Address, actual: Address },
    ChannelVoucherNotMonotonic { cumulative: u64, settled: u64 },
    ChannelOverspend { cumulative: u64, available: u64 },
    ChannelGraceNotElapsed { now_ms: u64, earliest_ms: u64 },
    ChannelCloseAlreadyPending,
    ChannelNoCloseRequest,
    ChannelInvalidVoucherSignature { reason: String },
    ChannelAmountZero,
    ChannelInvalidInput { reason: String },
    ChannelCoinTypeMismatch,
    NotAChannel { object_id: Address },
    ChannelClockMissing,

    // Generic error for cases not covered by specific variants
    OtherError(String),
}
