use thiserror::Error;

/// Errors produced by bridge node operations.
#[derive(Error, Debug)]
pub enum BridgeError {
    // --- Ethereum RPC errors ---
    #[error("Ethereum provider error: {0}")]
    ProviderError(String),

    #[error("Transient Ethereum provider error (retryable): {0}")]
    TransientProviderError(String),

    #[error("Transaction {0} not found on Ethereum")]
    TxNotFound(String),

    #[error("Transaction {0} not yet finalized")]
    TxNotFinalized(String),

    #[error("Deposit event not found in tx {0}")]
    DepositEventNotFound(String),

    // --- Signature errors ---
    #[error("Invalid ECDSA signature: {0}")]
    InvalidSignature(String),

    #[error("Insufficient committee stake: got {got}, need {required}")]
    InsufficientStake { got: u64, required: u64 },

    #[error("Unknown signer index {0}")]
    UnknownSigner(u32),

    #[error("Duplicate signature from signer {0}")]
    DuplicateSignature(u32),

    // --- Bridge state errors ---
    #[error("Bridge is paused")]
    BridgePaused,

    #[error("Deposit nonce {0} already processed")]
    NonceAlreadyProcessed(u64),

    // --- gRPC / network errors ---
    #[error("gRPC error: {0}")]
    GrpcError(#[from] tonic::Status),

    #[error("Peer connection failed: {0}")]
    PeerConnectionFailed(String),

    // --- Config errors ---
    #[error("Configuration error: {0}")]
    ConfigError(String),

    // --- Generic ---
    #[error("Internal bridge error: {0}")]
    Internal(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type BridgeResult<T> = Result<T, BridgeError>;
