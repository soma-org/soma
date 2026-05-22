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

    /// The Eth-side `SomaBridge` contract reports this nonce as already
    /// processed (the `isMessageProcessed[nonce]` mapping is `true`).
    /// Surfaced when an outbound withdrawal release tx reverts because
    /// a peer bridge-node already submitted the same nonce, or this node
    /// previously submitted and crashed before persisting the WAL
    /// idempotency entry. Callers should treat this as terminal-success
    /// for the nonce and `mark_relayed` to stop further retries.
    #[error("Nonce {0} already processed on Eth")]
    NonceAlreadyProcessed(u64),

    // --- gRPC / network errors ---
    #[error("gRPC error: {0}")]
    GrpcError(#[from] tonic::Status),

    #[error("Peer connection failed: {0}")]
    PeerConnectionFailed(String),

    // --- Config errors ---
    #[error("Configuration error: {0}")]
    ConfigError(String),

    // --- Governance / server errors (Sui parity) ---
    #[error("Action is not a governance action")]
    ActionIsNotGovernanceAction,

    #[error("Governance action is not in the operator-approved whitelist")]
    GovernanceActionIsNotApproved,

    #[error("Invalid bridge client request: {0}")]
    InvalidBridgeClientRequest(String),

    #[error("Bridge event found in unrecognized Eth contract")]
    BridgeEventInUnrecognizedEthContract,

    #[error("No bridge event at the claimed tx + index")]
    NoBridgeEventsInTxPosition,

    #[error("Mismatched action between request and signature")]
    MismatchedAction,

    // --- Generic ---
    #[error("Internal bridge error: {0}")]
    Internal(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type BridgeResult<T> = Result<T, BridgeError>;
