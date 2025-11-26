/// Chain ID of the current chain
pub const X_SOMA_CHAIN_ID: &str = "x-soma-chain-id";

/// Chain name of the current chain
pub const X_SOMA_CHAIN: &str = "x-soma-chain";

/// Current checkpoint height
pub const X_SOMA_CHECKPOINT_HEIGHT: &str = "x-soma-checkpoint-height";

/// Lowest available checkpoint for which transaction and checkpoint data can be requested.
///
/// Specifically this is the lowest checkpoint for which the following data can be requested:
///  - checkpoints
///  - transactions
///  - effects
pub const X_SOMA_LOWEST_AVAILABLE_CHECKPOINT: &str = "x-soma-lowest-available-checkpoint";

/// Lowest available checkpoint for which object data can be requested.
///
/// Specifically this is the lowest checkpoint for which input/output object data will be
/// available.
pub const X_SOMA_LOWEST_AVAILABLE_CHECKPOINT_OBJECTS: &str =
    "x-soma-lowest-available-checkpoint-objects";

/// Current epoch of the chain
pub const X_SOMA_EPOCH: &str = "x-soma-epoch";

/// Current timestamp of the chain - represented as number of milliseconds from the Unix epoch
pub const X_SOMA_TIMESTAMP_MS: &str = "x-soma-timestamp-ms";

/// Current timestamp of the chain - encoded in the [RFC 3339] format.
///
/// [RFC 3339]: https://www.ietf.org/rfc/rfc3339.txt
pub const X_SOMA_TIMESTAMP: &str = "x-soma-timestamp";
