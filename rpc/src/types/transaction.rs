use serde::{Deserialize, Serialize};

/// A transaction
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// transaction = %x00 transaction-v1
///
/// transaction-v1 = transaction-kind address gas-payment transaction-expiration
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transaction {
    pub kind: TransactionKind,
    pub sender: Address,
    pub gas_payment: GasPayment,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct SignedTransaction {
    pub transaction: Transaction,
    pub signatures: Vec<UserSignature>,
}

/// Payment information for executing a transaction
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// gas-payment = (vector object-ref) ; gas coin objects
///               address             ; owner
///               u64                 ; price
///               u64                 ; budget
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct GasPayment {
    pub objects: Vec<ObjectReference>,

    /// Owner of the gas objects, either the transaction sender or a sponsor
    pub owner: Address,
}

/// Transaction type
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// transaction-kind    =  %x00 ptb
///                     =/ %x01 change-epoch
///                     =/ %x02 genesis-transaction
///                     =/ %x03 consensus-commit-prologue
///                     =/ %x04 authenticator-state-update
///                     =/ %x05 (vector end-of-epoch-transaction-kind)
///                     =/ %x06 randomness-state-update
///                     =/ %x07 consensus-commit-prologue-v2
///                     =/ %x08 consensus-commit-prologue-v3
///                     =/ %x09 consensus-commit-prologue-v4
///                     =/ %x0A ptb
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[non_exhaustive]
pub enum TransactionKind {
    /// System transaction used to end an epoch.
    ///
    /// The ChangeEpoch variant is now deprecated (but the ChangeEpoch struct is still used by
    /// EndOfEpochTransaction below).
    ChangeEpoch(ChangeEpoch),

    /// Transaction used to initialize the chain state.
    ///
    /// Only valid if in the genesis checkpoint (0) and if this is the very first transaction ever
    /// executed on the chain.
    Genesis(GenesisTransaction),

    /// Consensus commit update
    ConsensusCommitPrologue(ConsensusCommitPrologue),
}

/// V1 of the consensus commit prologue system transaction
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// consensus-commit-prologue = u64 u64 u64
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ConsensusCommitPrologue {
    /// Epoch of the commit prologue transaction
    pub epoch: u64,

    /// Consensus round of the commit
    pub round: u64,

    /// Unix timestamp from consensus
    pub commit_timestamp_ms: CheckpointTimestamp,
}

/// System transaction used to change the epoch
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// change-epoch = u64  ; next epoch
///                u64  ; epoch start timestamp
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChangeEpoch {
    /// The next (to become) epoch ID.
    pub epoch: EpochId,

    /// Unix timestamp when epoch started
    pub epoch_start_timestamp_ms: u64,
}

/// The genesis transaction
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// genesis-transaction = (vector genesis-object)
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct GenesisTransaction {
    pub objects: Vec<GenesisObject>,
}
