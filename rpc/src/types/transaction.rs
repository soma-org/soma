use serde::{Deserialize, Serialize};

use crate::types::{Address, CommitTimestamp, EpochId, Object, ObjectReference, UserSignature};

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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct GasPayment {
    pub objects: Vec<ObjectReference>,
}

/// Transaction type
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
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

    // Validator management
    AddValidator(AddValidatorArgs),
    RemoveValidator(RemoveValidatorArgs),
    ReportValidator {
        reportee: Address,
    },
    UndoReportValidator {
        reportee: Address,
    },
    UpdateValidatorMetadata(UpdateValidatorMetadataArgs),
    SetCommissionRate {
        new_rate: u64,
    },

    // Encoder management
    AddEncoder(AddEncoderArgs),
    RemoveEncoder(RemoveEncoderArgs),
    ReportEncoder {
        reportee: Address,
    },
    UndoReportEncoder {
        reportee: Address,
    },
    UpdateEncoderMetadata(UpdateEncoderMetadataArgs),
    SetEncoderCommissionRate {
        new_rate: u64,
    },
    SetEncoderBytePrice {
        new_price: u64,
    },

    // Transfers and payments
    TransferCoin {
        coin: ObjectReference,
        amount: u64,
        recipient: Address,
    },
    PayCoins {
        coins: Vec<ObjectReference>,
        amounts: Vec<u64>,
        recipients: Vec<Address>,
    },
    TransferObjects {
        objects: Vec<ObjectReference>,
        recipient: Address,
    },

    // Staking
    AddStake {
        address: Address,
        coin_ref: ObjectReference,
        amount: u64,
    },
    AddStakeToEncoder {
        encoder_address: Address,
        coin_ref: ObjectReference,
        amount: u64,
    },
    WithdrawStake {
        staked_soma: ObjectReference,
    },

    // Shard operations
    EmbedData {
        digest: String,
        data_size_bytes: usize,
        coin_ref: ObjectReference,
    },
    ClaimEscrow {
        shard_input_ref: ObjectReference,
    },
    ReportScores {
        shard_input_ref: ObjectReference,
        scores: Vec<u8>,
        encoder_aggregate_signature: Vec<u8>,
        signers: Vec<String>,
    },
}

// Supporting types for validator management

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct AddValidatorArgs {
    pub pubkey_bytes: Vec<u8>,
    pub network_pubkey_bytes: Vec<u8>,
    pub worker_pubkey_bytes: Vec<u8>,
    pub net_address: Vec<u8>,
    pub p2p_address: Vec<u8>,
    pub primary_address: Vec<u8>,
    pub encoder_validator_address: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct RemoveValidatorArgs {
    pub pubkey_bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct UpdateValidatorMetadataArgs {
    pub next_epoch_network_address: Option<Vec<u8>>,
    pub next_epoch_p2p_address: Option<Vec<u8>>,
    pub next_epoch_primary_address: Option<Vec<u8>>,
    pub next_epoch_protocol_pubkey: Option<Vec<u8>>,
    pub next_epoch_worker_pubkey: Option<Vec<u8>>,
    pub next_epoch_network_pubkey: Option<Vec<u8>>,
}

// Supporting types for encoder management

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct AddEncoderArgs {
    pub encoder_pubkey_bytes: Vec<u8>,
    pub network_pubkey_bytes: Vec<u8>,
    pub internal_network_address: Vec<u8>,
    pub external_network_address: Vec<u8>,
    pub object_server_address: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct RemoveEncoderArgs {
    pub encoder_pubkey_bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct UpdateEncoderMetadataArgs {
    pub next_epoch_external_network_address: Option<Vec<u8>>,
    pub next_epoch_internal_network_address: Option<Vec<u8>>,
    pub next_epoch_network_pubkey: Option<Vec<u8>>,
    pub next_epoch_object_server_address: Option<Vec<u8>>,
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
    pub commit_timestamp_ms: CommitTimestamp,
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
    pub objects: Vec<Object>,
}
