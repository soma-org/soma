use serde::{Deserialize, Serialize};

use crate::types::Digest;
use crate::types::{Address, CommitTimestamp, EpochId, Object, ObjectReference, UserSignature};
use serde::ser::SerializeSeq;

use serde::Deserializer;
use serde::Serializer;
use serde_with::DeserializeAs;
use serde_with::SerializeAs;

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
    pub gas_payment: Vec<ObjectReference>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct SignedTransaction {
    pub transaction: Transaction,
    pub signatures: Vec<UserSignature>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct MetadataV1 {
    pub checksum: Vec<u8>,
    pub size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub enum Metadata {
    V1(MetadataV1),
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct DefaultDownloadMetadataV1 {
    pub url: String,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub enum DefaultDownloadMetadata {
    V1(DefaultDownloadMetadataV1),
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct MtlsDownloadMetadataV1 {
    pub peer: Vec<u8>, // NetworkPublicKey bytes
    pub url: String,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub enum MtlsDownloadMetadata {
    V1(MtlsDownloadMetadataV1),
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub enum DownloadMetadata {
    Default(DefaultDownloadMetadata),
    Mtls(MtlsDownloadMetadata),
}

/// Transaction type
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
#[non_exhaustive]
pub enum TransactionKind {
    /// Transaction used to initialize the chain state.
    ///
    /// Only valid if in the genesis checkpoint (0) and if this is the very first transaction ever
    /// executed on the chain.
    Genesis(GenesisTransaction),

    /// Consensus commit update
    ConsensusCommitPrologue(ConsensusCommitPrologue),

    /// System transaction used to end an epoch.
    ///
    /// The ChangeEpoch variant is now deprecated (but the ChangeEpoch struct is still used by
    /// EndOfEpochTransaction below).
    ChangeEpoch(ChangeEpoch),

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
        amount: Option<u64>,
        recipient: Address,
    },
    PayCoins {
        coins: Vec<ObjectReference>,
        amounts: Option<Vec<u64>>,
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
        amount: Option<u64>,
    },
    AddStakeToEncoder {
        encoder_address: Address,
        coin_ref: ObjectReference,
        amount: Option<u64>,
    },
    WithdrawStake {
        staked_soma: ObjectReference,
    },

    // Shard operations
    EmbedData {
        download_metadata: DownloadMetadata,
        coin_ref: ObjectReference,
    },
    ClaimEscrow {
        shard_input_ref: ObjectReference,
    },
    ReportWinner {
        shard_input_ref: ObjectReference,
        signed_report: Vec<u8>,
        encoder_aggregate_signature: Vec<u8>,
        signers: Vec<String>,
        shard_auth_token: Vec<u8>,
    },
}

// Supporting types for validator management

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct AddValidatorArgs {
    pub pubkey_bytes: Vec<u8>,
    pub network_pubkey_bytes: Vec<u8>,
    pub worker_pubkey_bytes: Vec<u8>,
    pub net_address: Vec<u8>,
    pub p2p_address: Vec<u8>,
    pub primary_address: Vec<u8>,
    pub encoder_validator_address: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct RemoveValidatorArgs {
    pub pubkey_bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct UpdateValidatorMetadataArgs {
    pub next_epoch_network_address: Option<Vec<u8>>,
    pub next_epoch_p2p_address: Option<Vec<u8>>,
    pub next_epoch_primary_address: Option<Vec<u8>>,
    pub next_epoch_protocol_pubkey: Option<Vec<u8>>,
    pub next_epoch_worker_pubkey: Option<Vec<u8>>,
    pub next_epoch_network_pubkey: Option<Vec<u8>>,
}

// Supporting types for encoder management

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct AddEncoderArgs {
    pub encoder_pubkey_bytes: Vec<u8>,
    pub network_pubkey_bytes: Vec<u8>,
    pub internal_network_address: Vec<u8>,
    pub external_network_address: Vec<u8>,
    pub object_server_address: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct RemoveEncoderArgs {
    pub encoder_pubkey_bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ConsensusCommitPrologue {
    /// Epoch of the commit prologue transaction
    pub epoch: u64,

    /// Consensus round of the commit
    pub round: u64,

    /// The sub DAG index of the consensus commit
    /// This field will be populated if there are multiple consensus commits per round
    pub sub_dag_index: Option<u64>,

    /// Unix timestamp from consensus
    pub commit_timestamp_ms: CommitTimestamp,

    /// Digest of consensus output for verification
    pub consensus_commit_digest: Digest,

    /// Digest of any additional state computed by the consensus handler.
    /// Used to detect forking bugs as early as possible.
    pub additional_state_digest: Digest,
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
#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct GenesisTransaction {
    pub objects: Vec<Object>,
}

/// serde implementation that serializes a transaction prefixed with the signing intent. See
/// [struct Intent] for more info.
///
/// So we need to serialize Transaction as (0, 0, 0, Transaction)
struct IntentMessageWrappedTransaction;

impl SerializeAs<Transaction> for IntentMessageWrappedTransaction {
    fn serialize_as<S>(transaction: &Transaction, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeTuple;

        let mut s = serializer.serialize_tuple(4)?;
        s.serialize_element(&0u8)?;
        s.serialize_element(&0u8)?;
        s.serialize_element(&0u8)?;
        s.serialize_element(transaction)?;
        s.end()
    }
}

impl<'de> DeserializeAs<'de, Transaction> for IntentMessageWrappedTransaction {
    fn deserialize_as<D>(deserializer: D) -> Result<Transaction, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (scope, version, app, transaction): (u8, u8, u8, Transaction) =
            Deserialize::deserialize(deserializer)?;
        match (scope, version, app) {
            (0, 0, 0) => {}
            _ => {
                return Err(serde::de::Error::custom(format!(
                    "invalid intent message ({scope}, {version}, {app})"
                )));
            }
        }

        Ok(transaction)
    }
}

pub(crate) struct SignedTransactionWithIntentMessage;

#[derive(serde_derive::Serialize)]
struct BinarySignedTransactionWithIntentMessageRef<'a> {
    #[serde(with = "::serde_with::As::<IntentMessageWrappedTransaction>")]
    transaction: &'a Transaction,
    signatures: &'a Vec<UserSignature>,
}

#[derive(serde_derive::Deserialize)]
struct BinarySignedTransactionWithIntentMessage {
    #[serde(with = "::serde_with::As::<IntentMessageWrappedTransaction>")]
    transaction: Transaction,
    signatures: Vec<UserSignature>,
}

impl SerializeAs<SignedTransaction> for SignedTransactionWithIntentMessage {
    fn serialize_as<S>(transaction: &SignedTransaction, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            transaction.serialize(serializer)
        } else {
            let SignedTransaction {
                transaction,
                signatures,
            } = transaction;
            let binary = BinarySignedTransactionWithIntentMessageRef {
                transaction,
                signatures,
            };

            let mut s = serializer.serialize_seq(Some(1))?;
            s.serialize_element(&binary)?;
            s.end()
        }
    }
}

impl<'de> DeserializeAs<'de, SignedTransaction> for SignedTransactionWithIntentMessage {
    fn deserialize_as<D>(deserializer: D) -> Result<SignedTransaction, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            SignedTransaction::deserialize(deserializer)
        } else {
            struct V;
            impl<'de> serde::de::Visitor<'de> for V {
                type Value = SignedTransaction;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    formatter.write_str("expected a sequence with length 1")
                }

                fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                where
                    A: serde::de::SeqAccess<'de>,
                {
                    if seq.size_hint().is_some_and(|size| size != 1) {
                        return Err(serde::de::Error::custom(
                            "expected a sequence with length 1",
                        ));
                    }

                    let BinarySignedTransactionWithIntentMessage {
                        transaction,
                        signatures,
                    } = seq.next_element()?.ok_or_else(|| {
                        serde::de::Error::custom("expected a sequence with length 1")
                    })?;
                    Ok(SignedTransaction {
                        transaction,
                        signatures,
                    })
                }
            }

            deserializer.deserialize_seq(V)
        }
    }
}

#[derive(serde_derive::Serialize)]
#[serde(rename = "Transaction")]
struct TransactionDataRef<'a> {
    kind: &'a TransactionKind,
    sender: &'a Address,
    gas_payment: &'a Vec<ObjectReference>,
}

#[derive(serde_derive::Deserialize)]
#[serde(rename = "Transaction")]
struct TransactionData {
    kind: TransactionKind,
    sender: Address,
    gas_payment: Vec<ObjectReference>,
}

impl Serialize for Transaction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let transaction = TransactionDataRef {
            kind: &self.kind,
            sender: &self.sender,
            gas_payment: &self.gas_payment,
        };

        transaction.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Transaction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let TransactionData {
            kind,
            sender,
            gas_payment,
        } = Deserialize::deserialize(deserializer)?;

        Ok(Transaction {
            kind,
            sender,
            gas_payment,
        })
    }
}
