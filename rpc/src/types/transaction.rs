use serde::{Deserialize, Serialize};

use crate::types::{Address, CheckpointTimestamp, EpochId, Object, ObjectReference, UserSignature};
use crate::types::{Digest, ProtocolVersion};
use serde::Deserializer;
use serde::Serializer;
use serde::ser::SerializeSeq;
use serde_with::DeserializeAs;
use serde_with::SerializeAs;
use url::Url;

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
pub struct ManifestV1 {
    pub url: Url,
    pub metadata: Metadata,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub enum Manifest {
    V1(ManifestV1),
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

    WithdrawStake {
        staked_soma: ObjectReference,
    },

    // Model transactions
    CommitModel(CommitModelArgs),
    RevealModel(RevealModelArgs),
    CommitModelUpdate(CommitModelUpdateArgs),
    RevealModelUpdate(RevealModelUpdateArgs),
    AddStakeToModel {
        model_id: Address,
        coin_ref: ObjectReference,
        amount: Option<u64>,
    },
    SetModelCommissionRate {
        model_id: Address,
        new_rate: u64,
    },
    DeactivateModel {
        model_id: Address,
    },
    ReportModel {
        model_id: Address,
    },
    UndoReportModel {
        model_id: Address,
    },

    // Submission transactions
    SubmitData(SubmitDataArgs),
    ClaimRewards(ClaimRewardsArgs),
    /// Report a submission as fraudulent (validators only)
    ReportSubmission {
        target_id: Address,
        /// Optional challenger to attribute fraud to (for tally-based bond distribution)
        challenger: Option<Address>,
    },
    /// Undo a previous submission report
    UndoReportSubmission {
        target_id: Address,
    },

    // Challenge transactions
    InitiateChallenge(InitiateChallengeArgs),
    /// Report a challenge (validators only).
    /// Reports indicate "the challenger is wrong" (i.e., the submission is valid).
    /// If 2f+1 validators report, the challenger loses their bond.
    ReportChallenge {
        challenge_id: Address,
    },
    /// Undo a previous challenge report
    UndoReportChallenge {
        challenge_id: Address,
    },
    /// Claim the challenger's bond after challenge window closes
    ClaimChallengeBond {
        challenge_id: Address,
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
    pub proxy_address: Vec<u8>,
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
    pub next_epoch_proxy_address: Option<Vec<u8>>,
    pub next_epoch_protocol_pubkey: Option<Vec<u8>>,
    pub next_epoch_worker_pubkey: Option<Vec<u8>>,
    pub next_epoch_network_pubkey: Option<Vec<u8>>,
}

// Supporting types for model transactions

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ModelWeightsManifest {
    pub manifest: Manifest,
    pub decryption_key: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct CommitModelArgs {
    pub model_id: Address,
    pub weights_url_commitment: Vec<u8>,
    pub weights_commitment: Vec<u8>,
    pub architecture_version: u64,
    pub stake_amount: u64,
    pub commission_rate: u64,
    pub staking_pool_id: Address,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct RevealModelArgs {
    pub model_id: Address,
    pub weights_manifest: ModelWeightsManifest,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct CommitModelUpdateArgs {
    pub model_id: Address,
    pub weights_url_commitment: Vec<u8>,
    pub weights_commitment: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct RevealModelUpdateArgs {
    pub model_id: Address,
    pub weights_manifest: ModelWeightsManifest,
}

// Supporting types for submission transactions

/// Manifest for submitted data (URL + metadata, no encryption key)
#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct SubmissionManifest {
    pub manifest: Manifest,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct SubmitDataArgs {
    pub target_id: Address,
    pub data_commitment: Vec<u8>,
    pub data_manifest: SubmissionManifest,
    pub model_id: Address,
    pub embedding: Vec<i64>,
    pub distance_score: i64,
    pub bond_coin: ObjectReference,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ClaimRewardsArgs {
    pub target_id: Address,
}

// Supporting types for challenge transactions

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct InitiateChallengeArgs {
    /// Target being challenged (must be filled and within challenge window)
    pub target_id: Address,
    /// Coin to pay challenger bond (must cover challenger_bond_per_byte * data_size)
    pub bond_coin: ObjectReference,
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
    pub commit_timestamp_ms: CheckpointTimestamp,

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

    /// The protocol version in effect in the new epoch.
    pub protocol_version: ProtocolVersion,

    /// Unix timestamp when epoch started
    pub epoch_start_timestamp_ms: u64,

    /// The total amount of fees charged during the epoch.
    pub fees: u64,

    /// Epoch randomness
    pub epoch_randomness: Vec<u8>,
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
