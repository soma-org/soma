// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use crate::types::TransactionFee;
use crate::types::ValidatorNetworkMetadata;

use super::Digest;
use super::Object;
use super::SignedTransaction;
use super::TransactionEffects;
use super::UserSignature;
use super::ValidatorAggregatedSignature;
use super::ValidatorCommittee;

pub type CheckpointSequenceNumber = u64;
pub type CheckpointTimestamp = u64;
pub type EpochId = u64;
pub type StakeUnit = u64;
pub type ProtocolVersion = u64;

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub enum CheckpointCommitment {
    /// An Elliptic Curve Multiset Hash attesting to the set of Objects that comprise the live
    /// state of the Sui blockchain.
    EcmhLiveObjectSet { digest: Digest },

    /// Digest of the checkpoint artifacts.
    CheckpointArtifacts { digest: Digest },
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct EndOfEpochData {
    /// The set of Validators that will be in the ValidatorCommittee for the next epoch.
    pub next_epoch_validator_committee: ValidatorCommittee,

    /// The protocol version that is in effect during the next epoch.
    pub next_epoch_protocol_version: ProtocolVersion,

    /// Commitments to epoch specific state (e.g. live object set)
    pub epoch_commitments: Vec<CheckpointCommitment>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct CheckpointSummary {
    /// Epoch that this checkpoint belongs to.
    pub epoch: EpochId,

    /// The height of this checkpoint.
    pub sequence_number: CheckpointSequenceNumber,

    /// Total number of transactions committed since genesis, including those in this
    /// checkpoint.
    pub network_total_transactions: u64,

    /// The hash of the [`CheckpointContents`] for this checkpoint.
    pub content_digest: Digest,

    /// The hash of the previous `CheckpointSummary`.
    ///
    /// This will be only be `None` for the first, or genesis checkpoint.
    pub previous_digest: Option<Digest>,

    /// The running total fees of all transactions included in the current epoch so far
    /// until this checkpoint.
    pub epoch_rolling_transaction_fees: TransactionFee,

    /// Timestamp of the checkpoint - number of milliseconds from the Unix epoch
    /// Checkpoint timestamps are monotonic, but not strongly monotonic - subsequent
    /// checkpoints can have same timestamp if they originate from the same underlining consensus commit
    pub timestamp_ms: CheckpointTimestamp,

    /// Commitments to checkpoint-specific state.
    pub checkpoint_commitments: Vec<CheckpointCommitment>,

    /// Extra data only present in the final checkpoint of an epoch.
    pub end_of_epoch_data: Option<EndOfEpochData>,

    /// Opaque version-specific data for forward compatibility.
    #[serde(default)]
    pub version_specific_data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct SignedCheckpointSummary {
    pub checkpoint: CheckpointSummary,
    pub signature: ValidatorAggregatedSignature,
}

/// The committed to contents of a checkpoint.
///
/// `CheckpointContents` contains a list of digests of Transactions, their effects, and the user
/// signatures that authorized their execution included in a checkpoint.
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// checkpoint-contents = %x00 checkpoint-contents-v1 ; variant 0
///
/// checkpoint-contents-v1 = (vector (digest digest)) ; vector of transaction and effect digests
///                          (vector (vector bcs-user-signature)) ; set of user signatures for each
///                                                               ; transaction. MUST be the same
///                                                               ; length as the vector of digests
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointContents(Vec<CheckpointTransactionInfo>);

impl CheckpointContents {
    pub fn new(transactions: Vec<CheckpointTransactionInfo>) -> Self {
        Self(transactions)
    }

    pub fn transactions(&self) -> &[CheckpointTransactionInfo] {
        &self.0
    }

    pub fn into_v1(self) -> Vec<CheckpointTransactionInfo> {
        self.0
    }
}

/// Transaction information committed to in a checkpoint
#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct CheckpointTransactionInfo {
    pub transaction: Digest,
    pub effects: Digest,
    pub signatures: Vec<UserSignature>,
}

#[derive(Clone, Debug, PartialEq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct CheckpointData {
    pub checkpoint_summary: SignedCheckpointSummary,
    pub checkpoint_contents: CheckpointContents,
    pub transactions: Vec<CheckpointTransaction>,
}

#[derive(Clone, Debug, PartialEq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct CheckpointTransaction {
    /// The input Transaction
    #[serde(with = "::serde_with::As::<crate::types::_serde::SignedTransactionWithIntentMessage>")]
    pub transaction: SignedTransaction,
    /// The effects produced by executing this transaction
    pub effects: TransactionEffects,
    /// The state of all inputs to this transaction as they were prior to execution.
    pub input_objects: Vec<Object>,
    /// The state of all output objects created or mutated by this transaction.
    pub output_objects: Vec<Object>,
}

mod serialization {
    use super::*;

    use serde::Deserialize;
    use serde::Deserializer;
    use serde::Serialize;
    use serde::Serializer;

    impl Serialize for CheckpointContents {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            use serde::ser::SerializeSeq;
            use serde::ser::SerializeTupleVariant;

            #[derive(serde_derive::Serialize)]
            struct Digests<'a> {
                transaction: &'a Digest,
                effects: &'a Digest,
            }

            struct DigestSeq<'a>(&'a CheckpointContents);
            impl Serialize for DigestSeq<'_> {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: Serializer,
                {
                    let mut seq = serializer.serialize_seq(Some(self.0.0.len()))?;
                    for txn in &self.0.0 {
                        let digests =
                            Digests { transaction: &txn.transaction, effects: &txn.effects };
                        seq.serialize_element(&digests)?;
                    }
                    seq.end()
                }
            }

            struct SignatureSeq<'a>(&'a CheckpointContents);
            impl Serialize for SignatureSeq<'_> {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: Serializer,
                {
                    let mut seq = serializer.serialize_seq(Some(self.0.0.len()))?;
                    for txn in &self.0.0 {
                        seq.serialize_element(&txn.signatures)?;
                    }
                    seq.end()
                }
            }

            let mut s = serializer.serialize_tuple_variant("CheckpointContents", 0, "V1", 2)?;
            s.serialize_field(&DigestSeq(self))?;
            s.serialize_field(&SignatureSeq(self))?;
            s.end()
        }
    }

    #[derive(serde_derive::Deserialize)]
    struct ExecutionDigests {
        transaction: Digest,
        effects: Digest,
    }

    #[derive(serde_derive::Deserialize)]
    struct BinaryContentsV1 {
        digests: Vec<ExecutionDigests>,
        signatures: Vec<Vec<UserSignature>>,
    }

    #[derive(serde_derive::Deserialize)]
    enum BinaryContents {
        V1(BinaryContentsV1),
    }

    impl<'de> Deserialize<'de> for CheckpointContents {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let BinaryContents::V1(BinaryContentsV1 { digests, signatures }) =
                Deserialize::deserialize(deserializer)?;

            if digests.len() != signatures.len() {
                return Err(serde::de::Error::custom(
                    "must have same number of signatures as transactions",
                ));
            }

            Ok(Self(
                digests
                    .into_iter()
                    .zip(signatures)
                    .map(|(ExecutionDigests { transaction, effects }, signatures)| {
                        CheckpointTransactionInfo { transaction, effects, signatures }
                    })
                    .collect(),
            ))
        }
    }
}
