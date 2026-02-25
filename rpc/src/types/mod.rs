// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

mod address;
mod balance_change;
mod bitmap;
mod checkpoint;
mod crypto;
mod digest;
mod effects;
mod execution_status;
mod fee;
pub mod hash;
mod object;
mod transaction;

pub use address::Address;
pub use balance_change::BalanceChange;
pub use bitmap::Bitmap;
pub use checkpoint::{
    CheckpointCommitment, CheckpointContents, CheckpointData, CheckpointSequenceNumber,
    CheckpointSummary, CheckpointTimestamp, CheckpointTransaction, CheckpointTransactionInfo,
    EndOfEpochData, EpochId, ProtocolVersion, SignedCheckpointSummary, StakeUnit,
};
pub use crypto::{
    Bls12381PublicKey, Bls12381Signature, Ed25519PublicKey, Ed25519Signature, Intent, IntentAppId,
    IntentScope, IntentVersion, MultisigAggregatedSignature, MultisigCommittee, MultisigMember,
    MultisigMemberPublicKey, MultisigMemberSignature, SignatureScheme, SimpleSignature,
    UserSignature, ValidatorAggregatedSignature, ValidatorCommittee, ValidatorCommitteeMember,
    ValidatorNetworkMetadata, ValidatorSignature,
};
pub use digest::{Digest, SigningDigest};
pub use effects::{
    ChangedObject, IdOperation, ObjectIn, ObjectOut, TransactionEffects, UnchangedSharedKind,
    UnchangedSharedObject,
};
pub use execution_status::{ExecutionError, ExecutionStatus};
pub use fee::TransactionFee;
pub use object::{Object, ObjectReference, ObjectType, Owner, Version};
pub(crate) use transaction::SignedTransactionWithIntentMessage;
pub use transaction::{
    AddValidatorArgs, ChangeEpoch, ClaimRewardsArgs, CommitModelArgs, CommitModelUpdateArgs,
    ConsensusCommitPrologue, GenesisTransaction, InitiateChallengeArgs, Manifest, ManifestV1,
    Metadata, MetadataV1, ModelWeightsManifest, RemoveValidatorArgs, RevealModelArgs,
    RevealModelUpdateArgs, SignedTransaction, SubmissionManifest, SubmitDataArgs, Transaction,
    TransactionKind, UpdateValidatorMetadataArgs,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersonalMessage<'a>(pub std::borrow::Cow<'a, [u8]>);

mod _serde {
    use std::borrow::Cow;

    use base64ct::{Base64, Encoding};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde_with::{DeserializeAs, SerializeAs};

    pub(crate) type ReadableBase64Encoded =
        ::serde_with::As<::serde_with::IfIsHumanReadable<Base64Encoded, ::serde_with::Bytes>>;

    pub(crate) struct Base64Encoded;

    impl<T: AsRef<[u8]>> SerializeAs<T> for Base64Encoded {
        fn serialize_as<S>(source: &T, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let bytes = source.as_ref();
            let b64 = Base64::encode_string(bytes);
            b64.serialize(serializer)
        }
    }

    impl<'de, T: TryFrom<Vec<u8>>> DeserializeAs<'de, T> for Base64Encoded {
        fn deserialize_as<D>(deserializer: D) -> Result<T, D::Error>
        where
            D: Deserializer<'de>,
        {
            let b64: Cow<'de, str> = Deserialize::deserialize(deserializer)?;
            let bytes = Base64::decode_vec(&b64).map_err(serde::de::Error::custom)?;
            let length = bytes.len();
            T::try_from(bytes).map_err(|_| {
                serde::de::Error::custom(format_args!(
                    "Can't convert a Byte Vector of length {length} to the output type."
                ))
            })
        }
    }

    pub(crate) use super::SignedTransactionWithIntentMessage;
}
