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
pub use checkpoint::CheckpointCommitment;
pub use checkpoint::CheckpointContents;
pub use checkpoint::CheckpointData;
pub use checkpoint::CheckpointSequenceNumber;
pub use checkpoint::CheckpointSummary;
pub use checkpoint::CheckpointTimestamp;
pub use checkpoint::CheckpointTransaction;
pub use checkpoint::CheckpointTransactionInfo;
pub use checkpoint::EndOfEpochData;
pub use checkpoint::EpochId;
pub use checkpoint::ProtocolVersion;
pub use checkpoint::SignedCheckpointSummary;
pub use checkpoint::StakeUnit;
pub use crypto::Bls12381PublicKey;
pub use crypto::Bls12381Signature;
pub use crypto::Ed25519PublicKey;
pub use crypto::Ed25519Signature;
pub use crypto::Intent;
pub use crypto::IntentAppId;
pub use crypto::IntentScope;
pub use crypto::IntentVersion;
pub use crypto::MultisigAggregatedSignature;
pub use crypto::MultisigCommittee;
pub use crypto::MultisigMember;
pub use crypto::MultisigMemberPublicKey;
pub use crypto::MultisigMemberSignature;
pub use crypto::SignatureScheme;
pub use crypto::SimpleSignature;
pub use crypto::UserSignature;
pub use crypto::ValidatorAggregatedSignature;
pub use crypto::ValidatorCommittee;
pub use crypto::ValidatorCommitteeMember;
pub use crypto::ValidatorNetworkMetadata;
pub use crypto::ValidatorSignature;
pub use digest::Digest;
pub use digest::SigningDigest;
pub use effects::ChangedObject;
pub use effects::IdOperation;
pub use effects::ObjectIn;
pub use effects::ObjectOut;
pub use effects::TransactionEffects;
pub use effects::UnchangedSharedKind;
pub use effects::UnchangedSharedObject;
pub use execution_status::ExecutionError;
pub use execution_status::ExecutionStatus;
pub use fee::TransactionFee;
pub use object::Object;
pub use object::ObjectReference;
pub use object::ObjectType;
pub use object::Owner;
pub use object::Version;
pub use transaction::AddValidatorArgs;
pub use transaction::ChangeEpoch;
pub use transaction::ConsensusCommitPrologue;
pub use transaction::GenesisTransaction;
pub use transaction::RemoveValidatorArgs;
pub use transaction::SignedTransaction;
pub(crate) use transaction::SignedTransactionWithIntentMessage;
pub use transaction::Transaction;
pub use transaction::TransactionKind;
pub use transaction::UpdateValidatorMetadataArgs;
pub use transaction::{
    ClaimRewardsArgs, CommitModelArgs, CommitModelUpdateArgs, InitiateChallengeArgs, Manifest,
    ManifestV1, Metadata, MetadataV1, ModelWeightsManifest, RevealModelArgs, RevealModelUpdateArgs,
    SubmissionManifest, SubmitDataArgs,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersonalMessage<'a>(pub std::borrow::Cow<'a, [u8]>);

mod _serde {
    use base64ct::Base64;
    use base64ct::Encoding;
    use serde::Deserialize;
    use serde::Deserializer;
    use serde::Serialize;
    use serde::Serializer;
    use serde_with::DeserializeAs;
    use serde_with::SerializeAs;
    use std::borrow::Cow;

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
