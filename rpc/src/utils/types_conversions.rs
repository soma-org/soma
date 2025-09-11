use std::collections::BTreeMap;

use crate::types::*;
use fastcrypto::traits::ToFromBytes;
use tap::Pipe;
use types::crypto::SomaSignature;

#[derive(Debug)]
pub struct SdkTypeConversionError(String);

impl std::fmt::Display for SdkTypeConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SdkTypeConversionError {}

impl From<anyhow::Error> for SdkTypeConversionError {
    fn from(value: anyhow::Error) -> Self {
        Self(value.to_string())
    }
}

impl From<bcs::Error> for SdkTypeConversionError {
    fn from(value: bcs::Error) -> Self {
        Self(value.to_string())
    }
}

impl From<std::array::TryFromSliceError> for SdkTypeConversionError {
    fn from(value: std::array::TryFromSliceError) -> Self {
        Self(value.to_string())
    }
}

macro_rules! bcs_convert_impl {
    ($core:ty, $external:ty) => {
        impl TryFrom<$core> for $external {
            type Error = bcs::Error;

            fn try_from(value: $core) -> Result<Self, Self::Error> {
                let bytes = bcs::to_bytes(&value)?;
                bcs::from_bytes(&bytes)
            }
        }

        impl TryFrom<$external> for $core {
            type Error = bcs::Error;

            fn try_from(value: $external) -> Result<Self, Self::Error> {
                let bytes = bcs::to_bytes(&value)?;
                bcs::from_bytes(&bytes)
            }
        }
    };
}

bcs_convert_impl!(types::object::Object, Object);
bcs_convert_impl!(types::transaction::TransactionData, Transaction);
bcs_convert_impl!(types::effects::TransactionEffects, TransactionEffects);
bcs_convert_impl!(types::crypto::GenericSignature, UserSignature);
bcs_convert_impl!(types::transaction::TransactionKind, TransactionKind);

impl<const T: bool> From<types::crypto::AuthorityQuorumSignInfo<T>>
    for ValidatorAggregatedSignature
{
    fn from(value: types::crypto::AuthorityQuorumSignInfo<T>) -> Self {
        let types::crypto::AuthorityQuorumSignInfo {
            epoch,
            signature,
            signers_map,
        } = value;

        Self {
            epoch,
            signature: Bls12381Signature::from_bytes(signature.as_ref()).unwrap(),
            bitmap: Bitmap::from_iter(signers_map),
        }
    }
}

impl<const T: bool> From<ValidatorAggregatedSignature>
    for types::crypto::AuthorityQuorumSignInfo<T>
{
    fn from(value: ValidatorAggregatedSignature) -> Self {
        let ValidatorAggregatedSignature {
            epoch,
            signature,
            bitmap,
        } = value;

        Self {
            epoch,
            signature: types::crypto::AggregateAuthoritySignature::from_bytes(signature.as_bytes())
                .unwrap(),
            signers_map: roaring::RoaringBitmap::from_iter(bitmap.iter()),
        }
    }
}

impl From<types::object::Owner> for Owner {
    fn from(value: types::object::Owner) -> Self {
        match value {
            types::object::Owner::AddressOwner(address) => Self::Address(address.into()),
            types::object::Owner::Shared {
                initial_shared_version,
            } => Self::Shared(initial_shared_version.value()),
            types::object::Owner::Immutable => Self::Immutable,
        }
    }
}

impl From<Owner> for types::object::Owner {
    fn from(value: Owner) -> Self {
        match value {
            Owner::Address(address) => types::object::Owner::AddressOwner(address.into()),
            Owner::Shared(initial_shared_version) => types::object::Owner::Shared {
                initial_shared_version: types::object::Version::from_u64(initial_shared_version),
            },
            Owner::Immutable => types::object::Owner::Immutable,
        }
    }
}

impl From<types::base::SomaAddress> for Address {
    fn from(value: types::base::SomaAddress) -> Self {
        Self::new(value.to_inner())
    }
}

impl From<Address> for types::base::SomaAddress {
    fn from(value: Address) -> Self {
        types::object::ObjectID::new(value.into_inner()).into()
    }
}

impl From<types::object::ObjectID> for Address {
    fn from(value: types::object::ObjectID) -> Self {
        Self::new(value.into_bytes())
    }
}

impl From<Address> for types::object::ObjectID {
    fn from(value: Address) -> Self {
        Self::new(value.into_inner())
    }
}

impl TryFrom<types::transaction::SenderSignedData> for SignedTransaction {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::transaction::SenderSignedData) -> Result<Self, Self::Error> {
        let types::transaction::SenderSignedTransaction {
            intent_message,
            tx_signatures,
        } = value.into_inner();

        Self {
            transaction: intent_message.value.try_into()?,
            signatures: tx_signatures
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        }
        .pipe(Ok)
    }
}

impl TryFrom<SignedTransaction> for types::transaction::SenderSignedData {
    type Error = SdkTypeConversionError;

    fn try_from(value: SignedTransaction) -> Result<Self, Self::Error> {
        let SignedTransaction {
            transaction,
            signatures,
        } = value;

        Self::new(
            transaction.try_into()?,
            signatures
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        )
        .pipe(Ok)
    }
}

impl TryFrom<types::transaction::Transaction> for SignedTransaction {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::transaction::Transaction) -> Result<Self, Self::Error> {
        value.into_data().try_into()
    }
}

impl TryFrom<SignedTransaction> for types::transaction::Transaction {
    type Error = SdkTypeConversionError;

    fn try_from(value: SignedTransaction) -> Result<Self, Self::Error> {
        Ok(Self::new(value.try_into()?))
    }
}

impl From<types::digests::ObjectDigest> for Digest {
    fn from(value: types::digests::ObjectDigest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<Digest> for types::digests::ObjectDigest {
    fn from(value: Digest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<types::digests::TransactionDigest> for Digest {
    fn from(value: types::digests::TransactionDigest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<Digest> for types::digests::TransactionDigest {
    fn from(value: Digest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<types::digests::Digest> for Digest {
    fn from(value: types::digests::Digest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<Digest> for types::digests::Digest {
    fn from(value: Digest) -> Self {
        Self::new(value.into_inner())
    }
}
impl From<types::committee::Committee> for ValidatorCommittee {
    fn from(value: types::committee::Committee) -> Self {
        let members = value
            .voting_rights
            .into_iter()
            .map(|(name, weight)| {
                let authority = value
                    .authorities
                    .get(&name)
                    .expect("Authority must exist for each voting right");

                ValidatorCommitteeMember {
                    // AuthorityName is just bytes, get the underlying bytes
                    authority_key: authority.authority_key.as_bytes().to_vec(),
                    stake: weight,
                    network_metadata: ValidatorNetworkMetadata {
                        consensus_address: authority.address.to_string(),
                        hostname: authority.hostname.clone(),
                        protocol_key: authority.protocol_key.to_bytes().to_vec(),
                        network_key: authority.network_key.to_bytes().to_vec(),
                    },
                }
            })
            .collect();

        ValidatorCommittee {
            epoch: value.epoch,
            members,
        }
    }
}

// Define the conversion error type
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("Invalid authority key: {0}")]
    InvalidAuthorityKey(String),

    #[error("Invalid protocol key: {0}")]
    InvalidProtocolKey(String),

    #[error("Invalid network key: {0}")]
    InvalidNetworkKey(String),

    #[error("Invalid multiaddr: {0}")]
    InvalidMultiaddr(String),
}

// From domain ValidatorCommittee to types::committee::Committee - with error handling
impl TryFrom<ValidatorCommittee> for types::committee::Committee {
    type Error = ConversionError;

    fn try_from(value: ValidatorCommittee) -> Result<Self, Self::Error> {
        let ValidatorCommittee { epoch, members } = value;

        let mut voting_rights = BTreeMap::new();
        let mut authorities = BTreeMap::new();

        for member in members {
            // Convert the authority key bytes to AuthorityPublicKey first
            let authority_public_key =
                fastcrypto::bls12381::min_sig::BLS12381PublicKey::from_bytes(&member.authority_key)
                    .map_err(|e| ConversionError::InvalidAuthorityKey(e.to_string()))?;

            // Create AuthorityName from the public key
            let authority_name = types::base::AuthorityName::from(&authority_public_key);

            voting_rights.insert(authority_name, member.stake);

            // Parse the multiaddr
            let address = member
                .network_metadata
                .consensus_address
                .parse()
                .map_err(|e| ConversionError::InvalidMultiaddr(format!("{:?}", e)))?;

            // Parse the protocol key
            let protocol_key = fastcrypto::ed25519::Ed25519PublicKey::from_bytes(
                &member.network_metadata.protocol_key,
            )
            .map_err(|e| ConversionError::InvalidProtocolKey(e.to_string()))?;

            // Parse the network key
            let network_key = fastcrypto::ed25519::Ed25519PublicKey::from_bytes(
                &member.network_metadata.network_key,
            )
            .map_err(|e| ConversionError::InvalidNetworkKey(e.to_string()))?;

            let authority = types::committee::Authority {
                stake: member.stake,
                address,
                hostname: member.network_metadata.hostname,
                protocol_key: types::crypto::ProtocolPublicKey::new(protocol_key),
                network_key: types::crypto::NetworkPublicKey::new(network_key),
                authority_key: authority_public_key,
            };

            authorities.insert(authority_name, authority);
        }

        Ok(Self::new(epoch, voting_rights, authorities))
    }
}

impl From<types::crypto::AuthorityPublicKeyBytes> for Bls12381PublicKey {
    fn from(value: types::crypto::AuthorityPublicKeyBytes) -> Self {
        Self::new(value.0)
    }
}

impl From<Bls12381PublicKey> for types::crypto::AuthorityPublicKeyBytes {
    fn from(value: Bls12381PublicKey) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<UnchangedSharedKind> for types::effects::UnchangedSharedKind {
    fn from(value: UnchangedSharedKind) -> Self {
        match value {
            UnchangedSharedKind::ReadOnlyRoot { version, digest } => {
                Self::ReadOnlyRoot((types::object::Version::from_u64(version), digest.into()))
            }
            UnchangedSharedKind::MutateDeleted { version } => {
                Self::MutateDeleted(types::object::Version::from_u64(version))
            }
            UnchangedSharedKind::ReadDeleted { version } => {
                Self::ReadDeleted(types::object::Version::from_u64(version))
            }
            UnchangedSharedKind::Canceled { version } => {
                Self::Cancelled(types::object::Version::from_u64(version))
            }

            _ => unreachable!("sdk shouldn't have a variant that the mono repo doesn't"),
        }
    }
}

impl From<types::effects::UnchangedSharedKind> for UnchangedSharedKind {
    fn from(value: types::effects::UnchangedSharedKind) -> Self {
        match value {
            types::effects::UnchangedSharedKind::ReadOnlyRoot((version, digest)) => {
                Self::ReadOnlyRoot {
                    version: version.value(),
                    digest: digest.into(),
                }
            }
            types::effects::UnchangedSharedKind::MutateDeleted(version) => Self::MutateDeleted {
                version: version.value(),
            },
            types::effects::UnchangedSharedKind::ReadDeleted(version) => Self::ReadDeleted {
                version: version.value(),
            },
            types::effects::UnchangedSharedKind::Cancelled(version) => Self::Canceled {
                version: version.value(),
            },
        }
    }
}

impl From<types::effects::object_change::ObjectIn> for ObjectIn {
    fn from(value: types::effects::object_change::ObjectIn) -> Self {
        match value {
            types::effects::object_change::ObjectIn::NotExist => Self::NotExist,
            types::effects::object_change::ObjectIn::Exist(((version, digest), owner)) => {
                Self::Exist {
                    version: version.value(),
                    digest: digest.into(),
                    owner: owner.into(),
                }
            }
        }
    }
}

impl From<types::effects::object_change::ObjectOut> for ObjectOut {
    fn from(value: types::effects::object_change::ObjectOut) -> Self {
        match value {
            types::effects::object_change::ObjectOut::NotExist => Self::NotExist,
            types::effects::object_change::ObjectOut::ObjectWrite((digest, owner)) => {
                Self::ObjectWrite {
                    digest: digest.into(),
                    owner: owner.into(),
                }
            }
        }
    }
}

impl From<types::effects::object_change::IDOperation> for IdOperation {
    fn from(value: types::effects::object_change::IDOperation) -> Self {
        match value {
            types::effects::object_change::IDOperation::None => Self::None,
            types::effects::object_change::IDOperation::Created => Self::Created,
            types::effects::object_change::IDOperation::Deleted => Self::Deleted,
        }
    }
}

impl From<types::effects::ExecutionFailureStatus> for ExecutionError {
    fn from(value: types::effects::ExecutionFailureStatus) -> Self {
        match value {
            types::effects::ExecutionFailureStatus::InsufficientGas => Self::InsufficientGas,
            types::effects::ExecutionFailureStatus::InvalidOwnership {
                object_id,
                expected_owner,
                actual_owner,
            } => Self::InvalidOwnership {
                object_id: object_id.into(),
            },
            types::effects::ExecutionFailureStatus::ObjectNotFound { object_id } => {
                Self::ObjectNotFound {
                    object_id: object_id.into(),
                }
            }
            types::effects::ExecutionFailureStatus::InvalidObjectType {
                object_id,
                expected_type,
                actual_type,
            } => Self::InvalidObjectType {
                object_id: object_id.into(),
            },
            types::effects::ExecutionFailureStatus::InvalidTransactionType => {
                Self::InvalidTransactionType
            }
            types::effects::ExecutionFailureStatus::InvalidArguments { reason } => {
                Self::InvalidArguments { reason }
            }
            types::effects::ExecutionFailureStatus::DuplicateValidator => Self::DuplicateValidator,
            types::effects::ExecutionFailureStatus::NotAValidator => Self::NotAValidator,
            types::effects::ExecutionFailureStatus::ValidatorAlreadyRemoved => {
                Self::ValidatorAlreadyRemoved
            }
            types::effects::ExecutionFailureStatus::AdvancedToWrongEpoch => {
                Self::AdvancedToWrongEpoch
            }
            types::effects::ExecutionFailureStatus::DuplicateEncoder => Self::DuplicateEncoder,
            types::effects::ExecutionFailureStatus::NotAnEncoder => Self::NotAnEncoder,
            types::effects::ExecutionFailureStatus::EncoderAlreadyRemoved => {
                Self::EncoderAlreadyRemoved
            }
            types::effects::ExecutionFailureStatus::InsufficientCoinBalance => {
                Self::InsufficientCoinBalance
            }
            types::effects::ExecutionFailureStatus::CoinBalanceOverflow => {
                Self::CoinBalanceOverflow
            }
            types::effects::ExecutionFailureStatus::ValidatorNotFound => Self::ValidatorNotFound,
            types::effects::ExecutionFailureStatus::EncoderNotFound => Self::EncoderNotFound,
            types::effects::ExecutionFailureStatus::StakingPoolNotFound => {
                Self::StakingPoolNotFound
            }
            types::effects::ExecutionFailureStatus::CannotReportOneself => {
                Self::CannotReportOneself
            }
            types::effects::ExecutionFailureStatus::ReportRecordNotFound => {
                Self::ReportRecordNotFound
            }
            types::effects::ExecutionFailureStatus::SomaError(soma_error) => {
                Self::OtherError(soma_error.to_string())
            }
        }
    }
}

impl From<ExecutionError> for types::effects::ExecutionFailureStatus {
    fn from(value: ExecutionError) -> Self {
        match value {
            ExecutionError::InsufficientGas => Self::InsufficientGas,
            _ => unreachable!("sdk shouldn't have a variant that the mono repo doesn't"),
        }
    }
}

impl From<types::effects::ExecutionStatus> for ExecutionStatus {
    fn from(value: types::effects::ExecutionStatus) -> Self {
        match value {
            types::effects::ExecutionStatus::Success => Self::Success,
            types::effects::ExecutionStatus::Failure { error } => Self::Failure {
                error: error.into(),
            },
        }
    }
}

impl TryFrom<types::crypto::Signature> for SimpleSignature {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::crypto::Signature) -> Result<Self, Self::Error> {
        match value {
            types::crypto::Signature::Ed25519SomaSignature(ed25519_sui_signature) => {
                Self::Ed25519 {
                    signature: Ed25519Signature::from_bytes(
                        ed25519_sui_signature.signature_bytes(),
                    )?,
                    public_key: Ed25519PublicKey::from_bytes(
                        ed25519_sui_signature.public_key_bytes(),
                    )?,
                }
            }
        }
        .pipe(Ok)
    }
}

impl From<types::crypto::SignatureScheme> for SignatureScheme {
    fn from(value: types::crypto::SignatureScheme) -> Self {
        match value {
            types::crypto::SignatureScheme::ED25519 => Self::Ed25519,

            types::crypto::SignatureScheme::BLS12381 => Self::Bls12381,
        }
    }
}

impl From<types::transaction::ChangeEpoch> for ChangeEpoch {
    fn from(
        types::transaction::ChangeEpoch {
            epoch,
            epoch_start_timestamp_ms,
        }: types::transaction::ChangeEpoch,
    ) -> Self {
        Self {
            epoch,
            epoch_start_timestamp_ms,
        }
    }
}
