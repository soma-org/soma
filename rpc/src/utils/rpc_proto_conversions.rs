use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use crate::proto::{TryFromProtoError, soma::*};
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;
use crate::utils::types_conversions::SdkTypeConversionError;
use fastcrypto::bls12381::min_sig::BLS12381PublicKey;
use fastcrypto::traits::ToFromBytes;
use types::base::SomaAddress;
use types::crypto::SomaSignature;
use types::envelope::Message as _;
use types::metadata::MetadataAPI as _;

//
// TransactionFee
//

impl From<types::tx_fee::TransactionFee> for TransactionFee {
    fn from(
        types::tx_fee::TransactionFee {
            base_fee,
            operation_fee,
            value_fee,
            total_fee,
        }: types::tx_fee::TransactionFee,
    ) -> Self {
        let mut message = Self::default();
        message.base_fee = Some(base_fee);
        message.operation_fee = Some(operation_fee);
        message.value_fee = Some(value_fee);
        message.total_fee = Some(total_fee);
        message
    }
}

impl From<types::effects::ExecutionStatus> for ExecutionStatus {
    fn from(value: types::effects::ExecutionStatus) -> Self {
        let mut message = Self::default();
        match value {
            types::effects::ExecutionStatus::Success => {
                message.success = Some(true);
            }
            types::effects::ExecutionStatus::Failure { error } => {
                let description = format!("{error:?}");
                let mut error_message = ExecutionError::from(error);

                error_message.description = Some(description);

                message.success = Some(false);
                message.error = Some(error_message);
            }
        }

        message
    }
}

impl From<types::effects::ExecutionFailureStatus> for ExecutionError {
    fn from(value: types::effects::ExecutionFailureStatus) -> Self {
        use execution_error::ErrorDetails;
        use execution_error::ExecutionErrorKind;
        use types::effects::ExecutionFailureStatus as E;

        let mut message = Self::default();

        let (kind, details) = match value {
            E::InsufficientGas => (ExecutionErrorKind::InsufficientGas, None),
            E::InvalidOwnership { object_id, .. } => (
                ExecutionErrorKind::InvalidOwnership,
                Some(object_id.to_hex()),
            ),
            E::ObjectNotFound { object_id } => {
                (ExecutionErrorKind::ObjectNotFound, Some(object_id.to_hex()))
            }
            E::InvalidObjectType { object_id, .. } => (
                ExecutionErrorKind::InvalidObjectType,
                Some(object_id.to_hex()),
            ),
            E::InvalidTransactionType => (ExecutionErrorKind::InvalidTransactionType, None),
            E::InvalidArguments { reason } => (ExecutionErrorKind::InvalidArguments, Some(reason)),
            E::DuplicateValidator => (ExecutionErrorKind::DuplicateValidator, None),
            E::NotAValidator => (ExecutionErrorKind::NotAValidator, None),
            E::ValidatorAlreadyRemoved => (ExecutionErrorKind::ValidatorAlreadyRemoved, None),
            E::AdvancedToWrongEpoch => (ExecutionErrorKind::AdvancedToWrongEpoch, None),
            E::DuplicateEncoder => (ExecutionErrorKind::DuplicateEncoder, None),
            E::NotAnEncoder => (ExecutionErrorKind::NotAnEncoder, None),
            E::EncoderAlreadyRemoved => (ExecutionErrorKind::EncoderAlreadyRemoved, None),
            E::InsufficientCoinBalance => (ExecutionErrorKind::InsufficientCoinBalance, None),
            E::CoinBalanceOverflow => (ExecutionErrorKind::CoinBalanceOverflow, None),
            E::ValidatorNotFound => (ExecutionErrorKind::ValidatorNotFound, None),
            E::EncoderNotFound => (ExecutionErrorKind::EncoderNotFound, None),
            E::StakingPoolNotFound => (ExecutionErrorKind::StakingPoolNotFound, None),
            E::CannotReportOneself => (ExecutionErrorKind::CannotReportOneself, None),
            E::ReportRecordNotFound => (ExecutionErrorKind::ReportRecordNotFound, None),
            E::InputObjectDeleted => (ExecutionErrorKind::InputObjectDeleted, None),
            E::CertificateDenied => (ExecutionErrorKind::CertificateDenied, None),
            E::ExecutionCancelledDueToSharedObjectCongestion => {
                (ExecutionErrorKind::SharedObjectCongestion, None)
            }
            E::SomaError(e) => (ExecutionErrorKind::OtherError, Some(e.to_string())),
        };

        message.set_kind(kind);
        message
    }
}

//
// AuthorityQuorumSignInfo aka ValidatorAggregatedSignature
//

impl<const T: bool> From<types::crypto::AuthorityQuorumSignInfo<T>>
    for ValidatorAggregatedSignature
{
    fn from(value: types::crypto::AuthorityQuorumSignInfo<T>) -> Self {
        let mut message = Self::default();
        message.epoch = Some(value.epoch);
        message.signature = Some(value.signature.as_ref().to_vec().into());
        message.bitmap = value.signers_map.iter().collect();
        message
    }
}

//
// ValidatorCommittee
//

impl From<types::committee::Committee> for ValidatorCommittee {
    fn from(value: types::committee::Committee) -> Self {
        let mut message = Self::default();
        message.epoch = Some(value.epoch);

        let authorities: Vec<_> = value.authorities().collect();

        message.members = authorities
            .into_iter()
            .map(|(i, authority)| {
                let network_key = authority.network_key.clone();
                let authority_key_bytes = authority.authority_key.as_bytes().to_vec();
                let protocol_key_bytes = authority.protocol_key.to_bytes().to_vec();
                let network_key_bytes = network_key.into_inner().as_bytes().to_vec();

                let mut member = ValidatorCommitteeMember::default();
                member.authority_key = Some(authority_key_bytes.into());
                member.weight = Some(authority.stake);

                member.network_metadata = Some(ValidatorNetworkMetadata {
                    consensus_address: Some(authority.address.to_string()),
                    hostname: Some(authority.hostname.clone()),
                    protocol_key: Some(protocol_key_bytes.into()),
                    network_key: Some(network_key_bytes.into()),
                });
                member
            })
            .collect();
        message
    }
}

//
// SignatureScheme
//

impl From<types::crypto::SignatureScheme> for SignatureScheme {
    fn from(value: types::crypto::SignatureScheme) -> Self {
        use types::crypto::SignatureScheme as S;

        match value {
            S::ED25519 => Self::Ed25519,
            S::BLS12381 => Self::Bls12381,
            S::MultiSig => Self::Multisig,
        }
    }
}

//
// SimpleSignature
//

impl From<types::crypto::Signature> for SimpleSignature {
    fn from(value: types::crypto::Signature) -> Self {
        let scheme: SignatureScheme = value.scheme().into();
        let signature = value.signature_bytes();
        let public_key = value.public_key_bytes();

        let mut message = Self::default();
        message.scheme = Some(scheme.into());
        message.signature = Some(signature.to_vec().into());
        message.public_key = Some(public_key.to_vec().into());
        message
    }
}

//
// MultisigMemberPublicKey
//

impl From<&types::crypto::PublicKey> for MultisigMemberPublicKey {
    fn from(value: &types::crypto::PublicKey) -> Self {
        let mut message = Self::default();

        match value {
            types::crypto::PublicKey::Ed25519(_) => {
                message.public_key = Some(value.as_ref().to_vec().into());
            }
        }

        message.set_scheme(value.scheme().into());
        message
    }
}

//
// MultisigCommittee
//

impl From<&types::multisig::MultiSigPublicKey> for MultisigCommittee {
    fn from(value: &types::multisig::MultiSigPublicKey) -> Self {
        let mut message = Self::default();
        message.members = value
            .pubkeys()
            .iter()
            .map(|(pk, weight)| {
                let mut member = MultisigMember::default();
                member.public_key = Some(pk.into());
                member.weight = Some((*weight).into());
                member
            })
            .collect();
        message.threshold = Some((*value.threshold()).into());
        message
    }
}

//
// MultisigMemberSignature
//

impl From<&types::crypto::CompressedSignature> for MultisigMemberSignature {
    fn from(value: &types::crypto::CompressedSignature) -> Self {
        let mut message = Self::default();

        let scheme = match value {
            types::crypto::CompressedSignature::Ed25519(b) => {
                message.signature = Some(b.0.to_vec().into());
                SignatureScheme::Ed25519
            }
        };

        message.set_scheme(scheme);
        message
    }
}

//
// MultisigAggregatedSignature
//

impl From<&types::multisig::MultiSig> for MultisigAggregatedSignature {
    fn from(value: &types::multisig::MultiSig) -> Self {
        let mut message = Self::default();
        message.signatures = value.get_sigs().iter().map(Into::into).collect();
        message.bitmap = Some(value.get_bitmap().into());
        message.committee = Some(value.get_pk().into());
        message
    }
}

//
// UserSignature
//

impl From<types::crypto::GenericSignature> for UserSignature {
    fn from(value: types::crypto::GenericSignature) -> Self {
        Self::merge_from(&value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<&types::crypto::GenericSignature> for UserSignature {
    fn merge(&mut self, source: &types::crypto::GenericSignature, mask: &FieldMaskTree) {
        use user_signature::Signature;

        let scheme = match source {
            types::crypto::GenericSignature::Signature(signature) => {
                let scheme = signature.scheme().into();
                if mask.contains(Self::SIMPLE_FIELD) {
                    self.signature = Some(Signature::Simple(signature.clone().into()));
                }
                scheme
            }
            types::crypto::GenericSignature::MultiSig(multi_sig) => {
                if mask.contains(Self::MULTISIG_FIELD) {
                    self.signature = Some(Signature::Multisig(multi_sig.into()));
                }
                SignatureScheme::Multisig
            }
        };

        if mask.contains(Self::SCHEME_FIELD) {
            self.set_scheme(scheme);
        }
    }
}

//
// BalanceChange
//

impl From<types::balance_change::BalanceChange> for BalanceChange {
    fn from(value: types::balance_change::BalanceChange) -> Self {
        let mut message = Self::default();
        message.address = Some(value.address.to_string());
        message.amount = Some(value.amount.to_string());
        message
    }
}

impl TryFrom<&BalanceChange> for types::balance_change::BalanceChange {
    type Error = TryFromProtoError;

    fn try_from(value: &BalanceChange) -> Result<Self, Self::Error> {
        let address = value
            .address
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("address"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid("address", e))?;

        let amount: i128 = value
            .amount
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("amount"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid("amount", e))?;

        Ok(types::balance_change::BalanceChange { address, amount })
    }
}

impl From<types::object::Object> for Object {
    fn from(value: types::object::Object) -> Self {
        Self::merge_from(&value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<&types::object::Object> for Object {
    fn merge(&mut self, source: &types::object::Object, mask: &FieldMaskTree) {
        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = Some(source.digest().to_string());
        }

        if mask.contains(Self::OBJECT_ID_FIELD.name) {
            self.object_id = Some(source.id().to_hex());
        }

        if mask.contains(Self::VERSION_FIELD.name) {
            self.version = Some(source.version().value());
        }

        if mask.contains(Self::OWNER_FIELD.name) {
            self.owner = Some(source.owner().to_owned().into());
        }

        if mask.contains(Self::PREVIOUS_TRANSACTION_FIELD.name) {
            self.previous_transaction = Some(source.previous_transaction.to_string());
        }

        if mask.contains(Self::OBJECT_TYPE_FIELD.name) {
            self.object_type = Some(source.data.object_type().to_string());
        }

        if mask.contains(Self::CONTENTS_FIELD.name) {
            self.contents = Some(source.data.contents().to_vec().into());
        }
    }
}

//
// ObjectReference
//

fn object_ref_to_proto(value: types::object::ObjectRef) -> ObjectReference {
    let (object_id, version, digest) = value;
    let mut message = ObjectReference::default();
    message.object_id = Some(object_id.to_hex());
    message.version = Some(version.value());
    message.digest = Some(digest.to_string());
    message
}

//
// Owner
//

impl From<types::object::Owner> for Owner {
    fn from(value: types::object::Owner) -> Self {
        use owner::OwnerKind;
        use types::object::Owner as O;

        let mut message = Self::default();

        let kind = match value {
            O::AddressOwner(address) => {
                message.address = Some(address.to_string());
                OwnerKind::Address
            }
            O::Shared {
                initial_shared_version,
            } => {
                message.version = Some(initial_shared_version.value());
                OwnerKind::Shared
            }
            O::Immutable => OwnerKind::Immutable,
        };

        message.set_kind(kind);
        message
    }
}

//
// Transaction
//

impl From<types::transaction::TransactionData> for Transaction {
    fn from(value: types::transaction::TransactionData) -> Self {
        Self::merge_from(&value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<&types::transaction::TransactionData> for Transaction {
    fn merge(&mut self, source: &types::transaction::TransactionData, mask: &FieldMaskTree) {
        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = Some(source.digest().to_string());
        }

        if mask.contains(Self::KIND_FIELD.name) {
            self.kind = Some(source.kind.clone().into());
        }

        if mask.contains(Self::SENDER_FIELD.name) {
            self.sender = Some(source.sender.to_string());
        }

        if mask.contains(Self::GAS_PAYMENT_FIELD.name) {
            self.gas_payment = source
                .gas_payment
                .clone()
                .into_iter()
                .map(|g| object_ref_to_proto(g))
                .collect();
        }
    }
}

//
// TransactionKind
//

impl From<types::transaction::TransactionKind> for TransactionKind {
    fn from(value: types::transaction::TransactionKind) -> Self {
        use transaction_kind::Kind;
        use types::transaction::TransactionKind as K;

        let kind = match value {
            K::Genesis(genesis) => Kind::Genesis(GenesisTransaction {
                objects: genesis.objects.into_iter().map(Into::into).collect(),
            }),
            K::ConsensusCommitPrologue(prologue) => Kind::ConsensusCommitPrologue(prologue.into()),
            K::ChangeEpoch(change_epoch) => Kind::ChangeEpoch(change_epoch.into()),
            K::AddValidator(args) => Kind::AddValidator(args.into()),
            K::RemoveValidator(args) => Kind::RemoveValidator(args.into()),
            K::ReportValidator { reportee } => Kind::ReportValidator(ReportValidator {
                reportee: Some(reportee.to_string()),
            }),
            K::UndoReportValidator { reportee } => Kind::UndoReportValidator(UndoReportValidator {
                reportee: Some(reportee.to_string()),
            }),
            K::UpdateValidatorMetadata(args) => Kind::UpdateValidatorMetadata(args.into()),
            K::SetCommissionRate { new_rate } => Kind::SetCommissionRate(SetCommissionRate {
                new_rate: Some(new_rate),
            }),
            K::AddEncoder(args) => Kind::AddEncoder(args.into()),
            K::RemoveEncoder(args) => Kind::RemoveEncoder(args.into()),
            K::ReportEncoder { reportee } => Kind::ReportEncoder(ReportEncoder {
                reportee: Some(reportee.to_string()),
            }),
            K::UndoReportEncoder { reportee } => Kind::UndoReportEncoder(UndoReportEncoder {
                reportee: Some(reportee.to_string()),
            }),
            K::UpdateEncoderMetadata(args) => Kind::UpdateEncoderMetadata(args.into()),
            K::SetEncoderCommissionRate { new_rate } => {
                Kind::SetEncoderCommissionRate(SetEncoderCommissionRate {
                    new_rate: Some(new_rate),
                })
            }
            K::SetEncoderBytePrice { new_price } => {
                Kind::SetEncoderBytePrice(SetEncoderBytePrice {
                    new_price: Some(new_price),
                })
            }
            K::TransferCoin {
                coin,
                amount,
                recipient,
            } => Kind::TransferCoin(TransferCoin {
                coin: Some(object_ref_to_proto(coin)),
                amount: amount,
                recipient: Some(recipient.to_string()),
            }),
            K::PayCoins {
                coins,
                amounts,
                recipients,
            } => Kind::PayCoins(PayCoins {
                coins: coins.into_iter().map(|c| object_ref_to_proto(c)).collect(),
                amounts: amounts.unwrap_or_default(),
                recipients: recipients.into_iter().map(|r| r.to_string()).collect(),
            }),
            K::TransferObjects { objects, recipient } => Kind::TransferObjects(TransferObjects {
                objects: objects
                    .into_iter()
                    .map(|o| object_ref_to_proto(o))
                    .collect(),
                recipient: Some(recipient.to_string()),
            }),
            K::AddStake {
                address,
                coin_ref,
                amount,
            } => Kind::AddStake(AddStake {
                address: Some(address.to_string()),
                coin_ref: Some(object_ref_to_proto(coin_ref)),
                amount: amount,
            }),
            K::AddStakeToEncoder {
                encoder_address,
                coin_ref,
                amount,
            } => Kind::AddStakeToEncoder(AddStakeToEncoder {
                encoder_address: Some(encoder_address.to_string()),
                coin_ref: Some(object_ref_to_proto(coin_ref)),
                amount: amount,
            }),
            K::WithdrawStake { staked_soma } => Kind::WithdrawStake(WithdrawStake {
                staked_soma: Some(object_ref_to_proto(staked_soma)),
            }),
            K::EmbedData {
                download_metadata,
                coin_ref,
                target_ref,
            } => Kind::EmbedData(EmbedData {
                download_metadata: Some(download_metadata.into()),
                coin_ref: Some(object_ref_to_proto(coin_ref)),
                target_ref: target_ref.map(|r| object_ref_to_proto(r)),
            }),
            K::ClaimEscrow { shard_ref } => Kind::ClaimEscrow(ClaimEscrow {
                shard_ref: Some(object_ref_to_proto(shard_ref)),
            }),
            K::ReportWinner {
                shard_ref,
                target_ref,
                report,
                signature,
                signers,
                shard_auth_token,
            } => Kind::ReportWinner(ReportWinner {
                shard_ref: Some(object_ref_to_proto(shard_ref)),
                target_ref: target_ref.map(|r| object_ref_to_proto(r)),
                report: Some(report.into()),
                signature: Some(signature.into()),
                signers: signers
                    .into_iter()
                    .map(|s| {
                        // Assuming EncoderPublicKey has a to_bytes() or similar method
                        format!("0x{}", hex::encode(s.to_bytes()))
                    })
                    .collect(),
                shard_auth_token: Some(shard_auth_token.into()),
            }),
            K::ClaimReward { target_ref } => Kind::ClaimReward(ClaimReward {
                target_ref: Some(object_ref_to_proto(target_ref)),
            }),
        };

        let mut message = Self::default();
        message.kind = Some(kind);
        message
    }
}

// Implement conversions for Args types
impl From<types::transaction::AddValidatorArgs> for AddValidator {
    fn from(args: types::transaction::AddValidatorArgs) -> Self {
        Self {
            pubkey_bytes: Some(args.pubkey_bytes.into()),
            network_pubkey_bytes: Some(args.network_pubkey_bytes.into()),
            worker_pubkey_bytes: Some(args.worker_pubkey_bytes.into()),
            net_address: Some(args.net_address.into()),
            p2p_address: Some(args.p2p_address.into()),
            primary_address: Some(args.primary_address.into()),
            encoder_validator_address: Some(args.encoder_validator_address.into()),
        }
    }
}

impl From<types::transaction::RemoveValidatorArgs> for RemoveValidator {
    fn from(args: types::transaction::RemoveValidatorArgs) -> Self {
        Self {
            pubkey_bytes: Some(args.pubkey_bytes.into()),
        }
    }
}

impl From<types::transaction::UpdateValidatorMetadataArgs> for UpdateValidatorMetadata {
    fn from(args: types::transaction::UpdateValidatorMetadataArgs) -> Self {
        Self {
            next_epoch_network_address: args.next_epoch_network_address.map(|bytes| bytes.into()),
            next_epoch_p2p_address: args.next_epoch_p2p_address.map(|bytes| bytes.into()),
            next_epoch_primary_address: args.next_epoch_primary_address.map(|bytes| bytes.into()),
            next_epoch_protocol_pubkey: args.next_epoch_protocol_pubkey.map(|bytes| bytes.into()),
            next_epoch_worker_pubkey: args.next_epoch_worker_pubkey.map(|bytes| bytes.into()),
            next_epoch_network_pubkey: args.next_epoch_network_pubkey.map(|bytes| bytes.into()),
        }
    }
}

impl From<types::transaction::AddEncoderArgs> for AddEncoder {
    fn from(args: types::transaction::AddEncoderArgs) -> Self {
        Self {
            encoder_pubkey_bytes: Some(args.encoder_pubkey_bytes.into()),
            network_pubkey_bytes: Some(args.network_pubkey_bytes.into()),
            internal_network_address: Some(args.internal_network_address.into()),
            external_network_address: Some(args.external_network_address.into()),
            object_server_address: Some(args.object_server_address.into()),
            probe: Some(args.probe.into()),
        }
    }
}

impl From<types::transaction::RemoveEncoderArgs> for RemoveEncoder {
    fn from(args: types::transaction::RemoveEncoderArgs) -> Self {
        Self {
            encoder_pubkey_bytes: Some(args.encoder_pubkey_bytes.into()),
        }
    }
}

impl From<types::transaction::UpdateEncoderMetadataArgs> for UpdateEncoderMetadata {
    fn from(args: types::transaction::UpdateEncoderMetadataArgs) -> Self {
        Self {
            next_epoch_external_network_address: args
                .next_epoch_external_network_address
                .map(|bytes| bytes.into()),
            next_epoch_internal_network_address: args
                .next_epoch_internal_network_address
                .map(|bytes| bytes.into()),
            next_epoch_network_pubkey: args.next_epoch_network_pubkey.map(|bytes| bytes.into()),
            next_epoch_object_server_address: args
                .next_epoch_object_server_address
                .map(|bytes| bytes.into()),
            next_epoch_probe: args.next_epoch_probe.map(|bytes| bytes.into()),
        }
    }
}

//
// ConsensusCommitPrologue
//

impl From<types::consensus::ConsensusCommitPrologue> for ConsensusCommitPrologue {
    fn from(value: types::consensus::ConsensusCommitPrologue) -> Self {
        let mut message = Self::default();
        message.epoch = Some(value.epoch);
        message.round = Some(value.round);
        message.commit_timestamp = Some(crate::proto::timestamp_ms_to_proto(
            value.commit_timestamp_ms,
        ));
        message
    }
}

//
// GenesisTransaction
//

impl From<types::transaction::GenesisTransaction> for GenesisTransaction {
    fn from(value: types::transaction::GenesisTransaction) -> Self {
        let mut message = Self::default();
        message.objects = value.objects.into_iter().map(Into::into).collect();
        message
    }
}

//
// ChangeEpoch
//

impl From<types::transaction::ChangeEpoch> for ChangeEpoch {
    fn from(value: types::transaction::ChangeEpoch) -> Self {
        Self {
            epoch: Some(value.epoch),
            epoch_start_timestamp: Some(crate::proto::timestamp_ms_to_proto(
                value.epoch_start_timestamp_ms,
            )),
            protocol_version: Some(value.protocol_version.as_u64()),
            fees: Some(value.fees),
            epoch_randomness: Some(value.epoch_randomness.into()),
        }
    }
}

//
// Shard
//
// Proto to SDK
impl TryFrom<&Shard> for crate::types::Shard {
    type Error = TryFromProtoError;

    fn try_from(value: &Shard) -> Result<Self, Self::Error> {
        let quorum_threshold = value
            .quorum_threshold
            .ok_or_else(|| TryFromProtoError::missing("quorum_threshold"))?;

        // Now much simpler!
        let encoders = value
            .encoders
            .iter()
            .enumerate()
            .map(|(idx, encoder_str)| {
                types::shard_crypto::keys::EncoderPublicKey::from_str(encoder_str)
                    .map(|key| key.to_bytes().to_vec())
                    .map_err(|e| TryFromProtoError::invalid(format!("encoders[{}]", idx), e))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let seed = value
            .seed
            .clone()
            .ok_or_else(|| TryFromProtoError::missing("seed"))?
            .to_vec();

        let epoch = value
            .epoch
            .ok_or_else(|| TryFromProtoError::missing("epoch"))?;

        Ok(crate::types::Shard::new(
            quorum_threshold,
            encoders,
            seed,
            epoch,
        ))
    }
}

// SDK to Proto
impl From<crate::types::Shard> for Shard {
    fn from(value: crate::types::Shard) -> Self {
        Self {
            quorum_threshold: Some(value.quorum_threshold),
            // Create EncoderPublicKey from bytes, then convert to string
            encoders: value
                .encoders
                .into_iter()
                .filter_map(|bytes| {
                    types::shard_crypto::keys::EncoderPublicKey::from_bytes(&bytes)
                        .ok()
                        .map(|key| key.to_hex_string())
                })
                .collect(),
            seed: Some(value.seed.into()),
            epoch: Some(value.epoch),
        }
    }
}

//
// TransactionEffects
//

impl From<types::effects::TransactionEffects> for TransactionEffects {
    fn from(value: types::effects::TransactionEffects) -> Self {
        Self::merge_from(&value, &FieldMaskTree::new_wildcard())
    }
}
impl Merge<&types::effects::TransactionEffects> for TransactionEffects {
    fn merge(&mut self, source: &types::effects::TransactionEffects, mask: &FieldMaskTree) {
        if mask.contains(Self::STATUS_FIELD.name) {
            self.status = Some(source.status.clone().into());
        }

        if mask.contains(Self::EPOCH_FIELD.name) {
            self.epoch = Some(source.executed_epoch);
        }

        if mask.contains(Self::FEE_FIELD.name) {
            self.fee = Some(source.transaction_fee.clone().into());
        }

        if mask.contains(Self::TRANSACTION_DIGEST_FIELD.name) {
            self.transaction_digest = Some(source.transaction_digest.to_string());
        }

        if mask.contains(Self::GAS_OBJECT_INDEX_FIELD.name) {
            self.gas_object_index = source.gas_object_index;
        }

        if mask.contains(Self::DEPENDENCIES_FIELD.name) {
            self.dependencies = source
                .dependencies
                .iter()
                .map(ToString::to_string)
                .collect();
        }

        if mask.contains(Self::LAMPORT_VERSION_FIELD.name) {
            self.lamport_version = Some(source.version.value());
        }

        if mask.contains(Self::CHANGED_OBJECTS_FIELD.name) {
            self.changed_objects = source
                .changed_objects
                .iter()
                .map(|(id, change)| {
                    let mut message = ChangedObject::from(change.clone());
                    message.object_id = Some(id.to_hex());
                    message
                })
                .collect();
        }

        // Set version for all objects that have output_digest but no output_version
        for object in self.changed_objects.iter_mut() {
            if object.output_digest.is_some() && object.output_version.is_none() {
                object.output_version = Some(source.version.value());
            }
        }

        if mask.contains(Self::UNCHANGED_SHARED_OBJECTS_FIELD.name) {
            self.unchanged_shared_objects = source
                .unchanged_shared_objects
                .iter()
                .map(|(id, unchanged)| {
                    let mut message = UnchangedSharedObject::from(unchanged.clone());
                    message.object_id = Some(id.to_hex());
                    message
                })
                .collect();
        }
    }
}

//
// ChangedObject
//

impl From<types::effects::object_change::EffectsObjectChange> for ChangedObject {
    fn from(value: types::effects::object_change::EffectsObjectChange) -> Self {
        use changed_object::InputObjectState;
        use changed_object::OutputObjectState;
        use types::effects::object_change::ObjectIn;
        use types::effects::object_change::ObjectOut;

        let mut message = Self::default();

        // Input State
        let input_state = match value.input_state {
            ObjectIn::NotExist => InputObjectState::DoesNotExist,
            ObjectIn::Exist(((version, digest), owner)) => {
                message.input_version = Some(version.value());
                message.input_digest = Some(digest.to_string());
                message.input_owner = Some(owner.into());
                InputObjectState::Exists
            }
        };
        message.set_input_state(input_state);

        // Output State
        let output_state = match value.output_state {
            ObjectOut::NotExist => OutputObjectState::DoesNotExist,
            ObjectOut::ObjectWrite((digest, owner)) => {
                message.output_digest = Some(digest.to_string());
                message.output_owner = Some(owner.into());
                OutputObjectState::ObjectWrite
            }
        };
        message.set_output_state(output_state);

        message.set_id_operation(value.id_operation.into());
        message
    }
}

//
// IdOperation
//

impl From<types::effects::object_change::IDOperation> for changed_object::IdOperation {
    fn from(value: types::effects::object_change::IDOperation) -> Self {
        use types::effects::object_change::IDOperation as I;

        match value {
            I::None => Self::None,
            I::Created => Self::Created,
            I::Deleted => Self::Deleted,
        }
    }
}

//
// UnchangedSharedObject
//

impl From<types::effects::UnchangedSharedKind> for UnchangedSharedObject {
    fn from(value: types::effects::UnchangedSharedKind) -> Self {
        use types::effects::UnchangedSharedKind as K;
        use unchanged_shared_object::UnchangedSharedObjectKind;

        let mut message = Self::default();

        let kind = match value {
            K::ReadOnlyRoot((version, digest)) => {
                message.version = Some(version.value());
                message.digest = Some(digest.to_string());
                UnchangedSharedObjectKind::ReadOnlyRoot
            }
            K::MutateDeleted(version) => {
                message.version = Some(version.value());
                UnchangedSharedObjectKind::MutatedDeleted
            }
            K::ReadDeleted(version) => {
                message.version = Some(version.value());
                UnchangedSharedObjectKind::ReadDeleted
            }
            K::Cancelled(version) => {
                message.version = Some(version.value());
                UnchangedSharedObjectKind::Canceled
            }
        };

        message.set_kind(kind);
        message
    }
}

impl TryFrom<SystemState> for types::system_state::SystemState {
    type Error = String;

    fn try_from(proto_state: SystemState) -> Result<Self, Self::Error> {
        let epoch = proto_state.epoch.ok_or("Missing epoch")?;
        let protocol_version = proto_state
            .protocol_version
            .ok_or("Missing protocol_version")?;
        let epoch_start_timestamp_ms = proto_state
            .epoch_start_timestamp_ms
            .ok_or("Missing epoch_start_timestamp_ms")?;

        let parameters: protocol_config::SystemParameters = proto_state
            .parameters
            .ok_or("Missing parameters")?
            .try_into()?;

        let validators = proto_state
            .validators
            .ok_or("Missing validators")?
            .try_into()?;

        let encoders = proto_state.encoders.ok_or("Missing encoders")?.try_into()?;

        let emission_pool = proto_state
            .emission_pool
            .ok_or("Missing emission_pool")?
            .try_into()?;

        let reference_byte_price = proto_state
            .reference_byte_price
            .ok_or("Missing reference_byte_price")?;

        // Convert map fields
        let target_rewards_per_epoch = proto_state.target_rewards_per_epoch;
        let targets_created_per_epoch = proto_state.targets_created_per_epoch;
        let epoch_seeds = proto_state
            .epoch_seeds
            .into_iter()
            .map(|(k, v)| (k, v.to_vec()))
            .collect();

        // Convert validator report records
        let validator_report_records =
            convert_report_records(proto_state.validator_report_records)?;

        // Convert encoder report records
        let encoder_report_records = convert_report_records(proto_state.encoder_report_records)?;

        let vdf_iterations = parameters.vdf_iterations;

        // Build initial committees
        let mut system_state = types::system_state::SystemState {
            epoch,
            protocol_version,
            epoch_start_timestamp_ms,
            parameters,
            validators,
            encoders,
            validator_report_records,
            encoder_report_records,
            emission_pool,
            reference_byte_price,
            target_rewards_per_epoch,
            targets_created_per_epoch,
            epoch_seeds,
            committees: [None, None],
        };

        // Initialize current epoch committees
        let current_committees = system_state.build_committees_for_epoch(epoch, vdf_iterations);
        system_state.committees[1] = Some(current_committees);

        Ok(system_state)
    }
}

impl TryFrom<SystemParameters> for protocol_config::SystemParameters {
    type Error = String;

    fn try_from(proto_params: SystemParameters) -> Result<Self, Self::Error> {
        Ok(protocol_config::SystemParameters {
            epoch_duration_ms: proto_params
                .epoch_duration_ms
                .ok_or("Missing epoch_duration_ms")?,
            vdf_iterations: proto_params
                .vdf_iterations
                .ok_or("Missing vdf_iterations")?,
            target_selection_rate_bps: proto_params
                .target_selection_rate_bps
                .ok_or("Missing target_selection_rate_bps")?,
            target_reward_allocation_bps: proto_params
                .target_reward_allocation_bps
                .ok_or("Missing target_reward_allocation_bps")?,
            encoder_tally_slash_rate_bps: proto_params
                .encoder_tally_slash_rate_bps
                .ok_or("Missing encoder_tally_slash_rate_bps")?,
            target_epoch_fee_collection: proto_params
                .target_epoch_fee_collection
                .ok_or("Missing target_epoch_fee_collection")?,
            base_fee: proto_params.base_fee.ok_or("Missing base_fee")?,
            write_object_fee: proto_params
                .write_object_fee
                .ok_or("Missing write_object_fee")?,
            value_fee_bps: proto_params.value_fee_bps.ok_or("Missing value_fee_bps")?,
            min_value_fee_bps: proto_params
                .min_value_fee_bps
                .ok_or("Missing min_value_fee_bps")?,
            max_value_fee_bps: proto_params
                .max_value_fee_bps
                .ok_or("Missing max_value_fee_bps")?,
            fee_adjustment_rate_bps: proto_params
                .fee_adjustment_rate_bps
                .ok_or("Missing fee_adjustment_rate_bps")?,
            claim_incentive_bps: proto_params
                .claim_incentive_bps
                .ok_or("Missing claim_incentive_bps")?,
        })
    }
}

impl TryFrom<EmissionPool> for types::system_state::emission::EmissionPool {
    type Error = String;

    fn try_from(proto_emission_pool: EmissionPool) -> Result<Self, Self::Error> {
        Ok(types::system_state::emission::EmissionPool {
            balance: proto_emission_pool.balance.ok_or("Missing balance")?,
            emission_per_epoch: proto_emission_pool
                .emission_per_epoch
                .ok_or("Missing emission_per_epoch")?,
        })
    }
}

impl TryFrom<ValidatorSet> for types::system_state::validator::ValidatorSet {
    type Error = String;

    fn try_from(proto_set: ValidatorSet) -> Result<Self, Self::Error> {
        let consensus_validators = proto_set
            .consensus_validators
            .into_iter()
            .map(|v| v.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let networking_validators = proto_set
            .networking_validators
            .into_iter()
            .map(|v| v.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let pending_validators = proto_set
            .pending_validators
            .into_iter()
            .map(|v| v.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let pending_removals = proto_set
            .pending_removals
            .into_iter()
            .map(|r| {
                Ok((
                    r.is_consensus.ok_or("Missing is_consensus")?,
                    r.index.ok_or("Missing index")? as usize,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let staking_pool_mappings = proto_set
            .staking_pool_mappings
            .into_iter()
            .map(|(k, v)| {
                let pool_id = k.parse().map_err(|_| "Invalid ObjectID")?;
                let address = v.parse().map_err(|_| "Invalid SomaAddress")?;
                Ok((pool_id, address))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let inactive_validators = proto_set
            .inactive_validators
            .into_iter()
            .map(|(k, v)| {
                let pool_id = k.parse().map_err(|_| "Invalid ObjectID")?;
                let validator = v.try_into()?;
                Ok((pool_id, validator))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let at_risk_validators = proto_set
            .at_risk_validators
            .into_iter()
            .map(|(k, v)| {
                let address = k.parse().map_err(|_| "Invalid SomaAddress")?;
                Ok((address, v))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        Ok(types::system_state::validator::ValidatorSet {
            total_stake: proto_set.total_stake.ok_or("Missing total_stake")?,
            consensus_validators,
            networking_validators,
            pending_validators,
            pending_removals,
            staking_pool_mappings,
            inactive_validators,
            at_risk_validators,
        })
    }
}

impl TryFrom<Validator> for types::system_state::validator::Validator {
    type Error = String;

    fn try_from(proto_val: Validator) -> Result<Self, Self::Error> {
        use fastcrypto::traits::ToFromBytes;
        use std::str::FromStr;

        let soma_address = proto_val
            .soma_address
            .ok_or("Missing soma_address")?
            .parse()
            .map_err(|_| "Invalid SomaAddress")?;

        let protocol_pubkey = proto_val
            .protocol_pubkey
            .ok_or("Missing protocol_pubkey")?
            .to_vec();
        let protocol_pubkey = BLS12381PublicKey::from_bytes(&protocol_pubkey)
            .map_err(|e| format!("Invalid protocol_pubkey: {}", e))?;

        let network_pubkey = proto_val
            .network_pubkey
            .ok_or("Missing network_pubkey")?
            .to_vec();
        let network_pubkey = fastcrypto::ed25519::Ed25519PublicKey::from_bytes(&network_pubkey)
            .map_err(|e| format!("Invalid network_pubkey: {}", e))?;
        let network_pubkey = types::crypto::NetworkPublicKey::new(network_pubkey);

        let worker_pubkey = proto_val
            .worker_pubkey
            .ok_or("Missing worker_pubkey")?
            .to_vec();
        let worker_pubkey = fastcrypto::ed25519::Ed25519PublicKey::from_bytes(&worker_pubkey)
            .map_err(|e| format!("Invalid worker_pubkey: {}", e))?;
        let worker_pubkey = types::crypto::NetworkPublicKey::new(worker_pubkey);

        let net_address = types::multiaddr::Multiaddr::from_str(
            &proto_val.net_address.ok_or("Missing net_address")?,
        )
        .map_err(|e| format!("Invalid net_address: {}", e))?;

        let p2p_address = types::multiaddr::Multiaddr::from_str(
            &proto_val.p2p_address.ok_or("Missing p2p_address")?,
        )
        .map_err(|e| format!("Invalid p2p_address: {}", e))?;

        let primary_address = types::multiaddr::Multiaddr::from_str(
            &proto_val.primary_address.ok_or("Missing primary_address")?,
        )
        .map_err(|e| format!("Invalid primary_address: {}", e))?;

        let encoder_validator_address = types::multiaddr::Multiaddr::from_str(
            &proto_val
                .encoder_validator_address
                .ok_or("Missing encoder_validator_address")?,
        )
        .map_err(|e| format!("Invalid encoder_validator_address: {}", e))?;

        // Convert optional next epoch fields
        let next_epoch_protocol_pubkey = proto_val
            .next_epoch_protocol_pubkey
            .map(|bytes| {
                BLS12381PublicKey::from_bytes(&bytes)
                    .map_err(|e| format!("Invalid next_epoch_protocol_pubkey: {}", e))
            })
            .transpose()?;

        let next_epoch_network_pubkey = proto_val
            .next_epoch_network_pubkey
            .map(|bytes| {
                fastcrypto::ed25519::Ed25519PublicKey::from_bytes(&bytes)
                    .map(types::crypto::NetworkPublicKey::new)
                    .map_err(|e| format!("Invalid next_epoch_network_pubkey: {}", e))
            })
            .transpose()?;

        let next_epoch_worker_pubkey = proto_val
            .next_epoch_worker_pubkey
            .map(|bytes| {
                fastcrypto::ed25519::Ed25519PublicKey::from_bytes(&bytes)
                    .map(types::crypto::NetworkPublicKey::new)
                    .map_err(|e| format!("Invalid next_epoch_worker_pubkey: {}", e))
            })
            .transpose()?;

        let next_epoch_net_address = proto_val
            .next_epoch_net_address
            .map(|addr| {
                types::multiaddr::Multiaddr::from_str(&addr)
                    .map_err(|e| format!("Invalid next_epoch_net_address: {}", e))
            })
            .transpose()?;

        let next_epoch_p2p_address = proto_val
            .next_epoch_p2p_address
            .map(|addr| {
                types::multiaddr::Multiaddr::from_str(&addr)
                    .map_err(|e| format!("Invalid next_epoch_p2p_address: {}", e))
            })
            .transpose()?;

        let next_epoch_primary_address = proto_val
            .next_epoch_primary_address
            .map(|addr| {
                types::multiaddr::Multiaddr::from_str(&addr)
                    .map_err(|e| format!("Invalid next_epoch_primary_address: {}", e))
            })
            .transpose()?;

        let next_epoch_encoder_validator_address = proto_val
            .next_epoch_encoder_validator_address
            .map(|addr| {
                types::multiaddr::Multiaddr::from_str(&addr)
                    .map_err(|e| format!("Invalid next_epoch_encoder_validator_address: {}", e))
            })
            .transpose()?;

        let metadata = types::system_state::validator::ValidatorMetadata {
            soma_address,
            protocol_pubkey,
            network_pubkey,
            worker_pubkey,
            net_address,
            p2p_address,
            primary_address,
            encoder_validator_address,
            next_epoch_protocol_pubkey,
            next_epoch_network_pubkey,
            next_epoch_net_address,
            next_epoch_p2p_address,
            next_epoch_primary_address,
            next_epoch_worker_pubkey,
            next_epoch_encoder_validator_address,
        };

        let staking_pool = proto_val
            .staking_pool
            .ok_or("Missing staking_pool")?
            .try_into()?;

        Ok(types::system_state::validator::Validator {
            metadata,
            voting_power: proto_val.voting_power.ok_or("Missing voting_power")?,
            staking_pool,
            commission_rate: proto_val.commission_rate.ok_or("Missing commission_rate")?,
            next_epoch_stake: proto_val
                .next_epoch_stake
                .ok_or("Missing next_epoch_stake")?,
            next_epoch_commission_rate: proto_val
                .next_epoch_commission_rate
                .ok_or("Missing next_epoch_commission_rate")?,
        })
    }
}

impl TryFrom<StakingPool> for types::system_state::staking::StakingPool {
    type Error = String;

    fn try_from(proto_pool: StakingPool) -> Result<Self, Self::Error> {
        let id = proto_pool
            .id
            .ok_or("Missing id")?
            .parse()
            .map_err(|_| "Invalid ObjectID")?;

        let exchange_rates = proto_pool
            .exchange_rates
            .into_iter()
            .map(|(k, v)| {
                let rate = v.try_into()?;
                Ok((k, rate))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        Ok(types::system_state::staking::StakingPool {
            id,
            activation_epoch: proto_pool.activation_epoch,
            deactivation_epoch: proto_pool.deactivation_epoch,
            soma_balance: proto_pool.soma_balance.ok_or("Missing soma_balance")?,
            rewards_pool: proto_pool.rewards_pool.ok_or("Missing rewards_pool")?,
            pool_token_balance: proto_pool
                .pool_token_balance
                .ok_or("Missing pool_token_balance")?,
            exchange_rates,
            pending_stake: proto_pool.pending_stake.ok_or("Missing pending_stake")?,
            pending_total_soma_withdraw: proto_pool
                .pending_total_soma_withdraw
                .ok_or("Missing pending_total_soma_withdraw")?,
            pending_pool_token_withdraw: proto_pool
                .pending_pool_token_withdraw
                .ok_or("Missing pending_pool_token_withdraw")?,
        })
    }
}

impl TryFrom<PoolTokenExchangeRate> for types::system_state::staking::PoolTokenExchangeRate {
    type Error = String;

    fn try_from(proto_rate: PoolTokenExchangeRate) -> Result<Self, Self::Error> {
        Ok(types::system_state::staking::PoolTokenExchangeRate {
            soma_amount: proto_rate.soma_amount.ok_or("Missing soma_amount")?,
            pool_token_amount: proto_rate
                .pool_token_amount
                .ok_or("Missing pool_token_amount")?,
        })
    }
}

impl TryFrom<EncoderSet> for types::system_state::encoder::EncoderSet {
    type Error = String;

    fn try_from(proto_set: EncoderSet) -> Result<Self, Self::Error> {
        let active_encoders = proto_set
            .active_encoders
            .into_iter()
            .map(|e| e.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let pending_active_encoders = proto_set
            .pending_active_encoders
            .into_iter()
            .map(|e| e.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let pending_removals = proto_set
            .pending_removals
            .into_iter()
            .map(|idx| idx as usize)
            .collect();

        let staking_pool_mappings = proto_set
            .staking_pool_mappings
            .into_iter()
            .map(|(k, v)| {
                let pool_id = k.parse().map_err(|_| "Invalid ObjectID")?;
                let address = v.parse().map_err(|_| "Invalid SomaAddress")?;
                Ok((pool_id, address))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let inactive_encoders = proto_set
            .inactive_encoders
            .into_iter()
            .map(|(k, v)| {
                let pool_id = k.parse().map_err(|_| "Invalid ObjectID")?;
                let encoder = v.try_into()?;
                Ok((pool_id, encoder))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let at_risk_encoders = proto_set
            .at_risk_encoders
            .into_iter()
            .map(|(k, v)| {
                let address = k.parse().map_err(|_| "Invalid SomaAddress")?;
                Ok((address, v))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        Ok(types::system_state::encoder::EncoderSet {
            total_stake: proto_set.total_stake.ok_or("Missing total_stake")?,
            active_encoders,
            pending_active_encoders,
            pending_removals,
            staking_pool_mappings,
            inactive_encoders,
            at_risk_encoders,
        })
    }
}

impl TryFrom<Encoder> for types::system_state::encoder::Encoder {
    type Error = String;

    fn try_from(proto_enc: Encoder) -> Result<Self, Self::Error> {
        use fastcrypto::traits::ToFromBytes;
        use std::str::FromStr;

        let soma_address = proto_enc
            .soma_address
            .ok_or("Missing soma_address")?
            .parse()
            .map_err(|_| "Invalid SomaAddress")?;

        let encoder_pubkey = proto_enc
            .encoder_pubkey
            .ok_or("Missing encoder_pubkey")?
            .to_vec();
        let encoder_pubkey =
            fastcrypto::bls12381::min_sig::BLS12381PublicKey::from_bytes(&encoder_pubkey)
                .map_err(|e| format!("Invalid encoder_pubkey: {}", e))?;
        let encoder_pubkey = types::shard_crypto::keys::EncoderPublicKey::new(encoder_pubkey);

        let network_pubkey = proto_enc
            .network_pubkey
            .ok_or("Missing network_pubkey")?
            .to_vec();
        let network_pubkey = fastcrypto::ed25519::Ed25519PublicKey::from_bytes(&network_pubkey)
            .map_err(|e| format!("Invalid network_pubkey: {}", e))?;
        let network_pubkey = types::crypto::NetworkPublicKey::new(network_pubkey);

        let internal_network_address = types::multiaddr::Multiaddr::from_str(
            &proto_enc
                .internal_network_address
                .ok_or("Missing internal_network_address")?,
        )
        .map_err(|e| format!("Invalid internal_network_address: {}", e))?;

        let external_network_address = types::multiaddr::Multiaddr::from_str(
            &proto_enc
                .external_network_address
                .ok_or("Missing external_network_address")?,
        )
        .map_err(|e| format!("Invalid external_network_address: {}", e))?;

        let object_server_address = types::multiaddr::Multiaddr::from_str(
            &proto_enc
                .object_server_address
                .ok_or("Missing object_server_address")?,
        )
        .map_err(|e| format!("Invalid object_server_address: {}", e))?;

        let probe = bcs::from_bytes::<types::metadata::DownloadMetadata>(
            &proto_enc.probe.ok_or("Missing probe")?,
        )
        .map_err(|e| format!("Invalid probe: {}", e))?;

        // Convert optional next epoch fields
        let next_epoch_network_pubkey = proto_enc
            .next_epoch_network_pubkey
            .map(|bytes| {
                fastcrypto::ed25519::Ed25519PublicKey::from_bytes(&bytes)
                    .map(types::crypto::NetworkPublicKey::new)
                    .map_err(|e| format!("Invalid next_epoch_network_pubkey: {}", e))
            })
            .transpose()?;

        let next_epoch_internal_network_address = proto_enc
            .next_epoch_internal_network_address
            .map(|addr| {
                types::multiaddr::Multiaddr::from_str(&addr)
                    .map_err(|e| format!("Invalid next_epoch_internal_network_address: {}", e))
            })
            .transpose()?;

        let next_epoch_external_network_address = proto_enc
            .next_epoch_external_network_address
            .map(|addr| {
                types::multiaddr::Multiaddr::from_str(&addr)
                    .map_err(|e| format!("Invalid next_epoch_external_network_address: {}", e))
            })
            .transpose()?;

        let next_epoch_object_server_address = proto_enc
            .next_epoch_object_server_address
            .map(|addr| {
                types::multiaddr::Multiaddr::from_str(&addr)
                    .map_err(|e| format!("Invalid next_epoch_object_server_address: {}", e))
            })
            .transpose()?;

        let next_epoch_probe: Option<types::metadata::DownloadMetadata> = proto_enc
            .next_epoch_probe
            .map(|addr| {
                bcs::from_bytes::<types::metadata::DownloadMetadata>(&addr)
                    .map_err(|e| format!("Invalid next_epoch_probe: {}", e))
            })
            .transpose()?;

        let metadata = types::system_state::encoder::EncoderMetadata {
            soma_address,
            encoder_pubkey,
            network_pubkey,
            internal_network_address,
            external_network_address,
            object_server_address,
            probe,
            next_epoch_network_pubkey,
            next_epoch_internal_network_address,
            next_epoch_external_network_address,
            next_epoch_object_server_address,
            next_epoch_probe,
        };

        let staking_pool = proto_enc
            .staking_pool
            .ok_or("Missing staking_pool")?
            .try_into()?;

        Ok(types::system_state::encoder::Encoder {
            metadata,
            voting_power: proto_enc.voting_power.ok_or("Missing voting_power")?,
            staking_pool,
            commission_rate: proto_enc.commission_rate.ok_or("Missing commission_rate")?,
            next_epoch_stake: proto_enc
                .next_epoch_stake
                .ok_or("Missing next_epoch_stake")?,
            next_epoch_commission_rate: proto_enc
                .next_epoch_commission_rate
                .ok_or("Missing next_epoch_commission_rate")?,
            byte_price: proto_enc.byte_price.ok_or("Missing byte_price")?,
            next_epoch_byte_price: proto_enc
                .next_epoch_byte_price
                .ok_or("Missing next_epoch_byte_price")?,
        })
    }
}

// Helper functions
fn convert_report_records(
    proto_records: BTreeMap<String, ReporterSet>,
) -> Result<BTreeMap<SomaAddress, BTreeSet<SomaAddress>>, String> {
    proto_records
        .into_iter()
        .map(|(k, v)| {
            let key = k.parse().map_err(|_| "Invalid SomaAddress")?;
            let reporters = v
                .reporters
                .into_iter()
                .map(|r| r.parse().map_err(|_| "Invalid SomaAddress"))
                .collect::<Result<BTreeSet<_>, _>>()?;
            Ok((key, reporters))
        })
        .collect()
}

impl TryFrom<types::system_state::SystemState> for SystemState {
    type Error = String;

    fn try_from(domain_state: types::system_state::SystemState) -> Result<Self, Self::Error> {
        // Convert validator report records
        let validator_report_records =
            convert_report_records_to_proto(domain_state.validator_report_records)?;

        // Convert encoder report records
        let encoder_report_records =
            convert_report_records_to_proto(domain_state.encoder_report_records)?;

        // Convert epoch_seeds: BTreeMap<u64, Vec<u8>> -> BTreeMap<u64, Bytes>
        let epoch_seeds = domain_state
            .epoch_seeds
            .into_iter()
            .map(|(k, v)| (k, v.into()))
            .collect();

        Ok(SystemState {
            epoch: Some(domain_state.epoch),
            protocol_version: Some(domain_state.protocol_version),
            epoch_start_timestamp_ms: Some(domain_state.epoch_start_timestamp_ms),
            parameters: Some(domain_state.parameters.try_into()?),
            validators: Some(domain_state.validators.try_into()?),
            encoders: Some(domain_state.encoders.try_into()?),
            validator_report_records,
            encoder_report_records,
            emission_pool: Some(domain_state.emission_pool.try_into()?),
            reference_byte_price: Some(domain_state.reference_byte_price),
            target_rewards_per_epoch: domain_state.target_rewards_per_epoch,
            targets_created_per_epoch: domain_state.targets_created_per_epoch,
            epoch_seeds,
        })
    }
}

impl TryFrom<protocol_config::SystemParameters> for SystemParameters {
    type Error = String;

    fn try_from(domain_params: protocol_config::SystemParameters) -> Result<Self, Self::Error> {
        Ok(SystemParameters {
            epoch_duration_ms: Some(domain_params.epoch_duration_ms),
            vdf_iterations: Some(domain_params.vdf_iterations),
            target_selection_rate_bps: Some(domain_params.target_selection_rate_bps),
            target_reward_allocation_bps: Some(domain_params.target_reward_allocation_bps),
            encoder_tally_slash_rate_bps: Some(domain_params.encoder_tally_slash_rate_bps),
            target_epoch_fee_collection: Some(domain_params.target_epoch_fee_collection),
            base_fee: Some(domain_params.base_fee),
            write_object_fee: Some(domain_params.write_object_fee),
            value_fee_bps: Some(domain_params.value_fee_bps),
            min_value_fee_bps: Some(domain_params.min_value_fee_bps),
            max_value_fee_bps: Some(domain_params.max_value_fee_bps),
            fee_adjustment_rate_bps: Some(domain_params.fee_adjustment_rate_bps),
            claim_incentive_bps: Some(domain_params.claim_incentive_bps),
        })
    }
}

impl TryFrom<types::system_state::emission::EmissionPool> for EmissionPool {
    type Error = String;

    fn try_from(
        domain_emission_pool: types::system_state::emission::EmissionPool,
    ) -> Result<Self, Self::Error> {
        Ok(EmissionPool {
            balance: Some(domain_emission_pool.balance),
            emission_per_epoch: Some(domain_emission_pool.emission_per_epoch),
        })
    }
}

impl TryFrom<types::system_state::validator::ValidatorSet> for ValidatorSet {
    type Error = String;

    fn try_from(
        domain_set: types::system_state::validator::ValidatorSet,
    ) -> Result<Self, Self::Error> {
        let consensus_validators = domain_set
            .consensus_validators
            .into_iter()
            .map(|v| v.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let networking_validators = domain_set
            .networking_validators
            .into_iter()
            .map(|v| v.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let pending_validators = domain_set
            .pending_validators
            .into_iter()
            .map(|v| v.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let pending_removals = domain_set
            .pending_removals
            .into_iter()
            .map(|(is_consensus, index)| PendingRemoval {
                is_consensus: Some(is_consensus),
                index: Some(index as u32),
            })
            .collect();

        let staking_pool_mappings = domain_set
            .staking_pool_mappings
            .into_iter()
            .map(|(pool_id, address)| (pool_id.to_string(), address.to_string()))
            .collect();

        let inactive_validators = domain_set
            .inactive_validators
            .into_iter()
            .map(|(pool_id, validator)| {
                let proto_validator: Validator = validator.try_into()?;
                Ok((pool_id.to_string(), proto_validator))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let at_risk_validators = domain_set
            .at_risk_validators
            .into_iter()
            .map(|(address, epochs)| (address.to_string(), epochs))
            .collect();

        Ok(ValidatorSet {
            total_stake: Some(domain_set.total_stake),
            consensus_validators,
            networking_validators,
            pending_validators,
            pending_removals,
            staking_pool_mappings,
            inactive_validators,
            at_risk_validators,
        })
    }
}

impl TryFrom<types::system_state::validator::Validator> for Validator {
    type Error = String;

    fn try_from(
        domain_val: types::system_state::validator::Validator,
    ) -> Result<Self, Self::Error> {
        use bytes::Bytes;
        use fastcrypto::traits::ToFromBytes;

        let metadata = domain_val.metadata;

        // Convert optional next epoch fields
        let next_epoch_protocol_pubkey = metadata
            .next_epoch_protocol_pubkey
            .map(|key| Bytes::from(key.as_bytes().to_vec()));

        let next_epoch_network_pubkey = metadata
            .next_epoch_network_pubkey
            .map(|key| Bytes::from(key.to_bytes().to_vec()));

        let next_epoch_worker_pubkey = metadata
            .next_epoch_worker_pubkey
            .map(|key| Bytes::from(key.to_bytes().to_vec()));

        let next_epoch_net_address = metadata.next_epoch_net_address.map(|addr| addr.to_string());

        let next_epoch_p2p_address = metadata.next_epoch_p2p_address.map(|addr| addr.to_string());

        let next_epoch_primary_address = metadata
            .next_epoch_primary_address
            .map(|addr| addr.to_string());

        let next_epoch_encoder_validator_address = metadata
            .next_epoch_encoder_validator_address
            .map(|addr| addr.to_string());

        Ok(Validator {
            soma_address: Some(metadata.soma_address.to_string()),
            protocol_pubkey: Some(Bytes::from(metadata.protocol_pubkey.as_bytes().to_vec())),
            network_pubkey: Some(Bytes::from(metadata.network_pubkey.to_bytes().to_vec())),
            worker_pubkey: Some(Bytes::from(metadata.worker_pubkey.to_bytes().to_vec())),
            net_address: Some(metadata.net_address.to_string()),
            p2p_address: Some(metadata.p2p_address.to_string()),
            primary_address: Some(metadata.primary_address.to_string()),
            encoder_validator_address: Some(metadata.encoder_validator_address.to_string()),
            voting_power: Some(domain_val.voting_power),
            commission_rate: Some(domain_val.commission_rate),
            next_epoch_stake: Some(domain_val.next_epoch_stake),
            next_epoch_commission_rate: Some(domain_val.next_epoch_commission_rate),
            staking_pool: Some(domain_val.staking_pool.try_into()?),
            next_epoch_protocol_pubkey,
            next_epoch_network_pubkey,
            next_epoch_worker_pubkey,
            next_epoch_net_address,
            next_epoch_p2p_address,
            next_epoch_primary_address,
            next_epoch_encoder_validator_address,
        })
    }
}

impl TryFrom<types::system_state::staking::StakingPool> for StakingPool {
    type Error = String;

    fn try_from(
        domain_pool: types::system_state::staking::StakingPool,
    ) -> Result<Self, Self::Error> {
        let exchange_rates = domain_pool
            .exchange_rates
            .into_iter()
            .map(|(k, v)| {
                let proto_rate: PoolTokenExchangeRate = v.try_into()?;
                Ok((k, proto_rate))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        Ok(StakingPool {
            id: Some(domain_pool.id.to_string()),
            activation_epoch: domain_pool.activation_epoch,
            deactivation_epoch: domain_pool.deactivation_epoch,
            soma_balance: Some(domain_pool.soma_balance),
            rewards_pool: Some(domain_pool.rewards_pool),
            pool_token_balance: Some(domain_pool.pool_token_balance),
            exchange_rates,
            pending_stake: Some(domain_pool.pending_stake),
            pending_total_soma_withdraw: Some(domain_pool.pending_total_soma_withdraw),
            pending_pool_token_withdraw: Some(domain_pool.pending_pool_token_withdraw),
        })
    }
}

impl TryFrom<types::system_state::staking::PoolTokenExchangeRate> for PoolTokenExchangeRate {
    type Error = String;

    fn try_from(
        domain_rate: types::system_state::staking::PoolTokenExchangeRate,
    ) -> Result<Self, Self::Error> {
        Ok(PoolTokenExchangeRate {
            soma_amount: Some(domain_rate.soma_amount),
            pool_token_amount: Some(domain_rate.pool_token_amount),
        })
    }
}

impl TryFrom<types::system_state::encoder::EncoderSet> for EncoderSet {
    type Error = String;

    fn try_from(domain_set: types::system_state::encoder::EncoderSet) -> Result<Self, Self::Error> {
        let active_encoders = domain_set
            .active_encoders
            .into_iter()
            .map(|e| e.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let pending_active_encoders = domain_set
            .pending_active_encoders
            .into_iter()
            .map(|e| e.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let pending_removals = domain_set
            .pending_removals
            .into_iter()
            .map(|idx| idx as u32)
            .collect();

        let staking_pool_mappings = domain_set
            .staking_pool_mappings
            .into_iter()
            .map(|(pool_id, address)| (pool_id.to_string(), address.to_string()))
            .collect();

        let inactive_encoders = domain_set
            .inactive_encoders
            .into_iter()
            .map(|(pool_id, encoder)| {
                let proto_encoder: Encoder = encoder.try_into()?;
                Ok((pool_id.to_string(), proto_encoder))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let at_risk_encoders = domain_set
            .at_risk_encoders
            .into_iter()
            .map(|(address, epochs)| (address.to_string(), epochs))
            .collect();

        Ok(EncoderSet {
            total_stake: Some(domain_set.total_stake),
            active_encoders,
            pending_active_encoders,
            pending_removals,
            staking_pool_mappings,
            inactive_encoders,
            at_risk_encoders,
        })
    }
}

impl TryFrom<types::system_state::encoder::Encoder> for Encoder {
    type Error = String;

    fn try_from(domain_enc: types::system_state::encoder::Encoder) -> Result<Self, Self::Error> {
        let metadata = domain_enc.metadata;

        let probe = bcs::to_bytes(&metadata.probe)
            .map_err(|e| format!("Invalid probe download metadata: {}", e))?
            .into();

        // Convert optional next epoch fields
        let next_epoch_network_pubkey = metadata
            .next_epoch_network_pubkey
            .map(|key| key.to_bytes().to_vec().into());

        let next_epoch_internal_network_address = metadata
            .next_epoch_internal_network_address
            .map(|addr| addr.to_string());

        let next_epoch_external_network_address = metadata
            .next_epoch_external_network_address
            .map(|addr| addr.to_string());

        let next_epoch_object_server_address = metadata
            .next_epoch_object_server_address
            .map(|addr| addr.to_string());

        let next_epoch_probe = metadata
            .next_epoch_probe
            .map(|addr| {
                bcs::to_bytes(&addr)
                    .map_err(|e| format!("Invalid next epoch probe download metadata: {}", e))
            })
            .transpose()?
            .map(Into::into);

        Ok(Encoder {
            soma_address: Some(metadata.soma_address.to_string()),
            encoder_pubkey: Some(metadata.encoder_pubkey.to_bytes().to_vec().into()),
            network_pubkey: Some(metadata.network_pubkey.to_bytes().to_vec().into()),
            internal_network_address: Some(metadata.internal_network_address.to_string()),
            external_network_address: Some(metadata.external_network_address.to_string()),
            object_server_address: Some(metadata.object_server_address.to_string()),
            voting_power: Some(domain_enc.voting_power),
            commission_rate: Some(domain_enc.commission_rate),
            next_epoch_stake: Some(domain_enc.next_epoch_stake),
            next_epoch_commission_rate: Some(domain_enc.next_epoch_commission_rate),
            byte_price: Some(domain_enc.byte_price),
            next_epoch_byte_price: Some(domain_enc.next_epoch_byte_price),
            probe: Some(probe),
            staking_pool: Some(domain_enc.staking_pool.try_into()?),
            next_epoch_network_pubkey,
            next_epoch_internal_network_address,
            next_epoch_external_network_address,
            next_epoch_object_server_address,
            next_epoch_probe,
        })
    }
}

// Helper functions for reverse conversion
fn convert_report_records_to_proto(
    domain_records: BTreeMap<types::base::SomaAddress, BTreeSet<types::base::SomaAddress>>,
) -> Result<BTreeMap<String, ReporterSet>, String> {
    domain_records
        .into_iter()
        .map(|(k, v)| {
            let key = k.to_string();
            let reporters = v.into_iter().map(|r| r.to_string()).collect();
            Ok((key, ReporterSet { reporters }))
        })
        .collect()
}

//
// TransactionChecks
//

impl From<simulate_transaction_request::TransactionChecks>
    for types::transaction_executor::TransactionChecks
{
    fn from(value: simulate_transaction_request::TransactionChecks) -> Self {
        match value {
            simulate_transaction_request::TransactionChecks::Enabled => Self::Enabled,
            simulate_transaction_request::TransactionChecks::Disabled => Self::Disabled,
            // Default to enabled
            _ => Self::Enabled,
        }
    }
}

impl From<types::metadata::Metadata> for Metadata {
    fn from(value: types::metadata::Metadata) -> Self {
        let mut message = Self::default();
        match value {
            types::metadata::Metadata::V1(v1) => {
                let mut proto_v1 = MetadataV1::default();
                proto_v1.checksum = Some(v1.checksum().as_bytes().to_vec().into());
                proto_v1.size = Some(v1.size() as u64);
                message.version = Some(crate::proto::soma::metadata::Version::V1(proto_v1));
            }
        }
        message
    }
}

// Add this conversion for Metadata
impl TryFrom<&Metadata> for types::metadata::Metadata {
    type Error = TryFromProtoError;

    fn try_from(value: &crate::proto::soma::Metadata) -> Result<Self, Self::Error> {
        use crate::proto::soma::metadata::Version;

        match value
            .version
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("metadata version"))?
        {
            Version::V1(v1) => {
                let checksum_bytes = v1
                    .checksum
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("checksum"))?
                    .as_ref();

                // Convert bytes to Checksum
                let checksum = types::checksum::Checksum::from_bytes(checksum_bytes)
                    .map_err(|e| TryFromProtoError::invalid("checksum", e))?;

                let size = v1
                    .size
                    .ok_or_else(|| TryFromProtoError::missing("size"))?
                    .try_into()
                    .map_err(|e| TryFromProtoError::invalid("size", e))?;

                Ok(types::metadata::Metadata::V1(
                    types::metadata::MetadataV1::new(checksum, size),
                ))
            }
        }
    }
}

// Also add the owned version
impl TryFrom<Metadata> for types::metadata::Metadata {
    type Error = TryFromProtoError;

    fn try_from(value: Metadata) -> Result<Self, Self::Error> {
        (&value).try_into()
    }
}

impl From<types::metadata::DownloadMetadata> for DownloadMetadata {
    fn from(value: types::metadata::DownloadMetadata) -> Self {
        use download_metadata::Kind;
        use types::metadata::{DefaultDownloadMetadataAPI, MetadataAPI, MtlsDownloadMetadataAPI};

        let kind = match value {
            types::metadata::DownloadMetadata::Default(dm) => {
                Kind::Default(DefaultDownloadMetadata {
                    version: Some(default_download_metadata::Version::V1(
                        DefaultDownloadMetadataV1 {
                            url: Some(dm.url().to_string()),
                            metadata: Some(dm.metadata().clone().into()),
                        },
                    )),
                })
            }
            types::metadata::DownloadMetadata::Mtls(dm) => Kind::Mtls(MtlsDownloadMetadata {
                version: Some(mtls_download_metadata::Version::V1(
                    MtlsDownloadMetadataV1 {
                        peer: Some(dm.peer().to_bytes().to_vec().into()),
                        url: Some(dm.url().to_string()),
                        metadata: Some(dm.metadata().clone().into()),
                    },
                )),
            }),
        };

        DownloadMetadata { kind: Some(kind) }
    }
}

impl TryFrom<&DownloadMetadata> for types::metadata::DownloadMetadata {
    type Error = TryFromProtoError;

    fn try_from(value: &DownloadMetadata) -> Result<Self, Self::Error> {
        use download_metadata::Kind;

        match value
            .kind
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("kind"))?
        {
            Kind::Default(dm) => {
                let v1 = match dm
                    .version
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("version"))?
                {
                    default_download_metadata::Version::V1(v1) => v1,
                };

                let url = v1
                    .url
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("url"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("url", e))?;
                let metadata = v1
                    .metadata
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("metadata"))?
                    .try_into()?;

                Ok(types::metadata::DownloadMetadata::Default(
                    types::metadata::DefaultDownloadMetadata::V1(
                        types::metadata::DefaultDownloadMetadataV1::new(url, metadata),
                    ),
                ))
            }
            Kind::Mtls(dm) => {
                let v1 = match dm
                    .version
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("version"))?
                {
                    mtls_download_metadata::Version::V1(v1) => v1,
                };

                let peer_bytes = v1
                    .peer
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("peer"))?;
                let network_pubkey = fastcrypto::ed25519::Ed25519PublicKey::from_bytes(peer_bytes)
                    .map_err(|e| TryFromProtoError::invalid("peer", e))?;
                let peer = types::crypto::NetworkPublicKey::new(network_pubkey);

                let url = v1
                    .url
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("url"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("url", e))?;
                let metadata = v1
                    .metadata
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("metadata"))?
                    .try_into()?;

                Ok(types::metadata::DownloadMetadata::Mtls(
                    types::metadata::MtlsDownloadMetadata::V1(
                        types::metadata::MtlsDownloadMetadataV1::new(peer, url, metadata),
                    ),
                ))
            }
        }
    }
}

//
// CheckpointSummary
//

impl Merge<&types::full_checkpoint_content::Checkpoint> for Checkpoint {
    fn merge(&mut self, source: &types::full_checkpoint_content::Checkpoint, mask: &FieldMaskTree) {
        let sequence_number = source.summary.sequence_number;
        let timestamp_ms = source.summary.timestamp_ms;

        let summary = source.summary.data();
        let signature = source.summary.auth_sig();

        self.merge(summary, mask);
        self.merge(signature.clone(), mask);

        if mask.contains(Checkpoint::CONTENTS_FIELD.name) {
            self.merge(&source.contents, mask);
        }

        if let Some(submask) = mask
            .subtree(Checkpoint::OBJECTS_FIELD)
            .and_then(|submask| submask.subtree(ObjectSet::OBJECTS_FIELD))
        {
            let set = source
                .object_set
                .iter()
                .map(|o| crate::proto::soma::Object::merge_from(o, &submask))
                .collect();
            self.objects = Some(ObjectSet::default().with_objects(set));
        }

        if let Some(submask) = mask.subtree(Checkpoint::TRANSACTIONS_FIELD.name) {
            self.transactions = source
                .transactions
                .iter()
                .map(|t| {
                    let mut transaction = ExecutedTransaction::merge_from(t, &submask);
                    transaction.checkpoint = submask
                        .contains(ExecutedTransaction::CHECKPOINT_FIELD)
                        .then_some(sequence_number);
                    transaction.timestamp = submask
                        .contains(ExecutedTransaction::TIMESTAMP_FIELD)
                        .then(|| crate::proto::timestamp_ms_to_proto(timestamp_ms));
                    transaction
                })
                .collect();
        }
    }
}

impl Merge<&types::full_checkpoint_content::ExecutedTransaction> for ExecutedTransaction {
    fn merge(
        &mut self,
        source: &types::full_checkpoint_content::ExecutedTransaction,
        mask: &FieldMaskTree,
    ) {
        if mask.contains(ExecutedTransaction::DIGEST_FIELD) {
            self.digest = Some(source.transaction.digest().to_string());
        }

        if let Some(submask) = mask.subtree(ExecutedTransaction::TRANSACTION_FIELD) {
            self.transaction = Some(Transaction::merge_from(&source.transaction, &submask));
        }

        if let Some(submask) = mask.subtree(ExecutedTransaction::SIGNATURES_FIELD) {
            self.signatures = source
                .signatures
                .iter()
                .map(|s| UserSignature::merge_from(s, &submask))
                .collect();
        }

        if let Some(submask) = mask.subtree(ExecutedTransaction::EFFECTS_FIELD) {
            let mut effects = TransactionEffects::merge_from(&source.effects, &submask);
            self.effects = Some(effects);
        }
    }
}

impl TryFrom<&Checkpoint> for types::full_checkpoint_content::Checkpoint {
    type Error = TryFromProtoError;

    fn try_from(checkpoint: &Checkpoint) -> Result<Self, Self::Error> {
        // Convert proto CheckpointSummary -> crate::types::CheckpointSummary -> types::checkpoints::CheckpointSummary
        let summary = {
            let proto_summary = checkpoint.summary();
            let crate_summary: crate::types::CheckpointSummary = proto_summary
                .try_into()
                .map_err(|e| TryFromProtoError::invalid("summary", e))?;

            let domain_summary: types::checkpoints::CheckpointSummary = crate_summary
                .try_into()
                .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("summary", e))?;

            // Get signature and combine into CertifiedCheckpointSummary
            let crate_sig: crate::types::ValidatorAggregatedSignature =
                checkpoint.signature().try_into()?;
            let signature = types::crypto::AuthorityStrongQuorumSignInfo::try_from(crate_sig)
                .map_err(|e| TryFromProtoError::invalid("signature", e))?;

            types::checkpoints::CertifiedCheckpointSummary::new_from_data_and_sig(
                domain_summary,
                signature,
            )
        };

        // Convert proto CheckpointContents -> crate::types::CheckpointContents -> types::checkpoints::CheckpointContents
        let contents = {
            let proto_contents = checkpoint.contents();
            let crate_contents: crate::types::CheckpointContents = proto_contents
                .try_into()
                .map_err(|e| TryFromProtoError::invalid("contents", e))?;

            crate_contents
                .try_into()
                .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("contents", e))?
        };

        let transactions = checkpoint
            .transactions()
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<_, _>>()?;

        let object_set = checkpoint.objects().try_into()?;

        Ok(Self {
            summary,
            contents,
            transactions,
            object_set,
        })
    }
}

impl TryFrom<&ObjectReference> for types::storage::ObjectKey {
    type Error = TryFromProtoError;

    fn try_from(value: &ObjectReference) -> Result<Self, Self::Error> {
        Ok(Self(
            value
                .object_id()
                .parse()
                .map_err(|e| TryFromProtoError::invalid("object_id", e))?,
            value.version().into(),
        ))
    }
}

//
// CheckpointSummary
//

impl From<types::checkpoints::CheckpointSummary> for CheckpointSummary {
    fn from(summary: types::checkpoints::CheckpointSummary) -> Self {
        Self::merge_from(summary, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<types::checkpoints::CheckpointSummary> for CheckpointSummary {
    fn merge(&mut self, source: types::checkpoints::CheckpointSummary, mask: &FieldMaskTree) {
        if mask.contains(Self::DIGEST_FIELD) {
            self.digest = Some(source.digest().to_string());
        }

        let types::checkpoints::CheckpointSummary {
            epoch,
            sequence_number,
            network_total_transactions,
            content_digest,
            previous_digest,
            epoch_rolling_transaction_fees,
            timestamp_ms,
            checkpoint_commitments,
            end_of_epoch_data,
        } = source;

        if mask.contains(Self::EPOCH_FIELD) {
            self.epoch = Some(epoch);
        }

        if mask.contains(Self::SEQUENCE_NUMBER_FIELD) {
            self.sequence_number = Some(sequence_number);
        }

        if mask.contains(Self::TOTAL_NETWORK_TRANSACTIONS_FIELD) {
            self.total_network_transactions = Some(network_total_transactions);
        }

        if mask.contains(Self::CONTENT_DIGEST_FIELD) {
            self.content_digest = Some(content_digest.to_string());
        }

        if mask.contains(Self::PREVIOUS_DIGEST_FIELD) {
            self.previous_digest = previous_digest.map(|d| d.to_string());
        }

        if mask.contains(Self::EPOCH_ROLLING_TRANSACTION_FEES_FIELD) {
            self.epoch_rolling_transaction_fees = Some(epoch_rolling_transaction_fees.into());
        }

        if mask.contains(Self::TIMESTAMP_FIELD) {
            self.timestamp = Some(crate::proto::timestamp_ms_to_proto(timestamp_ms));
        }

        if mask.contains(Self::COMMITMENTS_FIELD) {
            self.commitments = checkpoint_commitments.into_iter().map(Into::into).collect();
        }

        if mask.contains(Self::END_OF_EPOCH_DATA_FIELD) {
            self.end_of_epoch_data = end_of_epoch_data.map(Into::into);
        }
    }
}

//
// CheckpointCommitment
//

impl From<types::checkpoints::CheckpointCommitment> for CheckpointCommitment {
    fn from(value: types::checkpoints::CheckpointCommitment) -> Self {
        use checkpoint_commitment::CheckpointCommitmentKind;

        let mut message = Self::default();

        let kind = match value {
            types::checkpoints::CheckpointCommitment::ECMHLiveObjectSetDigest(digest) => {
                message.digest = Some(digest.digest.to_string());
                CheckpointCommitmentKind::EcmhLiveObjectSet
            }
            types::checkpoints::CheckpointCommitment::CheckpointArtifactsDigest(digest) => {
                message.digest = Some(digest.to_string());
                CheckpointCommitmentKind::CheckpointArtifacts
            }
        };

        message.set_kind(kind);
        message
    }
}

//
// EndOfEpochData
//

impl From<types::checkpoints::EndOfEpochData> for EndOfEpochData {
    fn from(value: types::checkpoints::EndOfEpochData) -> Self {
        Self {
            next_epoch_validator_committee: Some(value.next_epoch_validator_committee.into()),
            next_epoch_encoder_committee: Some(value.next_epoch_encoder_committee.into()),
            next_epoch_networking_committee: Some(value.next_epoch_networking_committee.into()),
            next_epoch_protocol_version: Some(value.next_epoch_protocol_version.as_u64()),
            epoch_commitments: value
                .epoch_commitments
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

//
// EncoderCommittee (domain -> proto)
//

impl From<types::encoder_committee::EncoderCommittee> for EncoderCommittee {
    fn from(value: types::encoder_committee::EncoderCommittee) -> Self {
        // Combine Encoder and EncoderNetworkMetadata into EncoderCommitteeMember
        let members: Vec<EncoderCommitteeMember> = value
            .members()
            .into_iter()
            .map(|(encoder_key, voting_power)| {
                let encoder = value.encoder_by_key(&encoder_key);
                let network_metadata = value.network_metadata.get(&encoder_key);

                let mut member = EncoderCommitteeMember::default();
                member.voting_power = Some(voting_power);
                member.encoder_key = Some(encoder_key.to_bytes().to_vec().into());

                // Add probe from Encoder if available
                if let Some(enc) = encoder {
                    member.probe = Some(enc.probe.clone().into());
                }

                // Add network metadata if available
                if let Some(net_meta) = network_metadata {
                    member.internal_network_address =
                        Some(net_meta.internal_network_address.to_string());
                    member.external_network_address =
                        Some(net_meta.external_network_address.to_string());
                    member.object_server_address = Some(net_meta.object_server_address.to_string());
                    member.network_key = Some(net_meta.network_key.to_bytes().to_vec().into());
                    member.hostname = Some(net_meta.hostname.clone());
                }

                member
            })
            .collect();

        Self {
            epoch: Some(value.epoch),
            members,
            shard_size: Some(value.shard_size()),
            quorum_threshold: Some(value.quorum_threshold()),
        }
    }
}

//
// EncoderCommittee (proto -> domain)
//

impl TryFrom<&EncoderCommittee> for types::encoder_committee::EncoderCommittee {
    type Error = TryFromProtoError;

    fn try_from(value: &EncoderCommittee) -> Result<Self, Self::Error> {
        use std::collections::BTreeMap;
        use types::encoder_committee::{Encoder, EncoderNetworkMetadata};
        use types::shard_crypto::keys::EncoderPublicKey;

        let epoch = value
            .epoch
            .ok_or_else(|| TryFromProtoError::missing("epoch"))?;
        let shard_size = value
            .shard_size
            .ok_or_else(|| TryFromProtoError::missing("shard_size"))?;
        let quorum_threshold = value
            .quorum_threshold
            .ok_or_else(|| TryFromProtoError::missing("quorum_threshold"))?;

        let mut encoders = Vec::new();
        let mut network_metadata = BTreeMap::new();

        for member in &value.members {
            let encoder_key_bytes = member
                .encoder_key
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("encoder_key"))?;
            let encoder_key = EncoderPublicKey::from_bytes(encoder_key_bytes)
                .map_err(|e| TryFromProtoError::invalid("encoder_key", e))?;

            let voting_power = member
                .voting_power
                .ok_or_else(|| TryFromProtoError::missing("voting_power"))?;

            let probe = member
                .probe
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("probe"))?
                .try_into()
                .map_err(|e| TryFromProtoError::invalid("probe", e))?;

            let encoder = Encoder {
                voting_power,
                encoder_key: encoder_key.clone(),
                probe,
            };
            encoders.push(encoder);

            // Build network metadata
            let internal_network_address = member
                .internal_network_address
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("internal_network_address"))?
                .parse()
                .map_err(|e| TryFromProtoError::invalid("internal_network_address", e))?;

            let external_network_address = member
                .external_network_address
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("external_network_address"))?
                .parse()
                .map_err(|e| TryFromProtoError::invalid("external_network_address", e))?;

            let object_server_address = member
                .object_server_address
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("object_server_address"))?
                .parse()
                .map_err(|e| TryFromProtoError::invalid("object_server_address", e))?;

            let network_key_bytes = member
                .network_key
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("network_key"))?;
            let ed25519_key = fastcrypto::ed25519::Ed25519PublicKey::from_bytes(network_key_bytes)
                .map_err(|e| TryFromProtoError::invalid("network_key", e))?;
            let network_key = types::crypto::NetworkPublicKey::new(ed25519_key);

            let hostname = member
                .hostname
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("hostname"))?;

            let net_meta = EncoderNetworkMetadata {
                internal_network_address,
                external_network_address,
                object_server_address,
                network_key,
                hostname,
            };
            network_metadata.insert(encoder_key, net_meta);
        }

        Ok(types::encoder_committee::EncoderCommittee::new(
            epoch,
            encoders,
            shard_size,
            quorum_threshold,
            network_metadata,
        ))
    }
}
//
// NetworkingCommittee (domain -> proto)
//

impl From<types::committee::NetworkingCommittee> for NetworkingCommittee {
    fn from(value: types::committee::NetworkingCommittee) -> Self {
        use fastcrypto::traits::ToFromBytes;

        let members: Vec<NetworkingCommitteeMember> = value
            .members
            .into_iter()
            .map(
                |(authority_name, network_metadata)| NetworkingCommitteeMember {
                    authority_key: Some(authority_name.as_ref().to_vec().into()),
                    network_metadata: Some(ValidatorNetworkMetadata {
                        consensus_address: Some(network_metadata.consensus_address.to_string()),
                        hostname: Some(network_metadata.hostname),
                        protocol_key: Some(
                            network_metadata.protocol_key.to_bytes().to_vec().into(),
                        ),
                        network_key: Some(network_metadata.network_key.to_bytes().to_vec().into()),
                    }),
                },
            )
            .collect();

        Self {
            epoch: Some(value.epoch),
            members,
        }
    }
}

//
// CheckpointContents
//

impl From<types::checkpoints::CheckpointContents> for CheckpointContents {
    fn from(value: types::checkpoints::CheckpointContents) -> Self {
        Self::merge_from(value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<types::checkpoints::CheckpointContents> for CheckpointContents {
    fn merge(&mut self, source: types::checkpoints::CheckpointContents, mask: &FieldMaskTree) {
        if mask.contains(Self::DIGEST_FIELD) {
            self.digest = Some(source.digest().to_string());
        }

        if mask.contains(Self::VERSION_FIELD) {
            self.version = Some(1);
        }

        if mask.contains(Self::TRANSACTIONS_FIELD) {
            self.transactions = source
                .into_iter_with_signatures()
                .map(|(digests, sigs)| {
                    let mut info = CheckpointedTransactionInfo::default();
                    info.transaction = Some(digests.transaction.to_string());
                    info.effects = Some(digests.effects.to_string());
                    info.signatures = sigs.into_iter().map(Into::into).collect();
                    info
                })
                .collect();
        }
    }
}

impl Merge<&types::checkpoints::CheckpointContents> for Checkpoint {
    fn merge(&mut self, source: &types::checkpoints::CheckpointContents, mask: &FieldMaskTree) {
        if let Some(submask) = mask.subtree(Self::CONTENTS_FIELD.name) {
            self.contents = Some(CheckpointContents::merge_from(source.to_owned(), &submask));
        }
    }
}

//
// Checkpoint
//

impl Merge<&types::checkpoints::CheckpointSummary> for Checkpoint {
    fn merge(&mut self, source: &types::checkpoints::CheckpointSummary, mask: &FieldMaskTree) {
        if mask.contains(Self::SEQUENCE_NUMBER_FIELD) {
            self.sequence_number = Some(source.sequence_number);
        }

        if mask.contains(Self::DIGEST_FIELD) {
            self.digest = Some(source.digest().to_string());
        }

        if let Some(submask) = mask.subtree(Self::SUMMARY_FIELD) {
            self.summary = Some(CheckpointSummary::merge_from(source.clone(), &submask));
        }
    }
}

impl<const T: bool> Merge<types::crypto::AuthorityQuorumSignInfo<T>> for Checkpoint {
    fn merge(&mut self, source: types::crypto::AuthorityQuorumSignInfo<T>, mask: &FieldMaskTree) {
        if mask.contains(Self::SIGNATURE_FIELD) {
            self.signature = Some(source.into());
        }
    }
}

impl Merge<types::checkpoints::CheckpointContents> for Checkpoint {
    fn merge(&mut self, source: types::checkpoints::CheckpointContents, mask: &FieldMaskTree) {
        if let Some(submask) = mask.subtree(Self::CONTENTS_FIELD) {
            self.contents = Some(CheckpointContents::merge_from(source, &submask));
        }
    }
}

impl TryFrom<&ObjectSet> for types::full_checkpoint_content::ObjectSet {
    type Error = TryFromProtoError;

    fn try_from(value: &ObjectSet) -> Result<Self, Self::Error> {
        let mut objects = Self::default();

        for o in value.objects() {
            let crate_object: crate::types::Object = o
                .try_into()
                .map_err(|e| TryFromProtoError::invalid("object", e))?;
            let object = crate_object
                .try_into()
                .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("object", e))?;
            objects.insert(object);
        }

        Ok(objects)
    }
}

impl TryFrom<&ExecutedTransaction> for types::full_checkpoint_content::ExecutedTransaction {
    type Error = TryFromProtoError;

    fn try_from(value: &ExecutedTransaction) -> Result<Self, Self::Error> {
        // Convert proto Transaction -> crate::types::Transaction -> types::transaction::TransactionData
        let transaction = {
            let proto_transaction = value.transaction();
            let crate_transaction: crate::types::Transaction = proto_transaction
                .try_into()
                .map_err(|e| TryFromProtoError::invalid("transaction", e))?;

            // Now convert crate::types::Transaction to types::transaction::TransactionData
            crate_transaction
                .try_into()
                .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("transaction", e))?
        };

        let signatures = {
            let proto_signature = value.signatures();
            let crate_signatures: Vec<crate::types::UserSignature> = proto_signature
                .iter()
                .map(|s| {
                    s.try_into()
                        .map_err(|e| TryFromProtoError::invalid("signature", e))
                })
                .collect::<Result<Vec<_>, _>>()?;

            crate_signatures
                .iter()
                .map(|s| {
                    (s.clone()).try_into().map_err(|e: SdkTypeConversionError| {
                        TryFromProtoError::invalid("signature", e)
                    })
                })
                .collect::<Result<Vec<types::crypto::GenericSignature>, _>>()?
        };

        // Convert proto TransactionEffects -> crate::types::TransactionEffects -> types::effects::TransactionEffects
        let effects = {
            let proto_effects = value.effects();
            let crate_effects: crate::types::TransactionEffects = proto_effects
                .try_into()
                .map_err(|e| TryFromProtoError::invalid("effects", e))?;

            // Now convert crate::types::TransactionEffects to types::effects::TransactionEffects
            crate_effects
                .try_into()
                .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("effects", e))?
        };

        Ok(Self {
            transaction,
            signatures,
            effects,
        })
    }
}
