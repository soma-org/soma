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
            gas_object_ref, // TODO: use gas_object_ref
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
            K::EmbedData { metadata, coin_ref } => Kind::EmbedData(EmbedData {
                metadata: Some(metadata.into()),
                coin_ref: Some(object_ref_to_proto(coin_ref)),
            }),
            K::ClaimEscrow { shard_input_ref } => Kind::ClaimEscrow(ClaimEscrow {
                shard_input_ref: Some(object_ref_to_proto(shard_input_ref)),
            }),
            K::ReportWinner {
                shard_input_ref,
                signed_report,
                signature,
                signers,
                shard_auth_token,
            } => Kind::ReportWinner(ReportWinner {
                shard_input_ref: Some(object_ref_to_proto(shard_input_ref)),
                signed_report: Some(signed_report.into()),
                encoder_aggregate_signature: Some(signature.into()),
                signers: signers
                    .into_iter()
                    .map(|s| {
                        // Assuming EncoderPublicKey has a to_bytes() or similar method
                        format!("0x{}", hex::encode(s.to_bytes()))
                    })
                    .collect(),
                shard_auth_token: Some(shard_auth_token.into()),
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
        let mut message = Self::default();
        message.epoch = Some(value.epoch);
        message.epoch_start_timestamp = Some(crate::proto::timestamp_ms_to_proto(
            value.epoch_start_timestamp_ms,
        ));
        message
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
            self.fee = source.transaction_fee.clone().map(|f| f.into());
        }

        if mask.contains(Self::TRANSACTION_DIGEST_FIELD.name) {
            self.transaction_digest = Some(source.transaction_digest.to_string());
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

        let epoch_start_timestamp_ms = proto_state
            .epoch_start_timestamp_ms
            .ok_or("Missing epoch_start_timestamp_ms")?;

        let parameters = proto_state
            .parameters
            .ok_or("Missing parameters")?
            .try_into()?;

        let validators = proto_state
            .validators
            .ok_or("Missing validators")?
            .try_into()?;

        let encoders = proto_state.encoders.ok_or("Missing encoders")?.try_into()?;

        let stake_subsidy = proto_state
            .stake_subsidy
            .ok_or("Missing stake_subsidy")?
            .try_into()?;

        // Convert validator report records
        let validator_report_records =
            convert_report_records(proto_state.validator_report_records)?;

        // Convert encoder report records
        let encoder_report_records = convert_report_records(proto_state.encoder_report_records)?;

        // Convert shard results
        let shard_results = convert_shard_results(proto_state.shard_results)?;

        // Build initial committees
        let mut system_state = types::system_state::SystemState {
            epoch,
            epoch_start_timestamp_ms,
            parameters,
            validators,
            encoders,
            validator_report_records,
            encoder_report_records,
            stake_subsidy,
            shard_results,
            committees: [None, None], // Will be initialized below
        };

        // Initialize current epoch committees
        let current_committees = system_state.build_committees_for_epoch(epoch);
        system_state.committees[1] = Some(current_committees);

        Ok(system_state)
    }
}

impl TryFrom<SystemParameters> for types::system_state::SystemParameters {
    type Error = String;

    fn try_from(proto_params: SystemParameters) -> Result<Self, Self::Error> {
        Ok(types::system_state::SystemParameters {
            epoch_duration_ms: proto_params
                .epoch_duration_ms
                .ok_or("Missing epoch_duration_ms")?,
            vdf_iterations: proto_params
                .vdf_iterations
                .ok_or("Missing vdf_iterations")?,
        })
    }
}

impl TryFrom<StakeSubsidy> for types::system_state::subsidy::StakeSubsidy {
    type Error = String;

    fn try_from(proto_subsidy: StakeSubsidy) -> Result<Self, Self::Error> {
        Ok(types::system_state::subsidy::StakeSubsidy {
            balance: proto_subsidy.balance.ok_or("Missing balance")?,
            distribution_counter: proto_subsidy
                .distribution_counter
                .ok_or("Missing distribution_counter")?,
            current_distribution_amount: proto_subsidy
                .current_distribution_amount
                .ok_or("Missing current_distribution_amount")?,
            period_length: proto_subsidy.period_length.ok_or("Missing period_length")?,
            decrease_rate: proto_subsidy.decrease_rate.ok_or("Missing decrease_rate")? as u16,
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
            reference_byte_price: proto_set
                .reference_byte_price
                .ok_or("Missing reference_byte_price")?,
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

        let metadata = types::system_state::encoder::EncoderMetadata {
            soma_address,
            encoder_pubkey,
            network_pubkey,
            internal_network_address,
            external_network_address,
            object_server_address,
            next_epoch_network_pubkey,
            next_epoch_internal_network_address,
            next_epoch_external_network_address,
            next_epoch_object_server_address,
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

impl TryFrom<ShardResult> for types::system_state::shard::ShardResult {
    type Error = String;

    fn try_from(proto_shard: ShardResult) -> Result<Self, Self::Error> {
        // Convert proto Metadata to domain Metadata
        let metadata = proto_shard
            .metadata
            .as_ref()
            .ok_or("Missing metadata")?
            .try_into()
            .map_err(|e: TryFromProtoError| format!("Invalid metadata: {}", e))?;

        let report_bytes = proto_shard.report.ok_or("Missing report")?.to_vec();

        // Deserialize the report from bytes
        let report = bcs::from_bytes(&report_bytes)
            .map_err(|e| format!("Failed to deserialize report: {}", e))?;

        Ok(types::system_state::shard::ShardResult {
            metadata,
            amount: proto_shard.amount.ok_or("Missing amount")?,
            report,
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

fn convert_shard_results(
    proto_results: BTreeMap<String, ShardResult>,
) -> Result<
    BTreeMap<
        types::shard_crypto::digest::Digest<types::shard::Shard>,
        types::system_state::shard::ShardResult,
    >,
    String,
> {
    proto_results
        .into_iter()
        .map(|(k, v)| {
            // The key is likely base64 encoded (based on the Display impl of Digest)
            let digest_bytes =
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &k)
                    .map_err(|e| format!("Failed to decode base64 digest key: {}", e))?;

            // Convert bytes to Digest<Shard> using TryFrom
            let digest: types::shard_crypto::digest::Digest<types::shard::Shard> =
                types::shard_crypto::digest::Digest::try_from(digest_bytes)
                    .map_err(|e| format!("Invalid digest key: {:?}", e))?;

            let result = v.try_into()?;
            Ok((digest, result))
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

        // Convert shard results
        let shard_results = convert_shard_results_to_proto(domain_state.shard_results)?;

        Ok(SystemState {
            epoch: Some(domain_state.epoch),
            epoch_start_timestamp_ms: Some(domain_state.epoch_start_timestamp_ms),
            parameters: Some(domain_state.parameters.try_into()?),
            validators: Some(domain_state.validators.try_into()?),
            encoders: Some(domain_state.encoders.try_into()?),
            validator_report_records,
            encoder_report_records,
            stake_subsidy: Some(domain_state.stake_subsidy.try_into()?),
            shard_results,
        })
    }
}

impl TryFrom<types::system_state::SystemParameters> for SystemParameters {
    type Error = String;

    fn try_from(domain_params: types::system_state::SystemParameters) -> Result<Self, Self::Error> {
        Ok(SystemParameters {
            epoch_duration_ms: Some(domain_params.epoch_duration_ms),
            vdf_iterations: Some(domain_params.vdf_iterations),
        })
    }
}

impl TryFrom<types::system_state::subsidy::StakeSubsidy> for StakeSubsidy {
    type Error = String;

    fn try_from(
        domain_subsidy: types::system_state::subsidy::StakeSubsidy,
    ) -> Result<Self, Self::Error> {
        Ok(StakeSubsidy {
            balance: Some(domain_subsidy.balance),
            distribution_counter: Some(domain_subsidy.distribution_counter),
            current_distribution_amount: Some(domain_subsidy.current_distribution_amount),
            period_length: Some(domain_subsidy.period_length),
            decrease_rate: Some(domain_subsidy.decrease_rate as u32),
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
            reference_byte_price: Some(domain_set.reference_byte_price),
        })
    }
}

impl TryFrom<types::system_state::encoder::Encoder> for Encoder {
    type Error = String;

    fn try_from(domain_enc: types::system_state::encoder::Encoder) -> Result<Self, Self::Error> {
        let metadata = domain_enc.metadata;

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
            staking_pool: Some(domain_enc.staking_pool.try_into()?),
            next_epoch_network_pubkey,
            next_epoch_internal_network_address,
            next_epoch_external_network_address,
            next_epoch_object_server_address,
        })
    }
}

impl TryFrom<types::system_state::shard::ShardResult> for ShardResult {
    type Error = String;

    fn try_from(
        domain_shard: types::system_state::shard::ShardResult,
    ) -> Result<Self, Self::Error> {
        use bytes::Bytes;

        // Convert domain Metadata to proto Metadata
        let metadata: crate::proto::soma::Metadata = domain_shard.metadata.into();

        // Serialize the report to bytes
        let report_bytes = bcs::to_bytes(&domain_shard.report)
            .map_err(|e| format!("Failed to serialize report: {}", e))?;

        Ok(ShardResult {
            metadata: Some(metadata),
            amount: Some(domain_shard.amount),
            report: Some(Bytes::from(report_bytes)),
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

fn convert_shard_results_to_proto(
    domain_results: BTreeMap<
        types::shard_crypto::digest::Digest<types::shard::Shard>,
        types::system_state::shard::ShardResult,
    >,
) -> Result<BTreeMap<String, ShardResult>, String> {
    domain_results
        .into_iter()
        .map(|(digest, result)| {
            // Convert Digest to base64 string (matching the Display impl)
            let bytes: &[u8] = digest.as_ref();
            let key = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes);
            let proto_result = result.try_into()?;
            Ok((key, proto_result))
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
            // epoch_rolling_gas_cost_summary,
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

        // if mask.contains(Self::EPOCH_ROLLING_GAS_COST_SUMMARY_FIELD) {
        //     self.epoch_rolling_gas_cost_summary = Some(epoch_rolling_gas_cost_summary.into());
        // }

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
    fn from(
        types::checkpoints::EndOfEpochData {
            next_epoch_validator_committee,
            // next_epoch_protocol_version,
            epoch_commitments,
            ..
        }: types::checkpoints::EndOfEpochData,
    ) -> Self {
        let mut message = Self::default();

        message.next_epoch_validator_committee = Some(next_epoch_validator_committee.into());
        // message.next_epoch_protocol_version = Some(next_epoch_protocol_version.as_u64());
        message.epoch_commitments = epoch_commitments.into_iter().map(Into::into).collect();

        message
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
