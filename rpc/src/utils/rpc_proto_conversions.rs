// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use fastcrypto::bls12381::min_sig::BLS12381PublicKey;
use fastcrypto::traits::ToFromBytes;
use types::base::SomaAddress;
use types::crypto::SomaSignature;
use types::envelope::Message as _;
use types::metadata::{ManifestAPI as _, MetadataAPI as _};
use url::Url;

use crate::proto::TryFromProtoError;
use crate::proto::soma::*;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;
use crate::utils::types_conversions::SdkTypeConversionError;

//
// TransactionFee
//

impl From<types::tx_fee::TransactionFee> for TransactionFee {
    fn from(types::tx_fee::TransactionFee { total_fee }: types::tx_fee::TransactionFee) -> Self {
        Self { total_fee: Some(total_fee) }
    }
}

impl From<types::effects::ExecutionStatus> for ExecutionStatus {
    fn from(value: types::effects::ExecutionStatus) -> Self {
        match value {
            types::effects::ExecutionStatus::Success => {
                Self { success: Some(true), ..Default::default() }
            }
            types::effects::ExecutionStatus::Failure { error } => {
                let description = format!("{error:?}");
                let mut error_message = ExecutionError::from(error);
                error_message.description = Some(description);

                Self { success: Some(false), error: Some(error_message), ..Default::default() }
            }
        }
    }
}

impl From<types::effects::ExecutionFailureStatus> for ExecutionError {
    fn from(value: types::effects::ExecutionFailureStatus) -> Self {
        use execution_error::{ErrorDetails, ExecutionErrorKind};
        use types::effects::ExecutionFailureStatus as E;

        let mut message = Self::default();

        let (kind, details) = match value {
            E::InsufficientGas => (ExecutionErrorKind::InsufficientGas, None),
            E::InvalidGasCoinType { object_id } => {
                (ExecutionErrorKind::InvalidGasCoinType, Some(object_id.to_hex()))
            }
            E::InvalidOwnership { object_id, .. } => {
                (ExecutionErrorKind::InvalidOwnership, Some(object_id.to_hex()))
            }
            E::ObjectNotFound { object_id } => {
                (ExecutionErrorKind::ObjectNotFound, Some(object_id.to_hex()))
            }
            E::InvalidObjectType { object_id, .. } => {
                (ExecutionErrorKind::InvalidObjectType, Some(object_id.to_hex()))
            }
            E::InvalidTransactionType => (ExecutionErrorKind::InvalidTransactionType, None),
            E::InvalidArguments { reason } => (ExecutionErrorKind::InvalidArguments, Some(reason)),
            E::DuplicateValidator => (ExecutionErrorKind::DuplicateValidator, None),
            E::DuplicateValidatorMetadata { field } => {
                (ExecutionErrorKind::DuplicateValidatorMetadata, Some(field))
            }
            E::MissingProofOfPossession => (ExecutionErrorKind::MissingProofOfPossession, None),
            E::InvalidProofOfPossession { reason } => {
                (ExecutionErrorKind::InvalidProofOfPossession, Some(reason))
            }
            E::NotAValidator => (ExecutionErrorKind::NotAValidator, None),
            E::ValidatorAlreadyRemoved => (ExecutionErrorKind::ValidatorAlreadyRemoved, None),
            E::AdvancedToWrongEpoch => (ExecutionErrorKind::AdvancedToWrongEpoch, None),
            E::ModelNotFound => (ExecutionErrorKind::ModelNotFound, None),
            E::NotModelOwner => (ExecutionErrorKind::NotModelOwner, None),
            E::ModelNotActive => (ExecutionErrorKind::ModelNotActive, None),
            E::ModelNotPending => (ExecutionErrorKind::ModelNotPending, None),
            E::ModelAlreadyInactive => (ExecutionErrorKind::ModelAlreadyInactive, None),
            E::ModelRevealEpochMismatch => (ExecutionErrorKind::ModelRevealEpochMismatch, None),
            E::ModelEmbeddingCommitmentMismatch => {
                (ExecutionErrorKind::ModelEmbeddingCommitmentMismatch, None)
            }
            E::ModelDecryptionKeyCommitmentMismatch => {
                (ExecutionErrorKind::ModelDecryptionKeyCommitmentMismatch, None)
            }
            E::ModelNoPendingUpdate => (ExecutionErrorKind::ModelNoPendingUpdate, None),
            E::ModelArchitectureVersionMismatch => {
                (ExecutionErrorKind::ModelArchitectureVersionMismatch, None)
            }
            E::ModelCommissionRateTooHigh => (ExecutionErrorKind::ModelCommissionRateTooHigh, None),
            E::ModelMinStakeNotMet => (ExecutionErrorKind::ModelMinStakeNotMet, None),
            E::InsufficientCoinBalance => (ExecutionErrorKind::InsufficientCoinBalance, None),
            E::CoinBalanceOverflow => (ExecutionErrorKind::CoinBalanceOverflow, None),
            E::ValidatorNotFound => (ExecutionErrorKind::ValidatorNotFound, None),
            E::StakingPoolNotFound => (ExecutionErrorKind::StakingPoolNotFound, None),
            E::CannotReportOneself => (ExecutionErrorKind::CannotReportOneself, None),
            E::ReportRecordNotFound => (ExecutionErrorKind::ReportRecordNotFound, None),
            E::InputObjectDeleted => (ExecutionErrorKind::InputObjectDeleted, None),
            E::CertificateDenied => (ExecutionErrorKind::CertificateDenied, None),
            E::ExecutionCancelledDueToSharedObjectCongestion => {
                (ExecutionErrorKind::SharedObjectCongestion, None)
            }
            // Target errors
            E::NoActiveModels => (ExecutionErrorKind::NoActiveModels, None),
            E::TargetNotFound => (ExecutionErrorKind::TargetNotFound, None),
            E::TargetNotOpen => (ExecutionErrorKind::TargetNotOpen, None),
            E::TargetExpired { generation_epoch, current_epoch } => (
                ExecutionErrorKind::TargetExpired,
                Some(format!(
                    "generation_epoch={}, current_epoch={}",
                    generation_epoch, current_epoch
                )),
            ),
            E::TargetNotFilled => (ExecutionErrorKind::TargetNotFilled, None),
            E::AuditWindowOpen { fill_epoch, current_epoch } => (
                ExecutionErrorKind::ChallengeWindowOpen,
                Some(format!("fill_epoch={}, current_epoch={}", fill_epoch, current_epoch)),
            ),
            E::TargetAlreadyClaimed => (ExecutionErrorKind::TargetAlreadyClaimed, None),
            // Submission errors
            E::ModelNotInTarget { model_id, target_id } => (
                ExecutionErrorKind::ModelNotInTarget,
                Some(format!("model_id={}, target_id={}", model_id, target_id)),
            ),
            E::EmbeddingDimensionMismatch { expected, actual } => (
                ExecutionErrorKind::EmbeddingDimensionMismatch,
                Some(format!("expected={}, actual={}", expected, actual)),
            ),
            E::InsufficientBond { required, provided } => (
                ExecutionErrorKind::InsufficientBond,
                Some(format!("required={}, provided={}", required, provided)),
            ),
            E::InsufficientEmissionBalance => {
                (ExecutionErrorKind::InsufficientEmissionBalance, None)
            }
            // Audit errors
            E::AuditWindowClosed { fill_epoch, current_epoch } => (
                ExecutionErrorKind::ChallengeWindowClosed,
                Some(format!("fill_epoch={}, current_epoch={}", fill_epoch, current_epoch)),
            ),
            E::DataExceedsMaxSize { size, max_size } => (
                ExecutionErrorKind::DataExceedsMaxSize,
                Some(format!("size={}, max_size={}", size, max_size)),
            ),
            E::ArithmeticOverflow => {
                (ExecutionErrorKind::OtherError, Some("Arithmetic overflow in execution".into()))
            }
            E::ModelNotCreated => (ExecutionErrorKind::ModelNotFound, None),
            E::ModelInvalidState => (
                ExecutionErrorKind::OtherError,
                Some("Model in invalid state for this operation".into()),
            ),
            E::SomaError(e) => (ExecutionErrorKind::OtherError, Some(e.to_string())),
            // Marketplace errors (legacy variants still in types crate)
            E::AskNotFound
            | E::AskNotOpen
            | E::AskExpired
            | E::AskAlreadyFilled
            | E::AskHasAcceptedBids
            | E::BidNotFound
            | E::BidNotPending
            | E::BidPriceTooHigh
            | E::SellerCannotBidOnOwnAsk
            | E::SettlementNotFound
            | E::SettlementAlreadyRatedNegative
            | E::RatingDeadlinePassed
            | E::VaultNotFound
            | E::InsufficientVaultBalance
            | E::WrongCoinTypeForPayment => {
                (ExecutionErrorKind::OtherError, Some(format!("{:?}", value)))
            }
            // Bridge errors
            E::BridgePaused => (ExecutionErrorKind::OtherError, Some("Bridge is paused".into())),
            E::BridgeNonceAlreadyProcessed => {
                (ExecutionErrorKind::OtherError, Some("Bridge nonce already processed".into()))
            }
            E::BridgeInsufficientSignatureStake => (
                ExecutionErrorKind::OtherError,
                Some("Bridge: insufficient signature stake".into()),
            ),
            E::BridgeSystemMessageSeqMismatch { expected, actual } => (
                ExecutionErrorKind::OtherError,
                Some(format!(
                    "Bridge: system-message seq mismatch (expected {expected}, got {actual})"
                )),
            ),
            E::BridgeAlreadyPaused => {
                (ExecutionErrorKind::OtherError, Some("Bridge already paused".into()))
            }
            E::BridgeNotPaused => {
                (ExecutionErrorKind::OtherError, Some("Bridge not paused".into()))
            }
            E::BridgeAmountZero => {
                (ExecutionErrorKind::OtherError, Some("Bridge amount must be non-zero".into()))
            }
            E::BridgeSupplyUnderflow => {
                (ExecutionErrorKind::OtherError, Some("Bridge USDC supply underflow".into()))
            }
            E::BridgeBlocklistPayloadTooLarge { got, max } => (
                ExecutionErrorKind::OtherError,
                Some(format!("Bridge blocklist payload too large ({got}/{max})")),
            ),
            E::BridgeUrlTooLong { got, max } => (
                ExecutionErrorKind::OtherError,
                Some(format!("Bridge http_url too long ({got}/{max})")),
            ),

            // Payment-channel errors
            E::ChannelCallerNotPayee { expected, actual } => (
                ExecutionErrorKind::ChannelCallerNotPayee,
                Some(format!("expected={expected}, actual={actual}")),
            ),
            E::ChannelCallerNotPayer { expected, actual } => (
                ExecutionErrorKind::ChannelCallerNotPayer,
                Some(format!("expected={expected}, actual={actual}")),
            ),
            E::ChannelVoucherNotMonotonic { cumulative, settled } => (
                ExecutionErrorKind::ChannelVoucherNotMonotonic,
                Some(format!("cumulative={cumulative}, settled={settled}")),
            ),
            E::ChannelOverspend { cumulative, available } => (
                ExecutionErrorKind::ChannelOverspend,
                Some(format!("cumulative={cumulative}, available={available}")),
            ),
            E::ChannelGraceNotElapsed { now_ms, earliest_ms } => (
                ExecutionErrorKind::ChannelGraceNotElapsed,
                Some(format!("now_ms={now_ms}, earliest_ms={earliest_ms}")),
            ),
            E::ChannelCloseAlreadyPending => (ExecutionErrorKind::ChannelCloseAlreadyPending, None),
            E::ChannelNoCloseRequest => (ExecutionErrorKind::ChannelNoCloseRequest, None),
            E::ChannelInvalidVoucherSignature { reason } => {
                (ExecutionErrorKind::ChannelInvalidVoucherSignature, Some(reason))
            }
            E::ChannelAmountZero => (ExecutionErrorKind::ChannelAmountZero, None),
            E::ChannelInvalidInput { reason } => {
                (ExecutionErrorKind::ChannelInvalidInput, Some(reason))
            }
            E::ChannelCoinTypeMismatch => (ExecutionErrorKind::ChannelCoinTypeMismatch, None),
            E::NotAChannel { object_id } => {
                (ExecutionErrorKind::NotAChannel, Some(object_id.to_hex()))
            }
            E::ChannelClockMissing => (ExecutionErrorKind::ChannelClockMissing, None),

            // Provider registry errors
            E::ProviderAlreadyExists => (ExecutionErrorKind::ProviderAlreadyExists, None),
            E::ProviderNotFound => (ExecutionErrorKind::ProviderNotFound, None),
            E::ProviderCallerMismatch => (ExecutionErrorKind::ProviderCallerMismatch, None),
            E::ProviderInvalidEndpoint { reason } => {
                (ExecutionErrorKind::ProviderInvalidEndpoint, Some(reason))
            }
            E::ProviderClockMissing => (ExecutionErrorKind::ProviderClockMissing, None),
            E::ChannelTooManyOpenForPair { current, max } => (
                ExecutionErrorKind::ChannelTooManyOpenForPair,
                Some(format!("current={}, max={}", current, max)),
            ),
            E::ChannelInboxPayeeMismatch { declared, actual } => (
                ExecutionErrorKind::ChannelInboxPayeeMismatch,
                Some(format!("declared={}, actual={}", declared, actual)),
            ),
            E::NotAProviderInbox { object_id } => {
                (ExecutionErrorKind::NotAProviderInbox, Some(object_id.to_hex()))
            }

            // Offering errors: dedicated kinds. Detail strings preserve
            // any inline fields so the client can surface useful messages.
            E::OfferingAlreadyExists => (ExecutionErrorKind::OfferingAlreadyExists, None),
            E::OfferingNotFound => (ExecutionErrorKind::OfferingNotFound, None),
            E::OfferingCallerMismatch => (ExecutionErrorKind::OfferingCallerMismatch, None),
            E::OfferingUnknownModel { model_id } => {
                (ExecutionErrorKind::OfferingUnknownModel, Some(format!("model_id={}", model_id)))
            }
            E::ChannelOfferingMissing { payee, model_id } => (
                ExecutionErrorKind::ChannelOfferingMissing,
                Some(format!("payee={}, model_id={}", payee, model_id)),
            ),
        };

        message.set_kind(kind);
        if let Some(detail_str) = details {
            message.error_details = Some(execution_error::ErrorDetails::OtherError(detail_str));
        }
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
        Self {
            epoch: Some(value.epoch),
            signature: Some(value.signature.as_ref().to_vec().into()),
            bitmap: value.signers_map.iter().collect(),
            ..Default::default()
        }
    }
}

//
// ValidatorCommittee
//

impl From<types::committee::Committee> for ValidatorCommittee {
    fn from(value: types::committee::Committee) -> Self {
        let authorities: Vec<_> = value.authorities().collect();

        let members = authorities
            .into_iter()
            .map(|(i, authority)| {
                let network_key = authority.network_key.clone();
                let authority_key_bytes = authority.authority_key.as_bytes().to_vec();
                let protocol_key_bytes = authority.protocol_key.to_bytes().to_vec();
                let network_key_bytes = network_key.into_inner().as_bytes().to_vec();

                ValidatorCommitteeMember {
                    authority_key: Some(authority_key_bytes.into()),
                    weight: Some(authority.stake),
                    network_metadata: Some(ValidatorNetworkMetadata {
                        consensus_address: Some(authority.address.to_string()),
                        hostname: Some(authority.hostname.clone()),
                        protocol_key: Some(protocol_key_bytes.into()),
                        network_key: Some(network_key_bytes.into()),
                    }),
                    ..Default::default()
                }
            })
            .collect();

        Self { epoch: Some(value.epoch), members, ..Default::default() }
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

        Self {
            scheme: Some(scheme.into()),
            signature: Some(signature.to_vec().into()),
            public_key: Some(public_key.to_vec().into()),
            ..Default::default()
        }
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
        let members = value
            .pubkeys()
            .iter()
            .map(|(pk, weight)| MultisigMember {
                public_key: Some(pk.into()),
                weight: Some((*weight).into()),
                ..Default::default()
            })
            .collect();

        Self { members, threshold: Some((*value.threshold()).into()), ..Default::default() }
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
        Self {
            signatures: value.get_sigs().iter().map(Into::into).collect(),
            bitmap: Some(value.get_bitmap().into()),
            committee: Some(value.get_pk().into()),
            ..Default::default()
        }
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
        Self {
            address: Some(value.address.to_string()),
            amount: Some(value.amount.to_string()),
            coin_type: Some(value.coin_type.to_string()),
            ..Default::default()
        }
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

        let coin_type = value
            .coin_type
            .as_deref()
            .ok_or_else(|| TryFromProtoError::missing("coin_type"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid("coin_type", e))?;

        Ok(types::balance_change::BalanceChange { address, coin_type, amount })
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

/// Stable string label for a CoinType — round-trips with
/// `parse_coin_type` in `types_conversions.rs`.
/// Convert a typed bridge cert envelope into the flat proto repeated list.
fn envelope_to_proto(
    sigs: std::collections::BTreeMap<types::bridge::BridgePubkey, types::bridge::BridgeSignature>,
) -> Vec<PubkeySig> {
    sigs.into_iter()
        .map(|(pk, sig)| PubkeySig {
            signer_pubkey: Some(pk.as_bytes().to_vec().into()),
            signature: Some(sig.as_bytes().to_vec().into()),
        })
        .collect()
}

fn coin_type_label(t: types::object::CoinType) -> &'static str {
    match t {
        types::object::CoinType::Soma => "SOMA",
        types::object::CoinType::Usdc => "USDC",
    }
}

fn object_ref_to_proto(value: types::object::ObjectRef) -> ObjectReference {
    let (object_id, version, digest) = value;
    ObjectReference {
        object_id: Some(object_id.to_hex()),
        version: Some(version.value()),
        digest: Some(digest.to_string()),
        ..Default::default()
    }
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
            O::Shared { initial_shared_version } => {
                message.version = Some(initial_shared_version.value());
                OwnerKind::Shared
            }
            O::Immutable => OwnerKind::Immutable,
            O::Accumulator { kind: acc_kind } => {
                // Stage 14a: serialize the accumulator family as a
                // string so adding new families later is non-breaking.
                use types::object::AccumulatorKind;
                message.accumulator_kind = Some(
                    match acc_kind {
                        AccumulatorKind::Balance => "BALANCE",
                        AccumulatorKind::Delegation => "DELEGATION",
                    }
                    .to_string(),
                );
                OwnerKind::Accumulator
            }
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
            self.kind = Some(source.kind().clone().into());
        }

        if mask.contains(Self::SENDER_FIELD.name) {
            self.sender = Some(source.sender().to_string());
        }

        if mask.contains(Self::GAS_PAYMENT_FIELD.name) {
            self.gas_payment = source.gas().into_iter().map(object_ref_to_proto).collect();
        }

        // Stage 5.5/6c: expiration is part of the signed BCS payload —
        // dropping it on the wire breaks signature verification for
        // any tx with non-default expiration (e.g., balance-mode gas
        // txs that declare ValidDuring).
        self.expiration = Some(transaction_expiration_to_proto(source.expiration()));
    }
}

fn transaction_expiration_to_proto(
    src: &types::transaction::TransactionExpiration,
) -> crate::proto::soma::TransactionExpiration {
    use crate::proto::soma::transaction_expiration::Value;
    let value = match src {
        types::transaction::TransactionExpiration::None => Value::None(()),
        types::transaction::TransactionExpiration::ValidDuring {
            min_epoch,
            max_epoch,
            chain,
            nonce,
        } => Value::ValidDuring(crate::proto::soma::ValidDuring {
            min_epoch: *min_epoch,
            max_epoch: *max_epoch,
            chain: Some(bcs::to_bytes(chain).expect("ChainIdentifier serialize").into()),
            nonce: Some(*nonce),
        }),
    };
    crate::proto::soma::TransactionExpiration { value: Some(value) }
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
            K::ConsensusCommitPrologueV1(prologue) => {
                Kind::ConsensusCommitPrologue(prologue.into())
            }
            K::ChangeEpoch(change_epoch) => Kind::ChangeEpoch(change_epoch.into()),
            K::AddValidator(args) => Kind::AddValidator(args.into()),
            K::RemoveValidator(args) => Kind::RemoveValidator(args.into()),
            K::ReportValidator { reportee } => {
                Kind::ReportValidator(ReportValidator { reportee: Some(reportee.to_string()) })
            }
            K::UndoReportValidator { reportee } => Kind::UndoReportValidator(UndoReportValidator {
                reportee: Some(reportee.to_string()),
            }),
            K::UpdateValidatorMetadata(args) => Kind::UpdateValidatorMetadata(args.into()),
            K::SetCommissionRate { new_rate } => {
                Kind::SetCommissionRate(SetCommissionRate { new_rate: Some(new_rate) })
            }

            // Stage 13b: K::Transfer / K::MergeCoins deleted at the
            // domain layer.
            K::TransferObjects { objects, recipient } => Kind::TransferObjects(TransferObjects {
                objects: objects.into_iter().map(object_ref_to_proto).collect(),
                recipient: Some(recipient.to_string()),
            }),
            K::AddStake { validator, amount } => Kind::AddStake(AddStake {
                validator: Some(validator.to_string()),
                amount: Some(amount),
            }),

            K::WithdrawStake { pool_id, amount } => {
                Kind::WithdrawStake(WithdrawStake { pool_id: Some(pool_id.to_string()), amount })
            }

            // Bridge transactions
            K::BridgeDeposit(args) => Kind::BridgeDeposit(BridgeDeposit {
                nonce: Some(args.nonce),
                eth_tx_hash: Some(args.eth_tx_hash.to_vec().into()),
                recipient: Some(args.recipient.to_string()),
                amount: Some(args.amount),
                timestamp_ms: Some(args.timestamp_ms),
                sender_eth_address: Some(args.sender_eth_address.to_vec().into()),
                target_chain: Some(args.target_chain.as_u8() as u32),
                token_type: Some(args.token_type as u32),
                signatures: envelope_to_proto(args.signatures),
            }),
            K::BridgeWithdraw(args) => Kind::BridgeWithdraw(BridgeWithdraw {
                amount: Some(args.amount),
                recipient_eth_address: Some(args.recipient_eth_address.to_vec().into()),
                target_chain: Some(args.target_chain.as_u8() as u32),
            }),
            K::BridgeEmergencyPause(args) => Kind::BridgeEmergencyPause(BridgeEmergencyPause {
                nonce: Some(args.nonce),
                signatures: envelope_to_proto(args.signatures),
            }),
            K::BridgeEmergencyUnpause(args) => {
                Kind::BridgeEmergencyUnpause(BridgeEmergencyUnpause {
                    nonce: Some(args.nonce),
                    signatures: envelope_to_proto(args.signatures),
                })
            }
            K::BridgeAttachWithdrawalSignatures(args) => {
                Kind::BridgeAttachWithdrawalSignatures(BridgeAttachWithdrawalSignatures {
                    nonce: Some(args.nonce),
                    signatures: envelope_to_proto(args.signatures),
                })
            }
            K::BridgeUpdateCommitteeBlocklist(args) => {
                let mut flat = Vec::with_capacity(args.eth_addresses.len() * 20);
                for a in &args.eth_addresses {
                    flat.extend_from_slice(a);
                }
                Kind::BridgeUpdateCommitteeBlocklist(BridgeUpdateCommitteeBlocklist {
                    nonce: Some(args.nonce),
                    is_blocklist: Some(args.is_blocklist),
                    eth_addresses: Some(flat.into()),
                    signatures: envelope_to_proto(args.signatures),
                })
            }
            K::BridgeRegisterBridgeKey(args) => {
                Kind::BridgeRegisterBridgeKey(BridgeRegisterBridgeKey {
                    bridge_pubkey: Some(args.bridge_pubkey.as_bytes().to_vec().into()),
                    http_url: Some(args.http_url),
                })
            }

            // Payment-channel tx kinds.
            K::OpenChannel(args) => Kind::OpenChannel(OpenChannel {
                payee: Some(args.payee.to_string()),
                authorized_signer: Some(args.authorized_signer.to_string()),
                token: Some(coin_type_label(args.token).to_string()),
                deposit_amount: Some(args.deposit_amount),
                model_id: Some(args.model_id),
            }),
            K::Settle(args) => Kind::Settle(Settle {
                channel_id: Some(args.channel_id.to_string()),
                cumulative_amount: Some(args.cumulative_amount),
                cumulative_prompt_tokens: Some(args.cumulative_prompt_tokens),
                cumulative_completion_tokens: Some(args.cumulative_completion_tokens),
                cumulative_cache_read_tokens: Some(args.cumulative_cache_read_tokens),
                cumulative_cache_write_tokens: Some(args.cumulative_cache_write_tokens),
                cumulative_requests: Some(args.cumulative_requests),
                voucher_signature: Some(args.voucher_signature.as_ref().to_vec().into()),
            }),
            K::RequestClose(args) => {
                Kind::RequestClose(RequestClose { channel_id: Some(args.channel_id.to_string()) })
            }
            K::WithdrawAfterTimeout(args) => Kind::WithdrawAfterTimeout(WithdrawAfterTimeout {
                channel_id: Some(args.channel_id.to_string()),
                payee: Some(args.payee.to_string()),
            }),
            K::TopUp(args) => Kind::TopUp(TopUp {
                channel_id: Some(args.channel_id.to_string()),
                coin_type: Some(coin_type_label(args.coin_type).to_string()),
                amount: Some(args.amount),
            }),
            K::RateChannel(args) => Kind::RateChannel(RateChannel {
                channel_id: Some(args.channel_id.to_string()),
                negative: Some(args.negative),
                reason_code: Some(args.reason_code as u32),
            }),

            // Provider registry tx kinds.
            K::RegisterProvider(args) => {
                Kind::RegisterProvider(RegisterProvider { endpoint: Some(args.endpoint) })
            }
            K::UpdateProvider(args) => Kind::UpdateProvider(UpdateProvider {
                provider_id: Some(args.provider_id.to_string()),
                endpoint: Some(args.endpoint),
            }),

            // Per-(provider, model) offering tx kinds.
            K::RegisterOffering(args) => Kind::RegisterOffering(RegisterOffering {
                model_id: Some(args.model_id),
                prompt_micros_per_1k: Some(args.prompt_micros_per_1k),
                completion_micros_per_1k: Some(args.completion_micros_per_1k),
                cache_read_micros_per_1k: Some(args.cache_read_micros_per_1k),
                cache_write_micros_per_1k: Some(args.cache_write_micros_per_1k),
                request_micros: Some(args.request_micros),
                ttft_bound_ms: Some(args.ttft_bound_ms),
                ttot_bound_ms: Some(args.ttot_bound_ms),
            }),
            K::UpdateOffering(args) => Kind::UpdateOffering(UpdateOffering {
                offering_id: Some(args.offering_id.to_string()),
                model_id: Some(args.model_id),
                prompt_micros_per_1k: Some(args.prompt_micros_per_1k),
                completion_micros_per_1k: Some(args.completion_micros_per_1k),
                cache_read_micros_per_1k: Some(args.cache_read_micros_per_1k),
                cache_write_micros_per_1k: Some(args.cache_write_micros_per_1k),
                request_micros: Some(args.request_micros),
                ttft_bound_ms: Some(args.ttft_bound_ms),
                ttot_bound_ms: Some(args.ttot_bound_ms),
            }),
            K::DeactivateOffering(args) => Kind::DeactivateOffering(DeactivateOffering {
                offering_id: Some(args.offering_id.to_string()),
                model_id: Some(args.model_id),
            }),

            K::BalanceTransfer(args) => Kind::BalanceTransfer(BalanceTransfer {
                coin_type: Some(coin_type_label(args.coin_type).to_string()),
                transfers: args
                    .transfers
                    .into_iter()
                    .map(|(recipient, amount)| BalanceTransferEntry {
                        recipient: Some(recipient.to_string()),
                        amount: Some(amount),
                    })
                    .collect(),
            }),

            K::Settlement(settlement) => Kind::Settlement(Settlement {
                epoch: Some(settlement.epoch),
                round: Some(settlement.round),
                sub_dag_index: settlement.sub_dag_index,
                changes: settlement
                    .changes
                    .into_iter()
                    .map(|ev| {
                        let (owner, coin_type, amount, is_credit) = match ev {
                            types::balance::BalanceEvent::Deposit { owner, coin_type, amount } => {
                                (owner, coin_type, amount, true)
                            }
                            types::balance::BalanceEvent::Withdraw { owner, coin_type, amount } => {
                                (owner, coin_type, amount, false)
                            }
                        };
                        SettlementChange {
                            owner: Some(owner.to_string()),
                            coin_type: Some(coin_type_label(coin_type).to_string()),
                            amount: Some(amount),
                            is_credit: Some(is_credit),
                        }
                    })
                    .collect(),
            }),
        };

        Self { kind: Some(kind), ..Default::default() }
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
            proof_of_possession: Some(args.proof_of_possession.into()),
        }
    }
}

impl From<types::transaction::RemoveValidatorArgs> for RemoveValidator {
    fn from(args: types::transaction::RemoveValidatorArgs) -> Self {
        Self { pubkey_bytes: Some(args.pubkey_bytes.into()) }
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
            next_epoch_proof_of_possession: args
                .next_epoch_proof_of_possession
                .map(|bytes| bytes.into()),
        }
    }
}

//
// ConsensusCommitPrologue
//

impl From<types::consensus::ConsensusCommitPrologueV1> for ConsensusCommitPrologue {
    fn from(value: types::consensus::ConsensusCommitPrologueV1) -> Self {
        Self {
            epoch: Some(value.epoch),
            round: Some(value.round),
            commit_timestamp: Some(crate::proto::timestamp_ms_to_proto(value.commit_timestamp_ms)),
            consensus_commit_digest: Some(value.consensus_commit_digest.to_string()),
            additional_state_digest: Some(value.additional_state_digest.to_string()),
            sub_dag_index: value.sub_dag_index,
        }
    }
}

//
// GenesisTransaction
//

impl From<types::transaction::GenesisTransaction> for GenesisTransaction {
    fn from(value: types::transaction::GenesisTransaction) -> Self {
        Self { objects: value.objects.into_iter().map(Into::into).collect() }
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
// TransactionEffects
//

impl From<types::effects::TransactionEffects> for TransactionEffects {
    fn from(value: types::effects::TransactionEffects) -> Self {
        Self::merge_from(&value, &FieldMaskTree::new_wildcard())
    }
}
impl Merge<&types::effects::TransactionEffects> for TransactionEffects {
    fn merge(&mut self, source: &types::effects::TransactionEffects, mask: &FieldMaskTree) {
        match source {
            types::effects::TransactionEffects::V1(source) => {
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
                    self.dependencies =
                        source.dependencies.iter().map(ToString::to_string).collect();
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
    }
}

//
// ChangedObject
//

impl From<types::effects::object_change::EffectsObjectChange> for ChangedObject {
    fn from(value: types::effects::object_change::EffectsObjectChange) -> Self {
        use changed_object::{InputObjectState, OutputObjectState};
        use types::effects::object_change::{ObjectIn, ObjectOut};

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
            ObjectOut::AccumulatorWriteV1(write) => {
                use types::effects::object_change::AccumulatorOperation;
                message.accumulator_operation = Some(
                    match write.operation {
                        AccumulatorOperation::Merge => "Merge",
                        AccumulatorOperation::Split => "Split",
                    }
                    .to_string(),
                );
                message.accumulator_amount = Some(write.value.as_u64());
                OutputObjectState::AccumulatorWriteV1
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
        let protocol_version = proto_state.protocol_version.ok_or("Missing protocol_version")?;
        let epoch_start_timestamp_ms =
            proto_state.epoch_start_timestamp_ms.ok_or("Missing epoch_start_timestamp_ms")?;

        let parameters: protocol_config::SystemParameters =
            proto_state.parameters.ok_or("Missing parameters")?.try_into()?;

        let validators = proto_state.validators.ok_or("Missing validators")?.try_into()?;

        let emission_pool = proto_state.emission_pool.ok_or("Missing emission_pool")?.try_into()?;

        // Convert validator report records
        let validator_report_records =
            convert_report_records(proto_state.validator_report_records)?;

        // Build initial committees
        let system_state =
            types::system_state::SystemState::V1(types::system_state::SystemStateV1 {
                epoch,
                protocol_version,
                epoch_start_timestamp_ms,
                parameters,
                validators,

                validator_report_records,

                emission_pool,

                marketplace_params: types::bridge::MarketplaceParameters::default(),
                protocol_fund_balance: 0,
                bridge_state: proto_state
                    .bridge_state
                    .map(types::bridge::BridgeState::try_from)
                    .transpose()?
                    .unwrap_or_else(|| {
                        types::bridge::BridgeState::new(types::bridge::BridgeCommittee::empty())
                    }),

                safe_mode: proto_state.safe_mode.unwrap_or(false),
            });

        Ok(system_state)
    }
}

impl TryFrom<SystemParameters> for protocol_config::SystemParameters {
    type Error = String;

    fn try_from(proto_params: SystemParameters) -> Result<Self, Self::Error> {
        Ok(protocol_config::SystemParameters {
            epoch_duration_ms: proto_params.epoch_duration_ms.ok_or("Missing epoch_duration_ms")?,
            unit_fee: proto_params.unit_fee.ok_or("Missing unit_fee")?,
            // Default to mainnet's 10-minute grace when missing — keeps
            // older RPC clients compatible. Real on-chain SystemParameters
            // always carry a value (set in `build_system_parameters`).
            channel_grace_period_ms: proto_params.channel_grace_period_ms.unwrap_or(10 * 60 * 1000),
            // Per-(payer, payee) channel cap: not yet wired through
            // the SystemParameters proto schema; older RPC clients
            // default to the genesis value (8). Real on-chain
            // SystemParameters always carry a value (set in
            // `build_system_parameters`); the proto schema gets
            // updated in a follow-up.
            max_channels_per_pair: types::provider_inbox::DEFAULT_MAX_CHANNELS_PER_PAIR,
        })
    }
}

impl TryFrom<EmissionPool> for types::system_state::emission::EmissionPool {
    type Error = String;

    fn try_from(proto_emission_pool: EmissionPool) -> Result<Self, Self::Error> {
        Ok(types::system_state::emission::EmissionPool {
            balance: proto_emission_pool.balance.ok_or("Missing balance")?,
            distribution_counter: 0,
            current_distribution_amount: proto_emission_pool
                .emission_per_epoch
                .ok_or("Missing emission_per_epoch")?,
            period_length: 0,
            decrease_rate: 0,
        })
    }
}

impl TryFrom<ValidatorSet> for types::system_state::validator::ValidatorSet {
    type Error = String;

    fn try_from(proto_set: ValidatorSet) -> Result<Self, Self::Error> {
        let validators = proto_set
            .validators
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
            .map(|r| Ok(r as usize))
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
            validators,
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
        use std::str::FromStr;

        use fastcrypto::traits::ToFromBytes;

        let soma_address = proto_val
            .soma_address
            .ok_or("Missing soma_address")?
            .parse()
            .map_err(|_| "Invalid SomaAddress")?;

        let protocol_pubkey = proto_val.protocol_pubkey.ok_or("Missing protocol_pubkey")?.to_vec();
        let protocol_pubkey = BLS12381PublicKey::from_bytes(&protocol_pubkey)
            .map_err(|e| format!("Invalid protocol_pubkey: {}", e))?;

        let network_pubkey = proto_val.network_pubkey.ok_or("Missing network_pubkey")?.to_vec();
        let network_pubkey = fastcrypto::ed25519::Ed25519PublicKey::from_bytes(&network_pubkey)
            .map_err(|e| format!("Invalid network_pubkey: {}", e))?;
        let network_pubkey = types::crypto::NetworkPublicKey::new(network_pubkey);

        let worker_pubkey = proto_val.worker_pubkey.ok_or("Missing worker_pubkey")?.to_vec();
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

        let proof_of_possession = proto_val
            .proof_of_possession
            .map(|bytes| {
                types::crypto::AuthoritySignature::from_bytes(&bytes)
                    .map_err(|e| format!("Invalid proof_of_possession: {}", e))
            })
            .transpose()?
            .ok_or("Missing proof_of_possession")?;

        let next_epoch_proof_of_possession = proto_val
            .next_epoch_proof_of_possession
            .map(|bytes| {
                types::crypto::AuthoritySignature::from_bytes(&bytes)
                    .map_err(|e| format!("Invalid next_epoch_proof_of_possession: {}", e))
            })
            .transpose()?;

        let metadata = types::system_state::validator::ValidatorMetadata {
            soma_address,
            protocol_pubkey,
            network_pubkey,
            worker_pubkey,
            proof_of_possession,
            net_address,
            p2p_address,
            primary_address,
            next_epoch_protocol_pubkey,
            next_epoch_network_pubkey,
            next_epoch_net_address,
            next_epoch_p2p_address,
            next_epoch_primary_address,
            next_epoch_worker_pubkey,
            next_epoch_proof_of_possession,
            bridge_ecdsa_pubkey: None,
            next_epoch_bridge_ecdsa_pubkey: None,
        };

        let staking_pool = proto_val.staking_pool.ok_or("Missing staking_pool")?.try_into()?;

        Ok(types::system_state::validator::Validator {
            metadata,
            voting_power: proto_val.voting_power.ok_or("Missing voting_power")?,
            staking_pool,
            commission_rate: proto_val.commission_rate.ok_or("Missing commission_rate")?,
            next_epoch_commission_rate: proto_val
                .next_epoch_commission_rate
                .ok_or("Missing next_epoch_commission_rate")?,
        })
    }
}

impl TryFrom<StakingPool> for types::system_state::staking::StakingPool {
    type Error = String;

    fn try_from(proto_pool: StakingPool) -> Result<Self, Self::Error> {
        let id = proto_pool.id.ok_or("Missing id")?.parse().map_err(|_| "Invalid ObjectID")?;

        // Auto-compound F1: read `active_stake` out of `soma_balance`
        // (the proto schema still carries the legacy field — a
        // follow-up trims the proto). Other F1 fields default to zero
        // since the proto wire doesn't carry them; consumers
        // reconstructing here only get a dashboard-grade view.
        let _ = proto_pool.exchange_rates;
        Ok(types::system_state::staking::StakingPool {
            id,
            activation_epoch: proto_pool.activation_epoch,
            deactivation_epoch: proto_pool.deactivation_epoch,
            active_stake: proto_pool.soma_balance.ok_or("Missing soma_balance")?,
            pending_active_stake: 0,
            cumulative_index: 0,
            index_history: vec![0],
            commission_rate: 0,
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
        let types::system_state::SystemState::V1(v1) = domain_state;

        // Convert validator report records
        let validator_report_records =
            convert_report_records_to_proto(v1.validator_report_records)?;

        Ok(SystemState {
            epoch: Some(v1.epoch),
            protocol_version: Some(v1.protocol_version),
            epoch_start_timestamp_ms: Some(v1.epoch_start_timestamp_ms),
            parameters: Some(v1.parameters.try_into()?),
            validators: Some(v1.validators.try_into()?),
            validator_report_records,
            emission_pool: Some(v1.emission_pool.try_into()?),
            target_state: None,
            model_registry: None,
            submission_report_records: std::collections::BTreeMap::new(),
            safe_mode: Some(v1.safe_mode),
            safe_mode_accumulated_fees: None,
            safe_mode_accumulated_emissions: None,
            bridge_state: Some(v1.bridge_state.into()),
        })
    }
}

// -----------------------------------------------------------------------------
// Bridge state conversions
// -----------------------------------------------------------------------------
//
// Proto ↔ domain mappings for `types::bridge::BridgeState` and its nested
// pieces. Used by the SystemState conversions above so callers of
// `Client::get_latest_system_state` (notably bridge-node) see the live
// committee, registrations, deposit nonces, and watchdog supply rather
// than a hardcoded empty BridgeState.
//
// String keys for hex maps stay un-prefixed (no `0x`), matching the
// existing convention used elsewhere in this file (e.g. report records).

// ---- Read direction (proto → domain) ---------------------------------------

impl TryFrom<BridgeMember> for types::bridge::BridgeMember {
    type Error = String;

    fn try_from(proto: BridgeMember) -> Result<Self, Self::Error> {
        let soma_address_hex = proto.soma_address.ok_or("BridgeMember: missing soma_address")?;
        let soma_address = SomaAddress::from_str(&soma_address_hex)
            .map_err(|e| format!("BridgeMember: parse soma_address {soma_address_hex}: {e}"))?;
        Ok(types::bridge::BridgeMember {
            soma_address,
            voting_power: proto.voting_power.unwrap_or(0),
            http_url: proto.http_url.unwrap_or_default(),
            is_blocklisted: proto.is_blocklisted.unwrap_or(false),
        })
    }
}

impl TryFrom<BridgeRegistration> for types::bridge::BridgeRegistration {
    type Error = String;

    fn try_from(proto: BridgeRegistration) -> Result<Self, Self::Error> {
        let pubkey_bytes =
            proto.bridge_pubkey.ok_or("BridgeRegistration: missing bridge_pubkey")?;
        let bridge_pubkey = types::bridge::BridgePubkey::from_bytes(&pubkey_bytes)
            .map_err(|e| format!("BridgeRegistration: invalid bridge_pubkey bytes: {e}"))?;
        Ok(types::bridge::BridgeRegistration {
            bridge_pubkey,
            http_url: proto.http_url.unwrap_or_default(),
        })
    }
}

impl TryFrom<BridgeCommittee> for types::bridge::BridgeCommittee {
    type Error = String;

    fn try_from(proto: BridgeCommittee) -> Result<Self, Self::Error> {
        let mut members = BTreeMap::new();
        for (pubkey_hex, proto_member) in proto.members {
            let bytes = hex::decode(&pubkey_hex).map_err(|e| {
                format!("BridgeCommittee.members: invalid hex key {pubkey_hex}: {e}")
            })?;
            let pubkey =
                types::bridge::BridgePubkey::from_bytes(&bytes).map_err(|e| {
                    format!("BridgeCommittee.members: invalid pubkey {pubkey_hex}: {e}")
                })?;
            members.insert(pubkey, proto_member.try_into()?);
        }
        Ok(types::bridge::BridgeCommittee {
            members,
            threshold_deposit: proto.threshold_deposit.unwrap_or(0),
            threshold_withdraw: proto.threshold_withdraw.unwrap_or(0),
            threshold_pause: proto.threshold_pause.unwrap_or(0),
            threshold_unpause: proto.threshold_unpause.unwrap_or(0),
            threshold_blocklist: proto.threshold_blocklist.unwrap_or(0),
            threshold_limit_update: proto.threshold_limit_update.unwrap_or(0),
            threshold_evm_upgrade: proto.threshold_evm_upgrade.unwrap_or(0),
        })
    }
}

impl TryFrom<BridgeState> for types::bridge::BridgeState {
    type Error = String;

    fn try_from(proto: BridgeState) -> Result<Self, Self::Error> {
        let bridge_committee = proto
            .bridge_committee
            .map(types::bridge::BridgeCommittee::try_from)
            .transpose()?
            .unwrap_or_else(types::bridge::BridgeCommittee::empty);

        let processed_deposit_nonces: BTreeSet<u64> =
            proto.processed_deposit_nonces.into_iter().collect();

        let mut system_message_seq_nums = BTreeMap::new();
        for (raw_type, seq) in proto.system_message_seq_nums {
            let msg_type = bridge_message_type_from_u32(raw_type)?;
            system_message_seq_nums.insert(msg_type, seq);
        }

        let mut bridge_registrations = BTreeMap::new();
        for (addr_hex, proto_reg) in proto.bridge_registrations {
            let addr = SomaAddress::from_str(&addr_hex).map_err(|e| {
                format!("BridgeState.bridge_registrations: invalid soma address {addr_hex}: {e}")
            })?;
            bridge_registrations.insert(addr, proto_reg.try_into()?);
        }

        Ok(types::bridge::BridgeState {
            paused: proto.paused.unwrap_or(false),
            next_withdrawal_nonce: proto.next_withdrawal_nonce.unwrap_or(0),
            bridge_committee,
            processed_deposit_nonces,
            system_message_seq_nums,
            bridge_registrations,
            total_usdc_supply: proto.total_usdc_supply.unwrap_or(0),
        })
    }
}

// ---- Write direction (domain → proto) --------------------------------------

impl From<types::bridge::BridgeMember> for BridgeMember {
    fn from(member: types::bridge::BridgeMember) -> Self {
        Self {
            soma_address: Some(hex::encode(member.soma_address.to_inner())),
            voting_power: Some(member.voting_power),
            http_url: Some(member.http_url),
            is_blocklisted: Some(member.is_blocklisted),
        }
    }
}

impl From<types::bridge::BridgeRegistration> for BridgeRegistration {
    fn from(reg: types::bridge::BridgeRegistration) -> Self {
        Self {
            bridge_pubkey: Some(bytes::Bytes::copy_from_slice(reg.bridge_pubkey.as_bytes())),
            http_url: Some(reg.http_url),
        }
    }
}

impl From<types::bridge::BridgeCommittee> for BridgeCommittee {
    fn from(committee: types::bridge::BridgeCommittee) -> Self {
        let members = committee
            .members
            .into_iter()
            .map(|(pk, m)| (hex::encode(pk.as_bytes()), m.into()))
            .collect();
        Self {
            members,
            threshold_deposit: Some(committee.threshold_deposit),
            threshold_withdraw: Some(committee.threshold_withdraw),
            threshold_pause: Some(committee.threshold_pause),
            threshold_unpause: Some(committee.threshold_unpause),
            threshold_blocklist: Some(committee.threshold_blocklist),
            threshold_limit_update: Some(committee.threshold_limit_update),
            threshold_evm_upgrade: Some(committee.threshold_evm_upgrade),
        }
    }
}

impl From<types::bridge::BridgeState> for BridgeState {
    fn from(state: types::bridge::BridgeState) -> Self {
        let system_message_seq_nums = state
            .system_message_seq_nums
            .into_iter()
            .map(|(t, seq)| (t as u32, seq))
            .collect();
        let bridge_registrations = state
            .bridge_registrations
            .into_iter()
            .map(|(addr, reg)| (hex::encode(addr.to_inner()), reg.into()))
            .collect();
        Self {
            paused: Some(state.paused),
            next_withdrawal_nonce: Some(state.next_withdrawal_nonce),
            bridge_committee: Some(state.bridge_committee.into()),
            processed_deposit_nonces: state.processed_deposit_nonces.into_iter().collect(),
            system_message_seq_nums,
            bridge_registrations,
            total_usdc_supply: Some(state.total_usdc_supply),
        }
    }
}

/// Map the proto's `uint32` representation of `BridgeMessageType` back to
/// the enum. Mirrors the `#[repr(u8)]` discriminants on the domain enum;
/// any unknown discriminant is rejected so a future on-chain variant
/// doesn't silently misroute as an existing one.
fn bridge_message_type_from_u32(raw: u32) -> Result<types::bridge::BridgeMessageType, String> {
    use types::bridge::BridgeMessageType;
    match raw {
        0 => Ok(BridgeMessageType::UsdcDeposit),
        1 => Ok(BridgeMessageType::UsdcWithdraw),
        2 => Ok(BridgeMessageType::EmergencyOp),
        4 => Ok(BridgeMessageType::UpdateCommitteeBlocklist),
        5 => Ok(BridgeMessageType::LimitUpdate),
        6 => Ok(BridgeMessageType::EvmContractUpgrade),
        other => Err(format!("BridgeState: unknown BridgeMessageType discriminant {other}")),
    }
}

impl TryFrom<protocol_config::SystemParameters> for SystemParameters {
    type Error = String;

    fn try_from(domain_params: protocol_config::SystemParameters) -> Result<Self, Self::Error> {
        Ok(SystemParameters {
            epoch_duration_ms: Some(domain_params.epoch_duration_ms),
            unit_fee: Some(domain_params.unit_fee),
            channel_grace_period_ms: Some(domain_params.channel_grace_period_ms),
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
            emission_per_epoch: Some(domain_emission_pool.current_distribution_amount),
        })
    }
}

impl TryFrom<types::system_state::validator::ValidatorSet> for ValidatorSet {
    type Error = String;

    fn try_from(
        domain_set: types::system_state::validator::ValidatorSet,
    ) -> Result<Self, Self::Error> {
        let validators = domain_set
            .validators
            .into_iter()
            .map(|v| v.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let pending_validators = domain_set
            .pending_validators
            .into_iter()
            .map(|v| v.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        let pending_removals =
            domain_set.pending_removals.into_iter().map(|index| index as u32).collect();

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
            validators,
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
        let next_epoch_protocol_pubkey =
            metadata.next_epoch_protocol_pubkey.map(|key| Bytes::from(key.as_bytes().to_vec()));

        let next_epoch_network_pubkey =
            metadata.next_epoch_network_pubkey.map(|key| Bytes::from(key.to_bytes().to_vec()));

        let next_epoch_worker_pubkey =
            metadata.next_epoch_worker_pubkey.map(|key| Bytes::from(key.to_bytes().to_vec()));

        let next_epoch_net_address = metadata.next_epoch_net_address.map(|addr| addr.to_string());

        let next_epoch_p2p_address = metadata.next_epoch_p2p_address.map(|addr| addr.to_string());

        let next_epoch_primary_address =
            metadata.next_epoch_primary_address.map(|addr| addr.to_string());

        Ok(Validator {
            soma_address: Some(metadata.soma_address.to_string()),
            protocol_pubkey: Some(Bytes::from(metadata.protocol_pubkey.as_bytes().to_vec())),
            network_pubkey: Some(Bytes::from(metadata.network_pubkey.to_bytes().to_vec())),
            worker_pubkey: Some(Bytes::from(metadata.worker_pubkey.to_bytes().to_vec())),
            net_address: Some(metadata.net_address.to_string()),
            p2p_address: Some(metadata.p2p_address.to_string()),
            primary_address: Some(metadata.primary_address.to_string()),

            voting_power: Some(domain_val.voting_power),
            commission_rate: Some(domain_val.commission_rate),
            // Auto-compound: dashboards see "next epoch stake" =
            // current active + same-epoch additions waiting to
            // promote at the next boundary.
            next_epoch_stake: Some(
                domain_val.staking_pool.active_stake + domain_val.staking_pool.pending_active_stake,
            ),
            next_epoch_commission_rate: Some(domain_val.next_epoch_commission_rate),
            staking_pool: Some(domain_val.staking_pool.try_into()?),
            next_epoch_protocol_pubkey,
            next_epoch_network_pubkey,
            next_epoch_worker_pubkey,
            next_epoch_net_address,
            next_epoch_p2p_address,
            next_epoch_primary_address,
            proof_of_possession: Some(Bytes::from(metadata.proof_of_possession.as_ref().to_vec())),
            next_epoch_proof_of_possession: metadata
                .next_epoch_proof_of_possession
                .map(|pop| Bytes::from(pop.as_ref().to_vec())),
        })
    }
}

impl TryFrom<types::system_state::staking::StakingPool> for StakingPool {
    type Error = String;

    fn try_from(
        domain_pool: types::system_state::staking::StakingPool,
    ) -> Result<Self, Self::Error> {
        // Pool-token fields on the proto schema are populated with
        // zeros / empty for backward compatibility until a follow-up
        // commit reshapes the proto. `soma_balance` carries
        // `active_stake` since dashboards already read it as the
        // validator's stake; `pending_stake` carries the
        // pending-active bucket.
        Ok(StakingPool {
            id: Some(domain_pool.id.to_string()),
            activation_epoch: domain_pool.activation_epoch,
            deactivation_epoch: domain_pool.deactivation_epoch,
            soma_balance: Some(domain_pool.active_stake),
            rewards_pool: Some(0),
            pool_token_balance: Some(0),
            exchange_rates: BTreeMap::new(),
            pending_stake: Some(domain_pool.pending_active_stake),
            pending_total_soma_withdraw: Some(0),
            pending_pool_token_withdraw: Some(0),
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

// ///////////////////////////////////////////

impl From<types::metadata::Metadata> for Metadata {
    fn from(value: types::metadata::Metadata) -> Self {
        let mut message = Self::default();
        match value {
            types::metadata::Metadata::V1(v1) => {
                let proto_v1 = MetadataV1 {
                    checksum: Some(v1.checksum().as_bytes().to_vec().into()),
                    size: Some(v1.size() as u64),
                };
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

                Ok(types::metadata::Metadata::V1(types::metadata::MetadataV1::new(checksum, size)))
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

// ///////////////////////////////////////////
impl From<types::metadata::Manifest> for Manifest {
    fn from(value: types::metadata::Manifest) -> Self {
        let mut message = Self::default();
        match value {
            types::metadata::Manifest::V1(v1) => {
                let proto_v1 = ManifestV1 {
                    url: Some(v1.url().to_string()),
                    metadata: Some(v1.metadata().clone().into()),
                };
                message.version = Some(crate::proto::soma::manifest::Version::V1(proto_v1));
            }
        }
        message
    }
}

// Add this conversion for Manifest
impl TryFrom<&Manifest> for types::metadata::Manifest {
    type Error = TryFromProtoError;

    fn try_from(value: &crate::proto::soma::Manifest) -> Result<Self, Self::Error> {
        use crate::proto::soma::manifest::Version;

        match value
            .version
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("manifest version"))?
        {
            Version::V1(v1) => {
                let url = Url::parse(
                    v1.url.as_ref().ok_or_else(|| TryFromProtoError::missing("url"))?.as_str(),
                )
                .map_err(|e| TryFromProtoError::invalid("url", e))?;

                let metadata = v1
                    .metadata
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("metadata"))?
                    .try_into()
                    .map_err(|e| TryFromProtoError::invalid("metadata", e))?;

                Ok(types::metadata::Manifest::V1(types::metadata::ManifestV1::new(url, metadata)))
            }
        }
    }
}

// Also add the owned version
impl TryFrom<Manifest> for types::metadata::Manifest {
    type Error = TryFromProtoError;

    fn try_from(value: Manifest) -> Result<Self, Self::Error> {
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
            self.signatures =
                source.signatures.iter().map(|s| UserSignature::merge_from(s, &submask)).collect();
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
            let crate_summary: crate::types::CheckpointSummary =
                proto_summary.try_into().map_err(|e| TryFromProtoError::invalid("summary", e))?;

            let domain_summary: types::checkpoints::CheckpointSummary = crate_summary
                .try_into()
                .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("summary", e))?;

            // Get signature and combine into CertifiedCheckpointSummary
            let crate_sig: crate::types::ValidatorAggregatedSignature =
                checkpoint.signature().try_into()?;
            let signature: types::crypto::AuthorityStrongQuorumSignInfo = crate_sig.into();

            types::checkpoints::CertifiedCheckpointSummary::new_from_data_and_sig(
                domain_summary,
                signature,
            )
        };

        // Convert proto CheckpointContents -> crate::types::CheckpointContents -> types::checkpoints::CheckpointContents
        let contents = {
            let proto_contents = checkpoint.contents();
            let crate_contents: crate::types::CheckpointContents =
                proto_contents.try_into().map_err(|e| TryFromProtoError::invalid("contents", e))?;

            crate_contents
                .try_into()
                .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("contents", e))?
        };

        let transactions =
            checkpoint.transactions().iter().map(TryInto::try_into).collect::<Result<_, _>>()?;

        let object_set = checkpoint.objects().try_into()?;

        Ok(Self { summary, contents, transactions, object_set })
    }
}

impl TryFrom<&ObjectReference> for types::storage::ObjectKey {
    type Error = TryFromProtoError;

    fn try_from(value: &ObjectReference) -> Result<Self, Self::Error> {
        Ok(Self(
            value.object_id().parse().map_err(|e| TryFromProtoError::invalid("object_id", e))?,
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
            version_specific_data: _,
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
            next_epoch_protocol_version: Some(value.next_epoch_protocol_version.as_u64()),
            epoch_commitments: value.epoch_commitments.into_iter().map(Into::into).collect(),
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
                .map(|(digests, sigs)| CheckpointedTransactionInfo {
                    transaction: Some(digests.transaction.to_string()),
                    effects: Some(digests.effects.to_string()),
                    signatures: sigs.into_iter().map(Into::into).collect(),
                    ..Default::default()
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
            let crate_object: crate::types::Object =
                o.try_into().map_err(|e| TryFromProtoError::invalid("object", e))?;
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
                .map(|s| s.try_into().map_err(|e| TryFromProtoError::invalid("signature", e)))
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
            let crate_effects: crate::types::TransactionEffects =
                proto_effects.try_into().map_err(|e| TryFromProtoError::invalid("effects", e))?;

            // Now convert crate::types::TransactionEffects to types::effects::TransactionEffects
            crate_effects
                .try_into()
                .map_err(|e: SdkTypeConversionError| TryFromProtoError::invalid("effects", e))?
        };

        Ok(Self { transaction, signatures, effects })
    }
}

#[cfg(test)]
mod bridge_state_conversion_tests {
    use super::*;
    use types::bridge::{
        BridgeCommittee as DomainBridgeCommittee, BridgeMember as DomainBridgeMember,
        BridgeMessageType, BridgePubkey, BridgeRegistration as DomainBridgeRegistration,
        BridgeState as DomainBridgeState,
    };

    /// Round-trip a populated `BridgeState` through the proto layer and
    /// back, asserting equality. Catches field omissions, key-encoding
    /// mismatches, and discriminant mishandling in the conversions added
    /// for v0.1.24-rc1. Single big test by design — round-trip equality is
    /// the contract; field-by-field tests are easy to keep passing while
    /// silently dropping a new field.
    #[test]
    fn bridge_state_round_trip_preserves_every_field() {
        // Two known-valid compressed secp256k1 pubkeys — same fixtures used
        // by `types::bridge::test_encode_blocklist_payload_regression`.
        let pk_a = BridgePubkey::from_bytes(
            &hex::decode("02321ede33d2c2d7a8a152f275a1484edef2098f034121a602cb7d767d38680aa4")
                .unwrap(),
        )
        .unwrap();
        let pk_b = BridgePubkey::from_bytes(
            &hex::decode("027f1178ff417fc9f5b8290bd8876f0a157a505a6c52db100a8492203ddd1d4279")
                .unwrap(),
        )
        .unwrap();

        let addr_a = SomaAddress::from_str(
            "1111111111111111111111111111111111111111111111111111111111111111",
        )
        .unwrap();
        let addr_b = SomaAddress::from_str(
            "2222222222222222222222222222222222222222222222222222222222222222",
        )
        .unwrap();

        let mut members = BTreeMap::new();
        members.insert(
            pk_a.clone(),
            DomainBridgeMember {
                soma_address: addr_a,
                voting_power: 3333,
                http_url: "http://bridge-a.example:9191".to_string(),
                is_blocklisted: false,
            },
        );
        members.insert(
            pk_b.clone(),
            DomainBridgeMember {
                soma_address: addr_b,
                voting_power: 3334,
                http_url: "http://bridge-b.example:9191".to_string(),
                is_blocklisted: true, // exercise the bool default explicitly
            },
        );
        let bridge_committee = DomainBridgeCommittee {
            members,
            // Non-default thresholds catch a "preserve thresholds" omission.
            threshold_deposit: 1234,
            threshold_withdraw: 2345,
            threshold_pause: 456,
            threshold_unpause: 5001,
            threshold_blocklist: 5002,
            threshold_limit_update: 5003,
            threshold_evm_upgrade: 5004,
        };

        let mut processed_deposit_nonces = BTreeSet::new();
        processed_deposit_nonces.insert(0u64);
        processed_deposit_nonces.insert(1);
        processed_deposit_nonces.insert(42);

        let mut system_message_seq_nums = BTreeMap::new();
        system_message_seq_nums.insert(BridgeMessageType::EmergencyOp, 7u64);
        system_message_seq_nums.insert(BridgeMessageType::LimitUpdate, 3);

        let mut bridge_registrations = BTreeMap::new();
        bridge_registrations.insert(
            addr_a,
            DomainBridgeRegistration {
                bridge_pubkey: pk_a.clone(),
                http_url: "http://bridge-a-pending.example:9191".to_string(),
            },
        );

        let original = DomainBridgeState {
            paused: true,
            next_withdrawal_nonce: 99,
            bridge_committee,
            processed_deposit_nonces,
            system_message_seq_nums,
            bridge_registrations,
            total_usdc_supply: 1_234_567_890,
        };

        let proto: BridgeState = original.clone().into();
        let round_tripped: DomainBridgeState =
            proto.try_into().expect("proto -> domain round-trip");
        assert_eq!(original, round_tripped);
    }

    /// An unknown `BridgeMessageType` discriminant must be rejected — a
    /// future on-chain variant that the conversions don't recognize should
    /// fail loudly, not silently misroute as an existing variant.
    #[test]
    fn unknown_bridge_message_type_discriminant_is_rejected() {
        assert!(bridge_message_type_from_u32(99).is_err());
        // Byte 3 is intentionally unused in `BridgeMessageType`; gap must
        // also stay rejected so we don't add a wrong route by accident.
        assert!(bridge_message_type_from_u32(3).is_err());
    }
}
