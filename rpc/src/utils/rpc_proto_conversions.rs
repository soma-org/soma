// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

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
use types::metadata::{ManifestAPI, MetadataAPI};
use url::Url;

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
        Self {
            base_fee: Some(base_fee),
            operation_fee: Some(operation_fee),
            value_fee: Some(value_fee),
            total_fee: Some(total_fee),
        }
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
        use execution_error::ErrorDetails;
        use execution_error::ExecutionErrorKind;
        use types::effects::ExecutionFailureStatus as E;

        let mut message = Self::default();

        let (kind, details) = match value {
            E::InsufficientGas => (ExecutionErrorKind::InsufficientGas, None),
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
            E::ModelWeightsUrlMismatch => (ExecutionErrorKind::ModelWeightsUrlMismatch, None),
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
            E::TargetExpired { .. } => (ExecutionErrorKind::TargetExpired, None),
            E::TargetNotFilled => (ExecutionErrorKind::TargetNotFilled, None),
            E::ChallengeWindowOpen { .. } => (ExecutionErrorKind::ChallengeWindowOpen, None),
            E::TargetAlreadyClaimed => (ExecutionErrorKind::TargetAlreadyClaimed, None),
            // Submission errors
            E::ModelNotInTarget { .. } => (ExecutionErrorKind::ModelNotInTarget, None),
            E::EmbeddingDimensionMismatch { .. } => {
                (ExecutionErrorKind::EmbeddingDimensionMismatch, None)
            }
            E::DistanceExceedsThreshold { .. } => {
                (ExecutionErrorKind::DistanceExceedsThreshold, None)
            }
            E::InsufficientBond { .. } => (ExecutionErrorKind::InsufficientBond, None),
            E::InsufficientEmissionBalance => {
                (ExecutionErrorKind::InsufficientEmissionBalance, None)
            }
            // Challenge errors
            E::ChallengeWindowClosed { .. } => (ExecutionErrorKind::ChallengeWindowClosed, None),
            E::InsufficientChallengerBond { .. } => {
                (ExecutionErrorKind::InsufficientChallengerBond, None)
            }
            E::ChallengeNotFound { .. } => (ExecutionErrorKind::ChallengeNotFound, None),
            E::ChallengeNotPending { .. } => (ExecutionErrorKind::ChallengeNotPending, None),
            E::ChallengeExpired { .. } => (ExecutionErrorKind::ChallengeExpired, None),
            E::InvalidChallengeResult => (ExecutionErrorKind::InvalidChallengeResult, None),
            E::InvalidChallengeQuorum => (ExecutionErrorKind::InvalidChallengeQuorum, None),
            E::ChallengeAlreadyExists => (ExecutionErrorKind::ChallengeAlreadyExists, None),
            E::DataExceedsMaxSize { size, max_size } => (
                ExecutionErrorKind::DataExceedsMaxSize,
                Some(format!("size={}, max_size={}", size, max_size)),
            ),
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

            K::TransferCoin { coin, amount, recipient } => Kind::TransferCoin(TransferCoin {
                coin: Some(object_ref_to_proto(coin)),
                amount,
                recipient: Some(recipient.to_string()),
            }),
            K::PayCoins { coins, amounts, recipients } => Kind::PayCoins(PayCoins {
                coins: coins.into_iter().map(object_ref_to_proto).collect(),
                amounts: amounts.unwrap_or_default(),
                recipients: recipients.into_iter().map(|r| r.to_string()).collect(),
            }),
            K::TransferObjects { objects, recipient } => Kind::TransferObjects(TransferObjects {
                objects: objects.into_iter().map(object_ref_to_proto).collect(),
                recipient: Some(recipient.to_string()),
            }),
            K::AddStake { address, coin_ref, amount } => Kind::AddStake(AddStake {
                address: Some(address.to_string()),
                coin_ref: Some(object_ref_to_proto(coin_ref)),
                amount,
            }),

            K::WithdrawStake { staked_soma } => Kind::WithdrawStake(WithdrawStake {
                staked_soma: Some(object_ref_to_proto(staked_soma)),
            }),

            // Model transactions
            K::CommitModel(args) => Kind::CommitModel(CommitModel {
                model_id: Some(args.model_id.to_string()),
                weights_url_commitment: Some(
                    args.weights_url_commitment.into_inner().to_vec().into(),
                ),
                weights_commitment: Some(args.weights_commitment.into_inner().to_vec().into()),
                architecture_version: Some(args.architecture_version),
                stake_amount: Some(args.stake_amount),
                commission_rate: Some(args.commission_rate),
                staking_pool_id: Some(args.staking_pool_id.to_string()),
            }),
            K::RevealModel(args) => Kind::RevealModel(RevealModel {
                model_id: Some(args.model_id.to_string()),
                weights_manifest: Some(args.weights_manifest.into()),
                embedding: args.embedding.to_vec(),
            }),
            K::CommitModelUpdate(args) => Kind::CommitModelUpdate(CommitModelUpdate {
                model_id: Some(args.model_id.to_string()),
                weights_url_commitment: Some(
                    args.weights_url_commitment.into_inner().to_vec().into(),
                ),
                weights_commitment: Some(args.weights_commitment.into_inner().to_vec().into()),
            }),
            K::RevealModelUpdate(args) => Kind::RevealModelUpdate(RevealModelUpdate {
                model_id: Some(args.model_id.to_string()),
                weights_manifest: Some(args.weights_manifest.into()),
                embedding: args.embedding.to_vec(),
            }),
            K::AddStakeToModel { model_id, coin_ref, amount } => {
                Kind::AddStakeToModel(AddStakeToModel {
                    model_id: Some(model_id.to_string()),
                    coin_ref: Some(object_ref_to_proto(coin_ref)),
                    amount,
                })
            }
            K::SetModelCommissionRate { model_id, new_rate } => {
                Kind::SetModelCommissionRate(SetModelCommissionRate {
                    model_id: Some(model_id.to_string()),
                    new_rate: Some(new_rate),
                })
            }
            K::DeactivateModel { model_id } => {
                Kind::DeactivateModel(DeactivateModel { model_id: Some(model_id.to_string()) })
            }
            K::ReportModel { model_id } => {
                Kind::ReportModel(ReportModel { model_id: Some(model_id.to_string()) })
            }
            K::UndoReportModel { model_id } => {
                Kind::UndoReportModel(UndoReportModel { model_id: Some(model_id.to_string()) })
            }
            // Submission transactions
            K::SubmitData(args) => Kind::SubmitData(SubmitData {
                target_id: Some(args.target_id.to_string()),
                data_commitment: Some(args.data_commitment.into_inner().to_vec().into()),
                data_manifest: Some(crate::proto::soma::SubmissionManifest {
                    manifest: Some(args.data_manifest.manifest.into()),
                }),
                model_id: Some(args.model_id.to_string()),
                embedding: args.embedding.to_vec(),
                distance_score: Some(args.distance_score.as_scalar()),
                bond_coin: Some(object_ref_to_proto(args.bond_coin)),
            }),
            K::ClaimRewards(args) => {
                Kind::ClaimRewards(ClaimRewards { target_id: Some(args.target_id.to_string()) })
            }
            K::ReportSubmission { target_id, challenger } => {
                Kind::ReportSubmission(ReportSubmission {
                    target_id: Some(target_id.to_string()),
                    challenger: challenger.map(|c| c.to_string()),
                })
            }
            K::UndoReportSubmission { target_id } => {
                Kind::UndoReportSubmission(UndoReportSubmission {
                    target_id: Some(target_id.to_string()),
                })
            }
            // Challenge transactions
            // All challenges are fraud challenges now (simplified design v2)
            K::InitiateChallenge(args) => Kind::InitiateChallenge(InitiateChallenge {
                target_id: Some(args.target_id.to_string()),
                challenge_type: Some("Fraud".to_string()),
                model_id: None,
                bond_coin: Some(object_ref_to_proto(args.bond_coin)),
            }),
            // Tally-based challenge transactions (simplified: reports indicate "challenger is wrong")
            K::ReportChallenge { challenge_id } => Kind::ReportChallenge(ReportChallenge {
                challenge_id: Some(challenge_id.to_string()),
            }),
            K::UndoReportChallenge { challenge_id } => {
                Kind::UndoReportChallenge(UndoReportChallenge {
                    challenge_id: Some(challenge_id.to_string()),
                })
            }
            K::ClaimChallengeBond { challenge_id } => {
                Kind::ClaimChallengeBond(ClaimChallengeBond {
                    challenge_id: Some(challenge_id.to_string()),
                })
            }
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
            proxy_address: Some(args.proxy_address.into()),
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
            next_epoch_proxy_address: args.next_epoch_proxy_address.map(|bytes| bytes.into()),
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
            ..Default::default()
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

        // NOTE: submission_report_records has been moved to Target objects (tally-based approach)

        // Convert model registry
        let model_registry =
            proto_state.model_registry.map(|mr| mr.try_into()).transpose()?.unwrap_or_default();

        // Convert target state
        let target_state =
            proto_state.target_state.map(|ts| ts.try_into()).transpose()?.unwrap_or_default();

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

                model_registry,

                target_state,

                safe_mode: proto_state.safe_mode.unwrap_or(false),
                safe_mode_accumulated_fees: proto_state.safe_mode_accumulated_fees.unwrap_or(0),
                safe_mode_accumulated_emissions: proto_state
                    .safe_mode_accumulated_emissions
                    .unwrap_or(0),
            });

        Ok(system_state)
    }
}

impl TryFrom<SystemParameters> for protocol_config::SystemParameters {
    type Error = String;

    fn try_from(proto_params: SystemParameters) -> Result<Self, Self::Error> {
        Ok(protocol_config::SystemParameters {
            epoch_duration_ms: proto_params.epoch_duration_ms.ok_or("Missing epoch_duration_ms")?,
            validator_reward_allocation_bps: proto_params
                .validator_reward_allocation_bps
                .ok_or("Missing validator_reward_allocation_bps")?,
            model_min_stake: proto_params.model_min_stake.ok_or("Missing model_min_stake")?,
            model_architecture_version: proto_params
                .model_architecture_version
                .ok_or("Missing model_architecture_version")?,
            model_reveal_slash_rate_bps: proto_params
                .model_reveal_slash_rate_bps
                .ok_or("Missing model_reveal_slash_rate_bps")?,
            model_tally_slash_rate_bps: proto_params
                .model_tally_slash_rate_bps
                .ok_or("Missing model_tally_slash_rate_bps")?,
            target_epoch_fee_collection: proto_params
                .target_epoch_fee_collection
                .ok_or("Missing target_epoch_fee_collection")?,
            base_fee: proto_params.base_fee.ok_or("Missing base_fee")?,
            write_object_fee: proto_params.write_object_fee.ok_or("Missing write_object_fee")?,
            value_fee_bps: proto_params.value_fee_bps.ok_or("Missing value_fee_bps")?,
            min_value_fee_bps: proto_params.min_value_fee_bps.ok_or("Missing min_value_fee_bps")?,
            max_value_fee_bps: proto_params.max_value_fee_bps.ok_or("Missing max_value_fee_bps")?,
            fee_adjustment_rate_bps: proto_params
                .fee_adjustment_rate_bps
                .ok_or("Missing fee_adjustment_rate_bps")?,
            // Target/Submission parameters
            target_models_per_target: proto_params
                .target_models_per_target
                .ok_or("Missing target_models_per_target")?,
            target_embedding_dim: proto_params
                .target_embedding_dim
                .ok_or("Missing target_embedding_dim")?,
            target_initial_distance_threshold: types::tensor::SomaTensor::scalar(
                proto_params
                    .target_initial_distance_threshold
                    .ok_or("Missing target_initial_distance_threshold")?,
            ),
            target_reward_allocation_bps: proto_params
                .target_reward_allocation_bps
                .ok_or("Missing target_reward_allocation_bps")?,
            target_hits_per_epoch: proto_params
                .target_hits_per_epoch
                .ok_or("Missing target_hits_per_epoch")?,
            target_hits_ema_decay_bps: proto_params
                .target_hits_ema_decay_bps
                .ok_or("Missing target_hits_ema_decay_bps")?,
            target_difficulty_adjustment_rate_bps: proto_params
                .target_difficulty_adjustment_rate_bps
                .ok_or("Missing target_difficulty_adjustment_rate_bps")?,
            target_max_distance_threshold: types::tensor::SomaTensor::scalar(
                proto_params
                    .target_max_distance_threshold
                    .ok_or("Missing target_max_distance_threshold")?,
            ),
            target_min_distance_threshold: types::tensor::SomaTensor::scalar(
                proto_params
                    .target_min_distance_threshold
                    .ok_or("Missing target_min_distance_threshold")?,
            ),
            target_initial_targets_per_epoch: proto_params
                .target_initial_targets_per_epoch
                .ok_or("Missing target_initial_targets_per_epoch")?,
            // Reward distribution parameters
            target_submitter_reward_share_bps: proto_params
                .target_submitter_reward_share_bps
                .ok_or("Missing target_submitter_reward_share_bps")?,
            target_model_reward_share_bps: proto_params
                .target_model_reward_share_bps
                .ok_or("Missing target_model_reward_share_bps")?,
            target_claimer_incentive_bps: proto_params
                .target_claimer_incentive_bps
                .ok_or("Missing target_claimer_incentive_bps")?,
            // Submission parameters
            submission_bond_per_byte: proto_params
                .submission_bond_per_byte
                .ok_or("Missing submission_bond_per_byte")?,
            // Challenge parameters
            challenger_bond_per_byte: proto_params
                .challenger_bond_per_byte
                .ok_or("Missing challenger_bond_per_byte")?,
            // Data size limit
            max_submission_data_size: proto_params
                .max_submission_data_size
                .ok_or("Missing max_submission_data_size")?,
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
        use fastcrypto::traits::ToFromBytes;
        use std::str::FromStr;

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

        // Parse proxy addresses (optional)
        let proxy_address = proto_val
            .proxy_address
            .map(|addr| {
                types::multiaddr::Multiaddr::from_str(&addr)
                    .map_err(|e| format!("Invalid proxy_address: {}", e))
            })
            .transpose()?
            .unwrap_or_else(types::multiaddr::Multiaddr::empty);

        let next_epoch_proxy_address = proto_val
            .next_epoch_proxy_address
            .map(|addr| {
                types::multiaddr::Multiaddr::from_str(&addr)
                    .map_err(|e| format!("Invalid next_epoch_proxy_address: {}", e))
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
            proxy_address,
            next_epoch_protocol_pubkey,
            next_epoch_network_pubkey,
            next_epoch_net_address,
            next_epoch_p2p_address,
            next_epoch_primary_address,
            next_epoch_worker_pubkey,
            next_epoch_proxy_address,
            next_epoch_proof_of_possession,
        };

        let staking_pool = proto_val.staking_pool.ok_or("Missing staking_pool")?.try_into()?;

        Ok(types::system_state::validator::Validator {
            metadata,
            voting_power: proto_val.voting_power.ok_or("Missing voting_power")?,
            staking_pool,
            commission_rate: proto_val.commission_rate.ok_or("Missing commission_rate")?,
            next_epoch_stake: proto_val.next_epoch_stake.ok_or("Missing next_epoch_stake")?,
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
            pool_token_amount: proto_rate.pool_token_amount.ok_or("Missing pool_token_amount")?,
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

fn convert_submission_report_records(
    proto_records: BTreeMap<String, ReporterSet>,
) -> Result<BTreeMap<types::object::ObjectID, BTreeSet<SomaAddress>>, String> {
    proto_records
        .into_iter()
        .map(|(k, v)| {
            let key = k.parse().map_err(|_| "Invalid ObjectID")?;
            let reporters = v
                .reporters
                .into_iter()
                .map(|r| r.parse().map_err(|_| "Invalid SomaAddress"))
                .collect::<Result<BTreeSet<_>, _>>()?;
            Ok((key, reporters))
        })
        .collect()
}

fn convert_submission_report_records_to_proto(
    records: BTreeMap<types::object::ObjectID, BTreeSet<SomaAddress>>,
) -> Result<BTreeMap<String, ReporterSet>, String> {
    Ok(records
        .into_iter()
        .map(|(k, v)| {
            let reporters: Vec<String> = v.into_iter().map(|addr| addr.to_string()).collect();
            (k.to_string(), ReporterSet { reporters })
        })
        .collect())
}

impl TryFrom<types::system_state::SystemState> for SystemState {
    type Error = String;

    fn try_from(domain_state: types::system_state::SystemState) -> Result<Self, Self::Error> {
        let types::system_state::SystemState::V1(v1) = domain_state;

        // Convert validator report records
        let validator_report_records =
            convert_report_records_to_proto(v1.validator_report_records)?;

        // NOTE: submission_report_records has been moved to Target objects (tally-based approach)

        Ok(SystemState {
            epoch: Some(v1.epoch),
            protocol_version: Some(v1.protocol_version),
            epoch_start_timestamp_ms: Some(v1.epoch_start_timestamp_ms),
            parameters: Some(v1.parameters.try_into()?),
            validators: Some(v1.validators.try_into()?),
            validator_report_records,
            emission_pool: Some(v1.emission_pool.try_into()?),
            target_state: Some(v1.target_state.into()),
            model_registry: Some(v1.model_registry.try_into()?),
            submission_report_records: std::collections::BTreeMap::new(), // Empty for backward compat
            safe_mode: Some(v1.safe_mode),
            safe_mode_accumulated_fees: Some(v1.safe_mode_accumulated_fees),
            safe_mode_accumulated_emissions: Some(v1.safe_mode_accumulated_emissions),
        })
    }
}

//
// TargetState conversions
//

impl From<types::system_state::target_state::TargetState> for TargetState {
    fn from(domain: types::system_state::target_state::TargetState) -> Self {
        TargetState {
            distance_threshold: Some(domain.distance_threshold.as_scalar()),
            targets_generated_this_epoch: Some(domain.targets_generated_this_epoch),
            hits_this_epoch: Some(domain.hits_this_epoch),
            hits_ema: Some(domain.hits_ema),
            reward_per_target: Some(domain.reward_per_target),
        }
    }
}

impl TryFrom<TargetState> for types::system_state::target_state::TargetState {
    type Error = String;

    fn try_from(proto: TargetState) -> Result<Self, Self::Error> {
        Ok(types::system_state::target_state::TargetState {
            distance_threshold: types::tensor::SomaTensor::scalar(
                proto.distance_threshold.ok_or("Missing distance_threshold")?,
            ),
            targets_generated_this_epoch: proto
                .targets_generated_this_epoch
                .ok_or("Missing targets_generated_this_epoch")?,
            hits_this_epoch: proto.hits_this_epoch.ok_or("Missing hits_this_epoch")?,
            hits_ema: proto.hits_ema.ok_or("Missing hits_ema")?,
            reward_per_target: proto.reward_per_target.ok_or("Missing reward_per_target")?,
        })
    }
}

impl TryFrom<protocol_config::SystemParameters> for SystemParameters {
    type Error = String;

    fn try_from(domain_params: protocol_config::SystemParameters) -> Result<Self, Self::Error> {
        Ok(SystemParameters {
            epoch_duration_ms: Some(domain_params.epoch_duration_ms),
            validator_reward_allocation_bps: Some(domain_params.validator_reward_allocation_bps),
            model_min_stake: Some(domain_params.model_min_stake),
            model_architecture_version: Some(domain_params.model_architecture_version),
            model_reveal_slash_rate_bps: Some(domain_params.model_reveal_slash_rate_bps),
            model_tally_slash_rate_bps: Some(domain_params.model_tally_slash_rate_bps),
            target_epoch_fee_collection: Some(domain_params.target_epoch_fee_collection),
            base_fee: Some(domain_params.base_fee),
            write_object_fee: Some(domain_params.write_object_fee),
            value_fee_bps: Some(domain_params.value_fee_bps),
            min_value_fee_bps: Some(domain_params.min_value_fee_bps),
            max_value_fee_bps: Some(domain_params.max_value_fee_bps),
            fee_adjustment_rate_bps: Some(domain_params.fee_adjustment_rate_bps),
            // Target/Submission parameters
            target_models_per_target: Some(domain_params.target_models_per_target),
            target_embedding_dim: Some(domain_params.target_embedding_dim),
            target_initial_distance_threshold: Some(
                domain_params.target_initial_distance_threshold.as_scalar(),
            ),
            target_reward_allocation_bps: Some(domain_params.target_reward_allocation_bps),
            target_hits_per_epoch: Some(domain_params.target_hits_per_epoch),
            target_hits_ema_decay_bps: Some(domain_params.target_hits_ema_decay_bps),
            target_difficulty_adjustment_rate_bps: Some(
                domain_params.target_difficulty_adjustment_rate_bps,
            ),
            target_max_distance_threshold: Some(
                domain_params.target_max_distance_threshold.as_scalar(),
            ),
            target_min_distance_threshold: Some(
                domain_params.target_min_distance_threshold.as_scalar(),
            ),
            target_initial_targets_per_epoch: Some(domain_params.target_initial_targets_per_epoch),
            // Reward distribution parameters
            target_submitter_reward_share_bps: Some(domain_params.target_submitter_reward_share_bps),
            target_model_reward_share_bps: Some(domain_params.target_model_reward_share_bps),
            target_claimer_incentive_bps: Some(domain_params.target_claimer_incentive_bps),
            // Submission parameters
            submission_bond_per_byte: Some(domain_params.submission_bond_per_byte),
            // Challenge parameters
            challenger_bond_per_byte: Some(domain_params.challenger_bond_per_byte),
            // Data size limit
            max_submission_data_size: Some(domain_params.max_submission_data_size),
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

        let next_epoch_proxy_address =
            metadata.next_epoch_proxy_address.map(|addr| addr.to_string());

        // Convert proxy address (empty Multiaddr becomes None in proto)
        let proxy_address = if metadata.proxy_address.is_empty() {
            None
        } else {
            Some(metadata.proxy_address.to_string())
        };

        Ok(Validator {
            soma_address: Some(metadata.soma_address.to_string()),
            protocol_pubkey: Some(Bytes::from(metadata.protocol_pubkey.as_bytes().to_vec())),
            network_pubkey: Some(Bytes::from(metadata.network_pubkey.to_bytes().to_vec())),
            worker_pubkey: Some(Bytes::from(metadata.worker_pubkey.to_bytes().to_vec())),
            net_address: Some(metadata.net_address.to_string()),
            p2p_address: Some(metadata.p2p_address.to_string()),
            primary_address: Some(metadata.primary_address.to_string()),
            proxy_address,

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
            next_epoch_proxy_address,
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
// ModelWeightsManifest
//

impl From<types::model::ModelWeightsManifest> for ModelWeightsManifest {
    fn from(value: types::model::ModelWeightsManifest) -> Self {
        Self {
            manifest: Some(value.manifest.into()),
            decryption_key: Some(value.decryption_key.as_bytes().to_vec().into()),
        }
    }
}

impl TryFrom<ModelWeightsManifest> for types::model::ModelWeightsManifest {
    type Error = String;

    fn try_from(proto: ModelWeightsManifest) -> Result<Self, Self::Error> {
        let manifest = proto
            .manifest
            .ok_or("Missing manifest")?
            .try_into()
            .map_err(|e: TryFromProtoError| e.to_string())?;

        let key_bytes: Vec<u8> = proto.decryption_key.ok_or("Missing decryption_key")?.into();
        let key_array: [u8; 32] =
            key_bytes.try_into().map_err(|_| "decryption_key must be 32 bytes".to_string())?;

        Ok(types::model::ModelWeightsManifest {
            manifest,
            decryption_key: types::crypto::DecryptionKey::new(key_array),
        })
    }
}

//
// PendingModelUpdate
//

impl From<types::model::PendingModelUpdate> for PendingModelUpdate {
    fn from(value: types::model::PendingModelUpdate) -> Self {
        Self {
            weights_url_commitment: Some(value.weights_url_commitment.into_inner().to_vec().into()),
            weights_commitment: Some(value.weights_commitment.into_inner().to_vec().into()),
            commit_epoch: Some(value.commit_epoch),
        }
    }
}

impl TryFrom<PendingModelUpdate> for types::model::PendingModelUpdate {
    type Error = String;

    fn try_from(proto: PendingModelUpdate) -> Result<Self, Self::Error> {
        let url_bytes: Vec<u8> =
            proto.weights_url_commitment.ok_or("Missing weights_url_commitment")?.into();
        let url_array: [u8; 32] = url_bytes
            .try_into()
            .map_err(|_| "weights_url_commitment must be 32 bytes".to_string())?;

        let wt_bytes: Vec<u8> =
            proto.weights_commitment.ok_or("Missing weights_commitment")?.into();
        let wt_array: [u8; 32] =
            wt_bytes.try_into().map_err(|_| "weights_commitment must be 32 bytes".to_string())?;

        Ok(types::model::PendingModelUpdate {
            weights_url_commitment: types::digests::ModelWeightsUrlCommitment::new(url_array),
            weights_commitment: types::digests::ModelWeightsCommitment::new(wt_array),
            commit_epoch: proto.commit_epoch.ok_or("Missing commit_epoch")?,
        })
    }
}

//
// Model
//

impl TryFrom<types::model::ModelV1> for Model {
    type Error = String;

    fn try_from(domain: types::model::ModelV1) -> Result<Self, Self::Error> {
        Ok(Model {
            owner: Some(domain.owner.to_string()),
            architecture_version: Some(domain.architecture_version),
            weights_url_commitment: Some(
                domain.weights_url_commitment.into_inner().to_vec().into(),
            ),
            weights_commitment: Some(domain.weights_commitment.into_inner().to_vec().into()),
            commit_epoch: Some(domain.commit_epoch),
            weights_manifest: domain.weights_manifest.map(Into::into),
            embedding: domain.embedding.map(|e| e.to_vec()).unwrap_or_default(),
            staking_pool: Some(domain.staking_pool.try_into()?),
            commission_rate: Some(domain.commission_rate),
            next_epoch_commission_rate: Some(domain.next_epoch_commission_rate),
            pending_update: domain.pending_update.map(Into::into),
        })
    }
}

impl TryFrom<Model> for types::model::ModelV1 {
    type Error = String;

    fn try_from(proto: Model) -> Result<Self, Self::Error> {
        let owner = proto
            .owner
            .ok_or("Missing owner")?
            .parse()
            .map_err(|_| "Invalid SomaAddress".to_string())?;

        let url_bytes: Vec<u8> =
            proto.weights_url_commitment.ok_or("Missing weights_url_commitment")?.into();
        let url_array: [u8; 32] = url_bytes
            .try_into()
            .map_err(|_| "weights_url_commitment must be 32 bytes".to_string())?;

        let wt_bytes: Vec<u8> =
            proto.weights_commitment.ok_or("Missing weights_commitment")?.into();
        let wt_array: [u8; 32] =
            wt_bytes.try_into().map_err(|_| "weights_commitment must be 32 bytes".to_string())?;

        let weights_manifest = proto.weights_manifest.map(TryInto::try_into).transpose()?;

        let staking_pool = proto.staking_pool.ok_or("Missing staking_pool")?.try_into()?;

        let pending_update = proto.pending_update.map(TryInto::try_into).transpose()?;

        // Convert embedding from repeated float to Option<SomaTensor>
        let embedding = if proto.embedding.is_empty() {
            None
        } else {
            let dim = proto.embedding.len();
            Some(types::tensor::SomaTensor::new(proto.embedding, vec![dim]))
        };

        Ok(types::model::ModelV1 {
            owner,
            architecture_version: proto
                .architecture_version
                .ok_or("Missing architecture_version")?,
            weights_url_commitment: types::digests::ModelWeightsUrlCommitment::new(url_array),
            weights_commitment: types::digests::ModelWeightsCommitment::new(wt_array),
            commit_epoch: proto.commit_epoch.ok_or("Missing commit_epoch")?,
            weights_manifest,
            embedding,
            staking_pool,
            commission_rate: proto.commission_rate.ok_or("Missing commission_rate")?,
            next_epoch_commission_rate: proto
                .next_epoch_commission_rate
                .ok_or("Missing next_epoch_commission_rate")?,
            pending_update,
        })
    }
}

//
// ModelRegistry
//

impl TryFrom<types::system_state::model_registry::ModelRegistry> for ModelRegistry {
    type Error = String;

    fn try_from(
        domain: types::system_state::model_registry::ModelRegistry,
    ) -> Result<Self, Self::Error> {
        let active_models = domain
            .active_models
            .into_iter()
            .map(|(id, model)| {
                let proto_model: Model = model.try_into()?;
                Ok((id.to_string(), proto_model))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let pending_models = domain
            .pending_models
            .into_iter()
            .map(|(id, model)| {
                let proto_model: Model = model.try_into()?;
                Ok((id.to_string(), proto_model))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let staking_pool_mappings = domain
            .staking_pool_mappings
            .into_iter()
            .map(|(pool_id, model_id)| (pool_id.to_string(), model_id.to_string()))
            .collect();

        let inactive_models = domain
            .inactive_models
            .into_iter()
            .map(|(id, model)| {
                let proto_model: Model = model.try_into()?;
                Ok((id.to_string(), proto_model))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let model_report_records = domain
            .model_report_records
            .into_iter()
            .map(|(k, v)| {
                let key = k.to_string();
                let reporters = v.into_iter().map(|r| r.to_string()).collect();
                Ok((key, ReporterSet { reporters }))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        Ok(ModelRegistry {
            active_models,
            pending_models,
            staking_pool_mappings,
            inactive_models,
            total_model_stake: Some(domain.total_model_stake),
            model_report_records,
        })
    }
}

impl TryFrom<ModelRegistry> for types::system_state::model_registry::ModelRegistry {
    type Error = String;

    fn try_from(proto: ModelRegistry) -> Result<Self, Self::Error> {
        let active_models = proto
            .active_models
            .into_iter()
            .map(|(k, v)| {
                let id = k.parse().map_err(|_| "Invalid ModelId/ObjectID".to_string())?;
                let model: types::model::ModelV1 = v.try_into()?;
                Ok((id, model))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let pending_models = proto
            .pending_models
            .into_iter()
            .map(|(k, v)| {
                let id = k.parse().map_err(|_| "Invalid ModelId/ObjectID".to_string())?;
                let model: types::model::ModelV1 = v.try_into()?;
                Ok((id, model))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let staking_pool_mappings = proto
            .staking_pool_mappings
            .into_iter()
            .map(|(k, v)| {
                let pool_id = k.parse().map_err(|_| "Invalid ObjectID".to_string())?;
                let model_id = v.parse().map_err(|_| "Invalid ModelId".to_string())?;
                Ok((pool_id, model_id))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let inactive_models = proto
            .inactive_models
            .into_iter()
            .map(|(k, v)| {
                let id = k.parse().map_err(|_| "Invalid ModelId/ObjectID".to_string())?;
                let model: types::model::ModelV1 = v.try_into()?;
                Ok((id, model))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        let model_report_records = proto
            .model_report_records
            .into_iter()
            .map(|(k, v)| {
                let model_id = k.parse().map_err(|_| "Invalid ModelId/ObjectID".to_string())?;
                let reporters = v
                    .reporters
                    .into_iter()
                    .map(|r| r.parse().map_err(|_| "Invalid SomaAddress".to_string()))
                    .collect::<Result<BTreeSet<_>, _>>()?;
                Ok((model_id, reporters))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;

        Ok(types::system_state::model_registry::ModelRegistry {
            active_models,
            pending_models,
            staking_pool_mappings,
            inactive_models,
            total_model_stake: proto.total_model_stake.ok_or("Missing total_model_stake")?,
            model_report_records,
        })
    }
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
