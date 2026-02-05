use std::collections::BTreeMap;

use crate::{proto::TryFromProtoError, types::*};
use base64::Engine;
use fastcrypto::{
    bls12381::min_sig::BLS12381PublicKey, serde_helpers::BytesRepresentation, traits::ToFromBytes,
};
use tap::Pipe;
use tracing::info;
use types::{
    crypto::{DIGEST_LENGTH, SomaSignature},
    metadata::{ManifestAPI, MetadataAPI as _},
    multiaddr::Multiaddr,
};

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

impl TryFrom<types::object::Object> for Object {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::object::Object) -> Result<Self, Self::Error> {
        // Extract the object's data fields
        let object_id = value.id().into();
        let version = value.version().value();

        // Map ObjectType from domain to SDK
        let object_type = match value.data.object_type() {
            types::object::ObjectType::SystemState => ObjectType::SystemState,
            types::object::ObjectType::Coin => ObjectType::Coin,
            types::object::ObjectType::StakedSoma => ObjectType::StakedSoma,
            types::object::ObjectType::Target => ObjectType::Target,
            types::object::ObjectType::Submission => ObjectType::Submission,
        };

        // Get contents without the ID prefix (ObjectData stores ID in first bytes)
        let contents = value.data.contents().to_vec();

        Ok(Self::new(
            object_id,
            version,
            object_type,
            value.owner.clone().into(),
            value.previous_transaction.into(),
            contents,
        ))
    }
}

impl TryFrom<Object> for types::object::Object {
    type Error = SdkTypeConversionError;

    fn try_from(value: Object) -> Result<Self, Self::Error> {
        // Map ObjectType from SDK to domain
        let object_type = match value.object_type {
            ObjectType::SystemState => types::object::ObjectType::SystemState,
            ObjectType::Coin => types::object::ObjectType::Coin,
            ObjectType::StakedSoma => types::object::ObjectType::StakedSoma,
            ObjectType::Target => types::object::ObjectType::Target,
            ObjectType::Submission => types::object::ObjectType::Submission,
        };

        // Create ObjectData with the ID prepended to contents
        let data = types::object::ObjectData::new_with_id(
            value.object_id.into(),
            object_type,
            types::object::Version::from_u64(value.version),
            value.contents,
        );

        Ok(types::object::Object::new(
            data,
            value.owner.into(),
            value.previous_transaction.into(),
        ))
    }
}

impl TryFrom<types::transaction::TransactionData> for Transaction {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::transaction::TransactionData) -> Result<Self, Self::Error> {
        Ok(Self {
            kind: value.kind.try_into()?,
            sender: value.sender.into(),
            gas_payment: value.gas_payment.into_iter().map(Into::into).collect(),
        })
    }
}

impl TryFrom<Transaction> for types::transaction::TransactionData {
    type Error = SdkTypeConversionError;

    fn try_from(value: Transaction) -> Result<Self, Self::Error> {
        Ok(types::transaction::TransactionData::new(
            value.kind.try_into()?,
            value.sender.into(),
            value.gas_payment.into_iter().map(Into::into).collect(),
        ))
    }
}

impl TryFrom<types::crypto::GenericSignature> for UserSignature {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::crypto::GenericSignature) -> Result<Self, Self::Error> {
        match value {
            types::crypto::GenericSignature::Signature(sig) => {
                match sig {
                    types::crypto::Signature::Ed25519SomaSignature(ed25519_sig) => {
                        // Extract signature and public key bytes
                        let sig_bytes = ed25519_sig.signature_bytes();
                        let pk_bytes = ed25519_sig.public_key_bytes();

                        // Create SDK Ed25519Signature and Ed25519PublicKey
                        let signature = Ed25519Signature::from_bytes(sig_bytes)?;
                        let public_key = Ed25519PublicKey::from_bytes(pk_bytes)?;

                        Ok(UserSignature::Simple(SimpleSignature::Ed25519 {
                            signature,
                            public_key,
                        }))
                    }
                }
            }
            types::crypto::GenericSignature::MultiSig(sig) => {
                Ok(UserSignature::Multisig(sig.into()))
            }
        }
    }
}

impl TryFrom<UserSignature> for types::crypto::GenericSignature {
    type Error = SdkTypeConversionError;

    fn try_from(value: UserSignature) -> Result<Self, Self::Error> {
        match value {
            UserSignature::Simple(simple) => {
                // Convert SimpleSignature to types::crypto::Signature
                let signature = match simple {
                    SimpleSignature::Ed25519 {
                        signature,
                        public_key,
                    } => {
                        // Create the Ed25519SomaSignature format: [flag][signature][public_key]
                        let mut full_bytes = Vec::with_capacity(
                            1 + Ed25519Signature::LENGTH + Ed25519PublicKey::LENGTH,
                        );
                        full_bytes.push(SignatureScheme::Ed25519.to_u8());
                        full_bytes.extend_from_slice(signature.as_ref());
                        full_bytes.extend_from_slice(public_key.as_ref());

                        types::crypto::Ed25519SomaSignature::from_bytes(&full_bytes)
                            .map(types::crypto::Signature::Ed25519SomaSignature)
                            .map_err(|e| {
                                SdkTypeConversionError(format!("Invalid Ed25519 signature: {}", e))
                            })?
                    }
                };

                Ok(types::crypto::GenericSignature::Signature(signature))
            }
            UserSignature::Multisig(multisig) => {
                Ok(types::crypto::GenericSignature::MultiSig(multisig.into()))
            }
        }
    }
}

impl From<MultisigMemberPublicKey> for types::crypto::PublicKey {
    fn from(value: MultisigMemberPublicKey) -> Self {
        match value {
            MultisigMemberPublicKey::Ed25519(pk) => {
                types::crypto::PublicKey::Ed25519(BytesRepresentation(pk.into_inner()))
            }
        }
    }
}

impl From<MultisigMemberSignature> for types::crypto::CompressedSignature {
    fn from(value: MultisigMemberSignature) -> Self {
        match value {
            MultisigMemberSignature::Ed25519(sig) => {
                types::crypto::CompressedSignature::Ed25519(BytesRepresentation(sig.into_inner()))
            }
        }
    }
}

impl From<types::multisig::MultiSig> for MultisigAggregatedSignature {
    fn from(value: types::multisig::MultiSig) -> Self {
        // Convert compressed signatures to SDK member signatures
        let signatures: Vec<MultisigMemberSignature> = value
            .get_sigs()
            .iter()
            .cloned()
            .map(|sig| {
                sig.try_into()
                    .expect("CompressedSignature conversion should not fail")
            })
            .collect();

        let bitmap = value.get_bitmap();

        // Convert MultiSigPublicKey to MultisigCommittee
        let multisig_pk = value.get_pk();
        let members: Vec<MultisigMember> = multisig_pk
            .pubkeys()
            .iter()
            .map(|(pk, weight)| {
                let public_key: MultisigMemberPublicKey = pk
                    .clone()
                    .try_into()
                    .expect("PublicKey conversion should not fail");
                MultisigMember::new(public_key, *weight)
            })
            .collect();

        let committee = MultisigCommittee::new(members, *multisig_pk.threshold());

        MultisigAggregatedSignature::new(committee, signatures, bitmap)
    }
}

impl From<MultisigAggregatedSignature> for types::multisig::MultiSig {
    fn from(value: MultisigAggregatedSignature) -> Self {
        // Convert SDK member signatures to domain compressed signatures
        let sigs: Vec<types::crypto::CompressedSignature> =
            value.signatures().iter().cloned().map(Into::into).collect();

        let bitmap = value.bitmap();

        // Convert MultisigCommittee to MultiSigPublicKey
        let committee = value.committee();
        let pk_map: Vec<(types::crypto::PublicKey, types::multisig::WeightUnit)> = committee
            .members()
            .iter()
            .map(|member| {
                let pk: types::crypto::PublicKey = member.public_key().clone().into();
                (pk, member.weight())
            })
            .collect();

        let multisig_pk =
            types::multisig::MultiSigPublicKey::insecure_new(pk_map, committee.threshold());

        types::multisig::MultiSig::insecure_new(sigs, bitmap, multisig_pk)
    }
}

// Helper conversion for SimpleSignature to types::crypto::Signature
impl TryFrom<SimpleSignature> for types::crypto::Signature {
    type Error = SdkTypeConversionError;

    fn try_from(value: SimpleSignature) -> Result<Self, Self::Error> {
        match value {
            SimpleSignature::Ed25519 {
                signature,
                public_key,
            } => {
                // Combine signature and public key bytes with the scheme flag
                let mut bytes = Vec::with_capacity(
                    1 + signature.as_bytes().len() + public_key.as_bytes().len(),
                );
                bytes.push(types::crypto::SignatureScheme::ED25519.flag());
                bytes.extend_from_slice(signature.as_bytes());
                bytes.extend_from_slice(public_key.as_bytes());

                types::crypto::Ed25519SomaSignature::from_bytes(&bytes)
                    .map(types::crypto::Signature::Ed25519SomaSignature)
                    .map_err(|e| SdkTypeConversionError(e.to_string()))
            }
            _ => Err(SdkTypeConversionError(
                "Unsupported signature scheme".to_string(),
            )),
        }
    }
}

impl TryFrom<types::transaction::TransactionKind> for TransactionKind {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::transaction::TransactionKind) -> Result<Self, Self::Error> {
        use types::transaction::TransactionKind as TK;

        Ok(match value {
            TK::Genesis(genesis) => TransactionKind::Genesis(GenesisTransaction {
                objects: genesis
                    .objects
                    .into_iter()
                    .map(TryInto::try_into)
                    .collect::<Result<_, _>>()?,
            }),

            TK::ConsensusCommitPrologue(prologue) => {
                TransactionKind::ConsensusCommitPrologue(ConsensusCommitPrologue {
                    epoch: prologue.epoch,
                    round: prologue.round,
                    commit_timestamp_ms: prologue.commit_timestamp_ms,
                    sub_dag_index: prologue.sub_dag_index,
                    consensus_commit_digest: prologue.consensus_commit_digest.into_inner().into(),
                    additional_state_digest: prologue.additional_state_digest.into_inner().into(),
                })
            }

            TK::ChangeEpoch(change) => TransactionKind::ChangeEpoch(ChangeEpoch {
                epoch: change.epoch,
                protocol_version: change.protocol_version.as_u64(),
                fees: change.fees,
                epoch_start_timestamp_ms: change.epoch_start_timestamp_ms,
                epoch_randomness: change.epoch_randomness,
            }),

            // Validator operations
            TK::AddValidator(args) => TransactionKind::AddValidator(AddValidatorArgs {
                pubkey_bytes: args.pubkey_bytes,
                network_pubkey_bytes: args.network_pubkey_bytes,
                worker_pubkey_bytes: args.worker_pubkey_bytes,
                net_address: args.net_address,
                p2p_address: args.p2p_address,
                primary_address: args.primary_address,
            }),

            TK::RemoveValidator(args) => TransactionKind::RemoveValidator(RemoveValidatorArgs {
                pubkey_bytes: args.pubkey_bytes,
            }),

            TK::ReportValidator { reportee } => TransactionKind::ReportValidator {
                reportee: reportee.into(),
            },

            TK::UndoReportValidator { reportee } => TransactionKind::UndoReportValidator {
                reportee: reportee.into(),
            },

            TK::UpdateValidatorMetadata(args) => {
                TransactionKind::UpdateValidatorMetadata(UpdateValidatorMetadataArgs {
                    next_epoch_network_address: args.next_epoch_network_address,
                    next_epoch_p2p_address: args.next_epoch_p2p_address,
                    next_epoch_primary_address: args.next_epoch_primary_address,
                    next_epoch_protocol_pubkey: args.next_epoch_protocol_pubkey,
                    next_epoch_worker_pubkey: args.next_epoch_worker_pubkey,
                    next_epoch_network_pubkey: args.next_epoch_network_pubkey,
                })
            }

            TK::SetCommissionRate { new_rate } => TransactionKind::SetCommissionRate { new_rate },

            // Transfer operations
            TK::TransferCoin {
                coin,
                amount,
                recipient,
            } => TransactionKind::TransferCoin {
                coin: coin.into(),
                amount,
                recipient: recipient.into(),
            },

            TK::PayCoins {
                coins,
                amounts,
                recipients,
            } => TransactionKind::PayCoins {
                coins: coins.into_iter().map(Into::into).collect(),
                amounts,
                recipients: recipients.into_iter().map(Into::into).collect(),
            },

            TK::TransferObjects { objects, recipient } => TransactionKind::TransferObjects {
                objects: objects.into_iter().map(Into::into).collect(),
                recipient: recipient.into(),
            },

            // Staking operations
            TK::AddStake {
                address,
                coin_ref,
                amount,
            } => TransactionKind::AddStake {
                address: address.into(),
                coin_ref: coin_ref.into(),
                amount,
            },

            TK::WithdrawStake { staked_soma } => TransactionKind::WithdrawStake {
                staked_soma: staked_soma.into(),
            },

            // Model transactions
            TK::CommitModel(args) => TransactionKind::CommitModel(CommitModelArgs {
                model_id: args.model_id.into(),
                weights_url_commitment: args.weights_url_commitment.into_inner().to_vec(),
                weights_commitment: args.weights_commitment.into_inner().to_vec(),
                architecture_version: args.architecture_version,
                stake_amount: args.stake_amount,
                commission_rate: args.commission_rate,
                staking_pool_id: args.staking_pool_id.into(),
            }),

            TK::RevealModel(args) => TransactionKind::RevealModel(RevealModelArgs {
                model_id: args.model_id.into(),
                weights_manifest: args.weights_manifest.into(),
            }),

            TK::CommitModelUpdate(args) => {
                TransactionKind::CommitModelUpdate(CommitModelUpdateArgs {
                    model_id: args.model_id.into(),
                    weights_url_commitment: args.weights_url_commitment.into_inner().to_vec(),
                    weights_commitment: args.weights_commitment.into_inner().to_vec(),
                })
            }

            TK::RevealModelUpdate(args) => {
                TransactionKind::RevealModelUpdate(RevealModelUpdateArgs {
                    model_id: args.model_id.into(),
                    weights_manifest: args.weights_manifest.into(),
                })
            }

            TK::AddStakeToModel {
                model_id,
                coin_ref,
                amount,
            } => TransactionKind::AddStakeToModel {
                model_id: model_id.into(),
                coin_ref: coin_ref.into(),
                amount,
            },

            TK::SetModelCommissionRate { model_id, new_rate } => {
                TransactionKind::SetModelCommissionRate {
                    model_id: model_id.into(),
                    new_rate,
                }
            }

            TK::DeactivateModel { model_id } => TransactionKind::DeactivateModel {
                model_id: model_id.into(),
            },

            TK::ReportModel { model_id } => TransactionKind::ReportModel {
                model_id: model_id.into(),
            },

            TK::UndoReportModel { model_id } => TransactionKind::UndoReportModel {
                model_id: model_id.into(),
            },

            // Submission transactions
            TK::SubmitData(args) => TransactionKind::SubmitData(SubmitDataArgs {
                target_id: args.target_id.into(),
                data_commitment: args.data_commitment.into_inner().to_vec(),
                data_manifest: SubmissionManifest {
                    manifest: args.data_manifest.manifest.into(),
                },
                model_id: args.model_id.into(),
                embedding: args.embedding.to_vec(),
                distance_score: args.distance_score,
                reconstruction_score: args.reconstruction_score,
                bond_coin: args.bond_coin.into(),
            }),

            TK::ClaimRewards(args) => TransactionKind::ClaimRewards(ClaimRewardsArgs {
                target_id: args.target_id.into(),
            }),
        })
    }
}

impl TryFrom<TransactionKind> for types::transaction::TransactionKind {
    type Error = SdkTypeConversionError;

    fn try_from(value: TransactionKind) -> Result<Self, Self::Error> {
        use types::transaction::TransactionKind as TK;

        Ok(match value {
            TransactionKind::Genesis(genesis) => {
                TK::Genesis(types::transaction::GenesisTransaction {
                    objects: genesis
                        .objects
                        .into_iter()
                        .map(TryInto::try_into)
                        .collect::<Result<_, _>>()?,
                })
            }

            TransactionKind::ConsensusCommitPrologue(prologue) => {
                TK::ConsensusCommitPrologue(types::consensus::ConsensusCommitPrologue {
                    epoch: prologue.epoch,
                    round: prologue.round,
                    sub_dag_index: prologue.sub_dag_index,
                    commit_timestamp_ms: prologue.commit_timestamp_ms,
                    consensus_commit_digest: prologue.consensus_commit_digest.into_inner().into(),
                    additional_state_digest: prologue.additional_state_digest.into_inner().into(),
                })
            }

            TransactionKind::ChangeEpoch(change) => {
                TK::ChangeEpoch(types::transaction::ChangeEpoch {
                    epoch: change.epoch,
                    protocol_version: change.protocol_version.into(),
                    fees: change.fees,
                    epoch_start_timestamp_ms: change.epoch_start_timestamp_ms,
                    epoch_randomness: change.epoch_randomness,
                })
            }

            // Validator operations
            TransactionKind::AddValidator(args) => {
                TK::AddValidator(types::transaction::AddValidatorArgs {
                    pubkey_bytes: args.pubkey_bytes,
                    network_pubkey_bytes: args.network_pubkey_bytes,
                    worker_pubkey_bytes: args.worker_pubkey_bytes,
                    net_address: args.net_address,
                    p2p_address: args.p2p_address,
                    primary_address: args.primary_address,
                })
            }

            TransactionKind::RemoveValidator(args) => {
                TK::RemoveValidator(types::transaction::RemoveValidatorArgs {
                    pubkey_bytes: args.pubkey_bytes,
                })
            }

            TransactionKind::ReportValidator { reportee } => TK::ReportValidator {
                reportee: reportee.into(),
            },

            TransactionKind::UndoReportValidator { reportee } => TK::UndoReportValidator {
                reportee: reportee.into(),
            },

            TransactionKind::UpdateValidatorMetadata(args) => {
                TK::UpdateValidatorMetadata(types::transaction::UpdateValidatorMetadataArgs {
                    next_epoch_network_address: args.next_epoch_network_address,
                    next_epoch_p2p_address: args.next_epoch_p2p_address,
                    next_epoch_primary_address: args.next_epoch_primary_address,
                    next_epoch_protocol_pubkey: args.next_epoch_protocol_pubkey,
                    next_epoch_worker_pubkey: args.next_epoch_worker_pubkey,
                    next_epoch_network_pubkey: args.next_epoch_network_pubkey,
                })
            }

            TransactionKind::SetCommissionRate { new_rate } => TK::SetCommissionRate { new_rate },

            // Transfer operations
            TransactionKind::TransferCoin {
                coin,
                amount,
                recipient,
            } => TK::TransferCoin {
                coin: coin.into(),
                amount,
                recipient: recipient.into(),
            },

            TransactionKind::PayCoins {
                coins,
                amounts,
                recipients,
            } => TK::PayCoins {
                coins: coins.into_iter().map(Into::into).collect(),
                amounts,
                recipients: recipients.into_iter().map(Into::into).collect(),
            },

            TransactionKind::TransferObjects { objects, recipient } => TK::TransferObjects {
                objects: objects.into_iter().map(Into::into).collect(),
                recipient: recipient.into(),
            },

            // Staking operations
            TransactionKind::AddStake {
                address,
                coin_ref,
                amount,
            } => TK::AddStake {
                address: address.into(),
                coin_ref: coin_ref.into(),
                amount,
            },

            TransactionKind::WithdrawStake { staked_soma } => TK::WithdrawStake {
                staked_soma: staked_soma.into(),
            },

            // Model transactions
            TransactionKind::CommitModel(args) => {
                TK::CommitModel(types::transaction::CommitModelArgs {
                    model_id: args.model_id.into(),
                    weights_url_commitment: types::digests::ModelWeightsUrlCommitment::new(
                        args.weights_url_commitment
                            .try_into()
                            .map_err(|_| SdkTypeConversionError(
                                "weights_url_commitment must be 32 bytes".into(),
                            ))?,
                    ),
                    weights_commitment: types::digests::ModelWeightsCommitment::new(
                        args.weights_commitment
                            .try_into()
                            .map_err(|_| SdkTypeConversionError(
                                "weights_commitment must be 32 bytes".into(),
                            ))?,
                    ),
                    architecture_version: args.architecture_version,
                    stake_amount: args.stake_amount,
                    commission_rate: args.commission_rate,
                    staking_pool_id: args.staking_pool_id.into(),
                })
            }

            TransactionKind::RevealModel(args) => {
                TK::RevealModel(types::transaction::RevealModelArgs {
                    model_id: args.model_id.into(),
                    weights_manifest: args.weights_manifest.try_into()?,
                })
            }

            TransactionKind::CommitModelUpdate(args) => {
                TK::CommitModelUpdate(types::transaction::CommitModelUpdateArgs {
                    model_id: args.model_id.into(),
                    weights_url_commitment: types::digests::ModelWeightsUrlCommitment::new(
                        args.weights_url_commitment
                            .try_into()
                            .map_err(|_| SdkTypeConversionError(
                                "weights_url_commitment must be 32 bytes".into(),
                            ))?,
                    ),
                    weights_commitment: types::digests::ModelWeightsCommitment::new(
                        args.weights_commitment
                            .try_into()
                            .map_err(|_| SdkTypeConversionError(
                                "weights_commitment must be 32 bytes".into(),
                            ))?,
                    ),
                })
            }

            TransactionKind::RevealModelUpdate(args) => {
                TK::RevealModelUpdate(types::transaction::RevealModelUpdateArgs {
                    model_id: args.model_id.into(),
                    weights_manifest: args.weights_manifest.try_into()?,
                })
            }

            TransactionKind::AddStakeToModel {
                model_id,
                coin_ref,
                amount,
            } => TK::AddStakeToModel {
                model_id: model_id.into(),
                coin_ref: coin_ref.into(),
                amount,
            },

            TransactionKind::SetModelCommissionRate { model_id, new_rate } => {
                TK::SetModelCommissionRate {
                    model_id: model_id.into(),
                    new_rate,
                }
            }

            TransactionKind::DeactivateModel { model_id } => TK::DeactivateModel {
                model_id: model_id.into(),
            },

            TransactionKind::ReportModel { model_id } => TK::ReportModel {
                model_id: model_id.into(),
            },

            TransactionKind::UndoReportModel { model_id } => TK::UndoReportModel {
                model_id: model_id.into(),
            },

            // Submission transactions
            TransactionKind::SubmitData(args) => {
                let data_commitment_array: [u8; 32] = args
                    .data_commitment
                    .try_into()
                    .map_err(|_| SdkTypeConversionError("data_commitment must be 32 bytes".into()))?;
                let data_manifest = types::submission::SubmissionManifest::new(args.data_manifest.manifest.try_into()?);
                let embedding = ndarray::Array1::from_vec(args.embedding);

                TK::SubmitData(types::transaction::SubmitDataArgs {
                    target_id: args.target_id.into(),
                    // target_initial_shared_version is ignored from proto - use protocol constant
                    data_commitment: types::digests::DataCommitment::new(data_commitment_array),
                    data_manifest,
                    model_id: args.model_id.into(),
                    embedding,
                    distance_score: args.distance_score,
                    reconstruction_score: args.reconstruction_score,
                    bond_coin: args.bond_coin.into(),
                })
            }

            TransactionKind::ClaimRewards(args) => TK::ClaimRewards(types::transaction::ClaimRewardsArgs {
                target_id: args.target_id.into(),
                // target_initial_shared_version is ignored from proto - use protocol constant
            }),
        })
    }
}

impl TryFrom<types::effects::TransactionEffects> for crate::types::TransactionEffects {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::effects::TransactionEffects) -> Result<Self, Self::Error> {
        Ok(Self {
            status: value.status.into(),
            epoch: value.executed_epoch,
            fee: value.transaction_fee.into(),
            transaction_digest: value.transaction_digest.into(),
            dependencies: value.dependencies.into_iter().map(Into::into).collect(),
            lamport_version: value.version.value(),
            gas_object_index: value.gas_object_index,
            changed_objects: value
                .changed_objects
                .into_iter()
                .map(|(object_id, change)| crate::types::ChangedObject {
                    object_id: object_id.into(),
                    input_state: change.input_state.into(),
                    output_state: change.output_state.into(),
                    id_operation: change.id_operation.into(),
                })
                .collect(),
            unchanged_shared_objects: value
                .unchanged_shared_objects
                .into_iter()
                .map(|(object_id, kind)| crate::types::UnchangedSharedObject {
                    object_id: object_id.into(),
                    kind: kind.into(),
                })
                .collect(),
        })
    }
}

impl TryFrom<crate::types::TransactionEffects> for types::effects::TransactionEffects {
    type Error = SdkTypeConversionError;

    fn try_from(value: crate::types::TransactionEffects) -> Result<Self, Self::Error> {
        Ok(Self {
            status: value.status.into(),
            executed_epoch: value.epoch,
            transaction_fee: value.fee.into(),
            transaction_digest: value.transaction_digest.into(),
            dependencies: value.dependencies.into_iter().map(Into::into).collect(),
            version: types::object::Version::from_u64(value.lamport_version),
            gas_object_index: value.gas_object_index,
            changed_objects: value
                .changed_objects
                .into_iter()
                .map(|object| {
                    (
                        object.object_id.into(),
                        types::effects::object_change::EffectsObjectChange {
                            input_state: object.input_state.into(),
                            output_state: object.output_state.into(),
                            id_operation: object.id_operation.into(),
                        },
                    )
                })
                .collect(),
            unchanged_shared_objects: value
                .unchanged_shared_objects
                .into_iter()
                .map(|object| (object.object_id.into(), object.kind.into()))
                .collect(),
        })
    }
}

impl From<types::tx_fee::TransactionFee> for crate::types::TransactionFee {
    fn from(value: types::tx_fee::TransactionFee) -> Self {
        Self {
            base_fee: value.base_fee,
            operation_fee: value.operation_fee,
            value_fee: value.value_fee,
            total_fee: value.total_fee,
        }
    }
}

impl From<crate::types::TransactionFee> for types::tx_fee::TransactionFee {
    fn from(value: crate::types::TransactionFee) -> Self {
        Self {
            base_fee: value.base_fee,
            operation_fee: value.operation_fee,
            value_fee: value.value_fee,
            total_fee: value.total_fee,
        }
    }
}

impl From<ObjectReference> for types::object::ObjectRef {
    fn from(value: ObjectReference) -> Self {
        (
            (*value.object_id()).into(),
            types::object::Version::from_u64(value.version()),
            (*value.digest()).into(),
        )
    }
}

impl From<types::object::ObjectRef> for ObjectReference {
    fn from(value: types::object::ObjectRef) -> Self {
        ObjectReference::new((*value.0).into(), value.1.value(), (value.2).into())
    }
}

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
        let inner = value.into_inner();

        Ok(Self {
            transaction: inner.intent_message.value.try_into()?,
            signatures: inner
                .tx_signatures
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        })
    }
}

impl TryFrom<SignedTransaction> for types::transaction::SenderSignedData {
    type Error = SdkTypeConversionError;

    fn try_from(value: SignedTransaction) -> Result<Self, Self::Error> {
        let tx_data: types::transaction::TransactionData = value.transaction.try_into()?;

        let signatures: Vec<types::crypto::GenericSignature> = value
            .signatures
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<_, _>>()?;

        Ok(types::transaction::SenderSignedData::new(
            tx_data, signatures,
        ))
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

impl From<types::consensus::commit::CommitDigest> for Digest {
    fn from(value: types::consensus::commit::CommitDigest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<types::digests::CheckpointDigest> for Digest {
    fn from(value: types::digests::CheckpointDigest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<Digest> for types::digests::CheckpointDigest {
    fn from(value: Digest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<types::digests::CheckpointArtifactsDigest> for Digest {
    fn from(value: types::digests::CheckpointArtifactsDigest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<Digest> for types::digests::CheckpointArtifactsDigest {
    fn from(value: Digest) -> Self {
        Self::new(value.into_inner())
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

impl From<ObjectIn> for types::effects::object_change::ObjectIn {
    fn from(value: ObjectIn) -> Self {
        match value {
            ObjectIn::NotExist => Self::NotExist,
            ObjectIn::Exist {
                version,
                digest,
                owner,
            } => Self::Exist((
                (types::object::Version::from_u64(version), digest.into()),
                owner.into(),
            )),
        }
    }
}

impl From<ObjectOut> for types::effects::object_change::ObjectOut {
    fn from(value: ObjectOut) -> Self {
        match value {
            ObjectOut::NotExist => Self::NotExist,
            ObjectOut::ObjectWrite { digest, owner } => {
                Self::ObjectWrite((digest.into(), owner.into()))
            }
        }
    }
}

impl From<IdOperation> for types::effects::object_change::IDOperation {
    fn from(value: IdOperation) -> Self {
        match value {
            IdOperation::None => Self::None,
            IdOperation::Created => Self::Created,
            IdOperation::Deleted => Self::Deleted,
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
            types::effects::ExecutionFailureStatus::ModelNotFound => Self::ModelNotFound,
            types::effects::ExecutionFailureStatus::NotModelOwner => Self::NotModelOwner,
            types::effects::ExecutionFailureStatus::ModelNotActive => Self::ModelNotActive,
            types::effects::ExecutionFailureStatus::ModelNotPending => Self::ModelNotPending,
            types::effects::ExecutionFailureStatus::ModelAlreadyInactive => Self::ModelAlreadyInactive,
            types::effects::ExecutionFailureStatus::ModelRevealEpochMismatch => Self::ModelRevealEpochMismatch,
            types::effects::ExecutionFailureStatus::ModelWeightsUrlMismatch => Self::ModelWeightsUrlMismatch,
            types::effects::ExecutionFailureStatus::ModelNoPendingUpdate => Self::ModelNoPendingUpdate,
            types::effects::ExecutionFailureStatus::ModelArchitectureVersionMismatch => Self::ModelArchitectureVersionMismatch,
            types::effects::ExecutionFailureStatus::ModelCommissionRateTooHigh => Self::ModelCommissionRateTooHigh,
            types::effects::ExecutionFailureStatus::ModelMinStakeNotMet => Self::ModelMinStakeNotMet,

            // Target errors
            types::effects::ExecutionFailureStatus::NoActiveModels => Self::NoActiveModels,
            types::effects::ExecutionFailureStatus::TargetNotFound => Self::TargetNotFound,
            types::effects::ExecutionFailureStatus::TargetNotOpen => Self::TargetNotOpen,
            types::effects::ExecutionFailureStatus::TargetExpired {
                generation_epoch,
                current_epoch,
            } => Self::TargetExpired {
                generation_epoch,
                current_epoch,
            },
            types::effects::ExecutionFailureStatus::TargetNotFilled => Self::TargetNotFilled,
            types::effects::ExecutionFailureStatus::ChallengeWindowOpen {
                fill_epoch,
                current_epoch,
            } => Self::ChallengeWindowOpen {
                fill_epoch,
                current_epoch,
            },
            types::effects::ExecutionFailureStatus::TargetAlreadyClaimed => Self::TargetAlreadyClaimed,

            // Submission errors
            types::effects::ExecutionFailureStatus::ModelNotInTarget { model_id, target_id } => {
                Self::ModelNotInTarget {
                    model_id: model_id.into(),
                    target_id: target_id.into(),
                }
            }
            types::effects::ExecutionFailureStatus::EmbeddingDimensionMismatch { expected, actual } => {
                Self::EmbeddingDimensionMismatch { expected, actual }
            }
            types::effects::ExecutionFailureStatus::DistanceExceedsThreshold { score, threshold } => {
                Self::DistanceExceedsThreshold { score, threshold }
            }
            types::effects::ExecutionFailureStatus::ReconstructionExceedsThreshold { score, threshold } => {
                Self::ReconstructionExceedsThreshold { score, threshold }
            }
            types::effects::ExecutionFailureStatus::InsufficientBond { required, provided } => {
                Self::InsufficientBond { required, provided }
            }
            types::effects::ExecutionFailureStatus::InsufficientEmissionBalance => {
                Self::InsufficientEmissionBalance
            }

            types::effects::ExecutionFailureStatus::InsufficientCoinBalance => {
                Self::InsufficientCoinBalance
            }
            types::effects::ExecutionFailureStatus::CoinBalanceOverflow => {
                Self::CoinBalanceOverflow
            }
            types::effects::ExecutionFailureStatus::ValidatorNotFound => Self::ValidatorNotFound,
            types::effects::ExecutionFailureStatus::StakingPoolNotFound => {
                Self::StakingPoolNotFound
            }
            types::effects::ExecutionFailureStatus::CannotReportOneself => {
                Self::CannotReportOneself
            }
            types::effects::ExecutionFailureStatus::ReportRecordNotFound => {
                Self::ReportRecordNotFound
            }
            types::effects::ExecutionFailureStatus::InputObjectDeleted => {
                Self::InputObjectDeleted
            }
            types::effects::ExecutionFailureStatus::CertificateDenied => {
                Self::CertificateDenied
            }
            types::effects::ExecutionFailureStatus::ExecutionCancelledDueToSharedObjectCongestion => {
                Self::SharedObjectCongestion
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
            ExecutionError::InvalidOwnership { object_id } => Self::InvalidOwnership {
                object_id: object_id.into(),
                expected_owner: object_id.into(), // TODO: change this
                actual_owner: Some(object_id.into()), // TODO: change this
            },
            ExecutionError::ObjectNotFound { object_id } => Self::ObjectNotFound {
                object_id: object_id.into(),
            },
            ExecutionError::InvalidObjectType { object_id } => Self::InvalidObjectType {
                object_id: object_id.into(),
                expected_type: types::object::ObjectType::Coin, // TODO: change this
                actual_type: types::object::ObjectType::Coin,   // TODO: change this
            },
            ExecutionError::InvalidTransactionType => Self::InvalidTransactionType,
            ExecutionError::InvalidArguments { reason } => Self::InvalidArguments { reason },

            // Validator errors
            ExecutionError::DuplicateValidator => Self::DuplicateValidator,
            ExecutionError::NotAValidator => Self::NotAValidator,
            ExecutionError::ValidatorAlreadyRemoved => Self::ValidatorAlreadyRemoved,
            ExecutionError::AdvancedToWrongEpoch => Self::AdvancedToWrongEpoch,

            // Model errors
            ExecutionError::ModelNotFound => Self::ModelNotFound,
            ExecutionError::NotModelOwner => Self::NotModelOwner,
            ExecutionError::ModelNotActive => Self::ModelNotActive,
            ExecutionError::ModelNotPending => Self::ModelNotPending,
            ExecutionError::ModelAlreadyInactive => Self::ModelAlreadyInactive,
            ExecutionError::ModelRevealEpochMismatch => Self::ModelRevealEpochMismatch,
            ExecutionError::ModelWeightsUrlMismatch => Self::ModelWeightsUrlMismatch,
            ExecutionError::ModelNoPendingUpdate => Self::ModelNoPendingUpdate,
            ExecutionError::ModelArchitectureVersionMismatch => Self::ModelArchitectureVersionMismatch,
            ExecutionError::ModelCommissionRateTooHigh => Self::ModelCommissionRateTooHigh,
            ExecutionError::ModelMinStakeNotMet => Self::ModelMinStakeNotMet,

            // Target errors
            ExecutionError::NoActiveModels => Self::NoActiveModels,
            ExecutionError::TargetNotFound => Self::TargetNotFound,
            ExecutionError::TargetNotOpen => Self::TargetNotOpen,
            ExecutionError::TargetExpired {
                generation_epoch,
                current_epoch,
            } => Self::TargetExpired {
                generation_epoch,
                current_epoch,
            },
            ExecutionError::TargetNotFilled => Self::TargetNotFilled,
            ExecutionError::ChallengeWindowOpen {
                fill_epoch,
                current_epoch,
            } => Self::ChallengeWindowOpen {
                fill_epoch,
                current_epoch,
            },
            ExecutionError::TargetAlreadyClaimed => Self::TargetAlreadyClaimed,

            // Submission errors
            ExecutionError::ModelNotInTarget { model_id, target_id } => Self::ModelNotInTarget {
                model_id: model_id.into(),
                target_id: target_id.into(),
            },
            ExecutionError::EmbeddingDimensionMismatch { expected, actual } => {
                Self::EmbeddingDimensionMismatch { expected, actual }
            }
            ExecutionError::DistanceExceedsThreshold { score, threshold } => {
                Self::DistanceExceedsThreshold { score, threshold }
            }
            ExecutionError::ReconstructionExceedsThreshold { score, threshold } => {
                Self::ReconstructionExceedsThreshold { score, threshold }
            }
            ExecutionError::InsufficientBond { required, provided } => {
                Self::InsufficientBond { required, provided }
            }
            ExecutionError::InsufficientEmissionBalance => Self::InsufficientEmissionBalance,

            // Coin errors
            ExecutionError::InsufficientCoinBalance => Self::InsufficientCoinBalance,
            ExecutionError::CoinBalanceOverflow => Self::CoinBalanceOverflow,

            // Staking errors
            ExecutionError::ValidatorNotFound => Self::ValidatorNotFound,
            ExecutionError::StakingPoolNotFound => Self::StakingPoolNotFound,
            ExecutionError::CannotReportOneself => Self::CannotReportOneself,
            ExecutionError::ReportRecordNotFound => Self::ReportRecordNotFound,

            // Generic error for cases not covered by specific variants
            ExecutionError::OtherError(string) => {
                Self::SomaError(types::error::SomaError::from(string))
            }
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

impl From<ExecutionStatus> for types::effects::ExecutionStatus {
    fn from(value: ExecutionStatus) -> Self {
        match value {
            ExecutionStatus::Success => Self::Success,
            ExecutionStatus::Failure { error } => Self::Failure {
                error: error.into(),
            },
        }
    }
}

impl From<types::checkpoints::CheckpointCommitment> for CheckpointCommitment {
    fn from(value: types::checkpoints::CheckpointCommitment) -> Self {
        match value {
            types::checkpoints::CheckpointCommitment::ECMHLiveObjectSetDigest(digest) => {
                Self::EcmhLiveObjectSet {
                    digest: digest.digest.into(),
                }
            }
            types::checkpoints::CheckpointCommitment::CheckpointArtifactsDigest(digest) => {
                Self::CheckpointArtifacts {
                    digest: digest.into(),
                }
            }
        }
    }
}

impl TryFrom<types::crypto::PublicKey> for MultisigMemberPublicKey {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::crypto::PublicKey) -> Result<Self, Self::Error> {
        match value {
            types::crypto::PublicKey::Ed25519(bytes_representation) => {
                Self::Ed25519(Ed25519PublicKey::new(bytes_representation.0))
            }
        }
        .pipe(Ok)
    }
}

impl TryFrom<types::crypto::CompressedSignature> for MultisigMemberSignature {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::crypto::CompressedSignature) -> Result<Self, Self::Error> {
        match value {
            types::crypto::CompressedSignature::Ed25519(bytes_representation) => {
                Self::Ed25519(Ed25519Signature::new(bytes_representation.0))
            }
        }
        .pipe(Ok)
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
            types::crypto::SignatureScheme::MultiSig => Self::Multisig,
        }
    }
}

impl From<types::transaction::ChangeEpoch> for ChangeEpoch {
    fn from(
        types::transaction::ChangeEpoch {
            epoch,
            protocol_version,
            fees,
            epoch_start_timestamp_ms,
            epoch_randomness,
        }: types::transaction::ChangeEpoch,
    ) -> Self {
        Self {
            epoch,
            protocol_version: protocol_version.as_u64(),
            fees,
            epoch_start_timestamp_ms,
            epoch_randomness,
        }
    }
}

impl From<types::metadata::Metadata> for Metadata {
    fn from(value: types::metadata::Metadata) -> Self {
        match value {
            types::metadata::Metadata::V1(v1) => Metadata::V1(MetadataV1 {
                checksum: v1
                    .checksum()
                    .as_bytes()
                    .try_into()
                    .expect("checksum should be 32 bytes"),
                size: v1.size() as u64,
            }),
        }
    }
}

impl TryFrom<Metadata> for types::metadata::Metadata {
    type Error = SdkTypeConversionError;

    fn try_from(value: Metadata) -> Result<Self, Self::Error> {
        match value {
            Metadata::V1(v1) => {
                let checksum = types::checksum::Checksum::from_bytes(&v1.checksum)
                    .map_err(|e| SdkTypeConversionError(format!("Invalid checksum: {}", e)))?;

                let size = v1
                    .size
                    .try_into()
                    .map_err(|e| SdkTypeConversionError(format!("Invalid size: {}", e)))?;

                Ok(types::metadata::Metadata::V1(
                    types::metadata::MetadataV1::new(checksum, size),
                ))
            }
        }
    }
}

impl From<types::metadata::Manifest> for Manifest {
    fn from(value: types::metadata::Manifest) -> Self {
        match value {
            types::metadata::Manifest::V1(v1) => Manifest::V1(ManifestV1 {
                url: v1
                    .url().clone(),
                metadata: v1.metadata().clone().into(),
            }),
        }
    }
}

impl TryFrom<Manifest> for types::metadata::Manifest {
    type Error = SdkTypeConversionError;

    fn try_from(value: Manifest) -> Result<Self, Self::Error> {
        match value {
            Manifest::V1(v1) => {
                let url = v1.url.clone();

                let metadata = v1
                    .metadata
                    .try_into()
                    .map_err(|e| SdkTypeConversionError(format!("Invalid size: {}", e)))?;

                Ok(types::metadata::Manifest::V1(
                    types::metadata::ManifestV1::new(url, metadata),
                ))
            }
        }
    }
}

// ============================================================================
// ModelWeightsManifest conversions
// ============================================================================

impl From<types::model::ModelWeightsManifest> for ModelWeightsManifest {
    fn from(value: types::model::ModelWeightsManifest) -> Self {
        Self {
            manifest: value.manifest.into(),
            decryption_key: value.decryption_key.as_bytes().to_vec(),
        }
    }
}

impl TryFrom<ModelWeightsManifest> for types::model::ModelWeightsManifest {
    type Error = SdkTypeConversionError;

    fn try_from(value: ModelWeightsManifest) -> Result<Self, Self::Error> {
        let manifest = value.manifest.try_into()?;
        let key_array: [u8; 32] = value
            .decryption_key
            .try_into()
            .map_err(|_| SdkTypeConversionError("decryption_key must be 32 bytes".into()))?;
        Ok(Self {
            manifest,
            decryption_key: types::crypto::DecryptionKey::new(key_array),
        })
    }
}

// ============================================================================
// TransactionEffectsDigest conversions
// ============================================================================

impl From<types::digests::TransactionEffectsDigest> for Digest {
    fn from(value: types::digests::TransactionEffectsDigest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<Digest> for types::digests::TransactionEffectsDigest {
    fn from(value: Digest) -> Self {
        Self::new(value.into_inner())
    }
}

// ============================================================================
// CheckpointContentsDigest conversions
// ============================================================================

impl From<types::digests::CheckpointContentsDigest> for Digest {
    fn from(value: types::digests::CheckpointContentsDigest) -> Self {
        Self::new(value.into_inner())
    }
}

impl From<Digest> for types::digests::CheckpointContentsDigest {
    fn from(value: Digest) -> Self {
        Self::new(value.into_inner())
    }
}

// ============================================================================
// CheckpointCommitment: SDK -> Domain
// ============================================================================

impl From<crate::types::CheckpointCommitment> for types::checkpoints::CheckpointCommitment {
    fn from(value: crate::types::CheckpointCommitment) -> Self {
        match value {
            crate::types::CheckpointCommitment::EcmhLiveObjectSet { digest } => {
                Self::ECMHLiveObjectSetDigest(types::checkpoints::ECMHLiveObjectSetDigest {
                    digest: digest.into(),
                })
            }
            crate::types::CheckpointCommitment::CheckpointArtifacts { digest } => {
                Self::CheckpointArtifactsDigest(digest.into())
            }
        }
    }
}

// ============================================================================
// EndOfEpochData conversions
// ============================================================================
impl TryFrom<types::checkpoints::EndOfEpochData> for crate::types::EndOfEpochData {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::checkpoints::EndOfEpochData) -> Result<Self, Self::Error> {
        Ok(Self {
            next_epoch_validator_committee: value.next_epoch_validator_committee.into(),

            next_epoch_protocol_version: value.next_epoch_protocol_version.as_u64(),
            epoch_commitments: value
                .epoch_commitments
                .into_iter()
                .map(Into::into)
                .collect(),
        })
    }
}

impl TryFrom<crate::types::EndOfEpochData> for types::checkpoints::EndOfEpochData {
    type Error = SdkTypeConversionError;

    fn try_from(value: crate::types::EndOfEpochData) -> Result<Self, Self::Error> {
        let next_epoch_validator_committee: types::committee::Committee = value
            .next_epoch_validator_committee
            .try_into()
            .map_err(|e: ConversionError| SdkTypeConversionError(e.to_string()))?;

        Ok(Self {
            next_epoch_protocol_version: value.next_epoch_protocol_version.into(),
            next_epoch_validator_committee,

            epoch_commitments: value
                .epoch_commitments
                .into_iter()
                .map(Into::into)
                .collect(),
        })
    }
}

// ============================================================================
// CheckpointSummary conversions
// ============================================================================

impl TryFrom<types::checkpoints::CheckpointSummary> for crate::types::CheckpointSummary {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::checkpoints::CheckpointSummary) -> Result<Self, Self::Error> {
        Ok(Self {
            epoch: value.epoch,
            sequence_number: value.sequence_number,
            network_total_transactions: value.network_total_transactions,
            epoch_rolling_transaction_fees: value.epoch_rolling_transaction_fees.into(),
            content_digest: value.content_digest.into(),
            previous_digest: value.previous_digest.map(Into::into),
            timestamp_ms: value.timestamp_ms,
            checkpoint_commitments: value
                .checkpoint_commitments
                .into_iter()
                .map(Into::into)
                .collect(),
            end_of_epoch_data: value.end_of_epoch_data.map(TryInto::try_into).transpose()?,
        })
    }
}

impl TryFrom<crate::types::CheckpointSummary> for types::checkpoints::CheckpointSummary {
    type Error = SdkTypeConversionError;

    fn try_from(value: crate::types::CheckpointSummary) -> Result<Self, Self::Error> {
        Ok(Self {
            epoch: value.epoch,
            sequence_number: value.sequence_number,
            network_total_transactions: value.network_total_transactions,
            epoch_rolling_transaction_fees: value.epoch_rolling_transaction_fees.into(),
            content_digest: value.content_digest.into(),
            previous_digest: value.previous_digest.map(Into::into),
            timestamp_ms: value.timestamp_ms,
            checkpoint_commitments: value
                .checkpoint_commitments
                .into_iter()
                .map(Into::into)
                .collect(),
            end_of_epoch_data: value.end_of_epoch_data.map(TryInto::try_into).transpose()?,
        })
    }
}

// ============================================================================
// SignedCheckpointSummary (CertifiedCheckpointSummary) conversions
// ============================================================================

impl TryFrom<types::checkpoints::CertifiedCheckpointSummary>
    for crate::types::SignedCheckpointSummary
{
    type Error = SdkTypeConversionError;

    fn try_from(
        value: types::checkpoints::CertifiedCheckpointSummary,
    ) -> Result<Self, Self::Error> {
        let checkpoint: crate::types::CheckpointSummary = value.data().clone().try_into()?;
        let signature: ValidatorAggregatedSignature = value.auth_sig().clone().into();

        Ok(Self {
            checkpoint,
            signature,
        })
    }
}

impl TryFrom<crate::types::SignedCheckpointSummary>
    for types::checkpoints::CertifiedCheckpointSummary
{
    type Error = SdkTypeConversionError;

    fn try_from(value: crate::types::SignedCheckpointSummary) -> Result<Self, Self::Error> {
        let checkpoint: types::checkpoints::CheckpointSummary = value.checkpoint.try_into()?;
        let auth_sig: types::crypto::AuthorityStrongQuorumSignInfo = value.signature.into();

        Ok(
            types::checkpoints::CertifiedCheckpointSummary::new_from_data_and_sig(
                checkpoint, auth_sig,
            ),
        )
    }
}

// ============================================================================
// CheckpointContents conversions
// ============================================================================

impl TryFrom<types::checkpoints::CheckpointContents> for crate::types::CheckpointContents {
    type Error = SdkTypeConversionError;

    fn try_from(value: types::checkpoints::CheckpointContents) -> Result<Self, Self::Error> {
        let transactions: Vec<crate::types::CheckpointTransactionInfo> = value
            .into_iter_with_signatures()
            .map(|(digests, signatures)| {
                let user_signatures: Result<Vec<_>, _> =
                    signatures.into_iter().map(TryInto::try_into).collect();

                Ok(crate::types::CheckpointTransactionInfo {
                    transaction: digests.transaction.into(),
                    effects: digests.effects.into(),
                    signatures: user_signatures?,
                })
            })
            .collect::<Result<_, SdkTypeConversionError>>()?;

        Ok(crate::types::CheckpointContents::new(transactions))
    }
}

impl TryFrom<crate::types::CheckpointContents> for types::checkpoints::CheckpointContents {
    type Error = SdkTypeConversionError;

    fn try_from(value: crate::types::CheckpointContents) -> Result<Self, Self::Error> {
        let transactions = value.transactions();

        let execution_digests: Vec<types::base::ExecutionDigests> = transactions
            .iter()
            .map(|info| types::base::ExecutionDigests {
                transaction: info.transaction.into(),
                effects: info.effects.into(),
            })
            .collect();

        let user_signatures: Result<Vec<Vec<types::crypto::GenericSignature>>, _> = transactions
            .iter()
            .map(|info| {
                info.signatures
                    .iter()
                    .cloned()
                    .map(TryInto::try_into)
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect();

        Ok(
            types::checkpoints::CheckpointContents::new_with_digests_and_signatures(
                execution_digests,
                user_signatures?,
            ),
        )
    }
}

// ============================================================================
// CheckpointTransaction conversions
// ============================================================================

impl TryFrom<types::full_checkpoint_content::CheckpointTransaction>
    for crate::types::CheckpointTransaction
{
    type Error = SdkTypeConversionError;

    fn try_from(
        value: types::full_checkpoint_content::CheckpointTransaction,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            transaction: value.transaction.try_into()?,
            effects: value.effects.try_into()?,
            input_objects: value
                .input_objects
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
            output_objects: value
                .output_objects
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        })
    }
}

impl TryFrom<crate::types::CheckpointTransaction>
    for types::full_checkpoint_content::CheckpointTransaction
{
    type Error = SdkTypeConversionError;

    fn try_from(value: crate::types::CheckpointTransaction) -> Result<Self, Self::Error> {
        Ok(Self {
            transaction: value.transaction.try_into()?,
            effects: value.effects.try_into()?,
            input_objects: value
                .input_objects
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
            output_objects: value
                .output_objects
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        })
    }
}

// ============================================================================
// CheckpointData conversions
// ============================================================================

impl TryFrom<types::full_checkpoint_content::CheckpointData> for crate::types::CheckpointData {
    type Error = SdkTypeConversionError;

    fn try_from(
        value: types::full_checkpoint_content::CheckpointData,
    ) -> Result<Self, Self::Error> {
        Ok(Self {
            checkpoint_summary: value.checkpoint_summary.try_into()?,
            checkpoint_contents: value.checkpoint_contents.try_into()?,
            transactions: value
                .transactions
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        })
    }
}

impl TryFrom<crate::types::CheckpointData> for types::full_checkpoint_content::CheckpointData {
    type Error = SdkTypeConversionError;

    fn try_from(value: crate::types::CheckpointData) -> Result<Self, Self::Error> {
        Ok(Self {
            checkpoint_summary: value.checkpoint_summary.try_into()?,
            checkpoint_contents: value.checkpoint_contents.try_into()?,
            transactions: value
                .transactions
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        })
    }
}
