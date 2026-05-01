// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use tap::Pipe;
use url::Url;

use super::*;
use crate::proto::TryFromProtoError;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;

impl From<crate::types::Metadata> for Metadata {
    fn from(value: crate::types::Metadata) -> Self {
        use metadata::Version;

        let mut message = Self::default();
        match value {
            crate::types::Metadata::V1(v1) => {
                let proto_v1 =
                    MetadataV1 { checksum: Some(v1.checksum.into()), size: Some(v1.size) };
                message.version = Some(Version::V1(proto_v1));
            }
        }
        message
    }
}

impl TryFrom<&Metadata> for crate::types::Metadata {
    type Error = TryFromProtoError;

    fn try_from(value: &Metadata) -> Result<Self, Self::Error> {
        use metadata::Version;

        match value
            .version
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("metadata version"))?
        {
            Version::V1(v1) => {
                let checksum = v1
                    .checksum
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("checksum"))?
                    .to_vec();

                let size = v1.size.ok_or_else(|| TryFromProtoError::missing("size"))?;

                Ok(crate::types::Metadata::V1(crate::types::MetadataV1 { checksum, size }))
            }
        }
    }
}

impl From<crate::types::Manifest> for Manifest {
    fn from(value: crate::types::Manifest) -> Self {
        use manifest::Version;

        let mut message = Self::default();
        match value {
            crate::types::Manifest::V1(v1) => {
                let proto_v1 =
                    ManifestV1 { url: Some(v1.url.into()), metadata: Some(v1.metadata.into()) };
                message.version = Some(Version::V1(proto_v1));
            }
        }
        message
    }
}

impl TryFrom<&Manifest> for crate::types::Manifest {
    type Error = TryFromProtoError;

    fn try_from(value: &Manifest) -> Result<Self, Self::Error> {
        use manifest::Version;

        match value
            .version
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("Manifest version"))?
        {
            Version::V1(v1) => {
                let url = Url::parse(
                    v1.url.clone().ok_or_else(|| TryFromProtoError::missing("url"))?.as_str(),
                )
                .map_err(|e| TryFromProtoError::invalid("url", e))?;

                let metadata =
                    &v1.metadata.clone().ok_or_else(|| TryFromProtoError::missing("metadata"))?;

                let metadata = metadata.try_into().map_err(|_| {
                    TryFromProtoError::invalid("checksum", "invalid checksum length")
                })?;

                Ok(crate::types::Manifest::V1(crate::types::ManifestV1 { url, metadata }))
            }
        }
    }
}

//
// Transaction
//

impl From<crate::types::Transaction> for Transaction {
    fn from(value: crate::types::Transaction) -> Self {
        Self::merge_from(value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<crate::types::Transaction> for Transaction {
    fn merge(&mut self, source: crate::types::Transaction, mask: &FieldMaskTree) {
        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = Some(source.digest().to_string());
        }

        if mask.contains(Self::KIND_FIELD.name) {
            self.kind = Some(source.kind.into());
        }

        if mask.contains(Self::SENDER_FIELD.name) {
            self.sender = Some(source.sender.to_string());
        }

        if mask.contains(Self::GAS_PAYMENT_FIELD.name) {
            self.gas_payment = source.gas_payment.into_iter().map(Into::into).collect();
        }
    }
}

impl Merge<&Transaction> for Transaction {
    fn merge(&mut self, source: &Transaction, mask: &FieldMaskTree) {
        let Transaction { digest, kind, sender, gas_payment } = source;

        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = digest.clone();
        }

        if mask.contains(Self::KIND_FIELD.name) {
            self.kind = kind.clone();
        }

        if mask.contains(Self::SENDER_FIELD.name) {
            self.sender = sender.clone();
        }

        if mask.contains(Self::GAS_PAYMENT_FIELD.name) {
            self.gas_payment = gas_payment.clone();
        }
    }
}

impl TryFrom<&Transaction> for crate::types::Transaction {
    type Error = TryFromProtoError;

    fn try_from(value: &Transaction) -> Result<Self, Self::Error> {
        // if let Some(bcs) = &value.bcs {
        //     return bcs
        //         .deserialize()
        //         .map_err(|e| TryFromProtoError::invalid(Transaction::BCS_FIELD, e));
        // }

        let kind =
            value.kind.as_ref().ok_or_else(|| TryFromProtoError::missing("kind"))?.try_into()?;

        let sender = value
            .sender
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("sender"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(Transaction::SENDER_FIELD, e))?;

        let gas_payment =
            value.gas_payment.iter().map(TryInto::try_into).collect::<Result<_, _>>()?;

        Ok(Self { kind, sender, gas_payment })
    }
}

//
// TransactionKind
//
// Converting from domain types to protobuf
impl From<crate::types::TransactionKind> for TransactionKind {
    fn from(value: crate::types::TransactionKind) -> Self {
        use transaction_kind::Kind;

        use crate::types::TransactionKind::*;

        let kind = match value {
            Genesis(genesis) => Kind::Genesis(genesis.into()),
            ConsensusCommitPrologue(prologue) => Kind::ConsensusCommitPrologue(prologue.into()),
            ChangeEpoch(change_epoch) => Kind::ChangeEpoch(change_epoch.into()),

            // Validator management
            AddValidator(args) => Kind::AddValidator(args.into()),
            RemoveValidator(args) => Kind::RemoveValidator(args.into()),
            ReportValidator { reportee } => Kind::ReportValidator(reportee.into()),
            UndoReportValidator { reportee } => Kind::UndoReportValidator(reportee.into()),
            UpdateValidatorMetadata(args) => Kind::UpdateValidatorMetadata(args.into()),
            SetCommissionRate { new_rate } => Kind::SetCommissionRate(new_rate.into()),

            // Transfers and payments
            Transfer { coins, amounts, recipients } => {
                Kind::PayCoins(PayCoinsArgs { coins, amounts, recipients }.into())
            }
            MergeCoins { coins } => {
                Kind::PayCoins(PayCoinsArgs { coins, amounts: None, recipients: vec![] }.into())
            }
            TransferObjects { objects, recipient } => {
                Kind::TransferObjects(TransferObjectsArgs { objects, recipient }.into())
            }

            // Staking
            AddStake { address, coin_ref, amount } => {
                Kind::AddStake(AddStakeArgs { address, coin_ref, amount }.into())
            }
            WithdrawStake { staked_soma } => Kind::WithdrawStake(staked_soma.into()),

            // Model transactions
            CreateModel(args) => Kind::CreateModel(super::CreateModel {
                stake_amount: Some(args.stake_amount),
                commission_rate: Some(args.commission_rate),
                architecture_version: Some(args.architecture_version),
            }),
            CommitModel(args) => Kind::CommitModel(super::CommitModel {
                model_id: Some(args.model_id.to_string()),
                manifest: Some(args.manifest.clone().into()),
                weights_commitment: Some(args.weights_commitment.clone().into()),
                embedding_commitment: Some(args.embedding_commitment.clone().into()),
                decryption_key_commitment: Some(args.decryption_key_commitment.clone().into()),
            }),
            RevealModel(args) => Kind::RevealModel(super::RevealModel {
                model_id: Some(args.model_id.to_string()),
                decryption_key: Some(args.decryption_key.clone().into()),
                embedding: args.embedding.clone(),
            }),
            AddStakeToModel { model_id, coin_ref, amount } => {
                Kind::AddStakeToModel(super::AddStakeToModel {
                    model_id: Some(model_id.to_string()),
                    coin_ref: Some(coin_ref.into()),
                    amount,
                })
            }
            SetModelCommissionRate { model_id, new_rate } => {
                Kind::SetModelCommissionRate(super::SetModelCommissionRate {
                    model_id: Some(model_id.to_string()),
                    new_rate: Some(new_rate),
                })
            }
            DeactivateModel { model_id } => Kind::DeactivateModel(super::DeactivateModel {
                model_id: Some(model_id.to_string()),
            }),
            ReportModel { model_id } => {
                Kind::ReportModel(super::ReportModel { model_id: Some(model_id.to_string()) })
            }
            UndoReportModel { model_id } => Kind::UndoReportModel(super::UndoReportModel {
                model_id: Some(model_id.to_string()),
            }),

            // Submission transactions
            SubmitData(args) => Kind::SubmitData(super::SubmitData {
                target_id: Some(args.target_id.to_string()),
                data_manifest: Some(super::SubmissionManifest {
                    manifest: Some(args.data_manifest.manifest.clone().into()),
                }),
                model_id: Some(args.model_id.to_string()),
                embedding: args.embedding.clone(),
                distance_score: Some(args.distance_score),
                bond_coin: Some(args.bond_coin.clone().into()),
                loss_score: args.loss_score.clone(),
            }),
            ClaimRewards(args) => Kind::ClaimRewards(super::ClaimRewards {
                target_id: Some(args.target_id.to_string()),
            }),
            ReportSubmission { target_id } => Kind::ReportSubmission(super::ReportSubmission {
                target_id: Some(target_id.to_string()),
                challenger: None,
            }),
            UndoReportSubmission { target_id } => {
                Kind::UndoReportSubmission(super::UndoReportSubmission {
                    target_id: Some(target_id.to_string()),
                })
            }

            // Bridge transactions
            BridgeDeposit(args) => Kind::BridgeDeposit(super::BridgeDeposit {
                nonce: Some(args.nonce),
                eth_tx_hash: Some(args.eth_tx_hash.clone().into()),
                recipient: Some(args.recipient.to_string()),
                amount: Some(args.amount),
                aggregated_signature: Some(args.aggregated_signature.clone().into()),
                signer_bitmap: Some(args.signer_bitmap.clone().into()),
            }),
            BridgeWithdraw(args) => Kind::BridgeWithdraw(super::BridgeWithdraw {
                payment_coin: Some(args.payment_coin.clone().into()),
                amount: Some(args.amount),
                recipient_eth_address: Some(args.recipient_eth_address.clone().into()),
            }),
            BridgeEmergencyPause(args) => Kind::BridgeEmergencyPause(super::BridgeEmergencyPause {
                aggregated_signature: Some(args.aggregated_signature.clone().into()),
                signer_bitmap: Some(args.signer_bitmap.clone().into()),
            }),
            BridgeEmergencyUnpause(args) => Kind::BridgeEmergencyUnpause(super::BridgeEmergencyUnpause {
                aggregated_signature: Some(args.aggregated_signature.clone().into()),
                signer_bitmap: Some(args.signer_bitmap.clone().into()),
            }),

            // Payment-channel transactions
            OpenChannel(args) => Kind::OpenChannel(super::OpenChannel {
                payee: Some(args.payee.to_string()),
                authorized_signer: Some(args.authorized_signer.to_string()),
                token: Some(args.token.clone()),
                deposit_coin: Some(args.deposit_coin.into()),
                deposit_amount: Some(args.deposit_amount),
            }),
            Settle(args) => Kind::Settle(super::Settle {
                channel_id: Some(args.channel_id.to_string()),
                cumulative_amount: Some(args.cumulative_amount),
                voucher_signature: Some(args.voucher_signature.clone().into()),
            }),
            RequestClose(args) => Kind::RequestClose(super::RequestClose {
                channel_id: Some(args.channel_id.to_string()),
            }),
            WithdrawAfterTimeout(args) => Kind::WithdrawAfterTimeout(super::WithdrawAfterTimeout {
                channel_id: Some(args.channel_id.to_string()),
            }),

            Settlement(settlement) => Kind::Settlement(super::Settlement {
                epoch: Some(settlement.epoch),
                round: Some(settlement.round),
                sub_dag_index: settlement.sub_dag_index,
                changes: settlement
                    .changes
                    .iter()
                    .map(|ev| {
                        let (owner, coin_type, amount, is_credit) = match ev {
                            types::balance::BalanceEvent::Deposit {
                                owner,
                                coin_type,
                                amount,
                            } => (*owner, *coin_type, *amount, true),
                            types::balance::BalanceEvent::Withdraw {
                                owner,
                                coin_type,
                                amount,
                            } => (*owner, *coin_type, *amount, false),
                        };
                        super::SettlementChange {
                            owner: Some(owner.to_string()),
                            coin_type: Some(
                                match coin_type {
                                    types::object::CoinType::Usdc => "USDC",
                                    types::object::CoinType::Soma => "SOMA",
                                }
                                .to_string(),
                            ),
                            amount: Some(amount),
                            is_credit: Some(is_credit),
                        }
                    })
                    .collect(),
            }),
        };

        TransactionKind { kind: Some(kind) }
    }
}

// Converting from protobuf to domain types
impl TryFrom<&TransactionKind> for crate::types::TransactionKind {
    type Error = TryFromProtoError;

    fn try_from(value: &TransactionKind) -> Result<Self, Self::Error> {
        use transaction_kind::Kind;

        match value.kind.as_ref().ok_or_else(|| TryFromProtoError::missing("kind"))? {
            Kind::ChangeEpoch(change_epoch) => Self::ChangeEpoch(change_epoch.try_into()?),
            Kind::Genesis(genesis) => Self::Genesis(genesis.try_into()?),
            Kind::ConsensusCommitPrologue(prologue) => {
                Self::ConsensusCommitPrologue(prologue.try_into()?)
            }

            // Validator management
            Kind::AddValidator(args) => Self::AddValidator(args.try_into()?),
            Kind::RemoveValidator(args) => Self::RemoveValidator(args.try_into()?),
            Kind::ReportValidator(report) => Self::ReportValidator {
                reportee: report
                    .reportee
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("reportee"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("reportee", e))?,
            },
            Kind::UndoReportValidator(undo) => Self::UndoReportValidator {
                reportee: undo
                    .reportee
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("reportee"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("reportee", e))?,
            },
            Kind::UpdateValidatorMetadata(metadata) => {
                Self::UpdateValidatorMetadata(metadata.try_into()?)
            }
            Kind::SetCommissionRate(rate) => Self::SetCommissionRate {
                new_rate: rate.new_rate.ok_or_else(|| TryFromProtoError::missing("new_rate"))?,
            },

            // Transfers and payments
            Kind::TransferCoin(transfer) => {
                // Legacy proto: map single-coin TransferCoin to Transfer
                let coin = transfer
                    .coin
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("coin"))?
                    .try_into()?;
                let recipient = transfer
                    .recipient
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("recipient"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("recipient", e))?;
                Self::Transfer {
                    coins: vec![coin],
                    amounts: transfer.amount.map(|a| vec![a]),
                    recipients: vec![recipient],
                }
            }
            Kind::PayCoins(pay) => {
                let coins: Vec<_> =
                    pay.coins.iter().map(TryInto::try_into).collect::<Result<_, _>>()?;
                if pay.recipients.is_empty() {
                    // MergeCoins: no recipients means merge-only
                    Self::MergeCoins { coins }
                } else {
                    let recipients: Vec<_> = pay
                        .recipients
                        .iter()
                        .map(|r| {
                            r.parse().map_err(|e| TryFromProtoError::invalid("recipients", e))
                        })
                        .collect::<Result<_, _>>()?;
                    let amounts = if pay.amounts.is_empty() { None } else { Some(pay.amounts.clone()) };
                    Self::Transfer { coins, amounts, recipients }
                }
            }
            Kind::TransferObjects(transfer) => Self::TransferObjects {
                objects: transfer
                    .objects
                    .iter()
                    .map(TryInto::try_into)
                    .collect::<Result<_, _>>()?,
                recipient: transfer
                    .recipient
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("recipient"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("recipient", e))?,
            },

            // Staking
            Kind::AddStake(stake) => Self::AddStake {
                address: stake
                    .address
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("address"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("address", e))?,
                coin_ref: stake
                    .coin_ref
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("coin_ref"))?
                    .try_into()?,
                amount: stake.amount,
            },

            Kind::WithdrawStake(withdraw) => Self::WithdrawStake {
                staked_soma: withdraw
                    .staked_soma
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("staked_soma"))?
                    .try_into()?,
            },

            // Model transactions
            Kind::CreateModel(args) => Self::CreateModel(crate::types::CreateModelArgs {
                stake_amount: args
                    .stake_amount
                    .ok_or_else(|| TryFromProtoError::missing("stake_amount"))?,
                commission_rate: args
                    .commission_rate
                    .ok_or_else(|| TryFromProtoError::missing("commission_rate"))?,
                architecture_version: args
                    .architecture_version
                    .ok_or_else(|| TryFromProtoError::missing("architecture_version"))?,
            }),
            Kind::CommitModel(args) => Self::CommitModel(crate::types::CommitModelArgs {
                model_id: args
                    .model_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("model_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("model_id", e))?,
                manifest: args
                    .manifest
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("manifest"))?
                    .try_into()?,
                weights_commitment: args
                    .weights_commitment
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("weights_commitment"))?
                    .to_vec(),
                embedding_commitment: args
                    .embedding_commitment
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("embedding_commitment"))?
                    .to_vec(),
                decryption_key_commitment: args
                    .decryption_key_commitment
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("decryption_key_commitment"))?
                    .to_vec(),
            }),
            Kind::RevealModel(args) => Self::RevealModel(crate::types::RevealModelArgs {
                model_id: args
                    .model_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("model_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("model_id", e))?,
                decryption_key: args
                    .decryption_key
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("decryption_key"))?
                    .to_vec(),
                embedding: args.embedding.clone(),
            }),
            Kind::AddStakeToModel(args) => Self::AddStakeToModel {
                model_id: args
                    .model_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("model_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("model_id", e))?,
                coin_ref: args
                    .coin_ref
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("coin_ref"))?
                    .try_into()?,
                amount: args.amount,
            },
            Kind::SetModelCommissionRate(args) => Self::SetModelCommissionRate {
                model_id: args
                    .model_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("model_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("model_id", e))?,
                new_rate: args.new_rate.ok_or_else(|| TryFromProtoError::missing("new_rate"))?,
            },
            Kind::DeactivateModel(args) => Self::DeactivateModel {
                model_id: args
                    .model_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("model_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("model_id", e))?,
            },
            Kind::ReportModel(args) => Self::ReportModel {
                model_id: args
                    .model_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("model_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("model_id", e))?,
            },
            Kind::UndoReportModel(args) => Self::UndoReportModel {
                model_id: args
                    .model_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("model_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("model_id", e))?,
            },

            // Submission transactions
            Kind::SubmitData(args) => {
                let data_manifest = args
                    .data_manifest
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("data_manifest"))?;
                Self::SubmitData(crate::types::SubmitDataArgs {
                    target_id: args
                        .target_id
                        .as_ref()
                        .ok_or_else(|| TryFromProtoError::missing("target_id"))?
                        .parse()
                        .map_err(|e| TryFromProtoError::invalid("target_id", e))?,
                    data_manifest: crate::types::SubmissionManifest {
                        manifest: data_manifest
                            .manifest
                            .as_ref()
                            .ok_or_else(|| TryFromProtoError::missing("manifest"))?
                            .try_into()?,
                    },
                    model_id: args
                        .model_id
                        .as_ref()
                        .ok_or_else(|| TryFromProtoError::missing("model_id"))?
                        .parse()
                        .map_err(|e| TryFromProtoError::invalid("model_id", e))?,
                    embedding: args.embedding.clone(),
                    distance_score: args
                        .distance_score
                        .ok_or_else(|| TryFromProtoError::missing("distance_score"))?,
                    loss_score: args.loss_score.clone(),
                    bond_coin: args
                        .bond_coin
                        .as_ref()
                        .ok_or_else(|| TryFromProtoError::missing("bond_coin"))?
                        .try_into()?,
                })
            }
            Kind::ClaimRewards(args) => Self::ClaimRewards(crate::types::ClaimRewardsArgs {
                target_id: args
                    .target_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("target_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("target_id", e))?,
            }),
            Kind::ReportSubmission(args) => Self::ReportSubmission {
                target_id: args
                    .target_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("target_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("target_id", e))?,
            },
            Kind::UndoReportSubmission(args) => Self::UndoReportSubmission {
                target_id: args
                    .target_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("target_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("target_id", e))?,
            },
            // Bridge transactions
            Kind::BridgeDeposit(args) => Self::BridgeDeposit(crate::types::BridgeDepositArgs {
                nonce: args.nonce.unwrap_or(0),
                eth_tx_hash: args.eth_tx_hash.as_deref().unwrap_or_default().to_vec(),
                recipient: args.recipient.as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("recipient"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("recipient", e))?,
                amount: args.amount.unwrap_or(0),
                aggregated_signature: args.aggregated_signature.as_deref().unwrap_or_default().to_vec(),
                signer_bitmap: args.signer_bitmap.as_deref().unwrap_or_default().to_vec(),
            }),
            Kind::BridgeWithdraw(args) => Self::BridgeWithdraw(crate::types::BridgeWithdrawArgs {
                payment_coin: args.payment_coin.as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("payment_coin"))?
                    .try_into()?,
                amount: args.amount.unwrap_or(0),
                recipient_eth_address: args.recipient_eth_address.as_deref().unwrap_or_default().to_vec(),
            }),
            Kind::BridgeEmergencyPause(args) => Self::BridgeEmergencyPause(crate::types::BridgeEmergencyPauseArgs {
                aggregated_signature: args.aggregated_signature.as_deref().unwrap_or_default().to_vec(),
                signer_bitmap: args.signer_bitmap.as_deref().unwrap_or_default().to_vec(),
            }),
            Kind::BridgeEmergencyUnpause(args) => Self::BridgeEmergencyUnpause(crate::types::BridgeEmergencyUnpauseArgs {
                aggregated_signature: args.aggregated_signature.as_deref().unwrap_or_default().to_vec(),
                signer_bitmap: args.signer_bitmap.as_deref().unwrap_or_default().to_vec(),
            }),

            // Challenge variants removed — reject if received from proto
            Kind::InitiateChallenge(_)
            | Kind::ReportChallenge(_)
            | Kind::UndoReportChallenge(_)
            | Kind::ClaimChallengeBond(_) => {
                return Err(TryFromProtoError::invalid(
                    "kind",
                    "challenge transactions are no longer supported",
                ));
            }

            // Payment-channel variants
            Kind::OpenChannel(args) => Self::OpenChannel(crate::types::OpenChannelArgs {
                payee: args
                    .payee
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("payee"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("payee", e))?,
                authorized_signer: args
                    .authorized_signer
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("authorized_signer"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("authorized_signer", e))?,
                token: args
                    .token
                    .clone()
                    .ok_or_else(|| TryFromProtoError::missing("token"))?,
                deposit_coin: args
                    .deposit_coin
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("deposit_coin"))?
                    .try_into()?,
                deposit_amount: args
                    .deposit_amount
                    .ok_or_else(|| TryFromProtoError::missing("deposit_amount"))?,
            }),

            Kind::Settle(args) => Self::Settle(crate::types::SettleArgs {
                channel_id: args
                    .channel_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("channel_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("channel_id", e))?,
                cumulative_amount: args
                    .cumulative_amount
                    .ok_or_else(|| TryFromProtoError::missing("cumulative_amount"))?,
                voucher_signature: args
                    .voucher_signature
                    .as_deref()
                    .ok_or_else(|| TryFromProtoError::missing("voucher_signature"))?
                    .to_vec(),
            }),

            Kind::RequestClose(args) => Self::RequestClose(crate::types::RequestCloseArgs {
                channel_id: args
                    .channel_id
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("channel_id"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("channel_id", e))?,
            }),

            Kind::WithdrawAfterTimeout(args) => {
                Self::WithdrawAfterTimeout(crate::types::WithdrawAfterTimeoutArgs {
                    channel_id: args
                        .channel_id
                        .as_ref()
                        .ok_or_else(|| TryFromProtoError::missing("channel_id"))?
                        .parse()
                        .map_err(|e| TryFromProtoError::invalid("channel_id", e))?,
                })
            }

            Kind::Settlement(settlement) => {
                let mut changes = Vec::with_capacity(settlement.changes.len());
                for change in &settlement.changes {
                    let owner = change
                        .owner
                        .as_ref()
                        .ok_or_else(|| TryFromProtoError::missing("settlement.changes.owner"))?
                        .parse()
                        .map_err(|e| TryFromProtoError::invalid("settlement.changes.owner", e))?;
                    let coin_type_label = change
                        .coin_type
                        .as_ref()
                        .ok_or_else(|| {
                            TryFromProtoError::missing("settlement.changes.coin_type")
                        })?
                        .as_str();
                    let coin_type = match coin_type_label {
                        "USDC" => types::object::CoinType::Usdc,
                        "SOMA" => types::object::CoinType::Soma,
                        other => {
                            return Err(TryFromProtoError::invalid(
                                "settlement.changes.coin_type",
                                format!("unknown coin type {other:?}"),
                            ));
                        }
                    };
                    let amount = change
                        .amount
                        .ok_or_else(|| TryFromProtoError::missing("settlement.changes.amount"))?;
                    let is_credit = change
                        .is_credit
                        .ok_or_else(|| TryFromProtoError::missing("settlement.changes.is_credit"))?;
                    changes.push(if is_credit {
                        types::balance::BalanceEvent::deposit(owner, coin_type, amount)
                    } else {
                        types::balance::BalanceEvent::withdraw(owner, coin_type, amount)
                    });
                }
                Self::Settlement(crate::types::SettlementTransaction {
                    epoch: settlement
                        .epoch
                        .ok_or_else(|| TryFromProtoError::missing("settlement.epoch"))?,
                    round: settlement
                        .round
                        .ok_or_else(|| TryFromProtoError::missing("settlement.round"))?,
                    sub_dag_index: settlement.sub_dag_index,
                    changes,
                })
            }
        }
        .pipe(Ok)
    }
}

// Supporting type conversions

// AddValidatorArgs conversions
impl From<crate::types::AddValidatorArgs> for AddValidator {
    fn from(value: crate::types::AddValidatorArgs) -> Self {
        Self {
            pubkey_bytes: Some(value.pubkey_bytes.into()),
            network_pubkey_bytes: Some(value.network_pubkey_bytes.into()),
            worker_pubkey_bytes: Some(value.worker_pubkey_bytes.into()),
            net_address: Some(value.net_address.into()),
            p2p_address: Some(value.p2p_address.into()),
            primary_address: Some(value.primary_address.into()),
            proxy_address: Some(value.proxy_address.into()),
            proof_of_possession: Some(value.proof_of_possession.into()),
        }
    }
}

impl TryFrom<&AddValidator> for crate::types::AddValidatorArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &AddValidator) -> Result<Self, Self::Error> {
        Ok(Self {
            pubkey_bytes: value
                .pubkey_bytes
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("pubkey_bytes"))?
                .into(),
            network_pubkey_bytes: value
                .network_pubkey_bytes
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("network_pubkey_bytes"))?
                .into(),
            worker_pubkey_bytes: value
                .worker_pubkey_bytes
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("worker_pubkey_bytes"))?
                .into(),
            proof_of_possession: value
                .proof_of_possession
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("proof_of_possession"))?
                .into(),
            net_address: value
                .net_address
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("net_address"))?
                .into(),
            p2p_address: value
                .p2p_address
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("p2p_address"))?
                .into(),
            primary_address: value
                .primary_address
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("primary_address"))?
                .into(),
            proxy_address: value
                .proxy_address
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("proxy_address"))?
                .into(),
        })
    }
}

// RemoveValidatorArgs conversions
impl From<crate::types::RemoveValidatorArgs> for RemoveValidator {
    fn from(value: crate::types::RemoveValidatorArgs) -> Self {
        Self { pubkey_bytes: Some(value.pubkey_bytes.into()) }
    }
}

impl TryFrom<&RemoveValidator> for crate::types::RemoveValidatorArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &RemoveValidator) -> Result<Self, Self::Error> {
        Ok(Self {
            pubkey_bytes: value
                .pubkey_bytes
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("pubkey_bytes"))?
                .into(),
        })
    }
}

// UpdateValidatorMetadataArgs conversions
impl From<crate::types::UpdateValidatorMetadataArgs> for UpdateValidatorMetadata {
    fn from(value: crate::types::UpdateValidatorMetadataArgs) -> Self {
        Self {
            next_epoch_network_address: value.next_epoch_network_address.map(|v| v.into()),
            next_epoch_p2p_address: value.next_epoch_p2p_address.map(|v| v.into()),
            next_epoch_primary_address: value.next_epoch_primary_address.map(|v| v.into()),
            next_epoch_proxy_address: value.next_epoch_proxy_address.map(|v| v.into()),
            next_epoch_protocol_pubkey: value.next_epoch_protocol_pubkey.map(|v| v.into()),
            next_epoch_worker_pubkey: value.next_epoch_worker_pubkey.map(|v| v.into()),
            next_epoch_network_pubkey: value.next_epoch_network_pubkey.map(|v| v.into()),
            next_epoch_proof_of_possession: value.next_epoch_proof_of_possession.map(|v| v.into()),
        }
    }
}

impl TryFrom<&UpdateValidatorMetadata> for crate::types::UpdateValidatorMetadataArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &UpdateValidatorMetadata) -> Result<Self, Self::Error> {
        Ok(Self {
            next_epoch_network_address: value.next_epoch_network_address.clone().map(|v| v.into()),
            next_epoch_p2p_address: value.next_epoch_p2p_address.clone().map(|v| v.into()),
            next_epoch_primary_address: value.next_epoch_primary_address.clone().map(|v| v.into()),
            next_epoch_proxy_address: value.next_epoch_proxy_address.clone().map(|v| v.into()),
            next_epoch_protocol_pubkey: value.next_epoch_protocol_pubkey.clone().map(|v| v.into()),
            next_epoch_worker_pubkey: value.next_epoch_worker_pubkey.clone().map(|v| v.into()),
            next_epoch_network_pubkey: value.next_epoch_network_pubkey.clone().map(|v| v.into()),
            next_epoch_proof_of_possession: value
                .next_epoch_proof_of_possession
                .clone()
                .map(|v| v.into()),
        })
    }
}

//
// ConsensusCommitPrologue
//

impl From<crate::types::ConsensusCommitPrologue> for ConsensusCommitPrologue {
    fn from(value: crate::types::ConsensusCommitPrologue) -> Self {
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

impl TryFrom<&ConsensusCommitPrologue> for crate::types::ConsensusCommitPrologue {
    type Error = TryFromProtoError;

    fn try_from(value: &ConsensusCommitPrologue) -> Result<Self, Self::Error> {
        let epoch = value.epoch.ok_or_else(|| TryFromProtoError::missing("epoch"))?;
        let round = value.round.ok_or_else(|| TryFromProtoError::missing("round"))?;
        let commit_timestamp_ms = value
            .commit_timestamp
            .ok_or_else(|| TryFromProtoError::missing("commit_timestamp"))?
            .pipe(crate::proto::proto_to_timestamp_ms)?;
        let consensus_commit_digest = value
            .consensus_commit_digest
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("consensus_commit_digest"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid("consensus_commit_digest", e))?;
        let additional_state_digest = value
            .additional_state_digest
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("additional_state_digest"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid("additional_state_digest", e))?;

        Ok(Self {
            epoch,
            round,
            commit_timestamp_ms,
            sub_dag_index: value.sub_dag_index,
            consensus_commit_digest,
            additional_state_digest,
        })
    }
}

//
// GenesisTransaction
//
impl From<crate::types::GenesisTransaction> for GenesisTransaction {
    fn from(value: crate::types::GenesisTransaction) -> Self {
        Self { objects: value.objects.into_iter().map(Object::from).collect() }
    }
}

impl TryFrom<&GenesisTransaction> for crate::types::GenesisTransaction {
    type Error = TryFromProtoError;

    fn try_from(value: &GenesisTransaction) -> Result<Self, Self::Error> {
        let objects = value
            .objects
            .iter()
            .map(crate::types::Object::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { objects })
    }
}

//
// ChangeEpoch
//
impl From<crate::types::ChangeEpoch> for ChangeEpoch {
    fn from(value: crate::types::ChangeEpoch) -> Self {
        Self {
            epoch: Some(value.epoch),
            protocol_version: Some(value.protocol_version),
            fees: Some(value.fees),
            epoch_start_timestamp: Some(crate::proto::timestamp_ms_to_proto(
                value.epoch_start_timestamp_ms,
            )),
            epoch_randomness: Some(value.epoch_randomness.into()),
        }
    }
}

impl TryFrom<&ChangeEpoch> for crate::types::ChangeEpoch {
    type Error = TryFromProtoError;

    fn try_from(
        ChangeEpoch {
            epoch,
            protocol_version,
            fees,
            epoch_start_timestamp,
            epoch_randomness,
        }: &ChangeEpoch,
    ) -> Result<Self, Self::Error> {
        let epoch = epoch.ok_or_else(|| TryFromProtoError::missing("epoch"))?;
        let protocol_version =
            protocol_version.ok_or_else(|| TryFromProtoError::missing("protocol_version"))?;
        let fees = fees.ok_or_else(|| TryFromProtoError::missing("fees"))?;
        let epoch_start_timestamp_ms = epoch_start_timestamp
            .ok_or_else(|| TryFromProtoError::missing("epoch_start_timestamp"))?
            .pipe(crate::proto::proto_to_timestamp_ms)?;
        let epoch_randomness = epoch_randomness
            .clone()
            .ok_or_else(|| TryFromProtoError::missing("epoch_randomness"))?
            .into();

        Ok(Self { epoch, protocol_version, fees, epoch_start_timestamp_ms, epoch_randomness })
    }
}

// ReportValidator conversions
impl From<crate::types::Address> for ReportValidator {
    fn from(reportee: crate::types::Address) -> Self {
        Self { reportee: Some(reportee.to_string()) }
    }
}

// UndoReportValidator conversions
impl From<crate::types::Address> for UndoReportValidator {
    fn from(reportee: crate::types::Address) -> Self {
        Self { reportee: Some(reportee.to_string()) }
    }
}

// SetCommissionRate conversions
impl From<u64> for SetCommissionRate {
    fn from(new_rate: u64) -> Self {
        Self { new_rate: Some(new_rate) }
    }
}

// PayCoins conversions (used for both Transfer and MergeCoins)
pub struct PayCoinsArgs {
    pub coins: Vec<crate::types::ObjectReference>,
    pub amounts: Option<Vec<u64>>,
    pub recipients: Vec<crate::types::Address>,
}

impl From<PayCoinsArgs> for PayCoins {
    fn from(args: PayCoinsArgs) -> Self {
        Self {
            coins: args.coins.into_iter().map(Into::into).collect(),
            amounts: args.amounts.unwrap_or_default(),
            recipients: args.recipients.into_iter().map(|r| r.to_string()).collect(),
        }
    }
}

// TransferObjects conversions
pub struct TransferObjectsArgs {
    pub objects: Vec<crate::types::ObjectReference>,
    pub recipient: crate::types::Address,
}

impl From<TransferObjectsArgs> for TransferObjects {
    fn from(args: TransferObjectsArgs) -> Self {
        Self {
            objects: args.objects.into_iter().map(Into::into).collect(),
            recipient: Some(args.recipient.to_string()),
        }
    }
}

// AddStake conversions
pub struct AddStakeArgs {
    pub address: crate::types::Address,
    pub coin_ref: crate::types::ObjectReference,
    pub amount: Option<u64>,
}

impl From<AddStakeArgs> for AddStake {
    fn from(args: AddStakeArgs) -> Self {
        Self {
            address: Some(args.address.to_string()),
            coin_ref: Some(args.coin_ref.into()),
            amount: args.amount,
        }
    }
}

// WithdrawStake conversions
impl From<crate::types::ObjectReference> for WithdrawStake {
    fn from(staked_soma: crate::types::ObjectReference) -> Self {
        Self { staked_soma: Some(staked_soma.into()) }
    }
}
