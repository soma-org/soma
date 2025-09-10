use super::*;
use crate::proto::TryFromProtoError;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;
use tap::Pipe;

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
        if mask.contains(Self::BCS_FIELD.name) {
            let mut bcs = Bcs::serialize(&source).unwrap();
            bcs.name = Some("TransactionData".to_owned());
            self.bcs = Some(bcs);
        }

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
            self.gas_payment = Some(source.gas_payment.into());
        }
    }
}

impl Merge<&Transaction> for Transaction {
    fn merge(&mut self, source: &Transaction, mask: &FieldMaskTree) {
        let Transaction {
            bcs,
            digest,
            kind,
            sender,
            gas_payment,
        } = source;

        if mask.contains(Self::BCS_FIELD.name) {
            self.bcs = bcs.clone();
        }

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
        if let Some(bcs) = &value.bcs {
            return bcs
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid(Transaction::BCS_FIELD, e));
        }

        let kind = value
            .kind
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("kind"))?
            .try_into()?;

        let sender = value
            .sender
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("sender"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(Transaction::SENDER_FIELD, e))?;

        let gas_payment = value
            .gas_payment
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("gas_payment"))?
            .try_into()?;

        Ok(Self {
            kind,
            sender,
            gas_payment,
        })
    }
}

//
// GasPayment
//

impl From<crate::types::GasPayment> for GasPayment {
    fn from(value: crate::types::GasPayment) -> Self {
        Self {
            objects: value.objects.into_iter().map(Into::into).collect(),
            owner: Some(value.owner.to_string()),
        }
    }
}

impl TryFrom<&GasPayment> for crate::types::GasPayment {
    type Error = TryFromProtoError;

    fn try_from(value: &GasPayment) -> Result<Self, Self::Error> {
        let objects = value
            .objects
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<_, _>>()?;

        let owner = value
            .owner
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("owner"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(GasPayment::OWNER_FIELD, e))?;

        Ok(Self { objects, owner })
    }
}

//
// TransactionKind
//
// Converting from domain types to protobuf
impl From<crate::types::TransactionKind> for TransactionKind {
    fn from(value: crate::types::TransactionKind) -> Self {
        use crate::types::TransactionKind::*;
        use transaction_kind::Kind;

        let kind = match value {
            ChangeEpoch(change_epoch) => Kind::ChangeEpoch(change_epoch.into()),
            Genesis(genesis) => Kind::Genesis(genesis.into()),
            ConsensusCommitPrologue(prologue) => Kind::ConsensusCommitPrologue(prologue.into()),

            // Validator management
            AddValidator(args) => Kind::AddValidator(args.into()),
            RemoveValidator(args) => Kind::RemoveValidator(args.into()),
            ReportValidator { reportee } => Kind::ReportValidator(reportee.into()),
            UndoReportValidator { reportee } => Kind::UndoReportValidator(reportee.into()),
            UpdateValidatorMetadata(args) => Kind::UpdateValidatorMetadata(args.into()),
            SetCommissionRate { new_rate } => Kind::SetCommissionRate(new_rate.into()),

            // Encoder management
            AddEncoder(args) => Kind::AddEncoder(args.into()),
            RemoveEncoder(args) => Kind::RemoveEncoder(args.into()),
            ReportEncoder { reportee } => Kind::ReportEncoder(reportee.into()),
            UndoReportEncoder { reportee } => Kind::UndoReportEncoder(reportee.into()),
            UpdateEncoderMetadata(args) => Kind::UpdateEncoderMetadata(args.into()),
            SetEncoderCommissionRate { new_rate } => {
                Kind::SetEncoderCommissionRate(new_rate.into())
            }
            SetEncoderBytePrice { new_price } => Kind::SetEncoderBytePrice(new_price.into()),

            // Transfers and payments
            TransferCoin {
                coin,
                amount,
                recipient,
            } => Kind::TransferCoin(
                TransferCoinArgs {
                    coin,
                    amount: Some(amount),
                    recipient,
                }
                .into(),
            ),
            PayCoins {
                coins,
                amounts,
                recipients,
            } => Kind::PayCoins(
                PayCoinsArgs {
                    coins,
                    amounts: Some(amounts),
                    recipients,
                }
                .into(),
            ),
            TransferObjects { objects, recipient } => {
                Kind::TransferObjects(TransferObjectsArgs { objects, recipient }.into())
            }

            // Staking
            AddStake {
                address,
                coin_ref,
                amount,
            } => Kind::AddStake(
                AddStakeArgs {
                    address,
                    coin_ref,
                    amount: Some(amount),
                }
                .into(),
            ),
            AddStakeToEncoder {
                encoder_address,
                coin_ref,
                amount,
            } => Kind::AddStakeToEncoder(
                AddStakeToEncoderArgs {
                    encoder_address,
                    coin_ref,
                    amount: Some(amount),
                }
                .into(),
            ),
            WithdrawStake { staked_soma } => Kind::WithdrawStake(staked_soma.into()),

            // Shard operations
            EmbedData {
                digest,
                data_size_bytes,
                coin_ref,
            } => Kind::EmbedData(
                EmbedDataArgs {
                    digest,
                    data_size_bytes,
                    coin_ref,
                }
                .into(),
            ),
            ClaimEscrow { shard_input_ref } => Kind::ClaimEscrow(shard_input_ref.into()),
            ReportScores {
                shard_input_ref,
                scores,
                encoder_aggregate_signature,
                signers,
            } => Kind::ReportScores(
                ReportScoresArgs {
                    shard_input_ref,
                    scores,
                    encoder_aggregate_signature,
                    signers,
                }
                .into(),
            ),
        };

        TransactionKind { kind: Some(kind) }
    }
}

// Converting from protobuf to domain types
impl TryFrom<&TransactionKind> for crate::types::TransactionKind {
    type Error = TryFromProtoError;

    fn try_from(value: &TransactionKind) -> Result<Self, Self::Error> {
        use transaction_kind::Kind;

        match value
            .kind
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("kind"))?
        {
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
                new_rate: rate
                    .new_rate
                    .ok_or_else(|| TryFromProtoError::missing("new_rate"))?,
            },

            // Encoder management
            Kind::AddEncoder(args) => Self::AddEncoder(args.try_into()?),
            Kind::RemoveEncoder(args) => Self::RemoveEncoder(args.try_into()?),
            Kind::ReportEncoder(report) => Self::ReportEncoder {
                reportee: report
                    .reportee
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("reportee"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("reportee", e))?,
            },
            Kind::UndoReportEncoder(undo) => Self::UndoReportEncoder {
                reportee: undo
                    .reportee
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("reportee"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("reportee", e))?,
            },
            Kind::UpdateEncoderMetadata(metadata) => {
                Self::UpdateEncoderMetadata(metadata.try_into()?)
            }
            Kind::SetEncoderCommissionRate(rate) => Self::SetEncoderCommissionRate {
                new_rate: rate
                    .new_rate
                    .ok_or_else(|| TryFromProtoError::missing("new_rate"))?,
            },
            Kind::SetEncoderBytePrice(price) => Self::SetEncoderBytePrice {
                new_price: price
                    .new_price
                    .ok_or_else(|| TryFromProtoError::missing("new_price"))?,
            },

            // Transfers and payments
            Kind::TransferCoin(transfer) => Self::TransferCoin {
                coin: transfer
                    .coin
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("coin"))?
                    .try_into()?,
                amount: transfer
                    .amount
                    .ok_or_else(|| TryFromProtoError::missing("amount"))?, // Convert 0 back to None
                recipient: transfer
                    .recipient
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("recipient"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("recipient", e))?,
            },
            Kind::PayCoins(pay) => Self::PayCoins {
                coins: pay
                    .coins
                    .iter()
                    .map(TryInto::try_into)
                    .collect::<Result<_, _>>()?,
                amounts: pay.amounts.clone(),
                recipients: pay
                    .recipients
                    .iter()
                    .map(|r| {
                        r.parse()
                            .map_err(|e| TryFromProtoError::invalid("recipients", e))
                    })
                    .collect::<Result<_, _>>()?,
            },
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
                amount: stake
                    .amount
                    .ok_or_else(|| TryFromProtoError::missing("amount"))?,
            },
            Kind::AddStakeToEncoder(stake) => Self::AddStakeToEncoder {
                encoder_address: stake
                    .encoder_address
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("encoder_address"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("encoder_address", e))?,
                coin_ref: stake
                    .coin_ref
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("coin_ref"))?
                    .try_into()?,
                amount: stake
                    .amount
                    .ok_or_else(|| TryFromProtoError::missing("amount"))?,
            },
            Kind::WithdrawStake(withdraw) => Self::WithdrawStake {
                staked_soma: withdraw
                    .staked_soma
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("staked_soma"))?
                    .try_into()?,
            },

            // Shard operations
            Kind::EmbedData(embed) => Self::EmbedData {
                digest: embed
                    .digest
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("digest"))?
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid("digest", e))?,
                data_size_bytes: embed
                    .data_size_bytes
                    .ok_or_else(|| TryFromProtoError::missing("data_size_bytes"))?
                    as usize,
                coin_ref: embed
                    .coin_ref
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("coin_ref"))?
                    .try_into()?,
            },
            Kind::ClaimEscrow(claim) => Self::ClaimEscrow {
                shard_input_ref: claim
                    .shard_input_ref
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("shard_input_ref"))?
                    .try_into()?,
            },
            Kind::ReportScores(report) => Self::ReportScores {
                shard_input_ref: report
                    .shard_input_ref
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("shard_input_ref"))?
                    .try_into()?,
                scores: report
                    .scores
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("scores"))?
                    .deserialize()
                    .map_err(|e| TryFromProtoError::invalid("scores", e))?,
                encoder_aggregate_signature: report
                    .encoder_aggregate_signature
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("encoder_aggregate_signature"))?
                    .deserialize()
                    .map_err(|e| TryFromProtoError::invalid("encoder_aggregate_signature", e))?,
                signers: report
                    .signers
                    .iter()
                    .map(|s| {
                        s.parse()
                            .map_err(|e| TryFromProtoError::invalid("signers", e))
                    })
                    .collect::<Result<_, _>>()?,
            },
        }
        .pipe(Ok)
    }
}

// Supporting type conversions

// AddValidatorArgs conversions
impl From<crate::types::AddValidatorArgs> for AddValidator {
    fn from(value: crate::types::AddValidatorArgs) -> Self {
        Self {
            pubkey_bytes: Some(Bcs::serialize(&value.pubkey_bytes).unwrap()),
            network_pubkey_bytes: Some(Bcs::serialize(&value.network_pubkey_bytes).unwrap()),
            worker_pubkey_bytes: Some(Bcs::serialize(&value.worker_pubkey_bytes).unwrap()),
            net_address: Some(Bcs::serialize(&value.net_address).unwrap()),
            p2p_address: Some(Bcs::serialize(&value.p2p_address).unwrap()),
            primary_address: Some(Bcs::serialize(&value.primary_address).unwrap()),
            encoder_validator_address: Some(
                Bcs::serialize(&value.encoder_validator_address).unwrap(),
            ),
        }
    }
}

impl TryFrom<&AddValidator> for crate::types::AddValidatorArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &AddValidator) -> Result<Self, Self::Error> {
        Ok(Self {
            pubkey_bytes: value
                .pubkey_bytes
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("pubkey_bytes"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("pubkey_bytes", e))?,
            network_pubkey_bytes: value
                .network_pubkey_bytes
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("network_pubkey_bytes"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("network_pubkey_bytes", e))?,
            worker_pubkey_bytes: value
                .worker_pubkey_bytes
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("worker_pubkey_bytes"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("worker_pubkey_bytes", e))?,
            net_address: value
                .net_address
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("net_address"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("net_address", e))?,
            p2p_address: value
                .p2p_address
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("p2p_address"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("p2p_address", e))?,
            primary_address: value
                .primary_address
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("primary_address"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("primary_address", e))?,
            encoder_validator_address: value
                .encoder_validator_address
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("encoder_validator_address"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("encoder_validator_address", e))?,
        })
    }
}

// RemoveValidatorArgs conversions
impl From<crate::types::RemoveValidatorArgs> for RemoveValidator {
    fn from(value: crate::types::RemoveValidatorArgs) -> Self {
        Self {
            pubkey_bytes: Some(Bcs::serialize(&value.pubkey_bytes).unwrap()),
        }
    }
}

impl TryFrom<&RemoveValidator> for crate::types::RemoveValidatorArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &RemoveValidator) -> Result<Self, Self::Error> {
        Ok(Self {
            pubkey_bytes: value
                .pubkey_bytes
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("pubkey_bytes"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("pubkey_bytes", e))?,
        })
    }
}

// UpdateValidatorMetadataArgs conversions
impl From<crate::types::UpdateValidatorMetadataArgs> for UpdateValidatorMetadata {
    fn from(value: crate::types::UpdateValidatorMetadataArgs) -> Self {
        Self {
            next_epoch_network_address: value
                .next_epoch_network_address
                .map(|v| Bcs::serialize(&v).unwrap()),
            next_epoch_p2p_address: value
                .next_epoch_p2p_address
                .map(|v| Bcs::serialize(&v).unwrap()),
            next_epoch_primary_address: value
                .next_epoch_primary_address
                .map(|v| Bcs::serialize(&v).unwrap()),
            next_epoch_protocol_pubkey: value
                .next_epoch_protocol_pubkey
                .map(|v| Bcs::serialize(&v).unwrap()),
            next_epoch_worker_pubkey: value
                .next_epoch_worker_pubkey
                .map(|v| Bcs::serialize(&v).unwrap()),
            next_epoch_network_pubkey: value
                .next_epoch_network_pubkey
                .map(|v| Bcs::serialize(&v).unwrap()),
        }
    }
}

impl TryFrom<&UpdateValidatorMetadata> for crate::types::UpdateValidatorMetadataArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &UpdateValidatorMetadata) -> Result<Self, Self::Error> {
        Ok(Self {
            next_epoch_network_address: value
                .next_epoch_network_address
                .as_ref()
                .map(|v| {
                    v.deserialize()
                        .map_err(|e| TryFromProtoError::invalid("next_epoch_network_address", e))
                })
                .transpose()?,
            next_epoch_p2p_address: value
                .next_epoch_p2p_address
                .as_ref()
                .map(|v| {
                    v.deserialize()
                        .map_err(|e| TryFromProtoError::invalid("next_epoch_p2p_address", e))
                })
                .transpose()?,
            next_epoch_primary_address: value
                .next_epoch_primary_address
                .as_ref()
                .map(|v| {
                    v.deserialize()
                        .map_err(|e| TryFromProtoError::invalid("next_epoch_primary_address", e))
                })
                .transpose()?,
            next_epoch_protocol_pubkey: value
                .next_epoch_protocol_pubkey
                .as_ref()
                .map(|v| {
                    v.deserialize()
                        .map_err(|e| TryFromProtoError::invalid("next_epoch_protocol_pubkey", e))
                })
                .transpose()?,
            next_epoch_worker_pubkey: value
                .next_epoch_worker_pubkey
                .as_ref()
                .map(|v| {
                    v.deserialize()
                        .map_err(|e| TryFromProtoError::invalid("next_epoch_worker_pubkey", e))
                })
                .transpose()?,
            next_epoch_network_pubkey: value
                .next_epoch_network_pubkey
                .as_ref()
                .map(|v| {
                    v.deserialize()
                        .map_err(|e| TryFromProtoError::invalid("next_epoch_network_pubkey", e))
                })
                .transpose()?,
        })
    }
}

// AddEncoderArgs conversions
impl From<crate::types::AddEncoderArgs> for AddEncoder {
    fn from(value: crate::types::AddEncoderArgs) -> Self {
        Self {
            encoder_pubkey_bytes: Some(Bcs::serialize(&value.encoder_pubkey_bytes).unwrap()),
            network_pubkey_bytes: Some(Bcs::serialize(&value.network_pubkey_bytes).unwrap()),
            internal_network_address: Some(
                Bcs::serialize(&value.internal_network_address).unwrap(),
            ),
            external_network_address: Some(
                Bcs::serialize(&value.external_network_address).unwrap(),
            ),
            object_server_address: Some(Bcs::serialize(&value.object_server_address).unwrap()),
        }
    }
}

impl TryFrom<&AddEncoder> for crate::types::AddEncoderArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &AddEncoder) -> Result<Self, Self::Error> {
        Ok(Self {
            encoder_pubkey_bytes: value
                .encoder_pubkey_bytes
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("encoder_pubkey_bytes"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("encoder_pubkey_bytes", e))?,
            network_pubkey_bytes: value
                .network_pubkey_bytes
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("network_pubkey_bytes"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("network_pubkey_bytes", e))?,
            internal_network_address: value
                .internal_network_address
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("internal_network_address"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("internal_network_address", e))?,
            external_network_address: value
                .external_network_address
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("external_network_address"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("external_network_address", e))?,
            object_server_address: value
                .object_server_address
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("object_server_address"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("object_server_address", e))?,
        })
    }
}

// RemoveEncoderArgs conversions
impl From<crate::types::RemoveEncoderArgs> for RemoveEncoder {
    fn from(value: crate::types::RemoveEncoderArgs) -> Self {
        Self {
            encoder_pubkey_bytes: Some(Bcs::serialize(&value.encoder_pubkey_bytes).unwrap()),
        }
    }
}

impl TryFrom<&RemoveEncoder> for crate::types::RemoveEncoderArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &RemoveEncoder) -> Result<Self, Self::Error> {
        Ok(Self {
            encoder_pubkey_bytes: value
                .encoder_pubkey_bytes
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("encoder_pubkey_bytes"))?
                .deserialize()
                .map_err(|e| TryFromProtoError::invalid("encoder_pubkey_bytes", e))?,
        })
    }
}

// UpdateEncoderMetadataArgs conversions
impl From<crate::types::UpdateEncoderMetadataArgs> for UpdateEncoderMetadata {
    fn from(value: crate::types::UpdateEncoderMetadataArgs) -> Self {
        Self {
            next_epoch_external_network_address: value
                .next_epoch_external_network_address
                .map(|v| Bcs::serialize(&v).unwrap()),
            next_epoch_internal_network_address: value
                .next_epoch_internal_network_address
                .map(|v| Bcs::serialize(&v).unwrap()),
            next_epoch_network_pubkey: value
                .next_epoch_network_pubkey
                .map(|v| Bcs::serialize(&v).unwrap()),
            next_epoch_object_server_address: value
                .next_epoch_object_server_address
                .map(|v| Bcs::serialize(&v).unwrap()),
        }
    }
}

impl TryFrom<&UpdateEncoderMetadata> for crate::types::UpdateEncoderMetadataArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &UpdateEncoderMetadata) -> Result<Self, Self::Error> {
        Ok(Self {
            next_epoch_external_network_address: value
                .next_epoch_external_network_address
                .as_ref()
                .map(|v| {
                    v.deserialize().map_err(|e| {
                        TryFromProtoError::invalid("next_epoch_external_network_address", e)
                    })
                })
                .transpose()?,
            next_epoch_internal_network_address: value
                .next_epoch_internal_network_address
                .as_ref()
                .map(|v| {
                    v.deserialize().map_err(|e| {
                        TryFromProtoError::invalid("next_epoch_internal_network_address", e)
                    })
                })
                .transpose()?,
            next_epoch_network_pubkey: value
                .next_epoch_network_pubkey
                .as_ref()
                .map(|v| {
                    v.deserialize()
                        .map_err(|e| TryFromProtoError::invalid("next_epoch_network_pubkey", e))
                })
                .transpose()?,
            next_epoch_object_server_address: value
                .next_epoch_object_server_address
                .as_ref()
                .map(|v| {
                    v.deserialize().map_err(|e| {
                        TryFromProtoError::invalid("next_epoch_object_server_address", e)
                    })
                })
                .transpose()?,
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
            commit_timestamp: Some(crate::proto::timestamp_ms_to_proto(
                value.commit_timestamp_ms,
            )),
            consensus_commit_digest: None,
            sub_dag_index: None,
        }
    }
}

impl TryFrom<&ConsensusCommitPrologue> for crate::types::ConsensusCommitPrologue {
    type Error = TryFromProtoError;

    fn try_from(value: &ConsensusCommitPrologue) -> Result<Self, Self::Error> {
        let epoch = value
            .epoch
            .ok_or_else(|| TryFromProtoError::missing("epoch"))?;
        let round = value
            .round
            .ok_or_else(|| TryFromProtoError::missing("round"))?;
        let commit_timestamp_ms = value
            .commit_timestamp
            .ok_or_else(|| TryFromProtoError::missing("commit_timestamp"))?
            .pipe(crate::proto::proto_to_timestamp_ms)?;

        Ok(Self {
            epoch,
            round,
            commit_timestamp_ms,
        })
    }
}

//
// GenesisTransaction
//
impl From<crate::types::GenesisTransaction> for GenesisTransaction {
    fn from(value: crate::types::GenesisTransaction) -> Self {
        Self {
            objects: value
                .objects
                .into_iter()
                .map(|obj| Object::from(obj)) // Explicitly use proto::Object
                .collect(),
        }
    }
}

impl TryFrom<&GenesisTransaction> for crate::types::GenesisTransaction {
    type Error = TryFromProtoError;

    fn try_from(value: &GenesisTransaction) -> Result<Self, Self::Error> {
        let objects = value
            .objects
            .iter()
            .map(|obj| crate::types::Object::try_from(obj)) // Explicitly convert
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
            epoch_start_timestamp: Some(crate::proto::timestamp_ms_to_proto(
                value.epoch_start_timestamp_ms,
            )),
        }
    }
}

impl TryFrom<&ChangeEpoch> for crate::types::ChangeEpoch {
    type Error = TryFromProtoError;

    fn try_from(
        ChangeEpoch {
            epoch,
            epoch_start_timestamp,
        }: &ChangeEpoch,
    ) -> Result<Self, Self::Error> {
        let epoch = epoch.ok_or_else(|| TryFromProtoError::missing("epoch"))?;

        let epoch_start_timestamp_ms = epoch_start_timestamp
            .ok_or_else(|| TryFromProtoError::missing("epoch_start_timestamp_ms"))?
            .pipe(crate::proto::proto_to_timestamp_ms)?;

        Ok(Self {
            epoch,
            epoch_start_timestamp_ms,
        })
    }
}

// ReportValidator conversions
impl From<crate::types::Address> for ReportValidator {
    fn from(reportee: crate::types::Address) -> Self {
        Self {
            reportee: Some(reportee.to_string()),
        }
    }
}

// UndoReportValidator conversions
impl From<crate::types::Address> for UndoReportValidator {
    fn from(reportee: crate::types::Address) -> Self {
        Self {
            reportee: Some(reportee.to_string()),
        }
    }
}

// SetCommissionRate conversions
impl From<u64> for SetCommissionRate {
    fn from(new_rate: u64) -> Self {
        Self {
            new_rate: Some(new_rate),
        }
    }
}

// ReportEncoder conversions
impl From<crate::types::Address> for ReportEncoder {
    fn from(reportee: crate::types::Address) -> Self {
        Self {
            reportee: Some(reportee.to_string()),
        }
    }
}

// UndoReportEncoder conversions
impl From<crate::types::Address> for UndoReportEncoder {
    fn from(reportee: crate::types::Address) -> Self {
        Self {
            reportee: Some(reportee.to_string()),
        }
    }
}

// SetEncoderCommissionRate conversions
impl From<u64> for SetEncoderCommissionRate {
    fn from(new_rate: u64) -> Self {
        Self {
            new_rate: Some(new_rate),
        }
    }
}

// SetEncoderBytePrice conversions
impl From<u64> for SetEncoderBytePrice {
    fn from(new_price: u64) -> Self {
        Self {
            new_price: Some(new_price),
        }
    }
}

// TransferCoin conversions (create a wrapper struct)
pub struct TransferCoinArgs {
    pub coin: crate::types::ObjectReference,
    pub amount: Option<u64>,
    pub recipient: crate::types::Address,
}

impl From<TransferCoinArgs> for TransferCoin {
    fn from(args: TransferCoinArgs) -> Self {
        Self {
            coin: Some(args.coin.into()),
            amount: args.amount,
            recipient: Some(args.recipient.to_string()),
        }
    }
}

// PayCoins conversions
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

// AddStakeToEncoder conversions
pub struct AddStakeToEncoderArgs {
    pub encoder_address: crate::types::Address,
    pub coin_ref: crate::types::ObjectReference,
    pub amount: Option<u64>,
}

impl From<AddStakeToEncoderArgs> for AddStakeToEncoder {
    fn from(args: AddStakeToEncoderArgs) -> Self {
        Self {
            encoder_address: Some(args.encoder_address.to_string()),
            coin_ref: Some(args.coin_ref.into()),
            amount: args.amount,
        }
    }
}

// WithdrawStake conversions
impl From<crate::types::ObjectReference> for WithdrawStake {
    fn from(staked_soma: crate::types::ObjectReference) -> Self {
        Self {
            staked_soma: Some(staked_soma.into()),
        }
    }
}

// EmbedData conversions
pub struct EmbedDataArgs {
    pub digest: String, // Or whatever the digest type is
    pub data_size_bytes: usize,
    pub coin_ref: crate::types::ObjectReference,
}

impl From<EmbedDataArgs> for EmbedData {
    fn from(args: EmbedDataArgs) -> Self {
        Self {
            digest: Some(args.digest),
            data_size_bytes: Some(args.data_size_bytes as u32),
            coin_ref: Some(args.coin_ref.into()),
        }
    }
}

// ClaimEscrow conversions
impl From<crate::types::ObjectReference> for ClaimEscrow {
    fn from(shard_input_ref: crate::types::ObjectReference) -> Self {
        Self {
            shard_input_ref: Some(shard_input_ref.into()),
        }
    }
}

// ReportScores conversions
pub struct ReportScoresArgs {
    pub shard_input_ref: crate::types::ObjectReference,
    pub scores: Vec<u8>,
    pub encoder_aggregate_signature: Vec<u8>,
    pub signers: Vec<String>,
}

impl From<ReportScoresArgs> for ReportScores {
    fn from(args: ReportScoresArgs) -> Self {
        Self {
            shard_input_ref: Some(args.shard_input_ref.into()),
            scores: Some(Bcs::serialize(&args.scores).unwrap()),
            encoder_aggregate_signature: Some(
                Bcs::serialize(&args.encoder_aggregate_signature).unwrap(),
            ),
            signers: args.signers,
        }
    }
}
