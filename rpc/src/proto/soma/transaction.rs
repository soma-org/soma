use super::*;
use crate::proto::TryFromProtoError;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;
use tap::Pipe;

impl From<crate::types::Metadata> for Metadata {
    fn from(value: crate::types::Metadata) -> Self {
        use metadata::Version;

        let mut message = Self::default();
        match value {
            crate::types::Metadata::V1(v1) => {
                let mut proto_v1 = MetadataV1::default();
                proto_v1.checksum = Some(v1.checksum.into());
                proto_v1.size = Some(v1.size);
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
                    .as_ref()
                    .try_into()
                    .map_err(|_| {
                        TryFromProtoError::invalid("checksum", "invalid checksum length")
                    })?;

                let size = v1.size.ok_or_else(|| TryFromProtoError::missing("size"))?;

                Ok(crate::types::Metadata::V1(crate::types::MetadataV1 {
                    checksum,
                    size,
                }))
            }
        }
    }
}

impl From<crate::types::DownloadMetadata> for DownloadMetadata {
    fn from(value: crate::types::DownloadMetadata) -> Self {
        use download_metadata::Kind;

        let kind = match value {
            crate::types::DownloadMetadata::Default(dm) => Kind::Default(dm.into()),
            crate::types::DownloadMetadata::Mtls(dm) => Kind::Mtls(dm.into()),
        };

        DownloadMetadata { kind: Some(kind) }
    }
}
impl From<crate::types::DefaultDownloadMetadata> for DefaultDownloadMetadata {
    fn from(value: crate::types::DefaultDownloadMetadata) -> Self {
        use default_download_metadata::Version;

        match value {
            crate::types::DefaultDownloadMetadata::V1(v1) => DefaultDownloadMetadata {
                version: Some(Version::V1(DefaultDownloadMetadataV1 {
                    url: Some(v1.url.to_string()), // Url -> String
                    metadata: Some(v1.metadata.into()),
                })),
            },
        }
    }
}

impl From<crate::types::MtlsDownloadMetadata> for MtlsDownloadMetadata {
    fn from(value: crate::types::MtlsDownloadMetadata) -> Self {
        use mtls_download_metadata::Version;

        match value {
            crate::types::MtlsDownloadMetadata::V1(v1) => MtlsDownloadMetadata {
                version: Some(Version::V1(MtlsDownloadMetadataV1 {
                    peer: Some(v1.peer.into()),
                    url: Some(v1.url.to_string()), // Url -> String
                    metadata: Some(v1.metadata.into()),
                })),
            },
        }
    }
}

impl TryFrom<&DownloadMetadata> for crate::types::DownloadMetadata {
    type Error = TryFromProtoError;

    fn try_from(value: &DownloadMetadata) -> Result<Self, Self::Error> {
        use download_metadata::Kind;

        match value
            .kind
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("kind"))?
        {
            Kind::Default(dm) => Ok(Self::Default(dm.try_into()?)),
            Kind::Mtls(dm) => Ok(Self::Mtls(dm.try_into()?)),
        }
    }
}

impl TryFrom<&DefaultDownloadMetadata> for crate::types::DefaultDownloadMetadata {
    type Error = TryFromProtoError;

    fn try_from(value: &DefaultDownloadMetadata) -> Result<Self, Self::Error> {
        use default_download_metadata::Version;

        match value
            .version
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("version"))?
        {
            Version::V1(v1) => Ok(Self::V1(crate::types::DefaultDownloadMetadataV1 {
                url: v1
                    .url
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("url"))?
                    .parse() // String -> Url
                    .map_err(|e| TryFromProtoError::invalid("url", e))?,
                metadata: v1
                    .metadata
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("metadata"))?
                    .try_into()?,
            })),
        }
    }
}

impl TryFrom<&MtlsDownloadMetadata> for crate::types::MtlsDownloadMetadata {
    type Error = TryFromProtoError;

    fn try_from(value: &MtlsDownloadMetadata) -> Result<Self, Self::Error> {
        use mtls_download_metadata::Version;

        match value
            .version
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("version"))?
        {
            Version::V1(v1) => Ok(Self::V1(crate::types::MtlsDownloadMetadataV1 {
                peer: v1
                    .peer
                    .clone()
                    .ok_or_else(|| TryFromProtoError::missing("peer"))?
                    .to_vec(),
                url: v1
                    .url
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("url"))?
                    .parse() // String -> Url
                    .map_err(|e| TryFromProtoError::invalid("url", e))?,
                metadata: v1
                    .metadata
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("metadata"))?
                    .try_into()?,
            })),
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
        let Transaction {
            digest,
            kind,
            sender,
            gas_payment,
        } = source;

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
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<_, _>>()?;

        Ok(Self {
            kind,
            sender,
            gas_payment,
        })
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
                    amount,
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
                    amounts,
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
                    amount,
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
                    amount,
                }
                .into(),
            ),
            WithdrawStake { staked_soma } => Kind::WithdrawStake(staked_soma.into()),

            // Shard operations
            EmbedData {
                download_metadata,
                coin_ref,
                target_ref,
            } => Kind::EmbedData(
                EmbedDataArgs {
                    download_metadata,
                    coin_ref,
                    target_ref,
                }
                .into(),
            ),
            ClaimEscrow { shard_ref } => Kind::ClaimEscrow(shard_ref.into()),
            ReportWinner {
                shard_ref,
                target_ref,
                report,
                signature,
                signers,
                shard_auth_token,
            } => Kind::ReportWinner(
                ReportWinnerArgs {
                    shard_ref,
                    target_ref,
                    report,
                    signature,
                    signers,
                    shard_auth_token,
                }
                .into(),
            ),
            ClaimReward { target_ref } => Kind::ClaimReward(target_ref.into()),
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
                amount: transfer.amount,
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
                amounts: Some(pay.amounts.clone()),
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
                amount: stake.amount,
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
                amount: stake.amount,
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
                download_metadata: embed
                    .download_metadata
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("download_metadata"))?
                    .try_into()?,
                coin_ref: embed
                    .coin_ref
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("coin_ref"))?
                    .try_into()?,
                target_ref: embed
                    .target_ref
                    .as_ref()
                    .map(|r| r.try_into())
                    .transpose()?,
            },
            Kind::ClaimEscrow(claim) => Self::ClaimEscrow {
                shard_ref: claim
                    .shard_ref
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("shard_ref"))?
                    .try_into()?,
            },
            Kind::ReportWinner(report) => Self::ReportWinner {
                shard_ref: report
                    .shard_ref
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("shard_input_ref"))?
                    .try_into()?,
                target_ref: report
                    .target_ref
                    .as_ref()
                    .map(|r| r.try_into())
                    .transpose()?,
                report: report
                    .report
                    .clone()
                    .ok_or_else(|| TryFromProtoError::missing("scores"))?
                    .into(),
                signature: report
                    .signature
                    .clone()
                    .ok_or_else(|| TryFromProtoError::missing("signature"))?
                    .into(),
                signers: report
                    .signers
                    .iter()
                    .map(|s| {
                        s.parse()
                            .map_err(|e| TryFromProtoError::invalid("signers", e))
                    })
                    .collect::<Result<_, _>>()?,
                shard_auth_token: report
                    .shard_auth_token
                    .clone()
                    .ok_or_else(|| TryFromProtoError::missing("scores"))?
                    .into(),
            },
            Kind::ClaimReward(claim) => Self::ClaimReward {
                target_ref: claim
                    .target_ref
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("target_ref"))?
                    .try_into()?,
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
            pubkey_bytes: Some(value.pubkey_bytes.into()),
            network_pubkey_bytes: Some(value.network_pubkey_bytes.into()),
            worker_pubkey_bytes: Some(value.worker_pubkey_bytes.into()),
            net_address: Some(value.net_address.into()),
            p2p_address: Some(value.p2p_address.into()),
            primary_address: Some(value.primary_address.into()),
            encoder_validator_address: Some(value.encoder_validator_address.into()),
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
            encoder_validator_address: value
                .encoder_validator_address
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("encoder_validator_address"))?
                .into(),
        })
    }
}

// RemoveValidatorArgs conversions
impl From<crate::types::RemoveValidatorArgs> for RemoveValidator {
    fn from(value: crate::types::RemoveValidatorArgs) -> Self {
        Self {
            pubkey_bytes: Some(value.pubkey_bytes.into()),
        }
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
            next_epoch_protocol_pubkey: value.next_epoch_protocol_pubkey.map(|v| v.into()),
            next_epoch_worker_pubkey: value.next_epoch_worker_pubkey.map(|v| v.into()),
            next_epoch_network_pubkey: value.next_epoch_network_pubkey.map(|v| v.into()),
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
            next_epoch_protocol_pubkey: value.next_epoch_protocol_pubkey.clone().map(|v| v.into()),
            next_epoch_worker_pubkey: value.next_epoch_worker_pubkey.clone().map(|v| v.into()),
            next_epoch_network_pubkey: value.next_epoch_network_pubkey.clone().map(|v| v.into()),
        })
    }
}

// AddEncoderArgs conversions
impl From<crate::types::AddEncoderArgs> for AddEncoder {
    fn from(value: crate::types::AddEncoderArgs) -> Self {
        Self {
            encoder_pubkey_bytes: Some(value.encoder_pubkey_bytes.into()),
            network_pubkey_bytes: Some(value.network_pubkey_bytes.into()),
            internal_network_address: Some(value.internal_network_address.into()),
            external_network_address: Some(value.external_network_address.into()),
            object_server_address: Some(value.object_server_address.into()),
            probe: Some(value.probe.into()),
        }
    }
}

impl TryFrom<&AddEncoder> for crate::types::AddEncoderArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &AddEncoder) -> Result<Self, Self::Error> {
        Ok(Self {
            encoder_pubkey_bytes: value
                .encoder_pubkey_bytes
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("encoder_pubkey_bytes"))?
                .into(),
            network_pubkey_bytes: value
                .network_pubkey_bytes
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("network_pubkey_bytes"))?
                .into(),
            internal_network_address: value
                .internal_network_address
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("internal_network_address"))?
                .into(),
            external_network_address: value
                .external_network_address
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("external_network_address"))?
                .into(),
            object_server_address: value
                .object_server_address
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("object_server_address"))?
                .into(),
            probe: value
                .probe
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("probe"))?
                .into(),
        })
    }
}

// RemoveEncoderArgs conversions
impl From<crate::types::RemoveEncoderArgs> for RemoveEncoder {
    fn from(value: crate::types::RemoveEncoderArgs) -> Self {
        Self {
            encoder_pubkey_bytes: Some(value.encoder_pubkey_bytes.into()),
        }
    }
}

impl TryFrom<&RemoveEncoder> for crate::types::RemoveEncoderArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &RemoveEncoder) -> Result<Self, Self::Error> {
        Ok(Self {
            encoder_pubkey_bytes: value
                .encoder_pubkey_bytes
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("encoder_pubkey_bytes"))?
                .into(),
        })
    }
}

// UpdateEncoderMetadataArgs conversions
impl From<crate::types::UpdateEncoderMetadataArgs> for UpdateEncoderMetadata {
    fn from(value: crate::types::UpdateEncoderMetadataArgs) -> Self {
        Self {
            next_epoch_external_network_address: value
                .next_epoch_external_network_address
                .map(|v| v.into()),
            next_epoch_internal_network_address: value
                .next_epoch_internal_network_address
                .map(|v| v.into()),
            next_epoch_network_pubkey: value.next_epoch_network_pubkey.map(|v| v.into()),
            next_epoch_object_server_address: value
                .next_epoch_object_server_address
                .map(|v| v.into()),
            next_epoch_probe: value.next_epoch_probe.map(|v| v.into()),
        }
    }
}

impl TryFrom<&UpdateEncoderMetadata> for crate::types::UpdateEncoderMetadataArgs {
    type Error = TryFromProtoError;

    fn try_from(value: &UpdateEncoderMetadata) -> Result<Self, Self::Error> {
        Ok(Self {
            next_epoch_external_network_address: value
                .next_epoch_external_network_address
                .clone()
                .map(|v| v.into()),
            next_epoch_internal_network_address: value
                .next_epoch_internal_network_address
                .clone()
                .map(|v| v.into()),
            next_epoch_network_pubkey: value.next_epoch_network_pubkey.clone().map(|v| v.into()),
            next_epoch_object_server_address: value
                .next_epoch_object_server_address
                .clone()
                .map(|v| v.into()),
            next_epoch_probe: value.next_epoch_probe.clone().map(|v| v.into()),
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
            consensus_commit_digest: Some(value.consensus_commit_digest.to_string()),
            additional_state_digest: Some(value.additional_state_digest.to_string()),
            sub_dag_index: value.sub_dag_index,
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

        Ok(Self {
            epoch,
            protocol_version,
            fees,
            epoch_start_timestamp_ms,
            epoch_randomness,
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
    pub download_metadata: crate::types::DownloadMetadata,
    pub coin_ref: crate::types::ObjectReference,
    pub target_ref: Option<crate::types::ObjectReference>,
}

impl From<EmbedDataArgs> for EmbedData {
    fn from(args: EmbedDataArgs) -> Self {
        Self {
            download_metadata: Some(args.download_metadata.into()),
            coin_ref: Some(args.coin_ref.into()),
            target_ref: args.target_ref.map(|r| r.into()),
        }
    }
}

// ClaimEscrow conversions
impl From<crate::types::ObjectReference> for ClaimEscrow {
    fn from(shard_ref: crate::types::ObjectReference) -> Self {
        Self {
            shard_ref: Some(shard_ref.into()),
        }
    }
}

impl From<crate::types::ObjectReference> for ClaimReward {
    fn from(target_ref: crate::types::ObjectReference) -> Self {
        Self {
            target_ref: Some(target_ref.into()),
        }
    }
}

// ReportScores conversions
pub struct ReportWinnerArgs {
    pub shard_ref: crate::types::ObjectReference,
    pub target_ref: Option<crate::types::ObjectReference>,
    pub report: Vec<u8>,
    pub signature: Vec<u8>,
    pub signers: Vec<String>,
    pub shard_auth_token: Vec<u8>,
}

impl From<ReportWinnerArgs> for ReportWinner {
    fn from(args: ReportWinnerArgs) -> Self {
        Self {
            shard_ref: Some(args.shard_ref.into()),
            target_ref: args.target_ref.map(|r| r.into()),
            report: Some(args.report.into()),
            signature: Some(args.signature.into()),
            signers: args.signers,
            shard_auth_token: Some(args.shard_auth_token.into()),
        }
    }
}
