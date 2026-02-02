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
            WithdrawStake { staked_soma } => Kind::WithdrawStake(staked_soma.into()),
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

            Kind::WithdrawStake(withdraw) => Self::WithdrawStake {
                staked_soma: withdraw
                    .staked_soma
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("staked_soma"))?
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

// WithdrawStake conversions
impl From<crate::types::ObjectReference> for WithdrawStake {
    fn from(staked_soma: crate::types::ObjectReference) -> Self {
        Self {
            staked_soma: Some(staked_soma.into()),
        }
    }
}
