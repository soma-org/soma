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

impl From<crate::types::TransactionKind> for TransactionKind {
    fn from(value: crate::types::TransactionKind) -> Self {
        use crate::types::TransactionKind::*;
        use transaction_kind::Kind;

        let kind = match value {
            ChangeEpoch(change_epoch) => Kind::ChangeEpoch(change_epoch.into()),
            Genesis(genesis) => Kind::Genesis(genesis.into()),
            ConsensusCommitPrologue(prologue) => Kind::ConsensusCommitPrologue(prologue.into()),

            _ => return Self::default(),
        };

        Self { kind: Some(kind) }
    }
}

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
        }
        .pipe(Ok)
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
            objects: value.objects.into_iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<&GenesisTransaction> for crate::types::GenesisTransaction {
    type Error = TryFromProtoError;

    fn try_from(value: &GenesisTransaction) -> Result<Self, Self::Error> {
        let objects = value
            .objects
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<_, _>>()?;

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
