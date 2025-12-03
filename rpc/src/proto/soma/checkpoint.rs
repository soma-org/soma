use super::*;
use crate::proto::TryFromProtoError;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;
use tap::Pipe;

//
// CheckpointSummary
//

impl From<crate::types::CheckpointSummary> for CheckpointSummary {
    fn from(summary: crate::types::CheckpointSummary) -> Self {
        Self::merge_from(summary, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<crate::types::CheckpointSummary> for CheckpointSummary {
    fn merge(&mut self, source: crate::types::CheckpointSummary, mask: &FieldMaskTree) {
        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = Some(source.digest().to_string());
        }

        let crate::types::CheckpointSummary {
            epoch,
            sequence_number,
            network_total_transactions,
            content_digest,
            previous_digest,
            // TODO: epoch_rolling_gas_cost_summary,
            timestamp_ms,
            checkpoint_commitments,
            end_of_epoch_data,
        } = source;

        if mask.contains(Self::EPOCH_FIELD.name) {
            self.epoch = Some(epoch);
        }

        if mask.contains(Self::SEQUENCE_NUMBER_FIELD.name) {
            self.sequence_number = Some(sequence_number);
        }

        if mask.contains(Self::TOTAL_NETWORK_TRANSACTIONS_FIELD.name) {
            self.total_network_transactions = Some(network_total_transactions);
        }

        if mask.contains(Self::CONTENT_DIGEST_FIELD.name) {
            self.content_digest = Some(content_digest.to_string());
        }

        if mask.contains(Self::PREVIOUS_DIGEST_FIELD.name) {
            self.previous_digest = previous_digest.map(|d| d.to_string());
        }

        // if mask.contains(Self::EPOCH_ROLLING_GAS_COST_SUMMARY_FIELD.name) {
        //     self.epoch_rolling_gas_cost_summary = Some(epoch_rolling_gas_cost_summary.into());
        // }

        if mask.contains(Self::TIMESTAMP_FIELD.name) {
            self.timestamp = Some(crate::proto::timestamp_ms_to_proto(timestamp_ms));
        }

        if mask.contains(Self::COMMITMENTS_FIELD.name) {
            self.commitments = checkpoint_commitments.into_iter().map(Into::into).collect();
        }

        if mask.contains(Self::END_OF_EPOCH_DATA_FIELD.name) {
            self.end_of_epoch_data = end_of_epoch_data.map(Into::into);
        }
    }
}

impl Merge<&CheckpointSummary> for CheckpointSummary {
    fn merge(&mut self, source: &CheckpointSummary, mask: &FieldMaskTree) {
        let CheckpointSummary {
            digest,
            epoch,
            sequence_number,
            total_network_transactions,
            content_digest,
            previous_digest,
            // epoch_rolling_gas_cost_summary,
            timestamp,
            commitments,
            end_of_epoch_data,
        } = source;

        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = digest.clone();
        }

        if mask.contains(Self::EPOCH_FIELD.name) {
            self.epoch = *epoch;
        }

        if mask.contains(Self::SEQUENCE_NUMBER_FIELD.name) {
            self.sequence_number = *sequence_number;
        }

        if mask.contains(Self::TOTAL_NETWORK_TRANSACTIONS_FIELD.name) {
            self.total_network_transactions = *total_network_transactions;
        }

        if mask.contains(Self::CONTENT_DIGEST_FIELD.name) {
            self.content_digest = content_digest.clone();
        }

        if mask.contains(Self::PREVIOUS_DIGEST_FIELD.name) {
            self.previous_digest = previous_digest.clone();
        }

        // if mask.contains(Self::EPOCH_ROLLING_GAS_COST_SUMMARY_FIELD.name) {
        //     self.epoch_rolling_gas_cost_summary = *epoch_rolling_gas_cost_summary;
        // }

        if mask.contains(Self::TIMESTAMP_FIELD.name) {
            self.timestamp = *timestamp;
        }

        if mask.contains(Self::COMMITMENTS_FIELD.name) {
            self.commitments = commitments.clone();
        }

        if mask.contains(Self::END_OF_EPOCH_DATA_FIELD.name) {
            self.end_of_epoch_data = end_of_epoch_data.clone();
        }
    }
}

impl TryFrom<&CheckpointSummary> for crate::types::CheckpointSummary {
    type Error = TryFromProtoError;

    fn try_from(
        CheckpointSummary {
            digest: _,
            epoch,
            sequence_number,
            total_network_transactions,
            content_digest,
            previous_digest,
            // epoch_rolling_gas_cost_summary,
            timestamp,
            commitments,
            end_of_epoch_data,
        }: &CheckpointSummary,
    ) -> Result<Self, Self::Error> {
        let epoch = epoch.ok_or_else(|| TryFromProtoError::missing("epoch"))?;
        let sequence_number =
            sequence_number.ok_or_else(|| TryFromProtoError::missing("sequence_number"))?;
        let network_total_transactions = total_network_transactions
            .ok_or_else(|| TryFromProtoError::missing("total_network_transactions"))?;
        let content_digest = content_digest
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("content_digest"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(CheckpointSummary::CONTENT_DIGEST_FIELD, e))?;
        let previous_digest = previous_digest
            .as_ref()
            .map(|s| {
                s.parse().map_err(|e| {
                    TryFromProtoError::invalid(CheckpointSummary::PREVIOUS_DIGEST_FIELD, e)
                })
            })
            .transpose()?;
        // let epoch_rolling_gas_cost_summary = epoch_rolling_gas_cost_summary
        //     .as_ref()
        //     .ok_or_else(|| TryFromProtoError::missing("epoch_rolling_gas_cost_summary"))?
        //     .try_into()?;

        let timestamp_ms = timestamp
            .ok_or_else(|| TryFromProtoError::missing("timestamp_ms"))?
            .pipe(crate::proto::proto_to_timestamp_ms)?;

        let checkpoint_commitments = commitments
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<_, _>>()?;

        let end_of_epoch_data = end_of_epoch_data
            .as_ref()
            .map(TryInto::try_into)
            .transpose()?;

        Ok(Self {
            epoch,
            sequence_number,
            network_total_transactions,
            content_digest,
            previous_digest,
            // epoch_rolling_gas_cost_summary,
            timestamp_ms,
            checkpoint_commitments,
            end_of_epoch_data,
        })
    }
}

//
// TODO: GasCostSummary
//

//
// CheckpointCommitment
//

impl From<crate::types::CheckpointCommitment> for CheckpointCommitment {
    fn from(value: crate::types::CheckpointCommitment) -> Self {
        use checkpoint_commitment::CheckpointCommitmentKind;

        let mut message = Self::default();

        let kind = match value {
            crate::types::CheckpointCommitment::EcmhLiveObjectSet { digest } => {
                message.digest = Some(digest.to_string());
                CheckpointCommitmentKind::EcmhLiveObjectSet
            }
            crate::types::CheckpointCommitment::CheckpointArtifacts { digest } => {
                message.digest = Some(digest.to_string());
                CheckpointCommitmentKind::CheckpointArtifacts
            }
            _ => CheckpointCommitmentKind::Unknown,
        };

        message.set_kind(kind);
        message
    }
}

impl TryFrom<&CheckpointCommitment> for crate::types::CheckpointCommitment {
    type Error = TryFromProtoError;

    fn try_from(value: &CheckpointCommitment) -> Result<Self, Self::Error> {
        use checkpoint_commitment::CheckpointCommitmentKind;

        match value.kind() {
            CheckpointCommitmentKind::Unknown => {
                return Err(TryFromProtoError::invalid(
                    CheckpointCommitment::KIND_FIELD,
                    "unknown CheckpointCommitmentKind",
                ));
            }
            CheckpointCommitmentKind::EcmhLiveObjectSet => Self::EcmhLiveObjectSet {
                digest: value.digest().parse().map_err(|e| {
                    TryFromProtoError::invalid(CheckpointCommitment::DIGEST_FIELD, e)
                })?,
            },
            CheckpointCommitmentKind::CheckpointArtifacts => Self::CheckpointArtifacts {
                digest: value.digest().parse().map_err(|e| {
                    TryFromProtoError::invalid(CheckpointCommitment::DIGEST_FIELD, e)
                })?,
            },
        }
        .pipe(Ok)
    }
}

//
// EndOfEpochData
//

impl From<crate::types::EndOfEpochData> for EndOfEpochData {
    fn from(
        crate::types::EndOfEpochData {
            next_epoch_validator_committee,
            epoch_commitments,
        }: crate::types::EndOfEpochData,
    ) -> Self {
        Self {
            next_epoch_validator_committee: Some(next_epoch_validator_committee.into()),
            // TODO: next_epoch_protocol_version: Some(next_epoch_protocol_version),
            epoch_commitments: epoch_commitments.into_iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<&EndOfEpochData> for crate::types::EndOfEpochData {
    type Error = TryFromProtoError;

    fn try_from(
        EndOfEpochData {
            next_epoch_validator_committee,
            // next_epoch_protocol_version,
            epoch_commitments,
        }: &EndOfEpochData,
    ) -> Result<Self, Self::Error> {
        // let next_epoch_protocol_version = next_epoch_protocol_version
        //     .ok_or_else(|| TryFromProtoError::missing("next_epoch_protocol_version"))?;

        Ok(Self {
            next_epoch_validator_committee: next_epoch_validator_committee
                .clone()
                .ok_or_else(|| TryFromProtoError::missing("next_epoch_validator_committee"))?
                .try_into()
                .map_err(|e| TryFromProtoError::invalid("next_epoch_validator_committee", e))?,
            // next_epoch_protocol_version,
            epoch_commitments: epoch_commitments
                .iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        })
    }
}

//
// CheckpointedTransactionInfo
//

impl From<crate::types::CheckpointTransactionInfo> for CheckpointedTransactionInfo {
    fn from(value: crate::types::CheckpointTransactionInfo) -> Self {
        Self {
            transaction: Some(value.transaction.to_string()),
            effects: Some(value.effects.to_string()),
            signatures: value.signatures.into_iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<&CheckpointedTransactionInfo> for crate::types::CheckpointTransactionInfo {
    type Error = TryFromProtoError;

    fn try_from(value: &CheckpointedTransactionInfo) -> Result<Self, Self::Error> {
        let transaction = value
            .transaction
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("transaction"))?
            .parse()
            .map_err(|e| {
                TryFromProtoError::invalid(CheckpointedTransactionInfo::TRANSACTION_FIELD, e)
            })?;

        let effects = value
            .effects
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("effects"))?
            .parse()
            .map_err(|e| {
                TryFromProtoError::invalid(CheckpointedTransactionInfo::EFFECTS_FIELD, e)
            })?;

        let signatures = value
            .signatures
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<_, _>>()?;

        Ok(Self {
            transaction,
            effects,
            signatures,
        })
    }
}

//
// CheckpointContents
//

impl From<crate::types::CheckpointContents> for CheckpointContents {
    fn from(value: crate::types::CheckpointContents) -> Self {
        Self::merge_from(value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<crate::types::CheckpointContents> for CheckpointContents {
    fn merge(&mut self, source: crate::types::CheckpointContents, mask: &FieldMaskTree) {
        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = Some(source.digest().to_string());
        }

        if mask.contains(Self::VERSION_FIELD.name) {
            self.version = Some(1);
        }

        if mask.contains(Self::TRANSACTIONS_FIELD.name) {
            self.transactions = source.into_v1().into_iter().map(Into::into).collect();
        }
    }
}

impl Merge<&CheckpointContents> for CheckpointContents {
    fn merge(&mut self, source: &CheckpointContents, mask: &FieldMaskTree) {
        let CheckpointContents {
            digest,
            version,
            transactions,
        } = source;

        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = digest.clone();
        }

        if mask.contains(Self::VERSION_FIELD.name) {
            self.version = *version;
        }

        if mask.contains(Self::TRANSACTIONS_FIELD.name) {
            self.transactions = transactions.clone();
        }
    }
}

impl TryFrom<&CheckpointContents> for crate::types::CheckpointContents {
    type Error = TryFromProtoError;

    fn try_from(value: &CheckpointContents) -> Result<Self, Self::Error> {
        match value.version {
            Some(1) => {}
            v => {
                return Err(TryFromProtoError::invalid(
                    CheckpointContents::VERSION_FIELD,
                    format!("unknown type version {v:?}"),
                ));
            }
        }

        Ok(Self::new(
            value
                .transactions
                .iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        ))
    }
}

//
// Checkpoint
//

impl Merge<&crate::types::CheckpointSummary> for Checkpoint {
    fn merge(&mut self, source: &crate::types::CheckpointSummary, mask: &FieldMaskTree) {
        if mask.contains(Self::SEQUENCE_NUMBER_FIELD.name) {
            self.sequence_number = Some(source.sequence_number);
        }

        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = Some(source.digest().to_string());
        }

        if let Some(submask) = mask.subtree(Self::SUMMARY_FIELD.name) {
            self.summary = Some(CheckpointSummary::merge_from(source.clone(), &submask));
        }
    }
}

impl Merge<crate::types::ValidatorAggregatedSignature> for Checkpoint {
    fn merge(&mut self, source: crate::types::ValidatorAggregatedSignature, mask: &FieldMaskTree) {
        if mask.contains(Self::SIGNATURE_FIELD.name) {
            self.signature = Some(source.into());
        }
    }
}

impl Merge<crate::types::CheckpointContents> for Checkpoint {
    fn merge(&mut self, source: crate::types::CheckpointContents, mask: &FieldMaskTree) {
        if let Some(submask) = mask.subtree(Self::CONTENTS_FIELD.name) {
            self.contents = Some(CheckpointContents::merge_from(source, &submask));
        }
    }
}

impl Merge<&Checkpoint> for Checkpoint {
    fn merge(&mut self, source: &Checkpoint, mask: &FieldMaskTree) {
        let Checkpoint {
            sequence_number,
            digest,
            summary,
            signature,
            contents,
            transactions,
            objects,
        } = source;

        if mask.contains(Self::SEQUENCE_NUMBER_FIELD.name) {
            self.sequence_number = *sequence_number;
        }

        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = digest.clone();
        }

        if let Some(submask) = mask.subtree(Self::SUMMARY_FIELD.name) {
            self.summary = summary
                .as_ref()
                .map(|summary| CheckpointSummary::merge_from(summary, &submask));
        }

        if mask.contains(Self::SIGNATURE_FIELD.name) {
            self.signature = signature.clone();
        }

        if let Some(submask) = mask.subtree(Self::CONTENTS_FIELD.name) {
            self.contents = contents
                .as_ref()
                .map(|contents| CheckpointContents::merge_from(contents, &submask));
        }

        if let Some(submask) = mask.subtree(Self::TRANSACTIONS_FIELD.name) {
            self.transactions = transactions
                .iter()
                .map(|transaction| ExecutedTransaction::merge_from(transaction, &submask))
                .collect();
        }

        if let Some(submask) = mask.subtree(Self::OBJECTS_FIELD) {
            self.objects = objects
                .as_ref()
                .map(|objects| ObjectSet::merge_from(objects, &submask));
        }
    }
}
