use super::*;
use crate::proto::TryFromProtoError;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;
use tap::Pipe;

//
// TransactionEffects
//

impl From<crate::types::TransactionEffects> for TransactionEffects {
    fn from(value: crate::types::TransactionEffects) -> Self {
        Self::merge_from(&value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<&TransactionEffects> for TransactionEffects {
    fn merge(
        &mut self,
        TransactionEffects {
            status,
            epoch,
            fee,
            transaction_digest,
            gas_object_index,
            dependencies,
            lamport_version,
            changed_objects,
            unchanged_shared_objects,
        }: &TransactionEffects,
        mask: &FieldMaskTree,
    ) {
        if mask.contains(Self::STATUS_FIELD.name) {
            self.status = status.clone();
        }

        if mask.contains(Self::EPOCH_FIELD.name) {
            self.epoch = *epoch;
        }

        if mask.contains(Self::FEE_FIELD.name) {
            self.fee = fee.clone();
        }

        if mask.contains(Self::TRANSACTION_DIGEST_FIELD.name) {
            self.transaction_digest = transaction_digest.clone();
        }

        if mask.contains(Self::GAS_OBJECT_INDEX_FIELD.name) {
            self.gas_object_index = gas_object_index.clone();
        }

        if mask.contains(Self::DEPENDENCIES_FIELD.name) {
            self.dependencies = dependencies.clone();
        }

        if mask.contains(Self::LAMPORT_VERSION_FIELD.name) {
            self.lamport_version = *lamport_version;
        }

        if mask.contains(Self::CHANGED_OBJECTS_FIELD.name) {
            self.changed_objects = changed_objects.clone();
        }

        if mask.contains(Self::UNCHANGED_SHARED_OBJECTS_FIELD.name) {
            self.unchanged_shared_objects = unchanged_shared_objects.clone();
        }
    }
}

impl TryFrom<&TransactionEffects> for crate::types::TransactionEffects {
    type Error = TryFromProtoError;

    fn try_from(value: &TransactionEffects) -> Result<Self, Self::Error> {
        Ok(Self {
            status: value
                .status
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("status"))?
                .try_into()?,
            epoch: value.epoch.ok_or_else(|| TryFromProtoError::missing("epoch"))?,
            fee: value.fee.as_ref().ok_or_else(|| TryFromProtoError::missing("fee"))?.try_into()?,
            transaction_digest: value
                .transaction_digest
                .as_ref()
                .ok_or_else(|| TryFromProtoError::missing("transaction_digest"))?
                .parse()
                .map_err(|e| TryFromProtoError::invalid("transaction_digest", e))?,
            dependencies: value
                .dependencies
                .iter()
                .map(|d| d.parse().map_err(|e| TryFromProtoError::invalid("dependencies", e)))
                .collect::<Result<Vec<_>, _>>()?,
            gas_object_index: value.gas_object_index,
            lamport_version: value
                .lamport_version
                .ok_or_else(|| TryFromProtoError::missing("lamport_version"))?,
            changed_objects: value
                .changed_objects
                .iter()
                .map(TryInto::try_into)
                .collect::<Result<Vec<_>, _>>()?,
            unchanged_shared_objects: value
                .unchanged_shared_objects
                .iter()
                .map(TryInto::try_into)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

//
// TransactionEffects
//

impl Merge<&crate::types::TransactionEffects> for TransactionEffects {
    fn merge(
        &mut self,
        crate::types::TransactionEffects {
            status,
            epoch,
            fee,
            transaction_digest,
            gas_object_index,
            dependencies,
            lamport_version,
            changed_objects,
            unchanged_shared_objects,
        }: &crate::types::TransactionEffects,
        mask: &FieldMaskTree,
    ) {
        if mask.contains(Self::STATUS_FIELD.name) {
            self.status = Some(status.clone().into());
        }

        if mask.contains(Self::EPOCH_FIELD.name) {
            self.epoch = Some(*epoch);
        }

        if mask.contains(Self::FEE_FIELD.name) {
            self.fee = Some(fee.clone().into());
        }

        if mask.contains(Self::TRANSACTION_DIGEST_FIELD.name) {
            self.transaction_digest = Some(transaction_digest.to_string());
        }

        if mask.contains(Self::GAS_OBJECT_INDEX_FIELD.name) {
            self.gas_object_index = gas_object_index.clone();
        }

        if mask.contains(Self::DEPENDENCIES_FIELD.name) {
            self.dependencies = dependencies.iter().map(ToString::to_string).collect();
        }

        if mask.contains(Self::LAMPORT_VERSION_FIELD.name) {
            self.lamport_version = Some(*lamport_version);
        }

        if mask.contains(Self::CHANGED_OBJECTS_FIELD.name) {
            self.changed_objects = changed_objects.clone().into_iter().map(Into::into).collect();
        }

        for object in self.changed_objects.iter_mut() {
            if object.output_digest.is_some() && object.output_version.is_none() {
                object.output_version = Some(*lamport_version);
            }
        }

        if mask.contains(Self::UNCHANGED_SHARED_OBJECTS_FIELD.name) {
            self.unchanged_shared_objects =
                unchanged_shared_objects.clone().into_iter().map(Into::into).collect();
        }
    }
}

//
// ChangedObject
//

impl From<crate::types::ChangedObject> for ChangedObject {
    fn from(value: crate::types::ChangedObject) -> Self {
        use changed_object::InputObjectState;
        use changed_object::OutputObjectState;

        let mut message =
            Self { object_id: Some(value.object_id.to_string()), ..Default::default() };

        // Input State
        let input_state = match value.input_state {
            crate::types::ObjectIn::NotExist => InputObjectState::DoesNotExist,
            crate::types::ObjectIn::Exist { version, digest, owner } => {
                message.input_version = Some(version);
                message.input_digest = Some(digest.to_string());
                message.input_owner = Some(owner.into());
                InputObjectState::Exists
            }
            _ => InputObjectState::Unknown,
        };
        message.set_input_state(input_state);

        // Output State
        let output_state = match value.output_state {
            crate::types::ObjectOut::NotExist => OutputObjectState::DoesNotExist,
            crate::types::ObjectOut::ObjectWrite { digest, owner } => {
                message.output_digest = Some(digest.to_string());
                message.output_owner = Some(owner.into());
                OutputObjectState::ObjectWrite
            }

            _ => OutputObjectState::Unknown,
        };
        message.set_output_state(output_state);

        message.set_id_operation(value.id_operation.into());
        message
    }
}

impl TryFrom<&ChangedObject> for crate::types::ChangedObject {
    type Error = TryFromProtoError;

    fn try_from(value: &ChangedObject) -> Result<Self, Self::Error> {
        use changed_object::InputObjectState;
        use changed_object::OutputObjectState;

        let object_id = value
            .object_id
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("object_id"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(ChangedObject::OBJECT_ID_FIELD, e))?;

        let input_state = match value.input_state() {
            InputObjectState::Unknown => {
                return Err(TryFromProtoError::invalid(
                    ChangedObject::INPUT_STATE_FIELD,
                    "unknown InputObjectState",
                ));
            }
            InputObjectState::DoesNotExist => crate::types::ObjectIn::NotExist,
            InputObjectState::Exists => crate::types::ObjectIn::Exist {
                version: value
                    .input_version
                    .ok_or_else(|| TryFromProtoError::missing("version"))?,
                digest: value
                    .input_digest
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("digest"))?
                    .parse()
                    .map_err(|e| {
                        TryFromProtoError::invalid(ChangedObject::INPUT_DIGEST_FIELD, e)
                    })?,
                owner: value
                    .input_owner
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("owner"))?
                    .try_into()?,
            },
        };

        let output_state = match value.output_state() {
            OutputObjectState::Unknown => {
                return Err(TryFromProtoError::invalid(
                    ChangedObject::OUTPUT_STATE_FIELD,
                    "unknown OutputObjectState",
                ));
            }
            OutputObjectState::DoesNotExist => crate::types::ObjectOut::NotExist,
            OutputObjectState::ObjectWrite => crate::types::ObjectOut::ObjectWrite {
                digest: value
                    .output_digest
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("digest"))?
                    .parse()
                    .map_err(|e| {
                        TryFromProtoError::invalid(ChangedObject::OUTPUT_DIGEST_FIELD, e)
                    })?,

                owner: value
                    .output_owner
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("owner"))?
                    .try_into()?,
            },
        };

        let id_operation = value.id_operation().try_into()?;

        Ok(Self { object_id, input_state, output_state, id_operation })
    }
}

//
// IdOperation
//

impl From<crate::types::IdOperation> for changed_object::IdOperation {
    fn from(value: crate::types::IdOperation) -> Self {
        use crate::types::IdOperation::*;

        match value {
            None => Self::None,
            Created => Self::Created,
            Deleted => Self::Deleted,
            _ => Self::Unknown,
        }
    }
}

impl TryFrom<changed_object::IdOperation> for crate::types::IdOperation {
    type Error = TryFromProtoError;

    fn try_from(value: changed_object::IdOperation) -> Result<Self, Self::Error> {
        use changed_object::IdOperation;

        match value {
            IdOperation::Unknown => {
                return Err(TryFromProtoError::invalid("id_operation", "unknown IdOperation"));
            }
            IdOperation::None => Self::None,
            IdOperation::Created => Self::Created,
            IdOperation::Deleted => Self::Deleted,
        }
        .pipe(Ok)
    }
}

//
// UnchangedSharedObject
//

impl From<crate::types::UnchangedSharedObject> for UnchangedSharedObject {
    fn from(value: crate::types::UnchangedSharedObject) -> Self {
        use crate::types::UnchangedSharedKind::*;
        use unchanged_shared_object::UnchangedSharedObjectKind;

        let mut message =
            Self { object_id: Some(value.object_id.to_string()), ..Default::default() };

        let kind = match value.kind {
            ReadOnlyRoot { version, digest } => {
                message.version = Some(version);
                message.digest = Some(digest.to_string());
                UnchangedSharedObjectKind::ReadOnlyRoot
            }
            MutateDeleted { version } => {
                message.version = Some(version);
                UnchangedSharedObjectKind::MutatedDeleted
            }
            ReadDeleted { version } => {
                message.version = Some(version);
                UnchangedSharedObjectKind::ReadDeleted
            }
            Canceled { version } => {
                message.version = Some(version);
                UnchangedSharedObjectKind::Canceled
            }
            // PerEpochConfig => UnchangedSharedObjectKind::PerEpochConfig,
            _ => UnchangedSharedObjectKind::Unknown,
        };

        message.set_kind(kind);
        message
    }
}

impl TryFrom<&UnchangedSharedObject> for crate::types::UnchangedSharedObject {
    type Error = TryFromProtoError;

    fn try_from(value: &UnchangedSharedObject) -> Result<Self, Self::Error> {
        use crate::types::UnchangedSharedKind;
        use unchanged_shared_object::UnchangedSharedObjectKind;

        let object_id = value
            .object_id
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("object_id"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(UnchangedSharedObject::OBJECT_ID_FIELD, e))?;

        let kind = match value.kind() {
            UnchangedSharedObjectKind::Unknown => {
                return Err(TryFromProtoError::invalid(
                    UnchangedSharedObject::KIND_FIELD,
                    "unknown InputKind",
                ));
            }

            UnchangedSharedObjectKind::ReadOnlyRoot => UnchangedSharedKind::ReadOnlyRoot {
                version: value.version.ok_or_else(|| TryFromProtoError::missing("version"))?,

                digest: value
                    .digest
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("digest"))?
                    .parse()
                    .map_err(|e| {
                        TryFromProtoError::invalid(UnchangedSharedObject::DIGEST_FIELD, e)
                    })?,
            },
            UnchangedSharedObjectKind::MutatedDeleted => UnchangedSharedKind::MutateDeleted {
                version: value.version.ok_or_else(|| TryFromProtoError::missing("version"))?,
            },
            UnchangedSharedObjectKind::ReadDeleted => UnchangedSharedKind::ReadDeleted {
                version: value.version.ok_or_else(|| TryFromProtoError::missing("version"))?,
            },
            UnchangedSharedObjectKind::Canceled => UnchangedSharedKind::Canceled {
                version: value.version.ok_or_else(|| TryFromProtoError::missing("version"))?,
            },
        };

        Ok(Self { object_id, kind })
    }
}

impl From<crate::types::TransactionFee> for TransactionFee {
    fn from(value: crate::types::TransactionFee) -> Self {
        Self {
            base_fee: Some(value.base_fee),
            operation_fee: Some(value.operation_fee),
            value_fee: Some(value.value_fee),
            total_fee: Some(value.total_fee),
        }
    }
}

impl TryFrom<&TransactionFee> for crate::types::TransactionFee {
    type Error = TryFromProtoError;

    fn try_from(value: &TransactionFee) -> Result<Self, Self::Error> {
        Ok(Self {
            base_fee: value.base_fee.ok_or_else(|| TryFromProtoError::missing("base_fee"))?,
            operation_fee: value
                .operation_fee
                .ok_or_else(|| TryFromProtoError::missing("operation_fee"))?,
            value_fee: value.value_fee.ok_or_else(|| TryFromProtoError::missing("value_fee"))?,
            total_fee: value.total_fee.ok_or_else(|| TryFromProtoError::missing("total_fee"))?,
        })
    }
}
