mod _field_impls {
    #![allow(clippy::wrong_self_convention)]
    use super::*;
    use crate::utils::field::MessageFields;
    use crate::utils::field::MessageField;
    impl BalanceChange {
        pub const ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "address",
            json_name: "address",
            number: 1i32,
            message_fields: None,
        };
        pub const AMOUNT_FIELD: &'static MessageField = &MessageField {
            name: "amount",
            json_name: "amount",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for BalanceChange {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ADDRESS_FIELD,
            Self::AMOUNT_FIELD,
        ];
    }
    impl BalanceChange {
        pub fn path_builder() -> BalanceChangeFieldPathBuilder {
            BalanceChangeFieldPathBuilder::new()
        }
    }
    pub struct BalanceChangeFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl BalanceChangeFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn address(mut self) -> String {
            self.path.push(BalanceChange::ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn amount(mut self) -> String {
            self.path.push(BalanceChange::AMOUNT_FIELD.name);
            self.finish()
        }
    }
    impl TransactionEffects {
        pub const STATUS_FIELD: &'static MessageField = &MessageField {
            name: "status",
            json_name: "status",
            number: 1i32,
            message_fields: Some(ExecutionStatus::FIELDS),
        };
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 2i32,
            message_fields: None,
        };
        pub const FEE_FIELD: &'static MessageField = &MessageField {
            name: "fee",
            json_name: "fee",
            number: 3i32,
            message_fields: Some(TransactionFee::FIELDS),
        };
        pub const TRANSACTION_DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "transaction_digest",
            json_name: "transactionDigest",
            number: 4i32,
            message_fields: None,
        };
        pub const DEPENDENCIES_FIELD: &'static MessageField = &MessageField {
            name: "dependencies",
            json_name: "dependencies",
            number: 5i32,
            message_fields: None,
        };
        pub const LAMPORT_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "lamport_version",
            json_name: "lamportVersion",
            number: 6i32,
            message_fields: None,
        };
        pub const CHANGED_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "changed_objects",
            json_name: "changedObjects",
            number: 7i32,
            message_fields: Some(ChangedObject::FIELDS),
        };
        pub const UNCHANGED_SHARED_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "unchanged_shared_objects",
            json_name: "unchangedSharedObjects",
            number: 8i32,
            message_fields: Some(UnchangedSharedObject::FIELDS),
        };
    }
    impl MessageFields for TransactionEffects {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::STATUS_FIELD,
            Self::EPOCH_FIELD,
            Self::FEE_FIELD,
            Self::TRANSACTION_DIGEST_FIELD,
            Self::DEPENDENCIES_FIELD,
            Self::LAMPORT_VERSION_FIELD,
            Self::CHANGED_OBJECTS_FIELD,
            Self::UNCHANGED_SHARED_OBJECTS_FIELD,
        ];
    }
    impl TransactionEffects {
        pub fn path_builder() -> TransactionEffectsFieldPathBuilder {
            TransactionEffectsFieldPathBuilder::new()
        }
    }
    pub struct TransactionEffectsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TransactionEffectsFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn status(mut self) -> ExecutionStatusFieldPathBuilder {
            self.path.push(TransactionEffects::STATUS_FIELD.name);
            ExecutionStatusFieldPathBuilder::new_with_base(self.path)
        }
        pub fn epoch(mut self) -> String {
            self.path.push(TransactionEffects::EPOCH_FIELD.name);
            self.finish()
        }
        pub fn fee(mut self) -> TransactionFeeFieldPathBuilder {
            self.path.push(TransactionEffects::FEE_FIELD.name);
            TransactionFeeFieldPathBuilder::new_with_base(self.path)
        }
        pub fn transaction_digest(mut self) -> String {
            self.path.push(TransactionEffects::TRANSACTION_DIGEST_FIELD.name);
            self.finish()
        }
        pub fn dependencies(mut self) -> String {
            self.path.push(TransactionEffects::DEPENDENCIES_FIELD.name);
            self.finish()
        }
        pub fn lamport_version(mut self) -> String {
            self.path.push(TransactionEffects::LAMPORT_VERSION_FIELD.name);
            self.finish()
        }
        pub fn changed_objects(mut self) -> ChangedObjectFieldPathBuilder {
            self.path.push(TransactionEffects::CHANGED_OBJECTS_FIELD.name);
            ChangedObjectFieldPathBuilder::new_with_base(self.path)
        }
        pub fn unchanged_shared_objects(
            mut self,
        ) -> UnchangedSharedObjectFieldPathBuilder {
            self.path.push(TransactionEffects::UNCHANGED_SHARED_OBJECTS_FIELD.name);
            UnchangedSharedObjectFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ChangedObject {
        pub const OBJECT_ID_FIELD: &'static MessageField = &MessageField {
            name: "object_id",
            json_name: "objectId",
            number: 1i32,
            message_fields: None,
        };
        pub const INPUT_STATE_FIELD: &'static MessageField = &MessageField {
            name: "input_state",
            json_name: "inputState",
            number: 2i32,
            message_fields: None,
        };
        pub const INPUT_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "input_version",
            json_name: "inputVersion",
            number: 3i32,
            message_fields: None,
        };
        pub const INPUT_DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "input_digest",
            json_name: "inputDigest",
            number: 4i32,
            message_fields: None,
        };
        pub const INPUT_OWNER_FIELD: &'static MessageField = &MessageField {
            name: "input_owner",
            json_name: "inputOwner",
            number: 5i32,
            message_fields: Some(Owner::FIELDS),
        };
        pub const OUTPUT_STATE_FIELD: &'static MessageField = &MessageField {
            name: "output_state",
            json_name: "outputState",
            number: 6i32,
            message_fields: None,
        };
        pub const OUTPUT_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "output_version",
            json_name: "outputVersion",
            number: 7i32,
            message_fields: None,
        };
        pub const OUTPUT_DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "output_digest",
            json_name: "outputDigest",
            number: 8i32,
            message_fields: None,
        };
        pub const OUTPUT_OWNER_FIELD: &'static MessageField = &MessageField {
            name: "output_owner",
            json_name: "outputOwner",
            number: 9i32,
            message_fields: Some(Owner::FIELDS),
        };
        pub const ID_OPERATION_FIELD: &'static MessageField = &MessageField {
            name: "id_operation",
            json_name: "idOperation",
            number: 10i32,
            message_fields: None,
        };
        pub const OBJECT_TYPE_FIELD: &'static MessageField = &MessageField {
            name: "object_type",
            json_name: "objectType",
            number: 11i32,
            message_fields: None,
        };
    }
    impl MessageFields for ChangedObject {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECT_ID_FIELD,
            Self::INPUT_STATE_FIELD,
            Self::INPUT_VERSION_FIELD,
            Self::INPUT_DIGEST_FIELD,
            Self::INPUT_OWNER_FIELD,
            Self::OUTPUT_STATE_FIELD,
            Self::OUTPUT_VERSION_FIELD,
            Self::OUTPUT_DIGEST_FIELD,
            Self::OUTPUT_OWNER_FIELD,
            Self::ID_OPERATION_FIELD,
            Self::OBJECT_TYPE_FIELD,
        ];
    }
    impl ChangedObject {
        pub fn path_builder() -> ChangedObjectFieldPathBuilder {
            ChangedObjectFieldPathBuilder::new()
        }
    }
    pub struct ChangedObjectFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ChangedObjectFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn object_id(mut self) -> String {
            self.path.push(ChangedObject::OBJECT_ID_FIELD.name);
            self.finish()
        }
        pub fn input_state(mut self) -> String {
            self.path.push(ChangedObject::INPUT_STATE_FIELD.name);
            self.finish()
        }
        pub fn input_version(mut self) -> String {
            self.path.push(ChangedObject::INPUT_VERSION_FIELD.name);
            self.finish()
        }
        pub fn input_digest(mut self) -> String {
            self.path.push(ChangedObject::INPUT_DIGEST_FIELD.name);
            self.finish()
        }
        pub fn input_owner(mut self) -> OwnerFieldPathBuilder {
            self.path.push(ChangedObject::INPUT_OWNER_FIELD.name);
            OwnerFieldPathBuilder::new_with_base(self.path)
        }
        pub fn output_state(mut self) -> String {
            self.path.push(ChangedObject::OUTPUT_STATE_FIELD.name);
            self.finish()
        }
        pub fn output_version(mut self) -> String {
            self.path.push(ChangedObject::OUTPUT_VERSION_FIELD.name);
            self.finish()
        }
        pub fn output_digest(mut self) -> String {
            self.path.push(ChangedObject::OUTPUT_DIGEST_FIELD.name);
            self.finish()
        }
        pub fn output_owner(mut self) -> OwnerFieldPathBuilder {
            self.path.push(ChangedObject::OUTPUT_OWNER_FIELD.name);
            OwnerFieldPathBuilder::new_with_base(self.path)
        }
        pub fn id_operation(mut self) -> String {
            self.path.push(ChangedObject::ID_OPERATION_FIELD.name);
            self.finish()
        }
        pub fn object_type(mut self) -> String {
            self.path.push(ChangedObject::OBJECT_TYPE_FIELD.name);
            self.finish()
        }
    }
    impl UnchangedSharedObject {
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 1i32,
            message_fields: None,
        };
        pub const OBJECT_ID_FIELD: &'static MessageField = &MessageField {
            name: "object_id",
            json_name: "objectId",
            number: 2i32,
            message_fields: None,
        };
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 3i32,
            message_fields: None,
        };
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 4i32,
            message_fields: None,
        };
        pub const OBJECT_TYPE_FIELD: &'static MessageField = &MessageField {
            name: "object_type",
            json_name: "objectType",
            number: 5i32,
            message_fields: None,
        };
    }
    impl MessageFields for UnchangedSharedObject {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::KIND_FIELD,
            Self::OBJECT_ID_FIELD,
            Self::VERSION_FIELD,
            Self::DIGEST_FIELD,
            Self::OBJECT_TYPE_FIELD,
        ];
    }
    impl UnchangedSharedObject {
        pub fn path_builder() -> UnchangedSharedObjectFieldPathBuilder {
            UnchangedSharedObjectFieldPathBuilder::new()
        }
    }
    pub struct UnchangedSharedObjectFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl UnchangedSharedObjectFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn kind(mut self) -> String {
            self.path.push(UnchangedSharedObject::KIND_FIELD.name);
            self.finish()
        }
        pub fn object_id(mut self) -> String {
            self.path.push(UnchangedSharedObject::OBJECT_ID_FIELD.name);
            self.finish()
        }
        pub fn version(mut self) -> String {
            self.path.push(UnchangedSharedObject::VERSION_FIELD.name);
            self.finish()
        }
        pub fn digest(mut self) -> String {
            self.path.push(UnchangedSharedObject::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn object_type(mut self) -> String {
            self.path.push(UnchangedSharedObject::OBJECT_TYPE_FIELD.name);
            self.finish()
        }
    }
    impl Epoch {
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 1i32,
            message_fields: None,
        };
        pub const COMMITTEE_FIELD: &'static MessageField = &MessageField {
            name: "committee",
            json_name: "committee",
            number: 2i32,
            message_fields: Some(ValidatorCommittee::FIELDS),
        };
        pub const SYSTEM_STATE_FIELD: &'static MessageField = &MessageField {
            name: "system_state",
            json_name: "systemState",
            number: 3i32,
            message_fields: Some(SystemState::FIELDS),
        };
        pub const START_FIELD: &'static MessageField = &MessageField {
            name: "start",
            json_name: "start",
            number: 6i32,
            message_fields: None,
        };
        pub const END_FIELD: &'static MessageField = &MessageField {
            name: "end",
            json_name: "end",
            number: 7i32,
            message_fields: None,
        };
    }
    impl MessageFields for Epoch {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_FIELD,
            Self::COMMITTEE_FIELD,
            Self::SYSTEM_STATE_FIELD,
            Self::START_FIELD,
            Self::END_FIELD,
        ];
    }
    impl Epoch {
        pub fn path_builder() -> EpochFieldPathBuilder {
            EpochFieldPathBuilder::new()
        }
    }
    pub struct EpochFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl EpochFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn epoch(mut self) -> String {
            self.path.push(Epoch::EPOCH_FIELD.name);
            self.finish()
        }
        pub fn committee(mut self) -> ValidatorCommitteeFieldPathBuilder {
            self.path.push(Epoch::COMMITTEE_FIELD.name);
            ValidatorCommitteeFieldPathBuilder::new_with_base(self.path)
        }
        pub fn system_state(mut self) -> SystemStateFieldPathBuilder {
            self.path.push(Epoch::SYSTEM_STATE_FIELD.name);
            SystemStateFieldPathBuilder::new_with_base(self.path)
        }
        pub fn start(mut self) -> String {
            self.path.push(Epoch::START_FIELD.name);
            self.finish()
        }
        pub fn end(mut self) -> String {
            self.path.push(Epoch::END_FIELD.name);
            self.finish()
        }
    }
    impl ExecutedTransaction {
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 1i32,
            message_fields: None,
        };
        pub const TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "transaction",
            json_name: "transaction",
            number: 2i32,
            message_fields: Some(Transaction::FIELDS),
        };
        pub const SIGNATURES_FIELD: &'static MessageField = &MessageField {
            name: "signatures",
            json_name: "signatures",
            number: 3i32,
            message_fields: Some(UserSignature::FIELDS),
        };
        pub const EFFECTS_FIELD: &'static MessageField = &MessageField {
            name: "effects",
            json_name: "effects",
            number: 4i32,
            message_fields: Some(TransactionEffects::FIELDS),
        };
        pub const COMMIT_FIELD: &'static MessageField = &MessageField {
            name: "commit",
            json_name: "commit",
            number: 5i32,
            message_fields: None,
        };
        pub const TIMESTAMP_FIELD: &'static MessageField = &MessageField {
            name: "timestamp",
            json_name: "timestamp",
            number: 6i32,
            message_fields: None,
        };
        pub const BALANCE_CHANGES_FIELD: &'static MessageField = &MessageField {
            name: "balance_changes",
            json_name: "balanceChanges",
            number: 7i32,
            message_fields: Some(BalanceChange::FIELDS),
        };
        pub const INPUT_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "input_objects",
            json_name: "inputObjects",
            number: 8i32,
            message_fields: Some(Object::FIELDS),
        };
        pub const OUTPUT_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "output_objects",
            json_name: "outputObjects",
            number: 9i32,
            message_fields: Some(Object::FIELDS),
        };
        pub const SHARD_FIELD: &'static MessageField = &MessageField {
            name: "shard",
            json_name: "shard",
            number: 10i32,
            message_fields: Some(Shard::FIELDS),
        };
    }
    impl MessageFields for ExecutedTransaction {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::DIGEST_FIELD,
            Self::TRANSACTION_FIELD,
            Self::SIGNATURES_FIELD,
            Self::EFFECTS_FIELD,
            Self::COMMIT_FIELD,
            Self::TIMESTAMP_FIELD,
            Self::BALANCE_CHANGES_FIELD,
            Self::INPUT_OBJECTS_FIELD,
            Self::OUTPUT_OBJECTS_FIELD,
            Self::SHARD_FIELD,
        ];
    }
    impl ExecutedTransaction {
        pub fn path_builder() -> ExecutedTransactionFieldPathBuilder {
            ExecutedTransactionFieldPathBuilder::new()
        }
    }
    pub struct ExecutedTransactionFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ExecutedTransactionFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn digest(mut self) -> String {
            self.path.push(ExecutedTransaction::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn transaction(mut self) -> TransactionFieldPathBuilder {
            self.path.push(ExecutedTransaction::TRANSACTION_FIELD.name);
            TransactionFieldPathBuilder::new_with_base(self.path)
        }
        pub fn signatures(mut self) -> UserSignatureFieldPathBuilder {
            self.path.push(ExecutedTransaction::SIGNATURES_FIELD.name);
            UserSignatureFieldPathBuilder::new_with_base(self.path)
        }
        pub fn effects(mut self) -> TransactionEffectsFieldPathBuilder {
            self.path.push(ExecutedTransaction::EFFECTS_FIELD.name);
            TransactionEffectsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn commit(mut self) -> String {
            self.path.push(ExecutedTransaction::COMMIT_FIELD.name);
            self.finish()
        }
        pub fn timestamp(mut self) -> String {
            self.path.push(ExecutedTransaction::TIMESTAMP_FIELD.name);
            self.finish()
        }
        pub fn balance_changes(mut self) -> BalanceChangeFieldPathBuilder {
            self.path.push(ExecutedTransaction::BALANCE_CHANGES_FIELD.name);
            BalanceChangeFieldPathBuilder::new_with_base(self.path)
        }
        pub fn input_objects(mut self) -> ObjectFieldPathBuilder {
            self.path.push(ExecutedTransaction::INPUT_OBJECTS_FIELD.name);
            ObjectFieldPathBuilder::new_with_base(self.path)
        }
        pub fn output_objects(mut self) -> ObjectFieldPathBuilder {
            self.path.push(ExecutedTransaction::OUTPUT_OBJECTS_FIELD.name);
            ObjectFieldPathBuilder::new_with_base(self.path)
        }
        pub fn shard(mut self) -> ShardFieldPathBuilder {
            self.path.push(ExecutedTransaction::SHARD_FIELD.name);
            ShardFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ExecutionStatus {
        pub const SUCCESS_FIELD: &'static MessageField = &MessageField {
            name: "success",
            json_name: "success",
            number: 1i32,
            message_fields: None,
        };
        pub const ERROR_FIELD: &'static MessageField = &MessageField {
            name: "error",
            json_name: "error",
            number: 2i32,
            message_fields: Some(ExecutionError::FIELDS),
        };
    }
    impl MessageFields for ExecutionStatus {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SUCCESS_FIELD,
            Self::ERROR_FIELD,
        ];
    }
    impl ExecutionStatus {
        pub fn path_builder() -> ExecutionStatusFieldPathBuilder {
            ExecutionStatusFieldPathBuilder::new()
        }
    }
    pub struct ExecutionStatusFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ExecutionStatusFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn success(mut self) -> String {
            self.path.push(ExecutionStatus::SUCCESS_FIELD.name);
            self.finish()
        }
        pub fn error(mut self) -> ExecutionErrorFieldPathBuilder {
            self.path.push(ExecutionStatus::ERROR_FIELD.name);
            ExecutionErrorFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ExecutionError {
        pub const DESCRIPTION_FIELD: &'static MessageField = &MessageField {
            name: "description",
            json_name: "description",
            number: 1i32,
            message_fields: None,
        };
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 2i32,
            message_fields: None,
        };
        pub const OBJECT_ID_FIELD: &'static MessageField = &MessageField {
            name: "object_id",
            json_name: "objectId",
            number: 3i32,
            message_fields: None,
        };
        pub const OTHER_ERROR_FIELD: &'static MessageField = &MessageField {
            name: "other_error",
            json_name: "otherError",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for ExecutionError {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::DESCRIPTION_FIELD,
            Self::KIND_FIELD,
            Self::OBJECT_ID_FIELD,
            Self::OTHER_ERROR_FIELD,
        ];
    }
    impl ExecutionError {
        pub fn path_builder() -> ExecutionErrorFieldPathBuilder {
            ExecutionErrorFieldPathBuilder::new()
        }
    }
    pub struct ExecutionErrorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ExecutionErrorFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn description(mut self) -> String {
            self.path.push(ExecutionError::DESCRIPTION_FIELD.name);
            self.finish()
        }
        pub fn kind(mut self) -> String {
            self.path.push(ExecutionError::KIND_FIELD.name);
            self.finish()
        }
        pub fn object_id(mut self) -> String {
            self.path.push(ExecutionError::OBJECT_ID_FIELD.name);
            self.finish()
        }
        pub fn other_error(mut self) -> String {
            self.path.push(ExecutionError::OTHER_ERROR_FIELD.name);
            self.finish()
        }
    }
    impl GetObjectRequest {
        pub const OBJECT_ID_FIELD: &'static MessageField = &MessageField {
            name: "object_id",
            json_name: "objectId",
            number: 1i32,
            message_fields: None,
        };
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 2i32,
            message_fields: None,
        };
        pub const READ_MASK_FIELD: &'static MessageField = &MessageField {
            name: "read_mask",
            json_name: "readMask",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for GetObjectRequest {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECT_ID_FIELD,
            Self::VERSION_FIELD,
            Self::READ_MASK_FIELD,
        ];
    }
    impl GetObjectRequest {
        pub fn path_builder() -> GetObjectRequestFieldPathBuilder {
            GetObjectRequestFieldPathBuilder::new()
        }
    }
    pub struct GetObjectRequestFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GetObjectRequestFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn object_id(mut self) -> String {
            self.path.push(GetObjectRequest::OBJECT_ID_FIELD.name);
            self.finish()
        }
        pub fn version(mut self) -> String {
            self.path.push(GetObjectRequest::VERSION_FIELD.name);
            self.finish()
        }
        pub fn read_mask(mut self) -> String {
            self.path.push(GetObjectRequest::READ_MASK_FIELD.name);
            self.finish()
        }
    }
    impl GetObjectResponse {
        pub const OBJECT_FIELD: &'static MessageField = &MessageField {
            name: "object",
            json_name: "object",
            number: 1i32,
            message_fields: Some(Object::FIELDS),
        };
    }
    impl MessageFields for GetObjectResponse {
        const FIELDS: &'static [&'static MessageField] = &[Self::OBJECT_FIELD];
    }
    impl GetObjectResponse {
        pub fn path_builder() -> GetObjectResponseFieldPathBuilder {
            GetObjectResponseFieldPathBuilder::new()
        }
    }
    pub struct GetObjectResponseFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GetObjectResponseFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn object(mut self) -> ObjectFieldPathBuilder {
            self.path.push(GetObjectResponse::OBJECT_FIELD.name);
            ObjectFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl BatchGetObjectsRequest {
        pub const REQUESTS_FIELD: &'static MessageField = &MessageField {
            name: "requests",
            json_name: "requests",
            number: 1i32,
            message_fields: Some(GetObjectRequest::FIELDS),
        };
        pub const READ_MASK_FIELD: &'static MessageField = &MessageField {
            name: "read_mask",
            json_name: "readMask",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for BatchGetObjectsRequest {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::REQUESTS_FIELD,
            Self::READ_MASK_FIELD,
        ];
    }
    impl BatchGetObjectsRequest {
        pub fn path_builder() -> BatchGetObjectsRequestFieldPathBuilder {
            BatchGetObjectsRequestFieldPathBuilder::new()
        }
    }
    pub struct BatchGetObjectsRequestFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl BatchGetObjectsRequestFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn requests(mut self) -> GetObjectRequestFieldPathBuilder {
            self.path.push(BatchGetObjectsRequest::REQUESTS_FIELD.name);
            GetObjectRequestFieldPathBuilder::new_with_base(self.path)
        }
        pub fn read_mask(mut self) -> String {
            self.path.push(BatchGetObjectsRequest::READ_MASK_FIELD.name);
            self.finish()
        }
    }
    impl BatchGetObjectsResponse {
        pub const OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "objects",
            json_name: "objects",
            number: 1i32,
            message_fields: Some(GetObjectResult::FIELDS),
        };
    }
    impl MessageFields for BatchGetObjectsResponse {
        const FIELDS: &'static [&'static MessageField] = &[Self::OBJECTS_FIELD];
    }
    impl BatchGetObjectsResponse {
        pub fn path_builder() -> BatchGetObjectsResponseFieldPathBuilder {
            BatchGetObjectsResponseFieldPathBuilder::new()
        }
    }
    pub struct BatchGetObjectsResponseFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl BatchGetObjectsResponseFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn objects(mut self) -> GetObjectResultFieldPathBuilder {
            self.path.push(BatchGetObjectsResponse::OBJECTS_FIELD.name);
            GetObjectResultFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl GetObjectResult {
        pub const OBJECT_FIELD: &'static MessageField = &MessageField {
            name: "object",
            json_name: "object",
            number: 1i32,
            message_fields: Some(Object::FIELDS),
        };
        pub const ERROR_FIELD: &'static MessageField = &MessageField {
            name: "error",
            json_name: "error",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for GetObjectResult {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECT_FIELD,
            Self::ERROR_FIELD,
        ];
    }
    impl GetObjectResult {
        pub fn path_builder() -> GetObjectResultFieldPathBuilder {
            GetObjectResultFieldPathBuilder::new()
        }
    }
    pub struct GetObjectResultFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GetObjectResultFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn object(mut self) -> ObjectFieldPathBuilder {
            self.path.push(GetObjectResult::OBJECT_FIELD.name);
            ObjectFieldPathBuilder::new_with_base(self.path)
        }
        pub fn error(mut self) -> String {
            self.path.push(GetObjectResult::ERROR_FIELD.name);
            self.finish()
        }
    }
    impl GetTransactionRequest {
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 1i32,
            message_fields: None,
        };
        pub const READ_MASK_FIELD: &'static MessageField = &MessageField {
            name: "read_mask",
            json_name: "readMask",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for GetTransactionRequest {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::DIGEST_FIELD,
            Self::READ_MASK_FIELD,
        ];
    }
    impl GetTransactionRequest {
        pub fn path_builder() -> GetTransactionRequestFieldPathBuilder {
            GetTransactionRequestFieldPathBuilder::new()
        }
    }
    pub struct GetTransactionRequestFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GetTransactionRequestFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn digest(mut self) -> String {
            self.path.push(GetTransactionRequest::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn read_mask(mut self) -> String {
            self.path.push(GetTransactionRequest::READ_MASK_FIELD.name);
            self.finish()
        }
    }
    impl GetTransactionResponse {
        pub const TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "transaction",
            json_name: "transaction",
            number: 1i32,
            message_fields: Some(ExecutedTransaction::FIELDS),
        };
    }
    impl MessageFields for GetTransactionResponse {
        const FIELDS: &'static [&'static MessageField] = &[Self::TRANSACTION_FIELD];
    }
    impl GetTransactionResponse {
        pub fn path_builder() -> GetTransactionResponseFieldPathBuilder {
            GetTransactionResponseFieldPathBuilder::new()
        }
    }
    pub struct GetTransactionResponseFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GetTransactionResponseFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn transaction(mut self) -> ExecutedTransactionFieldPathBuilder {
            self.path.push(GetTransactionResponse::TRANSACTION_FIELD.name);
            ExecutedTransactionFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl BatchGetTransactionsRequest {
        pub const DIGESTS_FIELD: &'static MessageField = &MessageField {
            name: "digests",
            json_name: "digests",
            number: 1i32,
            message_fields: None,
        };
        pub const READ_MASK_FIELD: &'static MessageField = &MessageField {
            name: "read_mask",
            json_name: "readMask",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for BatchGetTransactionsRequest {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::DIGESTS_FIELD,
            Self::READ_MASK_FIELD,
        ];
    }
    impl BatchGetTransactionsRequest {
        pub fn path_builder() -> BatchGetTransactionsRequestFieldPathBuilder {
            BatchGetTransactionsRequestFieldPathBuilder::new()
        }
    }
    pub struct BatchGetTransactionsRequestFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl BatchGetTransactionsRequestFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn digests(mut self) -> String {
            self.path.push(BatchGetTransactionsRequest::DIGESTS_FIELD.name);
            self.finish()
        }
        pub fn read_mask(mut self) -> String {
            self.path.push(BatchGetTransactionsRequest::READ_MASK_FIELD.name);
            self.finish()
        }
    }
    impl BatchGetTransactionsResponse {
        pub const TRANSACTIONS_FIELD: &'static MessageField = &MessageField {
            name: "transactions",
            json_name: "transactions",
            number: 1i32,
            message_fields: Some(GetTransactionResult::FIELDS),
        };
    }
    impl MessageFields for BatchGetTransactionsResponse {
        const FIELDS: &'static [&'static MessageField] = &[Self::TRANSACTIONS_FIELD];
    }
    impl BatchGetTransactionsResponse {
        pub fn path_builder() -> BatchGetTransactionsResponseFieldPathBuilder {
            BatchGetTransactionsResponseFieldPathBuilder::new()
        }
    }
    pub struct BatchGetTransactionsResponseFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl BatchGetTransactionsResponseFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn transactions(mut self) -> GetTransactionResultFieldPathBuilder {
            self.path.push(BatchGetTransactionsResponse::TRANSACTIONS_FIELD.name);
            GetTransactionResultFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl GetTransactionResult {
        pub const TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "transaction",
            json_name: "transaction",
            number: 1i32,
            message_fields: Some(ExecutedTransaction::FIELDS),
        };
        pub const ERROR_FIELD: &'static MessageField = &MessageField {
            name: "error",
            json_name: "error",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for GetTransactionResult {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::TRANSACTION_FIELD,
            Self::ERROR_FIELD,
        ];
    }
    impl GetTransactionResult {
        pub fn path_builder() -> GetTransactionResultFieldPathBuilder {
            GetTransactionResultFieldPathBuilder::new()
        }
    }
    pub struct GetTransactionResultFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GetTransactionResultFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn transaction(mut self) -> ExecutedTransactionFieldPathBuilder {
            self.path.push(GetTransactionResult::TRANSACTION_FIELD.name);
            ExecutedTransactionFieldPathBuilder::new_with_base(self.path)
        }
        pub fn error(mut self) -> String {
            self.path.push(GetTransactionResult::ERROR_FIELD.name);
            self.finish()
        }
    }
    impl GetEpochRequest {
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 1i32,
            message_fields: None,
        };
        pub const READ_MASK_FIELD: &'static MessageField = &MessageField {
            name: "read_mask",
            json_name: "readMask",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for GetEpochRequest {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_FIELD,
            Self::READ_MASK_FIELD,
        ];
    }
    impl GetEpochRequest {
        pub fn path_builder() -> GetEpochRequestFieldPathBuilder {
            GetEpochRequestFieldPathBuilder::new()
        }
    }
    pub struct GetEpochRequestFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GetEpochRequestFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn epoch(mut self) -> String {
            self.path.push(GetEpochRequest::EPOCH_FIELD.name);
            self.finish()
        }
        pub fn read_mask(mut self) -> String {
            self.path.push(GetEpochRequest::READ_MASK_FIELD.name);
            self.finish()
        }
    }
    impl GetEpochResponse {
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 1i32,
            message_fields: Some(Epoch::FIELDS),
        };
    }
    impl MessageFields for GetEpochResponse {
        const FIELDS: &'static [&'static MessageField] = &[Self::EPOCH_FIELD];
    }
    impl GetEpochResponse {
        pub fn path_builder() -> GetEpochResponseFieldPathBuilder {
            GetEpochResponseFieldPathBuilder::new()
        }
    }
    pub struct GetEpochResponseFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GetEpochResponseFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn epoch(mut self) -> EpochFieldPathBuilder {
            self.path.push(GetEpochResponse::EPOCH_FIELD.name);
            EpochFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl GetBalanceRequest {
        pub const OWNER_FIELD: &'static MessageField = &MessageField {
            name: "owner",
            json_name: "owner",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for GetBalanceRequest {
        const FIELDS: &'static [&'static MessageField] = &[Self::OWNER_FIELD];
    }
    impl GetBalanceRequest {
        pub fn path_builder() -> GetBalanceRequestFieldPathBuilder {
            GetBalanceRequestFieldPathBuilder::new()
        }
    }
    pub struct GetBalanceRequestFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GetBalanceRequestFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn owner(mut self) -> String {
            self.path.push(GetBalanceRequest::OWNER_FIELD.name);
            self.finish()
        }
    }
    impl GetBalanceResponse {
        pub const BALANCE_FIELD: &'static MessageField = &MessageField {
            name: "balance",
            json_name: "balance",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for GetBalanceResponse {
        const FIELDS: &'static [&'static MessageField] = &[Self::BALANCE_FIELD];
    }
    impl GetBalanceResponse {
        pub fn path_builder() -> GetBalanceResponseFieldPathBuilder {
            GetBalanceResponseFieldPathBuilder::new()
        }
    }
    pub struct GetBalanceResponseFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GetBalanceResponseFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn balance(mut self) -> String {
            self.path.push(GetBalanceResponse::BALANCE_FIELD.name);
            self.finish()
        }
    }
    impl SimulateTransactionRequest {
        pub const TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "transaction",
            json_name: "transaction",
            number: 1i32,
            message_fields: Some(Transaction::FIELDS),
        };
        pub const READ_MASK_FIELD: &'static MessageField = &MessageField {
            name: "read_mask",
            json_name: "readMask",
            number: 2i32,
            message_fields: None,
        };
        pub const CHECKS_FIELD: &'static MessageField = &MessageField {
            name: "checks",
            json_name: "checks",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for SimulateTransactionRequest {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::TRANSACTION_FIELD,
            Self::READ_MASK_FIELD,
            Self::CHECKS_FIELD,
        ];
    }
    impl SimulateTransactionRequest {
        pub fn path_builder() -> SimulateTransactionRequestFieldPathBuilder {
            SimulateTransactionRequestFieldPathBuilder::new()
        }
    }
    pub struct SimulateTransactionRequestFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SimulateTransactionRequestFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn transaction(mut self) -> TransactionFieldPathBuilder {
            self.path.push(SimulateTransactionRequest::TRANSACTION_FIELD.name);
            TransactionFieldPathBuilder::new_with_base(self.path)
        }
        pub fn read_mask(mut self) -> String {
            self.path.push(SimulateTransactionRequest::READ_MASK_FIELD.name);
            self.finish()
        }
        pub fn checks(mut self) -> String {
            self.path.push(SimulateTransactionRequest::CHECKS_FIELD.name);
            self.finish()
        }
    }
    impl SimulateTransactionResponse {
        pub const TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "transaction",
            json_name: "transaction",
            number: 1i32,
            message_fields: Some(ExecutedTransaction::FIELDS),
        };
    }
    impl MessageFields for SimulateTransactionResponse {
        const FIELDS: &'static [&'static MessageField] = &[Self::TRANSACTION_FIELD];
    }
    impl SimulateTransactionResponse {
        pub fn path_builder() -> SimulateTransactionResponseFieldPathBuilder {
            SimulateTransactionResponseFieldPathBuilder::new()
        }
    }
    pub struct SimulateTransactionResponseFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SimulateTransactionResponseFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn transaction(mut self) -> ExecutedTransactionFieldPathBuilder {
            self.path.push(SimulateTransactionResponse::TRANSACTION_FIELD.name);
            ExecutedTransactionFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ListOwnedObjectsRequest {
        pub const OWNER_FIELD: &'static MessageField = &MessageField {
            name: "owner",
            json_name: "owner",
            number: 1i32,
            message_fields: None,
        };
        pub const PAGE_SIZE_FIELD: &'static MessageField = &MessageField {
            name: "page_size",
            json_name: "pageSize",
            number: 2i32,
            message_fields: None,
        };
        pub const PAGE_TOKEN_FIELD: &'static MessageField = &MessageField {
            name: "page_token",
            json_name: "pageToken",
            number: 3i32,
            message_fields: None,
        };
        pub const READ_MASK_FIELD: &'static MessageField = &MessageField {
            name: "read_mask",
            json_name: "readMask",
            number: 4i32,
            message_fields: None,
        };
        pub const OBJECT_TYPE_FIELD: &'static MessageField = &MessageField {
            name: "object_type",
            json_name: "objectType",
            number: 5i32,
            message_fields: None,
        };
    }
    impl MessageFields for ListOwnedObjectsRequest {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OWNER_FIELD,
            Self::PAGE_SIZE_FIELD,
            Self::PAGE_TOKEN_FIELD,
            Self::READ_MASK_FIELD,
            Self::OBJECT_TYPE_FIELD,
        ];
    }
    impl ListOwnedObjectsRequest {
        pub fn path_builder() -> ListOwnedObjectsRequestFieldPathBuilder {
            ListOwnedObjectsRequestFieldPathBuilder::new()
        }
    }
    pub struct ListOwnedObjectsRequestFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ListOwnedObjectsRequestFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn owner(mut self) -> String {
            self.path.push(ListOwnedObjectsRequest::OWNER_FIELD.name);
            self.finish()
        }
        pub fn page_size(mut self) -> String {
            self.path.push(ListOwnedObjectsRequest::PAGE_SIZE_FIELD.name);
            self.finish()
        }
        pub fn page_token(mut self) -> String {
            self.path.push(ListOwnedObjectsRequest::PAGE_TOKEN_FIELD.name);
            self.finish()
        }
        pub fn read_mask(mut self) -> String {
            self.path.push(ListOwnedObjectsRequest::READ_MASK_FIELD.name);
            self.finish()
        }
        pub fn object_type(mut self) -> String {
            self.path.push(ListOwnedObjectsRequest::OBJECT_TYPE_FIELD.name);
            self.finish()
        }
    }
    impl ListOwnedObjectsResponse {
        pub const OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "objects",
            json_name: "objects",
            number: 1i32,
            message_fields: Some(Object::FIELDS),
        };
        pub const NEXT_PAGE_TOKEN_FIELD: &'static MessageField = &MessageField {
            name: "next_page_token",
            json_name: "nextPageToken",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for ListOwnedObjectsResponse {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECTS_FIELD,
            Self::NEXT_PAGE_TOKEN_FIELD,
        ];
    }
    impl ListOwnedObjectsResponse {
        pub fn path_builder() -> ListOwnedObjectsResponseFieldPathBuilder {
            ListOwnedObjectsResponseFieldPathBuilder::new()
        }
    }
    pub struct ListOwnedObjectsResponseFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ListOwnedObjectsResponseFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn objects(mut self) -> ObjectFieldPathBuilder {
            self.path.push(ListOwnedObjectsResponse::OBJECTS_FIELD.name);
            ObjectFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_page_token(mut self) -> String {
            self.path.push(ListOwnedObjectsResponse::NEXT_PAGE_TOKEN_FIELD.name);
            self.finish()
        }
    }
    impl Object {
        pub const OBJECT_ID_FIELD: &'static MessageField = &MessageField {
            name: "object_id",
            json_name: "objectId",
            number: 1i32,
            message_fields: None,
        };
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 2i32,
            message_fields: None,
        };
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 3i32,
            message_fields: None,
        };
        pub const OWNER_FIELD: &'static MessageField = &MessageField {
            name: "owner",
            json_name: "owner",
            number: 4i32,
            message_fields: Some(Owner::FIELDS),
        };
        pub const OBJECT_TYPE_FIELD: &'static MessageField = &MessageField {
            name: "object_type",
            json_name: "objectType",
            number: 5i32,
            message_fields: None,
        };
        pub const CONTENTS_FIELD: &'static MessageField = &MessageField {
            name: "contents",
            json_name: "contents",
            number: 6i32,
            message_fields: None,
        };
        pub const PREVIOUS_TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "previous_transaction",
            json_name: "previousTransaction",
            number: 7i32,
            message_fields: None,
        };
    }
    impl MessageFields for Object {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECT_ID_FIELD,
            Self::VERSION_FIELD,
            Self::DIGEST_FIELD,
            Self::OWNER_FIELD,
            Self::OBJECT_TYPE_FIELD,
            Self::CONTENTS_FIELD,
            Self::PREVIOUS_TRANSACTION_FIELD,
        ];
    }
    impl Object {
        pub fn path_builder() -> ObjectFieldPathBuilder {
            ObjectFieldPathBuilder::new()
        }
    }
    pub struct ObjectFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ObjectFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn object_id(mut self) -> String {
            self.path.push(Object::OBJECT_ID_FIELD.name);
            self.finish()
        }
        pub fn version(mut self) -> String {
            self.path.push(Object::VERSION_FIELD.name);
            self.finish()
        }
        pub fn digest(mut self) -> String {
            self.path.push(Object::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn owner(mut self) -> OwnerFieldPathBuilder {
            self.path.push(Object::OWNER_FIELD.name);
            OwnerFieldPathBuilder::new_with_base(self.path)
        }
        pub fn object_type(mut self) -> String {
            self.path.push(Object::OBJECT_TYPE_FIELD.name);
            self.finish()
        }
        pub fn contents(mut self) -> String {
            self.path.push(Object::CONTENTS_FIELD.name);
            self.finish()
        }
        pub fn previous_transaction(mut self) -> String {
            self.path.push(Object::PREVIOUS_TRANSACTION_FIELD.name);
            self.finish()
        }
    }
    impl ObjectReference {
        pub const OBJECT_ID_FIELD: &'static MessageField = &MessageField {
            name: "object_id",
            json_name: "objectId",
            number: 1i32,
            message_fields: None,
        };
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 2i32,
            message_fields: None,
        };
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for ObjectReference {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECT_ID_FIELD,
            Self::VERSION_FIELD,
            Self::DIGEST_FIELD,
        ];
    }
    impl ObjectReference {
        pub fn path_builder() -> ObjectReferenceFieldPathBuilder {
            ObjectReferenceFieldPathBuilder::new()
        }
    }
    pub struct ObjectReferenceFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ObjectReferenceFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn object_id(mut self) -> String {
            self.path.push(ObjectReference::OBJECT_ID_FIELD.name);
            self.finish()
        }
        pub fn version(mut self) -> String {
            self.path.push(ObjectReference::VERSION_FIELD.name);
            self.finish()
        }
        pub fn digest(mut self) -> String {
            self.path.push(ObjectReference::DIGEST_FIELD.name);
            self.finish()
        }
    }
    impl Owner {
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 1i32,
            message_fields: None,
        };
        pub const ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "address",
            json_name: "address",
            number: 2i32,
            message_fields: None,
        };
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for Owner {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::KIND_FIELD,
            Self::ADDRESS_FIELD,
            Self::VERSION_FIELD,
        ];
    }
    impl Owner {
        pub fn path_builder() -> OwnerFieldPathBuilder {
            OwnerFieldPathBuilder::new()
        }
    }
    pub struct OwnerFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl OwnerFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn kind(mut self) -> String {
            self.path.push(Owner::KIND_FIELD.name);
            self.finish()
        }
        pub fn address(mut self) -> String {
            self.path.push(Owner::ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn version(mut self) -> String {
            self.path.push(Owner::VERSION_FIELD.name);
            self.finish()
        }
    }
    impl Shard {
        pub const QUORUM_THRESHOLD_FIELD: &'static MessageField = &MessageField {
            name: "quorum_threshold",
            json_name: "quorumThreshold",
            number: 1i32,
            message_fields: None,
        };
        pub const ENCODERS_FIELD: &'static MessageField = &MessageField {
            name: "encoders",
            json_name: "encoders",
            number: 2i32,
            message_fields: None,
        };
        pub const SEED_FIELD: &'static MessageField = &MessageField {
            name: "seed",
            json_name: "seed",
            number: 3i32,
            message_fields: None,
        };
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for Shard {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::QUORUM_THRESHOLD_FIELD,
            Self::ENCODERS_FIELD,
            Self::SEED_FIELD,
            Self::EPOCH_FIELD,
        ];
    }
    impl Shard {
        pub fn path_builder() -> ShardFieldPathBuilder {
            ShardFieldPathBuilder::new()
        }
    }
    pub struct ShardFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ShardFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn quorum_threshold(mut self) -> String {
            self.path.push(Shard::QUORUM_THRESHOLD_FIELD.name);
            self.finish()
        }
        pub fn encoders(mut self) -> String {
            self.path.push(Shard::ENCODERS_FIELD.name);
            self.finish()
        }
        pub fn seed(mut self) -> String {
            self.path.push(Shard::SEED_FIELD.name);
            self.finish()
        }
        pub fn epoch(mut self) -> String {
            self.path.push(Shard::EPOCH_FIELD.name);
            self.finish()
        }
    }
    impl UserSignature {
        pub const SCHEME_FIELD: &'static MessageField = &MessageField {
            name: "scheme",
            json_name: "scheme",
            number: 1i32,
            message_fields: None,
        };
        pub const SIMPLE_FIELD: &'static MessageField = &MessageField {
            name: "simple",
            json_name: "simple",
            number: 2i32,
            message_fields: Some(SimpleSignature::FIELDS),
        };
    }
    impl MessageFields for UserSignature {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SCHEME_FIELD,
            Self::SIMPLE_FIELD,
        ];
    }
    impl UserSignature {
        pub fn path_builder() -> UserSignatureFieldPathBuilder {
            UserSignatureFieldPathBuilder::new()
        }
    }
    pub struct UserSignatureFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl UserSignatureFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn scheme(mut self) -> String {
            self.path.push(UserSignature::SCHEME_FIELD.name);
            self.finish()
        }
        pub fn simple(mut self) -> SimpleSignatureFieldPathBuilder {
            self.path.push(UserSignature::SIMPLE_FIELD.name);
            SimpleSignatureFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl SimpleSignature {
        pub const SCHEME_FIELD: &'static MessageField = &MessageField {
            name: "scheme",
            json_name: "scheme",
            number: 1i32,
            message_fields: None,
        };
        pub const SIGNATURE_FIELD: &'static MessageField = &MessageField {
            name: "signature",
            json_name: "signature",
            number: 2i32,
            message_fields: None,
        };
        pub const PUBLIC_KEY_FIELD: &'static MessageField = &MessageField {
            name: "public_key",
            json_name: "publicKey",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for SimpleSignature {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SCHEME_FIELD,
            Self::SIGNATURE_FIELD,
            Self::PUBLIC_KEY_FIELD,
        ];
    }
    impl SimpleSignature {
        pub fn path_builder() -> SimpleSignatureFieldPathBuilder {
            SimpleSignatureFieldPathBuilder::new()
        }
    }
    pub struct SimpleSignatureFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SimpleSignatureFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn scheme(mut self) -> String {
            self.path.push(SimpleSignature::SCHEME_FIELD.name);
            self.finish()
        }
        pub fn signature(mut self) -> String {
            self.path.push(SimpleSignature::SIGNATURE_FIELD.name);
            self.finish()
        }
        pub fn public_key(mut self) -> String {
            self.path.push(SimpleSignature::PUBLIC_KEY_FIELD.name);
            self.finish()
        }
    }
    impl ValidatorCommittee {
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 1i32,
            message_fields: None,
        };
        pub const MEMBERS_FIELD: &'static MessageField = &MessageField {
            name: "members",
            json_name: "members",
            number: 2i32,
            message_fields: Some(ValidatorCommitteeMember::FIELDS),
        };
    }
    impl MessageFields for ValidatorCommittee {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_FIELD,
            Self::MEMBERS_FIELD,
        ];
    }
    impl ValidatorCommittee {
        pub fn path_builder() -> ValidatorCommitteeFieldPathBuilder {
            ValidatorCommitteeFieldPathBuilder::new()
        }
    }
    pub struct ValidatorCommitteeFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ValidatorCommitteeFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn epoch(mut self) -> String {
            self.path.push(ValidatorCommittee::EPOCH_FIELD.name);
            self.finish()
        }
        pub fn members(mut self) -> ValidatorCommitteeMemberFieldPathBuilder {
            self.path.push(ValidatorCommittee::MEMBERS_FIELD.name);
            ValidatorCommitteeMemberFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ValidatorCommitteeMember {
        pub const AUTHORITY_KEY_FIELD: &'static MessageField = &MessageField {
            name: "authority_key",
            json_name: "authorityKey",
            number: 1i32,
            message_fields: None,
        };
        pub const WEIGHT_FIELD: &'static MessageField = &MessageField {
            name: "weight",
            json_name: "weight",
            number: 2i32,
            message_fields: None,
        };
        pub const NETWORK_METADATA_FIELD: &'static MessageField = &MessageField {
            name: "network_metadata",
            json_name: "networkMetadata",
            number: 3i32,
            message_fields: Some(ValidatorNetworkMetadata::FIELDS),
        };
    }
    impl MessageFields for ValidatorCommitteeMember {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::AUTHORITY_KEY_FIELD,
            Self::WEIGHT_FIELD,
            Self::NETWORK_METADATA_FIELD,
        ];
    }
    impl ValidatorCommitteeMember {
        pub fn path_builder() -> ValidatorCommitteeMemberFieldPathBuilder {
            ValidatorCommitteeMemberFieldPathBuilder::new()
        }
    }
    pub struct ValidatorCommitteeMemberFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ValidatorCommitteeMemberFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn authority_key(mut self) -> String {
            self.path.push(ValidatorCommitteeMember::AUTHORITY_KEY_FIELD.name);
            self.finish()
        }
        pub fn weight(mut self) -> String {
            self.path.push(ValidatorCommitteeMember::WEIGHT_FIELD.name);
            self.finish()
        }
        pub fn network_metadata(mut self) -> ValidatorNetworkMetadataFieldPathBuilder {
            self.path.push(ValidatorCommitteeMember::NETWORK_METADATA_FIELD.name);
            ValidatorNetworkMetadataFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ValidatorNetworkMetadata {
        pub const CONSENSUS_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "consensus_address",
            json_name: "consensusAddress",
            number: 1i32,
            message_fields: None,
        };
        pub const HOSTNAME_FIELD: &'static MessageField = &MessageField {
            name: "hostname",
            json_name: "hostname",
            number: 2i32,
            message_fields: None,
        };
        pub const PROTOCOL_KEY_FIELD: &'static MessageField = &MessageField {
            name: "protocol_key",
            json_name: "protocolKey",
            number: 3i32,
            message_fields: None,
        };
        pub const NETWORK_KEY_FIELD: &'static MessageField = &MessageField {
            name: "network_key",
            json_name: "networkKey",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for ValidatorNetworkMetadata {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::CONSENSUS_ADDRESS_FIELD,
            Self::HOSTNAME_FIELD,
            Self::PROTOCOL_KEY_FIELD,
            Self::NETWORK_KEY_FIELD,
        ];
    }
    impl ValidatorNetworkMetadata {
        pub fn path_builder() -> ValidatorNetworkMetadataFieldPathBuilder {
            ValidatorNetworkMetadataFieldPathBuilder::new()
        }
    }
    pub struct ValidatorNetworkMetadataFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ValidatorNetworkMetadataFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn consensus_address(mut self) -> String {
            self.path.push(ValidatorNetworkMetadata::CONSENSUS_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn hostname(mut self) -> String {
            self.path.push(ValidatorNetworkMetadata::HOSTNAME_FIELD.name);
            self.finish()
        }
        pub fn protocol_key(mut self) -> String {
            self.path.push(ValidatorNetworkMetadata::PROTOCOL_KEY_FIELD.name);
            self.finish()
        }
        pub fn network_key(mut self) -> String {
            self.path.push(ValidatorNetworkMetadata::NETWORK_KEY_FIELD.name);
            self.finish()
        }
    }
    impl ValidatorAggregatedSignature {
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 1i32,
            message_fields: None,
        };
        pub const SIGNATURE_FIELD: &'static MessageField = &MessageField {
            name: "signature",
            json_name: "signature",
            number: 2i32,
            message_fields: None,
        };
        pub const BITMAP_FIELD: &'static MessageField = &MessageField {
            name: "bitmap",
            json_name: "bitmap",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for ValidatorAggregatedSignature {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_FIELD,
            Self::SIGNATURE_FIELD,
            Self::BITMAP_FIELD,
        ];
    }
    impl ValidatorAggregatedSignature {
        pub fn path_builder() -> ValidatorAggregatedSignatureFieldPathBuilder {
            ValidatorAggregatedSignatureFieldPathBuilder::new()
        }
    }
    pub struct ValidatorAggregatedSignatureFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ValidatorAggregatedSignatureFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn epoch(mut self) -> String {
            self.path.push(ValidatorAggregatedSignature::EPOCH_FIELD.name);
            self.finish()
        }
        pub fn signature(mut self) -> String {
            self.path.push(ValidatorAggregatedSignature::SIGNATURE_FIELD.name);
            self.finish()
        }
        pub fn bitmap(mut self) -> String {
            self.path.push(ValidatorAggregatedSignature::BITMAP_FIELD.name);
            self.finish()
        }
    }
    impl SystemState {
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 1i32,
            message_fields: None,
        };
        pub const EPOCH_START_TIMESTAMP_MS_FIELD: &'static MessageField = &MessageField {
            name: "epoch_start_timestamp_ms",
            json_name: "epochStartTimestampMs",
            number: 2i32,
            message_fields: None,
        };
        pub const PARAMETERS_FIELD: &'static MessageField = &MessageField {
            name: "parameters",
            json_name: "parameters",
            number: 3i32,
            message_fields: Some(SystemParameters::FIELDS),
        };
        pub const VALIDATORS_FIELD: &'static MessageField = &MessageField {
            name: "validators",
            json_name: "validators",
            number: 4i32,
            message_fields: Some(ValidatorSet::FIELDS),
        };
        pub const ENCODERS_FIELD: &'static MessageField = &MessageField {
            name: "encoders",
            json_name: "encoders",
            number: 5i32,
            message_fields: Some(EncoderSet::FIELDS),
        };
        pub const VALIDATOR_REPORT_RECORDS_FIELD: &'static MessageField = &MessageField {
            name: "validator_report_records",
            json_name: "validatorReportRecords",
            number: 6i32,
            message_fields: None,
        };
        pub const ENCODER_REPORT_RECORDS_FIELD: &'static MessageField = &MessageField {
            name: "encoder_report_records",
            json_name: "encoderReportRecords",
            number: 7i32,
            message_fields: None,
        };
        pub const STAKE_SUBSIDY_FIELD: &'static MessageField = &MessageField {
            name: "stake_subsidy",
            json_name: "stakeSubsidy",
            number: 8i32,
            message_fields: Some(StakeSubsidy::FIELDS),
        };
        pub const SHARD_RESULTS_FIELD: &'static MessageField = &MessageField {
            name: "shard_results",
            json_name: "shardResults",
            number: 9i32,
            message_fields: None,
        };
    }
    impl MessageFields for SystemState {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_FIELD,
            Self::EPOCH_START_TIMESTAMP_MS_FIELD,
            Self::PARAMETERS_FIELD,
            Self::VALIDATORS_FIELD,
            Self::ENCODERS_FIELD,
            Self::VALIDATOR_REPORT_RECORDS_FIELD,
            Self::ENCODER_REPORT_RECORDS_FIELD,
            Self::STAKE_SUBSIDY_FIELD,
            Self::SHARD_RESULTS_FIELD,
        ];
    }
    impl SystemState {
        pub fn path_builder() -> SystemStateFieldPathBuilder {
            SystemStateFieldPathBuilder::new()
        }
    }
    pub struct SystemStateFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SystemStateFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn epoch(mut self) -> String {
            self.path.push(SystemState::EPOCH_FIELD.name);
            self.finish()
        }
        pub fn epoch_start_timestamp_ms(mut self) -> String {
            self.path.push(SystemState::EPOCH_START_TIMESTAMP_MS_FIELD.name);
            self.finish()
        }
        pub fn parameters(mut self) -> SystemParametersFieldPathBuilder {
            self.path.push(SystemState::PARAMETERS_FIELD.name);
            SystemParametersFieldPathBuilder::new_with_base(self.path)
        }
        pub fn validators(mut self) -> ValidatorSetFieldPathBuilder {
            self.path.push(SystemState::VALIDATORS_FIELD.name);
            ValidatorSetFieldPathBuilder::new_with_base(self.path)
        }
        pub fn encoders(mut self) -> EncoderSetFieldPathBuilder {
            self.path.push(SystemState::ENCODERS_FIELD.name);
            EncoderSetFieldPathBuilder::new_with_base(self.path)
        }
        pub fn validator_report_records(mut self) -> String {
            self.path.push(SystemState::VALIDATOR_REPORT_RECORDS_FIELD.name);
            self.finish()
        }
        pub fn encoder_report_records(mut self) -> String {
            self.path.push(SystemState::ENCODER_REPORT_RECORDS_FIELD.name);
            self.finish()
        }
        pub fn stake_subsidy(mut self) -> StakeSubsidyFieldPathBuilder {
            self.path.push(SystemState::STAKE_SUBSIDY_FIELD.name);
            StakeSubsidyFieldPathBuilder::new_with_base(self.path)
        }
        pub fn shard_results(mut self) -> String {
            self.path.push(SystemState::SHARD_RESULTS_FIELD.name);
            self.finish()
        }
    }
    impl ReporterSet {
        pub const REPORTERS_FIELD: &'static MessageField = &MessageField {
            name: "reporters",
            json_name: "reporters",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for ReporterSet {
        const FIELDS: &'static [&'static MessageField] = &[Self::REPORTERS_FIELD];
    }
    impl ReporterSet {
        pub fn path_builder() -> ReporterSetFieldPathBuilder {
            ReporterSetFieldPathBuilder::new()
        }
    }
    pub struct ReporterSetFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ReporterSetFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn reporters(mut self) -> String {
            self.path.push(ReporterSet::REPORTERS_FIELD.name);
            self.finish()
        }
    }
    impl SystemParameters {
        pub const EPOCH_DURATION_MS_FIELD: &'static MessageField = &MessageField {
            name: "epoch_duration_ms",
            json_name: "epochDurationMs",
            number: 1i32,
            message_fields: None,
        };
        pub const VDF_ITERATIONS_FIELD: &'static MessageField = &MessageField {
            name: "vdf_iterations",
            json_name: "vdfIterations",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for SystemParameters {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_DURATION_MS_FIELD,
            Self::VDF_ITERATIONS_FIELD,
        ];
    }
    impl SystemParameters {
        pub fn path_builder() -> SystemParametersFieldPathBuilder {
            SystemParametersFieldPathBuilder::new()
        }
    }
    pub struct SystemParametersFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SystemParametersFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn epoch_duration_ms(mut self) -> String {
            self.path.push(SystemParameters::EPOCH_DURATION_MS_FIELD.name);
            self.finish()
        }
        pub fn vdf_iterations(mut self) -> String {
            self.path.push(SystemParameters::VDF_ITERATIONS_FIELD.name);
            self.finish()
        }
    }
    impl StakeSubsidy {
        pub const BALANCE_FIELD: &'static MessageField = &MessageField {
            name: "balance",
            json_name: "balance",
            number: 1i32,
            message_fields: None,
        };
        pub const DISTRIBUTION_COUNTER_FIELD: &'static MessageField = &MessageField {
            name: "distribution_counter",
            json_name: "distributionCounter",
            number: 2i32,
            message_fields: None,
        };
        pub const CURRENT_DISTRIBUTION_AMOUNT_FIELD: &'static MessageField = &MessageField {
            name: "current_distribution_amount",
            json_name: "currentDistributionAmount",
            number: 3i32,
            message_fields: None,
        };
        pub const PERIOD_LENGTH_FIELD: &'static MessageField = &MessageField {
            name: "period_length",
            json_name: "periodLength",
            number: 4i32,
            message_fields: None,
        };
        pub const DECREASE_RATE_FIELD: &'static MessageField = &MessageField {
            name: "decrease_rate",
            json_name: "decreaseRate",
            number: 5i32,
            message_fields: None,
        };
    }
    impl MessageFields for StakeSubsidy {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BALANCE_FIELD,
            Self::DISTRIBUTION_COUNTER_FIELD,
            Self::CURRENT_DISTRIBUTION_AMOUNT_FIELD,
            Self::PERIOD_LENGTH_FIELD,
            Self::DECREASE_RATE_FIELD,
        ];
    }
    impl StakeSubsidy {
        pub fn path_builder() -> StakeSubsidyFieldPathBuilder {
            StakeSubsidyFieldPathBuilder::new()
        }
    }
    pub struct StakeSubsidyFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl StakeSubsidyFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn balance(mut self) -> String {
            self.path.push(StakeSubsidy::BALANCE_FIELD.name);
            self.finish()
        }
        pub fn distribution_counter(mut self) -> String {
            self.path.push(StakeSubsidy::DISTRIBUTION_COUNTER_FIELD.name);
            self.finish()
        }
        pub fn current_distribution_amount(mut self) -> String {
            self.path.push(StakeSubsidy::CURRENT_DISTRIBUTION_AMOUNT_FIELD.name);
            self.finish()
        }
        pub fn period_length(mut self) -> String {
            self.path.push(StakeSubsidy::PERIOD_LENGTH_FIELD.name);
            self.finish()
        }
        pub fn decrease_rate(mut self) -> String {
            self.path.push(StakeSubsidy::DECREASE_RATE_FIELD.name);
            self.finish()
        }
    }
    impl ValidatorSet {
        pub const TOTAL_STAKE_FIELD: &'static MessageField = &MessageField {
            name: "total_stake",
            json_name: "totalStake",
            number: 1i32,
            message_fields: None,
        };
        pub const CONSENSUS_VALIDATORS_FIELD: &'static MessageField = &MessageField {
            name: "consensus_validators",
            json_name: "consensusValidators",
            number: 2i32,
            message_fields: Some(Validator::FIELDS),
        };
        pub const NETWORKING_VALIDATORS_FIELD: &'static MessageField = &MessageField {
            name: "networking_validators",
            json_name: "networkingValidators",
            number: 3i32,
            message_fields: Some(Validator::FIELDS),
        };
        pub const PENDING_VALIDATORS_FIELD: &'static MessageField = &MessageField {
            name: "pending_validators",
            json_name: "pendingValidators",
            number: 4i32,
            message_fields: Some(Validator::FIELDS),
        };
        pub const PENDING_REMOVALS_FIELD: &'static MessageField = &MessageField {
            name: "pending_removals",
            json_name: "pendingRemovals",
            number: 5i32,
            message_fields: Some(PendingRemoval::FIELDS),
        };
        pub const STAKING_POOL_MAPPINGS_FIELD: &'static MessageField = &MessageField {
            name: "staking_pool_mappings",
            json_name: "stakingPoolMappings",
            number: 6i32,
            message_fields: None,
        };
        pub const INACTIVE_VALIDATORS_FIELD: &'static MessageField = &MessageField {
            name: "inactive_validators",
            json_name: "inactiveValidators",
            number: 7i32,
            message_fields: None,
        };
        pub const AT_RISK_VALIDATORS_FIELD: &'static MessageField = &MessageField {
            name: "at_risk_validators",
            json_name: "atRiskValidators",
            number: 8i32,
            message_fields: None,
        };
    }
    impl MessageFields for ValidatorSet {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::TOTAL_STAKE_FIELD,
            Self::CONSENSUS_VALIDATORS_FIELD,
            Self::NETWORKING_VALIDATORS_FIELD,
            Self::PENDING_VALIDATORS_FIELD,
            Self::PENDING_REMOVALS_FIELD,
            Self::STAKING_POOL_MAPPINGS_FIELD,
            Self::INACTIVE_VALIDATORS_FIELD,
            Self::AT_RISK_VALIDATORS_FIELD,
        ];
    }
    impl ValidatorSet {
        pub fn path_builder() -> ValidatorSetFieldPathBuilder {
            ValidatorSetFieldPathBuilder::new()
        }
    }
    pub struct ValidatorSetFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ValidatorSetFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn total_stake(mut self) -> String {
            self.path.push(ValidatorSet::TOTAL_STAKE_FIELD.name);
            self.finish()
        }
        pub fn consensus_validators(mut self) -> ValidatorFieldPathBuilder {
            self.path.push(ValidatorSet::CONSENSUS_VALIDATORS_FIELD.name);
            ValidatorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn networking_validators(mut self) -> ValidatorFieldPathBuilder {
            self.path.push(ValidatorSet::NETWORKING_VALIDATORS_FIELD.name);
            ValidatorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn pending_validators(mut self) -> ValidatorFieldPathBuilder {
            self.path.push(ValidatorSet::PENDING_VALIDATORS_FIELD.name);
            ValidatorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn pending_removals(mut self) -> PendingRemovalFieldPathBuilder {
            self.path.push(ValidatorSet::PENDING_REMOVALS_FIELD.name);
            PendingRemovalFieldPathBuilder::new_with_base(self.path)
        }
        pub fn staking_pool_mappings(mut self) -> String {
            self.path.push(ValidatorSet::STAKING_POOL_MAPPINGS_FIELD.name);
            self.finish()
        }
        pub fn inactive_validators(mut self) -> String {
            self.path.push(ValidatorSet::INACTIVE_VALIDATORS_FIELD.name);
            self.finish()
        }
        pub fn at_risk_validators(mut self) -> String {
            self.path.push(ValidatorSet::AT_RISK_VALIDATORS_FIELD.name);
            self.finish()
        }
    }
    impl PendingRemoval {
        pub const IS_CONSENSUS_FIELD: &'static MessageField = &MessageField {
            name: "is_consensus",
            json_name: "isConsensus",
            number: 1i32,
            message_fields: None,
        };
        pub const INDEX_FIELD: &'static MessageField = &MessageField {
            name: "index",
            json_name: "index",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for PendingRemoval {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::IS_CONSENSUS_FIELD,
            Self::INDEX_FIELD,
        ];
    }
    impl PendingRemoval {
        pub fn path_builder() -> PendingRemovalFieldPathBuilder {
            PendingRemovalFieldPathBuilder::new()
        }
    }
    pub struct PendingRemovalFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl PendingRemovalFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn is_consensus(mut self) -> String {
            self.path.push(PendingRemoval::IS_CONSENSUS_FIELD.name);
            self.finish()
        }
        pub fn index(mut self) -> String {
            self.path.push(PendingRemoval::INDEX_FIELD.name);
            self.finish()
        }
    }
    impl Validator {
        pub const SOMA_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "soma_address",
            json_name: "somaAddress",
            number: 1i32,
            message_fields: None,
        };
        pub const PROTOCOL_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "protocol_pubkey",
            json_name: "protocolPubkey",
            number: 2i32,
            message_fields: None,
        };
        pub const NETWORK_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "network_pubkey",
            json_name: "networkPubkey",
            number: 3i32,
            message_fields: None,
        };
        pub const WORKER_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "worker_pubkey",
            json_name: "workerPubkey",
            number: 4i32,
            message_fields: None,
        };
        pub const NET_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "net_address",
            json_name: "netAddress",
            number: 5i32,
            message_fields: None,
        };
        pub const P2P_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "p2p_address",
            json_name: "p2pAddress",
            number: 6i32,
            message_fields: None,
        };
        pub const PRIMARY_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "primary_address",
            json_name: "primaryAddress",
            number: 7i32,
            message_fields: None,
        };
        pub const ENCODER_VALIDATOR_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "encoder_validator_address",
            json_name: "encoderValidatorAddress",
            number: 8i32,
            message_fields: None,
        };
        pub const VOTING_POWER_FIELD: &'static MessageField = &MessageField {
            name: "voting_power",
            json_name: "votingPower",
            number: 9i32,
            message_fields: None,
        };
        pub const COMMISSION_RATE_FIELD: &'static MessageField = &MessageField {
            name: "commission_rate",
            json_name: "commissionRate",
            number: 10i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_STAKE_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_stake",
            json_name: "nextEpochStake",
            number: 11i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_COMMISSION_RATE_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_commission_rate",
            json_name: "nextEpochCommissionRate",
            number: 12i32,
            message_fields: None,
        };
        pub const STAKING_POOL_FIELD: &'static MessageField = &MessageField {
            name: "staking_pool",
            json_name: "stakingPool",
            number: 13i32,
            message_fields: Some(StakingPool::FIELDS),
        };
        pub const NEXT_EPOCH_PROTOCOL_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_protocol_pubkey",
            json_name: "nextEpochProtocolPubkey",
            number: 14i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_NETWORK_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_network_pubkey",
            json_name: "nextEpochNetworkPubkey",
            number: 15i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_WORKER_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_worker_pubkey",
            json_name: "nextEpochWorkerPubkey",
            number: 16i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_NET_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_net_address",
            json_name: "nextEpochNetAddress",
            number: 17i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_P2P_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_p2p_address",
            json_name: "nextEpochP2pAddress",
            number: 18i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_PRIMARY_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_primary_address",
            json_name: "nextEpochPrimaryAddress",
            number: 19i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_ENCODER_VALIDATOR_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_encoder_validator_address",
            json_name: "nextEpochEncoderValidatorAddress",
            number: 20i32,
            message_fields: None,
        };
    }
    impl MessageFields for Validator {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SOMA_ADDRESS_FIELD,
            Self::PROTOCOL_PUBKEY_FIELD,
            Self::NETWORK_PUBKEY_FIELD,
            Self::WORKER_PUBKEY_FIELD,
            Self::NET_ADDRESS_FIELD,
            Self::P2P_ADDRESS_FIELD,
            Self::PRIMARY_ADDRESS_FIELD,
            Self::ENCODER_VALIDATOR_ADDRESS_FIELD,
            Self::VOTING_POWER_FIELD,
            Self::COMMISSION_RATE_FIELD,
            Self::NEXT_EPOCH_STAKE_FIELD,
            Self::NEXT_EPOCH_COMMISSION_RATE_FIELD,
            Self::STAKING_POOL_FIELD,
            Self::NEXT_EPOCH_PROTOCOL_PUBKEY_FIELD,
            Self::NEXT_EPOCH_NETWORK_PUBKEY_FIELD,
            Self::NEXT_EPOCH_WORKER_PUBKEY_FIELD,
            Self::NEXT_EPOCH_NET_ADDRESS_FIELD,
            Self::NEXT_EPOCH_P2P_ADDRESS_FIELD,
            Self::NEXT_EPOCH_PRIMARY_ADDRESS_FIELD,
            Self::NEXT_EPOCH_ENCODER_VALIDATOR_ADDRESS_FIELD,
        ];
    }
    impl Validator {
        pub fn path_builder() -> ValidatorFieldPathBuilder {
            ValidatorFieldPathBuilder::new()
        }
    }
    pub struct ValidatorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ValidatorFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn soma_address(mut self) -> String {
            self.path.push(Validator::SOMA_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn protocol_pubkey(mut self) -> String {
            self.path.push(Validator::PROTOCOL_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn network_pubkey(mut self) -> String {
            self.path.push(Validator::NETWORK_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn worker_pubkey(mut self) -> String {
            self.path.push(Validator::WORKER_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn net_address(mut self) -> String {
            self.path.push(Validator::NET_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn p2p_address(mut self) -> String {
            self.path.push(Validator::P2P_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn primary_address(mut self) -> String {
            self.path.push(Validator::PRIMARY_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn encoder_validator_address(mut self) -> String {
            self.path.push(Validator::ENCODER_VALIDATOR_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn voting_power(mut self) -> String {
            self.path.push(Validator::VOTING_POWER_FIELD.name);
            self.finish()
        }
        pub fn commission_rate(mut self) -> String {
            self.path.push(Validator::COMMISSION_RATE_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_stake(mut self) -> String {
            self.path.push(Validator::NEXT_EPOCH_STAKE_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_commission_rate(mut self) -> String {
            self.path.push(Validator::NEXT_EPOCH_COMMISSION_RATE_FIELD.name);
            self.finish()
        }
        pub fn staking_pool(mut self) -> StakingPoolFieldPathBuilder {
            self.path.push(Validator::STAKING_POOL_FIELD.name);
            StakingPoolFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_epoch_protocol_pubkey(mut self) -> String {
            self.path.push(Validator::NEXT_EPOCH_PROTOCOL_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_network_pubkey(mut self) -> String {
            self.path.push(Validator::NEXT_EPOCH_NETWORK_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_worker_pubkey(mut self) -> String {
            self.path.push(Validator::NEXT_EPOCH_WORKER_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_net_address(mut self) -> String {
            self.path.push(Validator::NEXT_EPOCH_NET_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_p2p_address(mut self) -> String {
            self.path.push(Validator::NEXT_EPOCH_P2P_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_primary_address(mut self) -> String {
            self.path.push(Validator::NEXT_EPOCH_PRIMARY_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_encoder_validator_address(mut self) -> String {
            self.path.push(Validator::NEXT_EPOCH_ENCODER_VALIDATOR_ADDRESS_FIELD.name);
            self.finish()
        }
    }
    impl StakingPool {
        pub const ID_FIELD: &'static MessageField = &MessageField {
            name: "id",
            json_name: "id",
            number: 1i32,
            message_fields: None,
        };
        pub const ACTIVATION_EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "activation_epoch",
            json_name: "activationEpoch",
            number: 2i32,
            message_fields: None,
        };
        pub const DEACTIVATION_EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "deactivation_epoch",
            json_name: "deactivationEpoch",
            number: 3i32,
            message_fields: None,
        };
        pub const SOMA_BALANCE_FIELD: &'static MessageField = &MessageField {
            name: "soma_balance",
            json_name: "somaBalance",
            number: 4i32,
            message_fields: None,
        };
        pub const REWARDS_POOL_FIELD: &'static MessageField = &MessageField {
            name: "rewards_pool",
            json_name: "rewardsPool",
            number: 5i32,
            message_fields: None,
        };
        pub const POOL_TOKEN_BALANCE_FIELD: &'static MessageField = &MessageField {
            name: "pool_token_balance",
            json_name: "poolTokenBalance",
            number: 6i32,
            message_fields: None,
        };
        pub const EXCHANGE_RATES_FIELD: &'static MessageField = &MessageField {
            name: "exchange_rates",
            json_name: "exchangeRates",
            number: 7i32,
            message_fields: None,
        };
        pub const PENDING_STAKE_FIELD: &'static MessageField = &MessageField {
            name: "pending_stake",
            json_name: "pendingStake",
            number: 8i32,
            message_fields: None,
        };
        pub const PENDING_TOTAL_SOMA_WITHDRAW_FIELD: &'static MessageField = &MessageField {
            name: "pending_total_soma_withdraw",
            json_name: "pendingTotalSomaWithdraw",
            number: 9i32,
            message_fields: None,
        };
        pub const PENDING_POOL_TOKEN_WITHDRAW_FIELD: &'static MessageField = &MessageField {
            name: "pending_pool_token_withdraw",
            json_name: "pendingPoolTokenWithdraw",
            number: 10i32,
            message_fields: None,
        };
    }
    impl MessageFields for StakingPool {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ID_FIELD,
            Self::ACTIVATION_EPOCH_FIELD,
            Self::DEACTIVATION_EPOCH_FIELD,
            Self::SOMA_BALANCE_FIELD,
            Self::REWARDS_POOL_FIELD,
            Self::POOL_TOKEN_BALANCE_FIELD,
            Self::EXCHANGE_RATES_FIELD,
            Self::PENDING_STAKE_FIELD,
            Self::PENDING_TOTAL_SOMA_WITHDRAW_FIELD,
            Self::PENDING_POOL_TOKEN_WITHDRAW_FIELD,
        ];
    }
    impl StakingPool {
        pub fn path_builder() -> StakingPoolFieldPathBuilder {
            StakingPoolFieldPathBuilder::new()
        }
    }
    pub struct StakingPoolFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl StakingPoolFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn id(mut self) -> String {
            self.path.push(StakingPool::ID_FIELD.name);
            self.finish()
        }
        pub fn activation_epoch(mut self) -> String {
            self.path.push(StakingPool::ACTIVATION_EPOCH_FIELD.name);
            self.finish()
        }
        pub fn deactivation_epoch(mut self) -> String {
            self.path.push(StakingPool::DEACTIVATION_EPOCH_FIELD.name);
            self.finish()
        }
        pub fn soma_balance(mut self) -> String {
            self.path.push(StakingPool::SOMA_BALANCE_FIELD.name);
            self.finish()
        }
        pub fn rewards_pool(mut self) -> String {
            self.path.push(StakingPool::REWARDS_POOL_FIELD.name);
            self.finish()
        }
        pub fn pool_token_balance(mut self) -> String {
            self.path.push(StakingPool::POOL_TOKEN_BALANCE_FIELD.name);
            self.finish()
        }
        pub fn exchange_rates(mut self) -> String {
            self.path.push(StakingPool::EXCHANGE_RATES_FIELD.name);
            self.finish()
        }
        pub fn pending_stake(mut self) -> String {
            self.path.push(StakingPool::PENDING_STAKE_FIELD.name);
            self.finish()
        }
        pub fn pending_total_soma_withdraw(mut self) -> String {
            self.path.push(StakingPool::PENDING_TOTAL_SOMA_WITHDRAW_FIELD.name);
            self.finish()
        }
        pub fn pending_pool_token_withdraw(mut self) -> String {
            self.path.push(StakingPool::PENDING_POOL_TOKEN_WITHDRAW_FIELD.name);
            self.finish()
        }
    }
    impl PoolTokenExchangeRate {
        pub const SOMA_AMOUNT_FIELD: &'static MessageField = &MessageField {
            name: "soma_amount",
            json_name: "somaAmount",
            number: 1i32,
            message_fields: None,
        };
        pub const POOL_TOKEN_AMOUNT_FIELD: &'static MessageField = &MessageField {
            name: "pool_token_amount",
            json_name: "poolTokenAmount",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for PoolTokenExchangeRate {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SOMA_AMOUNT_FIELD,
            Self::POOL_TOKEN_AMOUNT_FIELD,
        ];
    }
    impl PoolTokenExchangeRate {
        pub fn path_builder() -> PoolTokenExchangeRateFieldPathBuilder {
            PoolTokenExchangeRateFieldPathBuilder::new()
        }
    }
    pub struct PoolTokenExchangeRateFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl PoolTokenExchangeRateFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn soma_amount(mut self) -> String {
            self.path.push(PoolTokenExchangeRate::SOMA_AMOUNT_FIELD.name);
            self.finish()
        }
        pub fn pool_token_amount(mut self) -> String {
            self.path.push(PoolTokenExchangeRate::POOL_TOKEN_AMOUNT_FIELD.name);
            self.finish()
        }
    }
    impl EncoderSet {
        pub const TOTAL_STAKE_FIELD: &'static MessageField = &MessageField {
            name: "total_stake",
            json_name: "totalStake",
            number: 1i32,
            message_fields: None,
        };
        pub const ACTIVE_ENCODERS_FIELD: &'static MessageField = &MessageField {
            name: "active_encoders",
            json_name: "activeEncoders",
            number: 2i32,
            message_fields: Some(Encoder::FIELDS),
        };
        pub const PENDING_ACTIVE_ENCODERS_FIELD: &'static MessageField = &MessageField {
            name: "pending_active_encoders",
            json_name: "pendingActiveEncoders",
            number: 3i32,
            message_fields: Some(Encoder::FIELDS),
        };
        pub const PENDING_REMOVALS_FIELD: &'static MessageField = &MessageField {
            name: "pending_removals",
            json_name: "pendingRemovals",
            number: 4i32,
            message_fields: None,
        };
        pub const STAKING_POOL_MAPPINGS_FIELD: &'static MessageField = &MessageField {
            name: "staking_pool_mappings",
            json_name: "stakingPoolMappings",
            number: 5i32,
            message_fields: None,
        };
        pub const INACTIVE_ENCODERS_FIELD: &'static MessageField = &MessageField {
            name: "inactive_encoders",
            json_name: "inactiveEncoders",
            number: 6i32,
            message_fields: None,
        };
        pub const AT_RISK_ENCODERS_FIELD: &'static MessageField = &MessageField {
            name: "at_risk_encoders",
            json_name: "atRiskEncoders",
            number: 7i32,
            message_fields: None,
        };
        pub const REFERENCE_BYTE_PRICE_FIELD: &'static MessageField = &MessageField {
            name: "reference_byte_price",
            json_name: "referenceBytePrice",
            number: 8i32,
            message_fields: None,
        };
    }
    impl MessageFields for EncoderSet {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::TOTAL_STAKE_FIELD,
            Self::ACTIVE_ENCODERS_FIELD,
            Self::PENDING_ACTIVE_ENCODERS_FIELD,
            Self::PENDING_REMOVALS_FIELD,
            Self::STAKING_POOL_MAPPINGS_FIELD,
            Self::INACTIVE_ENCODERS_FIELD,
            Self::AT_RISK_ENCODERS_FIELD,
            Self::REFERENCE_BYTE_PRICE_FIELD,
        ];
    }
    impl EncoderSet {
        pub fn path_builder() -> EncoderSetFieldPathBuilder {
            EncoderSetFieldPathBuilder::new()
        }
    }
    pub struct EncoderSetFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl EncoderSetFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn total_stake(mut self) -> String {
            self.path.push(EncoderSet::TOTAL_STAKE_FIELD.name);
            self.finish()
        }
        pub fn active_encoders(mut self) -> EncoderFieldPathBuilder {
            self.path.push(EncoderSet::ACTIVE_ENCODERS_FIELD.name);
            EncoderFieldPathBuilder::new_with_base(self.path)
        }
        pub fn pending_active_encoders(mut self) -> EncoderFieldPathBuilder {
            self.path.push(EncoderSet::PENDING_ACTIVE_ENCODERS_FIELD.name);
            EncoderFieldPathBuilder::new_with_base(self.path)
        }
        pub fn pending_removals(mut self) -> String {
            self.path.push(EncoderSet::PENDING_REMOVALS_FIELD.name);
            self.finish()
        }
        pub fn staking_pool_mappings(mut self) -> String {
            self.path.push(EncoderSet::STAKING_POOL_MAPPINGS_FIELD.name);
            self.finish()
        }
        pub fn inactive_encoders(mut self) -> String {
            self.path.push(EncoderSet::INACTIVE_ENCODERS_FIELD.name);
            self.finish()
        }
        pub fn at_risk_encoders(mut self) -> String {
            self.path.push(EncoderSet::AT_RISK_ENCODERS_FIELD.name);
            self.finish()
        }
        pub fn reference_byte_price(mut self) -> String {
            self.path.push(EncoderSet::REFERENCE_BYTE_PRICE_FIELD.name);
            self.finish()
        }
    }
    impl Encoder {
        pub const SOMA_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "soma_address",
            json_name: "somaAddress",
            number: 1i32,
            message_fields: None,
        };
        pub const ENCODER_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "encoder_pubkey",
            json_name: "encoderPubkey",
            number: 2i32,
            message_fields: None,
        };
        pub const NETWORK_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "network_pubkey",
            json_name: "networkPubkey",
            number: 3i32,
            message_fields: None,
        };
        pub const INTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "internal_network_address",
            json_name: "internalNetworkAddress",
            number: 4i32,
            message_fields: None,
        };
        pub const EXTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "external_network_address",
            json_name: "externalNetworkAddress",
            number: 5i32,
            message_fields: None,
        };
        pub const OBJECT_SERVER_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "object_server_address",
            json_name: "objectServerAddress",
            number: 6i32,
            message_fields: None,
        };
        pub const VOTING_POWER_FIELD: &'static MessageField = &MessageField {
            name: "voting_power",
            json_name: "votingPower",
            number: 7i32,
            message_fields: None,
        };
        pub const COMMISSION_RATE_FIELD: &'static MessageField = &MessageField {
            name: "commission_rate",
            json_name: "commissionRate",
            number: 8i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_STAKE_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_stake",
            json_name: "nextEpochStake",
            number: 9i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_COMMISSION_RATE_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_commission_rate",
            json_name: "nextEpochCommissionRate",
            number: 10i32,
            message_fields: None,
        };
        pub const BYTE_PRICE_FIELD: &'static MessageField = &MessageField {
            name: "byte_price",
            json_name: "bytePrice",
            number: 11i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_BYTE_PRICE_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_byte_price",
            json_name: "nextEpochBytePrice",
            number: 12i32,
            message_fields: None,
        };
        pub const STAKING_POOL_FIELD: &'static MessageField = &MessageField {
            name: "staking_pool",
            json_name: "stakingPool",
            number: 13i32,
            message_fields: Some(StakingPool::FIELDS),
        };
        pub const NEXT_EPOCH_NETWORK_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_network_pubkey",
            json_name: "nextEpochNetworkPubkey",
            number: 14i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_INTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_internal_network_address",
            json_name: "nextEpochInternalNetworkAddress",
            number: 15i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_EXTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_external_network_address",
            json_name: "nextEpochExternalNetworkAddress",
            number: 16i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_OBJECT_SERVER_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_object_server_address",
            json_name: "nextEpochObjectServerAddress",
            number: 17i32,
            message_fields: None,
        };
    }
    impl MessageFields for Encoder {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SOMA_ADDRESS_FIELD,
            Self::ENCODER_PUBKEY_FIELD,
            Self::NETWORK_PUBKEY_FIELD,
            Self::INTERNAL_NETWORK_ADDRESS_FIELD,
            Self::EXTERNAL_NETWORK_ADDRESS_FIELD,
            Self::OBJECT_SERVER_ADDRESS_FIELD,
            Self::VOTING_POWER_FIELD,
            Self::COMMISSION_RATE_FIELD,
            Self::NEXT_EPOCH_STAKE_FIELD,
            Self::NEXT_EPOCH_COMMISSION_RATE_FIELD,
            Self::BYTE_PRICE_FIELD,
            Self::NEXT_EPOCH_BYTE_PRICE_FIELD,
            Self::STAKING_POOL_FIELD,
            Self::NEXT_EPOCH_NETWORK_PUBKEY_FIELD,
            Self::NEXT_EPOCH_INTERNAL_NETWORK_ADDRESS_FIELD,
            Self::NEXT_EPOCH_EXTERNAL_NETWORK_ADDRESS_FIELD,
            Self::NEXT_EPOCH_OBJECT_SERVER_ADDRESS_FIELD,
        ];
    }
    impl Encoder {
        pub fn path_builder() -> EncoderFieldPathBuilder {
            EncoderFieldPathBuilder::new()
        }
    }
    pub struct EncoderFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl EncoderFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn soma_address(mut self) -> String {
            self.path.push(Encoder::SOMA_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn encoder_pubkey(mut self) -> String {
            self.path.push(Encoder::ENCODER_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn network_pubkey(mut self) -> String {
            self.path.push(Encoder::NETWORK_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn internal_network_address(mut self) -> String {
            self.path.push(Encoder::INTERNAL_NETWORK_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn external_network_address(mut self) -> String {
            self.path.push(Encoder::EXTERNAL_NETWORK_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn object_server_address(mut self) -> String {
            self.path.push(Encoder::OBJECT_SERVER_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn voting_power(mut self) -> String {
            self.path.push(Encoder::VOTING_POWER_FIELD.name);
            self.finish()
        }
        pub fn commission_rate(mut self) -> String {
            self.path.push(Encoder::COMMISSION_RATE_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_stake(mut self) -> String {
            self.path.push(Encoder::NEXT_EPOCH_STAKE_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_commission_rate(mut self) -> String {
            self.path.push(Encoder::NEXT_EPOCH_COMMISSION_RATE_FIELD.name);
            self.finish()
        }
        pub fn byte_price(mut self) -> String {
            self.path.push(Encoder::BYTE_PRICE_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_byte_price(mut self) -> String {
            self.path.push(Encoder::NEXT_EPOCH_BYTE_PRICE_FIELD.name);
            self.finish()
        }
        pub fn staking_pool(mut self) -> StakingPoolFieldPathBuilder {
            self.path.push(Encoder::STAKING_POOL_FIELD.name);
            StakingPoolFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_epoch_network_pubkey(mut self) -> String {
            self.path.push(Encoder::NEXT_EPOCH_NETWORK_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_internal_network_address(mut self) -> String {
            self.path.push(Encoder::NEXT_EPOCH_INTERNAL_NETWORK_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_external_network_address(mut self) -> String {
            self.path.push(Encoder::NEXT_EPOCH_EXTERNAL_NETWORK_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_object_server_address(mut self) -> String {
            self.path.push(Encoder::NEXT_EPOCH_OBJECT_SERVER_ADDRESS_FIELD.name);
            self.finish()
        }
    }
    impl ShardResult {
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 1i32,
            message_fields: None,
        };
        pub const DATA_SIZE_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "data_size_bytes",
            json_name: "dataSizeBytes",
            number: 2i32,
            message_fields: None,
        };
        pub const AMOUNT_FIELD: &'static MessageField = &MessageField {
            name: "amount",
            json_name: "amount",
            number: 3i32,
            message_fields: None,
        };
        pub const REPORT_FIELD: &'static MessageField = &MessageField {
            name: "report",
            json_name: "report",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for ShardResult {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::DIGEST_FIELD,
            Self::DATA_SIZE_BYTES_FIELD,
            Self::AMOUNT_FIELD,
            Self::REPORT_FIELD,
        ];
    }
    impl ShardResult {
        pub fn path_builder() -> ShardResultFieldPathBuilder {
            ShardResultFieldPathBuilder::new()
        }
    }
    pub struct ShardResultFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ShardResultFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn digest(mut self) -> String {
            self.path.push(ShardResult::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn data_size_bytes(mut self) -> String {
            self.path.push(ShardResult::DATA_SIZE_BYTES_FIELD.name);
            self.finish()
        }
        pub fn amount(mut self) -> String {
            self.path.push(ShardResult::AMOUNT_FIELD.name);
            self.finish()
        }
        pub fn report(mut self) -> String {
            self.path.push(ShardResult::REPORT_FIELD.name);
            self.finish()
        }
    }
    impl Transaction {
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 1i32,
            message_fields: None,
        };
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 2i32,
            message_fields: Some(TransactionKind::FIELDS),
        };
        pub const SENDER_FIELD: &'static MessageField = &MessageField {
            name: "sender",
            json_name: "sender",
            number: 3i32,
            message_fields: None,
        };
        pub const GAS_PAYMENT_FIELD: &'static MessageField = &MessageField {
            name: "gas_payment",
            json_name: "gasPayment",
            number: 4i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
    }
    impl MessageFields for Transaction {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::DIGEST_FIELD,
            Self::KIND_FIELD,
            Self::SENDER_FIELD,
            Self::GAS_PAYMENT_FIELD,
        ];
    }
    impl Transaction {
        pub fn path_builder() -> TransactionFieldPathBuilder {
            TransactionFieldPathBuilder::new()
        }
    }
    pub struct TransactionFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TransactionFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn digest(mut self) -> String {
            self.path.push(Transaction::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn kind(mut self) -> TransactionKindFieldPathBuilder {
            self.path.push(Transaction::KIND_FIELD.name);
            TransactionKindFieldPathBuilder::new_with_base(self.path)
        }
        pub fn sender(mut self) -> String {
            self.path.push(Transaction::SENDER_FIELD.name);
            self.finish()
        }
        pub fn gas_payment(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(Transaction::GAS_PAYMENT_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl TransactionKind {
        pub const GENESIS_FIELD: &'static MessageField = &MessageField {
            name: "genesis",
            json_name: "genesis",
            number: 1i32,
            message_fields: Some(GenesisTransaction::FIELDS),
        };
        pub const CONSENSUS_COMMIT_PROLOGUE_FIELD: &'static MessageField = &MessageField {
            name: "consensus_commit_prologue",
            json_name: "consensusCommitPrologue",
            number: 2i32,
            message_fields: Some(ConsensusCommitPrologue::FIELDS),
        };
        pub const CHANGE_EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "change_epoch",
            json_name: "changeEpoch",
            number: 3i32,
            message_fields: Some(ChangeEpoch::FIELDS),
        };
        pub const ADD_VALIDATOR_FIELD: &'static MessageField = &MessageField {
            name: "add_validator",
            json_name: "addValidator",
            number: 4i32,
            message_fields: Some(AddValidator::FIELDS),
        };
        pub const REMOVE_VALIDATOR_FIELD: &'static MessageField = &MessageField {
            name: "remove_validator",
            json_name: "removeValidator",
            number: 5i32,
            message_fields: Some(RemoveValidator::FIELDS),
        };
        pub const REPORT_VALIDATOR_FIELD: &'static MessageField = &MessageField {
            name: "report_validator",
            json_name: "reportValidator",
            number: 6i32,
            message_fields: Some(ReportValidator::FIELDS),
        };
        pub const UNDO_REPORT_VALIDATOR_FIELD: &'static MessageField = &MessageField {
            name: "undo_report_validator",
            json_name: "undoReportValidator",
            number: 7i32,
            message_fields: Some(UndoReportValidator::FIELDS),
        };
        pub const UPDATE_VALIDATOR_METADATA_FIELD: &'static MessageField = &MessageField {
            name: "update_validator_metadata",
            json_name: "updateValidatorMetadata",
            number: 8i32,
            message_fields: Some(UpdateValidatorMetadata::FIELDS),
        };
        pub const SET_COMMISSION_RATE_FIELD: &'static MessageField = &MessageField {
            name: "set_commission_rate",
            json_name: "setCommissionRate",
            number: 9i32,
            message_fields: Some(SetCommissionRate::FIELDS),
        };
        pub const ADD_ENCODER_FIELD: &'static MessageField = &MessageField {
            name: "add_encoder",
            json_name: "addEncoder",
            number: 10i32,
            message_fields: Some(AddEncoder::FIELDS),
        };
        pub const REMOVE_ENCODER_FIELD: &'static MessageField = &MessageField {
            name: "remove_encoder",
            json_name: "removeEncoder",
            number: 11i32,
            message_fields: Some(RemoveEncoder::FIELDS),
        };
        pub const REPORT_ENCODER_FIELD: &'static MessageField = &MessageField {
            name: "report_encoder",
            json_name: "reportEncoder",
            number: 12i32,
            message_fields: Some(ReportEncoder::FIELDS),
        };
        pub const UNDO_REPORT_ENCODER_FIELD: &'static MessageField = &MessageField {
            name: "undo_report_encoder",
            json_name: "undoReportEncoder",
            number: 13i32,
            message_fields: Some(UndoReportEncoder::FIELDS),
        };
        pub const UPDATE_ENCODER_METADATA_FIELD: &'static MessageField = &MessageField {
            name: "update_encoder_metadata",
            json_name: "updateEncoderMetadata",
            number: 14i32,
            message_fields: Some(UpdateEncoderMetadata::FIELDS),
        };
        pub const SET_ENCODER_COMMISSION_RATE_FIELD: &'static MessageField = &MessageField {
            name: "set_encoder_commission_rate",
            json_name: "setEncoderCommissionRate",
            number: 15i32,
            message_fields: Some(SetEncoderCommissionRate::FIELDS),
        };
        pub const SET_ENCODER_BYTE_PRICE_FIELD: &'static MessageField = &MessageField {
            name: "set_encoder_byte_price",
            json_name: "setEncoderBytePrice",
            number: 16i32,
            message_fields: Some(SetEncoderBytePrice::FIELDS),
        };
        pub const TRANSFER_COIN_FIELD: &'static MessageField = &MessageField {
            name: "transfer_coin",
            json_name: "transferCoin",
            number: 17i32,
            message_fields: Some(TransferCoin::FIELDS),
        };
        pub const PAY_COINS_FIELD: &'static MessageField = &MessageField {
            name: "pay_coins",
            json_name: "payCoins",
            number: 18i32,
            message_fields: Some(PayCoins::FIELDS),
        };
        pub const TRANSFER_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "transfer_objects",
            json_name: "transferObjects",
            number: 19i32,
            message_fields: Some(TransferObjects::FIELDS),
        };
        pub const ADD_STAKE_FIELD: &'static MessageField = &MessageField {
            name: "add_stake",
            json_name: "addStake",
            number: 20i32,
            message_fields: Some(AddStake::FIELDS),
        };
        pub const ADD_STAKE_TO_ENCODER_FIELD: &'static MessageField = &MessageField {
            name: "add_stake_to_encoder",
            json_name: "addStakeToEncoder",
            number: 21i32,
            message_fields: Some(AddStakeToEncoder::FIELDS),
        };
        pub const WITHDRAW_STAKE_FIELD: &'static MessageField = &MessageField {
            name: "withdraw_stake",
            json_name: "withdrawStake",
            number: 22i32,
            message_fields: Some(WithdrawStake::FIELDS),
        };
        pub const EMBED_DATA_FIELD: &'static MessageField = &MessageField {
            name: "embed_data",
            json_name: "embedData",
            number: 23i32,
            message_fields: Some(EmbedData::FIELDS),
        };
        pub const CLAIM_ESCROW_FIELD: &'static MessageField = &MessageField {
            name: "claim_escrow",
            json_name: "claimEscrow",
            number: 24i32,
            message_fields: Some(ClaimEscrow::FIELDS),
        };
        pub const REPORT_WINNER_FIELD: &'static MessageField = &MessageField {
            name: "report_winner",
            json_name: "reportWinner",
            number: 25i32,
            message_fields: Some(ReportWinner::FIELDS),
        };
    }
    impl MessageFields for TransactionKind {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::GENESIS_FIELD,
            Self::CONSENSUS_COMMIT_PROLOGUE_FIELD,
            Self::CHANGE_EPOCH_FIELD,
            Self::ADD_VALIDATOR_FIELD,
            Self::REMOVE_VALIDATOR_FIELD,
            Self::REPORT_VALIDATOR_FIELD,
            Self::UNDO_REPORT_VALIDATOR_FIELD,
            Self::UPDATE_VALIDATOR_METADATA_FIELD,
            Self::SET_COMMISSION_RATE_FIELD,
            Self::ADD_ENCODER_FIELD,
            Self::REMOVE_ENCODER_FIELD,
            Self::REPORT_ENCODER_FIELD,
            Self::UNDO_REPORT_ENCODER_FIELD,
            Self::UPDATE_ENCODER_METADATA_FIELD,
            Self::SET_ENCODER_COMMISSION_RATE_FIELD,
            Self::SET_ENCODER_BYTE_PRICE_FIELD,
            Self::TRANSFER_COIN_FIELD,
            Self::PAY_COINS_FIELD,
            Self::TRANSFER_OBJECTS_FIELD,
            Self::ADD_STAKE_FIELD,
            Self::ADD_STAKE_TO_ENCODER_FIELD,
            Self::WITHDRAW_STAKE_FIELD,
            Self::EMBED_DATA_FIELD,
            Self::CLAIM_ESCROW_FIELD,
            Self::REPORT_WINNER_FIELD,
        ];
    }
    impl TransactionKind {
        pub fn path_builder() -> TransactionKindFieldPathBuilder {
            TransactionKindFieldPathBuilder::new()
        }
    }
    pub struct TransactionKindFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TransactionKindFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn genesis(mut self) -> GenesisTransactionFieldPathBuilder {
            self.path.push(TransactionKind::GENESIS_FIELD.name);
            GenesisTransactionFieldPathBuilder::new_with_base(self.path)
        }
        pub fn consensus_commit_prologue(
            mut self,
        ) -> ConsensusCommitPrologueFieldPathBuilder {
            self.path.push(TransactionKind::CONSENSUS_COMMIT_PROLOGUE_FIELD.name);
            ConsensusCommitPrologueFieldPathBuilder::new_with_base(self.path)
        }
        pub fn change_epoch(mut self) -> ChangeEpochFieldPathBuilder {
            self.path.push(TransactionKind::CHANGE_EPOCH_FIELD.name);
            ChangeEpochFieldPathBuilder::new_with_base(self.path)
        }
        pub fn add_validator(mut self) -> AddValidatorFieldPathBuilder {
            self.path.push(TransactionKind::ADD_VALIDATOR_FIELD.name);
            AddValidatorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn remove_validator(mut self) -> RemoveValidatorFieldPathBuilder {
            self.path.push(TransactionKind::REMOVE_VALIDATOR_FIELD.name);
            RemoveValidatorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn report_validator(mut self) -> ReportValidatorFieldPathBuilder {
            self.path.push(TransactionKind::REPORT_VALIDATOR_FIELD.name);
            ReportValidatorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn undo_report_validator(mut self) -> UndoReportValidatorFieldPathBuilder {
            self.path.push(TransactionKind::UNDO_REPORT_VALIDATOR_FIELD.name);
            UndoReportValidatorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn update_validator_metadata(
            mut self,
        ) -> UpdateValidatorMetadataFieldPathBuilder {
            self.path.push(TransactionKind::UPDATE_VALIDATOR_METADATA_FIELD.name);
            UpdateValidatorMetadataFieldPathBuilder::new_with_base(self.path)
        }
        pub fn set_commission_rate(mut self) -> SetCommissionRateFieldPathBuilder {
            self.path.push(TransactionKind::SET_COMMISSION_RATE_FIELD.name);
            SetCommissionRateFieldPathBuilder::new_with_base(self.path)
        }
        pub fn add_encoder(mut self) -> AddEncoderFieldPathBuilder {
            self.path.push(TransactionKind::ADD_ENCODER_FIELD.name);
            AddEncoderFieldPathBuilder::new_with_base(self.path)
        }
        pub fn remove_encoder(mut self) -> RemoveEncoderFieldPathBuilder {
            self.path.push(TransactionKind::REMOVE_ENCODER_FIELD.name);
            RemoveEncoderFieldPathBuilder::new_with_base(self.path)
        }
        pub fn report_encoder(mut self) -> ReportEncoderFieldPathBuilder {
            self.path.push(TransactionKind::REPORT_ENCODER_FIELD.name);
            ReportEncoderFieldPathBuilder::new_with_base(self.path)
        }
        pub fn undo_report_encoder(mut self) -> UndoReportEncoderFieldPathBuilder {
            self.path.push(TransactionKind::UNDO_REPORT_ENCODER_FIELD.name);
            UndoReportEncoderFieldPathBuilder::new_with_base(self.path)
        }
        pub fn update_encoder_metadata(
            mut self,
        ) -> UpdateEncoderMetadataFieldPathBuilder {
            self.path.push(TransactionKind::UPDATE_ENCODER_METADATA_FIELD.name);
            UpdateEncoderMetadataFieldPathBuilder::new_with_base(self.path)
        }
        pub fn set_encoder_commission_rate(
            mut self,
        ) -> SetEncoderCommissionRateFieldPathBuilder {
            self.path.push(TransactionKind::SET_ENCODER_COMMISSION_RATE_FIELD.name);
            SetEncoderCommissionRateFieldPathBuilder::new_with_base(self.path)
        }
        pub fn set_encoder_byte_price(mut self) -> SetEncoderBytePriceFieldPathBuilder {
            self.path.push(TransactionKind::SET_ENCODER_BYTE_PRICE_FIELD.name);
            SetEncoderBytePriceFieldPathBuilder::new_with_base(self.path)
        }
        pub fn transfer_coin(mut self) -> TransferCoinFieldPathBuilder {
            self.path.push(TransactionKind::TRANSFER_COIN_FIELD.name);
            TransferCoinFieldPathBuilder::new_with_base(self.path)
        }
        pub fn pay_coins(mut self) -> PayCoinsFieldPathBuilder {
            self.path.push(TransactionKind::PAY_COINS_FIELD.name);
            PayCoinsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn transfer_objects(mut self) -> TransferObjectsFieldPathBuilder {
            self.path.push(TransactionKind::TRANSFER_OBJECTS_FIELD.name);
            TransferObjectsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn add_stake(mut self) -> AddStakeFieldPathBuilder {
            self.path.push(TransactionKind::ADD_STAKE_FIELD.name);
            AddStakeFieldPathBuilder::new_with_base(self.path)
        }
        pub fn add_stake_to_encoder(mut self) -> AddStakeToEncoderFieldPathBuilder {
            self.path.push(TransactionKind::ADD_STAKE_TO_ENCODER_FIELD.name);
            AddStakeToEncoderFieldPathBuilder::new_with_base(self.path)
        }
        pub fn withdraw_stake(mut self) -> WithdrawStakeFieldPathBuilder {
            self.path.push(TransactionKind::WITHDRAW_STAKE_FIELD.name);
            WithdrawStakeFieldPathBuilder::new_with_base(self.path)
        }
        pub fn embed_data(mut self) -> EmbedDataFieldPathBuilder {
            self.path.push(TransactionKind::EMBED_DATA_FIELD.name);
            EmbedDataFieldPathBuilder::new_with_base(self.path)
        }
        pub fn claim_escrow(mut self) -> ClaimEscrowFieldPathBuilder {
            self.path.push(TransactionKind::CLAIM_ESCROW_FIELD.name);
            ClaimEscrowFieldPathBuilder::new_with_base(self.path)
        }
        pub fn report_winner(mut self) -> ReportWinnerFieldPathBuilder {
            self.path.push(TransactionKind::REPORT_WINNER_FIELD.name);
            ReportWinnerFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl AddValidator {
        pub const PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "pubkey_bytes",
            json_name: "pubkeyBytes",
            number: 1i32,
            message_fields: None,
        };
        pub const NETWORK_PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "network_pubkey_bytes",
            json_name: "networkPubkeyBytes",
            number: 2i32,
            message_fields: None,
        };
        pub const WORKER_PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "worker_pubkey_bytes",
            json_name: "workerPubkeyBytes",
            number: 3i32,
            message_fields: None,
        };
        pub const NET_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "net_address",
            json_name: "netAddress",
            number: 4i32,
            message_fields: None,
        };
        pub const P2P_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "p2p_address",
            json_name: "p2pAddress",
            number: 5i32,
            message_fields: None,
        };
        pub const PRIMARY_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "primary_address",
            json_name: "primaryAddress",
            number: 6i32,
            message_fields: None,
        };
        pub const ENCODER_VALIDATOR_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "encoder_validator_address",
            json_name: "encoderValidatorAddress",
            number: 7i32,
            message_fields: None,
        };
    }
    impl MessageFields for AddValidator {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::PUBKEY_BYTES_FIELD,
            Self::NETWORK_PUBKEY_BYTES_FIELD,
            Self::WORKER_PUBKEY_BYTES_FIELD,
            Self::NET_ADDRESS_FIELD,
            Self::P2P_ADDRESS_FIELD,
            Self::PRIMARY_ADDRESS_FIELD,
            Self::ENCODER_VALIDATOR_ADDRESS_FIELD,
        ];
    }
    impl AddValidator {
        pub fn path_builder() -> AddValidatorFieldPathBuilder {
            AddValidatorFieldPathBuilder::new()
        }
    }
    pub struct AddValidatorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl AddValidatorFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn pubkey_bytes(mut self) -> String {
            self.path.push(AddValidator::PUBKEY_BYTES_FIELD.name);
            self.finish()
        }
        pub fn network_pubkey_bytes(mut self) -> String {
            self.path.push(AddValidator::NETWORK_PUBKEY_BYTES_FIELD.name);
            self.finish()
        }
        pub fn worker_pubkey_bytes(mut self) -> String {
            self.path.push(AddValidator::WORKER_PUBKEY_BYTES_FIELD.name);
            self.finish()
        }
        pub fn net_address(mut self) -> String {
            self.path.push(AddValidator::NET_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn p2p_address(mut self) -> String {
            self.path.push(AddValidator::P2P_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn primary_address(mut self) -> String {
            self.path.push(AddValidator::PRIMARY_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn encoder_validator_address(mut self) -> String {
            self.path.push(AddValidator::ENCODER_VALIDATOR_ADDRESS_FIELD.name);
            self.finish()
        }
    }
    impl RemoveValidator {
        pub const PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "pubkey_bytes",
            json_name: "pubkeyBytes",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for RemoveValidator {
        const FIELDS: &'static [&'static MessageField] = &[Self::PUBKEY_BYTES_FIELD];
    }
    impl RemoveValidator {
        pub fn path_builder() -> RemoveValidatorFieldPathBuilder {
            RemoveValidatorFieldPathBuilder::new()
        }
    }
    pub struct RemoveValidatorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl RemoveValidatorFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn pubkey_bytes(mut self) -> String {
            self.path.push(RemoveValidator::PUBKEY_BYTES_FIELD.name);
            self.finish()
        }
    }
    impl ReportValidator {
        pub const REPORTEE_FIELD: &'static MessageField = &MessageField {
            name: "reportee",
            json_name: "reportee",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for ReportValidator {
        const FIELDS: &'static [&'static MessageField] = &[Self::REPORTEE_FIELD];
    }
    impl ReportValidator {
        pub fn path_builder() -> ReportValidatorFieldPathBuilder {
            ReportValidatorFieldPathBuilder::new()
        }
    }
    pub struct ReportValidatorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ReportValidatorFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn reportee(mut self) -> String {
            self.path.push(ReportValidator::REPORTEE_FIELD.name);
            self.finish()
        }
    }
    impl UndoReportValidator {
        pub const REPORTEE_FIELD: &'static MessageField = &MessageField {
            name: "reportee",
            json_name: "reportee",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for UndoReportValidator {
        const FIELDS: &'static [&'static MessageField] = &[Self::REPORTEE_FIELD];
    }
    impl UndoReportValidator {
        pub fn path_builder() -> UndoReportValidatorFieldPathBuilder {
            UndoReportValidatorFieldPathBuilder::new()
        }
    }
    pub struct UndoReportValidatorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl UndoReportValidatorFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn reportee(mut self) -> String {
            self.path.push(UndoReportValidator::REPORTEE_FIELD.name);
            self.finish()
        }
    }
    impl UpdateValidatorMetadata {
        pub const NEXT_EPOCH_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_network_address",
            json_name: "nextEpochNetworkAddress",
            number: 1i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_P2P_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_p2p_address",
            json_name: "nextEpochP2pAddress",
            number: 2i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_PRIMARY_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_primary_address",
            json_name: "nextEpochPrimaryAddress",
            number: 3i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_PROTOCOL_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_protocol_pubkey",
            json_name: "nextEpochProtocolPubkey",
            number: 4i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_WORKER_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_worker_pubkey",
            json_name: "nextEpochWorkerPubkey",
            number: 5i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_NETWORK_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_network_pubkey",
            json_name: "nextEpochNetworkPubkey",
            number: 6i32,
            message_fields: None,
        };
    }
    impl MessageFields for UpdateValidatorMetadata {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::NEXT_EPOCH_NETWORK_ADDRESS_FIELD,
            Self::NEXT_EPOCH_P2P_ADDRESS_FIELD,
            Self::NEXT_EPOCH_PRIMARY_ADDRESS_FIELD,
            Self::NEXT_EPOCH_PROTOCOL_PUBKEY_FIELD,
            Self::NEXT_EPOCH_WORKER_PUBKEY_FIELD,
            Self::NEXT_EPOCH_NETWORK_PUBKEY_FIELD,
        ];
    }
    impl UpdateValidatorMetadata {
        pub fn path_builder() -> UpdateValidatorMetadataFieldPathBuilder {
            UpdateValidatorMetadataFieldPathBuilder::new()
        }
    }
    pub struct UpdateValidatorMetadataFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl UpdateValidatorMetadataFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn next_epoch_network_address(mut self) -> String {
            self.path
                .push(UpdateValidatorMetadata::NEXT_EPOCH_NETWORK_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_p2p_address(mut self) -> String {
            self.path.push(UpdateValidatorMetadata::NEXT_EPOCH_P2P_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_primary_address(mut self) -> String {
            self.path
                .push(UpdateValidatorMetadata::NEXT_EPOCH_PRIMARY_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_protocol_pubkey(mut self) -> String {
            self.path
                .push(UpdateValidatorMetadata::NEXT_EPOCH_PROTOCOL_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_worker_pubkey(mut self) -> String {
            self.path.push(UpdateValidatorMetadata::NEXT_EPOCH_WORKER_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_network_pubkey(mut self) -> String {
            self.path
                .push(UpdateValidatorMetadata::NEXT_EPOCH_NETWORK_PUBKEY_FIELD.name);
            self.finish()
        }
    }
    impl SetCommissionRate {
        pub const NEW_RATE_FIELD: &'static MessageField = &MessageField {
            name: "new_rate",
            json_name: "newRate",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for SetCommissionRate {
        const FIELDS: &'static [&'static MessageField] = &[Self::NEW_RATE_FIELD];
    }
    impl SetCommissionRate {
        pub fn path_builder() -> SetCommissionRateFieldPathBuilder {
            SetCommissionRateFieldPathBuilder::new()
        }
    }
    pub struct SetCommissionRateFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SetCommissionRateFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn new_rate(mut self) -> String {
            self.path.push(SetCommissionRate::NEW_RATE_FIELD.name);
            self.finish()
        }
    }
    impl AddEncoder {
        pub const ENCODER_PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "encoder_pubkey_bytes",
            json_name: "encoderPubkeyBytes",
            number: 1i32,
            message_fields: None,
        };
        pub const NETWORK_PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "network_pubkey_bytes",
            json_name: "networkPubkeyBytes",
            number: 2i32,
            message_fields: None,
        };
        pub const INTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "internal_network_address",
            json_name: "internalNetworkAddress",
            number: 3i32,
            message_fields: None,
        };
        pub const EXTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "external_network_address",
            json_name: "externalNetworkAddress",
            number: 4i32,
            message_fields: None,
        };
        pub const OBJECT_SERVER_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "object_server_address",
            json_name: "objectServerAddress",
            number: 5i32,
            message_fields: None,
        };
    }
    impl MessageFields for AddEncoder {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ENCODER_PUBKEY_BYTES_FIELD,
            Self::NETWORK_PUBKEY_BYTES_FIELD,
            Self::INTERNAL_NETWORK_ADDRESS_FIELD,
            Self::EXTERNAL_NETWORK_ADDRESS_FIELD,
            Self::OBJECT_SERVER_ADDRESS_FIELD,
        ];
    }
    impl AddEncoder {
        pub fn path_builder() -> AddEncoderFieldPathBuilder {
            AddEncoderFieldPathBuilder::new()
        }
    }
    pub struct AddEncoderFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl AddEncoderFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn encoder_pubkey_bytes(mut self) -> String {
            self.path.push(AddEncoder::ENCODER_PUBKEY_BYTES_FIELD.name);
            self.finish()
        }
        pub fn network_pubkey_bytes(mut self) -> String {
            self.path.push(AddEncoder::NETWORK_PUBKEY_BYTES_FIELD.name);
            self.finish()
        }
        pub fn internal_network_address(mut self) -> String {
            self.path.push(AddEncoder::INTERNAL_NETWORK_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn external_network_address(mut self) -> String {
            self.path.push(AddEncoder::EXTERNAL_NETWORK_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn object_server_address(mut self) -> String {
            self.path.push(AddEncoder::OBJECT_SERVER_ADDRESS_FIELD.name);
            self.finish()
        }
    }
    impl RemoveEncoder {
        pub const ENCODER_PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "encoder_pubkey_bytes",
            json_name: "encoderPubkeyBytes",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for RemoveEncoder {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ENCODER_PUBKEY_BYTES_FIELD,
        ];
    }
    impl RemoveEncoder {
        pub fn path_builder() -> RemoveEncoderFieldPathBuilder {
            RemoveEncoderFieldPathBuilder::new()
        }
    }
    pub struct RemoveEncoderFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl RemoveEncoderFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn encoder_pubkey_bytes(mut self) -> String {
            self.path.push(RemoveEncoder::ENCODER_PUBKEY_BYTES_FIELD.name);
            self.finish()
        }
    }
    impl ReportEncoder {
        pub const REPORTEE_FIELD: &'static MessageField = &MessageField {
            name: "reportee",
            json_name: "reportee",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for ReportEncoder {
        const FIELDS: &'static [&'static MessageField] = &[Self::REPORTEE_FIELD];
    }
    impl ReportEncoder {
        pub fn path_builder() -> ReportEncoderFieldPathBuilder {
            ReportEncoderFieldPathBuilder::new()
        }
    }
    pub struct ReportEncoderFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ReportEncoderFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn reportee(mut self) -> String {
            self.path.push(ReportEncoder::REPORTEE_FIELD.name);
            self.finish()
        }
    }
    impl UndoReportEncoder {
        pub const REPORTEE_FIELD: &'static MessageField = &MessageField {
            name: "reportee",
            json_name: "reportee",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for UndoReportEncoder {
        const FIELDS: &'static [&'static MessageField] = &[Self::REPORTEE_FIELD];
    }
    impl UndoReportEncoder {
        pub fn path_builder() -> UndoReportEncoderFieldPathBuilder {
            UndoReportEncoderFieldPathBuilder::new()
        }
    }
    pub struct UndoReportEncoderFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl UndoReportEncoderFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn reportee(mut self) -> String {
            self.path.push(UndoReportEncoder::REPORTEE_FIELD.name);
            self.finish()
        }
    }
    impl UpdateEncoderMetadata {
        pub const NEXT_EPOCH_EXTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_external_network_address",
            json_name: "nextEpochExternalNetworkAddress",
            number: 1i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_INTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_internal_network_address",
            json_name: "nextEpochInternalNetworkAddress",
            number: 2i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_NETWORK_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_network_pubkey",
            json_name: "nextEpochNetworkPubkey",
            number: 3i32,
            message_fields: None,
        };
        pub const NEXT_EPOCH_OBJECT_SERVER_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_object_server_address",
            json_name: "nextEpochObjectServerAddress",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for UpdateEncoderMetadata {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::NEXT_EPOCH_EXTERNAL_NETWORK_ADDRESS_FIELD,
            Self::NEXT_EPOCH_INTERNAL_NETWORK_ADDRESS_FIELD,
            Self::NEXT_EPOCH_NETWORK_PUBKEY_FIELD,
            Self::NEXT_EPOCH_OBJECT_SERVER_ADDRESS_FIELD,
        ];
    }
    impl UpdateEncoderMetadata {
        pub fn path_builder() -> UpdateEncoderMetadataFieldPathBuilder {
            UpdateEncoderMetadataFieldPathBuilder::new()
        }
    }
    pub struct UpdateEncoderMetadataFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl UpdateEncoderMetadataFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn next_epoch_external_network_address(mut self) -> String {
            self.path
                .push(
                    UpdateEncoderMetadata::NEXT_EPOCH_EXTERNAL_NETWORK_ADDRESS_FIELD.name,
                );
            self.finish()
        }
        pub fn next_epoch_internal_network_address(mut self) -> String {
            self.path
                .push(
                    UpdateEncoderMetadata::NEXT_EPOCH_INTERNAL_NETWORK_ADDRESS_FIELD.name,
                );
            self.finish()
        }
        pub fn next_epoch_network_pubkey(mut self) -> String {
            self.path.push(UpdateEncoderMetadata::NEXT_EPOCH_NETWORK_PUBKEY_FIELD.name);
            self.finish()
        }
        pub fn next_epoch_object_server_address(mut self) -> String {
            self.path
                .push(
                    UpdateEncoderMetadata::NEXT_EPOCH_OBJECT_SERVER_ADDRESS_FIELD.name,
                );
            self.finish()
        }
    }
    impl SetEncoderCommissionRate {
        pub const NEW_RATE_FIELD: &'static MessageField = &MessageField {
            name: "new_rate",
            json_name: "newRate",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for SetEncoderCommissionRate {
        const FIELDS: &'static [&'static MessageField] = &[Self::NEW_RATE_FIELD];
    }
    impl SetEncoderCommissionRate {
        pub fn path_builder() -> SetEncoderCommissionRateFieldPathBuilder {
            SetEncoderCommissionRateFieldPathBuilder::new()
        }
    }
    pub struct SetEncoderCommissionRateFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SetEncoderCommissionRateFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn new_rate(mut self) -> String {
            self.path.push(SetEncoderCommissionRate::NEW_RATE_FIELD.name);
            self.finish()
        }
    }
    impl SetEncoderBytePrice {
        pub const NEW_PRICE_FIELD: &'static MessageField = &MessageField {
            name: "new_price",
            json_name: "newPrice",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for SetEncoderBytePrice {
        const FIELDS: &'static [&'static MessageField] = &[Self::NEW_PRICE_FIELD];
    }
    impl SetEncoderBytePrice {
        pub fn path_builder() -> SetEncoderBytePriceFieldPathBuilder {
            SetEncoderBytePriceFieldPathBuilder::new()
        }
    }
    pub struct SetEncoderBytePriceFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SetEncoderBytePriceFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn new_price(mut self) -> String {
            self.path.push(SetEncoderBytePrice::NEW_PRICE_FIELD.name);
            self.finish()
        }
    }
    impl TransferCoin {
        pub const COIN_FIELD: &'static MessageField = &MessageField {
            name: "coin",
            json_name: "coin",
            number: 1i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
        pub const AMOUNT_FIELD: &'static MessageField = &MessageField {
            name: "amount",
            json_name: "amount",
            number: 2i32,
            message_fields: None,
        };
        pub const RECIPIENT_FIELD: &'static MessageField = &MessageField {
            name: "recipient",
            json_name: "recipient",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for TransferCoin {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::COIN_FIELD,
            Self::AMOUNT_FIELD,
            Self::RECIPIENT_FIELD,
        ];
    }
    impl TransferCoin {
        pub fn path_builder() -> TransferCoinFieldPathBuilder {
            TransferCoinFieldPathBuilder::new()
        }
    }
    pub struct TransferCoinFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TransferCoinFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn coin(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(TransferCoin::COIN_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
        pub fn amount(mut self) -> String {
            self.path.push(TransferCoin::AMOUNT_FIELD.name);
            self.finish()
        }
        pub fn recipient(mut self) -> String {
            self.path.push(TransferCoin::RECIPIENT_FIELD.name);
            self.finish()
        }
    }
    impl PayCoins {
        pub const COINS_FIELD: &'static MessageField = &MessageField {
            name: "coins",
            json_name: "coins",
            number: 1i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
        pub const AMOUNTS_FIELD: &'static MessageField = &MessageField {
            name: "amounts",
            json_name: "amounts",
            number: 2i32,
            message_fields: None,
        };
        pub const RECIPIENTS_FIELD: &'static MessageField = &MessageField {
            name: "recipients",
            json_name: "recipients",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for PayCoins {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::COINS_FIELD,
            Self::AMOUNTS_FIELD,
            Self::RECIPIENTS_FIELD,
        ];
    }
    impl PayCoins {
        pub fn path_builder() -> PayCoinsFieldPathBuilder {
            PayCoinsFieldPathBuilder::new()
        }
    }
    pub struct PayCoinsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl PayCoinsFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn coins(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(PayCoins::COINS_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
        pub fn amounts(mut self) -> String {
            self.path.push(PayCoins::AMOUNTS_FIELD.name);
            self.finish()
        }
        pub fn recipients(mut self) -> String {
            self.path.push(PayCoins::RECIPIENTS_FIELD.name);
            self.finish()
        }
    }
    impl TransferObjects {
        pub const OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "objects",
            json_name: "objects",
            number: 1i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
        pub const RECIPIENT_FIELD: &'static MessageField = &MessageField {
            name: "recipient",
            json_name: "recipient",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for TransferObjects {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECTS_FIELD,
            Self::RECIPIENT_FIELD,
        ];
    }
    impl TransferObjects {
        pub fn path_builder() -> TransferObjectsFieldPathBuilder {
            TransferObjectsFieldPathBuilder::new()
        }
    }
    pub struct TransferObjectsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TransferObjectsFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn objects(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(TransferObjects::OBJECTS_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
        pub fn recipient(mut self) -> String {
            self.path.push(TransferObjects::RECIPIENT_FIELD.name);
            self.finish()
        }
    }
    impl AddStake {
        pub const ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "address",
            json_name: "address",
            number: 1i32,
            message_fields: None,
        };
        pub const COIN_REF_FIELD: &'static MessageField = &MessageField {
            name: "coin_ref",
            json_name: "coinRef",
            number: 2i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
        pub const AMOUNT_FIELD: &'static MessageField = &MessageField {
            name: "amount",
            json_name: "amount",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for AddStake {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ADDRESS_FIELD,
            Self::COIN_REF_FIELD,
            Self::AMOUNT_FIELD,
        ];
    }
    impl AddStake {
        pub fn path_builder() -> AddStakeFieldPathBuilder {
            AddStakeFieldPathBuilder::new()
        }
    }
    pub struct AddStakeFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl AddStakeFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn address(mut self) -> String {
            self.path.push(AddStake::ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn coin_ref(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(AddStake::COIN_REF_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
        pub fn amount(mut self) -> String {
            self.path.push(AddStake::AMOUNT_FIELD.name);
            self.finish()
        }
    }
    impl AddStakeToEncoder {
        pub const ENCODER_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "encoder_address",
            json_name: "encoderAddress",
            number: 1i32,
            message_fields: None,
        };
        pub const COIN_REF_FIELD: &'static MessageField = &MessageField {
            name: "coin_ref",
            json_name: "coinRef",
            number: 2i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
        pub const AMOUNT_FIELD: &'static MessageField = &MessageField {
            name: "amount",
            json_name: "amount",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for AddStakeToEncoder {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ENCODER_ADDRESS_FIELD,
            Self::COIN_REF_FIELD,
            Self::AMOUNT_FIELD,
        ];
    }
    impl AddStakeToEncoder {
        pub fn path_builder() -> AddStakeToEncoderFieldPathBuilder {
            AddStakeToEncoderFieldPathBuilder::new()
        }
    }
    pub struct AddStakeToEncoderFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl AddStakeToEncoderFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn encoder_address(mut self) -> String {
            self.path.push(AddStakeToEncoder::ENCODER_ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn coin_ref(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(AddStakeToEncoder::COIN_REF_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
        pub fn amount(mut self) -> String {
            self.path.push(AddStakeToEncoder::AMOUNT_FIELD.name);
            self.finish()
        }
    }
    impl WithdrawStake {
        pub const STAKED_SOMA_FIELD: &'static MessageField = &MessageField {
            name: "staked_soma",
            json_name: "stakedSoma",
            number: 1i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
    }
    impl MessageFields for WithdrawStake {
        const FIELDS: &'static [&'static MessageField] = &[Self::STAKED_SOMA_FIELD];
    }
    impl WithdrawStake {
        pub fn path_builder() -> WithdrawStakeFieldPathBuilder {
            WithdrawStakeFieldPathBuilder::new()
        }
    }
    pub struct WithdrawStakeFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl WithdrawStakeFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn staked_soma(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(WithdrawStake::STAKED_SOMA_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl EmbedData {
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 1i32,
            message_fields: None,
        };
        pub const DATA_SIZE_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "data_size_bytes",
            json_name: "dataSizeBytes",
            number: 2i32,
            message_fields: None,
        };
        pub const COIN_REF_FIELD: &'static MessageField = &MessageField {
            name: "coin_ref",
            json_name: "coinRef",
            number: 3i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
    }
    impl MessageFields for EmbedData {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::DIGEST_FIELD,
            Self::DATA_SIZE_BYTES_FIELD,
            Self::COIN_REF_FIELD,
        ];
    }
    impl EmbedData {
        pub fn path_builder() -> EmbedDataFieldPathBuilder {
            EmbedDataFieldPathBuilder::new()
        }
    }
    pub struct EmbedDataFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl EmbedDataFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn digest(mut self) -> String {
            self.path.push(EmbedData::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn data_size_bytes(mut self) -> String {
            self.path.push(EmbedData::DATA_SIZE_BYTES_FIELD.name);
            self.finish()
        }
        pub fn coin_ref(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(EmbedData::COIN_REF_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ClaimEscrow {
        pub const SHARD_INPUT_REF_FIELD: &'static MessageField = &MessageField {
            name: "shard_input_ref",
            json_name: "shardInputRef",
            number: 1i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
    }
    impl MessageFields for ClaimEscrow {
        const FIELDS: &'static [&'static MessageField] = &[Self::SHARD_INPUT_REF_FIELD];
    }
    impl ClaimEscrow {
        pub fn path_builder() -> ClaimEscrowFieldPathBuilder {
            ClaimEscrowFieldPathBuilder::new()
        }
    }
    pub struct ClaimEscrowFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ClaimEscrowFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn shard_input_ref(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(ClaimEscrow::SHARD_INPUT_REF_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ReportWinner {
        pub const SHARD_INPUT_REF_FIELD: &'static MessageField = &MessageField {
            name: "shard_input_ref",
            json_name: "shardInputRef",
            number: 1i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
        pub const SIGNED_REPORT_FIELD: &'static MessageField = &MessageField {
            name: "signed_report",
            json_name: "signedReport",
            number: 2i32,
            message_fields: None,
        };
        pub const ENCODER_AGGREGATE_SIGNATURE_FIELD: &'static MessageField = &MessageField {
            name: "encoder_aggregate_signature",
            json_name: "encoderAggregateSignature",
            number: 3i32,
            message_fields: None,
        };
        pub const SIGNERS_FIELD: &'static MessageField = &MessageField {
            name: "signers",
            json_name: "signers",
            number: 4i32,
            message_fields: None,
        };
        pub const SHARD_AUTH_TOKEN_FIELD: &'static MessageField = &MessageField {
            name: "shard_auth_token",
            json_name: "shardAuthToken",
            number: 5i32,
            message_fields: None,
        };
    }
    impl MessageFields for ReportWinner {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SHARD_INPUT_REF_FIELD,
            Self::SIGNED_REPORT_FIELD,
            Self::ENCODER_AGGREGATE_SIGNATURE_FIELD,
            Self::SIGNERS_FIELD,
            Self::SHARD_AUTH_TOKEN_FIELD,
        ];
    }
    impl ReportWinner {
        pub fn path_builder() -> ReportWinnerFieldPathBuilder {
            ReportWinnerFieldPathBuilder::new()
        }
    }
    pub struct ReportWinnerFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ReportWinnerFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn shard_input_ref(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(ReportWinner::SHARD_INPUT_REF_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
        pub fn signed_report(mut self) -> String {
            self.path.push(ReportWinner::SIGNED_REPORT_FIELD.name);
            self.finish()
        }
        pub fn encoder_aggregate_signature(mut self) -> String {
            self.path.push(ReportWinner::ENCODER_AGGREGATE_SIGNATURE_FIELD.name);
            self.finish()
        }
        pub fn signers(mut self) -> String {
            self.path.push(ReportWinner::SIGNERS_FIELD.name);
            self.finish()
        }
        pub fn shard_auth_token(mut self) -> String {
            self.path.push(ReportWinner::SHARD_AUTH_TOKEN_FIELD.name);
            self.finish()
        }
    }
    impl ChangeEpoch {
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 1i32,
            message_fields: None,
        };
        pub const EPOCH_START_TIMESTAMP_FIELD: &'static MessageField = &MessageField {
            name: "epoch_start_timestamp",
            json_name: "epochStartTimestamp",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for ChangeEpoch {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_FIELD,
            Self::EPOCH_START_TIMESTAMP_FIELD,
        ];
    }
    impl ChangeEpoch {
        pub fn path_builder() -> ChangeEpochFieldPathBuilder {
            ChangeEpochFieldPathBuilder::new()
        }
    }
    pub struct ChangeEpochFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ChangeEpochFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn epoch(mut self) -> String {
            self.path.push(ChangeEpoch::EPOCH_FIELD.name);
            self.finish()
        }
        pub fn epoch_start_timestamp(mut self) -> String {
            self.path.push(ChangeEpoch::EPOCH_START_TIMESTAMP_FIELD.name);
            self.finish()
        }
    }
    impl GenesisTransaction {
        pub const OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "objects",
            json_name: "objects",
            number: 1i32,
            message_fields: Some(Object::FIELDS),
        };
    }
    impl MessageFields for GenesisTransaction {
        const FIELDS: &'static [&'static MessageField] = &[Self::OBJECTS_FIELD];
    }
    impl GenesisTransaction {
        pub fn path_builder() -> GenesisTransactionFieldPathBuilder {
            GenesisTransactionFieldPathBuilder::new()
        }
    }
    pub struct GenesisTransactionFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GenesisTransactionFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn objects(mut self) -> ObjectFieldPathBuilder {
            self.path.push(GenesisTransaction::OBJECTS_FIELD.name);
            ObjectFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ConsensusCommitPrologue {
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 1i32,
            message_fields: None,
        };
        pub const ROUND_FIELD: &'static MessageField = &MessageField {
            name: "round",
            json_name: "round",
            number: 2i32,
            message_fields: None,
        };
        pub const COMMIT_TIMESTAMP_FIELD: &'static MessageField = &MessageField {
            name: "commit_timestamp",
            json_name: "commitTimestamp",
            number: 3i32,
            message_fields: None,
        };
        pub const CONSENSUS_COMMIT_DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "consensus_commit_digest",
            json_name: "consensusCommitDigest",
            number: 4i32,
            message_fields: None,
        };
        pub const SUB_DAG_INDEX_FIELD: &'static MessageField = &MessageField {
            name: "sub_dag_index",
            json_name: "subDagIndex",
            number: 5i32,
            message_fields: None,
        };
    }
    impl MessageFields for ConsensusCommitPrologue {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_FIELD,
            Self::ROUND_FIELD,
            Self::COMMIT_TIMESTAMP_FIELD,
            Self::CONSENSUS_COMMIT_DIGEST_FIELD,
            Self::SUB_DAG_INDEX_FIELD,
        ];
    }
    impl ConsensusCommitPrologue {
        pub fn path_builder() -> ConsensusCommitPrologueFieldPathBuilder {
            ConsensusCommitPrologueFieldPathBuilder::new()
        }
    }
    pub struct ConsensusCommitPrologueFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ConsensusCommitPrologueFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn epoch(mut self) -> String {
            self.path.push(ConsensusCommitPrologue::EPOCH_FIELD.name);
            self.finish()
        }
        pub fn round(mut self) -> String {
            self.path.push(ConsensusCommitPrologue::ROUND_FIELD.name);
            self.finish()
        }
        pub fn commit_timestamp(mut self) -> String {
            self.path.push(ConsensusCommitPrologue::COMMIT_TIMESTAMP_FIELD.name);
            self.finish()
        }
        pub fn consensus_commit_digest(mut self) -> String {
            self.path.push(ConsensusCommitPrologue::CONSENSUS_COMMIT_DIGEST_FIELD.name);
            self.finish()
        }
        pub fn sub_dag_index(mut self) -> String {
            self.path.push(ConsensusCommitPrologue::SUB_DAG_INDEX_FIELD.name);
            self.finish()
        }
    }
    impl ExecuteTransactionRequest {
        pub const TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "transaction",
            json_name: "transaction",
            number: 1i32,
            message_fields: Some(Transaction::FIELDS),
        };
        pub const SIGNATURES_FIELD: &'static MessageField = &MessageField {
            name: "signatures",
            json_name: "signatures",
            number: 2i32,
            message_fields: Some(UserSignature::FIELDS),
        };
        pub const READ_MASK_FIELD: &'static MessageField = &MessageField {
            name: "read_mask",
            json_name: "readMask",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for ExecuteTransactionRequest {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::TRANSACTION_FIELD,
            Self::SIGNATURES_FIELD,
            Self::READ_MASK_FIELD,
        ];
    }
    impl ExecuteTransactionRequest {
        pub fn path_builder() -> ExecuteTransactionRequestFieldPathBuilder {
            ExecuteTransactionRequestFieldPathBuilder::new()
        }
    }
    pub struct ExecuteTransactionRequestFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ExecuteTransactionRequestFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn transaction(mut self) -> TransactionFieldPathBuilder {
            self.path.push(ExecuteTransactionRequest::TRANSACTION_FIELD.name);
            TransactionFieldPathBuilder::new_with_base(self.path)
        }
        pub fn signatures(mut self) -> UserSignatureFieldPathBuilder {
            self.path.push(ExecuteTransactionRequest::SIGNATURES_FIELD.name);
            UserSignatureFieldPathBuilder::new_with_base(self.path)
        }
        pub fn read_mask(mut self) -> String {
            self.path.push(ExecuteTransactionRequest::READ_MASK_FIELD.name);
            self.finish()
        }
    }
    impl ExecuteTransactionResponse {
        pub const FINALITY_FIELD: &'static MessageField = &MessageField {
            name: "finality",
            json_name: "finality",
            number: 1i32,
            message_fields: Some(TransactionFinality::FIELDS),
        };
        pub const TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "transaction",
            json_name: "transaction",
            number: 2i32,
            message_fields: Some(ExecutedTransaction::FIELDS),
        };
    }
    impl MessageFields for ExecuteTransactionResponse {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::FINALITY_FIELD,
            Self::TRANSACTION_FIELD,
        ];
    }
    impl ExecuteTransactionResponse {
        pub fn path_builder() -> ExecuteTransactionResponseFieldPathBuilder {
            ExecuteTransactionResponseFieldPathBuilder::new()
        }
    }
    pub struct ExecuteTransactionResponseFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ExecuteTransactionResponseFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn finality(mut self) -> TransactionFinalityFieldPathBuilder {
            self.path.push(ExecuteTransactionResponse::FINALITY_FIELD.name);
            TransactionFinalityFieldPathBuilder::new_with_base(self.path)
        }
        pub fn transaction(mut self) -> ExecutedTransactionFieldPathBuilder {
            self.path.push(ExecuteTransactionResponse::TRANSACTION_FIELD.name);
            ExecutedTransactionFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl TransactionFinality {
        pub const CERTIFIED_FIELD: &'static MessageField = &MessageField {
            name: "certified",
            json_name: "certified",
            number: 1i32,
            message_fields: Some(ValidatorAggregatedSignature::FIELDS),
        };
    }
    impl MessageFields for TransactionFinality {
        const FIELDS: &'static [&'static MessageField] = &[Self::CERTIFIED_FIELD];
    }
    impl TransactionFinality {
        pub fn path_builder() -> TransactionFinalityFieldPathBuilder {
            TransactionFinalityFieldPathBuilder::new()
        }
    }
    pub struct TransactionFinalityFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TransactionFinalityFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn certified(mut self) -> ValidatorAggregatedSignatureFieldPathBuilder {
            self.path.push(TransactionFinality::CERTIFIED_FIELD.name);
            ValidatorAggregatedSignatureFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl TransactionFee {
        pub const BASE_FEE_FIELD: &'static MessageField = &MessageField {
            name: "base_fee",
            json_name: "baseFee",
            number: 1i32,
            message_fields: None,
        };
        pub const OPERATION_FEE_FIELD: &'static MessageField = &MessageField {
            name: "operation_fee",
            json_name: "operationFee",
            number: 2i32,
            message_fields: None,
        };
        pub const VALUE_FEE_FIELD: &'static MessageField = &MessageField {
            name: "value_fee",
            json_name: "valueFee",
            number: 3i32,
            message_fields: None,
        };
        pub const TOTAL_FEE_FIELD: &'static MessageField = &MessageField {
            name: "total_fee",
            json_name: "totalFee",
            number: 4i32,
            message_fields: None,
        };
        pub const GAS_OBJECT_REF_FIELD: &'static MessageField = &MessageField {
            name: "gas_object_ref",
            json_name: "gasObjectRef",
            number: 5i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
    }
    impl MessageFields for TransactionFee {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BASE_FEE_FIELD,
            Self::OPERATION_FEE_FIELD,
            Self::VALUE_FEE_FIELD,
            Self::TOTAL_FEE_FIELD,
            Self::GAS_OBJECT_REF_FIELD,
        ];
    }
    impl TransactionFee {
        pub fn path_builder() -> TransactionFeeFieldPathBuilder {
            TransactionFeeFieldPathBuilder::new()
        }
    }
    pub struct TransactionFeeFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TransactionFeeFieldPathBuilder {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            Self { path: Default::default() }
        }
        #[doc(hidden)]
        pub fn new_with_base(base: Vec<&'static str>) -> Self {
            Self { path: base }
        }
        pub fn finish(self) -> String {
            self.path.join(".")
        }
        pub fn base_fee(mut self) -> String {
            self.path.push(TransactionFee::BASE_FEE_FIELD.name);
            self.finish()
        }
        pub fn operation_fee(mut self) -> String {
            self.path.push(TransactionFee::OPERATION_FEE_FIELD.name);
            self.finish()
        }
        pub fn value_fee(mut self) -> String {
            self.path.push(TransactionFee::VALUE_FEE_FIELD.name);
            self.finish()
        }
        pub fn total_fee(mut self) -> String {
            self.path.push(TransactionFee::TOTAL_FEE_FIELD.name);
            self.finish()
        }
        pub fn gas_object_ref(mut self) -> ObjectReferenceFieldPathBuilder {
            self.path.push(TransactionFee::GAS_OBJECT_REF_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
    }
}
