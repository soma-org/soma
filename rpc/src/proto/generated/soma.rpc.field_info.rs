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
    impl Bcs {
        pub const NAME_FIELD: &'static MessageField = &MessageField {
            name: "name",
            json_name: "name",
            number: 1i32,
            message_fields: None,
        };
        pub const VALUE_FIELD: &'static MessageField = &MessageField {
            name: "value",
            json_name: "value",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for Bcs {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::NAME_FIELD,
            Self::VALUE_FIELD,
        ];
    }
    impl Bcs {
        pub fn path_builder() -> BcsFieldPathBuilder {
            BcsFieldPathBuilder::new()
        }
    }
    pub struct BcsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl BcsFieldPathBuilder {
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
        pub fn name(mut self) -> String {
            self.path.push(Bcs::NAME_FIELD.name);
            self.finish()
        }
        pub fn value(mut self) -> String {
            self.path.push(Bcs::VALUE_FIELD.name);
            self.finish()
        }
    }
    impl TransactionEffects {
        pub const BCS_FIELD: &'static MessageField = &MessageField {
            name: "bcs",
            json_name: "bcs",
            number: 1i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 2i32,
            message_fields: None,
        };
        pub const STATUS_FIELD: &'static MessageField = &MessageField {
            name: "status",
            json_name: "status",
            number: 3i32,
            message_fields: Some(ExecutionStatus::FIELDS),
        };
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 4i32,
            message_fields: None,
        };
        pub const FEE_FIELD: &'static MessageField = &MessageField {
            name: "fee",
            json_name: "fee",
            number: 5i32,
            message_fields: Some(TransactionFee::FIELDS),
        };
        pub const TRANSACTION_DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "transaction_digest",
            json_name: "transactionDigest",
            number: 6i32,
            message_fields: None,
        };
        pub const GAS_OBJECT_FIELD: &'static MessageField = &MessageField {
            name: "gas_object",
            json_name: "gasObject",
            number: 7i32,
            message_fields: Some(ChangedObject::FIELDS),
        };
        pub const DEPENDENCIES_FIELD: &'static MessageField = &MessageField {
            name: "dependencies",
            json_name: "dependencies",
            number: 8i32,
            message_fields: None,
        };
        pub const LAMPORT_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "lamport_version",
            json_name: "lamportVersion",
            number: 9i32,
            message_fields: None,
        };
        pub const CHANGED_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "changed_objects",
            json_name: "changedObjects",
            number: 10i32,
            message_fields: Some(ChangedObject::FIELDS),
        };
        pub const UNCHANGED_SHARED_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "unchanged_shared_objects",
            json_name: "unchangedSharedObjects",
            number: 11i32,
            message_fields: Some(UnchangedSharedObject::FIELDS),
        };
    }
    impl MessageFields for TransactionEffects {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BCS_FIELD,
            Self::DIGEST_FIELD,
            Self::STATUS_FIELD,
            Self::EPOCH_FIELD,
            Self::FEE_FIELD,
            Self::TRANSACTION_DIGEST_FIELD,
            Self::GAS_OBJECT_FIELD,
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
        pub fn bcs(mut self) -> BcsFieldPathBuilder {
            self.path.push(TransactionEffects::BCS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn digest(mut self) -> String {
            self.path.push(TransactionEffects::DIGEST_FIELD.name);
            self.finish()
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
        pub fn gas_object(mut self) -> ChangedObjectFieldPathBuilder {
            self.path.push(TransactionEffects::GAS_OBJECT_FIELD.name);
            ChangedObjectFieldPathBuilder::new_with_base(self.path)
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
    impl Object {
        pub const BCS_FIELD: &'static MessageField = &MessageField {
            name: "bcs",
            json_name: "bcs",
            number: 1i32,
            message_fields: Some(Bcs::FIELDS),
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
        pub const OWNER_FIELD: &'static MessageField = &MessageField {
            name: "owner",
            json_name: "owner",
            number: 5i32,
            message_fields: Some(Owner::FIELDS),
        };
        pub const OBJECT_TYPE_FIELD: &'static MessageField = &MessageField {
            name: "object_type",
            json_name: "objectType",
            number: 6i32,
            message_fields: None,
        };
        pub const CONTENTS_FIELD: &'static MessageField = &MessageField {
            name: "contents",
            json_name: "contents",
            number: 7i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const PREVIOUS_TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "previous_transaction",
            json_name: "previousTransaction",
            number: 8i32,
            message_fields: None,
        };
        pub const JSON_FIELD: &'static MessageField = &MessageField {
            name: "json",
            json_name: "json",
            number: 100i32,
            message_fields: None,
        };
        pub const BALANCE_FIELD: &'static MessageField = &MessageField {
            name: "balance",
            json_name: "balance",
            number: 101i32,
            message_fields: None,
        };
    }
    impl MessageFields for Object {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BCS_FIELD,
            Self::OBJECT_ID_FIELD,
            Self::VERSION_FIELD,
            Self::DIGEST_FIELD,
            Self::OWNER_FIELD,
            Self::OBJECT_TYPE_FIELD,
            Self::CONTENTS_FIELD,
            Self::PREVIOUS_TRANSACTION_FIELD,
            Self::JSON_FIELD,
            Self::BALANCE_FIELD,
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
        pub fn bcs(mut self) -> BcsFieldPathBuilder {
            self.path.push(Object::BCS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
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
        pub fn contents(mut self) -> BcsFieldPathBuilder {
            self.path.push(Object::CONTENTS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn previous_transaction(mut self) -> String {
            self.path.push(Object::PREVIOUS_TRANSACTION_FIELD.name);
            self.finish()
        }
        pub fn json(mut self) -> String {
            self.path.push(Object::JSON_FIELD.name);
            self.finish()
        }
        pub fn balance(mut self) -> String {
            self.path.push(Object::BALANCE_FIELD.name);
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
    impl UserSignature {
        pub const BCS_FIELD: &'static MessageField = &MessageField {
            name: "bcs",
            json_name: "bcs",
            number: 1i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const SCHEME_FIELD: &'static MessageField = &MessageField {
            name: "scheme",
            json_name: "scheme",
            number: 2i32,
            message_fields: None,
        };
        pub const SIMPLE_FIELD: &'static MessageField = &MessageField {
            name: "simple",
            json_name: "simple",
            number: 3i32,
            message_fields: Some(SimpleSignature::FIELDS),
        };
    }
    impl MessageFields for UserSignature {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BCS_FIELD,
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
        pub fn bcs(mut self) -> BcsFieldPathBuilder {
            self.path.push(UserSignature::BCS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
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
        pub const PUBLIC_KEY_FIELD: &'static MessageField = &MessageField {
            name: "public_key",
            json_name: "publicKey",
            number: 1i32,
            message_fields: None,
        };
        pub const WEIGHT_FIELD: &'static MessageField = &MessageField {
            name: "weight",
            json_name: "weight",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for ValidatorCommitteeMember {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::PUBLIC_KEY_FIELD,
            Self::WEIGHT_FIELD,
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
        pub fn public_key(mut self) -> String {
            self.path.push(ValidatorCommitteeMember::PUBLIC_KEY_FIELD.name);
            self.finish()
        }
        pub fn weight(mut self) -> String {
            self.path.push(ValidatorCommitteeMember::WEIGHT_FIELD.name);
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
    impl Transaction {
        pub const BCS_FIELD: &'static MessageField = &MessageField {
            name: "bcs",
            json_name: "bcs",
            number: 1i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 2i32,
            message_fields: None,
        };
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 4i32,
            message_fields: Some(TransactionKind::FIELDS),
        };
        pub const SENDER_FIELD: &'static MessageField = &MessageField {
            name: "sender",
            json_name: "sender",
            number: 5i32,
            message_fields: None,
        };
        pub const GAS_PAYMENT_FIELD: &'static MessageField = &MessageField {
            name: "gas_payment",
            json_name: "gasPayment",
            number: 6i32,
            message_fields: Some(GasPayment::FIELDS),
        };
    }
    impl MessageFields for Transaction {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BCS_FIELD,
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
        pub fn bcs(mut self) -> BcsFieldPathBuilder {
            self.path.push(Transaction::BCS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
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
        pub fn gas_payment(mut self) -> GasPaymentFieldPathBuilder {
            self.path.push(Transaction::GAS_PAYMENT_FIELD.name);
            GasPaymentFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl GasPayment {
        pub const OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "objects",
            json_name: "objects",
            number: 1i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
        pub const OWNER_FIELD: &'static MessageField = &MessageField {
            name: "owner",
            json_name: "owner",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for GasPayment {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECTS_FIELD,
            Self::OWNER_FIELD,
        ];
    }
    impl GasPayment {
        pub fn path_builder() -> GasPaymentFieldPathBuilder {
            GasPaymentFieldPathBuilder::new()
        }
    }
    pub struct GasPaymentFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GasPaymentFieldPathBuilder {
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
            self.path.push(GasPayment::OBJECTS_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
        pub fn owner(mut self) -> String {
            self.path.push(GasPayment::OWNER_FIELD.name);
            self.finish()
        }
    }
    impl TransactionKind {
        pub const ADD_VALIDATOR_FIELD: &'static MessageField = &MessageField {
            name: "add_validator",
            json_name: "addValidator",
            number: 1i32,
            message_fields: Some(AddValidator::FIELDS),
        };
        pub const REMOVE_VALIDATOR_FIELD: &'static MessageField = &MessageField {
            name: "remove_validator",
            json_name: "removeValidator",
            number: 2i32,
            message_fields: Some(RemoveValidator::FIELDS),
        };
        pub const REPORT_VALIDATOR_FIELD: &'static MessageField = &MessageField {
            name: "report_validator",
            json_name: "reportValidator",
            number: 3i32,
            message_fields: Some(ReportValidator::FIELDS),
        };
        pub const UNDO_REPORT_VALIDATOR_FIELD: &'static MessageField = &MessageField {
            name: "undo_report_validator",
            json_name: "undoReportValidator",
            number: 4i32,
            message_fields: Some(UndoReportValidator::FIELDS),
        };
        pub const UNDO_VALIDATOR_METADATA_FIELD: &'static MessageField = &MessageField {
            name: "undo_validator_metadata",
            json_name: "undoValidatorMetadata",
            number: 5i32,
            message_fields: Some(UpdateValidatorMetadata::FIELDS),
        };
        pub const SET_COMMISSION_RATE_FIELD: &'static MessageField = &MessageField {
            name: "set_commission_rate",
            json_name: "setCommissionRate",
            number: 6i32,
            message_fields: Some(SetCommissionRate::FIELDS),
        };
        pub const ADD_ENCODER_FIELD: &'static MessageField = &MessageField {
            name: "add_encoder",
            json_name: "addEncoder",
            number: 7i32,
            message_fields: Some(AddEncoder::FIELDS),
        };
        pub const REMOVE_ENCODER_FIELD: &'static MessageField = &MessageField {
            name: "remove_encoder",
            json_name: "removeEncoder",
            number: 8i32,
            message_fields: Some(RemoveEncoder::FIELDS),
        };
        pub const REPORT_ENCODER_FIELD: &'static MessageField = &MessageField {
            name: "report_encoder",
            json_name: "reportEncoder",
            number: 9i32,
            message_fields: Some(ReportEncoder::FIELDS),
        };
        pub const UNDO_REPORT_ENCODER_FIELD: &'static MessageField = &MessageField {
            name: "undo_report_encoder",
            json_name: "undoReportEncoder",
            number: 10i32,
            message_fields: Some(UndoReportEncoder::FIELDS),
        };
        pub const UPDATE_ENCODER_METADATA_FIELD: &'static MessageField = &MessageField {
            name: "update_encoder_metadata",
            json_name: "updateEncoderMetadata",
            number: 11i32,
            message_fields: Some(UpdateEncoderMetadata::FIELDS),
        };
        pub const SET_ENCODER_COMMISSION_RATE_FIELD: &'static MessageField = &MessageField {
            name: "set_encoder_commission_rate",
            json_name: "setEncoderCommissionRate",
            number: 12i32,
            message_fields: Some(SetEncoderCommissionRate::FIELDS),
        };
        pub const SET_ENCODER_BYTE_PRICE_FIELD: &'static MessageField = &MessageField {
            name: "set_encoder_byte_price",
            json_name: "setEncoderBytePrice",
            number: 13i32,
            message_fields: Some(SetEncoderBytePrice::FIELDS),
        };
        pub const TRANSFER_COIN_FIELD: &'static MessageField = &MessageField {
            name: "transfer_coin",
            json_name: "transferCoin",
            number: 14i32,
            message_fields: Some(TransferCoin::FIELDS),
        };
        pub const PAY_COINS_FIELD: &'static MessageField = &MessageField {
            name: "pay_coins",
            json_name: "payCoins",
            number: 15i32,
            message_fields: Some(PayCoins::FIELDS),
        };
        pub const TRANSFER_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "transfer_objects",
            json_name: "transferObjects",
            number: 16i32,
            message_fields: Some(TransferObjects::FIELDS),
        };
        pub const ADD_STAKE_FIELD: &'static MessageField = &MessageField {
            name: "add_stake",
            json_name: "addStake",
            number: 17i32,
            message_fields: Some(AddStake::FIELDS),
        };
        pub const ADD_STAKE_TO_ENCODER_FIELD: &'static MessageField = &MessageField {
            name: "add_stake_to_encoder",
            json_name: "addStakeToEncoder",
            number: 18i32,
            message_fields: Some(AddStakeToEncoder::FIELDS),
        };
        pub const WITHDRAW_STAKE_FIELD: &'static MessageField = &MessageField {
            name: "withdraw_stake",
            json_name: "withdrawStake",
            number: 19i32,
            message_fields: Some(WithdrawStake::FIELDS),
        };
        pub const EMBED_DATA_FIELD: &'static MessageField = &MessageField {
            name: "embed_data",
            json_name: "embedData",
            number: 20i32,
            message_fields: Some(EmbedData::FIELDS),
        };
        pub const CLAIM_ESCROW_FIELD: &'static MessageField = &MessageField {
            name: "claim_escrow",
            json_name: "claimEscrow",
            number: 21i32,
            message_fields: Some(ClaimEscrow::FIELDS),
        };
        pub const REPORT_SCORES_FIELD: &'static MessageField = &MessageField {
            name: "report_scores",
            json_name: "reportScores",
            number: 22i32,
            message_fields: Some(ReportScores::FIELDS),
        };
        pub const CHANGE_EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "change_epoch",
            json_name: "changeEpoch",
            number: 100i32,
            message_fields: Some(ChangeEpoch::FIELDS),
        };
        pub const GENESIS_FIELD: &'static MessageField = &MessageField {
            name: "genesis",
            json_name: "genesis",
            number: 101i32,
            message_fields: Some(GenesisTransaction::FIELDS),
        };
        pub const CONSENSUS_COMMIT_PROLOGUE_FIELD: &'static MessageField = &MessageField {
            name: "consensus_commit_prologue",
            json_name: "consensusCommitPrologue",
            number: 102i32,
            message_fields: Some(ConsensusCommitPrologue::FIELDS),
        };
    }
    impl MessageFields for TransactionKind {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ADD_VALIDATOR_FIELD,
            Self::REMOVE_VALIDATOR_FIELD,
            Self::REPORT_VALIDATOR_FIELD,
            Self::UNDO_REPORT_VALIDATOR_FIELD,
            Self::UNDO_VALIDATOR_METADATA_FIELD,
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
            Self::REPORT_SCORES_FIELD,
            Self::CHANGE_EPOCH_FIELD,
            Self::GENESIS_FIELD,
            Self::CONSENSUS_COMMIT_PROLOGUE_FIELD,
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
        pub fn undo_validator_metadata(
            mut self,
        ) -> UpdateValidatorMetadataFieldPathBuilder {
            self.path.push(TransactionKind::UNDO_VALIDATOR_METADATA_FIELD.name);
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
        pub fn report_scores(mut self) -> ReportScoresFieldPathBuilder {
            self.path.push(TransactionKind::REPORT_SCORES_FIELD.name);
            ReportScoresFieldPathBuilder::new_with_base(self.path)
        }
        pub fn change_epoch(mut self) -> ChangeEpochFieldPathBuilder {
            self.path.push(TransactionKind::CHANGE_EPOCH_FIELD.name);
            ChangeEpochFieldPathBuilder::new_with_base(self.path)
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
    }
    impl AddValidator {
        pub const PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "pubkey_bytes",
            json_name: "pubkeyBytes",
            number: 1i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NETWORK_PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "network_pubkey_bytes",
            json_name: "networkPubkeyBytes",
            number: 2i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const WORKER_PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "worker_pubkey_bytes",
            json_name: "workerPubkeyBytes",
            number: 3i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NET_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "net_address",
            json_name: "netAddress",
            number: 4i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const P2P_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "p2p_address",
            json_name: "p2pAddress",
            number: 5i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const PRIMARY_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "primary_address",
            json_name: "primaryAddress",
            number: 6i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const ENCODER_VALIDATOR_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "encoder_validator_address",
            json_name: "encoderValidatorAddress",
            number: 7i32,
            message_fields: Some(Bcs::FIELDS),
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
        pub fn pubkey_bytes(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddValidator::PUBKEY_BYTES_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn network_pubkey_bytes(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddValidator::NETWORK_PUBKEY_BYTES_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn worker_pubkey_bytes(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddValidator::WORKER_PUBKEY_BYTES_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn net_address(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddValidator::NET_ADDRESS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn p2p_address(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddValidator::P2P_ADDRESS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn primary_address(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddValidator::PRIMARY_ADDRESS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn encoder_validator_address(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddValidator::ENCODER_VALIDATOR_ADDRESS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl RemoveValidator {
        pub const PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "pubkey_bytes",
            json_name: "pubkeyBytes",
            number: 1i32,
            message_fields: Some(Bcs::FIELDS),
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
        pub fn pubkey_bytes(mut self) -> BcsFieldPathBuilder {
            self.path.push(RemoveValidator::PUBKEY_BYTES_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
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
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NEXT_EPOCH_P2P_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_p2p_address",
            json_name: "nextEpochP2pAddress",
            number: 2i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NEXT_EPOCH_PRIMARY_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_primary_address",
            json_name: "nextEpochPrimaryAddress",
            number: 3i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NEXT_EPOCH_PROTOCOL_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_protocol_pubkey",
            json_name: "nextEpochProtocolPubkey",
            number: 4i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NEXT_EPOCH_WORKER_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_worker_pubkey",
            json_name: "nextEpochWorkerPubkey",
            number: 5i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NEXT_EPOCH_NETWORK_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_network_pubkey",
            json_name: "nextEpochNetworkPubkey",
            number: 6i32,
            message_fields: Some(Bcs::FIELDS),
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
        pub fn next_epoch_network_address(mut self) -> BcsFieldPathBuilder {
            self.path
                .push(UpdateValidatorMetadata::NEXT_EPOCH_NETWORK_ADDRESS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_epoch_p2p_address(mut self) -> BcsFieldPathBuilder {
            self.path.push(UpdateValidatorMetadata::NEXT_EPOCH_P2P_ADDRESS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_epoch_primary_address(mut self) -> BcsFieldPathBuilder {
            self.path
                .push(UpdateValidatorMetadata::NEXT_EPOCH_PRIMARY_ADDRESS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_epoch_protocol_pubkey(mut self) -> BcsFieldPathBuilder {
            self.path
                .push(UpdateValidatorMetadata::NEXT_EPOCH_PROTOCOL_PUBKEY_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_epoch_worker_pubkey(mut self) -> BcsFieldPathBuilder {
            self.path.push(UpdateValidatorMetadata::NEXT_EPOCH_WORKER_PUBKEY_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_epoch_network_pubkey(mut self) -> BcsFieldPathBuilder {
            self.path
                .push(UpdateValidatorMetadata::NEXT_EPOCH_NETWORK_PUBKEY_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
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
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NETWORK_PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "network_pubkey_bytes",
            json_name: "networkPubkeyBytes",
            number: 2i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const INTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "internal_network_address",
            json_name: "internalNetworkAddress",
            number: 3i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const EXTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "external_network_address",
            json_name: "externalNetworkAddress",
            number: 4i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const OBJECT_SERVER_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "object_server_address",
            json_name: "objectServerAddress",
            number: 5i32,
            message_fields: Some(Bcs::FIELDS),
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
        pub fn encoder_pubkey_bytes(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddEncoder::ENCODER_PUBKEY_BYTES_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn network_pubkey_bytes(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddEncoder::NETWORK_PUBKEY_BYTES_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn internal_network_address(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddEncoder::INTERNAL_NETWORK_ADDRESS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn external_network_address(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddEncoder::EXTERNAL_NETWORK_ADDRESS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn object_server_address(mut self) -> BcsFieldPathBuilder {
            self.path.push(AddEncoder::OBJECT_SERVER_ADDRESS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl RemoveEncoder {
        pub const ENCODER_PUBKEY_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "encoder_pubkey_bytes",
            json_name: "encoderPubkeyBytes",
            number: 1i32,
            message_fields: Some(Bcs::FIELDS),
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
        pub fn encoder_pubkey_bytes(mut self) -> BcsFieldPathBuilder {
            self.path.push(RemoveEncoder::ENCODER_PUBKEY_BYTES_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
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
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NEXT_EPOCH_INTERNAL_NETWORK_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_internal_network_address",
            json_name: "nextEpochInternalNetworkAddress",
            number: 2i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NEXT_EPOCH_NETWORK_PUBKEY_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_network_pubkey",
            json_name: "nextEpochNetworkPubkey",
            number: 3i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const NEXT_EPOCH_OBJECT_SERVER_ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "next_epoch_object_server_address",
            json_name: "nextEpochObjectServerAddress",
            number: 4i32,
            message_fields: Some(Bcs::FIELDS),
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
        pub fn next_epoch_external_network_address(mut self) -> BcsFieldPathBuilder {
            self.path
                .push(
                    UpdateEncoderMetadata::NEXT_EPOCH_EXTERNAL_NETWORK_ADDRESS_FIELD.name,
                );
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_epoch_internal_network_address(mut self) -> BcsFieldPathBuilder {
            self.path
                .push(
                    UpdateEncoderMetadata::NEXT_EPOCH_INTERNAL_NETWORK_ADDRESS_FIELD.name,
                );
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_epoch_network_pubkey(mut self) -> BcsFieldPathBuilder {
            self.path.push(UpdateEncoderMetadata::NEXT_EPOCH_NETWORK_PUBKEY_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn next_epoch_object_server_address(mut self) -> BcsFieldPathBuilder {
            self.path
                .push(
                    UpdateEncoderMetadata::NEXT_EPOCH_OBJECT_SERVER_ADDRESS_FIELD.name,
                );
            BcsFieldPathBuilder::new_with_base(self.path)
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
    impl ReportScores {
        pub const SHARD_INPUT_REF_FIELD: &'static MessageField = &MessageField {
            name: "shard_input_ref",
            json_name: "shardInputRef",
            number: 1i32,
            message_fields: Some(ObjectReference::FIELDS),
        };
        pub const SCORES_FIELD: &'static MessageField = &MessageField {
            name: "scores",
            json_name: "scores",
            number: 2i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const ENCODER_AGGREGATE_SIGNATURE_FIELD: &'static MessageField = &MessageField {
            name: "encoder_aggregate_signature",
            json_name: "encoderAggregateSignature",
            number: 3i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const SIGNERS_FIELD: &'static MessageField = &MessageField {
            name: "signers",
            json_name: "signers",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for ReportScores {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SHARD_INPUT_REF_FIELD,
            Self::SCORES_FIELD,
            Self::ENCODER_AGGREGATE_SIGNATURE_FIELD,
            Self::SIGNERS_FIELD,
        ];
    }
    impl ReportScores {
        pub fn path_builder() -> ReportScoresFieldPathBuilder {
            ReportScoresFieldPathBuilder::new()
        }
    }
    pub struct ReportScoresFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ReportScoresFieldPathBuilder {
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
            self.path.push(ReportScores::SHARD_INPUT_REF_FIELD.name);
            ObjectReferenceFieldPathBuilder::new_with_base(self.path)
        }
        pub fn scores(mut self) -> BcsFieldPathBuilder {
            self.path.push(ReportScores::SCORES_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn encoder_aggregate_signature(mut self) -> BcsFieldPathBuilder {
            self.path.push(ReportScores::ENCODER_AGGREGATE_SIGNATURE_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn signers(mut self) -> String {
            self.path.push(ReportScores::SIGNERS_FIELD.name);
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
    }
    impl MessageFields for TransactionFee {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BASE_FEE_FIELD,
            Self::OPERATION_FEE_FIELD,
            Self::VALUE_FEE_FIELD,
            Self::TOTAL_FEE_FIELD,
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
    }
}
