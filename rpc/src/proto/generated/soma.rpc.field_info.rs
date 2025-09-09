mod _field_impls {
    #![allow(clippy::wrong_self_convention)]
    use super::*;
    use crate::utils::field::MessageFields;
    use crate::utils::field::MessageField;
    impl Argument {
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 1i32,
            message_fields: None,
        };
        pub const INPUT_FIELD: &'static MessageField = &MessageField {
            name: "input",
            json_name: "input",
            number: 2i32,
            message_fields: None,
        };
        pub const RESULT_FIELD: &'static MessageField = &MessageField {
            name: "result",
            json_name: "result",
            number: 3i32,
            message_fields: None,
        };
        pub const SUBRESULT_FIELD: &'static MessageField = &MessageField {
            name: "subresult",
            json_name: "subresult",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for Argument {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::KIND_FIELD,
            Self::INPUT_FIELD,
            Self::RESULT_FIELD,
            Self::SUBRESULT_FIELD,
        ];
    }
    impl Argument {
        pub fn path_builder() -> ArgumentFieldPathBuilder {
            ArgumentFieldPathBuilder::new()
        }
    }
    pub struct ArgumentFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ArgumentFieldPathBuilder {
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
            self.path.push(Argument::KIND_FIELD.name);
            self.finish()
        }
        pub fn input(mut self) -> String {
            self.path.push(Argument::INPUT_FIELD.name);
            self.finish()
        }
        pub fn result(mut self) -> String {
            self.path.push(Argument::RESULT_FIELD.name);
            self.finish()
        }
        pub fn subresult(mut self) -> String {
            self.path.push(Argument::SUBRESULT_FIELD.name);
            self.finish()
        }
    }
    impl BalanceChange {
        pub const ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "address",
            json_name: "address",
            number: 1i32,
            message_fields: None,
        };
        pub const COIN_TYPE_FIELD: &'static MessageField = &MessageField {
            name: "coin_type",
            json_name: "coinType",
            number: 2i32,
            message_fields: None,
        };
        pub const AMOUNT_FIELD: &'static MessageField = &MessageField {
            name: "amount",
            json_name: "amount",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for BalanceChange {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ADDRESS_FIELD,
            Self::COIN_TYPE_FIELD,
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
        pub fn coin_type(mut self) -> String {
            self.path.push(BalanceChange::COIN_TYPE_FIELD.name);
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
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 3i32,
            message_fields: None,
        };
        pub const STATUS_FIELD: &'static MessageField = &MessageField {
            name: "status",
            json_name: "status",
            number: 4i32,
            message_fields: Some(ExecutionStatus::FIELDS),
        };
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 5i32,
            message_fields: None,
        };
        pub const GAS_USED_FIELD: &'static MessageField = &MessageField {
            name: "gas_used",
            json_name: "gasUsed",
            number: 6i32,
            message_fields: Some(GasCostSummary::FIELDS),
        };
        pub const TRANSACTION_DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "transaction_digest",
            json_name: "transactionDigest",
            number: 7i32,
            message_fields: None,
        };
        pub const GAS_OBJECT_FIELD: &'static MessageField = &MessageField {
            name: "gas_object",
            json_name: "gasObject",
            number: 8i32,
            message_fields: Some(ChangedObject::FIELDS),
        };
        pub const EVENTS_DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "events_digest",
            json_name: "eventsDigest",
            number: 9i32,
            message_fields: None,
        };
        pub const DEPENDENCIES_FIELD: &'static MessageField = &MessageField {
            name: "dependencies",
            json_name: "dependencies",
            number: 10i32,
            message_fields: None,
        };
        pub const LAMPORT_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "lamport_version",
            json_name: "lamportVersion",
            number: 11i32,
            message_fields: None,
        };
        pub const CHANGED_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "changed_objects",
            json_name: "changedObjects",
            number: 12i32,
            message_fields: Some(ChangedObject::FIELDS),
        };
        pub const UNCHANGED_CONSENSUS_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "unchanged_consensus_objects",
            json_name: "unchangedConsensusObjects",
            number: 13i32,
            message_fields: Some(UnchangedConsensusObject::FIELDS),
        };
        pub const AUXILIARY_DATA_DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "auxiliary_data_digest",
            json_name: "auxiliaryDataDigest",
            number: 14i32,
            message_fields: None,
        };
    }
    impl MessageFields for TransactionEffects {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BCS_FIELD,
            Self::DIGEST_FIELD,
            Self::VERSION_FIELD,
            Self::STATUS_FIELD,
            Self::EPOCH_FIELD,
            Self::GAS_USED_FIELD,
            Self::TRANSACTION_DIGEST_FIELD,
            Self::GAS_OBJECT_FIELD,
            Self::EVENTS_DIGEST_FIELD,
            Self::DEPENDENCIES_FIELD,
            Self::LAMPORT_VERSION_FIELD,
            Self::CHANGED_OBJECTS_FIELD,
            Self::UNCHANGED_CONSENSUS_OBJECTS_FIELD,
            Self::AUXILIARY_DATA_DIGEST_FIELD,
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
        pub fn version(mut self) -> String {
            self.path.push(TransactionEffects::VERSION_FIELD.name);
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
        pub fn gas_used(mut self) -> GasCostSummaryFieldPathBuilder {
            self.path.push(TransactionEffects::GAS_USED_FIELD.name);
            GasCostSummaryFieldPathBuilder::new_with_base(self.path)
        }
        pub fn transaction_digest(mut self) -> String {
            self.path.push(TransactionEffects::TRANSACTION_DIGEST_FIELD.name);
            self.finish()
        }
        pub fn gas_object(mut self) -> ChangedObjectFieldPathBuilder {
            self.path.push(TransactionEffects::GAS_OBJECT_FIELD.name);
            ChangedObjectFieldPathBuilder::new_with_base(self.path)
        }
        pub fn events_digest(mut self) -> String {
            self.path.push(TransactionEffects::EVENTS_DIGEST_FIELD.name);
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
        pub fn unchanged_consensus_objects(
            mut self,
        ) -> UnchangedConsensusObjectFieldPathBuilder {
            self.path.push(TransactionEffects::UNCHANGED_CONSENSUS_OBJECTS_FIELD.name);
            UnchangedConsensusObjectFieldPathBuilder::new_with_base(self.path)
        }
        pub fn auxiliary_data_digest(mut self) -> String {
            self.path.push(TransactionEffects::AUXILIARY_DATA_DIGEST_FIELD.name);
            self.finish()
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
    impl UnchangedConsensusObject {
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
    impl MessageFields for UnchangedConsensusObject {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::KIND_FIELD,
            Self::OBJECT_ID_FIELD,
            Self::VERSION_FIELD,
            Self::DIGEST_FIELD,
            Self::OBJECT_TYPE_FIELD,
        ];
    }
    impl UnchangedConsensusObject {
        pub fn path_builder() -> UnchangedConsensusObjectFieldPathBuilder {
            UnchangedConsensusObjectFieldPathBuilder::new()
        }
    }
    pub struct UnchangedConsensusObjectFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl UnchangedConsensusObjectFieldPathBuilder {
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
            self.path.push(UnchangedConsensusObject::KIND_FIELD.name);
            self.finish()
        }
        pub fn object_id(mut self) -> String {
            self.path.push(UnchangedConsensusObject::OBJECT_ID_FIELD.name);
            self.finish()
        }
        pub fn version(mut self) -> String {
            self.path.push(UnchangedConsensusObject::VERSION_FIELD.name);
            self.finish()
        }
        pub fn digest(mut self) -> String {
            self.path.push(UnchangedConsensusObject::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn object_type(mut self) -> String {
            self.path.push(UnchangedConsensusObject::OBJECT_TYPE_FIELD.name);
            self.finish()
        }
    }
    impl TransactionEvents {
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
        pub const EVENTS_FIELD: &'static MessageField = &MessageField {
            name: "events",
            json_name: "events",
            number: 3i32,
            message_fields: Some(Event::FIELDS),
        };
    }
    impl MessageFields for TransactionEvents {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BCS_FIELD,
            Self::DIGEST_FIELD,
            Self::EVENTS_FIELD,
        ];
    }
    impl TransactionEvents {
        pub fn path_builder() -> TransactionEventsFieldPathBuilder {
            TransactionEventsFieldPathBuilder::new()
        }
    }
    pub struct TransactionEventsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TransactionEventsFieldPathBuilder {
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
            self.path.push(TransactionEvents::BCS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn digest(mut self) -> String {
            self.path.push(TransactionEvents::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn events(mut self) -> EventFieldPathBuilder {
            self.path.push(TransactionEvents::EVENTS_FIELD.name);
            EventFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl Event {
        pub const PACKAGE_ID_FIELD: &'static MessageField = &MessageField {
            name: "package_id",
            json_name: "packageId",
            number: 1i32,
            message_fields: None,
        };
        pub const MODULE_FIELD: &'static MessageField = &MessageField {
            name: "module",
            json_name: "module",
            number: 2i32,
            message_fields: None,
        };
        pub const SENDER_FIELD: &'static MessageField = &MessageField {
            name: "sender",
            json_name: "sender",
            number: 3i32,
            message_fields: None,
        };
        pub const EVENT_TYPE_FIELD: &'static MessageField = &MessageField {
            name: "event_type",
            json_name: "eventType",
            number: 4i32,
            message_fields: None,
        };
        pub const CONTENTS_FIELD: &'static MessageField = &MessageField {
            name: "contents",
            json_name: "contents",
            number: 5i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const JSON_FIELD: &'static MessageField = &MessageField {
            name: "json",
            json_name: "json",
            number: 6i32,
            message_fields: None,
        };
    }
    impl MessageFields for Event {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::PACKAGE_ID_FIELD,
            Self::MODULE_FIELD,
            Self::SENDER_FIELD,
            Self::EVENT_TYPE_FIELD,
            Self::CONTENTS_FIELD,
            Self::JSON_FIELD,
        ];
    }
    impl Event {
        pub fn path_builder() -> EventFieldPathBuilder {
            EventFieldPathBuilder::new()
        }
    }
    pub struct EventFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl EventFieldPathBuilder {
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
        pub fn package_id(mut self) -> String {
            self.path.push(Event::PACKAGE_ID_FIELD.name);
            self.finish()
        }
        pub fn module(mut self) -> String {
            self.path.push(Event::MODULE_FIELD.name);
            self.finish()
        }
        pub fn sender(mut self) -> String {
            self.path.push(Event::SENDER_FIELD.name);
            self.finish()
        }
        pub fn event_type(mut self) -> String {
            self.path.push(Event::EVENT_TYPE_FIELD.name);
            self.finish()
        }
        pub fn contents(mut self) -> BcsFieldPathBuilder {
            self.path.push(Event::CONTENTS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn json(mut self) -> String {
            self.path.push(Event::JSON_FIELD.name);
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
        pub const EVENTS_FIELD: &'static MessageField = &MessageField {
            name: "events",
            json_name: "events",
            number: 5i32,
            message_fields: Some(TransactionEvents::FIELDS),
        };
        pub const CHECKPOINT_FIELD: &'static MessageField = &MessageField {
            name: "checkpoint",
            json_name: "checkpoint",
            number: 6i32,
            message_fields: None,
        };
        pub const TIMESTAMP_FIELD: &'static MessageField = &MessageField {
            name: "timestamp",
            json_name: "timestamp",
            number: 7i32,
            message_fields: None,
        };
        pub const BALANCE_CHANGES_FIELD: &'static MessageField = &MessageField {
            name: "balance_changes",
            json_name: "balanceChanges",
            number: 8i32,
            message_fields: Some(BalanceChange::FIELDS),
        };
        pub const INPUT_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "input_objects",
            json_name: "inputObjects",
            number: 10i32,
            message_fields: Some(Object::FIELDS),
        };
        pub const OUTPUT_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "output_objects",
            json_name: "outputObjects",
            number: 11i32,
            message_fields: Some(Object::FIELDS),
        };
    }
    impl MessageFields for ExecutedTransaction {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::DIGEST_FIELD,
            Self::TRANSACTION_FIELD,
            Self::SIGNATURES_FIELD,
            Self::EFFECTS_FIELD,
            Self::EVENTS_FIELD,
            Self::CHECKPOINT_FIELD,
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
        pub fn events(mut self) -> TransactionEventsFieldPathBuilder {
            self.path.push(ExecutedTransaction::EVENTS_FIELD.name);
            TransactionEventsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn checkpoint(mut self) -> String {
            self.path.push(ExecutedTransaction::CHECKPOINT_FIELD.name);
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
        pub const COMMAND_FIELD: &'static MessageField = &MessageField {
            name: "command",
            json_name: "command",
            number: 2i32,
            message_fields: None,
        };
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 3i32,
            message_fields: None,
        };
        pub const ABORT_FIELD: &'static MessageField = &MessageField {
            name: "abort",
            json_name: "abort",
            number: 4i32,
            message_fields: Some(MoveAbort::FIELDS),
        };
        pub const SIZE_ERROR_FIELD: &'static MessageField = &MessageField {
            name: "size_error",
            json_name: "sizeError",
            number: 5i32,
            message_fields: Some(SizeError::FIELDS),
        };
        pub const COMMAND_ARGUMENT_ERROR_FIELD: &'static MessageField = &MessageField {
            name: "command_argument_error",
            json_name: "commandArgumentError",
            number: 6i32,
            message_fields: Some(CommandArgumentError::FIELDS),
        };
        pub const TYPE_ARGUMENT_ERROR_FIELD: &'static MessageField = &MessageField {
            name: "type_argument_error",
            json_name: "typeArgumentError",
            number: 7i32,
            message_fields: Some(TypeArgumentError::FIELDS),
        };
        pub const PACKAGE_UPGRADE_ERROR_FIELD: &'static MessageField = &MessageField {
            name: "package_upgrade_error",
            json_name: "packageUpgradeError",
            number: 8i32,
            message_fields: Some(PackageUpgradeError::FIELDS),
        };
        pub const INDEX_ERROR_FIELD: &'static MessageField = &MessageField {
            name: "index_error",
            json_name: "indexError",
            number: 9i32,
            message_fields: Some(IndexError::FIELDS),
        };
        pub const OBJECT_ID_FIELD: &'static MessageField = &MessageField {
            name: "object_id",
            json_name: "objectId",
            number: 10i32,
            message_fields: None,
        };
        pub const COIN_DENY_LIST_ERROR_FIELD: &'static MessageField = &MessageField {
            name: "coin_deny_list_error",
            json_name: "coinDenyListError",
            number: 11i32,
            message_fields: Some(CoinDenyListError::FIELDS),
        };
        pub const CONGESTED_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "congested_objects",
            json_name: "congestedObjects",
            number: 12i32,
            message_fields: Some(CongestedObjects::FIELDS),
        };
    }
    impl MessageFields for ExecutionError {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::DESCRIPTION_FIELD,
            Self::COMMAND_FIELD,
            Self::KIND_FIELD,
            Self::ABORT_FIELD,
            Self::SIZE_ERROR_FIELD,
            Self::COMMAND_ARGUMENT_ERROR_FIELD,
            Self::TYPE_ARGUMENT_ERROR_FIELD,
            Self::PACKAGE_UPGRADE_ERROR_FIELD,
            Self::INDEX_ERROR_FIELD,
            Self::OBJECT_ID_FIELD,
            Self::COIN_DENY_LIST_ERROR_FIELD,
            Self::CONGESTED_OBJECTS_FIELD,
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
        pub fn command(mut self) -> String {
            self.path.push(ExecutionError::COMMAND_FIELD.name);
            self.finish()
        }
        pub fn kind(mut self) -> String {
            self.path.push(ExecutionError::KIND_FIELD.name);
            self.finish()
        }
        pub fn abort(mut self) -> MoveAbortFieldPathBuilder {
            self.path.push(ExecutionError::ABORT_FIELD.name);
            MoveAbortFieldPathBuilder::new_with_base(self.path)
        }
        pub fn size_error(mut self) -> SizeErrorFieldPathBuilder {
            self.path.push(ExecutionError::SIZE_ERROR_FIELD.name);
            SizeErrorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn command_argument_error(mut self) -> CommandArgumentErrorFieldPathBuilder {
            self.path.push(ExecutionError::COMMAND_ARGUMENT_ERROR_FIELD.name);
            CommandArgumentErrorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn type_argument_error(mut self) -> TypeArgumentErrorFieldPathBuilder {
            self.path.push(ExecutionError::TYPE_ARGUMENT_ERROR_FIELD.name);
            TypeArgumentErrorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn package_upgrade_error(mut self) -> PackageUpgradeErrorFieldPathBuilder {
            self.path.push(ExecutionError::PACKAGE_UPGRADE_ERROR_FIELD.name);
            PackageUpgradeErrorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn index_error(mut self) -> IndexErrorFieldPathBuilder {
            self.path.push(ExecutionError::INDEX_ERROR_FIELD.name);
            IndexErrorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn object_id(mut self) -> String {
            self.path.push(ExecutionError::OBJECT_ID_FIELD.name);
            self.finish()
        }
        pub fn coin_deny_list_error(mut self) -> CoinDenyListErrorFieldPathBuilder {
            self.path.push(ExecutionError::COIN_DENY_LIST_ERROR_FIELD.name);
            CoinDenyListErrorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn congested_objects(mut self) -> CongestedObjectsFieldPathBuilder {
            self.path.push(ExecutionError::CONGESTED_OBJECTS_FIELD.name);
            CongestedObjectsFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl MoveAbort {
        pub const ABORT_CODE_FIELD: &'static MessageField = &MessageField {
            name: "abort_code",
            json_name: "abortCode",
            number: 1i32,
            message_fields: None,
        };
        pub const LOCATION_FIELD: &'static MessageField = &MessageField {
            name: "location",
            json_name: "location",
            number: 2i32,
            message_fields: Some(MoveLocation::FIELDS),
        };
        pub const CLEVER_ERROR_FIELD: &'static MessageField = &MessageField {
            name: "clever_error",
            json_name: "cleverError",
            number: 3i32,
            message_fields: Some(CleverError::FIELDS),
        };
    }
    impl MessageFields for MoveAbort {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ABORT_CODE_FIELD,
            Self::LOCATION_FIELD,
            Self::CLEVER_ERROR_FIELD,
        ];
    }
    impl MoveAbort {
        pub fn path_builder() -> MoveAbortFieldPathBuilder {
            MoveAbortFieldPathBuilder::new()
        }
    }
    pub struct MoveAbortFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl MoveAbortFieldPathBuilder {
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
        pub fn abort_code(mut self) -> String {
            self.path.push(MoveAbort::ABORT_CODE_FIELD.name);
            self.finish()
        }
        pub fn location(mut self) -> MoveLocationFieldPathBuilder {
            self.path.push(MoveAbort::LOCATION_FIELD.name);
            MoveLocationFieldPathBuilder::new_with_base(self.path)
        }
        pub fn clever_error(mut self) -> CleverErrorFieldPathBuilder {
            self.path.push(MoveAbort::CLEVER_ERROR_FIELD.name);
            CleverErrorFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl MoveLocation {
        pub const PACKAGE_FIELD: &'static MessageField = &MessageField {
            name: "package",
            json_name: "package",
            number: 1i32,
            message_fields: None,
        };
        pub const MODULE_FIELD: &'static MessageField = &MessageField {
            name: "module",
            json_name: "module",
            number: 2i32,
            message_fields: None,
        };
        pub const FUNCTION_FIELD: &'static MessageField = &MessageField {
            name: "function",
            json_name: "function",
            number: 3i32,
            message_fields: None,
        };
        pub const INSTRUCTION_FIELD: &'static MessageField = &MessageField {
            name: "instruction",
            json_name: "instruction",
            number: 4i32,
            message_fields: None,
        };
        pub const FUNCTION_NAME_FIELD: &'static MessageField = &MessageField {
            name: "function_name",
            json_name: "functionName",
            number: 5i32,
            message_fields: None,
        };
    }
    impl MessageFields for MoveLocation {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::PACKAGE_FIELD,
            Self::MODULE_FIELD,
            Self::FUNCTION_FIELD,
            Self::INSTRUCTION_FIELD,
            Self::FUNCTION_NAME_FIELD,
        ];
    }
    impl MoveLocation {
        pub fn path_builder() -> MoveLocationFieldPathBuilder {
            MoveLocationFieldPathBuilder::new()
        }
    }
    pub struct MoveLocationFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl MoveLocationFieldPathBuilder {
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
        pub fn package(mut self) -> String {
            self.path.push(MoveLocation::PACKAGE_FIELD.name);
            self.finish()
        }
        pub fn module(mut self) -> String {
            self.path.push(MoveLocation::MODULE_FIELD.name);
            self.finish()
        }
        pub fn function(mut self) -> String {
            self.path.push(MoveLocation::FUNCTION_FIELD.name);
            self.finish()
        }
        pub fn instruction(mut self) -> String {
            self.path.push(MoveLocation::INSTRUCTION_FIELD.name);
            self.finish()
        }
        pub fn function_name(mut self) -> String {
            self.path.push(MoveLocation::FUNCTION_NAME_FIELD.name);
            self.finish()
        }
    }
    impl CleverError {
        pub const ERROR_CODE_FIELD: &'static MessageField = &MessageField {
            name: "error_code",
            json_name: "errorCode",
            number: 1i32,
            message_fields: None,
        };
        pub const LINE_NUMBER_FIELD: &'static MessageField = &MessageField {
            name: "line_number",
            json_name: "lineNumber",
            number: 2i32,
            message_fields: None,
        };
        pub const CONSTANT_NAME_FIELD: &'static MessageField = &MessageField {
            name: "constant_name",
            json_name: "constantName",
            number: 3i32,
            message_fields: None,
        };
        pub const CONSTANT_TYPE_FIELD: &'static MessageField = &MessageField {
            name: "constant_type",
            json_name: "constantType",
            number: 4i32,
            message_fields: None,
        };
        pub const RENDERED_FIELD: &'static MessageField = &MessageField {
            name: "rendered",
            json_name: "rendered",
            number: 5i32,
            message_fields: None,
        };
        pub const RAW_FIELD: &'static MessageField = &MessageField {
            name: "raw",
            json_name: "raw",
            number: 6i32,
            message_fields: None,
        };
    }
    impl MessageFields for CleverError {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ERROR_CODE_FIELD,
            Self::LINE_NUMBER_FIELD,
            Self::CONSTANT_NAME_FIELD,
            Self::CONSTANT_TYPE_FIELD,
            Self::RENDERED_FIELD,
            Self::RAW_FIELD,
        ];
    }
    impl CleverError {
        pub fn path_builder() -> CleverErrorFieldPathBuilder {
            CleverErrorFieldPathBuilder::new()
        }
    }
    pub struct CleverErrorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl CleverErrorFieldPathBuilder {
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
        pub fn error_code(mut self) -> String {
            self.path.push(CleverError::ERROR_CODE_FIELD.name);
            self.finish()
        }
        pub fn line_number(mut self) -> String {
            self.path.push(CleverError::LINE_NUMBER_FIELD.name);
            self.finish()
        }
        pub fn constant_name(mut self) -> String {
            self.path.push(CleverError::CONSTANT_NAME_FIELD.name);
            self.finish()
        }
        pub fn constant_type(mut self) -> String {
            self.path.push(CleverError::CONSTANT_TYPE_FIELD.name);
            self.finish()
        }
        pub fn rendered(mut self) -> String {
            self.path.push(CleverError::RENDERED_FIELD.name);
            self.finish()
        }
        pub fn raw(mut self) -> String {
            self.path.push(CleverError::RAW_FIELD.name);
            self.finish()
        }
    }
    impl SizeError {
        pub const SIZE_FIELD: &'static MessageField = &MessageField {
            name: "size",
            json_name: "size",
            number: 1i32,
            message_fields: None,
        };
        pub const MAX_SIZE_FIELD: &'static MessageField = &MessageField {
            name: "max_size",
            json_name: "maxSize",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for SizeError {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SIZE_FIELD,
            Self::MAX_SIZE_FIELD,
        ];
    }
    impl SizeError {
        pub fn path_builder() -> SizeErrorFieldPathBuilder {
            SizeErrorFieldPathBuilder::new()
        }
    }
    pub struct SizeErrorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SizeErrorFieldPathBuilder {
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
        pub fn size(mut self) -> String {
            self.path.push(SizeError::SIZE_FIELD.name);
            self.finish()
        }
        pub fn max_size(mut self) -> String {
            self.path.push(SizeError::MAX_SIZE_FIELD.name);
            self.finish()
        }
    }
    impl IndexError {
        pub const INDEX_FIELD: &'static MessageField = &MessageField {
            name: "index",
            json_name: "index",
            number: 1i32,
            message_fields: None,
        };
        pub const SUBRESULT_FIELD: &'static MessageField = &MessageField {
            name: "subresult",
            json_name: "subresult",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for IndexError {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::INDEX_FIELD,
            Self::SUBRESULT_FIELD,
        ];
    }
    impl IndexError {
        pub fn path_builder() -> IndexErrorFieldPathBuilder {
            IndexErrorFieldPathBuilder::new()
        }
    }
    pub struct IndexErrorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl IndexErrorFieldPathBuilder {
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
        pub fn index(mut self) -> String {
            self.path.push(IndexError::INDEX_FIELD.name);
            self.finish()
        }
        pub fn subresult(mut self) -> String {
            self.path.push(IndexError::SUBRESULT_FIELD.name);
            self.finish()
        }
    }
    impl CoinDenyListError {
        pub const ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "address",
            json_name: "address",
            number: 1i32,
            message_fields: None,
        };
        pub const COIN_TYPE_FIELD: &'static MessageField = &MessageField {
            name: "coin_type",
            json_name: "coinType",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for CoinDenyListError {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ADDRESS_FIELD,
            Self::COIN_TYPE_FIELD,
        ];
    }
    impl CoinDenyListError {
        pub fn path_builder() -> CoinDenyListErrorFieldPathBuilder {
            CoinDenyListErrorFieldPathBuilder::new()
        }
    }
    pub struct CoinDenyListErrorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl CoinDenyListErrorFieldPathBuilder {
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
            self.path.push(CoinDenyListError::ADDRESS_FIELD.name);
            self.finish()
        }
        pub fn coin_type(mut self) -> String {
            self.path.push(CoinDenyListError::COIN_TYPE_FIELD.name);
            self.finish()
        }
    }
    impl CongestedObjects {
        pub const OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "objects",
            json_name: "objects",
            number: 1i32,
            message_fields: None,
        };
    }
    impl MessageFields for CongestedObjects {
        const FIELDS: &'static [&'static MessageField] = &[Self::OBJECTS_FIELD];
    }
    impl CongestedObjects {
        pub fn path_builder() -> CongestedObjectsFieldPathBuilder {
            CongestedObjectsFieldPathBuilder::new()
        }
    }
    pub struct CongestedObjectsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl CongestedObjectsFieldPathBuilder {
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
        pub fn objects(mut self) -> String {
            self.path.push(CongestedObjects::OBJECTS_FIELD.name);
            self.finish()
        }
    }
    impl CommandArgumentError {
        pub const ARGUMENT_FIELD: &'static MessageField = &MessageField {
            name: "argument",
            json_name: "argument",
            number: 1i32,
            message_fields: None,
        };
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 2i32,
            message_fields: None,
        };
        pub const INDEX_ERROR_FIELD: &'static MessageField = &MessageField {
            name: "index_error",
            json_name: "indexError",
            number: 3i32,
            message_fields: Some(IndexError::FIELDS),
        };
    }
    impl MessageFields for CommandArgumentError {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ARGUMENT_FIELD,
            Self::KIND_FIELD,
            Self::INDEX_ERROR_FIELD,
        ];
    }
    impl CommandArgumentError {
        pub fn path_builder() -> CommandArgumentErrorFieldPathBuilder {
            CommandArgumentErrorFieldPathBuilder::new()
        }
    }
    pub struct CommandArgumentErrorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl CommandArgumentErrorFieldPathBuilder {
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
        pub fn argument(mut self) -> String {
            self.path.push(CommandArgumentError::ARGUMENT_FIELD.name);
            self.finish()
        }
        pub fn kind(mut self) -> String {
            self.path.push(CommandArgumentError::KIND_FIELD.name);
            self.finish()
        }
        pub fn index_error(mut self) -> IndexErrorFieldPathBuilder {
            self.path.push(CommandArgumentError::INDEX_ERROR_FIELD.name);
            IndexErrorFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl PackageUpgradeError {
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 1i32,
            message_fields: None,
        };
        pub const PACKAGE_ID_FIELD: &'static MessageField = &MessageField {
            name: "package_id",
            json_name: "packageId",
            number: 2i32,
            message_fields: None,
        };
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 3i32,
            message_fields: None,
        };
        pub const POLICY_FIELD: &'static MessageField = &MessageField {
            name: "policy",
            json_name: "policy",
            number: 4i32,
            message_fields: None,
        };
        pub const TICKET_ID_FIELD: &'static MessageField = &MessageField {
            name: "ticket_id",
            json_name: "ticketId",
            number: 5i32,
            message_fields: None,
        };
    }
    impl MessageFields for PackageUpgradeError {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::KIND_FIELD,
            Self::PACKAGE_ID_FIELD,
            Self::DIGEST_FIELD,
            Self::POLICY_FIELD,
            Self::TICKET_ID_FIELD,
        ];
    }
    impl PackageUpgradeError {
        pub fn path_builder() -> PackageUpgradeErrorFieldPathBuilder {
            PackageUpgradeErrorFieldPathBuilder::new()
        }
    }
    pub struct PackageUpgradeErrorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl PackageUpgradeErrorFieldPathBuilder {
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
            self.path.push(PackageUpgradeError::KIND_FIELD.name);
            self.finish()
        }
        pub fn package_id(mut self) -> String {
            self.path.push(PackageUpgradeError::PACKAGE_ID_FIELD.name);
            self.finish()
        }
        pub fn digest(mut self) -> String {
            self.path.push(PackageUpgradeError::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn policy(mut self) -> String {
            self.path.push(PackageUpgradeError::POLICY_FIELD.name);
            self.finish()
        }
        pub fn ticket_id(mut self) -> String {
            self.path.push(PackageUpgradeError::TICKET_ID_FIELD.name);
            self.finish()
        }
    }
    impl TypeArgumentError {
        pub const TYPE_ARGUMENT_FIELD: &'static MessageField = &MessageField {
            name: "type_argument",
            json_name: "typeArgument",
            number: 1i32,
            message_fields: None,
        };
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for TypeArgumentError {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::TYPE_ARGUMENT_FIELD,
            Self::KIND_FIELD,
        ];
    }
    impl TypeArgumentError {
        pub fn path_builder() -> TypeArgumentErrorFieldPathBuilder {
            TypeArgumentErrorFieldPathBuilder::new()
        }
    }
    pub struct TypeArgumentErrorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TypeArgumentErrorFieldPathBuilder {
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
        pub fn type_argument(mut self) -> String {
            self.path.push(TypeArgumentError::TYPE_ARGUMENT_FIELD.name);
            self.finish()
        }
        pub fn kind(mut self) -> String {
            self.path.push(TypeArgumentError::KIND_FIELD.name);
            self.finish()
        }
    }
    impl GasCostSummary {
        pub const COMPUTATION_COST_FIELD: &'static MessageField = &MessageField {
            name: "computation_cost",
            json_name: "computationCost",
            number: 1i32,
            message_fields: None,
        };
        pub const STORAGE_COST_FIELD: &'static MessageField = &MessageField {
            name: "storage_cost",
            json_name: "storageCost",
            number: 2i32,
            message_fields: None,
        };
        pub const STORAGE_REBATE_FIELD: &'static MessageField = &MessageField {
            name: "storage_rebate",
            json_name: "storageRebate",
            number: 3i32,
            message_fields: None,
        };
        pub const NON_REFUNDABLE_STORAGE_FEE_FIELD: &'static MessageField = &MessageField {
            name: "non_refundable_storage_fee",
            json_name: "nonRefundableStorageFee",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for GasCostSummary {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::COMPUTATION_COST_FIELD,
            Self::STORAGE_COST_FIELD,
            Self::STORAGE_REBATE_FIELD,
            Self::NON_REFUNDABLE_STORAGE_FEE_FIELD,
        ];
    }
    impl GasCostSummary {
        pub fn path_builder() -> GasCostSummaryFieldPathBuilder {
            GasCostSummaryFieldPathBuilder::new()
        }
    }
    pub struct GasCostSummaryFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl GasCostSummaryFieldPathBuilder {
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
        pub fn computation_cost(mut self) -> String {
            self.path.push(GasCostSummary::COMPUTATION_COST_FIELD.name);
            self.finish()
        }
        pub fn storage_cost(mut self) -> String {
            self.path.push(GasCostSummary::STORAGE_COST_FIELD.name);
            self.finish()
        }
        pub fn storage_rebate(mut self) -> String {
            self.path.push(GasCostSummary::STORAGE_REBATE_FIELD.name);
            self.finish()
        }
        pub fn non_refundable_storage_fee(mut self) -> String {
            self.path.push(GasCostSummary::NON_REFUNDABLE_STORAGE_FEE_FIELD.name);
            self.finish()
        }
    }
    impl Input {
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 1i32,
            message_fields: None,
        };
        pub const PURE_FIELD: &'static MessageField = &MessageField {
            name: "pure",
            json_name: "pure",
            number: 2i32,
            message_fields: None,
        };
        pub const OBJECT_ID_FIELD: &'static MessageField = &MessageField {
            name: "object_id",
            json_name: "objectId",
            number: 3i32,
            message_fields: None,
        };
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 4i32,
            message_fields: None,
        };
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 5i32,
            message_fields: None,
        };
        pub const MUTABLE_FIELD: &'static MessageField = &MessageField {
            name: "mutable",
            json_name: "mutable",
            number: 6i32,
            message_fields: None,
        };
        pub const LITERAL_FIELD: &'static MessageField = &MessageField {
            name: "literal",
            json_name: "literal",
            number: 1000i32,
            message_fields: None,
        };
    }
    impl MessageFields for Input {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::KIND_FIELD,
            Self::PURE_FIELD,
            Self::OBJECT_ID_FIELD,
            Self::VERSION_FIELD,
            Self::DIGEST_FIELD,
            Self::MUTABLE_FIELD,
            Self::LITERAL_FIELD,
        ];
    }
    impl Input {
        pub fn path_builder() -> InputFieldPathBuilder {
            InputFieldPathBuilder::new()
        }
    }
    pub struct InputFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl InputFieldPathBuilder {
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
            self.path.push(Input::KIND_FIELD.name);
            self.finish()
        }
        pub fn pure(mut self) -> String {
            self.path.push(Input::PURE_FIELD.name);
            self.finish()
        }
        pub fn object_id(mut self) -> String {
            self.path.push(Input::OBJECT_ID_FIELD.name);
            self.finish()
        }
        pub fn version(mut self) -> String {
            self.path.push(Input::VERSION_FIELD.name);
            self.finish()
        }
        pub fn digest(mut self) -> String {
            self.path.push(Input::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn mutable(mut self) -> String {
            self.path.push(Input::MUTABLE_FIELD.name);
            self.finish()
        }
        pub fn literal(mut self) -> String {
            self.path.push(Input::LITERAL_FIELD.name);
            self.finish()
        }
    }
    impl Package {
        pub const STORAGE_ID_FIELD: &'static MessageField = &MessageField {
            name: "storage_id",
            json_name: "storageId",
            number: 1i32,
            message_fields: None,
        };
        pub const ORIGINAL_ID_FIELD: &'static MessageField = &MessageField {
            name: "original_id",
            json_name: "originalId",
            number: 2i32,
            message_fields: None,
        };
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 3i32,
            message_fields: None,
        };
        pub const MODULES_FIELD: &'static MessageField = &MessageField {
            name: "modules",
            json_name: "modules",
            number: 4i32,
            message_fields: Some(Module::FIELDS),
        };
        pub const TYPE_ORIGINS_FIELD: &'static MessageField = &MessageField {
            name: "type_origins",
            json_name: "typeOrigins",
            number: 5i32,
            message_fields: Some(TypeOrigin::FIELDS),
        };
        pub const LINKAGE_FIELD: &'static MessageField = &MessageField {
            name: "linkage",
            json_name: "linkage",
            number: 6i32,
            message_fields: Some(Linkage::FIELDS),
        };
    }
    impl MessageFields for Package {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::STORAGE_ID_FIELD,
            Self::ORIGINAL_ID_FIELD,
            Self::VERSION_FIELD,
            Self::MODULES_FIELD,
            Self::TYPE_ORIGINS_FIELD,
            Self::LINKAGE_FIELD,
        ];
    }
    impl Package {
        pub fn path_builder() -> PackageFieldPathBuilder {
            PackageFieldPathBuilder::new()
        }
    }
    pub struct PackageFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl PackageFieldPathBuilder {
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
        pub fn storage_id(mut self) -> String {
            self.path.push(Package::STORAGE_ID_FIELD.name);
            self.finish()
        }
        pub fn original_id(mut self) -> String {
            self.path.push(Package::ORIGINAL_ID_FIELD.name);
            self.finish()
        }
        pub fn version(mut self) -> String {
            self.path.push(Package::VERSION_FIELD.name);
            self.finish()
        }
        pub fn modules(mut self) -> ModuleFieldPathBuilder {
            self.path.push(Package::MODULES_FIELD.name);
            ModuleFieldPathBuilder::new_with_base(self.path)
        }
        pub fn type_origins(mut self) -> TypeOriginFieldPathBuilder {
            self.path.push(Package::TYPE_ORIGINS_FIELD.name);
            TypeOriginFieldPathBuilder::new_with_base(self.path)
        }
        pub fn linkage(mut self) -> LinkageFieldPathBuilder {
            self.path.push(Package::LINKAGE_FIELD.name);
            LinkageFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl Module {
        pub const NAME_FIELD: &'static MessageField = &MessageField {
            name: "name",
            json_name: "name",
            number: 1i32,
            message_fields: None,
        };
        pub const CONTENTS_FIELD: &'static MessageField = &MessageField {
            name: "contents",
            json_name: "contents",
            number: 2i32,
            message_fields: None,
        };
        pub const DATATYPES_FIELD: &'static MessageField = &MessageField {
            name: "datatypes",
            json_name: "datatypes",
            number: 3i32,
            message_fields: Some(DatatypeDescriptor::FIELDS),
        };
        pub const FUNCTIONS_FIELD: &'static MessageField = &MessageField {
            name: "functions",
            json_name: "functions",
            number: 4i32,
            message_fields: Some(FunctionDescriptor::FIELDS),
        };
    }
    impl MessageFields for Module {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::NAME_FIELD,
            Self::CONTENTS_FIELD,
            Self::DATATYPES_FIELD,
            Self::FUNCTIONS_FIELD,
        ];
    }
    impl Module {
        pub fn path_builder() -> ModuleFieldPathBuilder {
            ModuleFieldPathBuilder::new()
        }
    }
    pub struct ModuleFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ModuleFieldPathBuilder {
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
            self.path.push(Module::NAME_FIELD.name);
            self.finish()
        }
        pub fn contents(mut self) -> String {
            self.path.push(Module::CONTENTS_FIELD.name);
            self.finish()
        }
        pub fn datatypes(mut self) -> DatatypeDescriptorFieldPathBuilder {
            self.path.push(Module::DATATYPES_FIELD.name);
            DatatypeDescriptorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn functions(mut self) -> FunctionDescriptorFieldPathBuilder {
            self.path.push(Module::FUNCTIONS_FIELD.name);
            FunctionDescriptorFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl DatatypeDescriptor {
        pub const TYPE_NAME_FIELD: &'static MessageField = &MessageField {
            name: "type_name",
            json_name: "typeName",
            number: 1i32,
            message_fields: None,
        };
        pub const DEFINING_ID_FIELD: &'static MessageField = &MessageField {
            name: "defining_id",
            json_name: "definingId",
            number: 2i32,
            message_fields: None,
        };
        pub const MODULE_FIELD: &'static MessageField = &MessageField {
            name: "module",
            json_name: "module",
            number: 3i32,
            message_fields: None,
        };
        pub const NAME_FIELD: &'static MessageField = &MessageField {
            name: "name",
            json_name: "name",
            number: 4i32,
            message_fields: None,
        };
        pub const ABILITIES_FIELD: &'static MessageField = &MessageField {
            name: "abilities",
            json_name: "abilities",
            number: 5i32,
            message_fields: None,
        };
        pub const TYPE_PARAMETERS_FIELD: &'static MessageField = &MessageField {
            name: "type_parameters",
            json_name: "typeParameters",
            number: 6i32,
            message_fields: Some(TypeParameter::FIELDS),
        };
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 7i32,
            message_fields: None,
        };
        pub const FIELDS_FIELD: &'static MessageField = &MessageField {
            name: "fields",
            json_name: "fields",
            number: 8i32,
            message_fields: Some(FieldDescriptor::FIELDS),
        };
        pub const VARIANTS_FIELD: &'static MessageField = &MessageField {
            name: "variants",
            json_name: "variants",
            number: 9i32,
            message_fields: Some(VariantDescriptor::FIELDS),
        };
    }
    impl MessageFields for DatatypeDescriptor {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::TYPE_NAME_FIELD,
            Self::DEFINING_ID_FIELD,
            Self::MODULE_FIELD,
            Self::NAME_FIELD,
            Self::ABILITIES_FIELD,
            Self::TYPE_PARAMETERS_FIELD,
            Self::KIND_FIELD,
            Self::FIELDS_FIELD,
            Self::VARIANTS_FIELD,
        ];
    }
    impl DatatypeDescriptor {
        pub fn path_builder() -> DatatypeDescriptorFieldPathBuilder {
            DatatypeDescriptorFieldPathBuilder::new()
        }
    }
    pub struct DatatypeDescriptorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl DatatypeDescriptorFieldPathBuilder {
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
        pub fn type_name(mut self) -> String {
            self.path.push(DatatypeDescriptor::TYPE_NAME_FIELD.name);
            self.finish()
        }
        pub fn defining_id(mut self) -> String {
            self.path.push(DatatypeDescriptor::DEFINING_ID_FIELD.name);
            self.finish()
        }
        pub fn module(mut self) -> String {
            self.path.push(DatatypeDescriptor::MODULE_FIELD.name);
            self.finish()
        }
        pub fn name(mut self) -> String {
            self.path.push(DatatypeDescriptor::NAME_FIELD.name);
            self.finish()
        }
        pub fn abilities(mut self) -> String {
            self.path.push(DatatypeDescriptor::ABILITIES_FIELD.name);
            self.finish()
        }
        pub fn type_parameters(mut self) -> TypeParameterFieldPathBuilder {
            self.path.push(DatatypeDescriptor::TYPE_PARAMETERS_FIELD.name);
            TypeParameterFieldPathBuilder::new_with_base(self.path)
        }
        pub fn kind(mut self) -> String {
            self.path.push(DatatypeDescriptor::KIND_FIELD.name);
            self.finish()
        }
        pub fn fields(mut self) -> FieldDescriptorFieldPathBuilder {
            self.path.push(DatatypeDescriptor::FIELDS_FIELD.name);
            FieldDescriptorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn variants(mut self) -> VariantDescriptorFieldPathBuilder {
            self.path.push(DatatypeDescriptor::VARIANTS_FIELD.name);
            VariantDescriptorFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl TypeParameter {
        pub const CONSTRAINTS_FIELD: &'static MessageField = &MessageField {
            name: "constraints",
            json_name: "constraints",
            number: 1i32,
            message_fields: None,
        };
        pub const IS_PHANTOM_FIELD: &'static MessageField = &MessageField {
            name: "is_phantom",
            json_name: "isPhantom",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for TypeParameter {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::CONSTRAINTS_FIELD,
            Self::IS_PHANTOM_FIELD,
        ];
    }
    impl TypeParameter {
        pub fn path_builder() -> TypeParameterFieldPathBuilder {
            TypeParameterFieldPathBuilder::new()
        }
    }
    pub struct TypeParameterFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TypeParameterFieldPathBuilder {
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
        pub fn constraints(mut self) -> String {
            self.path.push(TypeParameter::CONSTRAINTS_FIELD.name);
            self.finish()
        }
        pub fn is_phantom(mut self) -> String {
            self.path.push(TypeParameter::IS_PHANTOM_FIELD.name);
            self.finish()
        }
    }
    impl FieldDescriptor {
        pub const NAME_FIELD: &'static MessageField = &MessageField {
            name: "name",
            json_name: "name",
            number: 1i32,
            message_fields: None,
        };
        pub const POSITION_FIELD: &'static MessageField = &MessageField {
            name: "position",
            json_name: "position",
            number: 2i32,
            message_fields: None,
        };
        pub const TYPE_FIELD: &'static MessageField = &MessageField {
            name: "type",
            json_name: "type",
            number: 3i32,
            message_fields: Some(OpenSignatureBody::FIELDS),
        };
    }
    impl MessageFields for FieldDescriptor {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::NAME_FIELD,
            Self::POSITION_FIELD,
            Self::TYPE_FIELD,
        ];
    }
    impl FieldDescriptor {
        pub fn path_builder() -> FieldDescriptorFieldPathBuilder {
            FieldDescriptorFieldPathBuilder::new()
        }
    }
    pub struct FieldDescriptorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl FieldDescriptorFieldPathBuilder {
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
            self.path.push(FieldDescriptor::NAME_FIELD.name);
            self.finish()
        }
        pub fn position(mut self) -> String {
            self.path.push(FieldDescriptor::POSITION_FIELD.name);
            self.finish()
        }
        pub fn r#type(mut self) -> OpenSignatureBodyFieldPathBuilder {
            self.path.push(FieldDescriptor::TYPE_FIELD.name);
            OpenSignatureBodyFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl VariantDescriptor {
        pub const NAME_FIELD: &'static MessageField = &MessageField {
            name: "name",
            json_name: "name",
            number: 1i32,
            message_fields: None,
        };
        pub const POSITION_FIELD: &'static MessageField = &MessageField {
            name: "position",
            json_name: "position",
            number: 2i32,
            message_fields: None,
        };
        pub const FIELDS_FIELD: &'static MessageField = &MessageField {
            name: "fields",
            json_name: "fields",
            number: 3i32,
            message_fields: Some(FieldDescriptor::FIELDS),
        };
    }
    impl MessageFields for VariantDescriptor {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::NAME_FIELD,
            Self::POSITION_FIELD,
            Self::FIELDS_FIELD,
        ];
    }
    impl VariantDescriptor {
        pub fn path_builder() -> VariantDescriptorFieldPathBuilder {
            VariantDescriptorFieldPathBuilder::new()
        }
    }
    pub struct VariantDescriptorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl VariantDescriptorFieldPathBuilder {
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
            self.path.push(VariantDescriptor::NAME_FIELD.name);
            self.finish()
        }
        pub fn position(mut self) -> String {
            self.path.push(VariantDescriptor::POSITION_FIELD.name);
            self.finish()
        }
        pub fn fields(mut self) -> FieldDescriptorFieldPathBuilder {
            self.path.push(VariantDescriptor::FIELDS_FIELD.name);
            FieldDescriptorFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl OpenSignatureBody {
        pub const TYPE_FIELD: &'static MessageField = &MessageField {
            name: "type",
            json_name: "type",
            number: 1i32,
            message_fields: None,
        };
        pub const TYPE_NAME_FIELD: &'static MessageField = &MessageField {
            name: "type_name",
            json_name: "typeName",
            number: 2i32,
            message_fields: None,
        };
        pub const TYPE_PARAMETER_INSTANTIATION_FIELD: &'static MessageField = &MessageField {
            name: "type_parameter_instantiation",
            json_name: "typeParameterInstantiation",
            number: 3i32,
            message_fields: None,
        };
        pub const TYPE_PARAMETER_FIELD: &'static MessageField = &MessageField {
            name: "type_parameter",
            json_name: "typeParameter",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for OpenSignatureBody {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::TYPE_FIELD,
            Self::TYPE_NAME_FIELD,
            Self::TYPE_PARAMETER_INSTANTIATION_FIELD,
            Self::TYPE_PARAMETER_FIELD,
        ];
    }
    impl OpenSignatureBody {
        pub fn path_builder() -> OpenSignatureBodyFieldPathBuilder {
            OpenSignatureBodyFieldPathBuilder::new()
        }
    }
    pub struct OpenSignatureBodyFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl OpenSignatureBodyFieldPathBuilder {
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
        pub fn r#type(mut self) -> String {
            self.path.push(OpenSignatureBody::TYPE_FIELD.name);
            self.finish()
        }
        pub fn type_name(mut self) -> String {
            self.path.push(OpenSignatureBody::TYPE_NAME_FIELD.name);
            self.finish()
        }
        pub fn type_parameter_instantiation(mut self) -> String {
            self.path.push(OpenSignatureBody::TYPE_PARAMETER_INSTANTIATION_FIELD.name);
            self.finish()
        }
        pub fn type_parameter(mut self) -> String {
            self.path.push(OpenSignatureBody::TYPE_PARAMETER_FIELD.name);
            self.finish()
        }
    }
    impl FunctionDescriptor {
        pub const NAME_FIELD: &'static MessageField = &MessageField {
            name: "name",
            json_name: "name",
            number: 1i32,
            message_fields: None,
        };
        pub const VISIBILITY_FIELD: &'static MessageField = &MessageField {
            name: "visibility",
            json_name: "visibility",
            number: 5i32,
            message_fields: None,
        };
        pub const IS_ENTRY_FIELD: &'static MessageField = &MessageField {
            name: "is_entry",
            json_name: "isEntry",
            number: 6i32,
            message_fields: None,
        };
        pub const TYPE_PARAMETERS_FIELD: &'static MessageField = &MessageField {
            name: "type_parameters",
            json_name: "typeParameters",
            number: 7i32,
            message_fields: Some(TypeParameter::FIELDS),
        };
        pub const PARAMETERS_FIELD: &'static MessageField = &MessageField {
            name: "parameters",
            json_name: "parameters",
            number: 8i32,
            message_fields: Some(OpenSignature::FIELDS),
        };
        pub const RETURNS_FIELD: &'static MessageField = &MessageField {
            name: "returns",
            json_name: "returns",
            number: 9i32,
            message_fields: Some(OpenSignature::FIELDS),
        };
    }
    impl MessageFields for FunctionDescriptor {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::NAME_FIELD,
            Self::VISIBILITY_FIELD,
            Self::IS_ENTRY_FIELD,
            Self::TYPE_PARAMETERS_FIELD,
            Self::PARAMETERS_FIELD,
            Self::RETURNS_FIELD,
        ];
    }
    impl FunctionDescriptor {
        pub fn path_builder() -> FunctionDescriptorFieldPathBuilder {
            FunctionDescriptorFieldPathBuilder::new()
        }
    }
    pub struct FunctionDescriptorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl FunctionDescriptorFieldPathBuilder {
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
            self.path.push(FunctionDescriptor::NAME_FIELD.name);
            self.finish()
        }
        pub fn visibility(mut self) -> String {
            self.path.push(FunctionDescriptor::VISIBILITY_FIELD.name);
            self.finish()
        }
        pub fn is_entry(mut self) -> String {
            self.path.push(FunctionDescriptor::IS_ENTRY_FIELD.name);
            self.finish()
        }
        pub fn type_parameters(mut self) -> TypeParameterFieldPathBuilder {
            self.path.push(FunctionDescriptor::TYPE_PARAMETERS_FIELD.name);
            TypeParameterFieldPathBuilder::new_with_base(self.path)
        }
        pub fn parameters(mut self) -> OpenSignatureFieldPathBuilder {
            self.path.push(FunctionDescriptor::PARAMETERS_FIELD.name);
            OpenSignatureFieldPathBuilder::new_with_base(self.path)
        }
        pub fn returns(mut self) -> OpenSignatureFieldPathBuilder {
            self.path.push(FunctionDescriptor::RETURNS_FIELD.name);
            OpenSignatureFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl OpenSignature {
        pub const REFERENCE_FIELD: &'static MessageField = &MessageField {
            name: "reference",
            json_name: "reference",
            number: 1i32,
            message_fields: None,
        };
        pub const BODY_FIELD: &'static MessageField = &MessageField {
            name: "body",
            json_name: "body",
            number: 2i32,
            message_fields: Some(OpenSignatureBody::FIELDS),
        };
    }
    impl MessageFields for OpenSignature {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::REFERENCE_FIELD,
            Self::BODY_FIELD,
        ];
    }
    impl OpenSignature {
        pub fn path_builder() -> OpenSignatureFieldPathBuilder {
            OpenSignatureFieldPathBuilder::new()
        }
    }
    pub struct OpenSignatureFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl OpenSignatureFieldPathBuilder {
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
        pub fn reference(mut self) -> String {
            self.path.push(OpenSignature::REFERENCE_FIELD.name);
            self.finish()
        }
        pub fn body(mut self) -> OpenSignatureBodyFieldPathBuilder {
            self.path.push(OpenSignature::BODY_FIELD.name);
            OpenSignatureBodyFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl TypeOrigin {
        pub const MODULE_NAME_FIELD: &'static MessageField = &MessageField {
            name: "module_name",
            json_name: "moduleName",
            number: 1i32,
            message_fields: None,
        };
        pub const DATATYPE_NAME_FIELD: &'static MessageField = &MessageField {
            name: "datatype_name",
            json_name: "datatypeName",
            number: 2i32,
            message_fields: None,
        };
        pub const PACKAGE_ID_FIELD: &'static MessageField = &MessageField {
            name: "package_id",
            json_name: "packageId",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for TypeOrigin {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::MODULE_NAME_FIELD,
            Self::DATATYPE_NAME_FIELD,
            Self::PACKAGE_ID_FIELD,
        ];
    }
    impl TypeOrigin {
        pub fn path_builder() -> TypeOriginFieldPathBuilder {
            TypeOriginFieldPathBuilder::new()
        }
    }
    pub struct TypeOriginFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TypeOriginFieldPathBuilder {
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
        pub fn module_name(mut self) -> String {
            self.path.push(TypeOrigin::MODULE_NAME_FIELD.name);
            self.finish()
        }
        pub fn datatype_name(mut self) -> String {
            self.path.push(TypeOrigin::DATATYPE_NAME_FIELD.name);
            self.finish()
        }
        pub fn package_id(mut self) -> String {
            self.path.push(TypeOrigin::PACKAGE_ID_FIELD.name);
            self.finish()
        }
    }
    impl Linkage {
        pub const ORIGINAL_ID_FIELD: &'static MessageField = &MessageField {
            name: "original_id",
            json_name: "originalId",
            number: 1i32,
            message_fields: None,
        };
        pub const UPGRADED_ID_FIELD: &'static MessageField = &MessageField {
            name: "upgraded_id",
            json_name: "upgradedId",
            number: 2i32,
            message_fields: None,
        };
        pub const UPGRADED_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "upgraded_version",
            json_name: "upgradedVersion",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for Linkage {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ORIGINAL_ID_FIELD,
            Self::UPGRADED_ID_FIELD,
            Self::UPGRADED_VERSION_FIELD,
        ];
    }
    impl Linkage {
        pub fn path_builder() -> LinkageFieldPathBuilder {
            LinkageFieldPathBuilder::new()
        }
    }
    pub struct LinkageFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl LinkageFieldPathBuilder {
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
        pub fn original_id(mut self) -> String {
            self.path.push(Linkage::ORIGINAL_ID_FIELD.name);
            self.finish()
        }
        pub fn upgraded_id(mut self) -> String {
            self.path.push(Linkage::UPGRADED_ID_FIELD.name);
            self.finish()
        }
        pub fn upgraded_version(mut self) -> String {
            self.path.push(Linkage::UPGRADED_VERSION_FIELD.name);
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
        pub const HAS_PUBLIC_TRANSFER_FIELD: &'static MessageField = &MessageField {
            name: "has_public_transfer",
            json_name: "hasPublicTransfer",
            number: 7i32,
            message_fields: None,
        };
        pub const CONTENTS_FIELD: &'static MessageField = &MessageField {
            name: "contents",
            json_name: "contents",
            number: 8i32,
            message_fields: Some(Bcs::FIELDS),
        };
        pub const PACKAGE_FIELD: &'static MessageField = &MessageField {
            name: "package",
            json_name: "package",
            number: 9i32,
            message_fields: Some(Package::FIELDS),
        };
        pub const PREVIOUS_TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "previous_transaction",
            json_name: "previousTransaction",
            number: 10i32,
            message_fields: None,
        };
        pub const STORAGE_REBATE_FIELD: &'static MessageField = &MessageField {
            name: "storage_rebate",
            json_name: "storageRebate",
            number: 11i32,
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
            Self::HAS_PUBLIC_TRANSFER_FIELD,
            Self::CONTENTS_FIELD,
            Self::PACKAGE_FIELD,
            Self::PREVIOUS_TRANSACTION_FIELD,
            Self::STORAGE_REBATE_FIELD,
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
        pub fn has_public_transfer(mut self) -> String {
            self.path.push(Object::HAS_PUBLIC_TRANSFER_FIELD.name);
            self.finish()
        }
        pub fn contents(mut self) -> BcsFieldPathBuilder {
            self.path.push(Object::CONTENTS_FIELD.name);
            BcsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn package(mut self) -> PackageFieldPathBuilder {
            self.path.push(Object::PACKAGE_FIELD.name);
            PackageFieldPathBuilder::new_with_base(self.path)
        }
        pub fn previous_transaction(mut self) -> String {
            self.path.push(Object::PREVIOUS_TRANSACTION_FIELD.name);
            self.finish()
        }
        pub fn storage_rebate(mut self) -> String {
            self.path.push(Object::STORAGE_REBATE_FIELD.name);
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
        pub const MULTISIG_FIELD: &'static MessageField = &MessageField {
            name: "multisig",
            json_name: "multisig",
            number: 4i32,
            message_fields: Some(MultisigAggregatedSignature::FIELDS),
        };
        pub const ZKLOGIN_FIELD: &'static MessageField = &MessageField {
            name: "zklogin",
            json_name: "zklogin",
            number: 5i32,
            message_fields: Some(ZkLoginAuthenticator::FIELDS),
        };
        pub const PASSKEY_FIELD: &'static MessageField = &MessageField {
            name: "passkey",
            json_name: "passkey",
            number: 6i32,
            message_fields: Some(PasskeyAuthenticator::FIELDS),
        };
    }
    impl MessageFields for UserSignature {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BCS_FIELD,
            Self::SCHEME_FIELD,
            Self::SIMPLE_FIELD,
            Self::MULTISIG_FIELD,
            Self::ZKLOGIN_FIELD,
            Self::PASSKEY_FIELD,
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
        pub fn multisig(mut self) -> MultisigAggregatedSignatureFieldPathBuilder {
            self.path.push(UserSignature::MULTISIG_FIELD.name);
            MultisigAggregatedSignatureFieldPathBuilder::new_with_base(self.path)
        }
        pub fn zklogin(mut self) -> ZkLoginAuthenticatorFieldPathBuilder {
            self.path.push(UserSignature::ZKLOGIN_FIELD.name);
            ZkLoginAuthenticatorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn passkey(mut self) -> PasskeyAuthenticatorFieldPathBuilder {
            self.path.push(UserSignature::PASSKEY_FIELD.name);
            PasskeyAuthenticatorFieldPathBuilder::new_with_base(self.path)
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
    impl ZkLoginPublicIdentifier {
        pub const ISS_FIELD: &'static MessageField = &MessageField {
            name: "iss",
            json_name: "iss",
            number: 1i32,
            message_fields: None,
        };
        pub const ADDRESS_SEED_FIELD: &'static MessageField = &MessageField {
            name: "address_seed",
            json_name: "addressSeed",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for ZkLoginPublicIdentifier {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ISS_FIELD,
            Self::ADDRESS_SEED_FIELD,
        ];
    }
    impl ZkLoginPublicIdentifier {
        pub fn path_builder() -> ZkLoginPublicIdentifierFieldPathBuilder {
            ZkLoginPublicIdentifierFieldPathBuilder::new()
        }
    }
    pub struct ZkLoginPublicIdentifierFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ZkLoginPublicIdentifierFieldPathBuilder {
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
        pub fn iss(mut self) -> String {
            self.path.push(ZkLoginPublicIdentifier::ISS_FIELD.name);
            self.finish()
        }
        pub fn address_seed(mut self) -> String {
            self.path.push(ZkLoginPublicIdentifier::ADDRESS_SEED_FIELD.name);
            self.finish()
        }
    }
    impl MultisigMemberPublicKey {
        pub const SCHEME_FIELD: &'static MessageField = &MessageField {
            name: "scheme",
            json_name: "scheme",
            number: 1i32,
            message_fields: None,
        };
        pub const PUBLIC_KEY_FIELD: &'static MessageField = &MessageField {
            name: "public_key",
            json_name: "publicKey",
            number: 2i32,
            message_fields: None,
        };
        pub const ZKLOGIN_FIELD: &'static MessageField = &MessageField {
            name: "zklogin",
            json_name: "zklogin",
            number: 3i32,
            message_fields: Some(ZkLoginPublicIdentifier::FIELDS),
        };
    }
    impl MessageFields for MultisigMemberPublicKey {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SCHEME_FIELD,
            Self::PUBLIC_KEY_FIELD,
            Self::ZKLOGIN_FIELD,
        ];
    }
    impl MultisigMemberPublicKey {
        pub fn path_builder() -> MultisigMemberPublicKeyFieldPathBuilder {
            MultisigMemberPublicKeyFieldPathBuilder::new()
        }
    }
    pub struct MultisigMemberPublicKeyFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl MultisigMemberPublicKeyFieldPathBuilder {
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
            self.path.push(MultisigMemberPublicKey::SCHEME_FIELD.name);
            self.finish()
        }
        pub fn public_key(mut self) -> String {
            self.path.push(MultisigMemberPublicKey::PUBLIC_KEY_FIELD.name);
            self.finish()
        }
        pub fn zklogin(mut self) -> ZkLoginPublicIdentifierFieldPathBuilder {
            self.path.push(MultisigMemberPublicKey::ZKLOGIN_FIELD.name);
            ZkLoginPublicIdentifierFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl MultisigMember {
        pub const PUBLIC_KEY_FIELD: &'static MessageField = &MessageField {
            name: "public_key",
            json_name: "publicKey",
            number: 1i32,
            message_fields: Some(MultisigMemberPublicKey::FIELDS),
        };
        pub const WEIGHT_FIELD: &'static MessageField = &MessageField {
            name: "weight",
            json_name: "weight",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for MultisigMember {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::PUBLIC_KEY_FIELD,
            Self::WEIGHT_FIELD,
        ];
    }
    impl MultisigMember {
        pub fn path_builder() -> MultisigMemberFieldPathBuilder {
            MultisigMemberFieldPathBuilder::new()
        }
    }
    pub struct MultisigMemberFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl MultisigMemberFieldPathBuilder {
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
        pub fn public_key(mut self) -> MultisigMemberPublicKeyFieldPathBuilder {
            self.path.push(MultisigMember::PUBLIC_KEY_FIELD.name);
            MultisigMemberPublicKeyFieldPathBuilder::new_with_base(self.path)
        }
        pub fn weight(mut self) -> String {
            self.path.push(MultisigMember::WEIGHT_FIELD.name);
            self.finish()
        }
    }
    impl MultisigCommittee {
        pub const MEMBERS_FIELD: &'static MessageField = &MessageField {
            name: "members",
            json_name: "members",
            number: 1i32,
            message_fields: Some(MultisigMember::FIELDS),
        };
        pub const THRESHOLD_FIELD: &'static MessageField = &MessageField {
            name: "threshold",
            json_name: "threshold",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for MultisigCommittee {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::MEMBERS_FIELD,
            Self::THRESHOLD_FIELD,
        ];
    }
    impl MultisigCommittee {
        pub fn path_builder() -> MultisigCommitteeFieldPathBuilder {
            MultisigCommitteeFieldPathBuilder::new()
        }
    }
    pub struct MultisigCommitteeFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl MultisigCommitteeFieldPathBuilder {
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
        pub fn members(mut self) -> MultisigMemberFieldPathBuilder {
            self.path.push(MultisigCommittee::MEMBERS_FIELD.name);
            MultisigMemberFieldPathBuilder::new_with_base(self.path)
        }
        pub fn threshold(mut self) -> String {
            self.path.push(MultisigCommittee::THRESHOLD_FIELD.name);
            self.finish()
        }
    }
    impl MultisigAggregatedSignature {
        pub const SIGNATURES_FIELD: &'static MessageField = &MessageField {
            name: "signatures",
            json_name: "signatures",
            number: 1i32,
            message_fields: Some(MultisigMemberSignature::FIELDS),
        };
        pub const BITMAP_FIELD: &'static MessageField = &MessageField {
            name: "bitmap",
            json_name: "bitmap",
            number: 2i32,
            message_fields: None,
        };
        pub const LEGACY_BITMAP_FIELD: &'static MessageField = &MessageField {
            name: "legacy_bitmap",
            json_name: "legacyBitmap",
            number: 3i32,
            message_fields: None,
        };
        pub const COMMITTEE_FIELD: &'static MessageField = &MessageField {
            name: "committee",
            json_name: "committee",
            number: 4i32,
            message_fields: Some(MultisigCommittee::FIELDS),
        };
    }
    impl MessageFields for MultisigAggregatedSignature {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SIGNATURES_FIELD,
            Self::BITMAP_FIELD,
            Self::LEGACY_BITMAP_FIELD,
            Self::COMMITTEE_FIELD,
        ];
    }
    impl MultisigAggregatedSignature {
        pub fn path_builder() -> MultisigAggregatedSignatureFieldPathBuilder {
            MultisigAggregatedSignatureFieldPathBuilder::new()
        }
    }
    pub struct MultisigAggregatedSignatureFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl MultisigAggregatedSignatureFieldPathBuilder {
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
        pub fn signatures(mut self) -> MultisigMemberSignatureFieldPathBuilder {
            self.path.push(MultisigAggregatedSignature::SIGNATURES_FIELD.name);
            MultisigMemberSignatureFieldPathBuilder::new_with_base(self.path)
        }
        pub fn bitmap(mut self) -> String {
            self.path.push(MultisigAggregatedSignature::BITMAP_FIELD.name);
            self.finish()
        }
        pub fn legacy_bitmap(mut self) -> String {
            self.path.push(MultisigAggregatedSignature::LEGACY_BITMAP_FIELD.name);
            self.finish()
        }
        pub fn committee(mut self) -> MultisigCommitteeFieldPathBuilder {
            self.path.push(MultisigAggregatedSignature::COMMITTEE_FIELD.name);
            MultisigCommitteeFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl MultisigMemberSignature {
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
        pub const ZKLOGIN_FIELD: &'static MessageField = &MessageField {
            name: "zklogin",
            json_name: "zklogin",
            number: 3i32,
            message_fields: Some(ZkLoginAuthenticator::FIELDS),
        };
        pub const PASSKEY_FIELD: &'static MessageField = &MessageField {
            name: "passkey",
            json_name: "passkey",
            number: 4i32,
            message_fields: Some(PasskeyAuthenticator::FIELDS),
        };
    }
    impl MessageFields for MultisigMemberSignature {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::SCHEME_FIELD,
            Self::SIGNATURE_FIELD,
            Self::ZKLOGIN_FIELD,
            Self::PASSKEY_FIELD,
        ];
    }
    impl MultisigMemberSignature {
        pub fn path_builder() -> MultisigMemberSignatureFieldPathBuilder {
            MultisigMemberSignatureFieldPathBuilder::new()
        }
    }
    pub struct MultisigMemberSignatureFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl MultisigMemberSignatureFieldPathBuilder {
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
            self.path.push(MultisigMemberSignature::SCHEME_FIELD.name);
            self.finish()
        }
        pub fn signature(mut self) -> String {
            self.path.push(MultisigMemberSignature::SIGNATURE_FIELD.name);
            self.finish()
        }
        pub fn zklogin(mut self) -> ZkLoginAuthenticatorFieldPathBuilder {
            self.path.push(MultisigMemberSignature::ZKLOGIN_FIELD.name);
            ZkLoginAuthenticatorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn passkey(mut self) -> PasskeyAuthenticatorFieldPathBuilder {
            self.path.push(MultisigMemberSignature::PASSKEY_FIELD.name);
            PasskeyAuthenticatorFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ZkLoginAuthenticator {
        pub const INPUTS_FIELD: &'static MessageField = &MessageField {
            name: "inputs",
            json_name: "inputs",
            number: 1i32,
            message_fields: Some(ZkLoginInputs::FIELDS),
        };
        pub const MAX_EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "max_epoch",
            json_name: "maxEpoch",
            number: 2i32,
            message_fields: None,
        };
        pub const SIGNATURE_FIELD: &'static MessageField = &MessageField {
            name: "signature",
            json_name: "signature",
            number: 3i32,
            message_fields: Some(SimpleSignature::FIELDS),
        };
    }
    impl MessageFields for ZkLoginAuthenticator {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::INPUTS_FIELD,
            Self::MAX_EPOCH_FIELD,
            Self::SIGNATURE_FIELD,
        ];
    }
    impl ZkLoginAuthenticator {
        pub fn path_builder() -> ZkLoginAuthenticatorFieldPathBuilder {
            ZkLoginAuthenticatorFieldPathBuilder::new()
        }
    }
    pub struct ZkLoginAuthenticatorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ZkLoginAuthenticatorFieldPathBuilder {
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
        pub fn inputs(mut self) -> ZkLoginInputsFieldPathBuilder {
            self.path.push(ZkLoginAuthenticator::INPUTS_FIELD.name);
            ZkLoginInputsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn max_epoch(mut self) -> String {
            self.path.push(ZkLoginAuthenticator::MAX_EPOCH_FIELD.name);
            self.finish()
        }
        pub fn signature(mut self) -> SimpleSignatureFieldPathBuilder {
            self.path.push(ZkLoginAuthenticator::SIGNATURE_FIELD.name);
            SimpleSignatureFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ZkLoginInputs {
        pub const PROOF_POINTS_FIELD: &'static MessageField = &MessageField {
            name: "proof_points",
            json_name: "proofPoints",
            number: 1i32,
            message_fields: Some(ZkLoginProof::FIELDS),
        };
        pub const ISS_BASE64_DETAILS_FIELD: &'static MessageField = &MessageField {
            name: "iss_base64_details",
            json_name: "issBase64Details",
            number: 2i32,
            message_fields: Some(ZkLoginClaim::FIELDS),
        };
        pub const HEADER_BASE64_FIELD: &'static MessageField = &MessageField {
            name: "header_base64",
            json_name: "headerBase64",
            number: 3i32,
            message_fields: None,
        };
        pub const ADDRESS_SEED_FIELD: &'static MessageField = &MessageField {
            name: "address_seed",
            json_name: "addressSeed",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for ZkLoginInputs {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::PROOF_POINTS_FIELD,
            Self::ISS_BASE64_DETAILS_FIELD,
            Self::HEADER_BASE64_FIELD,
            Self::ADDRESS_SEED_FIELD,
        ];
    }
    impl ZkLoginInputs {
        pub fn path_builder() -> ZkLoginInputsFieldPathBuilder {
            ZkLoginInputsFieldPathBuilder::new()
        }
    }
    pub struct ZkLoginInputsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ZkLoginInputsFieldPathBuilder {
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
        pub fn proof_points(mut self) -> ZkLoginProofFieldPathBuilder {
            self.path.push(ZkLoginInputs::PROOF_POINTS_FIELD.name);
            ZkLoginProofFieldPathBuilder::new_with_base(self.path)
        }
        pub fn iss_base64_details(mut self) -> ZkLoginClaimFieldPathBuilder {
            self.path.push(ZkLoginInputs::ISS_BASE64_DETAILS_FIELD.name);
            ZkLoginClaimFieldPathBuilder::new_with_base(self.path)
        }
        pub fn header_base64(mut self) -> String {
            self.path.push(ZkLoginInputs::HEADER_BASE64_FIELD.name);
            self.finish()
        }
        pub fn address_seed(mut self) -> String {
            self.path.push(ZkLoginInputs::ADDRESS_SEED_FIELD.name);
            self.finish()
        }
    }
    impl ZkLoginProof {
        pub const A_FIELD: &'static MessageField = &MessageField {
            name: "a",
            json_name: "a",
            number: 1i32,
            message_fields: Some(CircomG1::FIELDS),
        };
        pub const B_FIELD: &'static MessageField = &MessageField {
            name: "b",
            json_name: "b",
            number: 2i32,
            message_fields: Some(CircomG2::FIELDS),
        };
        pub const C_FIELD: &'static MessageField = &MessageField {
            name: "c",
            json_name: "c",
            number: 3i32,
            message_fields: Some(CircomG1::FIELDS),
        };
    }
    impl MessageFields for ZkLoginProof {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::A_FIELD,
            Self::B_FIELD,
            Self::C_FIELD,
        ];
    }
    impl ZkLoginProof {
        pub fn path_builder() -> ZkLoginProofFieldPathBuilder {
            ZkLoginProofFieldPathBuilder::new()
        }
    }
    pub struct ZkLoginProofFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ZkLoginProofFieldPathBuilder {
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
        pub fn a(mut self) -> CircomG1FieldPathBuilder {
            self.path.push(ZkLoginProof::A_FIELD.name);
            CircomG1FieldPathBuilder::new_with_base(self.path)
        }
        pub fn b(mut self) -> CircomG2FieldPathBuilder {
            self.path.push(ZkLoginProof::B_FIELD.name);
            CircomG2FieldPathBuilder::new_with_base(self.path)
        }
        pub fn c(mut self) -> CircomG1FieldPathBuilder {
            self.path.push(ZkLoginProof::C_FIELD.name);
            CircomG1FieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ZkLoginClaim {
        pub const VALUE_FIELD: &'static MessageField = &MessageField {
            name: "value",
            json_name: "value",
            number: 1i32,
            message_fields: None,
        };
        pub const INDEX_MOD_4_FIELD: &'static MessageField = &MessageField {
            name: "index_mod_4",
            json_name: "indexMod4",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for ZkLoginClaim {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::VALUE_FIELD,
            Self::INDEX_MOD_4_FIELD,
        ];
    }
    impl ZkLoginClaim {
        pub fn path_builder() -> ZkLoginClaimFieldPathBuilder {
            ZkLoginClaimFieldPathBuilder::new()
        }
    }
    pub struct ZkLoginClaimFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ZkLoginClaimFieldPathBuilder {
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
        pub fn value(mut self) -> String {
            self.path.push(ZkLoginClaim::VALUE_FIELD.name);
            self.finish()
        }
        pub fn index_mod_4(mut self) -> String {
            self.path.push(ZkLoginClaim::INDEX_MOD_4_FIELD.name);
            self.finish()
        }
    }
    impl CircomG1 {
        pub const E0_FIELD: &'static MessageField = &MessageField {
            name: "e0",
            json_name: "e0",
            number: 1i32,
            message_fields: None,
        };
        pub const E1_FIELD: &'static MessageField = &MessageField {
            name: "e1",
            json_name: "e1",
            number: 2i32,
            message_fields: None,
        };
        pub const E2_FIELD: &'static MessageField = &MessageField {
            name: "e2",
            json_name: "e2",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for CircomG1 {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::E0_FIELD,
            Self::E1_FIELD,
            Self::E2_FIELD,
        ];
    }
    impl CircomG1 {
        pub fn path_builder() -> CircomG1FieldPathBuilder {
            CircomG1FieldPathBuilder::new()
        }
    }
    pub struct CircomG1FieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl CircomG1FieldPathBuilder {
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
        pub fn e0(mut self) -> String {
            self.path.push(CircomG1::E0_FIELD.name);
            self.finish()
        }
        pub fn e1(mut self) -> String {
            self.path.push(CircomG1::E1_FIELD.name);
            self.finish()
        }
        pub fn e2(mut self) -> String {
            self.path.push(CircomG1::E2_FIELD.name);
            self.finish()
        }
    }
    impl CircomG2 {
        pub const E00_FIELD: &'static MessageField = &MessageField {
            name: "e00",
            json_name: "e00",
            number: 1i32,
            message_fields: None,
        };
        pub const E01_FIELD: &'static MessageField = &MessageField {
            name: "e01",
            json_name: "e01",
            number: 2i32,
            message_fields: None,
        };
        pub const E10_FIELD: &'static MessageField = &MessageField {
            name: "e10",
            json_name: "e10",
            number: 3i32,
            message_fields: None,
        };
        pub const E11_FIELD: &'static MessageField = &MessageField {
            name: "e11",
            json_name: "e11",
            number: 4i32,
            message_fields: None,
        };
        pub const E20_FIELD: &'static MessageField = &MessageField {
            name: "e20",
            json_name: "e20",
            number: 5i32,
            message_fields: None,
        };
        pub const E21_FIELD: &'static MessageField = &MessageField {
            name: "e21",
            json_name: "e21",
            number: 6i32,
            message_fields: None,
        };
    }
    impl MessageFields for CircomG2 {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::E00_FIELD,
            Self::E01_FIELD,
            Self::E10_FIELD,
            Self::E11_FIELD,
            Self::E20_FIELD,
            Self::E21_FIELD,
        ];
    }
    impl CircomG2 {
        pub fn path_builder() -> CircomG2FieldPathBuilder {
            CircomG2FieldPathBuilder::new()
        }
    }
    pub struct CircomG2FieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl CircomG2FieldPathBuilder {
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
        pub fn e00(mut self) -> String {
            self.path.push(CircomG2::E00_FIELD.name);
            self.finish()
        }
        pub fn e01(mut self) -> String {
            self.path.push(CircomG2::E01_FIELD.name);
            self.finish()
        }
        pub fn e10(mut self) -> String {
            self.path.push(CircomG2::E10_FIELD.name);
            self.finish()
        }
        pub fn e11(mut self) -> String {
            self.path.push(CircomG2::E11_FIELD.name);
            self.finish()
        }
        pub fn e20(mut self) -> String {
            self.path.push(CircomG2::E20_FIELD.name);
            self.finish()
        }
        pub fn e21(mut self) -> String {
            self.path.push(CircomG2::E21_FIELD.name);
            self.finish()
        }
    }
    impl PasskeyAuthenticator {
        pub const AUTHENTICATOR_DATA_FIELD: &'static MessageField = &MessageField {
            name: "authenticator_data",
            json_name: "authenticatorData",
            number: 1i32,
            message_fields: None,
        };
        pub const CLIENT_DATA_JSON_FIELD: &'static MessageField = &MessageField {
            name: "client_data_json",
            json_name: "clientDataJson",
            number: 2i32,
            message_fields: None,
        };
        pub const SIGNATURE_FIELD: &'static MessageField = &MessageField {
            name: "signature",
            json_name: "signature",
            number: 3i32,
            message_fields: Some(SimpleSignature::FIELDS),
        };
    }
    impl MessageFields for PasskeyAuthenticator {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::AUTHENTICATOR_DATA_FIELD,
            Self::CLIENT_DATA_JSON_FIELD,
            Self::SIGNATURE_FIELD,
        ];
    }
    impl PasskeyAuthenticator {
        pub fn path_builder() -> PasskeyAuthenticatorFieldPathBuilder {
            PasskeyAuthenticatorFieldPathBuilder::new()
        }
    }
    pub struct PasskeyAuthenticatorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl PasskeyAuthenticatorFieldPathBuilder {
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
        pub fn authenticator_data(mut self) -> String {
            self.path.push(PasskeyAuthenticator::AUTHENTICATOR_DATA_FIELD.name);
            self.finish()
        }
        pub fn client_data_json(mut self) -> String {
            self.path.push(PasskeyAuthenticator::CLIENT_DATA_JSON_FIELD.name);
            self.finish()
        }
        pub fn signature(mut self) -> SimpleSignatureFieldPathBuilder {
            self.path.push(PasskeyAuthenticator::SIGNATURE_FIELD.name);
            SimpleSignatureFieldPathBuilder::new_with_base(self.path)
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
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 3i32,
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
        pub const EXPIRATION_FIELD: &'static MessageField = &MessageField {
            name: "expiration",
            json_name: "expiration",
            number: 7i32,
            message_fields: Some(TransactionExpiration::FIELDS),
        };
    }
    impl MessageFields for Transaction {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::BCS_FIELD,
            Self::DIGEST_FIELD,
            Self::VERSION_FIELD,
            Self::KIND_FIELD,
            Self::SENDER_FIELD,
            Self::GAS_PAYMENT_FIELD,
            Self::EXPIRATION_FIELD,
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
        pub fn version(mut self) -> String {
            self.path.push(Transaction::VERSION_FIELD.name);
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
        pub fn expiration(mut self) -> TransactionExpirationFieldPathBuilder {
            self.path.push(Transaction::EXPIRATION_FIELD.name);
            TransactionExpirationFieldPathBuilder::new_with_base(self.path)
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
        pub const PRICE_FIELD: &'static MessageField = &MessageField {
            name: "price",
            json_name: "price",
            number: 3i32,
            message_fields: None,
        };
        pub const BUDGET_FIELD: &'static MessageField = &MessageField {
            name: "budget",
            json_name: "budget",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for GasPayment {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECTS_FIELD,
            Self::OWNER_FIELD,
            Self::PRICE_FIELD,
            Self::BUDGET_FIELD,
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
        pub fn price(mut self) -> String {
            self.path.push(GasPayment::PRICE_FIELD.name);
            self.finish()
        }
        pub fn budget(mut self) -> String {
            self.path.push(GasPayment::BUDGET_FIELD.name);
            self.finish()
        }
    }
    impl TransactionExpiration {
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 1i32,
            message_fields: None,
        };
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for TransactionExpiration {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::KIND_FIELD,
            Self::EPOCH_FIELD,
        ];
    }
    impl TransactionExpiration {
        pub fn path_builder() -> TransactionExpirationFieldPathBuilder {
            TransactionExpirationFieldPathBuilder::new()
        }
    }
    pub struct TransactionExpirationFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl TransactionExpirationFieldPathBuilder {
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
            self.path.push(TransactionExpiration::KIND_FIELD.name);
            self.finish()
        }
        pub fn epoch(mut self) -> String {
            self.path.push(TransactionExpiration::EPOCH_FIELD.name);
            self.finish()
        }
    }
    impl TransactionKind {
        pub const PROGRAMMABLE_TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "programmable_transaction",
            json_name: "programmableTransaction",
            number: 2i32,
            message_fields: Some(ProgrammableTransaction::FIELDS),
        };
        pub const PROGRAMMABLE_SYSTEM_TRANSACTION_FIELD: &'static MessageField = &MessageField {
            name: "programmable_system_transaction",
            json_name: "programmableSystemTransaction",
            number: 3i32,
            message_fields: Some(ProgrammableTransaction::FIELDS),
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
        pub const CONSENSUS_COMMIT_PROLOGUE_V1_FIELD: &'static MessageField = &MessageField {
            name: "consensus_commit_prologue_v1",
            json_name: "consensusCommitPrologueV1",
            number: 102i32,
            message_fields: Some(ConsensusCommitPrologue::FIELDS),
        };
        pub const AUTHENTICATOR_STATE_UPDATE_FIELD: &'static MessageField = &MessageField {
            name: "authenticator_state_update",
            json_name: "authenticatorStateUpdate",
            number: 103i32,
            message_fields: Some(AuthenticatorStateUpdate::FIELDS),
        };
        pub const END_OF_EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "end_of_epoch",
            json_name: "endOfEpoch",
            number: 104i32,
            message_fields: Some(EndOfEpochTransaction::FIELDS),
        };
        pub const RANDOMNESS_STATE_UPDATE_FIELD: &'static MessageField = &MessageField {
            name: "randomness_state_update",
            json_name: "randomnessStateUpdate",
            number: 105i32,
            message_fields: Some(RandomnessStateUpdate::FIELDS),
        };
        pub const CONSENSUS_COMMIT_PROLOGUE_V2_FIELD: &'static MessageField = &MessageField {
            name: "consensus_commit_prologue_v2",
            json_name: "consensusCommitPrologueV2",
            number: 106i32,
            message_fields: Some(ConsensusCommitPrologue::FIELDS),
        };
        pub const CONSENSUS_COMMIT_PROLOGUE_V3_FIELD: &'static MessageField = &MessageField {
            name: "consensus_commit_prologue_v3",
            json_name: "consensusCommitPrologueV3",
            number: 107i32,
            message_fields: Some(ConsensusCommitPrologue::FIELDS),
        };
        pub const CONSENSUS_COMMIT_PROLOGUE_V4_FIELD: &'static MessageField = &MessageField {
            name: "consensus_commit_prologue_v4",
            json_name: "consensusCommitPrologueV4",
            number: 108i32,
            message_fields: Some(ConsensusCommitPrologue::FIELDS),
        };
    }
    impl MessageFields for TransactionKind {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::PROGRAMMABLE_TRANSACTION_FIELD,
            Self::PROGRAMMABLE_SYSTEM_TRANSACTION_FIELD,
            Self::CHANGE_EPOCH_FIELD,
            Self::GENESIS_FIELD,
            Self::CONSENSUS_COMMIT_PROLOGUE_V1_FIELD,
            Self::AUTHENTICATOR_STATE_UPDATE_FIELD,
            Self::END_OF_EPOCH_FIELD,
            Self::RANDOMNESS_STATE_UPDATE_FIELD,
            Self::CONSENSUS_COMMIT_PROLOGUE_V2_FIELD,
            Self::CONSENSUS_COMMIT_PROLOGUE_V3_FIELD,
            Self::CONSENSUS_COMMIT_PROLOGUE_V4_FIELD,
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
        pub fn programmable_transaction(
            mut self,
        ) -> ProgrammableTransactionFieldPathBuilder {
            self.path.push(TransactionKind::PROGRAMMABLE_TRANSACTION_FIELD.name);
            ProgrammableTransactionFieldPathBuilder::new_with_base(self.path)
        }
        pub fn programmable_system_transaction(
            mut self,
        ) -> ProgrammableTransactionFieldPathBuilder {
            self.path.push(TransactionKind::PROGRAMMABLE_SYSTEM_TRANSACTION_FIELD.name);
            ProgrammableTransactionFieldPathBuilder::new_with_base(self.path)
        }
        pub fn change_epoch(mut self) -> ChangeEpochFieldPathBuilder {
            self.path.push(TransactionKind::CHANGE_EPOCH_FIELD.name);
            ChangeEpochFieldPathBuilder::new_with_base(self.path)
        }
        pub fn genesis(mut self) -> GenesisTransactionFieldPathBuilder {
            self.path.push(TransactionKind::GENESIS_FIELD.name);
            GenesisTransactionFieldPathBuilder::new_with_base(self.path)
        }
        pub fn consensus_commit_prologue_v1(
            mut self,
        ) -> ConsensusCommitPrologueFieldPathBuilder {
            self.path.push(TransactionKind::CONSENSUS_COMMIT_PROLOGUE_V1_FIELD.name);
            ConsensusCommitPrologueFieldPathBuilder::new_with_base(self.path)
        }
        pub fn authenticator_state_update(
            mut self,
        ) -> AuthenticatorStateUpdateFieldPathBuilder {
            self.path.push(TransactionKind::AUTHENTICATOR_STATE_UPDATE_FIELD.name);
            AuthenticatorStateUpdateFieldPathBuilder::new_with_base(self.path)
        }
        pub fn end_of_epoch(mut self) -> EndOfEpochTransactionFieldPathBuilder {
            self.path.push(TransactionKind::END_OF_EPOCH_FIELD.name);
            EndOfEpochTransactionFieldPathBuilder::new_with_base(self.path)
        }
        pub fn randomness_state_update(
            mut self,
        ) -> RandomnessStateUpdateFieldPathBuilder {
            self.path.push(TransactionKind::RANDOMNESS_STATE_UPDATE_FIELD.name);
            RandomnessStateUpdateFieldPathBuilder::new_with_base(self.path)
        }
        pub fn consensus_commit_prologue_v2(
            mut self,
        ) -> ConsensusCommitPrologueFieldPathBuilder {
            self.path.push(TransactionKind::CONSENSUS_COMMIT_PROLOGUE_V2_FIELD.name);
            ConsensusCommitPrologueFieldPathBuilder::new_with_base(self.path)
        }
        pub fn consensus_commit_prologue_v3(
            mut self,
        ) -> ConsensusCommitPrologueFieldPathBuilder {
            self.path.push(TransactionKind::CONSENSUS_COMMIT_PROLOGUE_V3_FIELD.name);
            ConsensusCommitPrologueFieldPathBuilder::new_with_base(self.path)
        }
        pub fn consensus_commit_prologue_v4(
            mut self,
        ) -> ConsensusCommitPrologueFieldPathBuilder {
            self.path.push(TransactionKind::CONSENSUS_COMMIT_PROLOGUE_V4_FIELD.name);
            ConsensusCommitPrologueFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ProgrammableTransaction {
        pub const INPUTS_FIELD: &'static MessageField = &MessageField {
            name: "inputs",
            json_name: "inputs",
            number: 1i32,
            message_fields: Some(Input::FIELDS),
        };
        pub const COMMANDS_FIELD: &'static MessageField = &MessageField {
            name: "commands",
            json_name: "commands",
            number: 2i32,
            message_fields: Some(Command::FIELDS),
        };
    }
    impl MessageFields for ProgrammableTransaction {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::INPUTS_FIELD,
            Self::COMMANDS_FIELD,
        ];
    }
    impl ProgrammableTransaction {
        pub fn path_builder() -> ProgrammableTransactionFieldPathBuilder {
            ProgrammableTransactionFieldPathBuilder::new()
        }
    }
    pub struct ProgrammableTransactionFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ProgrammableTransactionFieldPathBuilder {
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
        pub fn inputs(mut self) -> InputFieldPathBuilder {
            self.path.push(ProgrammableTransaction::INPUTS_FIELD.name);
            InputFieldPathBuilder::new_with_base(self.path)
        }
        pub fn commands(mut self) -> CommandFieldPathBuilder {
            self.path.push(ProgrammableTransaction::COMMANDS_FIELD.name);
            CommandFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl Command {
        pub const MOVE_CALL_FIELD: &'static MessageField = &MessageField {
            name: "move_call",
            json_name: "moveCall",
            number: 1i32,
            message_fields: Some(MoveCall::FIELDS),
        };
        pub const TRANSFER_OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "transfer_objects",
            json_name: "transferObjects",
            number: 2i32,
            message_fields: Some(TransferObjects::FIELDS),
        };
        pub const SPLIT_COINS_FIELD: &'static MessageField = &MessageField {
            name: "split_coins",
            json_name: "splitCoins",
            number: 3i32,
            message_fields: Some(SplitCoins::FIELDS),
        };
        pub const MERGE_COINS_FIELD: &'static MessageField = &MessageField {
            name: "merge_coins",
            json_name: "mergeCoins",
            number: 4i32,
            message_fields: Some(MergeCoins::FIELDS),
        };
        pub const PUBLISH_FIELD: &'static MessageField = &MessageField {
            name: "publish",
            json_name: "publish",
            number: 5i32,
            message_fields: Some(Publish::FIELDS),
        };
        pub const MAKE_MOVE_VECTOR_FIELD: &'static MessageField = &MessageField {
            name: "make_move_vector",
            json_name: "makeMoveVector",
            number: 6i32,
            message_fields: Some(MakeMoveVector::FIELDS),
        };
        pub const UPGRADE_FIELD: &'static MessageField = &MessageField {
            name: "upgrade",
            json_name: "upgrade",
            number: 7i32,
            message_fields: Some(Upgrade::FIELDS),
        };
    }
    impl MessageFields for Command {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::MOVE_CALL_FIELD,
            Self::TRANSFER_OBJECTS_FIELD,
            Self::SPLIT_COINS_FIELD,
            Self::MERGE_COINS_FIELD,
            Self::PUBLISH_FIELD,
            Self::MAKE_MOVE_VECTOR_FIELD,
            Self::UPGRADE_FIELD,
        ];
    }
    impl Command {
        pub fn path_builder() -> CommandFieldPathBuilder {
            CommandFieldPathBuilder::new()
        }
    }
    pub struct CommandFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl CommandFieldPathBuilder {
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
        pub fn move_call(mut self) -> MoveCallFieldPathBuilder {
            self.path.push(Command::MOVE_CALL_FIELD.name);
            MoveCallFieldPathBuilder::new_with_base(self.path)
        }
        pub fn transfer_objects(mut self) -> TransferObjectsFieldPathBuilder {
            self.path.push(Command::TRANSFER_OBJECTS_FIELD.name);
            TransferObjectsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn split_coins(mut self) -> SplitCoinsFieldPathBuilder {
            self.path.push(Command::SPLIT_COINS_FIELD.name);
            SplitCoinsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn merge_coins(mut self) -> MergeCoinsFieldPathBuilder {
            self.path.push(Command::MERGE_COINS_FIELD.name);
            MergeCoinsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn publish(mut self) -> PublishFieldPathBuilder {
            self.path.push(Command::PUBLISH_FIELD.name);
            PublishFieldPathBuilder::new_with_base(self.path)
        }
        pub fn make_move_vector(mut self) -> MakeMoveVectorFieldPathBuilder {
            self.path.push(Command::MAKE_MOVE_VECTOR_FIELD.name);
            MakeMoveVectorFieldPathBuilder::new_with_base(self.path)
        }
        pub fn upgrade(mut self) -> UpgradeFieldPathBuilder {
            self.path.push(Command::UPGRADE_FIELD.name);
            UpgradeFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl MoveCall {
        pub const PACKAGE_FIELD: &'static MessageField = &MessageField {
            name: "package",
            json_name: "package",
            number: 1i32,
            message_fields: None,
        };
        pub const MODULE_FIELD: &'static MessageField = &MessageField {
            name: "module",
            json_name: "module",
            number: 2i32,
            message_fields: None,
        };
        pub const FUNCTION_FIELD: &'static MessageField = &MessageField {
            name: "function",
            json_name: "function",
            number: 3i32,
            message_fields: None,
        };
        pub const TYPE_ARGUMENTS_FIELD: &'static MessageField = &MessageField {
            name: "type_arguments",
            json_name: "typeArguments",
            number: 4i32,
            message_fields: None,
        };
        pub const ARGUMENTS_FIELD: &'static MessageField = &MessageField {
            name: "arguments",
            json_name: "arguments",
            number: 5i32,
            message_fields: Some(Argument::FIELDS),
        };
    }
    impl MessageFields for MoveCall {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::PACKAGE_FIELD,
            Self::MODULE_FIELD,
            Self::FUNCTION_FIELD,
            Self::TYPE_ARGUMENTS_FIELD,
            Self::ARGUMENTS_FIELD,
        ];
    }
    impl MoveCall {
        pub fn path_builder() -> MoveCallFieldPathBuilder {
            MoveCallFieldPathBuilder::new()
        }
    }
    pub struct MoveCallFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl MoveCallFieldPathBuilder {
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
        pub fn package(mut self) -> String {
            self.path.push(MoveCall::PACKAGE_FIELD.name);
            self.finish()
        }
        pub fn module(mut self) -> String {
            self.path.push(MoveCall::MODULE_FIELD.name);
            self.finish()
        }
        pub fn function(mut self) -> String {
            self.path.push(MoveCall::FUNCTION_FIELD.name);
            self.finish()
        }
        pub fn type_arguments(mut self) -> String {
            self.path.push(MoveCall::TYPE_ARGUMENTS_FIELD.name);
            self.finish()
        }
        pub fn arguments(mut self) -> ArgumentFieldPathBuilder {
            self.path.push(MoveCall::ARGUMENTS_FIELD.name);
            ArgumentFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl TransferObjects {
        pub const OBJECTS_FIELD: &'static MessageField = &MessageField {
            name: "objects",
            json_name: "objects",
            number: 1i32,
            message_fields: Some(Argument::FIELDS),
        };
        pub const ADDRESS_FIELD: &'static MessageField = &MessageField {
            name: "address",
            json_name: "address",
            number: 2i32,
            message_fields: Some(Argument::FIELDS),
        };
    }
    impl MessageFields for TransferObjects {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECTS_FIELD,
            Self::ADDRESS_FIELD,
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
        pub fn objects(mut self) -> ArgumentFieldPathBuilder {
            self.path.push(TransferObjects::OBJECTS_FIELD.name);
            ArgumentFieldPathBuilder::new_with_base(self.path)
        }
        pub fn address(mut self) -> ArgumentFieldPathBuilder {
            self.path.push(TransferObjects::ADDRESS_FIELD.name);
            ArgumentFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl SplitCoins {
        pub const COIN_FIELD: &'static MessageField = &MessageField {
            name: "coin",
            json_name: "coin",
            number: 1i32,
            message_fields: Some(Argument::FIELDS),
        };
        pub const AMOUNTS_FIELD: &'static MessageField = &MessageField {
            name: "amounts",
            json_name: "amounts",
            number: 2i32,
            message_fields: Some(Argument::FIELDS),
        };
    }
    impl MessageFields for SplitCoins {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::COIN_FIELD,
            Self::AMOUNTS_FIELD,
        ];
    }
    impl SplitCoins {
        pub fn path_builder() -> SplitCoinsFieldPathBuilder {
            SplitCoinsFieldPathBuilder::new()
        }
    }
    pub struct SplitCoinsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SplitCoinsFieldPathBuilder {
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
        pub fn coin(mut self) -> ArgumentFieldPathBuilder {
            self.path.push(SplitCoins::COIN_FIELD.name);
            ArgumentFieldPathBuilder::new_with_base(self.path)
        }
        pub fn amounts(mut self) -> ArgumentFieldPathBuilder {
            self.path.push(SplitCoins::AMOUNTS_FIELD.name);
            ArgumentFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl MergeCoins {
        pub const COIN_FIELD: &'static MessageField = &MessageField {
            name: "coin",
            json_name: "coin",
            number: 1i32,
            message_fields: Some(Argument::FIELDS),
        };
        pub const COINS_TO_MERGE_FIELD: &'static MessageField = &MessageField {
            name: "coins_to_merge",
            json_name: "coinsToMerge",
            number: 2i32,
            message_fields: Some(Argument::FIELDS),
        };
    }
    impl MessageFields for MergeCoins {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::COIN_FIELD,
            Self::COINS_TO_MERGE_FIELD,
        ];
    }
    impl MergeCoins {
        pub fn path_builder() -> MergeCoinsFieldPathBuilder {
            MergeCoinsFieldPathBuilder::new()
        }
    }
    pub struct MergeCoinsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl MergeCoinsFieldPathBuilder {
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
        pub fn coin(mut self) -> ArgumentFieldPathBuilder {
            self.path.push(MergeCoins::COIN_FIELD.name);
            ArgumentFieldPathBuilder::new_with_base(self.path)
        }
        pub fn coins_to_merge(mut self) -> ArgumentFieldPathBuilder {
            self.path.push(MergeCoins::COINS_TO_MERGE_FIELD.name);
            ArgumentFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl Publish {
        pub const MODULES_FIELD: &'static MessageField = &MessageField {
            name: "modules",
            json_name: "modules",
            number: 1i32,
            message_fields: None,
        };
        pub const DEPENDENCIES_FIELD: &'static MessageField = &MessageField {
            name: "dependencies",
            json_name: "dependencies",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for Publish {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::MODULES_FIELD,
            Self::DEPENDENCIES_FIELD,
        ];
    }
    impl Publish {
        pub fn path_builder() -> PublishFieldPathBuilder {
            PublishFieldPathBuilder::new()
        }
    }
    pub struct PublishFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl PublishFieldPathBuilder {
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
        pub fn modules(mut self) -> String {
            self.path.push(Publish::MODULES_FIELD.name);
            self.finish()
        }
        pub fn dependencies(mut self) -> String {
            self.path.push(Publish::DEPENDENCIES_FIELD.name);
            self.finish()
        }
    }
    impl MakeMoveVector {
        pub const ELEMENT_TYPE_FIELD: &'static MessageField = &MessageField {
            name: "element_type",
            json_name: "elementType",
            number: 1i32,
            message_fields: None,
        };
        pub const ELEMENTS_FIELD: &'static MessageField = &MessageField {
            name: "elements",
            json_name: "elements",
            number: 2i32,
            message_fields: Some(Argument::FIELDS),
        };
    }
    impl MessageFields for MakeMoveVector {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ELEMENT_TYPE_FIELD,
            Self::ELEMENTS_FIELD,
        ];
    }
    impl MakeMoveVector {
        pub fn path_builder() -> MakeMoveVectorFieldPathBuilder {
            MakeMoveVectorFieldPathBuilder::new()
        }
    }
    pub struct MakeMoveVectorFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl MakeMoveVectorFieldPathBuilder {
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
        pub fn element_type(mut self) -> String {
            self.path.push(MakeMoveVector::ELEMENT_TYPE_FIELD.name);
            self.finish()
        }
        pub fn elements(mut self) -> ArgumentFieldPathBuilder {
            self.path.push(MakeMoveVector::ELEMENTS_FIELD.name);
            ArgumentFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl Upgrade {
        pub const MODULES_FIELD: &'static MessageField = &MessageField {
            name: "modules",
            json_name: "modules",
            number: 1i32,
            message_fields: None,
        };
        pub const DEPENDENCIES_FIELD: &'static MessageField = &MessageField {
            name: "dependencies",
            json_name: "dependencies",
            number: 2i32,
            message_fields: None,
        };
        pub const PACKAGE_FIELD: &'static MessageField = &MessageField {
            name: "package",
            json_name: "package",
            number: 3i32,
            message_fields: None,
        };
        pub const TICKET_FIELD: &'static MessageField = &MessageField {
            name: "ticket",
            json_name: "ticket",
            number: 4i32,
            message_fields: Some(Argument::FIELDS),
        };
    }
    impl MessageFields for Upgrade {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::MODULES_FIELD,
            Self::DEPENDENCIES_FIELD,
            Self::PACKAGE_FIELD,
            Self::TICKET_FIELD,
        ];
    }
    impl Upgrade {
        pub fn path_builder() -> UpgradeFieldPathBuilder {
            UpgradeFieldPathBuilder::new()
        }
    }
    pub struct UpgradeFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl UpgradeFieldPathBuilder {
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
        pub fn modules(mut self) -> String {
            self.path.push(Upgrade::MODULES_FIELD.name);
            self.finish()
        }
        pub fn dependencies(mut self) -> String {
            self.path.push(Upgrade::DEPENDENCIES_FIELD.name);
            self.finish()
        }
        pub fn package(mut self) -> String {
            self.path.push(Upgrade::PACKAGE_FIELD.name);
            self.finish()
        }
        pub fn ticket(mut self) -> ArgumentFieldPathBuilder {
            self.path.push(Upgrade::TICKET_FIELD.name);
            ArgumentFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl RandomnessStateUpdate {
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 1i32,
            message_fields: None,
        };
        pub const RANDOMNESS_ROUND_FIELD: &'static MessageField = &MessageField {
            name: "randomness_round",
            json_name: "randomnessRound",
            number: 2i32,
            message_fields: None,
        };
        pub const RANDOM_BYTES_FIELD: &'static MessageField = &MessageField {
            name: "random_bytes",
            json_name: "randomBytes",
            number: 3i32,
            message_fields: None,
        };
        pub const RANDOMNESS_OBJECT_INITIAL_SHARED_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "randomness_object_initial_shared_version",
            json_name: "randomnessObjectInitialSharedVersion",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for RandomnessStateUpdate {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_FIELD,
            Self::RANDOMNESS_ROUND_FIELD,
            Self::RANDOM_BYTES_FIELD,
            Self::RANDOMNESS_OBJECT_INITIAL_SHARED_VERSION_FIELD,
        ];
    }
    impl RandomnessStateUpdate {
        pub fn path_builder() -> RandomnessStateUpdateFieldPathBuilder {
            RandomnessStateUpdateFieldPathBuilder::new()
        }
    }
    pub struct RandomnessStateUpdateFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl RandomnessStateUpdateFieldPathBuilder {
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
            self.path.push(RandomnessStateUpdate::EPOCH_FIELD.name);
            self.finish()
        }
        pub fn randomness_round(mut self) -> String {
            self.path.push(RandomnessStateUpdate::RANDOMNESS_ROUND_FIELD.name);
            self.finish()
        }
        pub fn random_bytes(mut self) -> String {
            self.path.push(RandomnessStateUpdate::RANDOM_BYTES_FIELD.name);
            self.finish()
        }
        pub fn randomness_object_initial_shared_version(mut self) -> String {
            self.path
                .push(
                    RandomnessStateUpdate::RANDOMNESS_OBJECT_INITIAL_SHARED_VERSION_FIELD
                        .name,
                );
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
        pub const PROTOCOL_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "protocol_version",
            json_name: "protocolVersion",
            number: 2i32,
            message_fields: None,
        };
        pub const STORAGE_CHARGE_FIELD: &'static MessageField = &MessageField {
            name: "storage_charge",
            json_name: "storageCharge",
            number: 3i32,
            message_fields: None,
        };
        pub const COMPUTATION_CHARGE_FIELD: &'static MessageField = &MessageField {
            name: "computation_charge",
            json_name: "computationCharge",
            number: 4i32,
            message_fields: None,
        };
        pub const STORAGE_REBATE_FIELD: &'static MessageField = &MessageField {
            name: "storage_rebate",
            json_name: "storageRebate",
            number: 5i32,
            message_fields: None,
        };
        pub const NON_REFUNDABLE_STORAGE_FEE_FIELD: &'static MessageField = &MessageField {
            name: "non_refundable_storage_fee",
            json_name: "nonRefundableStorageFee",
            number: 6i32,
            message_fields: None,
        };
        pub const EPOCH_START_TIMESTAMP_FIELD: &'static MessageField = &MessageField {
            name: "epoch_start_timestamp",
            json_name: "epochStartTimestamp",
            number: 7i32,
            message_fields: None,
        };
        pub const SYSTEM_PACKAGES_FIELD: &'static MessageField = &MessageField {
            name: "system_packages",
            json_name: "systemPackages",
            number: 8i32,
            message_fields: Some(SystemPackage::FIELDS),
        };
    }
    impl MessageFields for ChangeEpoch {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_FIELD,
            Self::PROTOCOL_VERSION_FIELD,
            Self::STORAGE_CHARGE_FIELD,
            Self::COMPUTATION_CHARGE_FIELD,
            Self::STORAGE_REBATE_FIELD,
            Self::NON_REFUNDABLE_STORAGE_FEE_FIELD,
            Self::EPOCH_START_TIMESTAMP_FIELD,
            Self::SYSTEM_PACKAGES_FIELD,
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
        pub fn protocol_version(mut self) -> String {
            self.path.push(ChangeEpoch::PROTOCOL_VERSION_FIELD.name);
            self.finish()
        }
        pub fn storage_charge(mut self) -> String {
            self.path.push(ChangeEpoch::STORAGE_CHARGE_FIELD.name);
            self.finish()
        }
        pub fn computation_charge(mut self) -> String {
            self.path.push(ChangeEpoch::COMPUTATION_CHARGE_FIELD.name);
            self.finish()
        }
        pub fn storage_rebate(mut self) -> String {
            self.path.push(ChangeEpoch::STORAGE_REBATE_FIELD.name);
            self.finish()
        }
        pub fn non_refundable_storage_fee(mut self) -> String {
            self.path.push(ChangeEpoch::NON_REFUNDABLE_STORAGE_FEE_FIELD.name);
            self.finish()
        }
        pub fn epoch_start_timestamp(mut self) -> String {
            self.path.push(ChangeEpoch::EPOCH_START_TIMESTAMP_FIELD.name);
            self.finish()
        }
        pub fn system_packages(mut self) -> SystemPackageFieldPathBuilder {
            self.path.push(ChangeEpoch::SYSTEM_PACKAGES_FIELD.name);
            SystemPackageFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl SystemPackage {
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 1i32,
            message_fields: None,
        };
        pub const MODULES_FIELD: &'static MessageField = &MessageField {
            name: "modules",
            json_name: "modules",
            number: 2i32,
            message_fields: None,
        };
        pub const DEPENDENCIES_FIELD: &'static MessageField = &MessageField {
            name: "dependencies",
            json_name: "dependencies",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for SystemPackage {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::VERSION_FIELD,
            Self::MODULES_FIELD,
            Self::DEPENDENCIES_FIELD,
        ];
    }
    impl SystemPackage {
        pub fn path_builder() -> SystemPackageFieldPathBuilder {
            SystemPackageFieldPathBuilder::new()
        }
    }
    pub struct SystemPackageFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl SystemPackageFieldPathBuilder {
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
        pub fn version(mut self) -> String {
            self.path.push(SystemPackage::VERSION_FIELD.name);
            self.finish()
        }
        pub fn modules(mut self) -> String {
            self.path.push(SystemPackage::MODULES_FIELD.name);
            self.finish()
        }
        pub fn dependencies(mut self) -> String {
            self.path.push(SystemPackage::DEPENDENCIES_FIELD.name);
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
        pub const CONSENSUS_DETERMINED_VERSION_ASSIGNMENTS_FIELD: &'static MessageField = &MessageField {
            name: "consensus_determined_version_assignments",
            json_name: "consensusDeterminedVersionAssignments",
            number: 6i32,
            message_fields: Some(ConsensusDeterminedVersionAssignments::FIELDS),
        };
        pub const ADDITIONAL_STATE_DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "additional_state_digest",
            json_name: "additionalStateDigest",
            number: 7i32,
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
            Self::CONSENSUS_DETERMINED_VERSION_ASSIGNMENTS_FIELD,
            Self::ADDITIONAL_STATE_DIGEST_FIELD,
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
        pub fn consensus_determined_version_assignments(
            mut self,
        ) -> ConsensusDeterminedVersionAssignmentsFieldPathBuilder {
            self.path
                .push(
                    ConsensusCommitPrologue::CONSENSUS_DETERMINED_VERSION_ASSIGNMENTS_FIELD
                        .name,
                );
            ConsensusDeterminedVersionAssignmentsFieldPathBuilder::new_with_base(
                self.path,
            )
        }
        pub fn additional_state_digest(mut self) -> String {
            self.path.push(ConsensusCommitPrologue::ADDITIONAL_STATE_DIGEST_FIELD.name);
            self.finish()
        }
    }
    impl VersionAssignment {
        pub const OBJECT_ID_FIELD: &'static MessageField = &MessageField {
            name: "object_id",
            json_name: "objectId",
            number: 1i32,
            message_fields: None,
        };
        pub const START_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "start_version",
            json_name: "startVersion",
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
    impl MessageFields for VersionAssignment {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::OBJECT_ID_FIELD,
            Self::START_VERSION_FIELD,
            Self::VERSION_FIELD,
        ];
    }
    impl VersionAssignment {
        pub fn path_builder() -> VersionAssignmentFieldPathBuilder {
            VersionAssignmentFieldPathBuilder::new()
        }
    }
    pub struct VersionAssignmentFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl VersionAssignmentFieldPathBuilder {
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
            self.path.push(VersionAssignment::OBJECT_ID_FIELD.name);
            self.finish()
        }
        pub fn start_version(mut self) -> String {
            self.path.push(VersionAssignment::START_VERSION_FIELD.name);
            self.finish()
        }
        pub fn version(mut self) -> String {
            self.path.push(VersionAssignment::VERSION_FIELD.name);
            self.finish()
        }
    }
    impl CanceledTransaction {
        pub const DIGEST_FIELD: &'static MessageField = &MessageField {
            name: "digest",
            json_name: "digest",
            number: 1i32,
            message_fields: None,
        };
        pub const VERSION_ASSIGNMENTS_FIELD: &'static MessageField = &MessageField {
            name: "version_assignments",
            json_name: "versionAssignments",
            number: 2i32,
            message_fields: Some(VersionAssignment::FIELDS),
        };
    }
    impl MessageFields for CanceledTransaction {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::DIGEST_FIELD,
            Self::VERSION_ASSIGNMENTS_FIELD,
        ];
    }
    impl CanceledTransaction {
        pub fn path_builder() -> CanceledTransactionFieldPathBuilder {
            CanceledTransactionFieldPathBuilder::new()
        }
    }
    pub struct CanceledTransactionFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl CanceledTransactionFieldPathBuilder {
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
            self.path.push(CanceledTransaction::DIGEST_FIELD.name);
            self.finish()
        }
        pub fn version_assignments(mut self) -> VersionAssignmentFieldPathBuilder {
            self.path.push(CanceledTransaction::VERSION_ASSIGNMENTS_FIELD.name);
            VersionAssignmentFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ConsensusDeterminedVersionAssignments {
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 1i32,
            message_fields: None,
        };
        pub const CANCELED_TRANSACTIONS_FIELD: &'static MessageField = &MessageField {
            name: "canceled_transactions",
            json_name: "canceledTransactions",
            number: 3i32,
            message_fields: Some(CanceledTransaction::FIELDS),
        };
    }
    impl MessageFields for ConsensusDeterminedVersionAssignments {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::VERSION_FIELD,
            Self::CANCELED_TRANSACTIONS_FIELD,
        ];
    }
    impl ConsensusDeterminedVersionAssignments {
        pub fn path_builder() -> ConsensusDeterminedVersionAssignmentsFieldPathBuilder {
            ConsensusDeterminedVersionAssignmentsFieldPathBuilder::new()
        }
    }
    pub struct ConsensusDeterminedVersionAssignmentsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ConsensusDeterminedVersionAssignmentsFieldPathBuilder {
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
        pub fn version(mut self) -> String {
            self.path.push(ConsensusDeterminedVersionAssignments::VERSION_FIELD.name);
            self.finish()
        }
        pub fn canceled_transactions(mut self) -> CanceledTransactionFieldPathBuilder {
            self.path
                .push(
                    ConsensusDeterminedVersionAssignments::CANCELED_TRANSACTIONS_FIELD
                        .name,
                );
            CanceledTransactionFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl AuthenticatorStateUpdate {
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
        pub const NEW_ACTIVE_JWKS_FIELD: &'static MessageField = &MessageField {
            name: "new_active_jwks",
            json_name: "newActiveJwks",
            number: 3i32,
            message_fields: Some(ActiveJwk::FIELDS),
        };
        pub const AUTHENTICATOR_OBJECT_INITIAL_SHARED_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "authenticator_object_initial_shared_version",
            json_name: "authenticatorObjectInitialSharedVersion",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for AuthenticatorStateUpdate {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::EPOCH_FIELD,
            Self::ROUND_FIELD,
            Self::NEW_ACTIVE_JWKS_FIELD,
            Self::AUTHENTICATOR_OBJECT_INITIAL_SHARED_VERSION_FIELD,
        ];
    }
    impl AuthenticatorStateUpdate {
        pub fn path_builder() -> AuthenticatorStateUpdateFieldPathBuilder {
            AuthenticatorStateUpdateFieldPathBuilder::new()
        }
    }
    pub struct AuthenticatorStateUpdateFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl AuthenticatorStateUpdateFieldPathBuilder {
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
            self.path.push(AuthenticatorStateUpdate::EPOCH_FIELD.name);
            self.finish()
        }
        pub fn round(mut self) -> String {
            self.path.push(AuthenticatorStateUpdate::ROUND_FIELD.name);
            self.finish()
        }
        pub fn new_active_jwks(mut self) -> ActiveJwkFieldPathBuilder {
            self.path.push(AuthenticatorStateUpdate::NEW_ACTIVE_JWKS_FIELD.name);
            ActiveJwkFieldPathBuilder::new_with_base(self.path)
        }
        pub fn authenticator_object_initial_shared_version(mut self) -> String {
            self.path
                .push(
                    AuthenticatorStateUpdate::AUTHENTICATOR_OBJECT_INITIAL_SHARED_VERSION_FIELD
                        .name,
                );
            self.finish()
        }
    }
    impl ActiveJwk {
        pub const ID_FIELD: &'static MessageField = &MessageField {
            name: "id",
            json_name: "id",
            number: 1i32,
            message_fields: Some(JwkId::FIELDS),
        };
        pub const JWK_FIELD: &'static MessageField = &MessageField {
            name: "jwk",
            json_name: "jwk",
            number: 2i32,
            message_fields: Some(Jwk::FIELDS),
        };
        pub const EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "epoch",
            json_name: "epoch",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for ActiveJwk {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ID_FIELD,
            Self::JWK_FIELD,
            Self::EPOCH_FIELD,
        ];
    }
    impl ActiveJwk {
        pub fn path_builder() -> ActiveJwkFieldPathBuilder {
            ActiveJwkFieldPathBuilder::new()
        }
    }
    pub struct ActiveJwkFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ActiveJwkFieldPathBuilder {
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
        pub fn id(mut self) -> JwkIdFieldPathBuilder {
            self.path.push(ActiveJwk::ID_FIELD.name);
            JwkIdFieldPathBuilder::new_with_base(self.path)
        }
        pub fn jwk(mut self) -> JwkFieldPathBuilder {
            self.path.push(ActiveJwk::JWK_FIELD.name);
            JwkFieldPathBuilder::new_with_base(self.path)
        }
        pub fn epoch(mut self) -> String {
            self.path.push(ActiveJwk::EPOCH_FIELD.name);
            self.finish()
        }
    }
    impl JwkId {
        pub const ISS_FIELD: &'static MessageField = &MessageField {
            name: "iss",
            json_name: "iss",
            number: 1i32,
            message_fields: None,
        };
        pub const KID_FIELD: &'static MessageField = &MessageField {
            name: "kid",
            json_name: "kid",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for JwkId {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::ISS_FIELD,
            Self::KID_FIELD,
        ];
    }
    impl JwkId {
        pub fn path_builder() -> JwkIdFieldPathBuilder {
            JwkIdFieldPathBuilder::new()
        }
    }
    pub struct JwkIdFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl JwkIdFieldPathBuilder {
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
        pub fn iss(mut self) -> String {
            self.path.push(JwkId::ISS_FIELD.name);
            self.finish()
        }
        pub fn kid(mut self) -> String {
            self.path.push(JwkId::KID_FIELD.name);
            self.finish()
        }
    }
    impl Jwk {
        pub const KTY_FIELD: &'static MessageField = &MessageField {
            name: "kty",
            json_name: "kty",
            number: 1i32,
            message_fields: None,
        };
        pub const E_FIELD: &'static MessageField = &MessageField {
            name: "e",
            json_name: "e",
            number: 2i32,
            message_fields: None,
        };
        pub const N_FIELD: &'static MessageField = &MessageField {
            name: "n",
            json_name: "n",
            number: 3i32,
            message_fields: None,
        };
        pub const ALG_FIELD: &'static MessageField = &MessageField {
            name: "alg",
            json_name: "alg",
            number: 4i32,
            message_fields: None,
        };
    }
    impl MessageFields for Jwk {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::KTY_FIELD,
            Self::E_FIELD,
            Self::N_FIELD,
            Self::ALG_FIELD,
        ];
    }
    impl Jwk {
        pub fn path_builder() -> JwkFieldPathBuilder {
            JwkFieldPathBuilder::new()
        }
    }
    pub struct JwkFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl JwkFieldPathBuilder {
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
        pub fn kty(mut self) -> String {
            self.path.push(Jwk::KTY_FIELD.name);
            self.finish()
        }
        pub fn e(mut self) -> String {
            self.path.push(Jwk::E_FIELD.name);
            self.finish()
        }
        pub fn n(mut self) -> String {
            self.path.push(Jwk::N_FIELD.name);
            self.finish()
        }
        pub fn alg(mut self) -> String {
            self.path.push(Jwk::ALG_FIELD.name);
            self.finish()
        }
    }
    impl EndOfEpochTransaction {
        pub const TRANSACTIONS_FIELD: &'static MessageField = &MessageField {
            name: "transactions",
            json_name: "transactions",
            number: 1i32,
            message_fields: Some(EndOfEpochTransactionKind::FIELDS),
        };
    }
    impl MessageFields for EndOfEpochTransaction {
        const FIELDS: &'static [&'static MessageField] = &[Self::TRANSACTIONS_FIELD];
    }
    impl EndOfEpochTransaction {
        pub fn path_builder() -> EndOfEpochTransactionFieldPathBuilder {
            EndOfEpochTransactionFieldPathBuilder::new()
        }
    }
    pub struct EndOfEpochTransactionFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl EndOfEpochTransactionFieldPathBuilder {
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
        pub fn transactions(mut self) -> EndOfEpochTransactionKindFieldPathBuilder {
            self.path.push(EndOfEpochTransaction::TRANSACTIONS_FIELD.name);
            EndOfEpochTransactionKindFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl EndOfEpochTransactionKind {
        pub const CHANGE_EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "change_epoch",
            json_name: "changeEpoch",
            number: 2i32,
            message_fields: Some(ChangeEpoch::FIELDS),
        };
        pub const AUTHENTICATOR_STATE_EXPIRE_FIELD: &'static MessageField = &MessageField {
            name: "authenticator_state_expire",
            json_name: "authenticatorStateExpire",
            number: 3i32,
            message_fields: Some(AuthenticatorStateExpire::FIELDS),
        };
        pub const EXECUTION_TIME_OBSERVATIONS_FIELD: &'static MessageField = &MessageField {
            name: "execution_time_observations",
            json_name: "executionTimeObservations",
            number: 4i32,
            message_fields: Some(ExecutionTimeObservations::FIELDS),
        };
        pub const AUTHENTICATOR_STATE_CREATE_FIELD: &'static MessageField = &MessageField {
            name: "authenticator_state_create",
            json_name: "authenticatorStateCreate",
            number: 200i32,
            message_fields: None,
        };
        pub const RANDOMNESS_STATE_CREATE_FIELD: &'static MessageField = &MessageField {
            name: "randomness_state_create",
            json_name: "randomnessStateCreate",
            number: 201i32,
            message_fields: None,
        };
        pub const DENY_LIST_STATE_CREATE_FIELD: &'static MessageField = &MessageField {
            name: "deny_list_state_create",
            json_name: "denyListStateCreate",
            number: 202i32,
            message_fields: None,
        };
        pub const BRIDGE_STATE_CREATE_FIELD: &'static MessageField = &MessageField {
            name: "bridge_state_create",
            json_name: "bridgeStateCreate",
            number: 203i32,
            message_fields: None,
        };
        pub const BRIDGE_COMMITTEE_INIT_FIELD: &'static MessageField = &MessageField {
            name: "bridge_committee_init",
            json_name: "bridgeCommitteeInit",
            number: 204i32,
            message_fields: None,
        };
        pub const ACCUMULATOR_ROOT_CREATE_FIELD: &'static MessageField = &MessageField {
            name: "accumulator_root_create",
            json_name: "accumulatorRootCreate",
            number: 205i32,
            message_fields: None,
        };
        pub const COIN_REGISTRY_CREATE_FIELD: &'static MessageField = &MessageField {
            name: "coin_registry_create",
            json_name: "coinRegistryCreate",
            number: 206i32,
            message_fields: None,
        };
    }
    impl MessageFields for EndOfEpochTransactionKind {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::CHANGE_EPOCH_FIELD,
            Self::AUTHENTICATOR_STATE_EXPIRE_FIELD,
            Self::EXECUTION_TIME_OBSERVATIONS_FIELD,
            Self::AUTHENTICATOR_STATE_CREATE_FIELD,
            Self::RANDOMNESS_STATE_CREATE_FIELD,
            Self::DENY_LIST_STATE_CREATE_FIELD,
            Self::BRIDGE_STATE_CREATE_FIELD,
            Self::BRIDGE_COMMITTEE_INIT_FIELD,
            Self::ACCUMULATOR_ROOT_CREATE_FIELD,
            Self::COIN_REGISTRY_CREATE_FIELD,
        ];
    }
    impl EndOfEpochTransactionKind {
        pub fn path_builder() -> EndOfEpochTransactionKindFieldPathBuilder {
            EndOfEpochTransactionKindFieldPathBuilder::new()
        }
    }
    pub struct EndOfEpochTransactionKindFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl EndOfEpochTransactionKindFieldPathBuilder {
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
        pub fn change_epoch(mut self) -> ChangeEpochFieldPathBuilder {
            self.path.push(EndOfEpochTransactionKind::CHANGE_EPOCH_FIELD.name);
            ChangeEpochFieldPathBuilder::new_with_base(self.path)
        }
        pub fn authenticator_state_expire(
            mut self,
        ) -> AuthenticatorStateExpireFieldPathBuilder {
            self.path
                .push(EndOfEpochTransactionKind::AUTHENTICATOR_STATE_EXPIRE_FIELD.name);
            AuthenticatorStateExpireFieldPathBuilder::new_with_base(self.path)
        }
        pub fn execution_time_observations(
            mut self,
        ) -> ExecutionTimeObservationsFieldPathBuilder {
            self.path
                .push(EndOfEpochTransactionKind::EXECUTION_TIME_OBSERVATIONS_FIELD.name);
            ExecutionTimeObservationsFieldPathBuilder::new_with_base(self.path)
        }
        pub fn authenticator_state_create(mut self) -> String {
            self.path
                .push(EndOfEpochTransactionKind::AUTHENTICATOR_STATE_CREATE_FIELD.name);
            self.finish()
        }
        pub fn randomness_state_create(mut self) -> String {
            self.path
                .push(EndOfEpochTransactionKind::RANDOMNESS_STATE_CREATE_FIELD.name);
            self.finish()
        }
        pub fn deny_list_state_create(mut self) -> String {
            self.path.push(EndOfEpochTransactionKind::DENY_LIST_STATE_CREATE_FIELD.name);
            self.finish()
        }
        pub fn bridge_state_create(mut self) -> String {
            self.path.push(EndOfEpochTransactionKind::BRIDGE_STATE_CREATE_FIELD.name);
            self.finish()
        }
        pub fn bridge_committee_init(mut self) -> String {
            self.path.push(EndOfEpochTransactionKind::BRIDGE_COMMITTEE_INIT_FIELD.name);
            self.finish()
        }
        pub fn accumulator_root_create(mut self) -> String {
            self.path
                .push(EndOfEpochTransactionKind::ACCUMULATOR_ROOT_CREATE_FIELD.name);
            self.finish()
        }
        pub fn coin_registry_create(mut self) -> String {
            self.path.push(EndOfEpochTransactionKind::COIN_REGISTRY_CREATE_FIELD.name);
            self.finish()
        }
    }
    impl AuthenticatorStateExpire {
        pub const MIN_EPOCH_FIELD: &'static MessageField = &MessageField {
            name: "min_epoch",
            json_name: "minEpoch",
            number: 1i32,
            message_fields: None,
        };
        pub const AUTHENTICATOR_OBJECT_INITIAL_SHARED_VERSION_FIELD: &'static MessageField = &MessageField {
            name: "authenticator_object_initial_shared_version",
            json_name: "authenticatorObjectInitialSharedVersion",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for AuthenticatorStateExpire {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::MIN_EPOCH_FIELD,
            Self::AUTHENTICATOR_OBJECT_INITIAL_SHARED_VERSION_FIELD,
        ];
    }
    impl AuthenticatorStateExpire {
        pub fn path_builder() -> AuthenticatorStateExpireFieldPathBuilder {
            AuthenticatorStateExpireFieldPathBuilder::new()
        }
    }
    pub struct AuthenticatorStateExpireFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl AuthenticatorStateExpireFieldPathBuilder {
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
        pub fn min_epoch(mut self) -> String {
            self.path.push(AuthenticatorStateExpire::MIN_EPOCH_FIELD.name);
            self.finish()
        }
        pub fn authenticator_object_initial_shared_version(mut self) -> String {
            self.path
                .push(
                    AuthenticatorStateExpire::AUTHENTICATOR_OBJECT_INITIAL_SHARED_VERSION_FIELD
                        .name,
                );
            self.finish()
        }
    }
    impl ExecutionTimeObservations {
        pub const VERSION_FIELD: &'static MessageField = &MessageField {
            name: "version",
            json_name: "version",
            number: 1i32,
            message_fields: None,
        };
        pub const OBSERVATIONS_FIELD: &'static MessageField = &MessageField {
            name: "observations",
            json_name: "observations",
            number: 2i32,
            message_fields: Some(ExecutionTimeObservation::FIELDS),
        };
    }
    impl MessageFields for ExecutionTimeObservations {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::VERSION_FIELD,
            Self::OBSERVATIONS_FIELD,
        ];
    }
    impl ExecutionTimeObservations {
        pub fn path_builder() -> ExecutionTimeObservationsFieldPathBuilder {
            ExecutionTimeObservationsFieldPathBuilder::new()
        }
    }
    pub struct ExecutionTimeObservationsFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ExecutionTimeObservationsFieldPathBuilder {
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
        pub fn version(mut self) -> String {
            self.path.push(ExecutionTimeObservations::VERSION_FIELD.name);
            self.finish()
        }
        pub fn observations(mut self) -> ExecutionTimeObservationFieldPathBuilder {
            self.path.push(ExecutionTimeObservations::OBSERVATIONS_FIELD.name);
            ExecutionTimeObservationFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ExecutionTimeObservation {
        pub const KIND_FIELD: &'static MessageField = &MessageField {
            name: "kind",
            json_name: "kind",
            number: 1i32,
            message_fields: None,
        };
        pub const MOVE_ENTRY_POINT_FIELD: &'static MessageField = &MessageField {
            name: "move_entry_point",
            json_name: "moveEntryPoint",
            number: 2i32,
            message_fields: Some(MoveCall::FIELDS),
        };
        pub const VALIDATOR_OBSERVATIONS_FIELD: &'static MessageField = &MessageField {
            name: "validator_observations",
            json_name: "validatorObservations",
            number: 3i32,
            message_fields: Some(ValidatorExecutionTimeObservation::FIELDS),
        };
    }
    impl MessageFields for ExecutionTimeObservation {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::KIND_FIELD,
            Self::MOVE_ENTRY_POINT_FIELD,
            Self::VALIDATOR_OBSERVATIONS_FIELD,
        ];
    }
    impl ExecutionTimeObservation {
        pub fn path_builder() -> ExecutionTimeObservationFieldPathBuilder {
            ExecutionTimeObservationFieldPathBuilder::new()
        }
    }
    pub struct ExecutionTimeObservationFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ExecutionTimeObservationFieldPathBuilder {
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
            self.path.push(ExecutionTimeObservation::KIND_FIELD.name);
            self.finish()
        }
        pub fn move_entry_point(mut self) -> MoveCallFieldPathBuilder {
            self.path.push(ExecutionTimeObservation::MOVE_ENTRY_POINT_FIELD.name);
            MoveCallFieldPathBuilder::new_with_base(self.path)
        }
        pub fn validator_observations(
            mut self,
        ) -> ValidatorExecutionTimeObservationFieldPathBuilder {
            self.path.push(ExecutionTimeObservation::VALIDATOR_OBSERVATIONS_FIELD.name);
            ValidatorExecutionTimeObservationFieldPathBuilder::new_with_base(self.path)
        }
    }
    impl ValidatorExecutionTimeObservation {
        pub const VALIDATOR_FIELD: &'static MessageField = &MessageField {
            name: "validator",
            json_name: "validator",
            number: 1i32,
            message_fields: None,
        };
        pub const DURATION_FIELD: &'static MessageField = &MessageField {
            name: "duration",
            json_name: "duration",
            number: 2i32,
            message_fields: None,
        };
    }
    impl MessageFields for ValidatorExecutionTimeObservation {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::VALIDATOR_FIELD,
            Self::DURATION_FIELD,
        ];
    }
    impl ValidatorExecutionTimeObservation {
        pub fn path_builder() -> ValidatorExecutionTimeObservationFieldPathBuilder {
            ValidatorExecutionTimeObservationFieldPathBuilder::new()
        }
    }
    pub struct ValidatorExecutionTimeObservationFieldPathBuilder {
        path: Vec<&'static str>,
    }
    impl ValidatorExecutionTimeObservationFieldPathBuilder {
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
        pub fn validator(mut self) -> String {
            self.path.push(ValidatorExecutionTimeObservation::VALIDATOR_FIELD.name);
            self.finish()
        }
        pub fn duration(mut self) -> String {
            self.path.push(ValidatorExecutionTimeObservation::DURATION_FIELD.name);
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
        pub const CHECKPOINTED_FIELD: &'static MessageField = &MessageField {
            name: "checkpointed",
            json_name: "checkpointed",
            number: 2i32,
            message_fields: None,
        };
        pub const QUORUM_EXECUTED_FIELD: &'static MessageField = &MessageField {
            name: "quorum_executed",
            json_name: "quorumExecuted",
            number: 3i32,
            message_fields: None,
        };
    }
    impl MessageFields for TransactionFinality {
        const FIELDS: &'static [&'static MessageField] = &[
            Self::CERTIFIED_FIELD,
            Self::CHECKPOINTED_FIELD,
            Self::QUORUM_EXECUTED_FIELD,
        ];
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
        pub fn checkpointed(mut self) -> String {
            self.path.push(TransactionFinality::CHECKPOINTED_FIELD.name);
            self.finish()
        }
        pub fn quorum_executed(mut self) -> String {
            self.path.push(TransactionFinality::QUORUM_EXECUTED_FIELD.name);
            self.finish()
        }
    }
}
