mod _getter_impls {
    #![allow(clippy::useless_conversion)]
    use super::*;
    impl Argument {
        pub const fn const_default() -> Self {
            Self {
                kind: None,
                input: None,
                result: None,
                subresult: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Argument = Argument::const_default();
            &DEFAULT
        }
        pub fn with_input(mut self, field: u32) -> Self {
            self.input = Some(field.into());
            self
        }
        pub fn with_result(mut self, field: u32) -> Self {
            self.result = Some(field.into());
            self
        }
        pub fn with_subresult(mut self, field: u32) -> Self {
            self.subresult = Some(field.into());
            self
        }
    }
    impl BalanceChange {
        pub const fn const_default() -> Self {
            Self {
                address: None,
                coin_type: None,
                amount: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: BalanceChange = BalanceChange::const_default();
            &DEFAULT
        }
        pub fn with_address(mut self, field: String) -> Self {
            self.address = Some(field.into());
            self
        }
        pub fn with_coin_type(mut self, field: String) -> Self {
            self.coin_type = Some(field.into());
            self
        }
        pub fn with_amount(mut self, field: String) -> Self {
            self.amount = Some(field.into());
            self
        }
    }
    impl Bcs {
        pub const fn const_default() -> Self {
            Self { name: None, value: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Bcs = Bcs::const_default();
            &DEFAULT
        }
        pub fn with_name(mut self, field: String) -> Self {
            self.name = Some(field.into());
            self
        }
        pub fn with_value(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.value = Some(field.into());
            self
        }
    }
    impl TransactionEffects {
        pub const fn const_default() -> Self {
            Self {
                bcs: None,
                digest: None,
                version: None,
                status: None,
                epoch: None,
                gas_used: None,
                transaction_digest: None,
                gas_object: None,
                events_digest: None,
                dependencies: Vec::new(),
                lamport_version: None,
                changed_objects: Vec::new(),
                unchanged_consensus_objects: Vec::new(),
                auxiliary_data_digest: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TransactionEffects = TransactionEffects::const_default();
            &DEFAULT
        }
        pub fn bcs(&self) -> &Bcs {
            self.bcs
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Bcs::default_instance() as _)
        }
        pub fn bcs_opt(&self) -> Option<&Bcs> {
            self.bcs.as_ref().map(|field| field as _)
        }
        pub fn bcs_opt_mut(&mut self) -> Option<&mut Bcs> {
            self.bcs.as_mut().map(|field| field as _)
        }
        pub fn bcs_mut(&mut self) -> &mut Bcs {
            self.bcs.get_or_insert_default()
        }
        pub fn with_bcs(mut self, field: Bcs) -> Self {
            self.bcs = Some(field.into());
            self
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn with_version(mut self, field: i32) -> Self {
            self.version = Some(field.into());
            self
        }
        pub fn status(&self) -> &ExecutionStatus {
            self.status
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ExecutionStatus::default_instance() as _)
        }
        pub fn status_opt(&self) -> Option<&ExecutionStatus> {
            self.status.as_ref().map(|field| field as _)
        }
        pub fn status_opt_mut(&mut self) -> Option<&mut ExecutionStatus> {
            self.status.as_mut().map(|field| field as _)
        }
        pub fn status_mut(&mut self) -> &mut ExecutionStatus {
            self.status.get_or_insert_default()
        }
        pub fn with_status(mut self, field: ExecutionStatus) -> Self {
            self.status = Some(field.into());
            self
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
        pub fn gas_used(&self) -> &GasCostSummary {
            self.gas_used
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| GasCostSummary::default_instance() as _)
        }
        pub fn gas_used_opt(&self) -> Option<&GasCostSummary> {
            self.gas_used.as_ref().map(|field| field as _)
        }
        pub fn gas_used_opt_mut(&mut self) -> Option<&mut GasCostSummary> {
            self.gas_used.as_mut().map(|field| field as _)
        }
        pub fn gas_used_mut(&mut self) -> &mut GasCostSummary {
            self.gas_used.get_or_insert_default()
        }
        pub fn with_gas_used(mut self, field: GasCostSummary) -> Self {
            self.gas_used = Some(field.into());
            self
        }
        pub fn with_transaction_digest(mut self, field: String) -> Self {
            self.transaction_digest = Some(field.into());
            self
        }
        pub fn gas_object(&self) -> &ChangedObject {
            self.gas_object
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ChangedObject::default_instance() as _)
        }
        pub fn gas_object_opt(&self) -> Option<&ChangedObject> {
            self.gas_object.as_ref().map(|field| field as _)
        }
        pub fn gas_object_opt_mut(&mut self) -> Option<&mut ChangedObject> {
            self.gas_object.as_mut().map(|field| field as _)
        }
        pub fn gas_object_mut(&mut self) -> &mut ChangedObject {
            self.gas_object.get_or_insert_default()
        }
        pub fn with_gas_object(mut self, field: ChangedObject) -> Self {
            self.gas_object = Some(field.into());
            self
        }
        pub fn with_events_digest(mut self, field: String) -> Self {
            self.events_digest = Some(field.into());
            self
        }
        pub fn dependencies(&self) -> &[String] {
            &self.dependencies
        }
        pub fn with_dependencies(mut self, field: Vec<String>) -> Self {
            self.dependencies = field;
            self
        }
        pub fn with_lamport_version(mut self, field: u64) -> Self {
            self.lamport_version = Some(field.into());
            self
        }
        pub fn changed_objects(&self) -> &[ChangedObject] {
            &self.changed_objects
        }
        pub fn with_changed_objects(mut self, field: Vec<ChangedObject>) -> Self {
            self.changed_objects = field;
            self
        }
        pub fn unchanged_consensus_objects(&self) -> &[UnchangedConsensusObject] {
            &self.unchanged_consensus_objects
        }
        pub fn with_unchanged_consensus_objects(
            mut self,
            field: Vec<UnchangedConsensusObject>,
        ) -> Self {
            self.unchanged_consensus_objects = field;
            self
        }
        pub fn with_auxiliary_data_digest(mut self, field: String) -> Self {
            self.auxiliary_data_digest = Some(field.into());
            self
        }
    }
    impl ChangedObject {
        pub const fn const_default() -> Self {
            Self {
                object_id: None,
                input_state: None,
                input_version: None,
                input_digest: None,
                input_owner: None,
                output_state: None,
                output_version: None,
                output_digest: None,
                output_owner: None,
                id_operation: None,
                object_type: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ChangedObject = ChangedObject::const_default();
            &DEFAULT
        }
        pub fn with_object_id(mut self, field: String) -> Self {
            self.object_id = Some(field.into());
            self
        }
        pub fn with_input_version(mut self, field: u64) -> Self {
            self.input_version = Some(field.into());
            self
        }
        pub fn with_input_digest(mut self, field: String) -> Self {
            self.input_digest = Some(field.into());
            self
        }
        pub fn input_owner(&self) -> &Owner {
            self.input_owner
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Owner::default_instance() as _)
        }
        pub fn input_owner_opt(&self) -> Option<&Owner> {
            self.input_owner.as_ref().map(|field| field as _)
        }
        pub fn input_owner_opt_mut(&mut self) -> Option<&mut Owner> {
            self.input_owner.as_mut().map(|field| field as _)
        }
        pub fn input_owner_mut(&mut self) -> &mut Owner {
            self.input_owner.get_or_insert_default()
        }
        pub fn with_input_owner(mut self, field: Owner) -> Self {
            self.input_owner = Some(field.into());
            self
        }
        pub fn with_output_version(mut self, field: u64) -> Self {
            self.output_version = Some(field.into());
            self
        }
        pub fn with_output_digest(mut self, field: String) -> Self {
            self.output_digest = Some(field.into());
            self
        }
        pub fn output_owner(&self) -> &Owner {
            self.output_owner
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Owner::default_instance() as _)
        }
        pub fn output_owner_opt(&self) -> Option<&Owner> {
            self.output_owner.as_ref().map(|field| field as _)
        }
        pub fn output_owner_opt_mut(&mut self) -> Option<&mut Owner> {
            self.output_owner.as_mut().map(|field| field as _)
        }
        pub fn output_owner_mut(&mut self) -> &mut Owner {
            self.output_owner.get_or_insert_default()
        }
        pub fn with_output_owner(mut self, field: Owner) -> Self {
            self.output_owner = Some(field.into());
            self
        }
        pub fn with_object_type(mut self, field: String) -> Self {
            self.object_type = Some(field.into());
            self
        }
    }
    impl UnchangedConsensusObject {
        pub const fn const_default() -> Self {
            Self {
                kind: None,
                object_id: None,
                version: None,
                digest: None,
                object_type: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: UnchangedConsensusObject = UnchangedConsensusObject::const_default();
            &DEFAULT
        }
        pub fn with_object_id(mut self, field: String) -> Self {
            self.object_id = Some(field.into());
            self
        }
        pub fn with_version(mut self, field: u64) -> Self {
            self.version = Some(field.into());
            self
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn with_object_type(mut self, field: String) -> Self {
            self.object_type = Some(field.into());
            self
        }
    }
    impl TransactionEvents {
        pub const fn const_default() -> Self {
            Self {
                bcs: None,
                digest: None,
                events: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TransactionEvents = TransactionEvents::const_default();
            &DEFAULT
        }
        pub fn bcs(&self) -> &Bcs {
            self.bcs
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Bcs::default_instance() as _)
        }
        pub fn bcs_opt(&self) -> Option<&Bcs> {
            self.bcs.as_ref().map(|field| field as _)
        }
        pub fn bcs_opt_mut(&mut self) -> Option<&mut Bcs> {
            self.bcs.as_mut().map(|field| field as _)
        }
        pub fn bcs_mut(&mut self) -> &mut Bcs {
            self.bcs.get_or_insert_default()
        }
        pub fn with_bcs(mut self, field: Bcs) -> Self {
            self.bcs = Some(field.into());
            self
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn events(&self) -> &[Event] {
            &self.events
        }
        pub fn with_events(mut self, field: Vec<Event>) -> Self {
            self.events = field;
            self
        }
    }
    impl Event {
        pub const fn const_default() -> Self {
            Self {
                package_id: None,
                module: None,
                sender: None,
                event_type: None,
                contents: None,
                json: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Event = Event::const_default();
            &DEFAULT
        }
        pub fn with_package_id(mut self, field: String) -> Self {
            self.package_id = Some(field.into());
            self
        }
        pub fn with_module(mut self, field: String) -> Self {
            self.module = Some(field.into());
            self
        }
        pub fn with_sender(mut self, field: String) -> Self {
            self.sender = Some(field.into());
            self
        }
        pub fn with_event_type(mut self, field: String) -> Self {
            self.event_type = Some(field.into());
            self
        }
        pub fn contents(&self) -> &Bcs {
            self.contents
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Bcs::default_instance() as _)
        }
        pub fn contents_opt(&self) -> Option<&Bcs> {
            self.contents.as_ref().map(|field| field as _)
        }
        pub fn contents_opt_mut(&mut self) -> Option<&mut Bcs> {
            self.contents.as_mut().map(|field| field as _)
        }
        pub fn contents_mut(&mut self) -> &mut Bcs {
            self.contents.get_or_insert_default()
        }
        pub fn with_contents(mut self, field: Bcs) -> Self {
            self.contents = Some(field.into());
            self
        }
    }
    impl ExecutedTransaction {
        pub const fn const_default() -> Self {
            Self {
                digest: None,
                transaction: None,
                signatures: Vec::new(),
                effects: None,
                events: None,
                checkpoint: None,
                timestamp: None,
                balance_changes: Vec::new(),
                input_objects: Vec::new(),
                output_objects: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ExecutedTransaction = ExecutedTransaction::const_default();
            &DEFAULT
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn transaction(&self) -> &Transaction {
            self.transaction
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Transaction::default_instance() as _)
        }
        pub fn transaction_opt(&self) -> Option<&Transaction> {
            self.transaction.as_ref().map(|field| field as _)
        }
        pub fn transaction_opt_mut(&mut self) -> Option<&mut Transaction> {
            self.transaction.as_mut().map(|field| field as _)
        }
        pub fn transaction_mut(&mut self) -> &mut Transaction {
            self.transaction.get_or_insert_default()
        }
        pub fn with_transaction(mut self, field: Transaction) -> Self {
            self.transaction = Some(field.into());
            self
        }
        pub fn signatures(&self) -> &[UserSignature] {
            &self.signatures
        }
        pub fn with_signatures(mut self, field: Vec<UserSignature>) -> Self {
            self.signatures = field;
            self
        }
        pub fn effects(&self) -> &TransactionEffects {
            self.effects
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| TransactionEffects::default_instance() as _)
        }
        pub fn effects_opt(&self) -> Option<&TransactionEffects> {
            self.effects.as_ref().map(|field| field as _)
        }
        pub fn effects_opt_mut(&mut self) -> Option<&mut TransactionEffects> {
            self.effects.as_mut().map(|field| field as _)
        }
        pub fn effects_mut(&mut self) -> &mut TransactionEffects {
            self.effects.get_or_insert_default()
        }
        pub fn with_effects(mut self, field: TransactionEffects) -> Self {
            self.effects = Some(field.into());
            self
        }
        pub fn events(&self) -> &TransactionEvents {
            self.events
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| TransactionEvents::default_instance() as _)
        }
        pub fn events_opt(&self) -> Option<&TransactionEvents> {
            self.events.as_ref().map(|field| field as _)
        }
        pub fn events_opt_mut(&mut self) -> Option<&mut TransactionEvents> {
            self.events.as_mut().map(|field| field as _)
        }
        pub fn events_mut(&mut self) -> &mut TransactionEvents {
            self.events.get_or_insert_default()
        }
        pub fn with_events(mut self, field: TransactionEvents) -> Self {
            self.events = Some(field.into());
            self
        }
        pub fn with_checkpoint(mut self, field: u64) -> Self {
            self.checkpoint = Some(field.into());
            self
        }
        pub fn balance_changes(&self) -> &[BalanceChange] {
            &self.balance_changes
        }
        pub fn with_balance_changes(mut self, field: Vec<BalanceChange>) -> Self {
            self.balance_changes = field;
            self
        }
        pub fn input_objects(&self) -> &[Object] {
            &self.input_objects
        }
        pub fn with_input_objects(mut self, field: Vec<Object>) -> Self {
            self.input_objects = field;
            self
        }
        pub fn output_objects(&self) -> &[Object] {
            &self.output_objects
        }
        pub fn with_output_objects(mut self, field: Vec<Object>) -> Self {
            self.output_objects = field;
            self
        }
    }
    impl ExecutionStatus {
        pub const fn const_default() -> Self {
            Self { success: None, error: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ExecutionStatus = ExecutionStatus::const_default();
            &DEFAULT
        }
        pub fn with_success(mut self, field: bool) -> Self {
            self.success = Some(field.into());
            self
        }
        pub fn error(&self) -> &ExecutionError {
            self.error
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ExecutionError::default_instance() as _)
        }
        pub fn error_opt(&self) -> Option<&ExecutionError> {
            self.error.as_ref().map(|field| field as _)
        }
        pub fn error_opt_mut(&mut self) -> Option<&mut ExecutionError> {
            self.error.as_mut().map(|field| field as _)
        }
        pub fn error_mut(&mut self) -> &mut ExecutionError {
            self.error.get_or_insert_default()
        }
        pub fn with_error(mut self, field: ExecutionError) -> Self {
            self.error = Some(field.into());
            self
        }
    }
    impl ExecutionError {
        pub const fn const_default() -> Self {
            Self {
                description: None,
                command: None,
                kind: None,
                error_details: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ExecutionError = ExecutionError::const_default();
            &DEFAULT
        }
        pub fn with_description(mut self, field: String) -> Self {
            self.description = Some(field.into());
            self
        }
        pub fn with_command(mut self, field: u64) -> Self {
            self.command = Some(field.into());
            self
        }
        pub fn abort(&self) -> &MoveAbort {
            if let Some(execution_error::ErrorDetails::Abort(field)) = &self
                .error_details
            {
                field as _
            } else {
                MoveAbort::default_instance() as _
            }
        }
        pub fn abort_opt(&self) -> Option<&MoveAbort> {
            if let Some(execution_error::ErrorDetails::Abort(field)) = &self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn abort_opt_mut(&mut self) -> Option<&mut MoveAbort> {
            if let Some(execution_error::ErrorDetails::Abort(field)) = &mut self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn abort_mut(&mut self) -> &mut MoveAbort {
            if self.abort_opt_mut().is_none() {
                self.error_details = Some(
                    execution_error::ErrorDetails::Abort(MoveAbort::default()),
                );
            }
            self.abort_opt_mut().unwrap()
        }
        pub fn with_abort(mut self, field: MoveAbort) -> Self {
            self.error_details = Some(
                execution_error::ErrorDetails::Abort(field.into()),
            );
            self
        }
        pub fn size_error(&self) -> &SizeError {
            if let Some(execution_error::ErrorDetails::SizeError(field)) = &self
                .error_details
            {
                field as _
            } else {
                SizeError::default_instance() as _
            }
        }
        pub fn size_error_opt(&self) -> Option<&SizeError> {
            if let Some(execution_error::ErrorDetails::SizeError(field)) = &self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn size_error_opt_mut(&mut self) -> Option<&mut SizeError> {
            if let Some(execution_error::ErrorDetails::SizeError(field)) = &mut self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn size_error_mut(&mut self) -> &mut SizeError {
            if self.size_error_opt_mut().is_none() {
                self.error_details = Some(
                    execution_error::ErrorDetails::SizeError(SizeError::default()),
                );
            }
            self.size_error_opt_mut().unwrap()
        }
        pub fn with_size_error(mut self, field: SizeError) -> Self {
            self.error_details = Some(
                execution_error::ErrorDetails::SizeError(field.into()),
            );
            self
        }
        pub fn command_argument_error(&self) -> &CommandArgumentError {
            if let Some(execution_error::ErrorDetails::CommandArgumentError(field)) = &self
                .error_details
            {
                field as _
            } else {
                CommandArgumentError::default_instance() as _
            }
        }
        pub fn command_argument_error_opt(&self) -> Option<&CommandArgumentError> {
            if let Some(execution_error::ErrorDetails::CommandArgumentError(field)) = &self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn command_argument_error_opt_mut(
            &mut self,
        ) -> Option<&mut CommandArgumentError> {
            if let Some(execution_error::ErrorDetails::CommandArgumentError(field)) = &mut self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn command_argument_error_mut(&mut self) -> &mut CommandArgumentError {
            if self.command_argument_error_opt_mut().is_none() {
                self.error_details = Some(
                    execution_error::ErrorDetails::CommandArgumentError(
                        CommandArgumentError::default(),
                    ),
                );
            }
            self.command_argument_error_opt_mut().unwrap()
        }
        pub fn with_command_argument_error(
            mut self,
            field: CommandArgumentError,
        ) -> Self {
            self.error_details = Some(
                execution_error::ErrorDetails::CommandArgumentError(field.into()),
            );
            self
        }
        pub fn type_argument_error(&self) -> &TypeArgumentError {
            if let Some(execution_error::ErrorDetails::TypeArgumentError(field)) = &self
                .error_details
            {
                field as _
            } else {
                TypeArgumentError::default_instance() as _
            }
        }
        pub fn type_argument_error_opt(&self) -> Option<&TypeArgumentError> {
            if let Some(execution_error::ErrorDetails::TypeArgumentError(field)) = &self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn type_argument_error_opt_mut(&mut self) -> Option<&mut TypeArgumentError> {
            if let Some(execution_error::ErrorDetails::TypeArgumentError(field)) = &mut self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn type_argument_error_mut(&mut self) -> &mut TypeArgumentError {
            if self.type_argument_error_opt_mut().is_none() {
                self.error_details = Some(
                    execution_error::ErrorDetails::TypeArgumentError(
                        TypeArgumentError::default(),
                    ),
                );
            }
            self.type_argument_error_opt_mut().unwrap()
        }
        pub fn with_type_argument_error(mut self, field: TypeArgumentError) -> Self {
            self.error_details = Some(
                execution_error::ErrorDetails::TypeArgumentError(field.into()),
            );
            self
        }
        pub fn package_upgrade_error(&self) -> &PackageUpgradeError {
            if let Some(execution_error::ErrorDetails::PackageUpgradeError(field)) = &self
                .error_details
            {
                field as _
            } else {
                PackageUpgradeError::default_instance() as _
            }
        }
        pub fn package_upgrade_error_opt(&self) -> Option<&PackageUpgradeError> {
            if let Some(execution_error::ErrorDetails::PackageUpgradeError(field)) = &self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn package_upgrade_error_opt_mut(
            &mut self,
        ) -> Option<&mut PackageUpgradeError> {
            if let Some(execution_error::ErrorDetails::PackageUpgradeError(field)) = &mut self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn package_upgrade_error_mut(&mut self) -> &mut PackageUpgradeError {
            if self.package_upgrade_error_opt_mut().is_none() {
                self.error_details = Some(
                    execution_error::ErrorDetails::PackageUpgradeError(
                        PackageUpgradeError::default(),
                    ),
                );
            }
            self.package_upgrade_error_opt_mut().unwrap()
        }
        pub fn with_package_upgrade_error(mut self, field: PackageUpgradeError) -> Self {
            self.error_details = Some(
                execution_error::ErrorDetails::PackageUpgradeError(field.into()),
            );
            self
        }
        pub fn index_error(&self) -> &IndexError {
            if let Some(execution_error::ErrorDetails::IndexError(field)) = &self
                .error_details
            {
                field as _
            } else {
                IndexError::default_instance() as _
            }
        }
        pub fn index_error_opt(&self) -> Option<&IndexError> {
            if let Some(execution_error::ErrorDetails::IndexError(field)) = &self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn index_error_opt_mut(&mut self) -> Option<&mut IndexError> {
            if let Some(execution_error::ErrorDetails::IndexError(field)) = &mut self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn index_error_mut(&mut self) -> &mut IndexError {
            if self.index_error_opt_mut().is_none() {
                self.error_details = Some(
                    execution_error::ErrorDetails::IndexError(IndexError::default()),
                );
            }
            self.index_error_opt_mut().unwrap()
        }
        pub fn with_index_error(mut self, field: IndexError) -> Self {
            self.error_details = Some(
                execution_error::ErrorDetails::IndexError(field.into()),
            );
            self
        }
        pub fn object_id(&self) -> &str {
            if let Some(execution_error::ErrorDetails::ObjectId(field)) = &self
                .error_details
            {
                field as _
            } else {
                ""
            }
        }
        pub fn object_id_opt(&self) -> Option<&str> {
            if let Some(execution_error::ErrorDetails::ObjectId(field)) = &self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn object_id_opt_mut(&mut self) -> Option<&mut String> {
            if let Some(execution_error::ErrorDetails::ObjectId(field)) = &mut self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn object_id_mut(&mut self) -> &mut String {
            if self.object_id_opt_mut().is_none() {
                self.error_details = Some(
                    execution_error::ErrorDetails::ObjectId(String::default()),
                );
            }
            self.object_id_opt_mut().unwrap()
        }
        pub fn with_object_id(mut self, field: String) -> Self {
            self.error_details = Some(
                execution_error::ErrorDetails::ObjectId(field.into()),
            );
            self
        }
        pub fn coin_deny_list_error(&self) -> &CoinDenyListError {
            if let Some(execution_error::ErrorDetails::CoinDenyListError(field)) = &self
                .error_details
            {
                field as _
            } else {
                CoinDenyListError::default_instance() as _
            }
        }
        pub fn coin_deny_list_error_opt(&self) -> Option<&CoinDenyListError> {
            if let Some(execution_error::ErrorDetails::CoinDenyListError(field)) = &self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn coin_deny_list_error_opt_mut(
            &mut self,
        ) -> Option<&mut CoinDenyListError> {
            if let Some(execution_error::ErrorDetails::CoinDenyListError(field)) = &mut self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn coin_deny_list_error_mut(&mut self) -> &mut CoinDenyListError {
            if self.coin_deny_list_error_opt_mut().is_none() {
                self.error_details = Some(
                    execution_error::ErrorDetails::CoinDenyListError(
                        CoinDenyListError::default(),
                    ),
                );
            }
            self.coin_deny_list_error_opt_mut().unwrap()
        }
        pub fn with_coin_deny_list_error(mut self, field: CoinDenyListError) -> Self {
            self.error_details = Some(
                execution_error::ErrorDetails::CoinDenyListError(field.into()),
            );
            self
        }
        pub fn congested_objects(&self) -> &CongestedObjects {
            if let Some(execution_error::ErrorDetails::CongestedObjects(field)) = &self
                .error_details
            {
                field as _
            } else {
                CongestedObjects::default_instance() as _
            }
        }
        pub fn congested_objects_opt(&self) -> Option<&CongestedObjects> {
            if let Some(execution_error::ErrorDetails::CongestedObjects(field)) = &self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn congested_objects_opt_mut(&mut self) -> Option<&mut CongestedObjects> {
            if let Some(execution_error::ErrorDetails::CongestedObjects(field)) = &mut self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn congested_objects_mut(&mut self) -> &mut CongestedObjects {
            if self.congested_objects_opt_mut().is_none() {
                self.error_details = Some(
                    execution_error::ErrorDetails::CongestedObjects(
                        CongestedObjects::default(),
                    ),
                );
            }
            self.congested_objects_opt_mut().unwrap()
        }
        pub fn with_congested_objects(mut self, field: CongestedObjects) -> Self {
            self.error_details = Some(
                execution_error::ErrorDetails::CongestedObjects(field.into()),
            );
            self
        }
    }
    impl MoveAbort {
        pub const fn const_default() -> Self {
            Self {
                abort_code: None,
                location: None,
                clever_error: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: MoveAbort = MoveAbort::const_default();
            &DEFAULT
        }
        pub fn with_abort_code(mut self, field: u64) -> Self {
            self.abort_code = Some(field.into());
            self
        }
        pub fn location(&self) -> &MoveLocation {
            self.location
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| MoveLocation::default_instance() as _)
        }
        pub fn location_opt(&self) -> Option<&MoveLocation> {
            self.location.as_ref().map(|field| field as _)
        }
        pub fn location_opt_mut(&mut self) -> Option<&mut MoveLocation> {
            self.location.as_mut().map(|field| field as _)
        }
        pub fn location_mut(&mut self) -> &mut MoveLocation {
            self.location.get_or_insert_default()
        }
        pub fn with_location(mut self, field: MoveLocation) -> Self {
            self.location = Some(field.into());
            self
        }
        pub fn clever_error(&self) -> &CleverError {
            self.clever_error
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| CleverError::default_instance() as _)
        }
        pub fn clever_error_opt(&self) -> Option<&CleverError> {
            self.clever_error.as_ref().map(|field| field as _)
        }
        pub fn clever_error_opt_mut(&mut self) -> Option<&mut CleverError> {
            self.clever_error.as_mut().map(|field| field as _)
        }
        pub fn clever_error_mut(&mut self) -> &mut CleverError {
            self.clever_error.get_or_insert_default()
        }
        pub fn with_clever_error(mut self, field: CleverError) -> Self {
            self.clever_error = Some(field.into());
            self
        }
    }
    impl MoveLocation {
        pub const fn const_default() -> Self {
            Self {
                package: None,
                module: None,
                function: None,
                instruction: None,
                function_name: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: MoveLocation = MoveLocation::const_default();
            &DEFAULT
        }
        pub fn with_package(mut self, field: String) -> Self {
            self.package = Some(field.into());
            self
        }
        pub fn with_module(mut self, field: String) -> Self {
            self.module = Some(field.into());
            self
        }
        pub fn with_function(mut self, field: u32) -> Self {
            self.function = Some(field.into());
            self
        }
        pub fn with_instruction(mut self, field: u32) -> Self {
            self.instruction = Some(field.into());
            self
        }
        pub fn with_function_name(mut self, field: String) -> Self {
            self.function_name = Some(field.into());
            self
        }
    }
    impl CleverError {
        pub const fn const_default() -> Self {
            Self {
                error_code: None,
                line_number: None,
                constant_name: None,
                constant_type: None,
                value: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: CleverError = CleverError::const_default();
            &DEFAULT
        }
        pub fn with_error_code(mut self, field: u64) -> Self {
            self.error_code = Some(field.into());
            self
        }
        pub fn with_line_number(mut self, field: u64) -> Self {
            self.line_number = Some(field.into());
            self
        }
        pub fn with_constant_name(mut self, field: String) -> Self {
            self.constant_name = Some(field.into());
            self
        }
        pub fn with_constant_type(mut self, field: String) -> Self {
            self.constant_type = Some(field.into());
            self
        }
        pub fn rendered(&self) -> &str {
            if let Some(clever_error::Value::Rendered(field)) = &self.value {
                field as _
            } else {
                ""
            }
        }
        pub fn rendered_opt(&self) -> Option<&str> {
            if let Some(clever_error::Value::Rendered(field)) = &self.value {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn rendered_opt_mut(&mut self) -> Option<&mut String> {
            if let Some(clever_error::Value::Rendered(field)) = &mut self.value {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn rendered_mut(&mut self) -> &mut String {
            if self.rendered_opt_mut().is_none() {
                self.value = Some(clever_error::Value::Rendered(String::default()));
            }
            self.rendered_opt_mut().unwrap()
        }
        pub fn with_rendered(mut self, field: String) -> Self {
            self.value = Some(clever_error::Value::Rendered(field.into()));
            self
        }
        pub fn raw(&self) -> &[u8] {
            if let Some(clever_error::Value::Raw(field)) = &self.value {
                field as _
            } else {
                &[]
            }
        }
        pub fn raw_opt(&self) -> Option<&[u8]> {
            if let Some(clever_error::Value::Raw(field)) = &self.value {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn raw_opt_mut(&mut self) -> Option<&mut ::prost::bytes::Bytes> {
            if let Some(clever_error::Value::Raw(field)) = &mut self.value {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn raw_mut(&mut self) -> &mut ::prost::bytes::Bytes {
            if self.raw_opt_mut().is_none() {
                self.value = Some(
                    clever_error::Value::Raw(::prost::bytes::Bytes::default()),
                );
            }
            self.raw_opt_mut().unwrap()
        }
        pub fn with_raw(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.value = Some(clever_error::Value::Raw(field.into()));
            self
        }
    }
    impl SizeError {
        pub const fn const_default() -> Self {
            Self { size: None, max_size: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SizeError = SizeError::const_default();
            &DEFAULT
        }
        pub fn with_size(mut self, field: u64) -> Self {
            self.size = Some(field.into());
            self
        }
        pub fn with_max_size(mut self, field: u64) -> Self {
            self.max_size = Some(field.into());
            self
        }
    }
    impl IndexError {
        pub const fn const_default() -> Self {
            Self {
                index: None,
                subresult: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: IndexError = IndexError::const_default();
            &DEFAULT
        }
        pub fn with_index(mut self, field: u32) -> Self {
            self.index = Some(field.into());
            self
        }
        pub fn with_subresult(mut self, field: u32) -> Self {
            self.subresult = Some(field.into());
            self
        }
    }
    impl CoinDenyListError {
        pub const fn const_default() -> Self {
            Self {
                address: None,
                coin_type: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: CoinDenyListError = CoinDenyListError::const_default();
            &DEFAULT
        }
        pub fn with_address(mut self, field: String) -> Self {
            self.address = Some(field.into());
            self
        }
        pub fn with_coin_type(mut self, field: String) -> Self {
            self.coin_type = Some(field.into());
            self
        }
    }
    impl CongestedObjects {
        pub const fn const_default() -> Self {
            Self { objects: Vec::new() }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: CongestedObjects = CongestedObjects::const_default();
            &DEFAULT
        }
        pub fn objects(&self) -> &[String] {
            &self.objects
        }
        pub fn with_objects(mut self, field: Vec<String>) -> Self {
            self.objects = field;
            self
        }
    }
    impl CommandArgumentError {
        pub const fn const_default() -> Self {
            Self {
                argument: None,
                kind: None,
                index_error: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: CommandArgumentError = CommandArgumentError::const_default();
            &DEFAULT
        }
        pub fn with_argument(mut self, field: u32) -> Self {
            self.argument = Some(field.into());
            self
        }
        pub fn index_error(&self) -> &IndexError {
            self.index_error
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| IndexError::default_instance() as _)
        }
        pub fn index_error_opt(&self) -> Option<&IndexError> {
            self.index_error.as_ref().map(|field| field as _)
        }
        pub fn index_error_opt_mut(&mut self) -> Option<&mut IndexError> {
            self.index_error.as_mut().map(|field| field as _)
        }
        pub fn index_error_mut(&mut self) -> &mut IndexError {
            self.index_error.get_or_insert_default()
        }
        pub fn with_index_error(mut self, field: IndexError) -> Self {
            self.index_error = Some(field.into());
            self
        }
    }
    impl PackageUpgradeError {
        pub const fn const_default() -> Self {
            Self {
                kind: None,
                package_id: None,
                digest: None,
                policy: None,
                ticket_id: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: PackageUpgradeError = PackageUpgradeError::const_default();
            &DEFAULT
        }
        pub fn with_package_id(mut self, field: String) -> Self {
            self.package_id = Some(field.into());
            self
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn with_policy(mut self, field: u32) -> Self {
            self.policy = Some(field.into());
            self
        }
        pub fn with_ticket_id(mut self, field: String) -> Self {
            self.ticket_id = Some(field.into());
            self
        }
    }
    impl TypeArgumentError {
        pub const fn const_default() -> Self {
            Self {
                type_argument: None,
                kind: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TypeArgumentError = TypeArgumentError::const_default();
            &DEFAULT
        }
        pub fn with_type_argument(mut self, field: u32) -> Self {
            self.type_argument = Some(field.into());
            self
        }
    }
    impl GasCostSummary {
        pub const fn const_default() -> Self {
            Self {
                computation_cost: None,
                storage_cost: None,
                storage_rebate: None,
                non_refundable_storage_fee: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GasCostSummary = GasCostSummary::const_default();
            &DEFAULT
        }
        pub fn with_computation_cost(mut self, field: u64) -> Self {
            self.computation_cost = Some(field.into());
            self
        }
        pub fn with_storage_cost(mut self, field: u64) -> Self {
            self.storage_cost = Some(field.into());
            self
        }
        pub fn with_storage_rebate(mut self, field: u64) -> Self {
            self.storage_rebate = Some(field.into());
            self
        }
        pub fn with_non_refundable_storage_fee(mut self, field: u64) -> Self {
            self.non_refundable_storage_fee = Some(field.into());
            self
        }
    }
    impl Input {
        pub const fn const_default() -> Self {
            Self {
                kind: None,
                pure: None,
                object_id: None,
                version: None,
                digest: None,
                mutable: None,
                literal: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Input = Input::const_default();
            &DEFAULT
        }
        pub fn with_pure(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.pure = Some(field.into());
            self
        }
        pub fn with_object_id(mut self, field: String) -> Self {
            self.object_id = Some(field.into());
            self
        }
        pub fn with_version(mut self, field: u64) -> Self {
            self.version = Some(field.into());
            self
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn with_mutable(mut self, field: bool) -> Self {
            self.mutable = Some(field.into());
            self
        }
    }
    impl Package {
        pub const fn const_default() -> Self {
            Self {
                storage_id: None,
                original_id: None,
                version: None,
                modules: Vec::new(),
                type_origins: Vec::new(),
                linkage: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Package = Package::const_default();
            &DEFAULT
        }
        pub fn with_storage_id(mut self, field: String) -> Self {
            self.storage_id = Some(field.into());
            self
        }
        pub fn with_original_id(mut self, field: String) -> Self {
            self.original_id = Some(field.into());
            self
        }
        pub fn with_version(mut self, field: u64) -> Self {
            self.version = Some(field.into());
            self
        }
        pub fn modules(&self) -> &[Module] {
            &self.modules
        }
        pub fn with_modules(mut self, field: Vec<Module>) -> Self {
            self.modules = field;
            self
        }
        pub fn type_origins(&self) -> &[TypeOrigin] {
            &self.type_origins
        }
        pub fn with_type_origins(mut self, field: Vec<TypeOrigin>) -> Self {
            self.type_origins = field;
            self
        }
        pub fn linkage(&self) -> &[Linkage] {
            &self.linkage
        }
        pub fn with_linkage(mut self, field: Vec<Linkage>) -> Self {
            self.linkage = field;
            self
        }
    }
    impl Module {
        pub const fn const_default() -> Self {
            Self {
                name: None,
                contents: None,
                datatypes: Vec::new(),
                functions: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Module = Module::const_default();
            &DEFAULT
        }
        pub fn with_name(mut self, field: String) -> Self {
            self.name = Some(field.into());
            self
        }
        pub fn with_contents(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.contents = Some(field.into());
            self
        }
        pub fn datatypes(&self) -> &[DatatypeDescriptor] {
            &self.datatypes
        }
        pub fn with_datatypes(mut self, field: Vec<DatatypeDescriptor>) -> Self {
            self.datatypes = field;
            self
        }
        pub fn functions(&self) -> &[FunctionDescriptor] {
            &self.functions
        }
        pub fn with_functions(mut self, field: Vec<FunctionDescriptor>) -> Self {
            self.functions = field;
            self
        }
    }
    impl DatatypeDescriptor {
        pub const fn const_default() -> Self {
            Self {
                type_name: None,
                defining_id: None,
                module: None,
                name: None,
                abilities: Vec::new(),
                type_parameters: Vec::new(),
                kind: None,
                fields: Vec::new(),
                variants: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: DatatypeDescriptor = DatatypeDescriptor::const_default();
            &DEFAULT
        }
        pub fn with_type_name(mut self, field: String) -> Self {
            self.type_name = Some(field.into());
            self
        }
        pub fn with_defining_id(mut self, field: String) -> Self {
            self.defining_id = Some(field.into());
            self
        }
        pub fn with_module(mut self, field: String) -> Self {
            self.module = Some(field.into());
            self
        }
        pub fn with_name(mut self, field: String) -> Self {
            self.name = Some(field.into());
            self
        }
        pub fn type_parameters(&self) -> &[TypeParameter] {
            &self.type_parameters
        }
        pub fn with_type_parameters(mut self, field: Vec<TypeParameter>) -> Self {
            self.type_parameters = field;
            self
        }
        pub fn fields(&self) -> &[FieldDescriptor] {
            &self.fields
        }
        pub fn with_fields(mut self, field: Vec<FieldDescriptor>) -> Self {
            self.fields = field;
            self
        }
        pub fn variants(&self) -> &[VariantDescriptor] {
            &self.variants
        }
        pub fn with_variants(mut self, field: Vec<VariantDescriptor>) -> Self {
            self.variants = field;
            self
        }
    }
    impl TypeParameter {
        pub const fn const_default() -> Self {
            Self {
                constraints: Vec::new(),
                is_phantom: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TypeParameter = TypeParameter::const_default();
            &DEFAULT
        }
        pub fn with_is_phantom(mut self, field: bool) -> Self {
            self.is_phantom = Some(field.into());
            self
        }
    }
    impl FieldDescriptor {
        pub const fn const_default() -> Self {
            Self {
                name: None,
                position: None,
                r#type: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: FieldDescriptor = FieldDescriptor::const_default();
            &DEFAULT
        }
        pub fn with_name(mut self, field: String) -> Self {
            self.name = Some(field.into());
            self
        }
        pub fn with_position(mut self, field: u32) -> Self {
            self.position = Some(field.into());
            self
        }
        pub fn r#type(&self) -> &OpenSignatureBody {
            self.r#type
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| OpenSignatureBody::default_instance() as _)
        }
        pub fn type_opt(&self) -> Option<&OpenSignatureBody> {
            self.r#type.as_ref().map(|field| field as _)
        }
        pub fn type_opt_mut(&mut self) -> Option<&mut OpenSignatureBody> {
            self.r#type.as_mut().map(|field| field as _)
        }
        pub fn type_mut(&mut self) -> &mut OpenSignatureBody {
            self.r#type.get_or_insert_default()
        }
        pub fn with_type(mut self, field: OpenSignatureBody) -> Self {
            self.r#type = Some(field.into());
            self
        }
    }
    impl VariantDescriptor {
        pub const fn const_default() -> Self {
            Self {
                name: None,
                position: None,
                fields: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: VariantDescriptor = VariantDescriptor::const_default();
            &DEFAULT
        }
        pub fn with_name(mut self, field: String) -> Self {
            self.name = Some(field.into());
            self
        }
        pub fn with_position(mut self, field: u32) -> Self {
            self.position = Some(field.into());
            self
        }
        pub fn fields(&self) -> &[FieldDescriptor] {
            &self.fields
        }
        pub fn with_fields(mut self, field: Vec<FieldDescriptor>) -> Self {
            self.fields = field;
            self
        }
    }
    impl OpenSignatureBody {
        pub const fn const_default() -> Self {
            Self {
                r#type: None,
                type_name: None,
                type_parameter_instantiation: Vec::new(),
                type_parameter: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: OpenSignatureBody = OpenSignatureBody::const_default();
            &DEFAULT
        }
        pub fn with_type_name(mut self, field: String) -> Self {
            self.type_name = Some(field.into());
            self
        }
        pub fn type_parameter_instantiation(&self) -> &[OpenSignatureBody] {
            &self.type_parameter_instantiation
        }
        pub fn with_type_parameter_instantiation(
            mut self,
            field: Vec<OpenSignatureBody>,
        ) -> Self {
            self.type_parameter_instantiation = field;
            self
        }
        pub fn with_type_parameter(mut self, field: u32) -> Self {
            self.type_parameter = Some(field.into());
            self
        }
    }
    impl FunctionDescriptor {
        pub const fn const_default() -> Self {
            Self {
                name: None,
                visibility: None,
                is_entry: None,
                type_parameters: Vec::new(),
                parameters: Vec::new(),
                returns: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: FunctionDescriptor = FunctionDescriptor::const_default();
            &DEFAULT
        }
        pub fn with_name(mut self, field: String) -> Self {
            self.name = Some(field.into());
            self
        }
        pub fn with_is_entry(mut self, field: bool) -> Self {
            self.is_entry = Some(field.into());
            self
        }
        pub fn type_parameters(&self) -> &[TypeParameter] {
            &self.type_parameters
        }
        pub fn with_type_parameters(mut self, field: Vec<TypeParameter>) -> Self {
            self.type_parameters = field;
            self
        }
        pub fn parameters(&self) -> &[OpenSignature] {
            &self.parameters
        }
        pub fn with_parameters(mut self, field: Vec<OpenSignature>) -> Self {
            self.parameters = field;
            self
        }
        pub fn returns(&self) -> &[OpenSignature] {
            &self.returns
        }
        pub fn with_returns(mut self, field: Vec<OpenSignature>) -> Self {
            self.returns = field;
            self
        }
    }
    impl OpenSignature {
        pub const fn const_default() -> Self {
            Self {
                reference: None,
                body: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: OpenSignature = OpenSignature::const_default();
            &DEFAULT
        }
        pub fn body(&self) -> &OpenSignatureBody {
            self.body
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| OpenSignatureBody::default_instance() as _)
        }
        pub fn body_opt(&self) -> Option<&OpenSignatureBody> {
            self.body.as_ref().map(|field| field as _)
        }
        pub fn body_opt_mut(&mut self) -> Option<&mut OpenSignatureBody> {
            self.body.as_mut().map(|field| field as _)
        }
        pub fn body_mut(&mut self) -> &mut OpenSignatureBody {
            self.body.get_or_insert_default()
        }
        pub fn with_body(mut self, field: OpenSignatureBody) -> Self {
            self.body = Some(field.into());
            self
        }
    }
    impl TypeOrigin {
        pub const fn const_default() -> Self {
            Self {
                module_name: None,
                datatype_name: None,
                package_id: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TypeOrigin = TypeOrigin::const_default();
            &DEFAULT
        }
        pub fn with_module_name(mut self, field: String) -> Self {
            self.module_name = Some(field.into());
            self
        }
        pub fn with_datatype_name(mut self, field: String) -> Self {
            self.datatype_name = Some(field.into());
            self
        }
        pub fn with_package_id(mut self, field: String) -> Self {
            self.package_id = Some(field.into());
            self
        }
    }
    impl Linkage {
        pub const fn const_default() -> Self {
            Self {
                original_id: None,
                upgraded_id: None,
                upgraded_version: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Linkage = Linkage::const_default();
            &DEFAULT
        }
        pub fn with_original_id(mut self, field: String) -> Self {
            self.original_id = Some(field.into());
            self
        }
        pub fn with_upgraded_id(mut self, field: String) -> Self {
            self.upgraded_id = Some(field.into());
            self
        }
        pub fn with_upgraded_version(mut self, field: u64) -> Self {
            self.upgraded_version = Some(field.into());
            self
        }
    }
    impl Object {
        pub const fn const_default() -> Self {
            Self {
                bcs: None,
                object_id: None,
                version: None,
                digest: None,
                owner: None,
                object_type: None,
                has_public_transfer: None,
                contents: None,
                package: None,
                previous_transaction: None,
                storage_rebate: None,
                json: None,
                balance: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Object = Object::const_default();
            &DEFAULT
        }
        pub fn bcs(&self) -> &Bcs {
            self.bcs
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Bcs::default_instance() as _)
        }
        pub fn bcs_opt(&self) -> Option<&Bcs> {
            self.bcs.as_ref().map(|field| field as _)
        }
        pub fn bcs_opt_mut(&mut self) -> Option<&mut Bcs> {
            self.bcs.as_mut().map(|field| field as _)
        }
        pub fn bcs_mut(&mut self) -> &mut Bcs {
            self.bcs.get_or_insert_default()
        }
        pub fn with_bcs(mut self, field: Bcs) -> Self {
            self.bcs = Some(field.into());
            self
        }
        pub fn with_object_id(mut self, field: String) -> Self {
            self.object_id = Some(field.into());
            self
        }
        pub fn with_version(mut self, field: u64) -> Self {
            self.version = Some(field.into());
            self
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn owner(&self) -> &Owner {
            self.owner
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Owner::default_instance() as _)
        }
        pub fn owner_opt(&self) -> Option<&Owner> {
            self.owner.as_ref().map(|field| field as _)
        }
        pub fn owner_opt_mut(&mut self) -> Option<&mut Owner> {
            self.owner.as_mut().map(|field| field as _)
        }
        pub fn owner_mut(&mut self) -> &mut Owner {
            self.owner.get_or_insert_default()
        }
        pub fn with_owner(mut self, field: Owner) -> Self {
            self.owner = Some(field.into());
            self
        }
        pub fn with_object_type(mut self, field: String) -> Self {
            self.object_type = Some(field.into());
            self
        }
        pub fn with_has_public_transfer(mut self, field: bool) -> Self {
            self.has_public_transfer = Some(field.into());
            self
        }
        pub fn contents(&self) -> &Bcs {
            self.contents
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Bcs::default_instance() as _)
        }
        pub fn contents_opt(&self) -> Option<&Bcs> {
            self.contents.as_ref().map(|field| field as _)
        }
        pub fn contents_opt_mut(&mut self) -> Option<&mut Bcs> {
            self.contents.as_mut().map(|field| field as _)
        }
        pub fn contents_mut(&mut self) -> &mut Bcs {
            self.contents.get_or_insert_default()
        }
        pub fn with_contents(mut self, field: Bcs) -> Self {
            self.contents = Some(field.into());
            self
        }
        pub fn package(&self) -> &Package {
            self.package
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Package::default_instance() as _)
        }
        pub fn package_opt(&self) -> Option<&Package> {
            self.package.as_ref().map(|field| field as _)
        }
        pub fn package_opt_mut(&mut self) -> Option<&mut Package> {
            self.package.as_mut().map(|field| field as _)
        }
        pub fn package_mut(&mut self) -> &mut Package {
            self.package.get_or_insert_default()
        }
        pub fn with_package(mut self, field: Package) -> Self {
            self.package = Some(field.into());
            self
        }
        pub fn with_previous_transaction(mut self, field: String) -> Self {
            self.previous_transaction = Some(field.into());
            self
        }
        pub fn with_storage_rebate(mut self, field: u64) -> Self {
            self.storage_rebate = Some(field.into());
            self
        }
        pub fn with_balance(mut self, field: u64) -> Self {
            self.balance = Some(field.into());
            self
        }
    }
    impl ObjectReference {
        pub const fn const_default() -> Self {
            Self {
                object_id: None,
                version: None,
                digest: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ObjectReference = ObjectReference::const_default();
            &DEFAULT
        }
        pub fn with_object_id(mut self, field: String) -> Self {
            self.object_id = Some(field.into());
            self
        }
        pub fn with_version(mut self, field: u64) -> Self {
            self.version = Some(field.into());
            self
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
    }
    impl Owner {
        pub const fn const_default() -> Self {
            Self {
                kind: None,
                address: None,
                version: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Owner = Owner::const_default();
            &DEFAULT
        }
        pub fn with_address(mut self, field: String) -> Self {
            self.address = Some(field.into());
            self
        }
        pub fn with_version(mut self, field: u64) -> Self {
            self.version = Some(field.into());
            self
        }
    }
    impl UserSignature {
        pub const fn const_default() -> Self {
            Self {
                bcs: None,
                scheme: None,
                signature: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: UserSignature = UserSignature::const_default();
            &DEFAULT
        }
        pub fn bcs(&self) -> &Bcs {
            self.bcs
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Bcs::default_instance() as _)
        }
        pub fn bcs_opt(&self) -> Option<&Bcs> {
            self.bcs.as_ref().map(|field| field as _)
        }
        pub fn bcs_opt_mut(&mut self) -> Option<&mut Bcs> {
            self.bcs.as_mut().map(|field| field as _)
        }
        pub fn bcs_mut(&mut self) -> &mut Bcs {
            self.bcs.get_or_insert_default()
        }
        pub fn with_bcs(mut self, field: Bcs) -> Self {
            self.bcs = Some(field.into());
            self
        }
        pub fn simple(&self) -> &SimpleSignature {
            if let Some(user_signature::Signature::Simple(field)) = &self.signature {
                field as _
            } else {
                SimpleSignature::default_instance() as _
            }
        }
        pub fn simple_opt(&self) -> Option<&SimpleSignature> {
            if let Some(user_signature::Signature::Simple(field)) = &self.signature {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn simple_opt_mut(&mut self) -> Option<&mut SimpleSignature> {
            if let Some(user_signature::Signature::Simple(field)) = &mut self.signature {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn simple_mut(&mut self) -> &mut SimpleSignature {
            if self.simple_opt_mut().is_none() {
                self.signature = Some(
                    user_signature::Signature::Simple(SimpleSignature::default()),
                );
            }
            self.simple_opt_mut().unwrap()
        }
        pub fn with_simple(mut self, field: SimpleSignature) -> Self {
            self.signature = Some(user_signature::Signature::Simple(field.into()));
            self
        }
        pub fn multisig(&self) -> &MultisigAggregatedSignature {
            if let Some(user_signature::Signature::Multisig(field)) = &self.signature {
                field as _
            } else {
                MultisigAggregatedSignature::default_instance() as _
            }
        }
        pub fn multisig_opt(&self) -> Option<&MultisigAggregatedSignature> {
            if let Some(user_signature::Signature::Multisig(field)) = &self.signature {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn multisig_opt_mut(&mut self) -> Option<&mut MultisigAggregatedSignature> {
            if let Some(user_signature::Signature::Multisig(field)) = &mut self.signature
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn multisig_mut(&mut self) -> &mut MultisigAggregatedSignature {
            if self.multisig_opt_mut().is_none() {
                self.signature = Some(
                    user_signature::Signature::Multisig(
                        MultisigAggregatedSignature::default(),
                    ),
                );
            }
            self.multisig_opt_mut().unwrap()
        }
        pub fn with_multisig(mut self, field: MultisigAggregatedSignature) -> Self {
            self.signature = Some(user_signature::Signature::Multisig(field.into()));
            self
        }
        pub fn zklogin(&self) -> &ZkLoginAuthenticator {
            if let Some(user_signature::Signature::Zklogin(field)) = &self.signature {
                field as _
            } else {
                ZkLoginAuthenticator::default_instance() as _
            }
        }
        pub fn zklogin_opt(&self) -> Option<&ZkLoginAuthenticator> {
            if let Some(user_signature::Signature::Zklogin(field)) = &self.signature {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn zklogin_opt_mut(&mut self) -> Option<&mut ZkLoginAuthenticator> {
            if let Some(user_signature::Signature::Zklogin(field)) = &mut self.signature
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn zklogin_mut(&mut self) -> &mut ZkLoginAuthenticator {
            if self.zklogin_opt_mut().is_none() {
                self.signature = Some(
                    user_signature::Signature::Zklogin(ZkLoginAuthenticator::default()),
                );
            }
            self.zklogin_opt_mut().unwrap()
        }
        pub fn with_zklogin(mut self, field: ZkLoginAuthenticator) -> Self {
            self.signature = Some(user_signature::Signature::Zklogin(field.into()));
            self
        }
        pub fn passkey(&self) -> &PasskeyAuthenticator {
            if let Some(user_signature::Signature::Passkey(field)) = &self.signature {
                field as _
            } else {
                PasskeyAuthenticator::default_instance() as _
            }
        }
        pub fn passkey_opt(&self) -> Option<&PasskeyAuthenticator> {
            if let Some(user_signature::Signature::Passkey(field)) = &self.signature {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn passkey_opt_mut(&mut self) -> Option<&mut PasskeyAuthenticator> {
            if let Some(user_signature::Signature::Passkey(field)) = &mut self.signature
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn passkey_mut(&mut self) -> &mut PasskeyAuthenticator {
            if self.passkey_opt_mut().is_none() {
                self.signature = Some(
                    user_signature::Signature::Passkey(PasskeyAuthenticator::default()),
                );
            }
            self.passkey_opt_mut().unwrap()
        }
        pub fn with_passkey(mut self, field: PasskeyAuthenticator) -> Self {
            self.signature = Some(user_signature::Signature::Passkey(field.into()));
            self
        }
    }
    impl SimpleSignature {
        pub const fn const_default() -> Self {
            Self {
                scheme: None,
                signature: None,
                public_key: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SimpleSignature = SimpleSignature::const_default();
            &DEFAULT
        }
        pub fn with_signature(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.signature = Some(field.into());
            self
        }
        pub fn with_public_key(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.public_key = Some(field.into());
            self
        }
    }
    impl ZkLoginPublicIdentifier {
        pub const fn const_default() -> Self {
            Self {
                iss: None,
                address_seed: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ZkLoginPublicIdentifier = ZkLoginPublicIdentifier::const_default();
            &DEFAULT
        }
        pub fn with_iss(mut self, field: String) -> Self {
            self.iss = Some(field.into());
            self
        }
        pub fn with_address_seed(mut self, field: String) -> Self {
            self.address_seed = Some(field.into());
            self
        }
    }
    impl MultisigMemberPublicKey {
        pub const fn const_default() -> Self {
            Self {
                scheme: None,
                public_key: None,
                zklogin: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: MultisigMemberPublicKey = MultisigMemberPublicKey::const_default();
            &DEFAULT
        }
        pub fn with_public_key(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.public_key = Some(field.into());
            self
        }
        pub fn zklogin(&self) -> &ZkLoginPublicIdentifier {
            self.zklogin
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ZkLoginPublicIdentifier::default_instance() as _)
        }
        pub fn zklogin_opt(&self) -> Option<&ZkLoginPublicIdentifier> {
            self.zklogin.as_ref().map(|field| field as _)
        }
        pub fn zklogin_opt_mut(&mut self) -> Option<&mut ZkLoginPublicIdentifier> {
            self.zklogin.as_mut().map(|field| field as _)
        }
        pub fn zklogin_mut(&mut self) -> &mut ZkLoginPublicIdentifier {
            self.zklogin.get_or_insert_default()
        }
        pub fn with_zklogin(mut self, field: ZkLoginPublicIdentifier) -> Self {
            self.zklogin = Some(field.into());
            self
        }
    }
    impl MultisigMember {
        pub const fn const_default() -> Self {
            Self {
                public_key: None,
                weight: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: MultisigMember = MultisigMember::const_default();
            &DEFAULT
        }
        pub fn public_key(&self) -> &MultisigMemberPublicKey {
            self.public_key
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| MultisigMemberPublicKey::default_instance() as _)
        }
        pub fn public_key_opt(&self) -> Option<&MultisigMemberPublicKey> {
            self.public_key.as_ref().map(|field| field as _)
        }
        pub fn public_key_opt_mut(&mut self) -> Option<&mut MultisigMemberPublicKey> {
            self.public_key.as_mut().map(|field| field as _)
        }
        pub fn public_key_mut(&mut self) -> &mut MultisigMemberPublicKey {
            self.public_key.get_or_insert_default()
        }
        pub fn with_public_key(mut self, field: MultisigMemberPublicKey) -> Self {
            self.public_key = Some(field.into());
            self
        }
        pub fn with_weight(mut self, field: u32) -> Self {
            self.weight = Some(field.into());
            self
        }
    }
    impl MultisigCommittee {
        pub const fn const_default() -> Self {
            Self {
                members: Vec::new(),
                threshold: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: MultisigCommittee = MultisigCommittee::const_default();
            &DEFAULT
        }
        pub fn members(&self) -> &[MultisigMember] {
            &self.members
        }
        pub fn with_members(mut self, field: Vec<MultisigMember>) -> Self {
            self.members = field;
            self
        }
        pub fn with_threshold(mut self, field: u32) -> Self {
            self.threshold = Some(field.into());
            self
        }
    }
    impl MultisigAggregatedSignature {
        pub const fn const_default() -> Self {
            Self {
                signatures: Vec::new(),
                bitmap: None,
                legacy_bitmap: Vec::new(),
                committee: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: MultisigAggregatedSignature = MultisigAggregatedSignature::const_default();
            &DEFAULT
        }
        pub fn signatures(&self) -> &[MultisigMemberSignature] {
            &self.signatures
        }
        pub fn with_signatures(mut self, field: Vec<MultisigMemberSignature>) -> Self {
            self.signatures = field;
            self
        }
        pub fn with_bitmap(mut self, field: u32) -> Self {
            self.bitmap = Some(field.into());
            self
        }
        pub fn legacy_bitmap(&self) -> &[u32] {
            &self.legacy_bitmap
        }
        pub fn with_legacy_bitmap(mut self, field: Vec<u32>) -> Self {
            self.legacy_bitmap = field;
            self
        }
        pub fn committee(&self) -> &MultisigCommittee {
            self.committee
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| MultisigCommittee::default_instance() as _)
        }
        pub fn committee_opt(&self) -> Option<&MultisigCommittee> {
            self.committee.as_ref().map(|field| field as _)
        }
        pub fn committee_opt_mut(&mut self) -> Option<&mut MultisigCommittee> {
            self.committee.as_mut().map(|field| field as _)
        }
        pub fn committee_mut(&mut self) -> &mut MultisigCommittee {
            self.committee.get_or_insert_default()
        }
        pub fn with_committee(mut self, field: MultisigCommittee) -> Self {
            self.committee = Some(field.into());
            self
        }
    }
    impl MultisigMemberSignature {
        pub const fn const_default() -> Self {
            Self {
                scheme: None,
                signature: None,
                zklogin: None,
                passkey: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: MultisigMemberSignature = MultisigMemberSignature::const_default();
            &DEFAULT
        }
        pub fn with_signature(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.signature = Some(field.into());
            self
        }
        pub fn zklogin(&self) -> &ZkLoginAuthenticator {
            self.zklogin
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ZkLoginAuthenticator::default_instance() as _)
        }
        pub fn zklogin_opt(&self) -> Option<&ZkLoginAuthenticator> {
            self.zklogin.as_ref().map(|field| field as _)
        }
        pub fn zklogin_opt_mut(&mut self) -> Option<&mut ZkLoginAuthenticator> {
            self.zklogin.as_mut().map(|field| field as _)
        }
        pub fn zklogin_mut(&mut self) -> &mut ZkLoginAuthenticator {
            self.zklogin.get_or_insert_default()
        }
        pub fn with_zklogin(mut self, field: ZkLoginAuthenticator) -> Self {
            self.zklogin = Some(field.into());
            self
        }
        pub fn passkey(&self) -> &PasskeyAuthenticator {
            self.passkey
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| PasskeyAuthenticator::default_instance() as _)
        }
        pub fn passkey_opt(&self) -> Option<&PasskeyAuthenticator> {
            self.passkey.as_ref().map(|field| field as _)
        }
        pub fn passkey_opt_mut(&mut self) -> Option<&mut PasskeyAuthenticator> {
            self.passkey.as_mut().map(|field| field as _)
        }
        pub fn passkey_mut(&mut self) -> &mut PasskeyAuthenticator {
            self.passkey.get_or_insert_default()
        }
        pub fn with_passkey(mut self, field: PasskeyAuthenticator) -> Self {
            self.passkey = Some(field.into());
            self
        }
    }
    impl ZkLoginAuthenticator {
        pub const fn const_default() -> Self {
            Self {
                inputs: None,
                max_epoch: None,
                signature: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ZkLoginAuthenticator = ZkLoginAuthenticator::const_default();
            &DEFAULT
        }
        pub fn inputs(&self) -> &ZkLoginInputs {
            self.inputs
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ZkLoginInputs::default_instance() as _)
        }
        pub fn inputs_opt(&self) -> Option<&ZkLoginInputs> {
            self.inputs.as_ref().map(|field| field as _)
        }
        pub fn inputs_opt_mut(&mut self) -> Option<&mut ZkLoginInputs> {
            self.inputs.as_mut().map(|field| field as _)
        }
        pub fn inputs_mut(&mut self) -> &mut ZkLoginInputs {
            self.inputs.get_or_insert_default()
        }
        pub fn with_inputs(mut self, field: ZkLoginInputs) -> Self {
            self.inputs = Some(field.into());
            self
        }
        pub fn with_max_epoch(mut self, field: u64) -> Self {
            self.max_epoch = Some(field.into());
            self
        }
        pub fn signature(&self) -> &SimpleSignature {
            self.signature
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| SimpleSignature::default_instance() as _)
        }
        pub fn signature_opt(&self) -> Option<&SimpleSignature> {
            self.signature.as_ref().map(|field| field as _)
        }
        pub fn signature_opt_mut(&mut self) -> Option<&mut SimpleSignature> {
            self.signature.as_mut().map(|field| field as _)
        }
        pub fn signature_mut(&mut self) -> &mut SimpleSignature {
            self.signature.get_or_insert_default()
        }
        pub fn with_signature(mut self, field: SimpleSignature) -> Self {
            self.signature = Some(field.into());
            self
        }
    }
    impl ZkLoginInputs {
        pub const fn const_default() -> Self {
            Self {
                proof_points: None,
                iss_base64_details: None,
                header_base64: None,
                address_seed: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ZkLoginInputs = ZkLoginInputs::const_default();
            &DEFAULT
        }
        pub fn proof_points(&self) -> &ZkLoginProof {
            self.proof_points
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ZkLoginProof::default_instance() as _)
        }
        pub fn proof_points_opt(&self) -> Option<&ZkLoginProof> {
            self.proof_points.as_ref().map(|field| field as _)
        }
        pub fn proof_points_opt_mut(&mut self) -> Option<&mut ZkLoginProof> {
            self.proof_points.as_mut().map(|field| field as _)
        }
        pub fn proof_points_mut(&mut self) -> &mut ZkLoginProof {
            self.proof_points.get_or_insert_default()
        }
        pub fn with_proof_points(mut self, field: ZkLoginProof) -> Self {
            self.proof_points = Some(field.into());
            self
        }
        pub fn iss_base64_details(&self) -> &ZkLoginClaim {
            self.iss_base64_details
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ZkLoginClaim::default_instance() as _)
        }
        pub fn iss_base64_details_opt(&self) -> Option<&ZkLoginClaim> {
            self.iss_base64_details.as_ref().map(|field| field as _)
        }
        pub fn iss_base64_details_opt_mut(&mut self) -> Option<&mut ZkLoginClaim> {
            self.iss_base64_details.as_mut().map(|field| field as _)
        }
        pub fn iss_base64_details_mut(&mut self) -> &mut ZkLoginClaim {
            self.iss_base64_details.get_or_insert_default()
        }
        pub fn with_iss_base64_details(mut self, field: ZkLoginClaim) -> Self {
            self.iss_base64_details = Some(field.into());
            self
        }
        pub fn with_header_base64(mut self, field: String) -> Self {
            self.header_base64 = Some(field.into());
            self
        }
        pub fn with_address_seed(mut self, field: String) -> Self {
            self.address_seed = Some(field.into());
            self
        }
    }
    impl ZkLoginProof {
        pub const fn const_default() -> Self {
            Self { a: None, b: None, c: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ZkLoginProof = ZkLoginProof::const_default();
            &DEFAULT
        }
        pub fn a(&self) -> &CircomG1 {
            self.a
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| CircomG1::default_instance() as _)
        }
        pub fn a_opt(&self) -> Option<&CircomG1> {
            self.a.as_ref().map(|field| field as _)
        }
        pub fn a_opt_mut(&mut self) -> Option<&mut CircomG1> {
            self.a.as_mut().map(|field| field as _)
        }
        pub fn a_mut(&mut self) -> &mut CircomG1 {
            self.a.get_or_insert_default()
        }
        pub fn with_a(mut self, field: CircomG1) -> Self {
            self.a = Some(field.into());
            self
        }
        pub fn b(&self) -> &CircomG2 {
            self.b
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| CircomG2::default_instance() as _)
        }
        pub fn b_opt(&self) -> Option<&CircomG2> {
            self.b.as_ref().map(|field| field as _)
        }
        pub fn b_opt_mut(&mut self) -> Option<&mut CircomG2> {
            self.b.as_mut().map(|field| field as _)
        }
        pub fn b_mut(&mut self) -> &mut CircomG2 {
            self.b.get_or_insert_default()
        }
        pub fn with_b(mut self, field: CircomG2) -> Self {
            self.b = Some(field.into());
            self
        }
        pub fn c(&self) -> &CircomG1 {
            self.c
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| CircomG1::default_instance() as _)
        }
        pub fn c_opt(&self) -> Option<&CircomG1> {
            self.c.as_ref().map(|field| field as _)
        }
        pub fn c_opt_mut(&mut self) -> Option<&mut CircomG1> {
            self.c.as_mut().map(|field| field as _)
        }
        pub fn c_mut(&mut self) -> &mut CircomG1 {
            self.c.get_or_insert_default()
        }
        pub fn with_c(mut self, field: CircomG1) -> Self {
            self.c = Some(field.into());
            self
        }
    }
    impl ZkLoginClaim {
        pub const fn const_default() -> Self {
            Self {
                value: None,
                index_mod_4: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ZkLoginClaim = ZkLoginClaim::const_default();
            &DEFAULT
        }
        pub fn with_value(mut self, field: String) -> Self {
            self.value = Some(field.into());
            self
        }
        pub fn with_index_mod_4(mut self, field: u32) -> Self {
            self.index_mod_4 = Some(field.into());
            self
        }
    }
    impl CircomG1 {
        pub const fn const_default() -> Self {
            Self {
                e0: None,
                e1: None,
                e2: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: CircomG1 = CircomG1::const_default();
            &DEFAULT
        }
        pub fn with_e0(mut self, field: String) -> Self {
            self.e0 = Some(field.into());
            self
        }
        pub fn with_e1(mut self, field: String) -> Self {
            self.e1 = Some(field.into());
            self
        }
        pub fn with_e2(mut self, field: String) -> Self {
            self.e2 = Some(field.into());
            self
        }
    }
    impl CircomG2 {
        pub const fn const_default() -> Self {
            Self {
                e00: None,
                e01: None,
                e10: None,
                e11: None,
                e20: None,
                e21: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: CircomG2 = CircomG2::const_default();
            &DEFAULT
        }
        pub fn with_e00(mut self, field: String) -> Self {
            self.e00 = Some(field.into());
            self
        }
        pub fn with_e01(mut self, field: String) -> Self {
            self.e01 = Some(field.into());
            self
        }
        pub fn with_e10(mut self, field: String) -> Self {
            self.e10 = Some(field.into());
            self
        }
        pub fn with_e11(mut self, field: String) -> Self {
            self.e11 = Some(field.into());
            self
        }
        pub fn with_e20(mut self, field: String) -> Self {
            self.e20 = Some(field.into());
            self
        }
        pub fn with_e21(mut self, field: String) -> Self {
            self.e21 = Some(field.into());
            self
        }
    }
    impl PasskeyAuthenticator {
        pub const fn const_default() -> Self {
            Self {
                authenticator_data: None,
                client_data_json: None,
                signature: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: PasskeyAuthenticator = PasskeyAuthenticator::const_default();
            &DEFAULT
        }
        pub fn with_authenticator_data(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.authenticator_data = Some(field.into());
            self
        }
        pub fn with_client_data_json(mut self, field: String) -> Self {
            self.client_data_json = Some(field.into());
            self
        }
        pub fn signature(&self) -> &SimpleSignature {
            self.signature
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| SimpleSignature::default_instance() as _)
        }
        pub fn signature_opt(&self) -> Option<&SimpleSignature> {
            self.signature.as_ref().map(|field| field as _)
        }
        pub fn signature_opt_mut(&mut self) -> Option<&mut SimpleSignature> {
            self.signature.as_mut().map(|field| field as _)
        }
        pub fn signature_mut(&mut self) -> &mut SimpleSignature {
            self.signature.get_or_insert_default()
        }
        pub fn with_signature(mut self, field: SimpleSignature) -> Self {
            self.signature = Some(field.into());
            self
        }
    }
    impl ValidatorCommittee {
        pub const fn const_default() -> Self {
            Self {
                epoch: None,
                members: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ValidatorCommittee = ValidatorCommittee::const_default();
            &DEFAULT
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
        pub fn members(&self) -> &[ValidatorCommitteeMember] {
            &self.members
        }
        pub fn with_members(mut self, field: Vec<ValidatorCommitteeMember>) -> Self {
            self.members = field;
            self
        }
    }
    impl ValidatorCommitteeMember {
        pub const fn const_default() -> Self {
            Self {
                public_key: None,
                weight: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ValidatorCommitteeMember = ValidatorCommitteeMember::const_default();
            &DEFAULT
        }
        pub fn with_public_key(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.public_key = Some(field.into());
            self
        }
        pub fn with_weight(mut self, field: u64) -> Self {
            self.weight = Some(field.into());
            self
        }
    }
    impl ValidatorAggregatedSignature {
        pub const fn const_default() -> Self {
            Self {
                epoch: None,
                signature: None,
                bitmap: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ValidatorAggregatedSignature = ValidatorAggregatedSignature::const_default();
            &DEFAULT
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
        pub fn with_signature(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.signature = Some(field.into());
            self
        }
        pub fn bitmap(&self) -> &[u32] {
            &self.bitmap
        }
        pub fn with_bitmap(mut self, field: Vec<u32>) -> Self {
            self.bitmap = field;
            self
        }
    }
    impl Transaction {
        pub const fn const_default() -> Self {
            Self {
                bcs: None,
                digest: None,
                version: None,
                kind: None,
                sender: None,
                gas_payment: None,
                expiration: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Transaction = Transaction::const_default();
            &DEFAULT
        }
        pub fn bcs(&self) -> &Bcs {
            self.bcs
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Bcs::default_instance() as _)
        }
        pub fn bcs_opt(&self) -> Option<&Bcs> {
            self.bcs.as_ref().map(|field| field as _)
        }
        pub fn bcs_opt_mut(&mut self) -> Option<&mut Bcs> {
            self.bcs.as_mut().map(|field| field as _)
        }
        pub fn bcs_mut(&mut self) -> &mut Bcs {
            self.bcs.get_or_insert_default()
        }
        pub fn with_bcs(mut self, field: Bcs) -> Self {
            self.bcs = Some(field.into());
            self
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn with_version(mut self, field: i32) -> Self {
            self.version = Some(field.into());
            self
        }
        pub fn kind(&self) -> &TransactionKind {
            self.kind
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| TransactionKind::default_instance() as _)
        }
        pub fn kind_opt(&self) -> Option<&TransactionKind> {
            self.kind.as_ref().map(|field| field as _)
        }
        pub fn kind_opt_mut(&mut self) -> Option<&mut TransactionKind> {
            self.kind.as_mut().map(|field| field as _)
        }
        pub fn kind_mut(&mut self) -> &mut TransactionKind {
            self.kind.get_or_insert_default()
        }
        pub fn with_kind(mut self, field: TransactionKind) -> Self {
            self.kind = Some(field.into());
            self
        }
        pub fn with_sender(mut self, field: String) -> Self {
            self.sender = Some(field.into());
            self
        }
        pub fn gas_payment(&self) -> &GasPayment {
            self.gas_payment
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| GasPayment::default_instance() as _)
        }
        pub fn gas_payment_opt(&self) -> Option<&GasPayment> {
            self.gas_payment.as_ref().map(|field| field as _)
        }
        pub fn gas_payment_opt_mut(&mut self) -> Option<&mut GasPayment> {
            self.gas_payment.as_mut().map(|field| field as _)
        }
        pub fn gas_payment_mut(&mut self) -> &mut GasPayment {
            self.gas_payment.get_or_insert_default()
        }
        pub fn with_gas_payment(mut self, field: GasPayment) -> Self {
            self.gas_payment = Some(field.into());
            self
        }
        pub fn expiration(&self) -> &TransactionExpiration {
            self.expiration
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| TransactionExpiration::default_instance() as _)
        }
        pub fn expiration_opt(&self) -> Option<&TransactionExpiration> {
            self.expiration.as_ref().map(|field| field as _)
        }
        pub fn expiration_opt_mut(&mut self) -> Option<&mut TransactionExpiration> {
            self.expiration.as_mut().map(|field| field as _)
        }
        pub fn expiration_mut(&mut self) -> &mut TransactionExpiration {
            self.expiration.get_or_insert_default()
        }
        pub fn with_expiration(mut self, field: TransactionExpiration) -> Self {
            self.expiration = Some(field.into());
            self
        }
    }
    impl GasPayment {
        pub const fn const_default() -> Self {
            Self {
                objects: Vec::new(),
                owner: None,
                price: None,
                budget: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GasPayment = GasPayment::const_default();
            &DEFAULT
        }
        pub fn objects(&self) -> &[ObjectReference] {
            &self.objects
        }
        pub fn with_objects(mut self, field: Vec<ObjectReference>) -> Self {
            self.objects = field;
            self
        }
        pub fn with_owner(mut self, field: String) -> Self {
            self.owner = Some(field.into());
            self
        }
        pub fn with_price(mut self, field: u64) -> Self {
            self.price = Some(field.into());
            self
        }
        pub fn with_budget(mut self, field: u64) -> Self {
            self.budget = Some(field.into());
            self
        }
    }
    impl TransactionExpiration {
        pub const fn const_default() -> Self {
            Self { kind: None, epoch: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TransactionExpiration = TransactionExpiration::const_default();
            &DEFAULT
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
    }
    impl TransactionKind {
        pub const fn const_default() -> Self {
            Self { kind: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TransactionKind = TransactionKind::const_default();
            &DEFAULT
        }
        pub fn programmable_transaction(&self) -> &ProgrammableTransaction {
            if let Some(transaction_kind::Kind::ProgrammableTransaction(field)) = &self
                .kind
            {
                field as _
            } else {
                ProgrammableTransaction::default_instance() as _
            }
        }
        pub fn programmable_transaction_opt(&self) -> Option<&ProgrammableTransaction> {
            if let Some(transaction_kind::Kind::ProgrammableTransaction(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn programmable_transaction_opt_mut(
            &mut self,
        ) -> Option<&mut ProgrammableTransaction> {
            if let Some(transaction_kind::Kind::ProgrammableTransaction(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn programmable_transaction_mut(&mut self) -> &mut ProgrammableTransaction {
            if self.programmable_transaction_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ProgrammableTransaction(
                        ProgrammableTransaction::default(),
                    ),
                );
            }
            self.programmable_transaction_opt_mut().unwrap()
        }
        pub fn with_programmable_transaction(
            mut self,
            field: ProgrammableTransaction,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::ProgrammableTransaction(field.into()),
            );
            self
        }
        pub fn programmable_system_transaction(&self) -> &ProgrammableTransaction {
            if let Some(transaction_kind::Kind::ProgrammableSystemTransaction(field)) = &self
                .kind
            {
                field as _
            } else {
                ProgrammableTransaction::default_instance() as _
            }
        }
        pub fn programmable_system_transaction_opt(
            &self,
        ) -> Option<&ProgrammableTransaction> {
            if let Some(transaction_kind::Kind::ProgrammableSystemTransaction(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn programmable_system_transaction_opt_mut(
            &mut self,
        ) -> Option<&mut ProgrammableTransaction> {
            if let Some(transaction_kind::Kind::ProgrammableSystemTransaction(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn programmable_system_transaction_mut(
            &mut self,
        ) -> &mut ProgrammableTransaction {
            if self.programmable_system_transaction_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ProgrammableSystemTransaction(
                        ProgrammableTransaction::default(),
                    ),
                );
            }
            self.programmable_system_transaction_opt_mut().unwrap()
        }
        pub fn with_programmable_system_transaction(
            mut self,
            field: ProgrammableTransaction,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::ProgrammableSystemTransaction(field.into()),
            );
            self
        }
        pub fn change_epoch(&self) -> &ChangeEpoch {
            if let Some(transaction_kind::Kind::ChangeEpoch(field)) = &self.kind {
                field as _
            } else {
                ChangeEpoch::default_instance() as _
            }
        }
        pub fn change_epoch_opt(&self) -> Option<&ChangeEpoch> {
            if let Some(transaction_kind::Kind::ChangeEpoch(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn change_epoch_opt_mut(&mut self) -> Option<&mut ChangeEpoch> {
            if let Some(transaction_kind::Kind::ChangeEpoch(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn change_epoch_mut(&mut self) -> &mut ChangeEpoch {
            if self.change_epoch_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ChangeEpoch(ChangeEpoch::default()),
                );
            }
            self.change_epoch_opt_mut().unwrap()
        }
        pub fn with_change_epoch(mut self, field: ChangeEpoch) -> Self {
            self.kind = Some(transaction_kind::Kind::ChangeEpoch(field.into()));
            self
        }
        pub fn genesis(&self) -> &GenesisTransaction {
            if let Some(transaction_kind::Kind::Genesis(field)) = &self.kind {
                field as _
            } else {
                GenesisTransaction::default_instance() as _
            }
        }
        pub fn genesis_opt(&self) -> Option<&GenesisTransaction> {
            if let Some(transaction_kind::Kind::Genesis(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn genesis_opt_mut(&mut self) -> Option<&mut GenesisTransaction> {
            if let Some(transaction_kind::Kind::Genesis(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn genesis_mut(&mut self) -> &mut GenesisTransaction {
            if self.genesis_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::Genesis(GenesisTransaction::default()),
                );
            }
            self.genesis_opt_mut().unwrap()
        }
        pub fn with_genesis(mut self, field: GenesisTransaction) -> Self {
            self.kind = Some(transaction_kind::Kind::Genesis(field.into()));
            self
        }
        pub fn consensus_commit_prologue_v1(&self) -> &ConsensusCommitPrologue {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV1(field)) = &self
                .kind
            {
                field as _
            } else {
                ConsensusCommitPrologue::default_instance() as _
            }
        }
        pub fn consensus_commit_prologue_v1_opt(
            &self,
        ) -> Option<&ConsensusCommitPrologue> {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV1(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn consensus_commit_prologue_v1_opt_mut(
            &mut self,
        ) -> Option<&mut ConsensusCommitPrologue> {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV1(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn consensus_commit_prologue_v1_mut(
            &mut self,
        ) -> &mut ConsensusCommitPrologue {
            if self.consensus_commit_prologue_v1_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ConsensusCommitPrologueV1(
                        ConsensusCommitPrologue::default(),
                    ),
                );
            }
            self.consensus_commit_prologue_v1_opt_mut().unwrap()
        }
        pub fn with_consensus_commit_prologue_v1(
            mut self,
            field: ConsensusCommitPrologue,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::ConsensusCommitPrologueV1(field.into()),
            );
            self
        }
        pub fn authenticator_state_update(&self) -> &AuthenticatorStateUpdate {
            if let Some(transaction_kind::Kind::AuthenticatorStateUpdate(field)) = &self
                .kind
            {
                field as _
            } else {
                AuthenticatorStateUpdate::default_instance() as _
            }
        }
        pub fn authenticator_state_update_opt(
            &self,
        ) -> Option<&AuthenticatorStateUpdate> {
            if let Some(transaction_kind::Kind::AuthenticatorStateUpdate(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn authenticator_state_update_opt_mut(
            &mut self,
        ) -> Option<&mut AuthenticatorStateUpdate> {
            if let Some(transaction_kind::Kind::AuthenticatorStateUpdate(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn authenticator_state_update_mut(
            &mut self,
        ) -> &mut AuthenticatorStateUpdate {
            if self.authenticator_state_update_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::AuthenticatorStateUpdate(
                        AuthenticatorStateUpdate::default(),
                    ),
                );
            }
            self.authenticator_state_update_opt_mut().unwrap()
        }
        pub fn with_authenticator_state_update(
            mut self,
            field: AuthenticatorStateUpdate,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::AuthenticatorStateUpdate(field.into()),
            );
            self
        }
        pub fn end_of_epoch(&self) -> &EndOfEpochTransaction {
            if let Some(transaction_kind::Kind::EndOfEpoch(field)) = &self.kind {
                field as _
            } else {
                EndOfEpochTransaction::default_instance() as _
            }
        }
        pub fn end_of_epoch_opt(&self) -> Option<&EndOfEpochTransaction> {
            if let Some(transaction_kind::Kind::EndOfEpoch(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn end_of_epoch_opt_mut(&mut self) -> Option<&mut EndOfEpochTransaction> {
            if let Some(transaction_kind::Kind::EndOfEpoch(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn end_of_epoch_mut(&mut self) -> &mut EndOfEpochTransaction {
            if self.end_of_epoch_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::EndOfEpoch(EndOfEpochTransaction::default()),
                );
            }
            self.end_of_epoch_opt_mut().unwrap()
        }
        pub fn with_end_of_epoch(mut self, field: EndOfEpochTransaction) -> Self {
            self.kind = Some(transaction_kind::Kind::EndOfEpoch(field.into()));
            self
        }
        pub fn randomness_state_update(&self) -> &RandomnessStateUpdate {
            if let Some(transaction_kind::Kind::RandomnessStateUpdate(field)) = &self
                .kind
            {
                field as _
            } else {
                RandomnessStateUpdate::default_instance() as _
            }
        }
        pub fn randomness_state_update_opt(&self) -> Option<&RandomnessStateUpdate> {
            if let Some(transaction_kind::Kind::RandomnessStateUpdate(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn randomness_state_update_opt_mut(
            &mut self,
        ) -> Option<&mut RandomnessStateUpdate> {
            if let Some(transaction_kind::Kind::RandomnessStateUpdate(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn randomness_state_update_mut(&mut self) -> &mut RandomnessStateUpdate {
            if self.randomness_state_update_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::RandomnessStateUpdate(
                        RandomnessStateUpdate::default(),
                    ),
                );
            }
            self.randomness_state_update_opt_mut().unwrap()
        }
        pub fn with_randomness_state_update(
            mut self,
            field: RandomnessStateUpdate,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::RandomnessStateUpdate(field.into()),
            );
            self
        }
        pub fn consensus_commit_prologue_v2(&self) -> &ConsensusCommitPrologue {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV2(field)) = &self
                .kind
            {
                field as _
            } else {
                ConsensusCommitPrologue::default_instance() as _
            }
        }
        pub fn consensus_commit_prologue_v2_opt(
            &self,
        ) -> Option<&ConsensusCommitPrologue> {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV2(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn consensus_commit_prologue_v2_opt_mut(
            &mut self,
        ) -> Option<&mut ConsensusCommitPrologue> {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV2(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn consensus_commit_prologue_v2_mut(
            &mut self,
        ) -> &mut ConsensusCommitPrologue {
            if self.consensus_commit_prologue_v2_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ConsensusCommitPrologueV2(
                        ConsensusCommitPrologue::default(),
                    ),
                );
            }
            self.consensus_commit_prologue_v2_opt_mut().unwrap()
        }
        pub fn with_consensus_commit_prologue_v2(
            mut self,
            field: ConsensusCommitPrologue,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::ConsensusCommitPrologueV2(field.into()),
            );
            self
        }
        pub fn consensus_commit_prologue_v3(&self) -> &ConsensusCommitPrologue {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV3(field)) = &self
                .kind
            {
                field as _
            } else {
                ConsensusCommitPrologue::default_instance() as _
            }
        }
        pub fn consensus_commit_prologue_v3_opt(
            &self,
        ) -> Option<&ConsensusCommitPrologue> {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV3(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn consensus_commit_prologue_v3_opt_mut(
            &mut self,
        ) -> Option<&mut ConsensusCommitPrologue> {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV3(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn consensus_commit_prologue_v3_mut(
            &mut self,
        ) -> &mut ConsensusCommitPrologue {
            if self.consensus_commit_prologue_v3_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ConsensusCommitPrologueV3(
                        ConsensusCommitPrologue::default(),
                    ),
                );
            }
            self.consensus_commit_prologue_v3_opt_mut().unwrap()
        }
        pub fn with_consensus_commit_prologue_v3(
            mut self,
            field: ConsensusCommitPrologue,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::ConsensusCommitPrologueV3(field.into()),
            );
            self
        }
        pub fn consensus_commit_prologue_v4(&self) -> &ConsensusCommitPrologue {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV4(field)) = &self
                .kind
            {
                field as _
            } else {
                ConsensusCommitPrologue::default_instance() as _
            }
        }
        pub fn consensus_commit_prologue_v4_opt(
            &self,
        ) -> Option<&ConsensusCommitPrologue> {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV4(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn consensus_commit_prologue_v4_opt_mut(
            &mut self,
        ) -> Option<&mut ConsensusCommitPrologue> {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologueV4(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn consensus_commit_prologue_v4_mut(
            &mut self,
        ) -> &mut ConsensusCommitPrologue {
            if self.consensus_commit_prologue_v4_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ConsensusCommitPrologueV4(
                        ConsensusCommitPrologue::default(),
                    ),
                );
            }
            self.consensus_commit_prologue_v4_opt_mut().unwrap()
        }
        pub fn with_consensus_commit_prologue_v4(
            mut self,
            field: ConsensusCommitPrologue,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::ConsensusCommitPrologueV4(field.into()),
            );
            self
        }
    }
    impl ProgrammableTransaction {
        pub const fn const_default() -> Self {
            Self {
                inputs: Vec::new(),
                commands: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ProgrammableTransaction = ProgrammableTransaction::const_default();
            &DEFAULT
        }
        pub fn inputs(&self) -> &[Input] {
            &self.inputs
        }
        pub fn with_inputs(mut self, field: Vec<Input>) -> Self {
            self.inputs = field;
            self
        }
        pub fn commands(&self) -> &[Command] {
            &self.commands
        }
        pub fn with_commands(mut self, field: Vec<Command>) -> Self {
            self.commands = field;
            self
        }
    }
    impl Command {
        pub const fn const_default() -> Self {
            Self { command: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Command = Command::const_default();
            &DEFAULT
        }
        pub fn move_call(&self) -> &MoveCall {
            if let Some(command::Command::MoveCall(field)) = &self.command {
                field as _
            } else {
                MoveCall::default_instance() as _
            }
        }
        pub fn move_call_opt(&self) -> Option<&MoveCall> {
            if let Some(command::Command::MoveCall(field)) = &self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn move_call_opt_mut(&mut self) -> Option<&mut MoveCall> {
            if let Some(command::Command::MoveCall(field)) = &mut self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn move_call_mut(&mut self) -> &mut MoveCall {
            if self.move_call_opt_mut().is_none() {
                self.command = Some(command::Command::MoveCall(MoveCall::default()));
            }
            self.move_call_opt_mut().unwrap()
        }
        pub fn with_move_call(mut self, field: MoveCall) -> Self {
            self.command = Some(command::Command::MoveCall(field.into()));
            self
        }
        pub fn transfer_objects(&self) -> &TransferObjects {
            if let Some(command::Command::TransferObjects(field)) = &self.command {
                field as _
            } else {
                TransferObjects::default_instance() as _
            }
        }
        pub fn transfer_objects_opt(&self) -> Option<&TransferObjects> {
            if let Some(command::Command::TransferObjects(field)) = &self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn transfer_objects_opt_mut(&mut self) -> Option<&mut TransferObjects> {
            if let Some(command::Command::TransferObjects(field)) = &mut self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn transfer_objects_mut(&mut self) -> &mut TransferObjects {
            if self.transfer_objects_opt_mut().is_none() {
                self.command = Some(
                    command::Command::TransferObjects(TransferObjects::default()),
                );
            }
            self.transfer_objects_opt_mut().unwrap()
        }
        pub fn with_transfer_objects(mut self, field: TransferObjects) -> Self {
            self.command = Some(command::Command::TransferObjects(field.into()));
            self
        }
        pub fn split_coins(&self) -> &SplitCoins {
            if let Some(command::Command::SplitCoins(field)) = &self.command {
                field as _
            } else {
                SplitCoins::default_instance() as _
            }
        }
        pub fn split_coins_opt(&self) -> Option<&SplitCoins> {
            if let Some(command::Command::SplitCoins(field)) = &self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn split_coins_opt_mut(&mut self) -> Option<&mut SplitCoins> {
            if let Some(command::Command::SplitCoins(field)) = &mut self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn split_coins_mut(&mut self) -> &mut SplitCoins {
            if self.split_coins_opt_mut().is_none() {
                self.command = Some(command::Command::SplitCoins(SplitCoins::default()));
            }
            self.split_coins_opt_mut().unwrap()
        }
        pub fn with_split_coins(mut self, field: SplitCoins) -> Self {
            self.command = Some(command::Command::SplitCoins(field.into()));
            self
        }
        pub fn merge_coins(&self) -> &MergeCoins {
            if let Some(command::Command::MergeCoins(field)) = &self.command {
                field as _
            } else {
                MergeCoins::default_instance() as _
            }
        }
        pub fn merge_coins_opt(&self) -> Option<&MergeCoins> {
            if let Some(command::Command::MergeCoins(field)) = &self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn merge_coins_opt_mut(&mut self) -> Option<&mut MergeCoins> {
            if let Some(command::Command::MergeCoins(field)) = &mut self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn merge_coins_mut(&mut self) -> &mut MergeCoins {
            if self.merge_coins_opt_mut().is_none() {
                self.command = Some(command::Command::MergeCoins(MergeCoins::default()));
            }
            self.merge_coins_opt_mut().unwrap()
        }
        pub fn with_merge_coins(mut self, field: MergeCoins) -> Self {
            self.command = Some(command::Command::MergeCoins(field.into()));
            self
        }
        pub fn publish(&self) -> &Publish {
            if let Some(command::Command::Publish(field)) = &self.command {
                field as _
            } else {
                Publish::default_instance() as _
            }
        }
        pub fn publish_opt(&self) -> Option<&Publish> {
            if let Some(command::Command::Publish(field)) = &self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn publish_opt_mut(&mut self) -> Option<&mut Publish> {
            if let Some(command::Command::Publish(field)) = &mut self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn publish_mut(&mut self) -> &mut Publish {
            if self.publish_opt_mut().is_none() {
                self.command = Some(command::Command::Publish(Publish::default()));
            }
            self.publish_opt_mut().unwrap()
        }
        pub fn with_publish(mut self, field: Publish) -> Self {
            self.command = Some(command::Command::Publish(field.into()));
            self
        }
        pub fn make_move_vector(&self) -> &MakeMoveVector {
            if let Some(command::Command::MakeMoveVector(field)) = &self.command {
                field as _
            } else {
                MakeMoveVector::default_instance() as _
            }
        }
        pub fn make_move_vector_opt(&self) -> Option<&MakeMoveVector> {
            if let Some(command::Command::MakeMoveVector(field)) = &self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn make_move_vector_opt_mut(&mut self) -> Option<&mut MakeMoveVector> {
            if let Some(command::Command::MakeMoveVector(field)) = &mut self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn make_move_vector_mut(&mut self) -> &mut MakeMoveVector {
            if self.make_move_vector_opt_mut().is_none() {
                self.command = Some(
                    command::Command::MakeMoveVector(MakeMoveVector::default()),
                );
            }
            self.make_move_vector_opt_mut().unwrap()
        }
        pub fn with_make_move_vector(mut self, field: MakeMoveVector) -> Self {
            self.command = Some(command::Command::MakeMoveVector(field.into()));
            self
        }
        pub fn upgrade(&self) -> &Upgrade {
            if let Some(command::Command::Upgrade(field)) = &self.command {
                field as _
            } else {
                Upgrade::default_instance() as _
            }
        }
        pub fn upgrade_opt(&self) -> Option<&Upgrade> {
            if let Some(command::Command::Upgrade(field)) = &self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn upgrade_opt_mut(&mut self) -> Option<&mut Upgrade> {
            if let Some(command::Command::Upgrade(field)) = &mut self.command {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn upgrade_mut(&mut self) -> &mut Upgrade {
            if self.upgrade_opt_mut().is_none() {
                self.command = Some(command::Command::Upgrade(Upgrade::default()));
            }
            self.upgrade_opt_mut().unwrap()
        }
        pub fn with_upgrade(mut self, field: Upgrade) -> Self {
            self.command = Some(command::Command::Upgrade(field.into()));
            self
        }
    }
    impl MoveCall {
        pub const fn const_default() -> Self {
            Self {
                package: None,
                module: None,
                function: None,
                type_arguments: Vec::new(),
                arguments: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: MoveCall = MoveCall::const_default();
            &DEFAULT
        }
        pub fn with_package(mut self, field: String) -> Self {
            self.package = Some(field.into());
            self
        }
        pub fn with_module(mut self, field: String) -> Self {
            self.module = Some(field.into());
            self
        }
        pub fn with_function(mut self, field: String) -> Self {
            self.function = Some(field.into());
            self
        }
        pub fn type_arguments(&self) -> &[String] {
            &self.type_arguments
        }
        pub fn with_type_arguments(mut self, field: Vec<String>) -> Self {
            self.type_arguments = field;
            self
        }
        pub fn arguments(&self) -> &[Argument] {
            &self.arguments
        }
        pub fn with_arguments(mut self, field: Vec<Argument>) -> Self {
            self.arguments = field;
            self
        }
    }
    impl TransferObjects {
        pub const fn const_default() -> Self {
            Self {
                objects: Vec::new(),
                address: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TransferObjects = TransferObjects::const_default();
            &DEFAULT
        }
        pub fn objects(&self) -> &[Argument] {
            &self.objects
        }
        pub fn with_objects(mut self, field: Vec<Argument>) -> Self {
            self.objects = field;
            self
        }
        pub fn address(&self) -> &Argument {
            self.address
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Argument::default_instance() as _)
        }
        pub fn address_opt(&self) -> Option<&Argument> {
            self.address.as_ref().map(|field| field as _)
        }
        pub fn address_opt_mut(&mut self) -> Option<&mut Argument> {
            self.address.as_mut().map(|field| field as _)
        }
        pub fn address_mut(&mut self) -> &mut Argument {
            self.address.get_or_insert_default()
        }
        pub fn with_address(mut self, field: Argument) -> Self {
            self.address = Some(field.into());
            self
        }
    }
    impl SplitCoins {
        pub const fn const_default() -> Self {
            Self {
                coin: None,
                amounts: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SplitCoins = SplitCoins::const_default();
            &DEFAULT
        }
        pub fn coin(&self) -> &Argument {
            self.coin
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Argument::default_instance() as _)
        }
        pub fn coin_opt(&self) -> Option<&Argument> {
            self.coin.as_ref().map(|field| field as _)
        }
        pub fn coin_opt_mut(&mut self) -> Option<&mut Argument> {
            self.coin.as_mut().map(|field| field as _)
        }
        pub fn coin_mut(&mut self) -> &mut Argument {
            self.coin.get_or_insert_default()
        }
        pub fn with_coin(mut self, field: Argument) -> Self {
            self.coin = Some(field.into());
            self
        }
        pub fn amounts(&self) -> &[Argument] {
            &self.amounts
        }
        pub fn with_amounts(mut self, field: Vec<Argument>) -> Self {
            self.amounts = field;
            self
        }
    }
    impl MergeCoins {
        pub const fn const_default() -> Self {
            Self {
                coin: None,
                coins_to_merge: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: MergeCoins = MergeCoins::const_default();
            &DEFAULT
        }
        pub fn coin(&self) -> &Argument {
            self.coin
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Argument::default_instance() as _)
        }
        pub fn coin_opt(&self) -> Option<&Argument> {
            self.coin.as_ref().map(|field| field as _)
        }
        pub fn coin_opt_mut(&mut self) -> Option<&mut Argument> {
            self.coin.as_mut().map(|field| field as _)
        }
        pub fn coin_mut(&mut self) -> &mut Argument {
            self.coin.get_or_insert_default()
        }
        pub fn with_coin(mut self, field: Argument) -> Self {
            self.coin = Some(field.into());
            self
        }
        pub fn coins_to_merge(&self) -> &[Argument] {
            &self.coins_to_merge
        }
        pub fn with_coins_to_merge(mut self, field: Vec<Argument>) -> Self {
            self.coins_to_merge = field;
            self
        }
    }
    impl Publish {
        pub const fn const_default() -> Self {
            Self {
                modules: Vec::new(),
                dependencies: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Publish = Publish::const_default();
            &DEFAULT
        }
        pub fn modules(&self) -> &[::prost::bytes::Bytes] {
            &self.modules
        }
        pub fn with_modules(mut self, field: Vec<::prost::bytes::Bytes>) -> Self {
            self.modules = field;
            self
        }
        pub fn dependencies(&self) -> &[String] {
            &self.dependencies
        }
        pub fn with_dependencies(mut self, field: Vec<String>) -> Self {
            self.dependencies = field;
            self
        }
    }
    impl MakeMoveVector {
        pub const fn const_default() -> Self {
            Self {
                element_type: None,
                elements: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: MakeMoveVector = MakeMoveVector::const_default();
            &DEFAULT
        }
        pub fn with_element_type(mut self, field: String) -> Self {
            self.element_type = Some(field.into());
            self
        }
        pub fn elements(&self) -> &[Argument] {
            &self.elements
        }
        pub fn with_elements(mut self, field: Vec<Argument>) -> Self {
            self.elements = field;
            self
        }
    }
    impl Upgrade {
        pub const fn const_default() -> Self {
            Self {
                modules: Vec::new(),
                dependencies: Vec::new(),
                package: None,
                ticket: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Upgrade = Upgrade::const_default();
            &DEFAULT
        }
        pub fn modules(&self) -> &[::prost::bytes::Bytes] {
            &self.modules
        }
        pub fn with_modules(mut self, field: Vec<::prost::bytes::Bytes>) -> Self {
            self.modules = field;
            self
        }
        pub fn dependencies(&self) -> &[String] {
            &self.dependencies
        }
        pub fn with_dependencies(mut self, field: Vec<String>) -> Self {
            self.dependencies = field;
            self
        }
        pub fn with_package(mut self, field: String) -> Self {
            self.package = Some(field.into());
            self
        }
        pub fn ticket(&self) -> &Argument {
            self.ticket
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Argument::default_instance() as _)
        }
        pub fn ticket_opt(&self) -> Option<&Argument> {
            self.ticket.as_ref().map(|field| field as _)
        }
        pub fn ticket_opt_mut(&mut self) -> Option<&mut Argument> {
            self.ticket.as_mut().map(|field| field as _)
        }
        pub fn ticket_mut(&mut self) -> &mut Argument {
            self.ticket.get_or_insert_default()
        }
        pub fn with_ticket(mut self, field: Argument) -> Self {
            self.ticket = Some(field.into());
            self
        }
    }
    impl RandomnessStateUpdate {
        pub const fn const_default() -> Self {
            Self {
                epoch: None,
                randomness_round: None,
                random_bytes: None,
                randomness_object_initial_shared_version: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: RandomnessStateUpdate = RandomnessStateUpdate::const_default();
            &DEFAULT
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
        pub fn with_randomness_round(mut self, field: u64) -> Self {
            self.randomness_round = Some(field.into());
            self
        }
        pub fn with_random_bytes(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.random_bytes = Some(field.into());
            self
        }
        pub fn with_randomness_object_initial_shared_version(
            mut self,
            field: u64,
        ) -> Self {
            self.randomness_object_initial_shared_version = Some(field.into());
            self
        }
    }
    impl ChangeEpoch {
        pub const fn const_default() -> Self {
            Self {
                epoch: None,
                protocol_version: None,
                storage_charge: None,
                computation_charge: None,
                storage_rebate: None,
                non_refundable_storage_fee: None,
                epoch_start_timestamp: None,
                system_packages: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ChangeEpoch = ChangeEpoch::const_default();
            &DEFAULT
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
        pub fn with_protocol_version(mut self, field: u64) -> Self {
            self.protocol_version = Some(field.into());
            self
        }
        pub fn with_storage_charge(mut self, field: u64) -> Self {
            self.storage_charge = Some(field.into());
            self
        }
        pub fn with_computation_charge(mut self, field: u64) -> Self {
            self.computation_charge = Some(field.into());
            self
        }
        pub fn with_storage_rebate(mut self, field: u64) -> Self {
            self.storage_rebate = Some(field.into());
            self
        }
        pub fn with_non_refundable_storage_fee(mut self, field: u64) -> Self {
            self.non_refundable_storage_fee = Some(field.into());
            self
        }
        pub fn system_packages(&self) -> &[SystemPackage] {
            &self.system_packages
        }
        pub fn with_system_packages(mut self, field: Vec<SystemPackage>) -> Self {
            self.system_packages = field;
            self
        }
    }
    impl SystemPackage {
        pub const fn const_default() -> Self {
            Self {
                version: None,
                modules: Vec::new(),
                dependencies: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SystemPackage = SystemPackage::const_default();
            &DEFAULT
        }
        pub fn with_version(mut self, field: u64) -> Self {
            self.version = Some(field.into());
            self
        }
        pub fn modules(&self) -> &[::prost::bytes::Bytes] {
            &self.modules
        }
        pub fn with_modules(mut self, field: Vec<::prost::bytes::Bytes>) -> Self {
            self.modules = field;
            self
        }
        pub fn dependencies(&self) -> &[String] {
            &self.dependencies
        }
        pub fn with_dependencies(mut self, field: Vec<String>) -> Self {
            self.dependencies = field;
            self
        }
    }
    impl GenesisTransaction {
        pub const fn const_default() -> Self {
            Self { objects: Vec::new() }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GenesisTransaction = GenesisTransaction::const_default();
            &DEFAULT
        }
        pub fn objects(&self) -> &[Object] {
            &self.objects
        }
        pub fn with_objects(mut self, field: Vec<Object>) -> Self {
            self.objects = field;
            self
        }
    }
    impl ConsensusCommitPrologue {
        pub const fn const_default() -> Self {
            Self {
                epoch: None,
                round: None,
                commit_timestamp: None,
                consensus_commit_digest: None,
                sub_dag_index: None,
                consensus_determined_version_assignments: None,
                additional_state_digest: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ConsensusCommitPrologue = ConsensusCommitPrologue::const_default();
            &DEFAULT
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
        pub fn with_round(mut self, field: u64) -> Self {
            self.round = Some(field.into());
            self
        }
        pub fn with_consensus_commit_digest(mut self, field: String) -> Self {
            self.consensus_commit_digest = Some(field.into());
            self
        }
        pub fn with_sub_dag_index(mut self, field: u64) -> Self {
            self.sub_dag_index = Some(field.into());
            self
        }
        pub fn consensus_determined_version_assignments(
            &self,
        ) -> &ConsensusDeterminedVersionAssignments {
            self.consensus_determined_version_assignments
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| {
                    ConsensusDeterminedVersionAssignments::default_instance() as _
                })
        }
        pub fn consensus_determined_version_assignments_opt(
            &self,
        ) -> Option<&ConsensusDeterminedVersionAssignments> {
            self.consensus_determined_version_assignments
                .as_ref()
                .map(|field| field as _)
        }
        pub fn consensus_determined_version_assignments_opt_mut(
            &mut self,
        ) -> Option<&mut ConsensusDeterminedVersionAssignments> {
            self.consensus_determined_version_assignments
                .as_mut()
                .map(|field| field as _)
        }
        pub fn consensus_determined_version_assignments_mut(
            &mut self,
        ) -> &mut ConsensusDeterminedVersionAssignments {
            self.consensus_determined_version_assignments.get_or_insert_default()
        }
        pub fn with_consensus_determined_version_assignments(
            mut self,
            field: ConsensusDeterminedVersionAssignments,
        ) -> Self {
            self.consensus_determined_version_assignments = Some(field.into());
            self
        }
        pub fn with_additional_state_digest(mut self, field: String) -> Self {
            self.additional_state_digest = Some(field.into());
            self
        }
    }
    impl VersionAssignment {
        pub const fn const_default() -> Self {
            Self {
                object_id: None,
                start_version: None,
                version: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: VersionAssignment = VersionAssignment::const_default();
            &DEFAULT
        }
        pub fn with_object_id(mut self, field: String) -> Self {
            self.object_id = Some(field.into());
            self
        }
        pub fn with_start_version(mut self, field: u64) -> Self {
            self.start_version = Some(field.into());
            self
        }
        pub fn with_version(mut self, field: u64) -> Self {
            self.version = Some(field.into());
            self
        }
    }
    impl CanceledTransaction {
        pub const fn const_default() -> Self {
            Self {
                digest: None,
                version_assignments: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: CanceledTransaction = CanceledTransaction::const_default();
            &DEFAULT
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn version_assignments(&self) -> &[VersionAssignment] {
            &self.version_assignments
        }
        pub fn with_version_assignments(
            mut self,
            field: Vec<VersionAssignment>,
        ) -> Self {
            self.version_assignments = field;
            self
        }
    }
    impl ConsensusDeterminedVersionAssignments {
        pub const fn const_default() -> Self {
            Self {
                version: None,
                canceled_transactions: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ConsensusDeterminedVersionAssignments = ConsensusDeterminedVersionAssignments::const_default();
            &DEFAULT
        }
        pub fn with_version(mut self, field: i32) -> Self {
            self.version = Some(field.into());
            self
        }
        pub fn canceled_transactions(&self) -> &[CanceledTransaction] {
            &self.canceled_transactions
        }
        pub fn with_canceled_transactions(
            mut self,
            field: Vec<CanceledTransaction>,
        ) -> Self {
            self.canceled_transactions = field;
            self
        }
    }
    impl AuthenticatorStateUpdate {
        pub const fn const_default() -> Self {
            Self {
                epoch: None,
                round: None,
                new_active_jwks: Vec::new(),
                authenticator_object_initial_shared_version: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: AuthenticatorStateUpdate = AuthenticatorStateUpdate::const_default();
            &DEFAULT
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
        pub fn with_round(mut self, field: u64) -> Self {
            self.round = Some(field.into());
            self
        }
        pub fn new_active_jwks(&self) -> &[ActiveJwk] {
            &self.new_active_jwks
        }
        pub fn with_new_active_jwks(mut self, field: Vec<ActiveJwk>) -> Self {
            self.new_active_jwks = field;
            self
        }
        pub fn with_authenticator_object_initial_shared_version(
            mut self,
            field: u64,
        ) -> Self {
            self.authenticator_object_initial_shared_version = Some(field.into());
            self
        }
    }
    impl ActiveJwk {
        pub const fn const_default() -> Self {
            Self {
                id: None,
                jwk: None,
                epoch: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ActiveJwk = ActiveJwk::const_default();
            &DEFAULT
        }
        pub fn id(&self) -> &JwkId {
            self.id
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| JwkId::default_instance() as _)
        }
        pub fn id_opt(&self) -> Option<&JwkId> {
            self.id.as_ref().map(|field| field as _)
        }
        pub fn id_opt_mut(&mut self) -> Option<&mut JwkId> {
            self.id.as_mut().map(|field| field as _)
        }
        pub fn id_mut(&mut self) -> &mut JwkId {
            self.id.get_or_insert_default()
        }
        pub fn with_id(mut self, field: JwkId) -> Self {
            self.id = Some(field.into());
            self
        }
        pub fn jwk(&self) -> &Jwk {
            self.jwk
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Jwk::default_instance() as _)
        }
        pub fn jwk_opt(&self) -> Option<&Jwk> {
            self.jwk.as_ref().map(|field| field as _)
        }
        pub fn jwk_opt_mut(&mut self) -> Option<&mut Jwk> {
            self.jwk.as_mut().map(|field| field as _)
        }
        pub fn jwk_mut(&mut self) -> &mut Jwk {
            self.jwk.get_or_insert_default()
        }
        pub fn with_jwk(mut self, field: Jwk) -> Self {
            self.jwk = Some(field.into());
            self
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
    }
    impl JwkId {
        pub const fn const_default() -> Self {
            Self { iss: None, kid: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: JwkId = JwkId::const_default();
            &DEFAULT
        }
        pub fn with_iss(mut self, field: String) -> Self {
            self.iss = Some(field.into());
            self
        }
        pub fn with_kid(mut self, field: String) -> Self {
            self.kid = Some(field.into());
            self
        }
    }
    impl Jwk {
        pub const fn const_default() -> Self {
            Self {
                kty: None,
                e: None,
                n: None,
                alg: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Jwk = Jwk::const_default();
            &DEFAULT
        }
        pub fn with_kty(mut self, field: String) -> Self {
            self.kty = Some(field.into());
            self
        }
        pub fn with_e(mut self, field: String) -> Self {
            self.e = Some(field.into());
            self
        }
        pub fn with_n(mut self, field: String) -> Self {
            self.n = Some(field.into());
            self
        }
        pub fn with_alg(mut self, field: String) -> Self {
            self.alg = Some(field.into());
            self
        }
    }
    impl EndOfEpochTransaction {
        pub const fn const_default() -> Self {
            Self { transactions: Vec::new() }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: EndOfEpochTransaction = EndOfEpochTransaction::const_default();
            &DEFAULT
        }
        pub fn transactions(&self) -> &[EndOfEpochTransactionKind] {
            &self.transactions
        }
        pub fn with_transactions(
            mut self,
            field: Vec<EndOfEpochTransactionKind>,
        ) -> Self {
            self.transactions = field;
            self
        }
    }
    impl EndOfEpochTransactionKind {
        pub const fn const_default() -> Self {
            Self { kind: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: EndOfEpochTransactionKind = EndOfEpochTransactionKind::const_default();
            &DEFAULT
        }
        pub fn change_epoch(&self) -> &ChangeEpoch {
            if let Some(end_of_epoch_transaction_kind::Kind::ChangeEpoch(field)) = &self
                .kind
            {
                field as _
            } else {
                ChangeEpoch::default_instance() as _
            }
        }
        pub fn change_epoch_opt(&self) -> Option<&ChangeEpoch> {
            if let Some(end_of_epoch_transaction_kind::Kind::ChangeEpoch(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn change_epoch_opt_mut(&mut self) -> Option<&mut ChangeEpoch> {
            if let Some(end_of_epoch_transaction_kind::Kind::ChangeEpoch(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn change_epoch_mut(&mut self) -> &mut ChangeEpoch {
            if self.change_epoch_opt_mut().is_none() {
                self.kind = Some(
                    end_of_epoch_transaction_kind::Kind::ChangeEpoch(
                        ChangeEpoch::default(),
                    ),
                );
            }
            self.change_epoch_opt_mut().unwrap()
        }
        pub fn with_change_epoch(mut self, field: ChangeEpoch) -> Self {
            self.kind = Some(
                end_of_epoch_transaction_kind::Kind::ChangeEpoch(field.into()),
            );
            self
        }
        pub fn authenticator_state_expire(&self) -> &AuthenticatorStateExpire {
            if let Some(
                end_of_epoch_transaction_kind::Kind::AuthenticatorStateExpire(field),
            ) = &self.kind
            {
                field as _
            } else {
                AuthenticatorStateExpire::default_instance() as _
            }
        }
        pub fn authenticator_state_expire_opt(
            &self,
        ) -> Option<&AuthenticatorStateExpire> {
            if let Some(
                end_of_epoch_transaction_kind::Kind::AuthenticatorStateExpire(field),
            ) = &self.kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn authenticator_state_expire_opt_mut(
            &mut self,
        ) -> Option<&mut AuthenticatorStateExpire> {
            if let Some(
                end_of_epoch_transaction_kind::Kind::AuthenticatorStateExpire(field),
            ) = &mut self.kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn authenticator_state_expire_mut(
            &mut self,
        ) -> &mut AuthenticatorStateExpire {
            if self.authenticator_state_expire_opt_mut().is_none() {
                self.kind = Some(
                    end_of_epoch_transaction_kind::Kind::AuthenticatorStateExpire(
                        AuthenticatorStateExpire::default(),
                    ),
                );
            }
            self.authenticator_state_expire_opt_mut().unwrap()
        }
        pub fn with_authenticator_state_expire(
            mut self,
            field: AuthenticatorStateExpire,
        ) -> Self {
            self.kind = Some(
                end_of_epoch_transaction_kind::Kind::AuthenticatorStateExpire(
                    field.into(),
                ),
            );
            self
        }
        pub fn execution_time_observations(&self) -> &ExecutionTimeObservations {
            if let Some(
                end_of_epoch_transaction_kind::Kind::ExecutionTimeObservations(field),
            ) = &self.kind
            {
                field as _
            } else {
                ExecutionTimeObservations::default_instance() as _
            }
        }
        pub fn execution_time_observations_opt(
            &self,
        ) -> Option<&ExecutionTimeObservations> {
            if let Some(
                end_of_epoch_transaction_kind::Kind::ExecutionTimeObservations(field),
            ) = &self.kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn execution_time_observations_opt_mut(
            &mut self,
        ) -> Option<&mut ExecutionTimeObservations> {
            if let Some(
                end_of_epoch_transaction_kind::Kind::ExecutionTimeObservations(field),
            ) = &mut self.kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn execution_time_observations_mut(
            &mut self,
        ) -> &mut ExecutionTimeObservations {
            if self.execution_time_observations_opt_mut().is_none() {
                self.kind = Some(
                    end_of_epoch_transaction_kind::Kind::ExecutionTimeObservations(
                        ExecutionTimeObservations::default(),
                    ),
                );
            }
            self.execution_time_observations_opt_mut().unwrap()
        }
        pub fn with_execution_time_observations(
            mut self,
            field: ExecutionTimeObservations,
        ) -> Self {
            self.kind = Some(
                end_of_epoch_transaction_kind::Kind::ExecutionTimeObservations(
                    field.into(),
                ),
            );
            self
        }
        pub fn bridge_state_create(&self) -> &str {
            if let Some(end_of_epoch_transaction_kind::Kind::BridgeStateCreate(field)) = &self
                .kind
            {
                field as _
            } else {
                ""
            }
        }
        pub fn bridge_state_create_opt(&self) -> Option<&str> {
            if let Some(end_of_epoch_transaction_kind::Kind::BridgeStateCreate(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn bridge_state_create_opt_mut(&mut self) -> Option<&mut String> {
            if let Some(end_of_epoch_transaction_kind::Kind::BridgeStateCreate(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn bridge_state_create_mut(&mut self) -> &mut String {
            if self.bridge_state_create_opt_mut().is_none() {
                self.kind = Some(
                    end_of_epoch_transaction_kind::Kind::BridgeStateCreate(
                        String::default(),
                    ),
                );
            }
            self.bridge_state_create_opt_mut().unwrap()
        }
        pub fn with_bridge_state_create(mut self, field: String) -> Self {
            self.kind = Some(
                end_of_epoch_transaction_kind::Kind::BridgeStateCreate(field.into()),
            );
            self
        }
        pub fn bridge_committee_init(&self) -> u64 {
            if let Some(
                end_of_epoch_transaction_kind::Kind::BridgeCommitteeInit(field),
            ) = &self.kind
            {
                *field
            } else {
                0u64
            }
        }
        pub fn bridge_committee_init_opt(&self) -> Option<u64> {
            if let Some(
                end_of_epoch_transaction_kind::Kind::BridgeCommitteeInit(field),
            ) = &self.kind
            {
                Some(*field)
            } else {
                None
            }
        }
        pub fn bridge_committee_init_opt_mut(&mut self) -> Option<&mut u64> {
            if let Some(
                end_of_epoch_transaction_kind::Kind::BridgeCommitteeInit(field),
            ) = &mut self.kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn bridge_committee_init_mut(&mut self) -> &mut u64 {
            if self.bridge_committee_init_opt_mut().is_none() {
                self.kind = Some(
                    end_of_epoch_transaction_kind::Kind::BridgeCommitteeInit(
                        u64::default(),
                    ),
                );
            }
            self.bridge_committee_init_opt_mut().unwrap()
        }
        pub fn with_bridge_committee_init(mut self, field: u64) -> Self {
            self.kind = Some(
                end_of_epoch_transaction_kind::Kind::BridgeCommitteeInit(field.into()),
            );
            self
        }
    }
    impl AuthenticatorStateExpire {
        pub const fn const_default() -> Self {
            Self {
                min_epoch: None,
                authenticator_object_initial_shared_version: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: AuthenticatorStateExpire = AuthenticatorStateExpire::const_default();
            &DEFAULT
        }
        pub fn with_min_epoch(mut self, field: u64) -> Self {
            self.min_epoch = Some(field.into());
            self
        }
        pub fn with_authenticator_object_initial_shared_version(
            mut self,
            field: u64,
        ) -> Self {
            self.authenticator_object_initial_shared_version = Some(field.into());
            self
        }
    }
    impl ExecutionTimeObservations {
        pub const fn const_default() -> Self {
            Self {
                version: None,
                observations: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ExecutionTimeObservations = ExecutionTimeObservations::const_default();
            &DEFAULT
        }
        pub fn with_version(mut self, field: i32) -> Self {
            self.version = Some(field.into());
            self
        }
        pub fn observations(&self) -> &[ExecutionTimeObservation] {
            &self.observations
        }
        pub fn with_observations(
            mut self,
            field: Vec<ExecutionTimeObservation>,
        ) -> Self {
            self.observations = field;
            self
        }
    }
    impl ExecutionTimeObservation {
        pub const fn const_default() -> Self {
            Self {
                kind: None,
                move_entry_point: None,
                validator_observations: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ExecutionTimeObservation = ExecutionTimeObservation::const_default();
            &DEFAULT
        }
        pub fn move_entry_point(&self) -> &MoveCall {
            self.move_entry_point
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| MoveCall::default_instance() as _)
        }
        pub fn move_entry_point_opt(&self) -> Option<&MoveCall> {
            self.move_entry_point.as_ref().map(|field| field as _)
        }
        pub fn move_entry_point_opt_mut(&mut self) -> Option<&mut MoveCall> {
            self.move_entry_point.as_mut().map(|field| field as _)
        }
        pub fn move_entry_point_mut(&mut self) -> &mut MoveCall {
            self.move_entry_point.get_or_insert_default()
        }
        pub fn with_move_entry_point(mut self, field: MoveCall) -> Self {
            self.move_entry_point = Some(field.into());
            self
        }
        pub fn validator_observations(&self) -> &[ValidatorExecutionTimeObservation] {
            &self.validator_observations
        }
        pub fn with_validator_observations(
            mut self,
            field: Vec<ValidatorExecutionTimeObservation>,
        ) -> Self {
            self.validator_observations = field;
            self
        }
    }
    impl ValidatorExecutionTimeObservation {
        pub const fn const_default() -> Self {
            Self {
                validator: None,
                duration: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ValidatorExecutionTimeObservation = ValidatorExecutionTimeObservation::const_default();
            &DEFAULT
        }
        pub fn with_validator(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.validator = Some(field.into());
            self
        }
    }
    impl ExecuteTransactionRequest {
        pub const fn const_default() -> Self {
            Self {
                transaction: None,
                signatures: Vec::new(),
                read_mask: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ExecuteTransactionRequest = ExecuteTransactionRequest::const_default();
            &DEFAULT
        }
        pub fn transaction(&self) -> &Transaction {
            self.transaction
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Transaction::default_instance() as _)
        }
        pub fn transaction_opt(&self) -> Option<&Transaction> {
            self.transaction.as_ref().map(|field| field as _)
        }
        pub fn transaction_opt_mut(&mut self) -> Option<&mut Transaction> {
            self.transaction.as_mut().map(|field| field as _)
        }
        pub fn transaction_mut(&mut self) -> &mut Transaction {
            self.transaction.get_or_insert_default()
        }
        pub fn with_transaction(mut self, field: Transaction) -> Self {
            self.transaction = Some(field.into());
            self
        }
        pub fn signatures(&self) -> &[UserSignature] {
            &self.signatures
        }
        pub fn with_signatures(mut self, field: Vec<UserSignature>) -> Self {
            self.signatures = field;
            self
        }
    }
    impl ExecuteTransactionResponse {
        pub const fn const_default() -> Self {
            Self {
                finality: None,
                transaction: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ExecuteTransactionResponse = ExecuteTransactionResponse::const_default();
            &DEFAULT
        }
        pub fn finality(&self) -> &TransactionFinality {
            self.finality
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| TransactionFinality::default_instance() as _)
        }
        pub fn finality_opt(&self) -> Option<&TransactionFinality> {
            self.finality.as_ref().map(|field| field as _)
        }
        pub fn finality_opt_mut(&mut self) -> Option<&mut TransactionFinality> {
            self.finality.as_mut().map(|field| field as _)
        }
        pub fn finality_mut(&mut self) -> &mut TransactionFinality {
            self.finality.get_or_insert_default()
        }
        pub fn with_finality(mut self, field: TransactionFinality) -> Self {
            self.finality = Some(field.into());
            self
        }
        pub fn transaction(&self) -> &ExecutedTransaction {
            self.transaction
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ExecutedTransaction::default_instance() as _)
        }
        pub fn transaction_opt(&self) -> Option<&ExecutedTransaction> {
            self.transaction.as_ref().map(|field| field as _)
        }
        pub fn transaction_opt_mut(&mut self) -> Option<&mut ExecutedTransaction> {
            self.transaction.as_mut().map(|field| field as _)
        }
        pub fn transaction_mut(&mut self) -> &mut ExecutedTransaction {
            self.transaction.get_or_insert_default()
        }
        pub fn with_transaction(mut self, field: ExecutedTransaction) -> Self {
            self.transaction = Some(field.into());
            self
        }
    }
    impl TransactionFinality {
        pub const fn const_default() -> Self {
            Self { finality: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TransactionFinality = TransactionFinality::const_default();
            &DEFAULT
        }
        pub fn certified(&self) -> &ValidatorAggregatedSignature {
            if let Some(transaction_finality::Finality::Certified(field)) = &self
                .finality
            {
                field as _
            } else {
                ValidatorAggregatedSignature::default_instance() as _
            }
        }
        pub fn certified_opt(&self) -> Option<&ValidatorAggregatedSignature> {
            if let Some(transaction_finality::Finality::Certified(field)) = &self
                .finality
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn certified_opt_mut(
            &mut self,
        ) -> Option<&mut ValidatorAggregatedSignature> {
            if let Some(transaction_finality::Finality::Certified(field)) = &mut self
                .finality
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn certified_mut(&mut self) -> &mut ValidatorAggregatedSignature {
            if self.certified_opt_mut().is_none() {
                self.finality = Some(
                    transaction_finality::Finality::Certified(
                        ValidatorAggregatedSignature::default(),
                    ),
                );
            }
            self.certified_opt_mut().unwrap()
        }
        pub fn with_certified(mut self, field: ValidatorAggregatedSignature) -> Self {
            self.finality = Some(
                transaction_finality::Finality::Certified(field.into()),
            );
            self
        }
        pub fn checkpointed(&self) -> u64 {
            if let Some(transaction_finality::Finality::Checkpointed(field)) = &self
                .finality
            {
                *field
            } else {
                0u64
            }
        }
        pub fn checkpointed_opt(&self) -> Option<u64> {
            if let Some(transaction_finality::Finality::Checkpointed(field)) = &self
                .finality
            {
                Some(*field)
            } else {
                None
            }
        }
        pub fn checkpointed_opt_mut(&mut self) -> Option<&mut u64> {
            if let Some(transaction_finality::Finality::Checkpointed(field)) = &mut self
                .finality
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn checkpointed_mut(&mut self) -> &mut u64 {
            if self.checkpointed_opt_mut().is_none() {
                self.finality = Some(
                    transaction_finality::Finality::Checkpointed(u64::default()),
                );
            }
            self.checkpointed_opt_mut().unwrap()
        }
        pub fn with_checkpointed(mut self, field: u64) -> Self {
            self.finality = Some(
                transaction_finality::Finality::Checkpointed(field.into()),
            );
            self
        }
    }
}
