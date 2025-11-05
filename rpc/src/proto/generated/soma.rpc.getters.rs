mod _getter_impls {
    #![allow(clippy::useless_conversion)]
    use super::*;
    impl BalanceChange {
        pub const fn const_default() -> Self {
            Self {
                address: None,
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
        pub fn with_amount(mut self, field: String) -> Self {
            self.amount = Some(field.into());
            self
        }
    }
    impl Commit {
        pub const fn const_default() -> Self {
            Self {
                index: None,
                digest: None,
                timestamp_ms: None,
                transactions: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Commit = Commit::const_default();
            &DEFAULT
        }
        pub fn with_index(mut self, field: u32) -> Self {
            self.index = Some(field.into());
            self
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn with_timestamp_ms(mut self, field: u64) -> Self {
            self.timestamp_ms = Some(field.into());
            self
        }
        pub fn transactions(&self) -> &[ExecutedTransaction] {
            &self.transactions
        }
        pub fn transactions_mut(&mut self) -> &mut Vec<ExecutedTransaction> {
            &mut self.transactions
        }
        pub fn with_transactions(mut self, field: Vec<ExecutedTransaction>) -> Self {
            self.transactions = field;
            self
        }
    }
    impl TransactionEffects {
        pub const fn const_default() -> Self {
            Self {
                status: None,
                epoch: None,
                fee: None,
                transaction_digest: None,
                dependencies: Vec::new(),
                lamport_version: None,
                changed_objects: Vec::new(),
                unchanged_shared_objects: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TransactionEffects = TransactionEffects::const_default();
            &DEFAULT
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
        pub fn fee(&self) -> &TransactionFee {
            self.fee
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| TransactionFee::default_instance() as _)
        }
        pub fn fee_opt(&self) -> Option<&TransactionFee> {
            self.fee.as_ref().map(|field| field as _)
        }
        pub fn fee_opt_mut(&mut self) -> Option<&mut TransactionFee> {
            self.fee.as_mut().map(|field| field as _)
        }
        pub fn fee_mut(&mut self) -> &mut TransactionFee {
            self.fee.get_or_insert_default()
        }
        pub fn with_fee(mut self, field: TransactionFee) -> Self {
            self.fee = Some(field.into());
            self
        }
        pub fn with_transaction_digest(mut self, field: String) -> Self {
            self.transaction_digest = Some(field.into());
            self
        }
        pub fn dependencies(&self) -> &[String] {
            &self.dependencies
        }
        pub fn dependencies_mut(&mut self) -> &mut Vec<String> {
            &mut self.dependencies
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
        pub fn changed_objects_mut(&mut self) -> &mut Vec<ChangedObject> {
            &mut self.changed_objects
        }
        pub fn with_changed_objects(mut self, field: Vec<ChangedObject>) -> Self {
            self.changed_objects = field;
            self
        }
        pub fn unchanged_shared_objects(&self) -> &[UnchangedSharedObject] {
            &self.unchanged_shared_objects
        }
        pub fn unchanged_shared_objects_mut(
            &mut self,
        ) -> &mut Vec<UnchangedSharedObject> {
            &mut self.unchanged_shared_objects
        }
        pub fn with_unchanged_shared_objects(
            mut self,
            field: Vec<UnchangedSharedObject>,
        ) -> Self {
            self.unchanged_shared_objects = field;
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
    impl UnchangedSharedObject {
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
            static DEFAULT: UnchangedSharedObject = UnchangedSharedObject::const_default();
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
    impl Epoch {
        pub const fn const_default() -> Self {
            Self {
                epoch: None,
                committee: None,
                system_state: None,
                start: None,
                end: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Epoch = Epoch::const_default();
            &DEFAULT
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
        pub fn committee(&self) -> &ValidatorCommittee {
            self.committee
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ValidatorCommittee::default_instance() as _)
        }
        pub fn committee_opt(&self) -> Option<&ValidatorCommittee> {
            self.committee.as_ref().map(|field| field as _)
        }
        pub fn committee_opt_mut(&mut self) -> Option<&mut ValidatorCommittee> {
            self.committee.as_mut().map(|field| field as _)
        }
        pub fn committee_mut(&mut self) -> &mut ValidatorCommittee {
            self.committee.get_or_insert_default()
        }
        pub fn with_committee(mut self, field: ValidatorCommittee) -> Self {
            self.committee = Some(field.into());
            self
        }
        pub fn system_state(&self) -> &SystemState {
            self.system_state
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| SystemState::default_instance() as _)
        }
        pub fn system_state_opt(&self) -> Option<&SystemState> {
            self.system_state.as_ref().map(|field| field as _)
        }
        pub fn system_state_opt_mut(&mut self) -> Option<&mut SystemState> {
            self.system_state.as_mut().map(|field| field as _)
        }
        pub fn system_state_mut(&mut self) -> &mut SystemState {
            self.system_state.get_or_insert_default()
        }
        pub fn with_system_state(mut self, field: SystemState) -> Self {
            self.system_state = Some(field.into());
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
                commit: None,
                timestamp: None,
                balance_changes: Vec::new(),
                input_objects: Vec::new(),
                output_objects: Vec::new(),
                shard: None,
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
        pub fn signatures_mut(&mut self) -> &mut Vec<UserSignature> {
            &mut self.signatures
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
        pub fn with_commit(mut self, field: u64) -> Self {
            self.commit = Some(field.into());
            self
        }
        pub fn balance_changes(&self) -> &[BalanceChange] {
            &self.balance_changes
        }
        pub fn balance_changes_mut(&mut self) -> &mut Vec<BalanceChange> {
            &mut self.balance_changes
        }
        pub fn with_balance_changes(mut self, field: Vec<BalanceChange>) -> Self {
            self.balance_changes = field;
            self
        }
        pub fn input_objects(&self) -> &[Object] {
            &self.input_objects
        }
        pub fn input_objects_mut(&mut self) -> &mut Vec<Object> {
            &mut self.input_objects
        }
        pub fn with_input_objects(mut self, field: Vec<Object>) -> Self {
            self.input_objects = field;
            self
        }
        pub fn output_objects(&self) -> &[Object] {
            &self.output_objects
        }
        pub fn output_objects_mut(&mut self) -> &mut Vec<Object> {
            &mut self.output_objects
        }
        pub fn with_output_objects(mut self, field: Vec<Object>) -> Self {
            self.output_objects = field;
            self
        }
        pub fn shard(&self) -> &Shard {
            self.shard
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Shard::default_instance() as _)
        }
        pub fn shard_opt(&self) -> Option<&Shard> {
            self.shard.as_ref().map(|field| field as _)
        }
        pub fn shard_opt_mut(&mut self) -> Option<&mut Shard> {
            self.shard.as_mut().map(|field| field as _)
        }
        pub fn shard_mut(&mut self) -> &mut Shard {
            self.shard.get_or_insert_default()
        }
        pub fn with_shard(mut self, field: Shard) -> Self {
            self.shard = Some(field.into());
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
        pub fn other_error(&self) -> &str {
            if let Some(execution_error::ErrorDetails::OtherError(field)) = &self
                .error_details
            {
                field as _
            } else {
                ""
            }
        }
        pub fn other_error_opt(&self) -> Option<&str> {
            if let Some(execution_error::ErrorDetails::OtherError(field)) = &self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn other_error_opt_mut(&mut self) -> Option<&mut String> {
            if let Some(execution_error::ErrorDetails::OtherError(field)) = &mut self
                .error_details
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn other_error_mut(&mut self) -> &mut String {
            if self.other_error_opt_mut().is_none() {
                self.error_details = Some(
                    execution_error::ErrorDetails::OtherError(String::default()),
                );
            }
            self.other_error_opt_mut().unwrap()
        }
        pub fn with_other_error(mut self, field: String) -> Self {
            self.error_details = Some(
                execution_error::ErrorDetails::OtherError(field.into()),
            );
            self
        }
    }
    impl GetObjectRequest {
        pub const fn const_default() -> Self {
            Self {
                object_id: None,
                version: None,
                read_mask: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetObjectRequest = GetObjectRequest::const_default();
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
    }
    impl GetObjectResponse {
        pub const fn const_default() -> Self {
            Self { object: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetObjectResponse = GetObjectResponse::const_default();
            &DEFAULT
        }
        pub fn object(&self) -> &Object {
            self.object
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Object::default_instance() as _)
        }
        pub fn object_opt(&self) -> Option<&Object> {
            self.object.as_ref().map(|field| field as _)
        }
        pub fn object_opt_mut(&mut self) -> Option<&mut Object> {
            self.object.as_mut().map(|field| field as _)
        }
        pub fn object_mut(&mut self) -> &mut Object {
            self.object.get_or_insert_default()
        }
        pub fn with_object(mut self, field: Object) -> Self {
            self.object = Some(field.into());
            self
        }
    }
    impl BatchGetObjectsRequest {
        pub const fn const_default() -> Self {
            Self {
                requests: Vec::new(),
                read_mask: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: BatchGetObjectsRequest = BatchGetObjectsRequest::const_default();
            &DEFAULT
        }
        pub fn requests(&self) -> &[GetObjectRequest] {
            &self.requests
        }
        pub fn requests_mut(&mut self) -> &mut Vec<GetObjectRequest> {
            &mut self.requests
        }
        pub fn with_requests(mut self, field: Vec<GetObjectRequest>) -> Self {
            self.requests = field;
            self
        }
    }
    impl BatchGetObjectsResponse {
        pub const fn const_default() -> Self {
            Self { objects: Vec::new() }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: BatchGetObjectsResponse = BatchGetObjectsResponse::const_default();
            &DEFAULT
        }
        pub fn objects(&self) -> &[GetObjectResult] {
            &self.objects
        }
        pub fn objects_mut(&mut self) -> &mut Vec<GetObjectResult> {
            &mut self.objects
        }
        pub fn with_objects(mut self, field: Vec<GetObjectResult>) -> Self {
            self.objects = field;
            self
        }
    }
    impl GetObjectResult {
        pub const fn const_default() -> Self {
            Self { result: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetObjectResult = GetObjectResult::const_default();
            &DEFAULT
        }
        pub fn object(&self) -> &Object {
            if let Some(get_object_result::Result::Object(field)) = &self.result {
                field as _
            } else {
                Object::default_instance() as _
            }
        }
        pub fn object_opt(&self) -> Option<&Object> {
            if let Some(get_object_result::Result::Object(field)) = &self.result {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn object_opt_mut(&mut self) -> Option<&mut Object> {
            if let Some(get_object_result::Result::Object(field)) = &mut self.result {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn object_mut(&mut self) -> &mut Object {
            if self.object_opt_mut().is_none() {
                self.result = Some(get_object_result::Result::Object(Object::default()));
            }
            self.object_opt_mut().unwrap()
        }
        pub fn with_object(mut self, field: Object) -> Self {
            self.result = Some(get_object_result::Result::Object(field.into()));
            self
        }
    }
    impl GetTransactionRequest {
        pub const fn const_default() -> Self {
            Self {
                digest: None,
                read_mask: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetTransactionRequest = GetTransactionRequest::const_default();
            &DEFAULT
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
            self
        }
    }
    impl GetTransactionResponse {
        pub const fn const_default() -> Self {
            Self { transaction: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetTransactionResponse = GetTransactionResponse::const_default();
            &DEFAULT
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
    impl BatchGetTransactionsRequest {
        pub const fn const_default() -> Self {
            Self {
                digests: Vec::new(),
                read_mask: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: BatchGetTransactionsRequest = BatchGetTransactionsRequest::const_default();
            &DEFAULT
        }
        pub fn digests(&self) -> &[String] {
            &self.digests
        }
        pub fn digests_mut(&mut self) -> &mut Vec<String> {
            &mut self.digests
        }
        pub fn with_digests(mut self, field: Vec<String>) -> Self {
            self.digests = field;
            self
        }
    }
    impl BatchGetTransactionsResponse {
        pub const fn const_default() -> Self {
            Self { transactions: Vec::new() }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: BatchGetTransactionsResponse = BatchGetTransactionsResponse::const_default();
            &DEFAULT
        }
        pub fn transactions(&self) -> &[GetTransactionResult] {
            &self.transactions
        }
        pub fn transactions_mut(&mut self) -> &mut Vec<GetTransactionResult> {
            &mut self.transactions
        }
        pub fn with_transactions(mut self, field: Vec<GetTransactionResult>) -> Self {
            self.transactions = field;
            self
        }
    }
    impl GetTransactionResult {
        pub const fn const_default() -> Self {
            Self { result: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetTransactionResult = GetTransactionResult::const_default();
            &DEFAULT
        }
        pub fn transaction(&self) -> &ExecutedTransaction {
            if let Some(get_transaction_result::Result::Transaction(field)) = &self
                .result
            {
                field as _
            } else {
                ExecutedTransaction::default_instance() as _
            }
        }
        pub fn transaction_opt(&self) -> Option<&ExecutedTransaction> {
            if let Some(get_transaction_result::Result::Transaction(field)) = &self
                .result
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn transaction_opt_mut(&mut self) -> Option<&mut ExecutedTransaction> {
            if let Some(get_transaction_result::Result::Transaction(field)) = &mut self
                .result
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn transaction_mut(&mut self) -> &mut ExecutedTransaction {
            if self.transaction_opt_mut().is_none() {
                self.result = Some(
                    get_transaction_result::Result::Transaction(
                        ExecutedTransaction::default(),
                    ),
                );
            }
            self.transaction_opt_mut().unwrap()
        }
        pub fn with_transaction(mut self, field: ExecutedTransaction) -> Self {
            self.result = Some(
                get_transaction_result::Result::Transaction(field.into()),
            );
            self
        }
    }
    impl GetCommitRequest {
        pub const fn const_default() -> Self {
            Self {
                read_mask: None,
                commit_id: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetCommitRequest = GetCommitRequest::const_default();
            &DEFAULT
        }
        pub fn index(&self) -> u32 {
            if let Some(get_commit_request::CommitId::Index(field)) = &self.commit_id {
                *field
            } else {
                0u32
            }
        }
        pub fn index_opt(&self) -> Option<u32> {
            if let Some(get_commit_request::CommitId::Index(field)) = &self.commit_id {
                Some(*field)
            } else {
                None
            }
        }
        pub fn index_opt_mut(&mut self) -> Option<&mut u32> {
            if let Some(get_commit_request::CommitId::Index(field)) = &mut self.commit_id
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn index_mut(&mut self) -> &mut u32 {
            if self.index_opt_mut().is_none() {
                self.commit_id = Some(
                    get_commit_request::CommitId::Index(u32::default()),
                );
            }
            self.index_opt_mut().unwrap()
        }
        pub fn with_index(mut self, field: u32) -> Self {
            self.commit_id = Some(get_commit_request::CommitId::Index(field.into()));
            self
        }
        pub fn digest(&self) -> &str {
            if let Some(get_commit_request::CommitId::Digest(field)) = &self.commit_id {
                field as _
            } else {
                ""
            }
        }
        pub fn digest_opt(&self) -> Option<&str> {
            if let Some(get_commit_request::CommitId::Digest(field)) = &self.commit_id {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn digest_opt_mut(&mut self) -> Option<&mut String> {
            if let Some(get_commit_request::CommitId::Digest(field)) = &mut self
                .commit_id
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn digest_mut(&mut self) -> &mut String {
            if self.digest_opt_mut().is_none() {
                self.commit_id = Some(
                    get_commit_request::CommitId::Digest(String::default()),
                );
            }
            self.digest_opt_mut().unwrap()
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.commit_id = Some(get_commit_request::CommitId::Digest(field.into()));
            self
        }
    }
    impl GetCommitResponse {
        pub const fn const_default() -> Self {
            Self { commit: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetCommitResponse = GetCommitResponse::const_default();
            &DEFAULT
        }
        pub fn commit(&self) -> &Commit {
            self.commit
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Commit::default_instance() as _)
        }
        pub fn commit_opt(&self) -> Option<&Commit> {
            self.commit.as_ref().map(|field| field as _)
        }
        pub fn commit_opt_mut(&mut self) -> Option<&mut Commit> {
            self.commit.as_mut().map(|field| field as _)
        }
        pub fn commit_mut(&mut self) -> &mut Commit {
            self.commit.get_or_insert_default()
        }
        pub fn with_commit(mut self, field: Commit) -> Self {
            self.commit = Some(field.into());
            self
        }
    }
    impl GetEpochRequest {
        pub const fn const_default() -> Self {
            Self {
                epoch: None,
                read_mask: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetEpochRequest = GetEpochRequest::const_default();
            &DEFAULT
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
    }
    impl GetEpochResponse {
        pub const fn const_default() -> Self {
            Self { epoch: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetEpochResponse = GetEpochResponse::const_default();
            &DEFAULT
        }
        pub fn epoch(&self) -> &Epoch {
            self.epoch
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Epoch::default_instance() as _)
        }
        pub fn epoch_opt(&self) -> Option<&Epoch> {
            self.epoch.as_ref().map(|field| field as _)
        }
        pub fn epoch_opt_mut(&mut self) -> Option<&mut Epoch> {
            self.epoch.as_mut().map(|field| field as _)
        }
        pub fn epoch_mut(&mut self) -> &mut Epoch {
            self.epoch.get_or_insert_default()
        }
        pub fn with_epoch(mut self, field: Epoch) -> Self {
            self.epoch = Some(field.into());
            self
        }
    }
    impl GetBalanceRequest {
        pub const fn const_default() -> Self {
            Self { owner: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetBalanceRequest = GetBalanceRequest::const_default();
            &DEFAULT
        }
        pub fn with_owner(mut self, field: String) -> Self {
            self.owner = Some(field.into());
            self
        }
    }
    impl GetBalanceResponse {
        pub const fn const_default() -> Self {
            Self { balance: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: GetBalanceResponse = GetBalanceResponse::const_default();
            &DEFAULT
        }
        pub fn with_balance(mut self, field: u64) -> Self {
            self.balance = Some(field.into());
            self
        }
    }
    impl SimulateTransactionRequest {
        pub const fn const_default() -> Self {
            Self {
                transaction: None,
                read_mask: None,
                checks: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SimulateTransactionRequest = SimulateTransactionRequest::const_default();
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
    }
    impl SimulateTransactionResponse {
        pub const fn const_default() -> Self {
            Self { transaction: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SimulateTransactionResponse = SimulateTransactionResponse::const_default();
            &DEFAULT
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
    impl ListOwnedObjectsRequest {
        pub const fn const_default() -> Self {
            Self {
                owner: None,
                page_size: None,
                page_token: None,
                read_mask: None,
                object_type: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ListOwnedObjectsRequest = ListOwnedObjectsRequest::const_default();
            &DEFAULT
        }
        pub fn with_owner(mut self, field: String) -> Self {
            self.owner = Some(field.into());
            self
        }
        pub fn with_page_size(mut self, field: u32) -> Self {
            self.page_size = Some(field.into());
            self
        }
        pub fn with_page_token(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.page_token = Some(field.into());
            self
        }
        pub fn with_object_type(mut self, field: String) -> Self {
            self.object_type = Some(field.into());
            self
        }
    }
    impl ListOwnedObjectsResponse {
        pub const fn const_default() -> Self {
            Self {
                objects: Vec::new(),
                next_page_token: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ListOwnedObjectsResponse = ListOwnedObjectsResponse::const_default();
            &DEFAULT
        }
        pub fn objects(&self) -> &[Object] {
            &self.objects
        }
        pub fn objects_mut(&mut self) -> &mut Vec<Object> {
            &mut self.objects
        }
        pub fn with_objects(mut self, field: Vec<Object>) -> Self {
            self.objects = field;
            self
        }
        pub fn with_next_page_token(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.next_page_token = Some(field.into());
            self
        }
    }
    impl Object {
        pub const fn const_default() -> Self {
            Self {
                object_id: None,
                version: None,
                digest: None,
                owner: None,
                object_type: None,
                contents: None,
                previous_transaction: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Object = Object::const_default();
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
        pub fn with_contents(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.contents = Some(field.into());
            self
        }
        pub fn with_previous_transaction(mut self, field: String) -> Self {
            self.previous_transaction = Some(field.into());
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
    impl Shard {
        pub const fn const_default() -> Self {
            Self {
                quorum_threshold: None,
                encoders: Vec::new(),
                seed: None,
                epoch: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Shard = Shard::const_default();
            &DEFAULT
        }
        pub fn with_quorum_threshold(mut self, field: u32) -> Self {
            self.quorum_threshold = Some(field.into());
            self
        }
        pub fn encoders(&self) -> &[String] {
            &self.encoders
        }
        pub fn encoders_mut(&mut self) -> &mut Vec<String> {
            &mut self.encoders
        }
        pub fn with_encoders(mut self, field: Vec<String>) -> Self {
            self.encoders = field;
            self
        }
        pub fn with_seed(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.seed = Some(field.into());
            self
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
    }
    impl UserSignature {
        pub const fn const_default() -> Self {
            Self {
                scheme: None,
                signature: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: UserSignature = UserSignature::const_default();
            &DEFAULT
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
        pub fn members_mut(&mut self) -> &mut Vec<ValidatorCommitteeMember> {
            &mut self.members
        }
        pub fn with_members(mut self, field: Vec<ValidatorCommitteeMember>) -> Self {
            self.members = field;
            self
        }
    }
    impl ValidatorCommitteeMember {
        pub const fn const_default() -> Self {
            Self {
                authority_key: None,
                weight: None,
                network_metadata: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ValidatorCommitteeMember = ValidatorCommitteeMember::const_default();
            &DEFAULT
        }
        pub fn with_authority_key(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.authority_key = Some(field.into());
            self
        }
        pub fn with_weight(mut self, field: u64) -> Self {
            self.weight = Some(field.into());
            self
        }
        pub fn network_metadata(&self) -> &ValidatorNetworkMetadata {
            self.network_metadata
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ValidatorNetworkMetadata::default_instance() as _)
        }
        pub fn network_metadata_opt(&self) -> Option<&ValidatorNetworkMetadata> {
            self.network_metadata.as_ref().map(|field| field as _)
        }
        pub fn network_metadata_opt_mut(
            &mut self,
        ) -> Option<&mut ValidatorNetworkMetadata> {
            self.network_metadata.as_mut().map(|field| field as _)
        }
        pub fn network_metadata_mut(&mut self) -> &mut ValidatorNetworkMetadata {
            self.network_metadata.get_or_insert_default()
        }
        pub fn with_network_metadata(mut self, field: ValidatorNetworkMetadata) -> Self {
            self.network_metadata = Some(field.into());
            self
        }
    }
    impl ValidatorNetworkMetadata {
        pub const fn const_default() -> Self {
            Self {
                consensus_address: None,
                hostname: None,
                protocol_key: None,
                network_key: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ValidatorNetworkMetadata = ValidatorNetworkMetadata::const_default();
            &DEFAULT
        }
        pub fn with_consensus_address(mut self, field: String) -> Self {
            self.consensus_address = Some(field.into());
            self
        }
        pub fn with_hostname(mut self, field: String) -> Self {
            self.hostname = Some(field.into());
            self
        }
        pub fn with_protocol_key(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.protocol_key = Some(field.into());
            self
        }
        pub fn with_network_key(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.network_key = Some(field.into());
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
        pub fn bitmap_mut(&mut self) -> &mut Vec<u32> {
            &mut self.bitmap
        }
        pub fn with_bitmap(mut self, field: Vec<u32>) -> Self {
            self.bitmap = field;
            self
        }
    }
    impl SubscribeCommitsRequest {
        pub const fn const_default() -> Self {
            Self { read_mask: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SubscribeCommitsRequest = SubscribeCommitsRequest::const_default();
            &DEFAULT
        }
    }
    impl SubscribeCommitsResponse {
        pub const fn const_default() -> Self {
            Self { cursor: None, commit: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SubscribeCommitsResponse = SubscribeCommitsResponse::const_default();
            &DEFAULT
        }
        pub fn with_cursor(mut self, field: u64) -> Self {
            self.cursor = Some(field.into());
            self
        }
        pub fn commit(&self) -> &Commit {
            self.commit
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| Commit::default_instance() as _)
        }
        pub fn commit_opt(&self) -> Option<&Commit> {
            self.commit.as_ref().map(|field| field as _)
        }
        pub fn commit_opt_mut(&mut self) -> Option<&mut Commit> {
            self.commit.as_mut().map(|field| field as _)
        }
        pub fn commit_mut(&mut self) -> &mut Commit {
            self.commit.get_or_insert_default()
        }
        pub fn with_commit(mut self, field: Commit) -> Self {
            self.commit = Some(field.into());
            self
        }
    }
    impl SystemState {
        pub const fn const_default() -> Self {
            Self {
                epoch: None,
                epoch_start_timestamp_ms: None,
                parameters: None,
                validators: None,
                encoders: None,
                validator_report_records: std::collections::BTreeMap::new(),
                encoder_report_records: std::collections::BTreeMap::new(),
                stake_subsidy: None,
                shard_results: std::collections::BTreeMap::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SystemState = SystemState::const_default();
            &DEFAULT
        }
        pub fn with_epoch(mut self, field: u64) -> Self {
            self.epoch = Some(field.into());
            self
        }
        pub fn with_epoch_start_timestamp_ms(mut self, field: u64) -> Self {
            self.epoch_start_timestamp_ms = Some(field.into());
            self
        }
        pub fn parameters(&self) -> &SystemParameters {
            self.parameters
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| SystemParameters::default_instance() as _)
        }
        pub fn parameters_opt(&self) -> Option<&SystemParameters> {
            self.parameters.as_ref().map(|field| field as _)
        }
        pub fn parameters_opt_mut(&mut self) -> Option<&mut SystemParameters> {
            self.parameters.as_mut().map(|field| field as _)
        }
        pub fn parameters_mut(&mut self) -> &mut SystemParameters {
            self.parameters.get_or_insert_default()
        }
        pub fn with_parameters(mut self, field: SystemParameters) -> Self {
            self.parameters = Some(field.into());
            self
        }
        pub fn validators(&self) -> &ValidatorSet {
            self.validators
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ValidatorSet::default_instance() as _)
        }
        pub fn validators_opt(&self) -> Option<&ValidatorSet> {
            self.validators.as_ref().map(|field| field as _)
        }
        pub fn validators_opt_mut(&mut self) -> Option<&mut ValidatorSet> {
            self.validators.as_mut().map(|field| field as _)
        }
        pub fn validators_mut(&mut self) -> &mut ValidatorSet {
            self.validators.get_or_insert_default()
        }
        pub fn with_validators(mut self, field: ValidatorSet) -> Self {
            self.validators = Some(field.into());
            self
        }
        pub fn encoders(&self) -> &EncoderSet {
            self.encoders
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| EncoderSet::default_instance() as _)
        }
        pub fn encoders_opt(&self) -> Option<&EncoderSet> {
            self.encoders.as_ref().map(|field| field as _)
        }
        pub fn encoders_opt_mut(&mut self) -> Option<&mut EncoderSet> {
            self.encoders.as_mut().map(|field| field as _)
        }
        pub fn encoders_mut(&mut self) -> &mut EncoderSet {
            self.encoders.get_or_insert_default()
        }
        pub fn with_encoders(mut self, field: EncoderSet) -> Self {
            self.encoders = Some(field.into());
            self
        }
        pub fn stake_subsidy(&self) -> &StakeSubsidy {
            self.stake_subsidy
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| StakeSubsidy::default_instance() as _)
        }
        pub fn stake_subsidy_opt(&self) -> Option<&StakeSubsidy> {
            self.stake_subsidy.as_ref().map(|field| field as _)
        }
        pub fn stake_subsidy_opt_mut(&mut self) -> Option<&mut StakeSubsidy> {
            self.stake_subsidy.as_mut().map(|field| field as _)
        }
        pub fn stake_subsidy_mut(&mut self) -> &mut StakeSubsidy {
            self.stake_subsidy.get_or_insert_default()
        }
        pub fn with_stake_subsidy(mut self, field: StakeSubsidy) -> Self {
            self.stake_subsidy = Some(field.into());
            self
        }
    }
    impl ReporterSet {
        pub const fn const_default() -> Self {
            Self { reporters: Vec::new() }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ReporterSet = ReporterSet::const_default();
            &DEFAULT
        }
        pub fn reporters(&self) -> &[String] {
            &self.reporters
        }
        pub fn reporters_mut(&mut self) -> &mut Vec<String> {
            &mut self.reporters
        }
        pub fn with_reporters(mut self, field: Vec<String>) -> Self {
            self.reporters = field;
            self
        }
    }
    impl SystemParameters {
        pub const fn const_default() -> Self {
            Self {
                epoch_duration_ms: None,
                vdf_iterations: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SystemParameters = SystemParameters::const_default();
            &DEFAULT
        }
        pub fn with_epoch_duration_ms(mut self, field: u64) -> Self {
            self.epoch_duration_ms = Some(field.into());
            self
        }
        pub fn with_vdf_iterations(mut self, field: u64) -> Self {
            self.vdf_iterations = Some(field.into());
            self
        }
    }
    impl StakeSubsidy {
        pub const fn const_default() -> Self {
            Self {
                balance: None,
                distribution_counter: None,
                current_distribution_amount: None,
                period_length: None,
                decrease_rate: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: StakeSubsidy = StakeSubsidy::const_default();
            &DEFAULT
        }
        pub fn with_balance(mut self, field: u64) -> Self {
            self.balance = Some(field.into());
            self
        }
        pub fn with_distribution_counter(mut self, field: u64) -> Self {
            self.distribution_counter = Some(field.into());
            self
        }
        pub fn with_current_distribution_amount(mut self, field: u64) -> Self {
            self.current_distribution_amount = Some(field.into());
            self
        }
        pub fn with_period_length(mut self, field: u64) -> Self {
            self.period_length = Some(field.into());
            self
        }
        pub fn with_decrease_rate(mut self, field: u32) -> Self {
            self.decrease_rate = Some(field.into());
            self
        }
    }
    impl ValidatorSet {
        pub const fn const_default() -> Self {
            Self {
                total_stake: None,
                consensus_validators: Vec::new(),
                networking_validators: Vec::new(),
                pending_validators: Vec::new(),
                pending_removals: Vec::new(),
                staking_pool_mappings: std::collections::BTreeMap::new(),
                inactive_validators: std::collections::BTreeMap::new(),
                at_risk_validators: std::collections::BTreeMap::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ValidatorSet = ValidatorSet::const_default();
            &DEFAULT
        }
        pub fn with_total_stake(mut self, field: u64) -> Self {
            self.total_stake = Some(field.into());
            self
        }
        pub fn consensus_validators(&self) -> &[Validator] {
            &self.consensus_validators
        }
        pub fn consensus_validators_mut(&mut self) -> &mut Vec<Validator> {
            &mut self.consensus_validators
        }
        pub fn with_consensus_validators(mut self, field: Vec<Validator>) -> Self {
            self.consensus_validators = field;
            self
        }
        pub fn networking_validators(&self) -> &[Validator] {
            &self.networking_validators
        }
        pub fn networking_validators_mut(&mut self) -> &mut Vec<Validator> {
            &mut self.networking_validators
        }
        pub fn with_networking_validators(mut self, field: Vec<Validator>) -> Self {
            self.networking_validators = field;
            self
        }
        pub fn pending_validators(&self) -> &[Validator] {
            &self.pending_validators
        }
        pub fn pending_validators_mut(&mut self) -> &mut Vec<Validator> {
            &mut self.pending_validators
        }
        pub fn with_pending_validators(mut self, field: Vec<Validator>) -> Self {
            self.pending_validators = field;
            self
        }
        pub fn pending_removals(&self) -> &[PendingRemoval] {
            &self.pending_removals
        }
        pub fn pending_removals_mut(&mut self) -> &mut Vec<PendingRemoval> {
            &mut self.pending_removals
        }
        pub fn with_pending_removals(mut self, field: Vec<PendingRemoval>) -> Self {
            self.pending_removals = field;
            self
        }
    }
    impl PendingRemoval {
        pub const fn const_default() -> Self {
            Self {
                is_consensus: None,
                index: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: PendingRemoval = PendingRemoval::const_default();
            &DEFAULT
        }
        pub fn with_is_consensus(mut self, field: bool) -> Self {
            self.is_consensus = Some(field.into());
            self
        }
        pub fn with_index(mut self, field: u32) -> Self {
            self.index = Some(field.into());
            self
        }
    }
    impl Validator {
        pub const fn const_default() -> Self {
            Self {
                soma_address: None,
                protocol_pubkey: None,
                network_pubkey: None,
                worker_pubkey: None,
                net_address: None,
                p2p_address: None,
                primary_address: None,
                encoder_validator_address: None,
                voting_power: None,
                commission_rate: None,
                next_epoch_stake: None,
                next_epoch_commission_rate: None,
                staking_pool: None,
                next_epoch_protocol_pubkey: None,
                next_epoch_network_pubkey: None,
                next_epoch_worker_pubkey: None,
                next_epoch_net_address: None,
                next_epoch_p2p_address: None,
                next_epoch_primary_address: None,
                next_epoch_encoder_validator_address: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Validator = Validator::const_default();
            &DEFAULT
        }
        pub fn with_soma_address(mut self, field: String) -> Self {
            self.soma_address = Some(field.into());
            self
        }
        pub fn with_protocol_pubkey(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.protocol_pubkey = Some(field.into());
            self
        }
        pub fn with_network_pubkey(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.network_pubkey = Some(field.into());
            self
        }
        pub fn with_worker_pubkey(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.worker_pubkey = Some(field.into());
            self
        }
        pub fn with_net_address(mut self, field: String) -> Self {
            self.net_address = Some(field.into());
            self
        }
        pub fn with_p2p_address(mut self, field: String) -> Self {
            self.p2p_address = Some(field.into());
            self
        }
        pub fn with_primary_address(mut self, field: String) -> Self {
            self.primary_address = Some(field.into());
            self
        }
        pub fn with_encoder_validator_address(mut self, field: String) -> Self {
            self.encoder_validator_address = Some(field.into());
            self
        }
        pub fn with_voting_power(mut self, field: u64) -> Self {
            self.voting_power = Some(field.into());
            self
        }
        pub fn with_commission_rate(mut self, field: u64) -> Self {
            self.commission_rate = Some(field.into());
            self
        }
        pub fn with_next_epoch_stake(mut self, field: u64) -> Self {
            self.next_epoch_stake = Some(field.into());
            self
        }
        pub fn with_next_epoch_commission_rate(mut self, field: u64) -> Self {
            self.next_epoch_commission_rate = Some(field.into());
            self
        }
        pub fn staking_pool(&self) -> &StakingPool {
            self.staking_pool
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| StakingPool::default_instance() as _)
        }
        pub fn staking_pool_opt(&self) -> Option<&StakingPool> {
            self.staking_pool.as_ref().map(|field| field as _)
        }
        pub fn staking_pool_opt_mut(&mut self) -> Option<&mut StakingPool> {
            self.staking_pool.as_mut().map(|field| field as _)
        }
        pub fn staking_pool_mut(&mut self) -> &mut StakingPool {
            self.staking_pool.get_or_insert_default()
        }
        pub fn with_staking_pool(mut self, field: StakingPool) -> Self {
            self.staking_pool = Some(field.into());
            self
        }
        pub fn with_next_epoch_protocol_pubkey(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_protocol_pubkey = Some(field.into());
            self
        }
        pub fn with_next_epoch_network_pubkey(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_network_pubkey = Some(field.into());
            self
        }
        pub fn with_next_epoch_worker_pubkey(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_worker_pubkey = Some(field.into());
            self
        }
        pub fn with_next_epoch_net_address(mut self, field: String) -> Self {
            self.next_epoch_net_address = Some(field.into());
            self
        }
        pub fn with_next_epoch_p2p_address(mut self, field: String) -> Self {
            self.next_epoch_p2p_address = Some(field.into());
            self
        }
        pub fn with_next_epoch_primary_address(mut self, field: String) -> Self {
            self.next_epoch_primary_address = Some(field.into());
            self
        }
        pub fn with_next_epoch_encoder_validator_address(
            mut self,
            field: String,
        ) -> Self {
            self.next_epoch_encoder_validator_address = Some(field.into());
            self
        }
    }
    impl StakingPool {
        pub const fn const_default() -> Self {
            Self {
                id: None,
                activation_epoch: None,
                deactivation_epoch: None,
                soma_balance: None,
                rewards_pool: None,
                pool_token_balance: None,
                exchange_rates: std::collections::BTreeMap::new(),
                pending_stake: None,
                pending_total_soma_withdraw: None,
                pending_pool_token_withdraw: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: StakingPool = StakingPool::const_default();
            &DEFAULT
        }
        pub fn with_id(mut self, field: String) -> Self {
            self.id = Some(field.into());
            self
        }
        pub fn with_activation_epoch(mut self, field: u64) -> Self {
            self.activation_epoch = Some(field.into());
            self
        }
        pub fn with_deactivation_epoch(mut self, field: u64) -> Self {
            self.deactivation_epoch = Some(field.into());
            self
        }
        pub fn with_soma_balance(mut self, field: u64) -> Self {
            self.soma_balance = Some(field.into());
            self
        }
        pub fn with_rewards_pool(mut self, field: u64) -> Self {
            self.rewards_pool = Some(field.into());
            self
        }
        pub fn with_pool_token_balance(mut self, field: u64) -> Self {
            self.pool_token_balance = Some(field.into());
            self
        }
        pub fn with_pending_stake(mut self, field: u64) -> Self {
            self.pending_stake = Some(field.into());
            self
        }
        pub fn with_pending_total_soma_withdraw(mut self, field: u64) -> Self {
            self.pending_total_soma_withdraw = Some(field.into());
            self
        }
        pub fn with_pending_pool_token_withdraw(mut self, field: u64) -> Self {
            self.pending_pool_token_withdraw = Some(field.into());
            self
        }
    }
    impl PoolTokenExchangeRate {
        pub const fn const_default() -> Self {
            Self {
                soma_amount: None,
                pool_token_amount: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: PoolTokenExchangeRate = PoolTokenExchangeRate::const_default();
            &DEFAULT
        }
        pub fn with_soma_amount(mut self, field: u64) -> Self {
            self.soma_amount = Some(field.into());
            self
        }
        pub fn with_pool_token_amount(mut self, field: u64) -> Self {
            self.pool_token_amount = Some(field.into());
            self
        }
    }
    impl EncoderSet {
        pub const fn const_default() -> Self {
            Self {
                total_stake: None,
                active_encoders: Vec::new(),
                pending_active_encoders: Vec::new(),
                pending_removals: Vec::new(),
                staking_pool_mappings: std::collections::BTreeMap::new(),
                inactive_encoders: std::collections::BTreeMap::new(),
                at_risk_encoders: std::collections::BTreeMap::new(),
                reference_byte_price: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: EncoderSet = EncoderSet::const_default();
            &DEFAULT
        }
        pub fn with_total_stake(mut self, field: u64) -> Self {
            self.total_stake = Some(field.into());
            self
        }
        pub fn active_encoders(&self) -> &[Encoder] {
            &self.active_encoders
        }
        pub fn active_encoders_mut(&mut self) -> &mut Vec<Encoder> {
            &mut self.active_encoders
        }
        pub fn with_active_encoders(mut self, field: Vec<Encoder>) -> Self {
            self.active_encoders = field;
            self
        }
        pub fn pending_active_encoders(&self) -> &[Encoder] {
            &self.pending_active_encoders
        }
        pub fn pending_active_encoders_mut(&mut self) -> &mut Vec<Encoder> {
            &mut self.pending_active_encoders
        }
        pub fn with_pending_active_encoders(mut self, field: Vec<Encoder>) -> Self {
            self.pending_active_encoders = field;
            self
        }
        pub fn pending_removals(&self) -> &[u32] {
            &self.pending_removals
        }
        pub fn pending_removals_mut(&mut self) -> &mut Vec<u32> {
            &mut self.pending_removals
        }
        pub fn with_pending_removals(mut self, field: Vec<u32>) -> Self {
            self.pending_removals = field;
            self
        }
        pub fn with_reference_byte_price(mut self, field: u64) -> Self {
            self.reference_byte_price = Some(field.into());
            self
        }
    }
    impl Encoder {
        pub const fn const_default() -> Self {
            Self {
                soma_address: None,
                encoder_pubkey: None,
                network_pubkey: None,
                internal_network_address: None,
                external_network_address: None,
                object_server_address: None,
                voting_power: None,
                commission_rate: None,
                next_epoch_stake: None,
                next_epoch_commission_rate: None,
                byte_price: None,
                next_epoch_byte_price: None,
                staking_pool: None,
                next_epoch_network_pubkey: None,
                next_epoch_internal_network_address: None,
                next_epoch_external_network_address: None,
                next_epoch_object_server_address: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Encoder = Encoder::const_default();
            &DEFAULT
        }
        pub fn with_soma_address(mut self, field: String) -> Self {
            self.soma_address = Some(field.into());
            self
        }
        pub fn with_encoder_pubkey(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.encoder_pubkey = Some(field.into());
            self
        }
        pub fn with_network_pubkey(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.network_pubkey = Some(field.into());
            self
        }
        pub fn with_internal_network_address(mut self, field: String) -> Self {
            self.internal_network_address = Some(field.into());
            self
        }
        pub fn with_external_network_address(mut self, field: String) -> Self {
            self.external_network_address = Some(field.into());
            self
        }
        pub fn with_object_server_address(mut self, field: String) -> Self {
            self.object_server_address = Some(field.into());
            self
        }
        pub fn with_voting_power(mut self, field: u64) -> Self {
            self.voting_power = Some(field.into());
            self
        }
        pub fn with_commission_rate(mut self, field: u64) -> Self {
            self.commission_rate = Some(field.into());
            self
        }
        pub fn with_next_epoch_stake(mut self, field: u64) -> Self {
            self.next_epoch_stake = Some(field.into());
            self
        }
        pub fn with_next_epoch_commission_rate(mut self, field: u64) -> Self {
            self.next_epoch_commission_rate = Some(field.into());
            self
        }
        pub fn with_byte_price(mut self, field: u64) -> Self {
            self.byte_price = Some(field.into());
            self
        }
        pub fn with_next_epoch_byte_price(mut self, field: u64) -> Self {
            self.next_epoch_byte_price = Some(field.into());
            self
        }
        pub fn staking_pool(&self) -> &StakingPool {
            self.staking_pool
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| StakingPool::default_instance() as _)
        }
        pub fn staking_pool_opt(&self) -> Option<&StakingPool> {
            self.staking_pool.as_ref().map(|field| field as _)
        }
        pub fn staking_pool_opt_mut(&mut self) -> Option<&mut StakingPool> {
            self.staking_pool.as_mut().map(|field| field as _)
        }
        pub fn staking_pool_mut(&mut self) -> &mut StakingPool {
            self.staking_pool.get_or_insert_default()
        }
        pub fn with_staking_pool(mut self, field: StakingPool) -> Self {
            self.staking_pool = Some(field.into());
            self
        }
        pub fn with_next_epoch_network_pubkey(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_network_pubkey = Some(field.into());
            self
        }
        pub fn with_next_epoch_internal_network_address(
            mut self,
            field: String,
        ) -> Self {
            self.next_epoch_internal_network_address = Some(field.into());
            self
        }
        pub fn with_next_epoch_external_network_address(
            mut self,
            field: String,
        ) -> Self {
            self.next_epoch_external_network_address = Some(field.into());
            self
        }
        pub fn with_next_epoch_object_server_address(mut self, field: String) -> Self {
            self.next_epoch_object_server_address = Some(field.into());
            self
        }
    }
    impl ShardResult {
        pub const fn const_default() -> Self {
            Self {
                digest: None,
                data_size_bytes: None,
                amount: None,
                report: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ShardResult = ShardResult::const_default();
            &DEFAULT
        }
        pub fn with_digest(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn with_data_size_bytes(mut self, field: u64) -> Self {
            self.data_size_bytes = Some(field.into());
            self
        }
        pub fn with_amount(mut self, field: u64) -> Self {
            self.amount = Some(field.into());
            self
        }
        pub fn with_report(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.report = Some(field.into());
            self
        }
    }
    impl Transaction {
        pub const fn const_default() -> Self {
            Self {
                digest: None,
                kind: None,
                sender: None,
                gas_payment: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: Transaction = Transaction::const_default();
            &DEFAULT
        }
        pub fn with_digest(mut self, field: String) -> Self {
            self.digest = Some(field.into());
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
        pub fn gas_payment(&self) -> &[ObjectReference] {
            &self.gas_payment
        }
        pub fn gas_payment_mut(&mut self) -> &mut Vec<ObjectReference> {
            &mut self.gas_payment
        }
        pub fn with_gas_payment(mut self, field: Vec<ObjectReference>) -> Self {
            self.gas_payment = field;
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
        pub fn consensus_commit_prologue(&self) -> &ConsensusCommitPrologue {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologue(field)) = &self
                .kind
            {
                field as _
            } else {
                ConsensusCommitPrologue::default_instance() as _
            }
        }
        pub fn consensus_commit_prologue_opt(&self) -> Option<&ConsensusCommitPrologue> {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologue(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn consensus_commit_prologue_opt_mut(
            &mut self,
        ) -> Option<&mut ConsensusCommitPrologue> {
            if let Some(transaction_kind::Kind::ConsensusCommitPrologue(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn consensus_commit_prologue_mut(&mut self) -> &mut ConsensusCommitPrologue {
            if self.consensus_commit_prologue_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ConsensusCommitPrologue(
                        ConsensusCommitPrologue::default(),
                    ),
                );
            }
            self.consensus_commit_prologue_opt_mut().unwrap()
        }
        pub fn with_consensus_commit_prologue(
            mut self,
            field: ConsensusCommitPrologue,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::ConsensusCommitPrologue(field.into()),
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
        pub fn add_validator(&self) -> &AddValidator {
            if let Some(transaction_kind::Kind::AddValidator(field)) = &self.kind {
                field as _
            } else {
                AddValidator::default_instance() as _
            }
        }
        pub fn add_validator_opt(&self) -> Option<&AddValidator> {
            if let Some(transaction_kind::Kind::AddValidator(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn add_validator_opt_mut(&mut self) -> Option<&mut AddValidator> {
            if let Some(transaction_kind::Kind::AddValidator(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn add_validator_mut(&mut self) -> &mut AddValidator {
            if self.add_validator_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::AddValidator(AddValidator::default()),
                );
            }
            self.add_validator_opt_mut().unwrap()
        }
        pub fn with_add_validator(mut self, field: AddValidator) -> Self {
            self.kind = Some(transaction_kind::Kind::AddValidator(field.into()));
            self
        }
        pub fn remove_validator(&self) -> &RemoveValidator {
            if let Some(transaction_kind::Kind::RemoveValidator(field)) = &self.kind {
                field as _
            } else {
                RemoveValidator::default_instance() as _
            }
        }
        pub fn remove_validator_opt(&self) -> Option<&RemoveValidator> {
            if let Some(transaction_kind::Kind::RemoveValidator(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn remove_validator_opt_mut(&mut self) -> Option<&mut RemoveValidator> {
            if let Some(transaction_kind::Kind::RemoveValidator(field)) = &mut self.kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn remove_validator_mut(&mut self) -> &mut RemoveValidator {
            if self.remove_validator_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::RemoveValidator(RemoveValidator::default()),
                );
            }
            self.remove_validator_opt_mut().unwrap()
        }
        pub fn with_remove_validator(mut self, field: RemoveValidator) -> Self {
            self.kind = Some(transaction_kind::Kind::RemoveValidator(field.into()));
            self
        }
        pub fn report_validator(&self) -> &ReportValidator {
            if let Some(transaction_kind::Kind::ReportValidator(field)) = &self.kind {
                field as _
            } else {
                ReportValidator::default_instance() as _
            }
        }
        pub fn report_validator_opt(&self) -> Option<&ReportValidator> {
            if let Some(transaction_kind::Kind::ReportValidator(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn report_validator_opt_mut(&mut self) -> Option<&mut ReportValidator> {
            if let Some(transaction_kind::Kind::ReportValidator(field)) = &mut self.kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn report_validator_mut(&mut self) -> &mut ReportValidator {
            if self.report_validator_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ReportValidator(ReportValidator::default()),
                );
            }
            self.report_validator_opt_mut().unwrap()
        }
        pub fn with_report_validator(mut self, field: ReportValidator) -> Self {
            self.kind = Some(transaction_kind::Kind::ReportValidator(field.into()));
            self
        }
        pub fn undo_report_validator(&self) -> &UndoReportValidator {
            if let Some(transaction_kind::Kind::UndoReportValidator(field)) = &self.kind
            {
                field as _
            } else {
                UndoReportValidator::default_instance() as _
            }
        }
        pub fn undo_report_validator_opt(&self) -> Option<&UndoReportValidator> {
            if let Some(transaction_kind::Kind::UndoReportValidator(field)) = &self.kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn undo_report_validator_opt_mut(
            &mut self,
        ) -> Option<&mut UndoReportValidator> {
            if let Some(transaction_kind::Kind::UndoReportValidator(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn undo_report_validator_mut(&mut self) -> &mut UndoReportValidator {
            if self.undo_report_validator_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::UndoReportValidator(
                        UndoReportValidator::default(),
                    ),
                );
            }
            self.undo_report_validator_opt_mut().unwrap()
        }
        pub fn with_undo_report_validator(mut self, field: UndoReportValidator) -> Self {
            self.kind = Some(transaction_kind::Kind::UndoReportValidator(field.into()));
            self
        }
        pub fn update_validator_metadata(&self) -> &UpdateValidatorMetadata {
            if let Some(transaction_kind::Kind::UpdateValidatorMetadata(field)) = &self
                .kind
            {
                field as _
            } else {
                UpdateValidatorMetadata::default_instance() as _
            }
        }
        pub fn update_validator_metadata_opt(&self) -> Option<&UpdateValidatorMetadata> {
            if let Some(transaction_kind::Kind::UpdateValidatorMetadata(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn update_validator_metadata_opt_mut(
            &mut self,
        ) -> Option<&mut UpdateValidatorMetadata> {
            if let Some(transaction_kind::Kind::UpdateValidatorMetadata(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn update_validator_metadata_mut(&mut self) -> &mut UpdateValidatorMetadata {
            if self.update_validator_metadata_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::UpdateValidatorMetadata(
                        UpdateValidatorMetadata::default(),
                    ),
                );
            }
            self.update_validator_metadata_opt_mut().unwrap()
        }
        pub fn with_update_validator_metadata(
            mut self,
            field: UpdateValidatorMetadata,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::UpdateValidatorMetadata(field.into()),
            );
            self
        }
        pub fn set_commission_rate(&self) -> &SetCommissionRate {
            if let Some(transaction_kind::Kind::SetCommissionRate(field)) = &self.kind {
                field as _
            } else {
                SetCommissionRate::default_instance() as _
            }
        }
        pub fn set_commission_rate_opt(&self) -> Option<&SetCommissionRate> {
            if let Some(transaction_kind::Kind::SetCommissionRate(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn set_commission_rate_opt_mut(&mut self) -> Option<&mut SetCommissionRate> {
            if let Some(transaction_kind::Kind::SetCommissionRate(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn set_commission_rate_mut(&mut self) -> &mut SetCommissionRate {
            if self.set_commission_rate_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::SetCommissionRate(
                        SetCommissionRate::default(),
                    ),
                );
            }
            self.set_commission_rate_opt_mut().unwrap()
        }
        pub fn with_set_commission_rate(mut self, field: SetCommissionRate) -> Self {
            self.kind = Some(transaction_kind::Kind::SetCommissionRate(field.into()));
            self
        }
        pub fn add_encoder(&self) -> &AddEncoder {
            if let Some(transaction_kind::Kind::AddEncoder(field)) = &self.kind {
                field as _
            } else {
                AddEncoder::default_instance() as _
            }
        }
        pub fn add_encoder_opt(&self) -> Option<&AddEncoder> {
            if let Some(transaction_kind::Kind::AddEncoder(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn add_encoder_opt_mut(&mut self) -> Option<&mut AddEncoder> {
            if let Some(transaction_kind::Kind::AddEncoder(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn add_encoder_mut(&mut self) -> &mut AddEncoder {
            if self.add_encoder_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::AddEncoder(AddEncoder::default()),
                );
            }
            self.add_encoder_opt_mut().unwrap()
        }
        pub fn with_add_encoder(mut self, field: AddEncoder) -> Self {
            self.kind = Some(transaction_kind::Kind::AddEncoder(field.into()));
            self
        }
        pub fn remove_encoder(&self) -> &RemoveEncoder {
            if let Some(transaction_kind::Kind::RemoveEncoder(field)) = &self.kind {
                field as _
            } else {
                RemoveEncoder::default_instance() as _
            }
        }
        pub fn remove_encoder_opt(&self) -> Option<&RemoveEncoder> {
            if let Some(transaction_kind::Kind::RemoveEncoder(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn remove_encoder_opt_mut(&mut self) -> Option<&mut RemoveEncoder> {
            if let Some(transaction_kind::Kind::RemoveEncoder(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn remove_encoder_mut(&mut self) -> &mut RemoveEncoder {
            if self.remove_encoder_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::RemoveEncoder(RemoveEncoder::default()),
                );
            }
            self.remove_encoder_opt_mut().unwrap()
        }
        pub fn with_remove_encoder(mut self, field: RemoveEncoder) -> Self {
            self.kind = Some(transaction_kind::Kind::RemoveEncoder(field.into()));
            self
        }
        pub fn report_encoder(&self) -> &ReportEncoder {
            if let Some(transaction_kind::Kind::ReportEncoder(field)) = &self.kind {
                field as _
            } else {
                ReportEncoder::default_instance() as _
            }
        }
        pub fn report_encoder_opt(&self) -> Option<&ReportEncoder> {
            if let Some(transaction_kind::Kind::ReportEncoder(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn report_encoder_opt_mut(&mut self) -> Option<&mut ReportEncoder> {
            if let Some(transaction_kind::Kind::ReportEncoder(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn report_encoder_mut(&mut self) -> &mut ReportEncoder {
            if self.report_encoder_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ReportEncoder(ReportEncoder::default()),
                );
            }
            self.report_encoder_opt_mut().unwrap()
        }
        pub fn with_report_encoder(mut self, field: ReportEncoder) -> Self {
            self.kind = Some(transaction_kind::Kind::ReportEncoder(field.into()));
            self
        }
        pub fn undo_report_encoder(&self) -> &UndoReportEncoder {
            if let Some(transaction_kind::Kind::UndoReportEncoder(field)) = &self.kind {
                field as _
            } else {
                UndoReportEncoder::default_instance() as _
            }
        }
        pub fn undo_report_encoder_opt(&self) -> Option<&UndoReportEncoder> {
            if let Some(transaction_kind::Kind::UndoReportEncoder(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn undo_report_encoder_opt_mut(&mut self) -> Option<&mut UndoReportEncoder> {
            if let Some(transaction_kind::Kind::UndoReportEncoder(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn undo_report_encoder_mut(&mut self) -> &mut UndoReportEncoder {
            if self.undo_report_encoder_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::UndoReportEncoder(
                        UndoReportEncoder::default(),
                    ),
                );
            }
            self.undo_report_encoder_opt_mut().unwrap()
        }
        pub fn with_undo_report_encoder(mut self, field: UndoReportEncoder) -> Self {
            self.kind = Some(transaction_kind::Kind::UndoReportEncoder(field.into()));
            self
        }
        pub fn update_encoder_metadata(&self) -> &UpdateEncoderMetadata {
            if let Some(transaction_kind::Kind::UpdateEncoderMetadata(field)) = &self
                .kind
            {
                field as _
            } else {
                UpdateEncoderMetadata::default_instance() as _
            }
        }
        pub fn update_encoder_metadata_opt(&self) -> Option<&UpdateEncoderMetadata> {
            if let Some(transaction_kind::Kind::UpdateEncoderMetadata(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn update_encoder_metadata_opt_mut(
            &mut self,
        ) -> Option<&mut UpdateEncoderMetadata> {
            if let Some(transaction_kind::Kind::UpdateEncoderMetadata(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn update_encoder_metadata_mut(&mut self) -> &mut UpdateEncoderMetadata {
            if self.update_encoder_metadata_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::UpdateEncoderMetadata(
                        UpdateEncoderMetadata::default(),
                    ),
                );
            }
            self.update_encoder_metadata_opt_mut().unwrap()
        }
        pub fn with_update_encoder_metadata(
            mut self,
            field: UpdateEncoderMetadata,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::UpdateEncoderMetadata(field.into()),
            );
            self
        }
        pub fn set_encoder_commission_rate(&self) -> &SetEncoderCommissionRate {
            if let Some(transaction_kind::Kind::SetEncoderCommissionRate(field)) = &self
                .kind
            {
                field as _
            } else {
                SetEncoderCommissionRate::default_instance() as _
            }
        }
        pub fn set_encoder_commission_rate_opt(
            &self,
        ) -> Option<&SetEncoderCommissionRate> {
            if let Some(transaction_kind::Kind::SetEncoderCommissionRate(field)) = &self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn set_encoder_commission_rate_opt_mut(
            &mut self,
        ) -> Option<&mut SetEncoderCommissionRate> {
            if let Some(transaction_kind::Kind::SetEncoderCommissionRate(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn set_encoder_commission_rate_mut(
            &mut self,
        ) -> &mut SetEncoderCommissionRate {
            if self.set_encoder_commission_rate_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::SetEncoderCommissionRate(
                        SetEncoderCommissionRate::default(),
                    ),
                );
            }
            self.set_encoder_commission_rate_opt_mut().unwrap()
        }
        pub fn with_set_encoder_commission_rate(
            mut self,
            field: SetEncoderCommissionRate,
        ) -> Self {
            self.kind = Some(
                transaction_kind::Kind::SetEncoderCommissionRate(field.into()),
            );
            self
        }
        pub fn set_encoder_byte_price(&self) -> &SetEncoderBytePrice {
            if let Some(transaction_kind::Kind::SetEncoderBytePrice(field)) = &self.kind
            {
                field as _
            } else {
                SetEncoderBytePrice::default_instance() as _
            }
        }
        pub fn set_encoder_byte_price_opt(&self) -> Option<&SetEncoderBytePrice> {
            if let Some(transaction_kind::Kind::SetEncoderBytePrice(field)) = &self.kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn set_encoder_byte_price_opt_mut(
            &mut self,
        ) -> Option<&mut SetEncoderBytePrice> {
            if let Some(transaction_kind::Kind::SetEncoderBytePrice(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn set_encoder_byte_price_mut(&mut self) -> &mut SetEncoderBytePrice {
            if self.set_encoder_byte_price_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::SetEncoderBytePrice(
                        SetEncoderBytePrice::default(),
                    ),
                );
            }
            self.set_encoder_byte_price_opt_mut().unwrap()
        }
        pub fn with_set_encoder_byte_price(
            mut self,
            field: SetEncoderBytePrice,
        ) -> Self {
            self.kind = Some(transaction_kind::Kind::SetEncoderBytePrice(field.into()));
            self
        }
        pub fn transfer_coin(&self) -> &TransferCoin {
            if let Some(transaction_kind::Kind::TransferCoin(field)) = &self.kind {
                field as _
            } else {
                TransferCoin::default_instance() as _
            }
        }
        pub fn transfer_coin_opt(&self) -> Option<&TransferCoin> {
            if let Some(transaction_kind::Kind::TransferCoin(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn transfer_coin_opt_mut(&mut self) -> Option<&mut TransferCoin> {
            if let Some(transaction_kind::Kind::TransferCoin(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn transfer_coin_mut(&mut self) -> &mut TransferCoin {
            if self.transfer_coin_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::TransferCoin(TransferCoin::default()),
                );
            }
            self.transfer_coin_opt_mut().unwrap()
        }
        pub fn with_transfer_coin(mut self, field: TransferCoin) -> Self {
            self.kind = Some(transaction_kind::Kind::TransferCoin(field.into()));
            self
        }
        pub fn pay_coins(&self) -> &PayCoins {
            if let Some(transaction_kind::Kind::PayCoins(field)) = &self.kind {
                field as _
            } else {
                PayCoins::default_instance() as _
            }
        }
        pub fn pay_coins_opt(&self) -> Option<&PayCoins> {
            if let Some(transaction_kind::Kind::PayCoins(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn pay_coins_opt_mut(&mut self) -> Option<&mut PayCoins> {
            if let Some(transaction_kind::Kind::PayCoins(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn pay_coins_mut(&mut self) -> &mut PayCoins {
            if self.pay_coins_opt_mut().is_none() {
                self.kind = Some(transaction_kind::Kind::PayCoins(PayCoins::default()));
            }
            self.pay_coins_opt_mut().unwrap()
        }
        pub fn with_pay_coins(mut self, field: PayCoins) -> Self {
            self.kind = Some(transaction_kind::Kind::PayCoins(field.into()));
            self
        }
        pub fn transfer_objects(&self) -> &TransferObjects {
            if let Some(transaction_kind::Kind::TransferObjects(field)) = &self.kind {
                field as _
            } else {
                TransferObjects::default_instance() as _
            }
        }
        pub fn transfer_objects_opt(&self) -> Option<&TransferObjects> {
            if let Some(transaction_kind::Kind::TransferObjects(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn transfer_objects_opt_mut(&mut self) -> Option<&mut TransferObjects> {
            if let Some(transaction_kind::Kind::TransferObjects(field)) = &mut self.kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn transfer_objects_mut(&mut self) -> &mut TransferObjects {
            if self.transfer_objects_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::TransferObjects(TransferObjects::default()),
                );
            }
            self.transfer_objects_opt_mut().unwrap()
        }
        pub fn with_transfer_objects(mut self, field: TransferObjects) -> Self {
            self.kind = Some(transaction_kind::Kind::TransferObjects(field.into()));
            self
        }
        pub fn add_stake(&self) -> &AddStake {
            if let Some(transaction_kind::Kind::AddStake(field)) = &self.kind {
                field as _
            } else {
                AddStake::default_instance() as _
            }
        }
        pub fn add_stake_opt(&self) -> Option<&AddStake> {
            if let Some(transaction_kind::Kind::AddStake(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn add_stake_opt_mut(&mut self) -> Option<&mut AddStake> {
            if let Some(transaction_kind::Kind::AddStake(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn add_stake_mut(&mut self) -> &mut AddStake {
            if self.add_stake_opt_mut().is_none() {
                self.kind = Some(transaction_kind::Kind::AddStake(AddStake::default()));
            }
            self.add_stake_opt_mut().unwrap()
        }
        pub fn with_add_stake(mut self, field: AddStake) -> Self {
            self.kind = Some(transaction_kind::Kind::AddStake(field.into()));
            self
        }
        pub fn add_stake_to_encoder(&self) -> &AddStakeToEncoder {
            if let Some(transaction_kind::Kind::AddStakeToEncoder(field)) = &self.kind {
                field as _
            } else {
                AddStakeToEncoder::default_instance() as _
            }
        }
        pub fn add_stake_to_encoder_opt(&self) -> Option<&AddStakeToEncoder> {
            if let Some(transaction_kind::Kind::AddStakeToEncoder(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn add_stake_to_encoder_opt_mut(
            &mut self,
        ) -> Option<&mut AddStakeToEncoder> {
            if let Some(transaction_kind::Kind::AddStakeToEncoder(field)) = &mut self
                .kind
            {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn add_stake_to_encoder_mut(&mut self) -> &mut AddStakeToEncoder {
            if self.add_stake_to_encoder_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::AddStakeToEncoder(
                        AddStakeToEncoder::default(),
                    ),
                );
            }
            self.add_stake_to_encoder_opt_mut().unwrap()
        }
        pub fn with_add_stake_to_encoder(mut self, field: AddStakeToEncoder) -> Self {
            self.kind = Some(transaction_kind::Kind::AddStakeToEncoder(field.into()));
            self
        }
        pub fn withdraw_stake(&self) -> &WithdrawStake {
            if let Some(transaction_kind::Kind::WithdrawStake(field)) = &self.kind {
                field as _
            } else {
                WithdrawStake::default_instance() as _
            }
        }
        pub fn withdraw_stake_opt(&self) -> Option<&WithdrawStake> {
            if let Some(transaction_kind::Kind::WithdrawStake(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn withdraw_stake_opt_mut(&mut self) -> Option<&mut WithdrawStake> {
            if let Some(transaction_kind::Kind::WithdrawStake(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn withdraw_stake_mut(&mut self) -> &mut WithdrawStake {
            if self.withdraw_stake_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::WithdrawStake(WithdrawStake::default()),
                );
            }
            self.withdraw_stake_opt_mut().unwrap()
        }
        pub fn with_withdraw_stake(mut self, field: WithdrawStake) -> Self {
            self.kind = Some(transaction_kind::Kind::WithdrawStake(field.into()));
            self
        }
        pub fn embed_data(&self) -> &EmbedData {
            if let Some(transaction_kind::Kind::EmbedData(field)) = &self.kind {
                field as _
            } else {
                EmbedData::default_instance() as _
            }
        }
        pub fn embed_data_opt(&self) -> Option<&EmbedData> {
            if let Some(transaction_kind::Kind::EmbedData(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn embed_data_opt_mut(&mut self) -> Option<&mut EmbedData> {
            if let Some(transaction_kind::Kind::EmbedData(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn embed_data_mut(&mut self) -> &mut EmbedData {
            if self.embed_data_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::EmbedData(EmbedData::default()),
                );
            }
            self.embed_data_opt_mut().unwrap()
        }
        pub fn with_embed_data(mut self, field: EmbedData) -> Self {
            self.kind = Some(transaction_kind::Kind::EmbedData(field.into()));
            self
        }
        pub fn claim_escrow(&self) -> &ClaimEscrow {
            if let Some(transaction_kind::Kind::ClaimEscrow(field)) = &self.kind {
                field as _
            } else {
                ClaimEscrow::default_instance() as _
            }
        }
        pub fn claim_escrow_opt(&self) -> Option<&ClaimEscrow> {
            if let Some(transaction_kind::Kind::ClaimEscrow(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn claim_escrow_opt_mut(&mut self) -> Option<&mut ClaimEscrow> {
            if let Some(transaction_kind::Kind::ClaimEscrow(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn claim_escrow_mut(&mut self) -> &mut ClaimEscrow {
            if self.claim_escrow_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ClaimEscrow(ClaimEscrow::default()),
                );
            }
            self.claim_escrow_opt_mut().unwrap()
        }
        pub fn with_claim_escrow(mut self, field: ClaimEscrow) -> Self {
            self.kind = Some(transaction_kind::Kind::ClaimEscrow(field.into()));
            self
        }
        pub fn report_winner(&self) -> &ReportWinner {
            if let Some(transaction_kind::Kind::ReportWinner(field)) = &self.kind {
                field as _
            } else {
                ReportWinner::default_instance() as _
            }
        }
        pub fn report_winner_opt(&self) -> Option<&ReportWinner> {
            if let Some(transaction_kind::Kind::ReportWinner(field)) = &self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn report_winner_opt_mut(&mut self) -> Option<&mut ReportWinner> {
            if let Some(transaction_kind::Kind::ReportWinner(field)) = &mut self.kind {
                Some(field as _)
            } else {
                None
            }
        }
        pub fn report_winner_mut(&mut self) -> &mut ReportWinner {
            if self.report_winner_opt_mut().is_none() {
                self.kind = Some(
                    transaction_kind::Kind::ReportWinner(ReportWinner::default()),
                );
            }
            self.report_winner_opt_mut().unwrap()
        }
        pub fn with_report_winner(mut self, field: ReportWinner) -> Self {
            self.kind = Some(transaction_kind::Kind::ReportWinner(field.into()));
            self
        }
    }
    impl AddValidator {
        pub const fn const_default() -> Self {
            Self {
                pubkey_bytes: None,
                network_pubkey_bytes: None,
                worker_pubkey_bytes: None,
                net_address: None,
                p2p_address: None,
                primary_address: None,
                encoder_validator_address: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: AddValidator = AddValidator::const_default();
            &DEFAULT
        }
        pub fn with_pubkey_bytes(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.pubkey_bytes = Some(field.into());
            self
        }
        pub fn with_network_pubkey_bytes(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.network_pubkey_bytes = Some(field.into());
            self
        }
        pub fn with_worker_pubkey_bytes(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.worker_pubkey_bytes = Some(field.into());
            self
        }
        pub fn with_net_address(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.net_address = Some(field.into());
            self
        }
        pub fn with_p2p_address(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.p2p_address = Some(field.into());
            self
        }
        pub fn with_primary_address(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.primary_address = Some(field.into());
            self
        }
        pub fn with_encoder_validator_address(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.encoder_validator_address = Some(field.into());
            self
        }
    }
    impl RemoveValidator {
        pub const fn const_default() -> Self {
            Self { pubkey_bytes: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: RemoveValidator = RemoveValidator::const_default();
            &DEFAULT
        }
        pub fn with_pubkey_bytes(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.pubkey_bytes = Some(field.into());
            self
        }
    }
    impl ReportValidator {
        pub const fn const_default() -> Self {
            Self { reportee: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ReportValidator = ReportValidator::const_default();
            &DEFAULT
        }
        pub fn with_reportee(mut self, field: String) -> Self {
            self.reportee = Some(field.into());
            self
        }
    }
    impl UndoReportValidator {
        pub const fn const_default() -> Self {
            Self { reportee: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: UndoReportValidator = UndoReportValidator::const_default();
            &DEFAULT
        }
        pub fn with_reportee(mut self, field: String) -> Self {
            self.reportee = Some(field.into());
            self
        }
    }
    impl UpdateValidatorMetadata {
        pub const fn const_default() -> Self {
            Self {
                next_epoch_network_address: None,
                next_epoch_p2p_address: None,
                next_epoch_primary_address: None,
                next_epoch_protocol_pubkey: None,
                next_epoch_worker_pubkey: None,
                next_epoch_network_pubkey: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: UpdateValidatorMetadata = UpdateValidatorMetadata::const_default();
            &DEFAULT
        }
        pub fn with_next_epoch_network_address(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_network_address = Some(field.into());
            self
        }
        pub fn with_next_epoch_p2p_address(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_p2p_address = Some(field.into());
            self
        }
        pub fn with_next_epoch_primary_address(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_primary_address = Some(field.into());
            self
        }
        pub fn with_next_epoch_protocol_pubkey(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_protocol_pubkey = Some(field.into());
            self
        }
        pub fn with_next_epoch_worker_pubkey(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_worker_pubkey = Some(field.into());
            self
        }
        pub fn with_next_epoch_network_pubkey(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_network_pubkey = Some(field.into());
            self
        }
    }
    impl SetCommissionRate {
        pub const fn const_default() -> Self {
            Self { new_rate: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SetCommissionRate = SetCommissionRate::const_default();
            &DEFAULT
        }
        pub fn with_new_rate(mut self, field: u64) -> Self {
            self.new_rate = Some(field.into());
            self
        }
    }
    impl AddEncoder {
        pub const fn const_default() -> Self {
            Self {
                encoder_pubkey_bytes: None,
                network_pubkey_bytes: None,
                internal_network_address: None,
                external_network_address: None,
                object_server_address: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: AddEncoder = AddEncoder::const_default();
            &DEFAULT
        }
        pub fn with_encoder_pubkey_bytes(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.encoder_pubkey_bytes = Some(field.into());
            self
        }
        pub fn with_network_pubkey_bytes(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.network_pubkey_bytes = Some(field.into());
            self
        }
        pub fn with_internal_network_address(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.internal_network_address = Some(field.into());
            self
        }
        pub fn with_external_network_address(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.external_network_address = Some(field.into());
            self
        }
        pub fn with_object_server_address(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.object_server_address = Some(field.into());
            self
        }
    }
    impl RemoveEncoder {
        pub const fn const_default() -> Self {
            Self { encoder_pubkey_bytes: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: RemoveEncoder = RemoveEncoder::const_default();
            &DEFAULT
        }
        pub fn with_encoder_pubkey_bytes(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.encoder_pubkey_bytes = Some(field.into());
            self
        }
    }
    impl ReportEncoder {
        pub const fn const_default() -> Self {
            Self { reportee: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ReportEncoder = ReportEncoder::const_default();
            &DEFAULT
        }
        pub fn with_reportee(mut self, field: String) -> Self {
            self.reportee = Some(field.into());
            self
        }
    }
    impl UndoReportEncoder {
        pub const fn const_default() -> Self {
            Self { reportee: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: UndoReportEncoder = UndoReportEncoder::const_default();
            &DEFAULT
        }
        pub fn with_reportee(mut self, field: String) -> Self {
            self.reportee = Some(field.into());
            self
        }
    }
    impl UpdateEncoderMetadata {
        pub const fn const_default() -> Self {
            Self {
                next_epoch_external_network_address: None,
                next_epoch_internal_network_address: None,
                next_epoch_network_pubkey: None,
                next_epoch_object_server_address: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: UpdateEncoderMetadata = UpdateEncoderMetadata::const_default();
            &DEFAULT
        }
        pub fn with_next_epoch_external_network_address(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_external_network_address = Some(field.into());
            self
        }
        pub fn with_next_epoch_internal_network_address(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_internal_network_address = Some(field.into());
            self
        }
        pub fn with_next_epoch_network_pubkey(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_network_pubkey = Some(field.into());
            self
        }
        pub fn with_next_epoch_object_server_address(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.next_epoch_object_server_address = Some(field.into());
            self
        }
    }
    impl SetEncoderCommissionRate {
        pub const fn const_default() -> Self {
            Self { new_rate: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SetEncoderCommissionRate = SetEncoderCommissionRate::const_default();
            &DEFAULT
        }
        pub fn with_new_rate(mut self, field: u64) -> Self {
            self.new_rate = Some(field.into());
            self
        }
    }
    impl SetEncoderBytePrice {
        pub const fn const_default() -> Self {
            Self { new_price: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: SetEncoderBytePrice = SetEncoderBytePrice::const_default();
            &DEFAULT
        }
        pub fn with_new_price(mut self, field: u64) -> Self {
            self.new_price = Some(field.into());
            self
        }
    }
    impl TransferCoin {
        pub const fn const_default() -> Self {
            Self {
                coin: None,
                amount: None,
                recipient: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TransferCoin = TransferCoin::const_default();
            &DEFAULT
        }
        pub fn coin(&self) -> &ObjectReference {
            self.coin
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ObjectReference::default_instance() as _)
        }
        pub fn coin_opt(&self) -> Option<&ObjectReference> {
            self.coin.as_ref().map(|field| field as _)
        }
        pub fn coin_opt_mut(&mut self) -> Option<&mut ObjectReference> {
            self.coin.as_mut().map(|field| field as _)
        }
        pub fn coin_mut(&mut self) -> &mut ObjectReference {
            self.coin.get_or_insert_default()
        }
        pub fn with_coin(mut self, field: ObjectReference) -> Self {
            self.coin = Some(field.into());
            self
        }
        pub fn with_amount(mut self, field: u64) -> Self {
            self.amount = Some(field.into());
            self
        }
        pub fn with_recipient(mut self, field: String) -> Self {
            self.recipient = Some(field.into());
            self
        }
    }
    impl PayCoins {
        pub const fn const_default() -> Self {
            Self {
                coins: Vec::new(),
                amounts: Vec::new(),
                recipients: Vec::new(),
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: PayCoins = PayCoins::const_default();
            &DEFAULT
        }
        pub fn coins(&self) -> &[ObjectReference] {
            &self.coins
        }
        pub fn coins_mut(&mut self) -> &mut Vec<ObjectReference> {
            &mut self.coins
        }
        pub fn with_coins(mut self, field: Vec<ObjectReference>) -> Self {
            self.coins = field;
            self
        }
        pub fn amounts(&self) -> &[u64] {
            &self.amounts
        }
        pub fn amounts_mut(&mut self) -> &mut Vec<u64> {
            &mut self.amounts
        }
        pub fn with_amounts(mut self, field: Vec<u64>) -> Self {
            self.amounts = field;
            self
        }
        pub fn recipients(&self) -> &[String] {
            &self.recipients
        }
        pub fn recipients_mut(&mut self) -> &mut Vec<String> {
            &mut self.recipients
        }
        pub fn with_recipients(mut self, field: Vec<String>) -> Self {
            self.recipients = field;
            self
        }
    }
    impl TransferObjects {
        pub const fn const_default() -> Self {
            Self {
                objects: Vec::new(),
                recipient: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TransferObjects = TransferObjects::const_default();
            &DEFAULT
        }
        pub fn objects(&self) -> &[ObjectReference] {
            &self.objects
        }
        pub fn objects_mut(&mut self) -> &mut Vec<ObjectReference> {
            &mut self.objects
        }
        pub fn with_objects(mut self, field: Vec<ObjectReference>) -> Self {
            self.objects = field;
            self
        }
        pub fn with_recipient(mut self, field: String) -> Self {
            self.recipient = Some(field.into());
            self
        }
    }
    impl AddStake {
        pub const fn const_default() -> Self {
            Self {
                address: None,
                coin_ref: None,
                amount: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: AddStake = AddStake::const_default();
            &DEFAULT
        }
        pub fn with_address(mut self, field: String) -> Self {
            self.address = Some(field.into());
            self
        }
        pub fn coin_ref(&self) -> &ObjectReference {
            self.coin_ref
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ObjectReference::default_instance() as _)
        }
        pub fn coin_ref_opt(&self) -> Option<&ObjectReference> {
            self.coin_ref.as_ref().map(|field| field as _)
        }
        pub fn coin_ref_opt_mut(&mut self) -> Option<&mut ObjectReference> {
            self.coin_ref.as_mut().map(|field| field as _)
        }
        pub fn coin_ref_mut(&mut self) -> &mut ObjectReference {
            self.coin_ref.get_or_insert_default()
        }
        pub fn with_coin_ref(mut self, field: ObjectReference) -> Self {
            self.coin_ref = Some(field.into());
            self
        }
        pub fn with_amount(mut self, field: u64) -> Self {
            self.amount = Some(field.into());
            self
        }
    }
    impl AddStakeToEncoder {
        pub const fn const_default() -> Self {
            Self {
                encoder_address: None,
                coin_ref: None,
                amount: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: AddStakeToEncoder = AddStakeToEncoder::const_default();
            &DEFAULT
        }
        pub fn with_encoder_address(mut self, field: String) -> Self {
            self.encoder_address = Some(field.into());
            self
        }
        pub fn coin_ref(&self) -> &ObjectReference {
            self.coin_ref
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ObjectReference::default_instance() as _)
        }
        pub fn coin_ref_opt(&self) -> Option<&ObjectReference> {
            self.coin_ref.as_ref().map(|field| field as _)
        }
        pub fn coin_ref_opt_mut(&mut self) -> Option<&mut ObjectReference> {
            self.coin_ref.as_mut().map(|field| field as _)
        }
        pub fn coin_ref_mut(&mut self) -> &mut ObjectReference {
            self.coin_ref.get_or_insert_default()
        }
        pub fn with_coin_ref(mut self, field: ObjectReference) -> Self {
            self.coin_ref = Some(field.into());
            self
        }
        pub fn with_amount(mut self, field: u64) -> Self {
            self.amount = Some(field.into());
            self
        }
    }
    impl WithdrawStake {
        pub const fn const_default() -> Self {
            Self { staked_soma: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: WithdrawStake = WithdrawStake::const_default();
            &DEFAULT
        }
        pub fn staked_soma(&self) -> &ObjectReference {
            self.staked_soma
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ObjectReference::default_instance() as _)
        }
        pub fn staked_soma_opt(&self) -> Option<&ObjectReference> {
            self.staked_soma.as_ref().map(|field| field as _)
        }
        pub fn staked_soma_opt_mut(&mut self) -> Option<&mut ObjectReference> {
            self.staked_soma.as_mut().map(|field| field as _)
        }
        pub fn staked_soma_mut(&mut self) -> &mut ObjectReference {
            self.staked_soma.get_or_insert_default()
        }
        pub fn with_staked_soma(mut self, field: ObjectReference) -> Self {
            self.staked_soma = Some(field.into());
            self
        }
    }
    impl EmbedData {
        pub const fn const_default() -> Self {
            Self {
                digest: None,
                data_size_bytes: None,
                coin_ref: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: EmbedData = EmbedData::const_default();
            &DEFAULT
        }
        pub fn with_digest(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.digest = Some(field.into());
            self
        }
        pub fn with_data_size_bytes(mut self, field: u64) -> Self {
            self.data_size_bytes = Some(field.into());
            self
        }
        pub fn coin_ref(&self) -> &ObjectReference {
            self.coin_ref
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ObjectReference::default_instance() as _)
        }
        pub fn coin_ref_opt(&self) -> Option<&ObjectReference> {
            self.coin_ref.as_ref().map(|field| field as _)
        }
        pub fn coin_ref_opt_mut(&mut self) -> Option<&mut ObjectReference> {
            self.coin_ref.as_mut().map(|field| field as _)
        }
        pub fn coin_ref_mut(&mut self) -> &mut ObjectReference {
            self.coin_ref.get_or_insert_default()
        }
        pub fn with_coin_ref(mut self, field: ObjectReference) -> Self {
            self.coin_ref = Some(field.into());
            self
        }
    }
    impl ClaimEscrow {
        pub const fn const_default() -> Self {
            Self { shard_input_ref: None }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ClaimEscrow = ClaimEscrow::const_default();
            &DEFAULT
        }
        pub fn shard_input_ref(&self) -> &ObjectReference {
            self.shard_input_ref
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ObjectReference::default_instance() as _)
        }
        pub fn shard_input_ref_opt(&self) -> Option<&ObjectReference> {
            self.shard_input_ref.as_ref().map(|field| field as _)
        }
        pub fn shard_input_ref_opt_mut(&mut self) -> Option<&mut ObjectReference> {
            self.shard_input_ref.as_mut().map(|field| field as _)
        }
        pub fn shard_input_ref_mut(&mut self) -> &mut ObjectReference {
            self.shard_input_ref.get_or_insert_default()
        }
        pub fn with_shard_input_ref(mut self, field: ObjectReference) -> Self {
            self.shard_input_ref = Some(field.into());
            self
        }
    }
    impl ReportWinner {
        pub const fn const_default() -> Self {
            Self {
                shard_input_ref: None,
                signed_report: None,
                encoder_aggregate_signature: None,
                signers: Vec::new(),
                shard_auth_token: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: ReportWinner = ReportWinner::const_default();
            &DEFAULT
        }
        pub fn shard_input_ref(&self) -> &ObjectReference {
            self.shard_input_ref
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ObjectReference::default_instance() as _)
        }
        pub fn shard_input_ref_opt(&self) -> Option<&ObjectReference> {
            self.shard_input_ref.as_ref().map(|field| field as _)
        }
        pub fn shard_input_ref_opt_mut(&mut self) -> Option<&mut ObjectReference> {
            self.shard_input_ref.as_mut().map(|field| field as _)
        }
        pub fn shard_input_ref_mut(&mut self) -> &mut ObjectReference {
            self.shard_input_ref.get_or_insert_default()
        }
        pub fn with_shard_input_ref(mut self, field: ObjectReference) -> Self {
            self.shard_input_ref = Some(field.into());
            self
        }
        pub fn with_signed_report(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.signed_report = Some(field.into());
            self
        }
        pub fn with_encoder_aggregate_signature(
            mut self,
            field: ::prost::bytes::Bytes,
        ) -> Self {
            self.encoder_aggregate_signature = Some(field.into());
            self
        }
        pub fn signers(&self) -> &[String] {
            &self.signers
        }
        pub fn signers_mut(&mut self) -> &mut Vec<String> {
            &mut self.signers
        }
        pub fn with_signers(mut self, field: Vec<String>) -> Self {
            self.signers = field;
            self
        }
        pub fn with_shard_auth_token(mut self, field: ::prost::bytes::Bytes) -> Self {
            self.shard_auth_token = Some(field.into());
            self
        }
    }
    impl ChangeEpoch {
        pub const fn const_default() -> Self {
            Self {
                epoch: None,
                epoch_start_timestamp: None,
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
        pub fn objects_mut(&mut self) -> &mut Vec<Object> {
            &mut self.objects
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
        pub fn signatures_mut(&mut self) -> &mut Vec<UserSignature> {
            &mut self.signatures
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
    }
    impl TransactionFee {
        pub const fn const_default() -> Self {
            Self {
                base_fee: None,
                operation_fee: None,
                value_fee: None,
                total_fee: None,
                gas_object_ref: None,
            }
        }
        #[doc(hidden)]
        pub fn default_instance() -> &'static Self {
            static DEFAULT: TransactionFee = TransactionFee::const_default();
            &DEFAULT
        }
        pub fn with_base_fee(mut self, field: u64) -> Self {
            self.base_fee = Some(field.into());
            self
        }
        pub fn with_operation_fee(mut self, field: u64) -> Self {
            self.operation_fee = Some(field.into());
            self
        }
        pub fn with_value_fee(mut self, field: u64) -> Self {
            self.value_fee = Some(field.into());
            self
        }
        pub fn with_total_fee(mut self, field: u64) -> Self {
            self.total_fee = Some(field.into());
            self
        }
        pub fn gas_object_ref(&self) -> &ObjectReference {
            self.gas_object_ref
                .as_ref()
                .map(|field| field as _)
                .unwrap_or_else(|| ObjectReference::default_instance() as _)
        }
        pub fn gas_object_ref_opt(&self) -> Option<&ObjectReference> {
            self.gas_object_ref.as_ref().map(|field| field as _)
        }
        pub fn gas_object_ref_opt_mut(&mut self) -> Option<&mut ObjectReference> {
            self.gas_object_ref.as_mut().map(|field| field as _)
        }
        pub fn gas_object_ref_mut(&mut self) -> &mut ObjectReference {
            self.gas_object_ref.get_or_insert_default()
        }
        pub fn with_gas_object_ref(mut self, field: ObjectReference) -> Self {
            self.gas_object_ref = Some(field.into());
            self
        }
    }
}
