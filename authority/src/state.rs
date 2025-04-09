//! # Authority State
//!
//! ## Overview
//! The Authority State module is the core state management component of the Soma blockchain validator.
//! It manages the validator's view of the blockchain state, processes transactions, and handles
//! epoch transitions.
//!
//! ## Responsibilities
//! - Maintaining the validator's state (objects, transactions, effects)
//! - Processing and executing transactions and certificates
//! - Managing epoch transitions and reconfiguration
//! - Coordinating transaction execution with consensus
//! - Providing transaction status and effects
//! - Ensuring thread-safe access to state
//!
//! ## Component Relationships
//! - Interacts with Consensus module to process ordered transactions
//! - Uses TransactionManager to track and execute pending transactions
//! - Manages AuthorityPerEpochStore for epoch-specific state
//! - Coordinates with StateAccumulator for state verification
//! - Provides interfaces for external services to query state
//!
//! ## Key Workflows
//! 1. Transaction processing: validation, execution, and effects generation
//! 2. Certificate execution: processing verified certificates from consensus
//! 3. Epoch reconfiguration: transitioning between epochs with validator set changes
//! 4. State synchronization: ensuring consistent state across validators
//!
//! ## Design Patterns
//! - Thread-safe state access via Arc<RwLock<>> and ArcSwap patterns
//! - Epoch-based isolation for reconfiguration safety
//! - Transactional execution with atomic commits
//! - Lock-based concurrency control for shared objects

use std::ops::Add;
use std::{pin::Pin, sync::Arc};

use arc_swap::{ArcSwap, Guard};
use fastcrypto::hash::MultisetHash;
use parking_lot::Mutex;
use tap::TapFallible;
use tokio::sync::{mpsc::unbounded_channel, oneshot};
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracing::{debug, error, info, instrument, warn, Instrument};
use types::digests::{ECMHLiveObjectSetDigest, TransactionEffectsDigest};
use types::effects::{
    self, SignedTransactionEffects, TransactionEffects, TransactionEffectsAPI,
    VerifiedCertifiedTransactionEffects, VerifiedSignedTransactionEffects,
};
use types::envelope::Message;
use types::error::ExecutionError;
use types::object::{Object, ObjectID, ObjectRef};
use types::state_sync::CommitTimestamp;
use types::storage::object_store::ObjectStore;
use types::system_state::{epoch_start::EpochStartSystemStateTrait, SystemState};
use types::temporary_store::InnerTemporaryStore;
use types::transaction::{InputObjects, SenderSignedData};
use types::tx_outputs::{TransactionOutputs, WrittenObjects};
use types::SYSTEM_STATE_OBJECT_ID;
use types::{
    accumulator::{AccumulatorStore, CommitIndex},
    base::AuthorityName,
    committee::{Committee, EpochId},
    config::node_config::NodeConfig,
    crypto::{AuthoritySignInfo, AuthoritySignature, Signer},
    digests::TransactionDigest,
    error::{SomaError, SomaResult},
    grpc::{HandleTransactionResponse, TransactionStatus},
    intent::{Intent, IntentScope},
    transaction::{
        VerifiedCertificate, VerifiedExecutableTransaction, VerifiedSignedTransaction,
        VerifiedTransaction,
    },
};

use crate::cache::{
    ExecutionCacheCommit, ExecutionCacheReconfigAPI, ExecutionCacheTraitPointers,
    ExecutionCacheWrite, ObjectCacheRead, TransactionCacheRead,
};
use crate::consensus_quarantine;
use crate::epoch_store::{CertLockGuard, CertTxGuard};
use crate::execution::execute_transaction;
use crate::execution_driver::execution_process;
use crate::start_epoch::EpochStartConfigTrait;
use crate::state_accumulator::StateAccumulator;
use crate::store::{AuthorityStore, ObjectLockStatus};
use crate::tx_input_loader::TransactionInputLoader;
use crate::{
    client::NetworkAuthorityClient, epoch_store::AuthorityPerEpochStore,
    start_epoch::EpochStartConfiguration, tx_manager::TransactionManager,
};
use types::storage::committee_store::CommitteeStore;

#[cfg(test)]
#[path = "unit_tests/authority_tests.rs"]
pub mod authority_tests;

/// # StableSyncAuthoritySigner
///
/// A trait object for `Signer` that provides thread-safe and memory-safe access to authority signing capabilities.
///
/// ## Purpose
/// Provides a secure way to handle cryptographic signing operations for an authority without
/// exposing or copying private key material.
///
/// ## Thread Safety
/// - `Pin`: Ensures the signer is confined to one place in memory for security (prevents copying private keys)
/// - `Arc<dyn Signer>`: Allows safe sharing between threads
/// - `Send + Sync`: Enables concurrent access from multiple threads
///
/// ## Usage
/// Typically instantiated with `Box::pin(keypair)` where keypair is a `KeyPair` implementation
/// that provides the cryptographic signing capabilities for the authority.
///
pub type StableSyncAuthoritySigner = Pin<Arc<dyn Signer<AuthoritySignature> + Send + Sync>>;

/// # AuthorityState
///
/// The core state management component of a Soma blockchain validator or fullnode.
///
/// ## Purpose
/// AuthorityState is responsible for maintaining the validator's view of the blockchain state,
/// processing transactions, executing certificates, and managing epoch transitions.
/// It serves as the central coordination point for transaction processing and state management.
///
/// ## Lifecycle
/// - Created during node startup with validator identity and initial epoch information
/// - Persists throughout the node's lifetime, managing epoch transitions
/// - Handles transaction processing and certificate execution
/// - Coordinates with consensus for transaction ordering
///
/// ## Thread Safety
/// This struct is designed for concurrent access with careful lock management:
/// - Uses `ArcSwap` for epoch store updates
/// - Employs `RwLock` for execution epoch management
/// - Coordinates transaction execution with transaction-specific locks
///
/// ## Key Components
/// - Epoch management via epoch_store and execution_lock
/// - Transaction processing via transaction_manager
/// - State verification via accumulator
/// - Object and transaction storage via execution_cache_trait_pointers
pub struct AuthorityState {
    /// The name (public key) of this authority.
    pub name: AuthorityName,

    /// The signature key of the authority used for signing transactions and effects.
    pub secret: StableSyncAuthoritySigner,

    /// The epoch-specific store, swapped atomically during reconfiguration.
    /// Contains epoch-specific committee information, protocol configs, and state.
    epoch_store: ArcSwap<AuthorityPerEpochStore>,

    /// Lock that denotes current 'execution epoch'.
    /// - Execution acquires read lock, checks certificate epoch and holds it until all writes are complete.
    /// - Reconfiguration acquires write lock, changes the epoch and reverts all transactions
    ///   from previous epoch that are executed but did not make it into a checkpoint.
    execution_lock: RwLock<EpochId>,

    /// Store for committee information across epochs.
    committee_store: Arc<CommitteeStore>,

    /// Manages pending certificates and their missing input objects.
    /// Responsible for transaction dependency tracking and execution ordering.
    transaction_manager: Arc<TransactionManager>,

    /// Node configuration parameters.
    pub config: NodeConfig,

    /// Channel sender to shut down the execution task. Used only in testing.
    #[allow(unused)]
    tx_execution_shutdown: Mutex<Option<oneshot::Sender<()>>>,

    /// Loader for transaction input objects.
    input_loader: TransactionInputLoader,

    /// Trait objects for accessing the execution cache and storage.
    execution_cache_trait_pointers: ExecutionCacheTraitPointers,

    /// The state accumulator for verifying state consistency.
    accumulator: Arc<StateAccumulator>,
}

impl AuthorityState {
    /// # Create a new AuthorityState
    ///
    /// Creates and initializes a new authority state with the provided components.
    ///
    /// ## Arguments
    /// * `name` - The authority's name (public key)
    /// * `secret` - The authority's signing key
    /// * `epoch_store` - The initial epoch store for the authority
    /// * `committee_store` - Store for committee information across epochs
    /// * `config` - Node configuration parameters
    /// * `execution_cache_trait_pointers` - Trait objects for accessing execution cache and storage
    /// * `accumulator` - State accumulator for verifying state consistency
    ///
    /// ## Returns
    /// An Arc-wrapped AuthorityState instance
    ///
    /// ## Behavior
    /// - Initializes the transaction manager
    /// - Sets up the execution lock with the current epoch
    /// - Spawns a background task to process ready certificates
    #[allow(clippy::disallowed_methods)] // allow unbounded_channel()
    pub async fn new(
        name: AuthorityName,
        secret: StableSyncAuthoritySigner,
        epoch_store: Arc<AuthorityPerEpochStore>,
        committee_store: Arc<CommitteeStore>,
        config: NodeConfig,
        execution_cache_trait_pointers: ExecutionCacheTraitPointers,
        accumulator: Arc<StateAccumulator>,
    ) -> Arc<Self> {
        let (tx_ready_certificates, rx_ready_certificates) = unbounded_channel();
        let transaction_manager = Arc::new(TransactionManager::new(
            execution_cache_trait_pointers.object_cache_reader.clone(),
            &epoch_store,
            tx_ready_certificates,
            execution_cache_trait_pointers
                .transaction_cache_reader
                .clone(),
        ));
        let (tx_execution_shutdown, rx_execution_shutdown) = oneshot::channel();

        let epoch = epoch_store.epoch();
        let input_loader =
            TransactionInputLoader::new(execution_cache_trait_pointers.object_cache_reader.clone());

        let state = Arc::new(AuthorityState {
            name,
            secret,
            execution_lock: RwLock::new(epoch),
            epoch_store: ArcSwap::new(epoch_store.clone()),
            tx_execution_shutdown: Mutex::new(Some(tx_execution_shutdown)),
            committee_store,
            transaction_manager,
            config,
            input_loader,
            execution_cache_trait_pointers,
            accumulator,
        });

        // Start a task to execute ready certificates.
        let authority_state = Arc::downgrade(&state);
        tokio::spawn(execution_process(
            authority_state,
            rx_ready_certificates,
            rx_execution_shutdown,
        ));

        state
    }

    /// # Get transaction cache reader
    ///
    /// Returns a reference to the transaction cache reader component.
    ///
    /// ## Purpose
    /// Provides access to transaction-related data in the cache, including transaction
    /// blocks, effects, and execution status.
    ///
    /// ## Usage
    /// Used throughout the authority state to read transaction data without directly
    /// accessing the underlying storage.
    pub fn get_transaction_cache_reader(&self) -> &Arc<dyn TransactionCacheRead> {
        &self.execution_cache_trait_pointers.transaction_cache_reader
    }

    /// # Get cache writer
    ///
    /// Returns a reference to the execution cache writer component.
    ///
    /// ## Purpose
    /// Provides the ability to write transaction outputs, acquire locks, and
    /// update the execution cache.
    ///
    /// ## Usage
    /// Used during transaction execution to write results to the cache.
    pub fn get_cache_writer(&self) -> &Arc<dyn ExecutionCacheWrite> {
        &self.execution_cache_trait_pointers.cache_writer
    }

    /// # Get object store
    ///
    /// Returns a reference to the object store component.
    ///
    /// ## Purpose
    /// Provides access to the persistent storage of objects.
    ///
    /// ## Usage
    /// Used during transaction execution and state verification to access
    /// the current state of objects.
    pub fn get_object_store(&self) -> &Arc<dyn ObjectStore + Send + Sync> {
        &self.execution_cache_trait_pointers.object_store
    }

    /// # Get object cache reader
    ///
    /// Returns a reference to the object cache reader component.
    ///
    /// ## Purpose
    /// Provides access to object-related data in the cache, including
    /// object versions, locks, and the system state object.
    ///
    /// ## Usage
    /// Used throughout the authority state to read object data without
    /// directly accessing the underlying storage.
    pub fn get_object_cache_reader(&self) -> &Arc<dyn ObjectCacheRead> {
        &self.execution_cache_trait_pointers.object_cache_reader
    }

    /// # Get accumulator store
    ///
    /// Returns a reference to the accumulator store component.
    ///
    /// ## Purpose
    /// Provides access to the storage for the state accumulator, which is used
    /// to verify the consistency of the blockchain state.
    ///
    /// ## Usage
    /// Used during state verification and accumulation processes.
    pub fn get_accumulator_store(&self) -> &Arc<dyn AccumulatorStore> {
        &self.execution_cache_trait_pointers.accumulator_store
    }

    /// # Get cache commit
    ///
    /// Returns a reference to the execution cache commit component.
    ///
    /// ## Purpose
    /// Provides the ability to commit transaction outputs to persistent storage.
    ///
    /// ## Usage
    /// Used after transaction execution to ensure durability of the results.
    pub fn get_cache_commit(&self) -> &Arc<dyn ExecutionCacheCommit> {
        &self.execution_cache_trait_pointers.cache_commit
    }

    pub fn get_reconfig_api(&self) -> &Arc<dyn ExecutionCacheReconfigAPI> {
        &self.execution_cache_trait_pointers.reconfig_api
    }

    pub fn database_for_testing(&self) -> Arc<AuthorityStore> {
        self.execution_cache_trait_pointers
            .testing_api
            .database_for_testing()
    }

    /// This is a private method and should be kept that way. It doesn't check whether
    /// the provided transaction is a system transaction, and hence can only be called internally.
    #[instrument(level = "trace", skip_all)]
    async fn handle_transaction_impl(
        &self,
        transaction: VerifiedTransaction,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<VerifiedSignedTransaction> {
        // Ensure that validator cannot reconfigure while we are signing the tx
        let _execution_lock = self.execution_lock_for_signing();

        let tx_digest = transaction.digest();
        let tx_data = transaction.data().transaction_data();

        let input_object_kinds = tx_data.input_objects()?;
        let receiving_objects_refs = tx_data.receiving_objects();
        let (input_objects, receiving_objects) = self.input_loader.read_objects_for_signing(
            Some(tx_digest),
            &input_object_kinds,
            &receiving_objects_refs,
            epoch_store.epoch(),
        )?;

        let owned_objects = input_objects.filter_owned_objects();

        let signed_transaction = VerifiedSignedTransaction::new(
            epoch_store.epoch(),
            transaction.clone(),
            self.name,
            &*self.secret,
        );

        // TODO: check_transaction_input
        // TODO: check_transaction_for_signing

        // Check and write locks, to signed transaction, into the database
        // The call to self.set_transaction_lock checks the lock is not conflicting,
        // and returns ConflictingTransaction error in case there is a lock on a different
        // existing transaction.
        self.get_cache_writer().acquire_transaction_locks(
            epoch_store,
            &owned_objects,
            *tx_digest,
            signed_transaction.clone(),
        )?;

        Ok(signed_transaction)
    }

    /// # Initiate a new transaction
    ///
    /// Processes a new transaction by validating it, signing it, and acquiring locks on input objects.
    ///
    /// ## Arguments
    /// * `epoch_store` - The current epoch store
    /// * `transaction` - The verified transaction to process
    ///
    /// ## Returns
    /// A response containing the transaction status, which may be:
    /// - `TransactionStatus::Signed` - If the transaction was successfully signed
    /// - Other status if the transaction was already processed
    ///
    /// ## Behavior
    /// 1. Checks if the transaction has already been processed
    /// 2. Verifies the validator is not halted for epoch end
    /// 3. Reads input objects and validates transaction
    /// 4. Signs the transaction and acquires locks on owned objects
    /// 5. Returns the transaction status
    ///
    /// ## Errors
    /// - `ValidatorHaltedAtEpochEnd` - If the validator is no longer accepting transactions
    /// - Various errors from transaction validation or lock acquisition
    #[instrument(level = "trace", skip_all)]
    pub async fn handle_transaction(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        transaction: VerifiedTransaction,
    ) -> SomaResult<HandleTransactionResponse> {
        let tx_digest = *transaction.digest();
        debug!("handle_transaction");

        // Ensure an idempotent answer.
        if let Some((_, status)) = self.get_transaction_status(&tx_digest, epoch_store)? {
            return Ok(HandleTransactionResponse { status });
        }

        // The should_accept_user_certs check here is best effort, because
        // between a validator signs a tx and a cert is formed, the validator
        // could close the window.
        if !epoch_store
            .get_reconfig_state_read_lock_guard()
            .should_accept_user_certs()
        {
            return Err(SomaError::ValidatorHaltedAtEpochEnd);
        }

        let signed = self.handle_transaction_impl(transaction, epoch_store).await;
        match signed {
            Ok(s) => {
                if self.is_validator(epoch_store) {
                    // if let Some(validator_tx_finalizer) = &self.validator_tx_finalizer {
                    //     let tx = s.clone();
                    //     let validator_tx_finalizer = validator_tx_finalizer.clone();
                    //     // let cache_reader = self.get_transaction_cache_reader().clone();
                    //     let epoch_store = epoch_store.clone();
                    //     tokio::spawn(epoch_store.within_alive_epoch(
                    //         validator_tx_finalizer.track_signed_tx(cache_reader, tx),
                    //     ));
                    // }
                }
                Ok(HandleTransactionResponse {
                    status: TransactionStatus::Signed(s.into_inner().into_sig()),
                })
            }
            // It happens frequently that while we are checking the validity of the transaction, it
            // has just been executed.
            // In that case, we could still return Ok to avoid showing confusing errors.
            Err(err) => Ok(HandleTransactionResponse {
                status: self
                    .get_transaction_status(&tx_digest, epoch_store)?
                    .ok_or(err)?
                    .1,
            }),
        }
    }

    /// Executes a transaction that's known to have correct effects.
    /// For such transaction, we don't have to wait for consensus to set shared object
    /// locks because we already know the shared object versions based on the effects.
    /// This function can be called by a fullnode only.
    #[instrument(level = "trace", skip_all)]
    pub async fn fullnode_execute_certificate_with_effects(
        &self,
        transaction: &VerifiedExecutableTransaction,
        // NOTE: the caller of this must promise to wait until it
        // knows for sure this tx is finalized, namely, it has seen a
        // CertifiedTransactionEffects or at least f+1 identifical effects
        // digests matching this TransactionEffectsEnvelope, before calling
        // this function, in order to prevent a byzantine validator from
        // giving us incorrect effects.
        effects: &VerifiedCertifiedTransactionEffects,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult {
        assert!(self.is_fullnode(epoch_store));
        // NOTE: the fullnode can change epoch during local execution. It should not cause
        // data inconsistency, but can be problematic for certain tests.
        // The check below mitigates the issue, but it is not a fundamental solution to
        // avoid race between local execution and reconfiguration.
        if self.epoch_store.load().epoch() != epoch_store.epoch() {
            return Err(SomaError::EpochEnded(epoch_store.epoch()));
        }

        let digest = *transaction.digest();
        debug!("execute_certificate_with_effects");

        if *effects.data().transaction_digest() != digest {
            return Err(SomaError::ErrorWhileProcessingCertificate {
                err: "effects/tx digest mismatch".to_string(),
            });
        }

        let expected_effects_digest = effects.digest();

        self.transaction_manager
            .enqueue(vec![transaction.clone()], epoch_store, None);

        let observed_effects = self
            .get_transaction_cache_reader()
            .notify_read_executed_effects(&[digest])
            .instrument(tracing::debug_span!(
                "notify_read_effects_in_execute_certificate_with_effects"
            ))
            .await?
            .pop()
            .expect("notify_read_effects should return exactly 1 element");

        let observed_effects_digest = observed_effects.digest();
        if &observed_effects_digest != expected_effects_digest {
            panic!(
                "Locally executed effects do not match canonical effects! expected_effects_digest={:?} observed_effects_digest={:?} expected_effects={:?} observed_effects={:?}",
                expected_effects_digest, observed_effects_digest, effects.data(), observed_effects
            );
        }
        Ok(())
    }

    /// # Execute a certificate for its effects
    ///
    /// Processes a verified certificate by executing it and returning its effects.
    ///
    /// ## Arguments
    /// * `certificate` - The verified certificate to execute
    /// * `epoch_store` - The current epoch store
    ///
    /// ## Returns
    /// The transaction effects resulting from execution
    ///
    /// ## Behavior
    /// 1. For owned object transactions (no shared objects), immediately enqueues for execution
    /// 2. For shared object transactions, relies on consensus to sequence them first
    /// 3. Waits for transaction execution to complete and returns the effects
    /// 4. Ensures execution happens within the current epoch
    ///
    /// ## Errors
    /// - `EpochEnded` - If the epoch ends during execution
    /// - Various errors from transaction execution
    #[instrument(level = "trace", skip_all)]
    pub async fn execute_certificate(
        &self,
        certificate: &VerifiedCertificate,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<TransactionEffects> {
        debug!("execute_certificate");

        // self.enqueue_certificates_for_execution(vec![certificate.clone()], epoch_store);
        // self.notify_read_effects(certificate).await

        // TODO(fastpath): use a separate function to check if a transaction should be executed in fastpath.
        if !certificate.contains_shared_object() {
            // Shared object transactions need to be sequenced by the consensus before enqueueing
            // for execution, done in AuthorityPerEpochStore::handle_consensus_transaction().
            // For owned object transactions, they can be enqueued for execution immediately.
            self.enqueue_certificates_for_execution(vec![certificate.clone()], epoch_store);
        }

        // tx could be reverted when epoch ends, so we must be careful not to return a result
        // here after the epoch ends.
        epoch_store
            .within_alive_epoch(self.notify_read_effects(certificate))
            .await
            .map_err(|_| SomaError::EpochEnded(epoch_store.epoch()))
            .and_then(|r| r)
    }

    /// # Wait for transaction effects
    ///
    /// Waits for a transaction's execution to complete and returns its effects.
    ///
    /// ## Arguments
    /// * `certificate` - The verified certificate whose effects to wait for
    ///
    /// ## Returns
    /// The transaction effects once execution is complete
    ///
    /// ## Behavior
    /// This is a blocking operation that waits until the transaction has been
    /// executed and its effects are available in the transaction cache.
    ///
    /// ## Usage
    /// Used after enqueueing a transaction for execution to wait for its completion.
    pub async fn notify_read_effects(
        &self,
        certificate: &VerifiedCertificate,
    ) -> SomaResult<TransactionEffects> {
        self.get_transaction_cache_reader()
            .notify_read_executed_effects(&[*certificate.digest()])
            .await
            .map(|mut r| r.pop().expect("must return correct number of effects"))
    }

    /// Test only wrapper for `try_execute_immediately()` above, useful for checking errors if the
    /// pre-conditions are not satisfied, and executing change epoch transactions.
    pub async fn try_execute_for_test(
        &self,
        certificate: &VerifiedCertificate,
    ) -> SomaResult<(VerifiedSignedTransactionEffects, Option<ExecutionError>)> {
        let epoch_store = self.epoch_store_for_testing();
        let (effects, execution_error_opt) = self
            .try_execute_immediately(
                &VerifiedExecutableTransaction::new_from_certificate(certificate.clone()),
                None,
                None,
                &epoch_store,
            )
            .await?;
        let signed_effects = self.sign_effects(effects, &epoch_store)?;
        Ok((signed_effects, execution_error_opt))
    }

    /// Internal logic to execute a certificate.
    ///
    /// Guarantees that
    /// - If input objects are available, return no permanent failure.
    /// - Execution and output commit are atomic. i.e. outputs are only written to storage,
    /// on successful execution; crashed execution has no observable effect and can be retried.
    ///
    /// It is caller's responsibility to ensure input objects are available and locks are set.
    /// If this cannot be satisfied by the caller, execute_certificate() should be called instead.
    ///
    /// Should only be called within core.
    #[instrument(level = "debug", skip_all)]
    pub async fn try_execute_immediately(
        &self,
        certificate: &VerifiedExecutableTransaction,
        mut expected_effects_digest: Option<TransactionEffectsDigest>,
        commit: Option<CommitIndex>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<(TransactionEffects, Option<ExecutionError>)> {
        debug!("execute_certificate_internal");

        let tx_digest = certificate.digest();

        // This acquires a lock on the tx digest to prevent multiple concurrent executions of the
        // same tx. While we don't need this for safety (tx sequencing is ultimately atomic), it is
        // very common to receive the same tx multiple times simultaneously due to gossip, so we
        // may as well hold the lock and save the cpu time for other requests.
        let tx_guard = epoch_store.acquire_tx_guard(certificate).await?;

        // The cert could have been processed by a concurrent attempt of the same cert, so check if
        // the effects have already been written.
        if let Some(effects) = self
            .get_transaction_cache_reader()
            .get_executed_effects(tx_digest)?
        {
            info!("processed by concurrent attempt of the same cert");
            tx_guard.release();
            return Ok((effects, None));
        }

        let input_objects =
            self.read_objects_for_execution(tx_guard.as_lock_guard(), certificate, epoch_store)?;

        if expected_effects_digest.is_none() {
            // We could be re-executing a previously executed but uncommitted transaction, perhaps after
            // restarting with a new binary. In this situation, if we have published an effects signature,
            // we must be sure not to equivocate.
            // TODO: read from cache instead of DB
            expected_effects_digest = epoch_store.get_signed_effects_digest(tx_digest)?;
        }

        self.process_certificate(
            tx_guard,
            certificate,
            input_objects,
            expected_effects_digest,
            commit,
            epoch_store,
        )
        .await
        .tap_err(|e| info!(?tx_digest, "process_certificate failed: {e}"))
    }

    /// # Read objects for transaction execution
    ///
    /// Loads all input objects required for executing a transaction.
    ///
    /// ## Arguments
    /// * `tx_lock` - Lock guard for the transaction
    /// * `certificate` - The verified transaction to execute
    /// * `epoch_store` - The current epoch store
    ///
    /// ## Returns
    /// The input objects needed for transaction execution
    ///
    /// ## Behavior
    /// Uses the transaction input loader to read all objects specified in the transaction's
    /// input objects list, ensuring they are available and at the correct versions.
    ///
    /// ## Errors
    /// - If any input object is not found
    /// - If object versions don't match what's specified in the transaction
    /// - If locks cannot be acquired for owned objects
    pub fn read_objects_for_execution(
        &self,
        tx_lock: &CertLockGuard,
        certificate: &VerifiedExecutableTransaction,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<InputObjects> {
        let input_objects = &certificate.data().transaction_data().input_objects()?;
        self.input_loader.read_objects_for_execution(
            epoch_store,
            &certificate.key(),
            tx_lock,
            input_objects,
            epoch_store.epoch(),
        )
    }

    #[instrument(level = "debug", skip_all)]
    pub(crate) async fn process_certificate(
        &self,
        tx_guard: CertTxGuard,
        certificate: &VerifiedExecutableTransaction,
        input_objects: InputObjects,
        expected_effects_digest: Option<TransactionEffectsDigest>,
        commit: Option<CommitIndex>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<(TransactionEffects, Option<ExecutionError>)> {
        let process_certificate_start_time = tokio::time::Instant::now();
        let digest = *certificate.digest();

        // The cert could have been processed by a concurrent attempt of the same cert, so check if
        // the effects have already been written.
        if let Some(effects) = self
            .get_transaction_cache_reader()
            .get_executed_effects(&digest)?
        {
            tx_guard.release();
            info!(
                "Cert processed by a concurrent attempt of the same cert, not processed: {}",
                digest
            );
            return Ok((effects, None));
        }
        let execution_guard = self
            .execution_lock_for_executable_transaction(certificate)
            .await;
        // Any caller that verifies the signatures on the certificate will have already checked the
        // epoch. But paths that don't verify sigs (e.g. execution from checkpoint, reading from db)
        // present the possibility of an epoch mismatch. If this cert is not finalzied in previous
        // epoch, then it's invalid.
        let execution_guard = match execution_guard {
            Ok(execution_guard) => execution_guard,
            Err(err) => {
                tx_guard.release();
                return Err(err);
            }
        };
        // Since we obtain a reference to the epoch store before taking the execution lock, it's
        // possible that reconfiguration has happened and they no longer match.

        if *execution_guard != epoch_store.epoch() {
            tx_guard.release();
            info!("The epoch of the execution_guard doesn't match the epoch store");
            return Err(SomaError::WrongEpoch {
                expected_epoch: epoch_store.epoch(),
                actual_epoch: *execution_guard,
            });
        }

        // Errors originating from prepare_certificate may be transient (failure to read locks) or
        // non-transient (transaction input is invalid, move vm errors). However, all errors from
        // this function occur before we have written anything to the db, so we commit the tx
        // guard and rely on the client to retry the tx (if it was transient).
        let (inner, effects, execution_error_opt) = match self.prepare_certificate(
            &execution_guard,
            certificate,
            input_objects,
            epoch_store,
        ) {
            Err(e) => {
                info!(name = ?self.name, ?digest, "Error preparing transaction: {e}");
                tx_guard.release();
                return Err(e);
            }
            Ok(res) => res,
        };

        self.commit_certificate(
            certificate,
            inner,
            &effects,
            tx_guard,
            execution_guard,
            epoch_store,
        )
        .await?;

        Ok((effects, execution_error_opt))
    }

    /// # Check if transaction is already executed
    ///
    /// Determines whether a transaction has already been executed.
    ///
    /// ## Arguments
    /// * `digest` - The transaction digest to check
    ///
    /// ## Returns
    /// * `true` - If the transaction has been executed
    /// * `false` - If the transaction has not been executed
    ///
    /// ## Purpose
    /// Used to avoid re-executing transactions and to determine
    /// if a transaction needs to be processed.
    pub fn is_tx_already_executed(&self, digest: &TransactionDigest) -> SomaResult<bool> {
        self.get_transaction_cache_reader()
            .is_tx_already_executed(digest)
    }

    #[instrument(level = "trace", skip_all)]
    async fn commit_certificate(
        &self,
        certificate: &VerifiedExecutableTransaction,
        inner_temporary_store: InnerTemporaryStore,
        effects: &TransactionEffects,
        tx_guard: CertTxGuard,
        _execution_guard: ExecutionLockReadGuard<'_>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult {
        let tx_key = certificate.key();
        let tx_digest = certificate.digest();
        let output_keys = inner_temporary_store.get_output_keys(effects);

        // Only need to sign effects if we are a validator, and if the executed_in_epoch_table is not yet enabled.
        // TODO: once executed_in_epoch_table is enabled everywhere, we can remove the code below entirely.
        let should_sign_effects = self.is_validator(epoch_store);

        let effects_sig = if should_sign_effects {
            Some(AuthoritySignInfo::new(
                epoch_store.epoch(),
                effects,
                Intent::soma_transaction(),
                self.name,
                &*self.secret,
            ))
        } else {
            None
        };

        // index certificate
        // let _ = self
        //     .post_process_one_tx(certificate, effects, &inner_temporary_store, epoch_store)
        //     .await
        //     .tap_err(|e| {
        //         error!(?tx_digest, "tx post processing failed: {e}");
        //     });

        info!(
            tx_digest=?tx_digest,
            "About to commit certificate for transaction. Inputs: {:?}",
            inner_temporary_store.input_objects
        );

        // The insertion to epoch_store is not atomic with the insertion to the perpetual store. This is OK because
        // we insert to the epoch store first. And during lookups we always look up in the perpetual store first.
        epoch_store.insert_tx_key_and_effects_signature(
            &tx_key,
            tx_digest,
            &effects.digest(),
            effects_sig.as_ref(),
        )?;

        let transaction_outputs = TransactionOutputs::build_transaction_outputs(
            certificate.clone().into_unsigned(),
            effects.clone(),
            inner_temporary_store,
        );
        self.get_cache_writer()
            .write_transaction_outputs(epoch_store.epoch(), transaction_outputs.into())
            .await?;

        // commit_certificate finished, the tx is fully committed to the store.
        tx_guard.commit_tx();

        // Notifies transaction manager about transaction and output objects committed.
        // This provides necessary information to transaction manager to start executing
        // additional ready transactions.
        self.transaction_manager
            .notify_commit(tx_digest, output_keys, epoch_store);

        Ok(())
    }

    /// prepare_certificate validates the transaction input, and executes the certificate,
    /// returning effects, output objects, events, etc.
    ///
    /// It reads state from the db (both owned and shared locks), but it has no side effects.
    ///
    /// It can be generally understood that a failure of prepare_certificate indicates a
    /// non-transient error, e.g. the transaction input is somehow invalid, the correct
    /// locks are not held, etc. However, this is not entirely true, as a transient db read error
    /// may also cause this function to fail.
    #[instrument(level = "trace", skip_all)]
    fn prepare_certificate(
        &self,
        _execution_guard: &ExecutionLockReadGuard<'_>,
        certificate: &VerifiedExecutableTransaction,
        input_objects: InputObjects,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<(
        InnerTemporaryStore,
        TransactionEffects,
        Option<ExecutionError>,
    )> {
        let prepare_certificate_start_time = tokio::time::Instant::now();

        // TODO: We need to move this to a more appropriate place to avoid redundant checks.
        let tx_data = certificate.data().transaction_data();

        // TODO: check_certificate_input

        let owned_object_refs = input_objects.filter_owned_objects();
        // TODO: CRASHING self.check_owned_locks(&owned_object_refs)?;
        let tx_digest = *certificate.digest();
        let protocol_config = epoch_store.protocol_config();
        let transaction_data = &certificate.data().intent_message().value;
        let (kind, signer, gas_payment) = transaction_data.execution_parts();

        let (inner, effects, execution_error_opt) = execute_transaction(
            epoch_store.epoch(),
            self.get_object_store().as_ref(),
            tx_digest,
            kind,
            signer,
            gas_payment,
            input_objects,
        );

        Ok((inner, effects, execution_error_opt))
    }

    /// # Check if owned objects are live
    ///
    /// Verifies that all owned objects referenced in a transaction are live and available.
    ///
    /// ## Arguments
    /// * `owned_object_refs` - References to owned objects to check
    ///
    /// ## Returns
    /// * `Ok(())` - If all objects are live and available
    /// * `Err(...)` - If any object is not live or not available
    ///
    /// ## Purpose
    /// Used during transaction validation to ensure that all owned objects
    /// referenced by the transaction are available for use.
    fn check_owned_locks(&self, owned_object_refs: &[ObjectRef]) -> SomaResult {
        self.get_object_cache_reader()
            .check_owned_objects_are_live(owned_object_refs)
    }

    #[instrument(level = "error", skip_all)]
    pub async fn reconfigure(
        &self,
        cur_epoch_store: &AuthorityPerEpochStore,
        new_committee: Committee,
        epoch_start_configuration: EpochStartConfiguration,
        epoch_last_commit: CommitIndex,
        // expensive_safety_check_config: &ExpensiveSafetyCheckConfig,
    ) -> SomaResult<Arc<AuthorityPerEpochStore>> {
        let new_epoch = new_committee.epoch;
        self.committee_store
            .insert_new_committee(new_committee.clone())?;
        let mut execution_lock = self.execution_lock_for_reconfiguration().await;

        // TODO: check system consistency using accumulator after reverting uncommitted epoch transactions

        let new_epoch_store = self
            .reopen_epoch_db(
                cur_epoch_store,
                new_committee,
                epoch_start_configuration,
                epoch_last_commit,
            )
            .await?;
        assert_eq!(new_epoch_store.epoch(), new_epoch);
        self.transaction_manager.reconfigure(new_epoch);
        *execution_lock = new_epoch;
        // drop execution_lock after epoch store was updated
        // see also assert in AuthorityState::process_certificate
        // on the epoch store and execution lock epoch match
        Ok(new_epoch_store)
    }

    /// Advance the epoch store to the next epoch for testing only.
    /// This only manually sets all the places where we have the epoch number.
    /// It doesn't properly reconfigure the node, hence should be only used for testing.
    pub async fn reconfigure_for_testing(&self) {
        let mut execution_lock = self.execution_lock_for_reconfiguration().await;
        let epoch_store = self.epoch_store_for_testing().clone();
        let protocol_config = epoch_store.protocol_config().clone();
        // The current protocol config used in the epoch store may have been overridden and diverged from
        // the protocol config definitions. That override may have now been dropped when the initial guard was dropped.
        // We reapply the override before creating the new epoch store, to make sure that
        // the new epoch store has the same protocol config as the current one.
        // Since this is for testing only, we mostly like to keep the protocol config the same
        // across epochs.

        let new_epoch_store = epoch_store.new_at_next_epoch_for_testing();
        let new_epoch = new_epoch_store.epoch();
        self.transaction_manager.reconfigure(new_epoch);
        self.epoch_store.store(new_epoch_store);
        epoch_store.epoch_terminated().await;
        *execution_lock = new_epoch;
    }

    /// # Load the current epoch store
    ///
    /// Provides access to the current epoch store, which contains epoch-specific state and configuration.
    ///
    /// ## Returns
    /// A guard containing a reference to the current epoch store
    ///
    /// ## Thread Safety
    /// This method is designed for concurrent access with careful usage patterns:
    /// - Should be called only once per task to ensure consistent epoch view
    /// - Uses ArcSwap for atomic updates during reconfiguration
    ///
    /// ## Usage Guidelines
    /// - Call this method only when there is no other way to obtain the epoch store
    /// - Minimize call sites to reduce the risk of inconsistent epoch state
    /// - Each call must be carefully audited to ensure proper usage
    pub fn load_epoch_store_one_call_per_task(&self) -> Guard<Arc<AuthorityPerEpochStore>> {
        self.epoch_store.load()
    }

    /// # Load epoch store for testing
    ///
    /// Provides access to the current epoch store for testing purposes.
    ///
    /// ## Returns
    /// A guard containing a reference to the current epoch store
    ///
    /// ## Usage
    /// This method should only be used in test code, not in production paths.
    /// It's a convenience wrapper around load_epoch_store_one_call_per_task.
    pub fn epoch_store_for_testing(&self) -> Guard<Arc<AuthorityPerEpochStore>> {
        self.load_epoch_store_one_call_per_task()
    }

    /// # Clone committee for testing
    ///
    /// Creates a clone of the current committee for testing purposes.
    ///
    /// ## Returns
    /// A clone of the current committee
    ///
    /// ## Usage
    /// This method should only be used in test code, not in production paths.
    /// It provides a convenient way to access the current committee configuration.
    pub fn clone_committee_for_testing(&self) -> Committee {
        Committee::clone(self.epoch_store_for_testing().committee())
    }

    /// # Enqueue certificates for execution
    ///
    /// Adds certificates to the transaction manager for ordered execution.
    ///
    /// ## Arguments
    /// * `certs` - A vector of verified certificates to execute
    /// * `epoch_store` - The current epoch store
    ///
    /// ## Purpose
    /// Submits transactions to the transaction manager, which will
    /// handle dependency tracking and execution ordering.
    ///
    /// ## Behavior
    /// The transaction manager will:
    /// 1. Track dependencies between transactions
    /// 2. Execute transactions when their dependencies are satisfied
    /// 3. Handle retries and error conditions
    pub fn enqueue_certificates_for_execution(
        &self,
        certs: Vec<VerifiedCertificate>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) {
        self.transaction_manager
            .enqueue_certificates(certs, epoch_store, None)
    }

    /// # Enqueue transactions with expected effects digest
    ///
    /// Adds transactions to the transaction manager with their expected effects digests.
    ///
    /// ## Arguments
    /// * `certs` - A vector of tuples containing verified executable transactions and their expected effects digests
    /// * `epoch_store` - The current epoch store
    ///
    /// ## Purpose
    /// Used when executing transactions where the effects are already known,
    /// such as during state synchronization or when re-executing transactions
    /// after a node restart.
    ///
    /// ## Behavior
    /// The transaction manager will verify that the actual effects match the
    /// expected effects digest after execution.
    pub(crate) fn enqueue_with_expected_effects_digest(
        &self,
        certs: Vec<(VerifiedExecutableTransaction, TransactionEffectsDigest)>,
        epoch_store: &AuthorityPerEpochStore,
    ) {
        self.transaction_manager
            .enqueue_with_expected_effects_digest(certs, epoch_store)
    }

    /// # Get transaction manager
    ///
    /// Returns a reference to the transaction manager.
    ///
    /// ## Returns
    /// A reference to the transaction manager
    ///
    /// ## Purpose
    /// Provides access to the transaction manager for coordinating
    /// transaction execution and dependency tracking.
    pub fn transaction_manager(&self) -> &Arc<TransactionManager> {
        &self.transaction_manager
    }

    /// # Clone committee store
    ///
    /// Returns a cloned reference to the committee store.
    ///
    /// ## Returns
    /// An Arc-wrapped reference to the committee store
    ///
    /// ## Purpose
    /// Provides access to committee information across epochs,
    /// which is useful for validators that need to verify transactions
    /// from different epochs.
    pub fn clone_committee_store(&self) -> Arc<CommitteeStore> {
        self.committee_store.clone()
    }

    /// # Get system state object for testing
    ///
    /// Retrieves the current system state object.
    ///
    /// ## Returns
    /// The current system state object
    ///
    /// ## Purpose
    /// Used in testing to access the system state, which contains
    /// information about the current epoch, validator set, and other
    /// system-wide parameters.
    ///
    /// ## Usage
    /// This method should only be used in test code, not in production paths.
    pub fn get_system_state_object_for_testing(&self) -> SystemState {
        self.get_object_cache_reader()
            .get_system_state_object()
            .unwrap()
    }

    #[instrument(level = "trace", skip_all)]
    pub async fn get_object(&self, object_id: &ObjectID) -> Option<Object> {
        self.get_object_store().get_object(object_id).unwrap()
    }

    /// Get the TransactionEnvelope that currently locks the given object, if any.
    /// Since object locks are only valid for one epoch, we also need the epoch_id in the query.
    /// Returns ObjectNotFound if no lock records for the given object can be found.
    /// Returns ObjectVersionUnavailableForConsumption if the object record is at a different version.
    /// Returns Some(VerifiedEnvelope) if the given ObjectRef is locked by a certain transaction.
    /// Returns None if a lock record is initialized for the given ObjectRef but not yet locked by any transaction,
    ///     or cannot find the transaction in transaction table, because of data race etc.
    #[instrument(level = "trace", skip_all)]
    pub async fn get_transaction_lock(
        &self,
        object_ref: &ObjectRef,
        epoch_store: &AuthorityPerEpochStore,
    ) -> SomaResult<Option<VerifiedSignedTransaction>> {
        let lock_info = self
            .get_object_cache_reader()
            .get_lock(*object_ref, epoch_store)?;
        let lock_info = match lock_info {
            ObjectLockStatus::LockedAtDifferentVersion { locked_ref } => {
                return Err(SomaError::ObjectVersionUnavailableForConsumption {
                    provided_obj_ref: *object_ref,
                    current_version: locked_ref.1,
                }
                .into());
            }
            ObjectLockStatus::Initialized => {
                return Ok(None);
            }
            ObjectLockStatus::LockedToTx { locked_by_tx } => locked_by_tx,
        };

        epoch_store.get_signed_transaction(&lock_info)
    }

    pub async fn get_object_or_tombstone(&self, object_id: ObjectID) -> Option<ObjectRef> {
        self.get_object_cache_reader()
            .get_latest_object_ref_or_tombstone(object_id)
            .unwrap()
    }

    pub async fn insert_genesis_object(&self, object: Object) {
        self.get_reconfig_api().insert_genesis_object(object);
    }

    pub async fn insert_genesis_objects(&self, objects: &[Object]) {
        futures::future::join_all(
            objects
                .iter()
                .map(|o| self.insert_genesis_object(o.clone())),
        )
        .await;
    }

    /// Attempts to acquire execution lock for an executable transaction.
    /// Returns the lock if the transaction is matching current executed epoch
    /// Returns None otherwise
    pub async fn execution_lock_for_executable_transaction(
        &self,
        transaction: &VerifiedExecutableTransaction,
    ) -> SomaResult<ExecutionLockReadGuard> {
        let lock = self.execution_lock.read().await;

        if *lock == transaction.auth_sig().epoch() {
            Ok(lock)
        } else {
            Err(SomaError::WrongEpoch {
                expected_epoch: *lock,
                actual_epoch: transaction.auth_sig().epoch(),
            })
        }
    }

    /// Acquires the execution lock for the duration of a transaction signing request.
    /// This prevents reconfiguration from starting until we are finished handling the signing request.
    /// Otherwise, in-memory lock state could be cleared (by `ObjectLocks::clear_cached_locks`)
    /// while we are attempting to acquire locks for the transaction.
    pub fn execution_lock_for_signing(&self) -> SomaResult<ExecutionLockReadGuard> {
        self.execution_lock
            .try_read()
            .map_err(|_| SomaError::ValidatorHaltedAtEpochEnd)
    }

    pub async fn execution_lock_for_reconfiguration(&self) -> ExecutionLockWriteGuard {
        self.execution_lock.write().await
    }

    /// # Check if node is a validator
    ///
    /// Determines whether this node is a validator in the current epoch.
    ///
    /// ## Arguments
    /// * `epoch_store` - The current epoch store
    ///
    /// ## Returns
    /// * `true` - If this node is a validator in the current epoch
    /// * `false` - If this node is not a validator (i.e., it's a fullnode)
    ///
    /// ## Purpose
    /// Used to determine the node's role and responsibilities in the network.
    /// Validators have additional responsibilities like signing transactions and
    /// participating in consensus.
    pub fn is_validator(&self, epoch_store: &AuthorityPerEpochStore) -> bool {
        epoch_store.committee().authority_exists(&self.name)
    }

    /// # Check if node is a fullnode
    ///
    /// Determines whether this node is a fullnode (non-validator) in the current epoch.
    ///
    /// ## Arguments
    /// * `epoch_store` - The current epoch store
    ///
    /// ## Returns
    /// * `true` - If this node is a fullnode
    /// * `false` - If this node is a validator
    ///
    /// ## Purpose
    /// Used to determine the node's role and responsibilities in the network.
    /// Fullnodes have different behavior than validators, such as not signing
    /// transaction effects and not participating in consensus.
    pub fn is_fullnode(&self, epoch_store: &AuthorityPerEpochStore) -> bool {
        !self.is_validator(epoch_store)
    }

    /// # Reopen epoch database
    ///
    /// Creates a new epoch store for the next epoch during reconfiguration.
    ///
    /// ## Arguments
    /// * `cur_epoch_store` - The current epoch store
    /// * `new_committee` - The committee for the new epoch
    /// * `epoch_start_configuration` - Configuration for the new epoch
    /// * `epoch_last_commit` - The last commit index of the current epoch
    ///
    /// ## Returns
    /// A new epoch store for the next epoch
    ///
    /// ## Behavior
    /// 1. Creates a new epoch store with the new committee and configuration
    /// 2. Atomically swaps the current epoch store with the new one
    /// 3. Signals the current epoch store that it has been terminated
    ///
    /// ## Purpose
    /// Used during epoch transitions to set up the state for the new epoch
    /// while ensuring a clean handoff from the previous epoch.
    #[instrument(level = "error", skip_all)]
    async fn reopen_epoch_db(
        &self,
        cur_epoch_store: &AuthorityPerEpochStore,
        new_committee: Committee,
        epoch_start_configuration: EpochStartConfiguration,
        epoch_last_commit: CommitIndex,
    ) -> SomaResult<Arc<AuthorityPerEpochStore>> {
        let new_epoch = new_committee.epoch;
        info!(new_epoch = ?new_epoch, "re-opening AuthorityEpochTables for new epoch");
        assert_eq!(
            epoch_start_configuration.epoch_start_state().epoch(),
            new_committee.epoch
        );

        let new_epoch_store = cur_epoch_store.new_at_next_epoch(
            self.name,
            new_committee,
            epoch_start_configuration,
            epoch_last_commit,
        );
        self.epoch_store.store(new_epoch_store.clone());
        cur_epoch_store.epoch_terminated().await;
        Ok(new_epoch_store)
    }

    /// # Create and execute epoch advancement transaction
    ///
    /// Creates and executes a special system transaction to advance to the next epoch.
    ///
    /// ## Arguments
    /// * `epoch_store` - The current epoch store
    /// * `epoch_start_timestamp_ms` - The timestamp for the start of the new epoch
    ///
    /// ## Returns
    /// A tuple containing the new system state and transaction effects
    ///
    /// ## Behavior
    /// 1. Creates a special end-of-epoch transaction
    /// 2. Executes it directly without going through consensus
    /// 3. Updates the system state with the new epoch information
    /// 4. Writes the transaction outputs to storage
    ///
    /// ## Purpose
    /// Used during epoch transitions to update the system state object
    /// with the new epoch information. This is a critical part of the
    /// reconfiguration process.
    #[instrument(level = "error", skip_all)]
    pub async fn create_and_execute_advance_epoch_tx(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        epoch_start_timestamp_ms: CommitTimestamp,
    ) -> anyhow::Result<(SystemState, TransactionEffects)> {
        let next_epoch = epoch_store.epoch() + 1;

        let tx =
            VerifiedTransaction::new_change_epoch_transaction(next_epoch, epoch_start_timestamp_ms);

        let executable_tx =
            VerifiedExecutableTransaction::new_system(tx.clone(), epoch_store.epoch());

        let tx_digest = executable_tx.digest();

        info!(
            ?next_epoch,
            ?tx_digest,
            "Creating advance epoch transaction"
        );

        let tx_lock = epoch_store.acquire_tx_lock(tx_digest).await;

        // The tx could have been executed by state sync already - if so simply return an error.
        // The checkpoint builder will shortly be terminated by reconfiguration anyway.
        if self
            .get_transaction_cache_reader()
            .is_tx_already_executed(tx_digest)
            .expect("read cannot fail")
        {
            warn!("change epoch tx has already been executed via state sync");
            return Err(anyhow::anyhow!(
                "change epoch tx has already been executed via state sync"
            ));
        }

        let execution_guard = self
            .execution_lock_for_executable_transaction(&executable_tx)
            .await?;

        // We must manually assign the shared object versions to the transaction before executing it.
        // This is because we do not sequence end-of-epoch transactions through consensus.
        epoch_store.assign_shared_object_versions_idempotent(
            self.get_object_cache_reader().as_ref(),
            &[executable_tx.clone()],
        )?;

        let input_objects =
            self.read_objects_for_execution(&tx_lock, &executable_tx, epoch_store)?;

        let (inner, effects, _execution_error_opt) =
            self.prepare_certificate(&execution_guard, &executable_tx, input_objects, epoch_store)?;

        let system_obj = inner.written.get(&SYSTEM_STATE_OBJECT_ID).unwrap();
        let system_state =
            bcs::from_bytes::<SystemState>(system_obj.as_inner().data.contents()).unwrap();

        self.get_cache_writer()
            .write_transaction_outputs(
                epoch_store.epoch(),
                TransactionOutputs::build_transaction_outputs(tx, effects.clone(), inner).into(),
            )
            .await?;

        //  TODO: instead of writing the advanced epoch state object to cache, write to long term storage self.get_state_sync_store()
        // .insert_transaction_and_effects(&tx, &effects)
        // .map_err(|err| {
        //     let err: anyhow::Error = err.into();
        //     err
        // })?;

        // The change epoch transaction cannot fail to execute.
        assert!(effects.status().is_ok());
        Ok((system_state, effects))
    }

    /// Make a status response for a transaction
    #[instrument(level = "trace", skip_all)]
    pub fn get_transaction_status(
        &self,
        transaction_digest: &TransactionDigest,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<Option<(SenderSignedData, TransactionStatus)>> {
        // TODO: In the case of read path, we should not have to re-sign the effects.
        if let Some(effects) =
            self.get_signed_effects_and_maybe_resign(transaction_digest, epoch_store)?
        {
            if let Some(transaction) = self
                .get_transaction_cache_reader()
                .get_transaction_block(transaction_digest)?
            {
                let cert_sig = epoch_store.get_transaction_cert_sig(transaction_digest)?;

                return Ok(Some((
                    (*transaction).clone().into_message(),
                    TransactionStatus::Executed(cert_sig, effects.into_inner()),
                )));
            } else {
                // The read of effects and read of transaction are not atomic. It's possible that we reverted
                // the transaction (during epoch change) in between the above two reads, and we end up
                // having effects but not transaction. In this case, we just fall through.
                debug!(tx_digest=?transaction_digest, "Signed effects exist but no transaction found");
            }
        }
        Ok(None)
    }

    /// Get the signed effects of the given transaction. If the effects was signed in a previous
    /// epoch, re-sign it so that the caller is able to form a cert of the effects in the current
    /// epoch.
    #[instrument(level = "trace", skip_all)]
    pub fn get_signed_effects_and_maybe_resign(
        &self,
        transaction_digest: &TransactionDigest,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<Option<VerifiedSignedTransactionEffects>> {
        let effects = self
            .get_transaction_cache_reader()
            .get_executed_effects(transaction_digest)?;
        match effects {
            Some(effects) => Ok(Some(self.sign_effects(effects, epoch_store)?)),
            None => Ok(None),
        }
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) fn sign_effects(
        &self,
        effects: TransactionEffects,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<VerifiedSignedTransactionEffects> {
        let tx_digest = *effects.transaction_digest();
        let signed_effects = match epoch_store.get_effects_signature(&tx_digest)? {
            Some(sig) if sig.epoch == epoch_store.epoch() => {
                SignedTransactionEffects::new_from_data_and_sig(effects, sig)
            }
            _ => {
                // If the transaction was executed in previous epochs, the validator will
                // re-sign the effects with new current epoch so that a client is always able to
                // obtain an effects certificate at the current epoch.
                //
                // Why is this necessary? Consider the following case:
                // - assume there are 4 validators
                // - Quorum driver gets 2 signed effects before reconfig halt
                // - The tx makes it into final checkpoint.
                // - 2 validators go away and are replaced in the new epoch.
                // - The new epoch begins.
                // - The quorum driver cannot complete the partial effects cert from the previous epoch,
                //   because it may not be able to reach either of the 2 former validators.
                // - But, if the 2 validators that stayed are willing to re-sign the effects in the new
                //   epoch, the QD can make a new effects cert and return it to the client.
                //
                // This is a considered a short-term workaround. Eventually, Quorum Driver should be able
                // to return either an effects certificate, -or- a proof of inclusion in a checkpoint. In
                // the case above, the Quorum Driver would return a proof of inclusion in the final
                // checkpoint, and this code would no longer be necessary.
                debug!(
                    ?tx_digest,
                    epoch=?epoch_store.epoch(),
                    "Re-signing the effects with the current epoch"
                );

                let sig = AuthoritySignInfo::new(
                    epoch_store.epoch(),
                    &effects,
                    Intent::soma_transaction(),
                    self.name,
                    &*self.secret,
                );

                let effects = SignedTransactionEffects::new_from_data_and_sig(effects, sig.clone());

                epoch_store.insert_effects_digest_and_signature(
                    &tx_digest,
                    effects.digest(),
                    &sig,
                )?;

                effects
            }
        };

        Ok(VerifiedSignedTransactionEffects::new_unchecked(
            signed_effects,
        ))
    }

    pub async fn get_root_state_digest(
        &self,
        commit_index: CommitIndex,
        effects: Vec<TransactionEffects>,
    ) -> SomaResult<ECMHLiveObjectSetDigest> {
        let epoch_store = self.epoch_store.load();
        let acc =
            self.accumulator
                .accumulate_commit(effects.clone(), commit_index, &epoch_store)?;
        self.accumulator
            .accumulate_running_root(&epoch_store, commit_index, Some(acc))
            .await?;
        self.accumulator
            .digest_epoch(epoch_store.clone(), commit_index)
            .await
    }

    /// NOTE: this function is only to be used for fuzzing and testing. Never use in prod
    pub async fn insert_objects_unsafe_for_testing_only(&self, objects: &[Object]) -> SomaResult {
        self.get_reconfig_api().bulk_insert_genesis_objects(objects);
        self.get_reconfig_api()
            .clear_state_end_of_epoch(&self.execution_lock_for_reconfiguration().await);
        Ok(())
    }

    #[cfg(test)]
    pub async fn get_latest_object_lock_for_testing(
        &self,
        object_id: ObjectID,
    ) -> SomaResult<Option<VerifiedSignedTransaction>> {
        let epoch_store = self.load_epoch_store_one_call_per_task();
        let (_, seq, _) = self
            .get_object_or_tombstone(object_id)
            .await
            .ok_or_else(|| SomaError::ObjectNotFound {
                object_id,
                version: None,
            })?;
        let object = self
            .get_object_store()
            .get_object_by_key(&object_id, seq)?
            .ok_or_else(|| SomaError::ObjectNotFound {
                object_id,
                version: Some(seq),
            })?;
        let lock = if !object.is_address_owned() {
            // Only address owned objects have locks.
            None
        } else {
            self.get_transaction_lock(&object.compute_object_reference(), &epoch_store)
                .await?
        };

        Ok(lock)
    }
}

pub type ExecutionLockReadGuard<'a> = RwLockReadGuard<'a, EpochId>;
pub type ExecutionLockWriteGuard<'a> = RwLockWriteGuard<'a, EpochId>;
