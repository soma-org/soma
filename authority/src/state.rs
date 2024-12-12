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
use types::state_sync::CommitTimestamp;
use types::storage::object_store::ObjectStore;
use types::system_state::{EpochStartSystemStateTrait, SystemState};
use types::transaction::{EndOfEpochTransactionKind, SenderSignedData};
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
    ExecutionCacheCommit, ExecutionCacheTraitPointers, ExecutionCacheWrite, ObjectCacheRead,
    TransactionCacheRead,
};
use crate::epoch_store::CertTxGuard;
use crate::execution_driver::execution_process;
use crate::start_epoch::EpochStartConfigTrait;
use crate::state_accumulator::StateAccumulator;
use crate::{
    client::NetworkAuthorityClient, committee_store::CommitteeStore,
    epoch_store::AuthorityPerEpochStore, start_epoch::EpochStartConfiguration,
    tx_manager::TransactionManager,
};

/// a Trait object for `Signer` that is:
/// - Pin, i.e. confined to one place in memory (we don't want to copy private keys).
/// - Sync, i.e. can be safely shared between threads.
///
/// Typically instantiated with Box::pin(keypair) where keypair is a `KeyPair`
///
pub type StableSyncAuthoritySigner = Pin<Arc<dyn Signer<AuthoritySignature> + Send + Sync>>;

pub struct AuthorityState {
    // Fixed size, static, identity of the authority
    /// The name of this authority.
    pub name: AuthorityName,
    /// The signature key of the authority.
    pub secret: StableSyncAuthoritySigner,

    epoch_store: ArcSwap<AuthorityPerEpochStore>,

    /// This lock denotes current 'execution epoch'.
    /// Execution acquires read lock, checks certificate epoch and holds it until all writes are complete.
    /// Reconfiguration acquires write lock, changes the epoch and revert all transactions
    /// from previous epoch that are executed but did not make into checkpoint.
    execution_lock: RwLock<EpochId>,

    committee_store: Arc<CommitteeStore>,

    // Manages pending certificates and their missing input objects.
    transaction_manager: Arc<TransactionManager>,
    pub config: NodeConfig,
    // pub validator_tx_finalizer: Option<Arc<ValidatorTxFinalizer<NetworkAuthorityClient>>>,
    /// Shuts down the execution task. Used only in testing.
    #[allow(unused)]
    tx_execution_shutdown: Mutex<Option<oneshot::Sender<()>>>,

    /// The database
    execution_cache_trait_pointers: ExecutionCacheTraitPointers,

    // The state accumulator
    accumulator: Arc<StateAccumulator>,
}

impl AuthorityState {
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
            &epoch_store,
            tx_ready_certificates,
            execution_cache_trait_pointers
                .transaction_cache_reader
                .clone(),
        ));
        let (tx_execution_shutdown, rx_execution_shutdown) = oneshot::channel();

        let epoch = epoch_store.epoch();
        let state = Arc::new(AuthorityState {
            name,
            secret,
            execution_lock: RwLock::new(epoch),
            epoch_store: ArcSwap::new(epoch_store.clone()),
            tx_execution_shutdown: Mutex::new(Some(tx_execution_shutdown)),
            committee_store,
            transaction_manager,
            config,
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

    pub fn get_transaction_cache_reader(&self) -> &Arc<dyn TransactionCacheRead> {
        &self.execution_cache_trait_pointers.transaction_cache_reader
    }

    pub fn get_cache_writer(&self) -> &Arc<dyn ExecutionCacheWrite> {
        &self.execution_cache_trait_pointers.cache_writer
    }

    pub fn get_object_store(&self) -> &Arc<dyn ObjectStore + Send + Sync> {
        &self.execution_cache_trait_pointers.object_store
    }

    pub fn get_object_cache_reader(&self) -> &Arc<dyn ObjectCacheRead> {
        &self.execution_cache_trait_pointers.object_cache_reader
    }

    pub fn get_accumulator_store(&self) -> &Arc<dyn AccumulatorStore> {
        &self.execution_cache_trait_pointers.accumulator_store
    }

    pub fn get_cache_commit(&self) -> &Arc<dyn ExecutionCacheCommit> {
        &self.execution_cache_trait_pointers.cache_commit
    }

    /// This is a private method and should be kept that way. It doesn't check whether
    /// the provided transaction is a system transaction, and hence can only be called internally.
    #[instrument(level = "trace", skip_all)]
    async fn handle_transaction_impl(
        &self,
        transaction: VerifiedTransaction,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<VerifiedSignedTransaction> {
        let signed_transaction = VerifiedSignedTransaction::new(
            epoch_store.epoch(),
            transaction,
            self.name,
            &*self.secret,
        );

        // Check and write locks, to signed transaction, into the database
        // The call to self.set_transaction_lock checks the lock is not conflicting,
        // and returns ConflictingTransaction error in case there is a lock on a different
        // existing transaction.
        // self.get_cache_writer()
        //     .acquire_transaction_locks(epoch_store, &owned_objects, signed_transaction.clone())
        //     .await?;

        Ok(signed_transaction)
    }

    /// Initiate a new transaction.
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

    /// Executes a certificate for its effects.
    #[instrument(level = "trace", skip_all)]
    pub async fn execute_certificate(
        &self,
        certificate: &VerifiedCertificate,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<TransactionEffects> {
        debug!("execute_certificate");

        self.enqueue_certificates_for_execution(vec![certificate.clone()], epoch_store);
        self.notify_read_effects(certificate).await
    }

    pub async fn notify_read_effects(
        &self,
        certificate: &VerifiedCertificate,
    ) -> SomaResult<TransactionEffects> {
        self.get_transaction_cache_reader()
            .notify_read_executed_effects(&[*certificate.digest()])
            .await
            .map(|mut r| r.pop().expect("must return correct number of effects"))
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
    /// Should only be called within sui-core.
    #[instrument(level = "trace", skip_all)]
    pub async fn try_execute_immediately(
        &self,
        certificate: &VerifiedExecutableTransaction,
        mut expected_effects_digest: Option<TransactionEffectsDigest>,
        commit: Option<CommitIndex>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<(TransactionEffects, Option<ExecutionError>)> {
        debug!("execute_certificate_internal");

        let tx_digest = certificate.digest();

        if expected_effects_digest.is_none() {
            // We could be re-executing a previously executed but uncommitted transaction, perhaps after
            // restarting with a new binary. In this situation, if we have published an effects signature,
            // we must be sure not to equivocate.
            // TODO: read from cache instead of DB
            expected_effects_digest = epoch_store.get_signed_effects_digest(tx_digest)?;
        }

        // This acquires a lock on the tx digest to prevent multiple concurrent executions of the
        // same tx. While we don't need this for safety (tx sequencing is ultimately atomic), it is
        // very common to receive the same tx multiple times simultaneously due to gossip, so we
        // may as well hold the lock and save the cpu time for other requests.
        let tx_guard = epoch_store.acquire_tx_guard(certificate).await?;

        self.process_certificate(
            tx_guard,
            certificate,
            expected_effects_digest,
            commit,
            epoch_store,
        )
        .await
        .tap_err(|e| info!(?tx_digest, "process_certificate failed: {e}"))
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) async fn process_certificate(
        &self,
        tx_guard: CertTxGuard,
        certificate: &VerifiedExecutableTransaction,
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

            if let Some(commit) = commit {
                if !effects.all_changed_objects().is_empty() {
                    let commit_acc = self.accumulator.accumulate_commit(
                        vec![effects.clone()],
                        commit,
                        epoch_store,
                    )?;
                    self.accumulator
                        .accumulate_running_root(epoch_store, commit, Some(commit_acc))
                        .await?;
                }
            }

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
        let (written_objects, effects, execution_error_opt) =
            match self.prepare_certificate(&execution_guard, certificate, epoch_store) {
                Err(e) => {
                    info!(name = ?self.name, ?digest, "Error preparing transaction: {e}");
                    tx_guard.release();
                    return Err(e);
                }
                Ok(res) => res,
            };

        self.commit_certificate(
            certificate,
            &written_objects,
            &effects,
            tx_guard,
            execution_guard,
            epoch_store,
        )
        .await?;

        if let Some(commit) = commit {
            if !effects.all_changed_objects().is_empty() {
                let commit_acc = self.accumulator.accumulate_commit(
                    vec![effects.clone()],
                    commit,
                    epoch_store,
                )?;
                self.accumulator
                    .accumulate_running_root(epoch_store, commit, Some(commit_acc))
                    .await?;
            }
        }

        Ok((effects, execution_error_opt))
    }

    pub fn is_tx_already_executed(&self, digest: &TransactionDigest) -> SomaResult<bool> {
        self.get_transaction_cache_reader()
            .is_tx_already_executed(digest)
    }

    #[instrument(level = "trace", skip_all)]
    async fn commit_certificate(
        &self,
        certificate: &VerifiedExecutableTransaction,
        written_objects: &WrittenObjects,
        effects: &TransactionEffects,
        tx_guard: CertTxGuard,
        _execution_guard: ExecutionLockReadGuard<'_>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult {
        let tx_key = certificate.key();
        let tx_digest = certificate.digest();

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
            written_objects.clone(),
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
            .notify_commit(tx_digest, epoch_store);

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
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<(WrittenObjects, TransactionEffects, Option<ExecutionError>)> {
        let prepare_certificate_start_time = tokio::time::Instant::now();

        // TODO: We need to move this to a more appropriate place to avoid redundant checks.
        let tx_data = certificate.data().transaction_data();

        let tx_digest = *certificate.digest();
        let protocol_config = epoch_store.protocol_config();
        let transaction_data = &certificate.data().intent_message().value;
        let (kind, signer) = transaction_data.execution_parts();

        let (written_objects, effects, execution_error_opt) = epoch_store.execute_transaction(
            self.get_object_store().as_ref(),
            tx_digest,
            kind,
            signer,
        );

        Ok((written_objects, effects, execution_error_opt))
    }

    #[instrument(level = "error", skip_all)]
    pub async fn reconfigure(
        &self,
        cur_epoch_store: &AuthorityPerEpochStore,
        new_committee: Committee,
        epoch_start_configuration: EpochStartConfiguration,
        // expensive_safety_check_config: &ExpensiveSafetyCheckConfig,
    ) -> SomaResult<Arc<AuthorityPerEpochStore>> {
        let new_epoch = new_committee.epoch;
        self.committee_store
            .insert_new_committee(new_committee.clone())?;
        let mut execution_lock = self.execution_lock_for_reconfiguration().await;

        // TODO: check system consistency using accumulator after reverting uncommitted epoch transactions

        let new_epoch_store = self
            .reopen_epoch_db(cur_epoch_store, new_committee, epoch_start_configuration)
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

    /// Load the current epoch store. This can change during reconfiguration. To ensure that
    /// we never end up accessing different epoch stores in a single task, we need to make sure
    /// that this is called once per task. Each call needs to be carefully audited to ensure it is
    /// the case. This also means we should minimize the number of call-sites. Only call it when
    /// there is no way to obtain it from somewhere else.
    pub fn load_epoch_store_one_call_per_task(&self) -> Guard<Arc<AuthorityPerEpochStore>> {
        self.epoch_store.load()
    }

    // Load the epoch store, should be used in tests only.
    pub fn epoch_store_for_testing(&self) -> Guard<Arc<AuthorityPerEpochStore>> {
        self.load_epoch_store_one_call_per_task()
    }

    pub fn clone_committee_for_testing(&self) -> Committee {
        Committee::clone(self.epoch_store_for_testing().committee())
    }

    /// Adds certificates to transaction manager for ordered execution.
    pub fn enqueue_certificates_for_execution(
        &self,
        certs: Vec<VerifiedCertificate>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) {
        self.transaction_manager
            .enqueue_certificates(certs, epoch_store, None)
    }

    pub(crate) fn enqueue_with_expected_effects_digest(
        &self,
        certs: Vec<(VerifiedExecutableTransaction, TransactionEffectsDigest)>,
        epoch_store: &AuthorityPerEpochStore,
    ) {
        self.transaction_manager
            .enqueue_with_expected_effects_digest(certs, epoch_store)
    }

    pub fn transaction_manager(&self) -> &Arc<TransactionManager> {
        &self.transaction_manager
    }

    pub fn clone_committee_store(&self) -> Arc<CommitteeStore> {
        self.committee_store.clone()
    }

    // This function is only used for testing.
    pub fn get_system_state_object_for_testing(&self) -> SystemState {
        self.get_object_cache_reader()
            .get_system_state_object()
            .unwrap()
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

    pub async fn execution_lock_for_reconfiguration(&self) -> ExecutionLockWriteGuard {
        self.execution_lock.write().await
    }

    pub fn is_validator(&self, epoch_store: &AuthorityPerEpochStore) -> bool {
        epoch_store.committee().authority_exists(&self.name)
    }

    pub fn is_fullnode(&self, epoch_store: &AuthorityPerEpochStore) -> bool {
        !self.is_validator(epoch_store)
    }

    #[instrument(level = "error", skip_all)]
    async fn reopen_epoch_db(
        &self,
        cur_epoch_store: &AuthorityPerEpochStore,
        new_committee: Committee,
        epoch_start_configuration: EpochStartConfiguration,
    ) -> SomaResult<Arc<AuthorityPerEpochStore>> {
        let new_epoch = new_committee.epoch;
        info!(new_epoch = ?new_epoch, "re-opening AuthorityEpochTables for new epoch");
        assert_eq!(
            epoch_start_configuration.epoch_start_state().epoch(),
            new_committee.epoch
        );

        let new_epoch_store =
            cur_epoch_store.new_at_next_epoch(self.name, new_committee, epoch_start_configuration);
        self.epoch_store.store(new_epoch_store.clone());
        cur_epoch_store.epoch_terminated().await;
        Ok(new_epoch_store)
    }

    #[instrument(level = "error", skip_all)]
    pub async fn create_and_execute_advance_epoch_tx(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        // epoch_start_timestamp_ms: CommitTimestamp,
    ) -> anyhow::Result<(SystemState, TransactionEffects)> {
        let next_epoch = epoch_store.epoch() + 1;

        let tx = VerifiedTransaction::new_end_of_epoch_transaction(
            EndOfEpochTransactionKind::new_change_epoch(next_epoch, 0), // TODO: change this to the correct value
        );

        let executable_tx =
            VerifiedExecutableTransaction::new_system(tx.clone(), epoch_store.epoch());

        let tx_digest = executable_tx.digest();

        info!(
            ?next_epoch,
            ?tx_digest,
            "Creating advance epoch transaction"
        );

        let _tx_lock = epoch_store.acquire_tx_lock(tx_digest).await;

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

        let (written_objects, effects, _execution_error_opt) =
            self.prepare_certificate(&execution_guard, &executable_tx, epoch_store)?;

        let system_obj = written_objects.get(&SYSTEM_STATE_OBJECT_ID).unwrap();
        let system_state =
            bcs::from_bytes::<SystemState>(system_obj.as_inner().data.contents()).unwrap();

        self.get_cache_writer()
            .write_transaction_outputs(
                epoch_store.epoch(),
                TransactionOutputs::build_transaction_outputs(tx, effects.clone(), written_objects)
                    .into(),
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
}

pub type ExecutionLockReadGuard<'a> = RwLockReadGuard<'a, EpochId>;
pub type ExecutionLockWriteGuard<'a> = RwLockWriteGuard<'a, EpochId>;
