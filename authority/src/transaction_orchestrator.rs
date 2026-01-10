/*
Transaction Orchestrator is a Node component that utilizes Transaction Driver to
submit transactions to validators for finality, and proactively executes
finalized transactions locally, when possible.
*/

use std::net::SocketAddr;
use std::ops::Deref;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::future::{select, Either, Future};
use futures::stream::{FuturesUnordered, StreamExt};
use futures::FutureExt;
use types::entropy::SimpleVDF;
use types::envelope::Message as _;
use types::finality::FinalityProof;
use types::object::ObjectRef;
use types::shard::{Input, InputV1, ShardAuthToken, ShardEntropy};
use types::shard_crypto::digest::Digest;
use types::storage::write_path_pending_tx_log::WritePathPendingTransactionLog;
use types::system_state::shard::{Shard, Target};
use utils::notify_read::NotifyRead;

use protocol_config::Chain;
use rand::Rng;
use types::config::node_config::NodeConfig;

use tokio::sync::broadcast::error::RecvError;
use tokio::sync::broadcast::{self, Receiver};
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout, Instant};
use tracing::{debug, error, error_span, info, instrument, warn, Instrument};
use types::digests::TransactionDigest;
use types::effects::TransactionEffectsAPI;
use types::error::{SomaError, SomaResult};
use types::messages_grpc::{SubmitTxRequest, TxType};
use types::quorum_driver::{
    EffectsFinalityInfo, ExecuteTransactionRequest, ExecuteTransactionRequestType,
    ExecuteTransactionResponse, FinalizedEffects, InitiateShardWorkRequest,
    InitiateShardWorkResponse, IsTransactionExecutedLocally, QuorumDriverEffectsQueueResult,
    QuorumDriverError, QuorumDriverResult,
};
use types::system_state::{SystemState, SystemStateTrait as _};
use types::transaction::{Transaction, TransactionData, TransactionKind, VerifiedTransaction};
use types::transaction_executor::{SimulateTransactionResult, TransactionChecks};

use crate::authority::AuthorityState;
use crate::authority_aggregator::AuthorityAggregator;
use crate::authority_client::{AuthorityAPI, NetworkAuthorityClient};
use crate::authority_per_epoch_store::AuthorityPerEpochStore;
use crate::encoder_client::EncoderClientService;
use crate::transaction_driver::reconfig_observer::{OnsiteReconfigObserver, ReconfigObserver};
use crate::transaction_driver::{
    choose_transaction_driver_percentage, QuorumTransactionResponse, SubmitTransactionOptions,
    TransactionDriver, TransactionDriverError,
};

// How long to wait for local execution (including parents) before a timeout
// is returned to client.
const LOCAL_EXECUTION_TIMEOUT: Duration = Duration::from_secs(10);

// Timeout for waiting for finality for each transaction.
const WAIT_FOR_FINALITY_TIMEOUT: Duration = Duration::from_secs(90);

pub type QuorumTransactionEffectsResult =
    Result<(Transaction, QuorumTransactionResponse), (TransactionDigest, QuorumDriverError)>;
pub struct TransactionOrchestrator<A: Clone> {
    validator_state: Arc<AuthorityState>,
    pending_tx_log: Arc<WritePathPendingTransactionLog>,
    notifier: Arc<NotifyRead<TransactionDigest, QuorumDriverResult>>,
    transaction_driver: Arc<TransactionDriver<A>>,
    td_allowed_submission_list: Vec<String>,
    td_blocked_submission_list: Vec<String>,
    enable_early_validation: bool,
    encoder_client: Option<Arc<EncoderClientService>>,
    vdf: Arc<SimpleVDF>,
}

impl TransactionOrchestrator<NetworkAuthorityClient> {
    pub fn new_with_auth_aggregator(
        validators: Arc<AuthorityAggregator<NetworkAuthorityClient>>,
        validator_state: Arc<AuthorityState>,
        reconfig_channel: Receiver<SystemState>,
        parent_path: &Path,
        node_config: &NodeConfig,
        encoder_client: Option<Arc<EncoderClientService>>,
    ) -> Self {
        let observer = OnsiteReconfigObserver::new(
            reconfig_channel,
            validator_state.get_object_cache_reader().clone(),
            validator_state.clone_committee_store(),
        );
        TransactionOrchestrator::new(
            validators,
            validator_state,
            parent_path,
            observer,
            node_config,
            encoder_client,
        )
    }
}

impl<A> TransactionOrchestrator<A>
where
    A: AuthorityAPI + Send + Sync + 'static + Clone,
    OnsiteReconfigObserver: ReconfigObserver<A>,
{
    pub fn new(
        validators: Arc<AuthorityAggregator<A>>,
        validator_state: Arc<AuthorityState>,
        parent_path: &Path,

        reconfig_observer: OnsiteReconfigObserver,
        node_config: &NodeConfig,
        encoder_client: Option<Arc<EncoderClientService>>,
    ) -> Self {
        let notifier = Arc::new(NotifyRead::new());
        let reconfig_observer = Arc::new(reconfig_observer);
        let epoch_store = validator_state.load_epoch_store_one_call_per_task();

        let transaction_driver = TransactionDriver::new(
            validators.clone(),
            reconfig_observer.clone(),
            Some(node_config),
        );

        let td_allowed_submission_list = node_config
            .transaction_driver_config
            .as_ref()
            .map(|config| config.allowed_submission_validators.clone())
            .unwrap_or_default();

        let td_blocked_submission_list = node_config
            .transaction_driver_config
            .as_ref()
            .map(|config| config.blocked_submission_validators.clone())
            .unwrap_or_default();

        if !td_allowed_submission_list.is_empty() && !td_blocked_submission_list.is_empty() {
            panic!(
                "Both allowed and blocked submission lists are set, this is not allowed, {:?} {:?}",
                td_allowed_submission_list, td_blocked_submission_list
            );
        }

        let pending_tx_log = Arc::new(WritePathPendingTransactionLog::new(
            parent_path.join("fullnode_pending_transactions"),
        ));
        Self::start_task_to_recover_txes_in_log(pending_tx_log.clone(), transaction_driver.clone());

        let enable_early_validation = node_config
            .transaction_driver_config
            .as_ref()
            .map(|config| config.enable_early_validation)
            .unwrap_or(true);

        Self {
            validator_state,
            pending_tx_log,
            notifier,
            transaction_driver,
            td_allowed_submission_list,
            td_blocked_submission_list,
            enable_early_validation,
            encoder_client,
            vdf: Arc::new(SimpleVDF::new(
                epoch_store.protocol_config().vdf_iterations(),
            )),
        }
    }
}

impl<A> TransactionOrchestrator<A>
where
    A: AuthorityAPI + Send + Sync + 'static + Clone,
{
    #[instrument(name = "tx_orchestrator_execute_transaction", level = "debug", skip_all,
    fields(
        tx_digest = ?request.transaction.digest(),
        tx_type = ?request_type,
    ))]
    pub async fn execute_transaction_block(
        &self,
        request: ExecuteTransactionRequest,
        request_type: ExecuteTransactionRequestType,
        client_addr: Option<SocketAddr>,
    ) -> Result<(ExecuteTransactionResponse, IsTransactionExecutedLocally), QuorumDriverError> {
        let timer = Instant::now();
        let tx_type = if request.transaction.is_consensus_tx() {
            TxType::SharedObject
        } else {
            TxType::SingleWriter
        };
        let tx_digest = *request.transaction.digest();

        let (response, mut executed_locally) = self
            .execute_transaction_with_effects_waiting(request, client_addr)
            .await?;

        if !executed_locally {
            executed_locally = if matches!(
                request_type,
                ExecuteTransactionRequestType::WaitForLocalExecution
            ) {
                let executed_locally = Self::wait_for_finalized_tx_executed_locally_with_timeout(
                    &self.validator_state,
                    tx_digest,
                    tx_type,
                )
                .await
                .is_ok();

                executed_locally
            } else {
                false
            };
        }

        let QuorumTransactionResponse {
            effects,

            input_objects,
            output_objects,
        } = response;

        let response = ExecuteTransactionResponse {
            effects,

            input_objects,
            output_objects,
        };

        Ok((response, executed_locally))
    }

    #[instrument(name = "tx_orchestrator_execute_transaction_v3", level = "debug", skip_all,
                 fields(tx_digest = ?request.transaction.digest()))]
    pub async fn execute_transaction(
        &self,
        request: ExecuteTransactionRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<ExecuteTransactionResponse, QuorumDriverError> {
        let timer = Instant::now();
        let tx_type = if request.transaction.is_consensus_tx() {
            TxType::SharedObject
        } else {
            TxType::SingleWriter
        };

        let (response, _) = self
            .execute_transaction_with_effects_waiting(request, client_addr)
            .await?;

        let QuorumTransactionResponse {
            effects,

            input_objects,
            output_objects,
        } = response;

        Ok(ExecuteTransactionResponse {
            effects,

            input_objects,
            output_objects,
        })
    }

    /// Shared implementation for executing transactions with parallel local effects waiting
    async fn execute_transaction_with_effects_waiting(
        &self,
        request: ExecuteTransactionRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<(QuorumTransactionResponse, IsTransactionExecutedLocally), QuorumDriverError> {
        let epoch_store = self.validator_state.load_epoch_store_one_call_per_task();
        let verified_transaction = epoch_store
            .verify_transaction(request.transaction.clone())
            .map_err(QuorumDriverError::InvalidUserSignature)?;
        let tx_digest = *verified_transaction.digest();

        // Early validation check against local state before submission to catch non-retriable errors
        // TODO: Consider moving this check to TransactionDriver for per-retry validation
        if self.enable_early_validation {
            if let Err(e) = self
                .validator_state
                .check_transaction_validity(&epoch_store, &verified_transaction)
            {
                let error_category = e.categorize();
                if !error_category.is_submission_retriable() {
                    // Skip early validation rejection if transaction has already been executed (allows retries to return cached results)
                    if !self.validator_state.is_tx_already_executed(&tx_digest) {
                        debug!(
                            error = ?e,
                            "Transaction rejected during early validation"
                        );

                        return Err(QuorumDriverError::TransactionFailed {
                            category: error_category,
                            details: e.to_string(),
                        });
                    }
                }
            }
        }

        // Add transaction to WAL log.
        let guard =
            TransactionSubmissionGuard::new(self.pending_tx_log.clone(), &verified_transaction);
        let is_new_transaction = guard.is_new_transaction();

        let include_input_objects = request.include_input_objects;
        let include_output_objects = request.include_output_objects;

        let finality_timeout = std::env::var("WAIT_FOR_FINALITY_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(WAIT_FOR_FINALITY_TIMEOUT);

        let num_submissions = if !is_new_transaction {
            // No need to submit when the transaction is already being processed.
            0
        } else if cfg!(msim) {
            // Allow duplicated submissions in tests.
            let r = rand::thread_rng().gen_range(1..=100);
            let n = if r <= 10 {
                3
            } else if r <= 30 {
                2
            } else {
                1
            };
            if n > 1 {
                debug!("Making {n} execution calls");
            }
            n
        } else {
            1
        };

        // Wait for one of the execution futures to succeed, or all of them to fail.
        let mut execution_futures = FuturesUnordered::new();
        for i in 0..num_submissions {
            // Generate jitter values outside the async block
            let should_delay = i > 0 && rand::thread_rng().gen_bool(0.8);
            let delay_ms = if should_delay {
                rand::thread_rng().gen_range(100..=500)
            } else {
                0
            };

            let epoch_store = epoch_store.clone();
            let request = request.clone();
            let verified_transaction = verified_transaction.clone();

            let future = async move {
                if delay_ms > 0 {
                    // Add jitters to duplicated submissions.
                    sleep(Duration::from_millis(delay_ms)).await;
                }
                self.execute_transaction_impl(
                    &epoch_store,
                    request,
                    verified_transaction,
                    client_addr,
                    Some(finality_timeout),
                )
                .await
            }
            .boxed();
            execution_futures.push(future);
        }

        // Track the last execution error.
        let mut last_execution_error: Option<QuorumDriverError> = None;

        // Wait for execution result outside of this call to become available.
        let digests = [tx_digest];
        let mut local_effects_future = epoch_store
            .within_alive_epoch(
                self.validator_state
                    .get_transaction_cache_reader()
                    .notify_read_executed_effects(&digests),
            )
            .boxed();

        // Wait for execution timeout.
        let mut timeout_future = tokio::time::sleep(finality_timeout).boxed();

        loop {
            tokio::select! {
                biased;

                // Local effects might be available
                local_effects_result = &mut local_effects_future => {
                    match local_effects_result {
                        Ok(effects) => {
                            debug!(
                                "Effects became available while execution was running"
                            );
                            if let Some(effects) = effects.into_iter().next() {

                                let epoch = effects.executed_epoch();

                                let input_objects = include_input_objects
                                    .then(|| self.validator_state.get_transaction_input_objects(&effects))
                                    .transpose()
                                    .map_err(QuorumDriverError::QuorumDriverInternalError)?;
                                let output_objects = include_output_objects
                                    .then(|| self.validator_state.get_transaction_output_objects(&effects))
                                    .transpose()
                                    .map_err(QuorumDriverError::QuorumDriverInternalError)?;
                                let response = QuorumTransactionResponse {
                                    effects: FinalizedEffects {
                                        effects,
                                        finality_info: EffectsFinalityInfo::QuorumExecuted(epoch),
                                    },

                                    input_objects,
                                    output_objects,

                                };
                                break Ok((response, true));
                            }
                        }
                        Err(_) => {
                            warn!("Epoch terminated before effects were available");
                        }
                    };

                    // Prevent this branch from being selected again
                    local_effects_future = futures::future::pending().boxed();
                }

                // This branch is disabled if execution_futures is empty.
                Some(result) = execution_futures.next() => {
                    match result {
                        Ok(resp) => {
                            // First success gets returned.
                            debug!("Execution succeeded, returning response");
                            let QuorumTransactionResponse {
                                effects,

                                input_objects,
                                output_objects,

                            } = resp;
                            // Filter fields based on request flags.
                            let resp = QuorumTransactionResponse {
                                effects,

                                input_objects: if include_input_objects { input_objects } else { None },
                                output_objects: if include_output_objects { output_objects } else { None },

                            };
                            break Ok((resp, false));
                        }
                        Err(QuorumDriverError::PendingExecutionInTransactionOrchestrator) => {
                            debug!(
                                "Transaction is already being processed"
                            );
                            // Avoid overriding errors with transaction already being processed.
                            if last_execution_error.is_none() {
                                last_execution_error = Some(QuorumDriverError::PendingExecutionInTransactionOrchestrator);
                            }
                        }
                        Err(e) => {
                            debug!(?e, "Execution attempt failed, wait for other attempts");
                            last_execution_error = Some(e);
                        }
                    };

                    // Last error must have been recorded.
                    if execution_futures.is_empty() {
                        break Err(last_execution_error.unwrap());
                    }
                }

                // A timeout has occurred while waiting for finality
                _ = &mut timeout_future => {
                    debug!("Timeout waiting for transaction finality.");


                    break Err(QuorumDriverError::TimeoutBeforeFinality);
                }
            }
        }
    }

    #[instrument(level = "error", skip_all)]
    async fn execute_transaction_impl(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        request: ExecuteTransactionRequest,
        verified_transaction: VerifiedTransaction,
        client_addr: Option<SocketAddr>,
        finality_timeout: Option<Duration>,
    ) -> Result<QuorumTransactionResponse, QuorumDriverError> {
        debug!("TO Received transaction execution request.");

        // Select TransactionDriver or QuorumDriver for submission.
        let response = self
            .submit_with_transaction_driver(
                &self.transaction_driver,
                &request,
                client_addr,
                &verified_transaction,
                finality_timeout,
            )
            .await?;

        Ok(response)
    }

    #[instrument(level = "error", skip_all, err(level = "info"))]
    async fn submit_with_transaction_driver(
        &self,
        td: &Arc<TransactionDriver<A>>,
        request: &ExecuteTransactionRequest,
        client_addr: Option<SocketAddr>,
        verified_transaction: &VerifiedTransaction,
        timeout_duration: Option<Duration>,
    ) -> Result<QuorumTransactionResponse, QuorumDriverError> {
        let tx_digest = *verified_transaction.digest();
        debug!("Using TransactionDriver for transaction {:?}", tx_digest);

        let td_response = td
            .drive_transaction(
                SubmitTxRequest::new_transaction(request.transaction.clone()),
                SubmitTransactionOptions {
                    forwarded_client_addr: client_addr,
                    allowed_validators: self.td_allowed_submission_list.clone(),
                    blocked_validators: self.td_blocked_submission_list.clone(),
                },
                timeout_duration,
            )
            .await
            .map_err(|e| match e {
                TransactionDriverError::TimeoutWithLastRetriableError {
                    last_error,
                    attempts,
                    timeout,
                } => QuorumDriverError::TimeoutBeforeFinalityWithErrors {
                    last_error: last_error.map(|e| e.to_string()).unwrap_or_default(),
                    attempts,
                    timeout,
                },
                other => QuorumDriverError::TransactionFailed {
                    category: other.categorize(),
                    details: other.to_string(),
                },
            });

        match td_response {
            Err(e) => {
                warn!("TransactionDriver error: {e:?}");
                Err(e)
            }
            Ok(quorum_transaction_response) => Ok(quorum_transaction_response),
        }
    }

    #[instrument(
        name = "tx_orchestrator_wait_for_finalized_tx_executed_locally_with_timeout",
        level = "debug",
        skip_all,
        err(level = "info")
    )]
    async fn wait_for_finalized_tx_executed_locally_with_timeout(
        validator_state: &Arc<AuthorityState>,
        tx_digest: TransactionDigest,
        tx_type: TxType,
    ) -> SomaResult {
        debug!("Waiting for finalized tx to be executed locally.");
        match timeout(
            LOCAL_EXECUTION_TIMEOUT,
            validator_state
                .get_transaction_cache_reader()
                .notify_read_executed_effects_digests(&[tx_digest]),
        )
        .instrument(error_span!(
            "transaction_orchestrator::local_execution",
            ?tx_digest
        ))
        .await
        {
            Err(_elapsed) => {
                debug!(
                    "Waiting for finalized tx to be executed locally timed out within {:?}.",
                    LOCAL_EXECUTION_TIMEOUT
                );

                Err(SomaError::TimeoutError.into())
            }
            Ok(_) => Ok(()),
        }
    }

    fn start_task_to_recover_txes_in_log(
        pending_tx_log: Arc<WritePathPendingTransactionLog>,
        transaction_driver: Arc<TransactionDriver<A>>,
    ) {
        tokio::spawn(async move {
            if std::env::var("SKIP_LOADING_FROM_PENDING_TX_LOG").is_ok() {
                info!("Skipping loading pending transactions from pending_tx_log.");
                return;
            }
            let pending_txes = pending_tx_log
                .load_all_pending_transactions()
                .expect("failed to load all pending transactions");
            let num_pending_txes = pending_txes.len();
            info!(
                "Recovering {} pending transactions from pending_tx_log.",
                num_pending_txes
            );
            let mut recovery = pending_txes
                .into_iter()
                .map(|tx| {
                    let pending_tx_log = pending_tx_log.clone();
                    let transaction_driver = transaction_driver.clone();
                    async move {
                        // TODO: ideally pending_tx_log would not contain VerifiedTransaction, but that
                        // requires a migration.
                        let tx = tx.into_inner();
                        let tx_digest = *tx.digest();
                        // It's not impossible we fail to enqueue a task but that's not the end of world.
                        // TODO(william) correctly extract client_addr from logs
                        if let Err(err) = transaction_driver
                            .drive_transaction(
                                SubmitTxRequest::new_transaction(tx),
                                SubmitTransactionOptions::default(),
                                Some(Duration::from_secs(60)),
                            )
                            .await
                        {
                            warn!(?tx_digest, "Failed to execute recovered transaction: {err}");
                        } else {
                            debug!(?tx_digest, "Executed recovered transaction");
                        }
                        if let Err(err) = pending_tx_log.finish_transaction(&tx_digest) {
                            warn!(
                                ?tx_digest,
                                "Failed to clean up transaction in pending log: {err}"
                            );
                        } else {
                            debug!(?tx_digest, "Cleaned up transaction in pending log");
                        }
                    }
                })
                .collect::<FuturesUnordered<_>>();

            let mut num_recovered = 0;
            while recovery.next().await.is_some() {
                num_recovered += 1;
                if num_recovered % 1000 == 0 {
                    info!(
                        "Recovered {} out of {} transactions from pending_tx_log.",
                        num_recovered, num_pending_txes
                    );
                }
            }
            info!(
                "Recovery finished. Recovered {} out of {} transactions from pending_tx_log.",
                num_recovered, num_pending_txes
            );
        });
    }

    pub fn load_all_pending_transactions_in_test(&self) -> SomaResult<Vec<VerifiedTransaction>> {
        self.pending_tx_log.load_all_pending_transactions()
    }

    pub fn authority_state(&self) -> &Arc<AuthorityState> {
        &self.validator_state
    }

    pub fn transaction_driver(&self) -> &Arc<TransactionDriver<A>> {
        &self.transaction_driver
    }

    pub fn clone_authority_aggregator(&self) -> Arc<AuthorityAggregator<A>> {
        self.transaction_driver.authority_aggregator().load_full()
    }

    #[instrument(level = "info", skip_all, fields(tx_digest = ?request.tx_digest, checkpoint_seq = ?request.checkpoint_seq))]
    pub async fn initiate_shard_work(
        &self,
        request: InitiateShardWorkRequest,
    ) -> Result<InitiateShardWorkResponse, SomaError> {
        let encoder_client = self
            .encoder_client
            .as_ref()
            .ok_or(SomaError::EncoderServiceUnavailable)?;

        let tx_digest = request.tx_digest;
        let checkpoint_seq = request.checkpoint_seq;

        // 1. Get the checkpoint and contents
        let checkpoint = self
            .validator_state
            .get_checkpoint_store()
            .get_checkpoint_by_sequence_number(checkpoint_seq)
            .map_err(|e| SomaError::Storage(e.to_string()))?
            .ok_or(SomaError::VerifiedCheckpointNotFound(checkpoint_seq))?;

        let checkpoint_contents = self
            .validator_state
            .get_checkpoint_store()
            .get_checkpoint_contents(&checkpoint.content_digest)
            .map_err(|e| SomaError::Storage(e.to_string()))?
            .ok_or(SomaError::CheckpointContentsNotFound(
                checkpoint.content_digest,
            ))?;

        // 2. Verify the transaction is in this checkpoint
        let tx_in_checkpoint = checkpoint_contents
            .iter()
            .find(|digests| digests.transaction == tx_digest);

        let expected_effects_digest = tx_in_checkpoint
            .ok_or_else(|| {
                SomaError::InvalidRequest(format!(
                    "Transaction {} not found in checkpoint {}",
                    tx_digest, checkpoint_seq
                ))
            })?
            .effects;

        // 3. Look up transaction and effects
        let transaction = self
            .validator_state
            .get_transaction_cache_reader()
            .get_transaction_block(&tx_digest)
            .ok_or(SomaError::TransactionNotFound { digest: tx_digest })?;

        let effects = self
            .validator_state
            .get_transaction_cache_reader()
            .get_executed_effects(&tx_digest)
            .ok_or(SomaError::TransactionNotFinalized)?;

        // 4. Verify effects digest matches checkpoint
        if effects.digest() != expected_effects_digest {
            return Err(SomaError::InvalidRequest(format!(
                "Effects digest mismatch: checkpoint has {:?}, local has {:?}",
                expected_effects_digest,
                effects.digest()
            )));
        }

        // 5. Verify it's an EmbedData transaction and extract refs
        let (download_metadata, target_ref) = match transaction.transaction_data().kind() {
            TransactionKind::EmbedData {
                download_metadata,
                target_ref,
                ..
            } => (download_metadata.clone(), target_ref.clone()),
            _ => return Err(SomaError::NotEmbedDataTransaction),
        };

        // 6. Verify transaction succeeded
        if !effects.status().is_ok() {
            return Err(SomaError::TransactionFailed(format!(
                "{:?}",
                effects.status()
            )));
        }

        // 7. Get the created Shard object (version from effects.created() is correct)
        let created = effects.created();
        let shard_obj_ref = created
            .first()
            .ok_or_else(|| SomaError::InvalidRequest("No created objects in EmbedData tx".into()))?
            .0;

        let shard_object: Shard = {
            let obj = self
                .validator_state
                .get_object_cache_reader()
                .get_object_by_key(&shard_obj_ref.0, shard_obj_ref.1)
                .ok_or_else(|| SomaError::ObjectNotFound {
                    object_id: shard_obj_ref.0,
                    version: Some(shard_obj_ref.1),
                })?;

            obj.as_shard().ok_or_else(|| {
                SomaError::InvalidRequest("Created Shard object could not be deserialized".into())
            })?
        };

        // 8. Load Target at the version actually used during execution
        let target_object: Option<(ObjectRef, Target)> = if let Some(target_ref) = target_ref {
            let target = self
                .validator_state
                .get_object_cache_reader()
                .get_object(&target_ref.0) // Gets latest version
                .ok_or_else(|| SomaError::ObjectNotFound {
                    object_id: target_ref.0,
                    version: None,
                })?;

            Some((
                target_ref,
                target.as_target().ok_or_else(|| {
                    SomaError::InvalidRequest("Target object could not be deserialized".into())
                })?,
            ))
        } else {
            None
        };

        let checkpoint_epoch = checkpoint.epoch();
        let current_epoch = self
            .validator_state
            .load_epoch_store_one_call_per_task()
            .epoch();

        let encoder_committee =
            if checkpoint_epoch == current_epoch || checkpoint_epoch + 1 == current_epoch {
                let system_state = self
                    .validator_state
                    .get_object_cache_reader()
                    .get_system_state_object()
                    .map_err(|e| SomaError::SystemStateReadError(e.to_string()))?;
                system_state.get_current_epoch_encoder_committee()
            } else {
                // Too old - reject
                return Err(SomaError::InvalidRequest(format!(
                    "Transaction from epoch {} is too old. Current epoch is {}. \
             Only current and previous epoch transactions are supported.",
                    checkpoint_epoch, current_epoch
                )));
            };

        encoder_client.update_encoder_committee(&encoder_committee);

        // Generate Merkle inclusion proof
        let inclusion_proof =
            checkpoint_contents.generate_inclusion_proof(&tx_digest, &effects.digest())?;

        // 10. Build finality proof
        let finality_proof = FinalityProof::new(
            (*transaction).clone().into_inner(),
            effects.clone(),
            checkpoint.clone().into_inner(),
            inclusion_proof,
        );

        // 11. Compute VDF
        let checkpoint_digest = finality_proof.checkpoint_digest().clone();
        let vdf = self.vdf.clone();

        let (checkpoint_entropy, entropy_proof) =
            tokio::task::spawn_blocking(move || vdf.get_entropy(&checkpoint_digest))
                .await
                .map_err(|e| SomaError::FailedVDF(format!("VDF task panicked: {}", e)))?
                .map_err(|e| SomaError::FailedVDF(e.to_string()))?;

        // 12. Compute shard selection
        let shard_entropy = ShardEntropy::new(
            download_metadata.metadata().clone(),
            checkpoint_entropy.clone(),
        );
        let shard_seed = Digest::new(&shard_entropy)
            .map_err(|e| SomaError::ShardSamplingError(e.to_string()))?;

        let shard = encoder_committee
            .sample_shard(shard_seed)
            .map_err(|e| SomaError::ShardSamplingError(e.to_string()))?;

        // 13. Build shard auth token
        let shard_auth_token = ShardAuthToken::new(
            finality_proof,
            checkpoint_entropy,
            entropy_proof,
            shard_obj_ref,
        );

        // 14. Build the full Input with on-chain objects
        let input = Input::V1(InputV1::new(
            shard_auth_token.clone(),
            download_metadata.clone(),
            shard_object,
            target_object,
        ));

        // 15. Send to shard members
        let encoders = shard.encoders();

        encoder_client
            .send_to_shard(encoders, input, Duration::from_secs(30))
            .await
            .map_err(|e| {
                SomaError::RpcError(
                    "encoder_shard".into(),
                    format!("Failed to contact shard: {}", e),
                )
            })?;

        Ok(InitiateShardWorkResponse { shard })
    }
}

#[async_trait::async_trait]
impl<A> types::transaction_executor::TransactionExecutor for TransactionOrchestrator<A>
where
    A: AuthorityAPI + Send + Sync + 'static + Clone,
{
    async fn execute_transaction(
        &self,
        request: ExecuteTransactionRequest,
        client_addr: Option<std::net::SocketAddr>,
    ) -> Result<ExecuteTransactionResponse, QuorumDriverError> {
        self.execute_transaction(request, client_addr).await
    }

    fn simulate_transaction(
        &self,
        transaction: TransactionData,
        checks: TransactionChecks,
    ) -> Result<SimulateTransactionResult, SomaError> {
        self.validator_state
            .simulate_transaction(transaction, checks)
    }

    async fn initiate_shard_work(
        &self,
        request: InitiateShardWorkRequest,
    ) -> Result<InitiateShardWorkResponse, SomaError> {
        self.initiate_shard_work(request).await
    }
}

/// Keeps track of inflight transactions being submitted, and helps recover transactions
/// on restart.
struct TransactionSubmissionGuard {
    pending_tx_log: Arc<WritePathPendingTransactionLog>,
    tx_digest: TransactionDigest,
    is_new_transaction: bool,
}

impl TransactionSubmissionGuard {
    pub fn new(
        pending_tx_log: Arc<WritePathPendingTransactionLog>,
        tx: &VerifiedTransaction,
    ) -> Self {
        let is_new_transaction = pending_tx_log.write_pending_transaction_maybe(tx);
        let tx_digest = *tx.digest();
        if is_new_transaction {
            debug!(?tx_digest, "Added transaction to inflight set");
        } else {
            debug!(
                ?tx_digest,
                "Transaction already being processed, no new submission will be made"
            );
        };
        Self {
            pending_tx_log,
            tx_digest,
            is_new_transaction,
        }
    }

    fn is_new_transaction(&self) -> bool {
        self.is_new_transaction
    }
}

impl Drop for TransactionSubmissionGuard {
    fn drop(&mut self) {
        if let Err(err) = self.pending_tx_log.finish_transaction(&self.tx_digest) {
            warn!(?self.tx_digest, "Failed to clean up transaction in pending log: {err}");
        } else {
            debug!(?self.tx_digest, "Cleaned up transaction in pending log");
        }
    }
}
