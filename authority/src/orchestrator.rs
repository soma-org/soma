use std::{future::Future, net::SocketAddr, sync::Arc, time::Duration};

use futures::{
    future::{select, Either},
    FutureExt,
};
use tokio::{
    sync::broadcast::{error::RecvError, Receiver},
    task::JoinHandle,
    time::timeout,
};
use tracing::{debug, error, error_span, info, instrument, warn, Instrument};
use types::{
    digests::TransactionDigest,
    effects::{TransactionEffectsAPI, VerifiedCertifiedTransactionEffects},
    error::{SomaError, SomaResult},
    quorum_driver::{
        ExecuteTransactionRequest, ExecuteTransactionRequestType, ExecuteTransactionResponse,
        FinalizedEffects, IsTransactionExecutedLocally, QuorumDriverEffectsQueueResult,
        QuorumDriverError, QuorumDriverResponse, QuorumDriverResult,
    },
    system_state::SystemState,
    transaction::{VerifiedExecutableTransaction, VerifiedTransaction},
};
use utils::notify_read::NotifyRead;

use crate::{
    aggregator::AuthorityAggregator,
    client::{AuthorityAPI, NetworkAuthorityClient},
    epoch_store::AuthorityPerEpochStore,
    quorum_driver::{
        OnsiteReconfigObserver, QuorumDriverHandler, QuorumDriverHandlerBuilder, ReconfigObserver,
    },
    state::AuthorityState,
};

// How long to wait for local execution (including parents) before a timeout
// is returned to client.
const LOCAL_EXECUTION_TIMEOUT: Duration = Duration::from_secs(10);

const WAIT_FOR_FINALITY_TIMEOUT: Duration = Duration::from_secs(30);

pub struct TransactiondOrchestrator<A: Clone> {
    quorum_driver_handler: Arc<QuorumDriverHandler<A>>,
    validator_state: Arc<AuthorityState>,
    _local_executor_handle: JoinHandle<()>,
    // pending_tx_log: Arc<WritePathPendingTransactionLog>,
    notifier: Arc<NotifyRead<TransactionDigest, QuorumDriverResult>>,
}

impl TransactiondOrchestrator<NetworkAuthorityClient> {
    pub fn new_with_auth_aggregator(
        validators: Arc<AuthorityAggregator<NetworkAuthorityClient>>,
        validator_state: Arc<AuthorityState>,
        reconfig_channel: Receiver<SystemState>,
    ) -> Self {
        let observer =
            OnsiteReconfigObserver::new(reconfig_channel, validator_state.clone_committee_store());
        TransactiondOrchestrator::new(validators, validator_state, observer)
    }
}

impl<A> TransactiondOrchestrator<A>
where
    A: AuthorityAPI + Send + Sync + 'static + Clone,
    OnsiteReconfigObserver: ReconfigObserver<A>,
{
    pub fn new(
        validators: Arc<AuthorityAggregator<A>>,
        validator_state: Arc<AuthorityState>,
        reconfig_observer: OnsiteReconfigObserver,
    ) -> Self {
        let notifier = Arc::new(NotifyRead::new());
        let quorum_driver_handler = Arc::new(
            QuorumDriverHandlerBuilder::new(validators)
                .with_notifier(notifier.clone())
                .with_reconfig_observer(Arc::new(reconfig_observer))
                .start(),
        );

        let effects_receiver = quorum_driver_handler.subscribe_to_effects();
        let state_clone = validator_state.clone();

        let _local_executor_handle = {
            tokio::spawn(async move {
                Self::loop_execute_finalized_tx_locally(state_clone, effects_receiver).await;
            })
        };

        Self {
            quorum_driver_handler,
            validator_state,
            _local_executor_handle,
            notifier,
        }
    }
}

impl<A> TransactiondOrchestrator<A>
where
    A: AuthorityAPI + Send + Sync + 'static + Clone,
{
    #[instrument(name = "tx_orchestrator_execute_transaction", level = "debug", skip_all,
    fields(
        tx_digest = ?request.transaction.digest(),
        tx_type = ?request_type,
    ),
    err)]
    pub async fn execute_transaction_block(
        &self,
        request: ExecuteTransactionRequest,
        request_type: ExecuteTransactionRequestType,
        client_addr: Option<SocketAddr>,
    ) -> Result<(ExecuteTransactionResponse, IsTransactionExecutedLocally), QuorumDriverError> {
        let epoch_store = self.validator_state.load_epoch_store_one_call_per_task();

        let (transaction, response) = self
            .execute_transaction_impl(&epoch_store, request, client_addr)
            .await?;

        let executed_locally = if matches!(
            request_type,
            ExecuteTransactionRequestType::WaitForLocalExecution
        ) {
            let executable_tx = VerifiedExecutableTransaction::new_from_quorum_execution(
                transaction,
                response.effects_cert.executed_epoch(),
            );
            Self::execute_finalized_tx_locally_with_timeout(
                &self.validator_state,
                &epoch_store,
                &executable_tx,
                &response.effects_cert,
            )
            .await
            .is_ok()
        } else {
            false
        };

        let QuorumDriverResponse { effects_cert } = response;

        let response = ExecuteTransactionResponse {
            effects: FinalizedEffects::new_from_effects_cert(effects_cert.into()),
        };

        Ok((response, executed_locally))
    }

    // TODO check if tx is already executed on this node.
    // Note: since EffectsCert is not stored today, we need to gather that from validators
    // (and maybe store it for caching purposes)
    pub async fn execute_transaction_impl(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        request: ExecuteTransactionRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<(VerifiedTransaction, QuorumDriverResponse), QuorumDriverError> {
        let transaction = epoch_store
            .verify_transaction(request.transaction.clone())
            .map_err(QuorumDriverError::InvalidUserSignature)?;
        let tx_digest = *transaction.digest();
        debug!(?tx_digest, "TO Received transaction execution request.");

        let ticket = self
            .submit(transaction.clone(), request, client_addr)
            .await
            .map_err(|e| {
                warn!(?tx_digest, "QuorumDriverInternalError: {e:?}");
                QuorumDriverError::QuorumDriverInternalError(e)
            })?;

        let Ok(result) = timeout(WAIT_FOR_FINALITY_TIMEOUT, ticket).await else {
            debug!(?tx_digest, "Timeout waiting for transaction finality.");
            return Err(QuorumDriverError::TimeoutBeforeFinality);
        };

        match result {
            Err(err) => {
                warn!(?tx_digest, "QuorumDriverInternalError: {err:?}");
                Err(QuorumDriverError::QuorumDriverInternalError(err))
            }
            Ok(Err(err)) => Err(err),
            Ok(Ok(response)) => Ok((transaction, response)),
        }
    }

    /// Submits the transaction to Quorum Driver for execution.
    /// Returns an awaitable Future.
    #[instrument(name = "tx_orchestrator_submit", level = "trace", skip_all)]
    async fn submit(
        &self,
        transaction: VerifiedTransaction,
        request: ExecuteTransactionRequest,
        client_addr: Option<SocketAddr>,
    ) -> SomaResult<impl Future<Output = SomaResult<QuorumDriverResult>> + '_> {
        let tx_digest = *transaction.digest();
        let ticket = self.notifier.register_one(&tx_digest);
        // TODO(william) need to also write client adr to pending tx log below
        // so that we can re-execute with this client addr if we restart
        self.quorum_driver()
            .submit_transaction_no_ticket(request.clone(), client_addr)
            .await?;

        // It's possible that the transaction effects is already stored in DB at this point.
        // So we also subscribe to that. If we hear from `effects_await` first, it means
        // the ticket misses the previous notification, and we want to ask quorum driver
        // to form a certificate for us again, to serve this request.
        let cache_reader = self.validator_state.get_transaction_cache_reader().clone();
        let qd = self.clone_quorum_driver();
        Ok(async move {
            let digests = [tx_digest];
            let effects_await = cache_reader.notify_read_executed_effects(&digests);
            // let-and-return necessary to satisfy borrow checker.
            #[allow(clippy::let_and_return)]
            let res = match select(ticket, effects_await.boxed()).await {
                Either::Left((quorum_driver_response, _)) => Ok(quorum_driver_response),
                Either::Right((_, unfinished_quorum_driver_task)) => {
                    debug!(
                        ?tx_digest,
                        "Effects are available in DB, use quorum driver to get a certificate"
                    );
                    qd.submit_transaction_no_ticket(request, client_addr)
                        .await?;
                    Ok(unfinished_quorum_driver_task.await)
                }
            };
            res
        })
    }

    #[instrument(name = "tx_orchestrator_execute_finalized_tx_locally_with_timeout", level = "debug", skip_all, fields(tx_digest = ?transaction.digest()), err)]
    async fn execute_finalized_tx_locally_with_timeout(
        validator_state: &Arc<AuthorityState>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        transaction: &VerifiedExecutableTransaction,
        effects_cert: &VerifiedCertifiedTransactionEffects,
    ) -> SomaResult {
        // TODO: attempt a finalized tx at most once per request.
        // Every WaitForLocalExecution request will be attempted to execute twice,
        // one from the subscriber queue, one from the proactive execution before
        // returning results to clients. This is not insanely bad because:
        // 1. it's possible that one attempt finishes before the other, so there's
        //      zero extra work except DB checks
        // 2. an up-to-date fullnode should have minimal overhead to sync parents
        //      (for one extra time)
        // 3. at the end of day, the tx will be executed at most once per lock guard.
        let tx_digest = transaction.digest();
        if validator_state.is_tx_already_executed(tx_digest)? {
            return Ok(());
        }

        debug!(?tx_digest, "Executing finalized tx locally.");
        match timeout(
            LOCAL_EXECUTION_TIMEOUT,
            validator_state.fullnode_execute_certificate_with_effects(
                transaction,
                effects_cert,
                epoch_store,
            ),
        )
        .instrument(error_span!(
            "transaction_orchestrator::local_execution",
            ?tx_digest
        ))
        .await
        {
            Err(_elapsed) => {
                debug!(
                    ?tx_digest,
                    "Executing tx locally by orchestrator timed out within {:?}.",
                    LOCAL_EXECUTION_TIMEOUT
                );

                Err(SomaError::TimeoutError)
            }
            Ok(Err(err)) => {
                debug!(
                    ?tx_digest,
                    "Executing tx locally by orchestrator failed with error: {:?}", err
                );

                Err(SomaError::TransactionOrchestratorLocalExecutionError {
                    error: err.to_string(),
                })
            }
            Ok(Ok(_)) => Ok(()),
        }
    }

    async fn loop_execute_finalized_tx_locally(
        validator_state: Arc<AuthorityState>,
        mut effects_receiver: Receiver<QuorumDriverEffectsQueueResult>,
    ) {
        loop {
            match effects_receiver.recv().await {
                Ok(Ok((transaction, QuorumDriverResponse { effects_cert }))) => {
                    let tx_digest = transaction.digest();

                    let epoch_store = validator_state.load_epoch_store_one_call_per_task();

                    // This is a redundant verification, but SignatureVerifier will cache the
                    // previous result.
                    let transaction = match epoch_store.verify_transaction(transaction) {
                        Ok(transaction) => transaction,
                        Err(err) => {
                            // This should be impossible, since we verified the transaction
                            // before sending it to quorum driver.
                            error!(
                                    ?err,
                                    "Transaction signature failed to verify after quorum driver execution."
                                );
                            continue;
                        }
                    };

                    let executable_tx = VerifiedExecutableTransaction::new_from_quorum_execution(
                        transaction,
                        effects_cert.executed_epoch(),
                    );

                    let _ = Self::execute_finalized_tx_locally_with_timeout(
                        &validator_state,
                        &epoch_store,
                        &executable_tx,
                        &effects_cert,
                    )
                    .await;
                }
                Ok(Err((tx_digest, _err))) => {}
                Err(RecvError::Closed) => {
                    error!("Sender of effects subscriber queue has been dropped!");
                    return;
                }
                Err(RecvError::Lagged(skipped_count)) => {
                    warn!("Skipped {skipped_count} transasctions in effects subscriber queue.");
                }
            }
        }
    }

    pub fn quorum_driver(&self) -> &Arc<QuorumDriverHandler<A>> {
        &self.quorum_driver_handler
    }

    pub fn clone_quorum_driver(&self) -> Arc<QuorumDriverHandler<A>> {
        self.quorum_driver_handler.clone()
    }

    pub fn clone_authority_aggregator(&self) -> Arc<AuthorityAggregator<A>> {
        self.quorum_driver().authority_aggregator().load_full()
    }

    pub fn subscribe_to_effects_queue(&self) -> Receiver<QuorumDriverEffectsQueueResult> {
        self.quorum_driver_handler.subscribe_to_effects()
    }
}
