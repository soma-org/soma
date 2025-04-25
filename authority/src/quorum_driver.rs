use crate::{
    aggregator::{
        AggregatorProcessCertificateError, AggregatorProcessTransactionError, AuthorityAggregator,
        ProcessTransactionResult,
    },
    client::{AuthorityAPI, NetworkAuthorityClient},
};
use arc_swap::ArcSwap;
use std::fmt::Write;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Formatter},
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};
use tap::TapFallible;
use tokio::{
    sync::{
        broadcast::error::RecvError,
        mpsc::{self, Receiver, Sender},
        Semaphore,
    },
    task::JoinHandle,
    time::sleep_until,
};
use tonic::async_trait;
use tracing::{debug, error, info, instrument, trace_span, warn};
use types::storage::committee_store::CommitteeStore;
use types::system_state::epoch_start::EpochStartSystemStateTrait;
use types::{
    base::AuthorityName,
    committee::{Committee, EpochId, VotingPower},
    digests::TransactionDigest,
    error::{SomaError, SomaResult},
    grpc::HandleCertificateRequest,
    quorum_driver::{
        ExecuteTransactionRequest, PlainTransactionInfoResponse, QuorumDriverEffectsQueueResult,
        QuorumDriverError, QuorumDriverResponse, QuorumDriverResult,
    },
    system_state::{SystemState, SystemStateTrait},
    transaction::{CertifiedTransaction, Transaction},
};
use utils::notify_read::{NotifyRead, Registration};

const TASK_QUEUE_SIZE: usize = 2000;
const EFFECTS_QUEUE_SIZE: usize = 10000;
const TX_MAX_RETRY_TIMES: u32 = 10;

#[derive(Clone)]
pub struct QuorumDriverTask {
    pub request: ExecuteTransactionRequest,
    pub tx_cert: Option<CertifiedTransaction>,
    pub retry_times: u32,
    pub next_retry_after: Instant,
    pub client_addr: Option<SocketAddr>,
    pub trace_span: Option<tracing::Span>,
}

impl Debug for QuorumDriverTask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = String::new();
        write!(writer, "tx_digest={:?} ", self.request.transaction.digest())?;
        write!(writer, "has_tx_cert={} ", self.tx_cert.is_some())?;
        write!(writer, "retry_times={} ", self.retry_times)?;
        write!(writer, "next_retry_after={:?} ", self.next_retry_after)?;
        write!(f, "{}", writer)
    }
}

pub struct QuorumDriver<A: Clone> {
    validators: ArcSwap<AuthorityAggregator<A>>,
    task_sender: Sender<QuorumDriverTask>,
    effects_subscribe_sender: tokio::sync::broadcast::Sender<QuorumDriverEffectsQueueResult>,
    notifier: Arc<NotifyRead<TransactionDigest, QuorumDriverResult>>,
    max_retry_times: u32,
}

impl<A: Clone> QuorumDriver<A> {
    pub(crate) fn new(
        validators: ArcSwap<AuthorityAggregator<A>>,
        task_sender: Sender<QuorumDriverTask>,
        effects_subscribe_sender: tokio::sync::broadcast::Sender<QuorumDriverEffectsQueueResult>,
        notifier: Arc<NotifyRead<TransactionDigest, QuorumDriverResult>>,
        max_retry_times: u32,
    ) -> Self {
        Self {
            validators,
            task_sender,
            effects_subscribe_sender,
            notifier,
            max_retry_times,
        }
    }

    pub fn authority_aggregator(&self) -> &ArcSwap<AuthorityAggregator<A>> {
        &self.validators
    }

    pub fn clone_committee(&self) -> Arc<Committee> {
        self.validators.load().committee.clone()
    }

    pub fn current_epoch(&self) -> EpochId {
        self.validators.load().committee.epoch
    }

    async fn enqueue_task(&self, task: QuorumDriverTask) -> SomaResult<()> {
        self.task_sender
            .send(task.clone())
            .await
            .tap_err(|e| debug!(?task, "Failed to enqueue task: {:?}", e))
            .tap_ok(|_| {
                debug!(?task, "Enqueued task.");
            })
            .map_err(|e| SomaError::QuorumDriverCommunicationError {
                error: e.to_string(),
            })
    }

    /// Enqueue the task again if it hasn't maxed out the total retry attempts.
    /// If it has, notify failure.
    async fn enqueue_again_maybe(
        &self,
        request: ExecuteTransactionRequest,
        tx_cert: Option<CertifiedTransaction>,
        old_retry_times: u32,
        client_addr: Option<SocketAddr>,
    ) -> SomaResult<()> {
        if old_retry_times >= self.max_retry_times {
            // max out the retry times, notify failure
            info!(tx_digest=?request.transaction.digest(), "Failed to reach finality after attempting for {} times", old_retry_times+1);
            self.notify(
                &request.transaction,
                &Err(
                    QuorumDriverError::FailedWithTransientErrorAfterMaximumAttempts {
                        total_attempts: old_retry_times + 1,
                    },
                ),
                old_retry_times + 1,
            );
            return Ok(());
        }
        self.backoff_and_enqueue(request, tx_cert, old_retry_times, client_addr, None)
            .await
    }

    /// Performs exponential backoff and enqueue the `transaction` to the execution queue.
    /// When `min_backoff_duration` is provided, the backoff duration will be at least `min_backoff_duration`.
    async fn backoff_and_enqueue(
        &self,
        request: ExecuteTransactionRequest,
        tx_cert: Option<CertifiedTransaction>,
        old_retry_times: u32,
        client_addr: Option<SocketAddr>,
        min_backoff_duration: Option<Duration>,
    ) -> SomaResult<()> {
        let next_retry_after = Instant::now()
            + Duration::from_millis(200 * u64::pow(2, old_retry_times))
                .max(min_backoff_duration.unwrap_or(Duration::from_secs(0)));
        sleep_until(next_retry_after.into()).await;

        let tx_cert = match tx_cert {
            // TxCert is only valid when its epoch matches current epoch.
            // Note, it's impossible that TxCert's epoch is larger than current epoch
            // because the TxCert will be considered invalid and cannot reach here.
            Some(tx_cert) if tx_cert.epoch() == self.current_epoch() => Some(tx_cert),
            _other => None,
        };

        self.enqueue_task(QuorumDriverTask {
            request,
            tx_cert,
            retry_times: old_retry_times + 1,
            next_retry_after,
            client_addr,
            trace_span: Some(tracing::Span::current()),
        })
        .await
    }

    pub fn notify(
        &self,
        transaction: &Transaction,
        response: &QuorumDriverResult,
        total_attempts: u32,
    ) {
        let tx_digest = transaction.digest();
        let effects_queue_result = match &response {
            Ok(resp) => Ok((transaction.clone(), resp.clone())),
            Err(err) => Err((*tx_digest, err.clone())),
        };
        // On fullnode we expect the send to always succeed because TransactionOrchestrator should be subscribing
        // to this queue all the time. However the if QuorumDriver is used elsewhere log may be noisy.
        if let Err(err) = self.effects_subscribe_sender.send(effects_queue_result) {
            warn!(?tx_digest, "No subscriber found for effects: {}", err);
        }
        debug!(?tx_digest, "notify QuorumDriver task result");
        self.notifier.notify(tx_digest, response);
    }
}

impl<A> QuorumDriver<A>
where
    A: AuthorityAPI + Send + Sync + 'static + Clone,
{
    #[instrument(level = "trace", skip_all)]
    pub async fn submit_transaction(
        &self,
        request: ExecuteTransactionRequest,
    ) -> SomaResult<Registration<TransactionDigest, QuorumDriverResult>> {
        let tx_digest = request.transaction.digest();
        debug!(?tx_digest, "Received transaction execution request.");

        let ticket = self.notifier.register_one(tx_digest);
        self.enqueue_task(QuorumDriverTask {
            request,
            tx_cert: None,
            retry_times: 0,
            next_retry_after: Instant::now(),
            client_addr: None,
            trace_span: Some(tracing::Span::current()),
        })
        .await?;
        Ok(ticket)
    }

    // Used when the it is called in a component holding the notifier, and a ticket is
    // already obtained prior to calling this function, for instance, TransactionOrchestrator
    #[instrument(level = "trace", skip_all)]
    pub async fn submit_transaction_no_ticket(
        &self,
        request: ExecuteTransactionRequest,
        client_addr: Option<SocketAddr>,
    ) -> SomaResult<()> {
        let tx_digest = request.transaction.digest();
        debug!(
            ?tx_digest,
            "Received transaction execution request, no ticket."
        );

        self.enqueue_task(QuorumDriverTask {
            request,
            tx_cert: None,
            retry_times: 0,
            next_retry_after: Instant::now(),
            client_addr,
            trace_span: Some(tracing::Span::current()),
        })
        .await
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) async fn process_transaction(
        &self,
        transaction: Transaction,
        client_addr: Option<SocketAddr>,
    ) -> Result<ProcessTransactionResult, Option<QuorumDriverError>> {
        let auth_agg = self.validators.load();
        let tx_digest = *transaction.digest();
        let result = auth_agg.process_transaction(transaction, client_addr).await;

        self.process_transaction_result(result, tx_digest, client_addr)
            .await
    }

    #[instrument(level = "trace", skip_all)]
    async fn process_transaction_result(
        &self,
        result: Result<ProcessTransactionResult, AggregatorProcessTransactionError>,
        tx_digest: TransactionDigest,
        client_addr: Option<SocketAddr>,
    ) -> Result<ProcessTransactionResult, Option<QuorumDriverError>> {
        match result {
            Ok(resp) => Ok(resp),
            Err(AggregatorProcessTransactionError::RetryableConflictingTransaction {
                conflicting_tx_digest_to_retry,
                errors,
                conflicting_tx_digests,
            }) => {
                debug!(
                    ?tx_digest,
                    "Observed {} conflicting transactions: {:?}",
                    conflicting_tx_digests.len(),
                    conflicting_tx_digests
                );

                // If no retryable conflicting transaction was returned that means we have >= 2f+1 good stake for
                // the original transaction + retryable stake. Will continue to retry the original transaction.
                debug!(
                        ?errors,
                        "Observed Tx {tx_digest:} is still in retryable state. Conflicting Txes: {conflicting_tx_digests:?}",
                    );
                Err(None)
            }

            Err(AggregatorProcessTransactionError::FatalConflictingTransaction {
                errors,
                conflicting_tx_digests,
            }) => {
                debug!(
                    ?errors,
                    "Observed Tx {tx_digest:} double spend attempted. Conflicting Txes: {conflicting_tx_digests:?}",
                );
                Err(Some(QuorumDriverError::ObjectsDoubleUsed {
                    conflicting_txes: conflicting_tx_digests,
                    retried_tx: None,
                    retried_tx_success: None,
                }))
            }

            Err(AggregatorProcessTransactionError::FatalTransaction { errors }) => {
                debug!(?tx_digest, ?errors, "Nonretryable transaction error");
                Err(Some(QuorumDriverError::NonRecoverableTransactionError {
                    errors,
                }))
            }

            Err(AggregatorProcessTransactionError::SystemOverload {
                overloaded_stake,
                errors,
            }) => {
                debug!(?tx_digest, ?errors, "System overload");
                Err(Some(QuorumDriverError::SystemOverload {
                    overloaded_stake,
                    errors,
                }))
            }

            Err(AggregatorProcessTransactionError::SystemOverloadRetryAfter {
                overload_stake,
                errors,
                retry_after_secs,
            }) => {
                debug!(
                    ?tx_digest,
                    ?errors,
                    "System overload and retry after secs {retry_after_secs}",
                );
                Err(Some(QuorumDriverError::SystemOverloadRetryAfter {
                    overload_stake,
                    errors,
                    retry_after_secs,
                }))
            }

            Err(AggregatorProcessTransactionError::RetryableTransaction { errors }) => {
                debug!(?tx_digest, ?errors, "Retryable transaction error");
                Err(None)
            }

            Err(
                AggregatorProcessTransactionError::TxAlreadyFinalizedWithDifferentUserSignatures,
            ) => {
                debug!(
                    ?tx_digest,
                    "Transaction is already finalized with different user signatures"
                );
                Err(Some(
                    QuorumDriverError::TxAlreadyFinalizedWithDifferentUserSignatures,
                ))
            }
        }
    }

    #[instrument(level = "trace", skip_all, fields(tx_digest = ?request.certificate.digest()))]
    pub(crate) async fn process_certificate(
        &self,
        request: HandleCertificateRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<QuorumDriverResponse, Option<QuorumDriverError>> {
        let auth_agg = self.validators.load();
        let tx_digest = *request.certificate.digest();
        let response = auth_agg
            .process_certificate(request.clone(), client_addr)
            .await
            .map_err(|agg_err| match agg_err {
                AggregatorProcessCertificateError::FatalExecuteCertificate {
                    non_retryable_errors,
                } => {
                    // Normally a certificate shouldn't have fatal errors.
                    error!(
                        ?tx_digest,
                        ?non_retryable_errors,
                        "[WATCHOUT] Unexpected Fatal error for certificate"
                    );
                    Some(QuorumDriverError::NonRecoverableTransactionError {
                        errors: non_retryable_errors,
                    })
                }
                AggregatorProcessCertificateError::RetryableExecuteCertificate {
                    retryable_errors,
                } => {
                    debug!(?retryable_errors, "Retryable certificate");
                    None
                }
            })?;

        Ok(response)
    }

    pub async fn update_validators(&self, new_validators: Arc<AuthorityAggregator<A>>) {
        info!(
            "Quorum Driver updating AuthorityAggregator with committee {}",
            new_validators.committee
        );
        self.validators.store(new_validators);
    }
}

pub struct QuorumDriverHandler<A: Clone> {
    quorum_driver: Arc<QuorumDriver<A>>,
    effects_subscriber: tokio::sync::broadcast::Receiver<QuorumDriverEffectsQueueResult>,
    reconfig_observer: Arc<dyn ReconfigObserver<A> + Sync + Send>,
    _processor_handle: JoinHandle<()>,
}

impl<A> QuorumDriverHandler<A>
where
    A: AuthorityAPI + Send + Sync + 'static + Clone,
{
    pub(crate) fn new(
        validators: Arc<AuthorityAggregator<A>>,
        notifier: Arc<NotifyRead<TransactionDigest, QuorumDriverResult>>,
        reconfig_observer: Arc<dyn ReconfigObserver<A> + Sync + Send>,
        max_retry_times: u32,
    ) -> Self {
        let (task_tx, task_rx) = mpsc::channel::<QuorumDriverTask>(TASK_QUEUE_SIZE);
        let (subscriber_tx, subscriber_rx) =
            tokio::sync::broadcast::channel::<_>(EFFECTS_QUEUE_SIZE);
        let quorum_driver = Arc::new(QuorumDriver::new(
            ArcSwap::new(validators),
            task_tx,
            subscriber_tx,
            notifier,
            max_retry_times,
        ));
        let processor_handle = {
            let quorum_driver_clone = quorum_driver.clone();
            tokio::spawn(Self::task_queue_processor(quorum_driver_clone, task_rx))
        };
        let reconfig_observer_clone = reconfig_observer.clone();
        {
            let quorum_driver_clone = quorum_driver.clone();
            tokio::spawn({
                async move {
                    let mut reconfig_observer_clone = reconfig_observer_clone.clone_boxed();
                    reconfig_observer_clone.run(quorum_driver_clone).await;
                }
            });
        };
        Self {
            quorum_driver,
            effects_subscriber: subscriber_rx,
            reconfig_observer,
            _processor_handle: processor_handle,
        }
    }

    // Used when the it is called in a component holding the notifier, and a ticket is
    // already obtained prior to calling this function, for instance, TransactionOrchestrator
    pub async fn submit_transaction_no_ticket(
        &self,
        request: ExecuteTransactionRequest,
        client_addr: Option<SocketAddr>,
    ) -> SomaResult<()> {
        self.quorum_driver
            .submit_transaction_no_ticket(request, client_addr)
            .await
    }

    pub async fn submit_transaction(
        &self,
        request: ExecuteTransactionRequest,
    ) -> SomaResult<Registration<TransactionDigest, QuorumDriverResult>> {
        self.quorum_driver.submit_transaction(request).await
    }

    /// Create a new `QuorumDriverHandler` based on the same AuthorityAggregator.
    /// Note: the new `QuorumDriverHandler` will have a new `ArcSwap<AuthorityAggregator>`
    /// that is NOT tied to the original one. So if there are multiple QuorumDriver(Handler)
    /// then all of them need to do reconfigs on their own.
    pub fn clone_new(&self) -> Self {
        let (task_sender, task_rx) = mpsc::channel::<QuorumDriverTask>(TASK_QUEUE_SIZE);
        let (effects_subscribe_sender, subscriber_rx) =
            tokio::sync::broadcast::channel::<_>(EFFECTS_QUEUE_SIZE);
        let validators = ArcSwap::new(self.quorum_driver.authority_aggregator().load_full());
        let quorum_driver = Arc::new(QuorumDriver {
            validators,
            task_sender,
            effects_subscribe_sender,
            notifier: Arc::new(NotifyRead::new()),
            max_retry_times: self.quorum_driver.max_retry_times,
        });

        let processor_handle = {
            let quorum_driver_copy = quorum_driver.clone();
            tokio::spawn(Self::task_queue_processor(quorum_driver_copy, task_rx))
        };
        {
            let quorum_driver_copy = quorum_driver.clone();
            let reconfig_observer = self.reconfig_observer.clone();
            tokio::spawn({
                async move {
                    let mut reconfig_observer_clone = reconfig_observer.clone_boxed();
                    reconfig_observer_clone.run(quorum_driver_copy).await;
                }
            })
        };

        Self {
            quorum_driver,
            effects_subscriber: subscriber_rx,
            reconfig_observer: self.reconfig_observer.clone(),
            _processor_handle: processor_handle,
        }
    }

    pub fn clone_quorum_driver(&self) -> Arc<QuorumDriver<A>> {
        self.quorum_driver.clone()
    }

    pub fn subscribe_to_effects(
        &self,
    ) -> tokio::sync::broadcast::Receiver<QuorumDriverEffectsQueueResult> {
        self.effects_subscriber.resubscribe()
    }

    pub fn authority_aggregator(&self) -> &ArcSwap<AuthorityAggregator<A>> {
        self.quorum_driver.authority_aggregator()
    }

    pub fn current_epoch(&self) -> EpochId {
        self.quorum_driver.current_epoch()
    }

    /// Process a QuorumDriverTask.
    /// The function has no return value - the corresponding actions of task result
    /// are performed in this call.
    #[instrument(level = "trace", parent = task.trace_span.as_ref().and_then(|s| s.id()), skip_all)]
    async fn process_task(quorum_driver: Arc<QuorumDriver<A>>, task: QuorumDriverTask) {
        debug!(?task, "Quorum Driver processing task");
        let QuorumDriverTask {
            request,
            tx_cert,
            retry_times: old_retry_times,
            client_addr,
            ..
        } = task;
        let transaction = &request.transaction;
        let tx_digest = *transaction.digest();
        let is_single_writer_tx = true;

        let timer = Instant::now();
        let (tx_cert, newly_formed) = match tx_cert {
            None => match quorum_driver
                .process_transaction(transaction.clone(), client_addr)
                .await
            {
                Ok(ProcessTransactionResult::Certified {
                    certificate,
                    newly_formed,
                }) => {
                    debug!(?tx_digest, "Transaction processing succeeded");
                    (certificate, newly_formed)
                }
                Ok(ProcessTransactionResult::Executed(effects_cert)) => {
                    debug!(
                        ?tx_digest,
                        "Transaction processing succeeded with effects directly"
                    );
                    let response = QuorumDriverResponse { effects_cert };
                    quorum_driver.notify(transaction, &Ok(response), old_retry_times + 1);
                    return;
                }
                Err(err) => {
                    Self::handle_error(
                        quorum_driver,
                        request,
                        err,
                        None,
                        old_retry_times,
                        "get tx cert",
                        client_addr,
                    );
                    return;
                }
            },
            Some(tx_cert) => (tx_cert, false),
        };

        let response = match quorum_driver
            .process_certificate(
                HandleCertificateRequest {
                    certificate: tx_cert.clone(),
                },
                client_addr,
            )
            .await
        {
            Ok(response) => {
                debug!(?tx_digest, "Certificate processing succeeded");
                response
            }
            // Note: non retryable failure when processing a cert
            // should be very rare.
            Err(err) => {
                Self::handle_error(
                    quorum_driver,
                    request,
                    err,
                    Some(tx_cert),
                    old_retry_times,
                    "get effects cert",
                    client_addr,
                );
                return;
            }
        };
        if newly_formed {
            let settlement_finality_latency = timer.elapsed().as_secs_f64();

            let is_out_of_expected_range =
                settlement_finality_latency >= 8.0 || settlement_finality_latency <= 0.1;
            debug!(
                ?tx_digest,
                ?is_single_writer_tx,
                ?is_out_of_expected_range,
                "QuorumDriver settlement finality latency: {:.3} seconds",
                settlement_finality_latency
            );
        }

        quorum_driver.notify(transaction, &Ok(response), old_retry_times + 1);
    }

    fn handle_error(
        quorum_driver: Arc<QuorumDriver<A>>,
        request: ExecuteTransactionRequest,
        err: Option<QuorumDriverError>,
        tx_cert: Option<CertifiedTransaction>,
        old_retry_times: u32,
        action: &'static str,
        client_addr: Option<SocketAddr>,
    ) {
        let tx_digest = *request.transaction.digest();
        match err {
            None => {
                debug!(?tx_digest, "Failed to {action} - Retrying");
                let qd_clone = Arc::clone(&quorum_driver);
                tokio::spawn(async move {
                    qd_clone
                        .enqueue_again_maybe(request, tx_cert, old_retry_times, client_addr)
                        .await
                });
            }
            Some(QuorumDriverError::SystemOverloadRetryAfter {
                retry_after_secs, ..
            }) => {
                debug!(?tx_digest, "Failed to {action} - Retrying");
                let qd_clone = Arc::clone(&quorum_driver);
                tokio::spawn(async move {
                    qd_clone
                        .backoff_and_enqueue(
                            request,
                            tx_cert,
                            old_retry_times,
                            client_addr,
                            Some(Duration::from_secs(retry_after_secs)),
                        )
                        .await
                });
            }
            Some(qd_error) => {
                debug!(?tx_digest, "Failed to {action}: {}", qd_error);
                quorum_driver.notify(&request.transaction, &Err(qd_error), old_retry_times + 1);
            }
        }
    }

    async fn task_queue_processor(
        quorum_driver: Arc<QuorumDriver<A>>,
        mut task_receiver: Receiver<QuorumDriverTask>,
    ) {
        let limit = Arc::new(Semaphore::new(TASK_QUEUE_SIZE));
        while let Some(task) = task_receiver.recv().await {
            let task_queue_span =
                trace_span!(parent: task.trace_span.as_ref().and_then(|s| s.id()), "task_queue");
            let task_span_guard = task_queue_span.enter();

            // hold semaphore permit until task completes. unwrap ok because we never close
            // the semaphore in this context.
            let limit = limit.clone();
            let permit = limit.acquire_owned().await.unwrap();

            // TODO check reconfig process here

            debug!(?task, "Dequeued task");
            if Instant::now()
                .checked_duration_since(task.next_retry_after)
                .is_none()
            {
                // Not ready for next attempt yet, re-enqueue
                let _ = quorum_driver.enqueue_task(task).await;
                continue;
            }
            let qd = quorum_driver.clone();
            drop(task_span_guard);
            tokio::spawn(async move {
                let _guard = permit;
                QuorumDriverHandler::process_task(qd, task).await
            });
        }
    }
}

pub struct QuorumDriverHandlerBuilder<A: Clone> {
    validators: Arc<AuthorityAggregator<A>>,
    notifier: Option<Arc<NotifyRead<TransactionDigest, QuorumDriverResult>>>,
    reconfig_observer: Option<Arc<dyn ReconfigObserver<A> + Sync + Send>>,
    max_retry_times: u32,
}

impl<A> QuorumDriverHandlerBuilder<A>
where
    A: AuthorityAPI + Send + Sync + 'static + Clone,
{
    pub fn new(validators: Arc<AuthorityAggregator<A>>) -> Self {
        Self {
            validators,
            notifier: None,
            reconfig_observer: None,
            max_retry_times: TX_MAX_RETRY_TIMES,
        }
    }

    pub(crate) fn with_notifier(
        mut self,
        notifier: Arc<NotifyRead<TransactionDigest, QuorumDriverResult>>,
    ) -> Self {
        self.notifier = Some(notifier);
        self
    }

    pub fn with_reconfig_observer(
        mut self,
        reconfig_observer: Arc<dyn ReconfigObserver<A> + Sync + Send>,
    ) -> Self {
        self.reconfig_observer = Some(reconfig_observer);
        self
    }

    /// Used in tests when smaller number of retries is desired
    pub fn with_max_retry_times(mut self, max_retry_times: u32) -> Self {
        self.max_retry_times = max_retry_times;
        self
    }

    pub fn start(self) -> QuorumDriverHandler<A> {
        QuorumDriverHandler::new(
            self.validators,
            self.notifier.unwrap_or_else(|| {
                Arc::new(NotifyRead::<TransactionDigest, QuorumDriverResult>::new())
            }),
            self.reconfig_observer
                .expect("Reconfig observer is missing"),
            self.max_retry_times,
        )
    }
}

#[async_trait]
pub trait ReconfigObserver<A: Clone> {
    async fn run(&mut self, quorum_driver: Arc<QuorumDriver<A>>);
    fn clone_boxed(&self) -> Box<dyn ReconfigObserver<A> + Send + Sync>;
}

/// A ReconfigObserver that subscribes to a reconfig channel of new committee.
/// This is used in TransactionOrchestrator.
pub struct OnsiteReconfigObserver {
    reconfig_rx: tokio::sync::broadcast::Receiver<SystemState>,
    committee_store: Arc<CommitteeStore>,
}

impl OnsiteReconfigObserver {
    pub fn new(
        reconfig_rx: tokio::sync::broadcast::Receiver<SystemState>,
        committee_store: Arc<CommitteeStore>,
    ) -> Self {
        Self {
            reconfig_rx,
            committee_store,
        }
    }
}

#[async_trait]
impl ReconfigObserver<NetworkAuthorityClient> for OnsiteReconfigObserver {
    fn clone_boxed(&self) -> Box<dyn ReconfigObserver<NetworkAuthorityClient> + Send + Sync> {
        Box::new(Self {
            reconfig_rx: self.reconfig_rx.resubscribe(),
            committee_store: self.committee_store.clone(),
        })
    }

    async fn run(&mut self, quorum_driver: Arc<QuorumDriver<NetworkAuthorityClient>>) {
        loop {
            match self.reconfig_rx.recv().await {
                Ok(system_state) => {
                    let epoch_start_state = system_state.into_epoch_start_state();
                    let committee = epoch_start_state.get_committee();
                    info!("Got reconfig message. New committee: {}", committee);
                    if committee.epoch() > quorum_driver.current_epoch() {
                        let new_auth_agg = quorum_driver
                            .authority_aggregator()
                            .load()
                            .recreate_with_new_epoch_start_state(&epoch_start_state);
                        quorum_driver
                            .update_validators(Arc::new(new_auth_agg))
                            .await;
                    } else {
                        // This should only happen when the node just starts
                        warn!("Epoch number decreased - ignoring committee: {}", committee);
                    }
                }
                // It's ok to miss messages due to overflow here
                Err(RecvError::Lagged(_)) => {
                    continue;
                }
                Err(RecvError::Closed) => {
                    // Closing the channel only happens in simtest when a node is shut down.
                    if cfg!(msim) {
                        return;
                    } else {
                        panic!("Do not expect the channel to be closed")
                    }
                }
            }
        }
    }
}
/// A dummy ReconfigObserver for testing.
pub struct DummyReconfigObserver;

#[async_trait]
impl<A> ReconfigObserver<A> for DummyReconfigObserver
where
    A: AuthorityAPI + Send + Sync + Clone + 'static,
{
    fn clone_boxed(&self) -> Box<dyn ReconfigObserver<A> + Send + Sync> {
        Box::new(Self {})
    }

    async fn run(&mut self, _quorum_driver: Arc<QuorumDriver<A>>) {}
}
