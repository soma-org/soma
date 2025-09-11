use std::{future::Future, net::SocketAddr, sync::Arc, time::Duration};

use fastcrypto::traits::KeyPair;
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
    checksum::Checksum,
    digests::TransactionDigest,
    effects::{TransactionEffectsAPI, VerifiedCertifiedTransactionEffects},
    encoder_committee::EncoderCommittee,
    entropy::SimpleVDF,
    error::{SomaError, SomaResult},
    finality::{CertifiedConsensusFinality, FinalityProof, VerifiedCertifiedConsensusFinality},
    metadata::{Metadata, MetadataCommitment, MetadataV1},
    multiaddr::Multiaddr,
    quorum_driver::{
        ExecuteTransactionRequest, ExecuteTransactionRequestType, ExecuteTransactionResponse,
        FinalizedEffects, IsTransactionExecutedLocally, QuorumDriverEffectsQueueResult,
        QuorumDriverError, QuorumDriverResponse, QuorumDriverResult,
    },
    shard::{Shard, ShardAuthToken, ShardEntropy},
    shard_crypto::{digest::Digest, keys::PeerPublicKey},
    system_state::{SystemState, SystemStateTrait},
    transaction::{
        Transaction, TransactionData, TransactionKind, VerifiedExecutableTransaction,
        VerifiedTransaction,
    },
    transaction_executor::{SimulateTransactionResult, TransactionChecks},
};
use utils::notify_read::NotifyRead;

use crate::{
    aggregator::AuthorityAggregator,
    client::{AuthorityAPI, NetworkAuthorityClient},
    encoder_client::EncoderClientService,
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

pub struct TransactionOrchestrator<A: Clone> {
    quorum_driver_handler: Arc<QuorumDriverHandler<A>>,
    validator_state: Arc<AuthorityState>,
    _local_executor_handle: JoinHandle<()>,
    // pending_tx_log: Arc<WritePathPendingTransactionLog>,
    notifier: Arc<NotifyRead<TransactionDigest, QuorumDriverResult>>,
    vdf: Arc<SimpleVDF>,
    encoder_client: Option<Arc<EncoderClientService>>,
}

impl TransactionOrchestrator<NetworkAuthorityClient> {
    pub fn new_with_encoder_client(
        validators: Arc<AuthorityAggregator<NetworkAuthorityClient>>,
        validator_state: Arc<AuthorityState>,
        reconfig_channel: Receiver<SystemState>,
        encoder_client: Option<Arc<EncoderClientService>>,
    ) -> Self {
        let observer =
            OnsiteReconfigObserver::new(reconfig_channel, validator_state.clone_committee_store());
        TransactionOrchestrator::new(validators, validator_state, observer, encoder_client)
    }

    pub fn new_with_auth_aggregator(
        validators: Arc<AuthorityAggregator<NetworkAuthorityClient>>,
        validator_state: Arc<AuthorityState>,
        reconfig_channel: Receiver<SystemState>,
    ) -> Self {
        let observer =
            OnsiteReconfigObserver::new(reconfig_channel, validator_state.clone_committee_store());
        TransactionOrchestrator::new(validators, validator_state, observer, None)
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
        reconfig_observer: OnsiteReconfigObserver,
        encoder_client: Option<Arc<EncoderClientService>>,
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

        // TODO: Configure VDF iterations
        const VDF_ITERATIONS: u64 = 1; // Adjust based on security requirements

        Self {
            quorum_driver_handler,
            validator_state,
            _local_executor_handle,
            notifier,
            vdf: Arc::new(SimpleVDF::new(VDF_ITERATIONS)),
            encoder_client,
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
                transaction.clone(),
                response.effects_cert.executed_epoch(),
            );
            Self::wait_for_finalized_tx_executed_locally_with_timeout(
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

        let QuorumDriverResponse {
            effects_cert,
            finality_cert,
            input_objects,
            output_objects,
        } = response;

        let shard = self
            .process_finality_and_encoder_work(&transaction, finality_cert)
            .await;

        let response = ExecuteTransactionResponse {
            effects: FinalizedEffects::new_from_effects_cert(effects_cert.into()),
            shard,
            input_objects,
            output_objects,
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

    #[instrument(name = "tx_orchestrator_wait_for_finalized_tx_executed_locally_with_timeout", level = "debug", skip_all, fields(tx_digest = ?transaction.digest()), err)]
    async fn wait_for_finalized_tx_executed_locally_with_timeout(
        validator_state: &Arc<AuthorityState>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        transaction: &VerifiedExecutableTransaction,
        effects_cert: &VerifiedCertifiedTransactionEffects,
    ) -> SomaResult {
        let tx_digest = transaction.digest();
        if validator_state.is_tx_already_executed(tx_digest)? {
            return Ok(());
        }

        debug!(
            ?tx_digest,
            "Waiting for finalized tx to be executed locally."
        );

        match timeout(
            LOCAL_EXECUTION_TIMEOUT,
            validator_state
                .get_transaction_cache_reader()
                .notify_read_executed_effects_digests(&[*tx_digest]),
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
                    "Waiting for finalized tx to be executed locally timed out within {:?}.",
                    LOCAL_EXECUTION_TIMEOUT
                );

                Err(SomaError::TimeoutError)
            }
            Ok(_) => {
                debug!(?tx_digest, "Successfully confirmed tx executed locally.");
                Ok(())
            }
        }
    }
    async fn loop_execute_finalized_tx_locally(
        validator_state: Arc<AuthorityState>,
        mut effects_receiver: Receiver<QuorumDriverEffectsQueueResult>,
    ) {
        loop {
            match effects_receiver.recv().await {
                Ok(Ok((
                    transaction,
                    QuorumDriverResponse {
                        effects_cert,
                        finality_cert,
                        input_objects,
                        output_objects,
                    },
                ))) => {
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
                                "Transaction signature failed to verify after quorum driver \
                                 execution."
                            );
                            continue;
                        }
                    };

                    let executable_tx = VerifiedExecutableTransaction::new_from_quorum_execution(
                        transaction,
                        effects_cert.executed_epoch(),
                    );

                    let _ = Self::wait_for_finalized_tx_executed_locally_with_timeout(
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

    /// Process finality proof generation and encoder work initiation for transactions requiring consensus finality
    async fn process_finality_and_encoder_work(
        &self,
        transaction: &VerifiedTransaction,
        finality_cert: Option<VerifiedCertifiedConsensusFinality>,
    ) -> Option<Shard> {
        info!(
            "Processing finality and encoder work for tx: {}",
            transaction.digest()
        );
        // Early return if transaction doesn't require consensus finality
        if !transaction
            .data()
            .transaction_data()
            .requires_consensus_finality()
        {
            info!("No finality required for tx: {}", transaction.digest());
            return None;
        }

        // Early return if no finality certificate
        let finality_cert = finality_cert?;

        // Generate finality proof
        let finality_proof =
            FinalityProof::new(transaction.inner().clone(), finality_cert.inner().clone());

        info!("Generated finality proof for tx: {}", transaction.digest());

        // TODO: define real Metadata and commitment
        let size_in_bytes = 1;
        let metadata = MetadataV1::new(Checksum::default(), size_in_bytes);
        let metadata_commitment = MetadataCommitment::new(Metadata::V1(metadata), [0u8; 32]);

        let tls_key = PeerPublicKey::new(
            self.validator_state
                .config
                .network_key_pair()
                .into_inner()
                .public()
                .clone(),
        );
        let address = self.validator_state.config.network_address.clone();

        if let Ok(shard) = self
            .initiate_encoder_work_for_embed_data(
                &finality_proof,
                metadata_commitment.clone(),
                tls_key,
                address,
                transaction.digest(),
            )
            .await
        {
            Some(shard)
        } else {
            None
        }
    }

    /// Initiate encoder work for EmbedData transactions
    async fn initiate_encoder_work_for_embed_data(
        &self,
        finality_proof: &FinalityProof,
        metadata_commitment: MetadataCommitment,
        tls_key: PeerPublicKey,
        address: Multiaddr,
        tx_digest: &TransactionDigest,
    ) -> SomaResult<Shard> {
        let (shard, shard_auth_token) = self
            .generate_shard_selection(finality_proof, metadata_commitment)
            .await?;

        if let Some(encoder_client) = &self.encoder_client {
            // Update encoder committee before sending
            if let Ok(committee) = self.get_current_encoder_committee().await {
                encoder_client.update_encoder_committee(&committee);
            }

            let client = encoder_client.clone();
            let token = shard_auth_token.clone();
            let digest = *tx_digest;
            let shard = shard.clone();

            tokio::spawn(async move {
                match client
                    .send_to_shard(
                        shard.encoders(),
                        token,
                        tls_key,
                        address,
                        Duration::from_secs(5),
                    )
                    .await
                {
                    Ok(()) => {
                        debug!(?digest, "Successfully sent shard input to all members");
                    }
                    Err(e) => {
                        error!(?digest, error = ?e, "Failed to send to shard members");
                    }
                }
            });
        }

        return Ok(shard);
    }

    /// Generate shard selection for EmbedData transactions
    async fn generate_shard_selection(
        &self,
        finality_proof: &FinalityProof,
        metadata_commitment: MetadataCommitment,
    ) -> SomaResult<(Shard, ShardAuthToken)> {
        let block_ref = finality_proof.block_ref().clone();
        let vdf = self.vdf.clone();

        // Move the VDF computation to a blocking thread pool
        let (block_entropy, block_entropy_proof) =
            tokio::task::spawn_blocking(move || vdf.get_entropy(block_ref))
                .await
                .map_err(|e| SomaError::from(format!("VDF task failed: {}", e)))??;
        // Get current encoder committee from system state
        let encoder_committee = self.get_current_encoder_committee().await?;

        // Create ShardEntropy
        let shard_entropy = ShardEntropy::new(metadata_commitment.clone(), block_entropy.clone());

        // Calculate entropy digest
        let entropy_digest = Digest::new(&shard_entropy)
            .map_err(|e| SomaError::from(format!("Failed to compute shard entropy: {}", e)))?;

        let shard = encoder_committee
            .sample_shard(entropy_digest)
            .map_err(|e| SomaError::from(e.to_string()))?;

        // Create ShardAuthToken
        let shard_auth_token = ShardAuthToken::new(
            finality_proof.clone(),
            block_entropy,
            block_entropy_proof,
            metadata_commitment,
        );

        Ok((shard, shard_auth_token))
    }

    /// Get current encoder committee from system state
    async fn get_current_encoder_committee(&self) -> SomaResult<EncoderCommittee> {
        // Get system state object
        let system_state = self
            .validator_state
            .get_object_cache_reader()
            .get_system_state_object()?;

        // Get encoder committee for current epoch
        Ok(system_state.get_current_epoch_encoder_committee())
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
        todo!()
        // self.validator_state
        //     .simulate_transaction(transaction, checks)
    }
}
