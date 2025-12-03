use crate::{
    authority::{AuthorityState, ExecutionEnv}, authority_per_epoch_store::AuthorityPerEpochStore, checkpoints::CheckpointStore, consensus_adapter::ConsensusAdapter, consensus_tx_status_cache::{ConsensusTxStatus, NotifyReadConsensusTxStatusResult}, execution_scheduler::SchedulingSource, mysticeti_adapter::LazyMysticetiClient, server::TLS_SERVER_NAME, shared_obj_version_manager::Schedulable, tonic_gen::validator_server::{Validator, ValidatorServer}
};
use anyhow::Result;
use async_trait::async_trait;
use fastcrypto::traits::KeyPair;
use futures::{future, TryFutureExt};
use itertools::Itertools as _;
use nonempty::{nonempty, NonEmpty};
use std::{
    cmp::Ordering,
    future::Future,
    io,
    net::{IpAddr, SocketAddr},
    pin::Pin,
    sync::Arc,
    time::{Duration},
};
use tap::TapFallible;
use tokio::sync::oneshot;
use tokio::time::timeout;
use tonic::metadata::{Ascii, MetadataValue};
use tonic::transport::server::TcpConnectInfo;
use tracing::{debug, error, error_span, info, instrument, Instrument};
use types::{
    checkpoints::{CheckpointRequest, CheckpointResponse}, config::local_ip_utils::new_local_tcp_address_for_testing, consensus::{ConsensusPosition, ConsensusTransactionKind}, digests::{TransactionDigest, TransactionEffectsDigest}, effects::{TransactionEffects}, error::SomaResult, object::Object, system_state::SystemState, transaction::{VerifiedCertificate, VerifiedExecutableTransaction}, transaction_outputs::TransactionOutputs
};
use types::{
    consensus::ConsensusTransaction,
    error::SomaError,
    messages_grpc::{
        ObjectInfoRequest, ObjectInfoResponse, TransactionInfoRequest, HandleTransactionResponse, TransactionInfoResponse,  SystemStateRequest, HandleCertificateResponse,HandleCertificateRequest,ExecutedData, SubmitTxResult, RawSubmitTxRequest, SubmitTxType,RawSubmitTxResponse,  PingType,  WaitForEffectsRequest, WaitForEffectsResponse, RawWaitForEffectsRequest, RawWaitForEffectsResponse
    },
    multiaddr::Multiaddr,
    transaction::{CertifiedTransaction, Transaction},
    envelope::Message as _,
};
use types::{ traffic_control::{Weight, ClientIdSource}};

pub struct AuthorityServerHandle {
    server_handle: crate::server::Server,
}

impl AuthorityServerHandle {
    pub async fn join(self) -> Result<(), io::Error> {
        self.server_handle.handle().wait_for_shutdown().await;
        Ok(())
    }

    pub async fn kill(self) -> Result<(), io::Error> {
        self.server_handle.handle().shutdown().await;
        Ok(())
    }

    pub fn address(&self) -> &Multiaddr {
        self.server_handle.local_addr()
    }
}

pub struct AuthorityServer {
    address: Multiaddr,
    pub state: Arc<AuthorityState>,
    consensus_adapter: Arc<ConsensusAdapter>,
}

impl AuthorityServer {
    pub fn new_for_test_with_consensus_adapter(
        state: Arc<AuthorityState>,
        consensus_adapter: Arc<ConsensusAdapter>,
    ) -> Self {
        let address = new_local_tcp_address_for_testing();
        Self {
            address,
            state,
            consensus_adapter,
        }
    }

    pub fn new_for_test(state: Arc<AuthorityState>) -> Self {
        let consensus_adapter = Arc::new(ConsensusAdapter::new(
            Arc::new(LazyMysticetiClient::new()),
            CheckpointStore::new_for_tests(),
            state.name,
            // Arc::new(ConnectionMonitorStatusForTests {}),
            100_000,
            100_000,
            None,
            None,
            state.epoch_store_for_testing().protocol_config().clone(),
        ));
        Self::new_for_test_with_consensus_adapter(state, consensus_adapter)
    }

    pub async fn spawn_for_test(self) -> Result<AuthorityServerHandle, io::Error> {
        let address = self.address.clone();
        self.spawn_with_bind_address_for_test(address).await
    }

    pub async fn spawn_with_bind_address_for_test(
        self,
        address: Multiaddr,
    ) -> Result<AuthorityServerHandle, io::Error> {
        let tls_config = soma_tls::create_rustls_server_config(
            self.state.config.network_key_pair().clone().private_key().into_inner(),
            TLS_SERVER_NAME.to_string(),
        );
        let config = types::client::Config::new();
        let server = crate::server::ServerBuilder::from_config(&config)
            .add_service(ValidatorServer::new(ValidatorService::new_for_tests(
                self.state,
                self.consensus_adapter,
            )))
            .bind(&address, Some(tls_config))
            .await
            .unwrap();
        let local_addr = server.local_addr().to_owned();
        info!("Listening to traffic on {local_addr}");
        let handle = AuthorityServerHandle {
            server_handle: server,
        };
        Ok(handle)
    }
}

#[derive(Clone)]
pub struct ValidatorService {
    state: Arc<AuthorityState>,
    consensus_adapter: Arc<ConsensusAdapter>,
    // traffic_controller: Option<Arc<TrafficController>>,
    // client_id_source: Option<ClientIdSource>,
}

impl ValidatorService {
    pub fn new(
        state: Arc<AuthorityState>,
        consensus_adapter: Arc<ConsensusAdapter>,
        // client_id_source: Option<ClientIdSource>,
    ) -> Self {
        // let traffic_controller = state.traffic_controller.clone();
        Self {
            state,
            consensus_adapter,
            // traffic_controller,
            // client_id_source,
        }
    }

    pub fn new_for_tests(
        state: Arc<AuthorityState>,
        consensus_adapter: Arc<ConsensusAdapter>,
    ) -> Self {
        Self {
            state,
            consensus_adapter,
        }
    }

    pub fn validator_state(&self) -> &Arc<AuthorityState> {
        &self.state
    }



    pub async fn handle_transaction_for_benchmarking(
        &self,
        transaction: Transaction,
    ) -> Result<tonic::Response<HandleTransactionResponse>, tonic::Status> {
        let request = make_tonic_request_for_testing(transaction);
        self.transaction(request).await
    }

    // When making changes to this function, see if the changes should be applied to
    // `Self::handle_submit_transaction()` and `SuiTxValidator::vote_transaction()` as well.
    async fn handle_transaction(
        &self,
        request: tonic::Request<Transaction>,
    ) -> WrappedServiceResponse<HandleTransactionResponse> {
        let Self {
            state,
            consensus_adapter,
        } = self.clone();
        let transaction = request.into_inner();
        let epoch_store = state.load_epoch_store_one_call_per_task();

        // When authority is overloaded and decide to reject this tx, we still lock the object
        // and ask the client to retry in the future. This is because without locking, the
        // input objects can be locked by a different tx in the future, however, the input objects
        // may already be locked by this tx in other validators. This can cause non of the txes
        // to have enough quorum to form a certificate, causing the objects to be locked for
        // the entire epoch. By doing locking but pushback, retrying transaction will have
        // higher chance to succeed.
        // let mut validator_pushback_error = None;
        // let overload_check_res = state.check_system_overload(
        //     &*consensus_adapter,
        //     transaction.data(),
        //     state.check_system_overload_at_signing(),
        // );
        // if let Err(error) = overload_check_res {
        //     // TODO: consider change the behavior for other types of overload errors.
        //     match error.as_inner() {
        //         SomaError::ValidatorOverloadedRetryAfter { .. } => {
        //             validator_pushback_error = Some(error)
        //         }
        //         _ => return Err(error.into()),
        //     }
        // }

        let transaction = epoch_store
            .verify_transaction(transaction)
            .tap_err(|_| {})?;

        let tx_digest = transaction.digest();

        // Enable Trace Propagation across spans/processes using tx_digest
        let span = error_span!("ValidatorService::validator_state_process_tx", ?tx_digest);

        let info = state
            .handle_transaction(&epoch_store, transaction.clone())
            .instrument(span)
            .await
            .tap_err(|e| if let SomaError::ValidatorHaltedAtEpochEnd = e {})?;

        // if let Some(error) = validator_pushback_error {
        //     // TODO: right now, we still sign the txn, but just don't return it. We can also skip signing
        //     // to save more CPU.
        //     return Err(error.into());
        // }

        Ok((tonic::Response::new(info), Weight::zero()))
    }

    #[instrument(
        name = "ValidatorService::handle_submit_transaction",
        level = "error",
        skip_all,
        err(level = "debug")
    )]
    async fn handle_submit_transaction(
        &self,
        request: tonic::Request<RawSubmitTxRequest>,
    ) -> WrappedServiceResponse<RawSubmitTxResponse> {
        let Self {
            state,
            consensus_adapter,
        } = self.clone();

        // let submitter_client_addr = if let Some(client_id_source) = &client_id_source {
        //     self.get_client_ip_addr(&request, client_id_source)
        // } else {
        //     self.get_client_ip_addr(&request, &ClientIdSource::SocketAddr)
        // };
        let submitter_client_addr =  self.get_client_ip_addr(&request, &ClientIdSource::SocketAddr);

        let epoch_store = state.load_epoch_store_one_call_per_task();

        let request = request.into_inner();
        let submit_type = SubmitTxType::try_from(request.submit_type).map_err(|e| {
            SomaError::GrpcMessageDeserializeError {
                type_info: "RawSubmitTxRequest.submit_type".to_string(),
                error: e.to_string(),
            }
        })?;

        let is_ping_request = submit_type == SubmitTxType::Ping;
        if is_ping_request {
            if !(request.transactions.is_empty()) {
                return Err(SomaError::InvalidRequest(format!(
                    "Ping request cannot contain {} transactions",
                    request.transactions.len()
                )).into());
            }
        } else {
            // Ensure default requests contain at least one transaction.
            if !(!request.transactions.is_empty()) {
                return Err(SomaError::InvalidRequest(
                    "At least one transaction needs to be submitted".to_string(),
                ).into());
            }
        }

       

        let max_num_transactions = 
            // Still enforce a limit even when transactions do not need to be in the same block.
            epoch_store
                .protocol_config()
                .max_num_transactions_in_block();

        if !(request.transactions.len() <= max_num_transactions as usize) {
            return Err(SomaError::InvalidRequest(format!(
                "Too many transactions in request: {} vs {}",
                request.transactions.len(),
                max_num_transactions
            )).into());
        }

        // Transaction digests.
        let mut tx_digests = Vec::with_capacity(request.transactions.len());
        // Transactions to submit to consensus.
        let mut consensus_transactions = Vec::with_capacity(request.transactions.len());
        // Indexes of transactions above in the request transactions.
        let mut transaction_indexes = Vec::with_capacity(request.transactions.len());
        // Results corresponding to each transaction in the request.
        let mut results: Vec<Option<SubmitTxResult>> = vec![None; request.transactions.len()];
        // Total size of all transactions in the request.
        let mut total_size_bytes = 0;

        let req_type = if is_ping_request {
            "ping"
        } else if request.transactions.len() == 1 {
            "single_transaction"
        } else {
            "batch"
        };

        
        for (idx, tx_bytes) in request.transactions.iter().enumerate() {
            let transaction = match bcs::from_bytes::<Transaction>(tx_bytes) {
                Ok(txn) => txn,
                Err(e) => {
                    // Ok to fail the request when any transaction is invalid.
                    return Err(SomaError::TransactionDeserializationError {
                        error: format!("Failed to deserialize transaction at index {}: {}", idx, e),
                    }.into());
                }
            };
            

            // Ok to fail the request when any signature is invalid.
            let verified_transaction = {
                match epoch_store.verify_transaction(transaction) {
                    Ok(txn) => txn,
                    Err(e) => {
                        return Err(e.into());
                    }
                }
            };

            let tx_digest = verified_transaction.digest();
            tx_digests.push(*tx_digest);

            debug!(
                ?tx_digest,
                "handle_submit_transaction: verified transaction"
            );

            // Check if the transaction has executed, before checking input objects
            // which could have been consumed.
            if let Some(effects) = self
                .state
                .get_transaction_cache_reader()
                .get_executed_effects(tx_digest)
            {
                let effects_digest = effects.digest();
                if let Ok(executed_data) = self.complete_executed_data(effects, None).await {
                    let executed_result = SubmitTxResult::Executed {
                        effects_digest,
                        details: Some(executed_data),
                        fast_path: false,
                    };
                    results[idx] = Some(executed_result);
                    debug!(?tx_digest, "handle_submit_transaction: already executed");
                    continue;
                }
            }

            debug!(
                ?tx_digest,
                "handle_submit_transaction: waiting for fastpath dependency objects"
            );
            if !state
                .wait_for_fastpath_dependency_objects(&verified_transaction, epoch_store.epoch())
                .await?
            {
                debug!(
                    ?tx_digest,
                    "fastpath input objects are still unavailable after waiting"
                );
            }

            match state.handle_vote_transaction(&epoch_store, verified_transaction.clone()) {
                Ok(_) => { /* continue processing */ }
                Err(e) => {
                    // Check if transaction has been executed while being validated.
                    // This is an edge case so checking executed effects twice is acceptable.
                    if let Some(effects) = self
                        .state
                        .get_transaction_cache_reader()
                        .get_executed_effects(tx_digest)
                    {
                        let effects_digest = effects.digest();
                        if let Ok(executed_data) = self.complete_executed_data(effects, None).await
                        {
                            let executed_result = SubmitTxResult::Executed {
                                effects_digest,
                                details: Some(executed_data),
                                fast_path: false,
                            };
                            results[idx] = Some(executed_result);
                            continue;
                        }
                    }

                    // When the transaction has not been executed, record the error for the transaction.
                    debug!(?tx_digest, "Transaction rejected during submission: {e}");
                    results[idx] = Some(SubmitTxResult::Rejected { error: e });
                    continue;
                }
            }

            consensus_transactions.push(ConsensusTransaction::new_user_transaction_message(
                &self.state.name,
                verified_transaction.into(),
            ));
            transaction_indexes.push(idx);
            total_size_bytes += tx_bytes.len();
        }

        if consensus_transactions.is_empty() && !is_ping_request {
            return Ok((
                tonic::Response::new(Self::try_from_submit_tx_response(results)?),
                Weight::zero(),
            ));
        }

        // Set the max bytes size of the soft bundle to be half of the consensus max transactions in block size.
        // We do this to account for serialization overheads and to ensure that the soft bundle is not too large
        // when is attempted to be posted via consensus.
        let max_transaction_bytes = 
            epoch_store
                .protocol_config()
                .consensus_max_transactions_in_block_bytes();
        if !(
            total_size_bytes <= max_transaction_bytes as usize) {
                return Err(SomaError::TotalTransactionSizeTooLargeInBatch {
                    size: total_size_bytes,
                    limit: max_transaction_bytes,
                }.into());
            }
            

        let consensus_positions = if is_ping_request {
            // We only allow the `consensus_transactions` to be empty for ping requests. This is how it should and is be treated from the downstream components.
            // For any other case, having an empty `consensus_transactions` vector is an invalid state and we should have never reached at this point.
            assert!(
                is_ping_request || !consensus_transactions.is_empty(),
                "A valid bundle must have at least one transaction"
            );
            debug!(
                "handle_submit_transaction: submitting consensus transactions ({}): {}",
                req_type,
                consensus_transactions
                    .iter()
                    .map(|t| t.local_display())
                    .join(", ")
            );
            self.handle_submit_to_consensus_for_position(
                consensus_transactions,
                &epoch_store,
                submitter_client_addr,
            )
            .await?
        } else {
            let futures = consensus_transactions.into_iter().map(|t| {
                debug!(
                    "handle_submit_transaction: submitting consensus transaction ({}): {}",
                    req_type,
                    t.local_display(),
                );
                self.handle_submit_to_consensus_for_position(
                    vec![t],
                    &epoch_store,
                    submitter_client_addr,
                )
            });
            future::try_join_all(futures)
                .await?
                .into_iter()
                .flatten()
                .collect()
        };

        if is_ping_request {
            // For ping requests, return the special consensus position.
            assert_eq!(consensus_positions.len(), 1);
            results.push(Some(SubmitTxResult::Submitted {
                consensus_position: consensus_positions[0],
            }));
        } else {
            // Otherwise, return the consensus position for each transaction.
            for ((idx, tx_digest), consensus_position) in transaction_indexes
                .into_iter()
                .zip(tx_digests)
                .zip(consensus_positions)
            {
                debug!(
                    ?tx_digest,
                    "handle_submit_transaction: submitted consensus transaction at {}",
                    consensus_position,
                );
                results[idx] = Some(SubmitTxResult::Submitted { consensus_position });
            }
        }

        Ok((
            tonic::Response::new(Self::try_from_submit_tx_response(results)?),
            Weight::zero(),
        ))
    }

    fn try_from_submit_tx_response(
        results: Vec<Option<SubmitTxResult>>,
    ) -> Result<RawSubmitTxResponse, SomaError> {
        let mut raw_results = Vec::new();
        for (i, result) in results.into_iter().enumerate() {
            let result = result.ok_or_else(|| SomaError::GenericAuthorityError {
                error: format!("Missing transaction result at {}", i),
            })?;
            let raw_result = result.try_into()?;
            raw_results.push(raw_result);
        }
        Ok(RawSubmitTxResponse {
            results: raw_results,
        })
    }

    // In addition to the response from handling the certificates,
    // returns a bool indicating whether the request should be tallied
    // toward spam count. In general, this should be set to true for
    // requests that are read-only and thus do not consume gas, such
    // as when the transaction is already executed.
    async fn handle_certificates(
        &self,
        certificates: NonEmpty<CertifiedTransaction>,
        include_input_objects: bool,
        include_output_objects: bool,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        wait_for_effects: bool,
    ) -> Result<(Option<Vec<HandleCertificateResponse>>, Weight), tonic::Status> {
        // Validate if cert can be executed
        // Fullnode does not serve handle_certificate call.
        if !(!self.state.is_fullnode(epoch_store)) {
            return Err( SomaError::FullNodeCantHandleCertificate.into());
        }
           

        let is_consensus_tx = certificates.iter().any(|cert| cert.is_consensus_tx());

        

        // 1) Check if the certificate is already executed.
        //    This is only needed when we have only one certificate (not a soft bundle).
        //    When multiple certificates are provided, we will either submit all of them or none of them to consensus.
        if certificates.len() == 1 {
            let tx_digest = *certificates[0].digest();
            debug!(tx_digest=?tx_digest, "Checking if certificate is already executed");

            if let Some(signed_effects) = self
                .state
                .get_signed_effects_and_maybe_resign(&tx_digest, epoch_store)?
            {

                return Ok((
                    Some(vec![HandleCertificateResponse {
                        effects: signed_effects.into_inner(),
                        input_objects: None,
                        output_objects: None,
                  
                    }]),
                    Weight::one(),
                ));
            };
        }

        // 2) Verify the certificates.
        let verified_certificates = {
            
            epoch_store
                .signature_verifier
                .multi_verify_certs(certificates.into())
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
        };
        let consensus_transactions =
            NonEmpty::collect(verified_certificates.iter().map(|certificate| {
                ConsensusTransaction::new_certificate_message(
                    &self.state.name,
                    certificate.clone().into(),
                )
            }))
            .unwrap();

        let (responses, weight) = self
            .handle_submit_to_consensus(
                consensus_transactions,
                include_input_objects,
                include_output_objects,
                epoch_store,
                wait_for_effects,
            )
            .await?;
        // Sign the returned TransactionEffects.
        let responses = if let Some(responses) = responses {
            Some(
                responses
                    .into_iter()
                    .map(|response| {
                        let signed_effects =
                            self.state.sign_effects(response.effects, epoch_store)?;
                        Ok(HandleCertificateResponse {
                            effects: signed_effects.into_inner(),
                         
                            input_objects: if response.input_objects.is_empty() {
                                None
                            } else {
                                Some(response.input_objects)
                            },
                            output_objects: if response.output_objects.is_empty() {
                                None
                            } else {
                                Some(response.output_objects)
                            },
                    
                        })
                    })
                    .collect::<Result<Vec<HandleCertificateResponse>, tonic::Status>>()?,
            )
        } else {
            None
        };

        Ok((responses, weight))
    }

    #[instrument(
        name = "ValidatorService::handle_submit_to_consensus_for_position",
        level = "debug",
        skip_all,
        err(level = "debug")
    )]
    async fn handle_submit_to_consensus_for_position(
        &self,
        // Empty when this is a ping request.
        consensus_transactions: Vec<ConsensusTransaction>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        submitter_client_addr: Option<IpAddr>,
    ) -> Result<Vec<ConsensusPosition>, tonic::Status> {
        let (tx_consensus_positions, rx_consensus_positions) = oneshot::channel();

        {
            // code block within reconfiguration lock
            let reconfiguration_lock = epoch_store.get_reconfig_state_read_lock_guard();
            if !reconfiguration_lock.should_accept_user_certs() {
          
                return Err(SomaError::ValidatorHaltedAtEpochEnd.into());
            }

            // Submit to consensus and wait for position, we do not check if tx
            // has been processed by consensus already as this method is called
            // to get back a consensus position.
         

            self.consensus_adapter.submit_batch(
                &consensus_transactions,
                Some(&reconfiguration_lock),
                epoch_store,
                Some(tx_consensus_positions),
                submitter_client_addr,
            )?;
        }

        Ok(rx_consensus_positions.await.map_err(|e| {
            SomaError::FailedToSubmitToConsensus(format!(
                "Failed to get consensus position: {e}"
            ))
        })?)
    }

    async fn handle_submit_to_consensus(
        &self,
        consensus_transactions: NonEmpty<ConsensusTransaction>,
        include_input_objects: bool,
        include_output_objects: bool,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        wait_for_effects: bool,
    ) -> Result<(Option<Vec<ExecutedData>>, Weight), tonic::Status> {
        let consensus_transactions: Vec<_> = consensus_transactions.into();
        {
            // code block within reconfiguration lock
            let reconfiguration_lock = epoch_store.get_reconfig_state_read_lock_guard();
            if !reconfiguration_lock.should_accept_user_certs() {

                return Err(SomaError::ValidatorHaltedAtEpochEnd.into());
            }

            // 3) All transactions are sent to consensus (at least by some authorities)
            // For certs with shared objects this will wait until either timeout or we have heard back from consensus.
            // For certs with owned objects this will return without waiting for certificate to be sequenced.
            // For uncertified transactions this will wait for fast path processing.
            // First do quick dirty non-async check.
            if !epoch_store.all_external_consensus_messages_processed(
                consensus_transactions.iter().map(|tx| tx.key()),
            )? {
               
                self.consensus_adapter.submit_batch(
                    &consensus_transactions,
                    Some(&reconfiguration_lock),
                    epoch_store,
                    None,
                    None, // not tracking submitter client addr for quorum driver path
                )?;
                // Do not wait for the result, because the transaction might have already executed.
                // Instead, check or wait for the existence of certificate effects below.
            }
        }

        if !wait_for_effects {
            // It is useful to enqueue owned object transaction for execution locally,
            // even when we are not returning effects to user
            let fast_path_certificates = consensus_transactions
                .iter()
                .filter_map(|tx| {
                    if let ConsensusTransactionKind::CertifiedTransaction(certificate) = &tx.kind {
                        (!certificate.is_consensus_tx())
                            // Certificates already verified by callers of this function.
                            .then_some((
                                VerifiedExecutableTransaction::new_from_certificate(
                                    VerifiedCertificate::new_unchecked(*(certificate.clone())),
                                ),
                                ExecutionEnv::new()
                                    .with_scheduling_source(SchedulingSource::NonFastPath),
                            ))
                    } else {
                        None
                    }
                })
                .map(|(tx, env)| (Schedulable::Transaction(tx), env))
                .collect::<Vec<_>>();
            if !fast_path_certificates.is_empty() {
                self.state
                    .execution_scheduler()
                    .enqueue(fast_path_certificates, epoch_store);
            }
            return Ok((None, Weight::zero()));
        }

        // 4) Execute the certificates immediately if they contain only owned object transactions,
        // or wait for the execution results if it contains shared objects.
        let responses = futures::future::try_join_all(consensus_transactions.into_iter().map(
            |tx| async move {
                let effects = match &tx.kind {
                    ConsensusTransactionKind::CertifiedTransaction(certificate) => {
                        // Certificates already verified by callers of this function.
                        let certificate = VerifiedCertificate::new_unchecked(*(certificate.clone()));
                        self.state
                            .wait_for_certificate_execution(&certificate, epoch_store)
                            .await?
                    }
                    ConsensusTransactionKind::UserTransaction(tx) => {
                        self.state.await_transaction_effects(*tx.digest(), epoch_store).await?
                    }
                    _ => panic!("`handle_submit_to_consensus` received transaction that is not a CertifiedTransaction or UserTransaction"),
                };
               

                let input_objects = if include_input_objects {
                    self.state.get_transaction_input_objects(&effects)?
                } else {
                    vec![]
                };

                let output_objects = if include_output_objects {
                    self.state.get_transaction_output_objects(&effects)?
                } else {
                    vec![]
                };

                if let ConsensusTransactionKind::CertifiedTransaction(certificate) = &tx.kind {
                    epoch_store.insert_tx_cert_sig(certificate.digest(), certificate.auth_sig())?;
                }

                Ok::<_, SomaError>(ExecutedData {
                    effects,
               
                    input_objects,
                    output_objects,
                })
            },
        ))
        .await?;

        Ok((Some(responses), Weight::zero()))
    }

    async fn collect_effects_data(
        &self,
        effects: &TransactionEffects,
        include_input_objects: bool,
        include_output_objects: bool,
        fastpath_outputs: Option<Arc<TransactionOutputs>>,
    ) -> SomaResult<( Vec<Object>, Vec<Object>)> {
        

        let input_objects = if include_input_objects {
            self.state.get_transaction_input_objects(effects)?
        } else {
            vec![]
        };

        let output_objects = if include_output_objects {
            if let Some(fastpath_outputs) = &fastpath_outputs {
                fastpath_outputs.written.values().cloned().collect()
            } else {
                self.state.get_transaction_output_objects(effects)?
            }
        } else {
            vec![]
        };

        Ok((input_objects, output_objects))
    }
}

type WrappedServiceResponse<T> = Result<(tonic::Response<T>, Weight), tonic::Status>;

impl ValidatorService {
    async fn transaction_impl(
        &self,
        request: tonic::Request<Transaction>,
    ) -> WrappedServiceResponse<HandleTransactionResponse> {
        self.handle_transaction(request).await
    }

    async fn handle_submit_transaction_impl(
        &self,
        request: tonic::Request<RawSubmitTxRequest>,
    ) -> WrappedServiceResponse<RawSubmitTxResponse> {
        self.handle_submit_transaction(request).await
    }



    async fn handle_certificate_impl(
        &self,
        request: tonic::Request<HandleCertificateRequest>,
    ) -> WrappedServiceResponse<HandleCertificateResponse> {
        let epoch_store = self.state.load_epoch_store_one_call_per_task();
        let request = request.into_inner();

        let span = error_span!("ValidatorService::handle_certificate_v3", tx_digest = ?request.certificate.digest());
        self.handle_certificates(
            nonempty![request.certificate],
            request.include_input_objects,
            request.include_output_objects,
            &epoch_store,
            true,
        )
        .instrument(span)
        .await
        .map(|(resp, spam_weight)| {
            (
                tonic::Response::new(
                    resp.expect(
                        "handle_certificate should not return none with wait_for_effects=true",
                    )
                    .remove(0),
                ),
                spam_weight,
            )
        })
    }

    async fn wait_for_effects_impl(
        &self,
        request: tonic::Request<RawWaitForEffectsRequest>,
    ) -> WrappedServiceResponse<RawWaitForEffectsResponse> {
        let request: WaitForEffectsRequest = request.into_inner().try_into()?;
        let epoch_store = self.state.load_epoch_store_one_call_per_task();
        let response = timeout(
            // TODO(fastpath): Tune this once we have a good estimate of the typical delay.
            Duration::from_secs(20),
            epoch_store
                .within_alive_epoch(self.wait_for_effects_response(request, &epoch_store))
                .map_err(|_| SomaError::EpochEnded(epoch_store.epoch())),
        )
        .await
        .map_err(|_| tonic::Status::internal("Timeout waiting for effects"))???
        .try_into()?;
        Ok((tonic::Response::new(response), Weight::zero()))
    }

    #[instrument(name= "ValidatorService::wait_for_effects_response", level = "error", skip_all, err(level = "debug"), fields(consensus_position = ?request.consensus_position, fast_path_effects = tracing::field::Empty))]
    async fn wait_for_effects_response(
        &self,
        request: WaitForEffectsRequest,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<WaitForEffectsResponse> {
        if request.ping_type.is_some() {
            return timeout(
                Duration::from_secs(10),
                self.ping_response(request, epoch_store),
            )
            .await
            .map_err(|_| SomaError::TimeoutError)?;
        }

        let Some(tx_digest) = request.transaction_digest else {
            return Err(SomaError::InvalidRequest(
                "Transaction digest is required for wait for effects requests".to_string(),
            )
            .into());
        };
        let tx_digests = [tx_digest];

        let fastpath_effects_future: Pin<Box<dyn Future<Output = _> + Send>> =
            if let Some(consensus_position) = request.consensus_position {
                Box::pin(self.wait_for_fastpath_effects(
                    consensus_position,
                    &tx_digests,
                    request.include_details,
                    epoch_store,
                ))
            } else {
                Box::pin(futures::future::pending())
            };

        tokio::select! {
            // Ensure that finalized effects are always prioritized.
            biased;
            // We always wait for effects regardless of consensus position via
            // notify_read_executed_effects. This is safe because we have separated
            // mysticeti fastpath outputs to a separate dirty cache
            // UncommittedData::fastpath_transaction_outputs that will only get flushed
            // once finalized. So the output of notify_read_executed_effects is
            // guaranteed to be finalized effects or effects from QD execution.
            mut effects = self.state
                .get_transaction_cache_reader()
                .notify_read_executed_effects(
                    &tx_digests,
                ) => {
                tracing::Span::current().record("fast_path_effects", false);
                let effects = effects.pop().unwrap();
                let details = if request.include_details {
                    Some(self.complete_executed_data(effects.clone(), None).await?)
                } else {
                    None
                };

                Ok(WaitForEffectsResponse::Executed {
                    effects_digest: effects.digest(),
                    details,
                    fast_path: false,
                })
            }

            fastpath_response = fastpath_effects_future => {
                tracing::Span::current().record("fast_path_effects", true);
                fastpath_response
            }
        }
    }

    #[instrument(level = "error", skip_all, err(level = "debug"))]
    async fn ping_response(
        &self,
        request: WaitForEffectsRequest,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<WaitForEffectsResponse> {
        let Some(consensus_tx_status_cache) = epoch_store.consensus_tx_status_cache.as_ref() else {
            return Err(SomaError::UnsupportedFeatureError {
                error: "Mysticeti fastpath".to_string(),
            }
            .into());
        };

        let Some(consensus_position) = request.consensus_position else {
            return Err(SomaError::InvalidRequest(
                "Consensus position is required for Ping requests".to_string(),
            )
            .into());
        };

        // We assume that the caller has already checked for the existence of the `ping` field, but handling it gracefully here.
        let Some(ping) = request.ping_type else {
            return Err(SomaError::InvalidRequest(
                "Ping type is required for ping requests".to_string(),
            )
            .into());
        };

       

        consensus_tx_status_cache.check_position_too_ahead(&consensus_position)?;

        let mut last_status = None;
        let details = if request.include_details {
            Some(Box::new(ExecutedData::default()))
        } else {
            None
        };

        loop {
            let status = consensus_tx_status_cache
                .notify_read_transaction_status_change(consensus_position, last_status)
                .await;
            match status {
                NotifyReadConsensusTxStatusResult::Status(status) => match status {
                    ConsensusTxStatus::FastpathCertified => {
                        // If the request is for consensus, we need to wait for the transaction to be finalised via Consensus.
                        if ping == PingType::Consensus {
                            last_status = Some(status);
                            continue;
                        }
                        return Ok(WaitForEffectsResponse::Executed {
                            effects_digest: TransactionEffectsDigest::ZERO,
                            details,
                            fast_path: true,
                        });
                    }
                    ConsensusTxStatus::Rejected => {
                        return Ok(WaitForEffectsResponse::Rejected { error: None });
                    }
                    ConsensusTxStatus::Finalized => {
                        return Ok(WaitForEffectsResponse::Executed {
                            effects_digest: TransactionEffectsDigest::ZERO,
                            details,
                            fast_path: false,
                        });
                    }
                },
                NotifyReadConsensusTxStatusResult::Expired(round) => {
                    return Ok(WaitForEffectsResponse::Expired {
                        epoch: epoch_store.epoch(),
                        round: Some(round),
                    });
                }
            }
        }
    }

    #[instrument(level = "error", skip_all, err(level = "debug"))]
    async fn wait_for_fastpath_effects(
        &self,
        consensus_position: ConsensusPosition,
        tx_digests: &[TransactionDigest],
        include_details: bool,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<WaitForEffectsResponse> {
        let Some(consensus_tx_status_cache) = epoch_store.consensus_tx_status_cache.as_ref() else {
            return Err(SomaError::UnsupportedFeatureError {
                error: "Mysticeti fastpath".to_string(),
            }
            .into());
        };

        let local_epoch = epoch_store.epoch();
        match consensus_position.epoch.cmp(&local_epoch) {
            Ordering::Less => {
                // Ask TransactionDriver to retry submitting the transaction and get a new ConsensusPosition,
                // if response from this validator is desired.
                let response = WaitForEffectsResponse::Expired {
                    epoch: local_epoch,
                    round: None,
                };
                return Ok(response);
            }
            Ordering::Greater => {
                // Ask TransactionDriver to retry this RPC until the validator's epoch catches up.
                return Err(SomaError::WrongEpoch {
                    expected_epoch: local_epoch,
                    actual_epoch: consensus_position.epoch,
                }
                .into());
            }
            Ordering::Equal => {
                // The validator's epoch is the same as the epoch of the transaction.
                // We can proceed with the normal flow.
            }
        };

        consensus_tx_status_cache.check_position_too_ahead(&consensus_position)?;

        let mut current_status = None;
        loop {
            tokio::select! {
                status_result = consensus_tx_status_cache
                    .notify_read_transaction_status_change(consensus_position, current_status) => {
                    match status_result {
                        NotifyReadConsensusTxStatusResult::Status(new_status) => {
                            match new_status {
                                ConsensusTxStatus::Rejected => {
                                    return Ok(WaitForEffectsResponse::Rejected {
                                        error: epoch_store.get_rejection_vote_reason(
                                            consensus_position
                                        )
                                    });
                                }
                                ConsensusTxStatus::FastpathCertified => {
                                    current_status = Some(new_status);
                                    continue;
                                }
                                ConsensusTxStatus::Finalized => {
                                    current_status = Some(new_status);
                                    continue;
                                }
                            }
                        }
                        NotifyReadConsensusTxStatusResult::Expired(round) => {
                            return Ok(WaitForEffectsResponse::Expired {
                                epoch: epoch_store.epoch(),
                                round: Some(round),
                            });
                        }
                    }
                }

                mut outputs = self.state.get_transaction_cache_reader().notify_read_fastpath_transaction_outputs(tx_digests),
                    if current_status == Some(ConsensusTxStatus::FastpathCertified) || current_status == Some(ConsensusTxStatus::Finalized) => {
                    let outputs = outputs.pop().unwrap();
                    let effects = outputs.effects.clone();

                    let details = if include_details {
                        Some(self.complete_executed_data(effects.clone(), Some(outputs)).await?)
                    } else {
                        None
                    };

                    return Ok(WaitForEffectsResponse::Executed {
                        effects_digest: effects.digest(),
                        details,
                        fast_path: current_status == Some(ConsensusTxStatus::FastpathCertified),
                    });
                }
            }
        }
    }

    async fn complete_executed_data(
        &self,
        effects: TransactionEffects,
        fastpath_outputs: Option<Arc<TransactionOutputs>>,
    ) -> SomaResult<Box<ExecutedData>> {
        let (input_objects, output_objects) = self
            .collect_effects_data(
                &effects,
                /* include_input_objects */ true,
                /* include_output_objects */ true,
                fastpath_outputs,
            )
            .await?;
        Ok(Box::new(ExecutedData {
            effects,
            input_objects,
            output_objects,
        }))
    }

    async fn object_info_impl(
        &self,
        request: tonic::Request<ObjectInfoRequest>,
    ) -> WrappedServiceResponse<ObjectInfoResponse> {
        let request = request.into_inner();
        let response = self.state.handle_object_info_request(request).await?;
        Ok((tonic::Response::new(response), Weight::one()))
    }

    async fn transaction_info_impl(
        &self,
        request: tonic::Request<TransactionInfoRequest>,
    ) -> WrappedServiceResponse<TransactionInfoResponse> {
        let request = request.into_inner();
        let response = self.state.handle_transaction_info_request(request).await?;
        Ok((tonic::Response::new(response), Weight::one()))
    }

    async fn checkpoint_impl(
        &self,
        request: tonic::Request<CheckpointRequest>,
    ) -> WrappedServiceResponse<CheckpointResponse> {
        let request = request.into_inner();
        let response = self.state.handle_checkpoint_request(&request)?;
        Ok((tonic::Response::new(response), Weight::one()))
    }


    async fn get_system_state_object_impl(
        &self,
        _request: tonic::Request<SystemStateRequest>,
    ) -> WrappedServiceResponse<SystemState> {
        let response = self
            .state
            .get_object_cache_reader()
            .get_system_state_object()?;
        Ok((tonic::Response::new(response), Weight::one()))
    }

     async fn validator_health_impl(
        &self,
        _request: tonic::Request<types::messages_grpc::RawValidatorHealthRequest>,
    ) -> WrappedServiceResponse<types::messages_grpc::RawValidatorHealthResponse> {
        let state = &self.state;

        // Get epoch store once for both metrics
        let epoch_store = state.load_epoch_store_one_call_per_task();

        // Get in-flight execution transactions from execution scheduler
        // let num_inflight_execution_transactions =
        //     state.execution_scheduler().num_pending_certificates() as u64;

        // Get in-flight consensus transactions from consensus adapter
        let num_inflight_consensus_transactions =
            self.consensus_adapter.num_inflight_transactions();

        // Get last committed leader round from epoch store
        let last_committed_leader_round = epoch_store
            .consensus_tx_status_cache
            .as_ref()
            .and_then(|cache| cache.get_last_committed_leader_round())
            .unwrap_or(0);

        // Get last locally built checkpoint sequence
        let last_locally_built_checkpoint = epoch_store
            .last_built_checkpoint_summary()
            .ok()
            .flatten()
            .map(|(_, summary)| summary.sequence_number)
            .unwrap_or(0);

        let typed_response = types::messages_grpc::ValidatorHealthResponse {
            num_inflight_consensus_transactions,
            // num_inflight_execution_transactions,
            last_locally_built_checkpoint,
            last_committed_leader_round,
        };

        let raw_response = typed_response
            .try_into()
            .map_err(|e: types::error::SomaError| {
                tonic::Status::internal(format!("Failed to serialize health response: {}", e))
            })?;

        Ok((tonic::Response::new(raw_response), Weight::one()))
    }

    
    fn get_client_ip_addr<T>(
        &self,
        request: &tonic::Request<T>,
        source: &ClientIdSource,
    ) -> Option<IpAddr> {
        let forwarded_header = request.metadata().get_all("x-forwarded-for").iter().next();

        if let Some(header) = forwarded_header {
            let num_hops = header
                .to_str()
                .map(|h| h.split(',').count().saturating_sub(1))
                .unwrap_or(0);

            
        }

        match source {
            ClientIdSource::SocketAddr => {
                let socket_addr: Option<SocketAddr> = request.remote_addr();

                // We will hit this case if the IO type used does not
                // implement Connected or when using a unix domain socket.
                // TODO: once we have confirmed that no legitimate traffic
                // is hitting this case, we should reject such requests that
                // hit this case.
                if let Some(socket_addr) = socket_addr {
                    Some(socket_addr.ip())
                } else {
                    if cfg!(msim) {
                        // Ignore the error from simtests.
                    } else if cfg!(test) {
                        panic!("Failed to get remote address from request");
                    } else {
                        
                        error!("Failed to get remote address from request");
                    }
                    None
                }
            }
            ClientIdSource::XForwardedFor(num_hops) => {
                let do_header_parse = |op: &MetadataValue<Ascii>| {
                    match op.to_str() {
                        Ok(header_val) => {
                            let header_contents =
                                header_val.split(',').map(str::trim).collect::<Vec<_>>();
                            if *num_hops == 0 {
                                error!(
                                    "x-forwarded-for: 0 specified. x-forwarded-for contents: {:?}. Please assign nonzero value for \
                                    number of hops here, or use `socket-addr` client-id-source type if requests are not being proxied \
                                    to this node. Skipping traffic controller request handling.",
                                    header_contents,
                                );
                                return None;
                            }
                            let contents_len = header_contents.len();
                            if contents_len < *num_hops {
                                error!(
                                    "x-forwarded-for header value of {:?} contains {} values, but {} hops were specified. \
                                    Expected at least {} values. Please correctly set the `x-forwarded-for` value under \
                                    `client-id-source` in the node config.",
                                    header_contents, contents_len, num_hops, contents_len,
                                );
                                
                                return None;
                            }
                            let Some(client_ip) = header_contents.get(contents_len - num_hops)
                            else {
                                error!(
                                    "x-forwarded-for header value of {:?} contains {} values, but {} hops were specified. \
                                    Expected at least {} values. Skipping traffic controller request handling.",
                                    header_contents, contents_len, num_hops, contents_len,
                                );
                                return None;
                            };
                            Self::parse_ip(client_ip).or_else(|| {
                                
                                None
                            })
                        }
                        Err(e) => {
                            // TODO: once we have confirmed that no legitimate traffic
                            // is hitting this case, we should reject such requests that
                            // hit this case.
                           
                            error!("Invalid UTF-8 in x-forwarded-for header: {:?}", e);
                            None
                        }
                    }
                };
                if let Some(op) = request.metadata().get("x-forwarded-for") {
                    do_header_parse(op)
                } else if let Some(op) = request.metadata().get("X-Forwarded-For") {
                    do_header_parse(op)
                } else {
                   
                    error!(
                        "x-forwarded-for header not present for request despite node configuring x-forwarded-for tracking type"
                    );
                    None
                }
            }
        }
    }

    // TODO: remove this after implementing traffic controller
    pub fn parse_ip(ip: &str) -> Option<IpAddr> {
        ip.parse::<IpAddr>().ok().or_else(|| {
            ip.parse::<SocketAddr>()
                .ok()
                .map(|socket_addr| socket_addr.ip())
                .or_else(|| {
                    error!("Failed to parse value of {:?} to ip address or socket.", ip,);
                    None
                })
        })
    }

    // async fn handle_traffic_req(&self, client: Option<IpAddr>) -> Result<(), tonic::Status> {
    //     if let Some(traffic_controller) = &self.traffic_controller {
    //         if !traffic_controller.check(&client, &None).await {
    //             // Entity in blocklist
    //             Err(tonic::Status::from_error(
    //                 SomaError::TooManyRequests.into(),
    //             ))
    //         } else {
    //             Ok(())
    //         }
    //     } else {
    //         Ok(())
    //     }
    // }

    fn handle_traffic_resp<T>(
        &self,
        client: Option<IpAddr>,
        wrapped_response: WrappedServiceResponse<T>,
    ) -> Result<tonic::Response<T>, tonic::Status> {
        let (error, spam_weight, unwrapped_response) = match wrapped_response {
            Ok((result, spam_weight)) => (None, spam_weight.clone(), Ok(result)),
            Err(status) => (
                Some(SomaError::from(status.clone())),
                Weight::zero(),
                Err(status.clone()),
            ),
        };

        // TODO: implement tallying with traffic controller
        // if let Some(traffic_controller) = self.traffic_controller.clone() {
        //     traffic_controller.tally(TrafficTally {
        //         direct: client,
        //         through_fullnode: None,
        //         error_info: error.map(|e| {
        //             let error_type = String::from(e.clone().as_ref());
        //             let error_weight = normalize(e);
        //             (error_weight, error_type)
        //         }),
        //         spam_weight,
        //         timestamp: SystemTime::now(),
        //     })
        // }
        unwrapped_response
    }
}

fn make_tonic_request_for_testing<T>(message: T) -> tonic::Request<T> {
    // simulate a TCP connection, which would have added extensions to
    // the request object that would be used downstream
    let mut request = tonic::Request::new(message);
    let tcp_connect_info = TcpConnectInfo {
        local_addr: None,
        remote_addr: Some(SocketAddr::new([127, 0, 0, 1].into(), 0)),
    };
    request.extensions_mut().insert(tcp_connect_info);
    request
}

// TODO: refine error matching here
fn normalize(err: SomaError) -> Weight {
    match err {
        SomaError::IncorrectUserSignature { .. } => Weight::one(),
        SomaError::InvalidSignature { .. }
        | SomaError::SignerSignatureAbsent { .. }
        | SomaError::SignerSignatureNumberMismatch { .. }
        | SomaError::IncorrectSigner { .. }
        | SomaError::UnknownSigner { .. }
        | SomaError::WrongEpoch { .. } => Weight::one(),
        _ => Weight::zero(),
    }
}

/// Implements generic pre- and post-processing. Since this is on the critical
/// path, any heavy lifting should be done in a separate non-blocking task
/// unless it is necessary to override the return value.
// #[macro_export]
// macro_rules! handle_with_decoration {
//     ($self:ident, $func_name:ident, $request:ident) => {{
//         if $self.client_id_source.is_none() {
//             return $self.$func_name($request).await.map(|(result, _)| result);
//         }

//         let client = $self.get_client_ip_addr(&$request, $self.client_id_source.as_ref().unwrap());

//         // check if either IP is blocked, in which case return early
//         $self.handle_traffic_req(client.clone()).await?;

//         // handle traffic tallying
//         let wrapped_response = $self.$func_name($request).await;
//         $self.handle_traffic_resp(client, wrapped_response)
//     }};
// }

#[async_trait]
impl Validator for ValidatorService {
    async fn submit_transaction(
        &self,
        request: tonic::Request<RawSubmitTxRequest>,
    ) -> Result<tonic::Response<RawSubmitTxResponse>, tonic::Status> {
        let validator_service = self.clone();

        // Spawns a task which handles the transaction. The task will unconditionally continue
        // processing in the event that the client connection is dropped.
        tokio::spawn(async move {
            // NB: traffic tally wrapping handled within the task rather than on task exit
            // to prevent an attacker from subverting traffic control by severing the connection
            // handle_with_decoration!(validator_service, handle_submit_transaction_impl, request)

            let wrapped_response = validator_service.handle_submit_transaction_impl(request).await;
            validator_service.handle_traffic_resp(None, wrapped_response)
        })
        .await
        .unwrap()
    }

    async fn transaction(
        &self,
        request: tonic::Request<Transaction>,
    ) -> Result<tonic::Response<HandleTransactionResponse>, tonic::Status> {
        let validator_service = self.clone();

        // Spawns a task which handles the transaction. The task will unconditionally continue
        // processing in the event that the client connection is dropped.
        tokio::spawn(async move {
            // NB: traffic tally wrapping handled within the task rather than on task exit
            // to prevent an attacker from subverting traffic control by severing the connection
            // handle_with_decoration!(validator_service, transaction_impl, request)

            let wrapped_response = validator_service.transaction_impl(request).await;
            validator_service.handle_traffic_resp(None, wrapped_response)
        })
        .await
        .unwrap()
    }


    async fn handle_certificate(
        &self,
        request: tonic::Request<HandleCertificateRequest>,
    ) -> Result<tonic::Response<HandleCertificateResponse>, tonic::Status> {
        let wrapped_response = self.handle_certificate_impl(request).await;
        self.handle_traffic_resp(None, wrapped_response)
    }

    async fn wait_for_effects(
        &self,
        request: tonic::Request<RawWaitForEffectsRequest>,
    ) -> Result<tonic::Response<RawWaitForEffectsResponse>, tonic::Status> {
        let wrapped_response = self.wait_for_effects_impl(request).await;
        self.handle_traffic_resp(None, wrapped_response)
    }

    async fn object_info(
        &self,
        request: tonic::Request<ObjectInfoRequest>,
    ) -> Result<tonic::Response<ObjectInfoResponse>, tonic::Status> {
        let wrapped_response = self.object_info_impl(request).await;
        self.handle_traffic_resp(None, wrapped_response)
    }

    async fn transaction_info(
        &self,
        request: tonic::Request<TransactionInfoRequest>,
    ) -> Result<tonic::Response<TransactionInfoResponse>, tonic::Status> {
        let wrapped_response = self.transaction_info_impl(request).await;
        self.handle_traffic_resp(None, wrapped_response)
    }

    async fn checkpoint(
        &self,
        request: tonic::Request<CheckpointRequest>,
    ) -> Result<tonic::Response<CheckpointResponse>, tonic::Status> {
        let wrapped_response = self.checkpoint_impl(request).await;
        self.handle_traffic_resp(None, wrapped_response)
    }

    async fn get_system_state_object(
        &self,
        request: tonic::Request<SystemStateRequest>,
    ) -> Result<tonic::Response<SystemState>, tonic::Status> {
        let wrapped_response = self.get_system_state_object_impl(request).await;
        self.handle_traffic_resp(None, wrapped_response)
    }

    async fn validator_health(
        &self,
        request: tonic::Request<types::messages_grpc::RawValidatorHealthRequest>,
    ) -> Result<tonic::Response<types::messages_grpc::RawValidatorHealthResponse>, tonic::Status>
    {
        let wrapped_response = self.validator_health_impl(request).await;
        self.handle_traffic_resp(None, wrapped_response)
    }
}
