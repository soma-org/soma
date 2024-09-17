use crate::{
    adapter::ConsensusAdapter, epoch_store::AuthorityPerEpochStore, state::AuthorityState,
    tonic_gen::validator_server::Validator,
};
use nonempty::nonempty;
use nonempty::NonEmpty;
use std::sync::Arc;
use tonic::{async_trait, Response};
use tracing::{error_span, info, Instrument};
use types::{
    consensus::ConsensusTransaction,
    error::SomaError,
    grpc::{
        HandleCertificateRequest, HandleCertificateResponse, HandleTransactionResponse,
        SubmitCertificateResponse,
    },
    transaction::{CertifiedTransaction, Transaction},
};

#[derive(Clone)]
pub struct ValidatorService {
    state: Arc<AuthorityState>,
    consensus_adapter: Arc<ConsensusAdapter>,
    // traffic_controller: Option<Arc<TrafficController>>,
    // client_id_source: Option<ClientIdSource>,
}

impl ValidatorService {
    pub fn new(state: Arc<AuthorityState>, consensus_adapter: Arc<ConsensusAdapter>) -> Self {
        Self {
            state,
            consensus_adapter,
        }
    }

    pub fn validator_state(&self) -> &Arc<AuthorityState> {
        &self.state
    }

    async fn handle_transaction(
        &self,
        request: tonic::Request<Transaction>,
    ) -> Result<Response<HandleTransactionResponse>, tonic::Status> {
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
        //     &consensus_adapter,
        //     transaction.data(),
        //     state.check_system_overload_at_signing(),
        // );
        // if let Err(error) = overload_check_res {
        //     // TODO: consider change the behavior for other types of overload errors.
        //     match error {
        //         SomaError::ValidatorOverloadedRetryAfter { .. } => {
        //             validator_pushback_error = Some(error)
        //         }
        //         _ => return Err(error.into()),
        //     }
        // }

        let transaction = epoch_store.verify_transaction(transaction)?;

        let tx_digest = transaction.digest();

        // Enable Trace Propagation across spans/processes using tx_digest
        let span = error_span!("validator_state_process_tx", ?tx_digest);

        let info = state
            .handle_transaction(&epoch_store, transaction.clone())
            .instrument(span)
            .await?;

        // if let Some(error) = validator_pushback_error {
        //     // TODO: right now, we still sign the txn, but just don't return it. We can also skip signing
        //     // to save more CPU.
        //     return Err(error.into());
        // }

        Ok(tonic::Response::new(info))
    }

    // In addition to the response from handling the certificates,
    // returns a bool indicating whether the request should be tallied
    // toward spam count. In general, this should be set to true for
    // requests that are read-only and thus do not consume gas, such
    // as when the transaction is already executed.
    async fn handle_certificates(
        &self,
        certificates: NonEmpty<CertifiedTransaction>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        wait_for_effects: bool,
    ) -> Result<Option<Vec<HandleCertificateResponse>>, tonic::Status> {
        // Todo: Validate if cert can be executed
        // Fullnode does not serve handle_certificate call.
        // fp_ensure!(
        //     !self.state.is_fullnode(epoch_store),
        //     SuiError::FullNodeCantHandleCertificate.into()
        // );

        // 1) Check if the certificate is already executed.
        //    This is only needed when we have only one certificate (not a soft bundle).
        //    When multiple certificates are provided, we will either submit all of them or none of them to consensus.
        // if certificates.len() == 1 {
        //     let tx_digest = *certificates[0].digest();

        //     // if let Some(signed_effects) = self
        //     //     .state
        //     //     .get_signed_effects_and_maybe_resign(&tx_digest, epoch_store)?
        //     // {
        //     //     return Ok(Some(vec![HandleCertificateResponse {
        //     //             // effects: signed_effects.into_inner(),

        //     //         }]));
        //     // };

        //     return Ok(Some(vec![HandleCertificateResponse {
        //         // effects: signed_effects.into_inner(),

        //     }]));
        // }

        // 2) Verify the certificates.
        // Check system overload
        // for certificate in &certificates {
        //     let overload_check_res = self.state.check_system_overload(
        //         &self.consensus_adapter,
        //         certificate.data(),
        //         self.state.check_system_overload_at_execution(),
        //     );
        //     if let Err(error) = overload_check_res {
        //         return Err(error.into());
        //     }
        // }

        let verified_certificates = {
            epoch_store
                .signature_verifier
                .multi_verify_certs(certificates.into())
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
        };

        {
            // code block within reconfiguration lock
            let reconfiguration_lock = epoch_store.get_reconfig_state_read_lock_guard();
            if !reconfiguration_lock.should_accept_user_certs() {
                return Err(SomaError::ValidatorHaltedAtEpochEnd.into());
            }

            // 3) All certificates are sent to consensus (at least by some authorities)
            // For shared objects this will wait until either timeout or we have heard back from consensus.
            // For owned objects this will return without waiting for certificate to be sequenced
            // First do quick dirty non-async check.
            if !epoch_store
                .is_all_tx_certs_consensus_message_processed(verified_certificates.iter())?
            {
                let transactions = verified_certificates
                    .iter()
                    .map(|certificate| {
                        ConsensusTransaction::new_certificate_message(
                            &self.state.name,
                            certificate.clone().into(),
                        )
                    })
                    .collect::<Vec<_>>();
                self.consensus_adapter.submit_batch(
                    &transactions,
                    Some(&reconfiguration_lock),
                    epoch_store,
                )?;
                // Do not wait for the result, because the transaction might have already executed.
                // Instead, check or wait for the existence of certificate effects below.
            }
        }

        if !wait_for_effects {
            // It is useful to enqueue owned object transaction for execution locally,
            // even when we are not returning effects to user
            let certificates_without_shared_objects =
                verified_certificates.iter().cloned().collect::<Vec<_>>();
            if !certificates_without_shared_objects.is_empty() {
                self.state.enqueue_certificates_for_execution(
                    certificates_without_shared_objects,
                    epoch_store,
                );
            }
            return Ok(None);
        }

        // 4) Execute the certificates immediately if they contain only owned object transactions,
        // or wait for the execution results if it contains shared objects.
        let responses = futures::future::try_join_all(verified_certificates.into_iter().map(
            |certificate| async move {
                let effects = self
                    .state
                    .execute_certificate(&certificate, epoch_store)
                    .await?;

                // let signed_effects = self.state.sign_effects(effects, epoch_store)?;
                // epoch_store.insert_tx_cert_sig(certificate.digest(), certificate.auth_sig())?;

                Ok::<_, SomaError>(HandleCertificateResponse {
                    // effects: signed_effects.into_inner(),
                    succeeded: true,
                })
            },
        ))
        .await?;

        Ok(Some(responses))
    }
}

impl ValidatorService {
    async fn submit_certificate_impl(
        &self,
        request: tonic::Request<CertifiedTransaction>,
    ) -> Result<tonic::Response<SubmitCertificateResponse>, tonic::Status> {
        let epoch_store = self.state.load_epoch_store_one_call_per_task();
        let certificate = request.into_inner();

        let span = error_span!("submit_certificate", tx_digest = ?certificate.digest());
        self.handle_certificates(nonempty![certificate], &epoch_store, false)
            .instrument(span)
            .await
            .map(|executed| {
                tonic::Response::new(SubmitCertificateResponse {
                    executed: executed.map(|mut x| x.remove(0)).map(Into::into),
                })
            })
    }

    async fn handle_certificate_impl(
        &self,
        request: tonic::Request<HandleCertificateRequest>,
    ) -> Result<tonic::Response<HandleCertificateResponse>, tonic::Status> {
        let epoch_store = self.state.load_epoch_store_one_call_per_task();
        let request = request.into_inner();

        let span = error_span!("handle_certificate", tx_digest = ?request.certificate.digest());
        self.handle_certificates(nonempty![request.certificate], &epoch_store, true)
            .instrument(span)
            .await
            .map(|resp| {
                tonic::Response::new(
                    resp.expect(
                        "handle_certificate should not return none with wait_for_effects=true",
                    )
                    .remove(0),
                )
            })
    }
}

#[async_trait]
impl Validator for ValidatorService {
    async fn transaction(
        &self,
        request: tonic::Request<Transaction>,
    ) -> Result<tonic::Response<HandleTransactionResponse>, tonic::Status> {
        self.handle_transaction(request).await
    }

    async fn submit_certificate(
        &self,
        request: tonic::Request<CertifiedTransaction>,
    ) -> Result<tonic::Response<SubmitCertificateResponse>, tonic::Status> {
        self.submit_certificate_impl(request).await
    }

    async fn handle_certificate(
        &self,
        request: tonic::Request<HandleCertificateRequest>,
    ) -> Result<tonic::Response<HandleCertificateResponse>, tonic::Status> {
        self.handle_certificate_impl(request).await
    }
}
