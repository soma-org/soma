use std::collections::HashMap;
use std::{net::SocketAddr, sync::Arc};

use crate::authority_client::AuthorityAPI;
use tap::TapFallible as _;
use tracing::{debug, error, instrument};
use types::base::ConciseableName as _;
use types::checkpoints::{
    CertifiedCheckpointSummary, CheckpointRequest, CheckpointResponse, CheckpointSequenceNumber,
    CheckpointSummaryResponse,
};
use types::crypto::AuthoritySignInfoTrait;
use types::intent::Intent;
use types::messages_grpc::{
    ExecutedData, ObjectInfoRequest, ObjectInfoResponse, SubmitTxRequest, SubmitTxResponse,
    SystemStateRequest, TransactionInfoRequest, ValidatorHealthRequest, ValidatorHealthResponse,
    VerifiedObjectInfoResponse, WaitForEffectsRequest, WaitForEffectsResponse,
};
use types::object::{Object, ObjectID, ObjectRef};
use types::storage::committee_store::CommitteeStore;
use types::system_state::SystemState;
use types::{
    committee::{Committee, EpochId},
    crypto::AuthorityPublicKeyBytes,
    digests::{TransactionDigest, TransactionEffectsDigest},
    effects::{self, SignedTransactionEffects, TransactionEffectsAPI},
    error::{SomaError, SomaResult},
    messages_grpc::{HandleCertificateRequest, HandleCertificateResponse, TransactionStatus},
    quorum_driver::PlainTransactionInfoResponse,
    transaction::{CertifiedTransaction, SignedTransaction, Transaction},
};

macro_rules! check_error {
    ($address:expr, $cond:expr, $msg:expr) => {
        $cond.tap_err(|err| {
            if err.individual_error_indicates_epoch_change() {
                debug!(?err, authority=?$address, "Not a real client error");
            } else {
                error!(?err, authority=?$address, $msg);
            }
        })
    }
}
#[derive(Clone)]
pub struct SafeClient<C>
where
    C: Clone,
{
    authority_client: C,
    committee_store: Arc<CommitteeStore>,
    address: AuthorityPublicKeyBytes,
}

impl<C: Clone> SafeClient<C> {
    pub fn new(
        authority_client: C,
        committee_store: Arc<CommitteeStore>,
        address: AuthorityPublicKeyBytes,
    ) -> Self {
        Self {
            authority_client,
            committee_store,
            address,
        }
    }
}

impl<C: Clone> SafeClient<C> {
    pub fn authority_client(&self) -> &C {
        &self.authority_client
    }

    #[cfg(test)]
    pub fn authority_client_mut(&mut self) -> &mut C {
        &mut self.authority_client
    }

    fn get_committee(&self, epoch_id: &EpochId) -> SomaResult<Arc<Committee>> {
        self.committee_store
            .get_committee(epoch_id)?
            .ok_or(SomaError::MissingCommitteeAtEpoch(*epoch_id).into())
    }

    fn check_signed_effects_plain(
        &self,
        digest: &TransactionDigest,
        signed_effects: SignedTransactionEffects,
        expected_effects_digest: Option<&TransactionEffectsDigest>,
    ) -> SomaResult<SignedTransactionEffects> {
        // Check it has the right signer
        if !(signed_effects.auth_sig().authority == self.address) {
            return Err(SomaError::ByzantineAuthoritySuspicion {
                authority: self.address,
                reason: format!(
                    "Unexpected validator address in the signed effects signature: {:?}",
                    signed_effects.auth_sig().authority
                ),
            }
            .into());
        }

        // Checks it concerns the right tx
        if !(signed_effects.data().transaction_digest() == digest) {
            return Err(SomaError::ByzantineAuthoritySuspicion {
                authority: self.address,
                reason: "Unexpected tx digest in the signed effects".to_string(),
            }
            .into());
        }

        // check that the effects digest is correct.
        if let Some(effects_digest) = expected_effects_digest {
            if !(signed_effects.digest() == effects_digest) {
                return Err(SomaError::ByzantineAuthoritySuspicion {
                    authority: self.address,
                    reason: "Effects digest does not match with expected digest".to_string(),
                }
                .into());
            }
        }
        self.get_committee(&signed_effects.epoch())?;
        Ok(signed_effects)
    }

    fn check_transaction_info(
        &self,
        digest: &TransactionDigest,
        transaction: Transaction,
        status: TransactionStatus,
    ) -> SomaResult<PlainTransactionInfoResponse> {
        if !(digest == transaction.digest()) {
            return Err(SomaError::ByzantineAuthoritySuspicion {
                authority: self.address,
                reason: "Signed transaction digest does not match with expected digest".to_string(),
            }
            .into());
        }

        match status {
            TransactionStatus::Signed(signed) => {
                self.get_committee(&signed.epoch)?;
                Ok(PlainTransactionInfoResponse::Signed(
                    SignedTransaction::new_from_data_and_sig(transaction.into_data(), signed),
                ))
            }
            TransactionStatus::Executed(cert_opt, effects) => {
                let signed_effects = self.check_signed_effects_plain(digest, effects, None)?;
                match cert_opt {
                    Some(cert) => {
                        let committee = self.get_committee(&cert.epoch)?;
                        let ct = CertifiedTransaction::new_from_data_and_sig(
                            transaction.into_data(),
                            cert,
                        );
                        ct.verify_committee_sigs_only(&committee).map_err(|e| {
                            SomaError::FailedToVerifyTxCertWithExecutedEffects {
                                validator_name: self.address,
                                error: e.to_string(),
                            }
                        })?;
                        Ok(PlainTransactionInfoResponse::ExecutedWithCert(
                            ct,
                            signed_effects,
                        ))
                    }
                    None => Ok(PlainTransactionInfoResponse::ExecutedWithoutCert(
                        transaction,
                        signed_effects,
                    )),
                }
            }
        }
    }

    fn check_object_response(
        &self,
        request: &ObjectInfoRequest,
        response: ObjectInfoResponse,
    ) -> SomaResult<VerifiedObjectInfoResponse> {
        let ObjectInfoResponse { object } = response;

        if !(request.object_id == object.id()) {
            return Err(SomaError::ByzantineAuthoritySuspicion {
                authority: self.address,
                reason: "Object id mismatch in the response".to_string(),
            }
            .into());
        }

        Ok(VerifiedObjectInfoResponse { object })
    }

    pub fn address(&self) -> &AuthorityPublicKeyBytes {
        &self.address
    }
}

impl<C> SafeClient<C>
where
    C: AuthorityAPI + Send + Sync + Clone + 'static,
{
    /// Submit a transaction for certification and execution.
    pub async fn submit_transaction(
        &self,
        request: SubmitTxRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<SubmitTxResponse, SomaError> {
        self.authority_client
            .submit_transaction(request, client_addr)
            .await
    }

    /// Wait for effects of a transaction that has been submitted to the network
    /// through the `submit_transaction` API.
    pub async fn wait_for_effects(
        &self,
        request: WaitForEffectsRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<WaitForEffectsResponse, SomaError> {
        let wait_for_effects_resp = self
            .authority_client
            .wait_for_effects(request, client_addr)
            .await?;

        match &wait_for_effects_resp {
            WaitForEffectsResponse::Executed {
                effects_digest: _,
                fast_path: _,
                details: Some(details),
            } => {
                self.verify_executed_data((**details).clone())?;
            }
            _ => {
                // No additional verification needed for other response types
            }
        };

        Ok(wait_for_effects_resp)
    }

    /// Initiate a new transfer to a Sui or Primary account.
    pub async fn handle_transaction(
        &self,
        transaction: Transaction,
        client_addr: Option<SocketAddr>,
    ) -> Result<PlainTransactionInfoResponse, SomaError> {
        let digest = *transaction.digest();
        let response = self
            .authority_client
            .handle_transaction(transaction.clone(), client_addr)
            .await?;
        let response = check_error!(
            self.address,
            self.check_transaction_info(&digest, transaction, response.status),
            "Client error in handle_transaction"
        )?;
        Ok(response)
    }

    fn verify_objects<I>(&self, objects: &Option<Vec<Object>>, expected_refs: I) -> SomaResult
    where
        I: IntoIterator<Item = (ObjectID, ObjectRef)>,
    {
        if let Some(objects) = objects {
            let expected: HashMap<_, _> = expected_refs.into_iter().collect();

            for object in objects {
                let object_ref = object.compute_object_reference();
                if expected
                    .get(&object_ref.0)
                    .is_none_or(|expect| &object_ref != expect)
                {
                    return Err(SomaError::ByzantineAuthoritySuspicion {
                        authority: self.address,
                        reason: "Returned object that wasn't present in effects".to_string(),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    fn verify_certificate_response(
        &self,
        digest: &TransactionDigest,
        HandleCertificateResponse {
            effects,
            input_objects,
            output_objects,
        }: HandleCertificateResponse,
    ) -> SomaResult<HandleCertificateResponse> {
        let effects = self.check_signed_effects_plain(digest, effects, None)?;

        // Check Input Objects
        self.verify_objects(
            &input_objects,
            effects
                .old_object_metadata()
                .into_iter()
                .map(|(object_ref, _owner)| (object_ref.0, object_ref)),
        )?;

        // Check Output Objects
        self.verify_objects(
            &output_objects,
            effects
                .all_changed_objects()
                .into_iter()
                .map(|(object_ref, _, _)| (object_ref.0, object_ref)),
        )?;

        Ok(HandleCertificateResponse {
            effects,
            input_objects,
            output_objects,
        })
    }

    fn verify_executed_data(
        &self,
        ExecutedData {
            effects,
            input_objects,
            output_objects,
        }: ExecutedData,
    ) -> SomaResult<()> {
        // Check Input Objects
        self.verify_objects(
            &Some(input_objects).filter(|v| !v.is_empty()),
            effects
                .old_object_metadata()
                .into_iter()
                .map(|(object_ref, _owner)| (object_ref.0, object_ref)),
        )?;

        // Check Output Objects
        self.verify_objects(
            &Some(output_objects).filter(|v| !v.is_empty()),
            effects
                .all_changed_objects()
                .into_iter()
                .map(|(object_ref, _, _)| (object_ref.0, object_ref)),
        )?;

        Ok(())
    }

    /// Execute a certificate.
    pub async fn handle_certificate(
        &self,
        request: HandleCertificateRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<HandleCertificateResponse, SomaError> {
        let digest = *request.certificate.digest();

        let response = self
            .authority_client
            .handle_certificate(request, client_addr)
            .await?;

        let verified = check_error!(
            self.address,
            self.verify_certificate_response(&digest, response),
            "Client error in handle_certificate"
        )?;
        Ok(verified)
    }

    pub async fn handle_object_info_request(
        &self,
        request: ObjectInfoRequest,
    ) -> Result<VerifiedObjectInfoResponse, SomaError> {
        let response = self
            .authority_client
            .handle_object_info_request(request.clone())
            .await?;
        let response = self
            .check_object_response(&request, response)
            .tap_err(|err| error!(?err, authority=?self.address, "Client error in handle_object_info_request"))?;

        Ok(response)
    }

    /// Handle Transaction information requests for a given digest.
    /// Only used for testing.
    #[instrument(level = "trace", skip_all, fields(authority = ?self.address.concise()))]
    pub async fn handle_transaction_info_request(
        &self,
        request: TransactionInfoRequest,
    ) -> Result<PlainTransactionInfoResponse, SomaError> {
        let transaction_info = self
            .authority_client
            .handle_transaction_info_request(request.clone())
            .await?;

        let transaction = Transaction::new(transaction_info.transaction);
        let transaction_info = self.check_transaction_info(
            &request.transaction_digest,
            transaction,
            transaction_info.status,
        ).tap_err(|err| {
            error!(?err, authority=?self.address, "Client error in handle_transaction_info_request");
        })?;

        Ok(transaction_info)
    }

    fn verify_checkpoint_sequence(
        &self,
        expected_seq: Option<CheckpointSequenceNumber>,
        checkpoint: &CertifiedCheckpointSummary,
    ) -> SomaResult {
        let observed_seq = checkpoint.sequence_number;

        if let (Some(e), o) = (expected_seq, observed_seq) {
            if !(e == o) {
                return Err(SomaError::from(
                    "Expected checkpoint number doesn't match with returned",
                ));
            }
        }
        Ok(())
    }

    fn verify_contents_exist<T, O>(
        &self,
        request_content: bool,
        checkpoint: &T,
        contents: &Option<O>,
    ) -> SomaResult {
        match (request_content, checkpoint, contents) {
            // If content is requested, checkpoint is not None, but we are not getting any content,
            // it's an error.
            // If content is not requested, or checkpoint is None, yet we are still getting content,
            // it's an error.
            (true, _, None) | (false, _, Some(_)) | (_, _, Some(_)) => Err(SomaError::from(
                "Checkpoint contents inconsistent with request",
            )),
            _ => Ok(()),
        }
    }

    fn verify_checkpoint_response(
        &self,
        request: &CheckpointRequest,
        response: &CheckpointResponse,
    ) -> SomaResult {
        // Verify response data was correct for request
        let CheckpointResponse {
            checkpoint,
            contents,
        } = &response;

        if let Some(CheckpointSummaryResponse::Certified(checkpoint)) = checkpoint {
            // Checks that the sequence number is correct.
            self.verify_checkpoint_sequence(request.sequence_number, checkpoint)?;
            self.verify_contents_exist(request.request_content, checkpoint, contents)?;
            // Verify signature.
            let epoch_id = checkpoint.epoch;
            checkpoint.verify_with_contents(&*self.get_committee(&epoch_id)?, contents.as_ref())
        } else {
            return Ok(());
        }
    }

    #[instrument(level = "trace", skip_all, fields(authority = ?self.address.concise()))]
    pub async fn handle_checkpoint(
        &self,
        request: CheckpointRequest,
    ) -> Result<CheckpointResponse, SomaError> {
        let resp = self
            .authority_client
            .handle_checkpoint(request.clone())
            .await?;
        self.verify_checkpoint_response(&request, &resp)
            .tap_err(|err| {
                error!(?err, authority=?self.address, "Client error in handle_checkpoint");
            })?;
        Ok(resp)
    }

    #[instrument(level = "trace", skip_all, fields(authority = ?self.address.concise()))]
    pub async fn handle_system_state_object(&self) -> Result<SystemState, SomaError> {
        self.authority_client
            .handle_system_state_object(SystemStateRequest { _unused: false })
            .await
    }

    /// Handle validator health check requests (for latency measurement)
    #[instrument(level = "trace", skip_all, fields(authority = ?self.address.concise()))]
    pub async fn validator_health(
        &self,
        request: ValidatorHealthRequest,
    ) -> Result<ValidatorHealthResponse, SomaError> {
        self.authority_client.validator_health(request).await
    }
}
