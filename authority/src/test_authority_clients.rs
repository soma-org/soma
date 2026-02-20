// Test infrastructure: MockAuthorityApi for testing authority aggregator,
// effects certifier, and transaction submitter.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use types::checkpoints::{CheckpointRequest, CheckpointResponse};
use types::crypto::{AuthorityKeyPair, AuthorityPublicKeyBytes, AuthoritySignInfo};
use types::effects::{ExecutionStatus, SignedTransactionEffects, TransactionEffects};
use types::envelope::Message;
use types::error::{SomaError, SomaResult};
use types::intent::{Intent, IntentScope};
use types::messages_grpc::{
    HandleCertificateRequest, HandleCertificateResponse, HandleTransactionResponse,
    ObjectInfoRequest, ObjectInfoResponse, SubmitTxRequest, SubmitTxResponse,
    SystemStateRequest, TransactionInfoRequest, TransactionInfoResponse, TransactionStatus,
    ValidatorHealthRequest, ValidatorHealthResponse, WaitForEffectsRequest, WaitForEffectsResponse,
};
use types::system_state::SystemState;
use types::transaction::Transaction;
use types::tx_fee::TransactionFee;

use crate::authority_client::AuthorityAPI;

/// Configurable responses for each mock authority method.
/// When a response queue is non-empty, pop and return. When empty, use default behavior.
#[derive(Default)]
struct MockState {
    /// Errors to return from handle_transaction (when empty, auto-sign)
    handle_transaction_errors: VecDeque<SomaError>,
    /// Errors to return from handle_certificate (when empty, auto-sign effects)
    handle_certificate_errors: VecDeque<SomaError>,
    /// Responses for submit_transaction
    submit_responses: VecDeque<Result<SubmitTxResponse, SomaError>>,
    /// Responses for wait_for_effects
    wait_for_effects_responses: VecDeque<Result<WaitForEffectsResponse, SomaError>>,
    /// Call counters
    handle_transaction_count: usize,
    handle_certificate_count: usize,
    submit_count: usize,
    wait_for_effects_count: usize,
}

/// A configurable mock implementation of AuthorityAPI for testing
/// authority aggregator, effects certifier, and transaction submitter.
///
/// Default behavior:
/// - `handle_transaction`: signs the transaction with this authority's keypair
/// - `handle_certificate`: creates and signs effects for the certificate's tx_digest
/// - `submit_transaction`: returns from queue, or panics if queue is empty
/// - `wait_for_effects`: returns from queue, or panics if queue is empty
#[derive(Clone)]
pub struct MockAuthorityApi {
    pub authority_key: Arc<AuthorityKeyPair>,
    pub name: AuthorityPublicKeyBytes,
    pub epoch: u64,
    pub delay: Duration,
    state: Arc<Mutex<MockState>>,
}

impl MockAuthorityApi {
    pub fn new(
        authority_key: AuthorityKeyPair,
        name: AuthorityPublicKeyBytes,
        epoch: u64,
    ) -> Self {
        Self {
            authority_key: Arc::new(authority_key),
            name,
            epoch,
            delay: Duration::ZERO,
            state: Arc::new(Mutex::new(MockState::default())),
        }
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Queue an error to be returned from handle_transaction.
    /// Errors are consumed in FIFO order; when exhausted, default signing resumes.
    pub fn enqueue_handle_transaction_error(&self, error: SomaError) {
        self.state.lock().unwrap().handle_transaction_errors.push_back(error);
    }

    /// Queue an error to be returned from handle_certificate.
    pub fn enqueue_handle_certificate_error(&self, error: SomaError) {
        self.state.lock().unwrap().handle_certificate_errors.push_back(error);
    }

    /// Queue a response for submit_transaction.
    pub fn enqueue_submit_response(&self, response: Result<SubmitTxResponse, SomaError>) {
        self.state.lock().unwrap().submit_responses.push_back(response);
    }

    /// Queue a response for wait_for_effects.
    pub fn enqueue_wait_for_effects_response(
        &self,
        response: Result<WaitForEffectsResponse, SomaError>,
    ) {
        self.state.lock().unwrap().wait_for_effects_responses.push_back(response);
    }

    pub fn handle_transaction_count(&self) -> usize {
        self.state.lock().unwrap().handle_transaction_count
    }

    pub fn handle_certificate_count(&self) -> usize {
        self.state.lock().unwrap().handle_certificate_count
    }

    pub fn submit_count(&self) -> usize {
        self.state.lock().unwrap().submit_count
    }

    pub fn wait_for_effects_count(&self) -> usize {
        self.state.lock().unwrap().wait_for_effects_count
    }

    /// Create a properly signed AuthoritySignInfo for the given transaction.
    fn sign_transaction(&self, transaction: &Transaction) -> AuthoritySignInfo {
        let data = transaction.data();
        AuthoritySignInfo::new(
            self.epoch,
            data,
            Intent::soma_app(IntentScope::SenderSignedTransaction),
            self.name,
            &*self.authority_key,
        )
    }

    /// Create a minimal TransactionEffects for the given tx digest and sign it.
    fn sign_effects(
        &self,
        tx_digest: types::digests::TransactionDigest,
    ) -> SignedTransactionEffects {
        let effects = TransactionEffects::V1(types::effects::TransactionEffectsV1 {
            status: ExecutionStatus::Success,
            executed_epoch: self.epoch,
            transaction_fee: TransactionFee::default(),
            transaction_digest: tx_digest,
            gas_object_index: None,
            dependencies: vec![],
            version: types::object::OBJECT_START_VERSION,
            changed_objects: vec![],
            unchanged_shared_objects: vec![],
        });
        let sig = AuthoritySignInfo::new(
            self.epoch,
            &effects,
            Intent::soma_app(IntentScope::TransactionData),
            self.name,
            &*self.authority_key,
        );
        SignedTransactionEffects::new_from_data_and_sig(effects, sig)
    }
}

#[async_trait]
impl AuthorityAPI for MockAuthorityApi {
    async fn handle_transaction(
        &self,
        transaction: Transaction,
        _client_addr: Option<SocketAddr>,
    ) -> Result<HandleTransactionResponse, SomaError> {
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }

        let mut state = self.state.lock().unwrap();
        state.handle_transaction_count += 1;

        if let Some(error) = state.handle_transaction_errors.pop_front() {
            return Err(error);
        }
        drop(state);

        let sig = self.sign_transaction(&transaction);
        Ok(HandleTransactionResponse { status: TransactionStatus::Signed(sig) })
    }

    async fn handle_certificate(
        &self,
        request: HandleCertificateRequest,
        _client_addr: Option<SocketAddr>,
    ) -> Result<HandleCertificateResponse, SomaError> {
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }

        let mut state = self.state.lock().unwrap();
        state.handle_certificate_count += 1;

        if let Some(error) = state.handle_certificate_errors.pop_front() {
            return Err(error);
        }
        drop(state);

        let tx_digest = *request.certificate.digest();
        let signed_effects = self.sign_effects(tx_digest);
        Ok(HandleCertificateResponse {
            effects: signed_effects,
            input_objects: None,
            output_objects: None,
        })
    }

    async fn submit_transaction(
        &self,
        _request: SubmitTxRequest,
        _client_addr: Option<SocketAddr>,
    ) -> Result<SubmitTxResponse, SomaError> {
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }

        let mut state = self.state.lock().unwrap();
        state.submit_count += 1;

        state
            .submit_responses
            .pop_front()
            .expect("MockAuthorityApi: no submit_transaction response queued")
    }

    async fn wait_for_effects(
        &self,
        _request: WaitForEffectsRequest,
        _client_addr: Option<SocketAddr>,
    ) -> Result<WaitForEffectsResponse, SomaError> {
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }

        let mut state = self.state.lock().unwrap();
        state.wait_for_effects_count += 1;

        state
            .wait_for_effects_responses
            .pop_front()
            .expect("MockAuthorityApi: no wait_for_effects response queued")
    }

    async fn handle_object_info_request(
        &self,
        _request: ObjectInfoRequest,
    ) -> Result<ObjectInfoResponse, SomaError> {
        Err(SomaError::UnsupportedFeatureError { error: "mock".to_string() })
    }

    async fn handle_transaction_info_request(
        &self,
        _request: TransactionInfoRequest,
    ) -> Result<TransactionInfoResponse, SomaError> {
        Err(SomaError::UnsupportedFeatureError { error: "mock".to_string() })
    }

    async fn handle_checkpoint(
        &self,
        _request: CheckpointRequest,
    ) -> Result<CheckpointResponse, SomaError> {
        Err(SomaError::UnsupportedFeatureError { error: "mock".to_string() })
    }

    async fn handle_system_state_object(
        &self,
        _request: SystemStateRequest,
    ) -> Result<SystemState, SomaError> {
        Err(SomaError::UnsupportedFeatureError { error: "mock".to_string() })
    }

    async fn validator_health(
        &self,
        _request: ValidatorHealthRequest,
    ) -> Result<ValidatorHealthResponse, SomaError> {
        Ok(ValidatorHealthResponse::default())
    }
}
