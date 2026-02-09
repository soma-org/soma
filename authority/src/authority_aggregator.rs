use crate::authority_client::{
    AuthorityAPI, NetworkAuthorityClient, make_authority_clients_with_timeout_config,
    make_network_authority_clients_with_network_config,
};
use crate::safe_client::SafeClient;
// #[cfg(test)]
// use crate::test_authority_clients::MockAuthorityApi;
use futures::StreamExt;
use std::convert::AsRef;
use std::net::SocketAddr;
use thiserror::Error;
use tracing::{Instrument, debug, error, instrument, trace, trace_span, warn};
use types::client::Config;
use types::config::network_config::NetworkConfig;
use types::crypto::{AuthorityPublicKeyBytes, AuthoritySignInfo};
use types::digests::{TransactionDigest, TransactionEffectsDigest};
use types::envelope::Message;
use types::genesis::Genesis;
use types::object::{Object, ObjectRef};
use types::quorum_driver::{GroupedErrors, PlainTransactionInfoResponse, QuorumDriverResponse};
use types::system_state::epoch_start::EpochStartSystemStateTrait;
use types::system_state::{SystemState, SystemStateTrait};
use types::{
    base::*,
    committee::Committee,
    error::{SomaError, SomaResult},
    transaction::*,
};

use crate::stake_aggregator::{InsertResult, MultiStakeAggregator, StakeAggregator};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::string::ToString;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use types::committee::{CommitteeWithNetworkMetadata, EpochId, StakeUnit};
use types::effects::{
    CertifiedTransactionEffects, SignedTransactionEffects, TransactionEffects,
    VerifiedCertifiedTransactionEffects,
};
use types::messages_grpc::{
    HandleCertificateRequest, HandleCertificateResponse, ObjectInfoRequest,
};
use types::object::ObjectID;
use types::storage::committee_store::CommitteeStore;
use types::system_state::epoch_start::EpochStartSystemState;

use futures::Future;
use futures::{future::BoxFuture, stream::FuturesUnordered};

use std::collections::BTreeSet;
use std::time::Instant;
use types::base::ConciseableName;
use types::committee::CommitteeTrait;

use tokio::time::timeout;

pub const DEFAULT_RETRIES: usize = 4;

// #[cfg(test)]
// #[path = "unit_tests/authority_aggregator_tests.rs"]
// pub mod authority_aggregator_tests;

#[derive(Clone)]
pub struct TimeoutConfig {
    pub pre_quorum_timeout: Duration,
    pub post_quorum_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            pre_quorum_timeout: Duration::from_secs(60),
            post_quorum_timeout: Duration::from_secs(7),
        }
    }
}

#[derive(Error, Debug, Eq, PartialEq)]
pub enum AggregatorProcessTransactionError {
    #[error(
        "Failed to execute transaction on a quorum of validators due to non-retryable errors. Validator errors: {:?}",
        errors
    )]
    FatalTransaction { errors: GroupedErrors },

    #[error(
        "Failed to execute transaction on a quorum of validators but state is still retryable. Validator errors: {:?}",
        errors
    )]
    RetryableTransaction { errors: GroupedErrors },

    #[error(
        "Failed to execute transaction on a quorum of validators due to conflicting transactions. Locked objects: {:?}. Validator errors: {:?}",
        conflicting_tx_digests,
        errors
    )]
    FatalConflictingTransaction {
        errors: GroupedErrors,
        conflicting_tx_digests:
            BTreeMap<TransactionDigest, (Vec<(AuthorityName, ObjectRef)>, StakeUnit)>,
    },

    #[error("Transaction is already finalized but with different user signatures")]
    TxAlreadyFinalizedWithDifferentUserSignatures,
}

#[derive(Error, Debug)]
pub enum AggregatorProcessCertificateError {
    #[error(
        "Failed to execute certificate on a quorum of validators. Non-retryable errors: {:?}",
        non_retryable_errors
    )]
    FatalExecuteCertificate { non_retryable_errors: GroupedErrors },

    #[error(
        "Failed to execute certificate on a quorum of validators but state is still retryable. Retryable errors: {:?}",
        retryable_errors
    )]
    RetryableExecuteCertificate { retryable_errors: GroupedErrors },
}

pub fn group_errors(errors: Vec<(SomaError, Vec<AuthorityName>, StakeUnit)>) -> GroupedErrors {
    #[allow(clippy::mutable_key_type)]
    let mut grouped_errors = HashMap::new();
    for (error, names, stake) in errors {
        let entry = grouped_errors.entry(error).or_insert((0, vec![]));
        entry.0 += stake;
        entry.1.extend(names.into_iter().map(|n| n.concise_owned()).collect::<Vec<_>>());
    }
    grouped_errors.into_iter().map(|(e, (s, n))| (e, s, n)).collect()
}

#[derive(Debug)]
struct ProcessTransactionState {
    // The list of signatures gathered at any point
    tx_signatures: StakeAggregator<AuthoritySignInfo, true>,
    effects_map: MultiStakeAggregator<TransactionEffectsDigest, TransactionEffects, true>,
    // The list of errors gathered at any point
    errors: Vec<(SomaError, Vec<AuthorityName>, StakeUnit)>,
    // This is exclusively non-retryable stake.
    non_retryable_stake: StakeUnit,
    // This includes both object and package not found sui errors.
    object_not_found_stake: StakeUnit,

    // If there are conflicting transactions, we note them down to report to user.
    conflicting_tx_digests:
        BTreeMap<TransactionDigest, (Vec<(AuthorityName, ObjectRef)>, StakeUnit)>,
    // As long as none of the exit criteria are met we consider the state retryable
    // 1) >= 2f+1 signatures
    // 2) >= f+1 non-retryable errors
    // 3) >= 2f+1 object not found errors
    retryable: bool,
    tx_finalized_with_different_user_sig: bool,
}

impl ProcessTransactionState {
    pub fn record_conflicting_transaction_if_any(
        &mut self,
        validator_name: AuthorityName,
        weight: StakeUnit,
        err: &SomaError,
    ) {
        if let SomaError::ObjectLockConflict { obj_ref, pending_transaction: transaction } = err {
            let (lock_records, total_stake) =
                self.conflicting_tx_digests.entry(*transaction).or_insert((Vec::new(), 0));
            lock_records.push((validator_name, *obj_ref));
            *total_stake += weight;
        }
    }

    pub fn check_if_error_indicates_tx_finalized_with_different_user_sig(
        &self,
        validity_threshold: StakeUnit,
    ) -> bool {
        // In some edge cases, the client may send the same transaction multiple times but with different user signatures.
        // When this happens, the "minority" tx will fail in safe_client because the certificate verification would fail
        // and return Sui::FailedToVerifyTxCertWithExecutedEffects.
        // Here, we check if there are f+1 validators return this error. If so, the transaction is already finalized
        // with a different set of user signatures. It's not trivial to return the results of that successful transaction
        // because we don't want fullnode to store the transaction with non-canonical user signatures. Given that this is
        // very rare, we simply return an error here.
        let invalid_sig_stake: StakeUnit = self
            .errors
            .iter()
            .filter_map(|(e, _, stake)| {
                if matches!(e, SomaError::FailedToVerifyTxCertWithExecutedEffects { .. }) {
                    Some(stake)
                } else {
                    None
                }
            })
            .sum();
        invalid_sig_stake >= validity_threshold
    }
}

struct ProcessCertificateState {
    // Different authorities could return different effects.  We want at least one effect to come
    // from 2f+1 authorities, which meets quorum and can be considered the approved effect.
    // The map here allows us to count the stake for each unique effect.
    effects_map:
        MultiStakeAggregator<(EpochId, TransactionEffectsDigest), TransactionEffects, true>,
    non_retryable_stake: StakeUnit,
    non_retryable_errors: Vec<(SomaError, Vec<AuthorityName>, StakeUnit)>,
    retryable_errors: Vec<(SomaError, Vec<AuthorityName>, StakeUnit)>,
    // As long as none of the exit criteria are met we consider the state retryable
    // 1) >= 2f+1 signatures
    // 2) >= f+1 non-retryable errors
    retryable: bool,

    // collection of extended data returned from the validators.
    // Not all validators will be asked to return this data so we need to hold onto it when one
    // validator has provided it
    input_objects: Option<Vec<Object>>,
    output_objects: Option<Vec<Object>>,
    auxiliary_data: Option<Vec<u8>>,
    request: HandleCertificateRequest,
}

#[derive(Debug)]
pub enum ProcessTransactionResult {
    Certified {
        certificate: CertifiedTransaction,
        /// Whether this certificate is newly created by aggregating 2f+1 signatures.
        /// If a validator returned a cert directly, this will be false.
        /// This is used to inform the quorum driver, which could make better decisions on telemetry
        /// such as settlement latency.
        newly_formed: bool,
    },
    Executed(VerifiedCertifiedTransactionEffects),
}

impl ProcessTransactionResult {
    pub fn into_cert_for_testing(self) -> CertifiedTransaction {
        match self {
            Self::Certified { certificate, .. } => certificate,
            Self::Executed(..) => panic!("Wrong type"),
        }
    }

    pub fn into_effects_for_testing(self) -> VerifiedCertifiedTransactionEffects {
        match self {
            Self::Certified { .. } => panic!("Wrong type"),
            Self::Executed(effects, ..) => effects,
        }
    }
}

#[derive(Clone)]
pub struct AuthorityAggregator<A: Clone> {
    /// Our committee.
    pub committee: Arc<Committee>,
    /// For more human readable metrics reporting.
    /// It's OK for this map to be empty or missing validators, it then defaults
    /// to use concise validator public keys.
    pub validator_display_names: Arc<HashMap<AuthorityName, String>>,

    /// How to talk to this committee.
    pub authority_clients: Arc<BTreeMap<AuthorityName, Arc<SafeClient<A>>>>,

    pub timeouts: TimeoutConfig,
    /// Store here for clone during re-config.
    pub committee_store: Arc<CommitteeStore>,
}

impl<A: Clone> AuthorityAggregator<A> {
    pub fn new(
        committee: Committee,
        validator_display_names: Arc<HashMap<AuthorityName, String>>,

        committee_store: Arc<CommitteeStore>,
        authority_clients: BTreeMap<AuthorityName, A>,

        timeouts: TimeoutConfig,
    ) -> Self {
        Self {
            committee: Arc::new(committee),
            validator_display_names,

            authority_clients: create_safe_clients(authority_clients, &committee_store),

            timeouts,
            committee_store,
        }
    }

    pub fn get_client(&self, name: &AuthorityName) -> Option<&Arc<SafeClient<A>>> {
        self.authority_clients.get(name)
    }

    pub fn clone_client_test_only(&self, name: &AuthorityName) -> Arc<SafeClient<A>>
    where
        A: Clone,
    {
        self.authority_clients[name].clone()
    }

    pub fn clone_committee_store(&self) -> Arc<CommitteeStore> {
        self.committee_store.clone()
    }

    pub fn clone_inner_committee_test_only(&self) -> Committee {
        (*self.committee).clone()
    }

    pub fn clone_inner_clients_test_only(&self) -> BTreeMap<AuthorityName, SafeClient<A>> {
        (*self.authority_clients).clone().into_iter().map(|(k, v)| (k, (*v).clone())).collect()
    }

    pub fn get_display_name(&self, name: &AuthorityName) -> String {
        self.validator_display_names
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.concise().to_string())
    }
}

fn create_safe_clients<A: Clone>(
    authority_clients: BTreeMap<AuthorityName, A>,
    committee_store: &Arc<CommitteeStore>,
) -> Arc<BTreeMap<AuthorityName, Arc<SafeClient<A>>>> {
    Arc::new(
        authority_clients
            .into_iter()
            .map(|(name, api)| {
                (name, Arc::new(SafeClient::new(api, committee_store.clone(), name)))
            })
            .collect(),
    )
}

impl AuthorityAggregator<NetworkAuthorityClient> {
    /// Create a new network authority aggregator by reading the committee and network addresses
    /// information from the given epoch start system state.
    pub fn new_from_epoch_start_state(
        epoch_start_state: &EpochStartSystemState,
        committee_store: &Arc<CommitteeStore>,
    ) -> Self {
        let committee = epoch_start_state.get_committee_with_network_metadata();
        let validator_display_names = epoch_start_state.get_authority_names_to_hostnames();
        Self::new_from_committee(committee, Arc::new(validator_display_names), committee_store)
    }

    /// Create a new AuthorityAggregator using information from the given epoch start system state.
    /// This is typically used during reconfiguration to create a new AuthorityAggregator with the
    /// new committee and network addresses.
    pub fn recreate_with_new_epoch_start_state(
        &self,
        epoch_start_state: &EpochStartSystemState,
    ) -> Self {
        Self::new_from_epoch_start_state(epoch_start_state, &self.committee_store)
    }

    pub fn new_from_committee(
        committee: CommitteeWithNetworkMetadata,
        validator_display_names: Arc<HashMap<AuthorityName, String>>,

        committee_store: &Arc<CommitteeStore>,
    ) -> Self {
        let net_config = default_network_config();
        let authority_clients =
            make_network_authority_clients_with_network_config(&committee, &net_config);
        Self::new(
            committee.committee().clone(),
            validator_display_names,
            committee_store.clone(),
            authority_clients,
            Default::default(),
        )
    }
}

impl<A> AuthorityAggregator<A>
where
    A: AuthorityAPI + Send + Sync + 'static + Clone,
{
    /// Query the object with highest version number from the authorities.
    /// We stop after receiving responses from 2f+1 validators.
    /// This function is untrusted because we simply assume each response is valid and there are no
    /// byzantine validators.
    /// Because of this, this function should only be used for testing or benchmarking.
    pub async fn get_latest_object_version_for_testing(
        &self,
        object_id: ObjectID,
    ) -> SomaResult<Object> {
        #[derive(Debug, Default)]
        struct State {
            latest_object_version: Option<Object>,
            total_weight: StakeUnit,
        }
        let initial_state = State::default();
        let result = quorum_map_then_reduce_with_timeout(
                self.committee.clone(),
                self.authority_clients.clone(),
                initial_state,
                |_name, client| {
                    Box::pin(async move {
                        let request =
                            ObjectInfoRequest { object_id };
                        let mut retry_count = 0;
                        loop {
                            match client.handle_object_info_request(request.clone()).await {
                                Ok(object_info) => return Ok(object_info),
                                Err(err) => {
                                    retry_count += 1;
                                    if retry_count > 3 {
                                        return Err(err);
                                    }
                                    tokio::time::sleep(Duration::from_secs(1)).await;
                                }
                            }
                        }
                    })
                },
                |mut state, name, weight, result| {
                    Box::pin(async move {
                        state.total_weight += weight;
                        match result {
                            Ok(object_info) => {
                                debug!("Received object info response from validator {:?} with version: {:?}", name.concise(), object_info.object.version());
                                if state.latest_object_version.as_ref().is_none_or(|latest| {
                                    object_info.object.version() > latest.version()
                                }) {
                                    state.latest_object_version = Some(object_info.object);
                                }
                            }
                            Err(err) => {
                                debug!("Received error from validator {:?}: {:?}", name.concise(), err);
                            }
                        };
                        if state.total_weight >= self.committee.quorum_threshold() {
                            if let Some(object) = state.latest_object_version {
                                return ReduceOutput::Success(object);
                            } else {
                                return ReduceOutput::Failed(state);
                            }
                        }
                        ReduceOutput::Continue(state)
                    })
                },
                // A long timeout before we hear back from a quorum
                self.timeouts.pre_quorum_timeout,
            )
            .await.map_err(|_state| SomaError::ObjectNotFound {
                object_id,
                version: None,
            })?;
        Ok(result.0)
    }

    /// Get the latest system state object from the authorities.
    /// This function assumes all validators are honest.
    /// It should only be used for testing or benchmarking.
    pub async fn get_latest_system_state_object_for_testing(&self) -> anyhow::Result<SystemState> {
        #[derive(Debug, Default)]
        struct State {
            latest_system_state: Option<SystemState>,
            total_weight: StakeUnit,
        }
        let initial_state = State::default();
        let result = quorum_map_then_reduce_with_timeout(
            self.committee.clone(),
            self.authority_clients.clone(),
            initial_state,
            |_name, client| Box::pin(async move { client.handle_system_state_object().await }),
            |mut state, name, weight, result| {
                Box::pin(async move {
                    state.total_weight += weight;
                    match result {
                        Ok(system_state) => {
                            debug!(
                                "Received system state object from validator {:?} with epoch: {:?}",
                                name.concise(),
                                system_state.epoch()
                            );
                            if state
                                .latest_system_state
                                .as_ref()
                                .is_none_or(|latest| system_state.epoch() > latest.epoch())
                            {
                                state.latest_system_state = Some(system_state);
                            }
                        }
                        Err(err) => {
                            debug!("Received error from validator {:?}: {:?}", name.concise(), err);
                        }
                    };
                    if state.total_weight >= self.committee.quorum_threshold() {
                        if let Some(system_state) = state.latest_system_state {
                            return ReduceOutput::Success(system_state);
                        } else {
                            return ReduceOutput::Failed(state);
                        }
                    }
                    ReduceOutput::Continue(state)
                })
            },
            // A long timeout before we hear back from a quorum
            self.timeouts.pre_quorum_timeout,
        )
        .await
        .map_err(|_| anyhow::anyhow!("Failed to get latest system state from the authorities"))?;
        Ok(result.0)
    }

    /// Submits the transaction to a quorum of validators to make a certificate.
    #[instrument(level = "trace", skip_all)]
    pub async fn process_transaction(
        &self,
        transaction: Transaction,
        client_addr: Option<SocketAddr>,
    ) -> Result<ProcessTransactionResult, AggregatorProcessTransactionError> {
        // Now broadcast the transaction to all authorities.
        let tx_digest = transaction.digest();
        debug!(
            tx_digest = ?tx_digest,
            "Broadcasting transaction request to authorities"
        );
        trace!("Transaction data: {:?}", transaction.data().intent_message().value);
        let committee = self.committee.clone();
        let state = ProcessTransactionState {
            tx_signatures: StakeAggregator::new(committee.clone()),
            effects_map: MultiStakeAggregator::new(committee.clone()),
            errors: vec![],
            object_not_found_stake: 0,
            non_retryable_stake: 0,

            retryable: true,
            conflicting_tx_digests: Default::default(),
            tx_finalized_with_different_user_sig: false,
        };

        let transaction_ref = &transaction;
        let validity_threshold = committee.validity_threshold();
        let quorum_threshold = committee.quorum_threshold();
        let validator_display_names = self.validator_display_names.clone();
        let result = quorum_map_then_reduce_with_timeout(
                committee.clone(),
                self.authority_clients.clone(),
                state,
                |name, client| {
                    Box::pin(
                        async move {
                          
                            let concise_name = name.concise_owned();
                            client.handle_transaction(transaction_ref.clone(), client_addr)
                                .instrument(trace_span!("handle_transaction", cancelled = false, authority =? concise_name))
                                .await
                        },
                    )
                },
                |mut state, name, weight, response| {
                    let display_name = validator_display_names.get(&name).unwrap_or(&name.concise().to_string()).clone();
                    Box::pin(async move {
                        match self.handle_process_transaction_response(
                            tx_digest, &mut state, response, name, weight,
                        ) {
                            Ok(Some(result)) => {
                                self.record_process_transaction_metrics(tx_digest, &state);
                                return ReduceOutput::Success(result);
                            }
                            Ok(None) => {},
                            Err(err) => {
                                let concise_name = name.concise();
                                debug!(?tx_digest, name=?concise_name, weight, "Error processing transaction from validator: {:?}", err);
                              
                                
                                // Record conflicting transactions if any to report to user.
                                state.record_conflicting_transaction_if_any(name, weight, &err);
                                let (retryable, categorized) = err.is_retryable();
                                if !categorized {
                                    // TODO: Should minimize possible uncategorized errors here
                                    // use ERROR for now to make them easier to spot.
                                    error!(?tx_digest, "uncategorized tx error: {err}");
                                }
                                if err.is_object_not_found() {
                                    // Special case for object not found because we can
                                    // retry if we have < 2f+1 object not found errors.
                                    // However once we reach >= 2f+1 object not found errors
                                    // we cannot retry.
                                    state.object_not_found_stake += weight;
                                }
                                else if !retryable {
                                    state.non_retryable_stake += weight;
                                }
                                state.errors.push((err, vec![name], weight));

                            }
                        };

                        let retryable_stake = self.get_retryable_stake(&state);
                        let good_stake = std::cmp::max(state.tx_signatures.total_votes(), state.effects_map.total_votes());
                        if good_stake + retryable_stake < quorum_threshold {
                            debug!(
                                tx_digest = ?tx_digest,
                                good_stake,
                                retryable_stake,
                                "No chance for any tx to get quorum, exiting. Conflicting_txes: {:?}",
                                state.conflicting_tx_digests
                            );
                            // If there is no chance for any tx to get quorum, exit.
                            state.retryable = false;
                            return ReduceOutput::Failed(state);
                        }

                        // TODO: add more comments to explain each condition.
                        let object_or_package_not_found_condition = state.object_not_found_stake >= quorum_threshold && std::env::var("NOT_RETRY_OBJECT_PACKAGE_NOT_FOUND").is_ok();
                        if state.non_retryable_stake >= validity_threshold
                            || object_or_package_not_found_condition // In normal case, object/package not found should be more than f+1
                            {
                            // We have hit an exit condition, f+1 non-retryable err or 2f+1 object not found,
                            // so we no longer consider the transaction state as retryable.
                            state.retryable = false;
                            ReduceOutput::Failed(state)
                        } else {
                            ReduceOutput::Continue(state)
                        }
                    })
                },
                // A long timeout before we hear back from a quorum
                self.timeouts.pre_quorum_timeout,
            )
            .await;

        match result {
            Ok((result, _)) => Ok(result),
            Err(state) => {
                self.record_process_transaction_metrics(tx_digest, &state);
                let state = self.record_non_quorum_effects_maybe(tx_digest, state);
                Err(self.handle_process_transaction_error(state))
            }
        }
    }

    fn handle_process_transaction_error(
        &self,
        state: ProcessTransactionState,
    ) -> AggregatorProcessTransactionError {
        let quorum_threshold = self.committee.quorum_threshold();

        if !state.retryable {
            if state.tx_finalized_with_different_user_sig
                || state.check_if_error_indicates_tx_finalized_with_different_user_sig(
                    self.committee.validity_threshold(),
                )
            {
                return AggregatorProcessTransactionError::TxAlreadyFinalizedWithDifferentUserSignatures;
            }

            // Handle conflicts first as `FatalConflictingTransaction` which is
            // more meaningful than `FatalTransaction`
            if !state.conflicting_tx_digests.is_empty() {
                let good_stake = state.tx_signatures.total_votes();
                warn!(
                    ?state.conflicting_tx_digests,
                    original_tx_stake = good_stake,
                    "Client double spend attempt detected!",
                );

                return AggregatorProcessTransactionError::FatalConflictingTransaction {
                    errors: group_errors(state.errors),
                    conflicting_tx_digests: state.conflicting_tx_digests,
                };
            }

            return AggregatorProcessTransactionError::FatalTransaction {
                errors: group_errors(state.errors),
            };
        }

        // The system is not overloaded and transaction state is still retryable.
        AggregatorProcessTransactionError::RetryableTransaction {
            errors: group_errors(state.errors),
        }
    }

    fn record_process_transaction_metrics(
        &self,
        tx_digest: &TransactionDigest,
        state: &ProcessTransactionState,
    ) {
        let num_signatures = state.tx_signatures.validator_sig_count();
        let good_stake = state.tx_signatures.total_votes();
        debug!(
            ?tx_digest,
            num_errors = state.errors.iter().map(|e| e.1.len()).sum::<usize>(),
            num_unique_errors = state.errors.len(),
            ?good_stake,
            non_retryable_stake = state.non_retryable_stake,
            ?num_signatures,
            "Received signatures response from validators handle_transaction"
        );
        if !state.errors.is_empty() {
            debug!(?tx_digest, "Errors received: {:?}", state.errors);
        }
    }

    fn handle_process_transaction_response(
        &self,
        tx_digest: &TransactionDigest,
        state: &mut ProcessTransactionState,
        response: SomaResult<PlainTransactionInfoResponse>,
        name: AuthorityName,
        weight: StakeUnit,
    ) -> SomaResult<Option<ProcessTransactionResult>> {
        match response {
            Ok(PlainTransactionInfoResponse::Signed(signed)) => {
                debug!(?tx_digest, name=?name.concise(), weight, "Received signed transaction from validator handle_transaction");
                self.handle_transaction_response_with_signed(state, signed)
            }
            Ok(PlainTransactionInfoResponse::ExecutedWithCert(cert, effects)) => {
                debug!(?tx_digest, name=?name.concise(), weight, "Received prev certificate and effects from validator handle_transaction");
                self.handle_transaction_response_with_executed(state, Some(cert), effects)
            }
            Ok(PlainTransactionInfoResponse::ExecutedWithoutCert(_, effects)) => {
                debug!(?tx_digest, name=?name.concise(), weight, "Received prev effects from validator handle_transaction");
                self.handle_transaction_response_with_executed(state, None, effects)
            }
            Err(err) => Err(err),
        }
    }

    fn handle_transaction_response_with_signed(
        &self,
        state: &mut ProcessTransactionState,
        plain_tx: SignedTransaction,
    ) -> SomaResult<Option<ProcessTransactionResult>> {
        match state.tx_signatures.insert(plain_tx.clone()) {
            InsertResult::NotEnoughVotes { bad_votes, bad_authorities } => {
                state.non_retryable_stake += bad_votes;
                if bad_votes > 0 {
                    state.errors.push((
                        SomaError::InvalidSignature {
                            error: "Individual signature verification failed".to_string(),
                        }
                        .into(),
                        bad_authorities,
                        bad_votes,
                    ));
                }
                Ok(None)
            }
            InsertResult::Failed { error } => Err(error),
            InsertResult::QuorumReached(cert_sig) => {
                let certificate =
                    CertifiedTransaction::new_from_data_and_sig(plain_tx.into_data(), cert_sig);
                certificate.verify_committee_sigs_only(&self.committee)?;
                Ok(Some(ProcessTransactionResult::Certified { certificate, newly_formed: true }))
            }
        }
    }

    fn handle_transaction_response_with_executed(
        &self,
        state: &mut ProcessTransactionState,
        certificate: Option<CertifiedTransaction>,
        plain_tx_effects: SignedTransactionEffects,
    ) -> SomaResult<Option<ProcessTransactionResult>> {
        match certificate {
            Some(certificate) if certificate.epoch() == self.committee.epoch => {
                // If we get a certificate in the same epoch, then we use it.
                // A certificate in a past epoch does not guarantee finality
                // and validators may reject to process it.
                Ok(Some(ProcessTransactionResult::Certified { certificate, newly_formed: false }))
            }
            _ => {
                // If we get 2f+1 effects, it's a proof that the transaction
                // has already been finalized. This works because validators would re-sign effects for transactions
                // that were finalized in previous epochs.
                let digest = plain_tx_effects.data().digest();
                match state.effects_map.insert(digest, plain_tx_effects.clone()) {
                    InsertResult::NotEnoughVotes { bad_votes, bad_authorities } => {
                        state.non_retryable_stake += bad_votes;
                        if bad_votes > 0 {
                            state.errors.push((
                                SomaError::InvalidSignature {
                                    error: "Individual signature verification failed".to_string(),
                                }
                                .into(),
                                bad_authorities,
                                bad_votes,
                            ));
                        }
                        Ok(None)
                    }
                    InsertResult::Failed { error } => Err(error),
                    InsertResult::QuorumReached(cert_sig) => {
                        let ct = CertifiedTransactionEffects::new_from_data_and_sig(
                            plain_tx_effects.into_data(),
                            cert_sig,
                        );
                        Ok(Some(ProcessTransactionResult::Executed(ct.verify(&self.committee)?)))
                    }
                }
            }
        }
    }

    /// Check if we have some signed TransactionEffects but not a quorum
    fn record_non_quorum_effects_maybe(
        &self,
        tx_digest: &TransactionDigest,
        mut state: ProcessTransactionState,
    ) -> ProcessTransactionState {
        if state.effects_map.unique_key_count() > 0 {
            let non_quorum_effects = state.effects_map.get_all_unique_values();
            warn!(
                ?tx_digest,
                "Received signed Effects but not with a quorum {:?}", non_quorum_effects
            );

            // Safe to unwrap because we know that there is at least one entry in the map
            // from the check above.
            let (_most_staked_effects_digest, (_, most_staked_effects_digest_stake)) =
                non_quorum_effects.iter().max_by_key(|&(_, (_, stake))| stake).unwrap();
            // We check if we have enough retryable stake to get quorum for the most staked
            // effects digest, otherwise it indicates we have violated safety assumptions
            // or we have forked.
            if most_staked_effects_digest_stake + self.get_retryable_stake(&state)
                < self.committee.quorum_threshold()
            {
                state.retryable = false;
                if state.check_if_error_indicates_tx_finalized_with_different_user_sig(
                    self.committee.validity_threshold(),
                ) {
                    state.tx_finalized_with_different_user_sig = true;
                } else {
                    // TODO: Figure out a more reliable way to detect invariance violations.
                    error!(
                        "We have seen signed effects but unable to reach quorum threshold even including retriable stakes. This is very rare. Tx: {tx_digest:?}. Non-quorum effects: {non_quorum_effects:?}."
                    );
                }
            }

            let mut involved_validators = Vec::new();
            let mut total_stake = 0;
            for (validators, stake) in non_quorum_effects.values() {
                involved_validators.extend_from_slice(validators);
                total_stake += stake;
            }
            // TODO: Instead of pushing a new error, we should add more information about the non-quorum effects
            // in the final error if state is no longer retryable
            state.errors.push((
                SomaError::QuorumFailedToGetEffectsQuorumWhenProcessingTransaction {
                    effects_map: non_quorum_effects,
                }
                .into(),
                involved_validators,
                total_stake,
            ));
        }
        state
    }

    fn get_retryable_stake(&self, state: &ProcessTransactionState) -> StakeUnit {
        self.committee.total_votes()
            - state.non_retryable_stake
            - state.effects_map.total_votes()
            - state.tx_signatures.total_votes()
    }

    #[instrument(level = "trace", skip_all)]
    pub async fn process_certificate(
        &self,
        request: HandleCertificateRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<QuorumDriverResponse, AggregatorProcessCertificateError> {
        let state = ProcessCertificateState {
            effects_map: MultiStakeAggregator::new(self.committee.clone()),
            non_retryable_stake: 0,
            non_retryable_errors: vec![],
            retryable_errors: vec![],
            retryable: true,

            input_objects: None,
            output_objects: None,
            auxiliary_data: None,
            request: request.clone(),
        };

        // create a set of validators that we should sample to request input/output objects from
        let validators_to_sample =
            if request.include_input_objects || request.include_output_objects {
                // Number of validators to request input/output objects from
                const NUMBER_TO_SAMPLE: usize = 10;

                self.committee.choose_multiple_weighted_iter(NUMBER_TO_SAMPLE).cloned().collect()
            } else {
                HashSet::new()
            };

        let tx_digest = *request.certificate.digest();
        let timeout_after_quorum = self.timeouts.post_quorum_timeout;

        let request_ref = request;
        let threshold = self.committee.quorum_threshold();
        let validity = self.committee.validity_threshold();

        debug!(
            ?tx_digest,
            quorum_threshold = threshold,
            validity_threshold = validity,
            ?timeout_after_quorum,
            "Broadcasting certificate to authorities"
        );
        let committee: Arc<Committee> = self.committee.clone();
        let authority_clients = self.authority_clients.clone();

        let validator_display_names = self.validator_display_names.clone();
        let (result, mut remaining_tasks) = quorum_map_then_reduce_with_timeout(
            committee.clone(),
            authority_clients.clone(),
            state,
            move |name, client| {
                Box::pin(async move {
                    let concise_name = name.concise_owned();
                    if request_ref.include_input_objects || request_ref.include_output_objects {

                        // adjust the request to validators we aren't planning on sampling
                        let req = if validators_to_sample.contains(&name) {
                            request_ref
                        } else {
                            HandleCertificateRequest {
                                include_input_objects: false,
                                include_output_objects: false,
                               
                                ..request_ref
                            }
                        };

                        client
                            .handle_certificate(req, client_addr)
                            .instrument(trace_span!("handle_certificate", authority =? concise_name))
                            .await
                    } else {
                        client
                            .handle_certificate(request_ref, client_addr)
                            .instrument(trace_span!("handle_certificate", authority =? concise_name))
                            .await
                            .map(|response| HandleCertificateResponse {
                                effects: response.effects,
                             
                                input_objects: None,
                                output_objects: None,
                              
                            })
                    }
                })
            },
            move |mut state, name, weight, response| {
                let committee_clone = committee.clone();
               
                let display_name = validator_display_names.get(&name).unwrap_or(&name.concise().to_string()).clone();
                Box::pin(async move {
                    // We aggregate the effects response, until we have more than 2f
                    // and return.
                    match AuthorityAggregator::<A>::handle_process_certificate_response(
                        committee_clone,
                      
                        &tx_digest, &mut state, response, name)
                    {
                        Ok(Some(effects)) => ReduceOutput::Success(effects),
                        Ok(None) => {
                            // When the result is none, it is possible that the
                            // non_retryable_stake had been incremented due to
                            // failed individual signature verification.
                            if state.non_retryable_stake >= validity {
                                state.retryable = false;
                                ReduceOutput::Failed(state)
                            } else {
                                ReduceOutput::Continue(state)
                            }
                        },
                        Err(err) => {
                            let concise_name = name.concise();
                            debug!(?tx_digest, name=?concise_name, "Error processing certificate from validator: {:?}", err);
                           
                           
                            let (retryable, categorized) = err.is_retryable();
                            if !categorized {
                                // TODO: Should minimize possible uncategorized errors here
                                // use ERROR for now to make them easier to spot.
                                error!(?tx_digest, "[WATCHOUT] uncategorized tx error: {err}");
                            }
                            if !retryable {
                                state.non_retryable_stake += weight;
                                state.non_retryable_errors.push((err, vec![name], weight));
                            } else {
                                state.retryable_errors.push((err, vec![name], weight));
                            }
                            if state.non_retryable_stake >= validity {
                                state.retryable = false;
                                ReduceOutput::Failed(state)
                            } else {
                                ReduceOutput::Continue(state)
                            }
                        }
                    }
                })
            },
            // A long timeout before we hear back from a quorum
            self.timeouts.pre_quorum_timeout,
        )
        .await
        .map_err(|state| {
            debug!(
                ?tx_digest,
                num_unique_effects = state.effects_map.unique_key_count(),
                non_retryable_stake = state.non_retryable_stake,
                "Received effects responses from validators"
            );

           
            if state.retryable {
                AggregatorProcessCertificateError::RetryableExecuteCertificate {
                    retryable_errors: group_errors(state.retryable_errors),
                }
            } else {
                AggregatorProcessCertificateError::FatalExecuteCertificate {
                    non_retryable_errors: group_errors(state.non_retryable_errors),
                }
            }
        })?;

        if !remaining_tasks.is_empty() {
            // Use best efforts to send the cert to remaining validators.
            tokio::spawn(async move {
                let mut timeout = Box::pin(sleep(timeout_after_quorum));
                loop {
                    tokio::select! {
                        _ = &mut timeout => {
                            debug!(?tx_digest, "Timed out in post quorum cert broadcasting: {:?}. Remaining tasks: {:?}", timeout_after_quorum, remaining_tasks.len());

                            break;
                        }
                        res = remaining_tasks.next() => {
                            if res.is_none() {
                                break;
                            }
                        }
                    }
                }
            });
        }
        Ok(result)
    }

    fn handle_process_certificate_response(
        committee: Arc<Committee>,

        tx_digest: &TransactionDigest,
        state: &mut ProcessCertificateState,
        response: SomaResult<HandleCertificateResponse>,
        name: AuthorityName,
    ) -> SomaResult<Option<QuorumDriverResponse>> {
        match response {
            Ok(HandleCertificateResponse {
                effects: signed_effects,

                input_objects,
                output_objects,
            }) => {
                debug!(
                    ?tx_digest,
                    name = ?name.concise(),
                    "Validator handled certificate successfully",
                );

                if input_objects.is_some() && state.input_objects.is_none() {
                    state.input_objects = input_objects;
                }

                if output_objects.is_some() && state.output_objects.is_none() {
                    state.output_objects = output_objects;
                }

                let effects_digest = *signed_effects.digest();
                // Note: here we aggregate votes by the hash of the effects structure
                match state
                    .effects_map
                    .insert((signed_effects.epoch(), effects_digest), signed_effects.clone())
                {
                    InsertResult::NotEnoughVotes { bad_votes, bad_authorities } => {
                        state.non_retryable_stake += bad_votes;
                        if bad_votes > 0 {
                            state.non_retryable_errors.push((
                                SomaError::InvalidSignature {
                                    error: "Individual signature verification failed".to_string(),
                                }
                                .into(),
                                bad_authorities,
                                bad_votes,
                            ));
                        }
                        Ok(None)
                    }
                    InsertResult::Failed { error } => Err(error),
                    InsertResult::QuorumReached(cert_sig) => {
                        let ct = CertifiedTransactionEffects::new_from_data_and_sig(
                            signed_effects.into_data(),
                            cert_sig,
                        );

                        if (state.request.include_input_objects && state.input_objects.is_none())
                            || (state.request.include_output_objects
                                && state.output_objects.is_none())
                        {
                            debug!(
                                ?tx_digest,
                                "Quorum Reached but requested input/output objects were not returned"
                            );
                        }

                        ct.verify(&committee).map(|ct| {
                            debug!(?tx_digest, "Got quorum for validators handle_certificate.");
                            Some(QuorumDriverResponse {
                                effects_cert: ct,

                                input_objects: state.input_objects.take(),
                                output_objects: state.output_objects.take(),
                            })
                        })
                    }
                }
            }
            Err(err) => Err(err),
        }
    }

    #[instrument(level = "trace", skip_all, fields(tx_digest = ?transaction.digest()))]
    pub async fn execute_transaction_block(
        &self,
        transaction: &Transaction,
        client_addr: Option<SocketAddr>,
    ) -> Result<VerifiedCertifiedTransactionEffects, anyhow::Error> {
        let result = self.process_transaction(transaction.clone(), client_addr).await?;
        let cert = match result {
            ProcessTransactionResult::Certified { certificate, .. } => certificate,
            ProcessTransactionResult::Executed(effects) => {
                return Ok(effects);
            }
        };

        let response = self
            .process_certificate(
                HandleCertificateRequest {
                    certificate: cert.clone(),

                    include_input_objects: false,
                    include_output_objects: false,
                },
                client_addr,
            )
            .await?;

        Ok(response.effects_cert)
    }
}

#[derive(Default)]
pub struct AuthorityAggregatorBuilder<'a> {
    network_config: Option<&'a NetworkConfig>,
    genesis: Option<&'a Genesis>,
    committee: Option<Committee>,
    reference_gas_price: Option<u64>,
    committee_store: Option<Arc<CommitteeStore>>,

    timeouts_config: Option<TimeoutConfig>,
}

impl<'a> AuthorityAggregatorBuilder<'a> {
    pub fn from_network_config(config: &'a NetworkConfig) -> Self {
        Self { network_config: Some(config), ..Default::default() }
    }

    pub fn from_genesis(genesis: &'a Genesis) -> Self {
        Self { genesis: Some(genesis), ..Default::default() }
    }

    pub fn from_committee(committee: Committee) -> Self {
        Self { committee: Some(committee), ..Default::default() }
    }

    #[cfg(test)]
    pub fn from_committee_size(committee_size: usize) -> Self {
        let (committee, _keypairs) = Committee::new_simple_test_committee_of_size(committee_size);
        Self::from_committee(committee)
    }

    pub fn with_committee_store(mut self, committee_store: Arc<CommitteeStore>) -> Self {
        self.committee_store = Some(committee_store);
        self
    }

    pub fn with_timeouts_config(mut self, timeouts_config: TimeoutConfig) -> Self {
        self.timeouts_config = Some(timeouts_config);
        self
    }

    fn get_network_committee(&self) -> CommitteeWithNetworkMetadata {
        self.get_genesis()
            .unwrap_or_else(|| panic!("need either NetworkConfig or Genesis."))
            .committee_with_network()
    }

    // fn get_committee_authority_names_to_hostnames(&self) -> HashMap<AuthorityName, String> {
    //     if let Some(genesis) = self.get_genesis() {
    //         let state = genesis
    //             .system_object()
    //             .into_genesis_version_for_tooling();
    //         state
    //             .validators
    //             .active_validators
    //             .iter()
    //             .map(|v| {
    //                 let metadata = v.verified_metadata();
    //                 let name = metadata.sui_pubkey_bytes();

    //                 (name, metadata.name.clone())
    //             })
    //             .collect()
    //     } else {
    //         HashMap::new()
    //     }
    // }

    fn get_genesis(&self) -> Option<&Genesis> {
        if let Some(network_config) = self.network_config {
            Some(&network_config.genesis)
        } else if let Some(genesis) = self.genesis {
            Some(genesis)
        } else {
            None
        }
    }

    fn get_committee(&self) -> Committee {
        self.committee.clone().unwrap_or_else(|| self.get_network_committee().committee().clone())
    }

    // pub fn build_network_clients(
    //     self,
    // ) -> (
    //     AuthorityAggregator<NetworkAuthorityClient>,
    //     BTreeMap<AuthorityPublicKeyBytes, NetworkAuthorityClient>,
    // ) {
    //     let network_committee = self.get_network_committee();
    //     let auth_clients = make_authority_clients_with_timeout_config(
    //         &network_committee,
    //         DEFAULT_CONNECT_TIMEOUT_SEC,
    //         DEFAULT_REQUEST_TIMEOUT_SEC,
    //     );
    //     let auth_agg = self.build_custom_clients(auth_clients.clone());
    //     (auth_agg, auth_clients)
    // }

    // pub fn build_custom_clients<C: Clone>(
    //     self,
    //     authority_clients: BTreeMap<AuthorityName, C>,
    // ) -> AuthorityAggregator<C> {
    //     let committee = self.get_committee();
    //     let validator_display_names = self.get_committee_authority_names_to_hostnames();

    //     let committee_store = self
    //         .committee_store
    //         .unwrap_or_else(|| Arc::new(CommitteeStore::new_for_testing(&committee)));

    //     let timeouts_config = self.timeouts_config.unwrap_or_default();

    //     AuthorityAggregator::new(
    //         committee,
    //         Arc::new(validator_display_names),

    //         committee_store,
    //         authority_clients,

    //         timeouts_config,
    //     )
    // }

    // #[cfg(test)]
    // pub fn build_mock_authority_aggregator(self) -> AuthorityAggregator<MockAuthorityApi> {
    //     let committee = self.get_committee();
    //     let clients = committee
    //         .names()
    //         .map(|name| {
    //             (
    //                 *name,
    //                 MockAuthorityApi::new(
    //                     Duration::from_millis(100),
    //                     Arc::new(std::sync::Mutex::new(30)),
    //                 ),
    //             )
    //         })
    //         .collect();
    //     self.build_custom_clients(clients)
    // }
}

pub type AsyncResult<'a, T, E> = BoxFuture<'a, Result<T, E>>;

pub struct SigRequestPrefs<K> {
    pub ordering_pref: BTreeSet<K>,
    pub prefetch_timeout: Duration,
}

pub enum ReduceOutput<R, S> {
    Continue(S),
    Failed(S),
    Success(R),
}

/// This function takes an initial state, than executes an asynchronous function (FMap) for each
/// authority, and folds the results as they become available into the state using an async function (FReduce).
///
/// prefetch_timeout: the minimum amount of time to spend trying to gather results from all authorities
/// before falling back to arrival order.
///
/// total_timeout: the maximum amount of total time to wait for results from all authorities, including
/// time spent prefetching.
pub async fn quorum_map_then_reduce_with_timeout_and_prefs<
    'a,
    C,
    K,
    Client: 'a,
    S,
    V,
    R,
    E,
    FMap,
    FReduce,
>(
    committee: Arc<C>,
    authority_clients: Arc<BTreeMap<K, Arc<Client>>>,
    authority_preferences: Option<SigRequestPrefs<K>>,
    initial_state: S,
    map_each_authority: FMap,
    reduce_result: FReduce,
    total_timeout: Duration,
) -> Result<(R, FuturesUnordered<impl Future<Output = (K, Result<V, E>)> + 'a>), S>
where
    K: Ord + ConciseableName<'a> + Clone + 'a,
    C: CommitteeTrait<K>,
    FMap: FnOnce(K, Arc<Client>) -> AsyncResult<'a, V, E> + Clone + 'a,
    FReduce: Fn(S, K, StakeUnit, Result<V, E>) -> BoxFuture<'a, ReduceOutput<R, S>>,
{
    let (preference, prefetch_timeout) =
        if let Some(SigRequestPrefs { ordering_pref, prefetch_timeout }) = authority_preferences {
            (Some(ordering_pref), Some(prefetch_timeout))
        } else {
            (None, None)
        };
    let authorities_shuffled = committee.shuffle_by_stake(preference.as_ref(), None);
    let mut accumulated_state = initial_state;
    let start_time = Instant::now();

    // First, execute in parallel for each authority FMap.
    let mut responses: futures::stream::FuturesUnordered<_> = authorities_shuffled
        .clone()
        .into_iter()
        .map(|name| {
            let client = authority_clients[&name].clone();
            let execute = map_each_authority.clone();
            async move { (name.clone(), execute(name, client).await) }
        })
        .collect();
    if let Some(prefetch_timeout) = prefetch_timeout {
        let prefetch_sleep = tokio::time::sleep(prefetch_timeout);
        let mut authority_to_result: BTreeMap<K, Result<V, E>> = BTreeMap::new();
        tokio::pin!(prefetch_sleep);
        // get all the sigs we can within prefetch_timeout
        loop {
            tokio::select! {
                resp = responses.next() => {
                    match resp {
                        Some((authority_name, result)) => {
                            authority_to_result.insert(authority_name, result);
                        }
                        None => {
                            // we have processed responses from the full committee so can stop early
                            break;
                        }
                    }
                }
                _ = &mut prefetch_sleep => {
                    break;
                }
            }
        }
        // process what we have up to this point
        for authority_name in authorities_shuffled {
            let authority_weight = committee.weight(&authority_name);
            if let Some(result) = authority_to_result.remove(&authority_name) {
                accumulated_state = match reduce_result(
                    accumulated_state,
                    authority_name,
                    authority_weight,
                    result,
                )
                .await
                {
                    // In the first two cases we are told to continue the iteration.
                    ReduceOutput::Continue(state) => state,
                    ReduceOutput::Failed(state) => {
                        return Err(state);
                    }
                    ReduceOutput::Success(result) => {
                        // The reducer tells us that we have the result needed. Just return it.
                        return Ok((result, responses));
                    }
                };
            }
        }
        // if we got here, fallback through the if statement to continue in arrival order on
        // the remaining validators
    }

    // As results become available fold them into the state using FReduce.
    while let Ok(Some((authority_name, result))) =
        timeout(total_timeout.saturating_sub(start_time.elapsed()), responses.next()).await
    {
        let authority_weight = committee.weight(&authority_name);
        accumulated_state =
            match reduce_result(accumulated_state, authority_name, authority_weight, result).await {
                // In the first two cases we are told to continue the iteration.
                ReduceOutput::Continue(state) => state,
                ReduceOutput::Failed(state) => {
                    return Err(state);
                }
                ReduceOutput::Success(result) => {
                    // The reducer tells us that we have the result needed. Just return it.
                    return Ok((result, responses));
                }
            }
    }
    // If we have exhausted all authorities and still have not returned a result, return
    // error with the accumulated state.
    Err(accumulated_state)
}

/// This function takes an initial state, than executes an asynchronous function (FMap) for each
/// authority, and folds the results as they become available into the state using an async function (FReduce).
///
/// FMap can do io, and returns a result V. An error there may not be fatal, and could be consumed by the
/// MReduce function to overall recover from it. This is necessary to ensure byzantine authorities cannot
/// interrupt the logic of this function.
///
/// FReduce returns a result to a ReduceOutput. If the result is Err the function
/// shortcuts and the Err is returned. An Ok ReduceOutput result can be used to shortcut and return
/// the resulting state (ReduceOutput::End), continue the folding as new states arrive (ReduceOutput::Continue).
///
/// This function provides a flexible way to communicate with a quorum of authorities, processing and
/// processing their results into a safe overall result, and also safely allowing operations to continue
/// past the quorum to ensure all authorities are up to date (up to a timeout).
pub async fn quorum_map_then_reduce_with_timeout<
    'a,
    C,
    K,
    Client: 'a,
    S: 'a,
    V: 'a,
    R: 'a,
    E,
    FMap,
    FReduce,
>(
    committee: Arc<C>,
    authority_clients: Arc<BTreeMap<K, Arc<Client>>>,
    // The initial state that will be used to fold in values from authorities.
    initial_state: S,
    // The async function used to apply to each authority. It takes an authority name,
    // and authority client parameter and returns a Result<V>.
    map_each_authority: FMap,
    // The async function that takes an accumulated state, and a new result for V from an
    // authority and returns a result to a ReduceOutput state.
    reduce_result: FReduce,
    // The initial timeout applied to all
    initial_timeout: Duration,
) -> Result<(R, FuturesUnordered<impl Future<Output = (K, Result<V, E>)> + 'a>), S>
where
    K: Ord + ConciseableName<'a> + Clone + 'a,
    C: CommitteeTrait<K>,
    FMap: FnOnce(K, Arc<Client>) -> AsyncResult<'a, V, E> + Clone + 'a,
    FReduce: Fn(S, K, StakeUnit, Result<V, E>) -> BoxFuture<'a, ReduceOutput<R, S>> + 'a,
{
    quorum_map_then_reduce_with_timeout_and_prefs(
        committee,
        authority_clients,
        None,
        initial_state,
        map_each_authority,
        reduce_result,
        initial_timeout,
    )
    .await
}

pub const DEFAULT_CONNECT_TIMEOUT_SEC: Duration = Duration::from_secs(10);
pub const DEFAULT_REQUEST_TIMEOUT_SEC: Duration = Duration::from_secs(30);
pub const DEFAULT_HTTP2_KEEPALIVE_SEC: Duration = Duration::from_secs(5);

pub fn default_network_config() -> Config {
    let mut net_config = Config::new();
    net_config.connect_timeout = Some(DEFAULT_CONNECT_TIMEOUT_SEC);
    net_config.request_timeout = Some(DEFAULT_REQUEST_TIMEOUT_SEC);
    net_config.http2_keepalive_interval = Some(DEFAULT_HTTP2_KEEPALIVE_SEC);
    net_config
}
