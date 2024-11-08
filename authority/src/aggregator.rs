use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};
use thiserror::Error;
use futures::{future::BoxFuture, stream::FuturesUnordered, StreamExt};
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, instrument, trace, trace_span, warn, Instrument};
use types::{
    base::{AuthorityName, ConciseableName}, client::Config, committee::{Committee, CommitteeWithNetworkMetadata, EpochId, VotingPower}, crypto::{AuthoritySignInfo, ConciseAuthorityPublicKeyBytes}, digests::{TransactionDigest, TransactionEffectsDigest}, effects::{self, CertifiedTransactionEffects, SignedTransactionEffects, TransactionEffects, VerifiedCertifiedTransactionEffects}, envelope::Message, error::{SomaError, SomaResult}, grpc::{HandleCertificateRequest, HandleCertificateResponse}, quorum_driver::{PlainTransactionInfoResponse, QuorumDriverResponse}, system_state::{EpochStartSystemState, EpochStartSystemStateTrait}, transaction::{CertifiedTransaction, SignedTransaction, Transaction}
};
use utils::agg::{quorum_map_then_reduce_with_timeout, AsyncResult, ReduceOutput};
use types::committee::CommitteeTrait;
use crate::{
    client::{
        make_network_authority_clients_with_network_config, AuthorityAPI, NetworkAuthorityClient,
    }, committee_store::CommitteeStore, safe_client::SafeClient, stake_aggregator::{InsertResult, MultiStakeAggregator, StakeAggregator}
};

#[derive(Debug)]
struct ProcessTransactionState {
    // The list of signatures gathered at any point
    tx_signatures: StakeAggregator<AuthoritySignInfo, true>,
    effects_map: MultiStakeAggregator<TransactionEffectsDigest, TransactionEffects, true>,
    // The list of errors gathered at any point
    errors: Vec<(SomaError, Vec<AuthorityName>, VotingPower)>,
    // If there are conflicting transactions, we note them down and may attempt to retry
    conflicting_tx_digests: BTreeMap<TransactionDigest, (Vec<AuthorityName>, VotingPower)>,
    // As long as none of the exit criteria are met we consider the state retryable
    // 1) >= 2f+1 signatures
    // 2) >= f+1 non-retryable errors
    // 3) >= 2f+1 object not found errors
    // Note: For conflicting transactions we collect as many responses as possible
    // before we know for sure no tx can reach quorum. Namely, stake of the most
    // promising tx + retryable stake < 2f+1.
    retryable: bool,
    tx_finalized_with_different_user_sig: bool,
}

impl ProcessTransactionState {
    #[allow(clippy::type_complexity)]
    pub fn conflicting_tx_digest_with_most_stake(
        &self,
    ) -> Option<(TransactionDigest, &Vec<AuthorityName>, VotingPower)> {
        self.conflicting_tx_digests
            .iter()
            .max_by_key(|(_, (_, stake))| *stake)
            .map(|(digest, (validators, stake))| (*digest, validators, *stake))
    }

    pub fn check_if_error_indicates_tx_finalized_with_different_user_sig(
        &self,
        validity_threshold: VotingPower,
    ) -> bool {
        // In some edge cases, the client may send the same transaction multiple times but with different user signatures.
        // When this happens, the "minority" tx will fail in safe_client because the certificate verification would fail
        // and return Sui::FailedToVerifyTxCertWithExecutedEffects.
        // Here, we check if there are f+1 validators return this error. If so, the transaction is already finalized
        // with a different set of user signatures. It's not trivial to return the results of that successful transaction
        // because we don't want fullnode to store the transaction with non-canonical user signatures. Given that this is
        // very rare, we simply return an error here.
        let invalid_sig_stake: VotingPower = self
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
    non_retryable_stake: VotingPower,
    non_retryable_errors: Vec<(SomaError, Vec<AuthorityName>, VotingPower)>,
    retryable_errors: Vec<(SomaError, Vec<AuthorityName>, VotingPower)>,
    // As long as none of the exit criteria are met we consider the state retryable
    // 1) >= 2f+1 signatures
    // 2) >= f+1 non-retryable errors
    retryable: bool,
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

#[derive(Clone)]
pub struct AuthorityAggregator<A: Clone> {
    /// Our Sui committee.
    pub committee: Arc<Committee>,
    /// For more human readable metrics reporting.
    /// It's OK for this map to be empty or missing validators, it then defaults
    /// to use concise validator public keys.
    pub validator_display_names: Arc<HashMap<AuthorityName, String>>,
    /// How to talk to this committee.
    pub authority_clients: Arc<BTreeMap<AuthorityName, Arc<SafeClient<A>>>>,

    // pub timeouts: TimeoutConfig,
    /// Store here for clone during re-config.
    pub committee_store: Arc<CommitteeStore>,
}

impl<A: Clone> AuthorityAggregator<A> {
    pub fn new(
        committee: Committee,
        committee_store: Arc<CommitteeStore>,
        authority_clients: BTreeMap<AuthorityName, A>,
        validator_display_names: Arc<HashMap<AuthorityName, String>>,
    ) -> Self {
        Self {
            committee: Arc::new(committee),
            validator_display_names,
            authority_clients: create_safe_clients(
                authority_clients,
                &committee_store,
            ),
            committee_store,
        }
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
                (
                    name,
                    Arc::new(SafeClient::new(
                        api,
                        committee_store.clone(),
                        name,
                    )),
                )
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
        Self::new_from_committee(
            committee,
            committee_store,
            Arc::new(validator_display_names),
        )
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
        committee_store: &Arc<CommitteeStore>,
        validator_display_names: Arc<HashMap<AuthorityName, String>>,
    ) -> Self {
        let net_config = Config::default();
        let authority_clients =
            make_network_authority_clients_with_network_config(&committee, &net_config);
        Self::new(
            committee.committee().clone(),
            committee_store.clone(),
            authority_clients,
            validator_display_names,
        )
    }
}

impl<A> AuthorityAggregator<A>
where
    A: AuthorityAPI + Send + Sync + 'static + Clone,
{
    // Repeatedly calls the provided closure on a randomly selected validator until it succeeds.
    // Once all validators have been attempted, starts over at the beginning. Intended for cases
    // that must eventually succeed as long as the network is up (or comes back up) eventually.
    async fn quorum_once_inner<'a, S, FMap>(
        &'a self,
        // try these authorities first
        preferences: Option<&BTreeSet<AuthorityName>>,
        // only attempt from these authorities.
        restrict_to: Option<&BTreeSet<AuthorityName>>,
        // The async function used to apply to each authority. It takes an authority name,
        // and authority client parameter and returns a Result<V>.
        map_each_authority: FMap,
        timeout_each_authority: Duration,
        authority_errors: &mut HashMap<AuthorityName, SomaError>,
    ) -> Result<S, SomaError>
    where
        FMap: Fn(AuthorityName, Arc<SafeClient<A>>) -> AsyncResult<'a, S, SomaError> + Send + Clone + 'a,
        S: Send,
    {
        let start = tokio::time::Instant::now();
        let mut delay = Duration::from_secs(1);
        loop {
            let authorities_shuffled = self.committee.shuffle_by_stake(preferences, restrict_to);
            let mut authorities_shuffled = authorities_shuffled.iter();

            type RequestResult<S> = Result<Result<S, SomaError>, tokio::time::error::Elapsed>;

            enum Event<S> {
                StartNext,
                Request(AuthorityName, RequestResult<S>),
            }

            let mut futures = FuturesUnordered::<BoxFuture<'a, Event<S>>>::new();

            let start_req = |name: AuthorityName, client: Arc<SafeClient<A>>| {
                let map_each_authority = map_each_authority.clone();
                Box::pin(async move {
                    trace!(name=?name.concise(), now = ?tokio::time::Instant::now() - start, "new request");
                    let map = map_each_authority(name, client);
                    Event::Request(name, timeout(timeout_each_authority, map).await)
                })
            };

            // This process is intended to minimize latency in the face of unreliable authorities,
            // without creating undue load on authorities.
            //
            // The fastest possible process from the
            // client's point of view would simply be to issue a concurrent request to every
            // authority and then take the winner - this would create unnecessary load on
            // authorities.
            //
            // The most efficient process from the network's point of view is to do one request at
            // a time, however if the first validator that the client contacts is unavailable or
            // slow, the client must wait for the serial_authority_request_interval period to elapse
            // before starting its next request.
            //
            // So, this process is designed as a compromise between these two extremes.
            // - We start one request, and schedule another request to begin after
            //   serial_authority_request_interval.
            // - Whenever a request finishes, if it succeeded, we return. if it failed, we start a
            //   new request.
            // - If serial_authority_request_interval elapses, we begin a new request even if the
            //   previous one is not finished, and schedule another future request.

            let name = authorities_shuffled.next().ok_or_else(|| {
                error!(
                    ?preferences,
                    ?restrict_to,
                    "Available authorities list is empty."
                );
                SomaError::from("Available authorities list is empty")
            })?;
            futures.push(start_req(*name, self.authority_clients[name].clone()));

            while let Some(res) = futures.next().await {
                match res {
                    Event::StartNext => {
                        trace!(now = ?tokio::time::Instant::now() - start, "eagerly beginning next request");
               
                    }
                    Event::Request(name, res) => {
                        match res {
                            // timeout
                            Err(_) => {
                                debug!(name=?name.concise(), "authority request timed out");
                                authority_errors.insert(name, SomaError::TimeoutError);
                            }
                            // request completed
                            Ok(inner_res) => {
                                trace!(name=?name.concise(), now = ?tokio::time::Instant::now() - start,
                                       "request completed successfully");
                                match inner_res {
                                    Err(e) => authority_errors.insert(name, e),
                                    Ok(res) => return Ok(res),
                                };
                            }
                        };
                    }
                }

                if let Some(next_authority) = authorities_shuffled.next() {
                    futures.push(start_req(
                        *next_authority,
                        self.authority_clients[next_authority].clone(),
                    ));
                } else {
                    break;
                }
            }

            info!(
                ?authority_errors,
                "quorum_once_with_timeout failed on all authorities, retrying in {:?}", delay
            );
            sleep(delay).await;
            delay = std::cmp::min(delay * 2, Duration::from_secs(5 * 60));
        }
    }

    /// Like quorum_map_then_reduce_with_timeout, but for things that need only a single
    /// successful response, such as fetching a Transaction from some authority.
    /// This is intended for cases in which byzantine authorities can time out or slow-loris, but
    /// can't give a false answer, because e.g. the digest of the response is known, or a
    /// quorum-signed object such as a checkpoint has been requested.
    pub(crate) async fn quorum_once_with_timeout<'a, S, FMap>(
        &'a self,
        // try these authorities first
        preferences: Option<&BTreeSet<AuthorityName>>,
        // only attempt from these authorities.
        restrict_to: Option<&BTreeSet<AuthorityName>>,
        // The async function used to apply to each authority. It takes an authority name,
        // and authority client parameter and returns a Result<V>.
        map_each_authority: FMap,
        timeout_each_authority: Duration,
        // When to give up on the attempt entirely.
        timeout_total: Option<Duration>,
        // The behavior that authorities expect to perform, used for logging and error
        description: String,
    ) -> Result<S, SomaError>
    where
        FMap: Fn(AuthorityName, Arc<SafeClient<A>>) -> AsyncResult<'a, S, SomaError> + Send + Clone + 'a,
        S: Send,
    {
        let mut authority_errors = HashMap::new();

        let fut = self.quorum_once_inner(
            preferences,
            restrict_to,
            map_each_authority,
            timeout_each_authority,
            &mut authority_errors,
        );

        if let Some(t) = timeout_total {
            timeout(t, fut).await.map_err(|_timeout_error| {
                if authority_errors.is_empty() {
                    SomaError::TimeoutError
                } else {
                    SomaError::TooManyIncorrectAuthorities {
                        errors: authority_errors
                            .iter()
                            .map(|(a, b)| (*a, b.clone()))
                            .collect(),
                        action: description,
                    }
                }
            })?
        } else {
            fut.await
        }
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
        trace!(
            "Transaction data: {:?}",
            transaction.data().intent_message().value
        );
        let committee = self.committee.clone();
        let state = ProcessTransactionState {
            tx_signatures: StakeAggregator::new(committee.clone()),
            errors: vec![],
            effects_map: MultiStakeAggregator::new(committee.clone()),
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
                                return ReduceOutput::Success(result);
                            }
                            Ok(None) => {},
                            Err(err) => {
                                let concise_name = name.concise();
                                debug!(?tx_digest, name=?concise_name, weight, "Error processing transaction from validator: {:?}", err);
                                // let (retryable, categorized) = err.is_retryable();
                                // if !categorized {
                                //     // TODO: Should minimize possible uncategorized errors here
                                //     // use ERROR for now to make them easier to spot.
                                //     error!(?tx_digest, "uncategorized tx error: {err}");
                                // }
                               
                                state.errors.push((err, vec![name], weight));

                            }
                        };
                        ReduceOutput::Continue(state)
                    })
                },
            )
            .await;

        match result {
            Ok((result, _)) => Ok(result),
            Err(state) => {
                let state = self.record_non_quorum_effects_maybe(tx_digest, state);
                Err(self.handle_process_transaction_error(tx_digest, state))
            }
        }
    }

    fn handle_process_transaction_response(
        &self,
        tx_digest: &TransactionDigest,
        state: &mut ProcessTransactionState,
        response: SomaResult<PlainTransactionInfoResponse>,
        name: AuthorityName,
        weight: VotingPower,
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
            InsertResult::NotEnoughVotes {
                bad_votes,
                bad_authorities,
            } => {
                // state.non_retryable_stake += bad_votes;
                if bad_votes > 0 {
                    state.errors.push((
                        SomaError::InvalidSignature {
                            error: "Individual signature verification failed".to_string(),
                        },
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
                Ok(Some(ProcessTransactionResult::Certified {
                    certificate,
                    newly_formed: true,
                }))
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
                Ok(Some(ProcessTransactionResult::Certified {
                    certificate,
                    newly_formed: false,
                }))
            }
            _ => {
                // If we get 2f+1 effects, it's a proof that the transaction
                // has already been finalized. This works because validators would re-sign effects for transactions
                // that were finalized in previous epochs.
                let digest = plain_tx_effects.data().digest();
                match state.effects_map.insert(digest, plain_tx_effects.clone()) {
                    InsertResult::NotEnoughVotes {
                        bad_votes,
                        bad_authorities,
                    } => {
                        // state.non_retryable_stake += bad_votes;
                        if bad_votes > 0 {
                            state.errors.push((
                                SomaError::InvalidSignature {
                                    error: "Individual signature verification failed".to_string(),
                                },
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
                        Ok(Some(ProcessTransactionResult::Executed(
                            ct.verify(&self.committee)?,
                        )))
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
                non_quorum_effects
                    .iter()
                    .max_by_key(|&(_, (_, stake))| stake)
                    .unwrap();
            // We check if we have enough retryable stake to get quorum for the most staked
            // effects digest, otherwise it indicates we have violated safety assumptions
            // or we have forked.
            if *most_staked_effects_digest_stake //+ self.get_retryable_stake(&state)
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
                },
                involved_validators,
                total_stake,
            ));
        }
        state
    }


    fn handle_process_transaction_error(
        &self,
        original_tx_digest: &TransactionDigest,
        state: ProcessTransactionState,
    ) -> AggregatorProcessTransactionError {
        let quorum_threshold = self.committee.quorum_threshold();

        // Handle possible conflicts first as `FatalConflictingTransaction` is
        // more meaningful than `FatalTransaction`.
        if let Some((most_staked_conflicting_tx, validators, most_staked_conflicting_tx_stake)) =
            state.conflicting_tx_digest_with_most_stake()
        {


            warn!(
                ?state.conflicting_tx_digests,
                ?most_staked_conflicting_tx,
                ?original_tx_digest,
       
                most_staked_conflicting_tx_stake = most_staked_conflicting_tx_stake,
                "Client double spend attempt detected: {:?}",
                validators
            );

            return AggregatorProcessTransactionError::FatalConflictingTransaction {
                errors: group_errors(state.errors),
                conflicting_tx_digests: state.conflicting_tx_digests,
            };
        }

        if !state.retryable {
            if state.tx_finalized_with_different_user_sig
                || state.check_if_error_indicates_tx_finalized_with_different_user_sig(
                    self.committee.validity_threshold(),
                )
            {
                return AggregatorProcessTransactionError::TxAlreadyFinalizedWithDifferentUserSignatures;
            }
            return AggregatorProcessTransactionError::FatalTransaction {
                errors: group_errors(state.errors),
            };
        }

        // No conflicting transaction, the system is not overloaded and transaction state is still retryable.
        AggregatorProcessTransactionError::RetryableTransaction {
            errors: group_errors(state.errors),
        }
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
        };

    

        let tx_digest = *request.certificate.digest();

        let request_ref = request;
        let threshold = self.committee.quorum_threshold();
        let validity = self.committee.validity_threshold();

        debug!(
            ?tx_digest,
            quorum_threshold = threshold,
            validity_threshold = validity,
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
                    // adjust the request to validators we aren't planning on sampling
                    let req =  HandleCertificateRequest {
                        ..request_ref
                    };

                    client
                        .handle_certificate(req, client_addr)
                        .instrument(trace_span!("handle_certificate", authority =? concise_name))
                        .await
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
                           
                            state.non_retryable_stake += weight;
                            state.non_retryable_errors.push((err, vec![name], weight));
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
  
        )
        .await
        .map_err(|state| {
            debug!(
                ?tx_digest,
   
                non_retryable_stake = state.non_retryable_stake,
                "Received effects responses from validators"
            );

            // record errors and tx retryable state
    
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

                loop {
                    let res = remaining_tasks.next().await;
                    if res.is_none() {
                        break;
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
                signed_effects,
            }) => {
                debug!(
                    ?tx_digest,
                    name = ?name.concise(),
                    "Validator handled certificate successfully",
                );

                let effects_digest = *signed_effects.digest();
                // Note: here we aggregate votes by the hash of the effects structure
                match state.effects_map.insert(
                    (signed_effects.epoch(), effects_digest),
                    signed_effects.clone(),
                ) {
                    InsertResult::NotEnoughVotes {
                        bad_votes,
                        bad_authorities,
                    } => {
                        warn!(
                            ?tx_digest,
                            ?effects_digest,
                            bad_votes,
                            "Not enough votes for effects"
                        );
                        // state.non_retryable_stake += bad_votes;
                        if bad_votes > 0 {
                            state.non_retryable_errors.push((
                                SomaError::InvalidSignature {
                                    error: "Individual signature verification failed".to_string(),
                                },
                                bad_authorities,
                                bad_votes,
                            ));
                        }
                        Ok(None)
                    }
                    InsertResult::Failed { error } => {
                        warn!(
                            ?tx_digest,
                            ?effects_digest,
                            "Failed to insert effects"
                        );    
                        Err(error)
                    },
                    InsertResult::QuorumReached(cert_sig) => {
                        let ct = CertifiedTransactionEffects::new_from_data_and_sig(
                            signed_effects.into_data(),
                            cert_sig,
                        );

                        ct.verify(&committee).map(|ct| {
                            debug!(?tx_digest, "Got quorum for validators handle_certificate.");
                            Some(QuorumDriverResponse {
                                effects_cert: ct,
                               
                            })
                        })
                    }
                }
            }
            Err(err) => Err(err),
        }
    }

}

#[derive(Error, Debug, Eq, PartialEq)]
pub enum AggregatorProcessTransactionError {
    #[error(
        "Failed to execute transaction on a quorum of validators due to non-retryable errors. Validator errors: {:?}",
        errors,
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
        errors,
    )]
    FatalConflictingTransaction {
        errors: GroupedErrors,
        conflicting_tx_digests:
            BTreeMap<TransactionDigest, (Vec<AuthorityName>, VotingPower)>,
    },

    #[error(
        "Validators returned conflicting transactions but it is potentially recoverable. Locked objects: {:?}. Validator errors: {:?}",
        conflicting_tx_digests,
        errors,
    )]
    RetryableConflictingTransaction {
        conflicting_tx_digest_to_retry: Option<TransactionDigest>,
        errors: GroupedErrors,
        conflicting_tx_digests:
            BTreeMap<TransactionDigest, (Vec<AuthorityName>, VotingPower)>,
    },

    #[error(
        "{} of the validators by stake are overloaded with transactions pending execution. Validator errors: {:?}",
        overloaded_stake,
        errors
    )]
    SystemOverload {
        overloaded_stake: VotingPower,
        errors: GroupedErrors,
    },

    #[error("Transaction is already finalized but with different user signatures")]
    TxAlreadyFinalizedWithDifferentUserSignatures,

    #[error(
        "{} of the validators by stake are overloaded and requested the client to retry after {} seconds. Validator errors: {:?}",
        overload_stake,
        retry_after_secs,
        errors
    )]
    SystemOverloadRetryAfter {
        overload_stake: VotingPower,
        errors: GroupedErrors,
        retry_after_secs: u64,
    },
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


pub fn group_errors(errors: Vec<(SomaError, Vec<AuthorityName>, VotingPower)>) -> GroupedErrors {
    let mut grouped_errors = HashMap::new();
    for (error, names, stake) in errors {
        let entry = grouped_errors.entry(error).or_insert((0, vec![]));
        entry.0 += stake;
        entry.1.extend(
            names
                .into_iter()
                .map(|n| n.concise_owned())
                .collect::<Vec<_>>(),
        );
    }
    grouped_errors
        .into_iter()
        .map(|(e, (s, n))| (e, s, n))
        .collect()
}

pub type GroupedErrors = Vec<(SomaError, VotingPower, Vec<ConciseAuthorityPublicKeyBytes>)>;