use std::{collections::VecDeque, sync::Arc};

use types::{base::AuthorityName, messages_grpc::TxType};

use crate::{
    authority_aggregator::AuthorityAggregator,
    safe_client::SafeClient,
    status_aggregator::StatusAggregator,
    transaction_driver::error::{
        AggregatedEffectsDigests, TransactionDriverError, TransactionRequestError,
        aggregate_request_errors,
    },
    validator_client_monitor::ValidatorClientMonitor,
};

/// Select validators with latencies within 2% of the lowest latency.
const SELECT_LATENCY_DELTA: f64 = 0.02;

/// Provides the next target validator to retry operations,
/// and gathers the errors along with the operations.
///
/// In TransactionDriver, submitting a transaction and getting full effects follow the same pattern:
/// 1. Retry against all validators until the operation succeeds.
/// 2. If non‑retriable errors from a quorum of validators are returned, the operation should fail permanently.
///
/// When an `allowed_validators` is provided, only the validators in the list will be used to submit the transaction to.
/// When the allowed validator list is empty, any validator can be used an then the validators are selected based on their scores.
///
/// When a `blocked_validators` is provided, the validators in the list cannot be used to submit the transaction to.
/// When the blocked validator list is empty, no restrictions are applied.
///
/// This component helps to manager this retry pattern.
pub(crate) struct RequestRetrier<A: Clone> {
    ranked_clients: VecDeque<(AuthorityName, Arc<SafeClient<A>>)>,
    pub(crate) non_retriable_errors_aggregator: StatusAggregator<TransactionRequestError>,
    pub(crate) retriable_errors_aggregator: StatusAggregator<TransactionRequestError>,
}

impl<A: Clone> RequestRetrier<A> {
    pub(crate) fn new(
        auth_agg: &Arc<AuthorityAggregator<A>>,
        client_monitor: &Arc<ValidatorClientMonitor<A>>,
        tx_type: TxType,
        allowed_validators: Vec<String>,
        blocked_validators: Vec<String>,
    ) -> Self {
        let ranked_validators = client_monitor.select_shuffled_preferred_validators(
            &auth_agg.committee,
            tx_type,
            SELECT_LATENCY_DELTA,
        );
        let ranked_clients = ranked_validators
            .into_iter()
            .map(|name| (name, auth_agg.get_display_name(&name)))
            .filter(|(_name, display_name)| {
                allowed_validators.is_empty() || allowed_validators.contains(display_name)
            })
            .filter(|(_name, display_name)| {
                blocked_validators.is_empty() || !blocked_validators.contains(display_name)
            })
            .filter_map(|(name, _display_name)| {
                // There is not guarantee that the `name` are in the `auth_agg.authority_clients` if those are coming from the list
                // of `allowed_validators`, as the provided `auth_agg` might have been updated with a new committee that doesn't contain the validator in question.
                auth_agg.authority_clients.get(&name).map(|client| (name, client.clone()))
            })
            .collect::<VecDeque<_>>();
        let non_retriable_errors_aggregator = StatusAggregator::new(auth_agg.committee.clone());
        let retriable_errors_aggregator = StatusAggregator::new(auth_agg.committee.clone());
        Self { ranked_clients, non_retriable_errors_aggregator, retriable_errors_aggregator }
    }

    // Selects the next target validator to attempt an operation.
    pub(crate) fn next_target(
        &mut self,
    ) -> Result<(AuthorityName, Arc<SafeClient<A>>), TransactionDriverError> {
        if let Some((name, client)) = self.ranked_clients.pop_front() {
            return Ok((name, client));
        };

        if self.non_retriable_errors_aggregator.reached_validity_threshold() {
            Err(TransactionDriverError::RejectedByValidators {
                submission_non_retriable_errors: aggregate_request_errors(
                    self.non_retriable_errors_aggregator.status_by_authority(),
                ),
                submission_retriable_errors: aggregate_request_errors(
                    self.retriable_errors_aggregator.status_by_authority(),
                ),
            })
        } else {
            Err(TransactionDriverError::Aborted {
                submission_non_retriable_errors: aggregate_request_errors(
                    self.non_retriable_errors_aggregator.status_by_authority(),
                ),
                submission_retriable_errors: aggregate_request_errors(
                    self.retriable_errors_aggregator.status_by_authority(),
                ),
                observed_effects_digests: AggregatedEffectsDigests { digests: Vec::new() },
            })
        }
    }

    // Adds an error associated with the operation against the authority.
    //
    // Returns an error if it has aggregated >= f+1 submission non-retriable errors.
    // In this case, the transaction cannot finalize unless there is a software bug
    // or > f malicious validators.
    pub(crate) fn add_error(
        &mut self,
        name: AuthorityName,
        error: TransactionRequestError,
    ) -> Result<(), TransactionDriverError> {
        if error.is_submission_retriable() {
            self.retriable_errors_aggregator.insert(name, error);
        } else {
            self.non_retriable_errors_aggregator.insert(name, error);
            if self.non_retriable_errors_aggregator.reached_validity_threshold() {
                return Err(TransactionDriverError::RejectedByValidators {
                    submission_non_retriable_errors: aggregate_request_errors(
                        self.non_retriable_errors_aggregator.status_by_authority(),
                    ),
                    submission_retriable_errors: aggregate_request_errors(
                        self.retriable_errors_aggregator.status_by_authority(),
                    ),
                });
            }
        }

        Ok(())
    }
}
