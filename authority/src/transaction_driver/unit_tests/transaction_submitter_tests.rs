// Tests for TransactionSubmitter: amplification factor, retry logic,
// and error aggregation.

use std::collections::BTreeMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use types::committee::Committee;
use types::config::validator_client_monitor_config::ValidatorClientMonitorConfig;
use types::consensus::ConsensusPosition;
use types::consensus::block::BlockRef;
use types::crypto::KeypairTraits;
use types::error::SomaError;
use types::messages_grpc::{SubmitTxRequest, SubmitTxResponse, SubmitTxResult, TxType};
use types::storage::committee_store::CommitteeStore;

use crate::authority_aggregator::AuthorityAggregator;
use crate::test_authority_clients::MockAuthorityApi;
use crate::transaction_driver::SubmitTransactionOptions;
use crate::transaction_driver::error::TransactionDriverError;
use crate::validator_client_monitor::ValidatorClientMonitor;

use super::TransactionSubmitter;

/// Create a test setup for transaction submitter tests.
fn setup() -> (
    Arc<AuthorityAggregator<MockAuthorityApi>>,
    Arc<ValidatorClientMonitor<MockAuthorityApi>>,
    TransactionSubmitter,
    Vec<MockAuthorityApi>,
    Committee,
) {
    let (committee, keypairs) = Committee::new_simple_test_committee_of_size(4);
    let committee_store = Arc::new(CommitteeStore::new_for_testing(&committee));

    let mut name_to_key: BTreeMap<_, _> = keypairs
        .into_iter()
        .map(|kp| (types::base::AuthorityName::from(kp.public()), kp))
        .collect();

    let mut authority_clients = BTreeMap::new();
    let mut mocks = Vec::new();

    for name in committee.names() {
        let keypair = name_to_key.remove(name).expect("keypair for committee member");
        let mock = MockAuthorityApi::new(keypair, *name, committee.epoch);
        authority_clients.insert(*name, mock.clone());
        mocks.push(mock);
    }

    let aggregator = Arc::new(AuthorityAggregator::new(
        committee.clone(),
        Arc::new(std::collections::HashMap::new()),
        committee_store,
        authority_clients,
        Default::default(),
    ));

    let agg_swap = Arc::new(ArcSwap::new(aggregator.clone()));
    let monitor = ValidatorClientMonitor::new(ValidatorClientMonitorConfig::default(), agg_swap);

    let submitter = TransactionSubmitter::new();

    (aggregator, monitor, submitter, mocks, committee)
}

fn make_submit_request() -> SubmitTxRequest {
    SubmitTxRequest { transaction: None, ping_type: None }
}

fn make_success_response() -> SubmitTxResponse {
    SubmitTxResponse {
        results: vec![SubmitTxResult::Submitted {
            consensus_position: ConsensusPosition { epoch: 0, block: BlockRef::MIN, index: 0 },
        }],
    }
}

fn make_error_response(msg: &str) -> SubmitTxResponse {
    SubmitTxResponse {
        results: vec![SubmitTxResult::Rejected {
            error: SomaError::InvalidSignature { error: msg.to_string() },
        }],
    }
}

fn default_options() -> SubmitTransactionOptions {
    SubmitTransactionOptions::default()
}

#[tokio::test]
async fn test_submit_transaction_success() {
    // First validator succeeds → transaction submitted
    let (aggregator, monitor, submitter, mocks, _committee) = setup();

    // All validators get a success response in case they're tried
    for mock in &mocks {
        mock.enqueue_submit_response(Ok(make_success_response()));
    }

    let result = submitter
        .submit_transaction(
            &aggregator,
            &monitor,
            TxType::SharedObject,
            1, // amplification_factor = 1
            make_submit_request(),
            &default_options(),
        )
        .await;

    assert!(result.is_ok(), "Expected success, got: {:?}", result.err());
    let (name, submit_result) = result.unwrap();
    match submit_result {
        SubmitTxResult::Submitted { .. } => {}
        other => panic!("Expected Submitted, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_submit_transaction_retry_on_rpc_error() {
    // First validator returns RPC error (retriable), second succeeds
    let (aggregator, monitor, submitter, mocks, _committee) = setup();

    // First validator: RPC error
    mocks[0].enqueue_submit_response(Err(SomaError::RpcError(
        "connection refused".to_string(),
        "test".to_string(),
    )));
    // Remaining validators: success
    for mock in &mocks[1..] {
        mock.enqueue_submit_response(Ok(make_success_response()));
    }

    let result = submitter
        .submit_transaction(
            &aggregator,
            &monitor,
            TxType::SharedObject,
            1,
            make_submit_request(),
            &default_options(),
        )
        .await;

    assert!(result.is_ok(), "Expected success after retry, got: {:?}", result.err());
}

#[tokio::test]
async fn test_submit_transaction_all_fail_aborted() {
    // All validators return RPC errors → Aborted
    let (aggregator, monitor, submitter, mocks, _committee) = setup();

    for mock in &mocks {
        mock.enqueue_submit_response(Err(SomaError::RpcError(
            "unavailable".to_string(),
            "test".to_string(),
        )));
    }

    let result = submitter
        .submit_transaction(
            &aggregator,
            &monitor,
            TxType::SharedObject,
            1,
            make_submit_request(),
            &default_options(),
        )
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        TransactionDriverError::Aborted { .. } => {}
        other => panic!("Expected Aborted, got: {}", other),
    }
}

#[tokio::test]
async fn test_submit_transaction_non_retriable_rejected() {
    // f+1 (2/4) validators return non-retriable rejection → RejectedByValidators
    let (aggregator, monitor, submitter, mocks, _committee) = setup();

    // All validators return InvalidTransaction (non-retriable rejection).
    // After f+1 non-retriable errors, the submitter returns RejectedByValidators.
    for mock in &mocks {
        mock.enqueue_submit_response(Ok(make_error_response("invalid nonce")));
    }

    let result = submitter
        .submit_transaction(
            &aggregator,
            &monitor,
            TxType::SharedObject,
            1,
            make_submit_request(),
            &default_options(),
        )
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        TransactionDriverError::RejectedByValidators {
            submission_non_retriable_errors, ..
        } => {
            assert!(
                submission_non_retriable_errors.total_stake > 0,
                "Expected non-retriable errors with stake"
            );
        }
        other => panic!("Expected RejectedByValidators, got: {}", other),
    }
}

#[tokio::test]
async fn test_submit_transaction_amplification_factor() {
    // With amplification_factor = 3, the submitter should try up to 3 validators
    // concurrently. If the first succeeds, the result is returned immediately.
    let (aggregator, monitor, submitter, mocks, _committee) = setup();

    for mock in &mocks {
        mock.enqueue_submit_response(Ok(make_success_response()));
    }

    let result = submitter
        .submit_transaction(
            &aggregator,
            &monitor,
            TxType::SharedObject,
            3, // amplification_factor = 3
            make_submit_request(),
            &default_options(),
        )
        .await;

    assert!(result.is_ok(), "Expected success with amplification, got: {:?}", result.err());
}

#[tokio::test]
async fn test_submit_transaction_amplification_with_partial_errors() {
    // amplification_factor = 4, some validators error, at least one succeeds
    let (aggregator, monitor, submitter, mocks, _committee) = setup();

    // First 3 validators: RPC error
    for mock in &mocks[..3] {
        mock.enqueue_submit_response(Err(SomaError::RpcError(
            "overloaded".to_string(),
            "test".to_string(),
        )));
    }
    // 4th validator: success
    mocks[3].enqueue_submit_response(Ok(make_success_response()));

    let result = submitter
        .submit_transaction(
            &aggregator,
            &monitor,
            TxType::SharedObject,
            4, // try all 4 concurrently
            make_submit_request(),
            &default_options(),
        )
        .await;

    assert!(result.is_ok(), "Expected success from 4th validator, got: {:?}", result.err());
}
