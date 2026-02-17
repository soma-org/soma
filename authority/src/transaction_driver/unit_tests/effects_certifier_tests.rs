// Tests for EffectsCertifier: effects digest quorum, forked execution,
// retries, and non-retriable error aggregation.

use std::collections::BTreeMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use types::base::SomaAddress;
use types::crypto::KeypairTraits;
use types::committee::Committee;
use types::config::validator_client_monitor_config::ValidatorClientMonitorConfig;
use types::consensus::block::BlockRef;
use types::consensus::ConsensusPosition;
use types::digests::TransactionEffectsDigest;
use types::effects::TransactionEffects;
use types::envelope::Message;
use types::error::SomaError;
use types::messages_grpc::{
    ExecutedData, SubmitTxResult, TxType, WaitForEffectsResponse,
};
use types::object::OBJECT_START_VERSION;
use types::storage::committee_store::CommitteeStore;

use crate::authority_aggregator::AuthorityAggregator;
use crate::test_authority_clients::MockAuthorityApi;
use crate::transaction_driver::SubmitTransactionOptions;
use crate::transaction_driver::error::TransactionDriverError;
use crate::validator_client_monitor::ValidatorClientMonitor;

use super::EffectsCertifier;

/// Create a test setup for effects certifier tests.
fn setup() -> (
    Arc<AuthorityAggregator<MockAuthorityApi>>,
    Arc<ValidatorClientMonitor<MockAuthorityApi>>,
    EffectsCertifier,
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
    let monitor = ValidatorClientMonitor::new(
        ValidatorClientMonitorConfig::default(),
        agg_swap,
    );

    let certifier = EffectsCertifier::new();

    (aggregator, monitor, certifier, mocks, committee)
}

fn default_effects_digest() -> TransactionEffectsDigest {
    TransactionEffects::default().digest()
}

fn default_options() -> SubmitTransactionOptions {
    SubmitTransactionOptions::default()
}

#[tokio::test]
async fn test_successful_certified_effects_with_submitted() {
    let (aggregator, monitor, certifier, mocks, committee) = setup();
    let effects_digest = default_effects_digest();
    let first_name = *committee.names().next().unwrap();

    // Queue wait_for_effects responses for all validators.
    // wait_for_acknowledgments calls ALL validators (1 call each).
    // get_full_effects calls retrier.next_target() which may pick any validator,
    // so we queue 2 responses for every validator to be safe.
    for mock in &mocks {
        // First response: for wait_for_acknowledgments (digest only, no details)
        mock.enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Executed {
            effects_digest,
            details: None,
            fast_path: false,
        }));
        // Second response: in case get_full_effects picks this validator
        mock.enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Executed {
            effects_digest,
            details: Some(Box::new(ExecutedData::default())),
            fast_path: false,
        }));
    }

    let submit_result = SubmitTxResult::Submitted {
        consensus_position: ConsensusPosition {
            epoch: 0,
            block: BlockRef::MIN,
            index: 0,
        },
    };

    let result = certifier
        .get_certified_finalized_effects(
            &aggregator,
            &monitor,
            None,
            TxType::SharedObject,
            first_name,
            submit_result,
            &default_options(),
        )
        .await;

    assert!(result.is_ok(), "Expected success, got: {:?}", result.err());
}

#[tokio::test]
async fn test_successful_certified_effects_with_executed() {
    // When SubmitTxResult::Executed is provided with details, get_full_effects
    // is skipped. Only wait_for_acknowledgments runs, needing 1 response per validator.
    let (aggregator, monitor, certifier, mocks, committee) = setup();
    let effects_digest = default_effects_digest();
    let first_name = *committee.names().next().unwrap();

    // Queue 1 response per validator for wait_for_acknowledgments
    for mock in &mocks {
        mock.enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Executed {
            effects_digest,
            details: None,
            fast_path: false,
        }));
    }

    let submit_result = SubmitTxResult::Executed {
        effects_digest,
        details: Some(Box::new(ExecutedData::default())),
        fast_path: false,
    };

    let result = certifier
        .get_certified_finalized_effects(
            &aggregator,
            &monitor,
            None,
            TxType::SingleWriter,
            first_name,
            submit_result,
            &default_options(),
        )
        .await;

    assert!(result.is_ok(), "Expected success, got: {:?}", result.err());
}

#[tokio::test]
async fn test_forked_execution_detection() {
    // Different validators return different effects digests.
    // With 2 digests each having 2 validators (5000 votes each), neither reaches
    // quorum (6667), and remaining_weight = 0, so it's a ForkedExecution.
    let (aggregator, monitor, certifier, mocks, committee) = setup();
    let first_name = *committee.names().next().unwrap();

    let digest_a = default_effects_digest();
    // Create a different digest by using a different TransactionEffects
    let mut effects_b = TransactionEffects::default();
    effects_b.executed_epoch = 999; // different content → different digest
    let digest_b = effects_b.digest();
    assert_ne!(digest_a, digest_b);

    // First 2 validators return digest_a
    mocks[0].enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Executed {
        effects_digest: digest_a,
        details: None,
        fast_path: false,
    }));
    mocks[1].enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Executed {
        effects_digest: digest_a,
        details: None,
        fast_path: false,
    }));
    // Last 2 validators return digest_b
    mocks[2].enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Executed {
        effects_digest: digest_b,
        details: None,
        fast_path: false,
    }));
    mocks[3].enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Executed {
        effects_digest: digest_b,
        details: None,
        fast_path: false,
    }));

    // We provide full effects to skip the get_full_effects path
    let submit_result = SubmitTxResult::Executed {
        effects_digest: digest_a,
        details: Some(Box::new(ExecutedData::default())),
        fast_path: false,
    };

    let result = certifier
        .get_certified_finalized_effects(
            &aggregator,
            &monitor,
            None,
            TxType::SharedObject,
            first_name,
            submit_result,
            &default_options(),
        )
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        TransactionDriverError::ForkedExecution {
            observed_effects_digests,
            ..
        } => {
            assert_eq!(
                observed_effects_digests.digests.len(),
                2,
                "Expected 2 different effects digests"
            );
        }
        other => panic!("Expected ForkedExecution, got: {}", other),
    }
}

#[tokio::test]
async fn test_non_retriable_rejection_by_validators() {
    // f+1 (2/4) validators return non-retriable rejection errors.
    // The certifier should aggregate and return RejectedByValidators.
    let (aggregator, monitor, certifier, mocks, committee) = setup();
    let effects_digest = default_effects_digest();
    let first_name = *committee.names().next().unwrap();

    let rejection_error = SomaError::InvalidSignature { error: "tx is invalid".to_string() };

    // 2 validators reject with non-retriable error
    mocks[0].enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Rejected {
        error: Some(rejection_error.clone()),
    }));
    mocks[1].enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Rejected {
        error: Some(rejection_error.clone()),
    }));
    // 2 validators succeed
    mocks[2].enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Executed {
        effects_digest,
        details: None,
        fast_path: false,
    }));
    mocks[3].enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Executed {
        effects_digest,
        details: None,
        fast_path: false,
    }));

    let submit_result = SubmitTxResult::Executed {
        effects_digest,
        details: Some(Box::new(ExecutedData::default())),
        fast_path: false,
    };

    let result = certifier
        .get_certified_finalized_effects(
            &aggregator,
            &monitor,
            None,
            TxType::SharedObject,
            first_name,
            submit_result,
            &default_options(),
        )
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        TransactionDriverError::RejectedByValidators {
            submission_non_retriable_errors,
            ..
        } => {
            assert!(
                submission_non_retriable_errors.total_stake > 0,
                "Expected non-retriable errors"
            );
        }
        other => panic!("Expected RejectedByValidators, got: {}", other),
    }
}

#[tokio::test]
async fn test_retriable_responses_result_in_aborted() {
    // All validators return Rejected with no error reason (treated as retriable).
    // Since no effects digest reaches quorum and all responses contribute to
    // retriable weight, the result should be Aborted.
    // Note: we use Rejected{error:None} instead of Err(RpcError) because
    // wait_for_acknowledgment_rpc retries RPC errors internally with backoff.
    let (aggregator, monitor, certifier, mocks, committee) = setup();
    let first_name = *committee.names().next().unwrap();
    let effects_digest = default_effects_digest();

    for mock in &mocks {
        mock.enqueue_wait_for_effects_response(Ok(WaitForEffectsResponse::Rejected {
            error: None,
        }));
    }

    let submit_result = SubmitTxResult::Executed {
        effects_digest,
        details: Some(Box::new(ExecutedData::default())),
        fast_path: false,
    };

    let result = certifier
        .get_certified_finalized_effects(
            &aggregator,
            &monitor,
            None,
            TxType::SharedObject,
            first_name,
            submit_result,
            &default_options(),
        )
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        TransactionDriverError::Aborted { .. } => {}
        other => panic!("Expected Aborted, got: {}", other),
    }
}
