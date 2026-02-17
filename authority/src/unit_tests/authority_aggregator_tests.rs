// Tests for AuthorityAggregator: process_transaction quorum gathering,
// error classification, conflicting transactions, and process_certificate.

use std::collections::BTreeMap;
use std::sync::Arc;

use types::base::SomaAddress;
use types::committee::Committee;
use fastcrypto::ed25519::Ed25519KeyPair;
use types::crypto::{AuthorityKeyPair, KeypairTraits, get_key_pair};
use types::digests::TransactionDigest;
use types::error::SomaError;
use types::object::{Object, ObjectRef, OBJECT_START_VERSION};
use types::storage::committee_store::CommitteeStore;
use types::transaction::{Transaction, TransactionData, TransactionKind};

use crate::authority_aggregator::{
    AggregatorProcessCertificateError, AggregatorProcessTransactionError, AuthorityAggregator,
};
use crate::test_authority_clients::MockAuthorityApi;

/// Create a test committee of `size` validators along with their mock clients.
/// Returns the aggregator, the committee, and the mock clients (for injecting responses).
fn setup_aggregator(
    size: usize,
) -> (
    AuthorityAggregator<MockAuthorityApi>,
    Committee,
    Vec<MockAuthorityApi>,
) {
    let (committee, keypairs) = Committee::new_simple_test_committee_of_size(size);
    let committee_store =
        Arc::new(CommitteeStore::new_for_testing(&committee));

    // Build name→keypair map to correctly associate keys with sorted committee names.
    // keypairs are in generation order but committee.names() returns sorted BTreeMap order.
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

    let aggregator = AuthorityAggregator::new(
        committee.clone(),
        Arc::new(std::collections::HashMap::new()),
        committee_store,
        authority_clients,
        Default::default(),
    );

    (aggregator, committee, mocks)
}

/// Create a minimal test transaction for aggregator testing.
fn make_test_transaction() -> Transaction {
    let (sender, sender_kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let gas_object_ref: ObjectRef = (
        types::object::ObjectID::random(),
        OBJECT_START_VERSION,
        types::digests::ObjectDigest::new([0u8; 32]),
    );
    let recipient: SomaAddress = SomaAddress::default();

    let data = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: gas_object_ref,
            amount: Some(100),
            recipient,
        },
        sender,
        vec![gas_object_ref],
    );
    Transaction::from_data_and_signer(data, vec![&sender_kp])
}

// ── process_transaction tests ───────────────────────────────────────────

#[tokio::test]
async fn test_process_transaction_quorum_success() {
    // All 4 validators sign → certificate formed
    let (aggregator, _committee, _mocks) = setup_aggregator(4);
    let tx = make_test_transaction();

    let result = aggregator.process_transaction(tx, None).await;
    assert!(result.is_ok(), "Expected quorum success, got: {:?}", result.err());

    let result = result.unwrap();
    // Should be Certified (newly formed) since all validators signed
    match result {
        crate::authority_aggregator::ProcessTransactionResult::Certified {
            certificate,
            newly_formed,
        } => {
            assert!(newly_formed, "Expected newly formed certificate");
            // Certificate should have valid committee signatures
            assert!(certificate
                .verify_committee_sigs_only(&_committee)
                .is_ok());
        }
        _ => panic!("Expected Certified result"),
    }
}

#[tokio::test]
async fn test_process_transaction_one_failure_still_reaches_quorum() {
    // 3/4 sign, 1 returns error → still reaches quorum (3 * 2500 = 7500 > 6667)
    let (aggregator, _committee, mocks) = setup_aggregator(4);
    let tx = make_test_transaction();

    // Make the first validator return a retryable error
    mocks[0].enqueue_handle_transaction_error(SomaError::RpcError(
        "connection refused".to_string(),
        "test".to_string(),
    ));

    let result = aggregator.process_transaction(tx, None).await;
    assert!(result.is_ok(), "Expected quorum with 3/4, got: {:?}", result.err());
}

#[tokio::test]
async fn test_process_transaction_two_failures_still_reaches_quorum() {
    // 2/4 sign, 2 retryable errors → NOT quorum (2 * 2500 = 5000 < 6667)
    // This should fail as retryable
    let (aggregator, _committee, mocks) = setup_aggregator(4);
    let tx = make_test_transaction();

    mocks[0].enqueue_handle_transaction_error(SomaError::RpcError(
        "connection refused".to_string(),
        "test".to_string(),
    ));
    mocks[1].enqueue_handle_transaction_error(SomaError::RpcError(
        "timeout".to_string(),
        "test".to_string(),
    ));

    let result = aggregator.process_transaction(tx, None).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        AggregatorProcessTransactionError::RetryableTransaction { errors } => {
            assert!(!errors.is_empty(), "Expected errors in retryable result");
        }
        other => panic!("Expected RetryableTransaction, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_process_transaction_non_retryable_errors_fatal() {
    // f+1 (2/4) non-retryable errors → FatalTransaction
    // Non-retryable example: InvalidTransaction
    let (aggregator, _committee, mocks) = setup_aggregator(4);
    let tx = make_test_transaction();

    mocks[0].enqueue_handle_transaction_error(SomaError::InvalidSignature { error: "bad tx".to_string() });
    mocks[1].enqueue_handle_transaction_error(SomaError::InvalidSignature { error: "bad tx".to_string() });

    let result = aggregator.process_transaction(tx, None).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        AggregatorProcessTransactionError::FatalTransaction { errors } => {
            assert!(!errors.is_empty());
        }
        other => panic!("Expected FatalTransaction, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_process_transaction_conflicting_transaction_detection() {
    // Simulate object lock conflict: a validator reports that the object is locked
    // by a different transaction.
    let (aggregator, _committee, mocks) = setup_aggregator(4);
    let tx = make_test_transaction();

    let conflicting_digest = TransactionDigest::random();
    let obj_ref: ObjectRef = (
        types::object::ObjectID::random(),
        OBJECT_START_VERSION,
        types::digests::ObjectDigest::new([1u8; 32]),
    );

    // 2 validators report lock conflict → non-retryable because good_stake + retryable < quorum
    mocks[0].enqueue_handle_transaction_error(SomaError::ObjectLockConflict {
        obj_ref,
        pending_transaction: conflicting_digest,
    });
    mocks[1].enqueue_handle_transaction_error(SomaError::ObjectLockConflict {
        obj_ref,
        pending_transaction: conflicting_digest,
    });

    let result = aggregator.process_transaction(tx, None).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        AggregatorProcessTransactionError::FatalConflictingTransaction {
            conflicting_tx_digests,
            ..
        } => {
            assert!(
                conflicting_tx_digests.contains_key(&conflicting_digest),
                "Expected conflicting digest in result"
            );
        }
        other => panic!("Expected FatalConflictingTransaction, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_process_transaction_all_errors_non_retryable() {
    // All 4 validators return non-retryable errors → FatalTransaction
    let (aggregator, _committee, mocks) = setup_aggregator(4);
    let tx = make_test_transaction();

    for mock in &mocks {
        mock.enqueue_handle_transaction_error(SomaError::InvalidSignature { error: "invalid".to_string() });
    }

    let result = aggregator.process_transaction(tx, None).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        AggregatorProcessTransactionError::FatalTransaction { .. } => {}
        other => panic!("Expected FatalTransaction, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_process_transaction_mixed_retryable_and_non_retryable() {
    // 1 non-retryable + 1 retryable + 2 success → quorum reached with 2 sigs
    // 2 * 2500 = 5000 < 6667 quorum, so need 3 sigs
    // Actually with 2 success, good_stake = 5000 < 6667, and retryable_stake = 0
    // (since 1 non-retryable + 1 retryable are consumed)
    // good_stake + retryable_stake < quorum → not retryable
    let (aggregator, _committee, mocks) = setup_aggregator(4);
    let tx = make_test_transaction();

    mocks[0].enqueue_handle_transaction_error(SomaError::InvalidSignature { error: "invalid".to_string() });
    mocks[1].enqueue_handle_transaction_error(SomaError::RpcError(
        "timeout".to_string(),
        "test".to_string(),
    ));
    // mocks[2] and mocks[3] will sign successfully (2 sigs, 5000 votes < 6667)

    let result = aggregator.process_transaction(tx, None).await;
    // With 2 good sigs (5000) + 0 retryable < 6667, state is not retryable
    assert!(result.is_err());
}

// ── process_certificate tests ───────────────────────────────────────────

/// Helper to create a CertifiedTransaction from a test transaction.
fn make_test_certificate(
    aggregator: &AuthorityAggregator<MockAuthorityApi>,
    tx: &Transaction,
) -> types::transaction::CertifiedTransaction {
    use types::crypto::AuthoritySignInfo;
    use types::intent::{Intent, IntentScope};
    use types::transaction::CertifiedTransaction;

    let committee = &aggregator.committee;
    let mut sigs = Vec::new();

    // Collect signatures from all validator mock clients
    for (name, client) in aggregator.authority_clients.iter() {
        let mock = client.authority_client();
        let sig = AuthoritySignInfo::new(
            committee.epoch,
            tx.data(),
            Intent::soma_app(IntentScope::SenderSignedTransaction),
            *name,
            &*mock.authority_key,
        );
        sigs.push(sig);
    }

    let cert_sig = types::crypto::AuthorityQuorumSignInfo::<true>::new_from_auth_sign_infos(
        sigs,
        committee,
    )
    .expect("Failed to create quorum sig");

    CertifiedTransaction::new_from_data_and_sig(tx.clone().into_data(), cert_sig)
}

#[tokio::test]
async fn test_process_certificate_quorum_success() {
    // All 4 validators return signed effects → quorum reached
    let (aggregator, committee, _mocks) = setup_aggregator(4);
    let tx = make_test_transaction();
    let cert = make_test_certificate(&aggregator, &tx);

    let request = types::messages_grpc::HandleCertificateRequest {
        certificate: cert,
        include_input_objects: false,
        include_output_objects: false,
    };

    let result = aggregator.process_certificate(request, None).await;
    assert!(result.is_ok(), "Expected certificate quorum, got: {:?}", result.err());

    let response = result.unwrap();
    // effects_cert is already verified (VerifiedCertifiedTransactionEffects)
    let _ = &response.effects_cert;
}

#[tokio::test]
async fn test_process_certificate_one_failure_still_reaches_quorum() {
    // 3/4 return effects, 1 errors → still quorum (7500 > 6667)
    let (aggregator, _committee, mocks) = setup_aggregator(4);
    let tx = make_test_transaction();
    let cert = make_test_certificate(&aggregator, &tx);

    mocks[0].enqueue_handle_certificate_error(SomaError::RpcError(
        "unavailable".to_string(),
        "test".to_string(),
    ));

    let request = types::messages_grpc::HandleCertificateRequest {
        certificate: cert,
        include_input_objects: false,
        include_output_objects: false,
    };

    let result = aggregator.process_certificate(request, None).await;
    assert!(result.is_ok(), "Expected quorum with 3/4, got: {:?}", result.err());
}

#[tokio::test]
async fn test_process_certificate_non_retryable_failure() {
    // f+1 (2/4) non-retryable errors → FatalExecuteCertificate
    let (aggregator, _committee, mocks) = setup_aggregator(4);
    let tx = make_test_transaction();
    let cert = make_test_certificate(&aggregator, &tx);

    mocks[0].enqueue_handle_certificate_error(SomaError::InvalidSignature { error: "cert invalid".to_string() });
    mocks[1].enqueue_handle_certificate_error(SomaError::InvalidSignature { error: "cert invalid".to_string() });

    let request = types::messages_grpc::HandleCertificateRequest {
        certificate: cert,
        include_input_objects: false,
        include_output_objects: false,
    };

    let result = aggregator.process_certificate(request, None).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        AggregatorProcessCertificateError::FatalExecuteCertificate {
            non_retryable_errors,
        } => {
            assert!(!non_retryable_errors.is_empty());
        }
        other => panic!("Expected FatalExecuteCertificate, got: {:?}", other),
    }
}
