use std::sync::Arc;

use fastcrypto::{ed25519::Ed25519KeyPair, hash::MultisetHash, traits::KeyPair};
use tracing::info;
use types::{
    base::SomaAddress,
    config::network_config::ConfigBuilder,
    consensus::ConsensusTransaction,
    crypto::AuthorityKeyPair,
    effects::SignedTransactionEffects,
    error::{ExecutionError, SomaError},
    genesis::Genesis,
    object::{Object, ObjectID, ObjectRef},
    transaction::{
        CertifiedTransaction, Transaction, TransactionData, VerifiedCertificate,
        VerifiedExecutableTransaction, VerifiedSignedTransaction, VerifiedTransaction,
    },
    unit_tests::utils::to_sender_signed_transaction,
};

use crate::{
    handler::SequencedConsensusTransaction, state::AuthorityState,
    state_accumulator::StateAccumulator, test_authority_builder::TestAuthorityBuilder,
};

pub async fn init_state_with_ids<I: IntoIterator<Item = (SomaAddress, ObjectID)>>(
    objects: I,
) -> Arc<AuthorityState> {
    let state = TestAuthorityBuilder::new().build().await;
    for (address, object_id) in objects {
        let obj = Object::with_id_owner_for_testing(object_id, address);
        // TODO: Make this part of genesis initialization instead of explicit insert.
        state.insert_genesis_object(obj).await;
    }
    state
}

pub async fn init_state_with_objects<I: IntoIterator<Item = Object>>(
    objects: I,
) -> Arc<AuthorityState> {
    let network_config = ConfigBuilder::new().build();
    let genesis = network_config.genesis;
    let keypair = network_config.validator_configs[0]
        .protocol_key_pair()
        .copy();
    init_state_with_objects_and_committee(objects, &genesis, &keypair).await
}

pub async fn init_state_with_objects_and_committee<I: IntoIterator<Item = Object>>(
    objects: I,
    genesis: &Genesis,
    authority_key: &AuthorityKeyPair,
) -> Arc<AuthorityState> {
    let state = init_state_with_committee(genesis, authority_key).await;
    for o in objects {
        state.insert_genesis_object(o).await;
    }
    state
}

pub async fn init_state_with_committee(
    genesis: &Genesis,
    authority_key: &AuthorityKeyPair,
) -> Arc<AuthorityState> {
    // let mut protocol_config =
    //     ProtocolConfig::get_for_version(ProtocolVersion::max(), Chain::Unknown);

    TestAuthorityBuilder::new()
        .with_genesis_and_keypair(genesis, authority_key)
        // .with_protocol_config(protocol_config)
        .build()
        .await
}

pub fn init_transfer_transaction(
    authority_state: &AuthorityState,
    sender: SomaAddress,
    secret: &Ed25519KeyPair,
    recipient: SomaAddress,
    object_ref: ObjectRef,
    // gas_object_ref: ObjectRef,
    // gas_budget: u64,
    // gas_price: u64,
) -> VerifiedTransaction {
    let data = TransactionData::new_transfer(
        recipient, object_ref,
        sender,
        // gas_object_ref,
        // gas_budget,
        // gas_price,
    );
    let tx = to_sender_signed_transaction(data, secret);
    authority_state
        .epoch_store_for_testing()
        .verify_transaction(tx)
        .unwrap()
}

pub fn init_certified_transaction(
    transaction: Transaction,
    authority_state: &AuthorityState,
) -> VerifiedCertificate {
    let epoch_store = authority_state.epoch_store_for_testing();
    let transaction = epoch_store.verify_transaction(transaction).unwrap();

    let vote = VerifiedSignedTransaction::new(
        0,
        transaction.clone(),
        authority_state.name,
        &*authority_state.secret,
    );
    CertifiedTransaction::new(
        transaction.into_message(),
        vec![vote.auth_sig().clone()],
        epoch_store.committee(),
    )
    .unwrap()
    .try_into_verified_for_testing(epoch_store.committee())
    .unwrap()
}

pub async fn send_and_confirm_transaction(
    authority: &AuthorityState,
    transaction: Transaction,
) -> Result<(CertifiedTransaction, SignedTransactionEffects), SomaError> {
    send_and_confirm_transaction_(
        authority,
        None, /* no fullnode_key_pair */
        transaction,
        false, /* no shared objects */
    )
    .await
}
pub async fn send_and_confirm_transaction_(
    authority: &AuthorityState,
    fullnode: Option<&AuthorityState>,
    transaction: Transaction,
    with_shared: bool, // transaction includes shared objects
) -> Result<(CertifiedTransaction, SignedTransactionEffects), SomaError> {
    let (txn, effects, _execution_error_opt) = send_and_confirm_transaction_with_execution_error(
        authority,
        fullnode,
        transaction,
        with_shared,
        true,
    )
    .await?;
    Ok((txn, effects))
}

pub async fn send_and_confirm_transaction_with_execution_error(
    authority: &AuthorityState,
    fullnode: Option<&AuthorityState>,
    transaction: Transaction,
    with_shared: bool,    // transaction includes shared objects
    fake_consensus: bool, // runs consensus handler if true
) -> Result<
    (
        CertifiedTransaction,
        SignedTransactionEffects,
        Option<ExecutionError>,
    ),
    SomaError,
> {
    let certificate = certify_transaction(authority, transaction).await?;
    execute_certificate_with_execution_error(
        authority,
        fullnode,
        certificate,
        with_shared,
        fake_consensus,
    )
    .await
}

pub async fn certify_transaction(
    authority: &AuthorityState,
    transaction: Transaction,
) -> Result<VerifiedCertificate, SomaError> {
    // Make the initial request
    let epoch_store = authority.load_epoch_store_one_call_per_task();
    // TODO: Move this check to a more appropriate place.
    // TODO: transaction.validity_check(epoch_store.protocol_config(), epoch_store.epoch())?;
    let transaction = epoch_store.verify_transaction(transaction).unwrap();

    let response = authority
        .handle_transaction(&epoch_store, transaction.clone())
        .await?;
    let vote = response.status.into_signed_for_testing();

    // Collect signatures from a quorum of authorities
    let committee = authority.clone_committee_for_testing();
    let certificate = CertifiedTransaction::new(transaction.into_message(), vec![vote], &committee)
        .unwrap()
        .try_into_verified_for_testing(&committee)
        .unwrap();
    Ok(certificate)
}

pub async fn execute_certificate_with_execution_error(
    authority: &AuthorityState,
    fullnode: Option<&AuthorityState>,
    certificate: VerifiedCertificate,
    with_shared: bool, // transaction includes shared objects
    fake_consensus: bool,
) -> Result<
    (
        CertifiedTransaction,
        SignedTransactionEffects,
        Option<ExecutionError>,
    ),
    SomaError,
> {
    let epoch_store = authority.load_epoch_store_one_call_per_task();
    // We also check the incremental effects of the transaction on the live object set against StateAccumulator
    // for testing and regression detection.
    // We must do this before sending to consensus, otherwise consensus may already
    // lead to transaction execution and state change.
    let state_acc = StateAccumulator::new(authority.get_accumulator_store().clone());
    let mut state = state_acc.accumulate_cached_live_object_set_for_testing();

    if with_shared {
        if fake_consensus {
            send_consensus(authority, &certificate).await;
        } else {
            // Just set object locks directly if send_consensus is not requested.
            authority
                .epoch_store_for_testing()
                .assign_shared_object_versions_for_tests(
                    authority.get_object_cache_reader().as_ref(),
                    &vec![VerifiedExecutableTransaction::new_from_certificate(
                        certificate.clone(),
                    )],
                )?;
        }
        if let Some(fullnode) = fullnode {
            fullnode
                .epoch_store_for_testing()
                .assign_shared_object_versions_for_tests(
                    fullnode.get_object_cache_reader().as_ref(),
                    &vec![VerifiedExecutableTransaction::new_from_certificate(
                        certificate.clone(),
                    )],
                )?;
        }
    }

    // Submit the confirmation. *Now* execution actually happens, and it should fail when we try to look up our dummy module.
    // we unfortunately don't get a very descriptive error message, but we can at least see that something went wrong inside the VM
    let (result, execution_error_opt) = authority.try_execute_for_test(&certificate).await?;
    let state_after = state_acc.accumulate_cached_live_object_set_for_testing();
    let effects_acc = state_acc.accumulate_effects(vec![result.inner().data().clone()]);
    state.union(&effects_acc);

    // TODO: assert_eq!(state_after.digest(), state.digest());

    if let Some(fullnode) = fullnode {
        fullnode.try_execute_for_test(&certificate).await?;
    }
    Ok((
        certificate.into_inner(),
        result.into_inner(),
        execution_error_opt,
    ))
}

pub async fn send_consensus(authority: &AuthorityState, cert: &VerifiedCertificate) {
    let transaction = SequencedConsensusTransaction::new_test(
        ConsensusTransaction::new_certificate_message(&authority.name, cert.clone().into_inner()),
    );

    let (certs, _) = authority
        .epoch_store_for_testing()
        .process_consensus_transactions_for_tests(
            vec![transaction],
            authority.get_object_cache_reader().as_ref(),
            true,
        )
        .await
        .unwrap();

    authority
        .transaction_manager()
        .enqueue(certs, &authority.epoch_store_for_testing(), None);
}
