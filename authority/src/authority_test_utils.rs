// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use core::default::Default;
use std::sync::Arc;

use fastcrypto::hash::MultisetHash;
use fastcrypto::traits::KeyPair;
use protocol_config::{Chain, ProtocolConfig, ProtocolVersion};
use types::base::{FullObjectRef, SomaAddress};
use types::config::node_config::ExpensiveSafetyCheckConfig;
use types::crypto::{AuthorityKeyPair, SomaKeyPair};
use types::effects::{SignedTransactionEffects, TransactionEffects};
use types::error::{ExecutionError, SomaError};
use types::genesis::Genesis;
use types::object::{Object, ObjectID, ObjectRef, Owner, Version};
use types::transaction::{
    CertifiedTransaction, Transaction, TransactionData, VerifiedCertificate,
    VerifiedExecutableTransaction, VerifiedSignedTransaction, VerifiedTransaction,
};
use types::unit_tests::utils::to_sender_signed_transaction;

use super::shared_obj_version_manager::AssignedVersions;
#[cfg(test)]
use super::shared_obj_version_manager::{AssignedTxAndVersions, Schedulable};
use super::test_authority_builder::TestAuthorityBuilder;
use crate::authority::*;
use crate::global_state_hasher::GlobalStateHasher;

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

pub async fn certify_transaction(
    authority: &AuthorityState,
    transaction: Transaction,
) -> Result<VerifiedCertificate, SomaError> {
    // Make the initial request
    let epoch_store = authority.load_epoch_store_one_call_per_task();

    let transaction = epoch_store.verify_transaction(transaction).unwrap();

    let response = authority.handle_transaction(&epoch_store, transaction.clone()).await?;
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
) -> Result<(CertifiedTransaction, SignedTransactionEffects, Option<ExecutionError>), SomaError> {
    let epoch_store = authority.load_epoch_store_one_call_per_task();
    // We also check the incremental effects of the transaction on the live object set against StateAccumulator
    // for testing and regression detection.
    // We must do this before sending to consensus, otherwise consensus may already
    // lead to transaction execution and state change.
    let state_acc = GlobalStateHasher::new(authority.get_global_state_hash_store().clone());

    let mut state = state_acc.accumulate_cached_live_object_set_for_testing(false);

    let assigned_versions = if with_shared {
        if fake_consensus {
            send_consensus(authority, &certificate).await
        } else {
            // Just set object locks directly if send_consensus is not requested.
            let assigned_versions =
                authority.epoch_store_for_testing().assign_shared_object_versions_for_tests(
                    authority.get_object_cache_reader().as_ref(),
                    &[VerifiedExecutableTransaction::new_from_certificate(certificate.clone())],
                )?;
            assigned_versions.into_map().get(&certificate.key()).cloned().unwrap()
        }
    } else {
        AssignedVersions::new(vec![])
    };

    // Submit the confirmation. *Now* execution actually happens, and it should fail when we try to look up our dummy module.
    // we unfortunately don't get a very descriptive error message, but we can at least see that something went wrong inside the VM
    let (result, execution_error_opt) = authority
        .try_execute_for_test(
            &certificate,
            ExecutionEnv::new().with_assigned_versions(assigned_versions.clone()),
        )
        .await;
    let state_after = state_acc.accumulate_cached_live_object_set_for_testing(false);
    let effects_acc = state_acc
        .accumulate_effects(&[result.inner().data().clone()], epoch_store.protocol_config());
    state.union(&effects_acc);

    assert_eq!(state_after.digest(), state.digest());

    if let Some(fullnode) = fullnode {
        fullnode
            .try_execute_for_test(
                &certificate,
                ExecutionEnv::new().with_assigned_versions(assigned_versions),
            )
            .await;
    }
    Ok((certificate.into_inner(), result.into_inner(), execution_error_opt))
}

pub async fn send_and_confirm_transaction_with_execution_error(
    authority: &AuthorityState,
    fullnode: Option<&AuthorityState>,
    transaction: Transaction,
    with_shared: bool,    // transaction includes shared objects
    fake_consensus: bool, // runs consensus handler if true
) -> Result<(CertifiedTransaction, SignedTransactionEffects, Option<ExecutionError>), SomaError> {
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

pub async fn init_state_validator_with_fullnode() -> (Arc<AuthorityState>, Arc<AuthorityState>) {
    use types::crypto::get_key_pair;

    let validator = TestAuthorityBuilder::new().build().await;
    let fullnode_key_pair = get_key_pair().1;
    let fullnode = TestAuthorityBuilder::new().with_keypair(&fullnode_key_pair).build().await;
    (validator, fullnode)
}

pub async fn init_state_with_committee(
    genesis: &Genesis,
    authority_key: &AuthorityKeyPair,
) -> Arc<AuthorityState> {
    let mut protocol_config =
        ProtocolConfig::get_for_version(ProtocolVersion::max(), Chain::Unknown);

    TestAuthorityBuilder::new()
        .with_genesis_and_keypair(genesis, authority_key)
        .with_protocol_config(protocol_config)
        .build()
        .await
}

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

pub async fn init_state_with_ids_and_versions<
    I: IntoIterator<Item = (SomaAddress, ObjectID, Version)>,
>(
    objects: I,
) -> Arc<AuthorityState> {
    let state = TestAuthorityBuilder::new().build().await;
    for (address, object_id, version) in objects {
        let obj = Object::with_id_owner_version_for_testing(
            object_id,
            version,
            Owner::AddressOwner(address),
        );
        state.insert_genesis_object(obj).await;
    }
    state
}

pub async fn init_state_with_objects<I: IntoIterator<Item = Object>>(
    objects: I,
) -> Arc<AuthorityState> {
    let dir = tempfile::TempDir::new().unwrap();
    let network_config = types::config::network_config::ConfigBuilder::new(dir.as_ref()).build();
    let genesis = network_config.genesis;
    let keypair = network_config.validator_configs[0].protocol_key_pair().copy();
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

pub async fn init_state_with_object_id(
    address: SomaAddress,
    object: ObjectID,
) -> Arc<AuthorityState> {
    init_state_with_ids(std::iter::once((address, object))).await
}

pub async fn init_state_with_ids_and_expensive_checks<
    I: IntoIterator<Item = (SomaAddress, ObjectID)>,
>(
    objects: I,
    config: ExpensiveSafetyCheckConfig,
) -> Arc<AuthorityState> {
    let state = TestAuthorityBuilder::new().with_expensive_safety_checks(config).build().await;
    for (address, object_id) in objects {
        let obj = Object::with_id_owner_for_testing(object_id, address);
        // TODO: Make this part of genesis initialization instead of explicit insert.
        state.insert_genesis_object(obj).await;
    }
    state
}

pub fn init_transfer_transaction(
    authority_state: &AuthorityState,
    sender: SomaAddress,
    secret: &SomaKeyPair,
    recipient: SomaAddress,
    object_ref: ObjectRef,
    gas_object_ref: ObjectRef,
) -> VerifiedTransaction {
    let data = TransactionData::new_transfer(recipient, object_ref, sender, vec![gas_object_ref]);
    let tx = to_sender_signed_transaction(data, secret);
    authority_state.epoch_store_for_testing().verify_transaction(tx).unwrap()
}

pub fn init_certified_transfer_transaction(
    sender: SomaAddress,
    secret: &SomaKeyPair,
    recipient: SomaAddress,
    object_ref: ObjectRef,
    gas_object_ref: ObjectRef,
    authority_state: &AuthorityState,
) -> VerifiedCertificate {
    let transfer_transaction = init_transfer_transaction(
        authority_state,
        sender,
        secret,
        recipient,
        object_ref,
        gas_object_ref,
    );
    init_certified_transaction(transfer_transaction.into(), authority_state)
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

pub async fn certify_shared_obj_transaction_no_execution(
    authority: &AuthorityState,
    transaction: Transaction,
) -> Result<(VerifiedCertificate, AssignedVersions), SomaError> {
    let epoch_store = authority.load_epoch_store_one_call_per_task();
    let transaction = epoch_store.verify_transaction(transaction).unwrap();
    let response = authority.handle_transaction(&epoch_store, transaction.clone()).await?;
    let vote = response.status.into_signed_for_testing();

    // Collect signatures from a quorum of authorities
    let committee = authority.clone_committee_for_testing();
    let certificate =
        CertifiedTransaction::new(transaction.into_message(), vec![vote.clone()], &committee)
            .unwrap()
            .try_into_verified_for_testing(&committee)
            .unwrap();

    let assigned_versions = send_consensus_no_execution(authority, &certificate).await;

    Ok((certificate, assigned_versions))
}

pub async fn enqueue_all_and_execute_all(
    authority: &AuthorityState,
    certificates: Vec<(VerifiedCertificate, ExecutionEnv)>,
) -> Result<Vec<TransactionEffects>, SomaError> {
    authority.execution_scheduler.enqueue(
        certificates
            .iter()
            .map(|(cert, env)| {
                (
                    VerifiedExecutableTransaction::new_from_certificate(cert.clone()).into(),
                    env.clone(),
                )
            })
            .collect(),
        &authority.epoch_store_for_testing(),
    );
    let mut output = Vec::new();
    for (cert, _) in certificates {
        let effects = authority.notify_read_effects(*cert.digest()).await?;
        output.push(effects);
    }
    Ok(output)
}

pub async fn execute_sequenced_certificate_to_effects(
    authority: &AuthorityState,
    certificate: VerifiedCertificate,
    assigned_versions: AssignedVersions,
) -> (TransactionEffects, Option<ExecutionError>) {
    let env = ExecutionEnv::new().with_assigned_versions(assigned_versions);
    authority.execution_scheduler.enqueue(
        vec![(
            VerifiedExecutableTransaction::new_from_certificate(certificate.clone()).into(),
            env.clone(),
        )],
        &authority.epoch_store_for_testing(),
    );

    let (result, execution_error_opt) = authority.try_execute_for_test(&certificate, env).await;
    let effects = result.inner().data().clone();
    (effects, execution_error_opt)
}

pub async fn send_consensus(
    authority: &AuthorityState,
    cert: &VerifiedCertificate,
) -> AssignedVersions {
    let assigned_versions = authority
        .epoch_store_for_testing()
        .assign_shared_object_versions_for_tests(
            authority.get_object_cache_reader().as_ref(),
            &[VerifiedExecutableTransaction::new_from_certificate(cert.clone())],
        )
        .unwrap();

    let assigned_versions = assigned_versions
        .into_map()
        .get(&cert.key())
        .cloned()
        .unwrap_or_else(|| AssignedVersions::new(vec![]));

    let certs = vec![(
        VerifiedExecutableTransaction::new_from_certificate(cert.clone()),
        ExecutionEnv::new().with_assigned_versions(assigned_versions.clone()),
    )];

    authority
        .execution_scheduler()
        .enqueue_transactions(certs, &authority.epoch_store_for_testing());

    assigned_versions
}

pub async fn send_consensus_no_execution(
    authority: &AuthorityState,
    cert: &VerifiedCertificate,
) -> AssignedVersions {
    // Use the simpler assign_shared_object_versions_for_tests API to avoid actually executing cert.
    // This allows testing cert execution independently.
    let assigned_versions = authority
        .epoch_store_for_testing()
        .assign_shared_object_versions_for_tests(
            authority.get_object_cache_reader().as_ref(),
            &[VerifiedExecutableTransaction::new_from_certificate(cert.clone())],
        )
        .unwrap();

    assigned_versions
        .into_map()
        .get(&cert.key())
        .cloned()
        .unwrap_or_else(|| AssignedVersions::new(vec![]))
}

#[cfg(test)]
pub async fn send_batch_consensus_no_execution<C>(
    authority: &AuthorityState,
    certificates: &[VerifiedCertificate],
    consensus_handler: &mut crate::consensus_handler::ConsensusHandler<C>,
    captured_transactions: &crate::consensus_test_utils::CapturedTransactions,
) -> (Vec<Schedulable>, AssignedTxAndVersions)
where
    C: crate::checkpoints::CheckpointServiceNotify + Send + Sync + 'static,
{
    use types::consensus::ConsensusTransaction;
    use types::system_state::epoch_start::EpochStartSystemStateTrait;

    use crate::consensus_test_utils::TestConsensusCommit;

    let consensus_transactions: Vec<ConsensusTransaction> = certificates
        .iter()
        .map(|cert| {
            ConsensusTransaction::new_certificate_message(&authority.name, cert.clone().into())
        })
        .collect();

    // Determine appropriate round and timestamp
    let epoch_store = authority.epoch_store_for_testing();
    let round = epoch_store.get_highest_pending_checkpoint_height() + 1;
    let timestamp_ms = epoch_store.epoch_start_state().epoch_start_timestamp_ms();
    let sub_dag_index = 0;

    let commit =
        TestConsensusCommit::new(consensus_transactions, round, timestamp_ms, sub_dag_index);

    consensus_handler.handle_consensus_commit_for_test(commit).await;

    // Wait for captured transactions to be available
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let (scheduled_txns, assigned_tx_and_versions) = {
        let mut captured = captured_transactions.lock();
        assert!(!captured.is_empty(), "Expected transactions to be scheduled");
        let (scheduled_txns, assigned_tx_and_versions, _) = captured.remove(0);
        (scheduled_txns, assigned_tx_and_versions)
    };

    (scheduled_txns, assigned_tx_and_versions)
}
