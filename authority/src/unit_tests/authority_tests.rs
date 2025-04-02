use fastcrypto::ed25519::Ed25519KeyPair;
use futures::{stream::FuturesUnordered, StreamExt};
use tracing::info;
use types::{
    base::dbg_addr,
    crypto::get_key_pair,
    effects::TransactionEffectsAPI,
    error::SomaError,
    object::{Object, ObjectID, Owner},
    transaction::TransactionData,
    unit_tests::utils::to_sender_signed_transaction,
};
use utils::logging::init_tracing;

use crate::authority_test_utils::{
    init_certified_transaction, init_state_with_ids, init_state_with_objects,
    init_transfer_transaction,
};

#[tokio::test]
async fn test_transfer_coin_no_amount() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(2);
    let coin_object_id = ObjectID::random();
    let coin_object = Object::with_id_owner_for_testing(coin_object_id, sender);
    let init_balance = coin_object.as_coin().unwrap();
    let authority_state = init_state_with_objects(vec![coin_object.clone()]).await;

    let epoch_store = authority_state.load_epoch_store_one_call_per_task();
    // let rgp = epoch_store.reference_gas_price();

    let coin_ref = coin_object.compute_object_reference();
    let tx_data = TransactionData::new_transfer_coin(
        recipient, sender, None,
        coin_ref,
        // rgp * TEST_ONLY_GAS_UNIT_FOR_TRANSFER,
        // rgp,
    );

    // Make sure transaction handling works as usual.
    let transaction = to_sender_signed_transaction(tx_data, &sender_key);
    let transaction = epoch_store.verify_transaction(transaction).unwrap();
    authority_state
        .handle_transaction(&epoch_store, transaction.clone())
        .await
        .unwrap();

    let certificate = init_certified_transaction(transaction.into(), &authority_state);
    let effects = authority_state
        .execute_certificate(&certificate, &authority_state.epoch_store_for_testing())
        .await
        .unwrap();
    // Check that the transaction was successful, and the gas object is the only mutated object,
    // and got transferred. Also check on its version and new balance.
    assert!(effects.status().is_ok());
    assert_eq!(effects.mutated().len(), 1);
    assert_eq!(effects.mutated()[0].1, Owner::AddressOwner(recipient));

    // TODO: after implementing gas
    // assert!(effects.mutated_excluding_gas().is_empty());
    // assert!(gas_ref.1 < effects.gas_object().0 .1);
    // assert_eq!(effects.gas_object().1, Owner::AddressOwner(recipient));
    let new_balance = authority_state
        .get_object(&coin_object_id)
        .await
        .unwrap()
        .as_coin()
        .unwrap();
    assert_eq!(
        new_balance, /*+ effects.gas_cost_summary().net_gas_usage()*/
        init_balance
    );
}

#[tokio::test]
async fn test_transfer_coin_with_amount() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(2);
    let gas_object_id = ObjectID::random();
    let gas_object = Object::with_id_owner_for_testing(gas_object_id, sender);
    let init_balance = gas_object.as_coin().unwrap();
    let authority_state = init_state_with_objects(vec![gas_object.clone()]).await;
    // let rgp = authority_state.reference_gas_price_for_testing().unwrap();

    let gas_ref = gas_object.compute_object_reference();
    let tx_data = TransactionData::new_transfer_coin(
        recipient,
        sender,
        Some(500),
        gas_ref,
        // rgp * TEST_ONLY_GAS_UNIT_FOR_TRANSFER,
        // rgp,
    );
    let transaction = to_sender_signed_transaction(tx_data, &sender_key);
    let certificate = init_certified_transaction(transaction, &authority_state);
    let effects = authority_state
        .execute_certificate(&certificate, &authority_state.epoch_store_for_testing())
        .await
        .unwrap();
    // Check that the transaction was successful, the gas object remains in the original owner,
    // and an amount is split out and send to the recipient.
    assert!(effects.status().is_ok());
    assert_eq!(effects.mutated().len(), 1);
    // TODO: assert!(effects.mutated_excluding_gas().is_empty());
    assert_eq!(effects.created().len(), 1);
    assert_eq!(effects.created()[0].1, Owner::AddressOwner(recipient));
    let new_gas = authority_state
        .get_object(&effects.created()[0].0 .0)
        .await
        .unwrap();
    assert_eq!(new_gas.as_coin().unwrap(), 500);
    // assert!(gas_ref.1 < effects.gas_object().0 .1);
    // assert_eq!(effects.gas_object().1, Owner::AddressOwner(sender));
    assert_eq!(effects.mutated()[0].1, Owner::AddressOwner(sender));
    let new_balance = authority_state
        .get_object(&gas_object_id)
        .await
        .unwrap()
        .as_coin()
        .unwrap();
    assert_eq!(
        new_balance as i64 + 500, //+ effects.gas_cost_summary().net_gas_usage() + 500,
        init_balance as i64
    );
}

#[tokio::test]
async fn test_handle_transfer_transaction_double_spend() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient = dbg_addr(2);
    let object_id = ObjectID::random();
    let gas_object_id = ObjectID::random();
    let authority_state =
        init_state_with_ids(vec![(sender, object_id), (sender, gas_object_id)]).await;

    // let rgp = authority_state.reference_gas_price_for_testing().unwrap();
    let epoch_store = authority_state.load_epoch_store_one_call_per_task();
    let object = authority_state.get_object(&object_id).await.unwrap();
    let gas_object = authority_state.get_object(&gas_object_id).await.unwrap();
    let transfer_transaction = init_transfer_transaction(
        &authority_state,
        sender,
        &sender_key,
        recipient,
        object.compute_object_reference(),
        gas_object.compute_object_reference(),
    );

    let signed_transaction = authority_state
        .handle_transaction(&epoch_store, transfer_transaction.clone())
        .await
        .unwrap();
    // calls to handlers are idempotent -- returns the same.
    let double_spend_signed_transaction = authority_state
        .handle_transaction(&epoch_store, transfer_transaction)
        .await
        .unwrap();
    // this is valid because our test authority should not change its certified transaction
    assert_eq!(signed_transaction, double_spend_signed_transaction);
}

#[cfg(msim)]
#[msim::sim_test]
async fn test_conflicting_transactions() {
    // let _ = tracing_subscriber::fmt::try_init();
    init_tracing();
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient1 = dbg_addr(2);
    let recipient2 = dbg_addr(3);
    let object_id = ObjectID::random();
    let gas_object_id = ObjectID::random();
    let authority_state =
        init_state_with_ids(vec![(sender, object_id), (sender, gas_object_id)]).await;

    // // let rgp = authority_state.reference_gas_price_for_testing().unwrap();
    let epoch_store = authority_state.load_epoch_store_one_call_per_task();
    let object = authority_state.get_object(&object_id).await.unwrap();
    let gas_object = authority_state.get_object(&gas_object_id).await.unwrap();

    let tx1 = init_transfer_transaction(
        &authority_state,
        sender,
        &sender_key,
        recipient1,
        object.compute_object_reference(),
        gas_object.compute_object_reference(),
    );

    let tx2 = init_transfer_transaction(
        &authority_state,
        sender,
        &sender_key,
        recipient2,
        object.compute_object_reference(),
        gas_object.compute_object_reference(),
    );

    // repeatedly attempt to submit conflicting transactions at the same time, and verify that
    // exactly one succeeds in every case.
    //
    // Note: this test fails immediately if we remove the acquire_locks() call in
    // acquire_transaction_locks() and then add a sleep after we read the locks.
    // for _ in 0..100 {
    let mut futures = FuturesUnordered::new();
    futures.push(authority_state.handle_transaction(&epoch_store, tx1.clone()));
    futures.push(authority_state.handle_transaction(&epoch_store, tx2.clone()));

    let first = futures.next().await.unwrap();
    let second = futures.next().await.unwrap();
    assert!(futures.next().await.is_none());

    // exactly one should fail.
    assert!(first.is_ok() != second.is_ok());

    let (ok, err) = if first.is_ok() {
        (first.unwrap(), second.unwrap_err())
    } else {
        (second.unwrap(), first.unwrap_err())
    };

    assert!(matches!(err, SomaError::ObjectLockConflict { .. }));

    let lock = authority_state
        .get_latest_object_lock_for_testing(object_id)
        .await
        .unwrap();
    let gas_lock = authority_state
        .get_latest_object_lock_for_testing(gas_object.id())
        .await
        .unwrap();

    let lock_digest = (lock.clone()).unwrap().digest().clone();

    assert_eq!(
        &ok.clone().status.into_signed_for_testing(),
        lock.clone().expect("object should be locked").auth_sig()
    );

    info!("Lock: {}", lock.unwrap().auth_sig());

    assert_eq!(
        &ok.clone().status.into_signed_for_testing(),
        gas_lock.expect("gas should be locked").auth_sig()
    );

    authority_state.database_for_testing().reset_locks_for_test(
        &[lock_digest],
        &[
            gas_object.compute_object_reference(),
            object.compute_object_reference(),
        ],
        &authority_state.epoch_store_for_testing(),
    );
    // }
}
