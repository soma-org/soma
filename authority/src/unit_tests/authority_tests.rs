use fastcrypto::ed25519::Ed25519KeyPair;
use futures::{stream::FuturesUnordered, StreamExt};
use tracing::info;
use types::{base::dbg_addr, crypto::get_key_pair, error::SomaError, object::ObjectID};
use utils::logging::init_tracing;

use crate::authority_test_utils::{init_state_with_ids, init_transfer_transaction};

#[cfg(msim)]
#[msim::sim_test]
async fn test_conflicting_transactions() {
    // let _ = tracing_subscriber::fmt::try_init();
    init_tracing();
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let recipient1 = dbg_addr(2);
    let recipient2 = dbg_addr(3);
    let object_id = ObjectID::random();
    // let gas_object_id = ObjectID::random();
    let authority_state = init_state_with_ids(vec![
        (sender, object_id), /* , (sender, gas_object_id)*/
    ])
    .await;

    // // let rgp = authority_state.reference_gas_price_for_testing().unwrap();
    let epoch_store = authority_state.load_epoch_store_one_call_per_task();
    let object = authority_state.get_object(&object_id).await.unwrap();
    // // let gas_object = authority_state.get_object(&gas_object_id).await.unwrap();

    let tx1 = init_transfer_transaction(
        &authority_state,
        sender,
        &sender_key,
        recipient1,
        object.compute_object_reference(),
        // gas_object.compute_object_reference(),
        // rgp * TEST_ONLY_GAS_UNIT_FOR_TRANSFER,
        // rgp,
    );

    let tx2 = init_transfer_transaction(
        &authority_state,
        sender,
        &sender_key,
        recipient2,
        object.compute_object_reference(),
        // gas_object.compute_object_reference(),
        // rgp * TEST_ONLY_GAS_UNIT_FOR_TRANSFER,
        // rgp,
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
    // let gas_lock = authority_state
    //     .get_latest_object_lock_for_testing(gas_object.id())
    //     .await
    //     .unwrap();

    let lock_digest = (lock.clone()).unwrap().digest().clone();

    assert_eq!(
        &ok.clone().status.into_signed_for_testing(),
        lock.clone().expect("object should be locked").auth_sig()
    );

    info!("Lock: {}", lock.unwrap().auth_sig());

    // assert_eq!(
    //     &ok.clone().status.into_signed_for_testing(),
    //     gas_lock
    //         .expect("gas should be locked")
    //         .auth_sig()
    // );

    authority_state.database_for_testing().reset_locks_for_test(
        &[lock_digest],
        &[
            // gas_object.compute_object_reference(),
            object.compute_object_reference(),
        ],
        &authority_state.epoch_store_for_testing(),
    );
    // }
}
