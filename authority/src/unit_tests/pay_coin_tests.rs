// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use fastcrypto::ed25519::Ed25519KeyPair;
use futures::future::join_all;
use tracing::info;
use types::{
    base::{SomaAddress, dbg_addr},
    crypto::{SomaKeyPair, get_key_pair},
    effects::{
        ExecutionFailureStatus, ExecutionStatus, SignedTransactionEffects, TransactionEffectsAPI,
    },
    error::SomaError,
    object::{Object, ObjectID, ObjectRef},
    transaction::TransactionData,
    unit_tests::utils::to_sender_signed_transaction,
};
use utils::logging::init_tracing;

use crate::{
    authority::AuthorityState, authority_test_utils::send_and_confirm_transaction,
    test_authority_builder::TestAuthorityBuilder,
};

#[tokio::test]
async fn test_pay_coin_success_one_input_coin() -> anyhow::Result<()> {
    init_tracing();
    // let _ = tracing_subscriber::fmt::try_init();
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let object_id = ObjectID::random();
    let coin_amount = 50000000;
    let coin_obj = Object::with_id_owner_coin_for_testing(object_id, sender, 50000000);
    let recipient1 = dbg_addr(1);
    let recipient2 = dbg_addr(2);
    let recipient3 = dbg_addr(3);
    let recipient_amount_map: HashMap<_, u64> =
        HashMap::from([(recipient1, 100), (recipient2, 200), (recipient3, 300)]);
    let res = execute_pay_coin(
        vec![coin_obj],
        vec![recipient1, recipient2, recipient3],
        Some(vec![100, 200, 300]),
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);
    info!("Effects: {:?}", effects);
    // make sure each recipient receives the specified amount
    assert_eq!(effects.created().len(), 3);
    let created_obj_id1 = effects.created()[0].0.0;
    let created_obj_id2 = effects.created()[1].0.0;
    let created_obj_id3 = effects.created()[2].0.0;
    let created_obj1 = res.authority_state.get_object(&created_obj_id1).await.unwrap();
    let created_obj2 = res.authority_state.get_object(&created_obj_id2).await.unwrap();
    let created_obj3 = res.authority_state.get_object(&created_obj_id3).await.unwrap();

    let addr1 = effects.created()[0].1.get_owner_address()?;
    let addr2 = effects.created()[1].1.get_owner_address()?;
    let addr3 = effects.created()[2].1.get_owner_address()?;
    let coin_val1 = *recipient_amount_map.get(&addr1).ok_or(SomaError::InvalidAddress)?;
    let coin_val2 = *recipient_amount_map.get(&addr2).ok_or(SomaError::InvalidAddress)?;
    let coin_val3 = *recipient_amount_map.get(&addr3).ok_or(SomaError::InvalidAddress)?;
    assert_eq!(created_obj1.as_coin().unwrap(), coin_val1);
    assert_eq!(created_obj2.as_coin().unwrap(), coin_val2);
    assert_eq!(created_obj3.as_coin().unwrap(), coin_val3);

    // make sure the first object still belongs to the sender,
    // the value is equal to all residual values after amounts transferred and gas payment.
    assert_eq!(effects.mutated()[0].0.0, object_id);
    assert_eq!(effects.mutated()[0].1.get_address_owner_address().unwrap(), sender);
    let gas_used = effects.transaction_fee().total_fee as u64;
    info!("{:?}", effects.transaction_fee());
    let gas_object = res.authority_state.get_object(&object_id).await.unwrap();
    assert_eq!(gas_object.as_coin().unwrap(), coin_amount - 100 - 200 - 300 - gas_used,);

    info!("Sender final balance: {}", gas_object.as_coin().unwrap());

    Ok(())
}

#[tokio::test]
async fn test_pay_coin_success_multiple_input_coins() -> anyhow::Result<()> {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let object_id1 = ObjectID::random();
    let object_id2 = ObjectID::random();
    let object_id3 = ObjectID::random();
    let coin_obj1 = Object::with_id_owner_coin_for_testing(object_id1, sender, 5000000);
    let coin_obj2 = Object::with_id_owner_coin_for_testing(object_id2, sender, 1000);
    let coin_obj3 = Object::with_id_owner_coin_for_testing(object_id3, sender, 1000);
    let recipient1 = dbg_addr(1);
    let recipient2 = dbg_addr(2);

    let res = execute_pay_coin(
        vec![coin_obj1, coin_obj2, coin_obj3],
        vec![recipient1, recipient2],
        Some(vec![500, 1500]),
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;
    let recipient_amount_map: HashMap<_, u64> =
        HashMap::from([(recipient1, 500), (recipient2, 1500)]);
    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // make sure each recipient receives the specified amount
    assert_eq!(effects.created().len(), 2);
    let created_obj_id1 = effects.created()[0].0.0;
    let created_obj_id2 = effects.created()[1].0.0;
    let created_obj1 = res.authority_state.get_object(&created_obj_id1).await.unwrap();
    let created_obj2 = res.authority_state.get_object(&created_obj_id2).await.unwrap();
    let addr1 = effects.created()[0].1.get_owner_address()?;
    let addr2 = effects.created()[1].1.get_owner_address()?;
    let coin_val1 = *recipient_amount_map.get(&addr1).ok_or(SomaError::InvalidAddress)?;
    let coin_val2 = *recipient_amount_map.get(&addr2).ok_or(SomaError::InvalidAddress)?;
    assert_eq!(created_obj1.as_coin().unwrap(), coin_val1);
    assert_eq!(created_obj2.as_coin().unwrap(), coin_val2);
    // make sure the first input coin still belongs to the sender,
    // the value is equal to all residual values after amounts transferred and gas payment.
    assert_eq!(effects.mutated()[0].0.0, object_id1);
    assert_eq!(effects.mutated()[0].1.get_address_owner_address().unwrap(), sender);
    let gas_used = effects.transaction_fee().total_fee as u64;
    let gas_object = res.authority_state.get_object(&object_id1).await.unwrap();
    assert_eq!(gas_object.as_coin().unwrap(), 5002000 - 500 - 1500 - gas_used,);

    // make sure the second and third input coins are deleted
    let deleted_ids: Vec<ObjectID> = effects.deleted().iter().map(|d| d.0).collect();
    assert!(deleted_ids.contains(&object_id2));
    assert!(deleted_ids.contains(&object_id3));
    Ok(())
}

#[tokio::test]
async fn test_pay_all_coins_success_one_input_coin() -> anyhow::Result<()> {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let object_id = ObjectID::random();
    let coin_obj = Object::with_id_owner_coin_for_testing(object_id, sender, 3000000);
    let recipient = dbg_addr(2);
    let res = execute_pay_coin(
        vec![coin_obj],
        vec![recipient],
        None,
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let obj_ref = &effects.created()[0].0;
    let deleted = &effects.deleted()[0];
    // input gas should be deleted
    assert_eq!(deleted.0, object_id);
    assert_eq!(effects.created()[0].1.get_address_owner_address().unwrap(), recipient);

    let gas_used = effects.transaction_fee().total_fee;
    let obj = res.authority_state.get_object(&obj_ref.0).await.unwrap();
    assert_eq!(obj.as_coin().unwrap(), 3000000 - gas_used);
    Ok(())
}

#[tokio::test]
async fn test_pay_all_coins_success_multiple_input_coins() -> anyhow::Result<()> {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let object_id1 = ObjectID::random();
    let object_id2 = ObjectID::random();
    let object_id3 = ObjectID::random();
    let coin_obj1 = Object::with_id_owner_coin_for_testing(object_id1, sender, 3000000);
    let coin_obj2 = Object::with_id_owner_coin_for_testing(object_id2, sender, 1000);
    let coin_obj3 = Object::with_id_owner_coin_for_testing(object_id3, sender, 1000);
    let recipient = dbg_addr(2);
    let res = execute_pay_coin(
        vec![coin_obj1, coin_obj2, coin_obj3],
        vec![recipient],
        None,
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    let obj_ref = &effects.created()[0].0;
    let deleted = &effects.deleted();
    // inputs should be deleted
    for input in [object_id1, object_id2, object_id3] {
        assert!(deleted.iter().map(|(id, _, _)| *id).collect::<Vec<ObjectID>>().contains(&input));
    }
    assert_eq!(effects.created()[0].1.get_address_owner_address().unwrap(), recipient);

    let gas_used = effects.transaction_fee().total_fee;
    let obj = res.authority_state.get_object(&obj_ref.0).await.unwrap();
    assert_eq!(obj.as_coin().unwrap(), 3002000 - gas_used);
    Ok(())
}

#[tokio::test]
async fn test_pay_coin_failure_insufficient_total_balance_multiple_input_coins() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin1 = Object::with_id_owner_coin_for_testing(
        ObjectID::random(),
        sender,
        4000 + 1000 + 4 * 300 + 7, // base fee + operation fee + value fee
    );
    let coin2 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 3000);
    let recipient1 = dbg_addr(1);
    let recipient2 = dbg_addr(2);

    let res = execute_pay_coin(
        vec![coin1, coin2],
        vec![recipient1, recipient2],
        Some(vec![4000, 4000]),
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;
    assert_eq!(
        res.txn_result.as_ref().unwrap().status(),
        &ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientCoinBalance },
    );
}

#[tokio::test]
async fn test_pay_coin_failure_insufficient_total_balance_one_input_coin() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin1 =
        Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 100 + 1000 + 3 * 300); // base fee + operation fee + operation fee + 0 value fee
    let recipient1 = dbg_addr(1);
    let recipient2 = dbg_addr(2);

    let res = execute_pay_coin(
        vec![coin1],
        vec![recipient1, recipient2],
        Some(vec![100, 100]),
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    assert_eq!(
        res.txn_result.as_ref().unwrap().status(),
        &ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientCoinBalance },
    );
}

#[tokio::test]
async fn test_pay_coin_failure_insufficient_gas_one_input_coin() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin1 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 900);
    let recipient1 = dbg_addr(1);
    let recipient2 = dbg_addr(2);

    let res = execute_pay_coin(
        vec![coin1],
        vec![recipient1, recipient2],
        Some(vec![100, 100]),
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    assert_eq!(
        res.txn_result.as_ref().unwrap().status(),
        &ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );
}

#[tokio::test]
async fn test_pay_coin_failure_insufficient_gas_multiple_input_coins() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin1 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 800);
    let coin2 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 700);
    let recipient1 = dbg_addr(1);
    let recipient2 = dbg_addr(2);

    let res = execute_pay_coin(
        vec![coin1, coin2],
        vec![recipient1, recipient2],
        Some(vec![100, 100]),
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    assert_eq!(
        res.txn_result.as_ref().unwrap().status(),
        &ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );
}

#[tokio::test]
async fn test_pay_all_coins_failure_insufficient_gas_one_input_coin() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin1 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 800);
    let recipient = dbg_addr(2);

    let res = execute_pay_coin(
        vec![coin1],
        vec![recipient],
        None,
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    assert_eq!(
        res.txn_result.as_ref().unwrap().status(),
        &ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );
}

#[tokio::test]
async fn test_pay_all_coins_failure_insufficient_gas_multiple_input_coins() {
    let (sender, sender_key): (_, Ed25519KeyPair) = get_key_pair();
    let coin1 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 750);
    let coin2 = Object::with_id_owner_coin_for_testing(ObjectID::random(), sender, 750);
    let recipient = dbg_addr(2);
    let res = execute_pay_coin(
        vec![coin1, coin2],
        vec![recipient],
        None,
        sender,
        SomaKeyPair::Ed25519(sender_key),
    )
    .await;

    assert_eq!(
        res.txn_result.as_ref().unwrap().status(),
        &ExecutionStatus::Failure { error: ExecutionFailureStatus::InsufficientGas },
    );
}

struct PayCoinTransactionBlockExecutionResult {
    pub authority_state: Arc<AuthorityState>,
    pub txn_result: Result<SignedTransactionEffects, SomaError>,
}

async fn execute_pay_coin(
    input_coin_objects: Vec<Object>,
    recipients: Vec<SomaAddress>,
    amounts: Option<Vec<u64>>,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
    // gas_budget: u64,
) -> PayCoinTransactionBlockExecutionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;

    let input_coin_refs: Vec<ObjectRef> =
        input_coin_objects.iter().map(|coin_obj| coin_obj.compute_object_reference()).collect();
    let handles: Vec<_> = input_coin_objects
        .into_iter()
        .map(|obj| authority_state.insert_genesis_object(obj))
        .collect();
    join_all(handles).await;
    // let rgp = authority_state.reference_gas_price_for_testing().unwrap();

    let data = TransactionData::new_pay_coins(input_coin_refs, amounts, recipients, sender);
    let tx = to_sender_signed_transaction(data, &sender_key);
    let txn_result =
        send_and_confirm_transaction(&authority_state, tx).await.map(|(_, effects)| effects);

    PayCoinTransactionBlockExecutionResult { authority_state, txn_result }
}
