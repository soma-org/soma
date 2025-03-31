use std::{collections::HashMap, sync::Arc};

use fastcrypto::ed25519::Ed25519KeyPair;
use futures::future::join_all;
use tracing::info;
use types::{
    base::{dbg_addr, SomaAddress},
    crypto::{get_key_pair, SomaKeyPair},
    effects::{ExecutionStatus, SignedTransactionEffects, TransactionEffectsAPI},
    error::SomaError,
    object::{Object, ObjectID, ObjectRef},
    transaction::TransactionData,
    unit_tests::utils::to_sender_signed_transaction,
};
use utils::logging::init_tracing;

use crate::{
    authority_test_utils::send_and_confirm_transaction, state::AuthorityState,
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
        vec![100, 200, 300],
        sender,
        SomaKeyPair::Ed25519(sender_key),
        // coin_amount - 300 - 200 - 100,
    )
    .await;

    let effects = res.txn_result.unwrap().into_data();
    assert_eq!(*effects.status(), ExecutionStatus::Success);
    info!("Effects: {:?}", effects);
    // make sure each recipient receives the specified amount
    assert_eq!(effects.created().len(), 3);
    let created_obj_id1 = effects.created()[0].0 .0;
    let created_obj_id2 = effects.created()[1].0 .0;
    let created_obj_id3 = effects.created()[2].0 .0;
    let created_obj1 = res
        .authority_state
        .get_object(&created_obj_id1)
        .await
        .unwrap();
    let created_obj2 = res
        .authority_state
        .get_object(&created_obj_id2)
        .await
        .unwrap();
    let created_obj3 = res
        .authority_state
        .get_object(&created_obj_id3)
        .await
        .unwrap();

    let addr1 = effects.created()[0].1.get_owner_address()?;
    let addr2 = effects.created()[1].1.get_owner_address()?;
    let addr3 = effects.created()[2].1.get_owner_address()?;
    let coin_val1 = *recipient_amount_map
        .get(&addr1)
        .ok_or(SomaError::InvalidAddress)?;
    let coin_val2 = *recipient_amount_map
        .get(&addr2)
        .ok_or(SomaError::InvalidAddress)?;
    let coin_val3 = *recipient_amount_map
        .get(&addr3)
        .ok_or(SomaError::InvalidAddress)?;
    info!(
        "Recipient {addr1} received: {}, {}",
        created_obj1.as_coin().unwrap(),
        coin_val1
    );
    assert_eq!(created_obj1.as_coin().unwrap(), coin_val1);
    info!(
        "Recipient {addr2} received: {}, {}",
        created_obj2.as_coin().unwrap(),
        coin_val2
    );
    assert_eq!(created_obj2.as_coin().unwrap(), coin_val2);
    info!(
        "Recipient {addr3} received: {}, {}",
        created_obj3.as_coin().unwrap(),
        coin_val3
    );
    assert_eq!(created_obj3.as_coin().unwrap(), coin_val3);

    // make sure the first object still belongs to the sender,
    // the value is equal to all residual values after amounts transferred and gas payment.
    assert_eq!(effects.mutated()[0].0 .0, object_id);
    assert_eq!(
        effects.mutated()[0].1.get_address_owner_address().unwrap(),
        sender
    );
    // TODO: let gas_used = effects.gas_cost_summary().net_gas_usage() as u64;
    let gas_object = res.authority_state.get_object(&object_id).await.unwrap();
    assert_eq!(
        gas_object.as_coin().unwrap(),
        coin_amount - 100 - 200 - 300, // - gas_used,
    );

    info!("Sender final balance: {}", gas_object.as_coin().unwrap());

    Ok(())
}

struct PaySuiTransactionBlockExecutionResult {
    pub authority_state: Arc<AuthorityState>,
    pub txn_result: Result<SignedTransactionEffects, SomaError>,
}

async fn execute_pay_coin(
    input_coin_objects: Vec<Object>,
    recipients: Vec<SomaAddress>,
    amounts: Vec<u64>,
    sender: SomaAddress,
    sender_key: SomaKeyPair,
    // gas_budget: u64,
) -> PaySuiTransactionBlockExecutionResult {
    let authority_state = TestAuthorityBuilder::new().build().await;

    let input_coin_refs: Vec<ObjectRef> = input_coin_objects
        .iter()
        .map(|coin_obj| coin_obj.compute_object_reference())
        .collect();
    let handles: Vec<_> = input_coin_objects
        .into_iter()
        .map(|obj| authority_state.insert_genesis_object(obj))
        .collect();
    join_all(handles).await;
    // let rgp = authority_state.reference_gas_price_for_testing().unwrap();

    let data = TransactionData::new_pay_coins(input_coin_refs, amounts, recipients, sender);
    let tx = to_sender_signed_transaction(data, &sender_key);
    let txn_result = send_and_confirm_transaction(&authority_state, tx)
        .await
        .map(|(_, effects)| effects);

    PaySuiTransactionBlockExecutionResult {
        authority_state,
        txn_result,
    }
}

// async fn execute_pay_all_coins(
//     input_coin_objects: Vec<&Object>,
//     recipient: SuiAddress,
//     sender: SuiAddress,
//     sender_key: AccountKeyPair,
//     gas_budget: u64,
// ) -> PaySuiTransactionBlockExecutionResult {
//     let dir = tempfile::TempDir::new().unwrap();
//     let network_config = sui_swarm_config::network_config_builder::ConfigBuilder::new(&dir)
//         .with_reference_gas_price(700)
//         .with_objects(
//             input_coin_objects
//                 .clone()
//                 .into_iter()
//                 .map(ToOwned::to_owned),
//         )
//         .build();
//     let genesis = network_config.genesis;
//     let keypair = network_config.validator_configs[0].protocol_key_pair();

//     let authority_state = init_state_with_committee(&genesis, keypair).await;
//     let rgp = authority_state.reference_gas_price_for_testing().unwrap();

//     let mut input_coins = Vec::new();
//     for coin in input_coin_objects {
//         let id = coin.id();
//         let object_ref = genesis
//             .objects()
//             .iter()
//             .find(|o| o.id() == id)
//             .unwrap()
//             .compute_object_reference();
//         input_coins.push(object_ref);
//     }

//     let mut builder = ProgrammableTransactionBuilder::new();
//     builder.pay_all_sui(recipient);
//     let pt = builder.finish();
//     let data = TransactionData::new_programmable(sender, input_coins, pt, gas_budget, rgp);
//     let tx = to_sender_signed_transaction(data, &sender_key);
//     let txn_result = send_and_confirm_transaction(&authority_state, tx)
//         .await
//         .map(|(_, effects)| effects);
//     PaySuiTransactionBlockExecutionResult {
//         authority_state,
//         txn_result,
//     }
// }
