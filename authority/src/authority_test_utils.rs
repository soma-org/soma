use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::{
    base::SomaAddress,
    object::{Object, ObjectID, ObjectRef},
    transaction::{TransactionData, VerifiedTransaction},
    unit_tests::utils::to_sender_signed_transaction,
};

use crate::{state::AuthorityState, test_authority_builder::TestAuthorityBuilder};

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
