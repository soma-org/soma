use crate::api::RpcService;
use crate::api::error::Result;
use crate::api::error::RpcError;
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::ErrorReason;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;

use crate::proto::soma::ExecutedTransaction;
use crate::proto::soma::Object;
use crate::proto::soma::SimulateTransactionRequest;
use crate::proto::soma::SimulateTransactionResponse;
use crate::proto::soma::Transaction;
use crate::proto::soma::TransactionEffects;
use types::balance_change::derive_balance_changes;
use types::object::ObjectID;
use types::transaction_executor::SimulateTransactionResult;
use types::transaction_executor::TransactionChecks;

pub fn simulate_transaction(
    service: &RpcService,
    request: SimulateTransactionRequest,
) -> Result<SimulateTransactionResponse> {
    let executor = service
        .executor
        .as_ref()
        .ok_or_else(|| RpcError::new(tonic::Code::Unimplemented, "no transaction executor"))?;

    let read_mask = request
        .read_mask
        .as_ref()
        .map(FieldMaskTree::from_field_mask)
        .unwrap_or_else(FieldMaskTree::new_wildcard);

    let transaction_proto = request
        .transaction
        .as_ref()
        .ok_or_else(|| FieldViolation::new("transaction").with_reason(ErrorReason::FieldMissing))?;

    let checks = TransactionChecks::from(request.checks());

    // Try to parse out a fully-formed transaction. If one wasn't provided then we will attempt to
    // perform transaction resolution.
    let mut transaction = match crate::types::Transaction::try_from(transaction_proto) {
        Ok(transaction) => types::transaction::TransactionData::try_from(transaction)?,
        Err(e) => {
            return Err(FieldViolation::new("transaction")
                .with_description(format!("invalid transaction: {e}"))
                .with_reason(ErrorReason::FieldInvalid)
                .into());
        }
    };

    let SimulateTransactionResult {
        input_objects,
        output_objects,
        effects,
    } = executor
        .simulate_transaction(transaction.clone(), checks)
        .map_err(anyhow::Error::from)?;

    let transaction = if let Some(submask) = read_mask.subtree("transaction") {
        let mut message = ExecutedTransaction::default();
        let transaction = crate::types::Transaction::try_from(transaction)?;

        let input_objects = input_objects.into_values().collect::<Vec<_>>();
        let output_objects = output_objects.into_values().collect::<Vec<_>>();

        message.balance_changes = read_mask
            .contains(ExecutedTransaction::BALANCE_CHANGES_FIELD.name)
            .then(|| {
                derive_balance_changes(&effects, &input_objects, &output_objects)
                    .into_iter()
                    .map(Into::into)
                    .collect()
            })
            .unwrap_or_default();

        message.effects = {
            let effects = crate::types::TransactionEffects::try_from(effects)?;
            submask
                .subtree(ExecutedTransaction::EFFECTS_FIELD)
                .map(|mask| {
                    let mut effects = TransactionEffects::merge_from(&effects, &mask);

                    if mask.contains(TransactionEffects::CHANGED_OBJECTS_FIELD.name) {
                        for changed_object in effects.changed_objects.iter_mut() {
                            let Ok(object_id) = changed_object.object_id().parse::<ObjectID>()
                            else {
                                continue;
                            };

                            if let Some(object) = input_objects
                                .iter()
                                .chain(&output_objects)
                                .find(|o| o.id() == object_id)
                            {
                                changed_object.object_type = Some((*object.type_()).to_string());
                            }
                        }
                    }

                    if mask.contains(TransactionEffects::UNCHANGED_SHARED_OBJECTS_FIELD.name) {
                        for unchanged_consensus_object in
                            effects.unchanged_shared_objects.iter_mut()
                        {
                            let Ok(object_id) =
                                unchanged_consensus_object.object_id().parse::<ObjectID>()
                            else {
                                continue;
                            };

                            if let Some(object) = input_objects.iter().find(|o| o.id() == object_id)
                            {
                                unchanged_consensus_object.object_type =
                                    Some((*object.type_()).to_string());
                            }
                        }
                    }

                    effects
                })
        };

        message.transaction = submask
            .subtree(ExecutedTransaction::TRANSACTION_FIELD.name)
            .map(|mask| Transaction::merge_from(transaction, &mask));

        let input_objects = input_objects
            .into_iter()
            .map(crate::types::Object::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let output_objects = output_objects
            .into_iter()
            .map(crate::types::Object::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        message.input_objects = submask
            .subtree(ExecutedTransaction::INPUT_OBJECTS_FIELD)
            .map(|mask| {
                input_objects
                    .into_iter()
                    .map(|o| Object::merge_from(o, &mask))
                    .collect()
            })
            .unwrap_or_default();

        message.output_objects = submask
            .subtree(ExecutedTransaction::OUTPUT_OBJECTS_FIELD)
            .map(|mask| {
                output_objects
                    .into_iter()
                    .map(|o| Object::merge_from(o, &mask))
                    .collect()
            })
            .unwrap_or_default();

        Some(message)
    } else {
        None
    };

    let mut response = SimulateTransactionResponse::default();
    response.transaction = transaction;

    Ok(response)
}
