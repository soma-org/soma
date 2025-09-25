use prost_types::FieldMask;
use tap::Pipe;
use tracing::info;
use types::{balance_change::derive_balance_changes, transaction_executor::TransactionExecutor};

use crate::{
    api::{RpcService, error::RpcError},
    proto::{
        google::rpc::bad_request::FieldViolation,
        soma::{
            ErrorReason, ExecuteTransactionRequest, ExecuteTransactionResponse,
            ExecutedTransaction, Object, Transaction, TransactionEffects, TransactionFinality,
            UserSignature, ValidatorAggregatedSignature,
            transaction_execution_service_server::TransactionExecutionService,
            transaction_finality::Finality,
        },
    },
    types::Address,
    utils::{
        field::{FieldMaskTree, FieldMaskUtil},
        merge::Merge,
    },
};

#[tonic::async_trait]
impl TransactionExecutionService for RpcService {
    async fn execute_transaction(
        &self,
        request: tonic::Request<ExecuteTransactionRequest>,
    ) -> Result<tonic::Response<ExecuteTransactionResponse>, tonic::Status> {
        let executor = self
            .executor
            .as_ref()
            .ok_or_else(|| tonic::Status::unimplemented("no transaction executor"))?;

        execute_transaction(self, executor, request.into_inner())
            .await
            .map(tonic::Response::new)
            .map_err(Into::into)
    }
}

pub const EXECUTE_TRANSACTION_READ_MASK_DEFAULT: &str = "finality";

#[tracing::instrument(skip(service, executor))]
pub async fn execute_transaction(
    service: &RpcService,
    executor: &std::sync::Arc<dyn TransactionExecutor>,
    request: ExecuteTransactionRequest,
) -> Result<ExecuteTransactionResponse, RpcError> {
    let transaction = request
        .transaction
        .as_ref()
        .ok_or_else(|| FieldViolation::new("transaction").with_reason(ErrorReason::FieldMissing))?
        .pipe(crate::types::Transaction::try_from)
        .map_err(|e| {
            FieldViolation::new("transaction")
                .with_description(format!("invalid transaction: {e}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;

    let signatures = request
        .signatures
        .iter()
        .enumerate()
        .map(|(i, signature)| {
            crate::types::UserSignature::try_from(signature).map_err(|e| {
                FieldViolation::new_at("signatures", i)
                    .with_description(format!("invalid signature: {e}"))
                    .with_reason(ErrorReason::FieldInvalid)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let signed_transaction = crate::types::SignedTransaction {
        transaction: transaction.clone(),
        signatures: signatures.clone(),
    };
    let read_mask = {
        let read_mask = request
            .read_mask
            .unwrap_or_else(|| FieldMask::from_str(EXECUTE_TRANSACTION_READ_MASK_DEFAULT));

        read_mask
            .validate::<ExecuteTransactionResponse>()
            .map_err(|path| {
                FieldViolation::new("read_mask")
                    .with_description(format!("invalid read_mask path: {path}"))
                    .with_reason(ErrorReason::FieldInvalid)
            })?;

        FieldMaskTree::from(read_mask)
    };

    let request = {
        let mask = read_mask
            .subtree(ExecuteTransactionResponse::TRANSACTION_FIELD.name)
            .unwrap_or_default();

        let executor_transaction = signed_transaction.try_into().map_err(|e| e)?;

        types::quorum_driver::ExecuteTransactionRequest {
            transaction: executor_transaction,
            include_input_objects: mask.contains(ExecutedTransaction::BALANCE_CHANGES_FIELD.name)
                || mask.contains(ExecutedTransaction::INPUT_OBJECTS_FIELD.name)
                || mask.contains(ExecutedTransaction::EFFECTS_FIELD.name),
            include_output_objects: mask.contains(ExecutedTransaction::BALANCE_CHANGES_FIELD.name)
                || mask.contains(ExecutedTransaction::OUTPUT_OBJECTS_FIELD.name)
                || mask.contains(ExecutedTransaction::EFFECTS_FIELD.name),
        }
    };

    let types::quorum_driver::ExecuteTransactionResponse {
        effects:
            types::quorum_driver::FinalizedEffects {
                effects,
                finality_info,
            },
        shard,
        input_objects,
        output_objects,
    } = executor.execute_transaction(request, None).await?;

    let finality = {
        let finality = match finality_info.clone() {
            types::quorum_driver::EffectsFinalityInfo::Certified(sig) => {
                Finality::Certified(ValidatorAggregatedSignature::from(sig).into())
            }
        };
        let mut message = TransactionFinality::default();
        message.finality = Some(finality);
        message
    };

    let executed_transaction = if let Some(mask) =
        read_mask.subtree(ExecuteTransactionResponse::TRANSACTION_FIELD.name)
    {
        let input_objects = input_objects.unwrap_or_default();
        let output_objects = output_objects.unwrap_or_default();

        let balance_changes = mask
            .contains(ExecutedTransaction::BALANCE_CHANGES_FIELD.name)
            .then(|| {
                derive_balance_changes(&effects, &input_objects, &output_objects)
                    .into_iter()
                    .map(Into::into)
                    .collect()
            })
            .unwrap_or_default();
        let input_objects = input_objects
            .into_iter()
            .map(crate::types::Object::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let output_objects = output_objects
            .into_iter()
            .map(crate::types::Object::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        // Convert effects with debugging
        info!("Converting effects to crate::types::TransactionEffects");
        let effects = match crate::types::TransactionEffects::try_from(effects) {
            Ok(e) => {
                info!("Successfully converted effects");
                e
            }
            Err(e) => {
                info!("ERROR converting effects: {:?}", e);
                return Err(e.into());
            }
        };

        let effects = mask
            .subtree(ExecutedTransaction::EFFECTS_FIELD.name)
            .map(|mask| {
                let mut effects = TransactionEffects::merge_from(&effects, &mask);

                if mask.contains(TransactionEffects::CHANGED_OBJECTS_FIELD.name) {
                    for changed_object in effects.changed_objects.iter_mut() {
                        let Ok(object_id) = changed_object.object_id().parse::<Address>() else {
                            continue;
                        };

                        // Rest of the logic...
                        if let Some(object) = input_objects
                            .iter()
                            .chain(&output_objects)
                            .find(|o| o.object_id() == object_id)
                        {
                            changed_object.object_type = Some(object.object_type.clone().into());
                        }
                    }
                }

                if mask.contains(TransactionEffects::UNCHANGED_SHARED_OBJECTS_FIELD.name) {
                    for unchanged_shared_object in effects.unchanged_shared_objects.iter_mut() {
                        let Ok(object_id) = unchanged_shared_object.object_id().parse::<Address>()
                        else {
                            continue;
                        };

                        if let Some(object) =
                            input_objects.iter().find(|o| o.object_id() == object_id)
                        {
                            unchanged_shared_object.object_type =
                                Some(object.object_type.clone().into());
                        }
                    }
                }

                effects
            });

        let proto_shard = mask
            .contains(ExecutedTransaction::SHARD_FIELD.name)
            .then(|| {
                shard.as_ref().and_then(|domain_shard| {
                    // Domain → SDK → Proto
                    let sdk_shard: crate::types::Shard = domain_shard.clone().try_into().ok()?; // Handle error gracefully
                    let proto_shard: crate::proto::soma::Shard = sdk_shard.into();
                    Some(proto_shard)
                })
            })
            .flatten();

        let mut message = ExecutedTransaction::default();
        message.digest = mask
            .contains(ExecutedTransaction::DIGEST_FIELD.name)
            .then(|| transaction.digest().to_string());
        message.transaction = mask
            .subtree(ExecutedTransaction::TRANSACTION_FIELD.name)
            .map(|mask| Transaction::merge_from(transaction, &mask));
        message.signatures = mask
            .subtree(ExecutedTransaction::SIGNATURES_FIELD.name)
            .map(|mask| {
                signatures
                    .into_iter()
                    .map(|s| UserSignature::merge_from(s, &mask))
                    .collect()
            })
            .unwrap_or_default();
        message.effects = effects;
        message.balance_changes = balance_changes;
        message.input_objects = mask
            .subtree(ExecutedTransaction::INPUT_OBJECTS_FIELD.name)
            .map(|mask| {
                input_objects
                    .into_iter()
                    .map(|o| Object::merge_from(o, &mask))
                    .collect()
            })
            .unwrap_or_default();
        message.output_objects = mask
            .subtree(ExecutedTransaction::OUTPUT_OBJECTS_FIELD.name)
            .map(|mask| {
                output_objects
                    .into_iter()
                    .map(|o| Object::merge_from(o, &mask))
                    .collect()
            })
            .unwrap_or_default();
        message.shard = proto_shard;
        Some(message)
    } else {
        None
    };

    let mut message = ExecuteTransactionResponse::default();
    message.finality = read_mask
        .contains(ExecuteTransactionResponse::FINALITY_FIELD.name)
        .then_some(finality);
    message.transaction = executed_transaction;

    Ok(message)
}
