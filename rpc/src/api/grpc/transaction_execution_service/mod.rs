use crate::api::RpcService;
use crate::api::error::RpcError;
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::ErrorReason;
use crate::proto::soma::ExecuteTransactionRequest;
use crate::proto::soma::ExecuteTransactionResponse;
use crate::proto::soma::ExecutedTransaction;
use crate::proto::soma::InitiateShardWorkRequest;
use crate::proto::soma::InitiateShardWorkResponse;
use crate::proto::soma::Object;
use crate::proto::soma::ObjectSet;
use crate::proto::soma::SimulateTransactionRequest;
use crate::proto::soma::SimulateTransactionResponse;
use crate::proto::soma::Transaction;
use crate::proto::soma::TransactionEffects;
use crate::proto::soma::UserSignature;
use crate::proto::soma::transaction_execution_service_server::TransactionExecutionService;
use crate::types::Address;
use crate::utils::field::FieldMaskTree;
use crate::utils::field::FieldMaskUtil;
use crate::utils::merge::Merge;
use prost_types::FieldMask;
use tap::Pipe;
use types::balance_change::derive_balance_changes;
use types::transaction_executor::TransactionExecutor;

mod initiate_shard_work;
mod simulate;

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

    async fn simulate_transaction(
        &self,
        request: tonic::Request<SimulateTransactionRequest>,
    ) -> Result<tonic::Response<SimulateTransactionResponse>, tonic::Status> {
        simulate::simulate_transaction(self, request.into_inner())
            .map(tonic::Response::new)
            .map_err(Into::into)
    }

    async fn initiate_shard_work(
        &self,
        request: tonic::Request<InitiateShardWorkRequest>,
    ) -> Result<tonic::Response<InitiateShardWorkResponse>, tonic::Status> {
        initiate_shard_work::initiate_shard_work(self, request.into_inner())
            .await
            .map(tonic::Response::new)
            .map_err(Into::into)
    }
}

pub const EXECUTE_TRANSACTION_READ_MASK_DEFAULT: &str = "effects";

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
            .validate::<ExecutedTransaction>()
            .map_err(|path| {
                FieldViolation::new("read_mask")
                    .with_description(format!("invalid read_mask path: {path}"))
                    .with_reason(ErrorReason::FieldInvalid)
            })?;
        FieldMaskTree::from(read_mask)
    };

    let request = types::quorum_driver::ExecuteTransactionRequest {
        transaction: signed_transaction.try_into()?,
        include_input_objects: read_mask.contains(ExecutedTransaction::BALANCE_CHANGES_FIELD.name)
            || read_mask.contains(ExecutedTransaction::OBJECTS_FIELD.name)
            || read_mask.contains(ExecutedTransaction::EFFECTS_FIELD.name),
        include_output_objects: read_mask.contains(ExecutedTransaction::BALANCE_CHANGES_FIELD.name)
            || read_mask.contains(ExecutedTransaction::OBJECTS_FIELD.name)
            || read_mask.contains(ExecutedTransaction::EFFECTS_FIELD.name),
    };

    let types::quorum_driver::ExecuteTransactionResponse {
        effects:
            types::quorum_driver::FinalizedEffects {
                effects,
                finality_info: _,
            },
        input_objects,
        output_objects,
    } = executor.execute_transaction(request, None).await?;

    let executed_transaction = {
        let input_objects = input_objects.unwrap_or_default();
        let output_objects = output_objects.unwrap_or_default();

        let balance_changes = if read_mask.contains(ExecutedTransaction::BALANCE_CHANGES_FIELD.name)
        {
            derive_balance_changes(&effects, &input_objects, &output_objects)
                .into_iter()
                .map(Into::into)
                .collect()
        } else {
            vec![]
        };

        let input_objects = input_objects
            .into_iter()
            .map(crate::types::Object::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let output_objects = output_objects
            .into_iter()
            .map(crate::types::Object::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        let effects = crate::types::TransactionEffects::try_from(effects)?;
        let effects = read_mask
            .subtree(ExecutedTransaction::EFFECTS_FIELD.name)
            .map(|mask| {
                let mut effects = TransactionEffects::merge_from(&effects, &mask);

                if mask.contains(TransactionEffects::CHANGED_OBJECTS_FIELD.name) {
                    for changed_object in effects.changed_objects.iter_mut() {
                        let Ok(object_id) = changed_object.object_id().parse::<Address>() else {
                            continue;
                        };

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
                    for unchanged_consensus_object in effects.unchanged_shared_objects.iter_mut() {
                        let Ok(object_id) =
                            unchanged_consensus_object.object_id().parse::<Address>()
                        else {
                            continue;
                        };

                        if let Some(object) =
                            input_objects.iter().find(|o| o.object_id() == object_id)
                        {
                            unchanged_consensus_object.object_type =
                                Some(object.object_type.clone().into());
                        }
                    }
                }

                effects
            });

        let mut message = ExecutedTransaction::default();
        message.digest = read_mask
            .contains(ExecutedTransaction::DIGEST_FIELD.name)
            .then(|| transaction.digest().to_string());
        message.transaction = read_mask
            .subtree(ExecutedTransaction::TRANSACTION_FIELD.name)
            .map(|mask| Transaction::merge_from(transaction, &mask));
        message.signatures = read_mask
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
        message.objects = read_mask
            .subtree(
                ExecutedTransaction::path_builder()
                    .objects()
                    .objects()
                    .finish(),
            )
            .map(|mask| {
                let set: std::collections::BTreeMap<_, _> = input_objects
                    .into_iter()
                    .chain(output_objects.into_iter())
                    .map(|object| ((object.object_id(), object.version()), object))
                    .collect();
                ObjectSet::default().with_objects(
                    set.into_values()
                        .map(|o| Object::merge_from(o, &mask))
                        .collect(),
                )
            });
        message
    };

    Ok(ExecuteTransactionResponse::default().with_transaction(executed_transaction))
}
