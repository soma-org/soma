use crate::api::RpcService;
use crate::api::error::Result;
use crate::api::error::RpcError;
use crate::api::reader::StateReader;
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::ErrorReason;
use crate::proto::soma::ExecutedTransaction;
use crate::proto::soma::Object;
use crate::proto::soma::ObjectSet;
use crate::proto::soma::SimulateTransactionRequest;
use crate::proto::soma::SimulateTransactionResponse;
use crate::proto::soma::Transaction;
use crate::proto::soma::TransactionEffects;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;
use itertools::Itertools;
use protocol_config::ProtocolConfig;
use types::balance_change::derive_balance_changes_2;
use types::base::SomaAddress;
use types::effects::TransactionEffectsAPI;
use types::object::ObjectID;
use types::object::ObjectRef;
use types::object::ObjectType;
use types::transaction_executor::SimulateTransactionResult;
use types::transaction_executor::TransactionChecks;

const GAS_COIN_SIZE_BYTES: u64 = 40;

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

    let protocol_config = {
        let system_state = service.reader.get_system_state()?;
        let protocol_config = ProtocolConfig::get_for_version_if_supported(
            system_state.protocol_version.into(),
            service.reader.inner().get_chain_identifier()?.chain(),
        )
        .ok_or_else(|| {
            RpcError::new(tonic::Code::Internal, "unable to get current protocol config")
        })?;

        protocol_config
    };

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

    if checks.enabled() {
        if transaction.gas().is_empty() {
            let input_objects = transaction
                .input_objects()
                .map_err(anyhow::Error::from)?
                .iter()
                .flat_map(|obj| match obj {
                    types::transaction::InputObjectKind::ImmOrOwnedObject((id, _, _)) => Some(*id),
                    _ => None,
                })
                .collect_vec();
            let gas_coins = select_gas(
                &service.reader,
                transaction.sender(),
                100, // TODO: protocol_config.max_gas_payment_objects(),
                &input_objects,
            )?;
            *transaction.gas_mut() = gas_coins;
        }
    }

    let SimulateTransactionResult { effects, objects, execution_result, mock_gas_id: _ } =
        executor.simulate_transaction(transaction.clone(), checks).map_err(anyhow::Error::from)?;

    let transaction = if let Some(submask) = read_mask.subtree("transaction") {
        let mut message = ExecutedTransaction::default();
        let transaction = crate::types::Transaction::try_from(transaction)?;

        message.balance_changes =
            if submask.contains(ExecutedTransaction::BALANCE_CHANGES_FIELD.name) {
                derive_balance_changes_2(&effects, &objects).into_iter().map(Into::into).collect()
            } else {
                vec![]
            };

        message.effects = {
            let effects = crate::types::TransactionEffects::try_from(effects)?;
            submask.subtree(ExecutedTransaction::EFFECTS_FIELD).map(|mask| {
                let mut effects = TransactionEffects::merge_from(&effects, &mask);

                if mask.contains(TransactionEffects::CHANGED_OBJECTS_FIELD.name) {
                    for changed_object in effects.changed_objects.iter_mut() {
                        let Ok(object_id) = changed_object.object_id().parse::<ObjectID>() else {
                            continue;
                        };

                        if let Some(object) = objects.iter().find(|o| o.id() == object_id) {
                            changed_object.object_type = Some((*object.type_()).to_string());
                        }
                    }
                }

                if mask.contains(TransactionEffects::UNCHANGED_SHARED_OBJECTS_FIELD.name) {
                    for unchanged_consensus_object in effects.unchanged_shared_objects.iter_mut() {
                        let Ok(object_id) =
                            unchanged_consensus_object.object_id().parse::<ObjectID>()
                        else {
                            continue;
                        };

                        if let Some(object) = objects.iter().find(|o| o.id() == object_id) {
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

        message.objects = submask
            .subtree(ExecutedTransaction::path_builder().objects().objects().finish())
            .map(|mask| {
                ObjectSet::default()
                    .with_objects(objects.iter().map(|o| Object::merge_from(o, &mask)).collect())
            });

        Some(message)
    } else {
        None
    };

    let mut response = SimulateTransactionResponse::default();
    response.transaction = transaction;
    Ok(response)
}

fn select_gas(
    reader: &StateReader,
    owner: SomaAddress,
    max_gas_payment_objects: u32,
    input_objects: &[ObjectID],
) -> Result<Vec<ObjectRef>> {
    let gas_coins = reader
        .inner()
        .indexes()
        .ok_or_else(RpcError::not_found)?
        .owned_objects_iter(owner, Some(ObjectType::Coin), None)?
        .filter_ok(|info| !input_objects.contains(&info.object_id))
        .filter_map_ok(|info| reader.inner().get_object(&info.object_id))
        // filter for objects which are not ConsensusAddress owned,
        // since only Address owned can be used for gas payments today
        .filter_ok(|object| !object.is_shared())
        .filter_map_ok(|object| {
            object.as_coin().map(|coin| (object.compute_object_reference(), coin))
        })
        .take(max_gas_payment_objects as usize);

    let mut selected_gas = vec![];
    let mut selected_gas_value = 0;

    for maybe_coin in gas_coins {
        let (object_ref, value) =
            maybe_coin.map_err(|e| RpcError::new(tonic::Code::Internal, e.to_string()))?;
        selected_gas.push(object_ref);
        selected_gas_value += value;
    }

    Ok(selected_gas)
}
