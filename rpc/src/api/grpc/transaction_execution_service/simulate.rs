// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use protocol_config::ProtocolConfig;
use types::balance_change::derive_balance_changes_2;
use types::effects::TransactionEffectsAPI;
use types::object::ObjectID;
use types::system_state::SystemStateTrait;
use types::transaction_executor::{SimulateTransactionResult, TransactionChecks};

use crate::api::RpcService;
use crate::api::error::{Result, RpcError};
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::{
    ErrorReason, ExecutedTransaction, Object, ObjectSet, SimulateTransactionRequest,
    SimulateTransactionResponse, Transaction, TransactionEffects,
};
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;

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
        ProtocolConfig::get_for_version_if_supported(
            system_state.protocol_version().into(),
            service.reader.inner().get_chain_identifier()?.chain(),
        )
        .ok_or_else(|| {
            RpcError::new(tonic::Code::Internal, "unable to get current protocol config")
        })?
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

    // Stage 13c: gas is balance-mode. `transaction.gas()` MUST be
    // empty for non-system txs — we no longer pre-fetch coin gas
    // objects on the caller's behalf.

    let SimulateTransactionResult { effects, objects, execution_result } =
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

    let response = SimulateTransactionResponse { transaction, ..Default::default() };
    Ok(response)
}

