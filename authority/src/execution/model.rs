use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    object::{Object, ObjectID, ObjectRef, ObjectType, Owner},
    system_state::SystemState,
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
};

use crate::execution::BPS_DENOMINATOR;

use super::{FeeCalculator, TransactionExecutor, object::check_ownership};

pub struct ModelExecutor;

impl ModelExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Deserialize SystemState from the temporary store.
    fn load_system_state(store: &TemporaryStore) -> ExecutionResult<(Object, SystemState)> {
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();

        let state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize system state: {}",
                    e
                )))
            })?;

        Ok((state_object, state))
    }

    /// Serialize and write back the updated SystemState.
    fn save_system_state(
        store: &mut TemporaryStore,
        state_object: Object,
        state: &SystemState,
    ) -> ExecutionResult<()> {
        let state_bytes = bcs::to_bytes(state).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to serialize updated system state: {}",
                e
            )))
        })?;

        let mut updated = state_object;
        updated.data.update_contents(state_bytes);
        store.mutate_input_object(updated);
        Ok(())
    }

    /// Execute CommitModel: validate parameters, split coin for stake, create StakedSoma.
    fn execute_commit_model(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        let TransactionKind::CommitModel(args) = kind else {
            return Err(ExecutionFailureStatus::InvalidTransactionType);
        };

        let (state_object, mut state) = Self::load_system_state(store)?;

        // Validate architecture version
        if args.architecture_version != state.parameters.model_architecture_version {
            return Err(ExecutionFailureStatus::ModelArchitectureVersionMismatch);
        }

        // Validate minimum stake
        if args.stake_amount < state.parameters.model_min_stake {
            return Err(ExecutionFailureStatus::ModelMinStakeNotMet);
        }

        // Validate commission rate
        if args.commission_rate > BPS_DENOMINATOR {
            return Err(ExecutionFailureStatus::ModelCommissionRateTooHigh);
        }

        // Commit the model in system state (creates pending model + staking pool)
        let staked_soma = state.request_commit_model(
            signer,
            args.model_id,
            args.weights_url_commitment,
            args.weights_commitment,
            args.architecture_version,
            args.stake_amount,
            args.commission_rate,
            args.staking_pool_id,
        )?;

        // Create StakedSoma object
        let staked_soma_object = Object::new_staked_soma_object(
            ObjectID::derive_id(tx_digest, store.next_creation_num()),
            staked_soma,
            Owner::AddressOwner(signer),
            tx_digest,
        );
        store.create_object(staked_soma_object);

        Self::save_system_state(store, state_object, &state)
    }

    /// Execute RevealModel: verify sender is owner, delegate to system state.
    fn execute_reveal_model(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
    ) -> ExecutionResult<()> {
        let TransactionKind::RevealModel(args) = kind else {
            return Err(ExecutionFailureStatus::InvalidTransactionType);
        };

        let (state_object, mut state) = Self::load_system_state(store)?;

        state.request_reveal_model(
            signer,
            &args.model_id,
            args.weights_manifest,
            args.embedding,
        )?;

        Self::save_system_state(store, state_object, &state)
    }

    /// Execute CommitModelUpdate: verify sender is owner, delegate to system state.
    fn execute_commit_model_update(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
    ) -> ExecutionResult<()> {
        let TransactionKind::CommitModelUpdate(args) = kind else {
            return Err(ExecutionFailureStatus::InvalidTransactionType);
        };

        let (state_object, mut state) = Self::load_system_state(store)?;

        state.request_commit_model_update(
            signer,
            &args.model_id,
            args.weights_url_commitment,
            args.weights_commitment,
        )?;

        Self::save_system_state(store, state_object, &state)
    }

    /// Execute RevealModelUpdate: verify sender is owner, delegate to system state.
    fn execute_reveal_model_update(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
    ) -> ExecutionResult<()> {
        let TransactionKind::RevealModelUpdate(args) = kind else {
            return Err(ExecutionFailureStatus::InvalidTransactionType);
        };

        let (state_object, mut state) = Self::load_system_state(store)?;

        state.request_reveal_model_update(
            signer,
            &args.model_id,
            args.weights_manifest,
            args.embedding,
        )?;

        Self::save_system_state(store, state_object, &state)
    }

    /// Execute AddStakeToModel: split coin for stake, create StakedSoma.
    /// Follows the same coin-splitting pattern as StakingExecutor::execute_add_stake.
    fn execute_add_stake_to_model(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        model_id: ObjectID,
        coin_ref: ObjectRef,
        amount: Option<u64>,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        let coin_id = coin_ref.0;
        let is_gas_coin = store.gas_object_id == Some(coin_id);

        // Get source coin
        let source_object = store
            .read_object(&coin_id)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: coin_id })?;

        // Check ownership
        check_ownership(&source_object, signer)?;

        // Verify it's a coin and get balance
        let source_balance = verify_coin(&source_object)?;

        let (state_object, mut state) = Self::load_system_state(store)?;

        match amount {
            Some(stake_amount) => {
                if is_gas_coin {
                    let write_fee = self.calculate_operation_fee(store, 2);
                    let total_fee = value_fee + write_fee;

                    if source_balance < stake_amount + total_fee {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    let staked_soma = state.request_add_stake_to_model(&model_id, stake_amount)?;

                    let staked_soma_object = Object::new_staked_soma_object(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        staked_soma,
                        Owner::AddressOwner(signer),
                        tx_digest,
                    );
                    store.create_object(staked_soma_object);

                    let remaining_balance = source_balance - stake_amount;
                    let mut updated_source = source_object.clone();
                    updated_source.update_coin_balance(remaining_balance);
                    store.mutate_input_object(updated_source);
                } else {
                    if source_balance < stake_amount {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    let staked_soma = state.request_add_stake_to_model(&model_id, stake_amount)?;

                    let staked_soma_object = Object::new_staked_soma_object(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        staked_soma,
                        Owner::AddressOwner(signer),
                        tx_digest,
                    );
                    store.create_object(staked_soma_object);

                    if stake_amount == source_balance {
                        store.delete_input_object(&coin_id);
                    } else {
                        let remaining_balance = source_balance - stake_amount;
                        let mut updated_source = source_object.clone();
                        updated_source.update_coin_balance(remaining_balance);
                        store.mutate_input_object(updated_source);
                    }
                }
            }
            None => {
                let stake_amount;

                if is_gas_coin {
                    let write_fee = self.calculate_operation_fee(store, 1);
                    let total_fee = value_fee + write_fee;

                    if source_balance <= total_fee {
                        return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
                    }

                    stake_amount = source_balance - total_fee;

                    let staked_soma = state.request_add_stake_to_model(&model_id, stake_amount)?;

                    let staked_soma_object = Object::new_staked_soma_object(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        staked_soma,
                        Owner::AddressOwner(signer),
                        tx_digest,
                    );
                    store.create_object(staked_soma_object);

                    store.delete_input_object(&coin_id);
                } else {
                    stake_amount = source_balance;

                    let staked_soma = state.request_add_stake_to_model(&model_id, stake_amount)?;

                    let staked_soma_object = Object::new_staked_soma_object(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        staked_soma,
                        Owner::AddressOwner(signer),
                        tx_digest,
                    );
                    store.create_object(staked_soma_object);

                    store.delete_input_object(&coin_id);
                }
            }
        }

        Self::save_system_state(store, state_object, &state)
    }

    /// Execute SetModelCommissionRate: delegate to system state.
    fn execute_set_model_commission_rate(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        model_id: ObjectID,
        new_rate: u64,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::load_system_state(store)?;

        state.request_set_model_commission_rate(signer, &model_id, new_rate)?;

        Self::save_system_state(store, state_object, &state)
    }

    /// Execute DeactivateModel: delegate to system state.
    fn execute_deactivate_model(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        model_id: ObjectID,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::load_system_state(store)?;

        state.request_deactivate_model(signer, &model_id)?;

        Self::save_system_state(store, state_object, &state)
    }

    /// Execute ReportModel: verify sender is active validator, delegate to system state.
    fn execute_report_model(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        model_id: ObjectID,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::load_system_state(store)?;

        state.report_model(signer, &model_id)?;

        Self::save_system_state(store, state_object, &state)
    }

    /// Execute UndoReportModel: delegate to system state.
    fn execute_undo_report_model(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        model_id: ObjectID,
    ) -> ExecutionResult<()> {
        let (state_object, mut state) = Self::load_system_state(store)?;

        state.undo_report_model(signer, &model_id)?;

        Self::save_system_state(store, state_object, &state)
    }
}

impl TransactionExecutor for ModelExecutor {
    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::CommitModel(_) => {
                self.execute_commit_model(store, signer, kind, tx_digest, value_fee)
            }
            TransactionKind::RevealModel(_) => self.execute_reveal_model(store, signer, kind),
            TransactionKind::CommitModelUpdate(_) => {
                self.execute_commit_model_update(store, signer, kind)
            }
            TransactionKind::RevealModelUpdate(_) => {
                self.execute_reveal_model_update(store, signer, kind)
            }
            TransactionKind::AddStakeToModel { model_id, coin_ref, amount } => self
                .execute_add_stake_to_model(
                    store, signer, model_id, coin_ref, amount, tx_digest, value_fee,
                ),
            TransactionKind::SetModelCommissionRate { model_id, new_rate } => {
                self.execute_set_model_commission_rate(store, signer, model_id, new_rate)
            }
            TransactionKind::DeactivateModel { model_id } => {
                self.execute_deactivate_model(store, signer, model_id)
            }
            TransactionKind::ReportModel { model_id } => {
                self.execute_report_model(store, signer, model_id)
            }
            TransactionKind::UndoReportModel { model_id } => {
                self.execute_undo_report_model(store, signer, model_id)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl FeeCalculator for ModelExecutor {
    fn calculate_value_fee(&self, store: &TemporaryStore, kind: &TransactionKind) -> u64 {
        // Model staking gets the same halved fee as validator staking
        let value_fee_bps = store.fee_parameters.value_fee_bps / 2;

        match kind {
            TransactionKind::CommitModel(args) => {
                if args.stake_amount == 0 {
                    return 0;
                }
                (args.stake_amount * value_fee_bps) / BPS_DENOMINATOR
            }
            TransactionKind::AddStakeToModel { coin_ref, amount, .. } => {
                let stake_amount = if let Some(specific_amount) = amount {
                    *specific_amount
                } else {
                    store.read_object(&coin_ref.0).and_then(|obj| obj.as_coin()).unwrap_or(0)
                };

                if stake_amount == 0 {
                    return 0;
                }

                (stake_amount * value_fee_bps) / BPS_DENOMINATOR
            }
            // Other model transactions have no value fee
            _ => 0,
        }
    }
}

/// Verifies an object is a coin and returns its balance
fn verify_coin(object: &Object) -> Result<u64, ExecutionFailureStatus> {
    object.as_coin().ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
        object_id: object.id(),
        expected_type: ObjectType::Coin,
        actual_type: object.type_().clone(),
    })
}
