use types::{
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    object::{Object, ObjectType, Owner},
    system_state::SystemState,
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
    SYSTEM_STATE_OBJECT_ID,
};

use super::{FeeCalculator, TransactionExecutor};

pub struct StakingExecutor;

impl StakingExecutor {
    pub fn new() -> Self {
        Self {}
    }
}

impl TransactionExecutor for StakingExecutor {
    fn execute(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        _value_fee: u64,
    ) -> ExecutionResult<()> {
        // Get system state object
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();

        // Deserialize system state
        let mut state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize system state: {}",
                    e
                )))
            })?;

        // Process transaction based on kind
        let result = match kind {
            TransactionKind::AddStake {
                validator_address,
                amount,
            } => {
                // TODO: modify such that there is a coin_ref here that may also be used for gas, in which case fees need to be adjusted accordingly

                // Request to add stake
                let staked_soma = state.request_add_stake(signer, validator_address, amount)?;

                // Create StakedSoma object and add to store
                let staked_soma_object = Object::new_staked_soma_object(
                    staked_soma,
                    Owner::AddressOwner(signer),
                    tx_digest,
                );

                // Add the new object to the store
                store.create_object(staked_soma_object);

                Ok(())
            }
            TransactionKind::WithdrawStake { staked_soma } => {
                // Get StakedSoma object
                let staked_soma_object = store
                    .read_object(&staked_soma.0)
                    .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                        object_id: staked_soma.0,
                    })?
                    .clone();

                // Extract StakedSoma data
                let staked_soma = Object::as_staked_soma(&staked_soma_object).ok_or_else(|| {
                    ExecutionFailureStatus::InvalidObjectType {
                        object_id: staked_soma_object.id(),
                        expected_type: ObjectType::StakedSoma,
                        actual_type: staked_soma_object.type_().clone(),
                    }
                })?;

                // Process withdrawal
                let withdrawn_amount = state.request_withdraw_stake(staked_soma)?;

                // Delete StakedSoma object
                store.delete_input_object(&staked_soma_object.id());

                // Create a Coin object with the withdrawn amount
                let new_coin =
                    Object::new_coin(withdrawn_amount, Owner::AddressOwner(signer), tx_digest);
                store.create_object(new_coin);

                Ok(())
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        };

        // Early return on error
        result?;

        // Update system state
        let state_bytes = bcs::to_bytes(&state).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to serialize updated system state: {}",
                e
            )))
        })?;

        let mut updated_state_object = state_object;
        updated_state_object.data.update_contents(state_bytes);
        store.mutate_input_object(updated_state_object);

        Ok(())
    }
}

impl FeeCalculator for StakingExecutor {}
