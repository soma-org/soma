use serde::{Deserialize, Serialize};

use crate::{
    committee::{Committee, CommitteeWithNetworkMetadata},
    effects::{self, TransactionEffects},
    error::SomaResult,
    system_state::{SystemState, SystemStateTrait},
    transaction::Transaction,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genesis {
    transaction: Transaction,
    system_state: SystemState,
    effects: TransactionEffects,
}

impl Genesis {
    pub fn new(
        transaction: Transaction,
        system_state: SystemState,
        effects: TransactionEffects,
    ) -> Self {
        Self {
            transaction,
            system_state,
            effects,
        }
    }

    pub fn system_state(&self) -> SystemState {
        self.system_state.clone()
    }

    pub fn committee_with_network(&self) -> CommitteeWithNetworkMetadata {
        self.system_state().get_current_epoch_committee()
    }

    pub fn committee(&self) -> SomaResult<Committee> {
        Ok(self.committee_with_network().committee().clone())
    }

    pub fn transaction(&self) -> &Transaction {
        &self.transaction
    }

    pub fn effects(&self) -> &TransactionEffects {
        &self.effects
    }
}
