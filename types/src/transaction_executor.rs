// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::{
    effects::TransactionEffects,
    error::{ExecutionError, ExecutionResult, SomaError},
    full_checkpoint_content::ObjectSet,
    object::{Object, ObjectID},
    quorum_driver::{ExecuteTransactionRequest, ExecuteTransactionResponse, QuorumDriverError},
    transaction::TransactionData,
};

/// Trait to define the interface for how the REST service interacts with a a QuorumDriver or a
/// simulated transaction executor.
#[async_trait::async_trait]
pub trait TransactionExecutor: Send + Sync {
    async fn execute_transaction(
        &self,
        request: ExecuteTransactionRequest,
        client_addr: Option<std::net::SocketAddr>,
    ) -> Result<ExecuteTransactionResponse, QuorumDriverError>;

    fn simulate_transaction(
        &self,
        transaction: TransactionData,
        checks: TransactionChecks,
    ) -> Result<SimulateTransactionResult, SomaError>;
}

pub struct SimulateTransactionResult {
    pub effects: TransactionEffects,
    pub objects: ObjectSet,
    pub execution_result: ExecutionResult,
    pub mock_gas_id: Option<ObjectID>,
}

#[derive(Default, Debug, Copy, Clone)]
pub enum TransactionChecks {
    #[default]
    Enabled,
    Disabled,
}

impl TransactionChecks {
    pub fn disabled(self) -> bool {
        matches!(self, Self::Disabled)
    }

    pub fn enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}
