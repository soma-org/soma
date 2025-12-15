use std::collections::BTreeMap;

use crate::{
    effects::TransactionEffects,
    error::{ExecutionError, ExecutionResult, SomaError},
    full_checkpoint_content::ObjectSet,
    object::{Object, ObjectID},
    quorum_driver::{
        ExecuteTransactionRequest, ExecuteTransactionResponse, InitiateShardWorkRequest,
        InitiateShardWorkResponse, QuorumDriverError,
    },
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

    /// Initiate shard work for a finalized EmbedData transaction.
    /// This computes the VDF, selects the appropriate shard, and sends
    /// the shard auth token to the encoder shard members.
    async fn initiate_shard_work(
        &self,
        request: InitiateShardWorkRequest,
    ) -> Result<InitiateShardWorkResponse, SomaError>;
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
