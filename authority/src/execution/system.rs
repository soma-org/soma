// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::ExecutionResult;
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;

use super::TransactionExecutor;

/// Executor for Genesis transactions
pub struct GenesisExecutor;

impl GenesisExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl TransactionExecutor for GenesisExecutor {
    fn fee_units(&self, _store: &TemporaryStore, _kind: &TransactionKind) -> u32 {
        // Gasless system tx — `is_system_tx()` short-circuits prepare_gas anyway.
        0
    }

    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        _signer: SomaAddress,
        kind: TransactionKind,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        if let TransactionKind::Genesis(genesis) = kind {
            for object in genesis.objects {
                store.create_object(object.clone());
            }
            Ok(())
        } else {
            Err(ExecutionFailureStatus::InvalidTransactionType)
        }
    }
}

/// Executor for consensus commit transactions
pub struct ConsensusCommitExecutor;

impl ConsensusCommitExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl TransactionExecutor for ConsensusCommitExecutor {
    fn fee_units(&self, _store: &TemporaryStore, _kind: &TransactionKind) -> u32 {
        0
    }

    fn execute(
        &mut self,
        _store: &mut TemporaryStore,
        _signer: SomaAddress,
        _kind: TransactionKind,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        Ok(())
    }
}
