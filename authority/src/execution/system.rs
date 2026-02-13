// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-core/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use types::{
    base::SomaAddress,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError, SomaResult},
    object::ObjectID,
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
};

use super::{FeeCalculator, TransactionExecutor};

/// Executor for Genesis transactions
pub struct GenesisExecutor;

impl GenesisExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl TransactionExecutor for GenesisExecutor {
    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        _signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        _value_fee: u64,
    ) -> ExecutionResult<()> {
        if let TransactionKind::Genesis(genesis) = kind {
            // Create all genesis objects
            for object in genesis.objects {
                store.create_object(object.clone());
            }
            Ok(())
        } else {
            Err(ExecutionFailureStatus::InvalidTransactionType)
        }
    }
}

impl FeeCalculator for GenesisExecutor {}

/// Executor for consensus commit transactions
pub struct ConsensusCommitExecutor;

impl ConsensusCommitExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl TransactionExecutor for ConsensusCommitExecutor {
    fn execute(
        &mut self,
        _store: &mut TemporaryStore,
        _signer: SomaAddress,
        _kind: TransactionKind,
        _tx_digest: TransactionDigest,
        _value_fee: u64,
    ) -> ExecutionResult<()> {
        // For consensus commit, we don't process any state changes, just return success
        Ok(())
    }
}

impl FeeCalculator for ConsensusCommitExecutor {}
