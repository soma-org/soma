// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::CLOCK_OBJECT_ID;
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
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
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // Sui parity: only the system address (`@0x0` in Move,
        // `SomaAddress::ZERO` here) may mutate the Clock. The consensus
        // handler builds the prologue tx via
        // `TransactionData::new_system_transaction`, which sets sender
        // to `SomaAddress::default() == ZERO`. Any other sender — e.g.
        // a user-signed CCP that slipped past upstream validation — is
        // rejected here. Mirrors `assert!(ctx.sender() == @0x0,
        // ENotSystemAddress)` in sui::clock::consensus_commit_prologue.
        if signer != SomaAddress::ZERO {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "ConsensusCommitPrologueV1 must be sent by the system address, got {}",
                signer
            )))
            .into());
        }

        let prologue = match kind {
            TransactionKind::ConsensusCommitPrologueV1(p) => p,
            _ => return Err(ExecutionFailureStatus::InvalidTransactionType.into()),
        };

        let clock_object = store.read_object(&CLOCK_OBJECT_ID).ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from(
                "Clock object missing from prologue inputs".to_string(),
            ))
        })?;

        // Monotonicity: validator-agreed commit timestamps come from
        // consensus output and are required to be non-decreasing across
        // commits. We treat a strictly-prior timestamp as a bug — fail
        // loudly so test runs surface the regression.
        let current_ts = clock_object.clock_timestamp_ms();
        if prologue.commit_timestamp_ms < current_ts {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Non-monotonic commit timestamp: prev={}, new={}",
                current_ts, prologue.commit_timestamp_ms
            )))
            .into());
        }

        let mut updated = clock_object.clone();
        updated.set_clock_timestamp_ms(prologue.commit_timestamp_ms);
        store.mutate_input_object(updated);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use protocol_config::Chain;
    use types::CLOCK_OBJECT_ID;
    use types::base::SomaAddress;
    use types::consensus::ConsensusCommitPrologueV1;
    use types::digests::{
        AdditionalConsensusStateDigest, ConsensusCommitDigest, TransactionDigest,
    };
    use types::object::Object;
    use types::system_state::FeeParameters;
    use types::temporary_store::TemporaryStore;
    use types::transaction::{
        InputObjectKind, InputObjects, ObjectReadResult, ObjectReadResultKind, TransactionKind,
    };

    use super::*;

    fn make_store_with_clock(initial_ts: u64) -> TemporaryStore {
        let clock = Object::new_clock_with_timestamp_for_testing(initial_ts);
        let oref = clock.compute_object_reference();
        let initial_shared_version = clock.owner.start_version().expect("clock is shared");

        let read_result = ObjectReadResult::new(
            InputObjectKind::SharedObject {
                id: oref.0,
                initial_shared_version,
                mutable: true,
            },
            ObjectReadResultKind::Object(clock),
        );

        let inputs = InputObjects::new(vec![read_result]);
        TemporaryStore::new(
            inputs,
            Vec::new(),
            TransactionDigest::default(),
            0,
            FeeParameters { unit_fee: 0 },
            0,
            Chain::Unknown,
        )
    }

    fn prologue_kind(commit_timestamp_ms: u64) -> TransactionKind {
        TransactionKind::ConsensusCommitPrologueV1(ConsensusCommitPrologueV1 {
            epoch: 0,
            round: 1,
            sub_dag_index: None,
            commit_timestamp_ms,
            consensus_commit_digest: ConsensusCommitDigest::default(),
            additional_state_digest: AdditionalConsensusStateDigest::ZERO,
        })
    }

    /// Prologue execution writes the new commit timestamp into the Clock
    /// object — the core happy path.
    #[test]
    fn prologue_writes_commit_timestamp_to_clock() {
        let mut store = make_store_with_clock(0);
        let mut executor = ConsensusCommitExecutor::new();
        executor
            .execute(
                &mut store,
                SomaAddress::ZERO,
                prologue_kind(1_700_000_000_000),
                TransactionDigest::default(),
            )
            .expect("prologue execution succeeds");

        let written = store.read_object(&CLOCK_OBJECT_ID).expect("Clock present after execution");
        assert_eq!(written.clock_timestamp_ms(), 1_700_000_000_000);
        // Helper convenience accessor on TemporaryStore
        assert_eq!(store.read_clock_timestamp_ms(), Some(1_700_000_000_000));
    }

    /// Repeated prologue executions advance the timestamp monotonically.
    #[test]
    fn consecutive_prologues_advance_timestamp() {
        let mut store = make_store_with_clock(100);
        let mut executor = ConsensusCommitExecutor::new();
        executor
            .execute(
                &mut store,
                SomaAddress::ZERO,
                prologue_kind(200),
                TransactionDigest::default(),
            )
            .unwrap();
        assert_eq!(store.read_clock_timestamp_ms(), Some(200));

        executor
            .execute(
                &mut store,
                SomaAddress::ZERO,
                prologue_kind(300),
                TransactionDigest::default(),
            )
            .unwrap();
        assert_eq!(store.read_clock_timestamp_ms(), Some(300));
    }

    /// Equal timestamps are accepted (consensus may legitimately stamp the
    /// same ms across two adjacent commits at sub-ms resolution).
    #[test]
    fn equal_timestamp_is_accepted() {
        let mut store = make_store_with_clock(500);
        let mut executor = ConsensusCommitExecutor::new();
        executor
            .execute(
                &mut store,
                SomaAddress::ZERO,
                prologue_kind(500),
                TransactionDigest::default(),
            )
            .expect("equal timestamps must not error");
        assert_eq!(store.read_clock_timestamp_ms(), Some(500));
    }

    /// Strictly-prior timestamps are rejected — defends against a bug in
    /// consensus output or a corrupted prologue tx.
    #[test]
    fn earlier_timestamp_is_rejected() {
        let mut store = make_store_with_clock(1_000);
        let mut executor = ConsensusCommitExecutor::new();
        let err = executor
            .execute(
                &mut store,
                SomaAddress::ZERO,
                prologue_kind(999),
                TransactionDigest::default(),
            )
            .expect_err("non-monotonic timestamp must error");
        let msg = format!("{:?}", err);
        assert!(msg.contains("Non-monotonic"), "error must mention monotonicity, got: {}", msg);
    }

    /// Sui parity: a CCP signed by a non-system address must be rejected
    /// even if it makes it to the executor. Mirrors Sui's
    /// `assert!(ctx.sender() == @0x0)`.
    #[test]
    fn rejects_non_system_sender() {
        let mut store = make_store_with_clock(0);
        let mut executor = ConsensusCommitExecutor::new();
        let user = SomaAddress::random();
        assert_ne!(user, SomaAddress::ZERO, "test invariant: user != system");

        let err = executor
            .execute(&mut store, user, prologue_kind(123), TransactionDigest::default())
            .expect_err("non-system signer must be rejected");
        let msg = format!("{:?}", err);
        assert!(
            msg.contains("must be sent by the system address"),
            "error must mention the system-address requirement, got: {}",
            msg
        );

        // And the Clock must NOT have moved.
        let clock = store.read_object(&CLOCK_OBJECT_ID).expect("Clock still present");
        assert_eq!(clock.clock_timestamp_ms(), 0, "rejected prologue must leave Clock unchanged");
    }

    /// The executor must reject any non-prologue TransactionKind (the
    /// dispatch invariant is verified upstream, but defense-in-depth here
    /// keeps the function honest in isolation).
    #[test]
    fn rejects_non_prologue_transaction_kind() {
        let mut store = make_store_with_clock(0);
        let mut executor = ConsensusCommitExecutor::new();
        let bogus = TransactionKind::SetCommissionRate { new_rate: 0 };
        let result = executor.execute(
            &mut store,
            SomaAddress::ZERO,
            bogus,
            TransactionDigest::default(),
        );
        assert!(result.is_err(), "non-prologue kind must be rejected");
    }
}
