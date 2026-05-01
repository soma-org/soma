// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-consensus-commit settlement of the account-balance accumulator.
//!
//! This executor takes a [`TransactionKind::Settlement`] whose payload is
//! the *aggregated* per-(owner, coin_type) net delta for every user tx
//! in a commit, and emits each entry as a `BalanceEvent` on the
//! `TemporaryStore`. The post-execution write path
//! ([`crate::authority_store::AuthorityStore::write_one_transaction_outputs`])
//! checks the tx kind and applies these events as deltas to the
//! `accumulator_balances` column family — atomically, in the same DB
//! batch as the commit's other transaction outputs.
//!
//! Stage 3 of the account-balance migration. No user executor emits
//! events yet (gas migration is Stage 6), so until then the consensus
//! handler either skips settlement injection entirely or feeds in an
//! empty `changes` list, which executes as a no-op.

use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::temporary_store::TemporaryStore;
use types::transaction::TransactionKind;

use super::TransactionExecutor;

pub struct SettlementExecutor;

impl SettlementExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl TransactionExecutor for SettlementExecutor {
    fn fee_units(&self, _store: &TemporaryStore, _kind: &TransactionKind) -> u32 {
        // System tx — gasless. Mirrors GenesisExecutor / ConsensusCommitExecutor.
        0
    }

    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // Only the system address may submit settlements. The consensus
        // handler builds them via `TransactionData::new_system_transaction`
        // which sets sender to `SomaAddress::ZERO`. A non-zero sender
        // would let any user inject arbitrary balance changes — defense
        // in depth even if upstream validation is correct.
        if signer != SomaAddress::ZERO {
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Settlement must be sent by the system address, got {}",
                signer
            )))
            .into());
        }

        let settlement = match kind {
            TransactionKind::Settlement(s) => s,
            _ => return Err(ExecutionFailureStatus::InvalidTransactionType.into()),
        };

        // Forward each pre-aggregated change to the temp store. The
        // perpetual-store write path drains these and applies them as
        // deltas to the accumulator. Order is preserved (already sorted
        // by `(owner, coin_type)` at aggregation time) so the digest is
        // deterministic.
        for change in settlement.changes {
            store.emit_balance_event(change);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use protocol_config::Chain;
    use types::balance::BalanceEvent;
    use types::base::SomaAddress;
    use types::digests::TransactionDigest;
    use types::object::CoinType;
    use types::system_state::FeeParameters;
    use types::temporary_store::TemporaryStore;
    use types::transaction::{InputObjects, SettlementTransaction, TransactionKind};

    use super::*;

    fn empty_store() -> TemporaryStore {
        TemporaryStore::new(
            InputObjects::new(Vec::new()),
            Vec::new(),
            TransactionDigest::default(),
            0,
            FeeParameters { unit_fee: 0 },
            0,
            Chain::Unknown,
        )
    }

    fn settlement(changes: Vec<BalanceEvent>) -> TransactionKind {
        TransactionKind::Settlement(SettlementTransaction {
            epoch: 0,
            round: 0,
            sub_dag_index: None,
            changes,
        })
    }

    /// The happy path: a settlement with two changes flushes them to
    /// the temp store's balance-event buffer in order.
    #[test]
    fn settlement_emits_changes_to_temp_store() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let bob = SomaAddress::random();
        let withdraw = BalanceEvent::withdraw(alice, CoinType::Usdc, 100);
        let deposit = BalanceEvent::deposit(bob, CoinType::Usdc, 100);

        let mut executor = SettlementExecutor::new();
        executor
            .execute(
                &mut store,
                SomaAddress::ZERO,
                settlement(vec![withdraw, deposit]),
                TransactionDigest::default(),
            )
            .expect("settlement must succeed");

        // Every change in the settlement appears in the buffer in
        // insertion order. The perpetual store's write path consumes
        // these as deltas.
        assert_eq!(store.balance_events(), &[withdraw, deposit]);
    }

    /// An empty settlement is a valid no-op — important because every
    /// commit injects a settlement tx whether or not user txs emitted
    /// events.
    #[test]
    fn empty_settlement_is_a_noop() {
        let mut store = empty_store();
        let mut executor = SettlementExecutor::new();
        executor
            .execute(
                &mut store,
                SomaAddress::ZERO,
                settlement(vec![]),
                TransactionDigest::default(),
            )
            .expect("empty settlement must succeed");
        assert!(store.balance_events().is_empty());
    }

    /// A non-system sender is rejected. Mirrors
    /// `ConsensusCommitExecutor::rejects_non_system_sender` — without
    /// this check, a user could mint balance arbitrarily.
    #[test]
    fn rejects_non_system_sender() {
        let mut store = empty_store();
        let mut executor = SettlementExecutor::new();
        let user = SomaAddress::random();
        assert_ne!(user, SomaAddress::ZERO);

        let alice = SomaAddress::random();
        let attack = BalanceEvent::deposit(alice, CoinType::Usdc, u64::MAX);

        let err = executor
            .execute(&mut store, user, settlement(vec![attack]), TransactionDigest::default())
            .expect_err("non-system sender must be rejected");
        let msg = format!("{:?}", err);
        assert!(
            msg.contains("must be sent by the system address"),
            "error must call out the system-address requirement, got: {}",
            msg
        );

        // Critical invariant: nothing leaked into the temp store buffer.
        assert!(store.balance_events().is_empty());
    }

    /// A non-Settlement TransactionKind reaching this executor is a
    /// dispatch bug — make the failure loud.
    #[test]
    fn rejects_non_settlement_kind() {
        let mut store = empty_store();
        let mut executor = SettlementExecutor::new();
        let bogus = TransactionKind::SetCommissionRate { new_rate: 0 };
        let result =
            executor.execute(&mut store, SomaAddress::ZERO, bogus, TransactionDigest::default());
        assert!(result.is_err(), "non-settlement kind must be rejected");
    }

    /// Two empty settlements at different commits must produce different
    /// BCS encodings. The transaction digest is computed over BCS, so
    /// this is the structural check that drives digest uniqueness.
    /// Without commit metadata baked into the kind, the second commit's
    /// settlement would collide with the first and be silently rejected
    /// by `is_tx_already_executed`. Sui solves this by routing
    /// settlement through the `AccumulatorRoot` shared object (whose
    /// version advances each commit); we don't have that object, so the
    /// kind itself carries `(epoch, round, sub_dag_index)`.
    #[test]
    fn empty_settlements_at_different_commits_have_distinct_encodings() {
        let s1 = TransactionKind::Settlement(SettlementTransaction {
            epoch: 5,
            round: 100,
            sub_dag_index: None,
            changes: vec![],
        });
        let s2 = TransactionKind::Settlement(SettlementTransaction {
            epoch: 5,
            round: 101,
            sub_dag_index: None,
            changes: vec![],
        });
        let s3 = TransactionKind::Settlement(SettlementTransaction {
            epoch: 5,
            round: 100,
            sub_dag_index: Some(1),
            changes: vec![],
        });
        let s4 = TransactionKind::Settlement(SettlementTransaction {
            epoch: 6,
            round: 100,
            sub_dag_index: None,
            changes: vec![],
        });

        let b1 = bcs::to_bytes(&s1).unwrap();
        let b2 = bcs::to_bytes(&s2).unwrap();
        let b3 = bcs::to_bytes(&s3).unwrap();
        let b4 = bcs::to_bytes(&s4).unwrap();

        assert_ne!(b1, b2, "different round must encode differently");
        assert_ne!(b1, b3, "different sub_dag_index must encode differently");
        assert_ne!(b1, b4, "different epoch must encode differently");
        assert_ne!(b2, b3);
        assert_ne!(b2, b4);
        assert_ne!(b3, b4);
    }

    /// Same commit metadata + same changes produce identical encodings —
    /// the digest is deterministic, which is exactly what consensus
    /// relies on for cross-validator agreement.
    #[test]
    fn identical_settlements_have_identical_encodings() {
        let alice = SomaAddress::random();
        let s1 = TransactionKind::Settlement(SettlementTransaction {
            epoch: 5,
            round: 100,
            sub_dag_index: Some(2),
            changes: vec![BalanceEvent::deposit(alice, CoinType::Usdc, 42)],
        });
        let s2 = TransactionKind::Settlement(SettlementTransaction {
            epoch: 5,
            round: 100,
            sub_dag_index: Some(2),
            changes: vec![BalanceEvent::deposit(alice, CoinType::Usdc, 42)],
        });
        assert_eq!(bcs::to_bytes(&s1).unwrap(), bcs::to_bytes(&s2).unwrap());
    }
}
