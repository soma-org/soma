// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Stage 7 — balance-mode value transfer.
//!
//! [`TransactionKind::BalanceTransfer`] is a stateless tx that moves
//! value between users via the account-balance accumulator. The sender
//! signs an intent message naming a coin type and a list of
//! `(recipient, amount)` pairs; the executor emits one
//! `BalanceEvent::Withdraw` for the total against the sender plus one
//! `BalanceEvent::Deposit` per recipient. The settlement system tx
//! injected by the consensus handler aggregates all events in the
//! commit and applies the net delta to the on-chain accumulator.
//!
//! Funds-availability is enforced by the reservation pre-pass in
//! [`crate::consensus_handler::run_reservation_prepass`]: the tx's
//! `reservations(unit_fee)` declare both the USDC fee and the transfer
//! total in the chosen coin type, and the pre-pass drops any tx that
//! would push the sender's balance negative across the commit. Reaching
//! this executor therefore implies sufficient balance.

use types::balance::BalanceEvent;
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::temporary_store::TemporaryStore;
use types::transaction::{BalanceTransferArgs, TransactionKind};

use super::{TransactionExecutor, checked_sum};

pub struct BalanceTransferExecutor;

impl BalanceTransferExecutor {
    pub(crate) fn new() -> Self {
        Self {}
    }

    fn execute_transfer(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        args: BalanceTransferArgs,
    ) -> ExecutionResult<()> {
        if args.transfers.is_empty() {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "BalanceTransfer must have at least one recipient".to_string(),
            }
            .into());
        }

        // A zero-amount entry is meaningless on-chain (no balance moves)
        // and almost always an encoding bug. Reject loudly so it doesn't
        // silently slip through.
        if args.transfers.iter().any(|(_, amt)| *amt == 0) {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "BalanceTransfer amounts must be non-zero".to_string(),
            }
            .into());
        }

        // Self-transfer would still settle correctly (Withdraw + Deposit
        // for the same address aggregate to net zero), but it wastes a
        // commit slot on a no-op. Reject so callers can't pad fees with
        // self-transfers.
        if args.transfers.iter().any(|(recipient, _)| *recipient == signer) {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "BalanceTransfer recipient must not equal sender".to_string(),
            }
            .into());
        }

        let total = checked_sum(args.transfers.iter().map(|(_, amt)| *amt))?;

        // Stage 14c.6 (SIP-58 cutover): user-tx executors emit ONLY
        // `AccumulatorWriteV1` records via `emit_accumulator_event`.
        // The per-cp `SettlementScheduler` aggregates these from
        // every tx's `effects.accumulator_events()`, builds a single
        // settlement tx whose `changes: Vec<BalanceEvent>` carries
        // the aggregated net deltas, and dispatches it. Settlement
        // is the SOLE driver of `apply_settlement_events`, so the
        // CF apply path runs exactly once per cp regardless of how
        // many user txs touched a balance.
        //
        // This matches Sui SIP-58: user txs emit delta records,
        // never direct mutations; per-cp settlement aggregates and
        // applies.
        use types::effects::object_change::{AccumulatorAddress, AccumulatorOperation};

        // Withdrawal: one record for the net debit.
        store.emit_accumulator_event(
            AccumulatorAddress::balance(signer, args.coin_type),
            AccumulatorOperation::Split,
            total,
        );

        for (recipient, amount) in args.transfers {
            store.emit_accumulator_event(
                AccumulatorAddress::balance(recipient, args.coin_type),
                AccumulatorOperation::Merge,
                amount,
            );
        }

        Ok(())
    }
}

impl TransactionExecutor for BalanceTransferExecutor {
    fn fee_units(&self, _store: &TemporaryStore, kind: &TransactionKind) -> u32 {
        // Mirror the protocol-level fee_units() on TransactionKind so
        // prepare_gas charges the same regardless of dispatch path.
        kind.fee_units()
    }

    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        _tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        if signer == SomaAddress::ZERO {
            // Defense in depth: BalanceTransfer is a user-only op. The
            // system-address sender (used for Genesis / Settlement /
            // ConsensusCommitPrologue) has no accumulator entry to debit
            // and would mint balance from nothing.
            return Err(ExecutionFailureStatus::SomaError(SomaError::from(
                "BalanceTransfer cannot be sent by the system address".to_string(),
            ))
            .into());
        }

        match kind {
            TransactionKind::BalanceTransfer(args) => self.execute_transfer(store, signer, args),
            _ => Err(ExecutionFailureStatus::InvalidTransactionType.into()),
        }
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
    use types::transaction::{BalanceTransferArgs, InputObjects, TransactionKind};

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

    fn xfer(coin_type: CoinType, transfers: Vec<(SomaAddress, u64)>) -> TransactionKind {
        TransactionKind::BalanceTransfer(BalanceTransferArgs { coin_type, transfers })
    }

    /// Happy path: one sender, two recipients. The executor emits a
    /// single Withdraw for the total followed by one Deposit per
    /// recipient, in order.
    #[test]
    fn emits_withdraw_then_deposits_per_recipient() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let bob = SomaAddress::random();
        let carol = SomaAddress::random();
        let mut executor = BalanceTransferExecutor::new();

        executor
            .execute(
                &mut store,
                alice,
                xfer(CoinType::Usdc, vec![(bob, 30), (carol, 70)]),
                TransactionDigest::default(),
            )
            .expect("transfer must succeed");

        // Stage 14c.6: user-tx executors emit ONLY AccumulatorWriteV1
        // records. The legacy `BalanceEvent` emission was dropped —
        // per-cp settlement aggregates the accumulator events and
        // drives CF apply.
        assert!(
            store.balance_events().is_empty(),
            "Stage 14c.6: user-tx executors must NOT emit BalanceEvents",
        );

        // AccumulatorWriteV1 records: three (one per accumulator),
        // each at the matching delta.
        use types::accumulator::BalanceAccumulator;
        use types::effects::object_change::{
            AccumulatorAddress, AccumulatorOperation, AccumulatorValue,
        };

        let writes = store.accumulator_writes();
        assert_eq!(
            writes.len(),
            3,
            "one AccumulatorWriteV1 per touched accumulator, got {writes:#?}",
        );

        let alice_id = BalanceAccumulator::derive_id(alice, CoinType::Usdc);
        let bob_id = BalanceAccumulator::derive_id(bob, CoinType::Usdc);
        let carol_id = BalanceAccumulator::derive_id(carol, CoinType::Usdc);

        let alice_write = writes.get(&alice_id).expect("alice's accumulator write must exist");
        assert_eq!(alice_write.address, AccumulatorAddress::balance(alice, CoinType::Usdc));
        assert_eq!(alice_write.operation, AccumulatorOperation::Split);
        assert_eq!(alice_write.value, AccumulatorValue::U64(100));

        let bob_write = writes.get(&bob_id).expect("bob's accumulator write must exist");
        assert_eq!(bob_write.operation, AccumulatorOperation::Merge);
        assert_eq!(bob_write.value, AccumulatorValue::U64(30));

        let carol_write = writes.get(&carol_id).expect("carol's accumulator write must exist");
        assert_eq!(carol_write.operation, AccumulatorOperation::Merge);
        assert_eq!(carol_write.value, AccumulatorValue::U64(70));
    }

    /// Stage 14c.2: a single tx that emits multiple events to the
    /// SAME accumulator (e.g., gas + transfer Withdraw against the
    /// sender's USDC) must net them into ONE AccumulatorWriteV1
    /// record. Sui SIP-58's `merge()` does the same — one entry per
    /// accumulator per tx.
    #[test]
    fn multiple_emissions_to_same_accumulator_net_within_a_tx() {
        use types::accumulator::BalanceAccumulator;
        use types::effects::object_change::{
            AccumulatorAddress, AccumulatorOperation, AccumulatorValue,
        };

        let mut store = empty_store();
        let alice = SomaAddress::random();
        let bob = SomaAddress::random();
        // Simulate prepare_gas Withdraw + BalanceTransfer Withdraw
        // on the same (sender, USDC) accumulator. Both call
        // emit_accumulator_event independently; the result is one
        // record with the net.
        store.emit_accumulator_event(
            AccumulatorAddress::balance(alice, CoinType::Usdc),
            AccumulatorOperation::Split,
            10,
        );
        store.emit_accumulator_event(
            AccumulatorAddress::balance(alice, CoinType::Usdc),
            AccumulatorOperation::Split,
            90,
        );
        // bob's accumulator stays untouched — separate ID, separate
        // record.
        store.emit_accumulator_event(
            AccumulatorAddress::balance(bob, CoinType::Usdc),
            AccumulatorOperation::Merge,
            100,
        );

        let writes = store.accumulator_writes();
        assert_eq!(writes.len(), 2, "one entry per touched accumulator: {writes:#?}");

        let alice_id = BalanceAccumulator::derive_id(alice, CoinType::Usdc);
        let alice_write = writes.get(&alice_id).unwrap();
        assert_eq!(alice_write.operation, AccumulatorOperation::Split);
        assert_eq!(alice_write.value, AccumulatorValue::U64(100), "10 + 90 netted");
    }

    /// A withdraw and an equal-magnitude deposit on the same
    /// accumulator within one tx net to zero — the record stays
    /// behind as a no-op (Merge with magnitude 0). Downstream
    /// consumers tolerate zero-magnitude records; settlement
    /// applies them as a no-op delta.
    #[test]
    fn offsetting_emissions_net_to_zero() {
        use types::accumulator::BalanceAccumulator;
        use types::effects::object_change::{
            AccumulatorAddress, AccumulatorOperation, AccumulatorValue,
        };

        let mut store = empty_store();
        let alice = SomaAddress::random();
        store.emit_accumulator_event(
            AccumulatorAddress::balance(alice, CoinType::Soma),
            AccumulatorOperation::Merge,
            500,
        );
        store.emit_accumulator_event(
            AccumulatorAddress::balance(alice, CoinType::Soma),
            AccumulatorOperation::Split,
            500,
        );

        let writes = store.accumulator_writes();
        let alice_id = BalanceAccumulator::derive_id(alice, CoinType::Soma);
        let w = writes.get(&alice_id).expect("net-zero entry stays behind");
        assert_eq!(w.value, AccumulatorValue::U64(0));
    }

    /// Empty transfer list is rejected — a no-op tx wastes a commit
    /// slot and almost always indicates a wallet bug.
    #[test]
    fn rejects_empty_transfers() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let mut executor = BalanceTransferExecutor::new();
        let err = executor
            .execute(
                &mut store,
                alice,
                xfer(CoinType::Usdc, vec![]),
                TransactionDigest::default(),
            )
            .expect_err("empty transfer list must be rejected");
        let msg = format!("{:?}", err);
        assert!(msg.contains("at least one recipient"), "got: {}", msg);
        assert!(store.balance_events().is_empty());
    }

    /// Zero-amount entries are rejected — the chain doesn't store
    /// zero-value movements, so a 0 entry is always a bug.
    #[test]
    fn rejects_zero_amount() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let bob = SomaAddress::random();
        let carol = SomaAddress::random();
        let mut executor = BalanceTransferExecutor::new();
        let err = executor
            .execute(
                &mut store,
                alice,
                xfer(CoinType::Usdc, vec![(bob, 50), (carol, 0)]),
                TransactionDigest::default(),
            )
            .expect_err("zero amount must be rejected");
        let msg = format!("{:?}", err);
        assert!(msg.contains("non-zero"), "got: {}", msg);
        // No partial events should leak into the store.
        assert!(store.balance_events().is_empty());
    }

    /// Self-transfers net to zero in settlement, but they're never the
    /// caller's intent — reject so we don't burn a commit slot on a
    /// no-op.
    #[test]
    fn rejects_self_transfer() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let mut executor = BalanceTransferExecutor::new();
        let err = executor
            .execute(
                &mut store,
                alice,
                xfer(CoinType::Usdc, vec![(alice, 10)]),
                TransactionDigest::default(),
            )
            .expect_err("self-transfer must be rejected");
        let msg = format!("{:?}", err);
        assert!(msg.contains("must not equal sender"), "got: {}", msg);
        assert!(store.balance_events().is_empty());
    }

    /// System sender is rejected. Mirrors `SettlementExecutor` and
    /// `ConsensusCommitExecutor` checks — the system address has no
    /// accumulator entry, so a Withdraw against it would underflow and
    /// effectively mint balance to recipients.
    #[test]
    fn rejects_system_sender() {
        let mut store = empty_store();
        let bob = SomaAddress::random();
        let mut executor = BalanceTransferExecutor::new();
        let err = executor
            .execute(
                &mut store,
                SomaAddress::ZERO,
                xfer(CoinType::Usdc, vec![(bob, 10)]),
                TransactionDigest::default(),
            )
            .expect_err("system sender must be rejected");
        let msg = format!("{:?}", err);
        assert!(msg.contains("system address"), "got: {}", msg);
        assert!(store.balance_events().is_empty());
    }

    /// Overflowing the total amount is reported as ArithmeticOverflow
    /// (not silently truncated).
    #[test]
    fn rejects_total_amount_overflow() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let bob = SomaAddress::random();
        let carol = SomaAddress::random();
        let mut executor = BalanceTransferExecutor::new();
        let err = executor
            .execute(
                &mut store,
                alice,
                xfer(CoinType::Usdc, vec![(bob, u64::MAX), (carol, 1)]),
                TransactionDigest::default(),
            )
            .expect_err("overflow must be rejected");
        let msg = format!("{:?}", err);
        assert!(msg.contains("ArithmeticOverflow"), "got: {}", msg);
    }
}
