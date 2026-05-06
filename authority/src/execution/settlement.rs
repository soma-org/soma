// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-checkpoint settlement of the account-balance accumulator
//! (SIP-58 single-path discipline, Stage 14d).
//!
//! ## What this executor does
//!
//! Takes a [`TransactionKind::Settlement`] whose payload is the
//! *aggregated* per-(owner, coin_type) and per-(pool_id, staker)
//! net deltas for the cp's user txs (built by the CheckpointBuilder
//! via [`crate::accumulators::AccumulatorSettlementTxBuilder`]) and:
//!
//! 1. Reads each touched `BalanceAccumulator` /
//!    `DelegationAccumulator` object from the canonical store and
//!    writes the new state via `mutate_input_object` (or
//!    `create_object` for first-touch).
//! 2. Re-emits each change as a `BalanceEvent` / `DelegationEvent`
//!    on the temp store, which the post-execution write path
//!    ([`crate::authority_store::AuthorityStore::write_one_transaction_outputs`])
//!    drains into the `accumulator_balances` / `delegations` CFs —
//!    atomically, in the same DB batch as the settlement's object
//!    mutations.
//!
//! ## "Single-path" vs ChangeEpoch
//!
//! For **user txs** the settlement system tx is the SOLE writer of
//! accumulator state (object world AND CFs) — user-tx executors emit
//! `AccumulatorWriteV1` records on `effects.changed_objects` that
//! the settlement aggregator consumes; user-tx writeback does NOT
//! drain accumulator events to the CFs.
//!
//! There is one **deliberate exception**: `TransactionKind::ChangeEpoch`
//! emits its own `delegation_events` (validator commission rewards)
//! and drains them inline via
//! `authority_store::write_one_transaction_outputs`. The cp builder
//! filters `ChangeEpoch` effects out of the settlement aggregation
//! to prevent double-apply (see `construct_and_execute_settlement`).
//! This asymmetry exists because Soma's F1 staking computes
//! commission distribution at the epoch boundary as part of the
//! `advance_epoch` invariant flow; routing it through the per-cp
//! settlement aggregation would invert the intended ordering. Sui
//! doesn't face this because they don't have F1; their epoch
//! transitions go through the same settlement path.

use types::accumulator::BalanceAccumulator;
use types::balance::BalanceEvent;
use types::base::SomaAddress;
use types::digests::TransactionDigest;
use types::effects::ExecutionFailureStatus;
use types::error::{ExecutionResult, SomaError};
use types::object::Object;
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
        tx_digest: TransactionDigest,
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

        // Stage 14d (SIP-58 single-path): settlement is the SOLE
        // writer of accumulator state — both the object world
        // (`mutate_input_object` / `create_object`) AND the
        // `accumulator_balances` / `delegations` CFs (via the
        // re-emitted `BalanceEvent` / `DelegationEvent` records the
        // perpetual store's `apply_settlement_events` /
        // `apply_delegation_events` consume from this tx's effects).
        //
        // Both paths handle first-touch (no prior input object) by
        // falling through to `create_object` — see
        // `apply_settlement_to_object_inputs`. So a "no input objects"
        // case is just "every change is first-touch", which is still
        // a valid execution path.
        apply_settlement_to_object_inputs(store, settlement.changes, tx_digest)?;
        apply_delegation_settlement_to_object_inputs(
            store,
            settlement.delegation_changes,
            tx_digest,
        )?;

        Ok(())
    }
}

/// Stage 14d (SIP-58 single-path): the SOLE writer of accumulator state.
///
/// For each pre-aggregated `BalanceEvent`:
///   1. Derive the canonical accumulator ID.
///   2. Locate the input object via `store.read_object`. Per Stage 14d
///      replay-safety, the accumulator object is loaded from the
///      canonical `ObjectStore` inside `execute_transaction` (the
///      `resolved_accumulators` block in `execution/mod.rs`), NOT
///      pre-loaded via `ExecutionEnv::pre_loaded_accumulators` (which
///      stays empty for replays / state-sync).
///   3. Apply the delta, write the new `BalanceAccumulator` object
///      via `mutate_input_object`/`create_object` (drives the object
///      world; the standard effects pipeline carries it into the
///      global state hash).
///   4. Emit a matching `BalanceEvent` so the perpetual store's
///      `apply_settlement_events` updates the `accumulator_balances`
///      CF inside the same write batch.
///
/// User-tx executors emit ONLY `AccumulatorWriteV1` records (delta
/// records on `effects.changed_objects`); the per-tx writeback path
/// does NOT apply them to the CF. The settlement system tx is the
/// only place CF + object are written, so the two stores can never
/// diverge.
fn apply_settlement_to_object_inputs(
    store: &mut TemporaryStore,
    changes: Vec<BalanceEvent>,
    tx_digest: TransactionDigest,
) -> ExecutionResult<()> {
    for change in changes {
        let owner = change.owner();
        let coin_type = change.coin_type();
        let magnitude = change.amount();
        let acc_id = BalanceAccumulator::derive_id(owner, coin_type);

        // `store.read_object` returns the input-or-written object
        // matching `id`. For accumulators the cp builder declared as
        // inputs, this returns the existing accumulator at its prior
        // version; the executor's mutation runs through
        // `mutate_input_object`. For first-touch deposits, no input
        // exists and we fall through to `create_object`.
        if let Some(existing) = store.read_object(&acc_id) {
            let mut acc = existing.as_balance_accumulator().ok_or_else(|| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Settlement input at deterministic accumulator ID {acc_id:?} for \
                     ({owner:?}, {coin_type:?}) is not a BalanceAccumulator object"
                )))
            })?;

            let new_balance = match change {
                BalanceEvent::Deposit { .. } => acc.balance.checked_add(magnitude).ok_or_else(
                    || {
                        ExecutionFailureStatus::SomaError(SomaError::from(format!(
                            "Settlement deposit overflow on ({owner:?}, {coin_type:?}): \
                             current={} + delta={} overflows u64",
                            acc.balance, magnitude
                        )))
                    },
                )?,
                BalanceEvent::Withdraw { .. } => acc.balance.checked_sub(magnitude).ok_or_else(
                    || {
                        ExecutionFailureStatus::SomaError(SomaError::from(format!(
                            "Settlement withdraw underflow on ({owner:?}, {coin_type:?}): \
                             current={} - delta={} underflows u64",
                            acc.balance, magnitude
                        )))
                    },
                )?,
            };
            acc.balance = new_balance;

            let mut new_obj = existing.clone();
            new_obj.set_balance_accumulator(&acc);
            store.mutate_input_object(new_obj);
        } else {
            // First-touch: only deposits make sense (a withdraw on
            // a non-existent accumulator would be a reservation-
            // pre-pass bug). Create at Version::MIN.
            match change {
                BalanceEvent::Deposit { .. } => {
                    let acc = BalanceAccumulator::new(owner, coin_type, magnitude);
                    let new_obj = Object::new_balance_accumulator(acc, tx_digest);
                    store.create_object(new_obj);
                }
                BalanceEvent::Withdraw { .. } => {
                    return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                        "Settlement withdraw on non-existent accumulator ({owner:?}, \
                         {coin_type:?}) — reservation pre-pass should have blocked this tx"
                    )))
                    .into());
                }
            }
        }

        // Stage 14d (cp-builder integration landed): settlement is
        // the SOLE writer of CF state now that the cp builder waits
        // for settlement effects before signing the cp. Re-emit the
        // BalanceEvent so `apply_settlement_events` updates the
        // `accumulator_balances` CF in the same atomic write batch
        // that lands the object mutation. The per-tx drain in
        // `authority_store.rs::write_one_transaction_outputs` was
        // dropped in this stage to avoid double-apply.
        store.emit_balance_event(change);
    }
    Ok(())
}

/// Stage 14d (SIP-58 single-path) — delegation accumulator side of
/// `apply_settlement_to_object_inputs`. Reads each touched
/// `DelegationAccumulator` object, applies the aggregated principal
/// delta (with optional F1 period advance), writes the new object
/// state, AND re-emits the `DelegationEvent` so the perpetual store's
/// `apply_delegation_events` keeps the `delegations` CF in sync.
fn apply_delegation_settlement_to_object_inputs(
    store: &mut TemporaryStore,
    changes: Vec<types::temporary_store::DelegationEvent>,
    tx_digest: TransactionDigest,
) -> ExecutionResult<()> {
    use types::accumulator::DelegationAccumulator;

    for change in changes {
        let pool_id = change.pool_id;
        let staker = change.staker;
        let acc_id = DelegationAccumulator::derive_id(pool_id, staker);

        if let Some(existing) = store.read_object(&acc_id) {
            let mut acc = existing.as_delegation_accumulator().ok_or_else(|| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Settlement input at deterministic delegation accumulator ID {acc_id:?} for \
                     ({pool_id:?}, {staker:?}) is not a DelegationAccumulator object"
                )))
            })?;

            let new_principal = match types::balance::apply_delta_to_balance(acc.principal, change.delta) {
                types::balance::BalanceUpdate::Ok(p) => p,
                types::balance::BalanceUpdate::Underflow { current, delta } => {
                    return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                        "Settlement delegation underflow on ({pool_id:?}, {staker:?}): \
                         current={current} delta={delta}"
                    )))
                    .into());
                }
                types::balance::BalanceUpdate::Overflow { current, delta } => {
                    return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                        "Settlement delegation overflow on ({pool_id:?}, {staker:?}): \
                         current={current} delta={delta}"
                    )))
                    .into());
                }
            };
            acc.principal = new_principal;
            if let Some(p) = change.set_period {
                acc.last_collected_period = p;
            }

            // Drain-to-zero CF/object alignment: `apply_delegation_events`
            // (via `write_delegation_cf_to_batch`) deletes the CF row
            // unconditionally when `principal == 0`. Mirror that here by
            // deleting the object so the two stores stay aligned and a
            // subsequent first-touch deposit goes through `create_object`
            // at `Version::MIN` instead of mutating a zero-principal
            // zombie. F4 audit fix.
            if new_principal == 0 {
                store.delete_input_object(&acc_id);
            } else {
                let mut new_obj = existing.clone();
                new_obj.set_delegation_accumulator(&acc);
                store.mutate_input_object(new_obj);
            }
        } else {
            // First-touch delegation: principal must be positive.
            if change.delta <= 0 {
                return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Settlement delegation withdraw on non-existent accumulator \
                     ({pool_id:?}, {staker:?}) — reservation pre-pass should have blocked this tx"
                )))
                .into());
            }
            let principal = change.delta as u64;
            let period = change.set_period.unwrap_or(0);
            let acc = DelegationAccumulator::new(pool_id, staker, principal, period);
            let new_obj = types::object::Object::new_delegation_accumulator(acc, tx_digest);
            store.create_object(new_obj);
        }

        // Stage 14d: settlement drives the `delegations` CF as well
        // as the DelegationAccumulator object world. The per-tx
        // delegation_events drain in
        // `authority_store.rs::write_one_transaction_outputs` is
        // gated to only fire for ChangeEpoch (which emits its own
        // validator-commission delegation events directly).
        store.emit_delegation_event(pool_id, staker, change.delta, change.set_period);
    }
    Ok(())
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
            delegation_changes: vec![],
        })
    }

    /// Happy path: each change in a settlement creates the
    /// corresponding accumulator object AND emits a matching CF event,
    /// in insertion order. Uses two deposits (both first-touch) to
    /// avoid needing to seed prior accumulators in this unit-test
    /// scaffolding — the sender-must-exist invariant for withdraws is
    /// covered separately in `first_touch_balance_withdraw_is_loud_error`.
    #[test]
    fn settlement_emits_changes_to_temp_store() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let bob = SomaAddress::random();
        let alice_dep = BalanceEvent::deposit(alice, CoinType::Usdc, 100);
        let bob_dep = BalanceEvent::deposit(bob, CoinType::Usdc, 200);

        let mut executor = SettlementExecutor::new();
        executor
            .execute(
                &mut store,
                SomaAddress::ZERO,
                settlement(vec![alice_dep, bob_dep]),
                TransactionDigest::default(),
            )
            .expect("settlement must succeed");

        // Every change in the settlement appears in the CF buffer in
        // insertion order. The perpetual store's write path consumes
        // these as deltas.
        assert_eq!(store.balance_events(), &[alice_dep, bob_dep]);
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
            delegation_changes: vec![],
        });
        let s2 = TransactionKind::Settlement(SettlementTransaction {
            epoch: 5,
            round: 101,
            sub_dag_index: None,
            changes: vec![],
            delegation_changes: vec![],
        });
        let s3 = TransactionKind::Settlement(SettlementTransaction {
            epoch: 5,
            round: 100,
            sub_dag_index: Some(1),
            changes: vec![],
            delegation_changes: vec![],
        });
        let s4 = TransactionKind::Settlement(SettlementTransaction {
            epoch: 6,
            round: 100,
            sub_dag_index: None,
            changes: vec![],
            delegation_changes: vec![],
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

    /// Stage 14d safety. First-touch deposit on a balance accumulator
    /// that doesn't exist yet must succeed: the helper detects the
    /// missing input via `store.read_object`, falls through to
    /// `create_object`, and emits the matching CF event. This mirrors
    /// Sui's behavior where the on-chain Move code creates the
    /// dynamic-field on first deposit; we have no Move so the executor
    /// owns object creation directly.
    #[test]
    fn first_touch_balance_deposit_creates_accumulator() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let deposit = BalanceEvent::deposit(alice, CoinType::Usdc, 500);

        apply_settlement_to_object_inputs(
            &mut store,
            vec![deposit],
            TransactionDigest::default(),
        )
        .expect("first-touch deposit must succeed");

        // CF event still gets emitted so the perpetual store's
        // `apply_settlement_events` lands the row.
        assert_eq!(store.balance_events(), &[deposit]);
        // The new accumulator object is in the temp store's writes.
        let acc_id = BalanceAccumulator::derive_id(alice, CoinType::Usdc);
        let created = store
            .execution_results
            .written_objects
            .get(&acc_id)
            .expect("new accumulator object must be in written set");
        let acc = created
            .as_balance_accumulator()
            .expect("created object is a BalanceAccumulator");
        assert_eq!(acc.balance, 500);
    }

    /// Stage 14d safety. First-touch *withdraw* must error loudly
    /// rather than silently underflow. The reservation pre-pass should
    /// have blocked any tx whose declared withdraw exceeded the
    /// sender's pre-commit balance, so a withdraw landing here without
    /// a prior accumulator is a programmer-error invariant violation,
    /// not user input.
    #[test]
    fn first_touch_balance_withdraw_is_loud_error() {
        let mut store = empty_store();
        let alice = SomaAddress::random();
        let bad = BalanceEvent::withdraw(alice, CoinType::Usdc, 1);

        let err = apply_settlement_to_object_inputs(
            &mut store,
            vec![bad],
            TransactionDigest::default(),
        )
        .expect_err("withdraw on non-existent acc must fail");
        let msg = format!("{:?}", err);
        assert!(
            msg.contains("withdraw on non-existent accumulator"),
            "error must call out the invariant, got: {msg}"
        );
        // Critical: the helper must not have leaked a CF event for
        // the failing change — the caller will revert the tx.
        assert!(store.balance_events().is_empty());
    }

    /// Stage 14d safety. First-touch delegation with positive delta
    /// creates the row at the supplied principal + period. This is the
    /// F1-equivalent of "first stake to a pool the staker has never
    /// touched"; the row must be born with `last_collected_period`
    /// set so the next reward calc picks the right start mark.
    #[test]
    fn first_touch_delegation_positive_delta_creates_with_period() {
        use types::accumulator::DelegationAccumulator;
        use types::object::ObjectID;
        use types::temporary_store::DelegationEvent;

        let mut store = empty_store();
        let pool_id = ObjectID::random();
        let staker = SomaAddress::random();
        let de = DelegationEvent {
            pool_id,
            staker,
            delta: 1_000,
            set_period: Some(42),
        };

        apply_delegation_settlement_to_object_inputs(
            &mut store,
            vec![de],
            TransactionDigest::default(),
        )
        .expect("first-touch delegation must succeed");

        let acc_id = DelegationAccumulator::derive_id(pool_id, staker);
        let created = store
            .execution_results
            .written_objects
            .get(&acc_id)
            .expect("new delegation accumulator must be in written set");
        let acc = created
            .as_delegation_accumulator()
            .expect("created object is a DelegationAccumulator");
        assert_eq!(acc.principal, 1_000);
        assert_eq!(acc.last_collected_period, 42);
    }

    /// Stage 14d safety. First-touch delegation with non-positive
    /// delta is an invariant violation: a withdraw or zero-delta on
    /// a pool/staker pair that has no row yet should never reach
    /// settlement — the staking executor's pre-checks (or the
    /// reservation pre-pass for pure withdraws) should have blocked
    /// it. The executor must surface this loudly rather than create
    /// an empty row.
    #[test]
    fn first_touch_delegation_non_positive_delta_is_loud_error() {
        use types::object::ObjectID;
        use types::temporary_store::DelegationEvent;

        let mut store = empty_store();
        let pool_id = ObjectID::random();
        let staker = SomaAddress::random();
        let bad_zero = DelegationEvent {
            pool_id,
            staker,
            delta: 0,
            set_period: Some(1),
        };
        let bad_neg = DelegationEvent {
            pool_id,
            staker,
            delta: -5,
            set_period: None,
        };

        for de in [bad_zero, bad_neg] {
            let err = apply_delegation_settlement_to_object_inputs(
                &mut store,
                vec![de],
                TransactionDigest::default(),
            )
            .expect_err("non-positive first-touch delegation must fail");
            let msg = format!("{:?}", err);
            assert!(
                msg.contains("delegation withdraw on non-existent accumulator"),
                "error must call out the invariant, got: {msg}"
            );
        }
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
            delegation_changes: vec![],
        });
        let s2 = TransactionKind::Settlement(SettlementTransaction {
            epoch: 5,
            round: 100,
            sub_dag_index: Some(2),
            changes: vec![BalanceEvent::deposit(alice, CoinType::Usdc, 42)],
            delegation_changes: vec![],
        });
        assert_eq!(bcs::to_bytes(&s1).unwrap(), bcs::to_bytes(&s2).unwrap());
    }
}
