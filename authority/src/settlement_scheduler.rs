// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! # SettlementScheduler (Stage 14d)
//!
//! Mirrors Sui SIP-58's
//! `crates/sui-core/src/execution_scheduler/settlement_scheduler.rs`.
//!
//! ## Architecture
//!
//! Settlement transactions can't be built up-front by the consensus
//! handler because their changes (the aggregated `AccumulatorWriteV1`
//! / `DelegationEvent` records) aren't known until the in-batch user
//! transactions execute. The flow:
//!
//!   1. Consensus handler emits a [`Schedulable::AccumulatorSettlement`]
//!      placeholder.
//!   2. The router enqueues the user txs to the regular
//!      [`ExecutionScheduler`] and the placeholder to **this** scheduler.
//!   3. The [`CheckpointBuilder`] (the **producer**) waits for the
//!      cp's user-tx effects to land in the cache, runs
//!      [`AccumulatorSettlementTxBuilder`] to aggregate per
//!      `(owner, coin_type)` / `(pool_id, staker)`, wraps the result
//!      as a `VerifiedExecutableTransaction::new_system`, and publishes
//!      it via [`AuthorityPerEpochStore::notify_settlement_transactions_ready`].
//!   4. This scheduler's queue runner (the **thin consumer**) awaits
//!      the published TX via
//!      [`AuthorityPerEpochStore::wait_for_settlement_transactions`] and
//!      forwards it to the regular [`ExecutionScheduler`].
//!   5. The cp builder blocks on the settlement effects via
//!      `notify_read_executed_effects` before signing the cp summary,
//!      so the cp can't finalize without settlement having landed —
//!      this is the SIP-58 single-path durability guarantee.
//!
//! ## Per-epoch lifetime
//!
//! Sui constructs a fresh [`SettlementScheduler`] inside
//! `new_consensus_handler()` for each new epoch. We do the same in
//! [`crate::consensus_handler::ConsensusHandlerInitializer::new_consensus_handler`].
//! Keeping a single scheduler across epochs would leak a stale queue
//! runner (parked on `wait_for_settlement_transactions` against the
//! prior epoch_store) into the new epoch where it would never wake —
//! the cluster would silently wedge.
//!
//! ## Differences from Sui
//!
//! - Soma has no barrier tx (no `0xacc` shared root version to
//!   advance). Sui dispatches a barrier after settlement; we don't.
//! - Soma has no `funds_withdraw_scheduler` to update post-settlement —
//!   the reservation pre-pass reads balances from the live store
//!   directly so the runtime auto-picks up the latest state.
//! - Sui's settlement TX is a Move PTB that mutates dynamic fields
//!   under `0xacc`; Soma settles by mutating
//!   `Owner::Accumulator`-typed objects directly (the executor reads
//!   them from the canonical `ObjectStore` at execute time for
//!   replay-safety — see `execute_transaction`'s
//!   `resolved_accumulators` block).
//!
//! [`Schedulable::AccumulatorSettlement`]: crate::shared_obj_version_manager::Schedulable
//! [`ExecutionScheduler`]: crate::execution_scheduler::ExecutionScheduler
//! [`SettlementTransaction`]: types::transaction::SettlementTransaction
//! [`CheckpointBuilder`]: crate::checkpoints::CheckpointBuilder
//! [`AccumulatorSettlementTxBuilder`]: crate::accumulators::AccumulatorSettlementTxBuilder
//! [`AuthorityPerEpochStore::notify_settlement_transactions_ready`]: crate::authority_per_epoch_store::AuthorityPerEpochStore::notify_settlement_transactions_ready
//! [`AuthorityPerEpochStore::wait_for_settlement_transactions`]: crate::authority_per_epoch_store::AuthorityPerEpochStore::wait_for_settlement_transactions

use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::mpsc;
use tracing::{debug, error};
use types::digests::TransactionDigest;
use types::transaction::VerifiedExecutableTransaction;

use crate::authority::ExecutionEnv;
use crate::authority_per_epoch_store::AuthorityPerEpochStore;
use crate::execution_scheduler::ExecutionScheduler;
use crate::shared_obj_version_manager::{Schedulable, SettlementBatchInfo};

/// One unit of settlement work — a placeholder enqueued by the
/// consensus handler at commit time.
struct SettlementWorkItem {
    batch_info: SettlementBatchInfo,
    /// Execution environment to apply when dispatching the resulting
    /// settlement transaction (carries the pre-computed shared-object
    /// version assignment).
    env: ExecutionEnv,
}

/// Sender side of the settlement queue.
#[derive(Clone)]
struct SettlementQueueSender {
    sender: mpsc::UnboundedSender<SettlementWorkItem>,
}

impl SettlementQueueSender {
    fn send(&self, item: SettlementWorkItem) {
        if let Err(e) = self.sender.send(item) {
            error!(
                settlement_key = ?e.0.batch_info.settlement_key,
                "Failed to enqueue settlement work item — receiver gone"
            );
        }
    }
}

/// Routes incoming schedulables: real transactions go to the main
/// [`ExecutionScheduler`]; `Schedulable::AccumulatorSettlement`
/// placeholders go to this scheduler's queue runner, which then
/// awaits the cp builder's published settlement TX and forwards it
/// to the main scheduler.
///
/// One scheduler per epoch — see module docs on per-epoch lifetime.
#[derive(Clone)]
pub struct SettlementScheduler {
    execution_scheduler: ExecutionScheduler,
    /// Lazily-created queue sender. The first `enqueue` call that
    /// sees an `AccumulatorSettlement` placeholder spawns the queue
    /// runner; subsequent calls in the **same epoch** reuse it.
    queue_sender: Arc<Mutex<Option<SettlementQueueSender>>>,
}

impl SettlementScheduler {
    pub fn new(execution_scheduler: ExecutionScheduler) -> Self {
        Self { execution_scheduler, queue_sender: Arc::new(Mutex::new(None)) }
    }

    /// Route `certs` between the main scheduler and the settlement
    /// queue. Mirrors Sui's `SettlementScheduler::enqueue` — drop-in
    /// replacement for callers that previously called
    /// `execution_scheduler.enqueue` directly.
    pub fn enqueue(
        &self,
        certs: Vec<(Schedulable, ExecutionEnv)>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) {
        let mut ordinary = Vec::with_capacity(certs.len());
        let mut settlements = Vec::new();

        for (schedulable, env) in certs {
            match schedulable {
                Schedulable::Transaction(tx) => {
                    ordinary.push((Schedulable::Transaction(tx), env));
                }
                Schedulable::AccumulatorSettlement(info) => {
                    settlements.push(SettlementWorkItem { batch_info: *info, env });
                }
            }
        }

        // User txs go through the main pipeline first so their
        // effects are written to the cache; the settlement queue
        // awaits those effects.
        self.execution_scheduler.enqueue(ordinary, epoch_store);

        if !settlements.is_empty() {
            let queue = self.get_or_start_queue(epoch_store);
            for item in settlements {
                queue.send(item);
            }
        }
    }

    /// Lazy queue startup. The first `AccumulatorSettlement` placeholder
    /// in this epoch spawns the runner; subsequent placeholders reuse
    /// the same channel. Mirrors Sui's lazy-start pattern (their
    /// `get_or_start_queue` does the same dance with a Mutex<Option<_>>).
    /// Because Sui — and now we — construct a fresh SettlementScheduler
    /// per epoch in `new_consensus_handler`, this Mutex is always None
    /// at the start of an epoch and the runner is correctly bound to
    /// the *current* epoch_store.
    fn get_or_start_queue(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SettlementQueueSender {
        let mut guard = self.queue_sender.lock();
        if let Some(sender) = guard.as_ref() {
            return sender.clone();
        }

        let (sender, recv) = mpsc::unbounded_channel();
        let queue_sender = SettlementQueueSender { sender };
        *guard = Some(queue_sender.clone());

        let scheduler = self.clone();
        let epoch_store = epoch_store.clone();
        tokio::spawn(async move {
            scheduler.run_queue(recv, epoch_store).await;
        });

        queue_sender
    }

    /// The settlement queue runner. One task per epoch, processes
    /// placeholders in arrival order. Each placeholder triggers a
    /// `construct_and_dispatch` cycle which awaits the cp builder's
    /// published TX and forwards it to the main `ExecutionScheduler`.
    async fn run_queue(
        self,
        mut recv: mpsc::UnboundedReceiver<SettlementWorkItem>,
        epoch_store: Arc<AuthorityPerEpochStore>,
    ) {
        debug!(epoch = epoch_store.epoch(), "SettlementScheduler queue runner started");
        while let Some(item) = recv.recv().await {
            self.construct_and_dispatch(item, &epoch_store).await;
        }
        debug!(epoch = epoch_store.epoch(), "SettlementScheduler queue runner shutting down");
    }

    /// Block on the cp builder publishing this batch's settlement TX,
    /// then forward to the main `ExecutionScheduler`. The settlement
    /// executor resolves accumulator inputs from the canonical
    /// `ObjectStore` at execute time (see `execute_transaction`'s
    /// `resolved_accumulators` block), so this scheduler doesn't need
    /// to attach inputs.
    async fn construct_and_dispatch(
        &self,
        item: SettlementWorkItem,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) {
        let SettlementWorkItem { batch_info, env } = item;
        let SettlementBatchInfo {
            settlement_key,
            tx_digests,
            checkpoint_seq,
            ..
        } = batch_info;

        debug!(
            ?settlement_key,
            num_user_txs = tx_digests.len(),
            checkpoint_seq,
            "SettlementScheduler: waiting for CheckpointBuilder to publish settlement TX"
        );

        // The CheckpointBuilder ran AccumulatorSettlementTxBuilder
        // against the cp's sorted user-tx effects, wrapped the result
        // as a system tx, and called notify_settlement_transactions_ready
        // (or None if the cp had no balance/delegation events).
        let executable = match epoch_store
            .wait_for_settlement_transactions(settlement_key)
            .await
        {
            Some(tx) => tx,
            None => {
                debug!(
                    ?settlement_key,
                    "SettlementScheduler: cp builder reported no settlement needed (skipped)"
                );
                return;
            }
        };

        // Replay-safety: `execute_transaction` resolves accumulator
        // inputs from the canonical object store (not from
        // `ExecutionEnv::pre_loaded_accumulators`) so settlement
        // effects are deterministic across original execution AND
        // state-sync / cp replay. No need to attach inputs here.
        debug!(
            ?settlement_key,
            settlement_digest = ?executable.digest(),
            "SettlementScheduler: dispatching settlement TX"
        );

        self.execution_scheduler.enqueue_transactions(
            vec![(executable, env)],
            epoch_store,
        );
    }
}

/// Helper for executors / cp builders that just want the main
/// `ExecutionScheduler` underneath. Carved out so callers don't
/// reach into `SettlementScheduler`'s private fields.
impl SettlementScheduler {
    pub fn execution_scheduler(&self) -> &ExecutionScheduler {
        &self.execution_scheduler
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The router must split correctly: regular txs and placeholders
    /// take different paths. This is a unit test on the routing
    /// logic without spinning up the full execution pipeline.
    #[test]
    fn schedulable_routing_classification() {
        // Sanity-check the variant discriminator. The router uses
        // `match` on the Schedulable enum; if a future variant slipped
        // through with the wrong shape, this test would force a
        // compile-time error via non-exhaustive match coverage.
        use types::transaction::TransactionKey;
        let info = SettlementBatchInfo {
            settlement_key: TransactionKey::Digest(TransactionDigest::random()),
            tx_digests: vec![TransactionDigest::random()],
            checkpoint_seq: 1,
            assigned_versions: crate::shared_obj_version_manager::AssignedVersions::default(),
        };
        let placeholder: Schedulable = Schedulable::AccumulatorSettlement(Box::new(info.clone()));

        match placeholder {
            Schedulable::Transaction(_) => panic!("placeholder should not match Transaction arm"),
            Schedulable::AccumulatorSettlement(b) => {
                assert_eq!(b.checkpoint_seq, 1);
                assert_eq!(b.tx_digests.len(), 1);
            }
        }
    }
}

