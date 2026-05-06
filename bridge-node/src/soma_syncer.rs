//! Soma-side checkpoint syncer.
//!
//! Mirrors Sui's `sui-bridge/src/sui_syncer.rs`. Polls the Soma RPC's
//! full-checkpoint endpoint sequentially from the last persisted cursor
//! and feeds each checkpoint into the [`CheckpointWatcher`], which
//! emits high-level `CheckpointEvent`s (new `PendingWithdrawal`, epoch
//! boundary).
//!
//! ## Why polling, not subscription
//!
//! Soma's gRPC `SubscribeCheckpoints` streams from the latest checkpoint
//! and provides no `start_after` parameter. That breaks crash-resume:
//! after a restart we must replay every checkpoint between the last
//! persisted cursor and "now". Polling `get_full_checkpoint(seq)` lets
//! the syncer advance one-by-one from the cursor — same shape as
//! [`crate::eth_syncer`] for the Eth side. The cost is one RPC per
//! checkpoint instead of one stream; cheap given Soma's checkpoint
//! cadence.
//!
//! ## Crash safety
//!
//! Soma cursor in the WAL is the **last successfully processed**
//! sequence number. After processing checkpoint N, we update the cursor
//! to N. On restart we resume from `cursor + 1`. If the bridge
//! observation logic crashes mid-checkpoint (e.g., `process_checkpoint`
//! panics), the cursor stays at N-1 and we'll re-process N — which is
//! idempotent: the WAL's `insert_pending_action` is keyed by digest so
//! a duplicate `BridgeAction::Withdrawal` for the same nonce is a
//! no-op overwrite.

use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use types::full_checkpoint_content::CheckpointData;
use types::object::Object;

use crate::checkpoint_watcher::CheckpointWatcher;
use crate::storage::BridgeOrchestratorTables;

/// How long to back off when `get_full_checkpoint(next_seq)` returns
/// `NotFound` — i.e. we've caught up with the chain and no new
/// checkpoint is ready yet. Mirrors the cadence at which Soma produces
/// checkpoints (~1-2s in practice).
const CATCH_UP_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// How long to back off after a transient RPC failure before retrying.
const ERROR_BACKOFF: Duration = Duration::from_secs(5);

/// Spawn the Soma checkpoint syncer task. Returns a single
/// [`JoinHandle`] — the syncer runs as one tokio task because the
/// processing per checkpoint is cheap and serialization is required to
/// emit `EpochBoundary` events in order.
///
/// Arguments:
///   - `client`: gRPC client to the local Soma fullnode.
///   - `watcher`: drives `process_checkpoint` and emits
///     [`crate::checkpoint_watcher::CheckpointEvent`]s for downstream
///     consumers (the withdrawal handler in `node.rs`).
///   - `wal`: persists the cursor so a restart resumes from the right
///     checkpoint.
///   - `start_seq`: first checkpoint to fetch on this run. Typically
///     `(wal.get_soma_cursor()? + 1)`; the caller owns the +1 so test
///     wiring can pass `0` to start from genesis.
pub fn run_soma_syncer(
    mut client: rpc::api::client::Client,
    mut watcher: CheckpointWatcher,
    wal: Arc<BridgeOrchestratorTables>,
    start_seq: u64,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut next_seq = start_seq;
        info!(start_seq = next_seq, "Soma checkpoint syncer started");
        loop {
            match client.get_full_checkpoint(next_seq).await {
                Ok(data) => {
                    if let Err(e) = process_checkpoint(&mut watcher, &data).await {
                        warn!(seq = next_seq, error = %e, "checkpoint processing failed");
                        // Don't advance cursor on processing failure — the
                        // observation will be retried on next iteration.
                        // (CheckpointWatcher's own logic handles "channel
                        // closed" as fatal; that's only when the consumer
                        // has dropped the rx, which means the bridge node
                        // is shutting down anyway.)
                        tokio::time::sleep(ERROR_BACKOFF).await;
                        continue;
                    }
                    if let Err(e) = wal.update_soma_cursor(next_seq) {
                        warn!(seq = next_seq, error = %e, "WAL cursor update failed");
                        // Cursor failure is non-fatal: we'll re-process
                        // this checkpoint on restart (idempotent).
                    }
                    debug!(seq = next_seq, "checkpoint processed");
                    next_seq += 1;
                }
                Err(s) => {
                    // gRPC NotFound code (5) means "not yet produced" —
                    // catch up wait, NOT an error. Anything else is a
                    // transient RPC failure; back off harder.
                    if s.code() as i32 == 5 {
                        debug!(seq = next_seq, "caught up; waiting for new checkpoints");
                        tokio::time::sleep(CATCH_UP_POLL_INTERVAL).await;
                    } else {
                        warn!(
                            seq = next_seq,
                            code = ?s.code(),
                            error = %s.message(),
                            "RPC failed; will retry",
                        );
                        tokio::time::sleep(ERROR_BACKOFF).await;
                    }
                }
            }
        }
    })
}

/// Extract created objects + epoch from a `CheckpointData` and drive
/// the watcher. Each transaction's freshly-created outputs are flat-
/// mapped together; `process_checkpoint` filters by object type.
async fn process_checkpoint(
    watcher: &mut CheckpointWatcher,
    data: &CheckpointData,
) -> crate::error::BridgeResult<()> {
    let epoch = data.checkpoint_summary.data().epoch;

    // Collect all created objects across all transactions in this
    // checkpoint. We clone here rather than borrowing because the
    // watcher's signature is `&[(ObjectID, Object)]` (owned values),
    // and the per-checkpoint volume is small enough that the clone
    // cost is negligible compared to the RPC roundtrip we just paid.
    let created: Vec<(types::object::ObjectID, Object)> = data
        .transactions
        .iter()
        .flat_map(|tx| tx.created_objects().cloned())
        .map(|obj| (obj.id(), obj))
        .collect();

    watcher.process_checkpoint(epoch, &created).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint_watcher::CheckpointWatcher;
    use crate::storage::BridgeOrchestratorTables;

    /// Smoke test: the syncer factory returns a JoinHandle — covers
    /// the type plumbing. End-to-end behavior requires a live Soma RPC
    /// (or a heavy mock of the proto stack) which is out of scope for
    /// unit tests; the executor's tests cover the action plumbing
    /// downstream of the watcher.
    #[tokio::test]
    async fn test_run_soma_syncer_returns_handle() {
        let temp = tempfile::tempdir().unwrap();
        let wal = BridgeOrchestratorTables::open(temp.path()).unwrap();
        let (watcher, _rx) = CheckpointWatcher::new(16);
        // Constructing a Client requires a URL; just confirm the type
        // signature compiles. Real wiring happens in `node.rs`.
        let result = rpc::api::client::Client::new("http://127.0.0.1:1");
        // If the URL is reachable we'd spawn; otherwise just exercise
        // the constructor. Both paths confirm types line up.
        if let Ok(client) = result {
            let _handle = run_soma_syncer(client, watcher, wal, 0);
            // Don't await — the syncer would loop forever.
            // The handle going out of scope cancels the task.
        }
    }
}
