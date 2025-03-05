use dashmap::DashMap;
use shared::digest::Digest;
use std::{sync::Arc, time::Duration};
use tokio::{
    sync::{oneshot, Semaphore},
    time::sleep,
};

use crate::types::shard::Shard;

#[derive(Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
enum SlotType {
    Commit,
    Reveal,
}
#[derive(Clone)]
pub(crate) struct SlotTracker {
    #[allow(clippy::type_complexity)]
    slots: Arc<DashMap<(Digest<Shard>, SlotType), oneshot::Sender<()>>>,
    semaphore: Arc<Semaphore>, // Limits concurrent tasks
}

impl SlotTracker {
    pub(crate) fn new(max_concurrent_tasks: usize) -> Self {
        Self {
            slots: Arc::new(DashMap::new()),
            semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
        }
    }

    pub(crate) async fn start_commit_vote_timer<F, Fut>(
        &self,
        shard_ref: Digest<Shard>,
        timeout: Duration,
        on_trigger: F,
    ) where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
        // Acquire a permit, blocking if the limit is reached
        let permit = self.semaphore.clone().acquire_owned().await.unwrap();
        let (tx, rx) = oneshot::channel();

        let slot_key = (shard_ref, SlotType::Commit);
        self.slots.insert(slot_key, tx);

        let slots = self.slots.clone();

        tokio::spawn(async move {
            tokio::select! {
                _ = sleep(timeout) => {
                    on_trigger().await;
                    // Timer hits, trigger work
                }
                _ = rx => {
                    on_trigger().await;
                    // Oneshot receives, trigger work
                }
            }
            slots.remove(&slot_key); // Clean up
            drop(permit); // Release the permit when the task completes
        });
    }

    pub(crate) async fn trigger_commit_vote(&self, shard_ref: Digest<Shard>) {
        let slot_key = (shard_ref, SlotType::Commit);
        if let Some((_, tx)) = self.slots.remove(&slot_key) {
            let _ = tx.send(());
        }
    }
    pub(crate) async fn start_reveal_vote_timer<F, Fut>(
        &self,
        shard_ref: Digest<Shard>,
        timeout: Duration,
        on_trigger: F,
    ) where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
        // Acquire a permit, blocking if the limit is reached
        let permit = self.semaphore.clone().acquire_owned().await.unwrap();
        let (tx, rx) = oneshot::channel();

        let slot_key = (shard_ref, SlotType::Reveal);
        self.slots.insert(slot_key, tx);

        let slots = self.slots.clone();

        tokio::spawn(async move {
            tokio::select! {
                _ = sleep(timeout) => {
                    on_trigger().await;
                    // Timer hits, trigger work
                }
                _ = rx => {
                    on_trigger().await;
                    // Oneshot receives, trigger work
                }
            }
            slots.remove(&slot_key); // Clean up
            drop(permit); // Release the permit when the task completes
        });
    }

    pub(crate) async fn trigger_reveal_vote(&self, shard_ref: Digest<Shard>) {
        let slot_key = (shard_ref, SlotType::Reveal);
        if let Some((_, tx)) = self.slots.remove(&slot_key) {
            let _ = tx.send(());
        }
    }
}
