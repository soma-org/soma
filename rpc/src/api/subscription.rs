// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};
use tracing::{info, trace};
use types::checkpoints::CheckpointSequenceNumber;
use types::full_checkpoint_content::Checkpoint;

const CHECKPOINT_MAILBOX_SIZE: usize = 1024;
const MAILBOX_SIZE: usize = 128;
const SUBSCRIPTION_CHANNEL_SIZE: usize = 256;
const MAX_SUBSCRIBERS: usize = 1024;
/// How many recently-committed checkpoints the service retains so it
/// can replay them into a freshly-registered subscriber.
///
/// Closes the race in `execute_transaction_and_wait_for_checkpoint`
/// where the client subscribes *after* its tx's checkpoint has already
/// been broadcast: a subscriber registered now would otherwise see only
/// *future* checkpoints, hit the 30s timeout, and erroneously conclude
/// the chain is stuck. Sized to comfortably cover the round-trip
/// between the client sending `subscribe_checkpoints` and the service
/// task processing the registration — at `min_checkpoint_interval_ms =
/// 200` (`protocol-config`), 16 entries ≈ 3.2s of history, well above
/// the few-ms window we actually need to cover.
const RECENT_CHECKPOINT_BUFFER_SIZE: usize = 16;

struct SubscriptionRequest {
    sender: oneshot::Sender<mpsc::Receiver<Arc<Checkpoint>>>,
}

#[derive(Clone)]
pub struct SubscriptionServiceHandle {
    sender: mpsc::Sender<SubscriptionRequest>,
}

impl SubscriptionServiceHandle {
    pub async fn register_subscription(&self) -> Option<mpsc::Receiver<Arc<Checkpoint>>> {
        let (sender, reciever) = oneshot::channel();
        let request = SubscriptionRequest { sender };
        self.sender.send(request).await.ok()?;

        reciever.await.ok()
    }
}

pub struct SubscriptionService {
    // Mailbox for recieving `Checkpoint` from the Checkpoint Executor
    //
    // Expectation is that checkpoints are recieved in-order
    checkpoint_mailbox: mpsc::Receiver<Checkpoint>,
    mailbox: mpsc::Receiver<SubscriptionRequest>,
    subscribers: Vec<mpsc::Sender<Arc<Checkpoint>>>,
    last_sequence_number: CheckpointSequenceNumber,
    /// Ring buffer of the last `RECENT_CHECKPOINT_BUFFER_SIZE` checkpoints
    /// the service has broadcast. Replayed into freshly-registered
    /// subscribers so the SDK's `execute_transaction_and_wait_for_checkpoint`
    /// doesn't miss a checkpoint that landed during the subscribe-then-
    /// register race window.
    recent_checkpoints: VecDeque<Arc<Checkpoint>>,
}

impl SubscriptionService {
    pub fn build() -> (mpsc::Sender<Checkpoint>, SubscriptionServiceHandle) {
        let (checkpoint_sender, checkpoint_mailbox) = mpsc::channel(CHECKPOINT_MAILBOX_SIZE);
        let (subscription_request_sender, mailbox) = mpsc::channel(MAILBOX_SIZE);

        tokio::spawn(
            Self {
                checkpoint_mailbox,
                mailbox,
                subscribers: Vec::new(),
                last_sequence_number: 0,
                recent_checkpoints: VecDeque::with_capacity(RECENT_CHECKPOINT_BUFFER_SIZE),
            }
            .start(),
        );

        (checkpoint_sender, SubscriptionServiceHandle { sender: subscription_request_sender })
    }

    async fn start(mut self) {
        // Start main loop.
        loop {
            tokio::select! {
                maybe_checkpoint = self.checkpoint_mailbox.recv() => {
                    // Once all handles to our checkpoint_mailbox have been dropped this
                    // will yield `None` and we can terminate the event loop
                    if let Some(checkpoint) = maybe_checkpoint {
                        self.handle_checkpoint(checkpoint);
                    } else {
                        break;
                    }
                },
                maybe_message = self.mailbox.recv() => {
                    // Once all handles to our mailbox have been dropped this
                    // will yield `None` and we can terminate the event loop
                    if let Some(message) = maybe_message {
                        self.handle_message(message);
                    } else {
                        break;
                    }
                },
            }
        }

        info!("RPC Subscription Services ended");
    }

    fn handle_checkpoint(&mut self, checkpoint: Checkpoint) {
        // Check that we recieved checkpoints in-order
        {
            let last_sequence_number = self.last_sequence_number;
            let sequence_number = *checkpoint.summary.sequence_number();

            if last_sequence_number != 0 && (last_sequence_number + 1) != sequence_number {
                panic!(
                    "recieved checkpoint out-of-order. expected checkpoint {}, recieved {}",
                    last_sequence_number + 1,
                    sequence_number
                );
            }
            self.last_sequence_number = sequence_number;
        }

        let checkpoint = Arc::new(checkpoint);

        // Buffer the checkpoint for replay to late-registering subscribers.
        // Eviction is oldest-first so the buffer always holds the most
        // recent `RECENT_CHECKPOINT_BUFFER_SIZE` checkpoints in order.
        if self.recent_checkpoints.len() == RECENT_CHECKPOINT_BUFFER_SIZE {
            self.recent_checkpoints.pop_front();
        }
        self.recent_checkpoints.push_back(Arc::clone(&checkpoint));

        // Try to send the latest checkpoint to all subscribers. If a subscriber's channel is full
        // then they are likely too slow so we drop them.
        self.subscribers.retain(|subscriber| {
            match subscriber.try_send(Arc::clone(&checkpoint)) {
                Ok(()) => {
                    trace!("successfully enqueued checkpont for subscriber");
                    true // Retain this subscriber
                }
                Err(e) => {
                    // It does not matter what the error is - channel full or closed, we drop the subscriber.
                    trace!("unable to enqueue checkpoint for subscriber: {e}");

                    false // Drop this subscriber
                }
            }
        });
    }

    fn handle_message(&mut self, request: SubscriptionRequest) {
        // Check if we've reached the limit to the number of subscribers we can have at one time.
        if self.subscribers.len() >= MAX_SUBSCRIBERS {
            trace!(
                "failed to register new subscriber: hit maximum number of subscribers {}",
                MAX_SUBSCRIBERS
            );
            return;
        }

        let (sender, reciever) = mpsc::channel(SUBSCRIPTION_CHANNEL_SIZE);

        // Replay the recent-checkpoint buffer into the new subscriber's
        // queue *before* handing back the receiver, so the subscriber's
        // stream begins with any checkpoint that landed during the race
        // window between `subscribe_checkpoints` being sent and the
        // request reaching this task. `SUBSCRIPTION_CHANNEL_SIZE` is
        // comfortably larger than `RECENT_CHECKPOINT_BUFFER_SIZE`, so
        // `try_send` only fails if the receiver is already closed — in
        // which case we skip registration entirely.
        for cp in self.recent_checkpoints.iter() {
            if let Err(e) = sender.try_send(Arc::clone(cp)) {
                trace!("dropping new subscriber during replay: {e}");
                return;
            }
        }

        match request.sender.send(reciever) {
            Ok(()) => {
                trace!("successfully registered new subscriber");

                self.subscribers.push(sender);
            }
            Err(e) => {
                trace!("failed to register new subscriber: {e:?}");
            }
        }
    }
}
