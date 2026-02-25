// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use parking_lot::{Mutex, RwLock};
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tracing::{debug, error, info};
use types::committee::AuthorityIndex;
use types::consensus::block::{BlockAPI as _, Round};
use types::consensus::context::Context;
use types::error::ConsensusError;

use crate::dag_state::DagState;
use crate::network::{NetworkClient, NetworkService};

/// Subscriber manages the block stream subscriptions to other peers, taking care of retrying
/// when subscription streams break. Blocks returned from the peer are sent to the authority
/// service for processing.
/// Currently subscription management for individual peer is not exposed, but it could become
/// useful in future.
pub(crate) struct Subscriber<C: NetworkClient, S: NetworkService> {
    context: Arc<Context>,
    network_client: Arc<C>,
    authority_service: Arc<S>,
    dag_state: Arc<RwLock<DagState>>,
    #[allow(clippy::type_complexity)]
    subscriptions: Arc<Mutex<Box<[Option<JoinHandle<()>>]>>>,
}

impl<C: NetworkClient, S: NetworkService> Subscriber<C, S> {
    pub(crate) fn new(
        context: Arc<Context>,
        network_client: Arc<C>,
        authority_service: Arc<S>,
        dag_state: Arc<RwLock<DagState>>,
    ) -> Self {
        let subscriptions = (0..context.committee.size()).map(|_| None).collect::<Vec<_>>();
        Self {
            context,
            network_client,
            authority_service,
            dag_state,
            subscriptions: Arc::new(Mutex::new(subscriptions.into_boxed_slice())),
        }
    }

    pub(crate) fn subscribe(&self, peer: AuthorityIndex) {
        if peer == self.context.own_index {
            error!("Attempt to subscribe to own validator {peer} is ignored!");
            return;
        }
        let context = self.context.clone();
        let network_client = self.network_client.clone();
        let authority_service = self.authority_service.clone();
        let (mut last_received, gc_round) = {
            let dag_state = self.dag_state.read();
            (dag_state.get_last_block_for_authority(peer).round(), dag_state.gc_round())
        };

        // If the latest block we have accepted by an authority is older than the current gc round,
        // then do not attempt to fetch any blocks from that point as they will simply be skipped. Instead
        // do attempt to fetch from the gc round.
        if last_received < gc_round {
            info!(
                "Last received block for peer {peer} is older than GC round, {last_received} < {gc_round}, fetching from GC round"
            );
            last_received = gc_round;
        }

        let mut subscriptions = self.subscriptions.lock();
        self.unsubscribe_locked(peer, &mut subscriptions[peer.value()]);
        subscriptions[peer.value()] = Some(tokio::spawn(Self::subscription_loop(
            context,
            network_client,
            authority_service,
            peer,
            last_received,
        )));
    }

    pub(crate) fn stop(&self) {
        let mut subscriptions = self.subscriptions.lock();
        for (peer, _) in self.context.committee.authorities() {
            self.unsubscribe_locked(peer, &mut subscriptions[peer.value()]);
        }
    }

    fn unsubscribe_locked(&self, peer: AuthorityIndex, subscription: &mut Option<JoinHandle<()>>) {
        if let Some(subscription) = subscription.take() {
            subscription.abort();
        }
        // There is a race between shutting down the subscription task and clearing the metric here.
        // TODO: fix the race when unsubscribe_locked() gets called outside of stop().
    }

    async fn subscription_loop(
        context: Arc<Context>,
        network_client: Arc<C>,
        authority_service: Arc<S>,
        peer: AuthorityIndex,
        last_received: Round,
    ) {
        const IMMEDIATE_RETRIES: i64 = 3;
        // When not immediately retrying, limit retry delay between 100ms and 10s.
        const INITIAL_RETRY_INTERVAL: Duration = Duration::from_millis(100);
        const MAX_RETRY_INTERVAL: Duration = Duration::from_secs(10);
        const RETRY_INTERVAL_MULTIPLIER: f32 = 1.2;
        let peer_hostname = &context
            .committee
            .authority_by_authority_index(peer)
            .expect("Peer should exist in committee")
            .hostname;
        let mut retries: i64 = 0;
        let mut delay = INITIAL_RETRY_INTERVAL;
        'subscription: loop {
            if retries > IMMEDIATE_RETRIES {
                debug!(
                    "Delaying retry {} of peer {} subscription, in {} seconds",
                    retries,
                    peer_hostname,
                    delay.as_secs_f32(),
                );
                sleep(delay).await;
                // Update delay for the next retry.
                delay = delay.mul_f32(RETRY_INTERVAL_MULTIPLIER).min(MAX_RETRY_INTERVAL);
            } else if retries > 0 {
                // Retry immediately, but still yield to avoid monopolizing the thread.
                tokio::task::yield_now().await;
            } else {
                // First attempt, reset delay for next retries but no waiting.
                delay = INITIAL_RETRY_INTERVAL;
            }
            retries += 1;

            let mut blocks = match network_client
                .subscribe_blocks(peer, last_received, MAX_RETRY_INTERVAL)
                .await
            {
                Ok(blocks) => {
                    debug!(
                        "Subscribed to peer {} {} after {} attempts",
                        peer, peer_hostname, retries
                    );

                    blocks
                }
                Err(e) => {
                    debug!(
                        "Failed to subscribe to blocks from peer {} {}: {}",
                        peer, peer_hostname, e
                    );

                    continue 'subscription;
                }
            };

            // Now can consider the subscription successful

            'stream: loop {
                match blocks.next().await {
                    Some(block) => {
                        let result = authority_service.handle_send_block(peer, block.clone()).await;
                        if let Err(e) = result {
                            match e {
                                ConsensusError::BlockRejected { block_ref, reason } => {
                                    debug!(
                                        "Failed to process block from peer {} {} for block {:?}: {}",
                                        peer, peer_hostname, block_ref, reason
                                    );
                                }
                                _ => {
                                    info!(
                                        "Invalid block received from peer {} {}: {}",
                                        peer, peer_hostname, e
                                    );
                                }
                            }
                        }
                        // Reset retries when a block is received.
                        retries = 0;
                    }
                    None => {
                        debug!("Subscription to blocks from peer {} {} ended", peer, peer_hostname);
                        retries += 1;
                        break 'stream;
                    }
                }
            }
        }
    }
}
