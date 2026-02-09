//! RoundProber periodically checks each peer for the latest rounds they received and accepted
//! from others. This provides insight into how effectively each authority's blocks are propagated
//! and accepted across the network.
//!
//! Unlike inferring accepted rounds from the DAG of each block, RoundProber has the benefit that
//! it remains active even when peers are not proposing. This makes it essential for determining
//! when to disable optimizations that improve DAG quality but may compromise liveness.
//!
//! RoundProber's data sources include the `highest_received_rounds` & `highest_accepted_rounds` tracked
//! by the CoreThreadDispatcher and DagState. The received rounds are updated after blocks are verified
//! but before checking for dependencies. This should make the values more indicative of how well authorities
//! propagate blocks, and less influenced by the quality of ancestors in the proposed blocks. The
//! accepted rounds are updated after checking for dependencies which should indicate the quality
//! of the proposed blocks including its ancestors.

use std::{sync::Arc, time::Duration};

use crate::{
    core_thread::CoreThreadDispatcher, dag_state::DagState, network::NetworkClient,
    round_tracker::PeerRoundTracker,
};

use futures::stream::{FuturesUnordered, StreamExt as _};
use parking_lot::RwLock;
use tokio::{task::JoinHandle, time::MissedTickBehavior};
use types::consensus::{
    block::{BlockAPI as _, Round},
    context::Context,
};
use utils::notify_once::NotifyOnce;

// Handle to control the RoundProber loop and read latest round gaps.
pub(crate) struct RoundProberHandle {
    prober_task: JoinHandle<()>,
    shutdown_notify: Arc<NotifyOnce>,
}

impl RoundProberHandle {
    pub(crate) async fn stop(self) {
        let _ = self.shutdown_notify.notify();
        // Do not abort prober task, which waits for requests to be cancelled.
        if let Err(e) = self.prober_task.await {
            if e.is_panic() {
                std::panic::resume_unwind(e.into_panic());
            }
        }
    }
}

pub(crate) struct RoundProber<C: NetworkClient> {
    context: Arc<Context>,
    core_thread_dispatcher: Arc<dyn CoreThreadDispatcher>,
    round_tracker: Arc<RwLock<PeerRoundTracker>>,
    dag_state: Arc<RwLock<DagState>>,
    network_client: Arc<C>,
    shutdown_notify: Arc<NotifyOnce>,
}

impl<C: NetworkClient> RoundProber<C> {
    pub(crate) fn new(
        context: Arc<Context>,
        core_thread_dispatcher: Arc<dyn CoreThreadDispatcher>,
        round_tracker: Arc<RwLock<PeerRoundTracker>>,
        dag_state: Arc<RwLock<DagState>>,
        network_client: Arc<C>,
    ) -> Self {
        Self {
            context,
            core_thread_dispatcher,
            round_tracker,
            dag_state,
            network_client,
            shutdown_notify: Arc::new(NotifyOnce::new()),
        }
    }

    pub(crate) fn start(self) -> RoundProberHandle {
        let shutdown_notify = self.shutdown_notify.clone();
        let loop_shutdown_notify = shutdown_notify.clone();
        let prober_task = tokio::spawn(async move {
            // With 200 validators, this would result in 200 * 4 * 200 / 2 = 80KB of additional
            // bandwidth usage per sec. We can consider using adaptive intervals, for example
            // 10s by default but reduced to 2s when the propagation delay is higher.
            let mut interval = tokio::time::interval(Duration::from_millis(
                self.context.parameters.round_prober_interval_ms,
            ));
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        self.probe().await;
                    }
                    _ = loop_shutdown_notify.wait() => {
                        break;
                    }
                }
            }
        });
        RoundProberHandle { prober_task, shutdown_notify }
    }

    // Probes each peer for the latest rounds they received from others.
    // Returns the propagation delay of own blocks.
    pub(crate) async fn probe(&self) -> Round {
        let request_timeout =
            Duration::from_millis(self.context.parameters.round_prober_request_timeout_ms);
        let own_index = self.context.own_index;
        let mut requests = FuturesUnordered::new();

        for (peer, _) in self.context.committee.authorities() {
            if peer == own_index {
                continue;
            }
            let network_client = self.network_client.clone();
            requests.push(async move {
                let result = tokio::time::timeout(
                    request_timeout,
                    network_client.get_latest_rounds(peer, request_timeout),
                )
                .await;
                (peer, result)
            });
        }

        let mut highest_received_rounds =
            vec![vec![0; self.context.committee.size()]; self.context.committee.size()];
        let mut highest_accepted_rounds =
            vec![vec![0; self.context.committee.size()]; self.context.committee.size()];

        let blocks = self.dag_state.read().get_last_cached_block_per_authority(Round::MAX);
        let local_highest_accepted_rounds =
            blocks.into_iter().map(|(block, _)| block.round()).collect::<Vec<_>>();
        let last_proposed_round = local_highest_accepted_rounds[own_index];

        // For our own index, the highest received & accepted round is our last
        // accepted round or our last proposed round.
        highest_received_rounds[own_index] = self.core_thread_dispatcher.highest_received_rounds();
        highest_accepted_rounds[own_index] = local_highest_accepted_rounds;
        highest_received_rounds[own_index][own_index] = last_proposed_round;
        highest_accepted_rounds[own_index][own_index] = last_proposed_round;

        loop {
            tokio::select! {
                result = requests.next() => {
                    let Some((peer, result)) = result else { break };
                    let peer_name = &self.context.committee.authority_by_authority_index(peer).expect("Peer should exist in committee").hostname;
                    match result {
                        Ok(Ok((received, accepted))) => {
                            if received.len() == self.context.committee.size()
                            {
                                highest_received_rounds[peer] = received;
                            } else {

                                tracing::warn!("Received invalid number of received rounds from peer {}", peer_name);
                            }

                            if accepted.len() == self.context.committee.size() {
                                highest_accepted_rounds[peer] = accepted;
                            } else {

                                tracing::warn!("Received invalid number of accepted rounds from peer {}", peer_name);
                            }
                        },
                        // When a request fails, the highest received rounds from that authority will be 0
                        // for the subsequent computations.
                        // For propagation delay, this behavior is desirable because the computed delay
                        // increases as this authority has more difficulty communicating with peers. Logic
                        // triggered by high delay should usually be triggered with frequent probing failures
                        // as well.
                        // For quorum rounds computed for peer, this means the values should be used for
                        // positive signals (peer A can propagate its blocks well) rather than negative signals
                        // (peer A cannot propagate its blocks well). It can be difficult to distinguish between
                        // own probing failures and actual propagation issues.
                        Ok(Err(err)) => {

                            tracing::debug!("Failed to get latest rounds from peer {}: {:?}", peer_name, err);
                        },
                        Err(_) => {

                            tracing::debug!("Timeout while getting latest rounds from peer {}", peer_name);
                        },
                    }
                }
                _ = self.shutdown_notify.wait() => break,
            }
        }

        self.round_tracker
            .write()
            .update_from_probe(highest_accepted_rounds, highest_received_rounds);
        let propagation_delay =
            self.round_tracker.read().calculate_propagation_delay(last_proposed_round);

        let _ = self.core_thread_dispatcher.set_propagation_delay(propagation_delay);

        propagation_delay
    }
}
