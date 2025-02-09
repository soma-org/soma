use bytes::Bytes;
use futures::{stream::FuturesOrdered, StreamExt};
use itertools::Itertools;
use parking_lot::RwLock;
use rand::Rng;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    ops::RangeInclusive,
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::{broadcast, mpsc},
    task::JoinSet,
    time::sleep,
};
use tonic::{transport::Channel, Request, Response};
use tracing::{debug, info, instrument, trace, warn};
use types::{
    accumulator::CommitIndex,
    committee::{Authority, Committee, Epoch, EpochId},
    config::state_sync_config::StateSyncConfig,
    consensus::{
        block::{BlockAPI, BlockRef, EndOfEpochData, SignedBlock, VerifiedBlock},
        block_verifier::{BlockVerifier, SignedBlockVerifier},
        commit::{Commit, CommitAPI, CommitDigest, CommitRef, CommittedSubDag, TrustedCommit},
        stake_aggregator::{QuorumThreshold, StakeAggregator},
    },
    crypto::NetworkPublicKey,
    dag::{
        block_manager::BlockManager, committer::universal_committer::UniversalCommitter,
        dag_state::DagState, linearizer::Linearizer,
    },
    error::{ConsensusError, SomaError},
    p2p::{
        active_peers::{ActivePeers, PeerState},
        PeerEvent,
    },
    peer_id::PeerId,
    state_sync::{
        FetchBlocksRequest, FetchCommitsRequest, FetchCommitsResponse,
        GetCommitAvailabilityRequest, GetCommitAvailabilityResponse, GetCommitInfoRequest,
        PushCommitRequest,
    },
    storage::write_store::WriteStore,
};

use crate::{
    discovery::now_unix,
    tonic_gen::{p2p_client::P2pClient, p2p_server::P2p},
};

pub mod tx_verifier;

const COMMIT_SUMMARY_DOWNLOAD_CONCURRENCY: usize = 400;

// Maximum total bytes fetched in a single fetch_blocks() call, after combining the responses.
const MAX_TOTAL_FETCHED_BYTES: usize = 128 * 1024 * 1024;

pub struct PeerHeights {
    /// Table used to track the highest commit for each of our peers.
    peers: HashMap<PeerId, PeerStateSyncInfo>,
    // The amount of time to wait before retry if there are no peers to sync content from.
    wait_interval_when_no_peer_to_sync_content: Duration,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PeerStateSyncInfo {
    /// The digest of the Peer's genesis commit.
    genesis_commit_digest: CommitDigest,
    // Indicates if this Peer is on the same chain as us.
    // on_same_chain_as_us: bool,
    /// Highest commit index we know of for this Peer.
    height: CommitIndex,
    /// lowest available commit for this Peer.
    /// This defaults to 0 for now.
    lowest: CommitIndex,
}

impl PeerHeights {
    pub fn new(wait_interval_when_no_peer_to_sync_content: Duration) -> Self {
        Self {
            peers: HashMap::new(),
            wait_interval_when_no_peer_to_sync_content,
        }
    }

    pub fn highest_known_commit_index(&self) -> Option<CommitIndex> {
        self.peers
            .values()
            .filter_map(|info| Some(info.height))
            .max()
    }

    // Returns a bool that indicates if the update was done successfully.
    //
    // This will return false if the given peer doesn't have an entry
    #[instrument(level = "debug", skip_all, fields(peer_id=?peer_id, commit=?commit))]
    pub fn update_peer_info(
        &mut self,
        peer_id: PeerId,
        commit: CommitIndex,
        low_watermark: Option<CommitIndex>,
    ) -> bool {
        debug!("Update peer info");

        let info = match self.peers.get_mut(&peer_id) {
            Some(info) => info,
            _ => return false,
        };

        info.height = std::cmp::max(commit, info.height);
        if let Some(low_watermark) = low_watermark {
            info.lowest = low_watermark;
        }

        true
    }

    #[instrument(level = "debug", skip_all, fields(peer_id=?peer_id, lowest = ?info.lowest, height = ?info.height))]
    pub fn insert_peer_info(&mut self, peer_id: PeerId, info: PeerStateSyncInfo) {
        use std::collections::hash_map::Entry;
        debug!("Insert peer info");

        match self.peers.entry(peer_id) {
            Entry::Occupied(mut entry) => {
                // If there's already an entry and the genesis commit digests match then update
                // the maximum height. Otherwise we'll use the more recent one
                let entry = entry.get_mut();
                if entry.genesis_commit_digest == info.genesis_commit_digest {
                    entry.height = std::cmp::max(entry.height, info.height);
                } else {
                    *entry = info;
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(info);
            }
        }
    }

    #[cfg(test)]
    pub fn set_wait_interval_when_no_peer_to_sync_content(&mut self, duration: Duration) {
        self.wait_interval_when_no_peer_to_sync_content = duration;
    }

    pub fn wait_interval_when_no_peer_to_sync_content(&self) -> Duration {
        self.wait_interval_when_no_peer_to_sync_content
    }
}

#[derive(Clone, Debug)]
pub enum StateSyncMessage {
    StartSyncJob,
    // Validators will send this to the StateSyncEventLoop in order to kick off notifying our peers
    // of the new commit.
    VerifiedCommit(Box<CommittedSubDag>),
    // Notification that the commit content sync task will send to the event loop in the event
    // it was able to successfully sync a commit's contents. If multiple commits were
    // synced at the same time, only the highest commit is sent.
    SyncedCommit(CommitIndex),
}

// PeerBalancer is an Iterator that selects peers based on RTT with some added randomness.
#[derive(Clone)]
struct PeerBalancer {
    peers: VecDeque<(PeerState, PeerStateSyncInfo)>,
    requested_commit: Option<CommitIndex>,
}

impl PeerBalancer {
    pub fn new(active_peers: ActivePeers, peer_heights: Arc<RwLock<PeerHeights>>) -> Self {
        let peers: Vec<_> = peer_heights
            .read()
            .peers
            .iter()
            .map(|(peer_id, info)| {
                active_peers.get_state(peer_id).map(|peer| (peer, *info)) // TODO: balance peers by rtt (peer.connection_rtt(), peer, *info)
            })
            .collect();
        // peers.sort_by(|(rtt_a, _, _), (rtt_b, _, _)| rtt_a.cmp(rtt_b));
        Self {
            peers: peers
                .into_iter()
                .filter_map(|peer| peer.is_some().then(|| peer.unwrap()))
                .collect(),
            requested_commit: None,
        }
    }

    pub fn with_commit(mut self, commit: CommitIndex) -> Self {
        self.requested_commit = Some(commit);
        self
    }
}

impl Iterator for PeerBalancer {
    type Item = PeerState;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.peers.is_empty() {
            const SELECTION_WINDOW: usize = 2;
            let idx =
                rand::thread_rng().gen_range(0..std::cmp::min(SELECTION_WINDOW, self.peers.len()));
            let (peer, info) = self.peers.remove(idx).unwrap();
            let requested_commit = self.requested_commit.unwrap_or(0);
            if info.height >= requested_commit && info.lowest <= requested_commit {
                return Some(peer);
            }
        }
        None
    }
}

pub struct StateSyncEventLoop<S> {
    config: StateSyncConfig,

    mailbox: mpsc::Receiver<StateSyncMessage>,
    /// Weak reference to our own mailbox
    weak_sender: mpsc::WeakSender<StateSyncMessage>,

    tasks: JoinSet<()>,

    store: S,
    peer_heights: Arc<RwLock<PeerHeights>>,
    commit_event_sender: broadcast::Sender<CommittedSubDag>,

    active_peers: ActivePeers,
    peer_event_receiver: broadcast::Receiver<PeerEvent>,
    block_verifier: Arc<SignedBlockVerifier>,

    dag_state: Arc<RwLock<DagState>>,
    committer: Arc<UniversalCommitter>,
    linearizer: Arc<RwLock<Linearizer>>,
    block_manager: Arc<RwLock<BlockManager>>,
}

impl<S> StateSyncEventLoop<S>
where
    S: WriteStore + Clone + Send + Sync + 'static,
{
    pub fn new(
        config: StateSyncConfig,
        mailbox: mpsc::Receiver<StateSyncMessage>,
        weak_sender: mpsc::WeakSender<StateSyncMessage>,
        store: S,
        peer_heights: Arc<RwLock<PeerHeights>>,
        commit_event_sender: broadcast::Sender<CommittedSubDag>,
        active_peers: ActivePeers,
        peer_event_receiver: broadcast::Receiver<PeerEvent>,
        block_verifier: Arc<SignedBlockVerifier>,
        dag_state: Arc<RwLock<DagState>>,
        committer: Arc<UniversalCommitter>,
        linearizer: Arc<RwLock<Linearizer>>,
        block_manager: Arc<RwLock<BlockManager>>,
    ) -> Self {
        Self {
            config,
            mailbox,
            weak_sender,
            tasks: JoinSet::new(),
            store,
            peer_heights,
            commit_event_sender,
            active_peers,
            peer_event_receiver,
            block_verifier,
            dag_state,
            committer,
            linearizer,
            block_manager,
        }
    }

    // Note: A great deal of care is taken to ensure that all event handlers are non-asynchronous
    // and that the only "await" points are from the select macro picking which event to handle.
    // This ensures that the event loop is able to process events at a high speed and reduce the
    // chance for building up a backlog of events to process.
    pub async fn start(mut self) {
        info!("State-Synchronizer started");

        let mut interval = tokio::time::interval(self.config.interval_period());
        for peer_id in self.active_peers.peers().iter() {
            self.spawn_get_latest_from_peer(*peer_id);
        }

        // Start main loop.
        loop {
            tokio::select! {
                now = interval.tick() => {
                    self.handle_tick(now.into_std());
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
                peer_event = self.peer_event_receiver.recv() => {
                    self.handle_peer_event(peer_event);
                },
                Some(task_result) = self.tasks.join_next() => {
                    match task_result {
                        Ok(()) => {},
                        Err(e) => {
                            if e.is_cancelled() {
                                // avoid crashing on ungraceful shutdown
                            } else if e.is_panic() {
                                // propagate panics.
                                std::panic::resume_unwind(e.into_panic());
                            } else {
                                panic!("task failed: {e}");
                            }
                        },
                    };
                },
            }

            // Schedule new fetches if we're behind
            self.maybe_start_sync_task();
        }

        info!("State-Synchronizer ended");
    }

    fn maybe_start_sync_task(&mut self) {
        let highest_synced_commit = self
            .store
            .get_highest_synced_commit()
            .expect("store operation should not fail")
            .commit_ref
            .index;

        let highest_known_commit = self.peer_heights.read().highest_known_commit_index();

        if Some(highest_synced_commit) < highest_known_commit {
            let task = sync_from_peer(
                self.active_peers.clone(),
                self.store.clone(),
                self.peer_heights.clone(),
                self.commit_event_sender.clone(),
                self.weak_sender.clone(),
                self.block_verifier.clone(),
                self.config.timeout(),
                // The if condition ensures this is Some
                highest_known_commit.unwrap(),
                self.dag_state.clone(),
                self.block_manager.clone(),
                self.committer.clone(),
                self.linearizer.clone(),
            );
            self.tasks.spawn(task);
        }
    }

    fn handle_message(&mut self, message: StateSyncMessage) {
        debug!("Received message: {:?}", message);
        match message {
            StateSyncMessage::StartSyncJob => self.maybe_start_sync_task(),
            StateSyncMessage::VerifiedCommit(commit) => self.handle_commit_from_consensus(commit),
            // After we've successfully synced a commit we can notify our peers
            StateSyncMessage::SyncedCommit(commit) => self.spawn_notify_peers_of_commit(commit),
        }
    }

    // Handle a commit that we received from consensus
    #[instrument(level = "debug", skip_all)]
    fn handle_commit_from_consensus(&mut self, commit: Box<CommittedSubDag>) {
        // TODO: Always check previous_digest matches in case there is a gap between
        // state sync and consensus.
        // let prev_digest = self
        //     .store
        //     .get_commit_by_index(commit.commit_ref.index - 1)
        //     .unwrap_or_else(|| {
        //         panic!(
        //             "Got commit {} from consensus but cannot find commit {} in certified_commits",
        //             commit.commit_ref.index,
        //             commit.commit_ref.index - 1
        //         )
        //     })
        //     .commit_ref
        //     .digest;
        // if commit.commit_ref.previous_digest != prev_digest {
        //     panic!("Commit {} from consensus has mismatched previous_digest, expected: {:?}, actual: {:?}", commit.commit_ref.index, Some(prev_digest), commit.previous_digest);
        // }

        let latest_commit = self
            .store
            .get_highest_synced_commit()
            .expect("store operation should not fail")
            .commit_ref
            .index;

        // If this is an older commit, just ignore it
        if latest_commit >= commit.commit_ref.index {
            return;
        }

        let commit = *commit;
        let next_index = latest_commit.checked_add(1).unwrap();
        if commit.commit_ref.index > next_index {
            debug!(
                "consensus sent too new of a commit, expecting: {}, got: {}",
                next_index, commit.commit_ref.index
            );
        }

        // Because commit from consensus sends in order, when we have commit n,
        // we must have all of the commits before n from either state sync or consensus.
        #[cfg(debug_assertions)]
        {
            let _ = (next_index..=commit.commit_ref.index)
                .map(|n| {
                    let commit = self
                        .store
                        .get_commit_by_index(n)
                        .unwrap_or_else(|| panic!("store should contain commit {n}"));
                })
                .collect::<Vec<_>>();
        }

        // TODO: If this is the last commit of a epoch, we need to make sure
        // new committee is in store before we verify newer commits in next epoch.
        // This could happen before this validator's reconfiguration finishes, because
        // state sync does not reconfig.
        // TODO: maybe we don't need to do this committee insertion in two places (other in StateSyncStore::insert_commit)
        if let Some(Some(EndOfEpochData {
            next_validator_set, ..
        })) = commit
            .get_end_of_epoch_block()
            .map(|b| b.end_of_epoch_data())
        {
            if let Some(next_validator_set) = next_validator_set {
                let voting_rights: BTreeMap<_, _> = next_validator_set
                    .0
                    .iter()
                    .map(|(name, stake, _)| (*name, *stake))
                    .collect();

                let authorities = next_validator_set
                    .0
                    .iter()
                    .map(|(name, stake, meta)| {
                        (
                            *name,
                            Authority {
                                stake: *stake,
                                address: meta.consensus_address.clone(),
                                hostname: meta.hostname.clone(),
                                protocol_key: meta.protocol_key.clone(),
                                network_key: meta.network_key.clone(),
                                authority_key: meta.authority_key.clone(),
                            },
                        )
                    })
                    .collect();
                let committee = Committee::new(
                    commit
                        .blocks
                        .last()
                        .unwrap()
                        .epoch()
                        .checked_add(1)
                        .unwrap(),
                    voting_rights,
                    authorities,
                );
                self.store
                    .insert_committee(committee)
                    .expect("insert committee operation should not fail");
            }
        }

        self.store
            .update_highest_synced_commit(&commit)
            .expect("store operation should not fail");

        // We don't care if no one is listening as this is a broadcast channel
        let _ = self.commit_event_sender.send(commit.clone());

        self.spawn_notify_peers_of_commit(commit.commit_ref.index);
    }

    fn handle_peer_event(
        &mut self,
        peer_event: Result<PeerEvent, tokio::sync::broadcast::error::RecvError>,
    ) {
        use tokio::sync::broadcast::error::RecvError;

        match peer_event {
            Ok(PeerEvent::NewPeer { peer_id, address }) => {
                self.spawn_get_latest_from_peer(peer_id);
            }
            Ok(PeerEvent::LostPeer { peer_id, reason }) => {
                self.peer_heights.write().peers.remove(&peer_id);
            }

            Err(RecvError::Closed) => {
                panic!("PeerEvent channel shouldn't be able to be closed");
            }

            Err(RecvError::Lagged(_)) => {
                trace!("State-Sync fell behind processing PeerEvents");
            }
        }
    }

    fn spawn_get_latest_from_peer(&mut self, peer_id: PeerId) {
        if let Some(peer) = self.active_peers.get_state(&peer_id) {
            let genesis_commit_digest = self
                .store
                .get_commit_by_index(0)
                .expect("store should contain genesis commit")
                .commit_ref
                .digest;
            let task = get_latest_from_peer(
                genesis_commit_digest,
                peer,
                self.peer_heights.clone(),
                self.config.timeout(),
            );
            self.tasks.spawn(task);
        }
    }

    fn handle_tick(&mut self, _now: std::time::Instant) {
        let task = query_peers_for_their_latest_commit(
            self.active_peers.clone(),
            self.peer_heights.clone(),
            self.weak_sender.clone(),
            self.config.timeout(),
        );
        self.tasks.spawn(task);
    }

    fn spawn_notify_peers_of_commit(&mut self, commit: CommitIndex) {
        let task = notify_peers_of_commit(
            self.active_peers.clone(),
            self.peer_heights.clone(),
            commit,
            self.config.timeout(),
        );
        self.tasks.spawn(task);
    }
}

async fn notify_peers_of_commit(
    active_peers: ActivePeers,
    peer_heights: Arc<RwLock<PeerHeights>>,
    commit: CommitIndex,
    timeout: Duration,
) {
    let futs = peer_heights
        .read()
        .peers
        .iter()
        // Filter out any peers who we know already have a commit higher than this one
        .filter_map(|(peer_id, info)| (commit > info.height).then_some(peer_id))
        // Filter out any peers who we aren't connected with
        .flat_map(|peer_id| active_peers.get(peer_id))
        .map(P2pClient::new)
        .map(|mut client| {
            let mut request = Request::new(PushCommitRequest { commit });
            request.set_timeout(timeout);
            async move { client.push_commit(request).await }
        })
        .collect::<Vec<_>>();
    futures::future::join_all(futs).await;
}

async fn get_latest_from_peer(
    our_genesis_commit_digest: CommitDigest,
    peer: PeerState,
    peer_heights: Arc<RwLock<PeerHeights>>,
    timeout: Duration,
) {
    let peer_id = peer.public_key.into();
    let mut client = P2pClient::new(peer.channel);

    let info = {
        let maybe_info = peer_heights.read().peers.get(&peer_id).copied();

        if let Some(info) = maybe_info {
            info
        } else {
            // TODO do we want to create a new API just for querying a node's chainid?
            //
            // We need to query this node's genesis commit to see if they're on the same chain
            // as us
            let mut request = Request::new(GetCommitInfoRequest::ByIndex(0));
            request.set_timeout(timeout);
            let response = client
                .get_commit_info(request)
                .await
                .map(Response::into_inner);
            let response = response.map(|r| r.commit_info);

            let info = match response {
                Ok(Some(commit)) => {
                    let digest = commit.digest;
                    PeerStateSyncInfo {
                        genesis_commit_digest: digest,
                        height: commit.index,
                        lowest: CommitIndex::default(),
                    }
                }
                Ok(None) => PeerStateSyncInfo {
                    genesis_commit_digest: CommitDigest::default(),
                    height: CommitIndex::default(),
                    lowest: CommitIndex::default(),
                },
                Err(status) => {
                    trace!("get_latest_commit_summary request failed: {status:?}");
                    return;
                }
            };
            peer_heights.write().insert_peer_info(peer_id, info);
            info
        }
    };

    let Some((highest_commit, low_watermark)) =
        query_peer_for_latest_info(&mut client, timeout).await
    else {
        return;
    };
    peer_heights
        .write()
        .update_peer_info(peer_id, highest_commit, Some(low_watermark));
}

/// Queries a peer for their highest_synced_commit and low commit watermark
async fn query_peer_for_latest_info(
    client: &mut P2pClient<Channel>,
    timeout: Duration,
) -> Option<(CommitIndex, CommitIndex)> {
    let mut request = Request::new(GetCommitAvailabilityRequest {
        timestamp_ms: now_unix(),
    });
    request.set_timeout(timeout);
    let response = client
        .get_commit_availability(request)
        .await
        .map(Response::into_inner);
    match response {
        Ok(GetCommitAvailabilityResponse {
            highest_synced_commit,
            lowest_available_commit,
        }) => {
            return Some((highest_synced_commit, lowest_available_commit));
        }
        Err(status) => {
            return None;
        }
    };
}

#[instrument(level = "debug", skip_all)]
async fn query_peers_for_their_latest_commit(
    active_peers: ActivePeers,
    peer_heights: Arc<RwLock<PeerHeights>>,
    sender: mpsc::WeakSender<StateSyncMessage>,
    timeout: Duration,
) {
    let peer_heights = &peer_heights;
    let futs = peer_heights
        .read()
        .peers
        .iter()
        // Filter out any peers who we aren't connected with
        .flat_map(|(peer_id, _info)| active_peers.get_state(peer_id))
        .map(|peer| {
            let peer_id = peer.public_key.into();
            let mut client = P2pClient::new(peer.channel);

            async move {
                let response = query_peer_for_latest_info(&mut client, timeout).await;
                match response {
                    Some((highest_commit, low_watermark)) => peer_heights
                        .write()
                        .update_peer_info(peer_id, highest_commit, Some(low_watermark))
                        .then_some(highest_commit),
                    None => None,
                }
            }
        })
        .collect::<Vec<_>>();

    debug!("Query {} peers for latest commit", futs.len());

    let commits = futures::future::join_all(futs).await.into_iter().flatten();

    let highest_commit = commits.max_by_key(|commit| *commit);

    let our_highest_commit = peer_heights.read().highest_known_commit_index();

    debug!(
        "Our highest commit {:?}, peers highest commit {:?}",
        our_highest_commit.as_ref(),
        highest_commit.as_ref()
    );

    let _new_commit = match (highest_commit, our_highest_commit) {
        (Some(theirs), None) => theirs,
        (Some(theirs), Some(ours)) if theirs > ours => theirs,
        _ => return,
    };

    if let Some(sender) = sender.upgrade() {
        let _ = sender.send(StateSyncMessage::StartSyncJob).await;
    }
}

async fn sync_from_peer<S>(
    active_peers: ActivePeers,
    store: S,
    peer_heights: Arc<RwLock<PeerHeights>>,
    commit_event_sender: broadcast::Sender<CommittedSubDag>,
    weak_sender: mpsc::WeakSender<StateSyncMessage>,
    block_verifier: Arc<SignedBlockVerifier>,
    timeout: Duration,
    target_commit: CommitIndex,
    dag_state: Arc<RwLock<DagState>>,
    block_manager: Arc<RwLock<BlockManager>>,
    committer: Arc<UniversalCommitter>,
    commit_interpreter: Arc<RwLock<Linearizer>>,
) where
    S: WriteStore + Clone,
{
    // Keep retrying with different peers until we succeed
    'retry: loop {
        let peer_balancer = PeerBalancer::new(active_peers.clone(), peer_heights.clone());

        let current_highest_synced_commit_index = store
            .get_highest_synced_commit()
            .expect("store operation should not fail")
            .commit_ref
            .index;

        for peer in peer_balancer.clone() {
            // Phase 1: Fetch commits
            let (verified_commits, mut unverified_commits) = match fetch_and_verify_commits(
                &peer,
                current_highest_synced_commit_index..=target_commit,
                &store,
                block_verifier.clone(),
                timeout,
            )
            .await
            {
                Ok(result) => result,
                Err(e) => {
                    warn!("Failed to fetch commits: {}", e);
                    // Try the next peer
                    continue;
                }
            };

            let block_refs: Vec<_> = verified_commits
                .iter()
                .flat_map(|c| c.blocks())
                .cloned()
                .collect();
            // Phase 2: Fetch blocks
            let blocks = match fetch_blocks_batch(
                &peer,
                block_refs,
                block_verifier.clone(),
                timeout,
            )
            .await
            {
                Ok(blocks) => blocks,
                Err(e) => {
                    warn!("Failed to fetch blocks: {}", e);
                    // Try the next peer
                    continue;
                }
            };

            // Phase 3: Process verified commits
            if let Err(e) = process_verified_commits(
                verified_commits,
                blocks,
                &store,
                &dag_state,
                &block_manager,
                &committer,
                &commit_interpreter,
                &commit_event_sender,
                &weak_sender,
            )
            .await
            {
                warn!("Failed to process verified commits: {}", e);
                continue;
            }

            // Phase 4: Try to verify unverified commits in epoch order
            if unverified_commits.has_pending() {
                // Get epochs in order
                let epochs: Vec<_> = unverified_commits.by_epoch.keys().copied().collect();

                // Try to process each epoch in sequence
                for epoch in epochs {
                    // Skip if we don't have the committee yet
                    if let Ok(Some(committee)) = store.get_committee(epoch) {
                        let unverified = unverified_commits.take_epoch(epoch);
                        let mut newly_verified = Vec::new();

                        // First verify all commits in this epoch
                        for commit in unverified {
                            match verify_commit(
                                commit.commit_digest,
                                commit.commit,
                                commit.blocks,
                                commit.serialized_commit,
                                committee.clone(),
                                block_verifier.clone(),
                                peer.public_key.clone(),
                            ) {
                                Ok(verified) => {
                                    newly_verified.push(verified);
                                }
                                Err(e) => {
                                    warn!("Failed to verify commit in epoch {}: {}", epoch, e);
                                    // If verification fails, abort
                                    continue 'retry;
                                }
                            }
                        }

                        if !newly_verified.is_empty() {
                            // Get all block refs for the verified commits
                            let block_refs: Vec<_> = newly_verified
                                .iter()
                                .flat_map(|c| c.blocks())
                                .cloned()
                                .collect();

                            // Fetch blocks for all verified commits in this epoch
                            match fetch_blocks_batch(
                                &peer,
                                block_refs,
                                block_verifier.clone(),
                                timeout,
                            )
                            .await
                            {
                                Ok(blocks) => {
                                    // Process all verified commits and their blocks together
                                    if let Err(e) = process_verified_commits(
                                        newly_verified,
                                        blocks,
                                        &store,
                                        &dag_state,
                                        &block_manager,
                                        &committer,
                                        &commit_interpreter,
                                        &commit_event_sender,
                                        &weak_sender,
                                    )
                                    .await
                                    {
                                        warn!(
                                            "Failed to process verified commits for epoch {}: {}",
                                            epoch, e
                                        );
                                        continue 'retry;
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to fetch blocks for epoch {}: {}", epoch, e);
                                    continue 'retry;
                                }
                            }
                        }
                    } else {
                        // If we don't have the committee for this epoch, abort and try again
                        debug!(
                            "Missing committee for epoch {}, will retry with new peer",
                            epoch
                        );
                        continue 'retry;
                    }
                }
            }

            // If we have no more pending commits, we're done
            if !unverified_commits.has_pending() {
                let highest_processed = store
                    .get_highest_synced_commit()
                    .expect("store operation should not fail")
                    .commit_ref
                    .index;

                if highest_processed < target_commit {
                    debug!(
                        "Fetched commits ended at {} but target was {}",
                        highest_processed, target_commit
                    );
                    // Try the next peer
                    continue;
                }

                break 'retry;
            }
        }

        sleep(Duration::from_secs(1)).await;
    }
}

// Phase 1: Fetch and verify commits
async fn fetch_and_verify_commits<S>(
    peer: &PeerState,
    commit_range: RangeInclusive<CommitIndex>,
    store: &S,
    block_verifier: Arc<SignedBlockVerifier>,
    timeout: Duration,
) -> Result<(Vec<TrustedCommit>, UnverifiedCommits), SomaError>
where
    S: WriteStore,
{
    const FETCH_COMMITS_TIMEOUT: Duration = Duration::from_secs(30);
    const FETCH_BLOCKS_TIMEOUT: Duration = Duration::from_secs(120);

    let mut client = P2pClient::new(peer.channel.clone());
    let public_key = peer.public_key.clone();

    // 1. Fetch commits in the commit range from the selected peer.
    let response = client
        .fetch_commits(FetchCommitsRequest {
            start: *commit_range.start(),
            end: *commit_range.end(),
        })
        .await?;

    let FetchCommitsResponse {
        commits: serialized_commits,
        certifier_blocks: serialized_blocks,
    } = response.into_inner();

    let mut verified_commits = Vec::new();
    let mut unverified_commits = UnverifiedCommits::default();
    let mut prev_digest: Option<(CommitDigest, Commit)> = None;
    let mut commits = Vec::new();
    // First validate commit sequence
    for (idx, serialized) in serialized_commits.iter().enumerate() {
        let commit: Commit =
            bcs::from_bytes(serialized).map_err(ConsensusError::MalformedCommit)?;

        // Validate sequence...
        if idx == 0 && commit.index() != *commit_range.start() {
            return Err(ConsensusError::UnexpectedStartCommit {
                peer: public_key.into_inner().to_string(),
                start: *commit_range.start(),
                commit: Box::new(commit),
            }
            .into());
        }

        let digest = TrustedCommit::compute_digest(serialized);

        if let Some((prev_commit_digest, prev_commit)) = prev_digest {
            if commit.index() != prev_commit.index() + 1
                || commit.previous_digest() != prev_commit_digest
            {
                return Err(ConsensusError::UnexpectedCommitSequence {
                    peer: public_key.into_inner().to_string(),
                    prev_commit: Box::new(prev_commit),
                    curr_commit: Box::new(commit.clone()),
                }
                .into());
            }
        }

        prev_digest = Some((digest, commit.clone()));

        // Do not process more commits past the end index.
        if commit.index() > *commit_range.end() {
            break;
        }
        commits.push((digest, commit, serialized.clone()));
    }

    if commits.is_empty() {
        return Err(ConsensusError::NoCommitReceived {
            peer: public_key.into_inner().to_string(),
        }
        .into());
    }

    // Group commits by epoch and collect relevant blocks.
    let mut commits_by_epoch: BTreeMap<Epoch, Vec<(CommitDigest, &Commit, Bytes)>> =
        BTreeMap::new();
    for (digest, commit, serialized) in &commits {
        commits_by_epoch.entry(commit.epoch()).or_default().push((
            *digest,
            commit,
            serialized.clone(),
        ));
    }

    // Parse and verify blocks.
    let blocks_by_commit_index: BTreeMap<CommitIndex, Vec<SignedBlock>> = serialized_blocks
        .iter()
        .filter_map(|serialized_block| {
            let block = bcs::from_bytes::<SignedBlock>(serialized_block)
                .map_err(|err| ConsensusError::MalformedBlock)
                .ok()?;

            Some(block)
        })
        .flat_map(|signed_block| {
            let votes = signed_block
                .commit_votes()
                .iter()
                .map(|vote| vote.index)
                .collect::<Vec<_>>();

            votes
                .into_iter()
                .map(move |index| (index, signed_block.clone()))
                .collect::<Vec<_>>()
        })
        .fold(BTreeMap::new(), |mut acc, (index, signed_block)| {
            acc.entry(index).or_default().push(signed_block);
            acc
        });

    // 3. Verify commits for each epoch.
    for (epoch, epoch_commits) in commits_by_epoch {
        // Try to verify commit with available committee
        match store.get_committee(epoch)? {
            Some(committee) => {
                for (commit_digest, commit, serialized_commit) in epoch_commits {
                    // Verify committee matches commit
                    let verified = verify_commit(
                        commit_digest,
                        commit.clone(),
                        blocks_by_commit_index
                            .get(&commit.index())
                            .cloned()
                            .unwrap_or_default(),
                        serialized_commit.clone(),
                        committee.clone(),
                        block_verifier.clone(),
                        public_key.clone(),
                    )?;

                    verified_commits.push(verified);
                }
            }
            None => {
                // Track any commits that couldn't be verified due to missing committees
                for commit in epoch_commits {
                    if let Some(blocks) = blocks_by_commit_index.get(&commit.1.index()).cloned() {
                        unverified_commits.add(
                            epoch,
                            UnverifiedCommit {
                                commit_digest: commit.0,
                                commit: commit.1.clone(),
                                blocks,
                                serialized_commit: commit.2.clone(),
                            },
                        );
                    } else {
                        return Err(ConsensusError::NoBlocksForCommit {
                            commit: Box::new(commit.1.clone()),
                            peer: public_key.into_inner().to_string(),
                        }
                        .into());
                    }
                }
            }
        }
    }

    Ok((verified_commits, unverified_commits))
}

fn verify_commit(
    digest: CommitDigest,
    commit: Commit,
    blocks: Vec<SignedBlock>,
    serialized_commit: Bytes,
    committee: Arc<Committee>,
    block_verifier: Arc<SignedBlockVerifier>,
    peer: NetworkPublicKey,
) -> Result<TrustedCommit, SomaError> {
    let commit_ref = CommitRef {
        index: commit.index(),
        digest,
    };

    let mut stake_aggregator = StakeAggregator::<QuorumThreshold>::new();
    for block in blocks {
        block_verifier.verify(&block)?;
        for vote in block.commit_votes() {
            if *vote == commit_ref {
                stake_aggregator.add(block.author(), &committee);
            }
        }
    }

    // Check if the commit has enough votes.
    if !stake_aggregator.reached_threshold(&committee) {
        return Err(ConsensusError::NotEnoughCommitVotes {
            stake: stake_aggregator.stake(),
            peer: peer.into_inner().to_string(),
            commit: Box::new(commit.clone()),
        }
        .into());
    }

    Ok(TrustedCommit::new_trusted(commit, serialized_commit))
}

async fn fetch_blocks_batch(
    peer: &PeerState,
    block_refs: Vec<BlockRef>,
    block_verifier: Arc<SignedBlockVerifier>,
    timeout: Duration,
) -> Result<Vec<VerifiedBlock>, SomaError> {
    const MAX_BLOCKS_PER_FETCH: usize = if cfg!(msim) {
        // Exercise hitting blocks per fetch limit.
        10
    } else {
        1000
    };

    let mut requests: FuturesOrdered<_> = block_refs
        .chunks(MAX_BLOCKS_PER_FETCH)
        .enumerate()
        .map(|(i, request_block_refs)| {
            let i = i as u32;
            let mut client = P2pClient::new(peer.channel.clone());
            let public_key = peer.public_key.clone();
            let block_verifier = block_verifier.clone();
            async move {
                // Pipeline the requests to avoid overloading the target.
                sleep(Duration::from_millis(200) * i).await;
                // TODO: add some retries.
                let mut stream = client
                    .fetch_blocks(FetchBlocksRequest {
                        block_refs: request_block_refs
                            .iter()
                            .filter_map(|r| match bcs::to_bytes(r) {
                                Ok(serialized) => Some(serialized),
                                Err(e) => {
                                    debug!("Failed to serialize block ref {:?}: {e:?}", r);
                                    None
                                }
                            })
                            .collect(),
                        highest_accepted_rounds: vec![],
                        epoch: 0,
                    })
                    .await.map_err(|e| ConsensusError::NetworkRequest(format!("Network error while streaming blocks")))?
                    .into_inner();

                let mut chunk_serialized_blocks = vec![];
                let mut total_fetched_bytes = 0;
                loop {
                    match stream.message().await {
                        Ok(Some(response)) => {
                            for b in &response.blocks {
                                total_fetched_bytes += b.len();
                            }
                            chunk_serialized_blocks.extend(response.blocks);
                            if total_fetched_bytes > MAX_TOTAL_FETCHED_BYTES {
                                info!(
                                    "fetch_blocks() fetched bytes exceeded limit: {} > {}, terminating stream.",
                                    total_fetched_bytes, MAX_TOTAL_FETCHED_BYTES,
                                );
                                break;
                            }
                        }
                        Ok(None) => {
                            break;
                        }
                        Err(e) => {
                            if chunk_serialized_blocks.is_empty() {
                                return Err(ConsensusError::NetworkRequest(format!(
                                    "fetch_blocks failed mid-stream: {e:?}"
                                )));
                            } else {
                                warn!("fetch_blocks failed mid-stream: {e:?}");
                                break;
                            }
                        }
                    }
                }

                // 4. Verify the same number of blocks are returned as requested.
                if request_block_refs.len() != chunk_serialized_blocks.len() {
                    return Err(ConsensusError::UnexpectedNumberOfBlocksFetched {
                        peer: public_key.into_inner().to_string(),
                        requested: request_block_refs.len(),
                        received: chunk_serialized_blocks.len(),
                    }.into());
                }

                let mut verified_blocks = Vec::new();
                for (requested_block_ref, serialized) in
                    request_block_refs.iter().zip(chunk_serialized_blocks.into_iter())
                {
                    let signed_block: SignedBlock =
                        bcs::from_bytes(&serialized).map_err(ConsensusError::MalformedBlock)?;
                    block_verifier.verify(&signed_block)?;

                    let signed_block_digest = VerifiedBlock::compute_digest(&serialized);
                    let received_block_ref =
                        BlockRef::new(signed_block.round(), signed_block.author(), signed_block_digest);

                    if *requested_block_ref != received_block_ref {
                        return Err(ConsensusError::UnexpectedBlockForCommit {
                            peer: public_key.into_inner().to_string(),
                            requested: *requested_block_ref,
                            received: received_block_ref,
                        }.into());
                    }

                    verified_blocks.push(VerifiedBlock::new_verified(signed_block, serialized));
                }

                Ok(verified_blocks)
            }
        })
        .collect();

    let mut fetched_blocks: Vec<VerifiedBlock> = Vec::new();
    while let Some(result) = requests.next().await {
        fetched_blocks.extend(result?);
    }

    Ok(fetched_blocks)
}

async fn process_verified_commits(
    verified_commits: Vec<TrustedCommit>,
    blocks: Vec<VerifiedBlock>,
    store: &impl WriteStore,
    dag_state: &RwLock<DagState>,
    block_manager: &RwLock<BlockManager>,
    committer: &UniversalCommitter,
    commit_interpreter: &RwLock<Linearizer>,
    commit_event_sender: &broadcast::Sender<CommittedSubDag>,
    weak_sender: &mpsc::WeakSender<StateSyncMessage>,
) -> Result<(), SomaError> {
    assert!(!verified_commits.is_empty());

    let commit_end = verified_commits.last().unwrap().index();

    // Group blocks by epoch
    let mut blocks_by_epoch: BTreeMap<EpochId, Vec<VerifiedBlock>> = BTreeMap::new();
    for block in blocks {
        blocks_by_epoch
            .entry(block.epoch())
            .or_default()
            .push(block);
    }

    // Process blocks by epoch
    for (epoch, epoch_blocks) in blocks_by_epoch {
        // Add blocks to block manager
        let (accepted_blocks, missing_blocks) =
            block_manager.write().try_accept_blocks(epoch_blocks);

        if !accepted_blocks.is_empty() {
            debug!(
                "Accepted blocks: {}",
                accepted_blocks
                    .iter()
                    .map(|b| b.reference().to_string())
                    .join(",")
            );

            // Try to decide on commits
            let decided_leaders =
                committer.try_decide(dag_state.read().last_commit_leader(), Some(epoch));

            let committed_leaders = decided_leaders
                .into_iter()
                .filter_map(|leader| leader.into_committed_block())
                .collect::<Vec<_>>();

            if !committed_leaders.is_empty() {
                debug!(
                    "Committing leaders: {}",
                    committed_leaders
                        .iter()
                        .map(|b| b.reference().to_string())
                        .join(",")
                );

                // Handle commits and update state
                let committed_sub_dags =
                    commit_interpreter.write().handle_commit(committed_leaders);

                for committed_sub_dag in committed_sub_dags {
                    // Send commit event
                    if let Err(err) = commit_event_sender.send(committed_sub_dag.clone()) {
                        tracing::error!(
                            "Failed to send committed sub-dag, probably due to shutdown: {err:?}"
                        );
                    }

                    tracing::debug!(
                        "Sending to execution commit {} leader {}",
                        committed_sub_dag.commit_ref,
                        committed_sub_dag.leader
                    );

                    // Update store
                    store.insert_commit(committed_sub_dag)?;
                }
            }
        }

        if !missing_blocks.is_empty() {
            debug!("Missing blocks: {:?}", missing_blocks);
            // TODO: Trigger fetching of missing blocks
        }
    }

    // Notify about synced commit
    if let Some(sender) = weak_sender.upgrade() {
        let _ = sender
            .send(StateSyncMessage::SyncedCommit(commit_end))
            .await;
    }

    Ok(())
}

struct UnverifiedCommit {
    blocks: Vec<SignedBlock>,
    commit: Commit,
    commit_digest: CommitDigest,
    serialized_commit: Bytes,
}

#[derive(Default)]
struct UnverifiedCommits {
    // Map from epoch to pending commits for that epoch
    by_epoch: BTreeMap<EpochId, Vec<UnverifiedCommit>>,
}

impl UnverifiedCommits {
    fn add(&mut self, epoch: EpochId, pending: UnverifiedCommit) {
        self.by_epoch.entry(epoch).or_default().push(pending);
    }

    fn take_epoch(&mut self, epoch: EpochId) -> Vec<UnverifiedCommit> {
        self.by_epoch.remove(&epoch).unwrap_or_default()
    }

    fn has_pending(&self) -> bool {
        !self.by_epoch.is_empty()
    }
}
