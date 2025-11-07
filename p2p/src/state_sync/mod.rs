use bytes::Bytes;
use data_ingestion::{executor::setup_single_workflow_with_options, reader::ReaderOptions};
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
    task::{AbortHandle, JoinSet},
    time::sleep,
};
use tonic::{transport::Channel, Request, Response};
use tracing::{debug, info, instrument, trace, warn};
use types::{
    accumulator::CommitIndex,
    committee::{Authority, Committee, Epoch, EpochId},
    config::{node_config::ArchiveReaderConfig, state_sync_config::StateSyncConfig},
    consensus::{
        block::{BlockAPI, BlockRef, EndOfEpochData, SignedBlock, VerifiedBlock},
        block_verifier::{BlockVerifier, SignedBlockVerifier},
        commit::{Commit, CommitAPI, CommitDigest, CommitRef, CommittedSubDag, TrustedCommit},
        stake_aggregator::{QuorumThreshold, StakeAggregator},
    },
    crypto::NetworkPublicKey,
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
    storage::{
        consensus::{ConsensusStore, WriteBatch},
        write_store::WriteStore,
    },
};

use crate::{
    discovery::now_unix,
    state_sync::worker::StateSyncWorker,
    tonic_gen::{p2p_client::P2pClient, p2p_server::P2p},
};

pub mod tx_verifier;
mod worker;

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
    sync_task: Option<AbortHandle>,

    store: S,
    peer_heights: Arc<RwLock<PeerHeights>>,
    commit_event_sender: broadcast::Sender<CommittedSubDag>,

    active_peers: ActivePeers,
    peer_event_receiver: broadcast::Receiver<PeerEvent>,
    block_verifier: Arc<SignedBlockVerifier>,

    archive_config: Option<ArchiveReaderConfig>,
}

impl<S> StateSyncEventLoop<S>
where
    S: ConsensusStore + WriteStore + Clone + Send + Sync + 'static,
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
        archive_config: Option<ArchiveReaderConfig>,
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
            sync_task: None,
            archive_config,
        }
    }

    // Note: A great deal of care is taken to ensure that all event handlers are non-asynchronous
    // and that the only "await" points are from the select macro picking which event to handle.
    // This ensures that the event loop is able to process events at a high speed and reduce the
    // chance for building up a backlog of events to process.
    pub async fn start(mut self) {
        info!("State-Synchronizer started");

        let mut interval = tokio::time::interval(Duration::from_millis(100));
        //TODO: self.config.interval_period()
        for peer_id in self.active_peers.peers().iter() {
            self.spawn_get_latest_from_peer(*peer_id);
        }

        let archive_task = sync_commits_from_archive(
            self.archive_config.clone(),
            self.store.clone(),
            self.peer_heights.clone(),
            self.block_verifier.clone(),
        );
        self.tasks.spawn(archive_task);

        // Start main loop.
        loop {
            tokio::select! {
                now = interval.tick() => {
                    self.handle_tick(now.into_std());
                },
                maybe_message = self.mailbox.recv() => {
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

                    if matches!(&self.sync_task, Some(t) if t.is_finished()) {
                        self.sync_task = None;
                    }
                },
            }

            // Schedule new fetches if we're behind
            self.maybe_start_sync_task();
        }

        info!("State-Synchronizer ended");
    }

    fn maybe_start_sync_task(&mut self) {
        // Only run one sync task at a time
        if self.sync_task.is_some() {
            return;
        }

        let highest_synced_commit = self
            .store
            .get_highest_synced_commit()
            .expect("store operation should not fail")
            .commit_ref
            .index;

        let highest_known_commit = self.peer_heights.read().highest_known_commit_index();

        info!(
            "Highest synced commit: {:?}, highest known commit: {:?}",
            highest_synced_commit, highest_known_commit
        );

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
            );
            let task_handle = self.tasks.spawn(task);
            self.sync_task = Some(task_handle);
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
        // Always check previous_digest matches in case there is a gap between
        // state sync and consensus.
        let prev_digest = self
            .store
            .get_commit_by_index(commit.commit_ref.index - 1)
            .unwrap_or_else(|| {
                panic!(
                    "Got commit {} from consensus but cannot find commit {} in certified_commits",
                    commit.commit_ref.index,
                    commit.commit_ref.index - 1
                )
            })
            .commit_ref
            .digest;
        if commit.previous_digest != prev_digest {
            panic!(
                "Commit {} from consensus has mismatched previous_digest, expected: {:?}, actual: \
                 {:?}",
                commit.commit_ref.index,
                Some(prev_digest),
                commit.previous_digest
            );
        }

        let latest_commit = self
            .store
            .get_highest_synced_commit()
            .expect("store operation should not fail")
            .commit_ref
            .index;

        // If this is an older commit, just ignore it
        if latest_commit >= commit.commit_ref.index {
            info!("Commit index is less than or equal to than synced commit, ignored");
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
        // #[cfg(debug_assertions)]
        // {
        //     let _ = (next_index..=commit.commit_ref.index)
        //         .map(|n| {
        //             let commit = self
        //                 .store
        //                 .get_commit_by_index(n)
        //                 .unwrap_or_else(|| panic!("store should contain commit {n}"));
        //         })
        //         .collect::<Vec<_>>();
        // }

        // Insert the commit to store
        self.store
            .insert_commit(commit.clone())
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

async fn sync_commits_from_archive<S>(
    archive_config: Option<ArchiveReaderConfig>,
    store: S,
    peer_heights: Arc<RwLock<PeerHeights>>,
    block_verifier: Arc<SignedBlockVerifier>,
) where
    S: ConsensusStore + WriteStore + Clone + Send + Sync + 'static,
{
    loop {
        sync_commits_from_archive_iteration(
            &archive_config,
            store.clone(),
            peer_heights.clone(),
            block_verifier.clone(),
        )
        .await;
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

async fn sync_commits_from_archive_iteration<S>(
    archive_config: &Option<ArchiveReaderConfig>,
    store: S,
    peer_heights: Arc<RwLock<PeerHeights>>,
    block_verifier: Arc<SignedBlockVerifier>,
) where
    S: ConsensusStore + WriteStore + Clone + Send + Sync + 'static,
{
    // Check if we need to sync from archive
    let highest_synced = store
        .get_highest_synced_commit()
        .expect("store operation should not fail")
        .commit_ref
        .index;

    let lowest_peer_commit = peer_heights
        .read()
        .peers
        .values()
        .map(|info| info.lowest)
        .min();

    let sync_from_archive = if let Some(lowest) = lowest_peer_commit {
        highest_synced < lowest
    } else {
        false
    };

    debug!(
        "Archive sync check: highest_synced={}, lowest_peer_commit={:?}, sync_needed={}",
        highest_synced, lowest_peer_commit, sync_from_archive
    );

    if !sync_from_archive {
        return;
    }

    let Some(archive_config) = archive_config else {
        warn!("Archive sync needed but no archive config provided");
        return;
    };

    let Some(ingestion_url) = &archive_config.ingestion_url else {
        warn!("Archive ingestion URL not configured");
        return;
    };

    let start = highest_synced
        .checked_add(1)
        .expect("Commit index overflow");
    let end = lowest_peer_commit.unwrap();

    info!("Starting archive sync for commits {} to {}", start, end);

    // Setup worker and executor
    let worker = StateSyncWorker::new(store, block_verifier);

    let reader_options = ReaderOptions {
        batch_size: archive_config.download_concurrency.into(),
        upper_limit: Some(end),
        ..Default::default()
    };

    match setup_single_workflow_with_options(
        worker,
        ingestion_url.clone(),
        archive_config.remote_store_options.clone(),
        start,
        1, // Single worker for ordered processing
        Some(reader_options),
    )
    .await
    {
        Ok((executor, _exit_sender)) => match executor.await {
            Ok(_) => {
                info!("Archive sync complete. Commits synced: {}", end - start + 1);
            }
            Err(err) => {
                warn!("Archive sync failed: {:?}", err);
            }
        },
        Err(err) => {
            warn!("Failed to setup archive sync: {:?}", err);
        }
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
) where
    S: ConsensusStore + WriteStore + Clone,
{
    // Keep retrying with different peers until we succeed
    'retry: loop {
        let peer_balancer = PeerBalancer::new(active_peers.clone(), peer_heights.clone())
            .with_commit(target_commit);

        'peer: for peer in peer_balancer.clone() {
            let mut sync_start = store
                .get_highest_synced_commit()
                .expect("store operation should not fail")
                .commit_ref
                .index
                .checked_add(1)
                .expect("commit index should not overflow");

            // Keep trying with the same peer until we hit a fatal error
            while sync_start <= target_commit {
                info!(
                    "Syncing from peer: {:?} for commits {} to {}",
                    peer.public_key, sync_start, target_commit
                );

                // Phase 1: Fetch and verify commits
                let verified_commits = match fetch_and_verify_commits(
                    &peer,
                    sync_start..=target_commit,
                    &store,
                    block_verifier.clone(),
                    timeout,
                )
                .await
                {
                    Ok(commits) => commits,
                    Err(e) => match e {
                        SomaError::NoCommitteeForEpoch(epoch) => {
                            // If we don't have the committee, wait for it to arrive
                            warn!(
                                "Missing committee for epoch {}, waiting for sync to catch up",
                                epoch
                            );
                            sleep(Duration::from_secs(1)).await;
                            continue;
                        }
                        _ => {
                            warn!("Failed to fetch commits: {}", e);
                            continue 'peer;
                        }
                    },
                };

                if verified_commits.is_empty() {
                    warn!(
                        "No verified commits received from peer: {:?}",
                        peer.public_key
                    );
                    continue 'peer;
                }

                // Phase 2: Fetch blocks for verified commits
                let block_refs: Vec<_> = verified_commits
                    .iter()
                    .flat_map(|c| c.blocks())
                    .cloned()
                    .collect();

                let blocks =
                    match fetch_blocks_batch(&peer, block_refs, block_verifier.clone(), timeout)
                        .await
                    {
                        Ok(blocks) => blocks,
                        Err(e) => {
                            warn!("Failed to fetch blocks: {}", e);
                            continue 'peer;
                        }
                    };

                info!(
                    "Fetched {} blocks from peer: {:?}",
                    blocks.len(),
                    peer.public_key,
                );

                // Phase 3: Process verified commits
                match process_verified_commits(
                    verified_commits,
                    blocks,
                    &store,
                    &commit_event_sender,
                    &weak_sender,
                )
                .await
                {
                    Ok(highest_processed) => {
                        info!(
                            "Successfully processed verified commits up to {}",
                            highest_processed
                        );

                        break 'retry;

                        // TODO: if highest_processed >= target_commit {
                        //     break 'retry;
                        // }

                        // // Update sync_start for next iteration with same peer
                        // sync_start = highest_processed
                        //     .checked_add(1)
                        //     .expect("commit index overflow");
                    }
                    Err(e) => {
                        warn!("Failed to process verified commits: {}", e);
                        continue 'peer;
                    }
                }
            }

            break 'retry;
        }

        // Retry timeout
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
) -> Result<Vec<TrustedCommit>, SomaError>
where
    S: WriteStore,
{
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

    // 2. Parse and validate commit sequence
    let mut commits = Vec::new();
    let mut prev_digest: Option<(CommitDigest, Commit)> = None;

    for (idx, serialized) in serialized_commits.iter().enumerate() {
        let commit: Commit =
            bcs::from_bytes(serialized).map_err(ConsensusError::MalformedCommit)?;

        // Validate first commit starts at requested index
        if idx == 0 && commit.index() != *commit_range.start() {
            return Err(ConsensusError::UnexpectedStartCommit {
                peer: public_key.into_inner().to_string(),
                start: *commit_range.start(),
                commit: Box::new(commit),
            }
            .into());
        }

        let digest = TrustedCommit::compute_digest(serialized);

        // Validate sequence
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

        // Don't process commits past the end
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

    // 3. Get the last commit and verify it has enough votes
    let (last_digest, last_commit, _) = commits.last().unwrap();

    // Get committee for the last commit
    let committee = store
        .get_committee(last_commit.epoch())?
        .ok_or_else(|| SomaError::NoCommitteeForEpoch(last_commit.epoch()))?;

    let last_commit_ref = CommitRef {
        index: last_commit.index(),
        digest: *last_digest,
    };

    // Parse blocks and accumulate votes for the last commit
    let mut stake_aggregator = StakeAggregator::<QuorumThreshold>::new();

    for block_bytes in &serialized_blocks {
        let block: SignedBlock =
            bcs::from_bytes(block_bytes).map_err(ConsensusError::MalformedBlock)?;

        block_verifier.verify(&block)?;

        for vote in block.commit_votes() {
            if *vote == last_commit_ref {
                stake_aggregator.add(block.author(), &committee);
            }
        }
    }

    // Verify the last commit has enough votes
    if !stake_aggregator.reached_threshold(&committee) {
        return Err(ConsensusError::NotEnoughCommitVotes {
            stake: stake_aggregator.stake(),
            peer: public_key.into_inner().to_string(),
            commit: Box::new(last_commit.clone()),
        }
        .into());
    }

    // 4. Convert all commits to TrustedCommit
    Ok(commits
        .into_iter()
        .map(|(_, commit, serialized)| TrustedCommit::new_trusted(commit, serialized))
        .collect())
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
                    })
                    .await
                    .map_err(|e| {
                        ConsensusError::NetworkRequest(format!(
                            "Network error while streaming blocks {}",
                            e
                        ))
                    })?
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
                                    "fetch_blocks() fetched bytes exceeded limit: {} > {}, \
                                     terminating stream.",
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
                    }
                    .into());
                }

                let mut verified_blocks = Vec::new();
                for (requested_block_ref, serialized) in request_block_refs
                    .iter()
                    .zip(chunk_serialized_blocks.into_iter())
                {
                    let signed_block: SignedBlock =
                        bcs::from_bytes(&serialized).map_err(ConsensusError::MalformedBlock)?;
                    block_verifier.verify(&signed_block)?;

                    let signed_block_digest = VerifiedBlock::compute_digest(&serialized);
                    let received_block_ref = BlockRef::new(
                        signed_block.round(),
                        signed_block.author(),
                        signed_block_digest,
                        signed_block.epoch(),
                    );

                    if *requested_block_ref != received_block_ref {
                        return Err(ConsensusError::UnexpectedBlockForCommit {
                            peer: public_key.into_inner().to_string(),
                            requested: *requested_block_ref,
                            received: received_block_ref,
                        }
                        .into());
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

async fn process_verified_commits<S>(
    verified_commits: Vec<TrustedCommit>,
    blocks: Vec<VerifiedBlock>,
    store: &S,
    commit_event_sender: &broadcast::Sender<CommittedSubDag>,
    weak_sender: &mpsc::WeakSender<StateSyncMessage>,
) -> Result<CommitIndex, SomaError>
where
    S: ConsensusStore + WriteStore,
{
    assert!(!verified_commits.is_empty());

    info!(
        "Processing verified commits: {}",
        verified_commits.iter().map(|c| c.index()).join(",")
    );

    let commit_end = verified_commits.last().unwrap().index();

    // First write blocks and commits to ConsensusStore
    store
        .write(WriteBatch::new(
            blocks.clone(),
            verified_commits.clone(),
            vec![],
        ))
        .unwrap_or_else(|e| panic!("Failed to write to storage: {:?}", e));

    for commit in verified_commits {
        let to_commit = commit
            .blocks()
            .iter()
            .map(|block_ref| {
                blocks
                    .iter()
                    .find(|b| b.reference() == *block_ref)
                    .unwrap_or_else(|| {
                        panic!(
                            "Failed to find block {:?} in blocks for commit {:?}",
                            block_ref, commit
                        )
                    })
            })
            .cloned()
            .collect::<Vec<_>>();

        let sub_dag = CommittedSubDag::new(
            commit.leader(),
            to_commit,
            commit.timestamp_ms(),
            commit.reference(),
            commit.previous_digest(),
        );

        if let Err(err) = commit_event_sender.send(sub_dag.clone()) {
            tracing::error!("Failed to send committed sub-dag, probably due to shutdown: {err:?}");
        }

        tracing::debug!(
            "Sending to execution commit {} leader {}",
            sub_dag.commit_ref,
            sub_dag.leader
        );

        // Update store
        store.insert_commit(sub_dag)?;
    }

    // Notify about synced commit
    if let Some(sender) = weak_sender.upgrade() {
        let _ = sender
            .send(StateSyncMessage::SyncedCommit(commit_end))
            .await;
    }

    Ok(commit_end)
}
