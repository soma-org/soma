use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};

use futures::{stream::FuturesOrdered, FutureExt, StreamExt};
use parking_lot::RwLock;
use rand::Rng;
use tap::{Pipe, TapFallible, TapOptional};
use tokio::{
    sync::{broadcast, mpsc},
    task::{watch, AbortHandle, JoinSet},
};
use tonic::{transport::Channel, Request, Response};
use tracing::{debug, info, instrument, trace};
use types::{
    accumulator::CommitIndex,
    committee::Committee,
    config::state_sync_config::StateSyncConfig,
    digests::CommitSummaryDigest,
    envelope::Message,
    error::{SomaError, SomaResult},
    p2p::{
        active_peers::{self, ActivePeers, PeerState},
        PeerEvent,
    },
    peer_id::{self, PeerId},
    state_sync::{
        CertifiedCommitSummary, FullCommitContents, GetCommitAvailabilityRequest,
        GetCommitAvailabilityResponse, GetCommitSummaryRequest, VerifiedCommitContents,
        VerifiedCommitSummary,
    },
    storage::write_store::WriteStore,
};

use crate::{
    discovery::now_unix,
    tonic_gen::{p2p_client::P2pClient, p2p_server::P2p},
};

const COMMIT_SUMMARY_DOWNLOAD_CONCURRENCY: usize = 400;

pub struct PeerHeights {
    /// Table used to track the highest commit for each of our peers.
    peers: HashMap<PeerId, PeerStateSyncInfo>,
    unprocessed_commits: HashMap<CommitSummaryDigest, CertifiedCommitSummary>,
    index_to_digest: HashMap<CommitIndex, CommitSummaryDigest>,

    // The amount of time to wait before retry if there are no peers to sync content from.
    wait_interval_when_no_peer_to_sync_content: Duration,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PeerStateSyncInfo {
    /// The digest of the Peer's genesis commit.
    genesis_commit_digest: CommitSummaryDigest,
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
            unprocessed_commits: HashMap::new(),
            index_to_digest: HashMap::new(),
            wait_interval_when_no_peer_to_sync_content,
        }
    }

    pub fn highest_known_commit(&self) -> Option<&CertifiedCommitSummary> {
        self.highest_known_commit_index()
            .and_then(|s| self.index_to_digest.get(&s))
            .and_then(|digest| self.unprocessed_commits.get(digest))
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
    #[instrument(level = "debug", skip_all, fields(peer_id=?peer_id, commit=?commit.index()))]
    pub fn update_peer_info(
        &mut self,
        peer_id: PeerId,
        commit: CertifiedCommitSummary,
        low_watermark: Option<CommitIndex>,
    ) -> bool {
        debug!("Update peer info");

        let info = match self.peers.get_mut(&peer_id) {
            Some(info) => info,
            _ => return false,
        };

        info.height = std::cmp::max(*commit.index(), info.height);
        if let Some(low_watermark) = low_watermark {
            info.lowest = low_watermark;
        }
        self.insert_commit(commit);

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

    pub fn cleanup_old_commits(&mut self, index: CommitIndex) {
        self.unprocessed_commits
            .retain(|_digest: &_, commit: &mut _| *commit.index() > index);
        self.index_to_digest.retain(|&i, _digest| i > index);
    }

    // TODO: also record who gives this commit info for peer quality measurement?
    pub fn insert_commit(&mut self, commit: CertifiedCommitSummary) {
        let digest = commit.digest();
        let index = *commit.index();
        self.unprocessed_commits.insert(*digest, commit.clone());
        self.index_to_digest.insert(index, *digest);
    }

    pub fn remove_commit(&mut self, digest: &CommitSummaryDigest) {
        if let Some(commit) = self.unprocessed_commits.remove(digest) {
            self.index_to_digest.remove(commit.index());
        }
    }

    pub fn get_commit_by_index(&self, index: CommitIndex) -> Option<&CertifiedCommitSummary> {
        self.index_to_digest
            .get(&index)
            .and_then(|digest| self.get_commit_by_digest(digest))
    }

    pub fn get_commit_by_digest(
        &self,
        digest: &CommitSummaryDigest,
    ) -> Option<&CertifiedCommitSummary> {
        self.unprocessed_commits.get(digest)
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
    VerifiedCommit(Box<VerifiedCommitSummary>),
    // Notification that the commit content sync task will send to the event loop in the event
    // it was able to successfully sync a commit's contents. If multiple commits were
    // synced at the same time, only the highest commit is sent.
    SyncedCommit(Box<VerifiedCommitSummary>),
}

// PeerBalancer is an Iterator that selects peers based on RTT with some added randomness.
#[derive(Clone)]
struct PeerBalancer {
    peers: VecDeque<(PeerState, PeerStateSyncInfo)>,
    requested_commit: Option<CommitIndex>,
    request_type: PeerCommitRequestType,
}

#[derive(Clone)]
enum PeerCommitRequestType {
    Summary,
    Content,
}

impl PeerBalancer {
    pub fn new(
        active_peers: ActivePeers,
        peer_heights: Arc<RwLock<PeerHeights>>,
        request_type: PeerCommitRequestType,
    ) -> Self {
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
            request_type,
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
            match &self.request_type {
                // Summary will never be pruned
                PeerCommitRequestType::Summary if info.height >= requested_commit => {
                    return Some(peer);
                }
                PeerCommitRequestType::Content
                    if info.height >= requested_commit && info.lowest <= requested_commit =>
                {
                    return Some(peer);
                }
                _ => {}
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
    sync_commit_summaries_task: Option<AbortHandle>,
    sync_commit_contents_task: Option<AbortHandle>,

    store: S,
    peer_heights: Arc<RwLock<PeerHeights>>,
    commit_event_sender: broadcast::Sender<VerifiedCommitSummary>,

    active_peers: ActivePeers,
    peer_event_receiver: broadcast::Receiver<PeerEvent>,
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
        commit_event_sender: broadcast::Sender<VerifiedCommitSummary>,
        active_peers: ActivePeers,
        peer_event_receiver: broadcast::Receiver<PeerEvent>,
    ) -> Self {
        Self {
            config,
            mailbox,
            weak_sender,
            tasks: JoinSet::new(),
            sync_commit_summaries_task: None,
            sync_commit_contents_task: None,
            store,
            peer_heights,
            commit_event_sender,
            active_peers,
            peer_event_receiver,
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

        let (target_commit_contents_index_sender, target_commit_contents_index_receiver) =
            watch::channel(0);
        // Start commit contents sync loop.
        let task = sync_commit_contents(
            self.active_peers.clone(),
            self.store.clone(),
            self.peer_heights.clone(),
            self.weak_sender.clone(),
            self.commit_event_sender.clone(),
            self.config
                .commit_content_download_concurrency()
                .try_into()
                .unwrap(),
            self.config.commit_content_download_concurrency(),
            self.config.commit_content_timeout(),
            target_commit_contents_index_receiver,
        );
        let task_handle = self.tasks.spawn(task);
        self.sync_commit_contents_task = Some(task_handle);

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

                    if matches!(&self.sync_commit_contents_task, Some(t) if t.is_finished()) {
                        panic!("sync_commit_contents task unexpectedly terminated")
                    }

                    if matches!(&self.sync_commit_summaries_task, Some(t) if t.is_finished()) {
                        self.sync_commit_summaries_task = None;
                    }


                },
            }

            self.maybe_start_commit_summary_sync_task();
            self.maybe_trigger_commit_contents_sync_task(&target_commit_contents_index_sender);
        }

        info!("State-Synchronizer ended");
    }

    fn handle_message(&mut self, message: StateSyncMessage) {
        debug!("Received message: {:?}", message);
        match message {
            StateSyncMessage::StartSyncJob => self.maybe_start_commit_summary_sync_task(),
            StateSyncMessage::VerifiedCommit(commit) => self.handle_commit_from_consensus(commit),
            // After we've successfully synced a commit we can notify our peers
            StateSyncMessage::SyncedCommit(commit) => self.spawn_notify_peers_of_commit(*commit),
        }
    }

    // Handle a commit that we received from consensus
    #[instrument(level = "debug", skip_all)]
    fn handle_commit_from_consensus(&mut self, commit: Box<VerifiedCommitSummary>) {
        // Always check previous_digest matches in case there is a gap between
        // state sync and consensus.
        let prev_digest = *self
            .store
            .get_commit_by_index(commit.index() - 1)
            .unwrap_or_else(|| {
                panic!(
                    "Got commit {} from consensus but cannot find commit {} in certified_commits",
                    commit.index(),
                    commit.index() - 1
                )
            })
            .digest();
        if commit.previous_digest != Some(prev_digest) {
            panic!("Commit {} from consensus has mismatched previous_digest, expected: {:?}, actual: {:?}", commit.index(), Some(prev_digest), commit.previous_digest);
        }

        let latest_commit = self
            .store
            .get_highest_verified_commit()
            .expect("store operation should not fail");

        // If this is an older commit, just ignore it
        if latest_commit.index() >= commit.index() {
            return;
        }

        let commit = *commit;
        let next_index = latest_commit.index().checked_add(1).unwrap();
        if *commit.index() > next_index {
            debug!(
                "consensus sent too new of a commit, expecting: {}, got: {}",
                next_index,
                commit.index()
            );
        }

        // Because commit from consensus sends in order, when we have commit n,
        // we must have all of the commits before n from either state sync or consensus.
        #[cfg(debug_assertions)]
        {
            let _ = (next_index..=*commit.index())
                .map(|n| {
                    let commit = self
                        .store
                        .get_commit_by_index(n)
                        .unwrap_or_else(|| panic!("store should contain commit {n}"));
                    self.store
                        .get_full_commit_contents(&commit.content_digest)
                        .unwrap_or_else(|| {
                            panic!(
                                "store should contain commit contents for {:?}",
                                commit.content_digest
                            )
                        });
                })
                .collect::<Vec<_>>();
        }

        self.store
            .update_highest_verified_commit(&commit)
            .expect("store operation should not fail");
        self.store
            .update_highest_synced_commit(&commit)
            .expect("store operation should not fail");

        // We don't care if no one is listening as this is a broadcast channel
        let _ = self.commit_event_sender.send(commit.clone());

        self.spawn_notify_peers_of_commit(commit);
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
            let genesis_commit_digest = *self
                .store
                .get_commit_by_index(0)
                .expect("store should contain genesis commit")
                .digest();
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

    fn maybe_start_commit_summary_sync_task(&mut self) {
        // Only run one sync task at a time
        if self.sync_commit_summaries_task.is_some() {
            return;
        }

        let highest_processed_commit = self
            .store
            .get_highest_verified_commit()
            .expect("store operation should not fail");

        let highest_known_commit = self.peer_heights.read().highest_known_commit().cloned();

        if Some(highest_processed_commit.index()) < highest_known_commit.as_ref().map(|x| x.index())
        {
            // start sync job
            let task = sync_to_commit(
                self.active_peers.clone(),
                self.store.clone(),
                self.peer_heights.clone(),
                self.config.timeout(),
                // The if condition should ensure that this is Some
                highest_known_commit.unwrap(),
            )
            .map(|result| match result {
                Ok(()) => {}
                Err(e) => {
                    debug!("error syncing commit {e}");
                }
            });
            let task_handle = self.tasks.spawn(task);
            self.sync_commit_summaries_task = Some(task_handle);
        }
    }

    fn maybe_trigger_commit_contents_sync_task(
        &mut self,
        target_index_channel: &watch::Sender<CommitIndex>,
    ) {
        let highest_verified_commit = self
            .store
            .get_highest_verified_commit()
            .expect("store operation should not fail");
        let highest_synced_commit = self
            .store
            .get_highest_synced_commit()
            .expect("store operation should not fail");

        if highest_verified_commit.index()
            > highest_synced_commit.index()
            // skip if we aren't connected to any peers that can help
            && self
                .peer_heights
                .read()
                .highest_known_commit_index()
                > Some(*highest_synced_commit.index())
        {
            let _ = target_index_channel.send_if_modified(|num| {
                let new_num = *highest_verified_commit.index();
                if *num == new_num {
                    return false;
                }
                *num = new_num;
                true
            });
        }
    }

    fn spawn_notify_peers_of_commit(&mut self, commit: VerifiedCommitSummary) {
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
    commit: VerifiedCommitSummary,
    timeout: Duration,
) {
    let futs = peer_heights
        .read()
        .peers
        .iter()
        // Filter out any peers who we know already have a commit higher than this one
        .filter_map(|(peer_id, info)| (*commit.index() > info.height).then_some(peer_id))
        // Filter out any peers who we aren't connected with
        .flat_map(|peer_id| active_peers.get(peer_id))
        .map(P2pClient::new)
        .map(|mut client| {
            let mut request = Request::new(commit.inner().clone());
            request.set_timeout(timeout);
            async move { client.push_commit_summary(request).await }
        })
        .collect::<Vec<_>>();
    futures::future::join_all(futs).await;
}

async fn get_latest_from_peer(
    our_genesis_commit_digest: CommitSummaryDigest,
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
            let mut request = Request::new(GetCommitSummaryRequest::ByIndex(0));
            request.set_timeout(timeout);
            let response = client
                .get_commit_summary(request)
                .await
                .map(Response::into_inner);

            let info = match response {
                Ok(Some(commit)) => {
                    let digest = *commit.digest();
                    PeerStateSyncInfo {
                        genesis_commit_digest: digest,
                        height: *commit.index(),
                        lowest: CommitIndex::default(),
                    }
                }
                Ok(None) => PeerStateSyncInfo {
                    genesis_commit_digest: CommitSummaryDigest::default(),
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
) -> Option<(CertifiedCommitSummary, CommitIndex)> {
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
                        .update_peer_info(peer_id, highest_commit.clone(), Some(low_watermark))
                        .then_some(highest_commit),
                    None => None,
                }
            }
        })
        .collect::<Vec<_>>();

    debug!("Query {} peers for latest commit", futs.len());

    let commits = futures::future::join_all(futs).await.into_iter().flatten();

    let highest_commit = commits.max_by_key(|commit| *commit.index());

    let our_highest_commit = peer_heights.read().highest_known_commit().cloned();

    debug!(
        "Our highest commit {:?}, peers highest commit {:?}",
        our_highest_commit.as_ref().map(|c| c.index()),
        highest_commit.as_ref().map(|c| c.index())
    );

    let _new_commit = match (highest_commit, our_highest_commit) {
        (Some(theirs), None) => theirs,
        (Some(theirs), Some(ours)) if theirs.index() > ours.index() => theirs,
        _ => return,
    };

    if let Some(sender) = sender.upgrade() {
        let _ = sender.send(StateSyncMessage::StartSyncJob).await;
    }
}

async fn sync_to_commit<S>(
    active_peers: ActivePeers,
    store: S,
    peer_heights: Arc<RwLock<PeerHeights>>,
    timeout: Duration,
    commit: CertifiedCommitSummary,
) -> anyhow::Result<()>
where
    S: WriteStore,
{
    let mut current = store
        .get_highest_verified_commit()
        .expect("store operation should not fail");
    if current.index() >= commit.index() {
        return Err(anyhow::anyhow!(
            "target commit {} is older than highest verified commit {}",
            commit.index(),
            current.index(),
        ));
    }

    let peer_balancer: PeerBalancer = PeerBalancer::new(
        active_peers,
        peer_heights.clone(),
        PeerCommitRequestType::Summary,
    );
    // range of the next indexs to fetch
    let mut request_stream = (current.index().checked_add(1).unwrap()
        ..=*commit.index())
        .map(|next| {
            let peers = peer_balancer.clone().with_commit(next);
            let peer_heights = peer_heights.clone();

            async move {
                if let Some(commit) = peer_heights
                    .read()
                    .get_commit_by_index(next)
                {
                    return (Some(commit.to_owned()), next, None::<PeerId>);
                }

                // Iterate through peers trying each one in turn until we're able to
                // successfully get the target commit
                for peer in peers {
                    let mut request = Request::new(GetCommitSummaryRequest::ByIndex(next));
                    request.set_timeout(timeout);
                    if let Some(commit) = P2pClient::new(peer.channel)
                        .get_commit_summary(request)
                        .await
                        .tap_err(|e| trace!("{e:?}"))
                        .ok()
                        .and_then(Response::into_inner)
                        .tap_none(|| trace!("peer unable to help sync"))
                    {
                        // peer didn't give us a commit with the height that we requested
                        if *commit.index() != next {
                            tracing::debug!(
                                "peer returned commit with wrong index number: expected {next}, got {}",
                                commit.index()
                            );
                            continue;
                        }

                        // Insert in our store in the event that things fail and we need to retry
                        peer_heights
                            .write()
                            .insert_commit(commit.clone());
                        return (Some(commit), next, Some(peer.public_key.into()));
                    }
                }
                (None, next, None)
            }
        })
        .pipe(futures::stream::iter).buffered(COMMIT_SUMMARY_DOWNLOAD_CONCURRENCY);

    while let Some((maybe_commit, next, maybe_peer_id)) = request_stream.next().await {
        assert_eq!(current.index().checked_add(1).expect("exhausted u64"), next);

        // Verify the commit
        let commit = 'cp: {
            let commit = maybe_commit
                .ok_or_else(|| anyhow::anyhow!("no peers were able to help sync commit {next}"))?;

            match verify_commit(&current, &store, commit) {
                Ok(verified_commit) => verified_commit,
                Err(commit) => {
                    let mut peer_heights = peer_heights.write();
                    // Remove the commit from our temporary store so that we can try querying
                    // another peer for a different one
                    peer_heights.remove_commit(commit.digest());

                    return Err(anyhow::anyhow!("unable to verify commit {commit:?}"));
                }
            }
        };

        debug!(commit_seq = ?commit.index(), "verified commit summary");

        current = commit.clone();
        // Insert the newly verified commit into our store, which will bump our highest
        // verified commit watermark as well.
        store
            .insert_commit(&commit)
            .expect("store operation should not fail");
    }

    peer_heights.write().cleanup_old_commits(*commit.index());

    Ok(())
}

async fn sync_commit_contents<S>(
    active_peers: ActivePeers,
    store: S,
    peer_heights: Arc<RwLock<PeerHeights>>,
    sender: mpsc::WeakSender<StateSyncMessage>,
    commit_event_sender: broadcast::Sender<VerifiedCommitSummary>,
    commit_content_download_concurrency: u64,
    commit_content_download_tx_concurrency: u64,
    timeout: Duration,
    mut target_index_channel: watch::Receiver<CommitIndex>,
) where
    S: WriteStore + Clone,
{
    let mut highest_synced = store
        .get_highest_synced_commit()
        .expect("store operation should not fail");

    let mut current_index = highest_synced.index().checked_add(1).unwrap();
    let mut target_index_cursor = 0;
    let mut commit_contents_tasks = FuturesOrdered::new();

    let mut tx_concurrency_remaining = commit_content_download_tx_concurrency;

    loop {
        tokio::select! {
            result = target_index_channel.changed() => {
                match result {
                    Ok(()) => {
                        target_index_cursor = (*target_index_channel.borrow_and_update()).checked_add(1).unwrap();
                    }
                    Err(_) => {
                        // Watch channel is closed, exit loop.
                        return
                    }
                }
            },
            Some(maybe_commit) = commit_contents_tasks.next() => {
                match maybe_commit {
                    Ok(commit) => {
                        let _: &VerifiedCommitSummary = &commit;  // type hint

                        store
                            .update_highest_synced_commit(&commit)
                            .expect("store operation should not fail");
                        // We don't care if no one is listening as this is a broadcast channel
                        let _ = commit_event_sender.send(commit.clone());

                        highest_synced = commit;

                    }
                    Err(commit) => {
                        let _: &VerifiedCommitSummary = &commit;  // type hint
                        if let Some(lowest_peer_commit) =
                            peer_heights.read().peers.iter().map(|(_, state_sync_info)|  state_sync_info.lowest).min() {
                            if commit.index() >= &lowest_peer_commit {
                                info!("unable to sync contents of commit through state sync {} with lowest peer commit: {}", commit.index(), lowest_peer_commit);
                            }
                        } else {
                            info!("unable to sync contents of commit through state sync {}", commit.index());

                        }
                        // Retry contents sync on failure.
                        commit_contents_tasks.push_front(sync_one_commit_contents(
                            active_peers.clone(),
                            &store,
                            peer_heights.clone(),
                            timeout,
                            commit,
                        ));
                    }
                }
            },
        }

        // Start new tasks up to configured concurrency limits.
        while current_index < target_index_cursor
            && commit_contents_tasks.len() < commit_content_download_concurrency as usize
        {
            let next_commit = store
                .get_commit_by_index(current_index)
                .expect("BUG: store should have all commits older than highest_verified_commit");

            current_index += 1;
            commit_contents_tasks.push_back(sync_one_commit_contents(
                active_peers.clone(),
                &store,
                peer_heights.clone(),
                timeout,
                next_commit,
            ));
        }

        if highest_synced.index() % commit_content_download_concurrency as u64 == 0
            || commit_contents_tasks.is_empty()
        {
            // Periodically notify event loop to notify our peers that we've synced to a new commit height
            if let Some(sender) = sender.upgrade() {
                let message = StateSyncMessage::SyncedCommit(Box::new(highest_synced.clone()));
                let _ = sender.send(message).await;
            }
        }
    }
}

#[instrument(level = "debug", skip_all, fields(index = ?commit.index()))]
async fn sync_one_commit_contents<S>(
    active_peers: ActivePeers,
    store: S,
    peer_heights: Arc<RwLock<PeerHeights>>,
    timeout: Duration,
    commit: VerifiedCommitSummary,
) -> Result<VerifiedCommitSummary, VerifiedCommitSummary>
where
    S: WriteStore + Clone,
{
    debug!("syncing commit contents");

    // Check if we already have produced this commit locally. If so, we don't need
    // to get it from peers anymore.
    if store
        .get_highest_synced_commit()
        .expect("store operation should not fail")
        .index()
        >= commit.index()
    {
        debug!("commit was already created via consensus output");
        return Ok(commit);
    }

    // Request commit contents from peers.
    let peers = PeerBalancer::new(
        active_peers,
        peer_heights.clone(),
        PeerCommitRequestType::Content,
    )
    .with_commit(*commit.index());
    let now = tokio::time::Instant::now();
    let Some(_contents) = get_full_commit_contents(peers, &store, &commit, timeout).await else {
        // Delay completion in case of error so we don't hammer the network with retries.
        let duration = peer_heights
            .read()
            .wait_interval_when_no_peer_to_sync_content();
        if now.elapsed() < duration {
            let duration = duration - now.elapsed();
            info!("retrying commit sync after {:?}", duration);
            tokio::time::sleep(duration).await;
        }
        return Err(commit);
    };
    debug!("completed commit contents sync");
    Ok(commit)
}

#[instrument(level = "debug", skip_all)]
async fn get_full_commit_contents<S>(
    peers: PeerBalancer,
    store: S,
    commit: &VerifiedCommitSummary,
    timeout: Duration,
) -> Option<FullCommitContents>
where
    S: WriteStore,
{
    let digest = commit.content_digest;
    if let Some(contents) = store
        .get_full_commit_contents_by_index(*commit.index())
        .or_else(|| store.get_full_commit_contents(&digest))
    {
        debug!("store already contains commit contents");
        return Some(contents);
    }

    // Iterate through our selected peers trying each one in turn until we're able to
    // successfully get the target commit
    for peer in peers {
        debug!(
            ?timeout,
            "requesting commit contents from {}",
            PeerId::from(peer.public_key),
        );
        let mut request = Request::new(digest);
        request.set_timeout(timeout);
        if let Some(contents) = P2pClient::new(peer.channel)
            .get_commit_contents(request)
            .await
            .tap_err(|e| trace!("{e:?}"))
            .ok()
            .and_then(Response::into_inner)
            .tap_none(|| trace!("peer unable to help sync"))
        {
            if contents.verify_digests(digest).is_ok() {
                let verified_contents = VerifiedCommitContents::new_unchecked(contents.clone());
                store
                    .insert_commit_contents(commit, verified_contents)
                    .expect("store operation should not fail");
                return Some(contents);
            }
        }
    }
    debug!("no peers had commit contents");
    None
}

pub fn verify_commit_with_committee(
    committee: Arc<Committee>,
    current: &VerifiedCommitSummary,
    commit: CertifiedCommitSummary,
) -> Result<VerifiedCommitSummary, CertifiedCommitSummary> {
    assert_eq!(*commit.index(), current.index().checked_add(1).unwrap());

    if Some(*current.digest()) != commit.previous_digest {
        debug!(
            current_commit_seq = current.index(),
            current_digest =% current.digest(),
            commit_seq = commit.index(),
            commit_digest =% commit.digest(),
            commit_previous_digest =? commit.previous_digest,
            "commit not on same chain"
        );
        return Err(commit);
    }

    let current_epoch = current.epoch();
    if commit.epoch() != current_epoch && commit.epoch() != current_epoch.checked_add(1).unwrap() {
        debug!(
            commit_seq = commit.index(),
            commit_epoch = commit.epoch(),
            current_commit_seq = current.index(),
            current_epoch = current_epoch,
            "cannot verify commit with too high of an epoch",
        );
        return Err(commit);
    }

    if commit.epoch() == current_epoch.checked_add(1).unwrap()
    // TODO: && current.next_epoch_committee().is_none()
    {
        debug!(
            commit_seq = commit.index(),
            commit_epoch = commit.epoch(),
            current_commit_seq = current.index(),
            current_epoch = current_epoch,
            "next commit claims to be from the next epoch but the latest verified \
            commit does not indicate that it is the last commit of an epoch"
        );
        return Err(commit);
    }

    commit
        .verify_authority_signatures(&committee)
        .map_err(|e| {
            debug!("error verifying commit: {e}");
            commit.clone()
        })?;
    Ok(VerifiedCommitSummary::new_unchecked(commit))
}

pub fn verify_commit<S>(
    current: &VerifiedCommitSummary,
    store: S,
    commit: CertifiedCommitSummary,
) -> Result<VerifiedCommitSummary, CertifiedCommitSummary>
where
    S: WriteStore,
{
    let committee = store
        .get_committee(commit.epoch())
        .unwrap_or_else(|e| {
            panic!(
            "BUG: should have committee for epoch {} before we try to verify commit {} - error {}",
            commit.epoch(),
            commit.index(),
            e
        )
        })
        .unwrap_or_else(|| {
            panic!(
                "BUG: should have committee for epoch {} before we try to verify commit {}",
                commit.epoch(),
                commit.index()
            )
        });

    verify_commit_with_committee(committee, current, commit)
}
