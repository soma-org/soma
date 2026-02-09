use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
    time::Duration,
};

use bytes::Bytes;
use futures::{StreamExt as _, stream::FuturesUnordered};
use itertools::Itertools as _;
use parking_lot::{Mutex, RwLock};
use rand::{prelude::SliceRandom as _, rngs::ThreadRng};
use tap::TapFallible;
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tokio::{
    runtime::Handle,
    sync::{mpsc::error::TrySendError, oneshot},
    task::{JoinError, JoinSet},
    time::{Instant, sleep, sleep_until, timeout},
};
use tracing::{debug, error, info, trace, warn};
use types::committee::AuthorityIndex;
use types::consensus::{
    block::{BlockAPI, BlockRef, Round, SignedBlock, VerifiedBlock},
    context::Context,
};
use types::error::{ConsensusError, ConsensusResult};

use crate::{
    authority_service::COMMIT_LAG_MULTIPLIER, core_thread::CoreThreadDispatcher,
    transaction_certifier::TransactionCertifier,
};
use crate::{
    block_verifier::BlockVerifier, commit_vote_monitor::CommitVoteMonitor, dag_state::DagState,
    network::NetworkClient,
};

/// The number of concurrent fetch blocks requests per authority
const FETCH_BLOCKS_CONCURRENCY: usize = 5;

/// Timeouts when fetching blocks.
const FETCH_REQUEST_TIMEOUT: Duration = Duration::from_millis(2_000);
const FETCH_FROM_PEERS_TIMEOUT: Duration = Duration::from_millis(4_000);

const MAX_AUTHORITIES_TO_FETCH_PER_BLOCK: usize = 2;

// Max number of peers to request missing blocks concurrently in periodic sync.
const MAX_PERIODIC_SYNC_PEERS: usize = 3;

struct BlocksGuard {
    map: Arc<InflightBlocksMap>,
    block_refs: BTreeSet<BlockRef>,
    peer: AuthorityIndex,
}

impl Drop for BlocksGuard {
    fn drop(&mut self) {
        self.map.unlock_blocks(&self.block_refs, self.peer);
    }
}

// Keeps a mapping between the missing blocks that have been instructed to be fetched and the authorities
// that are currently fetching them. For a block ref there is a maximum number of authorities that can
// concurrently fetch it. The authority ids that are currently fetching a block are set on the corresponding
// `BTreeSet` and basically they act as "locks".
struct InflightBlocksMap {
    inner: Mutex<HashMap<BlockRef, BTreeSet<AuthorityIndex>>>,
}

impl InflightBlocksMap {
    fn new() -> Arc<Self> {
        Arc::new(Self { inner: Mutex::new(HashMap::new()) })
    }

    /// Locks the blocks to be fetched for the assigned `peer_index`. We want to avoid re-fetching the
    /// missing blocks from too many authorities at the same time, thus we limit the concurrency
    /// per block by attempting to lock per block. If a block is already fetched by the maximum allowed
    /// number of authorities, then the block ref will not be included in the returned set. The method
    /// returns all the block refs that have been successfully locked and allowed to be fetched.
    fn lock_blocks(
        self: &Arc<Self>,
        missing_block_refs: BTreeSet<BlockRef>,
        peer: AuthorityIndex,
    ) -> Option<BlocksGuard> {
        let mut blocks = BTreeSet::new();
        let mut inner = self.inner.lock();

        for block_ref in missing_block_refs {
            // check that the number of authorities that are already instructed to fetch the block is not
            // higher than the allowed and the `peer_index` has not already been instructed to do that.
            let authorities = inner.entry(block_ref).or_default();
            if authorities.len() < MAX_AUTHORITIES_TO_FETCH_PER_BLOCK
                && authorities.get(&peer).is_none()
            {
                assert!(authorities.insert(peer));
                blocks.insert(block_ref);
            }
        }

        if blocks.is_empty() {
            None
        } else {
            Some(BlocksGuard { map: self.clone(), block_refs: blocks, peer })
        }
    }

    /// Unlocks the provided block references for the given `peer`. The unlocking is strict, meaning that
    /// if this method is called for a specific block ref and peer more times than the corresponding lock
    /// has been called, it will panic.
    fn unlock_blocks(self: &Arc<Self>, block_refs: &BTreeSet<BlockRef>, peer: AuthorityIndex) {
        // Now mark all the blocks as fetched from the map
        let mut blocks_to_fetch = self.inner.lock();
        for block_ref in block_refs {
            let authorities =
                blocks_to_fetch.get_mut(block_ref).expect("Should have found a non empty map");

            assert!(authorities.remove(&peer), "Peer index should be present!");

            // if the last one then just clean up
            if authorities.is_empty() {
                blocks_to_fetch.remove(block_ref);
            }
        }
    }

    /// Drops the provided `blocks_guard` which will force to unlock the blocks, and lock now again the
    /// referenced block refs. The swap is best effort and there is no guarantee that the `peer` will
    /// be able to acquire the new locks.
    fn swap_locks(
        self: &Arc<Self>,
        blocks_guard: BlocksGuard,
        peer: AuthorityIndex,
    ) -> Option<BlocksGuard> {
        let block_refs = blocks_guard.block_refs.clone();

        // Explicitly drop the guard
        drop(blocks_guard);

        // Now create new guard
        self.lock_blocks(block_refs, peer)
    }

    #[cfg(test)]
    fn num_of_locked_blocks(self: &Arc<Self>) -> usize {
        let inner = self.inner.lock();
        inner.len()
    }
}

enum Command {
    FetchBlocks {
        missing_block_refs: BTreeSet<BlockRef>,
        peer_index: AuthorityIndex,
        result: oneshot::Sender<Result<(), ConsensusError>>,
    },
    FetchOwnLastBlock,
    KickOffScheduler,
}

pub(crate) struct SynchronizerHandle {
    commands_sender: Sender<Command>,
    tasks: tokio::sync::Mutex<JoinSet<()>>,
}

impl SynchronizerHandle {
    /// Explicitly asks from the synchronizer to fetch the blocks - provided the block_refs set - from
    /// the peer authority.
    pub(crate) async fn fetch_blocks(
        &self,
        missing_block_refs: BTreeSet<BlockRef>,
        peer_index: AuthorityIndex,
    ) -> ConsensusResult<()> {
        let (sender, receiver) = oneshot::channel();
        self.commands_sender
            .send(Command::FetchBlocks { missing_block_refs, peer_index, result: sender })
            .await
            .map_err(|_err| ConsensusError::Shutdown)?;
        receiver.await.map_err(|_err| ConsensusError::Shutdown)?
    }

    pub(crate) async fn stop(&self) -> Result<(), JoinError> {
        let mut tasks = self.tasks.lock().await;
        tasks.abort_all();
        while let Some(result) = tasks.join_next().await {
            result?
        }
        Ok(())
    }
}

/// `Synchronizer` oversees live block synchronization, crucial for node progress. Live synchronization
/// refers to the process of retrieving missing blocks, particularly those essential for advancing a node
/// when data from only a few rounds is absent. If a node significantly lags behind the network,
/// `commit_syncer` handles fetching missing blocks via a more efficient approach. `Synchronizer`
/// aims for swift catch-up employing two mechanisms:
///
/// 1. Explicitly requesting missing blocks from designated authorities via the "block send" path.
///    This includes attempting to fetch any missing ancestors necessary for processing a received block.
///    Such requests prioritize the block author, maximizing the chance of prompt retrieval.
///    A locking mechanism allows concurrent requests for missing blocks from up to two authorities
///    simultaneously, enhancing the chances of timely retrieval. Notably, if additional missing blocks
///    arise during block processing, requests to the same authority are deferred to the scheduler.
///
/// 2. Periodically requesting missing blocks via a scheduler. This primarily serves to retrieve
///    missing blocks that were not ancestors of a received block via the "block send" path.
///    The scheduler operates on either a fixed periodic basis or is triggered immediately
///    after explicit fetches described in (1), ensuring continued block retrieval if gaps persist.
///
/// Additionally to the above, the synchronizer can synchronize and fetch the last own proposed block
/// from the network peers as best effort approach to recover node from amnesia and avoid making the
/// node equivocate.
pub(crate) struct Synchronizer<C: NetworkClient, V: BlockVerifier, D: CoreThreadDispatcher> {
    context: Arc<Context>,
    commands_receiver: Receiver<Command>,
    fetch_block_senders: BTreeMap<AuthorityIndex, Sender<BlocksGuard>>,
    core_dispatcher: Arc<D>,
    commit_vote_monitor: Arc<CommitVoteMonitor>,
    dag_state: Arc<RwLock<DagState>>,
    fetch_blocks_scheduler_task: JoinSet<()>,
    fetch_own_last_block_task: JoinSet<()>,
    network_client: Arc<C>,
    block_verifier: Arc<V>,
    transaction_certifier: TransactionCertifier,
    inflight_blocks_map: Arc<InflightBlocksMap>,
    commands_sender: Sender<Command>,
}

impl<C: NetworkClient, V: BlockVerifier, D: CoreThreadDispatcher> Synchronizer<C, V, D> {
    pub(crate) fn start(
        network_client: Arc<C>,
        context: Arc<Context>,
        core_dispatcher: Arc<D>,
        commit_vote_monitor: Arc<CommitVoteMonitor>,
        block_verifier: Arc<V>,
        transaction_certifier: TransactionCertifier,
        dag_state: Arc<RwLock<DagState>>,
        sync_last_known_own_block: bool,
    ) -> Arc<SynchronizerHandle> {
        let (commands_sender, commands_receiver) = channel(1_000);
        let inflight_blocks_map = InflightBlocksMap::new();

        // Spawn the tasks to fetch the blocks from the others
        let mut fetch_block_senders = BTreeMap::new();
        let mut tasks = JoinSet::new();
        for (index, _) in context.committee.authorities() {
            if index == context.own_index {
                continue;
            }
            let (sender, receiver) = channel(FETCH_BLOCKS_CONCURRENCY);
            let fetch_blocks_from_authority_async = Self::fetch_blocks_from_authority(
                index,
                network_client.clone(),
                block_verifier.clone(),
                transaction_certifier.clone(),
                commit_vote_monitor.clone(),
                context.clone(),
                core_dispatcher.clone(),
                dag_state.clone(),
                receiver,
                commands_sender.clone(),
            );
            tasks.spawn(fetch_blocks_from_authority_async);
            fetch_block_senders.insert(index, sender);
        }

        let commands_sender_clone = commands_sender.clone();

        if sync_last_known_own_block {
            commands_sender
                .try_send(Command::FetchOwnLastBlock)
                .expect("Failed to sync our last block");
        }

        // Spawn the task to listen to the requests & periodic runs
        tasks.spawn(async move {
            let mut s = Self {
                context,
                commands_receiver,
                fetch_block_senders,
                core_dispatcher,
                commit_vote_monitor,
                fetch_blocks_scheduler_task: JoinSet::new(),
                fetch_own_last_block_task: JoinSet::new(),
                network_client,
                block_verifier,
                transaction_certifier,
                inflight_blocks_map,
                commands_sender: commands_sender_clone,
                dag_state,
            };
            s.run().await;
        });

        Arc::new(SynchronizerHandle { commands_sender, tasks: tokio::sync::Mutex::new(tasks) })
    }

    // The main loop to listen for the submitted commands.
    async fn run(&mut self) {
        // We want the synchronizer to run periodically every 200ms to fetch any missing blocks.
        const PERIODIC_FETCH_INTERVAL: Duration = Duration::from_millis(200);
        let scheduler_timeout = sleep_until(Instant::now() + PERIODIC_FETCH_INTERVAL);

        tokio::pin!(scheduler_timeout);

        loop {
            tokio::select! {
                Some(command) = self.commands_receiver.recv() => {
                    match command {
                        Command::FetchBlocks{ missing_block_refs, peer_index, result } => {
                            if peer_index == self.context.own_index {
                                error!("We should never attempt to fetch blocks from our own node");
                                continue;
                            }

                            // Keep only the max allowed blocks to request. Additional missing blocks
                            // will be fetched via periodic sync.
                            // Fetch from the lowest to highest round, to ensure progress.
                            let missing_block_refs = missing_block_refs
                                .into_iter()
                                .take(self.context.parameters.max_blocks_per_sync)
                                .collect();

                            let blocks_guard = self.inflight_blocks_map.lock_blocks(missing_block_refs, peer_index);
                            let Some(blocks_guard) = blocks_guard else {
                                result.send(Ok(())).ok();
                                continue;
                            };

                            // We don't block if the corresponding peer task is saturated - but we rather drop the request. That's ok as the periodic
                            // synchronization task will handle any still missing blocks in next run.
                            let r = self
                                .fetch_block_senders
                                .get(&peer_index)
                                .expect("Fatal error, sender should be present")
                                .try_send(blocks_guard)
                                .map_err(|err| {
                                    match err {
                                        TrySendError::Full(_) => {

                                            ConsensusError::SynchronizerSaturated(peer_index)
                                        },
                                        TrySendError::Closed(_) => ConsensusError::Shutdown
                                    }
                                });

                            result.send(r).ok();
                        }
                        Command::FetchOwnLastBlock => {
                            if self.fetch_own_last_block_task.is_empty() {
                                self.start_fetch_own_last_block_task();
                            }
                        }
                        Command::KickOffScheduler => {
                            // just reset the scheduler timeout timer to run immediately if not already running.
                            // If the scheduler is already running then just reduce the remaining time to run.
                            let timeout = if self.fetch_blocks_scheduler_task.is_empty() {
                                Instant::now()
                            } else {
                                Instant::now() + PERIODIC_FETCH_INTERVAL.checked_div(2).unwrap()
                            };

                            // only reset if it is earlier than the next deadline
                            if timeout < scheduler_timeout.deadline() {
                                scheduler_timeout.as_mut().reset(timeout);
                            }
                        }
                    }
                },
                Some(result) = self.fetch_own_last_block_task.join_next(), if !self.fetch_own_last_block_task.is_empty() => {
                    match result {
                        Ok(()) => {},
                        Err(e) => {
                            if e.is_cancelled() {
                            } else if e.is_panic() {
                                std::panic::resume_unwind(e.into_panic());
                            } else {
                                panic!("fetch our last block task failed: {e}");
                            }
                        },
                    };
                },
                Some(result) = self.fetch_blocks_scheduler_task.join_next(), if !self.fetch_blocks_scheduler_task.is_empty() => {
                    match result {
                        Ok(()) => {},
                        Err(e) => {
                            if e.is_cancelled() {
                            } else if e.is_panic() {
                                std::panic::resume_unwind(e.into_panic());
                            } else {
                                panic!("fetch blocks scheduler task failed: {e}");
                            }
                        },
                    };
                },
                () = &mut scheduler_timeout => {
                    // we want to start a new task only if the previous one has already finished.
                    // TODO: consider starting backup fetches in parallel, when a fetch takes too long?
                    if self.fetch_blocks_scheduler_task.is_empty()
                         {
                            if let Err(err) = self.start_fetch_missing_blocks_task().await {
                                 debug!("Core is shutting down, synchronizer is shutting down: {err:?}");
                            return;
                            }

                        };

                    scheduler_timeout
                        .as_mut()
                        .reset(Instant::now() + PERIODIC_FETCH_INTERVAL);
                }
            }
        }
    }

    async fn fetch_blocks_from_authority(
        peer_index: AuthorityIndex,
        network_client: Arc<C>,
        block_verifier: Arc<V>,
        transaction_certifier: TransactionCertifier,
        commit_vote_monitor: Arc<CommitVoteMonitor>,
        context: Arc<Context>,
        core_dispatcher: Arc<D>,
        dag_state: Arc<RwLock<DagState>>,
        mut receiver: Receiver<BlocksGuard>,
        commands_sender: Sender<Command>,
    ) {
        const MAX_RETRIES: u32 = 3;

        let mut requests = FuturesUnordered::new();

        loop {
            tokio::select! {
                Some(blocks_guard) = receiver.recv(), if requests.len() < FETCH_BLOCKS_CONCURRENCY => {
                    // get the highest accepted rounds
                    let highest_rounds = Self::get_highest_accepted_rounds(dag_state.clone(), &context);

                    requests.push(Self::fetch_blocks_request(network_client.clone(), peer_index, blocks_guard, highest_rounds, true, FETCH_REQUEST_TIMEOUT, 1))
                },
                Some((response, blocks_guard, retries, _peer, highest_rounds)) = requests.next() => {
                    match response {
                        Ok(blocks) => {
                            if let Err(err) = Self::process_fetched_blocks(blocks,
                                peer_index,
                                blocks_guard,
                                core_dispatcher.clone(),
                                block_verifier.clone(),
                                transaction_certifier.clone(),
                                commit_vote_monitor.clone(),
                                context.clone(),
                                commands_sender.clone(),
                                "live"
                            ).await {
                                warn!("Error while processing fetched blocks from peer {peer_index}: {err}");

                            }
                        },
                        Err(_) => {

                            if retries <= MAX_RETRIES {
                                requests.push(Self::fetch_blocks_request(network_client.clone(), peer_index, blocks_guard, highest_rounds, true, FETCH_REQUEST_TIMEOUT, retries))
                            } else {
                                warn!("Max retries {retries} reached while trying to fetch blocks from peer {peer_index}.");
                                // we don't necessarily need to do, but dropping the guard here to unlock the blocks
                                drop(blocks_guard);
                            }
                        }
                    }
                },
                else => {
                    info!("Fetching blocks from authority {peer_index} task will now abort.");
                    break;
                }
            }
        }
    }

    /// Processes the requested raw fetched blocks from peer `peer_index`. If no error is returned then
    /// the verified blocks are immediately sent to Core for processing.
    async fn process_fetched_blocks(
        mut serialized_blocks: Vec<Bytes>,
        peer_index: AuthorityIndex,
        requested_blocks_guard: BlocksGuard,
        core_dispatcher: Arc<D>,
        block_verifier: Arc<V>,
        transaction_certifier: TransactionCertifier,
        commit_vote_monitor: Arc<CommitVoteMonitor>,
        context: Arc<Context>,
        commands_sender: Sender<Command>,
        sync_method: &str,
    ) -> ConsensusResult<()> {
        if serialized_blocks.is_empty() {
            return Ok(());
        }

        // Limit the number of the returned blocks processed.
        serialized_blocks.truncate(context.parameters.max_blocks_per_sync);

        // Verify all the fetched blocks
        let blocks = Handle::current()
            .spawn_blocking({
                let block_verifier = block_verifier.clone();
                let context = context.clone();
                move || {
                    Self::verify_blocks(
                        serialized_blocks,
                        block_verifier,
                        transaction_certifier,
                        &context,
                        peer_index,
                    )
                }
            })
            .await
            .expect("Spawn blocking should not fail")?;

        // Record commit votes from the verified blocks.
        for block in &blocks {
            commit_vote_monitor.observe_block(block);
        }

        debug!(
            "Synced {} missing blocks from peer {peer_index}: {}",
            blocks.len(),
            blocks.iter().map(|b| b.reference().to_string()).join(", "),
        );

        // Now send them to core for processing. Ignore the returned missing blocks as we don't want
        // this mechanism to keep feedback looping on fetching more blocks. The periodic synchronization
        // will take care of that.
        let missing_blocks =
            core_dispatcher.add_blocks(blocks).await.map_err(|_| ConsensusError::Shutdown)?;

        // now release all the locked blocks as they have been fetched, verified & processed
        drop(requested_blocks_guard);

        // kick off immediately the scheduled synchronizer
        if !missing_blocks.is_empty() {
            // do not block here, so we avoid any possible cycles.
            if let Err(TrySendError::Full(_)) = commands_sender.try_send(Command::KickOffScheduler)
            {
                warn!("Commands channel is full")
            }
        }

        Ok(())
    }

    fn get_highest_accepted_rounds(
        dag_state: Arc<RwLock<DagState>>,
        context: &Arc<Context>,
    ) -> Vec<Round> {
        let blocks = dag_state.read().get_last_cached_block_per_authority(Round::MAX);
        assert_eq!(blocks.len(), context.committee.size());

        blocks.into_iter().map(|(block, _)| block.round()).collect::<Vec<_>>()
    }

    fn verify_blocks(
        serialized_blocks: Vec<Bytes>,
        block_verifier: Arc<V>,
        transaction_certifier: TransactionCertifier,
        context: &Context,
        peer_index: AuthorityIndex,
    ) -> ConsensusResult<Vec<VerifiedBlock>> {
        let mut verified_blocks = Vec::new();
        let mut voted_blocks = Vec::new();
        for serialized_block in serialized_blocks {
            let signed_block: SignedBlock =
                bcs::from_bytes(&serialized_block).map_err(ConsensusError::MalformedBlock)?;

            // TODO: cache received and verified block refs to avoid duplicated work.
            let (verified_block, reject_txn_votes) =
                block_verifier.verify_and_vote(signed_block, serialized_block).tap_err(|e| {
                    info!("Invalid block received from {}: {}", peer_index, e);
                })?;

            // TODO: improve efficiency, maybe suspend and continue processing the block asynchronously.
            let now = context.clock.timestamp_utc_ms();
            let drift = verified_block.timestamp_ms().saturating_sub(now);
            if drift > 0 {
                trace!(
                    "Synced block {} timestamp {} is in the future (now={}).",
                    verified_block.reference(),
                    verified_block.timestamp_ms(),
                    now
                );
            }

            verified_blocks.push(verified_block.clone());
            voted_blocks.push((verified_block, reject_txn_votes));
        }

        transaction_certifier.add_voted_blocks(voted_blocks);

        Ok(verified_blocks)
    }

    async fn fetch_blocks_request(
        network_client: Arc<C>,
        peer: AuthorityIndex,
        blocks_guard: BlocksGuard,
        highest_rounds: Vec<Round>,
        breadth_first: bool,
        request_timeout: Duration,
        mut retries: u32,
    ) -> (ConsensusResult<Vec<Bytes>>, BlocksGuard, u32, AuthorityIndex, Vec<Round>) {
        let start = Instant::now();
        let resp = timeout(
            request_timeout,
            network_client.fetch_blocks(
                peer,
                blocks_guard.block_refs.clone().into_iter().collect::<Vec<_>>(),
                highest_rounds.clone().into_iter().collect::<Vec<_>>(),
                breadth_first,
                request_timeout,
            ),
        )
        .await;

        let resp = match resp {
            Ok(Err(err)) => {
                // Add a delay before retrying - if that is needed. If request has timed out then eventually
                // this will be a no-op.
                sleep_until(start + request_timeout).await;
                retries += 1;
                Err(err)
            } // network error
            Err(err) => {
                // timeout
                sleep_until(start + request_timeout).await;
                retries += 1;
                Err(ConsensusError::NetworkRequestTimeout(err.to_string()))
            }
            Ok(result) => result,
        };
        (resp, blocks_guard, retries, peer, highest_rounds)
    }

    fn start_fetch_own_last_block_task(&mut self) {
        const FETCH_OWN_BLOCK_RETRY_DELAY: Duration = Duration::from_millis(1_000);
        const MAX_RETRY_DELAY_STEP: Duration = Duration::from_millis(4_000);

        let context = self.context.clone();
        let dag_state = self.dag_state.clone();
        let network_client = self.network_client.clone();
        let block_verifier = self.block_verifier.clone();
        let core_dispatcher = self.core_dispatcher.clone();

        self.fetch_own_last_block_task
            .spawn(async move {
                
                let fetch_own_block = |authority_index: AuthorityIndex, fetch_own_block_delay: Duration| {
                    let network_client_cloned = network_client.clone();
                    let own_index = context.own_index;
                    async move {
                        sleep(fetch_own_block_delay).await;
                        let r = network_client_cloned.fetch_latest_blocks(authority_index, vec![own_index], FETCH_REQUEST_TIMEOUT).await;
                        (r, authority_index)
                    }
                };

                let process_blocks = |blocks: Vec<Bytes>, authority_index: AuthorityIndex| -> ConsensusResult<Vec<VerifiedBlock>> {
                    let mut result = Vec::new();
                    for serialized_block in blocks {
                        let signed_block = bcs::from_bytes(&serialized_block).map_err(ConsensusError::MalformedBlock)?;
                        let (verified_block, _) = block_verifier.verify_and_vote(signed_block, serialized_block).tap_err(|err|{
                          
                          
                            warn!("Invalid block received from {}: {}", authority_index, err);
                        })?;

                        if verified_block.author() != context.own_index {
                            return Err(ConsensusError::UnexpectedLastOwnBlock { index: authority_index, block_ref: verified_block.reference()});
                        }
                        result.push(verified_block);
                    }
                    Ok(result)
                };

                // Get the highest of all the results. Retry until at least `f+1` results have been gathered.
                let mut highest_round;
                let mut retries = 0;
                let mut retry_delay_step = Duration::from_millis(500);
                'main:loop {
                    if context.committee.size() == 1 {
                        highest_round = dag_state.read().get_last_proposed_block().round();
                        info!("Only one node in the network, will not try fetching own last block from peers.");
                        break 'main;
                    }

                    let mut total_stake = 0;
                    highest_round = 0;

                    // Ask all the other peers about our last block
                    let mut results = FuturesUnordered::new();

                    for (authority_index, _authority) in context.committee.authorities() {
                        if authority_index != context.own_index {
                            results.push(fetch_own_block(authority_index, Duration::from_millis(0)));
                        }
                    }

                    // Gather the results but wait to timeout as well
                    let timer = sleep_until(Instant::now() + context.parameters.sync_last_known_own_block_timeout);
                    tokio::pin!(timer);

                    'inner: loop {
                        tokio::select! {
                            result = results.next() => {
                                let Some((result, authority_index)) = result else {
                                    break 'inner;
                                };
                                match result {
                                    Ok(result) => {
                                        match process_blocks(result, authority_index) {
                                            Ok(blocks) => {
                                                let max_round = blocks.into_iter().map(|b|b.round()).max().unwrap_or(0);
                                                highest_round = highest_round.max(max_round);

                                                total_stake += context.committee.stake_by_index(authority_index);
                                            },
                                            Err(err) => {
                                                warn!("Invalid result returned from {authority_index} while fetching last own block: {err}");
                                            }
                                        }
                                    },
                                    Err(err) => {
                                        warn!("Error {err} while fetching our own block from peer {authority_index}. Will retry.");
                                        results.push(fetch_own_block(authority_index, FETCH_OWN_BLOCK_RETRY_DELAY));
                                    }
                                }
                            },
                            () = &mut timer => {
                                info!("Timeout while trying to sync our own last block from peers");
                                break 'inner;
                            }
                        }
                    }

                    // Request at least f+1 stake to have replied back.
                    if context.committee.reached_validity(total_stake) {
                        info!("{} out of {} total stake returned acceptable results for our own last block with highest round {}, with {retries} retries.", total_stake, context.committee.total_stake(), highest_round);
                        break 'main;
                    }

                    retries += 1;
                    
                    warn!("Not enough stake: {} out of {} total stake returned acceptable results for our own last block with highest round {}. Will now retry {retries}.", total_stake, context.committee.total_stake(), highest_round);

                    sleep(retry_delay_step).await;

                    retry_delay_step = Duration::from_secs_f64(retry_delay_step.as_secs_f64() * 1.5);
                    retry_delay_step = retry_delay_step.min(MAX_RETRY_DELAY_STEP);
                }

                // Update the Core with the highest detected round
               

                if let Err(err) = core_dispatcher.set_last_known_proposed_round(highest_round) {
                    warn!("Error received while calling dispatcher, probably dispatcher is shutting down, will now exit: {err:?}");
                }
            });
    }

    async fn start_fetch_missing_blocks_task(&mut self) -> ConsensusResult<()> {
        if self.context.committee.size() == 1 {
            trace!(
                "Only one node in the network, will not try fetching missing blocks from peers."
            );
            return Ok(());
        }

        let missing_blocks = self
            .core_dispatcher
            .get_missing_blocks()
            .await
            .map_err(|_err| ConsensusError::Shutdown)?;

        // No reason to kick off the scheduler if there are no missing blocks to fetch
        if missing_blocks.is_empty() {
            return Ok(());
        }

        let context = self.context.clone();
        let network_client = self.network_client.clone();
        let block_verifier = self.block_verifier.clone();
        let transaction_certifier = self.transaction_certifier.clone();
        let commit_vote_monitor = self.commit_vote_monitor.clone();
        let core_dispatcher = self.core_dispatcher.clone();
        let blocks_to_fetch = self.inflight_blocks_map.clone();
        let commands_sender = self.commands_sender.clone();
        let dag_state = self.dag_state.clone();

        // If we are commit lagging, then we don't want to enable the scheduler. As the node is sycnhronizing via the commit syncer, the certified commits
        // will bring all the necessary blocks to run the commits. As the commits are certified, we are guaranteed that all the necessary causal history is present.
        if self.is_commit_lagging() {
            return Ok(());
        }

        self.fetch_blocks_scheduler_task.spawn(async move {
            let total_requested = missing_blocks.len();

            // Fetch blocks from peers
            let results = Self::fetch_blocks_from_authorities(
                context.clone(),
                blocks_to_fetch.clone(),
                network_client,
                missing_blocks,
                dag_state,
            )
            .await;

            if results.is_empty() {
                return;
            }

            // Now process the returned results
            let mut total_fetched = 0;
            for (blocks_guard, fetched_blocks, peer) in results {
                total_fetched += fetched_blocks.len();

                if let Err(err) = Self::process_fetched_blocks(
                    fetched_blocks,
                    peer,
                    blocks_guard,
                    core_dispatcher.clone(),
                    block_verifier.clone(),
                    transaction_certifier.clone(),
                    commit_vote_monitor.clone(),
                    context.clone(),
                    commands_sender.clone(),
                    "periodic",
                )
                .await
                {
                    warn!("Error occurred while processing fetched blocks from peer {peer}: {err}");
                }
            }

            debug!(
                "Total blocks requested to fetch: {}, total fetched: {}",
                total_requested, total_fetched
            );
        });
        Ok(())
    }

    fn is_commit_lagging(&self) -> bool {
        let last_commit_index = self.dag_state.read().last_commit_index();
        let quorum_commit_index = self.commit_vote_monitor.quorum_commit_index();
        let commit_threshold = last_commit_index
            + self.context.parameters.commit_sync_batch_size * COMMIT_LAG_MULTIPLIER;
        commit_threshold < quorum_commit_index
    }

    /// Fetches the `missing_blocks` from peers. Requests the same number of authorities with missing blocks from each peer.
    /// Each response from peer can contain the requested blocks, and additional blocks from the last accepted round for
    /// authorities with missing blocks.
    /// Each element of the vector is a tuple which contains the requested missing block refs, the returned blocks and
    /// the peer authority index.
    async fn fetch_blocks_from_authorities(
        context: Arc<Context>,
        inflight_blocks: Arc<InflightBlocksMap>,
        network_client: Arc<C>,
        missing_blocks: BTreeSet<BlockRef>,
        dag_state: Arc<RwLock<DagState>>,
    ) -> Vec<(BlocksGuard, Vec<Bytes>, AuthorityIndex)> {
        // Preliminary truncation of missing blocks to fetch. Since each peer can have different
        // number of missing blocks and the fetching is batched by peer, so keep more than max_blocks_per_sync
        // per peer on average.
        let missing_blocks = missing_blocks
            .into_iter()
            .take(2 * MAX_PERIODIC_SYNC_PEERS * context.parameters.max_blocks_per_sync)
            .collect::<Vec<_>>();

        // Maps authorities to the missing blocks they have.
        let mut authorities = BTreeMap::<AuthorityIndex, Vec<BlockRef>>::new();
        for block_ref in &missing_blocks {
            authorities.entry(block_ref.author).or_default().push(*block_ref);
        }
        // Distribute the same number of authorities into each peer to sync.
        // When running this function, context.committee.size() is always greater than 1.
        let num_authorities_per_peer =
            authorities.len().div_ceil((context.committee.size() - 1).min(MAX_PERIODIC_SYNC_PEERS));

        // Update metrics related to missing blocks.
        let mut missing_blocks_per_authority = vec![0; context.committee.size()];
        for (authority, blocks) in &authorities {
            missing_blocks_per_authority[*authority] += blocks.len();
        }

        let mut peers = context
            .committee
            .authorities()
            .filter_map(|(peer_index, _)| (peer_index != context.own_index).then_some(peer_index))
            .collect::<Vec<_>>();

        // TODO: probably inject the RNG to allow unit testing - this is a work around for now.
        if cfg!(not(test)) {
            // Shuffle the peers
            peers.shuffle(&mut ThreadRng::default());
        }

        let mut peers = peers.into_iter();
        let mut request_futures = FuturesUnordered::new();

        let highest_rounds = Self::get_highest_accepted_rounds(dag_state, &context);

        // Shuffle the authorities for each request.
        let mut authorities = authorities.into_values().collect::<Vec<_>>();
        if cfg!(not(test)) {
            // Shuffle the authorities
            authorities.shuffle(&mut ThreadRng::default());
        }

        // Send the fetch requests
        for batch in authorities.chunks(num_authorities_per_peer) {
            let Some(peer) = peers.next() else {
                debug!("No more peers left to fetch blocks!");
                break;
            };

            // Fetch from the lowest round missing blocks to ensure progress.
            // This may reduce efficiency and increase the chance of duplicated data transfer in edge cases.
            let block_refs = batch
                .iter()
                .flatten()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .take(context.parameters.max_blocks_per_sync)
                .collect::<BTreeSet<_>>();

            // lock the blocks to be fetched. If no lock can be acquired for any of the blocks then don't bother
            if let Some(blocks_guard) = inflight_blocks.lock_blocks(block_refs.clone(), peer) {
                info!(
                    "Periodic sync of {} missing blocks from peer {}: {}",
                    block_refs.len(),
                    peer,
                    block_refs.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(", ")
                );
                request_futures.push(Self::fetch_blocks_request(
                    network_client.clone(),
                    peer,
                    blocks_guard,
                    highest_rounds.clone(),
                    false,
                    FETCH_REQUEST_TIMEOUT,
                    1,
                ));
            }
        }

        let mut results = Vec::new();
        let fetcher_timeout = sleep(FETCH_FROM_PEERS_TIMEOUT);

        tokio::pin!(fetcher_timeout);

        loop {
            tokio::select! {
                Some((response, blocks_guard, _retries, peer_index, highest_rounds)) = request_futures.next() => {

                    match response {
                        Ok(fetched_blocks) => {
                            results.push((blocks_guard, fetched_blocks, peer_index));

                            // no more pending requests are left, just break the loop
                            if request_futures.is_empty() {
                                break;
                            }
                        },
                        Err(_) => {

                            // try again if there is any peer left
                            if let Some(next_peer) = peers.next() {
                                // do best effort to lock guards. If we can't lock then don't bother at this run.
                                if let Some(blocks_guard) = inflight_blocks.swap_locks(blocks_guard, next_peer) {
                                    info!(
                                        "Retrying syncing {} missing blocks from peer: {}",
                                        blocks_guard.block_refs.len(),

                                        blocks_guard.block_refs
                                            .iter()
                                            .map(|b| b.to_string())
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    );
                                    request_futures.push(Self::fetch_blocks_request(
                                        network_client.clone(),
                                        next_peer,
                                        blocks_guard,
                                        highest_rounds,
                                        false,
                                        FETCH_REQUEST_TIMEOUT,
                                        1,
                                    ));
                                } else {
                                    debug!("Couldn't acquire locks to fetch blocks from peer {next_peer}.")
                                }
                            } else {
                                debug!("No more peers left to fetch blocks");
                            }
                        }
                    }
                },
                _ = &mut fetcher_timeout => {
                    debug!("Timed out while fetching missing blocks");
                    break;
                }
            }
        }

        results
    }
}
