use bytes::Bytes;
use parking_lot::RwLock;
use rand::seq::IteratorRandom;
use std::sync::Arc;
use std::{collections::HashMap, result::Result};
use tokio::sync::mpsc;
use tokio_stream::{iter, Iter};
use tonic::{async_trait, Request, Response, Status};
use tracing::{debug, info};
use types::committee::{Committee, EpochId};
use types::consensus::block::{BlockAPI, BlockRef, Round, VerifiedBlock, GENESIS_ROUND};
use types::consensus::commit::{CommitAPI, CommitRange, TrustedCommit};
use types::consensus::stake_aggregator::{QuorumThreshold, StakeAggregator};
use types::dag::dag_state::DagState;
use types::digests::CommitContentsDigest;
use types::discovery::{GetKnownPeersRequest, GetKnownPeersResponse};
use types::error::{ConsensusError, ConsensusResult};
use types::p2p::channel_manager::PeerInfo;
use types::state_sync::{
    FetchBlocksRequest, FetchBlocksResponse, FetchCommitsRequest, FetchCommitsResponse,
    GetCommitAvailabilityRequest, GetCommitAvailabilityResponse, GetCommitSummaryRequest,
    PushCommitSummaryResponse,
};
use types::storage::consensus::ConsensusStore;
use types::storage::write_store::WriteStore;

use crate::state_sync::{PeerHeights, StateSyncMessage};
use crate::tonic_gen::p2p_server::P2p;

use crate::discovery::{now_unix, DiscoveryState};

// Maximum bytes size in a single fetch_blocks() response.
const MAX_FETCH_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

const MAX_PEERS_TO_SEND: usize = 200;

const COMMIT_SYNC_BATCH_SIZE: u32 = 100;

pub struct P2pService<S> {
    pub discovery_state: Arc<RwLock<DiscoveryState>>,
    pub store: S,
    pub peer_heights: Arc<RwLock<PeerHeights>>,
    pub sender: mpsc::WeakSender<StateSyncMessage>,
    pub dag_state: Arc<RwLock<DagState>>,
}

impl<S> P2pService<S>
where
    S: ConsensusStore + WriteStore + Send + Sync + 'static,
{
    async fn handle_fetch_blocks(
        &self,
        block_refs: Vec<BlockRef>,
        highest_accepted_rounds: Vec<Round>,
        epoch: EpochId,
    ) -> ConsensusResult<Vec<Bytes>> {
        const MAX_ADDITIONAL_BLOCKS: usize = 10;
        // if block_refs.len() > self.context.parameters.max_blocks_per_fetch {
        //     return Err(ConsensusError::TooManyFetchBlocksRequested(peer));
        // }

        let Some(committee) = self
            .store
            .get_committee(epoch)?
            .map(|c| Committee::clone(&*c))
        else {
            return Err(ConsensusError::NoCommitteeForEpoch(epoch));
        };

        if !highest_accepted_rounds.is_empty() && highest_accepted_rounds.len() != committee.size()
        {
            return Err(ConsensusError::InvalidSizeOfHighestAcceptedRounds(
                highest_accepted_rounds.len(),
                committee.size(),
            ));
        }

        // Some quick validation of the requested block refs
        for block in &block_refs {
            if !committee.is_valid_index(block.author) {
                return Err(ConsensusError::InvalidAuthorityIndex {
                    index: block.author,
                    max: committee.size(),
                });
            }
            if block.round == GENESIS_ROUND {
                return Err(ConsensusError::UnexpectedGenesisBlockRequested);
            }
        }

        // For now ask dag state directly
        let blocks = self.dag_state.read().get_blocks(&block_refs);

        // Now check if an ancestor's round is higher than the one that the peer has. If yes, then serve
        // that ancestor blocks up to `MAX_ADDITIONAL_BLOCKS`.
        let mut ancestor_blocks = vec![];
        if !highest_accepted_rounds.is_empty() {
            let all_ancestors = blocks
                .iter()
                .flatten()
                .flat_map(|block| block.ancestors().to_vec())
                .filter(|block_ref| highest_accepted_rounds[block_ref.author] < block_ref.round)
                .take(MAX_ADDITIONAL_BLOCKS)
                .collect::<Vec<_>>();

            if !all_ancestors.is_empty() {
                ancestor_blocks = self.dag_state.read().get_blocks(&all_ancestors);
            }
        }

        // Return the serialised blocks & the ancestor blocks
        let result = blocks
            .into_iter()
            .chain(ancestor_blocks)
            .flatten()
            .map(|block| block.serialized().clone())
            .collect::<Vec<_>>();

        Ok(result)
    }

    async fn handle_fetch_commits(
        &self,
        commit_range: CommitRange,
    ) -> ConsensusResult<(Vec<TrustedCommit>, Vec<VerifiedBlock>)> {
        // Compute an inclusive end index and bound the maximum number of commits scanned.
        let inclusive_end = commit_range
            .end()
            .min(commit_range.start() + COMMIT_SYNC_BATCH_SIZE - 1);
        let mut commits = self
            .store
            .scan_commits((commit_range.start()..=inclusive_end).into())?;
        let mut certifier_block_refs = vec![];
        'commit: while let Some(c) = commits.last() {
            let index = c.index();
            let votes = self.store.read_commit_votes(index)?;
            let committee = self
                .store
                .get_committee(c.epoch())?
                .ok_or_else(|| ConsensusError::NoCommitteeForEpoch(c.epoch()))?;
            let mut stake_aggregator = StakeAggregator::<QuorumThreshold>::new();
            for v in &votes {
                stake_aggregator.add(v.author, &committee);
            }
            if stake_aggregator.reached_threshold(&committee) {
                certifier_block_refs = votes;
                break 'commit;
            } else {
                commits.pop();
            }
        }
        let certifier_blocks = self
            .store
            .read_blocks(&certifier_block_refs)?
            .into_iter()
            .flatten()
            .collect();
        Ok((commits, certifier_blocks))
    }
}

#[async_trait]
impl<S> P2p for P2pService<S>
where
    S: ConsensusStore + WriteStore + Send + Sync + 'static,
{
    type FetchBlocksStream = Iter<std::vec::IntoIter<Result<FetchBlocksResponse, tonic::Status>>>;

    async fn get_known_peers(
        &self,
        _request: Request<GetKnownPeersRequest>,
    ) -> Result<Response<GetKnownPeersResponse>, tonic::Status> {
        let state = self.discovery_state.read();
        let own_info = state
            .our_info
            .clone()
            .ok_or_else(|| tonic::Status::internal("own_info has not been initialized yet"))?;

        let known_peers = if state.known_peers.len() < MAX_PEERS_TO_SEND {
            state
                .known_peers
                .values()
                .map(|e| e.inner())
                .cloned()
                .collect()
        } else {
            let mut rng = rand::thread_rng();
            // TODO: prefer returning peers that we are connected to as they are known-good
            let mut known_peers = state
                .known_peers
                .values()
                .map(|info| (info.peer_id, info))
                .choose_multiple(&mut rng, MAX_PEERS_TO_SEND)
                .into_iter()
                .collect::<HashMap<_, _>>();

            if known_peers.len() <= MAX_PEERS_TO_SEND {
                // Fill the remaining space with other peers, randomly sampling at most MAX_PEERS_TO_SEND
                for info in state
                    .known_peers
                    .values()
                    // This randomly samples the iterator stream but the order of elements after
                    // sampling may not be random, this is ok though since we're just trying to do
                    // best-effort on sharing info of peers we haven't connected with ourselves.
                    .choose_multiple(&mut rng, MAX_PEERS_TO_SEND)
                {
                    if known_peers.len() >= MAX_PEERS_TO_SEND {
                        break;
                    }

                    known_peers.insert(info.peer_id, info);
                }
            }

            known_peers
                .into_values()
                .map(|e| e.inner())
                .cloned()
                .collect()
        };

        info!("Sending known peers and our info {}", own_info.peer_id);

        Ok(Response::new(GetKnownPeersResponse {
            own_info,
            known_peers,
        }))
    }

    // async fn push_commit_summary(
    //     &self,
    //     request: Request<CertifiedCommitSummary>,
    // ) -> Result<Response<PushCommitSummaryResponse>, tonic::Status> {
    //     let Some(peer_id) = request.extensions().get::<PeerInfo>().map(|p| p.peer_id) else {
    //         return Err(tonic::Status::internal("PeerInfo not found"));
    //     };

    //     let commit = request.into_inner();
    //     if !self
    //         .peer_heights
    //         .write()
    //         .update_peer_info(peer_id, commit.clone(), None)
    //     {
    //         return Ok(Response::new(PushCommitSummaryResponse {
    //             timestamp_ms: now_unix(),
    //         }));
    //     }

    //     let highest_verified_commit = *self
    //         .store
    //         .get_highest_verified_commit()
    //         .map_err(|e| Status::internal(e.to_string()))?
    //         .index();

    //     // If this commit is higher than our highest verified commit notify the
    //     // event loop to potentially sync it
    //     if *commit.index() > highest_verified_commit {
    //         if let Some(sender) = self.sender.upgrade() {
    //             sender.send(StateSyncMessage::StartSyncJob).await.unwrap();
    //         }
    //     }

    //     Ok(Response::new(PushCommitSummaryResponse {
    //         timestamp_ms: now_unix(),
    //     }))
    // }

    async fn get_commit_availability(
        &self,
        _request: Request<GetCommitAvailabilityRequest>,
    ) -> Result<Response<GetCommitAvailabilityResponse>, tonic::Status> {
        let highest_synced_commit = self
            .store
            .get_highest_synced_commit()
            .map_err(|e| Status::internal(e.to_string()))?;
        let lowest_available_commit = self
            .store
            .get_lowest_available_commit()
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(GetCommitAvailabilityResponse {
            highest_synced_commit,
            lowest_available_commit,
        }))
    }

    async fn fetch_blocks(
        &self,
        request: Request<FetchBlocksRequest>,
    ) -> Result<Response<Self::FetchBlocksStream>, tonic::Status> {
        let inner = request.into_inner();
        let block_refs = inner
            .block_refs
            .into_iter()
            .filter_map(|serialized| match bcs::from_bytes(&serialized) {
                Ok(r) => Some(r),
                Err(e) => {
                    debug!("Failed to deserialize block ref {:?}: {e:?}", serialized);
                    None
                }
            })
            .collect();
        let highest_accepted_rounds = inner.highest_accepted_rounds;
        let epoch = inner.epoch as EpochId;
        let blocks = self
            .handle_fetch_blocks(block_refs, highest_accepted_rounds, epoch)
            .await
            .map_err(|e| tonic::Status::internal(format!("{e:?}")))?;
        let responses: std::vec::IntoIter<Result<FetchBlocksResponse, tonic::Status>> =
            chunk_blocks(blocks, MAX_FETCH_RESPONSE_BYTES)
                .into_iter()
                .map(|blocks| Ok(FetchBlocksResponse { blocks }))
                .collect::<Vec<_>>()
                .into_iter();
        let stream = iter(responses);
        Ok(Response::new(stream))
    }

    async fn fetch_commits(
        &self,
        request: Request<FetchCommitsRequest>,
    ) -> Result<Response<FetchCommitsResponse>, tonic::Status> {
        let request = request.into_inner();
        let (commits, certifier_blocks) = self
            .handle_fetch_commits((request.start..=request.end).into())
            .await
            .map_err(|e| tonic::Status::internal(format!("{e:?}")))?;
        let commits = commits
            .into_iter()
            .map(|c| c.serialized().clone())
            .collect();
        let certifier_blocks = certifier_blocks
            .into_iter()
            .map(|b| b.serialized().clone())
            .collect();
        Ok(Response::new(FetchCommitsResponse {
            commits,
            certifier_blocks,
        }))
    }
}

fn chunk_blocks(blocks: Vec<Bytes>, chunk_limit: usize) -> Vec<Vec<Bytes>> {
    let mut chunks = vec![];
    let mut chunk = vec![];
    let mut chunk_size = 0;
    for block in blocks {
        let block_size = block.len();
        if !chunk.is_empty() && chunk_size + block_size > chunk_limit {
            chunks.push(chunk);
            chunk = vec![];
            chunk_size = 0;
        }
        chunk.push(block);
        chunk_size += block_size;
    }
    if !chunk.is_empty() {
        chunks.push(chunk);
    }
    chunks
}
