use parking_lot::RwLock;
use rand::seq::IteratorRandom;
use std::sync::Arc;
use std::{collections::HashMap, result::Result};
use tokio::sync::mpsc;
use tonic::{async_trait, Request, Response, Status};
use tracing::info;
use types::digests::CommitContentsDigest;
use types::discovery::{GetKnownPeersRequest, GetKnownPeersResponse};
use types::p2p::channel_manager::PeerInfo;
use types::state_sync::{
    CertifiedCommitSummary, FullCommitContents, GetCommitAvailabilityRequest,
    GetCommitAvailabilityResponse, GetCommitSummaryRequest, PushCommitSummaryResponse,
    VerifiedCommitSummary,
};
use types::storage::write_store::WriteStore;

use crate::state_sync::{PeerHeights, StateSyncMessage};
use crate::tonic_gen::p2p_server::P2p;

use crate::discovery::{now_unix, DiscoveryState};

const MAX_PEERS_TO_SEND: usize = 200;

pub struct P2pService<S> {
    pub discovery_state: Arc<RwLock<DiscoveryState>>,
    pub store: S,
    pub peer_heights: Arc<RwLock<PeerHeights>>,
    pub sender: mpsc::WeakSender<StateSyncMessage>,
}

#[async_trait]
impl<S> P2p for P2pService<S>
where
    S: WriteStore + Send + Sync + 'static,
{
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

    async fn push_commit_summary(
        &self,
        request: Request<CertifiedCommitSummary>,
    ) -> Result<Response<PushCommitSummaryResponse>, tonic::Status> {
        let Some(peer_id) = request.extensions().get::<PeerInfo>().map(|p| p.peer_id) else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };

        let commit = request.into_inner();
        if !self
            .peer_heights
            .write()
            .update_peer_info(peer_id, commit.clone(), None)
        {
            return Ok(Response::new(PushCommitSummaryResponse {
                timestamp_ms: now_unix(),
            }));
        }

        let highest_verified_commit = *self
            .store
            .get_highest_verified_commit()
            .map_err(|e| Status::internal(e.to_string()))?
            .index();

        // If this commit is higher than our highest verified commit notify the
        // event loop to potentially sync it
        if *commit.index() > highest_verified_commit {
            if let Some(sender) = self.sender.upgrade() {
                sender.send(StateSyncMessage::StartSyncJob).await.unwrap();
            }
        }

        Ok(Response::new(PushCommitSummaryResponse {
            timestamp_ms: now_unix(),
        }))
    }

    async fn get_commit_summary(
        &self,
        request: Request<GetCommitSummaryRequest>,
    ) -> Result<Response<Option<CertifiedCommitSummary>>, tonic::Status> {
        let commit = match request.into_inner() {
            GetCommitSummaryRequest::Latest => self
                .store
                .get_highest_synced_commit()
                .map(Some)
                .map_err(|e| Status::internal(e.to_string()))?,
            GetCommitSummaryRequest::ByDigest(digest) => self.store.get_commit_by_digest(&digest),
            GetCommitSummaryRequest::ByIndex(index) => self.store.get_commit_by_index(index),
        }
        .map(VerifiedCommitSummary::into_inner);

        Ok(Response::new(commit))
    }

    async fn get_commit_availability(
        &self,
        _request: Request<GetCommitAvailabilityRequest>,
    ) -> Result<Response<GetCommitAvailabilityResponse>, tonic::Status> {
        let highest_synced_commit = self
            .store
            .get_highest_synced_commit()
            .map_err(|e| Status::internal(e.to_string()))
            .map(VerifiedCommitSummary::into_inner)?;
        let lowest_available_commit = self
            .store
            .get_lowest_available_commit()
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(GetCommitAvailabilityResponse {
            highest_synced_commit,
            lowest_available_commit,
        }))
    }

    async fn get_commit_contents(
        &self,
        request: Request<CommitContentsDigest>,
    ) -> Result<Response<Option<FullCommitContents>>, tonic::Status> {
        let contents = self.store.get_full_commit_contents(&request.into_inner());
        Ok(Response::new(contents))
    }
}
