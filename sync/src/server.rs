use parking_lot::RwLock;
use rand::seq::IteratorRandom;
use std::sync::Arc;
use std::{collections::HashMap, result::Result};
use tokio::sync::mpsc;
use tonic::{async_trait, Request, Response, Status};
use tracing::{debug, info};
use types::checkpoints::VerifiedCheckpoint;
use types::sync::channel_manager::PeerInfo;

use types::sync::{
    GetCheckpointAvailabilityRequest, GetCheckpointAvailabilityResponse,
    GetCheckpointContentsRequest, GetCheckpointContentsResponse, GetCheckpointSummaryRequest,
    GetCheckpointSummaryResponse, GetKnownPeersRequest, GetKnownPeersResponse,
    PushCheckpointSummaryRequest, PushCheckpointSummaryResponse, SignedNodeInfo,
};

use types::storage::write_store::WriteStore;

use crate::state_sync::{PeerHeights, StateSyncMessage};
use crate::tonic_gen::p2p_server::P2p;

use crate::discovery::DiscoveryState;

const MAX_PEERS_TO_SEND: usize = 200;

pub struct P2pService<S> {
    pub discovery_state: Arc<RwLock<DiscoveryState>>,
    pub store: S,
    pub peer_heights: Arc<RwLock<PeerHeights>>,
    pub state_sync_sender: mpsc::WeakSender<StateSyncMessage>,
    pub discovery_sender: mpsc::Sender<SignedNodeInfo>,
}

#[async_trait]
impl<S> P2p for P2pService<S>
where
    S: WriteStore + Send + Sync + 'static,
{
    async fn push_checkpoint_summary(
        &self,
        request: Request<PushCheckpointSummaryRequest>,
    ) -> Result<Response<PushCheckpointSummaryResponse>, Status> {
        let Some(peer_id) = request.extensions().get::<PeerInfo>().map(|p| p.peer_id) else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };

        let PushCheckpointSummaryRequest { checkpoint } = request.into_inner();

        if !self
            .peer_heights
            .write()
            .update_peer_info(peer_id, checkpoint.clone(), None)
        {
            return Ok(Response::new(PushCheckpointSummaryResponse {
                _unused: true,
            }));
        }

        let highest_verified_checkpoint = *self
            .store
            .get_highest_verified_checkpoint()
            .map_err(|e| Status::internal(e.to_string()))?
            .sequence_number();

        // If this checkpoint is higher than our highest verified checkpoint notify the
        // event loop to potentially sync it
        if *checkpoint.sequence_number() > highest_verified_checkpoint {
            if let Some(sender) = self.state_sync_sender.upgrade() {
                sender.send(StateSyncMessage::StartSyncJob).await.unwrap();
            }
        }

        Ok(Response::new(PushCheckpointSummaryResponse {
            _unused: true,
        }))
    }

    async fn get_checkpoint_summary(
        &self,
        request: Request<GetCheckpointSummaryRequest>,
    ) -> Result<Response<GetCheckpointSummaryResponse>, Status> {
        let checkpoint = match request.into_inner() {
            GetCheckpointSummaryRequest::Latest => self
                .store
                .get_highest_synced_checkpoint()
                .map(Some)
                .map_err(|e| Status::internal(e.to_string()))?,
            GetCheckpointSummaryRequest::ByDigest(digest) => {
                self.store.get_checkpoint_by_digest(&digest)
            }
            GetCheckpointSummaryRequest::BySequenceNumber(sequence_number) => self
                .store
                .get_checkpoint_by_sequence_number(sequence_number),
        }
        .map(VerifiedCheckpoint::into_inner);

        Ok(Response::new(GetCheckpointSummaryResponse { checkpoint }))
    }

    async fn get_checkpoint_availability(
        &self,
        _request: Request<GetCheckpointAvailabilityRequest>,
    ) -> Result<Response<GetCheckpointAvailabilityResponse>, Status> {
        let highest_synced_checkpoint = self
            .store
            .get_highest_synced_checkpoint()
            .map_err(|e| Status::internal(e.to_string()))
            .map(VerifiedCheckpoint::into_inner)?;
        let lowest_available_checkpoint = self
            .store
            .get_lowest_available_checkpoint()
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(GetCheckpointAvailabilityResponse {
            highest_synced_checkpoint,
            lowest_available_checkpoint,
        }))
    }

    async fn get_checkpoint_contents(
        &self,
        request: Request<GetCheckpointContentsRequest>,
    ) -> Result<Response<GetCheckpointContentsResponse>, Status> {
        let digest = request.into_inner().digest;
        let contents = self.store.get_full_checkpoint_contents(None, &digest);
        Ok(Response::new(GetCheckpointContentsResponse { contents }))
    }

    async fn get_known_peers(
        &self,
        request: Request<GetKnownPeersRequest>,
    ) -> Result<Response<GetKnownPeersResponse>, tonic::Status> {
        let state = self.discovery_state.read();
        let own_info = state
            .our_info
            .clone()
            .ok_or_else(|| tonic::Status::internal("own_info has not been initialized yet"))?;

        let their_info = request.into_inner().own_info;

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

        if let Err(e) = self.discovery_sender.try_send(their_info.clone()) {
            debug!("Failed to send their info to connect back: {}", e);
        } else {
            info!("Sent their info to connect back: {}", their_info.peer_id);
        }
        info!("Sending known peers and our info {}", own_info.peer_id);

        Ok(Response::new(GetKnownPeersResponse {
            own_info,
            known_peers,
        }))
    }
}
