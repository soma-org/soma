use parking_lot::RwLock;
use rand::seq::IteratorRandom;
use std::sync::Arc;
use std::{collections::HashMap, result::Result};
use tonic::{async_trait, Request, Response};
use tracing::info;
use types::discovery::{GetKnownPeersRequest, GetKnownPeersResponse};

use crate::tonic_gen::p2p_server::P2p;

use super::DiscoveryState;

const MAX_PEERS_TO_SEND: usize = 200;

pub struct P2pService {
    pub discovery_state: Arc<RwLock<DiscoveryState>>,
}

#[async_trait]
impl P2p for P2pService {
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
            // prefer returning peers that we are connected to as they are known-good
            let mut known_peers = state
                .connected_peers
                .keys()
                .filter_map(|peer_id| state.known_peers.get(peer_id))
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
}
