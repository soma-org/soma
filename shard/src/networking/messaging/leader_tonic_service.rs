use async_trait::async_trait;
use bytes::Bytes;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tonic::transport::Server;
use tonic::{Request, Response};
use tower_http::add_extension::AddExtensionLayer;

use crate::crypto::keys::NetworkKeyPair;
use crate::error::{ShardError, ShardResult};
use crate::networking::messaging::channel_pool::{Channel, ChannelPool};
use crate::networking::messaging::to_socket_addr;
use crate::networking::messaging::tonic::PeerInfo;
use crate::networking::messaging::tonic_gen::leader_service_server::LeaderServiceServer;
use crate::types::context::{LeaderContext, NetworkingContext};
use crate::types::network_committee::NetworkIdentityIndex;
use tracing::info;

use super::encoder_tonic_service::EncoderTonicClient;
use super::tonic_gen::leader_service_client::LeaderServiceClient;
use super::tonic_gen::leader_service_server::LeaderService;
use super::{LeaderNetworkClient, LeaderNetworkManager, LeaderNetworkService};
use crate::types::{
    shard_commit::VerifiedSignedShardCommit, shard_endorsement::VerifiedSignedShardEndorsement,
};

// Implements Tonic RPC client for Consensus.
pub(crate) struct LeaderTonicClient<N: NetworkingContext> {
    network_keypair: NetworkKeyPair,
    channel_pool: Arc<ChannelPool<N>>,
}

impl<N> LeaderTonicClient<N>
where
    N: NetworkingContext,
{
    pub(crate) fn new(context: Arc<N>, network_keypair: NetworkKeyPair) -> Self {
        Self {
            network_keypair,
            channel_pool: Arc::new(ChannelPool::new(context)),
        }
    }

    async fn get_client(
        &self,
        peer: NetworkIdentityIndex,
        timeout: Duration,
    ) -> ShardResult<LeaderServiceClient<Channel>> {
        // let config = &self.context.parameters.tonic;
        let channel = self.channel_pool.get_channel(peer, timeout).await?;
        Ok(LeaderServiceClient::new(channel))
        // .max_encoding_message_size(config.message_size_limit)
        // .max_decoding_message_size(config.message_size_limit))
    }
}

#[async_trait]
impl<N> LeaderNetworkClient for LeaderTonicClient<N>
where
    N: NetworkingContext,
{
    async fn send_commit(
        &self,
        peer: NetworkIdentityIndex,
        commit: &VerifiedSignedShardCommit,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendCommitRequest {
            commit: commit.serialized().clone(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_commit(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("send_commit failed: {e:?}")))?;
        Ok(())
    }

    async fn send_endorsement(
        &self,
        peer: NetworkIdentityIndex,
        endorsement: &VerifiedSignedShardEndorsement,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendEndorsementRequest {
            endorsement: endorsement.serialized().clone(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_endorsement(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("send_endorsement failed: {e:?}")))?;
        Ok(())
    }
}

/// Proxies Tonic requests to NetworkService with actual handler implementation.
struct LeaderTonicServiceProxy<S: LeaderNetworkService> {
    context: Arc<LeaderContext>,
    service: Arc<S>,
}

impl<S: LeaderNetworkService> LeaderTonicServiceProxy<S> {
    fn new(context: Arc<LeaderContext>, service: Arc<S>) -> Self {
        Self { context, service }
    }
}

#[async_trait]
impl<S: LeaderNetworkService> LeaderService for LeaderTonicServiceProxy<S> {
    async fn send_commit(
        &self,
        request: Request<SendCommitRequest>,
    ) -> Result<Response<SendCommitResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let commit = request.into_inner().commit;
        self.service
            .handle_send_commit(peer_index, commit)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;

        Ok(Response::new(SendCommitResponse {}))
    }

    async fn send_endorsement(
        &self,
        request: Request<SendEndorsementRequest>,
    ) -> Result<Response<SendEndorsementResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let endorsement = request.into_inner().endorsement;
        self.service
            .handle_send_endorsement(peer_index, endorsement)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")))?;
        Ok(Response::new(SendEndorsementResponse {}))
    }
}

pub struct LeaderTonicManager {
    context: Arc<LeaderContext>,
    encoder_client: Arc<EncoderTonicClient<LeaderContext>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl LeaderTonicManager {
    /// creates a new leader tonic manager
    pub fn new(context: Arc<LeaderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self {
            context: context.clone(),
            encoder_client: Arc::new(EncoderTonicClient::new(context, network_keypair)),
            shutdown_tx: None,
        }
    }
}

impl<S: LeaderNetworkService> LeaderNetworkManager<S> for LeaderTonicManager {
    type Client = EncoderTonicClient<LeaderContext>;

    fn new(context: Arc<LeaderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self::new(context, network_keypair)
    }

    fn encoder_client(&self) -> Arc<Self::Client> {
        self.encoder_client.clone()
    }

    async fn start(&mut self, service: Arc<S>) {
        let network_identity = self
            .context
            .network_committee
            .identity(self.context.own_network_index);
        let own_network_index = self.context.own_network_index;
        // By default, bind to the unspecified address to allow the actual address to be assigned.
        // But bind to localhost if it is requested.
        let own_address = if network_identity.address.is_localhost_ip() {
            network_identity.address.clone()
        } else {
            network_identity.address.with_zero_ip()
        };
        let own_address = to_socket_addr(&own_address).unwrap();
        let svc =
            LeaderServiceServer::new(LeaderTonicServiceProxy::new(self.context.clone(), service));
        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        tokio::spawn(async move {
            // let leader_service = Server::builder().add_service(LeaderServiceServer::new(svc));

            let tower_layer = tower::ServiceBuilder::new()
                .layer(AddExtensionLayer::new(PeerInfo {
                    network_index: own_network_index,
                }))
                .into_inner();

            Server::builder()
                .layer(tower_layer)
                .add_service(svc)
                .serve_with_shutdown(own_address, async {
                    rx.await.ok();
                })
                .await
                .unwrap();
        });
        info!("Binding tonic server to address {:?}", own_address);
    }

    async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

// ////////////////////////////////////////////////////////////////////

/// This is called by the shard encoders as quickly as possible. They are incentivized to be faster than
/// the selection number for the shard, since the default leader behavior is to select the first
/// selection number of commits.
#[derive(Clone, prost::Message)]
pub(crate) struct SendCommitRequest {
    /// Signed shard commit
    /// Note: make changes in versioned struct, not here
    #[prost(bytes = "bytes", tag = "1")]
    commit: Bytes,
}

/// Empty response
#[derive(Clone, prost::Message)]
pub(crate) struct SendCommitResponse {}

// ////////////////////////////////////////////////////////////////////

/// This is called by the shard members, as to avoid being slashed.
/// It is assumed currently that the leader will submit the finalization transaction
/// that locks in the winning embedding. Currently this is susceptible to liveliness faults
/// if the leader disappears. May want to consider implementing a fallback?
#[derive(Clone, prost::Message)]
pub(crate) struct SendEndorsementRequest {
    /// Signed shard endorsement
    /// Note: make changes in versioned struct, not here
    #[prost(bytes = "bytes", tag = "1")]
    endorsement: Bytes,
}

/// Empty response
#[derive(Clone, prost::Message)]
pub(crate) struct SendEndorsementResponse {}
