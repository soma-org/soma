use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};
use tokio::sync::oneshot;
use tonic::{transport::Server, Request, Response};
use tower_http::add_extension::AddExtensionLayer;

use crate::{
    crypto::keys::NetworkKeyPair,
    error::{ShardError, ShardResult},
    networking::messaging::{
        to_socket_addr, tonic::PeerInfo, tonic_gen::encoder_service_server::EncoderServiceServer,
    },
    types::{
        context::{EncoderContext, NetworkingContext},
        network_committee::NetworkIdentityIndex,
        shard_input::VerifiedSignedShardInput,
        shard_selection::VerifiedSignedShardSelection,
    },
};
use tracing::info;

use crate::networking::messaging::tonic_gen::encoder_service_client::EncoderServiceClient;

use super::{
    channel_pool::{Channel, ChannelPool},
    leader_tonic_service::LeaderTonicClient,
    tonic_gen::encoder_service_server::EncoderService,
    EncoderNetworkClient, EncoderNetworkManager, EncoderNetworkService,
};

// Implements Tonic RPC client for Consensus.
pub(crate) struct EncoderTonicClient<N: NetworkingContext> {
    network_keypair: NetworkKeyPair,
    channel_pool: Arc<ChannelPool<N>>,
}

impl<N: NetworkingContext> EncoderTonicClient<N> {
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
    ) -> ShardResult<EncoderServiceClient<Channel>> {
        // let config = &self.context.parameters.tonic;
        let channel = self.channel_pool.get_channel(peer, timeout).await?;
        Ok(EncoderServiceClient::new(channel))
        // .max_encoding_message_size(config.message_size_limit)
        // .max_decoding_message_size(config.message_size_limit))
    }
}

#[async_trait]
impl<N: NetworkingContext> EncoderNetworkClient for EncoderTonicClient<N> {
    async fn send_input(
        &self,
        peer: NetworkIdentityIndex,
        input: &VerifiedSignedShardInput,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendInputRequest {
            input: input.serialized().clone(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_input(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("send_input failed: {e:?}")))?;
        Ok(())
    }

    async fn send_selection(
        &self,
        peer: NetworkIdentityIndex,
        selection: &VerifiedSignedShardSelection,
        timeout: Duration,
    ) -> ShardResult<()> {
        let mut request = Request::new(SendSelectionRequest {
            selection: selection.serialized().clone(),
        });
        request.set_timeout(timeout);
        self.get_client(peer, timeout)
            .await?
            .send_selection(request)
            .await
            .map_err(|e| ShardError::NetworkRequest(format!("send_input failed: {e:?}")))?;
        Ok(())
    }
}

/// Proxies Tonic requests to NetworkService with actual handler implementation.
struct EncoderTonicServiceProxy<S: EncoderNetworkService> {
    context: Arc<EncoderContext>,
    service: Arc<S>,
}

impl<S: EncoderNetworkService> EncoderTonicServiceProxy<S> {
    fn new(context: Arc<EncoderContext>, service: Arc<S>) -> Self {
        Self { context, service }
    }
}

#[async_trait]
impl<S: EncoderNetworkService> EncoderService for EncoderTonicServiceProxy<S> {
    async fn send_input(
        &self,
        request: Request<SendInputRequest>,
    ) -> Result<Response<SendInputResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let input = request.into_inner().input;
        self.service
            .handle_send_input(peer_index, input)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")));

        Ok(Response::new(SendInputResponse {}))
    }

    async fn send_selection(
        &self,
        request: Request<SendSelectionRequest>,
    ) -> Result<Response<SendSelectionResponse>, tonic::Status> {
        let Some(peer_index) = request
            .extensions()
            .get::<PeerInfo>()
            .map(|p| p.network_index)
        else {
            return Err(tonic::Status::internal("PeerInfo not found"));
        };
        let selection = request.into_inner().selection;
        self.service
            .handle_send_selection(peer_index, selection)
            .await
            .map_err(|e| tonic::Status::invalid_argument(format!("{e:?}")));
        Ok(Response::new(SendSelectionResponse {}))
    }
}

pub struct EncoderTonicManager {
    context: Arc<EncoderContext>,
    leader_client: Arc<LeaderTonicClient<EncoderContext>>,
    // encoder_client: Arc<EncoderTonicClient>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl EncoderTonicManager {
    pub fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        Self {
            context: context.clone(),
            leader_client: Arc::new(LeaderTonicClient::new(context, network_keypair)),
            // encoder_client: Arc::new(EncoderTonicClient::new(context, network_keypair)),
            shutdown_tx: None,
        }
    }
}

impl<S: EncoderNetworkService> EncoderNetworkManager<S> for EncoderTonicManager {
    type Client = LeaderTonicClient<EncoderContext>;

    fn new(context: Arc<EncoderContext>, network_keypair: NetworkKeyPair) -> Self {
        EncoderTonicManager::new(context, network_keypair)
    }

    fn leader_client(&self) -> Arc<Self::Client> {
        self.leader_client.clone()
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
            EncoderServiceServer::new(EncoderTonicServiceProxy::new(self.context.clone(), service));
        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        tokio::spawn(async move {
            // let leader_service = Server::builder().add_service(LeaderShardServiceServer::new(svc));

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

/// `SendShardInputRequest` contains a serialized signed shard input type.
/// A leader calls `SendInput` for each member of the shard.
#[derive(Clone, prost::Message)]
pub(crate) struct SendInputRequest {
    /// Signed but not verified shard input
    /// Note: make changes in versioned struct, not here
    #[prost(bytes = "bytes", tag = "1")]
    input: Bytes,
}

/// Empty response
#[derive(Clone, prost::Message)]
pub(crate) struct SendInputResponse {}

// ////////////////////////////////////////////////////////////////////

/// Contains a serialized shard selection. This is also called by the leader
/// hence the type definition existing for the encoder. It is also possible for
/// a encoder to reuse the signed message from the leader to broadcast to the
/// shard members if the leader fails after delivering a few signed shard selections.
#[derive(Clone, prost::Message)]
pub(crate) struct SendSelectionRequest {
    /// signed shard selection, can be resent by shard memebers if signature matches
    /// Note: make changes in versioned struct, not here
    #[prost(bytes = "bytes", tag = "1")]
    selection: Bytes,
}

/// Empty response
#[derive(Clone, prost::Message)]
pub(crate) struct SendSelectionResponse {}
