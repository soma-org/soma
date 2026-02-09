use bytes::Bytes;
use futures::Stream;
use rpc::api::ServerVersion;
use rpc::proto::soma::ListOwnedObjectsRequest;
use rpc::{api::client::Client, proto::soma::ledger_service_client::LedgerServiceClient};
use std::pin::Pin;
use std::{sync::Arc, time::Duration};
use tokio::io::{AsyncRead, AsyncSeek};
use tokio::sync::RwLock;
use types::base::SomaAddress;
use types::effects::TransactionEffects;
use types::object::{Object, ObjectID, Version};
use types::transaction::Transaction;
use url::Url;
use uuid::Uuid;

use crate::error::SomaRpcResult;

pub mod client_config;
pub mod error;
pub mod proxy_client;
pub mod transaction_builder;
pub mod wallet_context;

// TODO: define these when public rpcs are finalized
pub const SOMA_LOCAL_NETWORK_URL: &str = "http://127.0.0.1:9000";
pub const SOMA_LOCAL_NETWORK_URL_0: &str = "http://0.0.0.0:9000";
pub const SOMA_DEVNET_URL: &str = "https://fullnode.devnet.soma.org:443";
pub const SOMA_TESTNET_URL: &str = "https://fullnode.testnet.soma.org:443";
pub const SOMA_MAINNET_URL: &str = "https://fullnode.mainnet.soma.org:443";

/// Builder for configuring a SomaClient
pub struct SomaClientBuilder {
    request_timeout: Duration,
}

impl Default for SomaClientBuilder {
    fn default() -> Self {
        Self { request_timeout: Duration::from_secs(60) }
    }
}

impl SomaClientBuilder {
    /// Set the request timeout
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
    /// Build the client with RPC and object storage URLs
    pub async fn build(self, rpc_url: impl AsRef<str>) -> Result<SomaClient, error::Error> {
        // Create gRPC client
        let client = Client::new(rpc_url.as_ref())
            .map_err(|e| error::Error::ClientInitError(e.to_string()))?;

        Ok(SomaClient { inner: Arc::new(RwLock::new(client)) })
    }

    /// Build a client for the local network with default addresses
    pub async fn build_localnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_LOCAL_NETWORK_URL).await
    }

    /// Build a client for devnet with default addresses
    pub async fn build_devnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_DEVNET_URL).await
    }

    /// Build a client for testnet with default addresses
    pub async fn build_testnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_TESTNET_URL).await
    }
}

/// The main Soma client for interacting with the Soma network via gRPC and TUS
#[derive(Clone)]
pub struct SomaClient {
    inner: Arc<RwLock<Client>>,
}

impl SomaClient {
    /// Create a new client builder
    pub fn builder() -> SomaClientBuilder {
        SomaClientBuilder::default()
    }

    /// Execute a transaction
    pub async fn execute_transaction(
        &self,
        transaction: &types::transaction::Transaction,
    ) -> Result<rpc::api::client::TransactionExecutionResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.execute_transaction(transaction).await
    }

    /// Execute a transaction
    pub async fn execute_transaction_and_wait_for_checkpoint(
        &self,
        transaction: &types::transaction::Transaction,
        timeout: Duration,
    ) -> Result<rpc::api::client::TransactionExecutionResponseWithCheckpoint, tonic::Status> {
        let mut client = self.inner.write().await;
        client.execute_transaction_and_wait_for_checkpoint(transaction, timeout).await
    }

    /// Subscribe to checkpoints
    pub async fn subscribe_checkpoints(
        &self,
        request: impl tonic::IntoRequest<rpc::proto::soma::SubscribeCheckpointsRequest>,
    ) -> Result<tonic::Streaming<rpc::proto::soma::SubscribeCheckpointsResponse>, tonic::Status>
    {
        let mut client = self.inner.write().await;
        client.subscribe_checkpoints(request).await
    }

    /// Get an object by ID
    pub async fn get_object(&self, object_id: ObjectID) -> Result<Object, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_object(object_id).await
    }

    /// Get an object by ID and version
    pub async fn get_object_with_version(
        &self,
        object_id: ObjectID,
        version: Version,
    ) -> Result<Object, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_object_with_version(object_id, version).await
    }

    /// Stream objects owned by an address
    ///
    /// Returns a stream that automatically handles pagination.
    pub async fn list_owned_objects(
        &self,
        request: impl tonic::IntoRequest<ListOwnedObjectsRequest>,
    ) -> Pin<Box<dyn Stream<Item = Result<Object, tonic::Status>> + Send + 'static>> {
        self.inner.read().await.clone().list_owned_objects(request)
    }

    /// Get the chain identifier from the network
    pub async fn get_chain_identifier(&self) -> Result<String, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_chain_identifier().await
    }

    /// Get the server version from the network
    pub async fn get_server_version(&self) -> Result<String, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_server_version().await
    }

    /// Verifies if the API version matches the server version and returns an error if they do not match.
    pub async fn check_api_version(&self) -> SomaRpcResult<()> {
        let server_version =
            self.get_server_version().await.map_err(|e| crate::error::Error::RpcError(e.into()))?;
        let client_version = env!("CARGO_PKG_VERSION");
        if server_version != client_version {
            return Err(crate::error::Error::ServerVersionMismatch {
                client_version: client_version.to_string(),
                server_version,
            });
        };
        Ok(())
    }

    pub async fn get_latest_system_state(
        &self,
    ) -> Result<types::system_state::SystemState, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_latest_system_state().await
    }

    /// List targets with optional filtering by status and epoch.
    ///
    /// Returns a paginated list of targets. Use `status_filter` to filter by "open", "filled", or "claimed".
    /// Use `epoch_filter` to filter by generation epoch.
    pub async fn list_targets(
        &self,
        request: impl tonic::IntoRequest<rpc::proto::soma::ListTargetsRequest>,
    ) -> Result<rpc::proto::soma::ListTargetsResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.list_targets(request).await
    }

    /// Get a challenge by ID.
    ///
    /// Returns the challenge details including status, challenger, target, and audit data.
    pub async fn get_challenge(
        &self,
        request: impl tonic::IntoRequest<rpc::proto::soma::GetChallengeRequest>,
    ) -> Result<rpc::proto::soma::GetChallengeResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_challenge(request).await
    }

    /// List challenges with optional filtering by status, epoch, and target.
    ///
    /// Returns a paginated list of challenges. Use `status_filter` to filter by "pending" or "resolved".
    /// Use `epoch_filter` to filter by challenge epoch. Use `target_filter` to filter by target ID.
    pub async fn list_challenges(
        &self,
        request: impl tonic::IntoRequest<rpc::proto::soma::ListChallengesRequest>,
    ) -> Result<rpc::proto::soma::ListChallengesResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.list_challenges(request).await
    }

    /// Get epoch information
    pub async fn get_epoch(
        &self,
        epoch: Option<u64>,
    ) -> Result<rpc::proto::soma::GetEpochResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_epoch(epoch).await
    }

    /// Get the current protocol version from the network
    pub async fn get_protocol_version(&self) -> Result<u64, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_protocol_version().await
    }

    /// Simulate a transaction without executing it (no signature required)
    pub async fn simulate_transaction(
        &self,
        tx_data: &types::transaction::TransactionData,
    ) -> Result<rpc::api::client::SimulationResult, tonic::Status> {
        let mut client = self.inner.write().await;
        client.simulate_transaction(tx_data).await
    }

    /// Get a transaction by its digest
    pub async fn get_transaction(
        &self,
        digest: types::digests::TransactionDigest,
    ) -> Result<rpc::api::client::TransactionQueryResult, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_transaction(digest).await
    }
}
