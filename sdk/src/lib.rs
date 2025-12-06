use futures::Stream;
use rpc::proto::soma::ListOwnedObjectsRequest;
use rpc::{api::client::Client, proto::soma::ledger_service_client::LedgerServiceClient};
use std::pin::Pin;
use std::{sync::Arc, time::Duration};
use tokio::io::{AsyncRead, AsyncSeek};
use tokio::sync::RwLock;
use types::object::{Object, ObjectID, Version};
use types::transaction::Transaction;
use url::Url;
use uuid::Uuid;

pub mod client_config;
pub mod error;
pub mod wallet_context;
pub const SOMA_LOCAL_NETWORK_URL: &str = "http://127.0.0.1:9000";
pub const SOMA_LOCAL_NETWORK_URL_0: &str = "http://0.0.0.0:9000";
pub const SOMA_DEVNET_URL: &str = "https://fullnode.devnet.soma.org:443";
pub const SOMA_TESTNET_URL: &str = "https://fullnode.testnet.soma.org:443";
pub const SOMA_MAINNET_URL: &str = "https://fullnode.mainnet.soma.org:443";
// TODO: define default object storage urls for public RPCs and use it in the builder

/// Builder for configuring a SomaClient
pub struct SomaClientBuilder {
    request_timeout: Duration,
    tus_chunk_size: Option<usize>,
}

impl Default for SomaClientBuilder {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(60),
            tus_chunk_size: None,
        }
    }
}

impl SomaClientBuilder {
    /// Set the request timeout
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
    /// Build the client with RPC and object storage URLs
    pub async fn build(
        self,
        rpc_url: impl AsRef<str>,
        object_storage_url: impl AsRef<str>,
    ) -> Result<SomaClient, error::Error> {
        // Create gRPC client
        let client = Client::new(rpc_url.as_ref())
            .map_err(|e| error::Error::ClientInitError(e.to_string()))?;

        Ok(SomaClient {
            inner: Arc::new(RwLock::new(client)),
        })
    }

    /// Build a client for the local network with default addresses
    pub async fn build_localnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_LOCAL_NETWORK_URL, "http://127.0.0.1:8080")
            .await
    }

    /// Build a client for devnet with default addresses
    pub async fn build_devnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_DEVNET_URL, "http://fullnode.devnet.soma.org:8080")
            .await
    }

    /// Build a client for testnet with default addresses
    pub async fn build_testnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_TESTNET_URL, "http://fullnode.testnet.soma.org:8080")
            .await
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
    ) -> Result<rpc::api::client::TransactionExecutionResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client
            .execute_transaction_and_wait_for_checkpoint(transaction, timeout)
            .await
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
}
