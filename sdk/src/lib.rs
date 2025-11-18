use std::{collections::HashMap, sync::Arc, time::Duration};

use rpc::{
    api::client::{AuthInterceptor, Client, Result, TransactionExecutionResponse},
    proto::soma::{
        ledger_service_client::LedgerServiceClient, live_data_service_client::LiveDataServiceClient,
    },
};
use tokio::io::{AsyncRead, AsyncSeek};
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
    auth: Option<AuthInterceptor>,
    tus_chunk_size: Option<usize>,
}

impl Default for SomaClientBuilder {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(60),
            auth: None,
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

    /// Set basic auth credentials
    pub fn basic_auth(mut self, username: impl AsRef<str>, password: impl AsRef<str>) -> Self {
        self.auth = Some(AuthInterceptor::basic(
            username.as_ref(),
            Some(password.as_ref()),
        ));
        self
    }

    /// Build the client with RPC and object storage URLs
    pub async fn build(
        self,
        rpc_url: impl AsRef<str>,
        object_storage_url: impl AsRef<str>,
    ) -> Result<SomaClient, error::Error> {
        // Create gRPC client
        let mut client = Client::new(rpc_url.as_ref())
            .map_err(|e| error::Error::ClientInitError(e.to_string()))?;

        if let Some(auth) = self.auth.clone() {
            client = client.with_auth(auth);
        }

        Ok(SomaClient {
            inner: Arc::new(client),
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
    inner: Arc<Client>,
}

impl SomaClient {
    /// Create a new client builder
    pub fn builder() -> SomaClientBuilder {
        SomaClientBuilder::default()
    }

    /// Get the underlying gRPC client
    pub fn inner(&self) -> &Client {
        &self.inner
    }

    /// Execute a transaction
    pub async fn execute_transaction(
        &self,
        transaction: &Transaction,
    ) -> Result<TransactionExecutionResponse> {
        self.inner.execute_transaction(transaction).await
    }

    /// Get the live data service client
    pub fn live_data_client(
        &self,
    ) -> LiveDataServiceClient<
        tonic::service::interceptor::InterceptedService<tonic::transport::Channel, AuthInterceptor>,
    > {
        self.inner.live_data_client()
    }

    /// Get the ledger service client
    pub fn ledger_client(
        &self,
    ) -> LedgerServiceClient<
        tonic::service::interceptor::InterceptedService<tonic::transport::Channel, AuthInterceptor>,
    > {
        self.inner.raw_client()
    }
}
