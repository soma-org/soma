use std::{collections::HashMap, sync::Arc, time::Duration};

use rpc::api::client::{AuthInterceptor, Client, Result, TransactionExecutionResponse};
use types::transaction::Transaction;

pub mod client_config;
pub mod error;
pub mod wallet_context;
pub const SOMA_LOCAL_NETWORK_URL: &str = "http://127.0.0.1:9000";
pub const SOMA_LOCAL_NETWORK_URL_0: &str = "http://0.0.0.0:9000";
pub const SOMA_DEVNET_URL: &str = "https://fullnode.devnet.soma.org:443";
pub const SOMA_TESTNET_URL: &str = "https://fullnode.testnet.soma.org:443";
pub const SOMA_MAINNET_URL: &str = "https://fullnode.mainnet.soma.org:443";

/// Builder for configuring a SomaClient
pub struct SomaClientBuilder {
    request_timeout: Duration,
    auth: Option<AuthInterceptor>,
}

impl Default for SomaClientBuilder {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(60),
            auth: None,
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

    /// Set bearer token authentication
    pub fn bearer_token(mut self, token: impl AsRef<str>) -> Self {
        self.auth = Some(AuthInterceptor::bearer(token.as_ref()));
        self
    }

    /// Build the client with a custom URL
    pub async fn build(self, url: impl AsRef<str>) -> Result<SomaClient, error::Error> {
        let mut client =
            Client::new(url.as_ref()).map_err(|e| error::Error::ClientInitError(e.to_string()))?;

        if let Some(auth) = self.auth {
            client = client.with_auth(auth);
        }

        Ok(SomaClient {
            inner: Arc::new(client),
        })
    }

    /// Build a client for the local network
    pub async fn build_localnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_LOCAL_NETWORK_URL).await
    }

    /// Build a client for devnet
    pub async fn build_devnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_DEVNET_URL).await
    }

    /// Build a client for testnet
    pub async fn build_testnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_TESTNET_URL).await
    }

    /// Build a client for mainnet
    pub async fn build_mainnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_MAINNET_URL).await
    }
}

/// The main Soma client for interacting with the Soma network via gRPC
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
}
