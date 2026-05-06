// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;
use rand::Rng;
use rpc::api::client::Client;
use rpc::proto::soma::ListOwnedObjectsRequest;
use tokio::sync::{Mutex, RwLock};
use types::base::SomaAddress;
use types::effects::{TransactionEffects, TransactionEffectsAPI as _};
use types::object::{Object, ObjectID, Version};
use types::transaction::{Transaction, TransactionData, TransactionKind};

use crate::error::SomaRpcResult;

pub mod channel;
pub mod client_config;
pub mod crypto_utils;
pub mod error;
pub mod faucet_client;
pub mod keypair;
#[cfg(feature = "proxy")]
pub mod proxy_client;
pub mod transaction_builder;
pub mod wallet_context;

// Re-export types for downstream crates
#[cfg(feature = "grpc-services")]
pub use admin::admin_types;

// gRPC client type aliases (tonic 0.14.3 channels, separate from core ledger)
#[cfg(feature = "grpc-services")]
type AdminGrpcClient =
    admin::admin_gen::admin_client::AdminClient<admin::tonic::transport::Channel>;
// TODO: define these when public rpcs are finalized
pub const SOMA_LOCAL_NETWORK_URL: &str = "http://127.0.0.1:9000";
pub const SOMA_LOCAL_NETWORK_URL_0: &str = "http://0.0.0.0:9000";
pub const SOMA_TESTNET_URL: &str = "https://fullnode.testnet.soma.org:443";
// pub const SOMA_MAINNET_URL: &str = "https://fullnode.mainnet.soma.org:443";

/// Builder for configuring a SomaClient
pub struct SomaClientBuilder {
    request_timeout: Duration,
    faucet_url: Option<String>,
    #[cfg(feature = "grpc-services")]
    admin_url: Option<String>,
}

impl Default for SomaClientBuilder {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(60),
            faucet_url: None,
            #[cfg(feature = "grpc-services")]
            admin_url: None,
        }
    }
}

impl SomaClientBuilder {
    /// Set the request timeout
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Set the admin service URL (e.g. `http://127.0.0.1:9125`).
    #[cfg(feature = "grpc-services")]
    pub fn admin_url(mut self, url: impl Into<String>) -> Self {
        self.admin_url = Some(url.into());
        self
    }

    /// Set the faucet service URL (e.g. `http://127.0.0.1:9123`).
    pub fn faucet_url(mut self, url: impl Into<String>) -> Self {
        self.faucet_url = Some(url.into());
        self
    }

    /// Build the client with RPC and object storage URLs
    pub async fn build(self, rpc_url: impl AsRef<str>) -> Result<SomaClient, error::Error> {
        let client = Client::new(rpc_url.as_ref())
            .map_err(|e| error::Error::ClientInitError(e.to_string()))?;

        #[cfg(feature = "grpc-services")]
        let admin_client = match self.admin_url {
            Some(url) => {
                let ac = AdminGrpcClient::connect(url)
                    .await
                    .map_err(|e| error::Error::ClientInitError(e.to_string()))?;
                Some(Arc::new(Mutex::new(ac)))
            }
            None => None,
        };

        let faucet_client = match self.faucet_url {
            Some(url) => {
                let fc = faucet_client::FaucetClient::connect(url)
                    .await
                    .map_err(|e| error::Error::ClientInitError(e.to_string()))?;
                Some(Arc::new(Mutex::new(fc)))
            }
            None => None,
        };

        Ok(SomaClient {
            inner: Arc::new(RwLock::new(client)),
            faucet_client,
            #[cfg(feature = "grpc-services")]
            admin_client,
        })
    }

    /// Build a client for the local network with default addresses
    pub async fn build_localnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_LOCAL_NETWORK_URL).await
    }

    /// Build a client for testnet with default addresses
    pub async fn build_testnet(self) -> Result<SomaClient, error::Error> {
        self.build(SOMA_TESTNET_URL).await
    }
}

/// The main SOMA client for interacting with the SOMA network via gRPC and TUS.
///
/// Stage 13c: gas is balance-mode — there are no per-tx coin objects to
/// reserve / lock-retry against, so the SDK no longer tracks in-flight
/// coin IDs.
#[derive(Clone)]
pub struct SomaClient {
    inner: Arc<RwLock<Client>>,
    faucet_client: Option<Arc<Mutex<faucet_client::FaucetClient>>>,
    #[cfg(feature = "grpc-services")]
    admin_client: Option<Arc<Mutex<AdminGrpcClient>>>,
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

    /// Get the latest checkpoint summary
    pub async fn get_latest_checkpoint(
        &self,
    ) -> Result<types::checkpoints::CertifiedCheckpointSummary, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_latest_checkpoint().await
    }

    /// Get a checkpoint summary by sequence number
    pub async fn get_checkpoint_summary(
        &self,
        sequence_number: u64,
    ) -> Result<types::checkpoints::CertifiedCheckpointSummary, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_checkpoint_summary(sequence_number).await
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

    /// Get the USDC accumulator balance for an address. USDC is the
    /// gas / typical transferable currency.
    pub async fn get_balance(
        &self,
        owner: &types::base::SomaAddress,
    ) -> Result<u64, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_balance(owner).await
    }

    /// Stage 13c: read the accumulator balance for `(owner, coin_type)`.
    pub async fn get_balance_by_coin_type(
        &self,
        owner: &types::base::SomaAddress,
        coin_type: types::object::CoinType,
    ) -> Result<u64, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_balance_by_coin_type(owner, coin_type).await
    }

    /// Stage 9d: list a staker's active delegations from the on-chain
    /// `delegations` table.
    pub async fn list_delegations(
        &self,
        request: impl tonic::IntoRequest<rpc::proto::soma::ListDelegationsRequest>,
    ) -> Result<rpc::proto::soma::ListDelegationsResponse, tonic::Status> {
        let client = self.inner.read().await.clone();
        client.list_delegations(request).await
    }

    /// Get the chain identifier from the network
    pub async fn get_chain_identifier(&self) -> Result<String, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_chain_identifier().await
    }

    /// Get the human-readable chain name (e.g. "testnet", "localnet")
    pub async fn get_chain_name(&self) -> Result<String, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_chain_name().await
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
        // Server may report version as "soma-node/1.0.0" — strip the prefix for comparison
        let server_version_normalized =
            server_version.rsplit('/').next().unwrap_or(&server_version);
        if server_version_normalized != client_version {
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

    // -------------------------------------------------------------------
    // Transaction helpers
    // -------------------------------------------------------------------

    /// Build [`TransactionData`].
    ///
    /// Stage 13c: gas is balance-mode — `gas_payment` is always empty
    /// for non-system txs. The authority's `prepare_gas` debits the
    /// sender's USDC accumulator at execution time.
    pub fn build_transaction_data(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
    ) -> TransactionData {
        TransactionData::new(kind, sender, Vec::new())
    }

    /// Sign and execute a transaction, returning the effects.
    ///
    /// Waits for the transaction to be included in a checkpoint (i.e. indexed)
    /// so that subsequent reads (e.g. `get_balance`) reflect the new state.
    /// Returns an error if the transaction effects indicate failure.
    pub async fn sign_and_execute(
        &self,
        keypair: &keypair::Keypair,
        tx_data: TransactionData,
        label: &str,
    ) -> Result<TransactionEffects, error::Error> {
        let tx = keypair.sign_transaction(tx_data);
        let response = self
            .execute_transaction_and_wait_for_checkpoint(&tx, Duration::from_secs(30))
            .await
            .map_err(|e| error::Error::GrpcError(e.to_string()))?;
        if !response.effects.status().is_ok() {
            return Err(error::Error::TransactionFailed(format!(
                "{label} failed (digest: {}): {:?}",
                response.effects.transaction_digest(),
                response.effects.status()
            )));
        }
        Ok(response.effects)
    }

    /// Build, sign, and execute a transaction. Stage 13c: gas is
    /// balance-mode, so the old "retry on coin conflict" path is gone
    /// — there is no per-tx gas coin to lock or to come up stale.
    pub async fn sign_and_execute_with_retry(
        &self,
        keypair: &keypair::Keypair,
        sender: SomaAddress,
        kind: TransactionKind,
        label: &str,
    ) -> Result<TransactionEffects, error::Error> {
        let tx_data = self.build_transaction_data(sender, kind);
        self.sign_and_execute(keypair, tx_data, label).await
    }

    /// Merge all coins owned by the sender into as few as possible.
    ///
    /// Uses a single on-chain transaction: the smallest coin is
    // Stage 13b: `merge_coins` deleted along with the
    // TransactionKind::Transfer / MergeCoins variants. Coin objects
    // no longer exist (the balance accumulator is the sole record),
    // so there's nothing to merge.

    /// Request test tokens from the faucet. Returns the gas response.
    pub async fn request_faucet(
        &self,
        address: SomaAddress,
    ) -> Result<faucet_client::GasResponse, error::Error> {
        let fc = self
            .faucet_client
            .as_ref()
            .ok_or_else(|| {
                error::Error::ServiceNotConfigured(
                    "No faucet_url was provided when creating SomaClient".into(),
                )
            })?
            .clone();
        let mut client = fc.lock().await;
        let response = client
            .request_gas(faucet_client::GasRequest { recipient: address.to_string() })
            .await
            .map_err(|e| error::Error::GrpcError(e.to_string()))?
            .into_inner();
        Ok(response)
    }

    /// Trigger epoch advancement on localnet. Returns the new epoch number.
    #[cfg(feature = "grpc-services")]
    pub async fn advance_epoch(&self) -> Result<u64, error::Error> {
        let ac = self
            .admin_client
            .as_ref()
            .ok_or_else(|| {
                error::Error::ServiceNotConfigured(
                    "No admin_url was provided when creating SomaClient".into(),
                )
            })?
            .clone();
        let mut client = ac.lock().await;
        let response = client
            .advance_epoch(admin::admin_types::AdvanceEpochRequest {})
            .await
            .map_err(|e| error::Error::GrpcError(e.to_string()))?
            .into_inner();
        Ok(response.epoch)
    }
}

