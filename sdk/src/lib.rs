use futures::Stream;
use rpc::proto::soma::ListOwnedObjectsRequest;
use rpc::api::client::Client;
use std::pin::Pin;
use std::{sync::Arc, time::Duration};
use tokio::sync::{Mutex, RwLock};
use types::base::SomaAddress;
use types::effects::{TransactionEffects, TransactionEffectsAPI as _};
use types::object::{Object, ObjectID, Version};
use types::transaction::{Transaction, TransactionData, TransactionKind};

use crate::error::SomaRpcResult;

pub mod client_config;
pub mod crypto_utils;
pub mod error;
pub mod keypair;
#[cfg(feature = "proxy")]
pub mod proxy_client;
pub mod transaction_builder;
pub mod wallet_context;

// Re-export types for downstream crates
#[cfg(feature = "grpc-services")]
pub use scoring::types as scoring_types;
#[cfg(feature = "grpc-services")]
pub use admin::admin_types;

// gRPC client type aliases (tonic 0.14.3 channels, separate from core ledger)
#[cfg(feature = "grpc-services")]
type ScoringGrpcClient =
    scoring::tonic_gen::scoring_client::ScoringClient<scoring::tonic::transport::Channel>;
#[cfg(feature = "grpc-services")]
type AdminGrpcClient =
    admin::admin_gen::admin_client::AdminClient<admin::tonic::transport::Channel>;
// TODO: define these when public rpcs are finalized
pub const SOMA_LOCAL_NETWORK_URL: &str = "http://127.0.0.1:9000";
pub const SOMA_LOCAL_NETWORK_URL_0: &str = "http://0.0.0.0:9000";
pub const SOMA_DEVNET_URL: &str = "https://fullnode.devnet.soma.org:443";
pub const SOMA_TESTNET_URL: &str = "https://fullnode.testnet.soma.org:443";
pub const SOMA_MAINNET_URL: &str = "https://fullnode.mainnet.soma.org:443";

/// Builder for configuring a SomaClient
pub struct SomaClientBuilder {
    request_timeout: Duration,
    #[cfg(feature = "grpc-services")]
    scoring_url: Option<String>,
    #[cfg(feature = "grpc-services")]
    admin_url: Option<String>,
}

impl Default for SomaClientBuilder {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(60),
            #[cfg(feature = "grpc-services")]
            scoring_url: None,
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

    /// Set the scoring service URL (e.g. `http://127.0.0.1:9124`).
    #[cfg(feature = "grpc-services")]
    pub fn scoring_url(mut self, url: impl Into<String>) -> Self {
        self.scoring_url = Some(url.into());
        self
    }

    /// Set the admin service URL (e.g. `http://127.0.0.1:9125`).
    #[cfg(feature = "grpc-services")]
    pub fn admin_url(mut self, url: impl Into<String>) -> Self {
        self.admin_url = Some(url.into());
        self
    }

    /// Build the client with RPC and object storage URLs
    pub async fn build(self, rpc_url: impl AsRef<str>) -> Result<SomaClient, error::Error> {
        let client = Client::new(rpc_url.as_ref())
            .map_err(|e| error::Error::ClientInitError(e.to_string()))?;

        #[cfg(feature = "grpc-services")]
        let scoring_client = match self.scoring_url {
            Some(url) => {
                let sc = ScoringGrpcClient::connect(url)
                    .await
                    .map_err(|e| error::Error::ClientInitError(e.to_string()))?;
                Some(Arc::new(Mutex::new(sc)))
            }
            None => None,
        };

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

        Ok(SomaClient {
            inner: Arc::new(RwLock::new(client)),
            #[cfg(feature = "grpc-services")]
            scoring_client,
            #[cfg(feature = "grpc-services")]
            admin_client,
        })
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
    #[cfg(feature = "grpc-services")]
    scoring_client: Option<Arc<Mutex<ScoringGrpcClient>>>,
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

    /// Get the balance for an address
    pub async fn get_balance(
        &self,
        owner: &types::base::SomaAddress,
    ) -> Result<u64, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_balance(owner).await
    }

    /// Get the chain identifier from the network
    pub async fn get_chain_identifier(&self) -> Result<String, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_chain_identifier().await
    }

    /// Get the human-readable chain name (e.g. "mainnet", "testnet", "localnet")
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
        let server_version_normalized = server_version
            .rsplit('/')
            .next()
            .unwrap_or(&server_version);
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

    /// Get the current model architecture version from the network
    pub async fn get_architecture_version(&self) -> Result<u64, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_architecture_version().await
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

    /// Build [`TransactionData`] with automatic gas selection.
    ///
    /// If `gas` is `None`, queries the chain for the sender's first coin object.
    pub async fn build_transaction_data(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
        gas: Option<types::object::ObjectRef>,
    ) -> Result<TransactionData, error::Error> {
        let gas_payment = match gas {
            Some(gas_ref) => vec![gas_ref],
            None => {
                use futures::TryStreamExt as _;

                let mut request = ListOwnedObjectsRequest::default();
                request.owner = Some(sender.to_string());
                request.page_size = Some(1);
                request.object_type = Some(rpc::types::ObjectType::Coin.into());

                let stream = self.list_owned_objects(request).await;
                tokio::pin!(stream);

                let obj = stream
                    .try_next()
                    .await
                    .map_err(|e| error::Error::DataError(e.to_string()))?
                    .ok_or_else(|| {
                        error::Error::DataError(format!(
                            "No gas object found for address {sender}. \
                             Please ensure the address has coins."
                        ))
                    })?;
                vec![obj.compute_object_reference()]
            }
        };
        Ok(TransactionData::new(kind, sender, gas_payment))
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
                "{label} failed: {:?}",
                response.effects.status()
            )));
        }
        Ok(response.effects)
    }

    // -------------------------------------------------------------------
    // gRPC service methods (behind grpc-services feature)
    // -------------------------------------------------------------------

    /// Score model manifests against a data submission.
    #[cfg(feature = "grpc-services")]
    pub async fn score(
        &self,
        request: scoring::types::ScoreRequest,
    ) -> Result<scoring::types::ScoreResponse, error::Error> {
        let sc = self
            .scoring_client
            .as_ref()
            .ok_or_else(|| {
                error::Error::ServiceNotConfigured(
                    "No scoring_url was provided when creating SomaClient".into(),
                )
            })?
            .clone();
        let mut client = sc.lock().await;
        let response = client
            .score(request)
            .await
            .map_err(|e| error::Error::GrpcError(e.to_string()))?
            .into_inner();
        Ok(response)
    }

    /// Health check against the scoring service.
    #[cfg(feature = "grpc-services")]
    pub async fn scoring_health(&self) -> Result<bool, error::Error> {
        let sc = self
            .scoring_client
            .as_ref()
            .ok_or_else(|| {
                error::Error::ServiceNotConfigured(
                    "No scoring_url was provided when creating SomaClient".into(),
                )
            })?
            .clone();
        let mut client = sc.lock().await;
        let response = client
            .health(scoring::types::HealthRequest {})
            .await
            .map_err(|e| error::Error::GrpcError(e.to_string()))?
            .into_inner();
        Ok(response.ok)
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
