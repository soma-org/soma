use bytes::Bytes;
use futures::Stream;
use rpc::api::ServerVersion;
use rpc::api::client::{ShardCompletionInfo, ShardError};
use rpc::proto::soma::ListOwnedObjectsRequest;
use rpc::proto::soma::{InitiateShardWorkRequest, InitiateShardWorkResponse};
use rpc::{api::client::Client, proto::soma::ledger_service_client::LedgerServiceClient};
use std::pin::Pin;
use std::{sync::Arc, time::Duration};
use tokio::io::{AsyncRead, AsyncSeek};
use tokio::sync::RwLock;
use types::base::SomaAddress;
use types::effects::TransactionEffects;
use types::object::{Object, ObjectID, Version};
use types::shard_crypto::keys::EncoderPublicKey;
use types::transaction::Transaction;
use url::Url;
use uuid::Uuid;

use rpc::proto::soma::{
    GetClaimableEscrowsResponse, GetClaimableRewardsResponse, GetShardsByEncoderResponse,
    GetShardsByEpochResponse, GetShardsBySubmitterResponse, GetValidTargetsResponse,
};

use crate::error::SomaRpcResult;

pub mod client_config;
pub mod error;
pub mod transaction_builder;
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
}

impl Default for SomaClientBuilder {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(60),
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
    pub async fn build(self, rpc_url: impl AsRef<str>) -> Result<SomaClient, error::Error> {
        // Create gRPC client
        let client = Client::new(rpc_url.as_ref())
            .map_err(|e| error::Error::ClientInitError(e.to_string()))?;

        Ok(SomaClient {
            inner: Arc::new(RwLock::new(client)),
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

    /// Initiate shard work for encoding
    pub async fn initiate_shard_work(
        &self,
        request: impl tonic::IntoRequest<InitiateShardWorkRequest>,
    ) -> Result<InitiateShardWorkResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.initiate_shard_work(request).await
    }

    /// Verifies if the API version matches the server version and returns an error if they do not match.
    pub async fn check_api_version(&self) -> SomaRpcResult<()> {
        let server_version = self
            .get_server_version()
            .await
            .map_err(|e| crate::error::Error::RpcError(e.into()))?;
        let client_version = env!("CARGO_PKG_VERSION");
        if server_version != client_version {
            return Err(crate::error::Error::ServerVersionMismatch {
                client_version: client_version.to_string(),
                server_version: server_version,
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

    // =========================================================================
    // SHARD QUERIES
    // =========================================================================

    /// Get all shards created in a specific epoch
    pub async fn get_shards_by_epoch(
        &self,
        epoch: u64,
    ) -> Result<GetShardsByEpochResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_shards_by_epoch(epoch).await
    }

    /// Get all shards created in a specific epoch with pagination
    pub async fn get_shards_by_epoch_with_pagination(
        &self,
        epoch: u64,
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetShardsByEpochResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client
            .get_shards_by_epoch_with_pagination(epoch, cursor, limit)
            .await
    }

    /// Get shards submitted by a specific address
    pub async fn get_shards_by_submitter(
        &self,
        submitter: &[u8],
        epoch: Option<u64>,
    ) -> Result<GetShardsBySubmitterResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_shards_by_submitter(submitter, epoch).await
    }

    /// Get shards submitted by a specific address with pagination
    pub async fn get_shards_by_submitter_with_pagination(
        &self,
        submitter: &[u8],
        epoch: Option<u64>,
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetShardsBySubmitterResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client
            .get_shards_by_submitter_with_pagination(submitter, epoch, cursor, limit)
            .await
    }

    /// Get shards won by a specific encoder
    pub async fn get_shards_by_encoder(
        &self,
        encoder: &[u8],
    ) -> Result<GetShardsByEncoderResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_shards_by_encoder(encoder).await
    }

    /// Get shards won by a specific encoder with pagination
    pub async fn get_shards_by_encoder_with_pagination(
        &self,
        encoder: &[u8],
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetShardsByEncoderResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client
            .get_shards_by_encoder_with_pagination(encoder, cursor, limit)
            .await
    }

    /// Get claimable escrows for the current epoch
    pub async fn get_claimable_escrows(
        &self,
        current_epoch: u64,
    ) -> Result<GetClaimableEscrowsResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_claimable_escrows(current_epoch).await
    }

    /// Get claimable escrows with pagination
    pub async fn get_claimable_escrows_with_pagination(
        &self,
        current_epoch: u64,
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetClaimableEscrowsResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client
            .get_claimable_escrows_with_pagination(current_epoch, cursor, limit)
            .await
    }

    // =========================================================================
    // TARGET QUERIES
    // =========================================================================

    /// Get all targets valid for competition in the given epoch
    pub async fn get_valid_targets(
        &self,
        epoch: u64,
    ) -> Result<GetValidTargetsResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_valid_targets(epoch).await
    }

    /// Get all targets valid for competition with pagination
    pub async fn get_valid_targets_with_pagination(
        &self,
        epoch: u64,
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetValidTargetsResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client
            .get_valid_targets_with_pagination(epoch, cursor, limit)
            .await
    }

    /// Get claimable rewards for the current epoch
    pub async fn get_claimable_rewards(
        &self,
        current_epoch: u64,
    ) -> Result<GetClaimableRewardsResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_claimable_rewards(current_epoch).await
    }

    /// Get claimable rewards with pagination
    pub async fn get_claimable_rewards_with_pagination(
        &self,
        current_epoch: u64,
        cursor: Option<Bytes>,
        limit: Option<u32>,
    ) -> Result<GetClaimableRewardsResponse, tonic::Status> {
        let mut client = self.inner.write().await;
        client
            .get_claimable_rewards_with_pagination(current_epoch, cursor, limit)
            .await
    }

    // Get shards submitted by a SomaAddress
    pub async fn get_shards_by_submitter_address(
        &self,
        submitter: &SomaAddress,
        epoch: Option<u64>,
    ) -> Result<GetShardsBySubmitterResponse, tonic::Status> {
        self.get_shards_by_submitter(submitter.as_ref(), epoch)
            .await
    }

    /// Get shards won by an EncoderPublicKey
    pub async fn get_shards_by_encoder_key(
        &self,
        encoder: &EncoderPublicKey,
    ) -> Result<GetShardsByEncoderResponse, tonic::Status> {
        self.get_shards_by_encoder(encoder.to_bytes()).await
    }

    // =========================================================================
    // SHARD COMPLETION HELPERS
    // =========================================================================

    /// Extract the ShardInput object ID from EmbedData transaction effects.
    ///
    /// Use this after executing an EmbedData transaction to get the ID
    /// of the created ShardInput object.
    ///
    /// # Example
    /// ```no_run
    /// let response = client.execute_transaction(&embed_tx).await?;
    /// let shard_input_id = SomaClient::extract_shard_input_id(&response.effects)?;
    /// ```
    pub fn extract_shard_input_id(effects: &TransactionEffects) -> Result<ObjectID, ShardError> {
        rpc::api::client::Client::extract_shard_input_id(effects)
    }

    /// Wait for a shard to complete encoding.
    ///
    /// Subscribes to the checkpoint stream and watches for a ReportWinner
    /// transaction that references the given ShardInput object.
    ///
    /// # Arguments
    /// * `shard_input_id` - The ObjectID returned by `extract_shard_input_id`
    /// * `timeout` - Maximum time to wait for completion
    ///
    /// # Example
    /// ```no_run
    /// let response = client.execute_transaction(&embed_tx).await?;
    /// let shard_input_id = SomaClient::extract_shard_input_id(&response.effects)?;
    ///
    /// let completion = client.wait_for_shard_completion(&shard_input_id, Duration::from_secs(120)).await?;
    /// println!("Encoding complete! Winner tx: {}", completion.winner_tx_digest);
    ///
    /// // Now fetch the shard to get the embedding download metadata
    /// let shard = client.get_object(completion.shard_id).await?;
    /// ```
    pub async fn wait_for_shard_completion(
        &self,
        shard_input_id: &ObjectID,
        timeout: Duration,
    ) -> Result<ShardCompletionInfo, ShardError> {
        let mut client = self.inner.write().await;
        client
            .wait_for_shard_completion(shard_input_id, timeout)
            .await
    }

    /// Execute an EmbedData transaction and wait for the shard to complete.
    ///
    /// This is the recommended high-level API for submitting data and waiting
    /// for encoding to finish. It handles:
    /// 1. Transaction execution and checkpointing
    /// 2. Extracting the ShardInput object ID
    /// 3. Subscribing and waiting for ReportWinner
    ///
    /// After this returns, you can fetch the Shard object to get the
    /// `embedding_download_metadata` for accessing your embeddings.
    ///
    /// # Arguments
    /// * `transaction` - The EmbedData transaction
    /// * `timeout` - Maximum time to wait for shard completion
    ///
    /// # Example
    /// ```no_run
    /// // Build EmbedData transaction
    /// let tx = Transaction::from_data_and_signer(
    ///     TransactionData::new(
    ///         TransactionKind::EmbedData { download_metadata, coin_ref, target_ref: None },
    ///         address,
    ///         vec![gas_object],
    ///     ),
    ///     vec![&signer],
    /// );
    ///
    /// // Execute and wait for completion
    /// let (exec_response, completion) = client
    ///     .execute_embed_data_and_wait_for_completion(&tx, Duration::from_secs(120))
    ///     .await?;
    ///
    /// println!("Shard {} completed in checkpoint {}",
    ///     completion.shard_id,
    ///     completion.checkpoint_sequence
    /// );
    ///
    /// // Fetch the completed Shard to get embedding metadata
    /// let shard_object = client.get_object(completion.shard_id).await?;
    /// let shard: Shard = shard_object.try_into()?;
    /// let embedding_url = shard.embedding_download_metadata.url();
    /// ```
    pub async fn execute_embed_data_and_wait_for_completion(
        &self,
        transaction: &types::transaction::Transaction,
        timeout: Duration,
    ) -> Result<
        (
            rpc::api::client::TransactionExecutionResponseWithCheckpoint,
            ShardCompletionInfo,
        ),
        ShardError,
    > {
        let mut client = self.inner.write().await;
        client
            .execute_embed_data_and_wait_for_completion(transaction, timeout)
            .await
    }

    /// Convenience method to get a Shard object by ID.
    ///
    /// This fetches the object and attempts to deserialize it as a Shard.
    /// Use this after `wait_for_shard_completion` to get the embedding metadata.
    ///
    /// # Example
    /// ```no_run
    /// let completion = client.wait_for_shard_completion(&shard_input_id, timeout).await?;
    /// let shard = client.get_shard(completion.shard_id).await?;
    /// println!("Embeddings available at: {}", shard.embedding_download_metadata.url());
    /// ```
    pub async fn get_shard(
        &self,
        shard_id: ObjectID,
    ) -> Result<types::system_state::shard::Shard, tonic::Status> {
        let object = self.get_object(shard_id).await?;
        object
            .as_shard()
            .ok_or_else(|| tonic::Status::invalid_argument("Object is not a Shard"))
    }
}
