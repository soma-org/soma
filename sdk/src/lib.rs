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
            in_flight_coins: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
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

/// The main SOMA client for interacting with the SOMA network via gRPC and TUS
#[derive(Clone)]
pub struct SomaClient {
    inner: Arc<RwLock<Client>>,
    /// Coins currently used by in-flight transactions. Prevents concurrent
    /// calls from picking the same coin. Uses `std::sync::Mutex` (not tokio)
    /// since we only hold it briefly with no awaits.
    in_flight_coins: Arc<std::sync::Mutex<std::collections::HashSet<ObjectID>>>,
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

    /// Get the balance for an address
    pub async fn get_balance(
        &self,
        owner: &types::base::SomaAddress,
    ) -> Result<u64, tonic::Status> {
        let mut client = self.inner.write().await;
        client.get_balance(owner).await
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

    /// Build [`TransactionData`] with automatic gas selection.
    ///
    /// If `gas` is `None`, queries the chain for the sender's coin with the
    /// highest balance so that transfers and gas payments are less likely to
    /// fail due to picking a dust coin.
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
                request.page_size = Some(100);
                request.object_type = Some("Coin".to_string());

                let stream = self.list_owned_objects(request).await;
                tokio::pin!(stream);

                let mut best: Option<(types::object::ObjectRef, u64)> = None;
                while let Some(obj) =
                    stream.try_next().await.map_err(|e| error::Error::DataError(e.to_string()))?
                {
                    let balance = obj.as_coin().unwrap_or(0);
                    let obj_ref = obj.compute_object_reference();
                    if best.as_ref().map_or(true, |(_, b)| balance > *b) {
                        best = Some((obj_ref, balance));
                    }
                }

                let (obj_ref, _) = best.ok_or_else(|| {
                    error::Error::DataError(format!(
                        "No gas object found for address {sender}. \
                         Please ensure the address has coins."
                    ))
                })?;
                vec![obj_ref]
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
                "{label} failed (digest: {}): {:?}",
                response.effects.transaction_digest(),
                response.effects.status()
            )));
        }
        Ok(response.effects)
    }

    /// Build, sign, and execute a transaction with automatic retry on coin conflicts.
    ///
    /// Handles two conflict types that arise when coins are used concurrently:
    /// - **ObjectLockConflict** / "already locked": another in-flight tx holds the lock
    /// - **Stale version** / "not available for consumption": coin was already consumed
    ///
    /// Each attempt picks disjoint coins via randomized top-N selection and
    /// tracks them as in-flight so concurrent calls use different coins.
    pub async fn sign_and_execute_with_retry(
        &self,
        keypair: &keypair::Keypair,
        sender: SomaAddress,
        mut kind: TransactionKind,
        label: &str,
    ) -> Result<TransactionEffects, error::Error> {
        const MAX_RETRIES: usize = 5;
        let mut excluded_coins: std::collections::HashSet<ObjectID> = Default::default();

        for attempt in 0..=MAX_RETRIES {
            // 1. Build exclusion set: permanent excludes + in-flight snapshot
            let in_flight_snapshot = self.in_flight_coins.lock().unwrap().clone();
            let mut full_excluded = excluded_coins.clone();
            full_excluded.extend(&in_flight_snapshot);

            // 2. Select gas coin (single coin, random top-N, from full_excluded)
            let tx_data = match self
                .build_transaction_data_excluding(sender, kind.clone(), &full_excluded)
                .await
            {
                Ok(td) => td,
                Err(e) => return Err(e),
            };

            // 3. Reserve: insert gas IDs into in_flight_coins
            let mut reserved = Vec::new();
            for gas_ref in tx_data.gas() {
                reserved.push(gas_ref.0);
            }
            {
                let mut in_flight = self.in_flight_coins.lock().unwrap();
                for id in &reserved {
                    in_flight.insert(*id);
                }
            }

            // 5. Execute transaction
            let result = self.sign_and_execute(keypair, tx_data, label).await;

            // 6. Release: remove reserved IDs from in_flight_coins (always)
            {
                let mut in_flight = self.in_flight_coins.lock().unwrap();
                for id in &reserved {
                    in_flight.remove(id);
                }
            }

            match result {
                Ok(effects) => return Ok(effects),
                Err(e) => {
                    let err_str = e.to_string();
                    let is_coin_conflict = err_str.contains("already locked")
                        || err_str.contains("ObjectLockConflict")
                        || err_str.contains("not available for consumption");

                    if !is_coin_conflict || attempt == MAX_RETRIES {
                        return Err(e);
                    }

                    // 7. On conflict: add conflicting coin to permanent excludes, backoff
                    let backoff = Duration::from_secs(1 << attempt.min(3));
                    if let Some(obj_id) = Self::parse_conflict_object_id(&err_str) {
                        tracing::warn!(
                            "Coin {} conflict, excluding and retrying {label} in {:?} (attempt {}/{})",
                            obj_id,
                            backoff,
                            attempt + 1,
                            MAX_RETRIES
                        );
                        excluded_coins.insert(obj_id);
                    } else {
                        tracing::warn!(
                            "Coin conflict on {label}, retrying with fresh coins in {:?} (attempt {}/{})",
                            backoff,
                            attempt + 1,
                            MAX_RETRIES
                        );
                    }
                    tokio::time::sleep(backoff).await;
                }
            }
        }
        unreachable!()
    }

    /// Build transaction data, excluding specific coins from gas selection.
    ///
    /// Picks a **single** gas coin via randomized top-N selection so that
    /// concurrent callers are unlikely to choose the same coin.
    ///
    /// Scans at most one page of coins (page_size) to keep RPC calls to 1.
    async fn build_transaction_data_excluding(
        &self,
        sender: SomaAddress,
        kind: TransactionKind,
        excluded: &std::collections::HashSet<ObjectID>,
    ) -> Result<TransactionData, error::Error> {
        use futures::TryStreamExt as _;

        const TOP_N: usize = 8;
        const MAX_COINS: usize = 256;

        let mut request = ListOwnedObjectsRequest::default();
        request.owner = Some(sender.to_string());
        request.page_size = Some(MAX_COINS as u32);
        request.object_type = Some("Coin".to_string());

        let stream = self.list_owned_objects(request).await;
        tokio::pin!(stream);

        let mut coins: Vec<(types::object::ObjectRef, u64)> = Vec::new();
        while let Some(obj) =
            stream.try_next().await.map_err(|e| error::Error::DataError(e.to_string()))?
        {
            let obj_ref = obj.compute_object_reference();
            if excluded.contains(&obj_ref.0) {
                continue;
            }
            let balance = obj.as_coin().unwrap_or(0);
            coins.push((obj_ref, balance));
            if coins.len() >= MAX_COINS {
                break;
            }
        }

        if coins.is_empty() {
            return Err(error::Error::DataError(format!(
                "No available gas coin for address {sender} \
                 (excluded {} locked coins). \
                 Run `soma merge-coins` to consolidate your coins.",
                excluded.len()
            )));
        }

        // Sort richest-first, take top N, pick random
        coins.sort_by(|a, b| b.1.cmp(&a.1));
        let top = coins.len().min(TOP_N);
        let idx = rand::thread_rng().gen_range(0..top);
        let selected = coins[idx].0;
        Ok(TransactionData::new(kind, sender, vec![selected]))
    }

    /// Select a coin owned by `sender`, skipping coins in `excluded`.
    ///
    /// Uses randomized top-N selection so concurrent callers are unlikely
    /// to pick the same coin. This is a standalone coin-selection helper
    /// for callers that need to embed a coin reference in a
    /// [`TransactionKind`].
    ///
    /// Scans at most one page of coins to keep RPC calls to 1.
    pub async fn select_coin_excluding(
        &self,
        sender: SomaAddress,
        excluded: &std::collections::HashSet<ObjectID>,
    ) -> Result<types::object::ObjectRef, error::Error> {
        use futures::TryStreamExt as _;

        const TOP_N: usize = 8;
        const MAX_COINS: usize = 256;

        let mut request = ListOwnedObjectsRequest::default();
        request.owner = Some(sender.to_string());
        request.page_size = Some(MAX_COINS as u32);
        request.object_type = Some("Coin".to_string());

        let stream = self.list_owned_objects(request).await;
        tokio::pin!(stream);

        let mut coins: Vec<(types::object::ObjectRef, u64)> = Vec::new();
        while let Some(obj) =
            stream.try_next().await.map_err(|e| error::Error::DataError(e.to_string()))?
        {
            let obj_ref = obj.compute_object_reference();
            if excluded.contains(&obj_ref.0) {
                continue;
            }
            let balance = obj.as_coin().unwrap_or(0);
            coins.push((obj_ref, balance));
            if coins.len() >= MAX_COINS {
                break;
            }
        }

        if coins.is_empty() {
            return Err(error::Error::DataError(format!(
                "No available coin for address {sender} \
                 (excluded {} locked coins). \
                 Ensure the address has multiple coins for concurrent submissions.",
                excluded.len()
            )));
        }

        // Sort richest-first, take top N, pick random
        coins.sort_by(|a, b| b.1.cmp(&a.1));
        let top = coins.len().min(TOP_N);
        let idx = rand::thread_rng().gen_range(0..top);
        Ok(coins[idx].0)
    }

    /// Parse an object ID from a coin conflict error string.
    ///
    /// Handles both lock conflicts (`"Object (0x..., ...) already locked"`)
    /// and stale version errors (`"Object (0x..., ...) is not available"`).
    fn parse_conflict_object_id(err_str: &str) -> Option<ObjectID> {
        let start = err_str.find("Object (0x")?;
        let hex_start = start + "Object (".len();
        let hex_end = err_str[hex_start..].find(',')? + hex_start;
        let hex_str = &err_str[hex_start..hex_end];
        ObjectID::from_hex_literal(hex_str).ok()
    }

    /// Merge all coins owned by the sender into as few as possible.
    ///
    /// Uses a single on-chain transaction: the smallest coin is
    /// "transferred" to self while all other coins are passed as gas
    /// payment (smash_gas merges them). Result: 2 coins max.
    ///
    /// Merges up to 1000 coins per call (one RPC page). Run again if
    /// the address has more.
    pub async fn merge_coins(
        &self,
        keypair: &keypair::Keypair,
        sender: SomaAddress,
    ) -> Result<TransactionEffects, error::Error> {
        use futures::TryStreamExt as _;

        const MAX_COINS: usize = 256;

        let mut request = ListOwnedObjectsRequest::default();
        request.owner = Some(sender.to_string());
        request.page_size = Some(MAX_COINS as u32);
        request.object_type = Some("Coin".to_string());

        let stream = self.list_owned_objects(request).await;
        tokio::pin!(stream);

        let mut coins: Vec<(types::object::ObjectRef, u64)> = Vec::new();
        while let Some(obj) =
            stream.try_next().await.map_err(|e| error::Error::DataError(e.to_string()))?
        {
            let balance = obj.as_coin().unwrap_or(0);
            let obj_ref = obj.compute_object_reference();
            coins.push((obj_ref, balance));
            if coins.len() >= MAX_COINS {
                break;
            }
        }

        if coins.len() <= 1 {
            return Err(error::Error::DataError(
                "Nothing to merge: address has 0 or 1 coins.".to_string(),
            ));
        }

        // Sort by balance ascending — smallest first
        coins.sort_by(|a, b| a.1.cmp(&b.1));

        // Smallest coin is the "transfer coin"
        let transfer_coin = coins[0].0;

        // All other coins become gas payment (smash_gas merges them)
        let gas_payment: Vec<_> = coins[1..].iter().map(|(r, _)| *r).collect();

        let kind =
            TransactionKind::Transfer { coins: vec![transfer_coin], amounts: None, recipients: vec![sender] };
        let tx_data = TransactionData::new(kind, sender, gas_payment);
        self.sign_and_execute(keypair, tx_data, "MergeCoins").await
    }

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

