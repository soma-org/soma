use crate::SomaClient;
use crate::client_config::{SomaClientConfig, SomaEnv};
use anyhow::anyhow;
use futures::TryStreamExt as _;
use rpc::api::client::TransactionExecutionResponse;
use rpc::proto::soma::owner::OwnerKind;
use rpc::types::ObjectType;
use rpc::utils::field::{FieldMask, FieldMaskUtil};
use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::{AccountKeystore, Keystore};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::fs::File;
use tokio::io::{AsyncRead, AsyncSeek};
use tokio::sync::RwLock;
use tracing::info;
use types::base::SomaAddress;
use types::config::{Config, PersistedConfig};
use types::crypto::{Signature, SomaKeyPair};
use types::digests::ObjectDigest;
use types::intent::Intent;
use types::object::{ObjectID, ObjectRef, Version};
use types::transaction::{Transaction, TransactionData};

pub struct WalletContext {
    pub config: PersistedConfig<SomaClientConfig>,
    request_timeout: Option<std::time::Duration>,
    client: Arc<RwLock<Option<SomaClient>>>,
    env_override: Option<String>,
}

impl WalletContext {
    /// Create a new WalletContext from a config file path
    pub fn new(config_path: &Path) -> Result<Self, anyhow::Error> {
        let config: SomaClientConfig = PersistedConfig::read(config_path).map_err(|err| {
            anyhow!(
                "Cannot open wallet config file at {:?}. Err: {err}",
                config_path
            )
        })?;

        let config = config.persisted(config_path);
        let context = Self {
            config,
            request_timeout: None,
            client: Default::default(),
            env_override: None,
        };
        Ok(context)
    }

    /// Create a WalletContext for testing
    pub fn new_for_tests(
        keystore: Keystore,
        external: Option<Keystore>,
        path: Option<PathBuf>,
    ) -> Self {
        let mut config = SomaClientConfig::new(keystore)
            .persisted(&path.unwrap_or(PathBuf::from("test_config.yaml")));
        config.external_keys = external;
        Self {
            config,
            request_timeout: None,
            client: Arc::new(Default::default()),
            env_override: None,
        }
    }

    /// Set the request timeout
    pub fn with_request_timeout(mut self, request_timeout: std::time::Duration) -> Self {
        self.request_timeout = Some(request_timeout);
        self
    }

    /// Override the environment to use
    pub fn with_env_override(mut self, env_override: String) -> Self {
        self.env_override = Some(env_override);
        self
    }

    /// Get all addresses managed by the wallet
    pub fn get_addresses(&self) -> Vec<SomaAddress> {
        self.config.keystore.addresses()
    }

    /// Get the environment override if set
    pub fn get_env_override(&self) -> Option<String> {
        self.env_override.clone()
    }

    /// Get an address by key identity
    pub fn get_identity_address(
        &mut self,
        input: Option<KeyIdentity>,
    ) -> Result<SomaAddress, anyhow::Error> {
        if let Some(key_identity) = input {
            if let Ok(address) = self.config.keystore.get_by_identity(&key_identity) {
                return Ok(address);
            }
            if let Some(address) = self
                .config
                .external_keys
                .as_ref()
                .and_then(|external_keys| external_keys.get_by_identity(&key_identity).ok())
            {
                return Ok(address);
            }

            Err(anyhow!(
                "No address found for the provided key identity: {key_identity}"
            ))
        } else {
            self.active_address()
        }
    }

    /// Get or create the gRPC client
    pub async fn get_client(&self) -> Result<SomaClient, anyhow::Error> {
        let read = self.client.read().await;

        Ok(if let Some(client) = read.as_ref() {
            client.clone()
        } else {
            drop(read);
            let client = self
                .get_active_env()?
                .create_rpc_client(self.request_timeout)
                .await?;
            self.client.write().await.insert(client).clone()
        })
    }

    /// Load the chain ID corresponding to the active environment, or fetch and cache it if not
    /// present.
    ///
    /// The chain ID is cached in the `client.yaml` file to avoid redundant network requests.
    pub async fn load_or_cache_chain_id(
        &self,
        client: &SomaClient,
    ) -> Result<String, anyhow::Error> {
        self.internal_load_or_cache_chain_id(client, false).await
    }

    /// Try to load the cached chain ID for the active environment.
    pub async fn try_load_chain_id_from_cache(
        &self,
        env: Option<String>,
    ) -> Result<String, anyhow::Error> {
        let env = if let Some(env) = env {
            self.config
                .get_env(&Some(env.to_string()))
                .ok_or_else(|| anyhow!("Environment configuration not found for env [{}]", env))?
        } else {
            self.get_active_env()?
        };
        if let Some(chain_id) = &env.chain_id {
            Ok(chain_id.clone())
        } else {
            Err(anyhow!(
                "No cached chain ID found for env {}. Please pass `-e env_name` to your command",
                env.alias
            ))
        }
    }

    /// Cache (or recache) chain ID for the active environment by fetching it from the
    /// network
    pub async fn cache_chain_id(&self, client: &SomaClient) -> Result<String, anyhow::Error> {
        self.internal_load_or_cache_chain_id(client, true).await
    }

    async fn internal_load_or_cache_chain_id(
        &self,
        client: &SomaClient,
        force_recache: bool,
    ) -> Result<String, anyhow::Error> {
        let env = self.get_active_env()?;
        if !force_recache && env.chain_id.is_some() {
            let chain_id = env.chain_id.as_ref().unwrap();
            info!("Found cached chain ID for env {}: {}", env.alias, chain_id);
            return Ok(chain_id.clone());
        }
        let chain_id = client.get_chain_identifier().await?;
        let path = self.config.path();
        let mut config_result = SomaClientConfig::load_with_lock(path)?;

        config_result.update_env_chain_id(&env.alias, chain_id.clone())?;
        config_result.save_with_lock(path)?;
        Ok(chain_id)
    }

    /// Get the active environment configuration
    pub fn get_active_env(&self) -> Result<&SomaEnv, anyhow::Error> {
        if self.env_override.is_some() {
            self.config.get_env(&self.env_override).ok_or_else(|| {
                anyhow!(
                    "Environment configuration not found for env [{}]",
                    self.env_override.as_deref().unwrap_or("None")
                )
            })
        } else {
            self.config.get_active_env()
        }
    }

    /// Get the active address
    pub fn active_address(&mut self) -> Result<SomaAddress, anyhow::Error> {
        if self.config.keystore.entries().is_empty() {
            return Err(anyhow!(
                "No managed addresses. Create new address with `new-address` command."
            ));
        }

        // Set it if not exists
        self.config.active_address = Some(
            self.config
                .active_address
                .unwrap_or(*self.config.keystore.addresses().first().unwrap()),
        );

        Ok(self.config.active_address.unwrap())
    }

    pub async fn get_gas_objects_owned_by_address(
        &self,
        address: SomaAddress,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<ObjectRef>> {
        let client = self.get_client().await?;

        // Create the request for listing owned objects
        let mut request = rpc::proto::soma::ListOwnedObjectsRequest::default();
        request.owner = Some(address.to_string());

        // Set page size based on limit
        let page_size = limit.unwrap_or(100).min(1000) as u32;
        request.page_size = Some(page_size);

        request.object_type = Some((ObjectType::Coin).into());
        request.read_mask = Some(FieldMask::from_paths([
            "object_id",
            "version",
            "digest",
            "object_type",
            "owner",
            "contents",
            "previous_transaction",
        ]));

        // Call the live data service
        let stream = client.list_owned_objects(request).await;

        let object_refs: Vec<ObjectRef> = Vec::new();
        tokio::pin!(stream);

        // Convert the objects to ObjectRef
        let mut object_refs = Vec::new();
        while let Some(object) = stream.try_next().await? {
            object_refs.push(object.compute_object_reference());

            // Stop if we've reached the limit
            if let Some(limit) = limit {
                if object_refs.len() >= limit {
                    break;
                }
            }
        }

        Ok(object_refs)
    }

    pub async fn get_object_owner(&self, id: &ObjectID) -> Result<SomaAddress, anyhow::Error> {
        let client = self.get_client().await?;

        let object = client
            .get_object(id.clone())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get object: {}", e))?;

        // Extract owner from the proto object
        let owner = object
            .owner
            .get_owner_address()
            .map_err(|e| anyhow::anyhow!("Failed to get object: {}", e))?;

        Ok(owner)
    }

    /// Returns all the account addresses managed by the wallet and their owned gas objects.
    pub async fn get_all_accounts_and_gas_objects(
        &self,
    ) -> anyhow::Result<Vec<(SomaAddress, Vec<ObjectRef>)>> {
        let mut result = vec![];
        for address in self.get_addresses() {
            let objects = self.get_all_gas_objects_owned_by_address(address).await?;
            result.push((address, objects));
        }
        Ok(result)
    }

    pub async fn try_get_object_owner(
        &self,
        id: &Option<ObjectID>,
    ) -> Result<Option<SomaAddress>, anyhow::Error> {
        if let Some(id) = id {
            Ok(Some(self.get_object_owner(id).await?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_all_gas_objects_owned_by_address(
        &self,
        address: SomaAddress,
    ) -> anyhow::Result<Vec<ObjectRef>> {
        self.get_gas_objects_owned_by_address(address, None).await
    }

    /// Given an address, return one gas object owned by this address.
    /// The actual implementation just returns the first one returned by the read api.
    pub async fn get_one_gas_object_owned_by_address(
        &self,
        address: SomaAddress,
    ) -> anyhow::Result<Option<ObjectRef>> {
        Ok(self
            .get_gas_objects_owned_by_address(address, Some(1))
            .await?
            .pop())
    }

    /// Returns one address and all gas objects owned by that address.
    pub async fn get_one_account(&self) -> anyhow::Result<(SomaAddress, Vec<ObjectRef>)> {
        let address = self.get_addresses().pop().unwrap();
        Ok((
            address,
            self.get_all_gas_objects_owned_by_address(address).await?,
        ))
    }

    /// Return a gas object owned by an arbitrary address managed by the wallet.
    pub async fn get_one_gas_object(&self) -> anyhow::Result<Option<(SomaAddress, ObjectRef)>> {
        for address in self.get_addresses() {
            if let Some(gas_object) = self.get_one_gas_object_owned_by_address(address).await? {
                return Ok(Some((address, gas_object)));
            }
        }
        Ok(None)
    }

    /// Add an account to the wallet
    pub async fn add_account(&mut self, alias: Option<String>, keypair: SomaKeyPair) {
        self.config.keystore.import(alias, keypair).await.unwrap();
    }

    /// Get the keystore that contains the given key identity
    pub fn get_keystore_by_identity(
        &self,
        key_identity: &KeyIdentity,
    ) -> Result<&Keystore, anyhow::Error> {
        if self.config.keystore.get_by_identity(key_identity).is_ok() {
            return Ok(&self.config.keystore);
        }

        if let Some(external_keys) = self.config.external_keys.as_ref() {
            if external_keys.get_by_identity(key_identity).is_ok() {
                return Ok(external_keys);
            }
        }

        Err(anyhow!(
            "No keystore found for the provided key identity: {key_identity}"
        ))
    }

    /// Get a mutable reference to the keystore that contains the given key identity
    pub fn get_keystore_by_identity_mut(
        &mut self,
        key_identity: &KeyIdentity,
    ) -> Result<&mut Keystore, anyhow::Error> {
        if self.config.keystore.get_by_identity(key_identity).is_ok() {
            return Ok(&mut self.config.keystore);
        }

        if let Some(external_keys) = self.config.external_keys.as_mut() {
            if external_keys.get_by_identity(key_identity).is_ok() {
                return Ok(external_keys);
            }
        }

        Err(anyhow!(
            "No keystore found for the provided key identity: {key_identity}"
        ))
    }

    /// Sign transaction data with the specified key identity
    pub async fn sign_secure(
        &self,
        key_identity: &KeyIdentity,
        data: &TransactionData,
        intent: Intent,
    ) -> Result<Signature, anyhow::Error> {
        let keystore = self.get_keystore_by_identity(key_identity)?;
        let sig = keystore.sign_secure(&data.sender(), data, intent).await?;
        Ok(sig)
    }

    /// Sign a transaction with a key currently managed by the WalletContext
    pub async fn sign_transaction(&self, data: &TransactionData) -> Transaction {
        let sig = self
            .config
            .keystore
            .sign_secure(&data.sender(), data, Intent::soma_transaction())
            .await
            .unwrap();
        Transaction::from_data(data.clone(), vec![sig])
    }

    /// Execute a transaction and wait for it to be finalized
    /// Also expects the effects status to be successful
    pub async fn execute_transaction_must_succeed(
        &self,
        tx: Transaction,
    ) -> TransactionExecutionResponse {
        tracing::debug!("Executing transaction: {:?}", tx);
        let response = self.execute_transaction_may_fail(tx).await.unwrap();

        assert!(
            response.effects.status.is_ok(),
            "Transaction failed: {:?}",
            response
        );

        response
    }

    /// Execute a transaction and wait for it to be finalized
    /// The transaction execution is not guaranteed to succeed and may fail
    pub async fn execute_transaction_may_fail(
        &self,
        tx: Transaction,
    ) -> anyhow::Result<TransactionExecutionResponse> {
        let client = self.get_client().await?;
        Ok(client.execute_transaction(&tx).await?)
    }

    /// Execute a transaction and wait for it to be indexed (checkpointed)
    /// This ensures "read your writes" consistency for subsequent queries
    pub async fn execute_transaction_and_wait_for_indexing(
        &self,
        tx: Transaction,
    ) -> anyhow::Result<TransactionExecutionResponse> {
        let mut client = self.get_client().await?;
        client
            .execute_transaction_and_wait_for_checkpoint(&tx, Duration::from_secs(30))
            .await
            .map_err(|e| anyhow::anyhow!("Transaction execution failed: {}", e))
    }

    /// Get one address managed by the wallet (for testing)
    pub fn get_one_address(&self) -> Option<SomaAddress> {
        self.get_addresses().first().copied()
    }

    /// Check if the wallet has any addresses
    pub fn has_addresses(&self) -> bool {
        !self.config.keystore.addresses().is_empty()
    }

    /// Save the configuration to disk
    pub fn save_config(&self) -> Result<(), anyhow::Error> {
        self.config.save()?;
        Ok(())
    }
}
