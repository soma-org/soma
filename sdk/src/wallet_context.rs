use crate::SomaClient;
use crate::client_config::{SomaClientConfig, SomaEnv};
use anyhow::anyhow;
use rpc::api::client::TransactionExecutionResponse;
use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::{AccountKeystore, Keystore};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use types::base::SomaAddress;
use types::config::{Config, PersistedConfig};
use types::crypto::{Signature, SomaKeyPair};
use types::intent::Intent;
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

    // TODO: Implement these when ReadAPI is available
    /*
    /// Get the latest object reference given an object id
    pub async fn get_object_ref(&self, object_id: ObjectID) -> Result<ObjectRef, anyhow::Error> {
        // Will be implemented when ReadAPI is available
        todo!("ReadAPI not yet implemented")
    }

    /// Get all the gas objects for the address
    pub async fn gas_objects(
        &self,
        address: SomaAddress,
    ) -> Result<Vec<(u64, SomaObjectData)>, anyhow::Error> {
        // Will be implemented when ReadAPI is available
        todo!("ReadAPI not yet implemented")
    }

    pub async fn get_object_owner(&self, id: &ObjectID) -> Result<SomaAddress, anyhow::Error> {
        // Will be implemented when ReadAPI is available
        todo!("ReadAPI not yet implemented")
    }

    pub async fn get_reference_gas_price(&self) -> Result<u64, anyhow::Error> {
        // Will be implemented when governance API is available
        todo!("Governance API not yet implemented")
    }
    */

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
