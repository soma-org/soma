use std::{
    fmt::{Display, Formatter, Write},
    path::Path,
};

use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use soma_keys::keystore::{AccountKeystore, FileBasedKeystore, Keystore};
use types::{
    base::*,
    config::{
        Config, PersistedConfig, SOMA_CLIENT_CONFIG, SOMA_KEYSTORE_FILENAME,
        encoder_config::EncoderConfig,
    },
};

use crate::{
    SOMA_DEVNET_URL, SOMA_LOCAL_NETWORK_URL, SOMA_TESTNET_URL, SomaClient, SomaClientBuilder,
};

#[serde_as]
#[derive(Serialize, Deserialize)]
pub struct SomaClientConfig {
    /// The keystore that holds the user's private keys, typically filebased keystore
    pub keystore: Keystore,
    /// Optional external keystore for managing keys that are not stored in the main keystore.
    pub external_keys: Option<Keystore>,
    /// List of environments that the client can connect to.
    pub envs: Vec<SomaEnv>,
    /// The alias of the currently active environment.
    pub active_env: Option<String>,
    /// The address that is currently active in the keystore.
    pub active_address: Option<SomaAddress>,
}

impl SomaClientConfig {
    pub fn new(keystore: Keystore) -> Self {
        SomaClientConfig {
            keystore,
            external_keys: None,
            envs: vec![],
            active_env: None,
            active_address: None,
        }
    }

    pub fn get_env(&self, alias: &Option<String>) -> Option<&SomaEnv> {
        if let Some(alias) = alias {
            self.envs.iter().find(|env| &env.alias == alias)
        } else {
            self.envs.first()
        }
    }

    pub fn get_active_env(&self) -> Result<&SomaEnv, anyhow::Error> {
        self.get_env(&self.active_env).ok_or_else(|| {
            anyhow!(
                "Environment configuration not found for env [{}]",
                self.active_env.as_deref().unwrap_or("None")
            )
        })
    }

    pub fn add_env(&mut self, env: SomaEnv) {
        if !self
            .envs
            .iter()
            .any(|other_env| other_env.alias == env.alias)
        {
            self.envs.push(env)
        }
    }

    /// Update the cached chain ID for the specified environment.
    pub fn update_env_chain_id(
        &mut self,
        alias: &str,
        chain_id: String,
    ) -> Result<(), anyhow::Error> {
        let env = self
            .envs
            .iter_mut()
            .find(|env| env.alias == alias)
            .ok_or_else(|| anyhow!("Environment {} not found", alias))?;
        env.chain_id = Some(chain_id);
        Ok(())
    }
}

pub async fn encoder_config_to_client_config(
    encoder_config: &EncoderConfig,
    config_dir: &Path,
) -> Result<PersistedConfig<SomaClientConfig>, anyhow::Error> {
    let keystore_path = config_dir.join(SOMA_KEYSTORE_FILENAME);

    // Create file-based keystore
    let mut keystore = FileBasedKeystore::load_or_create(&keystore_path)?;

    // Import the account keypair
    let account_kp = encoder_config.account_keypair.keypair().copy();
    let address = SomaAddress::from(&account_kp.public());
    keystore
        .import(Some("encoder-account".to_string()), account_kp)
        .await?;

    // TODO: should encoder config have the internal object address of the rpc?
    let env = SomaEnv {
        alias: "encoder-env".to_string(),
        rpc: format!("http://{}", encoder_config.rpc_address),
        internal_object_address: format!("http://{}", encoder_config.rpc_address),
        basic_auth: None,
        chain_id: None, // TODO: change this chain_id?
    };

    let config = SomaClientConfig {
        keystore: Keystore::File(keystore),
        external_keys: None,
        envs: vec![env],
        active_env: Some("encoder-env".to_string()),
        active_address: Some(address),
    };

    let client_config_path = config_dir.join(SOMA_CLIENT_CONFIG);
    Ok(config.persisted(&client_config_path))
}

impl Config for SomaClientConfig {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaEnv {
    pub alias: String,
    pub rpc: String, // This is now the gRPC endpoint URL
    pub internal_object_address: String,
    pub basic_auth: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chain_id: Option<String>,
}

impl SomaEnv {
    pub async fn create_rpc_client(
        &self,
        request_timeout: Option<std::time::Duration>,
    ) -> Result<SomaClient, anyhow::Error> {
        let mut builder = SomaClientBuilder::default();

        if let Some(request_timeout) = request_timeout {
            builder = builder.request_timeout(request_timeout);
        }

        // TODO: add auth
        // if let Some(basic_auth) = &self.basic_auth {
        //     let fields: Vec<_> = basic_auth.split(':').collect();
        //     if fields.len() != 2 {
        //         return Err(anyhow!(
        //             "Basic auth should be in the format `username:password`"
        //         ));
        //     }
        //     builder = builder.basic_auth(fields[0], fields[1]);
        // }

        Ok(builder
            .build(&self.rpc, &self.internal_object_address)
            .await?)
    }

    pub fn devnet() -> Self {
        Self {
            alias: "devnet".to_string(),
            rpc: SOMA_DEVNET_URL.into(),
            internal_object_address: "http://fullnode.devnet.soma.org:8080".into(),
            basic_auth: None,
            chain_id: None,
        }
    }
    pub fn testnet() -> Self {
        Self {
            alias: "testnet".to_string(),
            rpc: SOMA_TESTNET_URL.into(),
            internal_object_address: "http://fullnode.testnet.soma.org:8080".into(),
            basic_auth: None,
            chain_id: None,
        }
    }

    pub fn localnet() -> Self {
        Self {
            alias: "local".to_string(),
            rpc: SOMA_LOCAL_NETWORK_URL.into(),
            internal_object_address: "http://127.0.0.1:8080".into(),
            basic_auth: None,
            chain_id: None,
        }
    }
}

impl Display for SomaEnv {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = String::new();
        writeln!(writer, "Active environment : {}", self.alias)?;
        write!(writer, "gRPC URL: {}", self.rpc)?;
        if let Some(basic_auth) = &self.basic_auth {
            writeln!(writer)?;
            write!(writer, "Basic Auth: {}", basic_auth)?;
        }
        if let Some(chain_id) = &self.chain_id {
            writeln!(writer)?;
            write!(writer, "Chain ID: {}", chain_id)?;
        }
        write!(f, "{}", writer)
    }
}

impl Display for SomaClientConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut writer = String::new();

        writeln!(
            writer,
            "Managed addresses : {}",
            self.keystore.addresses().len()
        )?;
        write!(writer, "Active address: ")?;
        match self.active_address {
            Some(r) => writeln!(writer, "{}", r)?,
            None => writeln!(writer, "None")?,
        };
        writeln!(writer, "{}", self.keystore)?;
        if let Ok(env) = self.get_active_env() {
            write!(writer, "{}", env)?;
        }
        write!(f, "{}", writer)
    }
}
