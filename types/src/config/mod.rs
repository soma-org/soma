use anyhow::Context;
use anyhow::Result;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::trace;

pub mod encoder_config;
pub mod genesis_config;
pub mod local_ip_utils;
pub mod network_config;
pub mod node_config;
pub mod p2p_config;
pub mod rpc_config;
pub mod state_sync_config;

const SOMA_DIR: &str = ".soma";
pub const SOMA_CONFIG_DIR: &str = "soma_config";
pub const SOMA_NETWORK_CONFIG: &str = "network.yaml";
pub const SOMA_FULLNODE_CONFIG: &str = "fullnode.yaml";
pub const SOMA_CLIENT_CONFIG: &str = "client.yaml";
pub const SOMA_KEYSTORE_FILENAME: &str = "soma.keystore";
pub const SOMA_KEYSTORE_ALIASES_FILENAME: &str = "soma.aliases";
pub const SOMA_BENCHMARK_GENESIS_GAS_KEYSTORE_FILENAME: &str = "benchmark.keystore";
pub const SOMA_GENESIS_FILENAME: &str = "genesis.blob";
pub const SOMA_DEV_NET_URL: &str = "https://fullnode.devnet.soma.io:443";

pub const AUTHORITIES_DB_NAME: &str = "authorities_db";
pub const CONSENSUS_DB_NAME: &str = "consensus_db";
pub const FULL_NODE_DB_PATH: &str = "full_node_db";

pub trait Config
where
    Self: DeserializeOwned + Serialize,
{
    fn persisted(self, path: &Path) -> PersistedConfig<Self> {
        PersistedConfig {
            inner: self,
            path: path.to_path_buf(),
        }
    }

    fn load<P: AsRef<Path>>(path: P) -> Result<Self, anyhow::Error> {
        let path = path.as_ref();
        trace!("Reading config from {}", path.display());
        let reader = fs::File::open(path)
            .with_context(|| format!("Unable to load config from {}", path.display()))?;
        Ok(serde_yaml::from_reader(reader)?)
    }

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), anyhow::Error> {
        let path = path.as_ref();
        trace!("Writing config to {}", path.display());
        let config = serde_yaml::to_string(&self)?;
        fs::write(path, config)
            .with_context(|| format!("Unable to save config to {}", path.display()))?;
        Ok(())
    }
}

pub struct PersistedConfig<C> {
    inner: C,
    path: PathBuf,
}

impl<C> PersistedConfig<C>
where
    C: Config,
{
    pub fn read(path: &Path) -> Result<C, anyhow::Error> {
        Config::load(path)
    }

    pub fn save(&self) -> Result<(), anyhow::Error> {
        self.inner.save(&self.path)
    }

    pub fn into_inner(self) -> C {
        self.inner
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl<C> std::ops::Deref for PersistedConfig<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<C> std::ops::DerefMut for PersistedConfig<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
