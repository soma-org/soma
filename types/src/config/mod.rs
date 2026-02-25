// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use anyhow::Result;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use tracing::debug;
use tracing::trace;

use crate::multiaddr::Multiaddr;

pub mod certificate_deny_config;
pub mod genesis_config;
pub mod local_ip_utils;
#[cfg(all(feature = "cloud-storage", feature = "ml"))]
pub mod network_config;
#[cfg(feature = "cloud-storage")]
pub mod node_config;
#[cfg(feature = "cloud-storage")]
pub mod object_store_config;
pub mod p2p_config;
pub mod rpc_config;
pub mod state_sync_config;
pub mod transaction_deny_config;
pub mod validator_client_monitor_config;

const SOMA_DIR: &str = ".soma";
pub const SOMA_CONFIG_DIR: &str = "soma_config";
pub const SOMA_NETWORK_CONFIG: &str = "network.yaml";
pub const SOMA_FULLNODE_CONFIG: &str = "fullnode.yaml";
pub const SOMA_CLIENT_CONFIG: &str = "client.yaml";
pub const SOMA_KEYSTORE_FILENAME: &str = "soma.keystore";
pub const SOMA_KEYSTORE_ALIASES_FILENAME: &str = "soma.aliases";
pub const SOMA_BENCHMARK_GENESIS_GAS_KEYSTORE_FILENAME: &str = "benchmark.keystore";
pub const SOMA_GENESIS_FILENAME: &str = "genesis.blob";
pub const AUTHORITIES_DB_NAME: &str = "authorities_db";
pub const CONSENSUS_DB_NAME: &str = "consensus_db";
pub const FULL_NODE_DB_PATH: &str = "full_node_db";

pub fn soma_config_dir() -> Result<PathBuf, anyhow::Error> {
    match std::env::var_os("SOMA_CONFIG_DIR") {
        Some(config_env) => Ok(config_env.into()),
        None => match dirs::home_dir() {
            Some(v) => Ok(v.join(SOMA_DIR).join(SOMA_CONFIG_DIR)),
            None => anyhow::bail!("Cannot obtain home directory path"),
        },
    }
    .and_then(|dir| {
        if !dir.exists() {
            fs::create_dir_all(dir.clone())?;
        }
        Ok(dir)
    })
}

/// Check if the genesis blob exists in the given directory or the default directory.
pub fn genesis_blob_exists(config_dir: Option<PathBuf>) -> bool {
    if let Some(dir) = config_dir {
        dir.join(SOMA_GENESIS_FILENAME).exists()
    } else if let Some(config_env) = std::env::var_os("SOMA_CONFIG_DIR") {
        Path::new(&config_env).join(SOMA_GENESIS_FILENAME).exists()
    } else if let Some(home) = dirs::home_dir() {
        let mut config = PathBuf::new();
        config.push(&home);
        config.extend([SOMA_DIR, SOMA_CONFIG_DIR, SOMA_GENESIS_FILENAME]);
        config.exists()
    } else {
        false
    }
}

pub fn validator_config_file(address: Multiaddr, i: usize) -> String {
    multiaddr_to_filename(address).unwrap_or(format!("validator-config-{}.yaml", i))
}

pub fn ssfn_config_file(address: Multiaddr, i: usize) -> String {
    multiaddr_to_filename(address).unwrap_or(format!("ssfn-config-{}.yaml", i))
}

fn multiaddr_to_filename(address: Multiaddr) -> Option<String> {
    if let Some(hostname) = address.hostname() {
        if let Some(port) = address.port() {
            return Some(format!("{}-{}.yaml", hostname, port));
        }
    }
    None
}

pub trait Config
where
    Self: DeserializeOwned + Serialize,
{
    fn persisted(self, path: &Path) -> PersistedConfig<Self> {
        PersistedConfig { inner: self, path: path.to_path_buf() }
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

    /// Load the config from the given path, acquiring a shared lock on the file during the read.
    fn load_with_lock<P: AsRef<Path>>(path: P) -> Result<Self, anyhow::Error> {
        let path = path.as_ref();
        debug!("Reading config with lock from {}", path.display());
        let file = fs::File::open(path)
            .with_context(|| format!("Unable to load config from {}", path.display()))?;
        file.lock_shared()?;
        let config: Self = serde_yaml::from_reader(&file)?;
        file.unlock()?;
        Ok(config)
    }

    /// Save the config to the given path, acquiring an exclusive lock on the file during the
    /// write.
    fn save_with_lock<P: AsRef<Path>>(&self, path: P) -> Result<(), anyhow::Error> {
        let path = path.as_ref();
        debug!("Writing config with lock to {}", path.display());
        let config_str = serde_yaml::to_string(&self)?;

        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .with_context(|| {
                format!("Unable to open config file for writing at {}", path.display())
            })?;

        file.lock()
            .with_context(|| format!("Unable to acquire exclusive lock on {}", path.display()))?;

        file.write_all(config_str.as_bytes())
            .with_context(|| format!("Unable to save config to {}", path.display()))?;

        file.unlock()?;
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
