//! Soma default configs
pub mod error;
use std::fs;
use std::path::PathBuf;

use error::{SomaConfigError, SomaConfigResult};

const SOMA_DIR: &str = ".soma";
// pub const SOMA_CONFIG_DIR: &str = "configs";

/// Keystore Filename
pub const SOMA_KEYSTORE_FILENAME: &str = "soma.keystore";

/// returns soma dir set by env or $HOME/.soma
pub fn soma_dir() -> SomaConfigResult<PathBuf> {
    match std::env::var_os("SOMA_DIR") {
        Some(config_env) => Ok(config_env.into()),
        None => match dirs::home_dir() {
            Some(home) => Ok(home.join(SOMA_DIR)),
            None => Err(SomaConfigError::HomeDirectoryError),
        },
    }
    .and_then(|dir| {
        if !dir.exists() {
            fs::create_dir_all(dir.clone())
                .map_err(|e| SomaConfigError::CreateDirectoryError(e.to_string()))?;
        }
        Ok(dir)
    })
}
