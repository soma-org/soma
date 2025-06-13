use soma_config::error::SomaConfigError;
use soma_keys::error::SomaKeyError;
use strum_macros::IntoStaticStr;
use thiserror::Error;

#[derive(Clone, Debug, Error, IntoStaticStr)]
pub enum CliError {
    #[error("Soma config error: {0}")]
    SomaConfig(SomaConfigError),
    #[error("Soma key error: {0}")]
    SomaKey(SomaKeyError),
    #[error("Address error: {0}")]
    AddressError(String),
    #[error("Alias error: {0}")]
    AliasError(String),
    #[error("keystore error: {0}")]
    KeyStoreError(String),
    #[error("Error reading key pair")]
    ErrorReadingKeyPair,
    #[error("Error decoding key pair")]
    ErrorDecodingKeyPair,
}

pub type CliResult<T> = Result<T, CliError>;
