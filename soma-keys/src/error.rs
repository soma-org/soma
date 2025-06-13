use strum_macros::IntoStaticStr;
use thiserror::Error;

#[derive(Clone, Debug, Error, IntoStaticStr)]
pub enum SomaKeyError {
    #[error("unsupported key scheme")]
    UnsupportedKeyScheme,
    #[error("Invalid word length")]
    InvalidWordLength,
    #[error("Invalid derivation path")]
    InvalidDerivationPath,
    #[error("error decoding key from: {0}")]
    ErrorDecoding(String),
    #[error("Invalid file path: {0}")]
    InvalidFilePath(String),
    #[error("file system error: {0}")]
    FileSystemError(String),
    #[error("Failed to generate key pair: {0}")]
    FailedToGenerateKeyPair(String),

    // keystore
    #[error("Alias error: {0}")]
    AliasError(String),
    #[error("Invalid mnemonic: {0}")]
    InvalidMnemonic(String),
    #[error("Address error: {0}")]
    AddressError(String),
    #[error("Keystore error: {0}")]
    KeyStoreError(String),
    #[error("Regex error: {0}")]
    RegexError(String),
}

pub type SomaKeyResult<T> = Result<T, SomaKeyError>;
