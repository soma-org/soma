use strum_macros::IntoStaticStr;
use thiserror::Error;

#[derive(Clone, Debug, Error, IntoStaticStr)]
pub enum SomaConfigError {
    #[error("Cannot obtain home directory path")]
    HomeDirectoryError,
    #[error("Failed creating directory: {0}")]
    CreateDirectoryError(String),
}

pub type SomaConfigResult<T> = Result<T, SomaConfigError>;
