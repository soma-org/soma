use strum_macros::IntoStaticStr;
use thiserror::Error;

pub type ModelResult<T> = Result<T, ModelError>;

#[derive(Clone, Debug, Error, IntoStaticStr)]
pub(crate) enum ModelError {
    #[error("Network config error: {0:?}")]
    NetworkConfig(String),
    #[error("Failed to parse URL: {0}")]
    UrlParseError(String),
    #[error("Network request error: {0:?}")]
    NetworkRequestError(String),
    #[error("Deserialize error: {0:?}")]
    DeserializeError(String),
    #[error("Validation error: {0:?}")]
    ValidationError(String),
}
