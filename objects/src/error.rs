use strum_macros::IntoStaticStr;
use thiserror::Error;

#[derive(Clone, Debug, Error, IntoStaticStr)]
pub enum ObjectError {
    #[error("fast crypto error: {0}")]
    FastCrypto(String),
    #[error("reqwest error: {0}")]
    ReqwestError(String),
    #[error("Network config error: {0:?}")]
    NetworkConfig(String),
    #[error("Failed to parse URL: {0}")]
    UrlParseError(String),
    #[error("Failed to send request: {0:?}")]
    NetworkRequest(String),
    #[error("write error: {0}")]
    WriteError(String),
    #[error("ObjectStorage: {0}")]
    ObjectStorage(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Verification error: {0}")]
    VerificationError(String),
}
pub type ObjectResult<T> = Result<T, ObjectError>;
