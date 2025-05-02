use strum_macros::IntoStaticStr;
use thiserror::Error;

pub type ProbeResult<T> = Result<T, ProbeError>;

#[derive(Clone, Debug, Error, IntoStaticStr)]
pub(crate) enum ProbeError {
    #[error("Network config error: {0:?}")]
    NetworkConfig(String),
    #[error("Failed to connect as client: {0:?}")]
    NetworkClientConnection(String),
    #[error("Failed to send request: {0:?}")]
    NetworkRequest(String),
    #[error("Error deserializing type: {0}")]
    MalformedType(bcs::Error),
}
