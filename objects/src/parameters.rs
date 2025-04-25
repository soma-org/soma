use std::{path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

/// Operational configurations of a consensus authority.
///
/// All fields should tolerate inconsistencies among authorities, without affecting safety of the
/// protocol. Otherwise, they need to be part of Sui protocol config or epoch state on-chain.
///
/// NOTE: fields with default values are specified in the serde default functions. Most operators
/// should not need to specify any field, except db_path.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Parameters {
    /// http2 network settings.
    #[serde(default = "Http2Parameters::default")]
    pub http2: Http2Parameters,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            http2: Http2Parameters::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Http2Parameters {
    /// Keepalive interval and timeouts for both client and server.
    ///
    /// If unspecified, this will default to 5s.
    #[serde(default = "Http2Parameters::default_keepalive_interval")]
    pub keepalive_interval: Duration,

    /// Set a timeout for only the connect phase of a `Client`.
    ///
    /// If unspecified, this will default to 5s.
    #[serde(default = "Http2Parameters::default_connect_timeout")]
    pub connect_timeout: Duration,

    /// Size of various per-connection buffers.
    ///
    /// If unspecified, this will default to 32MiB.
    #[serde(default = "Http2Parameters::default_connection_buffer_size")]
    pub connection_buffer_size: usize,

    #[serde(default = "Http2Parameters::default_channel_pool_capacity")]
    pub client_pool_capacity: usize,
}

impl Http2Parameters {
    fn default_keepalive_interval() -> Duration {
        Duration::from_secs(5)
    }
    fn default_connect_timeout() -> Duration {
        Duration::from_secs(5)
    }
    fn default_connection_buffer_size() -> usize {
        32 << 20
    }
    fn default_channel_pool_capacity() -> usize {
        1 << 8
    }
}

impl Default for Http2Parameters {
    fn default() -> Self {
        Self {
            keepalive_interval: Http2Parameters::default_keepalive_interval(),
            connect_timeout: Http2Parameters::default_connect_timeout(),
            connection_buffer_size: Http2Parameters::default_connection_buffer_size(),
            client_pool_capacity: Http2Parameters::default_channel_pool_capacity(),
        }
    }
}
