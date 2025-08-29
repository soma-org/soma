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
    /// Tonic network settings.
    #[serde(default = "TonicParameters::default")]
    pub tonic: TonicParameters,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            tonic: TonicParameters::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TonicParameters {
    /// Keepalive interval and timeouts for both client and server.
    ///
    /// If unspecified, this will default to 5s.
    #[serde(default = "TonicParameters::default_keepalive_interval")]
    pub keepalive_interval: Duration,

    /// Size of various per-connection buffers.
    ///
    /// If unspecified, this will default to 32MiB.
    #[serde(default = "TonicParameters::default_connection_buffer_size")]
    pub connection_buffer_size: usize,

    /// Messages over this size threshold will increment a counter.
    ///
    /// If unspecified, this will default to 16MiB.
    #[serde(default = "TonicParameters::default_excessive_message_size")]
    pub excessive_message_size: usize,

    /// Hard message size limit for both requests and responses.
    /// This value is higher than strictly necessary, to allow overheads.
    /// Message size targets and soft limits are computed based on this value.
    ///
    /// If unspecified, this will default to 1GiB.
    #[serde(default = "TonicParameters::default_message_size_limit")]
    pub message_size_limit: usize,

    #[serde(default = "TonicParameters::default_channel_pool_capacity")]
    pub channel_pool_capacity: usize,
}

impl TonicParameters {
    fn default_keepalive_interval() -> Duration {
        Duration::from_secs(5)
    }

    fn default_connection_buffer_size() -> usize {
        32 << 20
    }

    fn default_excessive_message_size() -> usize {
        16 << 20
    }

    fn default_message_size_limit() -> usize {
        64 << 20
    }
    fn default_channel_pool_capacity() -> usize {
        1 << 8
    }
}

impl Default for TonicParameters {
    fn default() -> Self {
        Self {
            keepalive_interval: TonicParameters::default_keepalive_interval(),
            connection_buffer_size: TonicParameters::default_connection_buffer_size(),
            excessive_message_size: TonicParameters::default_excessive_message_size(),
            message_size_limit: TonicParameters::default_message_size_limit(),
            channel_pool_capacity: TonicParameters::default_channel_pool_capacity(),
        }
    }
}
