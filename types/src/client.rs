use std::time::Duration;

use crate::multiaddr::{parse_dns, parse_ip4, parse_ip6, Multiaddr, Protocol};
use eyre::{eyre, Context, Result};
use serde::{Deserialize, Serialize};
use tonic::transport::{Channel, Endpoint, Uri};

pub async fn connect(address: &Multiaddr) -> Result<Channel> {
    let channel = endpoint_from_multiaddr(address)?.connect().await?;
    Ok(channel)
}

pub fn connect_lazy(address: &Multiaddr) -> Result<Channel> {
    let channel = endpoint_from_multiaddr(address)?.connect_lazy();
    Ok(channel)
}

fn endpoint_from_multiaddr(addr: &Multiaddr) -> Result<MyEndpoint> {
    let mut iter = addr.iter();

    let channel = match iter.next().ok_or_else(|| eyre!("address is empty"))? {
        Protocol::Dns(_) => {
            let (dns_name, tcp_port, http_or_https) = parse_dns(addr)?;
            let uri = format!("{http_or_https}://{dns_name}:{tcp_port}");
            MyEndpoint::try_from_uri(uri)?
        }
        Protocol::Ip4(_) => {
            let (socket_addr, http_or_https) = parse_ip4(addr)?;
            let uri = format!("{http_or_https}://{socket_addr}");
            MyEndpoint::try_from_uri(uri)?
        }
        Protocol::Ip6(_) => {
            let (socket_addr, http_or_https) = parse_ip6(addr)?;
            let uri = format!("{http_or_https}://{socket_addr}");
            MyEndpoint::try_from_uri(uri)?
        }
        unsupported => return Err(eyre!("unsupported protocol {unsupported}")),
    };

    Ok(channel)
}

pub(crate) async fn connect_with_config(address: &Multiaddr, config: &Config) -> Result<Channel> {
    let channel = endpoint_from_multiaddr(address)?
        .apply_config(config)
        .connect()
        .await?;
    Ok(channel)
}

pub(crate) fn connect_lazy_with_config(address: &Multiaddr, config: &Config) -> Result<Channel> {
    let channel = endpoint_from_multiaddr(address)?
        .apply_config(config)
        .connect_lazy();
    Ok(channel)
}

fn apply_config_to_endpoint(config: &Config, mut endpoint: Endpoint) -> Endpoint {
    if let Some(limit) = config.concurrency_limit_per_connection {
        endpoint = endpoint.concurrency_limit(limit);
    }

    if let Some(timeout) = config.request_timeout {
        endpoint = endpoint.timeout(timeout);
    }

    if let Some(timeout) = config.connect_timeout {
        endpoint = endpoint.connect_timeout(timeout);
    }

    if let Some(tcp_nodelay) = config.tcp_nodelay {
        endpoint = endpoint.tcp_nodelay(tcp_nodelay);
    }

    if let Some(http2_keepalive_interval) = config.http2_keepalive_interval {
        endpoint = endpoint.http2_keep_alive_interval(http2_keepalive_interval);
    }

    if let Some(http2_keepalive_timeout) = config.http2_keepalive_timeout {
        endpoint = endpoint.keep_alive_timeout(http2_keepalive_timeout);
    }

    if let Some((limit, duration)) = config.rate_limit {
        endpoint = endpoint.rate_limit(limit, duration);
    }

    endpoint
        .initial_stream_window_size(config.http2_initial_stream_window_size)
        .initial_connection_window_size(config.http2_initial_connection_window_size)
        .tcp_keepalive(config.tcp_keepalive)
}

struct MyEndpoint {
    endpoint: Endpoint,
}

impl MyEndpoint {
    fn new(endpoint: Endpoint) -> Self {
        Self { endpoint }
    }

    fn try_from_uri(uri: String) -> Result<Self> {
        let uri: Uri = uri
            .parse()
            .with_context(|| format!("unable to create Uri from '{uri}'"))?;
        let endpoint = Endpoint::from(uri);
        Ok(Self::new(endpoint))
    }

    fn connect_lazy(self) -> Channel {
        self.endpoint.connect_lazy()
    }

    async fn connect(self) -> Result<Channel> {
        self.endpoint.connect().await.map_err(Into::into)
    }

    fn apply_config(mut self, config: &Config) -> Self {
        self.endpoint = apply_config_to_endpoint(config, self.endpoint);
        self
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Config {
    /// Set the concurrency limit applied to on requests inbound per connection.
    pub concurrency_limit_per_connection: Option<usize>,

    /// Set a timeout for all request handlers.
    pub request_timeout: Option<Duration>,

    /// Set a timeout for establishing an outbound connection.
    pub connect_timeout: Option<Duration>,

    /// Sets the SETTINGS_INITIAL_WINDOW_SIZE option for HTTP2 stream-level flow control.
    /// Default is 65,535
    pub http2_initial_stream_window_size: Option<u32>,

    /// Sets the max connection-level flow control for HTTP2
    ///
    /// Default is 65,535
    pub http2_initial_connection_window_size: Option<u32>,

    /// Sets the SETTINGS_MAX_CONCURRENT_STREAMS option for HTTP2 connections.
    ///
    /// Default is no limit (None).
    pub http2_max_concurrent_streams: Option<u32>,

    /// Set whether TCP keepalive messages are enabled on accepted connections.
    ///
    /// If None is specified, keepalive is disabled, otherwise the duration specified will be the
    /// time to remain idle before sending TCP keepalive probes.
    ///
    /// Default is no keepalive (None)
    pub tcp_keepalive: Option<Duration>,

    /// Set the value of TCP_NODELAY option for accepted connections. Enabled by default.
    pub tcp_nodelay: Option<bool>,

    /// Set whether HTTP2 Ping frames are enabled on accepted connections.
    ///
    /// If None is specified, HTTP2 keepalive is disabled, otherwise the duration specified will be
    /// the time interval between HTTP2 Ping frames. The timeout for receiving an acknowledgement
    /// of the keepalive ping can be set with http2_keepalive_timeout.
    ///
    /// Default is no HTTP2 keepalive (None)
    pub http2_keepalive_interval: Option<Duration>,

    /// Sets a timeout for receiving an acknowledgement of the keepalive ping.
    ///
    /// If the ping is not acknowledged within the timeout, the connection will be closed. Does nothing
    /// if http2_keep_alive_interval is disabled.
    ///
    /// Default is 20 seconds.
    pub http2_keepalive_timeout: Option<Duration>,

    // Only affects servers
    pub load_shed: Option<bool>,

    /// Only affects clients
    pub rate_limit: Option<(u64, Duration)>,

    // Only affects servers
    pub global_concurrency_limit: Option<usize>,
}

impl Config {
    pub fn new() -> Self {
        Default::default()
    }

    pub async fn connect(&self, addr: &Multiaddr) -> Result<Channel> {
        connect_with_config(addr, self).await
    }

    pub fn connect_lazy(&self, addr: &Multiaddr) -> Result<Channel> {
        connect_lazy_with_config(addr, self)
    }
}
