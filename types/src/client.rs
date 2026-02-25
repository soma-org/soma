// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use crate::multiaddr::{Multiaddr, Protocol, parse_dns, parse_ip4, parse_ip6};
use eyre::{Context, Result, eyre};
use hyper_util::client::legacy::connect::{HttpConnector, dns::Name};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt,
    future::Future,
    io,
    net::{SocketAddr, ToSocketAddrs},
    pin::Pin,
    sync::{Arc, Mutex},
    task::{self, Poll},
    time::Instant,
    vec,
};
use tokio::task::JoinHandle;
use tokio_rustls::rustls::ClientConfig;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::Service;
use tracing::{info, trace};

pub async fn connect(address: &Multiaddr, tls_config: ClientConfig) -> Result<Channel> {
    let channel = endpoint_from_multiaddr(address, tls_config)?.connect().await?;
    Ok(channel)
}

pub fn connect_lazy(address: &Multiaddr, tls_config: ClientConfig) -> Result<Channel> {
    let channel = endpoint_from_multiaddr(address, tls_config)?.connect_lazy();
    Ok(channel)
}

pub(crate) async fn connect_with_config(
    address: &Multiaddr,
    tls_config: ClientConfig,
    config: &Config,
) -> Result<Channel> {
    let channel =
        endpoint_from_multiaddr(address, tls_config)?.apply_config(config).connect().await?;
    Ok(channel)
}

pub(crate) fn connect_lazy_with_config(
    address: &Multiaddr,
    tls_config: ClientConfig,
    config: &Config,
) -> Result<Channel> {
    let channel = endpoint_from_multiaddr(address, tls_config)?.apply_config(config).connect_lazy();
    Ok(channel)
}

fn endpoint_from_multiaddr(addr: &Multiaddr, tls_config: ClientConfig) -> Result<MyEndpoint> {
    let mut iter = addr.iter();

    let channel = match iter.next().ok_or_else(|| eyre!("address is empty"))? {
        Protocol::Dns(_) => {
            let (dns_name, tcp_port, http_or_https) = parse_dns(addr)?;
            let uri = format!("{http_or_https}://{dns_name}:{tcp_port}");
            MyEndpoint::try_from_uri(uri, tls_config)?
        }
        Protocol::Ip4(_) => {
            let (socket_addr, http_or_https) = parse_ip4(addr)?;
            let uri = format!("{http_or_https}://{socket_addr}");
            MyEndpoint::try_from_uri(uri, tls_config)?
        }
        Protocol::Ip6(_) => {
            let (socket_addr, http_or_https) = parse_ip6(addr)?;
            let uri = format!("{http_or_https}://{socket_addr}");
            MyEndpoint::try_from_uri(uri, tls_config)?
        }
        unsupported => return Err(eyre!("unsupported protocol {unsupported}")),
    };

    Ok(channel)
}

struct MyEndpoint {
    endpoint: Endpoint,
    tls_config: ClientConfig,
}

static DISABLE_CACHING_RESOLVER: OnceCell<bool> = OnceCell::new();

impl MyEndpoint {
    fn new(endpoint: Endpoint, tls_config: ClientConfig) -> Self {
        Self { endpoint, tls_config }
    }

    fn try_from_uri(uri: String, tls_config: ClientConfig) -> Result<Self> {
        let uri: Uri = uri.parse().with_context(|| format!("unable to create Uri from '{uri}'"))?;
        let endpoint = Endpoint::from(uri);
        Ok(Self::new(endpoint, tls_config))
    }

    fn apply_config(mut self, config: &Config) -> Self {
        self.endpoint = apply_config_to_endpoint(config, self.endpoint);
        self
    }

    fn connect_lazy(self) -> Channel {
        let disable_caching_resolver = *DISABLE_CACHING_RESOLVER.get_or_init(|| {
            let disable_caching_resolver = std::env::var("DISABLE_CACHING_RESOLVER").is_ok();
            info!("DISABLE_CACHING_RESOLVER: {disable_caching_resolver}");
            disable_caching_resolver
        });

        if disable_caching_resolver {
            let mut http = HttpConnector::new();
            http.enforce_http(false);
            http.set_nodelay(true);
            http.set_keepalive(None);
            http.set_connect_timeout(None);

            Channel::new(
                hyper_rustls::HttpsConnectorBuilder::new()
                    .with_tls_config(self.tls_config)
                    .https_only()
                    .enable_http2()
                    .wrap_connector(http),
                self.endpoint,
            )
        } else {
            let mut http = HttpConnector::new_with_resolver(CachingResolver::new());
            http.enforce_http(false);
            http.set_nodelay(true);
            http.set_keepalive(None);
            http.set_connect_timeout(None);

            let https = hyper_rustls::HttpsConnectorBuilder::new()
                .with_tls_config(self.tls_config)
                .https_only()
                .enable_http2()
                .wrap_connector(http);
            Channel::new(https, self.endpoint)
        }
    }

    async fn connect(self) -> Result<Channel> {
        let https_connector = hyper_rustls::HttpsConnectorBuilder::new()
            .with_tls_config(self.tls_config)
            .https_only()
            .enable_http2()
            .build();
        Channel::connect(https_connector, self.endpoint).await.map_err(Into::into)
    }
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

type CacheEntry = (Instant, Vec<SocketAddr>);

/// A caching resolver based on hyper_util GaiResolver
#[derive(Clone)]
pub struct CachingResolver {
    cache: Arc<Mutex<HashMap<Name, CacheEntry>>>,
}

type SocketAddrs = vec::IntoIter<SocketAddr>;

pub struct CachingFuture {
    inner: JoinHandle<Result<SocketAddrs, io::Error>>,
}

impl CachingResolver {
    pub fn new() -> Self {
        CachingResolver { cache: Arc::new(Mutex::new(HashMap::new())) }
    }
}

impl Default for CachingResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Service<Name> for CachingResolver {
    type Response = SocketAddrs;
    type Error = io::Error;
    type Future = CachingFuture;

    fn poll_ready(&mut self, _cx: &mut task::Context<'_>) -> Poll<Result<(), io::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, name: Name) -> Self::Future {
        let blocking = {
            let cache = self.cache.clone();
            tokio::task::spawn_blocking(move || {
                let entry = cache.lock().unwrap().get(&name).cloned();

                if let Some((when, addrs)) = entry {
                    trace!("cached host={:?}", name.as_str());

                    if when.elapsed().as_secs() > 60 {
                        trace!("refreshing cache for host={:?}", name.as_str());
                        // Start a new task to update the cache later.
                        tokio::task::spawn_blocking(move || {
                            if let Ok(addrs) = (name.as_str(), 0).to_socket_addrs() {
                                let addrs: Vec<_> = addrs.collect();
                                trace!("updating cached host={:?}", name.as_str());
                                cache.lock().unwrap().insert(name, (Instant::now(), addrs.clone()));
                            }
                        });
                    }

                    Ok(addrs.into_iter())
                } else {
                    trace!("resolving host={:?}", name.as_str());
                    match (name.as_str(), 0).to_socket_addrs() {
                        Ok(addrs) => {
                            let addrs: Vec<_> = addrs.collect();
                            cache.lock().unwrap().insert(name, (Instant::now(), addrs.clone()));
                            Ok(addrs.into_iter())
                        }
                        res => res,
                    }
                }
            })
        };

        CachingFuture { inner: blocking }
    }
}

impl fmt::Debug for CachingResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("CachingResolver")
    }
}

impl Future for CachingFuture {
    type Output = Result<SocketAddrs, io::Error>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.inner).poll(cx).map(|res| match res {
            Ok(Ok(addrs)) => Ok(addrs),
            Ok(Err(err)) => Err(err),
            Err(join_err) => {
                if join_err.is_cancelled() {
                    Err(io::Error::new(io::ErrorKind::Interrupted, join_err))
                } else {
                    panic!("background task failed: {:?}", join_err)
                }
            }
        })
    }
}

impl fmt::Debug for CachingFuture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("CachingFuture")
    }
}

impl Drop for CachingFuture {
    fn drop(&mut self) {
        self.inner.abort();
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
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

    pub async fn connect(&self, addr: &Multiaddr, tls_config: ClientConfig) -> Result<Channel> {
        connect_with_config(addr, tls_config, self).await
    }

    pub fn connect_lazy(&self, addr: &Multiaddr, tls_config: ClientConfig) -> Result<Channel> {
        connect_lazy_with_config(addr, tls_config, self)
    }

    #[cfg(feature = "server")]
    pub fn http_config(&self) -> soma_http::Config {
        soma_http::Config::default()
            .initial_stream_window_size(self.http2_initial_stream_window_size)
            .initial_connection_window_size(self.http2_initial_connection_window_size)
            .max_concurrent_streams(self.http2_max_concurrent_streams)
            .http2_keepalive_timeout(self.http2_keepalive_timeout)
            .http2_keepalive_interval(self.http2_keepalive_interval)
            .tcp_keepalive(self.tcp_keepalive)
    }
}
