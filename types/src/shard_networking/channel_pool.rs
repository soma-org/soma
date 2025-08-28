//! `ChannelPool` stores tonic channels for re-use.
use quick_cache::sync::Cache;
use shared::crypto::keys::{PeerKeyPair, PeerPublicKey};
use shared::error::{ShardError, ShardResult};
use shared::parameters::TonicParameters;
use soma_network::multiaddr::{to_host_port_str, Multiaddr};
use soma_network::CERTIFICATE_NAME;
use std::time::Duration;
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, TraceLayer};
use tracing::{debug, trace};

pub struct ChannelPool {
    channels: Cache<(PeerPublicKey, Multiaddr), Channel>,
}

/// Type alias since the type definition is so long
pub type Channel = tower_http::trace::Trace<
    tonic_rustls::Channel,
    tower_http::classify::SharedClassifier<tower_http::classify::GrpcErrorsAsFailures>,
>;

impl ChannelPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            channels: Cache::new(capacity),
        }
    }
    pub async fn get_channel(
        &self,
        address: &Multiaddr,
        peer_public_key: PeerPublicKey,
        config: &TonicParameters,
        peer_keypair: PeerKeyPair,
        timeout: Duration,
    ) -> ShardResult<Channel> {
        let cache_key = (peer_public_key.clone(), address.clone());

        // TODO: potentially add a TTL value to the channel such that past some amount of time, the
        // channel will be refreshed.
        if let Some(channel) = self.channels.get(&cache_key) {
            return Ok(channel);
        }

        let address = to_host_port_str(address).map_err(|e| {
            ShardError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        let address = format!("https://{address}");
        let buffer_size = config.connection_buffer_size;
        let client_tls_config = soma_tls::create_rustls_client_config(
            peer_public_key.clone().into_inner(),
            CERTIFICATE_NAME.to_string(),
            Some(peer_keypair.private_key().into_inner()),
        );
        let endpoint = tonic_rustls::Channel::from_shared(address.clone())
            .unwrap()
            .connect_timeout(timeout)
            .initial_connection_window_size(Some(buffer_size as u32))
            .initial_stream_window_size(Some(buffer_size as u32 / 2))
            .keep_alive_while_idle(true)
            .keep_alive_timeout(config.keepalive_interval)
            .http2_keep_alive_interval(config.keepalive_interval)
            // tcp keepalive is probably unnecessary and is unsupported by msim.
            .user_agent("soma")
            .unwrap()
            .tls_config(client_tls_config)
            .unwrap();

        let deadline = tokio::time::Instant::now() + timeout;
        let channel = loop {
            trace!("Connecting to endpoint at {address}");
            match endpoint.connect().await {
                Ok(channel) => break channel,
                Err(e) => {
                    debug!("Failed to connect to endpoint at {address}: {e:?}");
                    if tokio::time::Instant::now() >= deadline {
                        return Err(ShardError::NetworkClientConnection(format!(
                            "Timed out connecting to endpoint at {address}: {e:?}"
                        )));
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        };
        trace!("Connected to {address}");

        let channel = tower::ServiceBuilder::new()
            .layer(
                TraceLayer::new_for_grpc()
                    .make_span_with(DefaultMakeSpan::new().level(tracing::Level::TRACE))
                    .on_failure(DefaultOnFailure::new().level(tracing::Level::DEBUG)),
            )
            .service(channel);

        self.channels.insert(cache_key, channel.clone());
        Ok(channel)
    }
}
