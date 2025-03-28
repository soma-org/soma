//! `ChannelPool` stores tonic channels for re-use.
use crate::error::{ShardError, ShardResult};
use crate::messaging::tonic::CERTIFICATE_NAME;
use crate::types::parameters::TonicParameters;
use crate::utils::multiaddr::to_host_port_str;
use quick_cache::sync::Cache;
use shared::crypto::keys::{PeerKeyPair, PeerPublicKey};
use shared::multiaddr::Multiaddr;
use std::time::Duration;
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, TraceLayer};
use tracing::{debug, trace};

/// `ChannelPool` contains the encoder context and an efficient mapping between networking index
/// and an open channel. In the future it might be better to have a single `ChannelPool` for both
/// consensus networking and the encoders to reduce the code, at which time a generic would need to
/// be used for the context to support contexts for both encoders and authorities.
pub(crate) struct ChannelPool {
    channels: Cache<(PeerPublicKey, Multiaddr), Channel>,
}

/// Type alias since the type definition is so long
pub(crate) type Channel = tower_http::trace::Trace<
    tonic_rustls::Channel,
    tower_http::classify::SharedClassifier<tower_http::classify::GrpcErrorsAsFailures>,
>;

/// `ChannelPool` implements methods to create a new channel pool and get a channel from a
/// pre-existing channel pool. Note under the current construction, you would need to create
/// a new channel and reestablish the channels every epoch.
impl ChannelPool {
    /// the new fn takes an encoder context and establishes a new
    /// RwLock to hold the btree of index to channel maps
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            channels: Cache::new(capacity),
        }
    }
    pub(crate) async fn get_channel(
        &self,
        address: &Multiaddr,
        peer_public_key: PeerPublicKey,
        config: &TonicParameters,
        peer_keypair: PeerKeyPair,
        timeout: Duration,
    ) -> ShardResult<Channel> {
        let cache_key = (peer_public_key.clone(), address.clone());

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
