//! `ChannelPool` stores tonic channels for re-use.
use crate::networking::messaging::to_host_port_str;
use crate::types::context::EncoderContext;
use crate::types::network_committee::NetworkingIndex;
use crate::{
    error::{ShardError, ShardResult},
    types::context::NetworkingContext,
};
use parking_lot::RwLock;
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, TraceLayer};
use tracing::{trace, warn};

/// `ChannelPool` contains the encoder context and an efficient mapping between networking index
/// and an open channel. In the future it might be better to have a single `ChannelPool` for both
/// consensus networking and the encoders to reduce the code, at which time a generic would need to
/// be used for the context to support contexts for both encoders and authorities.
pub(crate) struct ChannelPool {
    /// context allows going from index to address
    context: Arc<EncoderContext>,
    /// channels stored using a RWLock to better support the concurrent usage
    channels: RwLock<BTreeMap<NetworkingIndex, Channel>>,
}

/// Type alias since the type definition is so long
pub(crate) type Channel = tower_http::trace::Trace<
    tonic::transport::Channel,
    tower_http::classify::SharedClassifier<tower_http::classify::GrpcErrorsAsFailures>,
>;

/// `ChannelPool` implements methods to create a new channel pool and get a channel from a
/// pre-existing channel pool. Note under the current construction, you would need to create
/// a new channel and reestablish the channels every epoch.
impl ChannelPool {
    /// the new fn takes an encoder context and establishes a new
    /// RwLock to hold the btree of index to channel maps
    pub(crate) const fn new(context: Arc<EncoderContext>) -> Self {
        Self {
            context,
            channels: RwLock::new(BTreeMap::new()),
        }
    }

    /// the get channel method first attempts to look up the channel inside the RwLocked BTreeMap. It drops
    /// the lock and returns if found. Otherwise it uses the network identity to look up the multiaddress, map to a suitable
    /// tonic address, and then establish a connection. The newly connected channel is stored in the map, and returned.
    pub(crate) async fn get_channel(
        &self,
        peer: NetworkingIndex,
        timeout: Duration,
    ) -> ShardResult<Channel> {
        {
            let channels = self.channels.read();
            if let Some(channel) = channels.get(&peer) {
                return Ok(channel.clone());
            }
        }
        let network_identity = self.context.network_committee().identity(peer);

        let address = to_host_port_str(&network_identity.messaging_address).map_err(|e| {
            ShardError::NetworkConfig(format!("Cannot convert address to host:port: {e:?}"))
        })?;
        let address = format!("http://{address}");

        let endpoint = tonic::transport::Channel::from_shared(address.clone())
            .map_err(|e| ShardError::NetworkConfig(format!("Failed to create URI: {e}")))?
            .keep_alive_while_idle(true)
            .connect_timeout(timeout);

        let deadline = tokio::time::Instant::now() + timeout;

        let channel = loop {
            trace!("Connecting to endpoint at {address}");
            match endpoint.connect().await {
                Ok(channel) => break channel,
                Err(e) => {
                    warn!("Failed to connect to endpoint at {address}: {e:?}");
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

        let mut channels = self.channels.write();
        let channel = channels.entry(peer).or_insert(channel);
        Ok(channel.clone())
    }
}
