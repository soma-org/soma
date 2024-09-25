use crate::networking::messaging::to_host_port_str;
use crate::types::multiaddr::{Multiaddr, Protocol};
use crate::types::network_committee::NetworkIdentityIndex;
use crate::{
    error::{ShardError, ShardResult},
    types::context::NetworkingContext,
};
use parking_lot::RwLock;
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, TraceLayer};
use tracing::{trace, warn};

pub(crate) struct ChannelPool<N: NetworkingContext> {
    context: Arc<N>,
    channels: RwLock<BTreeMap<NetworkIdentityIndex, Channel>>,
}

pub(crate) type Channel = tower_http::trace::Trace<
    tonic::transport::Channel,
    tower_http::classify::SharedClassifier<tower_http::classify::GrpcErrorsAsFailures>,
>;

impl<N> ChannelPool<N>
where
    N: NetworkingContext,
{
    pub(crate) fn new(context: Arc<N>) -> Self {
        Self {
            context,
            channels: RwLock::new(BTreeMap::new()),
        }
    }

    pub(crate) async fn get_channel(
        &self,
        peer: NetworkIdentityIndex,
        timeout: Duration,
    ) -> ShardResult<Channel> {
        {
            let channels = self.channels.read();
            if let Some(channel) = channels.get(&peer) {
                return Ok(channel.clone());
            }
        }
        let network_identity = self.context.network_committee().identity(peer);

        let address = to_host_port_str(&network_identity.address).map_err(|e| {
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
