pub(crate) mod service;
pub(crate) mod tonic;
use async_trait::async_trait;
use bytes::Bytes;
use soma_network::multiaddr::Multiaddr;
use std::{sync::Arc, time::Duration};

use crate::{error::ProbeResult, parameters::Parameters, ProbeInput};

#[async_trait]
pub(crate) trait ProbeNetworkClient: Send + Sync + Sized + 'static {
    async fn probe(&self, probe_input: ProbeInput, timeout: Duration) -> ProbeResult<()>;
}

#[async_trait]
pub(crate) trait ProbeNetworkService: Send + Sync + Sized + 'static {
    async fn handle_probe(&self, probe_input_bytes: Bytes) -> ProbeResult<Bytes>;
}

pub(crate) trait ProbeNetworkManager<P>: Send + Sync
where
    P: ProbeNetworkService,
{
    /// Creates a new network manager
    fn new(parameters: Arc<Parameters>, address: Multiaddr) -> Self;
    /// Starts the network services
    async fn start(&mut self, service: Arc<P>);
    /// Stops the network services
    async fn stop(&mut self);
}
