pub mod service;
pub mod tonic;
use async_trait::async_trait;
use bytes::Bytes;
use soma_network::multiaddr::Multiaddr;
use std::{sync::Arc, time::Duration};

use crate::{error::ProbeResult, parameters::Parameters, ProbeInput, ProbeOutput};

#[async_trait]
pub trait ProbeClient: Send + Sync + Sized + 'static {
    async fn probe(&self, probe_input: ProbeInput, timeout: Duration) -> ProbeResult<ProbeOutput>;
}

#[async_trait]
pub trait ProbeService: Send + Sync + Sized + 'static {
    async fn handle_probe(&self, probe_input_bytes: Bytes) -> ProbeResult<Bytes>;
}

pub trait ProbeManager<P>: Send + Sync
where
    P: ProbeService,
{
    /// Creates a new network manager
    fn new(parameters: Arc<Parameters>, address: Multiaddr) -> Self;
    /// Starts the network services
    async fn start(&mut self, service: Arc<P>);
    /// Stops the network services
    async fn stop(&mut self);
}
