use crate::inference::{InferenceInput, InferenceOutput};
use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};
use types::{error::InferenceResult, multiaddr::Multiaddr, parameters::TonicParameters};
pub mod service;
pub mod tonic;

#[async_trait]
pub trait InferenceClient: Send + Sync + Sized + 'static {
    async fn inference(
        &self,
        input: InferenceInput,
        timeout: Duration,
    ) -> InferenceResult<InferenceOutput>;
}

#[async_trait]
pub trait InferenceService: Send + Sync + Sized + 'static {
    async fn handle_inference(&self, input_bytes: Bytes) -> InferenceResult<Bytes>;
}

pub trait InferenceServiceManager<S>: Send + Sync
where
    S: InferenceService,
{
    /// Creates a new network manager
    fn new(parameters: Arc<TonicParameters>, address: Multiaddr) -> Self;
    /// Starts the network services
    async fn start(&mut self, service: Arc<S>);
    /// Stops the network services
    async fn stop(&mut self);
}
