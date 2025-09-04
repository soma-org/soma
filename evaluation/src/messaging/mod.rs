pub mod service;
pub mod tonic;
use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};
use types::error::EvaluationResult;
use types::multiaddr::Multiaddr;
use types::{
    evaluation::{EvaluationInput, EvaluationOutput},
    parameters::TonicParameters,
};

#[async_trait]
pub trait EvaluationClient: Send + Sync + Sized + 'static {
    async fn evaluation(
        &self,
        input: EvaluationInput,
        timeout: Duration,
    ) -> EvaluationResult<EvaluationOutput>;
}

#[async_trait]
pub trait EvaluationService: Send + Sync + Sized + 'static {
    async fn handle_evaluation(&self, input_bytes: Bytes) -> EvaluationResult<Bytes>;
}

pub trait EvaluationManager<P>: Send + Sync
where
    P: EvaluationService,
{
    /// Creates a new network manager
    fn new(parameters: Arc<TonicParameters>, address: Multiaddr) -> Self;
    /// Starts the network services
    async fn start(&mut self, service: Arc<P>);
    /// Stops the network services
    async fn stop(&mut self);
}
