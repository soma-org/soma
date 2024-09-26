use async_trait::async_trait;
use bytes::Bytes;
use std::sync::Arc;

use crate::{
    error::{ShardError, ShardResult},
    networking::messaging::EncoderNetworkService,
    types::{
        network_committee::NetworkIdentityIndex, shard_input::SignedShardInput,
        shard_selection::SignedShardSelection,
    },
};

use super::encoder_core_thread::EncoderCoreThreadDispatcher;

pub(crate) struct EncoderService<C: EncoderCoreThreadDispatcher> {
    core_dispatcher: Arc<C>,
}

impl<C: EncoderCoreThreadDispatcher> EncoderService<C> {
    pub(crate) fn new(core_dispatcher: Arc<C>) -> Self {
        println!("configured core thread");
        Self { core_dispatcher }
    }
}

#[async_trait]
impl<C: EncoderCoreThreadDispatcher> EncoderNetworkService for EncoderService<C> {
    async fn handle_send_input(&self, peer: NetworkIdentityIndex, input: Bytes) -> ShardResult<()> {
        let signed_input: SignedShardInput =
            bcs::from_bytes(&input).map_err(ShardError::MalformedInput)?;
        // TODO: look up key with the network identity index
        // signed_input.verify_signature()?;
        let verified_input = signed_input.verify(|input| Ok(()))?;

        self.core_dispatcher.process_input(verified_input).await?;
        Ok(())
    }
    async fn handle_send_selection(
        &self,
        peer: NetworkIdentityIndex,
        selection: Bytes,
    ) -> ShardResult<()> {
        let signed_selection: SignedShardSelection =
            bcs::from_bytes(&selection).map_err(ShardError::MalformedSelection)?;
        // TODO: verify signature
        let verified_selection = signed_selection.verify(|selection| Ok(()))?;
        self.core_dispatcher
            .process_selection(verified_selection)
            .await?;
        Ok(())
    }
}
