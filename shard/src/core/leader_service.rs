use async_trait::async_trait;
use bytes::Bytes;
use std::sync::Arc;

use crate::{
    error::{ShardError, ShardResult},
    networking::messaging::LeaderNetworkService,
    types::{
        network_committee::NetworkIdentityIndex, shard_commit::SignedShardCommit,
        shard_endorsement::SignedShardEndorsement, shard_input::SignedShardInput,
        shard_selection::SignedShardSelection,
    },
};

use super::leader_core_thread::LeaderCoreThreadDispatcher;

pub(crate) struct LeaderService<C: LeaderCoreThreadDispatcher> {
    core_dispatcher: Arc<C>,
}

impl<C: LeaderCoreThreadDispatcher> LeaderService<C> {
    pub(crate) fn new(core_dispatcher: Arc<C>) -> Self {
        println!("configured core thread");
        Self { core_dispatcher }
    }
}

#[async_trait]
impl<C: LeaderCoreThreadDispatcher> LeaderNetworkService for LeaderService<C> {
    async fn handle_send_commit(
        &self,
        peer: NetworkIdentityIndex,
        commit: Bytes,
    ) -> ShardResult<()> {
        let signed_commit: SignedShardCommit =
            bcs::from_bytes(&commit).map_err(ShardError::MalformedCommit)?;
        // TODO: look up key with the network identity index
        // TODO: verify signature
        let verified_commit = signed_commit.verify(|commit| Ok(()))?;

        self.core_dispatcher.process_commit(verified_commit).await?;
        Ok(())
    }
    async fn handle_send_endorsement(
        &self,
        peer: NetworkIdentityIndex,
        endorsement: Bytes,
    ) -> ShardResult<()> {
        let signed_endorsement: SignedShardEndorsement =
            bcs::from_bytes(&endorsement).map_err(ShardError::MalformedEndorsement)?;
        // TODO: verify signature
        let verified_endorsement = signed_endorsement.verify(|endorsement| Ok(()))?;
        self.core_dispatcher
            .process_endorsement(verified_endorsement)
            .await?;
        Ok(())
    }
}
