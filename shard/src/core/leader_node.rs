use std::sync::Arc;

use crate::{
    crypto::keys::NetworkKeyPair,
    networking::messaging::{
        encoder_tonic_service::EncoderTonicClient,
        leader_tonic_service::{LeaderTonicClient, LeaderTonicManager},
        LeaderNetworkManager,
    },
    types::context::LeaderContext,
};

use super::{
    leader_core::LeaderCore,
    leader_core_thread::{LeaderChannelCoreThreadDispatcher, LeaderCoreThreadHandle},
    leader_service::LeaderService,
};

pub struct Leader(LeaderNode<LeaderTonicManager>);

impl Leader {
    pub async fn start(
        leader_context: Arc<LeaderContext>,
        network_keypair: NetworkKeyPair,
        encoder_client: EncoderTonicClient<LeaderContext>,
    ) -> Self {
        let leader_node: LeaderNode<LeaderTonicManager> =
            LeaderNode::start(leader_context, network_keypair, encoder_client).await;
        Self(leader_node)
    }
    pub async fn stop(self) {
        self.0.stop().await;
    }
}

pub(crate) struct LeaderNode<N>
where
    N: LeaderNetworkManager<LeaderService<LeaderChannelCoreThreadDispatcher>>,
{
    core_thread_handle: LeaderCoreThreadHandle,
    network_manager: N,
}

impl<N> LeaderNode<N>
where
    N: LeaderNetworkManager<LeaderService<LeaderChannelCoreThreadDispatcher>>,
{
    pub(crate) async fn start(
        leader_context: Arc<LeaderContext>,
        network_keypair: NetworkKeyPair,
        encoder_client: EncoderTonicClient<LeaderContext>,
    ) -> Self {
        let mut network_manager = N::new(leader_context, network_keypair);
        let core = LeaderCore::new(100_usize, encoder_client);
        let (core_dispatcher, core_thread_handle) = LeaderChannelCoreThreadDispatcher::start(core);
        let core_dispatcher = Arc::new(core_dispatcher);
        let network_service = Arc::new(LeaderService::new(core_dispatcher));
        network_manager.start(network_service).await;
        Self {
            core_thread_handle,
            network_manager,
        }
    }

    pub(crate) async fn stop(mut self) {
        self.network_manager.stop().await;
    }
}
