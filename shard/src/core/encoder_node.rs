use std::sync::Arc;

use crate::{
    crypto::keys::NetworkKeyPair,
    networking::messaging::{
        encoder_tonic_service::EncoderTonicManager, leader_tonic_service::LeaderTonicClient,
        EncoderNetworkManager,
    },
    types::context::EncoderContext,
};

use super::{
    encoder_core::EncoderCore,
    encoder_core_thread::{EncoderChannelCoreThreadDispatcher, EncoderCoreThreadHandle},
    encoder_service::EncoderService,
};

pub struct Encoder(EncoderNode<EncoderTonicManager>);

impl Encoder {
    pub async fn start(
        encoder_context: Arc<EncoderContext>,
        network_keypair: NetworkKeyPair,
        leader_client: LeaderTonicClient<EncoderContext>,
    ) -> Self {
        let encoder_node: EncoderNode<EncoderTonicManager> =
            EncoderNode::start(encoder_context, network_keypair, leader_client).await;
        Self(encoder_node)
    }
    pub async fn stop(self) {
        self.0.stop().await;
    }
}

pub(crate) struct EncoderNode<N>
where
    N: EncoderNetworkManager<EncoderService<EncoderChannelCoreThreadDispatcher>>,
{
    core_thread_handle: EncoderCoreThreadHandle,
    network_manager: N,
}

impl<N> EncoderNode<N>
where
    N: EncoderNetworkManager<EncoderService<EncoderChannelCoreThreadDispatcher>>,
{
    pub(crate) async fn start(
        encoder_context: Arc<EncoderContext>,
        network_keypair: NetworkKeyPair,
        leader_client: LeaderTonicClient<EncoderContext>,
    ) -> Self {
        let mut network_manager = N::new(encoder_context, network_keypair);
        let core = EncoderCore::new(100_usize, leader_client);
        let (core_dispatcher, core_thread_handle) = EncoderChannelCoreThreadDispatcher::start(core);
        let core_dispatcher = Arc::new(core_dispatcher);
        let network_service = Arc::new(EncoderService::new(core_dispatcher));
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
