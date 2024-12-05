use std::sync::Arc;

use crate::{
    crypto::keys::NetworkKeyPair,
    networking::messaging::{tonic_network::EncoderTonicManager, EncoderNetworkManager},
    storage::datastore::mem_store::MemStore,
    types::context::EncoderContext,
    ProtocolKeyPair,
};

use super::{
    encoder_core::EncoderCore,
    encoder_service::EncoderService,
    task_manager::{ChannelTaskDispatcher, TaskManagerHandle},
};

pub struct Encoder(EncoderNode<EncoderTonicManager>);

impl Encoder {
    pub async fn start(
        encoder_context: Arc<EncoderContext>,
        network_keypair: NetworkKeyPair,
        protocol_keypair: ProtocolKeyPair,
    ) -> Self {
        let encoder_node: EncoderNode<EncoderTonicManager> =
            EncoderNode::start(encoder_context, network_keypair, protocol_keypair).await;
        Self(encoder_node)
    }
    pub async fn stop(self) {
        self.0.stop().await;
    }
}

pub(crate) struct EncoderNode<N>
where
    N: EncoderNetworkManager<EncoderService<ChannelTaskDispatcher, MemStore>>,
{
    task_manager_handle: TaskManagerHandle,
    network_manager: N,
}

impl<N> EncoderNode<N>
where
    N: EncoderNetworkManager<EncoderService<ChannelTaskDispatcher, MemStore>>,
{
    pub(crate) async fn start(
        encoder_context: Arc<EncoderContext>,
        network_keypair: NetworkKeyPair,
        protocol_keypair: ProtocolKeyPair,
    ) -> Self {
        let mut network_manager = N::new(encoder_context.clone(), network_keypair);
        let client = network_manager.client();
        let core = EncoderCore::new(100_usize, client);
        let (task_dispatcher, task_manager_handle) = ChannelTaskDispatcher::start(core);
        let task_dispatcher = Arc::new(task_dispatcher);
        let store = Arc::new(MemStore::new());
        let protocol_keypair = Arc::new(protocol_keypair);
        let network_service = Arc::new(EncoderService::new(
            encoder_context,
            task_dispatcher,
            store,
            protocol_keypair,
        ));
        network_manager.start(network_service).await;
        Self {
            task_manager_handle,
            network_manager,
        }
    }

    pub(crate) async fn stop(mut self) {
        self.network_manager.stop().await;
    }
}
