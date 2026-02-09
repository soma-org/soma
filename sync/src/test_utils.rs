use rand::{SeedableRng as _, rngs::StdRng};
use tokio::sync::{broadcast, mpsc};
use types::{
    crypto::NetworkKeyPair,
    multiaddr::Multiaddr,
    storage::write_store::WriteStore,
    sync::{
        PeerEvent,
        active_peers::ActivePeers,
        channel_manager::{ChannelManager, ChannelManagerRequest},
    },
};

use crate::{server::P2pService, tonic_gen::p2p_server::P2pServer};

// Helper function to create a test channel manager
// pub(crate) async fn create_test_channel_manager<S>(
//     own_address: Multiaddr,
//     server: P2pServer<P2pService<S>>,
// ) -> (
//     mpsc::Sender<ChannelManagerRequest>,
//     broadcast::Receiver<PeerEvent>,
//     ActivePeers,
//     NetworkKeyPair,
// )
// where
//     S: WriteStore + Clone + Send + Sync + 'static,
// {
//     let mut rng = StdRng::from_seed([0; 32]);
//     let active_peers = ActivePeers::new(1000);
//     let network_key_pair = NetworkKeyPair::generate(&mut rng);

//     let (manager, tx) = ChannelManager::new(
//         own_address,
//         network_key_pair.clone(),
//         server,
//         active_peers.clone(),
//     );
//     let rx = manager.subscribe();
//     tokio::spawn(manager.start());
//     (tx, rx, active_peers, network_key_pair)
// }
