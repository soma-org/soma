use crate::{
    server::P2pService,
    tonic_gen::p2p_server::{P2p, P2pServer},
};

use super::*;
use builder::{DiscoveryBuilder, UnstartedDiscovery};
use fastcrypto::ed25519::Ed25519Signature;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashSet;
use tokio::sync::{broadcast, mpsc};
use types::{error::SomaResult, p2p::channel_manager::ChannelManager};

// Helper function to create a test channel manager
async fn create_test_channel_manager(
    own_address: Multiaddr,
    server: P2pServer<P2pService>,
) -> (
    mpsc::Sender<ChannelManagerRequest>,
    broadcast::Receiver<PeerEvent>,
    ActivePeers,
    NetworkKeyPair,
) {
    let mut rng = StdRng::from_seed([0; 32]);
    let active_peers = ActivePeers::new(1000);
    let network_key_pair = NetworkKeyPair::generate(&mut rng);

    let (manager, tx) = ChannelManager::new(
        own_address,
        network_key_pair.clone(),
        server,
        active_peers.clone(),
    );
    let rx = manager.subscribe();
    tokio::spawn(manager.start());
    (tx, rx, active_peers, network_key_pair)
}

#[tokio::test]
async fn get_known_peers() -> SomaResult<()> {
    let peer_1_addr: Multiaddr = "/ip4/127.0.0.1/tcp/1234".parse().unwrap();
    let config = P2pConfig {
        external_address: Some(peer_1_addr.clone()),
        ..Default::default()
    };
    let (UnstartedDiscovery { state, .. }, server) =
        DiscoveryBuilder::new().config(config).build_internal();

    // Error when own_info not set
    server
        .get_known_peers(Request::new(GetKnownPeersRequest { timestamp_ms: 0 }))
        .await
        .unwrap_err();

    // Normal response with our_info

    let our_info = NodeInfo {
        peer_id: PeerId([9; 32]),
        address: peer_1_addr,
        timestamp_ms: now_unix(),
    };
    state.write().our_info = Some(SignedNodeInfo::new_from_data_and_sig(
        our_info.clone(),
        Ed25519Signature::default(),
    ));
    let response = server
        .get_known_peers(Request::new(GetKnownPeersRequest { timestamp_ms: 0 }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.own_info.data(), &our_info);
    assert!(response.known_peers.is_empty());

    // Normal response with some known peers
    let peer_2_addr = "/ip4/127.0.0.1/tcp/1235".parse().unwrap();
    let other_peer = NodeInfo {
        peer_id: PeerId([13; 32]),
        address: peer_2_addr,
        timestamp_ms: now_unix(),
    };
    state.write().known_peers.insert(
        other_peer.peer_id,
        VerifiedSignedNodeInfo::new_unchecked(SignedNodeInfo::new_from_data_and_sig(
            other_peer.clone(),
            Ed25519Signature::default(),
        )),
    );
    let response = server
        .get_known_peers(Request::new(GetKnownPeersRequest { timestamp_ms: 0 }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.own_info.data(), &our_info);
    assert_eq!(
        response
            .known_peers
            .into_iter()
            .map(|peer| peer.into_data())
            .collect::<Vec<_>>(),
        vec![other_peer]
    );

    Ok(())
}

#[tokio::test]
async fn make_connection_to_seed_peer() -> SomaResult<()> {
    let peer_1_addr: Multiaddr = "/ip4/127.0.0.1/tcp/1234".parse().unwrap();
    let config1 = P2pConfig {
        external_address: Some(peer_1_addr.clone()),
        ..Default::default()
    };

    // Setup first peer
    let (builder_1, server_1) = DiscoveryBuilder::new().config(config1).build();
    let (manager_tx_1, mut events_1, active_peers_1, network_key_pair_1) =
        create_test_channel_manager(peer_1_addr.clone(), server_1).await;

    // Setup second peer with first as seed
    let peer_2_addr: Multiaddr = "/ip4/127.0.0.1/tcp/1235".parse().unwrap();
    let config2 = P2pConfig {
        external_address: Some(peer_2_addr.clone()),
        seed_peers: vec![SeedPeer {
            peer_id: Some(network_key_pair_1.public().into()),
            address: peer_1_addr.clone(),
        }],
        ..Default::default()
    };

    let (builder_2, server_2) = DiscoveryBuilder::new().config(config2).build();
    let (manager_tx_2, mut events_2, active_peers_2, network_key_pair_2) =
        create_test_channel_manager(peer_2_addr.clone(), server_2).await;

    let (_event_loop_1, _) =
        builder_1.build(active_peers_1, manager_tx_1, network_key_pair_1.clone());
    let (mut event_loop_2, _) =
        builder_2.build(active_peers_2, manager_tx_2, network_key_pair_2.clone());

    // Trigger connection
    event_loop_2.handle_tick(std::time::Instant::now(), now_unix());

    // Verify connection events
    let peer_1_id = network_key_pair_1.public().into();
    let peer_2_id = network_key_pair_2.public().into();

    assert_eq!(
        events_2.recv().await.unwrap(),
        PeerEvent::NewPeer {
            peer_id: peer_1_id,
            address: peer_1_addr,
        }
    );
    assert_eq!(
        events_1.recv().await.unwrap(),
        PeerEvent::NewPeer {
            peer_id: peer_2_id,
            address: peer_2_addr,
        }
    );

    Ok(())
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn three_nodes_can_connect_via_discovery() -> SomaResult<()> {
    let peer_1_addr: Multiaddr = "/ip4/127.0.0.1/tcp/1234".parse().unwrap();
    let config1 = P2pConfig {
        external_address: Some(peer_1_addr.clone()),
        ..Default::default()
    };

    // Setup first peer (seed)
    let (builder_1, server_1) = DiscoveryBuilder::new().config(config1).build();
    let (manager_tx_1, mut events_1, active_peers_1, network_key_pair_1) =
        create_test_channel_manager(peer_1_addr.clone(), server_1).await;

    // Setup second peer with first as seed
    let peer_2_addr: Multiaddr = "/ip4/127.0.0.1/tcp/1235".parse().unwrap();
    let config2 = P2pConfig {
        external_address: Some(peer_2_addr.clone()),
        seed_peers: vec![SeedPeer {
            peer_id: Some(network_key_pair_1.public().into()),
            address: peer_1_addr.clone(),
        }],
        ..Default::default()
    };

    let (builder_2, server_2) = DiscoveryBuilder::new().config(config2).build();
    let (manager_tx_2, mut events_2, active_peers_2, network_key_pair_2) =
        create_test_channel_manager(peer_2_addr.clone(), server_2).await;

    // Setup third peer with same seed
    let peer_3_addr: Multiaddr = "/ip4/127.0.0.1/tcp/1236".parse().unwrap();
    let config3 = P2pConfig {
        external_address: Some(peer_3_addr.clone()),
        seed_peers: vec![SeedPeer {
            peer_id: Some(network_key_pair_1.public().into()),
            address: peer_1_addr.clone(),
        }],
        ..Default::default()
    };
    let (builder_3, server_3) = DiscoveryBuilder::new().config(config3).build();
    let (manager_tx_3, mut events_3, active_peers_3, network_key_pair_3) =
        create_test_channel_manager(peer_3_addr.clone(), server_3).await;

    // Start discovery loops
    let (event_loop_1, _) =
        builder_1.build(active_peers_1, manager_tx_1, network_key_pair_1.clone());
    let (event_loop_2, _) =
        builder_2.build(active_peers_2, manager_tx_2, network_key_pair_2.clone());
    let (event_loop_3, _) =
        builder_3.build(active_peers_3, manager_tx_3, network_key_pair_3.clone());

    tokio::spawn(event_loop_1.start());
    tokio::spawn(event_loop_2.start());
    tokio::spawn(event_loop_3.start());

    // Each peer should get two new peer events
    let mut peers_1 = HashSet::new();
    let mut peers_2 = HashSet::new();
    let mut peers_3 = HashSet::new();

    for _ in 0..2 {
        if let Ok(PeerEvent::NewPeer { peer_id, .. }) = events_1.recv().await {
            peers_1.insert(peer_id);
        }
    }

    for _ in 0..2 {
        if let Ok(PeerEvent::NewPeer { peer_id, .. }) = events_2.recv().await {
            peers_2.insert(peer_id);
        }
    }

    for _ in 0..2 {
        if let Ok(PeerEvent::NewPeer { peer_id, .. }) = events_3.recv().await {
            peers_3.insert(peer_id);
        }
    }

    // Verify connections
    assert!(peers_1.contains(&network_key_pair_2.public().into()));
    assert!(peers_1.contains(&network_key_pair_3.public().into()));
    assert!(peers_2.contains(&network_key_pair_1.public().into()));
    assert!(peers_2.contains(&network_key_pair_3.public().into()));
    assert!(peers_3.contains(&network_key_pair_1.public().into()));
    assert!(peers_3.contains(&network_key_pair_2.public().into()));

    Ok(())
}
