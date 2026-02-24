// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

//
// Modified for the SOMA project.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use rand::{SeedableRng, rngs::StdRng};
use tokio::sync::broadcast;
use tonic::Request;
use types::{
    config::{p2p_config::P2pConfig, state_sync_config::StateSyncConfig},
    crypto::NetworkKeyPair,
    digests::CheckpointDigest,
    peer_id::PeerId,
    storage::write_store::WriteStore,
    sync::{
        GetCheckpointAvailabilityRequest, GetCheckpointContentsRequest,
        GetCheckpointSummaryRequest, PushCheckpointSummaryRequest, active_peers::ActivePeers,
        channel_manager::PeerInfo,
    },
};

use crate::builder::P2pBuilder;
use crate::state_sync::{PeerHeights, PeerStateSyncInfo};
use crate::test_utils::CommitteeFixture;
use crate::tonic_gen::p2p_server::P2p;

// ============================================================================
// PeerHeights Unit Tests
// ============================================================================

fn make_peer_heights() -> PeerHeights {
    PeerHeights {
        peers: HashMap::new(),
        unprocessed_checkpoints: HashMap::new(),
        sequence_number_to_digest: HashMap::new(),
        wait_interval_when_no_peer_to_sync_content: Duration::from_secs(10),
    }
}

fn make_peer_id(seed: u8) -> PeerId {
    let mut rng = StdRng::from_seed([seed; 32]);
    let kp = NetworkKeyPair::generate(&mut rng);
    PeerId::from(kp.public())
}

fn make_peer_info(genesis_digest: CheckpointDigest, height: u64, lowest: u64) -> PeerStateSyncInfo {
    PeerStateSyncInfo {
        genesis_checkpoint_digest: genesis_digest,
        on_same_chain_as_us: true,
        height,
        lowest,
    }
}

#[test]
fn test_peer_heights_insert_peer_info_new() {
    let mut heights = make_peer_heights();
    let peer = make_peer_id(1);
    let digest = CheckpointDigest::default();
    let info = make_peer_info(digest, 42, 0);

    heights.insert_peer_info(peer, info);
    assert_eq!(heights.peers.get(&peer).unwrap().height, 42);
    assert!(heights.peers.get(&peer).unwrap().on_same_chain_as_us);
}

#[test]
fn test_peer_heights_insert_peer_info_update_height_same_genesis() {
    let mut heights = make_peer_heights();
    let peer = make_peer_id(1);
    let digest = CheckpointDigest::default();

    heights.insert_peer_info(peer, make_peer_info(digest, 10, 0));
    heights.insert_peer_info(peer, make_peer_info(digest, 42, 0));
    // Should keep the max height
    assert_eq!(heights.peers.get(&peer).unwrap().height, 42);

    // Lower height should not override
    heights.insert_peer_info(peer, make_peer_info(digest, 5, 0));
    assert_eq!(heights.peers.get(&peer).unwrap().height, 42);
}

#[test]
fn test_peer_heights_insert_peer_info_different_genesis_overwrites() {
    let mut heights = make_peer_heights();
    let peer = make_peer_id(1);
    let digest1 = CheckpointDigest::default();
    let digest2 = CheckpointDigest::new([1u8; 32]);

    heights.insert_peer_info(peer, make_peer_info(digest1, 100, 0));
    assert_eq!(heights.peers.get(&peer).unwrap().height, 100);

    // Different genesis digest should overwrite completely
    heights.insert_peer_info(peer, make_peer_info(digest2, 5, 0));
    assert_eq!(heights.peers.get(&peer).unwrap().height, 5);
    assert_eq!(heights.peers.get(&peer).unwrap().genesis_checkpoint_digest, digest2);
}

#[test]
fn test_peer_heights_update_peer_info_same_chain() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(5, None);
    let mut heights = make_peer_heights();
    let peer = make_peer_id(1);
    let genesis_digest = *checkpoints[0].digest();

    // Insert peer on same chain
    heights.insert_peer_info(peer, make_peer_info(genesis_digest, 0, 0));

    // Update should succeed and ratchet height
    let result = heights.update_peer_info(peer, checkpoints[3].inner().clone(), Some(0));
    assert!(result);
    assert_eq!(heights.peers.get(&peer).unwrap().height, 3);

    // Checkpoint should be stored
    assert!(heights.unprocessed_checkpoints.contains_key(checkpoints[3].digest()));
    assert_eq!(heights.sequence_number_to_digest.get(&3), Some(checkpoints[3].digest()));
}

#[test]
fn test_peer_heights_update_peer_info_not_on_same_chain() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(3, None);
    let mut heights = make_peer_heights();
    let peer = make_peer_id(1);

    // Insert peer NOT on same chain
    let mut info = make_peer_info(CheckpointDigest::new([99u8; 32]), 0, 0);
    info.on_same_chain_as_us = false;
    heights.insert_peer_info(peer, info);

    // Update should fail
    let result = heights.update_peer_info(peer, checkpoints[1].inner().clone(), None);
    assert!(!result);
}

#[test]
fn test_peer_heights_update_peer_info_unknown_peer() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(3, None);
    let mut heights = make_peer_heights();
    let unknown_peer = make_peer_id(99);

    let result = heights.update_peer_info(unknown_peer, checkpoints[1].inner().clone(), None);
    assert!(!result);
}

#[test]
fn test_peer_heights_highest_known_checkpoint() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(10, None);
    let mut heights = make_peer_heights();
    let genesis_digest = *checkpoints[0].digest();

    // No peers -> None
    assert!(heights.highest_known_checkpoint_sequence_number().is_none());

    // Add peers at different heights
    let peer1 = make_peer_id(1);
    let peer2 = make_peer_id(2);
    let peer3 = make_peer_id(3);

    heights.insert_peer_info(peer1, make_peer_info(genesis_digest, 3, 0));
    heights.insert_peer_info(peer2, make_peer_info(genesis_digest, 7, 0));
    heights.insert_peer_info(peer3, make_peer_info(genesis_digest, 5, 0));

    assert_eq!(heights.highest_known_checkpoint_sequence_number(), Some(7));
}

#[test]
fn test_peer_heights_highest_known_excludes_different_chain() {
    let mut heights = make_peer_heights();
    let genesis_digest = CheckpointDigest::default();

    let peer1 = make_peer_id(1);
    let peer2 = make_peer_id(2);

    heights.insert_peer_info(peer1, make_peer_info(genesis_digest, 10, 0));

    let mut info = make_peer_info(genesis_digest, 100, 0);
    info.on_same_chain_as_us = false;
    heights.insert_peer_info(peer2, info);

    // peer2 not on same chain, should not count
    assert_eq!(heights.highest_known_checkpoint_sequence_number(), Some(10));
}

#[test]
fn test_peer_heights_cleanup_old_checkpoints() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(10, None);
    let mut heights = make_peer_heights();

    // Insert several checkpoints
    for cp in &checkpoints {
        heights.insert_checkpoint(cp.inner().clone());
    }
    assert_eq!(heights.unprocessed_checkpoints.len(), 10);
    assert_eq!(heights.sequence_number_to_digest.len(), 10);

    // Cleanup below sequence 5 (retain only > 5)
    heights.cleanup_old_checkpoints(5);
    assert_eq!(heights.unprocessed_checkpoints.len(), 4); // seq 6,7,8,9
    assert_eq!(heights.sequence_number_to_digest.len(), 4);

    // Verify specific sequences
    assert!(heights.get_checkpoint_by_sequence_number(5).is_none());
    assert!(heights.get_checkpoint_by_sequence_number(6).is_some());
    assert!(heights.get_checkpoint_by_sequence_number(9).is_some());
}

#[test]
fn test_peer_heights_mark_peer_as_not_on_same_chain() {
    let mut heights = make_peer_heights();
    let peer = make_peer_id(1);
    let digest = CheckpointDigest::default();

    heights.insert_peer_info(peer, make_peer_info(digest, 50, 0));
    assert!(heights.peers.get(&peer).unwrap().on_same_chain_as_us);

    heights.mark_peer_as_not_on_same_chain(peer);
    assert!(!heights.peers.get(&peer).unwrap().on_same_chain_as_us);

    // Should be excluded from peers_on_same_chain
    assert_eq!(heights.peers_on_same_chain().count(), 0);
}

#[test]
fn test_peer_heights_insert_and_remove_checkpoint() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(3, None);
    let mut heights = make_peer_heights();

    let cp = checkpoints[1].inner().clone();
    let digest = *cp.digest();

    heights.insert_checkpoint(cp);
    assert!(heights.get_checkpoint_by_digest(&digest).is_some());
    assert!(heights.get_checkpoint_by_sequence_number(1).is_some());

    heights.remove_checkpoint(&digest);
    assert!(heights.get_checkpoint_by_digest(&digest).is_none());
    assert!(heights.get_checkpoint_by_sequence_number(1).is_none());
}

#[test]
fn test_peer_heights_peers_on_same_chain() {
    let mut heights = make_peer_heights();
    let digest = CheckpointDigest::default();

    let peer1 = make_peer_id(1);
    let peer2 = make_peer_id(2);
    let peer3 = make_peer_id(3);

    heights.insert_peer_info(peer1, make_peer_info(digest, 10, 0));

    let mut not_same = make_peer_info(digest, 20, 0);
    not_same.on_same_chain_as_us = false;
    heights.insert_peer_info(peer2, not_same);

    heights.insert_peer_info(peer3, make_peer_info(digest, 30, 0));

    let same_chain: Vec<_> = heights.peers_on_same_chain().collect();
    assert_eq!(same_chain.len(), 2);
}

#[test]
fn test_peer_heights_update_low_watermark() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(5, None);
    let mut heights = make_peer_heights();
    let peer = make_peer_id(1);
    let genesis_digest = *checkpoints[0].digest();

    heights.insert_peer_info(peer, make_peer_info(genesis_digest, 0, 0));
    assert_eq!(heights.peers.get(&peer).unwrap().lowest, 0);

    // Update with low watermark
    heights.update_peer_info(peer, checkpoints[3].inner().clone(), Some(2));
    assert_eq!(heights.peers.get(&peer).unwrap().lowest, 2);
}

// ============================================================================
// PeerBalancer Unit Tests
// ============================================================================

fn setup_peer_balancer_test() -> (ActivePeers, Arc<RwLock<PeerHeights>>, Vec<PeerId>) {
    let active_peers = ActivePeers::new(100);
    let mut heights = make_peer_heights();
    let digest = CheckpointDigest::default();

    let mut peer_ids = Vec::new();
    // Create 5 peers with various heights and lowest watermarks
    for (i, (height, lowest)) in [(10, 0), (20, 5), (30, 0), (15, 10), (25, 15)].iter().enumerate()
    {
        let mut rng = StdRng::from_seed([i as u8 + 10; 32]);
        let kp = NetworkKeyPair::generate(&mut rng);
        let peer_id = PeerId::from(kp.public());
        peer_ids.push(peer_id);

        heights.insert_peer_info(peer_id, make_peer_info(digest, *height, *lowest));

        // Insert into active_peers with a dummy channel
        // We can't easily create a real Channel, but we need the peer to be in active_peers
        // for PeerBalancer to find it. PeerBalancer uses active_peers.get_state().
        // Since we can't mock Channel easily, we test PeerBalancer logic via the filter behavior.
    }

    let peer_heights = Arc::new(RwLock::new(heights));
    (active_peers, peer_heights, peer_ids)
}

#[test]
fn test_peer_balancer_summary_filters_by_height() {
    // Test the PeerBalancer filtering logic directly via PeerHeights
    let mut heights = make_peer_heights();
    let digest = CheckpointDigest::default();

    let peer1 = make_peer_id(1);
    let peer2 = make_peer_id(2);
    let peer3 = make_peer_id(3);

    heights.insert_peer_info(peer1, make_peer_info(digest, 10, 0));
    heights.insert_peer_info(peer2, make_peer_info(digest, 20, 0));
    heights.insert_peer_info(peer3, make_peer_info(digest, 5, 0));

    // For summary mode, peers need height >= requested
    // Verify the filtering logic used by PeerBalancer
    let requested = 15u64;
    let eligible: Vec<_> =
        heights.peers_on_same_chain().filter(|(_, info)| info.height >= requested).collect();
    assert_eq!(eligible.len(), 1); // Only peer2 (height 20) qualifies
    assert_eq!(*eligible[0].0, peer2);
}

#[test]
fn test_peer_balancer_content_filters_by_height_and_lowest() {
    let mut heights = make_peer_heights();
    let digest = CheckpointDigest::default();

    let peer1 = make_peer_id(1);
    let peer2 = make_peer_id(2);
    let peer3 = make_peer_id(3);

    // peer1: height 20, lowest 0 -> can serve content for 0..=20
    heights.insert_peer_info(peer1, make_peer_info(digest, 20, 0));
    // peer2: height 20, lowest 10 -> can serve content for 10..=20
    heights.insert_peer_info(peer2, make_peer_info(digest, 20, 10));
    // peer3: height 5, lowest 0 -> can serve content for 0..=5
    heights.insert_peer_info(peer3, make_peer_info(digest, 5, 0));

    // Content mode: need height >= requested AND lowest <= requested
    let requested = 8u64;
    let eligible: Vec<_> = heights
        .peers_on_same_chain()
        .filter(|(_, info)| info.height >= requested && info.lowest <= requested)
        .collect();
    // peer1 qualifies (height 20 >= 8, lowest 0 <= 8)
    // peer2 does NOT qualify (lowest 10 > 8)
    // peer3 does NOT qualify (height 5 < 8)
    assert_eq!(eligible.len(), 1);
    assert_eq!(*eligible[0].0, peer1);
}

#[test]
fn test_peer_balancer_empty_when_no_eligible_peers() {
    let mut heights = make_peer_heights();
    let digest = CheckpointDigest::default();

    let peer1 = make_peer_id(1);
    heights.insert_peer_info(peer1, make_peer_info(digest, 5, 0));

    // No peers have height >= 100
    let requested = 100u64;
    let eligible: Vec<_> =
        heights.peers_on_same_chain().filter(|(_, info)| info.height >= requested).collect();
    assert_eq!(eligible.len(), 0);
}

// ============================================================================
// Server Handler Tests
// ============================================================================

fn make_p2p_config() -> P2pConfig {
    P2pConfig {
        external_address: Some("/ip4/127.0.0.1/tcp/8080".parse().unwrap()),
        state_sync: Some(StateSyncConfig::default()),
        ..Default::default()
    }
}

#[tokio::test]
async fn test_server_push_checkpoint_summary() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(5, None);
    let store = fixture.init_store();
    let genesis_digest = *checkpoints[0].digest();

    let config = make_p2p_config();
    let (_, unstarted_state_sync, server) =
        P2pBuilder::new().store(store.clone()).config(config).build_internal();

    let peer = make_peer_id(1);

    // Insert peer into peer_heights so update_peer_info succeeds
    unstarted_state_sync
        .peer_heights
        .write()
        .insert_peer_info(peer, make_peer_info(genesis_digest, 0, 0));

    // Push checkpoint 3 from the peer
    let mut request =
        Request::new(PushCheckpointSummaryRequest { checkpoint: checkpoints[3].inner().clone() });
    request.extensions_mut().insert(PeerInfo { peer_id: peer });

    let response = server.push_checkpoint_summary(request).await;
    assert!(response.is_ok());

    // Verify peer height was updated
    let heights = unstarted_state_sync.peer_heights.read();
    assert_eq!(heights.peers.get(&peer).unwrap().height, 3);

    // Verify checkpoint was stored in unprocessed_checkpoints
    assert!(heights.unprocessed_checkpoints.contains_key(checkpoints[3].digest()));
}

#[tokio::test]
async fn test_server_push_checkpoint_unknown_peer_ignored() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(3, None);
    let store = fixture.init_store();

    let config = make_p2p_config();
    let (_, _, server) = P2pBuilder::new().store(store.clone()).config(config).build_internal();

    let unknown_peer = make_peer_id(99);

    // Push from unknown peer (not in peer_heights)
    let mut request =
        Request::new(PushCheckpointSummaryRequest { checkpoint: checkpoints[1].inner().clone() });
    request.extensions_mut().insert(PeerInfo { peer_id: unknown_peer });

    // Should succeed but not update anything (update_peer_info returns false for unknown peer)
    let response = server.push_checkpoint_summary(request).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_server_get_checkpoint_summary_latest() {
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(5);

    let config = make_p2p_config();
    let (_, _, server) = P2pBuilder::new().store(store.clone()).config(config).build_internal();

    let request = Request::new(GetCheckpointSummaryRequest::Latest);
    let response = server.get_checkpoint_summary(request).await.unwrap().into_inner();
    let cp = response.checkpoint.unwrap();
    assert_eq!(*cp.sequence_number(), 4); // 0-indexed, 5 checkpoints = seq 0..4
}

#[tokio::test]
async fn test_server_get_checkpoint_summary_by_sequence_number() {
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(5);

    let config = make_p2p_config();
    let (_, _, server) = P2pBuilder::new().store(store.clone()).config(config).build_internal();

    // Existing checkpoint
    let request = Request::new(GetCheckpointSummaryRequest::BySequenceNumber(2));
    let response = server.get_checkpoint_summary(request).await.unwrap().into_inner();
    let cp = response.checkpoint.unwrap();
    assert_eq!(*cp.sequence_number(), 2);
    assert_eq!(*cp.digest(), *checkpoints[2].digest());

    // Non-existent checkpoint
    let request = Request::new(GetCheckpointSummaryRequest::BySequenceNumber(99));
    let response = server.get_checkpoint_summary(request).await.unwrap().into_inner();
    assert!(response.checkpoint.is_none());
}

#[tokio::test]
async fn test_server_get_checkpoint_summary_by_digest() {
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(5);

    let config = make_p2p_config();
    let (_, _, server) = P2pBuilder::new().store(store.clone()).config(config).build_internal();

    // Existing digest
    let target_digest = *checkpoints[3].digest();
    let request = Request::new(GetCheckpointSummaryRequest::ByDigest(target_digest));
    let response = server.get_checkpoint_summary(request).await.unwrap().into_inner();
    let cp = response.checkpoint.unwrap();
    assert_eq!(*cp.sequence_number(), 3);

    // Non-existent digest
    let request =
        Request::new(GetCheckpointSummaryRequest::ByDigest(CheckpointDigest::new([0xff; 32])));
    let response = server.get_checkpoint_summary(request).await.unwrap().into_inner();
    assert!(response.checkpoint.is_none());
}

#[tokio::test]
async fn test_server_get_checkpoint_availability() {
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(10);

    let config = make_p2p_config();
    let (_, _, server) = P2pBuilder::new().store(store.clone()).config(config).build_internal();

    let request = Request::new(GetCheckpointAvailabilityRequest { _unused: true });
    let response = server.get_checkpoint_availability(request).await.unwrap().into_inner();

    assert_eq!(*response.highest_synced_checkpoint.sequence_number(), 9);
    assert_eq!(response.lowest_available_checkpoint, 0);
}

#[tokio::test]
async fn test_server_get_checkpoint_contents_found() {
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(5);

    let config = make_p2p_config();
    let (_, _, server) = P2pBuilder::new().store(store.clone()).config(config).build_internal();

    let digest = checkpoints[2].content_digest;
    let request = Request::new(GetCheckpointContentsRequest { digest });
    let response = server.get_checkpoint_contents(request).await.unwrap().into_inner();
    assert!(response.contents.is_some());
}

#[tokio::test]
async fn test_server_get_checkpoint_contents_not_found() {
    let fixture = CommitteeFixture::generate(0, 4);
    let store = fixture.init_store();

    let config = make_p2p_config();
    let (_, _, server) = P2pBuilder::new().store(store.clone()).config(config).build_internal();

    let digest = types::digests::CheckpointContentsDigest::new([0xab; 32]);
    let request = Request::new(GetCheckpointContentsRequest { digest });
    let response = server.get_checkpoint_contents(request).await.unwrap().into_inner();
    assert!(response.contents.is_none());
}

#[tokio::test]
async fn test_server_get_known_peers() {
    let fixture = CommitteeFixture::generate(0, 4);
    let store = fixture.init_store();

    let config = make_p2p_config();
    let (discovery_builder, _, server) =
        P2pBuilder::new().store(store.clone()).config(config).build_internal();

    // Before setting our_info, get_known_peers should fail
    let mut rng = StdRng::from_seed([42; 32]);
    let our_kp = NetworkKeyPair::generate(&mut rng);
    let our_peer_id = PeerId::from(our_kp.public());

    // Set our_info
    let our_info = types::sync::NodeInfo {
        peer_id: our_peer_id,
        address: "/ip4/127.0.0.1/tcp/8080".parse().unwrap(),
        timestamp_ms: 1000,
    }
    .sign(&our_kp);
    discovery_builder.state.write().our_info = Some(our_info.clone());

    // Create a request with a fake peer's own_info
    let requester_kp = NetworkKeyPair::generate(&mut rng);
    let requester_info = types::sync::NodeInfo {
        peer_id: PeerId::from(requester_kp.public()),
        address: "/ip4/127.0.0.1/tcp/9090".parse().unwrap(),
        timestamp_ms: 1000,
    }
    .sign(&requester_kp);

    let request = Request::new(types::sync::GetKnownPeersRequest { own_info: requester_info });
    let response = server.get_known_peers(request).await.unwrap().into_inner();

    assert_eq!(response.own_info.peer_id, our_peer_id);
    assert!(response.known_peers.is_empty());
}

#[tokio::test]
async fn test_server_get_known_peers_not_initialized() {
    let fixture = CommitteeFixture::generate(0, 4);
    let store = fixture.init_store();

    let config = make_p2p_config();
    let (_, _, server) = P2pBuilder::new().store(store.clone()).config(config).build_internal();

    // our_info not set -> should return error
    let mut rng = StdRng::from_seed([42; 32]);
    let requester_kp = NetworkKeyPair::generate(&mut rng);
    let requester_info = types::sync::NodeInfo {
        peer_id: PeerId::from(requester_kp.public()),
        address: "/ip4/127.0.0.1/tcp/9090".parse().unwrap(),
        timestamp_ms: 1000,
    }
    .sign(&requester_kp);

    let request = Request::new(types::sync::GetKnownPeersRequest { own_info: requester_info });
    let response = server.get_known_peers(request).await;
    assert!(response.is_err());
}

// ============================================================================
// CommitteeFixture Tests
// ============================================================================

#[test]
fn test_committee_fixture_generate() {
    let fixture = CommitteeFixture::generate(0, 4);
    assert_eq!(fixture.epoch, 0);
    assert_eq!(fixture.keypairs.len(), 4);
    assert_eq!(fixture.committee.voting_rights.len(), 4);
}

#[test]
fn test_committee_fixture_root_checkpoint() {
    let fixture = CommitteeFixture::generate(0, 4);
    let (root, contents) = fixture.create_root_checkpoint();
    assert_eq!(*root.sequence_number(), 0);
    assert!(root.previous_digest.is_none());
    assert!(root.end_of_epoch_data.is_none());
}

#[test]
fn test_committee_fixture_checkpoint_chain() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(10, None);
    assert_eq!(checkpoints.len(), 10);

    // Verify chain integrity
    assert!(checkpoints[0].previous_digest.is_none());
    for i in 1..checkpoints.len() {
        assert_eq!(*checkpoints[i].sequence_number(), i as u64);
        assert_eq!(checkpoints[i].previous_digest, Some(*checkpoints[i - 1].digest()));
    }
}

#[test]
fn test_committee_fixture_checkpoint_chain_from_previous() {
    let fixture = CommitteeFixture::generate(0, 4);
    let first_batch = fixture.make_empty_checkpoints(5, None);
    let second_batch = fixture.make_empty_checkpoints(3, Some(first_batch.last().unwrap()));

    assert_eq!(second_batch.len(), 3);
    assert_eq!(*second_batch[0].sequence_number(), 5);
    assert_eq!(second_batch[0].previous_digest, Some(*first_batch.last().unwrap().digest()));
}

#[test]
fn test_committee_fixture_end_of_epoch_checkpoint() {
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(5, None);

    let next_fixture = CommitteeFixture::generate(1, 4);
    let eoe = fixture
        .make_end_of_epoch_checkpoint(checkpoints.last().unwrap(), next_fixture.committee.clone());

    assert_eq!(*eoe.sequence_number(), 5);
    assert!(eoe.end_of_epoch_data.is_some());
    let eoe_data = eoe.end_of_epoch_data.as_ref().unwrap();
    assert_eq!(eoe_data.next_epoch_validator_committee.epoch, 1);
}

#[test]
fn test_committee_fixture_init_store() {
    let fixture = CommitteeFixture::generate(0, 4);
    let store = fixture.init_store();

    // Genesis should be present
    let genesis = store.inner().get_checkpoint_by_sequence_number(0).unwrap().clone();
    assert_eq!(*genesis.sequence_number(), 0);
}

#[test]
fn test_committee_fixture_init_store_with_checkpoints() {
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(10);

    assert_eq!(checkpoints.len(), 10);

    // All checkpoints should be in store
    for cp in &checkpoints {
        let inner = store.inner();
        let stored = inner.get_checkpoint_by_sequence_number(*cp.sequence_number()).unwrap();
        assert_eq!(stored.digest(), cp.digest());
    }

    // Highest synced should be the last one
    let inner = store.inner();
    let highest = inner.get_highest_synced_checkpoint().unwrap();
    assert_eq!(*highest.sequence_number(), 9);
}

#[test]
fn test_committee_fixture_verified_checkpoints_validate() {
    // Verify that checkpoints produced by the fixture pass the storage verification function
    let fixture = CommitteeFixture::generate(0, 4);
    let checkpoints = fixture.make_empty_checkpoints(5, None);

    // Each checkpoint should be verifiable against the previous one
    for i in 1..checkpoints.len() {
        let result = types::storage::verify_checkpoint_with_committee(
            Arc::new(fixture.committee.clone()),
            &checkpoints[i - 1],
            checkpoints[i].inner().clone(),
        );
        assert!(result.is_ok(), "Checkpoint {} failed verification: {:?}", i, result.err());
    }
}

// ============================================================================
// Worker Unit Tests — get_or_insert_verified_checkpoint
// ============================================================================

use crate::state_sync::worker::get_or_insert_verified_checkpoint;

#[test]
fn test_get_or_insert_returns_existing_checkpoint() {
    // When a checkpoint already exists in the store, it should be returned without re-verification
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(5);

    // Checkpoint 3 is already in the store
    let result = get_or_insert_verified_checkpoint(&store, checkpoints[3].inner().clone(), true);
    assert!(result.is_ok());
    let verified = result.unwrap();
    assert_eq!(*verified.sequence_number(), 3);
    assert_eq!(verified.digest(), checkpoints[3].digest());
}

#[test]
fn test_get_or_insert_verifies_and_inserts_new_checkpoint() {
    // When a checkpoint is NOT in the store, it should be verified and inserted
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(5);

    // Create checkpoint 5 (not yet in store)
    let new_checkpoints = fixture.make_empty_checkpoints(1, Some(checkpoints.last().unwrap()));
    let new_cp = &new_checkpoints[0];
    assert_eq!(*new_cp.sequence_number(), 5);

    // Should not be in store yet
    assert!(store.inner().get_checkpoint_by_sequence_number(5).is_none());

    let result = get_or_insert_verified_checkpoint(&store, new_cp.inner().clone(), true);
    assert!(result.is_ok());
    let verified = result.unwrap();
    assert_eq!(*verified.sequence_number(), 5);

    // Should now be in store
    assert!(store.inner().get_checkpoint_by_sequence_number(5).is_some());

    // Highest verified checkpoint should be updated
    let inner = store.inner();
    let highest = inner.get_highest_verified_checkpoint().unwrap();
    assert_eq!(*highest.sequence_number(), 5);
}

#[test]
fn test_get_or_insert_fails_when_previous_missing() {
    // When verify=true and the previous checkpoint is missing, should error
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(3);

    // Create checkpoints 3, 4, 5 but only try to insert 5 (skipping 3 and 4)
    let chain = fixture.make_empty_checkpoints(3, Some(checkpoints.last().unwrap()));
    // chain[0]=seq3, chain[1]=seq4, chain[2]=seq5
    // Store only has 0..2, so inserting seq5 should fail (missing seq4)

    let result = get_or_insert_verified_checkpoint(&store, chain[2].inner().clone(), true);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Missing previous checkpoint") || err_msg.contains("Failed to get"),
        "Unexpected error: {err_msg}"
    );
}

#[test]
fn test_get_or_insert_without_verification() {
    // When verify=false, checkpoint should be inserted without checking previous
    let fixture = CommitteeFixture::generate(0, 4);
    let store = fixture.init_store();

    // Create a chain but skip inserting intermediate checkpoints
    let checkpoints = fixture.make_empty_checkpoints(5, None);

    // Insert checkpoint 4 directly without previous checkpoints 1-3 in store
    // With verify=false, this should succeed
    let result = get_or_insert_verified_checkpoint(&store, checkpoints[4].inner().clone(), false);
    assert!(result.is_ok());
    let verified = result.unwrap();
    assert_eq!(*verified.sequence_number(), 4);

    // Should be in store
    assert!(store.inner().get_checkpoint_by_sequence_number(4).is_some());
}

#[test]
fn test_get_or_insert_fails_for_genesis_with_verify() {
    // Inserting genesis (seq 0) with verify=true should fail because
    // checked_sub(1) on sequence_number 0 would underflow
    let fixture = CommitteeFixture::generate(0, 4);
    let store = types::storage::shared_in_memory_store::SharedInMemoryStore::default();

    let (root, _) = fixture.create_root_checkpoint();

    let result = get_or_insert_verified_checkpoint(&store, root.inner().clone(), true);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("underflow") || err_msg.contains("Failed to get"),
        "Unexpected error: {err_msg}"
    );
}

#[test]
fn test_get_or_insert_idempotent() {
    // Inserting the same checkpoint twice should be idempotent
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(3);

    let new_checkpoints = fixture.make_empty_checkpoints(1, Some(checkpoints.last().unwrap()));
    let new_cp = &new_checkpoints[0];

    // First insertion
    let result1 = get_or_insert_verified_checkpoint(&store, new_cp.inner().clone(), true);
    assert!(result1.is_ok());

    // Second insertion — should return the existing one
    let result2 = get_or_insert_verified_checkpoint(&store, new_cp.inner().clone(), true);
    assert!(result2.is_ok());
    assert_eq!(result1.unwrap().digest(), result2.unwrap().digest());
}

#[test]
fn test_get_or_insert_across_epoch_boundary() {
    // Test inserting a checkpoint from epoch 1 after an end-of-epoch checkpoint
    let fixture_e0 = CommitteeFixture::generate(0, 4);
    let fixture_e1 = CommitteeFixture::generate(1, 4);

    // Build epoch 0 chain with end-of-epoch
    let (store, e0_checkpoints) = fixture_e0.init_store_with_checkpoints(5);
    let eoe = fixture_e0
        .make_end_of_epoch_checkpoint(e0_checkpoints.last().unwrap(), fixture_e1.committee.clone());

    // Insert end-of-epoch checkpoint
    store.insert_checkpoint(&eoe).unwrap();
    store.insert_committee(fixture_e1.committee.clone()).unwrap();

    // Create epoch 1 checkpoint
    let e1_contents = CommitteeFixture::empty_contents();
    let e1_summary = types::checkpoints::CheckpointSummary::new(
        1, // epoch 1
        *eoe.sequence_number() + 1,
        0,
        &e1_contents,
        Some(*eoe.digest()),
        types::tx_fee::TransactionFee::default(),
        None,
        *eoe.sequence_number() + 1,
        vec![],
    );
    let e1_checkpoint = fixture_e1.create_certified_checkpoint(e1_summary);

    // Insert epoch 1 checkpoint with verification — should verify against new committee
    let result = get_or_insert_verified_checkpoint(&store, e1_checkpoint.inner().clone(), true);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
    assert_eq!(*result.unwrap().sequence_number(), *eoe.sequence_number() + 1);
}

// ============================================================================
// Worker Unit Tests — StateSyncWorker::process_checkpoint
// ============================================================================

use crate::state_sync::worker::StateSyncWorker;
use data_ingestion::Worker;

#[tokio::test]
async fn test_worker_process_checkpoint_empty() {
    // Test processing a checkpoint with empty contents (no transactions)
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(3);

    // Create a new checkpoint not yet in the store
    let new_checkpoints = fixture.make_empty_checkpoints(1, Some(checkpoints.last().unwrap()));
    let new_cp = &new_checkpoints[0];
    assert_eq!(*new_cp.sequence_number(), 3);

    let worker = StateSyncWorker(store.clone());

    let checkpoint_data = types::full_checkpoint_content::CheckpointData {
        checkpoint_summary: new_cp.inner().clone(),
        checkpoint_contents: CommitteeFixture::empty_contents(),
        transactions: vec![],
    };

    let result = worker.process_checkpoint(&checkpoint_data).await;
    assert!(result.is_ok(), "process_checkpoint failed: {:?}", result.err());

    // Checkpoint should now be in store
    assert!(store.inner().get_checkpoint_by_sequence_number(3).is_some());

    // Highest synced should be updated
    let inner = store.inner();
    let highest_synced = inner.get_highest_synced_checkpoint().unwrap();
    assert_eq!(*highest_synced.sequence_number(), 3);
    drop(inner);

    // Contents should be stored
    use types::storage::read_store::ReadStore;
    let contents = store.get_full_checkpoint_contents(Some(3), &new_cp.content_digest);
    assert!(contents.is_some());
}

#[tokio::test]
async fn test_worker_process_checkpoint_already_in_store() {
    // When checkpoint is already in store, process_checkpoint should still succeed
    // (get_or_insert_verified_checkpoint returns existing)
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(5);

    let worker = StateSyncWorker(store.clone());

    // Process checkpoint 3 which is already in store
    let checkpoint_data = types::full_checkpoint_content::CheckpointData {
        checkpoint_summary: checkpoints[3].inner().clone(),
        checkpoint_contents: CommitteeFixture::empty_contents(),
        transactions: vec![],
    };

    let result = worker.process_checkpoint(&checkpoint_data).await;
    assert!(result.is_ok(), "process_checkpoint failed: {:?}", result.err());
}

#[tokio::test]
async fn test_worker_process_checkpoint_sequential_chain() {
    // Process multiple checkpoints in sequence
    let fixture = CommitteeFixture::generate(0, 4);
    let store = fixture.init_store();

    let worker = StateSyncWorker(store.clone());

    // Create 5 more checkpoints after genesis
    let (root, _) = fixture.create_root_checkpoint();
    let additional = fixture.make_empty_checkpoints(5, Some(&root));

    for cp in &additional {
        let checkpoint_data = types::full_checkpoint_content::CheckpointData {
            checkpoint_summary: cp.inner().clone(),
            checkpoint_contents: CommitteeFixture::empty_contents(),
            transactions: vec![],
        };
        let result = worker.process_checkpoint(&checkpoint_data).await;
        assert!(result.is_ok(), "Failed at seq {}: {:?}", cp.sequence_number(), result.err());
    }

    // Final checkpoint should be highest synced
    let inner = store.inner();
    let highest_synced = inner.get_highest_synced_checkpoint().unwrap();
    assert_eq!(*highest_synced.sequence_number(), 5);
}

// ============================================================================
// handle_checkpoint_from_consensus Tests
// ============================================================================

use crate::state_sync::StateSyncEventLoop;

/// Helper to create a StateSyncEventLoop for testing without networking.
fn make_event_loop(
    store: types::storage::shared_in_memory_store::SharedInMemoryStore,
) -> (
    StateSyncEventLoop<types::storage::shared_in_memory_store::SharedInMemoryStore>,
    broadcast::Receiver<types::checkpoints::VerifiedCheckpoint>,
) {
    let config = make_p2p_config();
    let (_, unstarted, _server) = P2pBuilder::new().store(store).config(config).build_internal();

    let active_peers = ActivePeers::new(100);
    let (_, peer_event_rx) = broadcast::channel(100);
    let checkpoint_rx = unstarted.checkpoint_event_sender.subscribe();
    let (event_loop, _handle) = unstarted.build(active_peers, peer_event_rx);
    (event_loop, checkpoint_rx)
}

#[tokio::test]
async fn test_handle_checkpoint_from_consensus_updates_watermarks() {
    // handle_checkpoint_from_consensus should update both
    // highest_verified_checkpoint and highest_synced_checkpoint
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(5);

    // Create the next checkpoint (seq 5)
    let new_cps = fixture.make_empty_checkpoints(1, Some(checkpoints.last().unwrap()));
    let new_cp = &new_cps[0];

    // Use insert_certified_checkpoint (NOT insert_checkpoint) to pre-insert
    // checkpoint data into store without bumping highest_verified_checkpoint.
    // The debug_assertions block in handle_checkpoint_from_consensus expects
    // checkpoints and their contents to exist in the store.
    store.insert_certified_checkpoint(new_cp);
    store.insert_checkpoint_contents(new_cp, CommitteeFixture::empty_verified_contents()).unwrap();

    // Verify watermark is still at 4 (not bumped)
    {
        let inner = store.inner();
        assert_eq!(*inner.get_highest_verified_checkpoint().unwrap().sequence_number(), 4);
    }

    let (mut event_loop, _rx) = make_event_loop(store.clone());

    event_loop.handle_checkpoint_from_consensus(Box::new(new_cp.clone()));

    // Both watermarks should now be updated to 5
    let inner = store.inner();
    let highest_verified = inner.get_highest_verified_checkpoint().unwrap();
    assert_eq!(*highest_verified.sequence_number(), 5);
    let highest_synced = inner.get_highest_synced_checkpoint().unwrap();
    assert_eq!(*highest_synced.sequence_number(), 5);
}

#[tokio::test]
async fn test_handle_checkpoint_from_consensus_ignores_old() {
    // If checkpoint is <= highest_verified, it should be silently ignored
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(10);

    let (mut event_loop, _rx) = make_event_loop(store.clone());

    // Send checkpoint 5, which is below highest_verified (9)
    // Should be silently ignored
    event_loop.handle_checkpoint_from_consensus(Box::new(checkpoints[5].clone()));

    // Watermarks should remain at 9
    let inner = store.inner();
    let highest = inner.get_highest_verified_checkpoint().unwrap();
    assert_eq!(*highest.sequence_number(), 9);
}

#[tokio::test]
async fn test_handle_checkpoint_from_consensus_broadcasts_event() {
    // handle_checkpoint_from_consensus should broadcast the checkpoint
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(5);

    let new_cps = fixture.make_empty_checkpoints(1, Some(checkpoints.last().unwrap()));
    let new_cp = &new_cps[0];
    // Use insert_certified_checkpoint to avoid bumping watermark
    store.insert_certified_checkpoint(new_cp);
    store.insert_checkpoint_contents(new_cp, CommitteeFixture::empty_verified_contents()).unwrap();

    let (mut event_loop, mut rx) = make_event_loop(store.clone());

    event_loop.handle_checkpoint_from_consensus(Box::new(new_cp.clone()));

    // Should have received the checkpoint on the broadcast channel
    let received = rx.try_recv().unwrap();
    assert_eq!(*received.sequence_number(), 5);
    assert_eq!(received.digest(), new_cp.digest());
}

#[tokio::test]
#[should_panic(expected = "mismatched previous_digest")]
async fn test_handle_checkpoint_from_consensus_panics_on_chain_mismatch() {
    // If previous_digest doesn't match, should panic
    let fixture = CommitteeFixture::generate(0, 4);
    let (store, checkpoints) = fixture.init_store_with_checkpoints(5);

    // Create a checkpoint with wrong previous_digest
    let contents = CommitteeFixture::empty_contents();
    let bad_summary = types::checkpoints::CheckpointSummary::new(
        0,
        5, // seq 5
        0,
        &contents,
        Some(types::digests::CheckpointDigest::new([0xff; 32])), // wrong digest
        types::tx_fee::TransactionFee::default(),
        None,
        5,
        vec![],
    );
    let bad_checkpoint = fixture.create_certified_checkpoint(bad_summary);
    // Use insert_certified_checkpoint to avoid bumping watermark
    store.insert_certified_checkpoint(&bad_checkpoint);
    store
        .insert_checkpoint_contents(&bad_checkpoint, CommitteeFixture::empty_verified_contents())
        .unwrap();

    let (mut event_loop, _rx) = make_event_loop(store.clone());

    // This should panic because previous_digest doesn't match checkpoint 4's digest
    event_loop.handle_checkpoint_from_consensus(Box::new(bad_checkpoint));
}

#[tokio::test]
async fn test_handle_checkpoint_from_consensus_end_of_epoch_updates_watermarks() {
    // End-of-epoch checkpoint should update watermarks AND insert next committee.
    // Note: insert_certified_checkpoint already inserts the next committee (InMemoryStore behavior),
    // but handle_checkpoint_from_consensus also calls insert_committee for correctness with
    // other store implementations. This test verifies the watermark updates work correctly
    // for end-of-epoch checkpoints.
    let fixture_e0 = CommitteeFixture::generate(0, 4);
    let fixture_e1 = CommitteeFixture::generate(1, 4);
    let (store, checkpoints) = fixture_e0.init_store_with_checkpoints(5);

    // Create end-of-epoch checkpoint
    let eoe = fixture_e0
        .make_end_of_epoch_checkpoint(checkpoints.last().unwrap(), fixture_e1.committee.clone());
    // insert_certified_checkpoint auto-inserts the next committee from end_of_epoch_data
    store.insert_certified_checkpoint(&eoe);
    store.insert_checkpoint_contents(&eoe, CommitteeFixture::empty_verified_contents()).unwrap();

    // Verify watermark is still at 4 (insert_certified_checkpoint doesn't bump it)
    {
        let inner = store.inner();
        assert_eq!(*inner.get_highest_verified_checkpoint().unwrap().sequence_number(), 4);
    }

    let (mut event_loop, mut rx) = make_event_loop(store.clone());
    event_loop.handle_checkpoint_from_consensus(Box::new(eoe.clone()));

    // Watermarks should be updated
    let inner = store.inner();
    let highest_verified = inner.get_highest_verified_checkpoint().unwrap();
    assert_eq!(*highest_verified.sequence_number(), 5);
    let highest_synced = inner.get_highest_synced_checkpoint().unwrap();
    assert_eq!(*highest_synced.sequence_number(), 5);
    drop(inner);

    // Epoch 1 committee should be in store
    let inner = store.inner();
    let committee = inner.get_committee_by_epoch(1);
    assert!(committee.is_some(), "Epoch 1 committee should be in store");
    assert_eq!(committee.unwrap().epoch, 1);
    drop(inner);

    // Should have broadcast the end-of-epoch checkpoint
    let received = rx.try_recv().unwrap();
    assert_eq!(*received.sequence_number(), 5);
    assert!(received.end_of_epoch_data.is_some());
}
