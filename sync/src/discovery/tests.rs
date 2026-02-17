// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-network/src/discovery/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for the Soma project.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use rand::{SeedableRng, rngs::StdRng};
use types::{
    crypto::NetworkKeyPair,
    peer_id::PeerId,
    sync::{NodeInfo, SignedNodeInfo, VerifiedSignedNodeInfo},
};

use super::{DiscoveryState, ONE_DAY_MILLISECONDS, now_unix, update_known_peers};

fn make_keypair(seed: u8) -> NetworkKeyPair {
    let mut rng = StdRng::from_seed([seed; 32]);
    NetworkKeyPair::generate(&mut rng)
}

fn make_state_with_our_info(our_kp: &NetworkKeyPair) -> Arc<RwLock<DiscoveryState>> {
    let our_peer_id = PeerId::from(our_kp.public());
    let our_info = NodeInfo {
        peer_id: our_peer_id,
        address: "/ip4/127.0.0.1/tcp/8080".parse().unwrap(),
        timestamp_ms: now_unix(),
    }
    .sign(our_kp);

    Arc::new(RwLock::new(DiscoveryState { our_info: Some(our_info), known_peers: HashMap::new() }))
}

fn make_signed_node_info(kp: &NetworkKeyPair, timestamp_ms: u64) -> SignedNodeInfo {
    let peer_id = PeerId::from(kp.public());
    NodeInfo { peer_id, address: "/ip4/10.0.0.1/tcp/9090".parse().unwrap(), timestamp_ms }.sign(kp)
}

#[test]
fn test_update_known_peers_valid_signature() {
    let our_kp = make_keypair(1);
    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    let peer_kp = make_keypair(2);
    let peer_id = PeerId::from(peer_kp.public());
    let peer_info = make_signed_node_info(&peer_kp, now_unix());

    update_known_peers(state.clone(), vec![peer_info], allowlisted);

    assert!(state.read().known_peers.contains_key(&peer_id));
}

#[test]
fn test_update_known_peers_invalid_signature() {
    let our_kp = make_keypair(1);
    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    let peer_kp = make_keypair(2);
    let wrong_kp = make_keypair(3);
    let peer_id = PeerId::from(peer_kp.public());

    // Create NodeInfo with peer_kp's peer_id but sign with wrong_kp
    let node_info =
        NodeInfo { peer_id, address: "/ip4/10.0.0.1/tcp/9090".parse().unwrap(), timestamp_ms: now_unix() };
    let signed = node_info.sign(&wrong_kp);

    // Override the peer_id to be peer_kp's (so sig won't match)
    // Actually, sign() uses the keypair to sign, but the peer_id in NodeInfo is from peer_kp.
    // The verification checks peer_id (as public key) against the signature.
    // So signing with wrong_kp but having peer_id from peer_kp should fail verification.
    update_known_peers(state.clone(), vec![signed], allowlisted);

    // Should NOT be added because the signature doesn't match the peer_id
    assert!(!state.read().known_peers.contains_key(&peer_id));
}

#[test]
fn test_update_known_peers_future_timestamp_rejected() {
    let our_kp = make_keypair(1);
    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    let peer_kp = make_keypair(2);
    let peer_id = PeerId::from(peer_kp.public());

    // Timestamp 60 seconds in the future (limit is 30 seconds)
    let future_ts = now_unix() + 60_000;
    let peer_info = make_signed_node_info(&peer_kp, future_ts);

    update_known_peers(state.clone(), vec![peer_info], allowlisted);

    assert!(!state.read().known_peers.contains_key(&peer_id));
}

#[test]
fn test_update_known_peers_old_timestamp_rejected() {
    let our_kp = make_keypair(1);
    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    let peer_kp = make_keypair(2);
    let peer_id = PeerId::from(peer_kp.public());

    // Timestamp more than 1 day old
    let old_ts = now_unix().saturating_sub(ONE_DAY_MILLISECONDS + 1000);
    let peer_info = make_signed_node_info(&peer_kp, old_ts);

    update_known_peers(state.clone(), vec![peer_info], allowlisted);

    assert!(!state.read().known_peers.contains_key(&peer_id));
}

#[test]
fn test_update_known_peers_own_peer_id_skipped() {
    let our_kp = make_keypair(1);
    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    let our_peer_id = PeerId::from(our_kp.public());

    // Create a signed NodeInfo using our own keypair (same peer_id as our_info)
    let our_node_info = make_signed_node_info(&our_kp, now_unix());

    update_known_peers(state.clone(), vec![our_node_info], allowlisted);

    // Our own peer_id should NOT be in known_peers
    assert!(!state.read().known_peers.contains_key(&our_peer_id));
}

#[test]
fn test_update_known_peers_newer_replaces_older() {
    let our_kp = make_keypair(1);
    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    let peer_kp = make_keypair(2);
    let peer_id = PeerId::from(peer_kp.public());

    let ts1 = now_unix() - 5000;
    let ts2 = now_unix();

    // Insert older first
    let old_info = make_signed_node_info(&peer_kp, ts1);
    update_known_peers(state.clone(), vec![old_info], allowlisted.clone());
    assert_eq!(state.read().known_peers.get(&peer_id).unwrap().timestamp_ms, ts1);

    // Insert newer - should replace
    let new_info = make_signed_node_info(&peer_kp, ts2);
    update_known_peers(state.clone(), vec![new_info], allowlisted);
    assert_eq!(state.read().known_peers.get(&peer_id).unwrap().timestamp_ms, ts2);
}

#[test]
fn test_update_known_peers_older_does_not_replace_newer() {
    let our_kp = make_keypair(1);
    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    let peer_kp = make_keypair(2);
    let peer_id = PeerId::from(peer_kp.public());

    let ts_old = now_unix() - 5000;
    let ts_new = now_unix();

    // Insert newer first
    let new_info = make_signed_node_info(&peer_kp, ts_new);
    update_known_peers(state.clone(), vec![new_info], allowlisted.clone());
    assert_eq!(state.read().known_peers.get(&peer_id).unwrap().timestamp_ms, ts_new);

    // Insert older - should NOT replace
    let old_info = make_signed_node_info(&peer_kp, ts_old);
    update_known_peers(state.clone(), vec![old_info], allowlisted);
    assert_eq!(state.read().known_peers.get(&peer_id).unwrap().timestamp_ms, ts_new);
}

#[test]
fn test_update_known_peers_multiple_peers() {
    let our_kp = make_keypair(1);
    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    let ts = now_unix();
    let mut infos = Vec::new();
    let mut expected_ids = Vec::new();

    for i in 2..=6 {
        let kp = make_keypair(i);
        expected_ids.push(PeerId::from(kp.public()));
        infos.push(make_signed_node_info(&kp, ts));
    }

    update_known_peers(state.clone(), infos, allowlisted);

    let known = &state.read().known_peers;
    assert_eq!(known.len(), 5);
    for id in &expected_ids {
        assert!(known.contains_key(id));
    }
}

#[test]
fn test_update_known_peers_at_boundary_timestamps() {
    let our_kp = make_keypair(1);
    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    // Exactly at 30-second future boundary (should be accepted: > check, not >=)
    let peer_kp_ok = make_keypair(2);
    let peer_id_ok = PeerId::from(peer_kp_ok.public());
    let ts_boundary = now_unix() + 29_000; // 29 seconds in future, should pass
    let info_ok = make_signed_node_info(&peer_kp_ok, ts_boundary);

    // Exactly at 1-day-old boundary
    let peer_kp_old = make_keypair(3);
    let peer_id_old = PeerId::from(peer_kp_old.public());
    let ts_day = now_unix().saturating_sub(ONE_DAY_MILLISECONDS - 1000); // Just under 1 day, should pass
    let info_old = make_signed_node_info(&peer_kp_old, ts_day);

    update_known_peers(state.clone(), vec![info_ok, info_old], allowlisted);

    assert!(state.read().known_peers.contains_key(&peer_id_ok));
    assert!(state.read().known_peers.contains_key(&peer_id_old));
}

#[test]
fn test_discovery_state_peer_lost_removes_known_peer() {
    let our_kp = make_keypair(1);
    let peer_kp = make_keypair(2);
    let peer_id = PeerId::from(peer_kp.public());

    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    // Add peer
    let info = make_signed_node_info(&peer_kp, now_unix());
    update_known_peers(state.clone(), vec![info], allowlisted);
    assert!(state.read().known_peers.contains_key(&peer_id));

    // Simulate LostPeer by removing from known_peers (this is what handle_peer_event does)
    state.write().known_peers.remove(&peer_id);
    assert!(!state.read().known_peers.contains_key(&peer_id));
}

#[test]
fn test_peer_culling_removes_old_peers() {
    let our_kp = make_keypair(1);
    let state = make_state_with_our_info(&our_kp);
    let allowlisted = Arc::new(HashMap::new());

    let now = now_unix();

    // Add a fresh peer
    let fresh_kp = make_keypair(2);
    let fresh_id = PeerId::from(fresh_kp.public());
    let fresh_info = make_signed_node_info(&fresh_kp, now);
    update_known_peers(state.clone(), vec![fresh_info], allowlisted.clone());

    // Add a stale peer by manually inserting with old timestamp
    let stale_kp = make_keypair(3);
    let stale_id = PeerId::from(stale_kp.public());
    let stale_info = NodeInfo {
        peer_id: stale_id,
        address: "/ip4/10.0.0.2/tcp/9090".parse().unwrap(),
        timestamp_ms: now.saturating_sub(ONE_DAY_MILLISECONDS + 1000),
    }
    .sign(&stale_kp);
    let verified = VerifiedSignedNodeInfo::new_from_verified(stale_info);
    state.write().known_peers.insert(stale_id, verified);

    assert_eq!(state.read().known_peers.len(), 2);

    // Simulate the culling that happens in handle_tick
    state
        .write()
        .known_peers
        .retain(|_k, v| now.saturating_sub(v.timestamp_ms) < ONE_DAY_MILLISECONDS);

    // Only fresh peer should remain
    assert_eq!(state.read().known_peers.len(), 1);
    assert!(state.read().known_peers.contains_key(&fresh_id));
    assert!(!state.read().known_peers.contains_key(&stale_id));
}
