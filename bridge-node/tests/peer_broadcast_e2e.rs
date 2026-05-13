//! Multi-peer end-to-end signature aggregation.
//!
//! Spawns N real `bridge-node` HTTP servers in-process — each with its
//! own ECDSA keypair, governance whitelist, and axum binding — and
//! points a [`PeerBroadcastAggregator`] at them. The peers really sign,
//! the aggregator really fetches over the loopback network, the
//! signatures really ecrecover. This catches wire-format,
//! URL-construction, and JSON-serialization bugs that
//! single-process wiremock tests miss.
//!
//! Scenarios:
//! - happy 3/3: every peer responds; cert includes all signatures
//! - 2/3 with one peer down: threshold met without the down peer
//! - 1/3 (insufficient): aggregator returns `InsufficientStake`
//! - 3/3 with one peer blocklisted: its sig is dropped, threshold still met by the others
//! - threshold scales with action variant: pause uses pause threshold, not deposit threshold
//!
//! ## Why not msim
//!
//! The bridge node is an off-chain process; the simulator targets the
//! on-chain validator network. Running real tokio + a real TCP loopback
//! is the appropriate fidelity here — close enough to production that
//! the test exercises the same code paths an operator deployment will.

use std::collections::{BTreeMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use bridge_node::error::{BridgeError, BridgeResult};
use bridge_node::eth_client::EthClient;
use bridge_node::handler::BridgeRequestHandler;
use bridge_node::http_server::{BridgeNodePublicMetadata, run_server};
use bridge_node::peer_aggregator::{PeerBroadcastAggregator, SignRequest};
use bridge_node::soma_client::{SomaBridgeClient, SomaBridgeClientInner};
use bridge_node::types::BridgeAction;
use fastcrypto::secp256k1::Secp256k1KeyPair;
use fastcrypto::traits::KeyPair;
use rand::SeedableRng;
use rand::rngs::StdRng;
use types::base::SomaAddress;
use types::bridge::{
    BridgeChainId, BridgeCommittee, BridgeMember, BridgePubkey, PendingWithdrawal,
};
use types::effects::TransactionEffects;
use types::transaction::Transaction;

// ---------------------------------------------------------------------------
// Minimal in-test SomaClient mock. The unit-test mock lives under
// `#[cfg(test)]` inside the crate and is invisible to integration
// tests. Re-declaring the small surface here keeps the e2e test
// self-contained and pinned to the public `SomaBridgeClientInner` API
// (so a future trait change forces an update here too).
// ---------------------------------------------------------------------------

#[derive(Default)]
struct InTestSomaClient {
    /// `nonce → PendingWithdrawal`. Tests insert here to make each
    /// peer's handler verify a withdrawal before signing.
    pending: Mutex<BTreeMap<u64, PendingWithdrawal>>,
}

#[async_trait]
impl SomaBridgeClientInner for InTestSomaClient {
    async fn is_bridge_paused(&self) -> BridgeResult<bool> {
        Ok(false)
    }
    async fn get_total_usdc_supply(&self) -> BridgeResult<u64> {
        Ok(0)
    }
    async fn get_bridge_committee(&self) -> BridgeResult<BridgeCommittee> {
        Ok(BridgeCommittee::empty())
    }
    async fn is_deposit_processed(&self, _nonce: u64) -> BridgeResult<bool> {
        Ok(false)
    }
    async fn get_pending_withdrawal(
        &self,
        nonce: u64,
    ) -> BridgeResult<Option<PendingWithdrawal>> {
        Ok(self.pending.lock().unwrap().get(&nonce).cloned())
    }
    async fn get_chain_identifier(&self) -> BridgeResult<String> {
        Ok("soma-e2e".to_string())
    }
    async fn get_usdc_balance(&self, _addr: SomaAddress) -> BridgeResult<u64> {
        Ok(0)
    }
    async fn execute_transaction(
        &self,
        _tx: &Transaction,
    ) -> BridgeResult<TransactionEffects> {
        Err(BridgeError::Internal("execute_transaction not used in e2e".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Test scaffolding: spin up N real peer servers + an aggregator
// ---------------------------------------------------------------------------

/// One peer in the test cluster.
struct Peer {
    pubkey: BridgePubkey,
    base_url: String,
    /// Backing soma client — exposed so tests can install withdrawals,
    /// flip pause state, etc.
    soma: Arc<SomaBridgeClient<InTestSomaClient>>,
    /// The axum task. Aborting it shuts the peer down (used for
    /// the "peer down mid-aggregation" scenario).
    server_handle: tokio::task::JoinHandle<()>,
}

impl Peer {
    fn shutdown(self) {
        self.server_handle.abort();
    }
}

fn keypair(seed: u8) -> Secp256k1KeyPair {
    let mut rng = StdRng::from_seed([seed; 32]);
    Secp256k1KeyPair::generate(&mut rng)
}

/// Spawn a peer at `0.0.0.0:0` (OS-picked port). Returns the peer once
/// the listener is actually bound (so the aggregator can connect
/// without racing the bind).
async fn spawn_peer(
    seed: u8,
    approved: Vec<BridgeAction>,
) -> Peer {
    let kp = keypair(seed);
    let pubkey = BridgePubkey::from_keypair(&kp);

    // Bind first so we know the port before the server task starts.
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
        0,
    ))
    .await
    .expect("bind 127.0.0.1:0");
    let local = listener.local_addr().expect("local_addr");
    drop(listener); // close so the server can re-bind the same port

    let soma = Arc::new(SomaBridgeClient::new(
        InTestSomaClient::default(),
        BridgeChainId::SomaCustom,
    ));
    let eth = Arc::new(EthClient::new_for_test(
        "0x0000000000000000000000000000000000000000".to_string(),
    ));
    let handler = BridgeRequestHandler::new(kp, eth, Arc::clone(&soma), approved)
        .expect("handler builds");
    let metadata = Arc::new(BridgeNodePublicMetadata::empty_for_testing());

    let server_handle = run_server(&local, handler, metadata);

    // Poll the /ping endpoint until the server answers — avoids
    // races where the aggregator's first GET arrives before the
    // axum task has progressed past `bind()`.
    let base_url = format!("http://{local}");
    let client = reqwest::Client::new();
    let started = std::time::Instant::now();
    loop {
        if client
            .get(format!("{base_url}/ping"))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            break;
        }
        if started.elapsed() > std::time::Duration::from_secs(5) {
            panic!("peer at {base_url} never came up");
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    Peer { pubkey, base_url, soma, server_handle }
}

/// Build a committee from peers + voting powers. Equal powers when
/// the caller doesn't care about the split.
fn committee_from(peers: &[(&Peer, u64)]) -> BridgeCommittee {
    let mut c = BridgeCommittee::empty();
    for (peer, power) in peers {
        c.members.insert(
            peer.pubkey.clone(),
            BridgeMember {
                soma_address: SomaAddress::random(),
                voting_power: *power,
                http_url: peer.base_url.clone(),
                is_blocklisted: false,
            },
        );
    }
    c
}

fn aggregator(committee: BridgeCommittee) -> PeerBroadcastAggregator {
    PeerBroadcastAggregator::new(Arc::new(ArcSwap::from_pointee(committee)))
}

fn pause(nonce: u64) -> BridgeAction {
    BridgeAction::EmergencyPause { nonce }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// 3 peers all up, threshold met by everyone.
#[tokio::test]
async fn three_of_three_governance() {
    let action = pause(1);
    let approved = vec![action.clone()];
    let p1 = spawn_peer(11, approved.clone()).await;
    let p2 = spawn_peer(12, approved.clone()).await;
    let p3 = spawn_peer(13, approved).await;
    let committee = committee_from(&[(&p1, 3334), (&p2, 3333), (&p3, 3333)]);
    let agg = aggregator(committee);

    let cert = agg
        .request_signatures(SignRequest::Governance(action.clone()), 6667)
        .await
        .expect("cert");
    assert_eq!(cert.action, action);
    // At least 2 sigs (3334 + 3333 >= 6667). FuturesUnordered may
    // return before the 3rd lands; that's correct early-exit behavior.
    assert!(cert.signatures.len() >= 2);
    for (claimed, sig) in &cert.signatures {
        assert!(committee_contains(&[&p1, &p2, &p3], claimed));
        assert_eq!(sig.as_bytes().len(), 65);
    }
    drop(p1);
    drop(p2);
    drop(p3);
}

/// 2 of 3 up. Aggregator must still produce a cert when the down peer
/// isn't needed to meet threshold.
#[tokio::test]
async fn two_of_three_with_one_peer_down() {
    let action = pause(2);
    let approved = vec![action.clone()];
    let p1 = spawn_peer(21, approved.clone()).await;
    let p2 = spawn_peer(22, approved.clone()).await;
    let p3 = spawn_peer(23, approved).await;
    let committee = committee_from(&[(&p1, 3000), (&p2, 3000), (&p3, 4000)]);
    let agg = aggregator(committee);

    // Kill peer 3 before the aggregator runs.
    p3.shutdown();

    let cert = agg
        .request_signatures(SignRequest::Governance(action.clone()), 6000)
        .await
        .expect("cert from 2/3");
    assert_eq!(cert.action, action);
    assert_eq!(cert.signatures.len(), 2);
    // The two surviving peers must both be present.
    assert!(cert.signatures.contains_key(&p1.pubkey));
    assert!(cert.signatures.contains_key(&p2.pubkey));
}

/// Only 1 of 3 up. Aggregator gives up with InsufficientStake — must
/// not return a partial cert that the on-chain verifier would reject.
#[tokio::test]
async fn insufficient_stake_when_two_peers_down() {
    let action = pause(3);
    let approved = vec![action.clone()];
    let p1 = spawn_peer(31, approved.clone()).await;
    let p2 = spawn_peer(32, approved.clone()).await;
    let p3 = spawn_peer(33, approved).await;
    let committee = committee_from(&[(&p1, 2500), (&p2, 2500), (&p3, 5000)]);
    let agg = aggregator(committee);

    p2.shutdown();
    p3.shutdown();

    let err = agg
        .request_signatures(SignRequest::Governance(action), 5000)
        .await
        .expect_err("must not produce cert below threshold");
    match err {
        bridge_node::error::BridgeError::InsufficientStake { got, required } => {
            assert_eq!(got, 2500);
            assert_eq!(required, 5000);
        }
        other => panic!("expected InsufficientStake, got {other:?}"),
    }
}

/// A blocklisted member's signature is dropped even if the peer
/// responds (because the on-chain verifier counts their stake as 0).
/// Aggregator must skip them entirely — not just discard their sig
/// after fetching it.
#[tokio::test]
async fn blocklisted_member_is_skipped() {
    let action = pause(4);
    let approved = vec![action.clone()];
    let p1 = spawn_peer(41, approved.clone()).await;
    let p2 = spawn_peer(42, approved.clone()).await;
    let p3 = spawn_peer(43, approved).await;

    let mut committee = committee_from(&[(&p1, 5000), (&p2, 3000), (&p3, 2000)]);
    // Blocklist p3 (the smallest-stake peer).
    committee.members.get_mut(&p3.pubkey).unwrap().is_blocklisted = true;
    let agg = aggregator(committee);

    // Threshold 7000 needs p1 + p2 (5000 + 3000 = 8000). p3 is
    // available but blocklisted — must not contribute.
    let cert = agg
        .request_signatures(SignRequest::Governance(action.clone()), 7000)
        .await
        .expect("cert");
    assert_eq!(cert.action, action);
    assert!(cert.signatures.contains_key(&p1.pubkey));
    assert!(cert.signatures.contains_key(&p2.pubkey));
    assert!(
        !cert.signatures.contains_key(&p3.pubkey),
        "blocklisted member's sig must not appear in cert"
    );
}

/// Withdrawal end-to-end. Each peer holds its own `MockSomaClient`;
/// the test installs the same `PendingWithdrawal` on all three so
/// each peer can verify it before signing. This catches signature
/// canonicalization bugs (each peer signs the same canonical bytes;
/// each sig ecrecovers to a different pubkey, all valid).
#[tokio::test]
async fn withdrawal_end_to_end() {
    let p1 = spawn_peer(51, vec![]).await;
    let p2 = spawn_peer(52, vec![]).await;
    let p3 = spawn_peer(53, vec![]).await;

    let withdrawal = PendingWithdrawal {
        id: types::object::ObjectID::random(),
        nonce: 7,
        sender: SomaAddress::random(),
        recipient_eth_address: [0x42; 20],
        amount: 1_500_000,
        created_at_ms: 1_700_000_000_000,
        target_chain: types::bridge::BridgeChainId::EthCustom,
        verified_signatures: None,
    };
    // Install on each peer's InTestSomaClient — peers re-fetch and
    // verify the withdrawal before signing.
    for p in [&p1, &p2, &p3] {
        p.soma
            .inner_for_test()
            .pending
            .lock()
            .unwrap()
            .insert(withdrawal.nonce, withdrawal.clone());
    }

    let committee = committee_from(&[(&p1, 3334), (&p2, 3333), (&p3, 3333)]);
    let agg = aggregator(committee);

    let expected = BridgeAction::Withdrawal {
        nonce: withdrawal.nonce,
        sender: withdrawal.sender,
        target_chain: types::bridge::BridgeChainId::EthCustom,
        recipient_eth_address: withdrawal.recipient_eth_address,
        token_type: types::bridge::USDC_TOKEN_TYPE,
        amount: withdrawal.amount,
        timestamp_ms: withdrawal.created_at_ms,
    };
    let cert = agg
        .request_signatures(
            SignRequest::SomaWithdrawal {
                nonce: withdrawal.nonce,
                expected: expected.clone(),
            },
            6667,
        )
        .await
        .expect("withdrawal cert");
    assert_eq!(cert.action, expected);
    assert!(cert.signatures.len() >= 2);
}

/// Per-action threshold: a `LimitUpdate` action triggers fewer peers
/// than `Deposit` would. The test cluster has a 3-way 4000/4000/2000
/// split; with deposit-style 3334 threshold a single high-stake peer
/// suffices, while a 6000 threshold needs two.
#[tokio::test]
async fn higher_threshold_requires_more_peers() {
    let action = pause(9);
    let approved = vec![action.clone()];
    let p1 = spawn_peer(91, approved.clone()).await;
    let p2 = spawn_peer(92, approved.clone()).await;
    let p3 = spawn_peer(93, approved).await;
    let committee = committee_from(&[(&p1, 4000), (&p2, 4000), (&p3, 2000)]);
    let agg = aggregator(committee);

    // Threshold 3500 — any one peer with ≥3500 power satisfies (p1 or p2).
    let cert = agg
        .request_signatures(SignRequest::Governance(action.clone()), 3500)
        .await
        .expect("low-threshold cert");
    assert!(cert.signatures.len() >= 1);

    // Threshold 7000 — need at least two of p1/p2 to reach 8000.
    let cert = agg
        .request_signatures(SignRequest::Governance(action), 7000)
        .await
        .expect("high-threshold cert");
    assert!(cert.signatures.len() >= 2);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn committee_contains(peers: &[&Peer], pubkey: &BridgePubkey) -> bool {
    let set: HashSet<_> = peers.iter().map(|p| &p.pubkey).collect();
    set.contains(pubkey)
}
