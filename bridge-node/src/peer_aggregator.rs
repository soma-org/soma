//! Peer-broadcast signature aggregator.
//!
//! Mirrors Sui's `BridgeAuthorityAggregator::request_committee_signatures`
//! plus `quorum_map_then_reduce_with_timeout_and_prefs`. Given a
//! [`BridgeAction`] and the target voting-power threshold, fan out HTTP
//! sig requests to every non-blocklisted committee member, accumulate
//! responses as they stream back, and return a [`CertifiedBridgeAction`]
//! as soon as accumulated stake crosses the threshold.
//!
//! ```text
//!         action + threshold
//!                │
//!                ▼
//!     ┌───────────────────────────────┐
//!     │ snapshot live committee       │
//!     │ (Arc<ArcSwap<...>> from monitor)│
//!     └──────────────┬────────────────┘
//!                    │
//!         ┌──────────┴──────────┐
//!         │ FuturesUnordered    │ ← one BridgeClient GET per non-blocklisted member
//!         │ (parallel fan-out)  │
//!         └──────────┬──────────┘
//!                    │  (as completions arrive)
//!                    ▼
//!     ┌───────────────────────────────┐
//!     │ verify sig at client edge     │ ← in BridgeClient
//!     └──────────────┬────────────────┘
//!                    │ Ok(sig) → accumulate
//!                    ▼
//!         stake_so_far += member.voting_power
//!                    │
//!         stake_so_far >= threshold? ──yes──▶ return CertifiedBridgeAction
//!                    │ no
//!                    └────── more pending? continue : InsufficientStake
//! ```
//!
//! ## Why parallel fan-out, not preference-ordered
//!
//! Sui sorts peers by stake (`quorum_map_then_reduce_with_timeout_and_prefs`)
//! and issues calls in waves, holding back low-stake peers if high-stake
//! peers will satisfy the threshold first. The motivation is bandwidth +
//! peer load — don't bother low-stake peers if you don't need them.
//!
//! For Soma's scale (committee size 4–32 today, target ~64 at scale) the
//! optimization is premature: a parallel fan-out completes in one round
//! trip and saves dramatically on latency vs. sequential waves. If load
//! becomes a problem we can revisit by sorting peers and capping
//! concurrent in-flight calls.

use std::collections::BTreeMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use futures::stream::{FuturesUnordered, StreamExt};
use tracing::{debug, info, warn};
use types::bridge::{BridgeCommittee, BridgePubkey, BridgeSignature};

use crate::aggregator::{BridgeAuthorityAggregator, CertifiedBridgeAction};
use crate::bridge_client::BridgeClient;
use crate::error::{BridgeError, BridgeResult};
use crate::types::BridgeAction;

/// What action + routing info the aggregator should request sigs for.
///
/// Bundling the routing context (event_idx for deposits, nonce-only for
/// withdrawals, action-in-URL for governance) into one enum keeps the
/// public API a single method while the URL-construction logic lives
/// in [`crate::bridge_client::BridgeClient`].
pub enum SignRequest {
    /// Eth → Soma USDC deposit. `tx_hash + event_idx` route peers to
    /// `EthClient::get_finalized_bridge_action_maybe`. `expected` is
    /// what each peer's response is validated against (recipient,
    /// amount, nonce, etc.).
    EthDeposit { tx_hash: [u8; 32], event_idx: u16, expected: BridgeAction },
    /// Soma → Eth USDC withdrawal. Nonce-keyed on the Soma side.
    SomaWithdrawal { nonce: u64, expected: BridgeAction },
    /// Governance action (pause/blocklist/limit-update/EVM upgrade).
    /// The full action goes in the URL; peers validate against their
    /// operator-approved whitelist.
    Governance(BridgeAction),
}

impl SignRequest {
    pub fn expected_action(&self) -> &BridgeAction {
        match self {
            SignRequest::EthDeposit { expected, .. }
            | SignRequest::SomaWithdrawal { expected, .. }
            | SignRequest::Governance(expected) => expected,
        }
    }
}

/// Broadcast aggregator. Holds the live committee snapshot (kept fresh
/// by [`crate::monitor::BridgeMonitor`]) and produces certs by fanning
/// out to peers.
pub struct PeerBroadcastAggregator {
    /// Live committee — loaded fresh on each call so a mid-flight
    /// rotation reflects on the next request.
    committee: Arc<ArcSwap<BridgeCommittee>>,
}

impl PeerBroadcastAggregator {
    pub fn new(committee: Arc<ArcSwap<BridgeCommittee>>) -> Self {
        Self { committee }
    }

    /// Map a [`BridgeAction`] to the [`SignRequest`] the peer fan-out
    /// should issue. Token transfers carry their own routing data
    /// (`eth_tx_hash` + `eth_event_idx` for deposits, `nonce` for
    /// withdrawals); governance actions go fully in the URL path.
    fn dispatch(action: BridgeAction) -> SignRequest {
        match &action {
            BridgeAction::Deposit { eth_tx_hash, eth_event_idx, .. } => SignRequest::EthDeposit {
                tx_hash: *eth_tx_hash,
                event_idx: *eth_event_idx,
                expected: action.clone(),
            },
            BridgeAction::Withdrawal { nonce, .. } => {
                SignRequest::SomaWithdrawal { nonce: *nonce, expected: action.clone() }
            }
            BridgeAction::EmergencyPause { .. }
            | BridgeAction::EmergencyUnpause { .. }
            | BridgeAction::UpdateCommitteeBlocklist { .. }
            | BridgeAction::LimitUpdate { .. }
            | BridgeAction::EvmContractUpgrade { .. } => SignRequest::Governance(action),
        }
    }

    /// Block until either (a) sigs totaling `threshold` voting power
    /// arrive, in which case return the cert; or (b) every non-
    /// blocklisted member has responded (or failed) and the total is
    /// still under threshold, in which case return
    /// [`BridgeError::InsufficientStake`].
    ///
    /// This is the direct-call entry; the [`BridgeAuthorityAggregator`]
    /// trait impl is a thin shim that dispatches `&BridgeAction` to the
    /// appropriate [`SignRequest`] before calling here.
    pub async fn request_signatures(
        &self,
        req: SignRequest,
        threshold: u64,
    ) -> BridgeResult<CertifiedBridgeAction> {
        let committee = self.committee.load_full();

        // Build one BridgeClient per non-blocklisted member with a
        // valid http_url. We skip rather than error on individual
        // construction failures — a misconfigured peer should not
        // sink the whole round.
        let mut peers: Vec<(BridgeClient, u64)> = Vec::new();
        let mut skipped_blocklisted = 0usize;
        let mut skipped_unconfigured = 0usize;
        let mut skipped_bad_url = 0usize;
        for (pubkey, member) in &committee.members {
            if member.is_blocklisted {
                skipped_blocklisted += 1;
                continue;
            }
            if member.http_url.is_empty() {
                skipped_unconfigured += 1;
                continue;
            }
            match BridgeClient::new(pubkey.clone(), member.http_url.clone()) {
                Ok(c) => peers.push((c, member.voting_power)),
                Err(e) => {
                    warn!(?pubkey, err = %e, "skipping peer with bad http_url");
                    skipped_bad_url += 1;
                }
            }
        }

        // Fan out one request per peer.
        let mut futures = FuturesUnordered::new();
        let req = Arc::new(req);
        for (client, power) in peers {
            let req = Arc::clone(&req);
            futures.push(async move {
                let pk = client.peer_pubkey().clone();
                let res = match &*req {
                    SignRequest::EthDeposit { tx_hash, event_idx, expected } => {
                        client.request_sign_deposit(*tx_hash, *event_idx, expected).await
                    }
                    SignRequest::SomaWithdrawal { nonce, expected } => {
                        client.request_sign_withdrawal(*nonce, expected).await
                    }
                    SignRequest::Governance(action) => client.request_sign_governance(action).await,
                };
                (pk, power, res)
            });
        }

        // Accumulate signatures until threshold is met or no peers remain.
        let mut signatures: BTreeMap<BridgePubkey, BridgeSignature> = BTreeMap::new();
        let mut stake_so_far: u64 = 0;
        let mut peer_failures: usize = 0;
        while let Some((pk, power, res)) = futures.next().await {
            match res {
                Ok(signed) => {
                    let sig = match BridgeSignature::from_bytes(&signed.signature) {
                        Ok(s) => s,
                        Err(e) => {
                            warn!(?pk, err = ?e, "peer returned malformed sig bytes");
                            peer_failures += 1;
                            continue;
                        }
                    };
                    if signatures.insert(pk.clone(), sig).is_some() {
                        // Same peer responded twice somehow — shouldn't
                        // happen with FuturesUnordered, but be defensive.
                        warn!(?pk, "duplicate peer response — dropping");
                        continue;
                    }
                    stake_so_far = stake_so_far.saturating_add(power);
                    debug!(?pk, power, stake_so_far, threshold, "peer sig accepted");
                    if stake_so_far >= threshold {
                        info!(
                            sigs_collected = signatures.len(),
                            stake = stake_so_far,
                            threshold,
                            skipped_blocklisted,
                            skipped_unconfigured,
                            skipped_bad_url,
                            "cert assembled"
                        );
                        return Ok(CertifiedBridgeAction {
                            action: req.expected_action().clone(),
                            signatures,
                        });
                    }
                }
                Err(e) => {
                    warn!(?pk, err = %e, "peer sig request failed");
                    peer_failures += 1;
                }
            }
        }

        warn!(
            stake = stake_so_far,
            threshold,
            sigs_collected = signatures.len(),
            peer_failures,
            skipped_blocklisted,
            skipped_unconfigured,
            skipped_bad_url,
            "aggregation finished below threshold"
        );
        Err(BridgeError::InsufficientStake { got: stake_so_far, required: threshold })
    }
}

/// [`BridgeAuthorityAggregator`] adapter — lets the action executor
/// call `aggregator.request_committee_signatures(&action, threshold)`
/// (the existing trait signature) and have it dispatch internally to
/// the right [`SignRequest`] variant.
#[async_trait::async_trait]
impl BridgeAuthorityAggregator for PeerBroadcastAggregator {
    async fn request_committee_signatures(
        &self,
        action: &BridgeAction,
        threshold: u64,
    ) -> BridgeResult<CertifiedBridgeAction> {
        let req = Self::dispatch(action.clone());
        self.request_signatures(req, threshold).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SignedBridgeAction;
    use fastcrypto::secp256k1::Secp256k1KeyPair;
    use fastcrypto::traits::{KeyPair, ToFromBytes};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use types::base::SomaAddress;
    use types::bridge::{BridgeChainId, BridgeCommittee, BridgeMember, sign_bridge_message};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn fresh_kp(seed: u8) -> Secp256k1KeyPair {
        let mut rng = StdRng::from_seed([seed; 32]);
        Secp256k1KeyPair::generate(&mut rng)
    }

    fn pause() -> BridgeAction {
        BridgeAction::EmergencyPause { nonce: 1 }
    }

    fn sign(kp: &Secp256k1KeyPair, action: &BridgeAction) -> SignedBridgeAction {
        let msg = action.to_message_bytes();
        let sig = sign_bridge_message(kp, &msg);
        let pk = BridgePubkey::from_keypair(kp);
        SignedBridgeAction {
            action: action.clone(),
            signer_pubkey: pk.as_bytes().to_vec(),
            signature: BridgeSignature::from_bytes(sig.as_ref()).unwrap().as_bytes().to_vec(),
        }
    }

    /// Spawn a wiremock that responds with the given signed action for
    /// any sig endpoint. Returns the server (kept alive for test
    /// scope) + its base URL.
    async fn mock_peer(signed: SignedBridgeAction) -> (MockServer, String) {
        let body = serde_json::to_string(&signed).unwrap();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/sign/emergency_button/1/0"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;
        let uri = server.uri();
        (server, uri)
    }

    /// Build a committee where each `(pubkey, power, response)` becomes
    /// a [`BridgeMember`] pointing at a freshly spawned wiremock. If
    /// `response` is `None`, the member's URL points at a closed port —
    /// a connection refusal is the test's "peer down" signal.
    async fn committee_with_mocks(
        peers: Vec<(BridgePubkey, u64, Option<SignedBridgeAction>)>,
    ) -> (BridgeCommittee, Vec<Option<MockServer>>) {
        let mut committee = BridgeCommittee::empty();
        let mut servers = Vec::new();
        for (pk, power, resp) in peers {
            let url = if let Some(signed) = resp {
                let (server, uri) = mock_peer(signed).await;
                servers.push(Some(server));
                uri
            } else {
                servers.push(None);
                "http://127.0.0.1:1".to_string() // guaranteed-refused
            };
            committee.members.insert(
                pk,
                BridgeMember {
                    soma_address: SomaAddress::random(),
                    voting_power: power,
                    http_url: url,
                    is_blocklisted: false,
                },
            );
        }
        (committee, servers)
    }

    #[tokio::test]
    async fn test_meets_threshold_with_two_of_three_peers() {
        let action = pause();
        let kp_a = fresh_kp(10);
        let kp_b = fresh_kp(20);
        let kp_c = fresh_kp(30);
        let pk_a = BridgePubkey::from_keypair(&kp_a);
        let pk_b = BridgePubkey::from_keypair(&kp_b);
        let pk_c = BridgePubkey::from_keypair(&kp_c);
        let (committee, _servers) = committee_with_mocks(vec![
            (pk_a, 4000, Some(sign(&kp_a, &action))),
            (pk_b, 4000, Some(sign(&kp_b, &action))),
            (pk_c, 2000, Some(sign(&kp_c, &action))),
        ])
        .await;

        let agg = PeerBroadcastAggregator::new(Arc::new(ArcSwap::from_pointee(committee)));
        let cert =
            agg.request_signatures(SignRequest::Governance(action.clone()), 7000).await.unwrap();
        assert!(cert.signatures.len() >= 2);
        assert_eq!(cert.action, action);
    }

    #[tokio::test]
    async fn test_returns_insufficient_stake_when_peers_fail() {
        // Only one peer responds; threshold demands >1's stake worth.
        let action = pause();
        let kp_a = fresh_kp(11);
        let kp_b = fresh_kp(22);
        let kp_c = fresh_kp(33);
        let pk_a = BridgePubkey::from_keypair(&kp_a);
        let pk_b = BridgePubkey::from_keypair(&kp_b);
        let pk_c = BridgePubkey::from_keypair(&kp_c);
        let (committee, _s) = committee_with_mocks(vec![
            (pk_a, 2500, Some(sign(&kp_a, &action))),
            (pk_b, 2500, None), // down
            (pk_c, 5000, None), // down
        ])
        .await;

        let agg = PeerBroadcastAggregator::new(Arc::new(ArcSwap::from_pointee(committee)));
        let err = agg.request_signatures(SignRequest::Governance(action), 5000).await.unwrap_err();
        assert!(
            matches!(err, BridgeError::InsufficientStake { got: 2500, required: 5000 }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn test_skips_blocklisted_members() {
        let action = pause();
        let kp_a = fresh_kp(40);
        let kp_b = fresh_kp(50);
        let kp_c = fresh_kp(60);
        let pk_a = BridgePubkey::from_keypair(&kp_a);
        let pk_b = BridgePubkey::from_keypair(&kp_b);
        let pk_c = BridgePubkey::from_keypair(&kp_c);
        let (mut committee, _s) = committee_with_mocks(vec![
            (pk_a, 5000, Some(sign(&kp_a, &action))),
            // B's response would meet threshold on its own, but B is
            // blocklisted — must be skipped entirely. Stake from B must
            // not be counted.
            (pk_b.clone(), 5000, Some(sign(&kp_b, &action))),
            (pk_c, 2000, None),
        ])
        .await;
        // Blocklist member B.
        committee.members.get_mut(&pk_b).unwrap().is_blocklisted = true;

        let agg = PeerBroadcastAggregator::new(Arc::new(ArcSwap::from_pointee(committee)));
        // Threshold 6000 requires both A + B's stake; with B blocklisted
        // it's unreachable.
        let err = agg.request_signatures(SignRequest::Governance(action), 6000).await.unwrap_err();
        assert!(
            matches!(err, BridgeError::InsufficientStake { got: 5000, required: 6000 }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn test_skips_members_with_no_http_url() {
        // Same as blocklisted: an empty http_url means the peer hasn't
        // registered an endpoint and can't be contacted. Skipping it
        // is fine; counting its stake would lie to the caller.
        let action = pause();
        let kp_a = fresh_kp(70);
        let kp_b = fresh_kp(80);
        let kp_c = fresh_kp(90);
        // C is contactable (has URL) but we'll spawn no mock — the
        // simple way to model "registered but down". B has empty
        // http_url — definitely skipped.
        let mut committee = BridgeCommittee::empty();
        let (server_a, uri_a) = mock_peer(sign(&kp_a, &action)).await;
        committee.members.insert(
            BridgePubkey::from_keypair(&kp_a),
            BridgeMember {
                soma_address: SomaAddress::random(),
                voting_power: 3000,
                http_url: uri_a.clone(),
                is_blocklisted: false,
            },
        );
        committee.members.insert(
            BridgePubkey::from_keypair(&kp_b),
            BridgeMember {
                soma_address: SomaAddress::random(),
                voting_power: 4000,
                http_url: String::new(), // unconfigured
                is_blocklisted: false,
            },
        );
        committee.members.insert(
            BridgePubkey::from_keypair(&kp_c),
            BridgeMember {
                soma_address: SomaAddress::random(),
                voting_power: 3000,
                http_url: "http://127.0.0.1:1".to_string(), // down
                is_blocklisted: false,
            },
        );

        let agg = PeerBroadcastAggregator::new(Arc::new(ArcSwap::from_pointee(committee)));
        let err = agg.request_signatures(SignRequest::Governance(action), 5000).await.unwrap_err();
        // Only A's 3000 stake is counted.
        assert!(
            matches!(err, BridgeError::InsufficientStake { got: 3000, required: 5000 }),
            "{err:?}"
        );
        drop(server_a);
    }

    #[tokio::test]
    async fn test_early_return_when_high_stake_peer_satisfies_threshold() {
        // A single high-stake peer can clear the threshold. The
        // aggregator must NOT wait for slow peers in that case.
        let action = pause();
        let kp_a = fresh_kp(100);
        let kp_b = fresh_kp(110);
        let pk_a = BridgePubkey::from_keypair(&kp_a);
        let pk_b = BridgePubkey::from_keypair(&kp_b);
        let (committee, _s) = committee_with_mocks(vec![
            (pk_a, 8000, Some(sign(&kp_a, &action))),
            (pk_b, 2000, None), // down — would never respond
        ])
        .await;
        let agg = PeerBroadcastAggregator::new(Arc::new(ArcSwap::from_pointee(committee)));
        let cert =
            agg.request_signatures(SignRequest::Governance(action.clone()), 5000).await.unwrap();
        assert_eq!(cert.signatures.len(), 1);
        assert_eq!(cert.action, action);
    }

    /// Regression for the `BridgeAuthorityAggregator` trait impl: the
    /// executor passes `&BridgeAction` (not `SignRequest`), so we must
    /// dispatch a `Deposit` action correctly to `EthDeposit` (with the
    /// action's `eth_event_idx` carried through to the URL path).
    #[tokio::test]
    async fn test_trait_dispatch_deposit_uses_event_idx_from_action() {
        let kp = fresh_kp(150);
        let pk = BridgePubkey::from_keypair(&kp);
        let action = BridgeAction::Deposit {
            nonce: 1,
            eth_tx_hash: [0xCC; 32],
            eth_event_idx: 7,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0x42; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1_000_000,
            timestamp_ms: 1_700_000_000_000,
        };
        let signed = sign(&kp, &action);
        let body = serde_json::to_string(&signed).unwrap();
        let server = MockServer::start().await;
        // The dispatch must produce a URL whose event_idx is 7 (from
        // the action), not 0. Any other value won't match this mock.
        Mock::given(method("GET"))
            .and(path(format!("/sign/bridge_tx/eth/soma/0x{}/7", hex::encode([0xCC; 32]))))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;
        let mut committee = BridgeCommittee::empty();
        committee.members.insert(
            pk,
            BridgeMember {
                soma_address: SomaAddress::random(),
                voting_power: 10_000,
                http_url: server.uri(),
                is_blocklisted: false,
            },
        );
        let agg = PeerBroadcastAggregator::new(Arc::new(ArcSwap::from_pointee(committee)));
        // Call through the trait (the executor's call path), not the
        // direct `request_signatures(SignRequest, _)` entry.
        let cert =
            <PeerBroadcastAggregator as BridgeAuthorityAggregator>::request_committee_signatures(
                &agg, &action, 5_000,
            )
            .await
            .unwrap();
        assert_eq!(cert.action, action);
    }

    #[tokio::test]
    async fn test_unused_chain_id() {
        // Tiny coverage test that LimitUpdate URLs reach peers correctly.
        // (Routing is exhaustively tested in bridge_client::tests; this
        // is just a smoke test that the aggregator wires LimitUpdate
        // through SignRequest::Governance.)
        let action = BridgeAction::LimitUpdate {
            nonce: 1,
            chain_id: BridgeChainId::EthCustom,
            sending_chain_id: BridgeChainId::SomaCustom,
            new_usd_limit: 99,
        };
        let kp_a = fresh_kp(200);
        let signed = sign(&kp_a, &action);
        let body = serde_json::to_string(&signed).unwrap();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/sign/update_limit/12/1/2/99"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;
        let mut committee = BridgeCommittee::empty();
        committee.members.insert(
            BridgePubkey::from_keypair(&kp_a),
            BridgeMember {
                soma_address: SomaAddress::random(),
                voting_power: 10000,
                http_url: server.uri(),
                is_blocklisted: false,
            },
        );
        let agg = PeerBroadcastAggregator::new(Arc::new(ArcSwap::from_pointee(committee)));
        let cert =
            agg.request_signatures(SignRequest::Governance(action.clone()), 5000).await.unwrap();
        assert_eq!(cert.action, action);
    }
}
