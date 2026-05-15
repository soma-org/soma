//! HTTP REST client for talking to a single peer's bridge node.
//!
//! Mirrors Sui's `sui-bridge/src/client/bridge_client.rs`. Each peer
//! committee member runs an [`crate::http_server`]; this module is the
//! other end — a typed wrapper that GETs the route corresponding to
//! the requested action, parses the [`SignedBridgeAction`] response,
//! and verifies the signature against the *expected* signing pubkey
//! before returning.
//!
//! ## Verification at the client edge
//!
//! Three checks happen here, defense-in-depth against a malicious or
//! broken peer:
//!
//! 1. **Pubkey match**: the response's `signer_pubkey` must equal the
//!    committee member's known [`BridgePubkey`]. Otherwise a rogue
//!    peer could return a signature from a different (but also
//!    committee-member) identity and steal another member's stake
//!    weight in the aggregator's tally.
//! 2. **Signature integrity**: the 65-byte recoverable signature must
//!    ecrecover back to the labeled pubkey, signing the canonical
//!    message bytes of the *expected* action. A peer that returns a
//!    signature over a slightly different action (different recipient,
//!    different amount) fails this check.
//! 3. **Action match**: the action embedded in the response must equal
//!    the action we requested. This catches a peer that misroutes,
//!    e.g. returns a sig for nonce N+1 when N was requested.

use std::time::Duration;

use fastcrypto::hash::Keccak256;
use fastcrypto::secp256k1::Secp256k1PublicKey;
use fastcrypto::secp256k1::recoverable::Secp256k1RecoverableSignature;
use fastcrypto::traits::{RecoverableSignature, ToFromBytes};
use reqwest::Client;
use tracing::warn;
use types::bridge::BridgePubkey;

use crate::error::{BridgeError, BridgeResult};
use crate::types::{BridgeAction, SignedBridgeAction};

/// Default per-request timeout. Sui uses 5s for non-overloaded peers; we
/// match. The aggregator's per-round budget composes with this.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// HTTP client targeting a single peer.
///
/// Constructed once per committee member at startup (or on committee
/// rotation). Cheaply cloneable — wraps `reqwest::Client`, which itself
/// pools connections under the hood.
#[derive(Debug, Clone)]
pub struct BridgeClient {
    /// Pubkey of the peer being talked to. Used for response validation
    /// — the response's `signer_pubkey` must equal this.
    peer_pubkey: BridgePubkey,
    /// Base URL of the peer's HTTP server, e.g. `http://10.0.0.5:9191`.
    /// Endpoint paths are appended.
    base_url: String,
    http: Client,
}

impl BridgeClient {
    /// Construct a client targeting `base_url` for a peer whose
    /// signing identity is `peer_pubkey`. `base_url` must be a
    /// well-formed HTTP/HTTPS URL with no trailing slash.
    pub fn new(peer_pubkey: BridgePubkey, base_url: String) -> BridgeResult<Self> {
        let trimmed = base_url.trim_end_matches('/').to_string();
        if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
            return Err(BridgeError::ConfigError(format!(
                "BridgeClient base_url must be http(s): got {trimmed}"
            )));
        }
        let http = Client::builder()
            .timeout(DEFAULT_REQUEST_TIMEOUT)
            .build()
            .map_err(|e| BridgeError::ConfigError(format!("reqwest builder: {e}")))?;
        Ok(Self { peer_pubkey, base_url: trimmed, http })
    }

    /// Test/inspection-only helper. Lets the aggregator log who it's
    /// talking to.
    pub fn peer_pubkey(&self) -> &BridgePubkey {
        &self.peer_pubkey
    }

    /// Request a signed [`BridgeAction::Deposit`] from this peer.
    /// `expected` is the canonical action the peer should produce
    /// (used for response validation — the peer must ecrecover-prove
    /// it signed *this* deposit, not some other one).
    pub async fn request_sign_deposit(
        &self,
        tx_hash: [u8; 32],
        event_idx: u16,
        expected: &BridgeAction,
    ) -> BridgeResult<SignedBridgeAction> {
        let url = format!(
            "{}/sign/bridge_tx/eth/soma/0x{}/{}",
            self.base_url,
            hex::encode(tx_hash),
            event_idx
        );
        self.fetch_and_verify(&url, expected).await
    }

    /// Request a signed [`BridgeAction::Withdrawal`] from this peer.
    /// Soma withdrawals are nonce-keyed on chain.
    pub async fn request_sign_withdrawal(
        &self,
        nonce: u64,
        expected: &BridgeAction,
    ) -> BridgeResult<SignedBridgeAction> {
        let url = format!("{}/sign/bridge_action/soma/eth/{}", self.base_url, nonce);
        self.fetch_and_verify(&url, expected).await
    }

    /// Request a signed governance action. The full action goes in the
    /// URL path; the peer's [`crate::governance_verifier::GovernanceVerifier`]
    /// will reject any action not on its operator-approved whitelist.
    pub async fn request_sign_governance(
        &self,
        action: &BridgeAction,
    ) -> BridgeResult<SignedBridgeAction> {
        let url = governance_url(&self.base_url, action)?;
        self.fetch_and_verify(&url, action).await
    }

    async fn fetch_and_verify(
        &self,
        url: &str,
        expected: &BridgeAction,
    ) -> BridgeResult<SignedBridgeAction> {
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| BridgeError::PeerConnectionFailed(format!("GET {url}: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            warn!(%url, %status, %body, "peer signing request returned non-success");
            return Err(BridgeError::PeerConnectionFailed(format!(
                "GET {url} returned {status}: {body}"
            )));
        }
        let signed: SignedBridgeAction = resp.json().await.map_err(|e| {
            BridgeError::PeerConnectionFailed(format!("decode JSON from {url}: {e}"))
        })?;
        self.verify_response(&signed, expected)?;
        Ok(signed)
    }

    fn verify_response(
        &self,
        signed: &SignedBridgeAction,
        expected: &BridgeAction,
    ) -> BridgeResult<()> {
        // 1. Pubkey match — block stake-theft by a peer rebinding identities.
        if signed.signer_pubkey.as_slice() != self.peer_pubkey.as_bytes().as_slice() {
            return Err(BridgeError::InvalidSignature(format!(
                "peer returned sig with mismatched signer_pubkey: expected {:?}, got {:?}",
                self.peer_pubkey.as_bytes(),
                &signed.signer_pubkey,
            )));
        }
        // 2. Action match — block misrouting.
        if &signed.action != expected {
            return Err(BridgeError::MismatchedAction);
        }
        // 3. Signature integrity — ecrecover against canonical message bytes.
        let msg = signed.action.to_message_bytes();
        let sig = Secp256k1RecoverableSignature::from_bytes(&signed.signature)
            .map_err(|e| BridgeError::InvalidSignature(format!("decode sig: {e:?}")))?;
        let recovered: Secp256k1PublicKey = sig
            .recover_with_hash::<Keccak256>(&msg)
            .map_err(|e| BridgeError::InvalidSignature(format!("ecrecover: {e:?}")))?;
        if recovered.as_ref() != self.peer_pubkey.as_bytes().as_slice() {
            return Err(BridgeError::InvalidSignature(
                "ecrecover did not match peer pubkey".to_string(),
            ));
        }
        Ok(())
    }
}

/// Build the URL path for a governance action. Returns
/// `InvalidBridgeClientRequest` for non-governance actions or those
/// not yet routed.
pub fn governance_url(base_url: &str, action: &BridgeAction) -> BridgeResult<String> {
    use types::bridge::derive_eth_address;
    match action {
        BridgeAction::EmergencyPause { nonce } => {
            Ok(format!("{base_url}/sign/emergency_button/{nonce}/0"))
        }
        BridgeAction::EmergencyUnpause { nonce } => {
            Ok(format!("{base_url}/sign/emergency_button/{nonce}/1"))
        }
        BridgeAction::UpdateCommitteeBlocklist { nonce, chain_id, blocklist_type, members } => {
            // Encode each pubkey as compressed hex; the server re-derives
            // 20-byte eth addresses internally before signing.
            let keys = members
                .iter()
                .map(|pk| format!("0x{}", hex::encode(pk.as_bytes())))
                .collect::<Vec<_>>()
                .join(",");
            let _ = derive_eth_address; // suppress unused (kept for symmetry note)
            Ok(format!(
                "{base_url}/sign/update_committee_blocklist/{}/{nonce}/{}/{keys}",
                chain_id.as_u8(),
                *blocklist_type as u8,
            ))
        }
        BridgeAction::LimitUpdate { nonce, chain_id, sending_chain_id, new_usd_limit } => {
            Ok(format!(
                "{base_url}/sign/update_limit/{}/{nonce}/{}/{new_usd_limit}",
                chain_id.as_u8(),
                sending_chain_id.as_u8(),
            ))
        }
        BridgeAction::EvmContractUpgrade { nonce, chain_id, proxy, new_impl, call_data } => {
            let proxy_hex = format!("0x{}", hex::encode(proxy));
            let impl_hex = format!("0x{}", hex::encode(new_impl));
            if call_data.is_empty() {
                Ok(format!(
                    "{base_url}/sign/upgrade_evm_contract/{}/{nonce}/{proxy_hex}/{impl_hex}",
                    chain_id.as_u8(),
                ))
            } else {
                let calldata_hex = format!("0x{}", hex::encode(call_data));
                Ok(format!(
                    "{base_url}/sign/upgrade_evm_contract/{}/{nonce}/{proxy_hex}/{impl_hex}/{calldata_hex}",
                    chain_id.as_u8(),
                ))
            }
        }
        BridgeAction::Deposit { .. } | BridgeAction::Withdrawal { .. } => {
            Err(BridgeError::InvalidBridgeClientRequest(
                "token transfers must use request_sign_deposit / request_sign_withdrawal"
                    .to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SignedBridgeAction;
    use fastcrypto::secp256k1::Secp256k1KeyPair;
    use fastcrypto::traits::KeyPair;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use types::base::SomaAddress;
    use types::bridge::{BridgeChainId, BridgeSignature, sign_bridge_message};
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn fresh_kp(seed: u8) -> Secp256k1KeyPair {
        let mut rng = StdRng::from_seed([seed; 32]);
        Secp256k1KeyPair::generate(&mut rng)
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

    fn deposit() -> BridgeAction {
        BridgeAction::Deposit {
            nonce: 1,
            eth_tx_hash: [0xab; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0x42; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1_000_000,
            timestamp_ms: 1_700_000_000_000,
        }
    }

    fn pause() -> BridgeAction {
        BridgeAction::EmergencyPause { nonce: 7 }
    }

    #[tokio::test]
    async fn test_request_sign_deposit_happy_path() {
        let kp = fresh_kp(11);
        let pk = BridgePubkey::from_keypair(&kp);
        let action = deposit();
        let signed = sign(&kp, &action);
        let body = serde_json::to_string(&signed).unwrap();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/sign/bridge_tx/eth/soma/0x[0-9a-f]+/0$"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let client = BridgeClient::new(pk, server.uri()).unwrap();
        let got = client.request_sign_deposit([0xab; 32], 0, &action).await.unwrap();
        assert_eq!(got.action, action);
    }

    #[tokio::test]
    async fn test_request_sign_withdrawal_happy_path() {
        let kp = fresh_kp(22);
        let pk = BridgePubkey::from_keypair(&kp);
        let action = BridgeAction::Withdrawal {
            nonce: 42,
            sender: SomaAddress::random(),
            target_chain: types::bridge::BridgeChainId::EthCustom,
            recipient_eth_address: [9; 20],
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 500_000,
            timestamp_ms: 0,
        };
        let signed = sign(&kp, &action);
        let body = serde_json::to_string(&signed).unwrap();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/sign/bridge_action/soma/eth/42$"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let client = BridgeClient::new(pk, server.uri()).unwrap();
        let got = client.request_sign_withdrawal(42, &action).await.unwrap();
        assert_eq!(got.action, action);
    }

    #[tokio::test]
    async fn test_request_sign_governance_happy_path() {
        let kp = fresh_kp(33);
        let pk = BridgePubkey::from_keypair(&kp);
        let action = pause();
        let signed = sign(&kp, &action);
        let body = serde_json::to_string(&signed).unwrap();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/sign/emergency_button/7/0$"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let client = BridgeClient::new(pk, server.uri()).unwrap();
        let got = client.request_sign_governance(&action).await.unwrap();
        assert_eq!(got.action, action);
    }

    #[tokio::test]
    async fn test_peer_returns_wrong_signer_pubkey_rejected() {
        // Peer A's URL claims A's pubkey, but the response signature is
        // from peer B. Client must reject — otherwise B could feed sigs
        // through A's identity slot and inflate the aggregator's tally.
        let kp_a = fresh_kp(44);
        let pk_a = BridgePubkey::from_keypair(&kp_a);
        let kp_b = fresh_kp(45);
        let action = pause();
        let signed_by_b = sign(&kp_b, &action);
        let body = serde_json::to_string(&signed_by_b).unwrap();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let client = BridgeClient::new(pk_a, server.uri()).unwrap();
        let err = client.request_sign_governance(&action).await.unwrap_err();
        assert!(matches!(err, BridgeError::InvalidSignature(_)), "{err:?}");
    }

    #[tokio::test]
    async fn test_peer_returns_mismatched_action_rejected() {
        let kp = fresh_kp(55);
        let pk = BridgePubkey::from_keypair(&kp);
        let requested = pause();
        // Peer signs a different action (different nonce).
        let returned = BridgeAction::EmergencyPause { nonce: 99 };
        let signed = sign(&kp, &returned);
        let body = serde_json::to_string(&signed).unwrap();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let client = BridgeClient::new(pk, server.uri()).unwrap();
        let err = client.request_sign_governance(&requested).await.unwrap_err();
        assert!(matches!(err, BridgeError::MismatchedAction), "{err:?}");
    }

    #[tokio::test]
    async fn test_peer_returns_invalid_signature_rejected() {
        let kp = fresh_kp(66);
        let pk = BridgePubkey::from_keypair(&kp);
        let action = pause();
        // Self-consistent action + pubkey, but signature is garbage.
        let signed = SignedBridgeAction {
            action: action.clone(),
            signer_pubkey: pk.as_bytes().to_vec(),
            signature: vec![0u8; 65],
        };
        let body = serde_json::to_string(&signed).unwrap();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let client = BridgeClient::new(pk, server.uri()).unwrap();
        let err = client.request_sign_governance(&action).await.unwrap_err();
        assert!(matches!(err, BridgeError::InvalidSignature(_)), "{err:?}");
    }

    #[tokio::test]
    async fn test_peer_returns_5xx_propagates_error() {
        let kp = fresh_kp(77);
        let pk = BridgePubkey::from_keypair(&kp);

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500).set_body_string("oops"))
            .mount(&server)
            .await;

        let client = BridgeClient::new(pk, server.uri()).unwrap();
        let err = client.request_sign_governance(&pause()).await.unwrap_err();
        assert!(matches!(err, BridgeError::PeerConnectionFailed(_)), "{err:?}");
    }

    #[test]
    fn test_invalid_base_url_rejected() {
        let pk = BridgePubkey::from_keypair(&fresh_kp(88));
        assert!(matches!(
            BridgeClient::new(pk, "ftp://x".to_string()),
            Err(BridgeError::ConfigError(_)),
        ));
    }

    #[test]
    fn test_governance_url_emergency_pause() {
        let url = governance_url("http://x:9", &BridgeAction::EmergencyPause { nonce: 5 }).unwrap();
        assert_eq!(url, "http://x:9/sign/emergency_button/5/0");
    }

    #[test]
    fn test_governance_url_emergency_unpause() {
        let url =
            governance_url("http://x:9", &BridgeAction::EmergencyUnpause { nonce: 8 }).unwrap();
        assert_eq!(url, "http://x:9/sign/emergency_button/8/1");
    }

    #[test]
    fn test_governance_url_limit_update() {
        let action = BridgeAction::LimitUpdate {
            nonce: 3,
            chain_id: BridgeChainId::EthCustom,
            sending_chain_id: BridgeChainId::SomaCustom,
            new_usd_limit: 1_000_000,
        };
        let url = governance_url("http://x:9", &action).unwrap();
        assert_eq!(url, "http://x:9/sign/update_limit/12/3/2/1000000");
    }

    #[test]
    fn test_governance_url_evm_upgrade_no_calldata() {
        let action = BridgeAction::EvmContractUpgrade {
            nonce: 1,
            chain_id: BridgeChainId::EthCustom,
            proxy: [0x11; 20],
            new_impl: [0x22; 20],
            call_data: vec![],
        };
        let url = governance_url("http://x:9", &action).unwrap();
        assert_eq!(
            url,
            "http://x:9/sign/upgrade_evm_contract/12/1/0x1111111111111111111111111111111111111111/0x2222222222222222222222222222222222222222"
        );
    }

    #[test]
    fn test_governance_url_token_transfer_rejected() {
        assert!(matches!(
            governance_url("http://x:9", &deposit()),
            Err(BridgeError::InvalidBridgeClientRequest(_)),
        ));
    }
}
