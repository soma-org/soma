//! Bridge request handler — the fetch-and-sign core.
//!
//! Mirrors Sui's `sui-bridge/src/server/handler.rs`. Each REST endpoint
//! on [`crate::server`] delegates here. For every incoming sig request
//! the handler proves the action against authoritative state (Eth
//! tx receipt, Soma `PendingWithdrawal`, or the operator-approved
//! governance whitelist) *before* it produces a signature.
//!
//! ```text
//!     POST /sign/bridge_tx/eth/soma/{tx_hash}/{event_idx}
//!         │
//!         ▼
//!     handle_eth_tx_hash
//!         │
//!         ├─ EthClient::get_finalized_bridge_action_maybe(tx_hash, event_idx)
//!         │     (verifies finality + parses Deposit log)
//!         │
//!         └─ sign(action) → SignedBridgeAction
//! ```
//!
//! ## Why fetch-and-sign
//!
//! The alternative — accepting a peer-supplied `BridgeAction` and
//! signing it after a digest match — assumes the peer is honest about
//! what's on chain. Fetch-and-sign cuts that out: the signer is the
//! authority on what the chain says, and the requester only supplies a
//! *pointer* (tx hash + log index, or withdrawal nonce). A rogue peer
//! that mis-states the deposit recipient/amount produces a request the
//! handler won't sign.
//!
//! Governance is the one exception (there's no on-chain event to fetch
//! a pause/blocklist from — it's a future action). Those go through
//! [`GovernanceVerifier`] which is the operator's pre-authorization.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::info;
use types::bridge::{BridgePubkey, BridgeSignature, sign_bridge_message};

use crate::error::{BridgeError, BridgeResult};
use crate::eth_client::EthClient;
use crate::governance_verifier::GovernanceVerifier;
use crate::soma_client::{SomaBridgeClient, SomaBridgeClientInner};
use crate::types::{BridgeAction, SignedBridgeAction};

/// Bridge keypair alias (Sui parity: `BridgeAuthorityKeyPair`).
pub type BridgeAuthorityKeyPair = fastcrypto::secp256k1::Secp256k1KeyPair;

#[async_trait]
pub trait BridgeRequestHandlerTrait: Send + Sync {
    /// Sign a [`BridgeAction::Deposit`] proven by an Eth tx + log index.
    async fn handle_eth_tx_hash(
        &self,
        tx_hash_hex: String,
        event_idx: u16,
    ) -> BridgeResult<SignedBridgeAction>;

    /// Sign a [`BridgeAction::Withdrawal`] proven by an on-chain Soma
    /// `PendingWithdrawal` at `nonce`. Soma's outbound transfers are
    /// nonce-keyed rather than tx-digest-keyed (the on-chain object id
    /// is derived from `nonce`), so this is the Soma analog of Sui's
    /// `handle_sui_token_transfer`.
    async fn handle_soma_withdrawal(&self, nonce: u64) -> BridgeResult<SignedBridgeAction>;

    /// Sign a governance action (pause / unpause / blocklist update /
    /// limit update / EVM upgrade / committee update). The action must
    /// be byte-identical to one in the operator-approved whitelist.
    async fn handle_governance_action(
        &self,
        action: BridgeAction,
    ) -> BridgeResult<SignedBridgeAction>;
}

/// Concrete handler. Holds the long-lived dependencies — signer
/// keypair, Eth fetcher, Soma fetcher, governance whitelist — and
/// produces signatures on request.
pub struct BridgeRequestHandler<SC: SomaBridgeClientInner> {
    signer: Arc<BridgeAuthorityKeyPair>,
    signer_pubkey: BridgePubkey,
    eth_client: Arc<EthClient>,
    soma_client: Arc<SomaBridgeClient<SC>>,
    governance_verifier: GovernanceVerifier,
}

impl<SC: SomaBridgeClientInner> BridgeRequestHandler<SC> {
    pub fn new(
        signer: BridgeAuthorityKeyPair,
        eth_client: Arc<EthClient>,
        soma_client: Arc<SomaBridgeClient<SC>>,
        approved_governance_actions: Vec<BridgeAction>,
    ) -> BridgeResult<Self> {
        let signer_pubkey = BridgePubkey::from_keypair(&signer);
        Ok(Self {
            signer: Arc::new(signer),
            signer_pubkey,
            eth_client,
            soma_client,
            governance_verifier: GovernanceVerifier::new(approved_governance_actions)?,
        })
    }

    /// Produce a [`SignedBridgeAction`] from a verified action. This is
    /// the only place in the handler that touches the private key; every
    /// public entry point must verify first and call here last.
    fn sign(&self, action: BridgeAction) -> SignedBridgeAction {
        let msg = action.to_message_bytes();
        let sig = sign_bridge_message(&self.signer, &msg);
        let sig_bytes = BridgeSignature::from_bytes(sig.as_ref())
            .expect("recoverable secp256k1 signature is always 65 bytes");
        SignedBridgeAction {
            action,
            signer_pubkey: self.signer_pubkey.as_bytes().to_vec(),
            signature: sig_bytes.as_bytes().to_vec(),
        }
    }

    /// Verify an Eth-side deposit by re-fetching the receipt and parsing
    /// the log at `event_idx`. Encapsulates the only authority-relevant
    /// I/O — caller can wrap with caching if needed.
    async fn verify_eth(&self, tx_hash: [u8; 32], event_idx: u16) -> BridgeResult<BridgeAction> {
        let action = self.eth_client.get_finalized_bridge_action_maybe(tx_hash, event_idx).await?;
        info!(?action, "Eth bridge action verified");
        Ok(action)
    }

    /// Verify a Soma-side withdrawal by fetching the on-chain
    /// `PendingWithdrawal` at `nonce` and reconstructing the action from
    /// its observed fields. Rejects already-completed withdrawals (the
    /// cert is already attached — there's nothing to sign).
    async fn verify_soma_withdrawal(&self, nonce: u64) -> BridgeResult<BridgeAction> {
        let pw = self
            .soma_client
            .get_pending_withdrawal(nonce)
            .await?
            .ok_or(BridgeError::NoBridgeEventsInTxPosition)?;
        if pw.verified_signatures.is_some() {
            return Err(BridgeError::InvalidBridgeClientRequest(format!(
                "withdrawal nonce {nonce} already has cert attached"
            )));
        }
        // Token type is always USDC today; the chain enforces this in
        // the bridge executor. target_chain comes from the on-chain
        // PendingWithdrawal — the user who initiated the burn picked
        // the destination Eth chain at burn time.
        let action = BridgeAction::Withdrawal {
            nonce: pw.nonce,
            sender: pw.sender,
            target_chain: pw.target_chain,
            recipient_eth_address: pw.recipient_eth_address,
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: pw.amount,
            timestamp_ms: pw.created_at_ms,
        };
        info!(?action, "Soma withdrawal verified");
        Ok(action)
    }
}

#[async_trait]
impl<SC: SomaBridgeClientInner> BridgeRequestHandlerTrait for BridgeRequestHandler<SC> {
    async fn handle_eth_tx_hash(
        &self,
        tx_hash_hex: String,
        event_idx: u16,
    ) -> BridgeResult<SignedBridgeAction> {
        let tx_hash = parse_tx_hash(&tx_hash_hex)?;
        let action = self.verify_eth(tx_hash, event_idx).await?;
        Ok(self.sign(action))
    }

    async fn handle_soma_withdrawal(&self, nonce: u64) -> BridgeResult<SignedBridgeAction> {
        let action = self.verify_soma_withdrawal(nonce).await?;
        Ok(self.sign(action))
    }

    async fn handle_governance_action(
        &self,
        action: BridgeAction,
    ) -> BridgeResult<SignedBridgeAction> {
        if !action.is_governance_action() {
            return Err(BridgeError::ActionIsNotGovernanceAction);
        }
        let verified = self.governance_verifier.verify(action)?;
        Ok(self.sign(verified))
    }
}

fn parse_tx_hash(hex: &str) -> BridgeResult<[u8; 32]> {
    let stripped = hex.strip_prefix("0x").unwrap_or(hex);
    let raw = hex::decode(stripped).map_err(|e| {
        BridgeError::InvalidBridgeClientRequest(format!("tx_hash not valid hex: {e}"))
    })?;
    if raw.len() != 32 {
        return Err(BridgeError::InvalidBridgeClientRequest(format!(
            "tx_hash must be 32 bytes, got {}",
            raw.len()
        )));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&raw);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soma_client::tests::MockSomaClient;
    use fastcrypto::secp256k1::Secp256k1KeyPair;
    use fastcrypto::traits::KeyPair;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use types::base::SomaAddress;
    use types::bridge::{BridgeChainId, PendingWithdrawal};
    use types::object::ObjectID;

    fn fresh_kp() -> Secp256k1KeyPair {
        let mut rng = StdRng::from_seed([7; 32]);
        Secp256k1KeyPair::generate(&mut rng)
    }

    fn handler_with(
        approved: Vec<BridgeAction>,
    ) -> (BridgeRequestHandler<MockSomaClient>, Arc<SomaBridgeClient<MockSomaClient>>) {
        let mock = MockSomaClient::new();
        let soma = Arc::new(SomaBridgeClient::new(mock, BridgeChainId::SomaCustom));
        let eth = Arc::new(EthClient::new_for_test(
            "0x0000000000000000000000000000000000000001".to_string(),
        ));
        let h = BridgeRequestHandler::new(fresh_kp(), eth, Arc::clone(&soma), approved).unwrap();
        (h, soma)
    }

    fn install_withdrawal(soma: &SomaBridgeClient<MockSomaClient>, pw: PendingWithdrawal) {
        soma.inner_for_test().pending_withdrawals.lock().unwrap().insert(pw.nonce, pw);
    }

    #[tokio::test]
    async fn test_governance_action_signs_when_approved() {
        let action = BridgeAction::EmergencyPause { nonce: 1 };
        let (h, _) = handler_with(vec![action.clone()]);
        let signed = h.handle_governance_action(action.clone()).await.unwrap();
        assert_eq!(signed.action, action);
        assert_eq!(signed.signer_pubkey.len(), 33);
        assert_eq!(signed.signature.len(), 65);
    }

    #[tokio::test]
    async fn test_governance_action_rejected_when_not_approved() {
        let approved = BridgeAction::EmergencyPause { nonce: 1 };
        let requested = BridgeAction::EmergencyPause { nonce: 2 };
        let (h, _) = handler_with(vec![approved]);
        assert!(matches!(
            h.handle_governance_action(requested).await,
            Err(BridgeError::GovernanceActionIsNotApproved),
        ));
    }

    #[tokio::test]
    async fn test_governance_handler_rejects_token_transfer() {
        let (h, _) = handler_with(vec![]);
        let deposit = BridgeAction::Deposit {
            nonce: 1,
            eth_tx_hash: [0; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::random(),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1,
            timestamp_ms: 0,
        };
        assert!(matches!(
            h.handle_governance_action(deposit).await,
            Err(BridgeError::ActionIsNotGovernanceAction),
        ));
    }

    #[tokio::test]
    async fn test_soma_withdrawal_signs() {
        let (h, soma) = handler_with(vec![]);
        install_withdrawal(
            &soma,
            PendingWithdrawal {
                id: ObjectID::random(),
                nonce: 42,
                sender: SomaAddress::random(),
                recipient_eth_address: [9; 20],
                amount: 1_000_000,
                created_at_ms: 1_700_000_000_000,
                target_chain: types::bridge::BridgeChainId::EthCustom,
                verified_signatures: None,
            },
        );
        let signed = h.handle_soma_withdrawal(42).await.unwrap();
        match signed.action {
            BridgeAction::Withdrawal { nonce, amount, .. } => {
                assert_eq!(nonce, 42);
                assert_eq!(amount, 1_000_000);
            }
            _ => panic!("expected Withdrawal"),
        }
    }

    #[tokio::test]
    async fn test_soma_withdrawal_missing_returns_error() {
        let (h, _) = handler_with(vec![]);
        assert!(matches!(
            h.handle_soma_withdrawal(999).await,
            Err(BridgeError::NoBridgeEventsInTxPosition),
        ));
    }

    #[tokio::test]
    async fn test_soma_withdrawal_already_completed_rejected() {
        let (h, soma) = handler_with(vec![]);
        install_withdrawal(
            &soma,
            PendingWithdrawal {
                id: ObjectID::random(),
                nonce: 7,
                sender: SomaAddress::random(),
                recipient_eth_address: [9; 20],
                amount: 1_000_000,
                created_at_ms: 0,
                target_chain: types::bridge::BridgeChainId::EthCustom,
                verified_signatures: Some(types::bridge::WithdrawalCertificate {
                    signatures: Default::default(),
                    attached_at_epoch: 0,
                }),
            },
        );
        assert!(matches!(
            h.handle_soma_withdrawal(7).await,
            Err(BridgeError::InvalidBridgeClientRequest(_)),
        ));
    }

    #[test]
    fn test_parse_tx_hash_accepts_with_and_without_prefix() {
        let raw = [42u8; 32];
        let hex_with = format!("0x{}", hex::encode(raw));
        let hex_without = hex::encode(raw);
        assert_eq!(parse_tx_hash(&hex_with).unwrap(), raw);
        assert_eq!(parse_tx_hash(&hex_without).unwrap(), raw);
    }

    #[test]
    fn test_parse_tx_hash_rejects_wrong_length() {
        assert!(
            matches!(parse_tx_hash("0xabcd"), Err(BridgeError::InvalidBridgeClientRequest(_)),)
        );
    }
}
