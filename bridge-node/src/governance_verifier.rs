//! Operator-approved governance action whitelist.
//!
//! Mirrors Sui's `sui-bridge/src/server/governance_verifier.rs`. When the
//! bridge server receives a governance-action sig request (pause, blocklist,
//! limit update, EVM upgrade, …), it must look the action up in this
//! whitelist before producing a signature. The whitelist is populated from
//! `BridgeNodeConfig.approved_governance_actions` at startup; an operator
//! must edit the config and restart (or hot-reload, future work) to authorize
//! a new governance action.
//!
//! The keying is by [`crate::types::BridgeAction::digest`] — the same
//! canonical Keccak256 hash used by the on-chain executor — so an
//! approved action and an incoming request must be byte-identical
//! (down to nonce, chain id, blocklist members in canonical order, etc.)
//! to match.
//!
//! ## Security rationale (Sui parity)
//!
//! Sui's threat model: a rogue committee member could DM a peer's bridge
//! node a request like "sign pause action with nonce 99 on chain X" and
//! the peer's node would happily produce a sig (it has the private key).
//! Once the rogue collected enough sigs, the bridge could be paused
//! without operator consent. The whitelist prevents that — peers only
//! sign actions the operator pre-approved.
//!
//! Token transfers don't need this whitelist: their authority comes from
//! the underlying chain event, which the server verifies independently
//! via `EthClient::get_finalized_bridge_action_maybe` /
//! `SomaBridgeClient::get_pending_withdrawal`.

use std::collections::HashMap;

use crate::error::{BridgeError, BridgeResult};
use crate::storage::BridgeActionDigest;
use crate::types::BridgeAction;

/// Operator-approved governance action whitelist.
#[derive(Debug, Clone, Default)]
pub struct GovernanceVerifier {
    approved: HashMap<BridgeActionDigest, BridgeAction>,
}

impl GovernanceVerifier {
    /// Build a verifier from the operator-supplied list. Errors if any
    /// item is not actually a governance action (token transfers in
    /// the whitelist would be a configuration mistake — they have no
    /// business being pre-approved this way).
    pub fn new(approved_actions: Vec<BridgeAction>) -> BridgeResult<Self> {
        let mut approved = HashMap::new();
        for action in approved_actions {
            if !action.is_governance_action() {
                return Err(BridgeError::ActionIsNotGovernanceAction);
            }
            approved.insert(action.digest(), action);
        }
        Ok(Self { approved })
    }

    /// Echo the action back if it's in the whitelist; error otherwise.
    pub fn verify(&self, action: BridgeAction) -> BridgeResult<BridgeAction> {
        if !action.is_governance_action() {
            return Err(BridgeError::ActionIsNotGovernanceAction);
        }
        if self.approved.contains_key(&action.digest()) {
            Ok(action)
        } else {
            Err(BridgeError::GovernanceActionIsNotApproved)
        }
    }

    /// Test/inspection helper.
    pub fn approved_count(&self) -> usize {
        self.approved.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::base::SomaAddress;
    use types::bridge::{BridgeChainId, SOMA_BRIDGE_CHAIN_ID};

    fn pause(nonce: u64) -> BridgeAction {
        BridgeAction::EmergencyPause { nonce }
    }

    fn deposit(nonce: u64) -> BridgeAction {
        BridgeAction::Deposit {
            nonce,
            eth_tx_hash: [0; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::random(),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1,
            timestamp_ms: 0,
        }
    }

    fn limit_update() -> BridgeAction {
        BridgeAction::LimitUpdate {
            nonce: 1,
            chain_id: BridgeChainId::EthCustom,
            sending_chain_id: SOMA_BRIDGE_CHAIN_ID,
            new_usd_limit: 100_000,
        }
    }

    #[test]
    fn test_approved_action_passes() {
        let p = pause(7);
        let v = GovernanceVerifier::new(vec![p.clone()]).unwrap();
        assert_eq!(v.verify(p.clone()).unwrap(), p);
    }

    #[test]
    fn test_unapproved_action_rejected() {
        let v = GovernanceVerifier::new(vec![pause(7)]).unwrap();
        // Different nonce — different digest — not in whitelist.
        assert!(matches!(
            v.verify(pause(8)),
            Err(BridgeError::GovernanceActionIsNotApproved),
        ));
    }

    #[test]
    fn test_token_transfer_rejected_in_constructor() {
        // Putting a Deposit in the whitelist is a config mistake.
        assert!(matches!(
            GovernanceVerifier::new(vec![pause(1), deposit(1)]),
            Err(BridgeError::ActionIsNotGovernanceAction),
        ));
    }

    #[test]
    fn test_token_transfer_rejected_in_verify() {
        let v = GovernanceVerifier::new(vec![pause(7)]).unwrap();
        // Even with a non-empty whitelist, a Deposit can never be
        // approved via this path — token transfers are server-verified
        // via the chain (eth_client::get_finalized_bridge_action_maybe).
        assert!(matches!(
            v.verify(deposit(1)),
            Err(BridgeError::ActionIsNotGovernanceAction),
        ));
    }

    #[test]
    fn test_multiple_approved_actions() {
        let v = GovernanceVerifier::new(vec![pause(1), pause(2), limit_update()]).unwrap();
        assert_eq!(v.approved_count(), 3);
        assert!(v.verify(pause(1)).is_ok());
        assert!(v.verify(pause(2)).is_ok());
        assert!(v.verify(limit_update()).is_ok());
        assert!(v.verify(pause(3)).is_err());
    }

    #[test]
    fn test_empty_whitelist_rejects_everything() {
        let v = GovernanceVerifier::default();
        assert!(matches!(
            v.verify(pause(1)),
            Err(BridgeError::GovernanceActionIsNotApproved),
        ));
    }
}
