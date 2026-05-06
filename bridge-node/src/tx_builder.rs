//! Build signed Soma transactions from a [`CertifiedBridgeAction`].
//!
//! Mirrors Sui's `sui-bridge/src/sui_transaction_builder.rs`: pure
//! function — given a quorum-signed bridge action and the relayer's
//! identity, produce a fully-signed `Transaction` ready to submit to
//! Soma's `TransactionExecutionService`.
//!
//! Soma's tx model is dramatically simpler than Sui's here. There's no
//! `bridge_object_arg` to look up, no token-id → `TypeTag` map to
//! resolve, no reference gas price to fetch, and no gas object to
//! select — the relayer's USDC accumulator pays for the tx in
//! "accumulator mode," so `gas_payment` is `vec![]`. The only thing
//! this function does is map each [`BridgeAction`] variant to the
//! corresponding [`TransactionKind`], stuff in the cert's signatures,
//! and sign with the relayer's keypair.

use fastcrypto::traits::Signer;
use types::base::SomaAddress;
use types::crypto::Signature;
use types::transaction::{
    BridgeAttachWithdrawalSignaturesArgs, BridgeDepositArgs, BridgeEmergencyPauseArgs,
    BridgeEmergencyUnpauseArgs, BridgeUpdateCommitteeBlocklistArgs, Transaction,
    TransactionData, TransactionKind,
};

use crate::aggregator::CertifiedBridgeAction;
use crate::error::{BridgeError, BridgeResult};
use crate::types::BridgeAction;

/// Build and sign a Soma `Transaction` carrying `cert.action` plus the
/// quorum cert. The returned `Transaction` is ready for submission via
/// `SomaBridgeClient::execute_transaction`.
///
/// The signing key here is the relayer's Soma key — orthogonal to the
/// validator's bridge-committee secp256k1 key (the latter signs bridge
/// messages whose ecrecovered pubkeys are already inside `cert.signatures`).
/// The relayer key only authorizes the wrapper user-tx; it doesn't
/// participate in bridge consensus.
pub fn build_bridge_transaction(
    relayer_address: SomaAddress,
    relayer_signer: &dyn Signer<Signature>,
    cert: &CertifiedBridgeAction,
) -> BridgeResult<Transaction> {
    let kind = action_to_transaction_kind(cert)?;
    // Accumulator-mode pricing: USDC is debited from `relayer_address`
    // at execution; no explicit gas object is selected.
    let data = TransactionData::new(kind, relayer_address, vec![]);
    Ok(Transaction::from_data_and_signer(data, vec![relayer_signer]))
}

/// Map a quorum-signed `BridgeAction` to the matching [`TransactionKind`].
/// The cert's `signatures` envelope is moved into the args struct of the
/// chosen variant; the on-chain executor re-verifies it against the
/// canonical message bytes derived from the args.
fn action_to_transaction_kind(
    cert: &CertifiedBridgeAction,
) -> BridgeResult<TransactionKind> {
    let signatures = cert.signatures.clone();
    let kind = match &cert.action {
        BridgeAction::Deposit {
            nonce,
            eth_tx_hash,
            // eth_event_idx is a routing identifier — peers fetch the
            // receipt log to re-verify; on-chain the deposit is keyed
            // by (nonce, eth_tx_hash) and event_idx isn't in the
            // signed payload.
            eth_event_idx: _,
            // sender_eth_address / target_chain / token_type are
            // Sui-parity wire-format fields. The on-chain Soma
            // executor reconstructs the canonical signed bytes from
            // its own state (the on-chain `PendingDeposit` carries
            // these), so we drop them from the tx args.
            sender_eth_address: _,
            target_chain: _,
            token_type: _,
            recipient,
            amount,
            timestamp_ms,
        } => TransactionKind::BridgeDeposit(BridgeDepositArgs {
            nonce: *nonce,
            eth_tx_hash: *eth_tx_hash,
            recipient: *recipient,
            amount: *amount,
            timestamp_ms: *timestamp_ms,
            signatures,
        }),

        BridgeAction::Withdrawal { nonce, .. } => {
            // Outbound: cert authorizes attaching signatures to an
            // existing on-chain `PendingWithdrawal`. The on-chain
            // executor reconstructs the canonical message bytes from
            // the object's fields, NOT from the args — anything
            // smuggled here would be ignored.
            TransactionKind::BridgeAttachWithdrawalSignatures(
                BridgeAttachWithdrawalSignaturesArgs { nonce: *nonce, signatures },
            )
        }

        BridgeAction::EmergencyPause { nonce } => {
            TransactionKind::BridgeEmergencyPause(BridgeEmergencyPauseArgs {
                nonce: *nonce,
                signatures,
            })
        }

        BridgeAction::EmergencyUnpause { nonce } => {
            TransactionKind::BridgeEmergencyUnpause(BridgeEmergencyUnpauseArgs {
                nonce: *nonce,
                signatures,
            })
        }

        BridgeAction::UpdateCommitteeBlocklist {
            nonce,
            blocklist_type,
            members,
            ..
        } => {
            // The on-chain `BridgeUpdateCommitteeBlocklist` executor
            // operates on derived 20-byte Eth addresses, not raw pubkeys
            // (mirrors Sui's blocklist payload encoding). Convert here so
            // the executor can compare directly against
            // `derive_eth_address(member.pubkey)`.
            let eth_addresses: Vec<[u8; 20]> =
                members.iter().map(types::bridge::derive_eth_address).collect();
            let is_blocklist =
                matches!(blocklist_type, types::bridge::BlocklistType::Blocklist);
            TransactionKind::BridgeUpdateCommitteeBlocklist(
                BridgeUpdateCommitteeBlocklistArgs {
                    nonce: *nonce,
                    is_blocklist,
                    eth_addresses,
                    signatures,
                },
            )
        }

        // The remaining action variants target the Eth side (the off-chain
        // bridge node submits their certs to the EVM contract, not to
        // Soma). They have no Soma-side `TransactionKind`, so building a
        // Soma tx for them is a programmer error and we surface it as
        // such instead of silently constructing nothing.
        BridgeAction::CommitteeUpdate { .. }
        | BridgeAction::LimitUpdate { .. }
        | BridgeAction::EvmContractUpgrade { .. } => {
            return Err(BridgeError::Internal(format!(
                "BridgeAction targets Eth side, no Soma TransactionKind: {:?}",
                cert.action.message_type()
            )));
        }
    };
    Ok(kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregator::CertifiedBridgeAction;
    use std::collections::BTreeMap;
    use types::bridge::{
        BlocklistType, BridgeChainId, BridgePubkey, BridgeSignature, SOMA_BRIDGE_CHAIN_ID,
        generate_test_bridge_committee,
    };
    use types::crypto::{SomaKeyPair, get_key_pair};
    use types::base::SomaAddress;
    use types::crypto::Ed25519SomaSignature;

    /// Build a relayer keypair + address using the same helper that
    /// produces the test cluster's accounts.
    fn relayer() -> (SomaAddress, SomaKeyPair) {
        let (addr, kp): (_, fastcrypto::ed25519::Ed25519KeyPair) = get_key_pair();
        (addr, SomaKeyPair::Ed25519(kp))
    }

    fn empty_sigs() -> BTreeMap<BridgePubkey, BridgeSignature> {
        BTreeMap::new()
    }

    #[test]
    fn test_build_deposit_transaction() {
        let (sender, kp) = relayer();
        let action = BridgeAction::Deposit {
            nonce: 7,
            eth_tx_hash: [0xAB; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::random(),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1_000_000,
            timestamp_ms: 0,
        };
        let cert = CertifiedBridgeAction { action, signatures: empty_sigs() };
        let tx = build_bridge_transaction(sender, &kp, &cert).expect("tx built");
        // Should be a BridgeDeposit kind.
        let inner = &tx.inner().intent_message.value;
        assert!(matches!(inner.kind(), TransactionKind::BridgeDeposit(_)));
        assert_eq!(inner.sender(), sender);
    }

    #[test]
    fn test_build_withdrawal_attach_transaction() {
        let (sender, kp) = relayer();
        let action = BridgeAction::Withdrawal {
            nonce: 3,
            sender: SomaAddress::random(),
            target_chain: types::bridge::BridgeChainId::EthCustom,
            recipient_eth_address: [0xCC; 20],
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1_000,
            timestamp_ms: 0,
        };
        let cert = CertifiedBridgeAction { action, signatures: empty_sigs() };
        let tx = build_bridge_transaction(sender, &kp, &cert).expect("tx built");
        assert!(matches!(
            tx.inner().intent_message.value.kind(),
            TransactionKind::BridgeAttachWithdrawalSignatures(_)
        ));
    }

    #[test]
    fn test_build_emergency_pause_transaction() {
        let (sender, kp) = relayer();
        let cert = CertifiedBridgeAction {
            action: BridgeAction::EmergencyPause { nonce: 5 },
            signatures: empty_sigs(),
        };
        let tx = build_bridge_transaction(sender, &kp, &cert).unwrap();
        assert!(matches!(
            tx.inner().intent_message.value.kind(),
            TransactionKind::BridgeEmergencyPause(_)
        ));
    }

    #[test]
    fn test_build_emergency_unpause_transaction() {
        let (sender, kp) = relayer();
        let cert = CertifiedBridgeAction {
            action: BridgeAction::EmergencyUnpause { nonce: 5 },
            signatures: empty_sigs(),
        };
        let tx = build_bridge_transaction(sender, &kp, &cert).unwrap();
        assert!(matches!(
            tx.inner().intent_message.value.kind(),
            TransactionKind::BridgeEmergencyUnpause(_)
        ));
    }

    /// Blocklist actions carry typed `BridgePubkey` members that the
    /// builder must convert to derived 20-byte Eth addresses for the
    /// on-chain executor. Round-trip the conversion.
    #[test]
    fn test_build_blocklist_transaction_derives_eth_addresses() {
        let (sender, kp) = relayer();
        let (_committee, keypairs) = generate_test_bridge_committee(2);
        let pk0 = BridgePubkey::from_keypair(&keypairs[0]);
        let pk1 = BridgePubkey::from_keypair(&keypairs[1]);

        let action = BridgeAction::UpdateCommitteeBlocklist {
            nonce: 1,
            chain_id: SOMA_BRIDGE_CHAIN_ID,
            blocklist_type: BlocklistType::Blocklist,
            members: vec![pk0.clone(), pk1.clone()],
        };
        let cert = CertifiedBridgeAction { action, signatures: empty_sigs() };
        let tx = build_bridge_transaction(sender, &kp, &cert).unwrap();
        let kind = tx.inner().intent_message.value.kind();
        let TransactionKind::BridgeUpdateCommitteeBlocklist(args) = kind else {
            panic!("wrong tx kind");
        };
        assert_eq!(args.is_blocklist, true);
        assert_eq!(args.eth_addresses.len(), 2);
        assert_eq!(args.eth_addresses[0], types::bridge::derive_eth_address(&pk0));
        assert_eq!(args.eth_addresses[1], types::bridge::derive_eth_address(&pk1));
    }

    /// Eth-targeted actions (LimitUpdate, EvmContractUpgrade,
    /// CommitteeUpdate) have no Soma-side TransactionKind. Trying to
    /// build a Soma tx for them must error rather than silently produce
    /// garbage.
    #[test]
    fn test_eth_targeted_actions_reject() {
        let (sender, kp) = relayer();
        let limit = CertifiedBridgeAction {
            action: BridgeAction::LimitUpdate {
                nonce: 1,
                chain_id: BridgeChainId::EthCustom,
                sending_chain_id: SOMA_BRIDGE_CHAIN_ID,
                new_usd_limit: 1_000_000,
            },
            signatures: empty_sigs(),
        };
        assert!(build_bridge_transaction(sender, &kp, &limit).is_err());

        let upgrade = CertifiedBridgeAction {
            action: BridgeAction::EvmContractUpgrade {
                nonce: 1,
                chain_id: BridgeChainId::EthCustom,
                proxy: [0x06; 20],
                new_impl: [0x09; 20],
                call_data: vec![],
            },
            signatures: empty_sigs(),
        };
        assert!(build_bridge_transaction(sender, &kp, &upgrade).is_err());

        let committee_update = CertifiedBridgeAction {
            action: BridgeAction::CommitteeUpdate {
                nonce: 1,
                new_members: vec![],
            },
            signatures: empty_sigs(),
        };
        assert!(build_bridge_transaction(sender, &kp, &committee_update).is_err());
    }

    /// `Ed25519SomaSignature` import is here only to make the test file
    /// compile if/when we extend coverage to alternate signer types.
    #[allow(dead_code)]
    fn _ed25519_sig_marker() -> Option<Ed25519SomaSignature> { None }
}
