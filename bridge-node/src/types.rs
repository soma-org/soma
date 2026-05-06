use serde::{Deserialize, Serialize};
use types::base::SomaAddress;
use types::bridge::{
    BRIDGE_MESSAGE_VERSION, BlocklistType, BridgeCommittee, BridgeMessageType, EmergencyOpCode,
    SOMA_BRIDGE_CHAIN_ID, derive_eth_address, encode_blocklist_payload, encode_bridge_message,
    encode_deposit_payload, encode_emergency_payload, encode_evm_contract_upgrade_payload,
    encode_limit_update_payload, encode_withdraw_payload,
};
use types::object::ObjectID;

/// A bridge action that needs committee signatures before it can be executed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BridgeAction {
    /// USDC deposit from Ethereum → Soma. V2 token transfer carries
    /// `timestamp_ms` in the signed payload (Eth-side block time).
    ///
    /// Wire-format fields (Sui parity — see
    /// [`types::bridge::encode_deposit_payload`]): the signed payload
    /// includes the Eth `sender` (who locked USDC on Eth), the
    /// `target_chain` (which Soma chain), and the `token_type` (always
    /// USDC today). Soma's Eth-side contract asserts the address
    /// lengths + token type match the expected invariants before
    /// minting on the target.
    ///
    /// `eth_tx_hash` and `eth_event_idx` are NOT part of the signed
    /// payload — they're routing identifiers so peers can re-fetch and
    /// re-verify the same log via
    /// `GET /sign/bridge_tx/eth/soma/{tx_hash}/{event_idx}`. The signed
    /// message bytes (and therefore the on-chain digest) are invariant
    /// under changes to those fields — see [`Self::to_message_bytes`].
    Deposit {
        nonce: u64,
        eth_tx_hash: [u8; 32],
        eth_event_idx: u16,
        /// Eth address that locked USDC. In the signed V2 payload at
        /// the `senderAddress` slot (length byte = 20).
        sender_eth_address: [u8; 20],
        /// Destination chain — always a Soma chain (`SomaMainnet`,
        /// `SomaTestnet`, `SomaCustom`). In the signed payload at
        /// `targetChain`.
        target_chain: types::bridge::BridgeChainId,
        /// Soma recipient — in the signed payload at `targetAddress`
        /// (length byte = 32).
        recipient: SomaAddress,
        /// Wire-format token id. Always [`types::bridge::USDC_TOKEN_TYPE`]
        /// today; carried so a future multi-token expansion doesn't
        /// require a wire-format version bump.
        token_type: u8,
        amount: u64,
        timestamp_ms: u64,
    },
    /// USDC withdrawal from Soma → Ethereum. Observed from Soma checkpoints
    /// via PendingWithdrawal objects. V2 token transfer with `timestamp_ms`
    /// taken from the Soma-side `created_at_ms` of the PendingWithdrawal.
    /// Mirrors [`Self::Deposit`]'s wire-format fields with sender/target
    /// roles flipped (Soma sender → Eth recipient).
    Withdrawal {
        nonce: u64,
        /// Soma sender — in the signed V2 payload at `senderAddress`
        /// (length byte = 32).
        sender: SomaAddress,
        /// Destination chain — always an Eth chain (`EthMainnet`,
        /// `EthSepolia`, `EthCustom`). In the signed payload at
        /// `targetChain`.
        target_chain: types::bridge::BridgeChainId,
        /// Eth recipient — in the signed payload at `targetAddress`
        /// (length byte = 20).
        recipient_eth_address: [u8; 20],
        /// Wire-format token id. Always [`types::bridge::USDC_TOKEN_TYPE`]
        /// today.
        token_type: u8,
        amount: u64,
        timestamp_ms: u64,
    },
    /// Emergency pause — stops all bridge operations. Carries a real
    /// per-message-type nonce (Stage 3) so the on-chain executor's
    /// `expected_system_message_seq(EmergencyOp)` check matches.
    EmergencyPause { nonce: u64 },
    /// Emergency unpause — resumes bridge operations. Shares the EmergencyOp
    /// counter with pause.
    EmergencyUnpause { nonce: u64 },
    /// Committee update — sync Ethereum contract with new validator set at epoch boundary.
    /// Carries a per-message-type nonce so a quorum-signed cert can't be
    /// replayed against the Eth contract after a future committee rotation.
    /// (Soma's on-chain rotation happens implicitly at epoch boundary —
    /// Stage 5 — so this action exists purely for Eth-side sync today.)
    CommitteeUpdate {
        nonce: u64,
        new_members: Vec<(types::bridge::BridgePubkey, u64)>,
    },
    /// Surgically blocklist or unblocklist individual committee members without
    /// rotating the whole committee. Members are carried as typed
    /// [`types::bridge::BridgePubkey`]s; encoding derives 20-byte Eth
    /// addresses for the wire format so the Solidity-side contract can
    /// match against `ecrecover` outputs.
    UpdateCommitteeBlocklist {
        nonce: u64,
        chain_id: types::bridge::BridgeChainId,
        blocklist_type: BlocklistType,
        members: Vec<types::bridge::BridgePubkey>,
    },
    /// Update the per-route USD/day transfer limit. `chain_id` is the receiving
    /// chain (in the message header); `sending_chain_id` (in the payload)
    /// specifies the source. `new_usd_limit` uses [`types::bridge::USD_MULTIPLIER`]
    /// (4 decimal places: $1 = 10000).
    LimitUpdate {
        nonce: u64,
        chain_id: types::bridge::BridgeChainId,
        sending_chain_id: types::bridge::BridgeChainId,
        new_usd_limit: u64,
    },
    /// Ship an upgrade to the Ethereum-side bridge proxy. `chain_id` should be
    /// an Eth chain id; `call_data` is forwarded to the proxy's `upgradeToAndCall`.
    EvmContractUpgrade {
        nonce: u64,
        chain_id: types::bridge::BridgeChainId,
        proxy: [u8; 20],
        new_impl: [u8; 20],
        call_data: Vec<u8>,
    },
}

impl BridgeAction {
    /// Returns the bridge message type for this action.
    pub fn message_type(&self) -> BridgeMessageType {
        match self {
            BridgeAction::Deposit { .. } => BridgeMessageType::UsdcDeposit,
            BridgeAction::Withdrawal { .. } => BridgeMessageType::UsdcWithdraw,
            BridgeAction::EmergencyPause { .. } | BridgeAction::EmergencyUnpause { .. } => {
                BridgeMessageType::EmergencyOp
            }
            BridgeAction::CommitteeUpdate { .. } => BridgeMessageType::CommitteeUpdate,
            BridgeAction::UpdateCommitteeBlocklist { .. } => {
                BridgeMessageType::UpdateCommitteeBlocklist
            }
            BridgeAction::LimitUpdate { .. } => BridgeMessageType::LimitUpdate,
            BridgeAction::EvmContractUpgrade { .. } => BridgeMessageType::EvmContractUpgrade,
        }
    }

    /// Returns `true` for governance-class actions (pause/unpause,
    /// blocklist updates, limit updates, EVM contract upgrades,
    /// committee updates) and `false` for token transfers.
    ///
    /// Used by the bridge server's `GovernanceVerifier` to reject
    /// token-transfer sigs from governance endpoints and vice versa.
    /// Mirrors Sui's `BridgeAction::is_governance_action()`.
    pub fn is_governance_action(&self) -> bool {
        match self {
            BridgeAction::Deposit { .. } | BridgeAction::Withdrawal { .. } => false,
            BridgeAction::EmergencyPause { .. }
            | BridgeAction::EmergencyUnpause { .. }
            | BridgeAction::CommitteeUpdate { .. }
            | BridgeAction::UpdateCommitteeBlocklist { .. }
            | BridgeAction::LimitUpdate { .. }
            | BridgeAction::EvmContractUpgrade { .. } => true,
        }
    }

    /// Returns the nonce for this action.
    ///
    /// `EmergencyPause`/`EmergencyUnpause` carry the per-message-type seq
    /// num (Stage 3 — pause and unpause share the EmergencyOp counter,
    /// mirroring Sui's `execute_system_message`). `CommitteeUpdate` carries
    /// its own counter for Eth-side sync replay defense (L9).
    pub fn nonce(&self) -> u64 {
        match self {
            BridgeAction::Deposit { nonce, .. }
            | BridgeAction::Withdrawal { nonce, .. }
            | BridgeAction::EmergencyPause { nonce }
            | BridgeAction::EmergencyUnpause { nonce }
            | BridgeAction::CommitteeUpdate { nonce, .. }
            | BridgeAction::UpdateCommitteeBlocklist { nonce, .. }
            | BridgeAction::LimitUpdate { nonce, .. }
            | BridgeAction::EvmContractUpgrade { nonce, .. } => *nonce,
        }
    }

    /// Returns the chain id this action targets — i.e. the chain that will
    /// verify and execute the message. Token-transfer / emergency / committee-update
    /// actions are scoped to Soma; new governance actions carry their own chain id
    /// so an EVM contract upgrade can target Ethereum without colliding with
    /// Soma-targeted messages.
    pub fn chain_id(&self) -> types::bridge::BridgeChainId {
        match self {
            BridgeAction::Deposit { .. }
            | BridgeAction::Withdrawal { .. }
            | BridgeAction::EmergencyPause { .. }
            | BridgeAction::EmergencyUnpause { .. }
            | BridgeAction::CommitteeUpdate { .. } => SOMA_BRIDGE_CHAIN_ID,
            BridgeAction::UpdateCommitteeBlocklist { chain_id, .. }
            | BridgeAction::LimitUpdate { chain_id, .. }
            | BridgeAction::EvmContractUpgrade { chain_id, .. } => *chain_id,
        }
    }

    /// Approval threshold (BPS) required for this action under `committee`.
    pub fn approval_threshold(&self, committee: &BridgeCommittee) -> u64 {
        match self {
            BridgeAction::Deposit { .. } => committee.threshold_deposit,
            BridgeAction::Withdrawal { .. } => committee.threshold_withdraw,
            BridgeAction::EmergencyPause { .. } => committee.threshold_pause,
            BridgeAction::EmergencyUnpause { .. } => committee.threshold_unpause,
            BridgeAction::CommitteeUpdate { .. } => committee.threshold_unpause,
            BridgeAction::UpdateCommitteeBlocklist { .. } => committee.threshold_blocklist,
            BridgeAction::LimitUpdate { .. } => committee.threshold_limit_update,
            BridgeAction::EvmContractUpgrade { .. } => committee.threshold_evm_upgrade,
        }
    }

    /// Encode this action into the canonical bridge message bytes for signing.
    /// Format: PREFIX || type(1) || version(1) || nonce(8,BE) || chainID(8,BE) || payload
    ///
    /// Panics if [`BridgeAction::UpdateCommitteeBlocklist`] carries a pubkey
    /// that fails secp256k1 decompression. Pubkeys in the committee are
    /// validated at insertion time, so this is a programmer error rather than
    /// a runtime condition.
    pub fn to_message_bytes(&self) -> Vec<u8> {
        let payload = self.encode_payload();
        encode_bridge_message(
            self.message_type(),
            self.message_version(),
            self.nonce(),
            self.chain_id(),
            &payload,
        )
    }

    /// Wire-format version for this action's message. Token transfers
    /// (deposit / withdraw) use V2 (with timestamp_ms in payload);
    /// system messages use V1.
    pub fn message_version(&self) -> u8 {
        match self {
            BridgeAction::Deposit { .. } | BridgeAction::Withdrawal { .. } => {
                types::bridge::TOKEN_TRANSFER_MESSAGE_VERSION_V2
            }
            _ => types::bridge::BRIDGE_MESSAGE_VERSION,
        }
    }

    fn encode_payload(&self) -> Vec<u8> {
        match self {
            BridgeAction::Deposit {
                sender_eth_address,
                target_chain,
                recipient,
                token_type,
                amount,
                timestamp_ms,
                ..
            } => encode_deposit_payload(
                sender_eth_address,
                *target_chain,
                recipient,
                *token_type,
                *amount,
                *timestamp_ms,
            ),
            BridgeAction::Withdrawal {
                sender,
                target_chain,
                recipient_eth_address,
                token_type,
                amount,
                timestamp_ms,
                ..
            } => encode_withdraw_payload(
                sender,
                *target_chain,
                recipient_eth_address,
                *token_type,
                *amount,
                *timestamp_ms,
            ),
            BridgeAction::EmergencyPause { .. } => encode_emergency_payload(EmergencyOpCode::Freeze),
            BridgeAction::EmergencyUnpause { .. } => encode_emergency_payload(EmergencyOpCode::Unfreeze),
            BridgeAction::CommitteeUpdate { new_members, .. } => {
                // Encode: count(4,BE) || (pubkey(33) || voting_power(8,BE))*
                let mut payload = Vec::new();
                payload.extend_from_slice(&(new_members.len() as u32).to_be_bytes());
                for (pubkey, power) in new_members {
                    payload.extend_from_slice(pubkey.as_bytes());
                    payload.extend_from_slice(&power.to_be_bytes());
                }
                payload
            }
            BridgeAction::UpdateCommitteeBlocklist {
                blocklist_type,
                members,
                ..
            } => {
                let eth_addresses: Vec<[u8; 20]> =
                    members.iter().map(derive_eth_address).collect();
                encode_blocklist_payload(*blocklist_type, &eth_addresses)
            }
            BridgeAction::LimitUpdate {
                sending_chain_id,
                new_usd_limit,
                ..
            } => encode_limit_update_payload(*sending_chain_id, *new_usd_limit),
            BridgeAction::EvmContractUpgrade {
                proxy,
                new_impl,
                call_data,
                ..
            } => encode_evm_contract_upgrade_payload(proxy, new_impl, call_data),
        }
    }
}

/// A bridge action with a collected signature from one committee member.
/// The signer's identity is recovered on demand via ecrecover (Sui parity);
/// the signing pubkey is also kept here for cheap pubkey-keyed dedup
/// without re-running ecrecover at every site.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedBridgeAction {
    pub action: BridgeAction,
    /// 33-byte compressed secp256k1 pubkey of the signer.
    pub signer_pubkey: Vec<u8>,
    pub signature: Vec<u8>, // 65-byte recoverable ECDSA signature
}

/// A deposit event parsed from Ethereum logs.
///
/// Field set matches the on-chain Soma `TokensDeposited` event (Sui
/// parity): the bridge node syncer reads each of these fields from the
/// log so the signed payload it reconstructs is byte-identical to what
/// the contract would have produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepositEvent {
    pub nonce: u64,
    /// Eth address that locked USDC — `msg.sender` of the on-chain
    /// `deposit()` call. Goes into the V2 payload's `senderAddress`.
    pub eth_sender: [u8; 20],
    /// Destination chain id from the on-chain event. The bridge node
    /// trusts this from the log (the contract emits it from its own
    /// state) rather than inferring from `chainID`.
    pub destination_chain_id: types::bridge::BridgeChainId,
    /// Soma recipient — V2 payload's `targetAddress`.
    pub soma_recipient: [u8; 32],
    /// Wire-format token id (always [`types::bridge::USDC_TOKEN_TYPE`]
    /// today). Carried so a future multi-token deployment can grow
    /// without re-syncing every historical event.
    pub token_type: u8,
    pub amount: u64,
    pub tx_hash: [u8; 32],
    /// Log index within the Eth tx receipt. Routes peer sig requests
    /// back to this exact log via `/sign/bridge_tx/eth/soma/{tx_hash}/{event_idx}`.
    /// `u16` matches Sui's `EthLog::log_index_in_tx` width.
    pub event_idx: u16,
    pub block_number: u64,
    /// Eth-side block timestamp at which the deposit was emitted (V2).
    pub timestamp_ms: u64,
}

impl DepositEvent {
    /// Convert to a BridgeAction for signing.
    pub fn to_bridge_action(&self) -> BridgeAction {
        BridgeAction::Deposit {
            nonce: self.nonce,
            eth_tx_hash: self.tx_hash,
            eth_event_idx: self.event_idx,
            sender_eth_address: self.eth_sender,
            target_chain: self.destination_chain_id,
            recipient: SomaAddress::from(self.soma_recipient),
            token_type: self.token_type,
            amount: self.amount,
            timestamp_ms: self.timestamp_ms,
        }
    }
}

/// Summary of a pending withdrawal observed from Soma checkpoints.
#[derive(Debug, Clone)]
pub struct ObservedWithdrawal {
    pub id: ObjectID,
    pub nonce: u64,
    pub sender: SomaAddress,
    /// Destination chain — always an Eth chain
    /// (`EthMainnet`/`EthSepolia`/`EthCustom`). Read from the Soma-side
    /// `PendingWithdrawal` object so a future multi-Eth-chain
    /// deployment doesn't require a wire-format change.
    pub target_chain: types::bridge::BridgeChainId,
    pub recipient_eth_address: [u8; 20],
    /// Wire-format token id — always [`types::bridge::USDC_TOKEN_TYPE`]
    /// today.
    pub token_type: u8,
    pub amount: u64,
    /// Soma-side timestamp from `PendingWithdrawal.created_at_ms` (V2).
    pub timestamp_ms: u64,
}

impl ObservedWithdrawal {
    /// Convert to a BridgeAction for signing.
    pub fn to_bridge_action(&self) -> BridgeAction {
        BridgeAction::Withdrawal {
            nonce: self.nonce,
            sender: self.sender,
            target_chain: self.target_chain,
            recipient_eth_address: self.recipient_eth_address,
            token_type: self.token_type,
            amount: self.amount,
            timestamp_ms: self.timestamp_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastcrypto::hash::{HashFunction, Keccak256};
    use fastcrypto::secp256k1::recoverable::Secp256k1RecoverableSignature;
    use fastcrypto::traits::{KeyPair, RecoverableSignature, ToFromBytes};
    use types::bridge::{
        build_bridge_signatures, generate_test_bridge_committee, sign_bridge_message,
        TOKEN_TRANSFER_MESSAGE_VERSION_V2,
    };

    #[test]
    fn test_deposit_action_message_encoding() {
        let action = BridgeAction::Deposit {
            nonce: 42,
            eth_tx_hash: [0xAB; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0x01; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1_000_000, // 1 USDC
            timestamp_ms: 1_700_000_000_000,
        };

        let msg = action.to_message_bytes();

        // Verify prefix
        assert!(msg.starts_with(b"SOMA_BRIDGE_MESSAGE"));
        // Verify message type byte (UsdcDeposit = 0)
        assert_eq!(msg[19], 0);
        // Token transfers use V2 (with timestamp_ms suffix in payload).
        assert_eq!(msg[20], TOKEN_TRANSFER_MESSAGE_VERSION_V2);

        // Ensure deterministic: same action produces same bytes
        assert_eq!(msg, action.to_message_bytes());
    }

    #[test]
    fn test_withdrawal_action_message_encoding() {
        let action = BridgeAction::Withdrawal {
            nonce: 7,
            sender: SomaAddress::from([0x02; 32]),
            target_chain: types::bridge::BridgeChainId::EthCustom,
            recipient_eth_address: [0xCC; 20],
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 5_000_000,
            timestamp_ms: 1_700_000_000_000,
        };

        let msg = action.to_message_bytes();
        assert!(msg.starts_with(b"SOMA_BRIDGE_MESSAGE"));
        assert_eq!(msg[19], 1); // UsdcWithdraw = 1
        assert_eq!(msg[20], TOKEN_TRANSFER_MESSAGE_VERSION_V2);
    }

    #[test]
    fn test_emergency_actions_encoding() {
        let pause = BridgeAction::EmergencyPause { nonce: 0 };
        let unpause = BridgeAction::EmergencyUnpause { nonce: 0 };

        let pause_msg = pause.to_message_bytes();
        let unpause_msg = unpause.to_message_bytes();

        // Both are EmergencyOp type
        assert_eq!(pause_msg[19], 2);
        assert_eq!(unpause_msg[19], 2);
        // But different payloads (freeze=0 vs unfreeze=1)
        assert_ne!(pause_msg, unpause_msg);
    }

    #[test]
    fn test_deposit_event_to_action() {
        let event = DepositEvent {
            nonce: 1,
            eth_sender: [0xAA; 20],
            destination_chain_id: types::bridge::BridgeChainId::SomaCustom,
            soma_recipient: [0xBB; 32],
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 100_000,
            tx_hash: [0xCC; 32],
            event_idx: 0,
            block_number: 12345,
            timestamp_ms: 1_700_000_000_000,
        };

        let action = event.to_bridge_action();
        assert_eq!(action.nonce(), 1);
        assert!(matches!(action.message_type(), BridgeMessageType::UsdcDeposit));
    }

    #[test]
    fn test_action_message_is_signable() {
        // Generate a real keypair and sign the message bytes
        let (_committee, keypairs) = generate_test_bridge_committee(4);
        let action = BridgeAction::Deposit {
            nonce: 1,
            eth_tx_hash: [0; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0x01; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1_000_000,
            timestamp_ms: 0,
        };

        let msg_bytes = action.to_message_bytes();
        // Should be signable without panic
        let sig = sign_bridge_message(&keypairs[0], &msg_bytes);
        assert_eq!(sig.as_ref().len(), 65); // recoverable signature
    }

    // -----------------------------------------------------------------------
    // Crypto cross-verification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sign_and_ecrecover_roundtrip() {
        // Sign a deposit message, ecrecover the pubkey, verify it matches.
        let (_committee, keypairs) = generate_test_bridge_committee(4);
        let action = BridgeAction::Deposit {
            nonce: 100,
            eth_tx_hash: [0xDE; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0x42; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 10_000_000, // 10 USDC
            timestamp_ms: 0,
        };

        let msg_bytes = action.to_message_bytes();
        let sig = sign_bridge_message(&keypairs[0], &msg_bytes);

        // ecrecover: hash with Keccak256, then recover public key from signature
        let recovered_pubkey = sig
            .recover_with_hash::<Keccak256>(&msg_bytes)
            .expect("ecrecover should succeed");

        assert_eq!(
            recovered_pubkey.as_bytes(),
            keypairs[0].public().as_bytes(),
            "ecrecovered pubkey must match signer's pubkey"
        );
    }

    #[test]
    fn test_ecrecover_wrong_message_fails() {
        // Sign message A, try ecrecover with message B — should recover a different key.
        let (_committee, keypairs) = generate_test_bridge_committee(1);

        let action_a = BridgeAction::Deposit {
            nonce: 1,
            eth_tx_hash: [0; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0x01; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 100,
            timestamp_ms: 0,
        };
        let action_b = BridgeAction::Deposit {
            nonce: 2, // different nonce
            eth_tx_hash: [0; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0x01; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 100,
            timestamp_ms: 0,
        };

        let sig = sign_bridge_message(&keypairs[0], &action_a.to_message_bytes());

        // Recover with wrong message should produce a different pubkey
        let recovered = sig
            .recover_with_hash::<Keccak256>(&action_b.to_message_bytes())
            .expect("ecrecover still produces *some* key");

        assert_ne!(
            recovered.as_bytes(),
            keypairs[0].public().as_bytes(),
            "wrong message should recover different pubkey"
        );
    }

    #[test]
    fn test_build_bridge_signatures_format() {
        // Verify each independent signature in the Vec<Vec<u8>> format
        // ecrecovers to the correct signer (Sui parity).
        let (_committee, keypairs) = generate_test_bridge_committee(4);
        let action = BridgeAction::Deposit {
            nonce: 1,
            eth_tx_hash: [0; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0x01; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1_000_000,
            timestamp_ms: 0,
        };

        let msg_bytes = action.to_message_bytes();

        // Sign with members 0 and 2.
        let signers: Vec<&fastcrypto::secp256k1::Secp256k1KeyPair> =
            vec![&keypairs[0], &keypairs[2]];
        let signatures = build_bridge_signatures(&signers, &msg_bytes);

        // Labeled-pubkey envelope: 2 entries × 65 bytes each.
        assert_eq!(signatures.len(), 2);
        assert!(signatures.values().all(|s| s.as_bytes().len() == 65));

        // Each signature should ecrecover to its labeled pubkey.
        for (claimed_pk, sig) in &signatures {
            let recoverable = Secp256k1RecoverableSignature::from_bytes(sig.as_slice())
                .expect("valid 65-byte recoverable sig");
            let recovered = recoverable
                .recover_with_hash::<Keccak256>(&msg_bytes)
                .expect("ecrecover should work");
            assert_eq!(
                recovered.as_bytes(),
                claimed_pk.as_bytes(),
                "signature must ecrecover to its labeled pubkey"
            );
        }
    }

    #[test]
    fn test_message_encoding_known_values() {
        // Fixed-input test vector for cross-verification with Solidity.
        // This test uses deterministic inputs so the expected output can be
        // hardcoded once verified, and then used as a regression test for
        // the Solidity encoder.
        let action = BridgeAction::Deposit {
            nonce: 1,
            eth_tx_hash: [0; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0u8; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1_000_000,
            timestamp_ms: 0x0102030405060708,
        };

        let msg_bytes = action.to_message_bytes();

        // V2 (Sui-parity) wire format:
        //   Header: PREFIX(19) || type(1) || version(1) || nonce(8) || chainID(1) = 30
        //   Payload (Eth→Soma): senderLen(1) || sender(20) || targetChain(1)
        //                       || targetLen(1) || target(32) || tokenType(1)
        //                       || amount(8 BE) || timestamp(8 BE) = 72
        //   Total: 102 bytes.
        assert_eq!(msg_bytes.len(), 30 + 72);

        // PREFIX
        assert_eq!(&msg_bytes[0..19], b"SOMA_BRIDGE_MESSAGE");
        // Type = UsdcDeposit = 0
        assert_eq!(msg_bytes[19], 0);
        // Version = 2 (V2 token transfer)
        assert_eq!(msg_bytes[20], TOKEN_TRANSFER_MESSAGE_VERSION_V2);
        // Nonce = 1 (big-endian u64)
        assert_eq!(&msg_bytes[21..29], &1u64.to_be_bytes());
        // ChainID = SOMA_BRIDGE_CHAIN_ID (the chain that will verify; for a
        // deposit message bound for the Soma side, this is the Soma chain).
        assert_eq!(msg_bytes[29], SOMA_BRIDGE_CHAIN_ID.as_u8());

        // ---- Payload begins at offset 30 ----
        // Sender address length = 20 (Eth)
        assert_eq!(msg_bytes[30], 20);
        // Sender address (20 zero bytes for this fixture)
        assert_eq!(&msg_bytes[31..51], &[0u8; 20]);
        // Target chain = SomaCustom = 2
        assert_eq!(msg_bytes[51], types::bridge::BridgeChainId::SomaCustom.as_u8());
        // Target address length = 32 (Soma)
        assert_eq!(msg_bytes[52], 32);
        // Target address (32 zero bytes for this fixture)
        assert_eq!(&msg_bytes[53..85], &[0u8; 32]);
        // Token type = USDC (3)
        assert_eq!(msg_bytes[85], types::bridge::USDC_TOKEN_TYPE);
        // Amount (BE u64)
        assert_eq!(&msg_bytes[86..94], &1_000_000u64.to_be_bytes());
        // Timestamp (BE u64)
        assert_eq!(&msg_bytes[94..102], &0x0102030405060708u64.to_be_bytes());

        // Determinism check
        let hash1 = Keccak256::digest(&msg_bytes);
        let hash2 = Keccak256::digest(&action.to_message_bytes());
        assert_eq!(hash1.digest, hash2.digest);
    }

    #[test]
    fn test_withdrawal_encoding_known_values() {
        let action = BridgeAction::Withdrawal {
            nonce: 42,
            sender: SomaAddress::from([0xFF; 32]),
            target_chain: types::bridge::BridgeChainId::EthCustom,
            recipient_eth_address: [0xAA; 20],
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 5_000_000,
            timestamp_ms: 0x1122334455667788,
        };

        let msg_bytes = action.to_message_bytes();

        // V2 (Sui-parity) wire format:
        //   Header: 30 bytes (same as deposit).
        //   Payload (Soma→Eth): senderLen(1) || sender(32) || targetChain(1)
        //                       || targetLen(1) || target(20) || tokenType(1)
        //                       || amount(8 BE) || timestamp(8 BE) = 72
        //   Total: 102 bytes.
        assert_eq!(msg_bytes.len(), 30 + 72);

        assert_eq!(msg_bytes[19], 1); // UsdcWithdraw
        assert_eq!(msg_bytes[20], TOKEN_TRANSFER_MESSAGE_VERSION_V2);
        assert_eq!(&msg_bytes[21..29], &42u64.to_be_bytes());
        // ChainID = SOMA_BRIDGE_CHAIN_ID (the chain that *originated* the
        // outbound withdrawal — i.e. Soma, the source).
        assert_eq!(msg_bytes[29], SOMA_BRIDGE_CHAIN_ID.as_u8());

        // ---- Payload begins at offset 30 ----
        // Sender length = 32 (Soma)
        assert_eq!(msg_bytes[30], 32);
        // Sender = 0xFF repeated 32 times
        assert_eq!(&msg_bytes[31..63], &[0xFFu8; 32]);
        // Target chain = EthCustom = 12
        assert_eq!(msg_bytes[63], types::bridge::BridgeChainId::EthCustom.as_u8());
        // Target length = 20 (Eth)
        assert_eq!(msg_bytes[64], 20);
        // Target (Eth recipient)
        assert_eq!(&msg_bytes[65..85], &[0xAA; 20]);
        // Token type = USDC
        assert_eq!(msg_bytes[85], types::bridge::USDC_TOKEN_TYPE);
        // Amount + timestamp
        assert_eq!(&msg_bytes[86..94], &5_000_000u64.to_be_bytes());
        assert_eq!(&msg_bytes[94..102], &0x1122334455667788u64.to_be_bytes());
    }

    #[test]
    fn test_emergency_encoding_known_values() {
        let pause = BridgeAction::EmergencyPause { nonce: 0 };
        let unpause = BridgeAction::EmergencyUnpause { nonce: 0 };

        let pause_msg = pause.to_message_bytes();
        let unpause_msg = unpause.to_message_bytes();

        // PREFIX(19) || type(1) || version(1) || nonce(8) || chainID(1) || payload(1) = 31 bytes
        assert_eq!(pause_msg.len(), 19 + 1 + 1 + 8 + 1 + 1);
        assert_eq!(unpause_msg.len(), 31);

        // Type = EmergencyOp = 2
        assert_eq!(pause_msg[19], 2);
        assert_eq!(unpause_msg[19], 2);

        // Nonce = 0 for emergency ops
        assert_eq!(&pause_msg[21..29], &0u64.to_be_bytes());

        // Payload: Freeze = 0, Unfreeze = 1
        assert_eq!(pause_msg[30], 0); // Freeze
        assert_eq!(unpause_msg[30], 1); // Unfreeze
    }

    #[test]
    fn test_committee_update_encoding() {
        // Use real keypairs so the typed BridgePubkey is valid.
        let (_committee, keypairs) = generate_test_bridge_committee(2);
        let pk0 = types::bridge::BridgePubkey::from_keypair(&keypairs[0]);
        let pk1 = types::bridge::BridgePubkey::from_keypair(&keypairs[1]);
        let action = BridgeAction::CommitteeUpdate {
            nonce: 42,
            new_members: vec![(pk0, 5000), (pk1, 5000)],
        };

        let msg_bytes = action.to_message_bytes();

        assert_eq!(msg_bytes[19], 3); // CommitteeUpdate
        // L9: CommitteeUpdate now carries a real nonce (was hardcoded 0
        // before this fix — replay risk for Eth-side committee sync).
        assert_eq!(&msg_bytes[21..29], &42u64.to_be_bytes());

        // Header now ends at byte 30 (1-byte chain id).
        // Payload: count(4) + (pubkey(33) + power(8)) * 2
        let payload_start = 30;
        let expected_payload_len = 4 + 2 * (33 + 8); // 86 bytes
        assert_eq!(msg_bytes.len(), payload_start + expected_payload_len);

        // Count = 2
        assert_eq!(
            &msg_bytes[payload_start..payload_start + 4],
            &2u32.to_be_bytes()
        );
    }

    #[test]
    fn test_all_action_types_ecrecover_roundtrip() {
        // Verify sign → ecrecover works for every action type.
        let (_committee, keypairs) = generate_test_bridge_committee(1);
        let kp = &keypairs[0];

        let pubkey0 = types::bridge::BridgePubkey::from_keypair(kp);
        let actions: Vec<BridgeAction> = vec![
            BridgeAction::Deposit {
                nonce: 1,
                eth_tx_hash: [0; 32],
                eth_event_idx: 0,
                sender_eth_address: [0; 20],
                target_chain: types::bridge::BridgeChainId::SomaCustom,
                recipient: SomaAddress::from([1; 32]),
                token_type: types::bridge::USDC_TOKEN_TYPE,
                amount: 100,
                timestamp_ms: 0,
            },
            BridgeAction::Withdrawal {
                nonce: 2,
                sender: SomaAddress::from([2; 32]),
                target_chain: types::bridge::BridgeChainId::EthCustom,
                recipient_eth_address: [3; 20],
                token_type: types::bridge::USDC_TOKEN_TYPE,
                amount: 200,
                timestamp_ms: 0,
            },
            BridgeAction::EmergencyPause { nonce: 0 },
            BridgeAction::EmergencyUnpause { nonce: 0 },
            BridgeAction::CommitteeUpdate {
                nonce: 0,
                new_members: vec![(pubkey0.clone(), 10000)],
            },
            BridgeAction::UpdateCommitteeBlocklist {
                nonce: 7,
                chain_id: SOMA_BRIDGE_CHAIN_ID,
                blocklist_type: BlocklistType::Blocklist,
                members: vec![pubkey0],
            },
            BridgeAction::LimitUpdate {
                nonce: 8,
                chain_id: SOMA_BRIDGE_CHAIN_ID,
                sending_chain_id: types::bridge::BridgeChainId::EthCustom,
                new_usd_limit: 1_000_000 * 10_000,
            },
            BridgeAction::EvmContractUpgrade {
                nonce: 9,
                chain_id: types::bridge::BridgeChainId::EthCustom,
                proxy: [0x06; 20],
                new_impl: [0x09; 20],
                call_data: vec![0x5c, 0xd8, 0xa7, 0x6b],
            },
        ];

        for action in &actions {
            let msg = action.to_message_bytes();
            let sig = sign_bridge_message(kp, &msg);
            let recovered = sig
                .recover_with_hash::<Keccak256>(&msg)
                .expect("ecrecover should succeed");
            assert_eq!(
                recovered.as_bytes(),
                kp.public().as_bytes(),
                "ecrecover failed for {:?}",
                action.message_type()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Governance action tests (UpdateCommitteeBlocklist, LimitUpdate, EvmContractUpgrade)
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_committee_blocklist_action_encoding() {
        let (_committee, keypairs) = generate_test_bridge_committee(2);
        let pubkey0 = types::bridge::BridgePubkey::from_keypair(&keypairs[0]);
        let pubkey1 = types::bridge::BridgePubkey::from_keypair(&keypairs[1]);

        let action = BridgeAction::UpdateCommitteeBlocklist {
            nonce: 42,
            chain_id: SOMA_BRIDGE_CHAIN_ID,
            blocklist_type: BlocklistType::Blocklist,
            members: vec![pubkey0.clone(), pubkey1.clone()],
        };
        let msg = action.to_message_bytes();

        // Header: prefix(19) + type(1) + version(1) + nonce(8) + chainID(1) = 30 bytes
        assert!(msg.starts_with(b"SOMA_BRIDGE_MESSAGE"));
        assert_eq!(msg[19], BridgeMessageType::UpdateCommitteeBlocklist as u8);
        assert_eq!(msg[20], BRIDGE_MESSAGE_VERSION);
        assert_eq!(action.nonce(), 42);
        // Payload: blocklist_type(1) + count(1) + 2 × eth_address(20) = 42 bytes
        assert_eq!(msg.len(), 19 + 1 + 1 + 8 + 1 + 1 + 1 + 40);
        assert_eq!(msg[30], 0x00); // Blocklist
        assert_eq!(msg[31], 0x02); // count

        // Member addresses match what derive_eth_address produces.
        let expected_addr0 = types::bridge::derive_eth_address(&pubkey0);
        let expected_addr1 = types::bridge::derive_eth_address(&pubkey1);
        assert_eq!(&msg[32..52], &expected_addr0);
        assert_eq!(&msg[52..72], &expected_addr1);
    }

    #[test]
    fn test_update_committee_blocklist_unblocklist_differs() {
        let (_committee, keypairs) = generate_test_bridge_committee(1);
        let pubkey = types::bridge::BridgePubkey::from_keypair(&keypairs[0]);

        let block = BridgeAction::UpdateCommitteeBlocklist {
            nonce: 1,
            chain_id: SOMA_BRIDGE_CHAIN_ID,
            blocklist_type: BlocklistType::Blocklist,
            members: vec![pubkey.clone()],
        };
        let unblock = BridgeAction::UpdateCommitteeBlocklist {
            nonce: 1,
            chain_id: SOMA_BRIDGE_CHAIN_ID,
            blocklist_type: BlocklistType::Unblocklist,
            members: vec![pubkey],
        };
        // Same nonce, same members, different type byte → different message digest.
        assert_ne!(block.to_message_bytes(), unblock.to_message_bytes());
    }

    #[test]
    fn test_limit_update_action_encoding() {
        let action = BridgeAction::LimitUpdate {
            nonce: 15,
            chain_id: SOMA_BRIDGE_CHAIN_ID,
            sending_chain_id: types::bridge::BridgeChainId::EthCustom,
            new_usd_limit: 1_000_000 * 10_000, // $1M with USD_MULTIPLIER
        };
        let msg = action.to_message_bytes();

        assert_eq!(msg[19], BridgeMessageType::LimitUpdate as u8);
        // Header(30) + sending_chain_id(1) + new_usd_limit(8) = 39 bytes
        assert_eq!(msg.len(), 30 + 9);
        assert_eq!(msg[30], types::bridge::BridgeChainId::EthCustom.as_u8());
        assert_eq!(&msg[31..39], &(1_000_000u64 * 10_000).to_be_bytes());
        assert_eq!(action.nonce(), 15);
    }

    #[test]
    fn test_limit_update_chain_id_distinguishes_route() {
        // Same sending_chain_id but different receiving (header) chain_id
        // must produce different message bytes — these target different routes.
        let to_soma = BridgeAction::LimitUpdate {
            nonce: 1,
            chain_id: SOMA_BRIDGE_CHAIN_ID,
            sending_chain_id: types::bridge::BridgeChainId::EthCustom,
            new_usd_limit: 1_000_000,
        };
        let to_eth = BridgeAction::LimitUpdate {
            nonce: 1,
            chain_id: types::bridge::BridgeChainId::EthCustom,
            sending_chain_id: SOMA_BRIDGE_CHAIN_ID,
            new_usd_limit: 1_000_000,
        };
        assert_ne!(to_soma.to_message_bytes(), to_eth.to_message_bytes());
    }

    #[test]
    fn test_evm_contract_upgrade_action_encoding() {
        let action = BridgeAction::EvmContractUpgrade {
            nonce: 123,
            chain_id: types::bridge::BridgeChainId::EthCustom,
            proxy: [0x06; 20],
            new_impl: [0x09; 20],
            call_data: vec![0x5c, 0xd8, 0xa7, 0x6b],
        };
        let msg = action.to_message_bytes();

        assert_eq!(msg[19], BridgeMessageType::EvmContractUpgrade as u8);
        // Header(30) + ABI-encoded payload (4 head slots × 32 + 32 padded data) = 190
        assert_eq!(msg.len(), 30 + 160);
        // Proxy left-padded to 32, then new_impl left-padded to 32.
        assert_eq!(&msg[30..42], &[0u8; 12]);
        assert_eq!(&msg[42..62], &[0x06u8; 20]);
        assert_eq!(&msg[62..74], &[0u8; 12]);
        assert_eq!(&msg[74..94], &[0x09u8; 20]);
    }

    #[test]
    fn test_evm_contract_upgrade_chain_id_targets_eth() {
        // EVM upgrades default to a non-Soma chain id so a quorum-signed upgrade
        // can't be replayed against the Soma-side bridge contract.
        let action = BridgeAction::EvmContractUpgrade {
            nonce: 1,
            chain_id: types::bridge::BridgeChainId::EthCustom,
            proxy: [0; 20],
            new_impl: [0; 20],
            call_data: vec![],
        };
        assert_eq!(action.chain_id(), types::bridge::BridgeChainId::EthCustom);
        assert_ne!(action.chain_id(), SOMA_BRIDGE_CHAIN_ID);
    }

    #[test]
    fn test_governance_actions_carry_real_nonces() {
        // Token-transfer + new governance variants must NOT hardcode nonce to 0.
        // Emergency/CommitteeUpdate are still 0 today — see audit follow-up.
        for nonce in [1u64, 42, 1_000_000] {
            assert_eq!(
                BridgeAction::UpdateCommitteeBlocklist {
                    nonce,
                    chain_id: SOMA_BRIDGE_CHAIN_ID,
                    blocklist_type: BlocklistType::Blocklist,
                    members: vec![],
                }
                .nonce(),
                nonce
            );
            assert_eq!(
                BridgeAction::LimitUpdate {
                    nonce,
                    chain_id: SOMA_BRIDGE_CHAIN_ID,
                    sending_chain_id: types::bridge::BridgeChainId::EthCustom,
                    new_usd_limit: 0,
                }
                .nonce(),
                nonce
            );
            assert_eq!(
                BridgeAction::EvmContractUpgrade {
                    nonce,
                    chain_id: types::bridge::BridgeChainId::EthCustom,
                    proxy: [0; 20],
                    new_impl: [0; 20],
                    call_data: vec![],
                }
                .nonce(),
                nonce
            );
        }
    }

    #[test]
    fn test_approval_thresholds_per_action() {
        let (committee, _keypairs) = generate_test_bridge_committee(4);

        let dep = BridgeAction::Deposit {
            nonce: 1,
            eth_tx_hash: [0; 32],
            eth_event_idx: 0,
            sender_eth_address: [0; 20],
            target_chain: types::bridge::BridgeChainId::SomaCustom,
            recipient: SomaAddress::from([0; 32]),
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1,
            timestamp_ms: 0,
        };
        let pause = BridgeAction::EmergencyPause { nonce: 0 };
        let unpause = BridgeAction::EmergencyUnpause { nonce: 0 };
        let blocklist = BridgeAction::UpdateCommitteeBlocklist {
            nonce: 1,
            chain_id: SOMA_BRIDGE_CHAIN_ID,
            blocklist_type: BlocklistType::Blocklist,
            members: vec![],
        };
        let limit = BridgeAction::LimitUpdate {
            nonce: 1,
            chain_id: SOMA_BRIDGE_CHAIN_ID,
            sending_chain_id: types::bridge::BridgeChainId::EthCustom,
            new_usd_limit: 0,
        };
        let upgrade = BridgeAction::EvmContractUpgrade {
            nonce: 1,
            chain_id: types::bridge::BridgeChainId::EthCustom,
            proxy: [0; 20],
            new_impl: [0; 20],
            call_data: vec![],
        };

        // f+1 deposit < pause < unpause; governance actions all require majority.
        assert!(dep.approval_threshold(&committee) < unpause.approval_threshold(&committee));
        assert!(pause.approval_threshold(&committee) < unpause.approval_threshold(&committee));
        assert!(blocklist.approval_threshold(&committee) >= 5001);
        assert!(limit.approval_threshold(&committee) >= 5001);
        assert!(upgrade.approval_threshold(&committee) >= 5001);
    }
}
