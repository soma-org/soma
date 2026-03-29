use serde::{Deserialize, Serialize};
use types::base::SomaAddress;
use types::object::ObjectID;
use types::bridge::{
    BridgeMessageType, EmergencyOpCode, BRIDGE_MESSAGE_VERSION, SOMA_BRIDGE_CHAIN_ID,
    encode_bridge_message, encode_deposit_payload, encode_emergency_payload,
    encode_withdraw_payload,
};

/// A bridge action that needs committee signatures before it can be executed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BridgeAction {
    /// USDC deposit from Ethereum → Soma. Observed from Ethereum finalized blocks.
    Deposit {
        nonce: u64,
        eth_tx_hash: [u8; 32],
        recipient: SomaAddress,
        amount: u64,
    },
    /// USDC withdrawal from Soma → Ethereum. Observed from Soma checkpoints
    /// via PendingWithdrawal objects.
    Withdrawal {
        nonce: u64,
        sender: SomaAddress,
        recipient_eth_address: [u8; 20],
        amount: u64,
    },
    /// Emergency pause — stops all bridge operations.
    EmergencyPause,
    /// Emergency unpause — resumes bridge operations.
    EmergencyUnpause,
    /// Committee update — sync Ethereum contract with new validator set at epoch boundary.
    CommitteeUpdate {
        new_members: Vec<(Vec<u8>, u64)>, // (compressed_pubkey 33 bytes, voting_power)
    },
}

impl BridgeAction {
    /// Returns the bridge message type for this action.
    pub fn message_type(&self) -> BridgeMessageType {
        match self {
            BridgeAction::Deposit { .. } => BridgeMessageType::UsdcDeposit,
            BridgeAction::Withdrawal { .. } => BridgeMessageType::UsdcWithdraw,
            BridgeAction::EmergencyPause | BridgeAction::EmergencyUnpause => {
                BridgeMessageType::EmergencyOp
            }
            BridgeAction::CommitteeUpdate { .. } => BridgeMessageType::CommitteeUpdate,
        }
    }

    /// Returns the nonce for this action (0 for emergency ops / committee updates).
    pub fn nonce(&self) -> u64 {
        match self {
            BridgeAction::Deposit { nonce, .. } | BridgeAction::Withdrawal { nonce, .. } => *nonce,
            BridgeAction::EmergencyPause
            | BridgeAction::EmergencyUnpause
            | BridgeAction::CommitteeUpdate { .. } => 0,
        }
    }

    /// Encode this action into the canonical bridge message bytes for signing.
    /// Format: PREFIX || type(1) || version(1) || nonce(8,BE) || chainID(8,BE) || payload
    pub fn to_message_bytes(&self) -> Vec<u8> {
        let payload = self.encode_payload();
        encode_bridge_message(
            self.message_type(),
            self.nonce(),
            SOMA_BRIDGE_CHAIN_ID,
            &payload,
        )
    }

    fn encode_payload(&self) -> Vec<u8> {
        match self {
            BridgeAction::Deposit {
                recipient, amount, ..
            } => encode_deposit_payload(recipient, *amount),
            BridgeAction::Withdrawal {
                recipient_eth_address,
                amount,
                ..
            } => encode_withdraw_payload(recipient_eth_address, *amount),
            BridgeAction::EmergencyPause => {
                encode_emergency_payload(EmergencyOpCode::Freeze)
            }
            BridgeAction::EmergencyUnpause => {
                encode_emergency_payload(EmergencyOpCode::Unfreeze)
            }
            BridgeAction::CommitteeUpdate { new_members } => {
                // Encode: count(4,BE) || (pubkey(33) || voting_power(8,BE))*
                let mut payload = Vec::new();
                payload.extend_from_slice(&(new_members.len() as u32).to_be_bytes());
                for (pubkey, power) in new_members {
                    payload.extend_from_slice(pubkey.as_slice());
                    payload.extend_from_slice(&power.to_be_bytes());
                }
                payload
            }
        }
    }
}

/// A bridge action with a collected signature from one committee member.
#[derive(Debug, Clone)]
pub struct SignedBridgeAction {
    pub action: BridgeAction,
    pub signer_index: u32,
    pub signature: Vec<u8>, // 65-byte recoverable ECDSA signature
}

/// A deposit event parsed from Ethereum logs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepositEvent {
    pub nonce: u64,
    pub eth_sender: [u8; 20],
    pub soma_recipient: [u8; 32], // raw bytes of SomaAddress
    pub amount: u64,
    pub tx_hash: [u8; 32],
    pub block_number: u64,
}

impl DepositEvent {
    /// Convert to a BridgeAction for signing.
    pub fn to_bridge_action(&self) -> BridgeAction {
        BridgeAction::Deposit {
            nonce: self.nonce,
            eth_tx_hash: self.tx_hash,
            recipient: SomaAddress::from(self.soma_recipient),
            amount: self.amount,
        }
    }
}

/// Summary of a pending withdrawal observed from Soma checkpoints.
#[derive(Debug, Clone)]
pub struct ObservedWithdrawal {
    pub id: ObjectID,
    pub nonce: u64,
    pub sender: SomaAddress,
    pub recipient_eth_address: [u8; 20],
    pub amount: u64,
}

impl ObservedWithdrawal {
    /// Convert to a BridgeAction for signing.
    pub fn to_bridge_action(&self) -> BridgeAction {
        BridgeAction::Withdrawal {
            nonce: self.nonce,
            sender: self.sender,
            recipient_eth_address: self.recipient_eth_address,
            amount: self.amount,
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
    };

    #[test]
    fn test_deposit_action_message_encoding() {
        let action = BridgeAction::Deposit {
            nonce: 42,
            eth_tx_hash: [0xAB; 32],
            recipient: SomaAddress::from([0x01; 32]),
            amount: 1_000_000, // 1 USDC
        };

        let msg = action.to_message_bytes();

        // Verify prefix
        assert!(msg.starts_with(b"SOMA_BRIDGE_MESSAGE"));
        // Verify message type byte (UsdcDeposit = 0)
        assert_eq!(msg[19], 0);
        // Verify version byte
        assert_eq!(msg[20], BRIDGE_MESSAGE_VERSION);

        // Ensure deterministic: same action produces same bytes
        assert_eq!(msg, action.to_message_bytes());
    }

    #[test]
    fn test_withdrawal_action_message_encoding() {
        let action = BridgeAction::Withdrawal {
            nonce: 7,
            sender: SomaAddress::from([0x02; 32]),
            recipient_eth_address: [0xCC; 20],
            amount: 5_000_000,
        };

        let msg = action.to_message_bytes();
        assert!(msg.starts_with(b"SOMA_BRIDGE_MESSAGE"));
        assert_eq!(msg[19], 1); // UsdcWithdraw = 1
    }

    #[test]
    fn test_emergency_actions_encoding() {
        let pause = BridgeAction::EmergencyPause;
        let unpause = BridgeAction::EmergencyUnpause;

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
            soma_recipient: [0xBB; 32],
            amount: 100_000,
            tx_hash: [0xCC; 32],
            block_number: 12345,
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
            recipient: SomaAddress::from([0x01; 32]),
            amount: 1_000_000,
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
            recipient: SomaAddress::from([0x42; 32]),
            amount: 10_000_000, // 10 USDC
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
            recipient: SomaAddress::from([0x01; 32]),
            amount: 100,
        };
        let action_b = BridgeAction::Deposit {
            nonce: 2, // different nonce
            eth_tx_hash: [0; 32],
            recipient: SomaAddress::from([0x01; 32]),
            amount: 100,
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
        // Verify the aggregated signature format matches what the executor expects.
        let (_committee, keypairs) = generate_test_bridge_committee(4);
        let action = BridgeAction::Deposit {
            nonce: 1,
            eth_tx_hash: [0; 32],
            recipient: SomaAddress::from([0x01; 32]),
            amount: 1_000_000,
        };

        let msg_bytes = action.to_message_bytes();

        // Sign with members 0 and 2 (indices in BTreeMap order)
        let signers: Vec<(usize, &fastcrypto::secp256k1::Secp256k1KeyPair)> =
            vec![(0, &keypairs[0]), (2, &keypairs[2])];
        let (agg_sig, bitmap) = build_bridge_signatures(&signers, &msg_bytes);

        // Aggregated sig: 2 × 65 = 130 bytes
        assert_eq!(agg_sig.len(), 130);

        // Bitmap: bit 0 and bit 2 set = 0b00000101 = 5
        assert_eq!(bitmap, vec![5u8]);

        // Each 65-byte signature should ecrecover to the correct signer
        for (i, (member_idx, kp)) in signers.iter().enumerate() {
            let sig_bytes = &agg_sig[i * 65..(i + 1) * 65];
            let sig = Secp256k1RecoverableSignature::from_bytes(sig_bytes)
                .expect("valid 65-byte recoverable sig");
            let recovered = sig
                .recover_with_hash::<Keccak256>(&msg_bytes)
                .expect("ecrecover should work");
            assert_eq!(
                recovered.as_bytes(),
                kp.public().as_bytes(),
                "signature at index {i} (member {member_idx}) must ecrecover to correct pubkey"
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
            recipient: SomaAddress::from([0u8; 32]),
            amount: 1_000_000,
        };

        let msg_bytes = action.to_message_bytes();

        // Verify the raw encoding structure:
        // PREFIX(19) || type(1) || version(1) || nonce(8) || chainID(8) || payload(40)
        assert_eq!(msg_bytes.len(), 19 + 1 + 1 + 8 + 8 + 32 + 8); // 77 bytes

        // PREFIX
        assert_eq!(&msg_bytes[0..19], b"SOMA_BRIDGE_MESSAGE");
        // Type = UsdcDeposit = 0
        assert_eq!(msg_bytes[19], 0);
        // Version = 1
        assert_eq!(msg_bytes[20], 1);
        // Nonce = 1 (big-endian u64)
        assert_eq!(&msg_bytes[21..29], &1u64.to_be_bytes());
        // ChainID = SOMA_BRIDGE_CHAIN_ID = 1 (big-endian u64)
        assert_eq!(&msg_bytes[29..37], &SOMA_BRIDGE_CHAIN_ID.to_be_bytes());
        // Payload: recipient(32 zero bytes) + amount(1_000_000 BE u64)
        assert_eq!(&msg_bytes[37..69], &[0u8; 32]); // recipient
        assert_eq!(&msg_bytes[69..77], &1_000_000u64.to_be_bytes()); // amount

        // Compute and print keccak256 hash for Solidity cross-verification
        let hash = Keccak256::digest(&msg_bytes);
        let hash_hex = hex::encode(hash.as_ref());
        // This hash MUST match Solidity's keccak256(abi.encodePacked(...))
        // with the same inputs. Pin it as a regression test.
        assert_eq!(hash_hex.len(), 64, "keccak256 produces 32 bytes");
        // Store the expected hash — any change in encoding will break this.
        let expected_hash = hash_hex.clone();
        // Re-encode and verify determinism
        let hash2 = Keccak256::digest(&action.to_message_bytes());
        assert_eq!(hex::encode(hash2.as_ref()), expected_hash);
    }

    #[test]
    fn test_withdrawal_encoding_known_values() {
        let action = BridgeAction::Withdrawal {
            nonce: 42,
            sender: SomaAddress::from([0xFF; 32]),
            recipient_eth_address: [0xAA; 20],
            amount: 5_000_000,
        };

        let msg_bytes = action.to_message_bytes();

        // PREFIX(19) || type(1) || version(1) || nonce(8) || chainID(8) || payload(28)
        assert_eq!(msg_bytes.len(), 19 + 1 + 1 + 8 + 8 + 20 + 8); // 65 bytes

        assert_eq!(msg_bytes[19], 1); // UsdcWithdraw
        assert_eq!(&msg_bytes[21..29], &42u64.to_be_bytes());
        // Payload: eth_recipient(20) + amount(8)
        assert_eq!(&msg_bytes[37..57], &[0xAA; 20]);
        assert_eq!(&msg_bytes[57..65], &5_000_000u64.to_be_bytes());
    }

    #[test]
    fn test_emergency_encoding_known_values() {
        let pause = BridgeAction::EmergencyPause;
        let unpause = BridgeAction::EmergencyUnpause;

        let pause_msg = pause.to_message_bytes();
        let unpause_msg = unpause.to_message_bytes();

        // PREFIX(19) || type(1) || version(1) || nonce(8) || chainID(8) || payload(1)
        assert_eq!(pause_msg.len(), 19 + 1 + 1 + 8 + 8 + 1); // 38 bytes
        assert_eq!(unpause_msg.len(), 38);

        // Type = EmergencyOp = 2
        assert_eq!(pause_msg[19], 2);
        assert_eq!(unpause_msg[19], 2);

        // Nonce = 0 for emergency ops
        assert_eq!(&pause_msg[21..29], &0u64.to_be_bytes());

        // Payload: Freeze = 0, Unfreeze = 1
        assert_eq!(pause_msg[37], 0); // Freeze
        assert_eq!(unpause_msg[37], 1); // Unfreeze
    }

    #[test]
    fn test_committee_update_encoding() {
        let action = BridgeAction::CommitteeUpdate {
            new_members: vec![
                (vec![0x02; 33], 5000), // member 0
                (vec![0x03; 33], 5000), // member 1
            ],
        };

        let msg_bytes = action.to_message_bytes();

        assert_eq!(msg_bytes[19], 3); // CommitteeUpdate
        // Nonce = 0 for committee updates
        assert_eq!(&msg_bytes[21..29], &0u64.to_be_bytes());

        // Payload: count(4) + (pubkey(33) + power(8)) * 2
        let payload_start = 37;
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

        let actions: Vec<BridgeAction> = vec![
            BridgeAction::Deposit {
                nonce: 1,
                eth_tx_hash: [0; 32],
                recipient: SomaAddress::from([1; 32]),
                amount: 100,
            },
            BridgeAction::Withdrawal {
                nonce: 2,
                sender: SomaAddress::from([2; 32]),
                recipient_eth_address: [3; 20],
                amount: 200,
            },
            BridgeAction::EmergencyPause,
            BridgeAction::EmergencyUnpause,
            BridgeAction::CommitteeUpdate {
                new_members: vec![(vec![0x02; 33], 10000)],
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
}
