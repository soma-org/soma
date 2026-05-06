// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::object::ObjectID;

// ---------------------------------------------------------------------------
// Bridge message encoding — must match Solidity's SomaBridgeMessage.encodeMessage()
// ---------------------------------------------------------------------------

/// Prefix for all bridge messages, used to domain-separate bridge signatures.
pub const BRIDGE_MESSAGE_PREFIX: &[u8] = b"SOMA_BRIDGE_MESSAGE";

/// Bridge message types — must match Solidity constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BridgeMessageType {
    UsdcDeposit = 0,
    UsdcWithdraw = 1,
    EmergencyOp = 2,
    CommitteeUpdate = 3,
}

/// Emergency operation codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EmergencyOpCode {
    Freeze = 0,
    Unfreeze = 1,
}

/// Version of the bridge message format.
pub const BRIDGE_MESSAGE_VERSION: u8 = 1;

/// Encode a bridge message for signing. The format is:
/// `PREFIX || type(1) || version(1) || nonce(8, big-endian) || chainID(8, big-endian) || payload`
///
/// This matches Solidity's `abi.encodePacked(...)` used in `SomaBridgeMessage.encodeMessage()`.
/// The resulting bytes are hashed with Keccak256 before ECDSA signing.
pub fn encode_bridge_message(
    msg_type: BridgeMessageType,
    nonce: u64,
    chain_id: u64,
    payload: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(
        BRIDGE_MESSAGE_PREFIX.len() + 1 + 1 + 8 + 8 + payload.len(),
    );
    buf.extend_from_slice(BRIDGE_MESSAGE_PREFIX);
    buf.push(msg_type as u8);
    buf.push(BRIDGE_MESSAGE_VERSION);
    buf.extend_from_slice(&nonce.to_be_bytes());
    buf.extend_from_slice(&chain_id.to_be_bytes());
    buf.extend_from_slice(payload);
    buf
}

/// Encode the payload for a USDC deposit message.
/// `recipient(32 bytes) || amount(8 bytes, big-endian)`
pub fn encode_deposit_payload(recipient: &SomaAddress, amount: u64) -> Vec<u8> {
    let mut payload = Vec::with_capacity(32 + 8);
    payload.extend_from_slice(recipient.as_ref());
    payload.extend_from_slice(&amount.to_be_bytes());
    payload
}

/// Encode the payload for a USDC withdrawal message.
/// `eth_recipient(20 bytes) || amount(8 bytes, big-endian)`
pub fn encode_withdraw_payload(eth_recipient: &[u8; 20], amount: u64) -> Vec<u8> {
    let mut payload = Vec::with_capacity(20 + 8);
    payload.extend_from_slice(eth_recipient);
    payload.extend_from_slice(&amount.to_be_bytes());
    payload
}

/// Encode the payload for an emergency operation message.
/// `op_code(1 byte)`
pub fn encode_emergency_payload(op_code: EmergencyOpCode) -> Vec<u8> {
    vec![op_code as u8]
}

/// The Soma bridge chain ID. Used in message encoding to prevent cross-chain replay.
/// Testnet and mainnet should use different values; this is the default for testnet.
pub const SOMA_BRIDGE_CHAIN_ID: u64 = 1;

/// Bridge state stored in SystemState, tracking USDC bridge between Ethereum and Soma.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BridgeState {
    pub paused: bool,
    pub next_withdrawal_nonce: u64,
    /// Set of recent deposit nonces that have already been processed,
    /// for replay protection. Bounded by [`PROCESSED_NONCE_RETENTION_LIMIT`]:
    /// when the set hits the limit, the oldest nonces are pruned and
    /// `min_acceptable_deposit_nonce` advances accordingly. Audit F16
    /// fix — without bounding, this set grew with every deposit forever
    /// and inflated SystemState's BCS size on every system tx.
    pub processed_deposit_nonces: BTreeSet<u64>,
    /// Watermark below which deposit nonces are rejected outright as
    /// "too old" — they may have been pruned from
    /// `processed_deposit_nonces`, so we can no longer tell apart "new"
    /// from "replayed". Audit F16 fix.
    pub min_acceptable_deposit_nonce: u64,
    pub bridge_committee: BridgeCommittee,
    pub total_bridged_usdc: u64,
}

/// Maximum size of [`BridgeState::processed_deposit_nonces`] before
/// the oldest nonces are pruned. 100k nonces ≈ 800kB BCS-encoded —
/// large enough to absorb realistic out-of-order delivery, bounded
/// enough that SystemState size stays manageable.
pub const PROCESSED_NONCE_RETENTION_LIMIT: usize = 100_000;

impl BridgeState {
    pub fn new(committee: BridgeCommittee) -> Self {
        Self {
            paused: false,
            next_withdrawal_nonce: 0,
            processed_deposit_nonces: BTreeSet::new(),
            min_acceptable_deposit_nonce: 0,
            bridge_committee: committee,
            total_bridged_usdc: 0,
        }
    }

    /// Record a freshly-processed deposit nonce, evicting the oldest
    /// entries if the set has grown past
    /// [`PROCESSED_NONCE_RETENTION_LIMIT`]. The replay check
    /// (`nonce > min_acceptable_deposit_nonce && !set.contains(&nonce)`)
    /// stays correct because evicted nonces fall below the watermark
    /// and are rejected as "too old".
    pub fn record_processed_deposit_nonce(&mut self, nonce: u64) {
        self.processed_deposit_nonces.insert(nonce);
        while self.processed_deposit_nonces.len() > PROCESSED_NONCE_RETENTION_LIMIT {
            if let Some(&min) = self.processed_deposit_nonces.iter().next() {
                self.processed_deposit_nonces.remove(&min);
                if min >= self.min_acceptable_deposit_nonce {
                    self.min_acceptable_deposit_nonce = min + 1;
                }
            } else {
                break;
            }
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BridgeCommittee {
    pub members: BTreeMap<SomaAddress, BridgeMember>,
    /// f+1, ~3334/10000
    pub threshold_deposit: u64,
    /// f+1, ~3334/10000
    pub threshold_withdraw: u64,
    /// ~450/10000
    pub threshold_pause: u64,
    /// 2/3, ~6667/10000
    pub threshold_unpause: u64,
}

impl BridgeCommittee {
    pub fn empty() -> Self {
        Self {
            members: BTreeMap::new(),
            threshold_deposit: 3334,
            threshold_withdraw: 3334,
            threshold_pause: 450,
            threshold_unpause: 6667,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BridgeMember {
    /// Compressed Secp256k1 public key (33 bytes) for EVM-compatible bridge signing
    pub ecdsa_pubkey: Vec<u8>,
    pub voting_power: u64,
}

/// Created when a user initiates a USDC withdrawal from Soma to Ethereum.
/// Bridge nodes observe this in checkpoints and sign for Ethereum release.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PendingWithdrawal {
    pub id: ObjectID,
    pub nonce: u64,
    pub sender: SomaAddress,
    pub recipient_eth_address: [u8; 20],
    pub amount: u64,
    pub created_at_ms: u64,
}

/// Marketplace parameters stored in SystemState.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct MarketplaceParameters {
    /// Rating window in milliseconds (e.g., 48 hours = 172_800_000)
    pub rating_window_ms: u64,
    /// Minimum ask timeout (floor, e.g., 10 seconds = 10_000)
    pub min_ask_timeout_ms: u64,
    /// Maximum ask timeout (ceiling, e.g., 7 days = 604_800_000)
    pub max_ask_timeout_ms: u64,
    /// Marketplace fee in basis points (e.g., 250 = 2.5%)
    pub marketplace_fee_bps: u64,
}

impl Default for MarketplaceParameters {
    fn default() -> Self {
        Self {
            rating_window_ms: 172_800_000,       // 48 hours
            min_ask_timeout_ms: 10_000,          // 10 seconds
            max_ask_timeout_ms: 604_800_000,     // 7 days
            marketplace_fee_bps: 250,            // 2.5%
        }
    }
}

// ---------------------------------------------------------------------------
// Bridge signing utilities
// ---------------------------------------------------------------------------

use fastcrypto::hash::Keccak256;
use fastcrypto::secp256k1::Secp256k1KeyPair;
use fastcrypto::secp256k1::Secp256k1PublicKey;
use fastcrypto::secp256k1::recoverable::Secp256k1RecoverableSignature;
use fastcrypto::traits::{KeyPair, RecoverableSigner, ToFromBytes};

/// Sign a bridge message with a secp256k1 keypair using Keccak256 hash.
/// Returns a 65-byte recoverable signature (r[32] + s[32] + v[1]).
pub fn sign_bridge_message(
    keypair: &Secp256k1KeyPair,
    message: &[u8],
) -> Secp256k1RecoverableSignature {
    keypair.sign_recoverable_with_hash::<Keccak256>(message)
}

/// Build aggregated signature bytes and signer bitmap for a set of signers.
///
/// `signers` is a list of (member_index, keypair) pairs where member_index is
/// the position of the member in the BTreeMap iteration order.
///
/// Returns `(aggregated_signature, signer_bitmap)`:
/// - `aggregated_signature`: concatenated 65-byte recoverable signatures, ordered by member index
/// - `signer_bitmap`: bitmap where bit i is set if member i signed
pub fn build_bridge_signatures(
    signers: &[(usize, &Secp256k1KeyPair)],
    message: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    // Sort by member index to ensure signatures are in bitmap order
    let mut sorted: Vec<_> = signers.to_vec();
    sorted.sort_by_key(|(idx, _)| *idx);

    let max_index = sorted.iter().map(|(idx, _)| *idx).max().unwrap_or(0);
    let bitmap_len = max_index / 8 + 1;
    let mut bitmap = vec![0u8; bitmap_len];
    let mut aggregated = Vec::with_capacity(sorted.len() * 65);

    for (idx, keypair) in &sorted {
        // Set bit in bitmap
        bitmap[*idx / 8] |= 1 << (*idx % 8);
        // Sign and append 65-byte recoverable signature
        let sig = sign_bridge_message(keypair, message);
        aggregated.extend_from_slice(sig.as_ref());
    }

    (aggregated, bitmap)
}

/// Generate a test bridge committee with real secp256k1 keypairs.
/// Returns `(committee, keypairs)` where keypairs is a Vec in the same
/// order as the committee's BTreeMap iteration order.
pub fn generate_test_bridge_committee(
    num_members: usize,
) -> (BridgeCommittee, Vec<Secp256k1KeyPair>) {
    use std::collections::BTreeMap;

    let voting_power_each = 10000u64 / num_members as u64;
    let mut members = BTreeMap::new();
    let mut keypairs = Vec::with_capacity(num_members);
    // Collect (addr, keypair) pairs first, then sort by addr to match BTreeMap order
    let mut addr_kp_pairs: Vec<_> = (0..num_members)
        .map(|_| {
            let kp = Secp256k1KeyPair::generate(&mut rand::thread_rng());
            let pubkey_bytes = kp.public().as_bytes().to_vec();
            let addr = SomaAddress::random();
            (addr, kp, pubkey_bytes)
        })
        .collect();
    // Insert into BTreeMap (sorts by SomaAddress)
    for (addr, _kp, pubkey_bytes) in &addr_kp_pairs {
        members.insert(
            *addr,
            BridgeMember {
                ecdsa_pubkey: pubkey_bytes.clone(),
                voting_power: voting_power_each,
            },
        );
    }
    // Build keypairs Vec in BTreeMap iteration order (by address)
    let ordered_addrs: Vec<_> = members.keys().cloned().collect();
    for addr in &ordered_addrs {
        let idx = addr_kp_pairs.iter().position(|(a, _, _)| a == addr).unwrap();
        keypairs.push(addr_kp_pairs.remove(idx).1);
    }

    let committee = BridgeCommittee {
        members,
        threshold_deposit: 3334,
        threshold_withdraw: 3334,
        threshold_pause: 450,
        threshold_unpause: 6667,
    };
    (committee, keypairs)
}
