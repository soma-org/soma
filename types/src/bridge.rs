// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use fastcrypto::error::FastCryptoError;
use fastcrypto::hash::{HashFunction, Keccak256};
use fastcrypto::secp256k1::Secp256k1KeyPair;
use fastcrypto::secp256k1::Secp256k1PublicKey;
use fastcrypto::secp256k1::recoverable::Secp256k1RecoverableSignature;
use fastcrypto::traits::{KeyPair, RecoverableSigner, ToFromBytes};
use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::object::ObjectID;

// ---------------------------------------------------------------------------
// BridgeChainId — identifies a chain participating in the bridge.
// ---------------------------------------------------------------------------

/// Bridge chain identifier. Encoded as a single byte on the wire for parity
/// with Sui's `BridgeChainId`. Numbering is also Sui-compatible for the Eth
/// variants so Solidity-side parsers can be shared.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[repr(u8)]
pub enum BridgeChainId {
    SomaMainnet = 0,
    SomaTestnet = 1,
    SomaCustom = 2,
    EthMainnet = 10,
    EthSepolia = 11,
    EthCustom = 12,
}

impl BridgeChainId {
    pub const fn is_soma_chain(self) -> bool {
        matches!(self, Self::SomaMainnet | Self::SomaTestnet | Self::SomaCustom)
    }

    pub const fn is_eth_chain(self) -> bool {
        matches!(self, Self::EthMainnet | Self::EthSepolia | Self::EthCustom)
    }

    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Parse a single byte into a [`BridgeChainId`]. Returns `None` for
    /// values that don't correspond to any defined variant.
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::SomaMainnet),
            1 => Some(Self::SomaTestnet),
            2 => Some(Self::SomaCustom),
            10 => Some(Self::EthMainnet),
            11 => Some(Self::EthSepolia),
            12 => Some(Self::EthCustom),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// BridgePubkey — typed wrapper around a 33-byte compressed secp256k1 pubkey.
//
// Validates curve membership at construction time so downstream code (eth
// address derivation, committee membership, etc.) can be infallible.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BridgePubkey([u8; 33]);

impl Serialize for BridgePubkey {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.as_slice().to_vec().serialize(s)
    }
}

impl<'de> Deserialize<'de> for BridgePubkey {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let bytes = <Vec<u8>>::deserialize(d)?;
        Self::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}

impl BridgePubkey {
    /// Parse a 33-byte compressed secp256k1 pubkey, validating curve membership.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        // Round-trip through Secp256k1PublicKey to validate the curve point.
        let pk = Secp256k1PublicKey::from_bytes(bytes)?;
        let raw = pk.as_bytes();
        if raw.len() != 33 {
            return Err(FastCryptoError::InvalidInput);
        }
        let mut arr = [0u8; 33];
        arr.copy_from_slice(raw);
        Ok(Self(arr))
    }

    pub fn from_keypair(kp: &Secp256k1KeyPair) -> Self {
        // Public keys produced by the keypair are always valid.
        let bytes = kp.public().as_bytes().to_vec();
        Self::from_bytes(&bytes).expect("keypair pubkey is always valid")
    }

    pub fn as_bytes(&self) -> &[u8; 33] {
        &self.0
    }

    /// Standard EIP-55 derivation: decompress the pubkey to its 65-byte
    /// uncompressed form (`0x04 || X(32) || Y(32)`), keccak256 the 64-byte
    /// X||Y, and take the last 20 bytes of the digest.
    pub fn to_eth_address(&self) -> [u8; 20] {
        // Unwrap safe: Self only constructs from valid curve points.
        let pk = Secp256k1PublicKey::from_bytes(&self.0).expect("BridgePubkey invariant");
        let uncompressed = pk.pubkey.serialize_uncompressed();
        let digest = Keccak256::digest(&uncompressed[1..]);
        let mut addr = [0u8; 20];
        addr.copy_from_slice(&digest.digest[12..]);
        addr
    }
}

impl AsRef<[u8]> for BridgePubkey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Display for BridgePubkey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0[..4] {
            write!(f, "{:02x}", b)?;
        }
        write!(f, "..")
    }
}

// ---------------------------------------------------------------------------
// BridgeSignature — wraps a 65-byte recoverable secp256k1 signature
// (r[32] || s[32] || v[1]) with serde-friendly serialization. Cannot just
// use `[u8; 65]` because serde only impls Deserialize for arrays up to 32.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BridgeSignature(pub [u8; 65]);

impl BridgeSignature {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        if bytes.len() != 65 {
            return Err(FastCryptoError::InputLengthWrong(65));
        }
        let mut arr = [0u8; 65];
        arr.copy_from_slice(bytes);
        Ok(Self(arr))
    }

    pub fn as_bytes(&self) -> &[u8; 65] {
        &self.0
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<[u8]> for BridgeSignature {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Serialize for BridgeSignature {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Serialize as Vec<u8> (length-prefixed in BCS) for serde compatibility.
        // The length prefix costs 1 byte per cert entry; trivial.
        self.0.as_slice().to_vec().serialize(s)
    }
}

impl<'de> Deserialize<'de> for BridgeSignature {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let bytes = <Vec<u8>>::deserialize(d)?;
        Self::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// Bridge message encoding — must match Solidity's SomaBridgeMessage.encodeMessage()
// ---------------------------------------------------------------------------

/// Prefix for all bridge messages, used to domain-separate bridge signatures.
pub const BRIDGE_MESSAGE_PREFIX: &[u8] = b"SOMA_BRIDGE_MESSAGE";

/// Bridge message types — must match Solidity constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum BridgeMessageType {
    UsdcDeposit = 0,
    UsdcWithdraw = 1,
    EmergencyOp = 2,
    CommitteeUpdate = 3,
    UpdateCommitteeBlocklist = 4,
    LimitUpdate = 5,
    EvmContractUpgrade = 6,
}

/// Emergency operation codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EmergencyOpCode {
    Freeze = 0,
    Unfreeze = 1,
}

/// Blocklist operation codes for [`BridgeMessageType::UpdateCommitteeBlocklist`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum BlocklistType {
    Blocklist = 0,
    Unblocklist = 1,
}

/// USD value multiplier for [`BridgeMessageType::LimitUpdate`]: 4 decimal places, so $1 = 10000.
/// Mirrors sui-bridge's `USD_MULTIPLIER`.
pub const USD_MULTIPLIER: u64 = 10000;

/// Ethereum chain ID used in the message header when an action targets the
/// Ethereum-side bridge contract (e.g. [`BridgeMessageType::EvmContractUpgrade`]).
/// Distinct from [`SOMA_BRIDGE_CHAIN_ID`] so a quorum-signed EVM action cannot
/// be replayed against the Soma side and vice versa.
pub const ETH_BRIDGE_CHAIN_ID: BridgeChainId = BridgeChainId::EthCustom;

/// Default version for non-token bridge messages (emergency op,
/// blocklist, limit update, committee update, etc.). Mirrors Sui's
/// `CURRENT_MESSAGE_VERSION = 1`.
pub const BRIDGE_MESSAGE_VERSION: u8 = 1;

/// Token transfer messages (USDC deposit / withdraw) use version 2,
/// which appends a `timestamp_ms` field to the payload. Mirrors Sui's
/// `TOKEN_TRANSFER_MESSAGE_VERSION_V2 = 2`. The timestamp is what would
/// power a future Eth-side limiter's 48h bypass; even though Soma has
/// no on-chain limiter, the field exists for cross-chain wire-format
/// parity and audit trail.
pub const TOKEN_TRANSFER_MESSAGE_VERSION_V2: u8 = 2;

/// Wire-format token id for USDC. Matches Sui's `BridgeMessage.USDC = 3`
/// — Sui reserves 0..2 for SUI/BTC/ETH and 3+ for stablecoins, so we use
/// 3 even though Soma is USDC-only today. Keeps the namespace forward
/// -compatible if Soma ever adds more tokens (each gets its own id; the
/// Eth-side decoder asserts `token_type == USDC_TOKEN_TYPE` for now).
pub const USDC_TOKEN_TYPE: u8 = 3;

/// Sui-parity Soma address length (32 bytes). Used as the
/// `sender_address_length` / `target_address_length` framing byte in
/// the V2 token-transfer payload when the field is a Soma address.
pub const SOMA_ADDRESS_LENGTH: u8 = 32;

/// Sui-parity Eth address length (20 bytes). Used as the
/// `sender_address_length` / `target_address_length` framing byte in
/// the V2 token-transfer payload when the field is an Eth address.
pub const ETH_ADDRESS_LENGTH: u8 = 20;

/// Encode a bridge message for signing. The format is:
/// `PREFIX || type(1) || version(1) || nonce(8, big-endian) || chainID(1) || payload`
///
/// `version` is per-message-type: token transfers use V2 (with timestamp
/// in payload), system messages use V1. Chain id is a single byte to match
/// Sui's wire format and minimize Solidity verification gas.
pub fn encode_bridge_message(
    msg_type: BridgeMessageType,
    version: u8,
    nonce: u64,
    chain_id: BridgeChainId,
    payload: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(
        BRIDGE_MESSAGE_PREFIX.len() + 1 + 1 + 8 + 1 + payload.len(),
    );
    buf.extend_from_slice(BRIDGE_MESSAGE_PREFIX);
    buf.push(msg_type as u8);
    buf.push(version);
    buf.extend_from_slice(&nonce.to_be_bytes());
    buf.push(chain_id.as_u8());
    buf.extend_from_slice(payload);
    buf
}

/// Encode the V2 payload for a USDC deposit message (Eth → Soma).
///
/// Layout (Sui parity — `create_token_bridge_message_v2` in
/// `crates/sui-framework/packages/bridge/sources/message.move`):
/// ```text
///   senderAddressLength(1) || senderAddress(N) ||
///   targetChain(1) ||
///   targetAddressLength(1) || targetAddress(M) ||
///   tokenType(1) ||
///   amount(8 BE) ||
///   timestampMs(8 BE)
/// ```
/// For Eth → Soma the sender is a 20-byte Eth address and the target is a
/// 32-byte Soma address, so the total is `1+20+1+1+32+1+8+8 = 72 bytes`.
///
/// The framing bytes (`senderAddressLength`, `targetAddressLength`) keep
/// the format forward-compatible — if Soma ever bridges to a chain with
/// different address widths the decoder doesn't change. The Eth-side
/// Solidity decoder asserts the lengths are 20/32 to lock in the current
/// invariants.
pub fn encode_deposit_payload(
    sender_eth_address: &[u8; 20],
    target_chain: BridgeChainId,
    recipient: &SomaAddress,
    token_type: u8,
    amount: u64,
    timestamp_ms: u64,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(72);
    payload.push(ETH_ADDRESS_LENGTH); // 20
    payload.extend_from_slice(sender_eth_address);
    payload.push(target_chain.as_u8());
    payload.push(SOMA_ADDRESS_LENGTH); // 32
    payload.extend_from_slice(recipient.as_ref());
    payload.push(token_type);
    payload.extend_from_slice(&amount.to_be_bytes());
    payload.extend_from_slice(&timestamp_ms.to_be_bytes());
    payload
}

/// Encode the V2 payload for a USDC withdrawal message (Soma → Eth).
///
/// Same layout as [`encode_deposit_payload`] but the sender is a 32-byte
/// Soma address and the target is a 20-byte Eth address, so the total
/// is `1+32+1+1+20+1+8+8 = 72 bytes`.
pub fn encode_withdraw_payload(
    sender: &SomaAddress,
    target_chain: BridgeChainId,
    eth_recipient: &[u8; 20],
    token_type: u8,
    amount: u64,
    timestamp_ms: u64,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(72);
    payload.push(SOMA_ADDRESS_LENGTH); // 32
    payload.extend_from_slice(sender.as_ref());
    payload.push(target_chain.as_u8());
    payload.push(ETH_ADDRESS_LENGTH); // 20
    payload.extend_from_slice(eth_recipient);
    payload.push(token_type);
    payload.extend_from_slice(&amount.to_be_bytes());
    payload.extend_from_slice(&timestamp_ms.to_be_bytes());
    payload
}

/// Encode the payload for an emergency operation message.
/// `op_code(1 byte)`
pub fn encode_emergency_payload(op_code: EmergencyOpCode) -> Vec<u8> {
    vec![op_code as u8]
}

/// Encode the payload for an [`BridgeMessageType::UpdateCommitteeBlocklist`] message.
/// `blocklist_type(1) || count(1) || eth_address(20) * count`
///
/// Members are identified by their derived 20-byte Ethereum address so the
/// Solidity-side bridge contract can verify the message against `ecrecover`
/// outputs without holding the full pubkey list. Mirrors sui-bridge's
/// `BlocklistCommitteeAction` encoding.
pub fn encode_blocklist_payload(
    blocklist_type: BlocklistType,
    eth_addresses: &[[u8; 20]],
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(2 + eth_addresses.len() * 20);
    payload.push(blocklist_type as u8);
    payload.push(eth_addresses.len() as u8);
    for addr in eth_addresses {
        payload.extend_from_slice(addr);
    }
    payload
}

/// Encode the payload for a [`BridgeMessageType::LimitUpdate`] message.
/// `sending_chain_id(1) || new_usd_limit(8 BE)` = 9 bytes.
///
/// The receiving chain id is in the message header (i.e. the route is
/// `sending_chain_id → header chainID`), so it's not repeated here.
pub fn encode_limit_update_payload(
    sending_chain_id: BridgeChainId,
    new_usd_limit: u64,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(9);
    payload.push(sending_chain_id.as_u8());
    payload.extend_from_slice(&new_usd_limit.to_be_bytes());
    payload
}

/// Encode the payload for an [`BridgeMessageType::EvmContractUpgrade`] message
/// using Solidity ABI encoding. Mirrors sui-bridge's
/// `(proxy, new_impl, &call_data).abi_encode_params()`.
///
/// Layout (32-byte aligned, big-endian):
///   - proxy_address, left-padded to 32 bytes
///   - new_impl_address, left-padded to 32 bytes
///   - offset to call_data tail (= 0x60, three head slots)
///   - call_data length as uint256
///   - call_data, zero-padded to a multiple of 32 bytes
///
/// This is the only Soma bridge action that uses ABI (not packed) encoding,
/// so the Solidity-side bridge contract can `abi.decode((address, address, bytes), …)`
/// and forward to the proxy's `upgradeToAndCall` without manual parsing.
pub fn encode_evm_contract_upgrade_payload(
    proxy: &[u8; 20],
    new_impl: &[u8; 20],
    call_data: &[u8],
) -> Vec<u8> {
    let padded_data_len = call_data.len().div_ceil(32) * 32;
    let mut payload = Vec::with_capacity(32 * 4 + padded_data_len);

    // proxy: 32 bytes left-padded
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(proxy);
    // new_impl: 32 bytes left-padded
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(new_impl);
    // offset to call_data tail (3 head slots × 32 bytes = 0x60)
    let mut offset = [0u8; 32];
    offset[31] = 0x60;
    payload.extend_from_slice(&offset);
    // call_data length as uint256 (we cap at u64::MAX, plenty for any realistic calldata)
    let mut len_word = [0u8; 32];
    len_word[24..].copy_from_slice(&(call_data.len() as u64).to_be_bytes());
    payload.extend_from_slice(&len_word);
    // call_data, zero-padded to 32-byte boundary
    payload.extend_from_slice(call_data);
    payload.resize(payload.len() + (padded_data_len - call_data.len()), 0);

    payload
}

/// The Soma bridge chain ID. Used in message encoding to prevent cross-chain replay.
/// Testnet and mainnet should use different values; this defaults to `SomaCustom`
/// for development. Production deployments swap to `SomaMainnet` / `SomaTestnet`.
pub const SOMA_BRIDGE_CHAIN_ID: BridgeChainId = BridgeChainId::SomaCustom;

/// Domain-separation prefix for [`derive_bridge_record_id`]. Distinct from
/// the message-signing prefix so a record ID can't be mistaken for a
/// signed message hash.
const BRIDGE_RECORD_ID_PREFIX: &[u8] = b"SOMA_BRIDGE_RECORD/";

/// Derive the deterministic [`ObjectID`] for a bridge record (deposit or
/// withdrawal) keyed by `(source_chain_id, message_type, nonce)`.
///
/// Mirrors Sui's `BridgeMessageKey { source_chain, message_type, bridge_seq_num }`
/// (used as the `LinkedTable` key inside `BridgeInner`). Soma stores each
/// record as a separate shared object instead of a Move table entry, but
/// the addressing scheme is identical.
///
/// Replay protection: an executor refuses to create a second record with
/// the same `(source_chain, message_type, nonce)` because the derived ID
/// would already exist in the object store.
pub fn derive_bridge_record_id(
    source_chain_id: BridgeChainId,
    msg_type: BridgeMessageType,
    nonce: u64,
) -> ObjectID {
    let mut input = Vec::with_capacity(BRIDGE_RECORD_ID_PREFIX.len() + 1 + 1 + 8);
    input.extend_from_slice(BRIDGE_RECORD_ID_PREFIX);
    input.push(source_chain_id.as_u8());
    input.push(msg_type as u8);
    input.extend_from_slice(&nonce.to_be_bytes());
    let digest = Keccak256::digest(&input);
    ObjectID::new(digest.digest)
}

/// Bridge state stored in SystemState, tracking USDC bridge between Ethereum and Soma.
///
/// Per-deposit audit records live as separate [`BridgeRecord`] objects in
/// the regular object store, keyed by [`derive_bridge_record_id`] —
/// **never in this struct**. The queryable history is "the BridgeRecord
/// at the derived ID exists".
///
/// Replay defense is a single-u64 watermark plus an auto-draining set
/// for tolerating out-of-order delivery. Mirrors Sui's
/// `LinkedTable<MessageKey, BridgeRecord>` membership semantics:
/// once a nonce has been processed it stays in the set forever, and
/// any later attempt to reuse it is rejected. Per-deposit
/// [`BridgeRecord`] objects carry the audit data; this set is purely
/// the replay-defense oracle, growing by 8 bytes per deposit.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BridgeState {
    pub paused: bool,
    pub next_withdrawal_nonce: u64,
    pub bridge_committee: BridgeCommittee,
    /// Set of every deposit nonce ever processed. Grows monotonically.
    /// Matches Sui's LinkedTable membership semantics (Sui's table also
    /// grows unbounded with deposit volume).
    pub processed_deposit_nonces: BTreeSet<u64>,
    /// Per-message-type monotonic sequence numbers for system messages
    /// (emergency op, committee update, blocklist, etc.). Mirrors
    /// `BridgeInner.sequence_nums: VecMap<u8, u64>` in Sui's `bridge.move`.
    ///
    /// Each governance executor reads the expected nonce for its message
    /// type, asserts the incoming `nonce == expected`, then increments.
    /// This prevents replay of a quorum-signed pause / blocklist / etc.
    /// from a previous epoch — the threat that the original audit flagged
    /// when emergency-op was hardcoded to nonce=0.
    ///
    /// Token transfers do **not** consume from this map — their replay
    /// defense is the deposit watermark above (for inbound) and the
    /// `PendingWithdrawal` object existence (for outbound).
    #[serde(default)]
    pub system_message_seq_nums: BTreeMap<BridgeMessageType, u64>,
    /// Per-validator pre-registered bridge keys, populated by the
    /// `BridgeRegisterBridgeKey` tx. Each registration carries the 33-byte
    /// compressed secp256k1 pubkey the validator will sign bridge messages
    /// with, plus the public RPC URL its bridge node listens on.
    ///
    /// At epoch boundaries [`Self::try_rotate_committee`] reads the new
    /// validator set + this map and rebuilds `bridge_committee` weighted
    /// by validator voting power. Mirrors Sui's
    /// `committee::register` + `try_create_next_committee` flow.
    ///
    /// Cleared on each successful rotation; validators must re-register
    /// each epoch they want a seat (matches Sui's `member_registrations`
    /// clear-on-success behavior).
    #[serde(default)]
    pub bridge_registrations: BTreeMap<SomaAddress, BridgeRegistration>,
}

/// A validator's pre-registered bridge keypair + endpoint, awaiting
/// committee rotation. Mirrors Sui's `CommitteeMemberRegistration`.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BridgeRegistration {
    /// 33-byte compressed secp256k1 public key. Validated to be a real
    /// secp256k1 point at registration time so the rotation logic doesn't
    /// have to.
    pub bridge_pubkey: BridgePubkey,
    /// Public URL the validator's bridge node listens on for the
    /// pull-style sig-collection HTTP endpoint (Stage 6).
    pub http_url: String,
}

impl BridgeState {
    pub fn new(committee: BridgeCommittee) -> Self {
        Self {
            paused: false,
            next_withdrawal_nonce: 0,
            bridge_committee: committee,
            processed_deposit_nonces: BTreeSet::new(),
            system_message_seq_nums: BTreeMap::new(),
            bridge_registrations: BTreeMap::new(),
        }
    }

    /// Returns true iff `nonce` has already been processed.
    pub fn is_deposit_nonce_processed(&self, nonce: u64) -> bool {
        self.processed_deposit_nonces.contains(&nonce)
    }

    /// Record a freshly-processed deposit nonce. Mirrors Sui's
    /// `LinkedTable.push_back` for inbound records.
    pub fn record_processed_deposit_nonce(&mut self, nonce: u64) {
        self.processed_deposit_nonces.insert(nonce);
    }

    /// Attempt to rotate the bridge committee using `bridge_registrations`
    /// and the per-validator voting powers from `validator_voting_power`.
    ///
    /// For each registered validator that's still in the active set, build
    /// a [`BridgeMember`] with the registered pubkey and the validator's
    /// current voting power. If the **sum of participating voting power**
    /// is at least `min_stake_participation_bps` of the total active stake,
    /// the rotation succeeds: registrations are cleared and `bridge_committee`
    /// is replaced. Otherwise this is a **silent no-op** — the prior
    /// committee remains in force (matches Sui's `try_create_next_committee`,
    /// which prefers committee continuity over end-of-epoch failures).
    ///
    /// Returns `true` iff rotation occurred.
    pub fn try_rotate_committee(
        &mut self,
        validator_voting_power: &BTreeMap<SomaAddress, u64>,
        min_stake_participation_bps: u64,
    ) -> bool {
        let total_active_stake: u64 = validator_voting_power.values().copied().sum();
        if total_active_stake == 0 {
            return false;
        }

        // Build prospective member set from registrations whose owner is
        // still in the active validator set. Keyed by pubkey to match
        // Sui's `members: VecMap<vector<u8>, CommitteeMember>`.
        let mut new_members: BTreeMap<BridgePubkey, BridgeMember> = BTreeMap::new();
        let mut participation: u64 = 0;
        for (validator_addr, reg) in &self.bridge_registrations {
            let Some(&power) = validator_voting_power.get(validator_addr) else {
                continue; // validator no longer active
            };
            new_members.insert(
                reg.bridge_pubkey.clone(),
                BridgeMember {
                    soma_address: *validator_addr,
                    voting_power: power,
                    http_url: reg.http_url.clone(),
                    is_blocklisted: false,
                },
            );
            participation = participation.saturating_add(power);
        }

        // Quorum-of-participation check: sum of registered active stake
        // must be ≥ min_stake_participation_bps of total active stake.
        // Using bps (10000 = 100%) avoids float arithmetic.
        let participation_bps = participation
            .saturating_mul(10_000)
            .checked_div(total_active_stake)
            .unwrap_or(0);
        if participation_bps < min_stake_participation_bps {
            // Silent no-op — prior committee stays, no event emitted.
            return false;
        }

        // Carry over the existing thresholds so config doesn't drift between
        // committees within the same protocol version.
        let prior = &self.bridge_committee;
        self.bridge_committee = BridgeCommittee {
            members: new_members,
            threshold_deposit: prior.threshold_deposit,
            threshold_withdraw: prior.threshold_withdraw,
            threshold_pause: prior.threshold_pause,
            threshold_unpause: prior.threshold_unpause,
            threshold_blocklist: prior.threshold_blocklist,
            threshold_limit_update: prior.threshold_limit_update,
            threshold_evm_upgrade: prior.threshold_evm_upgrade,
        };
        self.bridge_registrations.clear();
        true
    }

    /// Return the next expected sequence number for a system-message
    /// type. Lazily initializes to 0 on first read.
    ///
    /// Mirrors Sui's `BridgeInner::get_current_seq_num_and_increment`,
    /// minus the increment — callers consume separately via
    /// [`Self::consume_system_message_seq`] only after verification
    /// succeeds, so a failing tx doesn't burn a nonce.
    pub fn expected_system_message_seq(&self, msg_type: BridgeMessageType) -> u64 {
        self.system_message_seq_nums
            .get(&msg_type)
            .copied()
            .unwrap_or(0)
    }

    /// Increment the sequence counter for `msg_type`. Called only after
    /// the executor has verified signatures and is committing the action,
    /// so a bad cert doesn't waste the nonce.
    pub fn consume_system_message_seq(&mut self, msg_type: BridgeMessageType) {
        let next = self
            .system_message_seq_nums
            .get(&msg_type)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        self.system_message_seq_nums.insert(msg_type, next);
    }

}

/// On-chain audit record for a successful Eth→Soma USDC deposit.
///
/// Created atomically by `BridgeDeposit` alongside the USDC mint; owner is
/// `Immutable` (created once, never mutated). The object's deterministic ID
/// `derive_bridge_record_id(source_chain, UsdcDeposit, nonce)` doubles as
/// the replay defense: a second deposit with the same `(chain, nonce)` would
/// collide on the same ID and the executor refuses.
///
/// Because the record persists permanently in the object store, anyone can
/// look up "did the bridge ever process Eth tx X with nonce Y?" without
/// trusting off-chain indexers — mirrors Sui's `LinkedTable` queryability
/// but using Soma's flat object addressing.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct BridgeRecord {
    pub id: ObjectID,
    /// Source chain that initiated the deposit (Eth in V1).
    pub source_chain_id: BridgeChainId,
    /// Per-source-chain monotonic deposit nonce.
    pub nonce: u64,
    /// Eth tx hash that emitted the originating Deposit event — load-bearing
    /// for cross-chain auditability.
    pub eth_tx_hash: [u8; 32],
    pub recipient: SomaAddress,
    pub amount: u64,
    /// Soma epoch at which the record was created (= when USDC was minted).
    pub created_at_epoch: u64,
    /// Soma wall-clock at creation, in milliseconds.
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BridgeCommittee {
    /// Committee members, keyed by their 33-byte compressed secp256k1
    /// pubkey ([`BridgePubkey`]) so verification can ecrecover and look
    /// up directly, mirroring Sui's
    /// `members: VecMap<vector<u8>, CommitteeMember>`.
    pub members: BTreeMap<BridgePubkey, BridgeMember>,
    /// f+1, ~3334/10000
    pub threshold_deposit: u64,
    /// f+1, ~3334/10000
    pub threshold_withdraw: u64,
    /// ~450/10000 — small minority can pause in a crisis.
    pub threshold_pause: u64,
    /// 5001/10000 (50.01%) — majority required to resume. Matches Sui's
    /// `bridge::message::required_voting_power(emergency_op::UNPAUSE) = 5001`.
    pub threshold_unpause: u64,
    /// Majority required to blocklist or unblocklist a committee member.
    pub threshold_blocklist: u64,
    /// Majority required to update a route's USD/day transfer limit.
    pub threshold_limit_update: u64,
    /// Majority required to ship an Ethereum-side contract upgrade.
    pub threshold_evm_upgrade: u64,
}

impl BridgeCommittee {
    pub fn empty() -> Self {
        Self {
            members: BTreeMap::new(),
            threshold_deposit: 3334,
            threshold_withdraw: 3334,
            threshold_pause: 450,
            threshold_unpause: 5001,
            threshold_blocklist: 5001,
            threshold_limit_update: 5001,
            threshold_evm_upgrade: 5001,
        }
    }

}

/// Committee member as stored in [`BridgeCommittee::members`]. The
/// 33-byte compressed pubkey is the map key; this struct holds the
/// remaining fields. Mirrors Sui's `CommitteeMember`.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BridgeMember {
    /// Soma validator address that registered this committee seat.
    /// Doesn't participate in sig verification — purely audit/lookup.
    pub soma_address: SomaAddress,
    pub voting_power: u64,
    /// Public HTTP URL the validator's bridge node listens on for
    /// peer sig-request RPCs (Sui's `BridgeAuthority.base_url` parity).
    /// Populated at committee rotation from the corresponding
    /// [`BridgeRegistration::http_url`]. Used by the peer-broadcast
    /// aggregator to dial committee members.
    #[serde(default)]
    pub http_url: String,
    /// Blocklisted members stay in the committee map (so their sigs are
    /// recognized rather than triggering "unknown signer") but contribute
    /// **0 voting power** to threshold checks. Mirrors Sui's
    /// `CommitteeMember.blocklisted` + `committee::execute_blocklist`.
    #[serde(default)]
    pub is_blocklisted: bool,
}

/// Created when a user initiates a USDC withdrawal from Soma to Ethereum.
/// Bridge nodes observe this in checkpoints, collect committee signatures
/// off-chain, then submit a `BridgeAttachWithdrawalSignatures` tx to attach
/// the cert on-chain. Once the cert is attached anyone can fetch the object
/// + cert and submit it to the Eth-side `SomaBridge` Solidity contract to
/// release the locked USDC.
///
/// Mirrors Sui's outbound `BridgeRecord` lifecycle (`bridge::send_token`
/// inserts with `verified_signatures: None`; `bridge::approve_token_transfer`
/// attaches signatures idempotently).
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PendingWithdrawal {
    pub id: ObjectID,
    pub nonce: u64,
    pub sender: SomaAddress,
    pub recipient_eth_address: [u8; 20],
    pub amount: u64,
    pub created_at_ms: u64,
    /// Aggregated committee signatures over the canonical withdrawal
    /// message bytes. `None` until a `BridgeAttachWithdrawalSignatures`
    /// tx attaches a quorum cert; `Some` afterward, at which point the
    /// cert is permanently on-chain and any actor can submit it to Eth.
    pub verified_signatures: Option<WithdrawalCertificate>,
}

/// Quorum-signed authorization to release USDC on Ethereum for a
/// specific [`PendingWithdrawal`].
///
/// `signatures` is a labeled-pubkey envelope: each entry maps a
/// [`BridgePubkey`] to its 65-byte recoverable secp256k1 signature. The
/// on-chain verifier ecrecovers each signature and confirms the recovered
/// pubkey matches the labeled one, then sums non-blocklisted voting power
/// against the configured threshold. Map keys give cheap dedup without
/// rerunning ecrecover.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct WithdrawalCertificate {
    pub signatures: BTreeMap<BridgePubkey, BridgeSignature>,
    /// Epoch at which the cert was attached. Lets the Eth-side contract
    /// reject certs signed by a committee that's been rotated out.
    pub attached_at_epoch: u64,
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

/// Derive a 20-byte Ethereum address from a [`BridgePubkey`].
///
/// Convenience function; equivalent to [`BridgePubkey::to_eth_address`].
/// Kept as a free function for sites that already hold a typed pubkey.
pub fn derive_eth_address(pubkey: &BridgePubkey) -> [u8; 20] {
    pubkey.to_eth_address()
}

/// Sign a bridge message with a secp256k1 keypair using Keccak256 hash.
/// Returns a 65-byte recoverable signature (r[32] + s[32] + v[1]).
pub fn sign_bridge_message(
    keypair: &Secp256k1KeyPair,
    message: &[u8],
) -> Secp256k1RecoverableSignature {
    keypair.sign_recoverable_with_hash::<Keccak256>(message)
}

/// Sign a bridge message with each keypair and return a labeled-pubkey
/// signature envelope. Each entry maps a [`BridgePubkey`] to its 65-byte
/// recoverable secp256k1 signature.
///
/// Mirrors Sui's `BTreeMap<BridgeAuthorityPublicKeyBytes, sig>` cert
/// shape. The on-chain verifier ecrecovers each sig and checks that the
/// recovered pubkey matches the labeled one before accepting.
pub fn build_bridge_signatures(
    signers: &[&Secp256k1KeyPair],
    message: &[u8],
) -> BTreeMap<BridgePubkey, BridgeSignature> {
    let mut out = BTreeMap::new();
    for kp in signers {
        let pk = BridgePubkey::from_keypair(kp);
        let sig = sign_bridge_message(kp, message);
        let sig = BridgeSignature::from_bytes(sig.as_ref())
            .expect("recoverable secp256k1 signature is always 65 bytes");
        out.insert(pk, sig);
    }
    out
}

/// Generate a test bridge committee with real secp256k1 keypairs.
/// Returns `(committee, keypairs)` where each keypair's public key is a
/// member of the committee.
pub fn generate_test_bridge_committee(
    num_members: usize,
) -> (BridgeCommittee, Vec<Secp256k1KeyPair>) {
    use std::collections::BTreeMap;

    let voting_power_each = 10000u64 / num_members as u64;
    let mut members = BTreeMap::new();
    let mut keypairs = Vec::with_capacity(num_members);
    for _ in 0..num_members {
        let kp = Secp256k1KeyPair::generate(&mut rand::thread_rng());
        let pubkey = BridgePubkey::from_keypair(&kp);
        let addr = SomaAddress::random();
        members.insert(
            pubkey,
            BridgeMember {
                soma_address: addr,
                voting_power: voting_power_each,
                http_url: "http://127.0.0.1:9191".to_string(),
                is_blocklisted: false,
            },
        );
        keypairs.push(kp);
    }

    let committee = BridgeCommittee {
        members,
        threshold_deposit: 3334,
        threshold_withdraw: 3334,
        threshold_pause: 450,
        threshold_unpause: 5001,
        threshold_blocklist: 5001,
        threshold_limit_update: 5001,
        threshold_evm_upgrade: 5001,
    };
    (committee, keypairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_decode(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn hex_encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Cross-validates `BridgePubkey::to_eth_address` against a known
    /// sui-bridge test vector. Compressed pubkey `02321ede…aa4` derives
    /// to Eth address `68b43fd9…7c65`.
    #[test]
    fn test_derive_eth_address_known_vector() {
        let compressed = hex_decode("02321ede33d2c2d7a8a152f275a1484edef2098f034121a602cb7d767d38680aa4");
        let pk = BridgePubkey::from_bytes(&compressed).expect("valid compressed pubkey");
        let addr = derive_eth_address(&pk);
        assert_eq!(
            hex_encode(&addr),
            "68b43fd906c0b8f024a18c56e06744f7c6157c65"
        );
    }

    #[test]
    fn test_derive_eth_address_rejects_invalid_pubkey() {
        let bad = vec![0xFFu8; 33];
        assert!(BridgePubkey::from_bytes(&bad).is_err());
    }

    /// Regression test: full UpdateCommitteeBlocklist message encoding.
    /// Locks the byte layout so future changes to this wire format are forced
    /// to be deliberate.
    #[test]
    fn test_encode_blocklist_payload_regression() {
        let pubkey1 = BridgePubkey::from_bytes(&hex_decode("02321ede33d2c2d7a8a152f275a1484edef2098f034121a602cb7d767d38680aa4")).unwrap();
        let pubkey2 = BridgePubkey::from_bytes(&hex_decode("027f1178ff417fc9f5b8290bd8876f0a157a505a6c52db100a8492203ddd1d4279")).unwrap();
        let addr1 = derive_eth_address(&pubkey1);
        let addr2 = derive_eth_address(&pubkey2);

        let payload = encode_blocklist_payload(BlocklistType::Blocklist, &[addr1, addr2]);
        // type=00 || count=02 || addr1(20) || addr2(20)
        assert_eq!(
            hex_encode(&payload),
            "000268b43fd906c0b8f024a18c56e06744f7c6157c65acaef39832cb995c4e049437a3e2ec6a7bad1ab5"
        );

        // Full message bytes: PREFIX || type=04 || version=01 || nonce=129
        //   || chainID(1 byte) || payload
        let msg = encode_bridge_message(
            BridgeMessageType::UpdateCommitteeBlocklist,
            BRIDGE_MESSAGE_VERSION,
            129,
            SOMA_BRIDGE_CHAIN_ID,
            &payload,
        );
        // 19 + 1 + 1 + 8 + 1 + 2 + 40 = 72
        assert_eq!(msg.len(), 72);
        assert!(msg.starts_with(BRIDGE_MESSAGE_PREFIX));
        assert_eq!(msg[19], BridgeMessageType::UpdateCommitteeBlocklist as u8);
        assert_eq!(msg[20], BRIDGE_MESSAGE_VERSION);

        // Unblocklist variant produces a different first payload byte.
        let unblock = encode_blocklist_payload(BlocklistType::Unblocklist, &[addr1]);
        assert_eq!(unblock[0], 0x01);
    }

    #[test]
    fn test_encode_limit_update_payload_regression() {
        // Mirrors sui-bridge's $1M limit test ($1M × USD_MULTIPLIER = 10_000_000_000 = 0x2540be400)
        let payload = encode_limit_update_payload(
            BridgeChainId::EthCustom,
            1_000_000 * USD_MULTIPLIER,
        );
        // sending_chain_id(1) || new_usd_limit(8 BE)
        // EthCustom = 0x0c, $1M*USD_MULTIPLIER big-endian = 00000002540be400
        assert_eq!(hex_encode(&payload), "0c00000002540be400");

        let msg = encode_bridge_message(
            BridgeMessageType::LimitUpdate,
            BRIDGE_MESSAGE_VERSION,
            15,
            SOMA_BRIDGE_CHAIN_ID,
            &payload,
        );
        // 19 + 1 + 1 + 8 + 1 + 9 = 39
        assert_eq!(msg.len(), 39);
        assert_eq!(msg[19], BridgeMessageType::LimitUpdate as u8);
    }

    #[test]
    fn test_encode_evm_contract_upgrade_payload_empty_calldata() {
        let proxy = [0x06u8; 20];
        let new_impl = [0x09u8; 20];
        let payload = encode_evm_contract_upgrade_payload(&proxy, &new_impl, &[]);
        // ABI layout matches sui-bridge's `(addr, addr, bytes).abi_encode_params()`:
        //   proxy padded to 32 || new_impl padded to 32 || offset=0x60 || length=0
        let expected = concat!(
            "0000000000000000000000000606060606060606060606060606060606060606",
            "0000000000000000000000000909090909090909090909090909090909090909",
            "0000000000000000000000000000000000000000000000000000000000000060",
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        assert_eq!(hex_encode(&payload), expected);
        assert_eq!(payload.len(), 128);
    }

    #[test]
    fn test_encode_evm_contract_upgrade_payload_with_calldata() {
        // Matches sui-bridge's `initializeV2()` selector test (0x5cd8a76b).
        let proxy = [0x06u8; 20];
        let new_impl = [0x09u8; 20];
        let call_data = vec![0x5c, 0xd8, 0xa7, 0x6b]; // 4-byte selector
        let payload = encode_evm_contract_upgrade_payload(&proxy, &new_impl, &call_data);
        let expected = concat!(
            "0000000000000000000000000606060606060606060606060606060606060606",
            "0000000000000000000000000909090909090909090909090909090909090909",
            "0000000000000000000000000000000000000000000000000000000000000060",
            "0000000000000000000000000000000000000000000000000000000000000004",
            "5cd8a76b00000000000000000000000000000000000000000000000000000000",
        );
        assert_eq!(hex_encode(&payload), expected);
        assert_eq!(payload.len(), 160); // 4 head slots + 32 bytes for padded calldata
    }

    /// Calldata that crosses 32-byte boundaries should pad to the next multiple of 32.
    #[test]
    fn test_encode_evm_contract_upgrade_padding() {
        let proxy = [0u8; 20];
        let new_impl = [0u8; 20];

        // 33 bytes of calldata → padded to 64.
        let call_data = vec![0xAAu8; 33];
        let payload = encode_evm_contract_upgrade_payload(&proxy, &new_impl, &call_data);
        // 4 head slots (128) + 64 bytes padded data = 192
        assert_eq!(payload.len(), 192);
        // Length field reads 33 (0x21).
        assert_eq!(&payload[120..128], &[0u8, 0, 0, 0, 0, 0, 0, 0x21]);
        // Last 31 bytes of padded calldata are zero.
        assert!(payload[128 + 33..].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_derive_bridge_record_id_is_deterministic() {
        let chain_a = BridgeChainId::EthCustom;
        let chain_b = BridgeChainId::SomaCustom;
        // Same inputs → same ID
        let id1 = derive_bridge_record_id(chain_a, BridgeMessageType::UsdcDeposit, 42);
        let id2 = derive_bridge_record_id(chain_a, BridgeMessageType::UsdcDeposit, 42);
        assert_eq!(id1, id2);

        // Different chain → different ID
        let other_chain = derive_bridge_record_id(chain_b, BridgeMessageType::UsdcDeposit, 42);
        assert_ne!(id1, other_chain);

        // Different message type → different ID
        let other_type = derive_bridge_record_id(chain_a, BridgeMessageType::UsdcWithdraw, 42);
        assert_ne!(id1, other_type);

        // Different nonce → different ID
        let other_nonce = derive_bridge_record_id(chain_a, BridgeMessageType::UsdcDeposit, 43);
        assert_ne!(id1, other_nonce);
    }

    #[test]
    fn test_derive_bridge_record_id_known_vector() {
        // Lock the byte derivation so a future change to the ID formula
        // would break this test (the ID is part of consensus — anyone with
        // the same (chain, msg_type, nonce) must derive the same address).
        let id = derive_bridge_record_id(SOMA_BRIDGE_CHAIN_ID, BridgeMessageType::UsdcWithdraw, 0);
        // Compute expected manually:
        // keccak256("SOMA_BRIDGE_RECORD/" || chain_id(1) || msg_type(1) || nonce_be(8))
        let mut input = Vec::new();
        input.extend_from_slice(b"SOMA_BRIDGE_RECORD/");
        input.push(SOMA_BRIDGE_CHAIN_ID.as_u8());
        input.push(1u8); // UsdcWithdraw
        input.extend_from_slice(&0u64.to_be_bytes());
        let expected = Keccak256::digest(&input).digest;
        assert_eq!(id.into_bytes(), expected);
    }

    #[test]
    fn test_replay_defense_records_processed_nonces() {
        // Mirrors Sui's `LinkedTable` membership semantics — the set
        // grows monotonically, every recorded nonce stays "processed"
        // forever, including out-of-order arrivals.
        let mut state = BridgeState::new(BridgeCommittee::empty());

        // Out-of-order delivery: 2, 1, 3, then 0.
        for n in [2u64, 1, 3, 0] {
            assert!(!state.is_deposit_nonce_processed(n));
            state.record_processed_deposit_nonce(n);
            assert!(state.is_deposit_nonce_processed(n));
        }
        assert_eq!(state.processed_deposit_nonces.len(), 4);

        // Re-recording is a no-op (BTreeSet semantics).
        state.record_processed_deposit_nonce(2);
        assert_eq!(state.processed_deposit_nonces.len(), 4);

        // Nonces never recorded are not yet processed.
        assert!(!state.is_deposit_nonce_processed(99));
    }

    #[test]
    fn test_try_rotate_committee_meets_threshold() {
        use crate::base::SomaAddress;
        let mut state = BridgeState::new(BridgeCommittee::empty());
        let v1 = SomaAddress::from([1; 32]);
        let v2 = SomaAddress::from([2; 32]);
        let v3 = SomaAddress::from([3; 32]);
        let kp1 = Secp256k1KeyPair::generate(&mut rand::thread_rng());
        let kp2 = Secp256k1KeyPair::generate(&mut rand::thread_rng());
        let pk1 = BridgePubkey::from_keypair(&kp1);
        let pk2 = BridgePubkey::from_keypair(&kp2);
        for (addr, pk) in [(v1, pk1.clone()), (v2, pk2.clone())] {
            state.bridge_registrations.insert(
                addr,
                BridgeRegistration { bridge_pubkey: pk, http_url: "u".into() },
            );
        }

        // Active set: v1=4000, v2=4000, v3=2000. v1+v2 = 8000/10000 = 80% > 50%.
        let voting = BTreeMap::from([(v1, 4000), (v2, 4000), (v3, 2000)]);
        let rotated = state.try_rotate_committee(&voting, 5001);
        assert!(rotated, "registered stake (80%) should clear 50.01% threshold");
        assert!(state.bridge_registrations.is_empty(), "registrations cleared on success");
        // Members are now keyed by pubkey (matching Sui).
        assert_eq!(state.bridge_committee.members.len(), 2);
        let m1 = &state.bridge_committee.members[&pk1];
        assert_eq!(m1.soma_address, v1);
        assert_eq!(m1.voting_power, 4000);
        assert!(!m1.is_blocklisted);
        let m2 = &state.bridge_committee.members[&pk2];
        assert_eq!(m2.soma_address, v2);
        assert_eq!(m2.voting_power, 4000);
    }

    #[test]
    fn test_try_rotate_committee_silent_no_op_below_threshold() {
        use crate::base::SomaAddress;
        // Pre-populate with a "current" committee we should NOT lose.
        let mut prior_committee = BridgeCommittee::empty();
        let prior_kp = Secp256k1KeyPair::generate(&mut rand::thread_rng());
        let prior_pubkey = BridgePubkey::from_keypair(&prior_kp);
        let prior_addr = SomaAddress::from([9; 32]);
        prior_committee.members.insert(
            prior_pubkey.clone(),
            BridgeMember {
                soma_address: prior_addr,
                voting_power: 10000,
                http_url: String::new(),
                is_blocklisted: false,
            },
        );
        let mut state = BridgeState::new(prior_committee);

        // Only one validator registers, with 1000/10000 = 10% — below 50.01%.
        let v1 = SomaAddress::from([1; 32]);
        let v2 = SomaAddress::from([2; 32]);
        let v1_kp = Secp256k1KeyPair::generate(&mut rand::thread_rng());
        state.bridge_registrations.insert(
            v1,
            BridgeRegistration {
                bridge_pubkey: BridgePubkey::from_keypair(&v1_kp),
                http_url: "u".into(),
            },
        );
        let voting = BTreeMap::from([(v1, 1000), (v2, 9000)]);

        let rotated = state.try_rotate_committee(&voting, 5001);
        assert!(!rotated, "below-threshold registration must not rotate");
        assert!(state.bridge_registrations.contains_key(&v1));
        // Prior committee stays intact, keyed by pubkey.
        assert_eq!(state.bridge_committee.members.len(), 1);
        assert!(state.bridge_committee.members.contains_key(&prior_pubkey));
    }

    #[test]
    fn test_try_rotate_committee_filters_inactive_validators() {
        use crate::base::SomaAddress;
        let mut state = BridgeState::new(BridgeCommittee::empty());
        let v1 = SomaAddress::from([1; 32]);
        let v_kicked = SomaAddress::from([2; 32]);
        let v1_kp = Secp256k1KeyPair::generate(&mut rand::thread_rng());
        let kicked_kp = Secp256k1KeyPair::generate(&mut rand::thread_rng());
        let v1_pk = BridgePubkey::from_keypair(&v1_kp);
        let kicked_pk = BridgePubkey::from_keypair(&kicked_kp);

        state.bridge_registrations.insert(
            v1,
            BridgeRegistration { bridge_pubkey: v1_pk.clone(), http_url: "u".into() },
        );
        state.bridge_registrations.insert(
            v_kicked,
            BridgeRegistration { bridge_pubkey: kicked_pk.clone(), http_url: "u".into() },
        );
        // Active set: v1=10000, v_kicked is gone.
        let voting = BTreeMap::from([(v1, 10000)]);
        let rotated = state.try_rotate_committee(&voting, 5001);
        assert!(rotated);
        assert_eq!(state.bridge_committee.members.len(), 1);
        assert!(state.bridge_committee.members.contains_key(&v1_pk));
        assert!(!state.bridge_committee.members.contains_key(&kicked_pk));
    }

    #[test]
    fn test_committee_thresholds_match_sui_design() {
        let c = BridgeCommittee::empty();
        // Asymmetric pause vs unpause — minority can pause, supermajority required to resume.
        assert!(c.threshold_pause < c.threshold_unpause);
        // All governance actions require majority.
        assert!(c.threshold_blocklist >= 5001);
        assert!(c.threshold_limit_update >= 5001);
        assert!(c.threshold_evm_upgrade >= 5001);
    }
}
