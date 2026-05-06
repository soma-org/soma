// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title BridgeMessage
/// @notice Defines the canonical message format and decoders for the Soma bridge.
/// @dev This is the Solidity counterpart of `types/src/bridge.rs::encode_bridge_message`
/// on the Rust side; the two MUST stay in sync because the off-chain bridge node
/// signs the bytes this library reconstructs.
///
/// Wire format:
/// ```text
///   PREFIX || type(1) || version(1) || nonce(8 BE) || chainID(1) || payload
/// ```
///
/// Adapted from Sui's `evm-sui-bridge`. Differences from Sui:
/// - Single-token (USDC) — no token id, no decimal adjustment, no token registry
/// - Single direction per message type: `USDC_DEPOSIT` is emitted *from* Eth (and
///   processed off-chain on Soma); `USDC_WITHDRAW` is the message Soma sends *to*
///   Eth for release. They are different message-type bytes; the Eth contract only
///   processes the latter
/// - `COMMITTEE_UPDATE` is a new Soma-only message type; its on-chain handler is
///   deferred (see `decodeCommitteeUpdatePayload` notes below) and committee
///   membership today changes only via blocklist
library BridgeMessage {
    /* ========== MESSAGE TYPES ========== */
    /// @dev These values MUST match `types/src/bridge.rs::BridgeMessageType`.
    /// The Rust side asserts this round-trips via the wire format tests.
    uint8 public constant USDC_DEPOSIT = 0;
    uint8 public constant USDC_WITHDRAW = 1;
    uint8 public constant EMERGENCY_OP = 2;
    uint8 public constant COMMITTEE_UPDATE = 3;
    uint8 public constant BLOCKLIST = 4;
    uint8 public constant UPDATE_LIMIT = 5;
    uint8 public constant UPGRADE = 6;

    /* ========== STAKE THRESHOLDS (BPS) ========== */
    /// @dev Mirror `BridgeCommittee` defaults in `types/src/bridge.rs`.
    /// Total committee stake is 10000 BPS by construction (enforced at init).
    uint32 public constant TRANSFER_STAKE_REQUIRED = 3334;        // ~33.3%
    uint32 public constant FREEZING_STAKE_REQUIRED = 450;         // 4.5% — deliberately low
    uint32 public constant UNFREEZING_STAKE_REQUIRED = 5001;      // >50%
    uint32 public constant BLOCKLIST_STAKE_REQUIRED = 5001;
    uint32 public constant LIMIT_UPDATE_STAKE_REQUIRED = 5001;
    uint32 public constant UPGRADE_STAKE_REQUIRED = 5001;
    uint32 public constant COMMITTEE_UPDATE_STAKE_REQUIRED = 5001;

    /* ========== VERSIONS ========== */
    /// @dev Token transfers (deposit/withdraw) carry a `timestamp_ms` and use V2.
    /// System messages (emergency op, blocklist, etc.) use V1. Matches Sui parity.
    uint8 public constant TOKEN_TRANSFER_VERSION_V2 = 2;
    uint8 public constant SYSTEM_MESSAGE_VERSION = 1;

    /* ========== CANONICAL MESSAGE PREFIX ========== */
    string public constant MESSAGE_PREFIX = "SOMA_BRIDGE_MESSAGE";

    /// @dev Generic message envelope. See module-level comment for wire layout.
    /// @param messageType One of the message-type constants above.
    /// @param version Per-type wire version (`TOKEN_TRANSFER_VERSION_V2` or `SYSTEM_MESSAGE_VERSION`).
    /// @param nonce Per-message-type sequence number (token transfers use their own counter).
    /// @param chainID For token transfers this is the SOURCE chain id; for system messages
    /// it is the chain id of the contract that should process it (i.e. this Eth chain).
    /// @param payload Message-type-specific body. See per-type `decode*` helpers below.
    struct Message {
        uint8 messageType;
        uint8 version;
        uint64 nonce;
        uint8 chainID;
        bytes payload;
    }

    /* ========== ENCODING ========== */

    /// @dev Encode a message to its canonical wire form for hashing/signing.
    /// MUST be byte-identical to the Rust side's `encode_bridge_message`.
    function encodeMessage(Message memory message) internal pure returns (bytes memory) {
        bytes memory header =
            abi.encodePacked(MESSAGE_PREFIX, message.messageType, message.version);
        bytes memory nonce = abi.encodePacked(message.nonce);
        bytes memory chainID = abi.encodePacked(message.chainID);
        return bytes.concat(header, nonce, chainID, message.payload);
    }

    /// @dev Compute the digest that committee members sign over.
    /// Keccak256 of the canonical encoded form.
    function computeHash(Message memory message) internal pure returns (bytes32) {
        return keccak256(encodeMessage(message));
    }

    /* ========== STAKE DISPATCH ========== */

    /// @dev Per-action threshold lookup. Mirrors `BridgeAction::approval_threshold`
    /// on the Rust side. The contract checks the recovered + counted stake against
    /// this return value before accepting a quorum-signed message.
    function getRequiredStake(Message memory message) internal pure returns (uint32) {
        if (message.messageType == USDC_WITHDRAW) {
            return TRANSFER_STAKE_REQUIRED;
        }
        if (message.messageType == EMERGENCY_OP) {
            // Freeze is intentionally cheap (450 BPS), unfreeze is expensive (5001).
            // Decode the op byte to pick the right threshold so a freezing action
            // can land with a single watchdog signature while unfreezing needs
            // majority consent.
            bool isFreezing = decodeEmergencyOpPayload(message.payload);
            return isFreezing ? FREEZING_STAKE_REQUIRED : UNFREEZING_STAKE_REQUIRED;
        }
        if (message.messageType == BLOCKLIST) return BLOCKLIST_STAKE_REQUIRED;
        if (message.messageType == UPDATE_LIMIT) return LIMIT_UPDATE_STAKE_REQUIRED;
        if (message.messageType == UPGRADE) return UPGRADE_STAKE_REQUIRED;
        if (message.messageType == COMMITTEE_UPDATE) return COMMITTEE_UPDATE_STAKE_REQUIRED;
        revert("BridgeMessage: Invalid message type");
    }

    /* ========== PAYLOAD DECODERS ========== */

    /// @notice Token id for USDC. Matches Sui's `BridgeMessage.USDC = 3`.
    /// The Eth-side decoder asserts the incoming `tokenType` equals this
    /// constant; Soma is USDC-only today, and the assertion future-proofs
    /// the contract if/when the wire format expands to multi-token.
    uint8 public constant USDC_TOKEN_TYPE = 3;

    /// @notice Maturity window for limiter bypass. A withdrawal whose
    /// signed `timestampMs` is older than this window is released without
    /// debiting the rolling-24h cap. Mirrors Sui's `BridgeUtilsV2.isMatureMessage`.
    /// Rationale: a quorum-signed message that sat in the mempool / WAL
    /// for days shouldn't burn today's bandwidth — operators expect the
    /// cap to throttle *fresh* outflow, not catch up an old backlog.
    uint256 public constant MATURITY_WINDOW_SECONDS = 48 * 3600;

    /// @notice `true` if a message's seconds-since-epoch timestamp is at
    /// least `MATURITY_WINDOW_SECONDS` older than `currentTimestamp`.
    function isMatureMessage(uint256 messageTimestampSeconds, uint256 currentTimestamp)
        internal
        pure
        returns (bool)
    {
        return currentTimestamp > messageTimestampSeconds + MATURITY_WINDOW_SECONDS;
    }

    /// @notice USDC withdrawal payload (Soma → Eth) — V2 token transfer
    /// format matching Sui's wire layout.
    ///
    /// Layout (Sui parity — `create_token_bridge_message_v2` in
    /// `crates/sui-framework/packages/bridge/sources/message.move`):
    /// ```text
    ///   senderAddressLength(1)  // = 32 (Soma)
    ///   senderAddress(32)
    ///   targetChain(1)          // an Eth chain id
    ///   targetAddressLength(1)  // = 20 (Eth)
    ///   targetAddress(20)
    ///   tokenType(1)            // = USDC = 3
    ///   amount(8 BE)
    ///   timestampMs(8 BE)
    /// ```
    /// Total: 72 bytes. The framing-byte assertions
    /// (`senderAddressLength == 32`, `targetAddressLength == 20`) lock in
    /// the current direction (Soma→Eth) so a payload encoded the other
    /// way around can't trick the contract into mis-routing.
    function decodeUsdcWithdrawPayload(bytes memory payload)
        internal
        pure
        returns (
            bytes32 somaSender,
            uint8 targetChain,
            address ethRecipient,
            uint8 tokenType,
            uint64 amount,
            uint64 timestampMs
        )
    {
        require(
            payload.length == 72,
            "BridgeMessage: USDC withdraw payload must be 72 bytes (V2)"
        );

        // senderAddressLength @ byte 0 — Soma addresses are 32 bytes.
        require(
            uint8(payload[0]) == 32,
            "BridgeMessage: Soma sender must be 32 bytes"
        );
        // senderAddress @ bytes 1..33 — load 32 bytes starting at offset 33.
        assembly {
            somaSender := mload(add(payload, add(0x20, 1)))
        }

        // targetChain @ byte 33
        targetChain = uint8(payload[33]);

        // targetAddressLength @ byte 34 — Eth addresses are 20 bytes.
        require(
            uint8(payload[34]) == 20,
            "BridgeMessage: Eth target must be 20 bytes"
        );
        // targetAddress @ bytes 35..55.
        assembly {
            ethRecipient := shr(96, mload(add(payload, add(0x20, 35))))
        }

        // tokenType @ byte 55
        tokenType = uint8(payload[55]);

        // amount @ bytes 56..64
        assembly {
            amount := shr(192, mload(add(payload, add(0x20, 56))))
        }

        // timestampMs @ bytes 64..72
        assembly {
            timestampMs := shr(192, mload(add(payload, add(0x20, 64))))
        }
    }

    /// @notice Emergency op payload — single op byte, 0 = freeze, 1 = unfreeze.
    /// @dev Mirrors `encode_emergency_payload`.
    function decodeEmergencyOpPayload(bytes memory payload) internal pure returns (bool) {
        require(payload.length == 1, "BridgeMessage: Emergency payload must be 1 byte");
        uint8 op = uint8(payload[0]);
        require(op <= 1, "BridgeMessage: Invalid emergency op code");
        return op == 0; // true == freezing
    }

    /// @notice Blocklist update payload.
    /// @dev Layout: `blocklistType(1) || count(1) || ethAddr(20)*count`.
    /// Mirrors `encode_blocklist_payload` — Soma pre-derives Eth addresses on the
    /// Rust side (see `derive_eth_address`) so the on-chain decoder doesn't need
    /// to do secp256k1 pubkey decompression.
    /// @return blocklisted `true` to add to blocklist, `false` to remove.
    /// @return members Eth addresses to update.
    function decodeBlocklistPayload(bytes memory payload)
        internal
        pure
        returns (bool blocklisted, address[] memory members)
    {
        require(payload.length >= 2, "BridgeMessage: Blocklist payload too short");
        uint8 blocklistType = uint8(payload[0]);
        require(blocklistType <= 1, "BridgeMessage: Invalid blocklist type");
        blocklisted = (blocklistType == 0);

        uint8 count = uint8(payload[1]);
        require(
            payload.length == uint256(2) + uint256(count) * 20,
            "BridgeMessage: Blocklist payload length mismatch"
        );

        members = new address[](count);
        for (uint8 i = 0; i < count; i++) {
            address member;
            // payload[2 + i*20 .. 2 + i*20 + 20]. mload reads 32 bytes ending
            // at byte (2 + i*20 + 20); shift right 96 to keep the high 20.
            uint256 offset = 0x20 + 2 + uint256(i) * 20;
            assembly {
                member := shr(96, mload(add(payload, offset)))
            }
            members[i] = member;
        }
    }

    /// @notice Limit-update payload.
    /// @dev Layout: `sendingChainID(1) || newUsdLimit(8 BE)` = 9 bytes.
    /// `newUsdLimit` uses Soma's USD_MULTIPLIER (1 USD = 10000).
    function decodeUpdateLimitPayload(bytes memory payload)
        internal
        pure
        returns (uint8 sendingChainID, uint64 newUsdLimit)
    {
        require(payload.length == 9, "BridgeMessage: Limit payload must be 9 bytes");
        sendingChainID = uint8(payload[0]);
        // Bytes 1..9. mload from offset 1 reads 32 bytes; shift right 192 to keep
        // the 8 BE bytes in the low position of the resulting word.
        assembly {
            newUsdLimit := shr(192, mload(add(payload, add(0x20, 1))))
        }
    }

    /// @notice EVM-upgrade payload.
    /// @dev Encoded with standard Solidity ABI (matches `abi.encode(proxy, impl, callData)`).
    /// Mirrors `encode_evm_contract_upgrade_payload` on the Rust side, which uses
    /// `abi_encode_params` against the same triple.
    function decodeUpgradePayload(bytes memory payload)
        internal
        pure
        returns (address proxy, address implementation, bytes memory callData)
    {
        (proxy, implementation, callData) =
            abi.decode(payload, (address, address, bytes));
    }

    /* ========== COMMITTEE UPDATE (deferred on-chain handler) ========== */
    //
    // `COMMITTEE_UPDATE` payload layout on Soma (`types/src/bridge.rs::encode_payload`):
    //   `count(4 BE) || (pubkey33 || power8 BE)*`
    //
    // Implementing the on-chain decoder requires deriving the 20-byte Eth address
    // from each 33-byte compressed secp256k1 pubkey, which needs a modular-sqrt
    // routine (not available natively in Solidity). Two clean paths:
    //   (a) Change the wire format to ship pre-derived Eth addresses + powers
    //       — mirrors the `BLOCKLIST` wire format. Cheap on chain, small Rust
    //       diff (Soma already has `derive_eth_address`).
    //   (b) Add a SECP256K1 decompression library in Solidity.
    //
    // Today the Eth contract supports committee membership changes via
    // `BLOCKLIST` only — matches Sui's design (Sui has no on-chain committee
    // update at all, blocklist is the only knob). Wholesale committee rotation
    // is deferred until the wire format is finalized.
}
