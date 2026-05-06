// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ISomaBridge
/// @notice Public surface of the SomaBridge entry contract — events that the
/// off-chain Soma bridge node syncer indexes, plus the user/quorum entry
/// points it exercises.
interface ISomaBridge {
    /// @notice Emitted on user deposit (Eth → Soma).
    /// @dev The bridge node's `eth_client::parse_deposit_log` reads these fields
    /// from event data. Field order + types MUST match that parser exactly:
    ///   word 0: uint64 nonce                 (right-aligned in the 32-byte slot)
    ///   word 1: address sender               (right-aligned, 20 bytes)
    ///   word 2: uint8 destinationChainID     (right-aligned, 1 byte) — added for Sui parity
    ///   word 3: bytes32 somaRecipient        (full 32 bytes)
    ///   word 4: uint8 tokenType              (right-aligned, 1 byte) — added for Sui parity
    ///   word 5: uint64 amount                (right-aligned, 8 bytes)
    ///   word 6: uint64 timestampMs           (right-aligned, 8 bytes — V2)
    /// NONE are `indexed` — the parser slices `data`, not topics.
    ///
    /// The `destinationChainID` + `tokenType` fields mirror Sui's
    /// `TokensDeposited` event so off-chain indexers can route + filter
    /// without needing extra contract state lookups.
    event TokensDeposited(
        uint64 nonce,
        address sender,
        uint8 destinationChainID,
        bytes32 somaRecipient,
        uint8 tokenType,
        uint64 amount,
        uint64 timestampMs
    );

    /// @notice Emitted on quorum-signed outbound transfer (Soma → Eth release).
    /// Includes the sender on the Soma side + token id for Sui parity (Sui
    /// emits `(sourceChainID, nonce, tokenId, amount, senderAddress, targetAddress)`).
    event BridgedTokensTransferred(
        uint8 sourceChainID,
        uint64 nonce,
        bytes32 somaSender,
        uint8 tokenType,
        uint64 amount,
        address indexed targetAddress
    );
}
