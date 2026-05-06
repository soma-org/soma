// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../utils/BridgeMessage.sol";

/// @title IBridgeCommittee
/// @notice Public surface of [`BridgeCommittee`] — the ecrecover-based stake
/// counter that every quorum-signed message goes through.
interface IBridgeCommittee {
    /// @notice Verifies that the supplied signatures account for the required
    /// stake against the given canonical message.
    /// @dev Reverts on insufficient stake. Skips blocklisted members and
    /// duplicate signatures.
    /// @param signatures 65-byte recoverable secp256k1 signatures, one per signer.
    /// @param message Canonical Soma bridge message.
    function verifySignatures(bytes[] memory signatures, BridgeMessage.Message memory message)
        external
        view;

    /// @notice Eth chain id (one of {10=Mainnet, 11=Sepolia, 12=Custom}).
    function chainID() external view returns (uint8);

    /// @notice Emitted when the blocklist is updated via a quorum-signed action.
    /// @param newMembers Eth addresses whose blocklist flag changed.
    /// @param isBlocklisted Final state — `true` if now blocklisted.
    event BlocklistUpdated(address[] newMembers, bool isBlocklisted);
}
