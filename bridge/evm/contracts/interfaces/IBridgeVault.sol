// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title IBridgeVault
/// @notice Owner-gated USDC custody. The owner is the [`SomaBridge`] contract;
/// users never call this directly.
interface IBridgeVault {
    /// @notice Transfer USDC from the vault to `targetAddress`.
    /// @dev Only callable by the owner ([`SomaBridge`]) after a quorum-signed
    /// withdrawal message has been verified.
    function transferUSDC(address targetAddress, uint256 amount) external;
}
