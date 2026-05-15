// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title IBridgeLimiter
/// @notice Daily-rolling-window cap on outbound USDC transfers (Soma → Eth).
/// @dev Tracks per-hour amounts in USD (Soma's USD_MULTIPLIER scale = 1 USD = 1e4).
/// Sliding 24-hour window: an outbound transfer is permitted iff the sum of the
/// last 24 hourly buckets plus the new amount stays under `totalLimit`.
interface IBridgeLimiter {
    /// @notice Would adding `usdcMicroAmount` push the rolling-24h total over the
    /// configured `totalLimit`? Pure view; doesn't mutate.
    function willAmountExceedLimit(uint256 usdcMicroAmount) external view returns (bool);

    /// @notice Account a fresh outbound transfer toward the current hour's bucket.
    /// Reverts if it would exceed the limit. Only callable by the owner
    /// ([`SomaBridge`]) — the bridge calls this *after* it has verified the
    /// quorum-signed withdrawal message but *before* releasing USDC from the vault.
    function recordUSDCTransfer(uint256 usdcMicroAmount) external;

    /// @notice Emitted when a fresh hourly bucket is debited.
    event HourlyTransferAmountUpdated(uint32 hourTimestamp, uint256 usdAmount);

    /// @notice Emitted when the daily limit is changed via a quorum-signed action.
    event LimitUpdated(uint8 sendingChainID, uint64 newUsdLimit);
}
