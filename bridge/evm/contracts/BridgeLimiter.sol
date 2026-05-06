// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";

import "./interfaces/IBridgeLimiter.sol";
import "./utils/CommitteeUpgradeable.sol";

/// @title BridgeLimiter
/// @notice Sliding-window rate limiter for outbound USDC transfers (Soma → Eth).
///
/// Adapted from Sui's `BridgeLimiter.sol`. Differences:
/// - Single token (USDC, 6 decimals) — no token id, no price oracle, no
///   per-token tracking. The limit is denominated in USD (Soma's
///   `USD_MULTIPLIER = 10000` scale; 1 USD = 1e4).
/// - No `updateAssetPriceWithSignatures` (no asset price to update).
/// - Hourly buckets in a `mapping(uint32 => uint256)`, summed over the
///   trailing 24-hour window on every check.
contract BridgeLimiter is IBridgeLimiter, CommitteeUpgradeable, OwnableUpgradeable {
    /// @notice USDC decimals. Hardcoded — USDC has been 6 decimals since
    /// inception on every major chain. If a future Soma deployment targets
    /// a different stablecoin, this becomes a constructor argument.
    uint8 public constant USDC_DECIMALS = 6;

    /// @notice Hour-timestamp (block.timestamp / 1h) → total USDC amount
    /// bridged in that hour, in USD-multiplier units (1 USD = 1e4).
    mapping(uint32 => uint256) public hourlyTransferAmount;

    /// @notice Configured per-24h USD cap. In Soma's USD_MULTIPLIER scale.
    /// Updated via quorum-signed `UPDATE_LIMIT` messages.
    uint64 public totalLimit;

    /// @notice Initialize the limiter against a committee.
    /// @dev The owner is `msg.sender` (the deployer); the deployer must
    /// `transferOwnership(address(somaBridge))` immediately so only the
    /// bridge can call `recordUSDCTransfer`.
    function initialize(address _committee, uint64 _totalLimit) external initializer {
        __CommitteeUpgradeable_init(_committee);
        __UUPSUpgradeable_init();
        __Ownable_init(msg.sender);
        totalLimit = _totalLimit;
    }

    /* ========== VIEWS ========== */

    /// @notice Current hour bucket key — `block.timestamp / 1 hours`.
    function currentHour() public view returns (uint32) {
        return uint32(block.timestamp / 1 hours);
    }

    /// @notice Convert a raw USDC amount (6 decimals) to Soma's USD-multiplier
    /// scale (4 decimals). 1 USDC = 1_000_000 raw = 10_000 USD-scale.
    /// @dev `amount * 10000 / 10^6 = amount / 100`. Done as a multiply-then-divide
    /// so callers can pass any amount without precision loss inside the cap.
    function usdcToUsdScale(uint256 usdcMicroAmount) public pure returns (uint256) {
        return (usdcMicroAmount * 10_000) / (10 ** USDC_DECIMALS);
    }

    /// @notice Sum of the 24 trailing hourly buckets, in USD-multiplier units.
    /// @dev `h - i` is allowed to underflow on a fresh chain (test timestamp = 0).
    /// `hourlyTransferAmount` is a mapping with default 0, so reading a wrapped
    /// key still returns 0 — the math is safe under `unchecked`. In production
    /// Unix `block.timestamp` is many orders of magnitude past 24 hours, so
    /// this branch never triggers anyway.
    function calculateWindowAmount() public view returns (uint256 total) {
        uint32 h = currentHour();
        unchecked {
            for (uint32 i = 0; i < 24; i++) {
                total += hourlyTransferAmount[h - i];
            }
        }
    }

    /// @inheritdoc IBridgeLimiter
    function willAmountExceedLimit(uint256 usdcMicroAmount)
        public
        view
        override
        returns (bool)
    {
        return calculateWindowAmount() + usdcToUsdScale(usdcMicroAmount) > totalLimit;
    }

    /* ========== EXTERNAL ========== */

    /// @inheritdoc IBridgeLimiter
    function recordUSDCTransfer(uint256 usdcMicroAmount) external override onlyOwner {
        require(usdcMicroAmount > 0, "BridgeLimiter: Amount must be > 0");
        uint256 usdAmount = usdcToUsdScale(usdcMicroAmount);
        require(
            calculateWindowAmount() + usdAmount <= totalLimit,
            "BridgeLimiter: Exceeds rolling 24h limit"
        );

        uint32 h = currentHour();
        // Best-effort garbage collection of the 25-hour-old bucket. Saves
        // a storage slot on hot paths; safe to skip if the slot is already
        // zero. Same underflow rationale as `calculateWindowAmount` above.
        unchecked {
            uint32 stale = h - 25;
            if (hourlyTransferAmount[stale] > 0) {
                delete hourlyTransferAmount[stale];
            }
        }
        hourlyTransferAmount[h] += usdAmount;
        emit HourlyTransferAmountUpdated(h, usdAmount);
    }

    /// @notice Update the 24-hour USD cap via a quorum-signed `UPDATE_LIMIT`.
    /// @dev `sendingChainID` in the payload is the source chain (Soma); we
    /// don't currently route limits per-source-chain so the value is
    /// surfaced in the event for off-chain consumption but not consulted
    /// for enforcement. Sui's contract has the same shape.
    function updateLimitWithSignatures(
        bytes[] memory signatures,
        BridgeMessage.Message memory message
    )
        external
        nonReentrant
        verifyMessageAndSignatures(message, signatures, BridgeMessage.UPDATE_LIMIT)
    {
        (uint8 sendingChainID, uint64 newLimit) =
            BridgeMessage.decodeUpdateLimitPayload(message.payload);
        totalLimit = newLimit;
        emit LimitUpdated(sendingChainID, newLimit);
    }
}
