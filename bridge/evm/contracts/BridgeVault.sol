// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

import "./interfaces/IBridgeVault.sol";

/// @title BridgeVault
/// @notice USDC-only custody contract owned by the [`SomaBridge`].
///
/// Sui's `BridgeVault` is multi-token (USDC, USDT, WBTC, WETH, ETH) and
/// includes a `wETH` unwrap path for native ETH. Soma is USDC-only — the
/// vault holds exactly one ERC20 (USDC, set at construction) and exposes
/// a single owner-gated `transferUSDC` to release funds when the bridge
/// has verified a quorum-signed withdrawal.
///
/// Construction note: `Ownable(msg.sender)` sets the deployer as the owner;
/// the deployer immediately transfers ownership to the `SomaBridge` proxy
/// via `transferOwnership` once the bridge address is known (see the
/// deploy script). The bridge is the only address that can call
/// `transferUSDC`; users never touch this contract.
contract BridgeVault is Ownable, IBridgeVault {
    /// @notice The USDC ERC20 this vault holds. Immutable per-deployment.
    IERC20 public immutable usdc;

    /// @param _usdc USDC ERC20 contract address.
    constructor(address _usdc) Ownable(msg.sender) {
        require(_usdc != address(0), "BridgeVault: USDC address required");
        usdc = IERC20(_usdc);
    }

    /// @inheritdoc IBridgeVault
    /// @dev `onlyOwner` so only the SomaBridge can release funds; the bridge
    /// itself only does so after `MessageVerifier.verifyMessageAndSignatures`
    /// has accepted the quorum-signed withdrawal.
    function transferUSDC(address targetAddress, uint256 amount)
        external
        override
        onlyOwner
    {
        require(targetAddress != address(0), "BridgeVault: Zero recipient");
        bool ok = usdc.transfer(targetAddress, amount);
        require(ok, "BridgeVault: Transfer failed");
    }
}
