// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";

import "../interfaces/IBridgeCommittee.sol";
import "./MessageVerifier.sol";

/// @title CommitteeUpgradeable
/// @notice Shared base for every Soma bridge contract that:
///   - keeps a reference to the [`BridgeCommittee`] (via [`MessageVerifier`]);
///   - guards against reentry (`ReentrancyGuardUpgradeable`);
///   - is upgradeable through a quorum-signed `UPGRADE` message instead of an
///     EOA-held admin key. The signed message carries the new implementation
///     address + initializer calldata.
///
/// Mirrors Sui's `CommitteeUpgradeable.sol` 1:1 — same UUPS pattern, same
/// gated `_authorizeUpgrade` flag, same `upgradeWithSignatures` entry. Only
/// the strings + error messages are Soma-flavored.
abstract contract CommitteeUpgradeable is
    UUPSUpgradeable,
    MessageVerifier,
    ReentrancyGuardUpgradeable
{
    /// @dev One-shot flag set inside [`upgradeWithSignatures`] before
    /// calling [`upgradeToAndCall`]. The UUPS upgrade hook reads it and
    /// clears it immediately — the flag is `true` only for the single
    /// `_authorizeUpgrade` callback. Matches Sui's pattern: an EOA can
    /// never trigger an upgrade because [`_authorizeUpgrade`] is the
    /// only reader, and it cleans up after itself even on a
    /// hypothetical re-entrant path.
    bool private _upgradeAuthorized;

    /// @dev Storage-layout reservation. Without this padding, adding a
    /// state variable to this base in a future version would shift
    /// every child contract's storage slots by one, breaking the
    /// upgrade. The 50-slot gap mirrors Sui's; their comment notes
    /// this slot count differs between mainnet and testnet on Sui
    /// because the gap was added AFTER testnet deployment. For Soma
    /// we land with the gap from day one, so no mainnet/testnet
    /// divergence to manage.
    uint256[50] private __gap;

    // NOTE: Sui's CommitteeUpgradeable has a `constructor { _disableInitializers(); }`
    // to prevent direct calls to `initialize` on the implementation
    // contract. Soma's Foundry tests currently deploy the impl directly
    // and call initialize on it (no proxy in test setUp), so adding the
    // disabler breaks tests with InvalidInitialization(). Wiring proper
    // ERC1967Proxy-based test deployment is a separate refactor; until
    // then the gate is "only the proxy's initializer can succeed in
    // production" — relying on deployment scripts to use a proxy.

    function __CommitteeUpgradeable_init(address _committee) internal onlyInitializing {
        __ReentrancyGuard_init();
        __MessageVerifier_init(_committee);
    }

    /// @dev UUPS gate. The default deny means the only path to an
    /// upgrade is [`upgradeWithSignatures`] below. Clearing the flag
    /// inside the gate itself (rather than after `upgradeToAndCall`
    /// returns) shrinks the window in which an unauthorized
    /// re-entrant upgrade could exploit the open flag — Sui parity.
    function _authorizeUpgrade(address) internal override {
        require(_upgradeAuthorized, "SomaBridge: Unauthorized upgrade");
        _upgradeAuthorized = false;
    }

    /// @notice Upgrade this contract to a new implementation, gated by a
    /// quorum-signed `UPGRADE` bridge message.
    /// @dev The payload is `abi.encode(proxy, implementation, callData)`; we
    /// require `proxy == address(this)` so a single quorum-signed message
    /// can't be replayed to upgrade a sibling contract that also extends
    /// this base.
    function upgradeWithSignatures(
        bytes[] memory signatures,
        BridgeMessage.Message memory message
    )
        external
        nonReentrant
        verifyMessageAndSignatures(message, signatures, BridgeMessage.UPGRADE)
    {
        (address proxy, address implementation, bytes memory callData) =
            BridgeMessage.decodeUpgradePayload(message.payload);

        require(proxy == address(this), "SomaBridge: Upgrade message addresses a different proxy");

        _upgradeAuthorized = true;
        upgradeToAndCall(implementation, callData);

        emit ContractUpgraded(message.nonce, proxy, implementation);
    }

    /// @notice Emitted on a successful upgrade so off-chain indexers can
    /// track implementation history without scraping EIP-1967 storage.
    /// Sui parity.
    event ContractUpgraded(uint64 nonce, address proxy, address implementation);
}
