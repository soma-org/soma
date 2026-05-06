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
    /// @dev One-shot flag set inside [`upgradeWithSignatures`] before calling
    /// [`upgradeToAndCall`]. The UUPS upgrade hook reads it; setting + clearing
    /// it in the same call frame means an EOA can never trigger an upgrade
    /// because [`_authorizeUpgrade`] is the only place that reads it.
    bool private _upgradeAuthorized;

    function __CommitteeUpgradeable_init(address _committee) internal onlyInitializing {
        __ReentrancyGuard_init();
        __MessageVerifier_init(_committee);
        // committee is set in __MessageVerifier_init; setting it again here
        // would be a redundant write but Sui parity keeps the two writes
        // separated. We don't repeat it.
    }

    /// @dev UUPS gate. The default deny means the only path to an upgrade
    /// is [`upgradeWithSignatures`] below, which flips the flag for exactly
    /// one call.
    function _authorizeUpgrade(address) internal view override {
        require(_upgradeAuthorized, "SomaBridge: Unauthorized upgrade");
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
        _upgradeAuthorized = false;
    }
}
