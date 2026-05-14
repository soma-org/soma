// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import {Upgrades} from "@openzeppelin/openzeppelin-foundry-upgrades/Upgrades.sol";
import {Options} from "@openzeppelin/openzeppelin-foundry-upgrades/Options.sol";

/// @title  Upgrade-safety validation for every UUPS bridge contract.
/// @notice Static analysis that runs in `forge test`. Failures here MUST block
///         any deploy or upgrade — they signal a storage-layout regression
///         (reordered/removed/retyped state variable) or a UUPS shape
///         violation (selfdestruct, raw delegatecall in impl, missing
///         _authorizeUpgrade, etc.) that would brick existing proxies.
///
/// When a `*V2` contract is added (e.g. `BridgeCommitteeV2.sol`), add a
/// V1->V2 test using the commented-out pattern below. Do NOT relax the
/// validation to silence a failure — fix the new contract's layout to
/// match the previous version's instead.
contract UpgradeValidationTest is Test {
    /// All bridge impls share `CommitteeUpgradeable`, whose constructor
    /// calls `_disableInitializers()`. The OZ plugin flags ANY constructor
    /// as suspect by default, so we explicitly allow it here. This does
    /// NOT skip storage-layout / UUPS shape checks — only the constructor
    /// safety lint.
    string internal constant ALLOW_CONSTRUCTOR = "constructor";

    function test_bridgeCommittee_is_upgrade_safe() public {
        Options memory opts;
        opts.unsafeAllow = ALLOW_CONSTRUCTOR;
        Upgrades.validateImplementation("BridgeCommittee.sol", opts);
    }

    function test_bridgeLimiter_is_upgrade_safe() public {
        Options memory opts;
        opts.unsafeAllow = ALLOW_CONSTRUCTOR;
        Upgrades.validateImplementation("BridgeLimiter.sol", opts);
    }

    function test_somaBridge_is_upgrade_safe() public {
        Options memory opts;
        opts.unsafeAllow = ALLOW_CONSTRUCTOR;
        Upgrades.validateImplementation("SomaBridge.sol", opts);
    }

    /// When v2 contracts land, add tests like:
    ///
    /// function test_bridgeCommittee_v1_to_v2_layout_compatible() public {
    ///     Options memory opts;
    ///     opts.unsafeAllow = "constructor";
    ///     opts.referenceContract = "BridgeCommittee.sol";
    ///     Upgrades.validateUpgrade("BridgeCommitteeV2.sol", opts);
    /// }
    ///
    /// function test_bridgeLimiter_v1_to_v2_layout_compatible() public {
    ///     Options memory opts;
    ///     opts.unsafeAllow = "constructor";
    ///     opts.referenceContract = "BridgeLimiter.sol";
    ///     Upgrades.validateUpgrade("BridgeLimiterV2.sol", opts);
    /// }
    ///
    /// function test_somaBridge_v1_to_v2_layout_compatible() public {
    ///     Options memory opts;
    ///     opts.unsafeAllow = "constructor";
    ///     opts.referenceContract = "SomaBridge.sol";
    ///     Upgrades.validateUpgrade("SomaBridgeV2.sol", opts);
    /// }
}
