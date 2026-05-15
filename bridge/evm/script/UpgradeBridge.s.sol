// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {Upgrades} from "@openzeppelin/openzeppelin-foundry-upgrades/Upgrades.sol";
import {Options} from "@openzeppelin/openzeppelin-foundry-upgrades/Options.sol";

import "../contracts/utils/BridgeMessage.sol";

/// @title UpgradeBridge
/// @notice Two-mode helper for UUPS upgrades of the Soma bridge contracts
/// (`BridgeCommittee`, `BridgeLimiter`, `SomaBridge` — all three inherit
/// `CommitteeUpgradeable` and accept upgrades via a quorum-signed
/// `EvmContractUpgrade` message).
///
/// The on-chain entry point is the parent `upgradeWithSignatures(bytes[],
/// BridgeMessage.Message)` — same selector across all three. So this script
/// is impl-agnostic: it dispatches through a minimal interface and never
/// imports the concrete contracts.
///
/// ============================================================================
/// MODE A — prepare
/// ============================================================================
///
/// Run on the deploy box BEFORE any signatures exist. Deploys the new
/// implementation, validates storage-layout compatibility against the
/// reference impl, and prints the canonical signed-bytes hex that the
/// off-chain bridge-node operators feed into the peer-broadcast sig
/// aggregator. No on-chain state change to the proxy itself yet.
///
/// USAGE:
///     UPGRADE_MODE=prepare \
///     UPGRADE_CONTRACT=BridgeCommittee \
///     UPGRADE_PROXY=0x... \
///     UPGRADE_NEW_IMPL_SOL=BridgeCommitteeV2.sol \
///     UPGRADE_REF_IMPL_SOL=BridgeCommittee.sol \
///     UPGRADE_NONCE=7 \
///     UPGRADE_CHAIN_ID=13 \
///     UPGRADE_CALL_DATA=0x \
///     forge script script/UpgradeBridge.s.sol:UpgradeBridge \
///         --rpc-url $SEPOLIA_RPC_URL \
///         --private-key $DEPLOYER_PK \
///         --broadcast
///
/// Output is `KEY=VALUE` lines: `NEW_IMPL_ADDRESS=...`, `SIGNED_BYTES=0x...`,
/// `EXPECTED_DIGEST=0x...`, `PROXY_ADDRESS=...`, `NONCE=...`.
///
/// ============================================================================
/// MODE B — submit
/// ============================================================================
///
/// Run on the operator box AFTER the bridge-node peer-broadcast aggregator
/// has collected a quorum of committee signatures. The new impl was already
/// deployed in Mode A — this mode only assembles the same
/// `BridgeMessage.Message` and submits it through `upgradeWithSignatures`.
///
/// IMPORTANT: the resolved config (proxy, new impl, payload triple, nonce,
/// chain id, callData) MUST match Mode A byte-for-byte. The contract
/// verifies `keccak256(encodeMessage(message))` against the recovered
/// signatures, so any drift makes the sigs invalid and the tx reverts.
///
/// USAGE:
///     UPGRADE_MODE=submit \
///     UPGRADE_CONTRACT=BridgeCommittee \
///     UPGRADE_PROXY=0x... \
///     UPGRADE_NEW_IMPL_SOL=BridgeCommitteeV2.sol \
///     UPGRADE_NONCE=7 \
///     UPGRADE_CHAIN_ID=13 \
///     UPGRADE_CALL_DATA=0x \
///     UPGRADE_SIGNATURES=0x... (concat of all 65-byte sigs) \
///     forge script script/UpgradeBridge.s.sol:UpgradeBridge \
///         --rpc-url $SEPOLIA_RPC_URL \
///         --private-key $DEPLOYER_PK \
///         --broadcast
///
/// Submit mode does NOT redeploy the impl — it reads `UPGRADE_NEW_IMPL_SOL`
/// only to recover the address via `prepareUpgrade`'s deterministic
/// CREATE2-equivalent? NO — prepareUpgrade uses CREATE; the redeployed impl
/// would have a different address than Mode A's. So submit mode takes the
/// impl address from an env var `UPGRADE_NEW_IMPL_ADDR` (computed once in
/// Mode A and persisted by the runbook).
///
/// ============================================================================
/// NOT in scope
/// ============================================================================
/// - Multi-contract upgrades in one invocation (one upgrade per script run)
/// - Rollback (it's the same flow: sign an upgrade message that points back
///   at the previous impl)
/// - Etherscan verification (pass `--verify` to forge as usual)
contract UpgradeBridge is Script {
    /* ========== env var names (centralized so a typo fails fast) ========== */

    string constant ENV_MODE = "UPGRADE_MODE";
    string constant ENV_CONTRACT = "UPGRADE_CONTRACT";
    string constant ENV_PROXY = "UPGRADE_PROXY";
    string constant ENV_NEW_IMPL_SOL = "UPGRADE_NEW_IMPL_SOL";
    string constant ENV_REF_IMPL_SOL = "UPGRADE_REF_IMPL_SOL";
    string constant ENV_NEW_IMPL_ADDR = "UPGRADE_NEW_IMPL_ADDR";
    string constant ENV_NONCE = "UPGRADE_NONCE";
    string constant ENV_CHAIN_ID = "UPGRADE_CHAIN_ID";
    string constant ENV_CALL_DATA = "UPGRADE_CALL_DATA";
    string constant ENV_SIGNATURES = "UPGRADE_SIGNATURES";

    string constant MODE_PREPARE = "prepare";
    string constant MODE_SUBMIT = "submit";

    /// @dev Per-committee-signature length: r(32) || s(32) || v(1) = 65 bytes.
    /// `BridgeCommittee.verifySignatures` slices the bundle the same way; if
    /// this ever changes, both sides must move together.
    uint256 constant SIG_LEN = 65;

    /// @dev Minimal interface for the upgrade entry point. Lives on every
    /// `CommitteeUpgradeable`-derived contract. We dispatch through this
    /// instead of importing the concrete BridgeCommittee/Limiter/SomaBridge
    /// so the script stays impl-agnostic and doesn't pull their full deps.
    function _upgradeSelector() internal pure returns (bytes4) {
        return bytes4(keccak256("upgradeWithSignatures(bytes[],(uint8,uint8,uint64,uint8,bytes))"));
    }

    function run() external {
        // ====================================================================
        // 1) Read + validate ALL env input BEFORE doing anything destructive.
        //    Both modes echo the resolved config first; an operator reading
        //    the output should see the inputs and bail before any RPC call
        //    if anything looks wrong.
        // ====================================================================

        string memory mode = vm.envString(ENV_MODE);
        bool isPrepare = _stringEq(mode, MODE_PREPARE);
        bool isSubmit = _stringEq(mode, MODE_SUBMIT);
        require(
            isPrepare || isSubmit,
            "UpgradeBridge: UPGRADE_MODE must be 'prepare' or 'submit'"
        );

        // Friendly name only — used in log lines so the operator can see
        // which of the three contracts is being upgraded. Not load-bearing.
        string memory contractName = vm.envString(ENV_CONTRACT);
        require(
            _stringEq(contractName, "BridgeCommittee")
                || _stringEq(contractName, "BridgeLimiter")
                || _stringEq(contractName, "SomaBridge"),
            "UpgradeBridge: UPGRADE_CONTRACT must be BridgeCommittee | BridgeLimiter | SomaBridge"
        );

        address proxy = vm.envAddress(ENV_PROXY);
        require(proxy != address(0), "UpgradeBridge: UPGRADE_PROXY is zero");

        string memory newImplSol = vm.envString(ENV_NEW_IMPL_SOL);
        require(bytes(newImplSol).length > 0, "UpgradeBridge: UPGRADE_NEW_IMPL_SOL is empty");

        // Nonce + chain id: narrow + validate.
        uint256 nonceRaw = vm.envUint(ENV_NONCE);
        require(nonceRaw <= type(uint64).max, "UpgradeBridge: UPGRADE_NONCE exceeds uint64");
        uint64 nonce = uint64(nonceRaw);

        uint256 chainIdRaw = vm.envUint(ENV_CHAIN_ID);
        require(
            chainIdRaw <= type(uint8).max,
            "UpgradeBridge: UPGRADE_CHAIN_ID exceeds uint8 (BridgeChainId byte)"
        );
        uint8 chainId = uint8(chainIdRaw);

        // Call data is optional — empty means a no-init upgrade.
        bytes memory callData = vm.envOr(ENV_CALL_DATA, bytes(""));

        // ====================================================================
        // 2) Echo resolved config. Same KEY=VALUE shape as DeployBridge.s.sol
        //    so operators / runbook templating can parse identically.
        // ====================================================================
        console.log("== Soma bridge upgrade ==");
        console.log("UPGRADE_MODE         =", mode);
        console.log("UPGRADE_CONTRACT     =", contractName);
        console.log("UPGRADE_PROXY        =", proxy);
        console.log("UPGRADE_NEW_IMPL_SOL =", newImplSol);
        console.log("UPGRADE_NONCE        =", nonce);
        console.log("UPGRADE_CHAIN_ID     =", chainId);
        console.log("UPGRADE_CALL_DATA_LEN=", callData.length);

        if (isPrepare) {
            _runPrepare(proxy, newImplSol, callData, nonce, chainId);
        } else {
            _runSubmit(proxy, newImplSol, callData, nonce, chainId);
        }
    }

    // ========================================================================
    // Mode A — prepare
    // ========================================================================

    function _runPrepare(
        address proxy,
        string memory newImplSol,
        bytes memory callData,
        uint64 nonce,
        uint8 chainId
    ) internal {
        // Reference impl: default to the un-suffixed name (e.g. if the new
        // impl is `BridgeCommitteeV2.sol`, the previous version is presumed
        // to live at `BridgeCommittee.sol`). The runbook can override via
        // UPGRADE_REF_IMPL_SOL when the naming doesn't follow that pattern.
        string memory refImplSol =
            vm.envOr(ENV_REF_IMPL_SOL, _stripVersionSuffix(newImplSol));
        require(
            bytes(refImplSol).length > 0,
            "UpgradeBridge: UPGRADE_REF_IMPL_SOL resolved to empty (couldn't infer)"
        );
        console.log("UPGRADE_REF_IMPL_SOL =", refImplSol);

        // OZ Upgrades plugin options:
        //  - referenceContract: file that defines the "current" impl. The
        //    plugin loads its storage layout from foundry's build artifacts
        //    and diffs it against the new impl. A renamed slot, a moved
        //    `__gap`, an inserted variable above existing state — all
        //    trigger a fail-stop here, BEFORE we cut a quorum signature.
        //  - unsafeAllow: "constructor" — every CommitteeUpgradeable child
        //    has a constructor that calls `_disableInitializers()`. The
        //    plugin's default rule flags ANY non-trivial constructor; the
        //    DeployBridge.s.sol script makes the same allowance, and we
        //    mirror it.
        Options memory opts;
        opts.referenceContract = refImplSol;
        opts.unsafeAllow = "constructor";

        // -- 1) Static validation. This call performs the storage-layout
        //    diff without deploying. Doing it explicitly (rather than
        //    relying on prepareUpgrade's implicit check) gives a clean
        //    point in the script log where a failure obviously means "the
        //    new impl is storage-incompatible with the old one".
        Upgrades.validateUpgrade(newImplSol, opts);
        console.log(unicode"✓ storage-layout validation passed");

        // -- 2) Deploy the new implementation. `prepareUpgrade` re-runs the
        //    same validations and then deploys. It does NOT touch the
        //    proxy — that's done in Mode B with a quorum signature.
        vm.startBroadcast();
        address newImpl = Upgrades.prepareUpgrade(newImplSol, opts);
        vm.stopBroadcast();

        require(newImpl != address(0), "UpgradeBridge: prepareUpgrade returned zero address");

        // -- 3) Build the canonical signed bytes. The payload encoding is
        //    standard abi.encode(proxy, newImpl, callData) — the on-chain
        //    decoder reverses it via `abi.decode(payload, (address,
        //    address, bytes))` in `BridgeMessage.decodeUpgradePayload`.
        //    The Rust side does the equivalent in
        //    `encode_evm_contract_upgrade_payload` (alloy `abi_encode_params`
        //    against the same triple).
        bytes memory payload = abi.encode(proxy, newImpl, callData);

        BridgeMessage.Message memory message = BridgeMessage.Message({
            messageType: BridgeMessage.UPGRADE,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: nonce,
            chainID: chainId,
            payload: payload
        });

        bytes memory signedBytes = BridgeMessage.encodeMessage(message);
        bytes32 digest = BridgeMessage.computeHash(message);

        // -- 4) Machine-parseable output. The runbook pipes this through
        //    grep + cut to populate the bridge-node config for the
        //    peer-broadcast aggregator. Anything that isn't KEY=VALUE
        //    here is going to make the operator sad.
        console.log("---");
        console.log(unicode"✓ prepare complete");
        console.log("NEW_IMPL_ADDRESS=", newImpl);
        console.log("PROXY_ADDRESS=", proxy);
        console.log("NONCE=", nonce);
        console.log("CHAIN_ID=", chainId);
        // SIGNED_BYTES and EXPECTED_DIGEST are emitted with console.logBytes
        // so they appear as a 0x-prefixed hex string — exactly what the
        // bridge-node operators feed into ecrecover off-chain.
        console.log("SIGNED_BYTES (hex):");
        console.logBytes(signedBytes);
        console.log("EXPECTED_DIGEST (hex):");
        console.logBytes32(digest);
    }

    // ========================================================================
    // Mode B — submit
    // ========================================================================

    function _runSubmit(
        address proxy,
        string memory newImplSol,
        bytes memory callData,
        uint64 nonce,
        uint8 chainId
    ) internal {
        // In submit mode the new impl already exists on-chain (Mode A
        // deployed it). We need the address explicitly — re-running
        // `prepareUpgrade` here would deploy a DIFFERENT impl (CREATE
        // bumps the nonce) and the resulting message wouldn't match what
        // the committee signed in Mode A.
        address newImpl = vm.envAddress(ENV_NEW_IMPL_ADDR);
        require(newImpl != address(0), "UpgradeBridge: UPGRADE_NEW_IMPL_ADDR is zero");
        console.log("UPGRADE_NEW_IMPL_ADDR=", newImpl);
        // `newImplSol` is logged for the operator's eyes only; the on-chain
        // call is impl-agnostic. We keep it in the env so a typo'd address
        // is harder to miss when scanning the log.
        console.log("NOTE: UPGRADE_NEW_IMPL_SOL is informational in submit mode");

        // Signature bundle: concatenated 65-byte sigs. Forge's
        // `vm.envBytes` parses a 0x-prefixed hex string into raw bytes.
        bytes memory sigBundle = vm.envBytes(ENV_SIGNATURES);
        require(sigBundle.length > 0, "UpgradeBridge: UPGRADE_SIGNATURES is empty");
        require(
            sigBundle.length % SIG_LEN == 0,
            "UpgradeBridge: UPGRADE_SIGNATURES length must be a multiple of 65 bytes"
        );

        uint256 sigCount = sigBundle.length / SIG_LEN;
        bytes[] memory signatures = new bytes[](sigCount);
        for (uint256 i = 0; i < sigCount; i++) {
            bytes memory sig = new bytes(SIG_LEN);
            for (uint256 j = 0; j < SIG_LEN; j++) {
                sig[j] = sigBundle[i * SIG_LEN + j];
            }
            signatures[i] = sig;
        }
        console.log("Decoded signature count:", sigCount);

        // Rebuild the exact same BridgeMessage as Mode A. If any field
        // here drifts from prepare-mode the recovered signers won't
        // match the committee and the tx reverts with "BridgeCommittee:
        // Insufficient stake".
        bytes memory payload = abi.encode(proxy, newImpl, callData);
        BridgeMessage.Message memory message = BridgeMessage.Message({
            messageType: BridgeMessage.UPGRADE,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: nonce,
            chainID: chainId,
            payload: payload
        });

        // Sanity log: print the reconstructed digest so an operator can
        // eyeball it against the EXPECTED_DIGEST from Mode A before the
        // tx broadcasts. If they don't match, kill the run now.
        bytes32 digest = BridgeMessage.computeHash(message);
        console.log("Reconstructed digest (must match Mode A EXPECTED_DIGEST):");
        console.logBytes32(digest);

        // Dispatch. Encoding by hand rather than importing the
        // CommitteeUpgradeable type keeps this script free of the
        // BridgeCommittee / BridgeLimiter / SomaBridge graph. The
        // selector is identical across all three; verified in
        // `_upgradeSelector` above.
        vm.startBroadcast();
        (bool ok, bytes memory ret) = proxy.call(
            abi.encodeWithSelector(_upgradeSelector(), signatures, message)
        );
        vm.stopBroadcast();

        if (!ok) {
            // Bubble up the revert reason if there is one.
            if (ret.length > 0) {
                assembly {
                    let size := mload(ret)
                    revert(add(32, ret), size)
                }
            }
            revert("UpgradeBridge: upgradeWithSignatures reverted with no data");
        }

        console.log("---");
        console.log(unicode"✓ submit complete");
        console.log("PROXY_ADDRESS=", proxy);
        console.log("NEW_IMPL_ADDRESS=", newImpl);
        console.log("BLOCK_NUMBER=", block.number);
    }

    // ========================================================================
    // helpers
    // ========================================================================

    function _stringEq(string memory a, string memory b) internal pure returns (bool) {
        return keccak256(bytes(a)) == keccak256(bytes(b));
    }

    /// @dev Best-effort "strip the version suffix" used to default
    /// UPGRADE_REF_IMPL_SOL. Examples:
    ///   `BridgeCommitteeV2.sol`  -> `BridgeCommittee.sol`
    ///   `BridgeCommitteeV12.sol` -> `BridgeCommittee.sol`
    ///   `SomaBridgeV3.sol`       -> `SomaBridge.sol`
    /// Anything else returns the input unchanged — the operator must then
    /// pass UPGRADE_REF_IMPL_SOL explicitly. We don't try to be clever
    /// (e.g. fall back to walking EIP-1967) because guessing wrong here
    /// would silently diff against the wrong reference and let an
    /// incompatible storage layout through.
    function _stripVersionSuffix(string memory s) internal pure returns (string memory) {
        bytes memory b = bytes(s);
        // Expect `.sol` suffix.
        if (b.length < 6) return s;
        if (
            b[b.length - 4] != "." || b[b.length - 3] != "s" || b[b.length - 2] != "o"
                || b[b.length - 1] != "l"
        ) {
            return s;
        }

        // Walk backwards from before ".sol", consuming digits until we
        // hit a non-digit. If that non-digit is 'V', strip "V<digits>"
        // and re-append ".sol".
        uint256 end = b.length - 4; // index just past the non-".sol" prefix
        uint256 i = end;
        while (i > 0) {
            bytes1 c = b[i - 1];
            if (c < "0" || c > "9") break;
            i--;
        }
        // No digits, or all digits — leave alone.
        if (i == end || i == 0) return s;
        // Character immediately before the digit run must be 'V'.
        if (b[i - 1] != "V") return s;

        // Rebuild: b[0 .. i-1] + ".sol".
        bytes memory stripped = new bytes(i - 1 + 4);
        for (uint256 k = 0; k < i - 1; k++) {
            stripped[k] = b[k];
        }
        stripped[i - 1] = ".";
        stripped[i] = "s";
        stripped[i + 1] = "o";
        stripped[i + 2] = "l";
        return string(stripped);
    }
}
