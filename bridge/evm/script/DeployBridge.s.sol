// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import {Upgrades} from "@openzeppelin/openzeppelin-foundry-upgrades/Upgrades.sol";
import {Options} from "@openzeppelin/openzeppelin-foundry-upgrades/Options.sol";

import "../contracts/BridgeCommittee.sol";
import "../contracts/BridgeLimiter.sol";
import "../contracts/BridgeVault.sol";
import "../contracts/SomaBridge.sol";

/// @title DeployBridge
/// @notice One-shot deployment script for the Soma <-> Eth USDC bridge.
///
/// Mirrors the proxy-based deployment sequence in
/// `test/SomaBridgeTest.t.sol::setUp` so operators get the exact same
/// contract topology in production that the test suite exercises.
///
/// CONFIGURATION:
///     The script reads its config from a per-chain JSON file at
///     `deploy_configs/{block.chainid}.json` (resolved relative to the
///     Foundry project root, i.e. `bridge/evm/deploy_configs/...`).
///     For example, Base Sepolia (EVM chain id 84532) reads from
///     `deploy_configs/84532.json`.
///
/// PATH OVERRIDE:
///     The `OVERRIDE_CONFIG_PATH` environment variable, if set, replaces
///     the per-chain default. This exists primarily for integration
///     tests + ad-hoc CI runs against an Anvil instance whose
///     `block.chainid` is meaningless. **DO NOT** rely on this in
///     production deploys — point at the canonical per-chain file so
///     the config under version control matches what was deployed.
///     If you must override in CI, audit the override path in the
///     run logs *before* `--broadcast` lands.
///
/// USAGE (Base Sepolia, EVM chain id 84532):
///     # 1) Edit deploy_configs/84532.json (USDC, limit, supported chains).
///     # 2) Refresh committee state from live Soma chain:
///     bridge-committee-export \
///         --soma-rpc http://<soma-rpc>:9000 \
///         --target-chain-id 13 \
///         --output deploy_configs/84532.json
///     # 3) Run the deploy:
///     forge script script/DeployBridge.s.sol:DeployBridge \
///         --rpc-url $BASE_SEPOLIA_RPC \
///         --private-key $DEPLOYER_PK \
///         --broadcast --verify
contract DeployBridge is Script {
    /// BPS total enforced on the committee stakes input. The on-chain
    /// thresholds in `BridgeCommittee` assume the total stake is 10000.
    uint16 constant TOTAL_STAKE_BPS = 10000;

    /// Strongly-typed projection of the on-disk JSON. JSON gives us
    /// uint256 for every integer; we narrow to the contract-side widths
    /// during validation.
    struct DeployConfig {
        address[] committeeMembers;
        uint256[] committeeStake;
        uint8 ethChainId;
        address usdcAddress;
        uint64 limiterTotalLimit;
        uint8[] supportedSomaChains;
        // Optional: SHA-256 of the (sorted-by-Eth-address) committee
        // state, copied verbatim from the `bridge-committee-export`
        // output. Echoed to stdout for the operator ledger. Empty bytes
        // (`""`) is tolerated — the deployer still proceeds because the
        // digest is informational, not enforced on-chain.
        bytes32 somaCommitteeDigest;
    }

    /// Parse the JSON file at `path` into a `DeployConfig`. Field-by-field
    /// `vm.parseJson` matches Sui's `deploy_bridge.s.sol` pattern: we
    /// cannot decode the whole struct at once because Solidity's tuple
    /// ABI is sensitive to alphabetical field ordering and silent
    /// corruption is worse than verbose code here.
    function parseDeployConfig(string memory path)
        public
        view
        returns (DeployConfig memory)
    {
        string memory json = vm.readFile(path);
        DeployConfig memory config;

        config.committeeMembers =
            abi.decode(vm.parseJson(json, ".committeeMembers"), (address[]));
        config.committeeStake =
            abi.decode(vm.parseJson(json, ".committeeStake"), (uint256[]));

        // JSON gives us uint256; narrow defensively to uint8.
        uint256 ethChainIdRaw =
            abi.decode(vm.parseJson(json, ".ethChainId"), (uint256));
        require(
            ethChainIdRaw <= type(uint8).max,
            "DeployBridge: ethChainId exceeds uint8"
        );
        config.ethChainId = uint8(ethChainIdRaw);

        config.usdcAddress =
            abi.decode(vm.parseJson(json, ".usdcAddress"), (address));

        // limiterTotalLimit is encoded as a string in the JSON to dodge
        // JSON number-precision loss (uint64 max exceeds JS Number
        // safe-integer range). Parse + narrow here.
        string memory limiterLimitStr =
            abi.decode(vm.parseJson(json, ".limiterTotalLimit"), (string));
        uint256 limiterLimitRaw = vm.parseUint(limiterLimitStr);
        require(
            limiterLimitRaw <= type(uint64).max,
            "DeployBridge: limiterTotalLimit exceeds uint64"
        );
        config.limiterTotalLimit = uint64(limiterLimitRaw);

        // supportedSomaChains: JSON gives uint256[], narrow to uint8[].
        uint256[] memory somaChainsRaw =
            abi.decode(vm.parseJson(json, ".supportedSomaChains"), (uint256[]));
        config.supportedSomaChains = new uint8[](somaChainsRaw.length);
        for (uint256 i = 0; i < somaChainsRaw.length; i++) {
            require(
                somaChainsRaw[i] <= type(uint8).max,
                "DeployBridge: supportedSomaChains entry exceeds uint8"
            );
            config.supportedSomaChains[i] = uint8(somaChainsRaw[i]);
        }

        // Committee digest is informational. `bridge-committee-export`
        // writes it as a `0x`-prefixed 64-char hex string, which foundry's
        // `vm.parseJson` auto-decodes as `bytes32` (32 raw bytes) rather
        // than as a string — so `abi.decode(..., (string))` reverts. Read
        // the raw ABI-encoded result instead and branch on its length:
        // 32 bytes → bytes32 payload; anything else (e.g. `""` from a
        // pre-export config) → fall back to the zero hash and let the
        // empty-committee check below give a clearer error.
        bytes memory digestRaw = vm.parseJson(json, ".somaCommitteeDigest");
        if (digestRaw.length == 32) {
            config.somaCommitteeDigest = abi.decode(digestRaw, (bytes32));
        } else {
            config.somaCommitteeDigest = bytes32(0);
        }

        return config;
    }

    function run() external {
        // ============================================================
        // 1) Resolve the config path. The default is keyed off
        //    `block.chainid` so each EVM target has its own file under
        //    version control. The `OVERRIDE_CONFIG_PATH` env var exists
        //    for integration tests against Anvil — see the docstring
        //    above for the (loud) warning about misusing it in CI.
        // ============================================================
        string memory root = vm.projectRoot();
        string memory defaultPath = string.concat(
            root,
            "/deploy_configs/",
            Strings.toString(block.chainid),
            ".json"
        );
        // `vm.envOr` returns the env var if set, else the fallback.
        // ANY non-empty value in the env wins — this is exactly the
        // kind of thing that gets misconfigured in CI; echo the
        // resolved path before doing anything else so the operator
        // can spot a stray override in run logs.
        string memory path = vm.envOr("OVERRIDE_CONFIG_PATH", defaultPath);

        console.log("== Soma bridge deployment ==");
        console.log("block.chainid       =", block.chainid);
        console.log("config path         =", path);
        if (
            keccak256(bytes(path)) != keccak256(bytes(defaultPath))
        ) {
            console.log(
                unicode"!! OVERRIDE_CONFIG_PATH is in effect — verify before broadcast !!"
            );
        }

        DeployConfig memory config = parseDeployConfig(path);

        // ============================================================
        // 2) Validate the parsed config (BEFORE broadcast so a bad
        //    config aborts without spending gas).
        // ============================================================

        // The single most likely operator mistake: forgetting to run
        // `bridge-committee-export` against this file. Catch it with a
        // clear message rather than letting the length mismatch /
        // BPS-sum check fire downstream.
        require(
            config.committeeMembers.length > 0,
            "DeployBridge: committeeMembers is empty - run `bridge-committee-export --output deploy_configs/<chainid>.json` first"
        );

        // Length match: members <-> stakes.
        require(
            config.committeeMembers.length == config.committeeStake.length,
            "DeployBridge: committeeMembers and committeeStake length mismatch"
        );

        // Narrow uint256 -> uint16 stakes, sum-check to TOTAL_STAKE_BPS.
        // A typo that produced 9999 or 10001 silently here would yield a
        // committee that can never form quorum (or worse, overflows the
        // threshold math), so we abort.
        uint16[] memory stakes = new uint16[](config.committeeStake.length);
        uint256 stakeSum = 0;
        for (uint256 i = 0; i < config.committeeStake.length; i++) {
            require(
                config.committeeStake[i] <= type(uint16).max,
                "DeployBridge: stake exceeds uint16"
            );
            stakes[i] = uint16(config.committeeStake[i]);
            stakeSum += config.committeeStake[i];
        }
        require(
            stakeSum == TOTAL_STAKE_BPS,
            "DeployBridge: committeeStake must sum to 10000 BPS"
        );

        // USDC must be non-zero (BridgeVault would reject anyway, but
        // fail here for a clearer error).
        require(
            config.usdcAddress != address(0),
            "DeployBridge: usdcAddress is zero"
        );

        // Limiter total limit must be > 0; the uint64 narrowing already
        // happened in parseDeployConfig.
        require(
            config.limiterTotalLimit > 0,
            "DeployBridge: limiterTotalLimit must be > 0"
        );

        // Supported Soma chains: require >= 1.
        require(
            config.supportedSomaChains.length >= 1,
            "DeployBridge: supportedSomaChains must have >= 1 entry"
        );

        // ============================================================
        // 3) Echo the resolved config to stdout. Operators piping the
        //    output to their ledger see exactly what's being deployed
        //    before any tx broadcasts.
        // ============================================================
        console.log("ethChainId          =", config.ethChainId);
        console.log("usdcAddress         =", config.usdcAddress);
        console.log("limiterTotalLimit   =", config.limiterTotalLimit);
        console.log("committee size      =", config.committeeMembers.length);
        for (uint256 i = 0; i < config.committeeMembers.length; i++) {
            console.log("  member", i, config.committeeMembers[i]);
            console.log("  stake ", i, stakes[i]);
        }
        console.log(
            "supportedSomaChains count =",
            config.supportedSomaChains.length
        );
        for (uint256 i = 0; i < config.supportedSomaChains.length; i++) {
            console.log("  soma chain", i, config.supportedSomaChains[i]);
        }
        console.logBytes32(config.somaCommitteeDigest);

        // ============================================================
        // 4) Broadcast deployment txs. Each `new X()` and external
        //    call inside this block becomes a real on-chain tx that
        //    `--broadcast` will sign and submit.
        // ============================================================
        vm.startBroadcast();

        // OZ Upgrades plugin validation: skip the ones that don't
        // apply to a UUPS-with-_disableInitializers-constructor impl.
        // We deliberately do NOT skip the storage-layout / UUPS-shape
        // checks — those are the point of using the plugin.
        Options memory opts;
        opts.unsafeAllow = "constructor";

        // 4a) BridgeCommittee — UUPS proxy via OZ Upgrades plugin.
        //     The plugin runs storage-layout + UUPS-shape validation
        //     at deploy time and refuses to compile if the impl is
        //     unsafe (selfdestruct, raw delegatecall in impl, etc.).
        BridgeCommittee committee = BridgeCommittee(
            Upgrades.deployUUPSProxy(
                "BridgeCommittee.sol",
                abi.encodeCall(
                    BridgeCommittee.initialize,
                    (config.committeeMembers, stakes, config.ethChainId)
                ),
                opts
            )
        );

        // 4b) BridgeVault — NOT upgradeable. Regular Ownable; usdc
        //     is locked in at construction.
        BridgeVault vault = new BridgeVault(config.usdcAddress);

        // 4c) BridgeLimiter — UUPS proxy via plugin.
        BridgeLimiter limiter = BridgeLimiter(
            Upgrades.deployUUPSProxy(
                "BridgeLimiter.sol",
                abi.encodeCall(
                    BridgeLimiter.initialize,
                    (address(committee), config.limiterTotalLimit)
                ),
                opts
            )
        );

        // 4d) SomaBridge — UUPS proxy, wires committee + usdc + vault
        //     + limiter together with the supported source-chain list.
        SomaBridge somaBridge = SomaBridge(
            Upgrades.deployUUPSProxy(
                "SomaBridge.sol",
                abi.encodeCall(
                    SomaBridge.initialize,
                    (
                        address(committee),
                        config.usdcAddress,
                        address(vault),
                        address(limiter),
                        config.supportedSomaChains
                    )
                ),
                opts
            )
        );

        // 4e) Hand ownership of the non-upgradeable vault and the
        //     limiter over to the bridge. After this the deployer EOA
        //     can no longer touch them; only the bridge (acting on
        //     verified committee certs) can.
        vault.transferOwnership(address(somaBridge));
        limiter.transferOwnership(address(somaBridge));

        vm.stopBroadcast();

        // ============================================================
        // 5) Emit the contract addresses + block in the exact format
        //    the runbook's templating step expects. Anything that
        //    isn't KEY=VALUE here is going to make the operator sad.
        // ============================================================
        console.log("BRIDGE_COMMITTEE_PROXY=", address(committee));
        console.log("BRIDGE_VAULT=", address(vault));
        console.log("BRIDGE_LIMITER_PROXY=", address(limiter));
        console.log("SOMA_BRIDGE_PROXY=", address(somaBridge));
        console.log("DEPLOYMENT_BLOCK=", block.number);
        console.log(unicode"✓ deployment complete");
    }

    // used to ignore for forge coverage
    function testSkip() public {}
}
