// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {Upgrades} from "@openzeppelin/openzeppelin-foundry-upgrades/Upgrades.sol";
import {Options} from "@openzeppelin/openzeppelin-foundry-upgrades/Options.sol";

import "../contracts/BridgeCommittee.sol";
import "../contracts/BridgeLimiter.sol";
import "../contracts/BridgeVault.sol";
import "../contracts/SomaBridge.sol";

/// @title Deploy
/// @notice One-shot deployment script for the Soma <-> Eth USDC bridge.
///
/// Mirrors the proxy-based deployment sequence in
/// `test/SomaBridgeTest.t.sol::setUp` so operators get the exact same
/// contract topology in production that the test suite exercises.
///
/// USAGE:
///     COMMITTEE_MEMBERS=0xaa..,0xbb..,0xcc..,0xdd.. \
///     COMMITTEE_STAKES=2500,2500,2500,2500 \
///     ETH_CHAIN_ID=13 \
///     USDC_ADDRESS=0x036CbD53842c5426634e7929541eC2318f3dCF7e \
///     LIMITER_TOTAL_LIMIT=1000000000000 \
///     SUPPORTED_SOMA_CHAINS=0,1,2 \
///     forge script script/Deploy.s.sol:Deploy \
///         --rpc-url $SEPOLIA_RPC_URL \
///         --private-key $DEPLOYER_PK \
///         --broadcast --verify
///
/// The script reads the lists with `vm.envXxx("VAR", ",")` so each list
/// is a single comma-separated string in the env. Pass `--verify` to
/// forge separately if you want Etherscan verification — this script
/// does not encode that decision.
contract Deploy is Script {
    // ---- env var names (centralized so a typo fails fast) ----
    string constant ENV_MEMBERS = "COMMITTEE_MEMBERS";
    string constant ENV_STAKES = "COMMITTEE_STAKES";
    string constant ENV_ETH_CHAIN_ID = "ETH_CHAIN_ID";
    string constant ENV_USDC = "USDC_ADDRESS";
    string constant ENV_LIMITER_LIMIT = "LIMITER_TOTAL_LIMIT";
    string constant ENV_SOMA_CHAINS = "SUPPORTED_SOMA_CHAINS";

    /// BPS total enforced on the committee stakes input. The on-chain
    /// thresholds in `BridgeCommittee` assume the total stake is 10000.
    uint16 constant TOTAL_STAKE_BPS = 10000;

    function run() external {
        // ============================================================
        // 1) Read + validate inputs (BEFORE broadcast so a bad config
        //    aborts without spending gas).
        // ============================================================

        address[] memory members = vm.envAddress(ENV_MEMBERS, ",");
        uint256[] memory stakesRaw = vm.envUint(ENV_STAKES, ",");
        uint256 ethChainIdRaw = vm.envUint(ENV_ETH_CHAIN_ID);
        address usdc = vm.envAddress(ENV_USDC);
        uint256 limiterLimitRaw = vm.envUint(ENV_LIMITER_LIMIT);
        uint256[] memory somaChainsRaw = vm.envUint(ENV_SOMA_CHAINS, ",");

        // Length match: members <-> stakes.
        require(
            members.length == stakesRaw.length,
            "Deploy: COMMITTEE_MEMBERS and COMMITTEE_STAKES length mismatch"
        );
        require(members.length > 0, "Deploy: committee must have >= 1 member");

        // Narrow uint256 -> uint16 stakes, sum-check to TOTAL_STAKE_BPS.
        // A typo that produced 9999 or 10001 silently here would yield a
        // committee that can never form quorum (or worse, overflows the
        // threshold math), so we abort.
        uint16[] memory stakes = new uint16[](stakesRaw.length);
        uint256 stakeSum = 0;
        for (uint256 i = 0; i < stakesRaw.length; i++) {
            require(stakesRaw[i] <= type(uint16).max, "Deploy: stake exceeds uint16");
            stakes[i] = uint16(stakesRaw[i]);
            stakeSum += stakesRaw[i];
        }
        require(
            stakeSum == TOTAL_STAKE_BPS,
            "Deploy: COMMITTEE_STAKES must sum to 10000 BPS"
        );

        // Eth chain id is one byte.
        require(ethChainIdRaw <= type(uint8).max, "Deploy: ETH_CHAIN_ID exceeds uint8");
        uint8 ethChainId = uint8(ethChainIdRaw);

        // USDC must be non-zero (BridgeVault would reject anyway, but
        // fail here for a clearer error).
        require(usdc != address(0), "Deploy: USDC_ADDRESS is zero");

        // Limiter total limit is uint64-scale.
        require(
            limiterLimitRaw <= type(uint64).max,
            "Deploy: LIMITER_TOTAL_LIMIT exceeds uint64"
        );
        require(limiterLimitRaw > 0, "Deploy: LIMITER_TOTAL_LIMIT must be > 0");
        uint64 limiterLimit = uint64(limiterLimitRaw);

        // Supported Soma chains: narrow to uint8, require >= 1.
        require(somaChainsRaw.length >= 1, "Deploy: SUPPORTED_SOMA_CHAINS must have >= 1 entry");
        uint8[] memory supportedSomaChains = new uint8[](somaChainsRaw.length);
        for (uint256 i = 0; i < somaChainsRaw.length; i++) {
            require(
                somaChainsRaw[i] <= type(uint8).max,
                "Deploy: SUPPORTED_SOMA_CHAINS entry exceeds uint8"
            );
            supportedSomaChains[i] = uint8(somaChainsRaw[i]);
        }

        // ============================================================
        // 2) Echo the resolved config to stdout. Operators piping the
        //    output to their ledger see exactly what's being deployed
        //    before any tx broadcasts.
        // ============================================================
        console.log("== Soma bridge deployment ==");
        console.log("ETH_CHAIN_ID         =", ethChainId);
        console.log("USDC_ADDRESS         =", usdc);
        console.log("LIMITER_TOTAL_LIMIT  =", limiterLimit);
        console.log("COMMITTEE size       =", members.length);
        for (uint256 i = 0; i < members.length; i++) {
            console.log("  member", i, members[i]);
            console.log("  stake ", i, stakes[i]);
        }
        console.log("SUPPORTED_SOMA_CHAINS count =", supportedSomaChains.length);
        for (uint256 i = 0; i < supportedSomaChains.length; i++) {
            console.log("  soma chain", i, supportedSomaChains[i]);
        }

        // ============================================================
        // 3) Broadcast deployment txs. Each `new X()` and external
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

        // 3a) BridgeCommittee — UUPS proxy via OZ Upgrades plugin.
        //     The plugin runs storage-layout + UUPS-shape validation
        //     at deploy time and refuses to compile if the impl is
        //     unsafe (selfdestruct, raw delegatecall in impl, etc.).
        BridgeCommittee committee = BridgeCommittee(
            Upgrades.deployUUPSProxy(
                "BridgeCommittee.sol",
                abi.encodeCall(
                    BridgeCommittee.initialize,
                    (members, stakes, ethChainId)
                ),
                opts
            )
        );

        // 3b) BridgeVault — NOT upgradeable. Regular Ownable; usdc
        //     is locked in at construction.
        BridgeVault vault = new BridgeVault(usdc);

        // 3c) BridgeLimiter — UUPS proxy via plugin.
        BridgeLimiter limiter = BridgeLimiter(
            Upgrades.deployUUPSProxy(
                "BridgeLimiter.sol",
                abi.encodeCall(
                    BridgeLimiter.initialize,
                    (address(committee), limiterLimit)
                ),
                opts
            )
        );

        // 3d) SomaBridge — UUPS proxy, wires committee + usdc + vault
        //     + limiter together with the supported source-chain list.
        SomaBridge somaBridge = SomaBridge(
            Upgrades.deployUUPSProxy(
                "SomaBridge.sol",
                abi.encodeCall(
                    SomaBridge.initialize,
                    (
                        address(committee),
                        usdc,
                        address(vault),
                        address(limiter),
                        supportedSomaChains
                    )
                ),
                opts
            )
        );

        // 3e) Hand ownership of the non-upgradeable vault and the
        //     limiter over to the bridge. After this the deployer EOA
        //     can no longer touch them; only the bridge (acting on
        //     verified committee certs) can.
        vault.transferOwnership(address(somaBridge));
        limiter.transferOwnership(address(somaBridge));

        vm.stopBroadcast();

        // ============================================================
        // 4) Emit the contract addresses + block in the exact format
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
}
