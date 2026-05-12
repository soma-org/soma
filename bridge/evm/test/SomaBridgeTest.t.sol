// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";

import "../contracts/BridgeCommittee.sol";
import "../contracts/BridgeLimiter.sol";
import "../contracts/BridgeVault.sol";
import "../contracts/SomaBridge.sol";
import "../contracts/test-mocks/MockUSDC.sol";
import "../contracts/utils/BridgeMessage.sol";

/// @title SomaBridgeTest
/// @notice Round-trips the full wire format with a real ecrecover-verified quorum.
/// Mirrors what `bridge-node`'s peer-broadcast aggregator produces; if these
/// tests pass, the Eth-side contract accepts certs the Rust side hands it.
contract SomaBridgeTest is Test {
    /// Mirror of `ISomaBridge.TokensDeposited` for `vm.expectEmit`. Solidity
    /// 0.8 requires the event to be declared on either the emitting contract
    /// or the test contract; can't be referenced via the interface from a
    /// non-inheriting contract.
    event TokensDeposited(
        uint64 nonce,
        address sender,
        uint8 destinationChainID,
        bytes32 somaRecipient,
        uint8 tokenType,
        uint64 amount,
        uint64 timestampMs
    );

    /// Eth chain id used in the deployed `BridgeCommittee`. Matches Soma's
    /// `BridgeChainId::EthCustom = 12`.
    uint8 constant ETH_CHAIN_ID = 12;
    /// Soma's source chain id for outbound withdrawals — Soma's
    /// `BridgeChainId::SomaCustom = 2`.
    uint8 constant SOMA_CHAIN_ID = 2;

    MockUSDC usdc;
    BridgeCommittee committee;
    BridgeVault vault;
    BridgeLimiter limiter;
    SomaBridge bridge;

    // 3-of-4 committee fixture. Each gets exactly 2500 BPS so any single
    // signer is below TRANSFER_STAKE_REQUIRED (3334); any two clear it.
    uint256[4] signerKeys;
    address[4] signerAddrs;

    function setUp() public {
        // 1) Build signer keypairs deterministically.
        for (uint256 i = 0; i < 4; i++) {
            signerKeys[i] = uint256(keccak256(abi.encode("soma-test-signer", i)));
            signerAddrs[i] = vm.addr(signerKeys[i]);
        }

        // 2) Deploy USDC + committee.
        usdc = new MockUSDC();
        committee = new BridgeCommittee();
        address[] memory members = new address[](4);
        uint16[] memory stakes = new uint16[](4);
        for (uint256 i = 0; i < 4; i++) {
            members[i] = signerAddrs[i];
            stakes[i] = 2500;
        }
        committee.initialize(members, stakes, ETH_CHAIN_ID);

        // 3) Vault — deploy then transfer ownership to bridge after deploy.
        vault = new BridgeVault(address(usdc));

        // 4) Limiter — deploy, init, then transfer ownership to bridge.
        // 1e12 in USD-multiplier scale (1e4 = $1) = $1e8 / day. Plenty.
        limiter = new BridgeLimiter();
        limiter.initialize(address(committee), 1_000_000_000_000);

        // 5) Bridge — deploy, init. Supports a single Soma chain
        // (SomaCustom = 2); production deployments may pass multiple
        // (mainnet + testnet + custom) so one Eth contract can route
        // across them.
        bridge = new SomaBridge();
        uint8[] memory supportedChains = new uint8[](1);
        supportedChains[0] = SOMA_CHAIN_ID;
        bridge.initialize(
            address(committee),
            address(usdc),
            address(vault),
            address(limiter),
            supportedChains
        );

        // 6) Ownership: vault + limiter → bridge.
        vault.transferOwnership(address(bridge));
        limiter.transferOwnership(address(bridge));
    }

    // ============================================================
    // Helpers
    // ============================================================

    /// Sign the canonical hash of a Soma bridge message with `signerKey`.
    /// Returns a 65-byte recoverable signature.
    function _sign(uint256 signerKey, BridgeMessage.Message memory m)
        internal
        pure
        returns (bytes memory)
    {
        bytes32 digest = BridgeMessage.computeHash(m);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(signerKey, digest);
        return abi.encodePacked(r, s, v);
    }

    function vm_sign_wrapper(uint256 k, bytes32 d)
        internal
        pure
        returns (uint8, bytes32, bytes32)
    {
        // forge-std's vm.sign isn't `pure`; this wrapper exists to silence
        // the helper above (it's logically pure since vm.sign is deterministic
        // given k+d, but the cheatcode signature is `view`).
        return vm.sign(k, d);
    }

    // ============================================================
    // Wire format
    // ============================================================

    /// Smoke: the SOMA_BRIDGE_MESSAGE prefix is what the Rust side encodes.
    function test_messagePrefixMatchesRust() public pure {
        assertEq(BridgeMessage.MESSAGE_PREFIX, "SOMA_BRIDGE_MESSAGE");
    }

    /// **Cross-language wire-format proof.** A deterministic V2
    /// withdrawal must `keccak` to a value pinned by BOTH this test
    /// and the matching Rust test
    /// (`bridge-node/src/types.rs::test_withdrawal_sentinel_hash_matches_solidity`).
    /// If either side changes its encoder, both tests break. Silent
    /// wire-format drift across the Rust↔Solidity boundary is
    /// therefore impossible.
    function test_withdrawalSentinelHashMatchesRust() public pure {
        bytes memory payload = abi.encodePacked(
            uint8(32),
            bytes32(0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA),
            uint8(12),       // EthCustom
            uint8(20),
            bytes20(0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB),
            BridgeMessage.USDC_TOKEN_TYPE,
            uint64(0x1122334455667788),
            uint64(0x99AABBCCDDEEFF00)
        );
        assertEq(payload.length, 72);

        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 0x0102030405060708,
            chainID: SOMA_CHAIN_ID,
            payload: payload
        });
        bytes32 hash = BridgeMessage.computeHash(m);
        bytes32 expected =
            0x920f174c925e333236cfc68cf2023e6028c7d843b004d4786f5fb37ac2c5db79;
        assertEq(hash, expected, "wire-format keccak diverged from Rust sentinel");
    }

    /// Smoke: known-value encoding. Sentinel inputs let us pin the wire format
    /// against the Rust-side test in `types/src/bridge.rs::test_message_encoding_known_values`.
    function test_encodeMessageLayout() public pure {
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 0x0102030405060708,
            chainID: SOMA_CHAIN_ID,
            payload: hex""
        });
        bytes memory enc = BridgeMessage.encodeMessage(m);
        // Layout: 19-byte prefix + 1 + 1 + 8 + 1 = 30 bytes header (no payload).
        assertEq(enc.length, 30);
        // Type byte at index 19.
        assertEq(uint8(enc[19]), BridgeMessage.USDC_WITHDRAW);
        // Version byte at index 20.
        assertEq(uint8(enc[20]), BridgeMessage.TOKEN_TRANSFER_VERSION_V2);
        // Nonce big-endian at 21..29.
        for (uint256 i = 0; i < 8; i++) {
            assertEq(uint8(enc[21 + i]), uint8(i + 1));
        }
        // chainID at 29.
        assertEq(uint8(enc[29]), SOMA_CHAIN_ID);
    }

    /// Smoke: decode of a 72-byte V2 withdraw payload yields exactly what
    /// the Rust side `encode_withdraw_payload` produces.
    function test_decodeUsdcWithdrawPayload() public pure {
        bytes32 somaSender = bytes32(uint256(0xCAFED00D));
        uint8 targetChain = ETH_CHAIN_ID;
        address recipient = address(uint160(0x42));
        uint8 tokenType = BridgeMessage.USDC_TOKEN_TYPE;
        uint64 amount = 1_500_000; // 1.5 USDC raw
        uint64 ts = 1_700_000_000_000;

        // V2 layout: senderLen(1) || sender(32) || targetChain(1)
        //          || targetLen(1) || target(20) || tokenType(1)
        //          || amount(8 BE) || timestamp(8 BE)
        bytes memory payload = abi.encodePacked(
            uint8(32),       // senderAddressLength
            somaSender,
            targetChain,
            uint8(20),       // targetAddressLength
            recipient,
            tokenType,
            amount,
            ts
        );
        assertEq(payload.length, 72);

        (
            bytes32 gotSender,
            uint8 gotTargetChain,
            address gotRecipient,
            uint8 gotTokenType,
            uint64 gotAmount,
            uint64 gotTs
        ) = BridgeMessage.decodeUsdcWithdrawPayload(payload);
        assertEq(gotSender, somaSender);
        assertEq(gotTargetChain, targetChain);
        assertEq(gotRecipient, recipient);
        assertEq(gotTokenType, tokenType);
        assertEq(gotAmount, amount);
        assertEq(gotTs, ts);
    }

    /// Build a V2 withdraw payload helper for downstream tests.
    function _buildWithdrawPayload(address recipient, uint64 amount)
        internal
        pure
        returns (bytes memory)
    {
        return abi.encodePacked(
            uint8(32),                       // senderLen
            bytes32(uint256(0xBABE)),        // somaSender
            ETH_CHAIN_ID,                    // targetChain
            uint8(20),                       // targetLen
            recipient,                       // target
            BridgeMessage.USDC_TOKEN_TYPE,   // tokenType
            amount,
            uint64(0)                        // timestampMs
        );
    }

    // ============================================================
    // Committee verification
    // ============================================================

    /// Two signers (5000 BPS) exceed TRANSFER_STAKE_REQUIRED (3334).
    function test_verifySignatures_twoOfFourMeetsTransferThreshold() public view {
        bytes memory payload = _buildWithdrawPayload(address(uint160(0x1234)), 100_000);
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 0,
            chainID: SOMA_CHAIN_ID,
            payload: payload
        });
        bytes[] memory sigs = new bytes[](2);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[1], m);
        committee.verifySignatures(sigs, m);
    }

    /// One signer (2500 BPS) is below TRANSFER_STAKE_REQUIRED — must revert.
    function test_verifySignatures_oneOfFourBelowThresholdReverts() public {
        bytes memory payload = _buildWithdrawPayload(address(uint160(0x1234)), 100_000);
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 0,
            chainID: SOMA_CHAIN_ID,
            payload: payload
        });
        bytes[] memory sigs = new bytes[](1);
        sigs[0] = _sign(signerKeys[0], m);
        vm.expectRevert(bytes("BridgeCommittee: Insufficient stake"));
        committee.verifySignatures(sigs, m);
    }

    /// Duplicate signature from the same signer is rejected as a
    /// malformed cert. Sui parity: not "silently skip duplicates" —
    /// fail loudly so the off-chain aggregator stops producing them.
    function test_verifySignatures_rejectsDuplicateSigner() public {
        bytes memory payload = _buildWithdrawPayload(address(uint160(0x1234)), 100_000);
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 0,
            chainID: SOMA_CHAIN_ID,
            payload: payload
        });
        bytes[] memory sigs = new bytes[](2);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[0], m);
        vm.expectRevert(bytes("BridgeCommittee: Duplicate signature provided"));
        committee.verifySignatures(sigs, m);
    }

    /// A signature from someone not in the committee fails with
    /// "Signer has no stake" — Sui parity for early rejection of
    /// malformed certs vs silently skipping unknown signers.
    function test_verifySignatures_nonCommitteeSignerRejected() public {
        bytes memory payload = _buildWithdrawPayload(address(uint160(0x1234)), 100_000);
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 0,
            chainID: SOMA_CHAIN_ID,
            payload: payload
        });
        uint256 strangerKey = uint256(keccak256(abi.encode("stranger")));
        bytes[] memory sigs = new bytes[](2);
        sigs[0] = _sign(signerKeys[0], m); // valid (2500 BPS)
        // Even though the second sig is well-formed, the signer isn't
        // in the committee, so the call reverts.
        bytes32 digest = BridgeMessage.computeHash(m);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(strangerKey, digest);
        sigs[1] = abi.encodePacked(r, s, v);
        vm.expectRevert(bytes("BridgeCommittee: Signer has no stake"));
        committee.verifySignatures(sigs, m);
    }

    /// Adding a member to the blocklist must cause subsequent sigs
    /// from that member to fail with `Signer is blocklisted` — not
    /// silently skipped.
    function test_blocklist_signerIsBlocklistedReverts() public {
        // Blocklist signer 0.
        address[] memory toBlock = new address[](1);
        toBlock[0] = signerAddrs[0];
        bytes memory payload = abi.encodePacked(uint8(0), uint8(1), toBlock[0]);
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.BLOCKLIST,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: 0,
            chainID: ETH_CHAIN_ID,
            payload: payload
        });
        bytes[] memory sigs = new bytes[](3);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[1], m);
        sigs[2] = _sign(signerKeys[2], m);
        committee.updateBlocklistWithSignatures(sigs, m);
        assertTrue(committee.blocklist(signerAddrs[0]));

        // Now ANY sig array that includes signer 0 must revert
        // outright — not just silently drop their stake.
        BridgeMessage.Message memory mt = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 100,
            chainID: SOMA_CHAIN_ID,
            payload: _buildWithdrawPayload(address(0x99), 1)
        });
        bytes[] memory sigsWithBlocklisted = new bytes[](2);
        sigsWithBlocklisted[0] = _sign(signerKeys[0], mt);
        sigsWithBlocklisted[1] = _sign(signerKeys[1], mt);
        vm.expectRevert(bytes("BridgeCommittee: Signer is blocklisted"));
        committee.verifySignatures(sigsWithBlocklisted, mt);
    }

    /// Full blocklist lifecycle: a quorum-signed remove flips the
    /// flag back, and the previously blocklisted member's sigs
    /// become valid again.
    function test_blocklist_addThenRemove() public {
        // ---- add ----
        address[] memory toBlock = new address[](1);
        toBlock[0] = signerAddrs[0];
        bytes memory addPayload = abi.encodePacked(uint8(0), uint8(1), toBlock[0]);
        BridgeMessage.Message memory addMsg = BridgeMessage.Message({
            messageType: BridgeMessage.BLOCKLIST,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: 0,
            chainID: ETH_CHAIN_ID,
            payload: addPayload
        });
        bytes[] memory addSigs = new bytes[](3);
        addSigs[0] = _sign(signerKeys[0], addMsg);
        addSigs[1] = _sign(signerKeys[1], addMsg);
        addSigs[2] = _sign(signerKeys[2], addMsg);
        committee.updateBlocklistWithSignatures(addSigs, addMsg);
        assertTrue(committee.blocklist(signerAddrs[0]));

        // ---- remove ----
        bytes memory rmPayload = abi.encodePacked(uint8(1), uint8(1), toBlock[0]);
        BridgeMessage.Message memory rmMsg = BridgeMessage.Message({
            messageType: BridgeMessage.BLOCKLIST,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: 1,
            chainID: ETH_CHAIN_ID,
            payload: rmPayload
        });
        // Need 3 NON-blocklisted signers to sign the remove. Signer 0
        // is blocklisted so they can't contribute. 1+2+3 = 7500 BPS,
        // clears the 5001 blocklist threshold.
        bytes[] memory rmSigs = new bytes[](3);
        rmSigs[0] = _sign(signerKeys[1], rmMsg);
        rmSigs[1] = _sign(signerKeys[2], rmMsg);
        rmSigs[2] = _sign(signerKeys[3], rmMsg);
        committee.updateBlocklistWithSignatures(rmSigs, rmMsg);
        assertFalse(committee.blocklist(signerAddrs[0]));

        // Signer 0's sigs are valid again.
        BridgeMessage.Message memory mt = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 100,
            chainID: SOMA_CHAIN_ID,
            payload: _buildWithdrawPayload(address(0x99), 1)
        });
        bytes[] memory sigs = new bytes[](2);
        sigs[0] = _sign(signerKeys[0], mt);
        sigs[1] = _sign(signerKeys[1], mt);
        committee.verifySignatures(sigs, mt);
    }

    /// Init guard: duplicate member addresses in the constructor
    /// array fail.
    function test_committeeInit_rejectsDuplicateMember() public {
        BridgeCommittee fresh = new BridgeCommittee();
        address[] memory dups = new address[](2);
        dups[0] = address(0x11);
        dups[1] = address(0x11);
        uint16[] memory stakes = new uint16[](2);
        stakes[0] = 5000;
        stakes[1] = 5000;
        vm.expectRevert(bytes("BridgeCommittee: Duplicate committee member"));
        fresh.initialize(dups, stakes, ETH_CHAIN_ID);
    }

    /// Init guard: mismatched array lengths.
    function test_committeeInit_rejectsLengthMismatch() public {
        BridgeCommittee fresh = new BridgeCommittee();
        address[] memory members = new address[](2);
        members[0] = address(0x11);
        members[1] = address(0x22);
        uint16[] memory stakes = new uint16[](1);
        stakes[0] = 10000;
        vm.expectRevert(
            bytes("BridgeCommittee: Committee and stake arrays must be of the same length")
        );
        fresh.initialize(members, stakes, ETH_CHAIN_ID);
    }

    /// Init guard: 256-member cap. Sui's dedup bitmap is uint256, so
    /// 255 is the hard ceiling on committee size.
    function test_committeeInit_rejectsOver255Members() public {
        BridgeCommittee fresh = new BridgeCommittee();
        address[] memory members = new address[](256);
        uint16[] memory stakes = new uint16[](256);
        for (uint256 i = 0; i < 256; i++) {
            members[i] = address(uint160(i + 1));
            stakes[i] = 0; // doesn't matter — must revert on length first
        }
        vm.expectRevert(
            bytes("BridgeCommittee: Committee length must be less than 256")
        );
        fresh.initialize(members, stakes, ETH_CHAIN_ID);
    }

    /// Init guard: total stake must equal exactly 10000 BPS. Reject
    /// either under-stake or over-stake.
    function test_committeeInit_rejectsTotalStakeNot10000() public {
        BridgeCommittee fresh = new BridgeCommittee();
        address[] memory members = new address[](2);
        members[0] = address(0x11);
        members[1] = address(0x22);
        uint16[] memory stakes = new uint16[](2);
        stakes[0] = 2000;
        stakes[1] = 2000; // total 4000, not 10000
        vm.expectRevert(bytes("BridgeCommittee: Total stake must equal 10000 BPS"));
        fresh.initialize(members, stakes, ETH_CHAIN_ID);
    }

    // ============================================================
    // End-to-end deposit + withdraw
    // ============================================================

    /// Deposit flow: user approves + calls deposit → vault holds USDC,
    /// event fires with the V2 fields.
    function test_deposit_locksUsdcAndEmitsEvent() public {
        address user = address(0xBEEF);
        bytes32 somaRecipient = bytes32(uint256(0xDEADCAFE));
        uint64 amount = 5_000_000; // 5 USDC

        usdc.mint(user, amount);
        vm.startPrank(user);
        usdc.approve(address(bridge), amount);

        vm.expectEmit(false, false, false, true);
        emit TokensDeposited(
            0,
            user,
            SOMA_CHAIN_ID,
            somaRecipient,
            BridgeMessage.USDC_TOKEN_TYPE,
            amount,
            uint64(block.timestamp * 1000)
        );
        bridge.deposit(SOMA_CHAIN_ID, somaRecipient, amount);
        vm.stopPrank();

        assertEq(usdc.balanceOf(user), 0);
        assertEq(usdc.balanceOf(address(vault)), amount);
        assertEq(bridge.nonces(BridgeMessage.USDC_DEPOSIT), 1);
    }

    /// Withdraw flow: quorum-signed message → vault releases USDC.
    /// Verifies the full path the Rust-side relayer will exercise.
    function test_transferBridgedTokens_endToEnd() public {
        // Pre-fund the vault — simulates a prior deposit having locked USDC.
        uint64 amount = 2_000_000; // 2 USDC
        usdc.mint(address(vault), amount);

        address recipient = address(0xCAFE);
        bytes memory payload = _buildWithdrawPayload(recipient, amount);
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 0,
            chainID: SOMA_CHAIN_ID, // SOURCE chain
            payload: payload
        });
        bytes[] memory sigs = new bytes[](2);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[1], m);

        bridge.transferBridgedTokensWithSignatures(sigs, m);

        assertEq(usdc.balanceOf(recipient), amount);
        assertEq(usdc.balanceOf(address(vault)), 0);
        assertTrue(bridge.isMessageProcessed(0));
    }

    /// Replay defense: same message twice → second call reverts.
    function test_transferBridgedTokens_rejectsReplay() public {
        uint64 amount = 1_000_000;
        usdc.mint(address(vault), amount * 2);

        address recipient = address(0xCAFE);
        bytes memory payload = _buildWithdrawPayload(recipient, amount);
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 7,
            chainID: SOMA_CHAIN_ID,
            payload: payload
        });
        bytes[] memory sigs = new bytes[](2);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[1], m);

        bridge.transferBridgedTokensWithSignatures(sigs, m);
        vm.expectRevert(bytes("SomaBridge: Withdrawal nonce already processed"));
        bridge.transferBridgedTokensWithSignatures(sigs, m);
    }

    // ============================================================
    // Emergency op
    // ============================================================

    /// Freeze: one signer below the 450 BPS freeze threshold reverts.
    function test_emergencyOp_freezeRequiresFreezeThreshold() public {
        // Single signer with 2500 BPS clears 450 freeze threshold trivially.
        BridgeMessage.Message memory freeze = BridgeMessage.Message({
            messageType: BridgeMessage.EMERGENCY_OP,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: 0,
            chainID: ETH_CHAIN_ID,
            payload: abi.encodePacked(uint8(0)) // freeze
        });
        bytes[] memory sigs = new bytes[](1);
        sigs[0] = _sign(signerKeys[0], freeze);

        bridge.executeEmergencyOpWithSignatures(sigs, freeze);
        assertTrue(bridge.paused());

        // Deposit must now revert.
        usdc.mint(address(this), 100);
        usdc.approve(address(bridge), 100);
        vm.expectRevert();
        bridge.deposit(SOMA_CHAIN_ID, bytes32(uint256(1)), 100);
    }

    /// Unfreeze needs 5001 BPS — single 2500 signer can't.
    function test_emergencyOp_unfreezeRequiresMajority() public {
        // First freeze.
        BridgeMessage.Message memory freeze = BridgeMessage.Message({
            messageType: BridgeMessage.EMERGENCY_OP,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: 0,
            chainID: ETH_CHAIN_ID,
            payload: abi.encodePacked(uint8(0))
        });
        bytes[] memory sigs1 = new bytes[](1);
        sigs1[0] = _sign(signerKeys[0], freeze);
        bridge.executeEmergencyOpWithSignatures(sigs1, freeze);
        assertTrue(bridge.paused());

        // Try unfreeze with one signer — reverts.
        BridgeMessage.Message memory unfreeze = BridgeMessage.Message({
            messageType: BridgeMessage.EMERGENCY_OP,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: 1,
            chainID: ETH_CHAIN_ID,
            payload: abi.encodePacked(uint8(1))
        });
        bytes[] memory sigsBad = new bytes[](1);
        sigsBad[0] = _sign(signerKeys[0], unfreeze);
        vm.expectRevert(bytes("BridgeCommittee: Insufficient stake"));
        bridge.executeEmergencyOpWithSignatures(sigsBad, unfreeze);

        // Three signers = 7500 BPS, clears 5001.
        bytes[] memory sigsGood = new bytes[](3);
        sigsGood[0] = _sign(signerKeys[0], unfreeze);
        sigsGood[1] = _sign(signerKeys[1], unfreeze);
        sigsGood[2] = _sign(signerKeys[2], unfreeze);
        bridge.executeEmergencyOpWithSignatures(sigsGood, unfreeze);
        assertFalse(bridge.paused());
    }

    // ============================================================
    // Multi-chain destination support (Sui V2 parity)
    // ============================================================

    /// `deposit()` rejects destinations that aren't in
    /// `isChainSupported`. Same gate inbound withdrawals go through
    /// — a user can't accidentally send USDC to a Soma chain the
    /// bridge node doesn't watch.
    function test_deposit_rejectsUnsupportedDestinationChain() public {
        address user = address(0xBEEF);
        usdc.mint(user, 1_000_000);
        vm.prank(user);
        usdc.approve(address(bridge), 1_000_000);

        // setUp() only configured SomaCustom (=2). SomaMainnet (=0)
        // must therefore be unsupported.
        vm.prank(user);
        vm.expectRevert(bytes("SomaBridge: Chain not supported"));
        bridge.deposit(/*destinationChainID=*/ 0, bytes32(uint256(1)), 1_000_000);
    }

    /// Inbound withdrawals from an unsupported SOURCE chain are
    /// rejected by `onlySupportedChain(message.chainID)`.
    function test_transferBridgedTokens_rejectsUnsupportedSourceChain() public {
        uint64 amount = 1_000_000;
        usdc.mint(address(vault), amount);

        address recipient = address(0xCAFE);
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 0,
            // Pretend the withdrawal came from a Soma chain id this
            // contract doesn't trust (0 = SomaMainnet, not in the
            // supported set in setUp()).
            chainID: 0,
            payload: _buildWithdrawPayload(recipient, amount)
        });
        bytes[] memory sigs = new bytes[](2);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[1], m);

        vm.expectRevert(bytes("SomaBridge: Chain not supported"));
        bridge.transferBridgedTokensWithSignatures(sigs, m);
    }

    // ============================================================
    // `isMatureMessage` 48h limiter bypass (Sui V2 parity)
    // ============================================================

    /// A mature (>48h old) signed withdrawal releases without
    /// touching the rolling-24h limiter — long-delayed backlog can't
    /// starve fresh outflow. Mirror of Sui's `limitNotExceededV2`.
    function test_transferBridgedTokens_matureMessageBypassesLimiter() public {
        // Warp ahead so we have headroom to set `timestamp = now - 49h`.
        vm.warp(7 days);
        uint64 oldTimestampMs = uint64((block.timestamp - 49 hours) * 1000);

        // Stand up a fresh limiter with a *very tight* cap so any debit
        // would be visible. Tight enough that a non-bypassed write
        // would revert the call.
        BridgeLimiter tightLimiter = new BridgeLimiter();
        tightLimiter.initialize(address(committee), /*totalLimit*/ 1);

        BridgeVault tightVault = new BridgeVault(address(usdc));
        SomaBridge tightBridge = new SomaBridge();
        uint8[] memory chains = new uint8[](1);
        chains[0] = SOMA_CHAIN_ID;
        tightBridge.initialize(
            address(committee),
            address(usdc),
            address(tightVault),
            address(tightLimiter),
            chains
        );
        tightVault.transferOwnership(address(tightBridge));
        tightLimiter.transferOwnership(address(tightBridge));

        uint64 amount = 1_000_000;
        usdc.mint(address(tightVault), amount);

        // Hand-built mature payload so we can set the timestamp.
        address recipient = address(0xCAFE);
        bytes memory payload = abi.encodePacked(
            uint8(32),
            bytes32(uint256(0xBABE)),
            ETH_CHAIN_ID,
            uint8(20),
            recipient,
            BridgeMessage.USDC_TOKEN_TYPE,
            amount,
            oldTimestampMs
        );
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 1,
            chainID: SOMA_CHAIN_ID,
            payload: payload
        });
        bytes[] memory sigs = new bytes[](2);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[1], m);

        // Releases despite the 1-unit cap — proof the bypass fired.
        tightBridge.transferBridgedTokensWithSignatures(sigs, m);
        assertEq(usdc.balanceOf(recipient), amount);
    }

    /// Mirror of the above with a *fresh* timestamp: the limiter
    /// debit fires and the call reverts under a tight cap. Together
    /// these two tests prove the maturity bypass is exactly the
    /// branch that gates the limiter.
    function test_transferBridgedTokens_freshMessageHitsLimiter() public {
        vm.warp(7 days);

        BridgeLimiter tightLimiter = new BridgeLimiter();
        tightLimiter.initialize(address(committee), /*totalLimit*/ 1);
        BridgeVault tightVault = new BridgeVault(address(usdc));
        SomaBridge tightBridge = new SomaBridge();
        uint8[] memory chains = new uint8[](1);
        chains[0] = SOMA_CHAIN_ID;
        tightBridge.initialize(
            address(committee),
            address(usdc),
            address(tightVault),
            address(tightLimiter),
            chains
        );
        tightVault.transferOwnership(address(tightBridge));
        tightLimiter.transferOwnership(address(tightBridge));

        uint64 amount = 1_000_000;
        usdc.mint(address(tightVault), amount);

        address recipient = address(0xCAFE);
        bytes memory payload = abi.encodePacked(
            uint8(32),
            bytes32(uint256(0xBABE)),
            ETH_CHAIN_ID,
            uint8(20),
            recipient,
            BridgeMessage.USDC_TOKEN_TYPE,
            amount,
            uint64(block.timestamp * 1000) // fresh
        );
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 1,
            chainID: SOMA_CHAIN_ID,
            payload: payload
        });
        bytes[] memory sigs = new bytes[](2);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[1], m);

        vm.expectRevert(bytes("BridgeLimiter: Exceeds rolling 24h limit"));
        tightBridge.transferBridgedTokensWithSignatures(sigs, m);
    }

    // ============================================================
    // Blocklist + limit-update via signatures (Sui V2 parity)
    // ============================================================

    /// A quorum-signed `BLOCKLIST` adds members; their signatures are
    /// then dropped at verification time (zero contribution to stake).
    function test_blocklist_addedMembersAreDroppedFromVerification() public {
        // Block out signer 0.
        address[] memory toBlock = new address[](1);
        toBlock[0] = signerAddrs[0];

        bytes memory payload = abi.encodePacked(
            uint8(0), // blocklist (not unblock)
            uint8(1),
            toBlock[0]
        );
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.BLOCKLIST,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: 0,
            chainID: ETH_CHAIN_ID,
            payload: payload
        });
        // 3 signers = 7500 BPS clears 5001 blocklist threshold.
        bytes[] memory sigs = new bytes[](3);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[1], m);
        sigs[2] = _sign(signerKeys[2], m);
        committee.updateBlocklistWithSignatures(sigs, m);
        assertTrue(committee.blocklist(signerAddrs[0]));

        // Now signer 0's signature contributes 0 stake. Two signers
        // (1 + 2 = 5000 BPS) clears the deposit threshold, but
        // signer 0's sig alone (would have been 2500) is dropped.
        BridgeMessage.Message memory mt = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 100,
            chainID: SOMA_CHAIN_ID,
            payload: _buildWithdrawPayload(address(0x99), 1)
        });
        // Sui parity: presence of a blocklisted signer in the array
        // reverts loudly rather than silently dropping their stake.
        bytes[] memory blockedSigs = new bytes[](2);
        blockedSigs[0] = _sign(signerKeys[0], mt); // blocklisted
        blockedSigs[1] = _sign(signerKeys[1], mt); // 2500
        vm.expectRevert(bytes("BridgeCommittee: Signer is blocklisted"));
        committee.verifySignatures(blockedSigs, mt);

        // The "would have been short" case: only non-blocklisted
        // signer 1 (2500 BPS) submits — clearly below 3334 threshold.
        bytes[] memory shortSigs = new bytes[](1);
        shortSigs[0] = _sign(signerKeys[1], mt);
        vm.expectRevert(bytes("BridgeCommittee: Insufficient stake"));
        committee.verifySignatures(shortSigs, mt);
    }

    /// Limit update via quorum-signed `UPDATE_LIMIT` — totalLimit is
    /// changed by the message.
    function test_limitUpdate_quorumSignedUpdatesTotalLimit() public {
        uint64 newLimit = 9_999_999;
        // payload: sendingChainID(1) || newLimit(8 BE)
        bytes memory payload = abi.encodePacked(SOMA_CHAIN_ID, newLimit);
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.UPDATE_LIMIT,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: 0,
            chainID: ETH_CHAIN_ID,
            payload: payload
        });
        bytes[] memory sigs = new bytes[](3);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[1], m);
        sigs[2] = _sign(signerKeys[2], m);
        limiter.updateLimitWithSignatures(sigs, m);
        assertEq(limiter.totalLimit(), newLimit);
    }

    // ============================================================
    // Limiter window calculation + garbage collection
    // ============================================================

    /// 24-hour sliding-window arithmetic. Two transfers an hour
    /// apart contribute to today's window; one slides off after 24h.
    /// Mirrors Sui's `testCalculateWindowLimit`.
    function test_limiter_windowSlidesOver24Hours() public {
        // Warp to a clean hour boundary so all `currentHour()` reads
        // line up with the buckets we write.
        vm.warp(block.timestamp - (block.timestamp % 1 hours));

        // Limiter's `recordUSDCTransfer` is gated by `onlyOwner` (the
        // bridge). Impersonate the bridge for the duration.
        vm.startPrank(address(bridge));

        uint64 amount = 1_000_000; // 1 USDC raw → 10_000 USD-scale
        limiter.recordUSDCTransfer(amount);
        // Now-window includes 1 USDC.
        assertEq(limiter.calculateWindowAmount(), 10_000);

        skip(1 hours);
        limiter.recordUSDCTransfer(amount * 2);
        // Now-window: 1 + 2 = 3 USDC.
        assertEq(limiter.calculateWindowAmount(), 30_000);

        skip(22 hours);
        // 23h since first; first bucket still in the 24h trailing window.
        assertEq(limiter.calculateWindowAmount(), 30_000);

        skip(59 minutes);
        // 23h 59m since first — still in the window.
        assertEq(limiter.calculateWindowAmount(), 30_000);

        skip(1 minutes);
        // 24h since first — first bucket has slid off. Only the 2-USDC
        // second bucket remains.
        assertEq(limiter.calculateWindowAmount(), 20_000);

        vm.stopPrank();
    }

    /// The 25-hour-old bucket is deleted on the next write (best-
    /// effort garbage collection). Mirrors Sui's
    /// `testrecordBridgeTransfersGarbageCollection`.
    function test_limiter_garbageCollectsOldBucket() public {
        vm.warp(block.timestamp - (block.timestamp % 1 hours));
        uint32 hourToDelete = limiter.currentHour();

        vm.startPrank(address(bridge));
        uint64 amount = 1_000_000;
        limiter.recordUSDCTransfer(amount);
        assertEq(limiter.hourlyTransferAmount(hourToDelete), 10_000);

        // Skip 25 hours and write again. The 25-hour-old bucket
        // (= hourToDelete) must be cleared.
        skip(25 hours);
        limiter.recordUSDCTransfer(amount);
        assertEq(limiter.hourlyTransferAmount(hourToDelete), 0);
        vm.stopPrank();
    }

    /// Direct check of `willAmountExceedLimit` — boundary case
    /// (just-under, just-over, and a value that brings the running
    /// window to exactly the cap).
    function test_limiter_willAmountExceedLimitBoundary() public {
        vm.warp(block.timestamp - (block.timestamp % 1 hours));
        vm.startPrank(address(bridge));

        // Configure a tight limit by re-deploying the limiter with
        // totalLimit = 30000 USD-scale = 3 USDC equivalent.
        BridgeLimiter tight = new BridgeLimiter();
        tight.initialize(address(committee), 30_000);
        tight.transferOwnership(address(bridge));
        vm.stopPrank();

        vm.startPrank(address(bridge));
        // 0 USDC consumed; 3 USDC fits exactly.
        assertFalse(tight.willAmountExceedLimit(3_000_000));
        // usdcToUsdScale rounds down — smallest amount that actually
        // bumps the USD-scale total is 100 raw (= 0.0001 USDC).
        assertTrue(tight.willAmountExceedLimit(3_000_100));

        // After spending 2 USDC, 1 more fits exactly; 1.01 over.
        tight.recordUSDCTransfer(2_000_000);
        assertFalse(tight.willAmountExceedLimit(1_000_000));
        assertTrue(tight.willAmountExceedLimit(1_010_000));
        vm.stopPrank();
    }

    // ============================================================
    // Payload decoder unit tests (Sui BridgeUtilsTest parity)
    // ============================================================

    /// Direct unit test of the blocklist payload decoder — Sui parity
    /// with `testDecodeBlocklistPayload`.
    function test_decodeBlocklistPayload() public pure {
        // blocklistType=0 (blocklist) + count=2 + two 20-byte addresses
        bytes memory payload = abi.encodePacked(
            uint8(0),
            uint8(2),
            address(0x68B43fD906C0B8F024a18C56e06744F7c6157c65),
            address(0xaCAEf39832CB995c4E049437A3E2eC6a7bad1Ab5)
        );
        (bool isBlocklisting, address[] memory members) =
            BridgeMessage.decodeBlocklistPayload(payload);
        assertTrue(isBlocklisting);
        assertEq(members.length, 2);
        assertEq(members[0], 0x68B43fD906C0B8F024a18C56e06744F7c6157c65);
        assertEq(members[1], 0xaCAEf39832CB995c4E049437A3E2eC6a7bad1Ab5);
    }

    /// 3-address blocklist payload + unblocklist flag — covers the
    /// loop offset arithmetic for more than 2 entries.
    function test_decodeBlocklistPayload_threeAddresses() public pure {
        address m1 = address(0x1111111111111111111111111111111111111111);
        address m2 = address(0x2222222222222222222222222222222222222222);
        address m3 = address(0x3333333333333333333333333333333333333333);
        bytes memory payload =
            abi.encodePacked(uint8(1), uint8(3), m1, m2, m3); // unblocklist
        (bool isBlocklisting, address[] memory members) =
            BridgeMessage.decodeBlocklistPayload(payload);
        assertFalse(isBlocklisting);
        assertEq(members.length, 3);
        assertEq(members[0], m1);
        assertEq(members[1], m2);
        assertEq(members[2], m3);
    }

    /// Blocklist payload with mismatched count + length is rejected.
    function test_decodeBlocklistPayload_rejectsBadLength() public {
        // count says 2 but only 1 address follows.
        bytes memory payload = abi.encodePacked(
            uint8(0),
            uint8(2),
            address(0x1111111111111111111111111111111111111111)
        );
        // vm.expectRevert requires the revert to happen one call frame
        // deeper than the test itself, so route through an external
        // wrapper on `this`.
        vm.expectRevert(bytes("BridgeMessage: Blocklist payload length mismatch"));
        this.callDecodeBlocklistPayload(payload);
    }

    /// External wrapper so internal-library calls can be caught by
    /// `vm.expectRevert` (which fires only on the next external call).
    function callDecodeBlocklistPayload(bytes calldata payload)
        external
        pure
        returns (bool, address[] memory)
    {
        return BridgeMessage.decodeBlocklistPayload(payload);
    }

    /// Direct unit test of the limit-update payload decoder.
    function test_decodeUpdateLimitPayload() public pure {
        // sendingChainID=2 (SomaCustom), newLimit=100 * 1e4 = 1_000_000
        bytes memory payload = abi.encodePacked(uint8(2), uint64(1_000_000));
        (uint8 src, uint64 lim) = BridgeMessage.decodeUpdateLimitPayload(payload);
        assertEq(src, 2);
        assertEq(lim, 1_000_000);
    }

    /// Emergency-op payload decoder.
    function test_decodeEmergencyOpPayload() public pure {
        bytes memory freeze = hex"00";
        bytes memory unfreeze = hex"01";
        assertTrue(BridgeMessage.decodeEmergencyOpPayload(freeze));
        assertFalse(BridgeMessage.decodeEmergencyOpPayload(unfreeze));
    }

    /// Emergency-op decoder rejects invalid op codes (anything > 1).
    function test_decodeEmergencyOpPayload_rejectsInvalidCode() public {
        bytes memory invalid = hex"02";
        vm.expectRevert(bytes("BridgeMessage: Invalid emergency op code"));
        this.callDecodeEmergencyOpPayload(invalid);
    }

    /// External wrapper (see callDecodeBlocklistPayload for rationale).
    function callDecodeEmergencyOpPayload(bytes calldata payload)
        external
        pure
        returns (bool)
    {
        return BridgeMessage.decodeEmergencyOpPayload(payload);
    }

    /// Upgrade payload decoder round-trips abi-encoded
    /// (proxy, impl, callData). Sui parity: `testDecodeUpgradePayload`.
    function test_decodeUpgradePayload() public pure {
        address proxy = address(100);
        address impl = address(200);
        bytes memory cd = hex"deadbeef";
        bytes memory payload = abi.encode(proxy, impl, cd);
        (address gotProxy, address gotImpl, bytes memory gotCd) =
            BridgeMessage.decodeUpgradePayload(payload);
        assertEq(gotProxy, proxy);
        assertEq(gotImpl, impl);
        assertEq(gotCd, cd);
    }

    /// `getRequiredStake` rejects unknown message types so a forged
    /// type byte can't pass the verifier with zero stake.
    function test_getRequiredStake_rejectsInvalidType() public {
        BridgeMessage.Message memory bogus = BridgeMessage.Message({
            messageType: 99,
            version: 1,
            nonce: 0,
            chainID: ETH_CHAIN_ID,
            payload: hex"00"
        });
        bytes[] memory empty = new bytes[](0);
        vm.expectRevert(bytes("BridgeMessage: Invalid message type"));
        committee.verifySignatures(empty, bogus);
    }

    /// Same-action-twice replay-defense: the per-message-type nonce
    /// is incremented after a successful call, so the same signed
    /// bytes can't be replayed.
    function test_limitUpdate_replayBlockedByNonce() public {
        bytes memory payload = abi.encodePacked(SOMA_CHAIN_ID, uint64(100));
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.UPDATE_LIMIT,
            version: BridgeMessage.SYSTEM_MESSAGE_VERSION,
            nonce: 0,
            chainID: ETH_CHAIN_ID,
            payload: payload
        });
        bytes[] memory sigs = new bytes[](3);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[1], m);
        sigs[2] = _sign(signerKeys[2], m);
        limiter.updateLimitWithSignatures(sigs, m);
        vm.expectRevert(bytes("MessageVerifier: Invalid nonce"));
        limiter.updateLimitWithSignatures(sigs, m);
    }
}
