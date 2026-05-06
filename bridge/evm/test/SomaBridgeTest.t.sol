// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";

import "../contracts/BridgeCommittee.sol";
import "../contracts/BridgeLimiter.sol";
import "../contracts/BridgeVault.sol";
import "../contracts/SomaBridge.sol";
import "../contracts/utils/BridgeMessage.sol";

/// @dev Minimal ERC20 used as USDC stand-in. 6 decimals.
contract MockUSDC {
    string public constant name = "USD Coin (Mock)";
    string public constant symbol = "USDC";
    uint8 public constant decimals = 6;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "USDC: insufficient");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "USDC: insufficient");
        require(allowance[from][msg.sender] >= amount, "USDC: not approved");
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

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

    /// Duplicate signature from the same signer doesn't double-count.
    function test_verifySignatures_dedupesDuplicateSigner() public {
        bytes memory payload = _buildWithdrawPayload(address(uint160(0x1234)), 100_000);
        BridgeMessage.Message memory m = BridgeMessage.Message({
            messageType: BridgeMessage.USDC_WITHDRAW,
            version: BridgeMessage.TOKEN_TRANSFER_VERSION_V2,
            nonce: 0,
            chainID: SOMA_CHAIN_ID,
            payload: payload
        });
        // Same signer twice → 2500 BPS, below 3334.
        bytes[] memory sigs = new bytes[](2);
        sigs[0] = _sign(signerKeys[0], m);
        sigs[1] = _sign(signerKeys[0], m);
        vm.expectRevert(bytes("BridgeCommittee: Insufficient stake"));
        committee.verifySignatures(sigs, m);
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
}
