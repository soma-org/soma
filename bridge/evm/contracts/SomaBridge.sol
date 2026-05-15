// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

import "./interfaces/IBridgeLimiter.sol";
import "./interfaces/IBridgeVault.sol";
import "./interfaces/ISomaBridge.sol";
import "./utils/CommitteeUpgradeable.sol";

/// @title SomaBridge
/// @notice Entry contract for the Soma ↔ Eth USDC bridge.
///
/// Adapted from Sui's `SuiBridge.sol`. Differences:
/// - **USDC-only**: no token registry, no multi-token branching, no WETH9
///   wrap/unwrap, no `IBridgeTokens` indirection.
/// - **One source chain (Soma)**: no `isChainSupported` mapping; the
///   `MessageVerifier` checks `chainID` directly against `committee.chainID()`
///   for system messages, and outbound withdrawals carry the Soma chain id
///   verbatim.
/// - **V2 token transfer payload**: 36-byte `(ethRecipient, amount, timestampMs)`
///   instead of Sui's 64-byte multi-token payload.
///
/// User flow (Eth → Soma):
///   1. User approves USDC to this contract.
///   2. User calls `deposit(somaRecipient, amount)` — locks USDC in the vault,
///      emits `TokensDeposited` with the V2 fields the bridge node syncer
///      indexes.
///
/// Quorum flow (Soma → Eth):
///   1. Soma chain creates a `PendingWithdrawal`. Off-chain bridge nodes sign.
///   2. A relayer calls `transferBridgedTokensWithSignatures` with the
///      quorum-signed `USDC_WITHDRAW` message.
///   3. The contract verifies signatures, checks the rolling limit, releases
///      USDC from the vault to the recipient.
contract SomaBridge is ISomaBridge, CommitteeUpgradeable, PausableUpgradeable {
    /// @notice USDC ERC20. Locked here from user deposits, transferred into the
    /// vault on the same call so this contract's USDC balance is always 0 at
    /// rest.
    IERC20 public usdc;

    /// @notice Vault holding locked USDC. Owner-gated; only this bridge can
    /// release.
    IBridgeVault public vault;

    /// @notice Sliding-window rate limiter for outbound transfers.
    IBridgeLimiter public limiter;

    /// @notice Per-nonce idempotency for outbound `USDC_WITHDRAW` messages.
    /// Soma's bridge node uses nonces from `BridgeState.next_withdrawal_nonce`;
    /// this map ensures the same quorum-signed message can't drain the vault
    /// twice if a peer accidentally re-submits.
    mapping(uint64 => bool) public isMessageProcessed;

    /// @notice Set of Soma chain ids this deployment trusts as
    /// source/destination. Mirrors Sui's `SuiBridge.isChainSupported`
    /// — gates both outbound deposits (the user-supplied
    /// `destinationChainID`) and inbound withdrawals (the source chain
    /// in the message header).
    ///
    /// A single Eth deployment can route to multiple Soma chains —
    /// e.g. one Eth Sepolia contract supporting both `SomaTestnet` and
    /// `SomaCustom`, or one Eth mainnet contract supporting only
    /// `SomaMainnet`. The active set is fixed at `initialize` and can
    /// only be changed through a quorum-signed upgrade.
    mapping(uint8 => bool) public isChainSupported;

    /// @notice Initialize the bridge against its committee + vault + limiter.
    /// @param _committee Deployed `BridgeCommittee`.
    /// @param _usdc USDC ERC20 address (must match the address the vault was deployed with).
    /// @param _vault Deployed `BridgeVault` (ownership must be transferred to this bridge).
    /// @param _limiter Deployed `BridgeLimiter` (ownership must be transferred to this bridge).
    /// @param _supportedChainIDs Soma chain ids this deployment trusts. At least one is required;
    /// none may equal the committee's own (Eth) `chainID`. See [`isChainSupported`].
    function initialize(
        address _committee,
        address _usdc,
        address _vault,
        address _limiter,
        uint8[] memory _supportedChainIDs
    ) external initializer {
        // Init parents in C3 linearization order; see
        // CommitteeUpgradeable.__CommitteeUpgradeable_init doc for the
        // OZ-plugin rationale.
        __ReentrancyGuard_init();
        __MessageVerifier_init(_committee);
        __UUPSUpgradeable_init();
        __CommitteeUpgradeable_init(_committee);
        __Pausable_init();
        require(_usdc != address(0), "SomaBridge: USDC address required");
        require(_vault != address(0), "SomaBridge: Vault address required");
        require(_limiter != address(0), "SomaBridge: Limiter address required");
        require(
            _supportedChainIDs.length > 0,
            "SomaBridge: At least one supported chain required"
        );
        usdc = IERC20(_usdc);
        vault = IBridgeVault(_vault);
        limiter = IBridgeLimiter(_limiter);

        uint8 selfChainID = IBridgeCommittee(_committee).chainID();
        for (uint256 i = 0; i < _supportedChainIDs.length; i++) {
            require(
                _supportedChainIDs[i] != selfChainID,
                "SomaBridge: Cannot support self as remote chain"
            );
            isChainSupported[_supportedChainIDs[i]] = true;
        }
    }

    /// @dev Gate any user-facing or message-processing call on whether
    /// the relevant chain id is in the trusted set.
    modifier onlySupportedChain(uint8 chainID_) {
        require(isChainSupported[chainID_], "SomaBridge: Chain not supported");
        _;
    }

    /* ========== USER ENTRY (Eth → Soma) ========== */

    /// @notice Lock `amount` USDC and emit a deposit event the Soma bridge
    /// node will pick up. Caller must have approved this contract for `amount`
    /// USDC beforehand.
    /// @param destinationChainID Soma chain to credit on. Must be in the
    /// `isChainSupported` set — same set the contract validates for
    /// inbound withdrawals. This lets a single Eth deployment route to
    /// multiple Soma chains (mainnet, testnet, custom) and avoids users
    /// silently sending funds to a chain the bridge node doesn't watch.
    /// @param somaRecipient 32-byte Soma address to credit on the Soma side.
    /// @param amount USDC raw units (6 decimals; not adjusted).
    function deposit(uint8 destinationChainID, bytes32 somaRecipient, uint64 amount)
        external
        whenNotPaused
        nonReentrant
        onlySupportedChain(destinationChainID)
    {
        require(amount > 0, "SomaBridge: Amount must be > 0");
        require(somaRecipient != bytes32(0), "SomaBridge: Recipient required");

        // Transfer USDC straight into the vault — the bridge contract itself
        // never holds funds.
        require(
            usdc.transferFrom(msg.sender, address(vault), amount),
            "SomaBridge: USDC transferFrom failed"
        );

        uint64 nonce = nonces[BridgeMessage.USDC_DEPOSIT];

        // Event field order/types MUST match `eth_client::parse_deposit_log`
        // on the bridge node side. See `ISomaBridge.TokensDeposited` doc.
        emit TokensDeposited(
            nonce,
            msg.sender,
            destinationChainID,
            somaRecipient,
            BridgeMessage.USDC_TOKEN_TYPE,
            amount,
            uint64(block.timestamp * 1000)
        );

        // Increment the per-message-type nonce. USDC_DEPOSIT is the only
        // outbound nonce the contract tracks; USDC_WITHDRAW is keyed by
        // the incoming message's nonce.
        nonces[BridgeMessage.USDC_DEPOSIT] = nonce + 1;
    }

    /* ========== QUORUM ENTRY (Soma → Eth) ========== */

    /// @notice Process a quorum-signed `USDC_WITHDRAW` message: verify sigs,
    /// check the rolling limit, release USDC from the vault.
    /// @dev The `message.chainID` is the SOURCE chain (Soma); we don't check
    /// it against this contract's chainID — `MessageVerifier` skips that
    /// check for `USDC_WITHDRAW`. Replay defense is `isMessageProcessed`
    /// keyed by the message nonce.
    function transferBridgedTokensWithSignatures(
        bytes[] memory signatures,
        BridgeMessage.Message memory message
    )
        external
        whenNotPaused
        nonReentrant
        verifyMessageAndSignatures(message, signatures, BridgeMessage.USDC_WITHDRAW)
        onlySupportedChain(message.chainID)
    {
        require(
            !isMessageProcessed[message.nonce],
            "SomaBridge: Withdrawal nonce already processed"
        );

        (
            bytes32 somaSender,
            uint8 targetChain,
            address ethRecipient,
            uint8 tokenType,
            uint64 amount,
            uint64 timestampMs
        ) = BridgeMessage.decodeUsdcWithdrawPayload(message.payload);

        // Withdrawal must terminate on *this* Eth chain. `message.chainID`
        // (which `onlySupportedChain` already gated as a known Soma source)
        // is the SOURCE for token-transfer messages; the destination is in
        // the payload — and must match this contract's chain id.
        require(
            targetChain == committee.chainID(),
            "SomaBridge: Withdrawal targets a different Eth chain"
        );
        // Lock in the current single-token deployment. When/if Soma adds
        // another token this check moves to per-token routing.
        require(
            tokenType == BridgeMessage.USDC_TOKEN_TYPE,
            "SomaBridge: Only USDC withdrawals supported"
        );
        require(ethRecipient != address(0), "SomaBridge: Zero recipient");
        require(amount > 0, "SomaBridge: Amount must be > 0");

        // Mark processed *before* the external transfer to defend against any
        // weird re-entry through a misbehaving USDC token; `nonReentrant`
        // already gates this but defense-in-depth costs nothing.
        isMessageProcessed[message.nonce] = true;

        // Limiter bypass for mature messages (Sui V2 parity — see
        // `BridgeMessage.isMatureMessage`). Quorum-signed withdrawals
        // older than 48h release without debiting the rolling-24h cap
        // so a long-delayed backlog doesn't starve fresh traffic.
        uint256 timestampSeconds = uint256(timestampMs) / 1000;
        if (!BridgeMessage.isMatureMessage(timestampSeconds, block.timestamp)) {
            limiter.recordUSDCTransfer(amount);
        }
        vault.transferUSDC(ethRecipient, amount);

        emit BridgedTokensTransferred(
            message.chainID,
            message.nonce,
            somaSender,
            tokenType,
            amount,
            ethRecipient
        );
    }

    /// @notice Pause/unpause the bridge via a quorum-signed `EMERGENCY_OP`.
    /// @dev Freeze threshold is intentionally low (450 BPS); unfreeze is 5001.
    /// See `BridgeMessage.getRequiredStake`.
    function executeEmergencyOpWithSignatures(
        bytes[] memory signatures,
        BridgeMessage.Message memory message
    )
        external
        nonReentrant
        verifyMessageAndSignatures(message, signatures, BridgeMessage.EMERGENCY_OP)
    {
        bool isFreezing = BridgeMessage.decodeEmergencyOpPayload(message.payload);
        if (isFreezing) {
            _pause();
        } else {
            _unpause();
        }
    }
}
