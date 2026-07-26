// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "../interfaces/IBridgeCommittee.sol";

/// @title MessageVerifier
/// @notice Mixin that combines two checks every quorum-signed bridge message
/// must pass before the calling contract acts on it:
///   1. The committee verifies the signatures meet the required stake for
///      the message's action type (delegated to [`BridgeCommittee.verifySignatures`]).
///   2. The message header is consistent with this Eth chain + nonce schedule:
///      - chain id matches (system messages only — token transfers carry the
///        SOURCE chain id, not the receiving one).
///      - per-type sequence number is exactly the expected next value
///        (token transfers use a separate per-message-key flag tracked in
///        the bridge contract).
///
/// On success the per-type nonce is incremented atomically with the action,
/// so a replay of the same quorum-signed bytes reverts.
abstract contract MessageVerifier is Initializable {
    IBridgeCommittee public committee;

    /// @dev `messageType => next expected nonce`. Token transfer messages are
    /// nonce-checked by the bridge contract via `isMessageProcessed` instead,
    /// so this map is not bumped for them.
    mapping(uint8 => uint64) public nonces;

    function __MessageVerifier_init(address _committee) internal onlyInitializing {
        committee = IBridgeCommittee(_committee);
    }

    /// @dev Modifier that runs every check before the wrapped body executes.
    /// Reverts if the message can't be acted on; otherwise increments the
    /// system-message nonce so a replay of the same signed bytes fails next
    /// time around.
    modifier verifyMessageAndSignatures(
        BridgeMessage.Message memory message,
        bytes[] memory signatures,
        uint8 messageType
    ) {
        require(
            message.messageType == messageType,
            "MessageVerifier: Message does not match expected type"
        );
        committee.verifySignatures(signatures, message);
        if (messageType != BridgeMessage.USDC_WITHDRAW) {
            // System messages (emergency op, blocklist, limit update, upgrade,
            // committee update) target this Eth chain; their header chain id
            // must match.
            require(
                message.chainID == committee.chainID(),
                "MessageVerifier: Invalid chain ID"
            );
            require(
                message.nonce == nonces[message.messageType],
                "MessageVerifier: Invalid nonce"
            );
            nonces[message.messageType]++;
        }
        _;
    }
}
