// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

import "./interfaces/IBridgeCommittee.sol";
import "./utils/CommitteeUpgradeable.sol";

/// @title BridgeCommittee
/// @notice The on-chain committee that authorizes every quorum-signed bridge
/// action. Stores stake-weighted membership + the chain id; verifies
/// recoverable secp256k1 signatures via `ecrecover` and counts non-blocklisted
/// stake until the action's required threshold is met.
///
/// Soma's design mirrors Sui's `BridgeCommittee.sol`:
/// - Members keyed by Eth address (recovered from the signature; the on-chain
///   contract never sees a 33-byte compressed pubkey directly).
/// - Per-member voting power in BPS; total must be exactly 10000 at init.
/// - Blocklist is a simple `address => bool`; flipped via a quorum-signed
///   `BLOCKLIST` message (`updateBlocklistWithSignatures`).
/// - Self-upgradeable through `CommitteeUpgradeable.upgradeWithSignatures`.
contract BridgeCommittee is IBridgeCommittee, CommitteeUpgradeable {
    /// @notice The Eth chain this committee authorizes against. Compared to
    /// `BridgeMessage.chainID` for system messages.
    uint8 public chainID;

    /// @notice Per-member voting power in BPS (basis points; 10000 = 100%).
    /// Zero means "not a committee member."
    mapping(address => uint16) public committeeStake;

    /// @notice Per-member position in the bitmap used during dedup-by-signer
    /// in `verifySignatures`. 0..N-1 in init order; unused for non-members.
    mapping(address => uint8) public committeeIndex;

    /// @notice Members on the blocklist contribute zero stake to verification
    /// regardless of `committeeStake`. Flipped via `updateBlocklistWithSignatures`.
    mapping(address => bool) public blocklist;

    /// @notice Initialize the committee.
    /// @dev Must be called immediately after proxy deployment (UUPS pattern).
    /// `members.length == stake.length` and `sum(stake) == 10000` are
    /// enforced — total stake must be exactly the full 10000 BPS so the
    /// threshold comparisons in `BridgeMessage.getRequiredStake` are absolute,
    /// not relative.
    function initialize(
        address[] memory _committeeMembers,
        uint16[] memory _stake,
        uint8 _chainID
    ) external initializer {
        // Init parents in C3 linearization order (deepest first), so
        // the OZ Foundry upgrades plugin's parent-init order check is
        // satisfied. See CommitteeUpgradeable.__CommitteeUpgradeable_init
        // doc for why these calls live here rather than in the wrapper.
        //
        // Bootstrap the committee reference *to ourselves* — verification
        // calls flow back through `committee.verifySignatures()`, so this
        // contract is both the verifier and the verified.
        __ReentrancyGuard_init();
        __MessageVerifier_init(address(this));
        __UUPSUpgradeable_init();
        __CommitteeUpgradeable_init(address(this));

        uint256 _committeeLength = _committeeMembers.length;
        // Sui parity: the dedup bitmap in verifySignatures is uint256
        // so 255 is the hard ceiling on member count. Use the same
        // verbatim error string Sui uses for byte-for-byte test parity.
        require(
            _committeeLength < 256,
            "BridgeCommittee: Committee length must be less than 256"
        );
        require(
            _committeeLength == _stake.length,
            "BridgeCommittee: Committee and stake arrays must be of the same length"
        );

        uint16 totalStake = 0;
        for (uint16 i = 0; i < _committeeLength; i++) {
            address member = _committeeMembers[i];
            require(
                committeeStake[member] == 0,
                "BridgeCommittee: Duplicate committee member"
            );
            committeeStake[member] = _stake[i];
            // forge-lint: disable-next-line(unsafe-typecast)
            committeeIndex[member] = uint8(i);
            totalStake += _stake[i];
        }
        require(totalStake == 10000, "BridgeCommittee: Total stake must equal 10000 BPS");

        // committee field on MessageVerifier is set inside
        // __CommitteeUpgradeable_init → __MessageVerifier_init(address(this)).
        chainID = _chainID;
    }

    /// @inheritdoc IBridgeCommittee
    function verifySignatures(
        bytes[] memory signatures,
        BridgeMessage.Message memory message
    ) public view override {
        uint32 requiredStake = BridgeMessage.getRequiredStake(message);

        uint16 approvalStake;
        address signer;
        uint256 bitmap;
        bytes32 digest = BridgeMessage.computeHash(message);

        for (uint16 i = 0; i < signatures.length; i++) {
            (bytes32 r, bytes32 s, uint8 v) = _splitSignature(signatures[i]);
            (signer,,) = ECDSA.tryRecover(digest, v, r, s);

            // Reject signers that aren't in the committee at all. Sui
            // parity: a sig from a non-member is a malformed cert, not
            // a "skip and try the rest" condition — fail loudly so the
            // off-chain aggregator doesn't waste gas re-submitting.
            require(
                committeeStake[signer] != 0,
                "BridgeCommittee: Signer has no stake"
            );
            // Blocklisted committee members contribute zero stake. Sui
            // parity: their presence in the array is rejected outright.
            require(!blocklist[signer], "BridgeCommittee: Signer is blocklisted");

            uint8 idx = committeeIndex[signer];
            uint256 mask = uint256(1) << idx;
            // Sui parity: a duplicate signer in the sig array is a
            // malformed cert; reject rather than silently skip.
            require(
                bitmap & mask == 0,
                "BridgeCommittee: Duplicate signature provided"
            );
            bitmap |= mask;
            approvalStake += committeeStake[signer];
        }

        require(
            approvalStake >= requiredStake,
            "BridgeCommittee: Insufficient stake"
        );
    }

    /// @notice Flip the blocklist flag for one or more members.
    /// @dev Gated by a quorum-signed `BLOCKLIST` message. The wire format
    /// pre-derives Eth addresses on the Rust side (see Soma's
    /// `derive_eth_address` + `encode_blocklist_payload`), so the on-chain
    /// decoder is straightforward — no secp256k1 pubkey decompression.
    function updateBlocklistWithSignatures(
        bytes[] memory signatures,
        BridgeMessage.Message memory message
    )
        external
        nonReentrant
        verifyMessageAndSignatures(message, signatures, BridgeMessage.BLOCKLIST)
    {
        (bool isBlocklisted, address[] memory members) =
            BridgeMessage.decodeBlocklistPayload(message.payload);

        for (uint16 i = 0; i < members.length; i++) {
            blocklist[members[i]] = isBlocklisted;
        }
        emit BlocklistUpdated(members, isBlocklisted);
    }

    /// @dev Split a 65-byte recoverable signature into `(r, s, v)`. Tolerates
    /// the two common conventions for the v byte (27/28 vs 0/1) by adjusting
    /// before returning.
    function _splitSignature(bytes memory sig)
        internal
        pure
        returns (bytes32 r, bytes32 s, uint8 v)
    {
        require(sig.length == 65, "BridgeCommittee: Signature must be 65 bytes");
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
        if (v < 27) v += 27;
    }
}
