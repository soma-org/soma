//! ABI encoding for the Eth-side bridge contract.
//!
//! Generates the calldata that goes into `eth_sendRawTransaction` for
//! the outbound release flow: `SomaBridge.transferBridgedTokensWith-
//! Signatures(bytes[] signatures, Message message)`. The contract's
//! function selector + arg layout are pinned via `alloy::sol!` so a
//! mismatch between the off-chain encoder and the on-chain ABI fails
//! at compile time.
//!
//! Companion to [`crate::outbound_relayer`]: that module owns the
//! polling loop + idempotency tracker; this module owns the bytes
//! that go on the wire.

use alloy::sol;
use alloy::sol_types::SolCall;
use types::bridge::WithdrawalCertificate;

use crate::error::{BridgeError, BridgeResult};
use crate::types::BridgeAction;

// `sol!` lifts the Solidity ABI signatures we care about into Rust
// types with `selector()` + `abi_encode()` impls. Names must match
// the on-chain contract verbatim — the compiler enforces this at
// every call site below.
sol! {
    /// Mirror of `BridgeMessage.Message` in `bridge/evm/`. Field
    /// order is load-bearing — `(uint8, uint8, uint64, uint8, bytes)`
    /// must match the Solidity struct exactly so the selector hashes
    /// the same way.
    struct Message {
        uint8 messageType;
        uint8 version;
        uint64 nonce;
        uint8 chainID;
        bytes payload;
    }

    /// The release function on `SomaBridge`. The off-chain relayer
    /// builds calldata for this and submits it via `eth_sendRawTrans-
    /// action`. Anyone holding the cert can call it; the contract's
    /// `verifyMessageAndSignatures` modifier re-checks the cert.
    function transferBridgedTokensWithSignatures(
        bytes[] memory signatures,
        Message memory message
    ) external;
}

/// Build the ABI-encoded calldata for releasing a withdrawal on
/// Ethereum. The caller (outbound relayer) wraps this in an EIP-1559
/// tx envelope, signs it with the operator's Eth wallet, and submits
/// via `eth_sendRawTransaction`.
///
/// **Crucial invariant**: the `message_bytes` passed here MUST be
/// the exact canonical bytes the committee signed. Reconstructing
/// from any subset of fields (e.g. recipient + amount) silently
/// drops the original sender / timestamp / etc. — the contract
/// re-hashes the full Message and any field mismatch makes
/// `ecrecover` recover the wrong address, failing verification.
///
/// Inputs:
///   - `message_bytes` — the 102-byte V2 canonical message
///     ([`BridgeAction::to_message_bytes`] output for a `Withdrawal`).
///   - `cert` — the labeled-pubkey signature envelope. Flattened
///     into a `bytes[]` (the contract's wire format) in BTreeMap
///     order. The contract's verifier ecrecovers each sig and
///     dedups by recovered pubkey, so order doesn't affect
///     acceptance.
pub fn encode_release_calldata(
    message_bytes: &[u8],
    cert: &WithdrawalCertificate,
) -> BridgeResult<Vec<u8>> {
    // 30-byte header: prefix(19) + type(1) + version(1) + nonce(8) + chainID(1).
    if message_bytes.len() != 30 + 72 {
        return Err(BridgeError::Internal(format!(
            "encode_release_calldata: expected 102-byte V2 withdrawal, got {}",
            message_bytes.len()
        )));
    }
    // Prefix sanity check — a 102-byte buffer that doesn't start with
    // SOMA_BRIDGE_MESSAGE is a programmer error, not a wire format
    // we should silently submit.
    if !message_bytes.starts_with(types::bridge::BRIDGE_MESSAGE_PREFIX) {
        return Err(BridgeError::Internal(
            "encode_release_calldata: message_bytes prefix mismatch".to_string(),
        ));
    }

    let message_type = message_bytes[19];
    if message_type != types::bridge::BridgeMessageType::UsdcWithdraw as u8 {
        return Err(BridgeError::Internal(format!(
            "encode_release_calldata: not a USDC_WITHDRAW message (type={message_type})"
        )));
    }
    let version = message_bytes[20];
    let mut nonce_bytes = [0u8; 8];
    nonce_bytes.copy_from_slice(&message_bytes[21..29]);
    let nonce = u64::from_be_bytes(nonce_bytes);
    let chain_id = message_bytes[29];
    let payload = message_bytes[30..].to_vec();

    let message = Message {
        messageType: message_type,
        version,
        nonce,
        chainID: chain_id,
        payload: payload.into(),
    };

    // Flatten the labeled-pubkey envelope into a Vec<Bytes>. The
    // contract iterates this in order, ecrecovers each, and tallies
    // stake.
    let signatures: Vec<alloy::primitives::Bytes> = cert
        .signatures
        .values()
        .map(|sig| alloy::primitives::Bytes::from(sig.as_bytes().to_vec()))
        .collect();

    let call = transferBridgedTokensWithSignaturesCall { signatures, message };
    Ok(call.abi_encode())
}

/// Convenience wrapper that derives the message bytes from a typed
/// `BridgeAction::Withdrawal`. Useful for callers that own the
/// action (e.g. tests). Production callers should prefer
/// [`encode_release_calldata`] directly so the bytes flow through
/// the system byte-identically to what the committee signed.
pub fn encode_release_calldata_from_action(
    action: &BridgeAction,
    cert: &WithdrawalCertificate,
) -> BridgeResult<Vec<u8>> {
    if !matches!(action, BridgeAction::Withdrawal { .. }) {
        return Err(BridgeError::Internal(format!(
            "encode_release_calldata_from_action: expected Withdrawal, got {:?}",
            action.message_type()
        )));
    }
    let bytes = action.to_message_bytes();
    encode_release_calldata(&bytes, cert)
}

/// Function selector for `transferBridgedTokensWithSignatures`. Used
/// by tests + log lines to assert the encoder generated the right
/// call without parsing the full calldata.
pub fn release_function_selector() -> [u8; 4] {
    transferBridgedTokensWithSignaturesCall::SELECTOR
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use types::base::SomaAddress;
    use types::bridge::{BridgeChainId, BridgePubkey, BridgeSignature};

    fn sample_withdrawal() -> BridgeAction {
        BridgeAction::Withdrawal {
            nonce: 7,
            sender: SomaAddress::from([0xAA; 32]),
            target_chain: BridgeChainId::EthCustom,
            recipient_eth_address: [0xBB; 20],
            token_type: types::bridge::USDC_TOKEN_TYPE,
            amount: 1_000_000,
            timestamp_ms: 1_700_000_000_000,
        }
    }

    fn sample_cert() -> WithdrawalCertificate {
        let mut sigs = BTreeMap::new();
        // Generator point — a real curve point so BridgePubkey ctor accepts it.
        let pk = BridgePubkey::from_bytes(&[
            0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce,
            0x87, 0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81,
            0x5b, 0x16, 0xf8, 0x17, 0x98,
        ])
        .unwrap();
        sigs.insert(pk, BridgeSignature::from_bytes(&[0xCC; 65]).unwrap());
        WithdrawalCertificate {
            signatures: sigs,
            attached_at_epoch: 0,
        }
    }

    /// **Cross-language selector proof.** The first 4 bytes of
    /// `keccak256("transferBridgedTokensWithSignatures(bytes[],(uint8,uint8,uint64,uint8,bytes))")`
    /// = `0xbeb0d55c`. Verified to match the Solidity contract's
    /// runtime selector via `forge inspect SomaBridge methodIdentifiers`.
    /// If either side renames the function or changes a field type,
    /// this test fails immediately — no silent ABI drift across the
    /// Rust↔Solidity boundary.
    #[test]
    fn selector_matches_solidity_signature() {
        const EXPECTED: [u8; 4] = [0xbe, 0xb0, 0xd5, 0x5c];
        assert_eq!(release_function_selector(), EXPECTED);
    }

    /// Calldata starts with the selector + has the expected length
    /// for a 1-signature, 72-byte-payload withdrawal release call.
    #[test]
    fn calldata_starts_with_selector() {
        let calldata = encode_release_calldata_from_action(
            &sample_withdrawal(),
            &sample_cert(),
        )
        .unwrap();
        assert_eq!(&calldata[..4], &release_function_selector());
        // Calldata is dynamic in length depending on payload + sig
        // count; a 1-sig 72-byte-payload call comes in at >= 4 + 32 * 12 bytes.
        assert!(calldata.len() > 200, "calldata implausibly short: {}", calldata.len());
    }

    /// Non-Withdrawal actions are programmer errors — the relayer
    /// only ever calls this from the outbound polling loop, which
    /// already filters to Withdrawal variants.
    #[test]
    fn rejects_non_withdrawal_action() {
        let pause = BridgeAction::EmergencyPause { nonce: 0 };
        let err =
            encode_release_calldata_from_action(&pause, &sample_cert()).unwrap_err();
        assert!(matches!(err, BridgeError::Internal(_)));
    }

    /// Defense-in-depth: a wrong-prefix buffer or a non-WITHDRAW
    /// message type is rejected, so an upstream bug can't silently
    /// produce calldata for the wrong message.
    #[test]
    fn rejects_wrong_length_or_prefix() {
        let cert = sample_cert();
        // Wrong length
        let err = encode_release_calldata(&[0; 50], &cert).unwrap_err();
        assert!(matches!(err, BridgeError::Internal(_)));
        // Right length, wrong prefix
        let err = encode_release_calldata(&[0; 102], &cert).unwrap_err();
        assert!(matches!(err, BridgeError::Internal(_)));
    }
}
