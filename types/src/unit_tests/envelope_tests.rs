// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::base::SomaAddress;
use crate::crypto::*;
use crate::digests::ObjectDigest;
use crate::envelope::{Envelope, TrustedEnvelope, VerifiedEnvelope};
use crate::object::{ObjectID, Version};
use crate::transaction::*;

/// Helper: build a minimal SenderSignedData for testing.
fn make_sender_signed_data() -> SenderSignedData {
    let sender = SomaAddress::default();
    let obj_id = ObjectID::random();
    let obj_ref = (obj_id, Version::from_u64(1), ObjectDigest::new([1u8; 32]));
    let recipient = SomaAddress::random();

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: obj_ref, amount: Some(100), recipient },
        sender,
        vec![obj_ref],
    );

    // Use a dummy signature (all zeros via Default).
    let dummy_sig = Ed25519SomaSignature::default();
    SenderSignedData::new_from_sender_signature(tx_data, dummy_sig.into())
}

#[test]
fn test_envelope_creation_empty_sig() {
    let data = make_sender_signed_data();
    let envelope = Envelope::<SenderSignedData, EmptySignInfo>::new(data.clone());

    // The envelope wraps the same data.
    assert_eq!(*envelope.data(), data);
    // The auth_signature is EmptySignInfo.
    assert_eq!(*envelope.auth_sig(), EmptySignInfo {});
}

#[test]
fn test_envelope_data_access() {
    let data = make_sender_signed_data();
    let envelope = Envelope::<SenderSignedData, EmptySignInfo>::new(data.clone());

    // .data() returns a reference equal to the original.
    assert_eq!(envelope.data(), &data);

    // .into_data() consumes the envelope and returns the inner data.
    let recovered = envelope.into_data();
    assert_eq!(recovered, data);
}

#[test]
fn test_envelope_digest_caching() {
    let data = make_sender_signed_data();
    let envelope = Envelope::<SenderSignedData, EmptySignInfo>::new(data);

    // First call computes the digest.
    let d1 = envelope.digest().clone();
    // Second call should return the cached value.
    let d2 = envelope.digest().clone();

    assert_eq!(d1, d2, "digest() must be deterministic and cached via OnceLock");
}

#[test]
fn test_envelope_into_data_and_sig() {
    let data = make_sender_signed_data();
    let envelope = Envelope::<SenderSignedData, EmptySignInfo>::new(data.clone());

    let (recovered_data, recovered_sig) = envelope.into_data_and_sig();
    assert_eq!(recovered_data, data);
    assert_eq!(recovered_sig, EmptySignInfo {});
}

#[test]
fn test_verified_envelope_from_trusted() {
    let data = make_sender_signed_data();
    let envelope = Envelope::<SenderSignedData, EmptySignInfo>::new(data.clone());

    // Wrap in TrustedEnvelope via serialization roundtrip.
    let bytes = bcs::to_bytes(&envelope).expect("BCS serialize should succeed");
    let trusted: TrustedEnvelope<SenderSignedData, EmptySignInfo> =
        bcs::from_bytes(&bytes).expect("BCS deserialize should succeed");

    // Convert TrustedEnvelope -> VerifiedEnvelope via From.
    let verified: VerifiedEnvelope<SenderSignedData, EmptySignInfo> = trusted.into();

    // The inner data should match the original.
    assert_eq!(*verified.data(), data);
}

#[test]
fn test_envelope_deref() {
    let data = make_sender_signed_data();
    let envelope = Envelope::<SenderSignedData, EmptySignInfo>::new(data.clone());

    // Envelope<T, S> derefs to T (SenderSignedData).
    // SenderSignedData has .transaction_data() method.
    let tx_data_via_deref = envelope.transaction_data();
    let tx_data_via_data = envelope.data().transaction_data();

    assert_eq!(tx_data_via_deref, tx_data_via_data);
}

#[test]
fn test_envelope_bcs_roundtrip() {
    let data = make_sender_signed_data();
    let envelope = Envelope::<SenderSignedData, EmptySignInfo>::new(data);

    let bytes = bcs::to_bytes(&envelope).expect("BCS serialize should succeed");
    let deserialized: Envelope<SenderSignedData, EmptySignInfo> =
        bcs::from_bytes(&bytes).expect("BCS deserialize should succeed");

    // Envelopes should be equal (data + auth_signature match).
    assert_eq!(envelope, deserialized);

    // Digests should match after deserialization.
    assert_eq!(envelope.digest(), deserialized.digest());
}
