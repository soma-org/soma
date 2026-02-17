use crate::intent::*;

#[test]
fn test_intent_soma_transaction() {
    let intent = Intent::soma_transaction();
    assert_eq!(intent.scope, IntentScope::TransactionData);
    assert_eq!(intent.version, IntentVersion::V0);
    assert_eq!(intent.app_id, AppId::Soma);
}

#[test]
fn test_intent_to_bytes() {
    let intent = Intent::soma_transaction();
    let bytes = intent.to_bytes();

    // Intent serializes to exactly 3 bytes.
    assert_eq!(bytes.len(), INTENT_PREFIX_LENGTH);
    assert_eq!(bytes.len(), 3);

    // First byte is the scope (TransactionData = 0).
    assert_eq!(bytes[0], IntentScope::TransactionData as u8);
    // Second byte is the version (V0 = 0).
    assert_eq!(bytes[1], IntentVersion::V0 as u8);
    // Third byte is the app_id (Soma = 1).
    assert_eq!(bytes[2], AppId::Soma as u8);
}

#[test]
fn test_intent_from_bytes_roundtrip() {
    let original = Intent::soma_transaction();
    let bytes = original.to_bytes();
    let recovered = Intent::from_bytes(&bytes).expect("from_bytes should succeed");
    assert_eq!(original, recovered);

    // Also test with consensus_app.
    let consensus_intent = Intent::consensus_app(IntentScope::ConsensusBlock);
    let bytes2 = consensus_intent.to_bytes();
    let recovered2 = Intent::from_bytes(&bytes2).expect("from_bytes should succeed");
    assert_eq!(consensus_intent, recovered2);
}

#[test]
fn test_intent_soma_app() {
    let scope = IntentScope::TransactionEffects;
    let intent = Intent::soma_app(scope);

    assert_eq!(intent.scope, scope);
    assert_eq!(intent.version, IntentVersion::V0);
    assert_eq!(intent.app_id, AppId::Soma);
}

#[test]
fn test_intent_consensus_app() {
    let scope = IntentScope::ConsensusBlock;
    let intent = Intent::consensus_app(scope);

    assert_eq!(intent.scope, scope);
    assert_eq!(intent.version, IntentVersion::V0);
    assert_eq!(intent.app_id, AppId::Consensus);
}

#[test]
fn test_intent_scope_variants() {
    // All IntentScope variants must have distinct u8 values.
    let scopes = [
        IntentScope::TransactionData,
        IntentScope::ConsensusBlock,
        IntentScope::SenderSignedTransaction,
        IntentScope::TransactionEffects,
        IntentScope::DiscoveryPeers,
        IntentScope::CommitSummary,
        IntentScope::ValidatorSet,
        IntentScope::CheckpointSummary,
    ];

    let mut seen = std::collections::HashSet::new();
    for scope in &scopes {
        let val = *scope as u8;
        assert!(
            seen.insert(val),
            "IntentScope variant {:?} has duplicate u8 value {}",
            scope,
            val
        );
    }

    // Verify we tested all variants (update count if new variants are added).
    assert_eq!(scopes.len(), 8);
}

#[test]
fn test_intent_message_wrapping() {
    let intent = Intent::soma_transaction();
    let value = 42u64;

    let msg = IntentMessage::new(intent.clone(), value);

    assert_eq!(msg.intent, intent);
    assert_eq!(msg.value, value);
}

#[test]
fn test_intent_bcs_serialization() {
    let intent = Intent::soma_transaction();

    let bytes = bcs::to_bytes(&intent).expect("BCS serialize should succeed");
    let deserialized: Intent = bcs::from_bytes(&bytes).expect("BCS deserialize should succeed");

    assert_eq!(intent, deserialized);

    // BCS serialization of an Intent should be compact (3 bytes):
    // each of the three enum fields is a single u8.
    assert_eq!(bytes.len(), 3, "Intent BCS should be exactly 3 bytes");
}
