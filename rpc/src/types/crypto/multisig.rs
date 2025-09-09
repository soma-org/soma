#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MultisigMemberPublicKey {
    Ed25519(Ed25519PublicKey),
    Secp256k1(Secp256k1PublicKey),
    Secp256r1(Secp256r1PublicKey),
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "proptest", derive(test_strategy::Arbitrary))]
#[non_exhaustive]
pub enum MultisigMemberSignature {
    Ed25519(Ed25519Signature),
    Secp256k1(Secp256k1Signature),
    Secp256r1(Secp256r1Signature),
}
