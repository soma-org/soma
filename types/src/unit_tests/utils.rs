use fastcrypto::traits::Signer;
use rand::{SeedableRng as _, rngs::StdRng};

use crate::{
    crypto::{Signature, SomaKeyPair, get_key_pair_from_rng},
    transaction::{Transaction, TransactionData},
};

// This is used to sign transaction with signer using default Intent.
pub fn to_sender_signed_transaction(
    data: TransactionData,
    signer: &dyn Signer<Signature>,
) -> Transaction {
    to_sender_signed_transaction_with_multi_signers(data, vec![signer])
}

pub fn to_sender_signed_transaction_with_multi_signers(
    data: TransactionData,
    signers: Vec<&dyn Signer<Signature>>,
) -> Transaction {
    Transaction::from_data_and_signer(data, signers)
}

pub fn keys() -> Vec<SomaKeyPair> {
    let mut seed = StdRng::from_seed([0; 32]);
    let kp1: SomaKeyPair = SomaKeyPair::Ed25519(get_key_pair_from_rng(&mut seed).1);
    let kp2: SomaKeyPair = SomaKeyPair::Ed25519(get_key_pair_from_rng(&mut seed).1);
    let kp3: SomaKeyPair = SomaKeyPair::Ed25519(get_key_pair_from_rng(&mut seed).1);
    vec![kp1, kp2, kp3]
}
