use types::base::SomaAddress;
use types::crypto::{Signature, SignatureScheme, SomaKeyPair};
use types::intent::{Intent, IntentMessage};
use types::transaction::{Transaction, TransactionData};

use crate::error::Error;

/// A convenient wrapper around [`SomaKeyPair`] providing clean key generation,
/// import/export, and transaction signing.
///
/// Uses the `soma-keys` crate for key derivation rather than raw byte manipulation.
pub struct Keypair {
    inner: SomaKeyPair,
}

impl Keypair {
    /// Generate a random Ed25519 keypair.
    pub fn generate() -> Self {
        let (_addr, kp, _scheme, _phrase) =
            soma_keys::key_derive::generate_new_key(SignatureScheme::ED25519, None, None)
                .expect("Ed25519 key generation should not fail");
        Self { inner: kp }
    }

    /// Generate a random keypair, returning `(Keypair, mnemonic_phrase)`.
    pub fn generate_with_mnemonic() -> (Self, String) {
        let (_addr, kp, _scheme, phrase) =
            soma_keys::key_derive::generate_new_key(SignatureScheme::ED25519, None, None)
                .expect("Ed25519 key generation should not fail");
        (Self { inner: kp }, phrase)
    }

    /// Derive a keypair from a BIP39 mnemonic phrase.
    ///
    /// Uses the default SLIP-0010 derivation path `m/44'/784'/0'/0'/0'`.
    pub fn from_mnemonic(mnemonic: &str) -> Result<Self, Error> {
        use bip39::{Language, Mnemonic, Seed};

        let m = Mnemonic::from_phrase(mnemonic, Language::English)
            .map_err(|e| Error::KeyError(format!("Invalid mnemonic: {e:?}")))?;
        let seed = Seed::new(&m, "");

        let (_addr, kp) = soma_keys::key_derive::derive_key_pair_from_path(
            seed.as_bytes(),
            None, // default path m/44'/784'/0'/0'/0'
            &SignatureScheme::ED25519,
        )
        .map_err(|e| Error::KeyError(format!("Key derivation failed: {e:?}")))?;

        Ok(Self { inner: kp })
    }

    /// Create a keypair from a raw 32-byte Ed25519 secret key.
    pub fn from_secret_key(bytes: &[u8]) -> Result<Self, Error> {
        use fastcrypto::ed25519::Ed25519PrivateKey;
        use fastcrypto::traits::ToFromBytes;

        let sk = Ed25519PrivateKey::from_bytes(bytes)
            .map_err(|e| Error::KeyError(format!("Invalid secret key: {e}")))?;
        let kp: fastcrypto::ed25519::Ed25519KeyPair = sk.into();
        Ok(Self { inner: SomaKeyPair::Ed25519(kp) })
    }

    /// Create a keypair from a Bech32-encoded secret key (`somaprivkey1...`).
    pub fn from_encoded(encoded: &str) -> Result<Self, Error> {
        let kp = SomaKeyPair::decode(encoded)
            .map_err(|e| Error::KeyError(format!("Failed to decode key: {e}")))?;
        Ok(Self { inner: kp })
    }

    /// Return the Soma address derived from this keypair's public key.
    pub fn address(&self) -> SomaAddress {
        SomaAddress::from(&self.inner.public())
    }

    /// Export the 32-byte Ed25519 secret key (no flag prefix).
    pub fn to_secret_key(&self) -> Vec<u8> {
        self.inner.to_bytes_no_flag()
    }

    /// Export as a Bech32-encoded string (`somaprivkey1...`).
    pub fn to_encoded(&self) -> String {
        self.inner.encode().expect("Bech32 encoding should not fail")
    }

    /// Sign [`TransactionData`], returning a signed [`Transaction`] ready for execution.
    pub fn sign_transaction(&self, tx_data: TransactionData) -> Transaction {
        let intent_msg = IntentMessage::new(Intent::soma_transaction(), &tx_data);
        let sig = Signature::new_secure(&intent_msg, &self.inner);
        Transaction::from_data(tx_data, vec![sig])
    }

    /// Access the inner [`SomaKeyPair`].
    pub fn inner(&self) -> &SomaKeyPair {
        &self.inner
    }

    /// Copy the inner keypair.
    pub fn copy_inner(&self) -> SomaKeyPair {
        self.inner.copy()
    }

    /// Create a `Keypair` from a raw [`SomaKeyPair`].
    pub fn from_inner(kp: SomaKeyPair) -> Self {
        Self { inner: kp }
    }

    /// Copy this keypair (mirrors [`SomaKeyPair::copy`]).
    pub fn copy(&self) -> Self {
        Self { inner: self.inner.copy() }
    }
}
