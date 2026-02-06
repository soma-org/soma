use crate::types::Address;
use crate::types::Digest;

use blake2::Digest as DigestTrait;

type Blake2b256 = blake2::Blake2b<blake2::digest::consts::U32>;

/// A Blake2b256 Hasher
#[derive(Debug, Default)]
pub struct Hasher(Blake2b256);

impl Hasher {
    /// Initialize a new Blake2b256 Hasher instance.
    pub fn new() -> Self {
        Self(Blake2b256::new())
    }

    /// Process the provided data, updating internal state.
    pub fn update<T: AsRef<[u8]>>(&mut self, data: T) {
        self.0.update(data)
    }

    /// Finalize hashing, consuming the Hasher instance and returning the resultant hash or
    /// `Digest`.
    pub fn finalize(self) -> Digest {
        let mut buf = [0; Digest::LENGTH];
        let result = self.0.finalize();

        buf.copy_from_slice(result.as_ref());

        Digest::new(buf)
    }

    /// Convenience function for creating a new Hasher instance, hashing the provided data, and
    /// returning the resultant `Digest`
    pub fn digest<T: AsRef<[u8]>>(data: T) -> Digest {
        let mut hasher = Self::new();
        hasher.update(data);
        hasher.finalize()
    }
}

impl std::io::Write for Hasher {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

impl crate::types::Ed25519PublicKey {
    /// Derive an `Address` from this Public Key
    ///
    /// An `Address` can be derived from an `Ed25519PublicKey` by hashing the bytes of the public
    /// key prefixed with the Ed25519 `SignatureScheme` flag (`0x00`).
    ///
    /// `hash( 0x00 || 32-byte ed25519 public key)`
    ///
    /// ```
    /// use sui_sdk_types::Address;
    /// use sui_sdk_types::Ed25519PublicKey;
    /// use sui_sdk_types::hash::Hasher;
    ///
    /// let public_key_bytes = [0; 32];
    /// let mut hasher = Hasher::new();
    /// hasher.update([0x00]); // The SignatureScheme flag for Ed25519 is `0`
    /// hasher.update(public_key_bytes);
    /// let address = Address::new(hasher.finalize().into_inner());
    /// println!("Address: {}", address);
    ///
    /// let public_key = Ed25519PublicKey::new(public_key_bytes);
    /// assert_eq!(address, public_key.derive_address());
    /// ```
    pub fn derive_address(&self) -> Address {
        let mut hasher = Hasher::new();
        self.write_into_hasher(&mut hasher);
        let digest = hasher.finalize();
        Address::new(digest.into_inner())
    }

    fn write_into_hasher(&self, hasher: &mut Hasher) {
        hasher.update([self.scheme().to_u8()]);
        hasher.update(self.inner());
    }
}

impl crate::types::SimpleSignature {
    pub fn derive_address(&self) -> Address {
        match self {
            crate::types::SimpleSignature::Ed25519 { public_key, .. } => {
                public_key.derive_address()
            }
        }
    }
}

impl crate::types::UserSignature {
    pub fn derive_address(&self) -> Address {
        self.derive_addresses().next().unwrap()
    }

    pub fn derive_addresses(&self) -> impl ExactSizeIterator<Item = Address> {
        match self {
            crate::types::UserSignature::Simple(simple) => {
                DerivedAddressIter::new(simple.derive_address())
            }
            crate::types::UserSignature::Multisig(multisig) => {
                DerivedAddressIter::new(multisig.derive_address())
            }
        }
    }
}

impl crate::types::MultisigCommittee {
    /// Derive an `Address` from this MultisigCommittee.
    ///
    /// A MultiSig address
    /// is defined as the 32-byte Blake2b hash of serializing the `SignatureScheme` flag (0x03), the
    /// threshold (in little endian), and the concatenation of all n flag, public keys and
    /// its weight.
    ///
    /// `hash(0x03 || threshold || flag_1 || pk_1 || weight_1
    /// || ... || flag_n || pk_n || weight_n)`.
    ///
    /// When flag_i is ZkLogin, the pk_i for the [`ZkLoginPublicIdentifier`] refers to the same
    /// input used when deriving the address using the
    /// [`ZkLoginPublicIdentifier::derive_address_padded`] method (using the full 32-byte
    /// `address_seed` value).
    ///
    /// [`ZkLoginPublicIdentifier`]: crate::ZkLoginPublicIdentifier
    /// [`ZkLoginPublicIdentifier::derive_address_padded`]: crate::ZkLoginPublicIdentifier::derive_address_padded
    pub fn derive_address(&self) -> Address {
        use crate::types::MultisigMemberPublicKey::*;

        let mut hasher = Hasher::new();
        hasher.update([self.scheme().to_u8()]);
        hasher.update(self.threshold().to_le_bytes());

        for member in self.members() {
            match member.public_key() {
                Ed25519(p) => p.write_into_hasher(&mut hasher),
            }

            hasher.update(member.weight().to_le_bytes());
        }

        let digest = hasher.finalize();
        Address::new(digest.into_inner())
    }
}

impl crate::types::MultisigAggregatedSignature {
    pub fn derive_address(&self) -> Address {
        self.committee().derive_address()
    }
}

struct DerivedAddressIter {
    primary: Option<Address>,
    extra: Option<Address>,
}

impl DerivedAddressIter {
    fn new(primary: Address) -> Self {
        Self { primary: Some(primary), extra: None }
    }
}

impl ExactSizeIterator for DerivedAddressIter {}
impl Iterator for DerivedAddressIter {
    type Item = Address;

    fn next(&mut self) -> Option<Self::Item> {
        if self.primary.is_some() {
            self.primary.take()
        } else if self.extra.is_some() {
            self.extra.take()
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.primary.iter().len() + self.extra.iter().len();
        (len, Some(len))
    }
}

mod type_digest {
    use super::Hasher;
    use crate::types::Digest;

    impl crate::types::Object {
        /// Calculate the digest of this `Object`
        ///
        /// This is done by hashing the BCS bytes of this `Object` prefixed
        pub fn digest(&self) -> Digest {
            const SALT: &str = "Object::";
            type_digest(SALT, self)
        }
    }

    impl crate::types::CheckpointSummary {
        pub fn digest(&self) -> Digest {
            const SALT: &str = "CheckpointSummary::";
            type_digest(SALT, self)
        }
    }

    impl crate::types::CheckpointContents {
        pub fn digest(&self) -> Digest {
            const SALT: &str = "CheckpointContents::";
            type_digest(SALT, self)
        }
    }

    impl crate::types::Transaction {
        pub fn digest(&self) -> Digest {
            const SALT: &str = "TransactionData::";
            type_digest(SALT, self)
        }
    }

    impl crate::types::TransactionEffects {
        pub fn digest(&self) -> Digest {
            const SALT: &str = "TransactionEffects::";
            type_digest(SALT, self)
        }
    }

    fn type_digest<T: serde::Serialize>(salt: &str, ty: &T) -> Digest {
        let mut hasher = Hasher::new();
        hasher.update(salt);
        bcs::serialize_into(&mut hasher, ty).unwrap();
        hasher.finalize()
    }
}

mod signing_message {
    use crate::types::Digest;
    use crate::types::Intent;
    use crate::types::IntentAppId;
    use crate::types::IntentScope;
    use crate::types::IntentVersion;
    use crate::types::PersonalMessage;
    use crate::types::SigningDigest;
    use crate::types::Transaction;
    use crate::types::hash::Hasher;

    impl Transaction {
        pub fn signing_digest(&self) -> SigningDigest {
            const INTENT: Intent = Intent {
                scope: IntentScope::TransactionData,
                version: IntentVersion::V0,
                app_id: IntentAppId::Soma,
            };
            let digest = signing_digest(INTENT, self);
            digest.into_inner()
        }
    }

    fn signing_digest<T: serde::Serialize + ?Sized>(intent: Intent, ty: &T) -> Digest {
        let mut hasher = Hasher::new();
        hasher.update(intent.to_bytes());
        bcs::serialize_into(&mut hasher, ty).unwrap();
        hasher.finalize()
    }

    impl PersonalMessage<'_> {
        pub fn signing_digest(&self) -> SigningDigest {
            const INTENT: Intent = Intent {
                scope: IntentScope::PersonalMessage,
                version: IntentVersion::V0,
                app_id: IntentAppId::Soma,
            };
            let digest = signing_digest(INTENT, &self.0);
            digest.into_inner()
        }
    }

    impl crate::types::CheckpointSummary {
        pub fn signing_message(&self) -> Vec<u8> {
            const INTENT: Intent = Intent {
                scope: IntentScope::CheckpointSummary,
                version: IntentVersion::V0,
                app_id: IntentAppId::Soma,
            };
            let mut message = Vec::new();
            message.extend(INTENT.to_bytes());
            bcs::serialize_into(&mut message, self).unwrap();
            bcs::serialize_into(&mut message, &self.epoch).unwrap();
            message
        }
    }
}

/// A 1-byte domain separator for deriving `ObjectId`s in Sui. It is starting from `0xf0` to ensure
/// no hashing collision for any ObjectId vs Address which is derived as the hash of `flag ||
/// pubkey`.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
enum HashingIntent {
    RegularObjectId = 0xf1,
}

impl crate::types::Address {
    /// Create an ObjectId from `TransactionDigest` and `count`.
    ///
    /// `count` is the number of objects that have been created during a transactions.
    pub fn derive_id(digest: crate::types::Digest, count: u64) -> Self {
        let mut hasher = Hasher::new();
        hasher.update([HashingIntent::RegularObjectId as u8]);
        hasher.update(digest);
        hasher.update(count.to_le_bytes());
        let digest = hasher.finalize();
        Self::new(digest.into_inner())
    }
}
