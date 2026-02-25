// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! # Cryptography Module
//!
//! ## Overview
//! This module provides the cryptographic primitives and utilities used throughout the SOMA blockchain.
//! It implements signature schemes, key management, verification mechanisms, and cryptographic
//! operations essential for blockchain security and consensus.
//!
//! ## Responsibilities
//! - Define cryptographic types for authorities and users
//! - Implement signature creation and verification
//! - Provide key pair generation and management
//! - Support aggregated signatures for quorum operations
//! - Implement verification obligations for batch verification
//!
//! ## Component Relationships
//! - Used by Authority module for transaction validation and state certification
//! - Used by Consensus module for block signing and verification
//! - Used by P2P module for secure communication
//! - Provides cryptographic primitives to all modules requiring security operations
//!
//! ## Key Workflows
//! 1. Authority signature creation and verification for consensus
//! 2. Transaction signing by users and verification by validators
//! 3. Quorum certificate creation and verification
//! 4. Batch signature verification for performance optimization
//!
//! ## Design Patterns
//! - Trait-based abstraction for different signature schemes
//! - Enum dispatch for runtime polymorphism of signature types
//! - Verification obligation pattern for efficient batch verification
//! - Serialization/deserialization with safety checks

use crate::base::{AuthorityName, ConciseableName, SomaAddress};
use crate::committee::{Committee, CommitteeTrait, EpochId, VotingPower};
use crate::error::{SomaError, SomaResult};
use crate::intent::{Intent, IntentMessage, IntentScope};
use crate::multisig::MultiSig;
use crate::serde::Readable;
use crate::serde::SomaBitmap;
use anyhow::{Error, anyhow};
use core::hash::Hash;
use derive_more::{AsMut, AsRef, From};
use enum_dispatch::enum_dispatch;
use eyre::eyre;
use fastcrypto::bls12381::min_sig::{
    BLS12381AggregateSignature, BLS12381AggregateSignatureAsBytes, BLS12381KeyPair,
    BLS12381PrivateKey, BLS12381PublicKey, BLS12381Signature,
};
use fastcrypto::ed25519::{
    self, Ed25519KeyPair, Ed25519PublicKey, Ed25519PublicKeyAsBytes, Ed25519Signature,
    Ed25519SignatureAsBytes,
};
use fastcrypto::encoding::{Base64, Bech32};
use fastcrypto::encoding::{Encoding, Hex};
use fastcrypto::error::{FastCryptoError, FastCryptoResult};
use fastcrypto::hash::{Blake2b256, HashFunction};
pub use fastcrypto::traits::KeyPair as KeypairTraits;
pub use fastcrypto::traits::Signer;
pub use fastcrypto::traits::{
    AggregateAuthenticator, Authenticator, EncodeDecodeBase64, SigningKey,
};
use fastcrypto::traits::{ToFromBytes, VerifyingKey};
use rand::rngs::OsRng;
use rand::{SeedableRng, rngs::StdRng};
use roaring::RoaringBitmap;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::{Bytes, serde_as};
use std::collections::BTreeMap;
use std::fmt::{Debug, Display, Formatter};
use std::str::FromStr;
use strum::EnumString;
use tracing::{instrument, warn};

/// Default hash function used throughout the SOMA blockchain
///
/// Blake2b256 is chosen for its security properties and performance characteristics.
/// It provides a good balance between security and efficiency for blockchain operations.
pub type DefaultHash = Blake2b256;

/// Length of hash digests produced by the default hash function
pub const DIGEST_LENGTH: usize = DefaultHash::OUTPUT_SIZE;

/// Bech32 prefix for SOMA private keys
///
/// Used when encoding private keys to string format for storage or transmission
pub const SOMA_PRIV_KEY_PREFIX: &str = "somaprivkey";

// Authority cryptographic types
/// Key pair used by authorities (validators) for signing consensus messages
///
/// Uses BLS12-381 signature scheme which supports signature aggregation,
/// essential for efficient quorum certificate creation
pub type AuthorityKeyPair = BLS12381KeyPair;

/// Public key type for authorities
///
/// Used to verify signatures produced by authorities and identify validators
pub type AuthorityPublicKey = BLS12381PublicKey;

/// Private key type for authorities
///
/// Used by validators to sign consensus messages and transactions
pub type AuthorityPrivateKey = BLS12381PrivateKey;

/// Signature type produced by authorities
///
/// Used to authenticate messages from specific validators
pub type AuthoritySignature = BLS12381Signature;

/// Aggregated signature type for multiple authority signatures
///
/// Enables efficient verification of signatures from multiple validators,
/// critical for performance in quorum-based consensus
pub type AggregateAuthoritySignature = BLS12381AggregateSignature;

/// Byte representation of aggregated authority signatures
///
/// Used for serialization and transmission of aggregated signatures
pub type AggregateAuthoritySignatureAsBytes = BLS12381AggregateSignatureAsBytes;

/// Compressed representation of an authority public key
///
/// ## Purpose
/// Provides a space-efficient representation of authority public keys for storage
/// and network transmission. This type is used throughout the system when referring
/// to authorities by their public key.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
///
/// ## Examples
/// ```ignore
/// // Convert from a full public key to bytes representation
/// let public_key_bytes = AuthorityPublicKeyBytes::from(&authority_public_key);
///
/// // Convert back to a full public key when needed for verification
/// let public_key = AuthorityPublicKey::try_from(public_key_bytes)?;
/// ```
#[serde_as]
#[derive(
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    AsRef
)]
#[as_ref(forward)]
pub struct AuthorityPublicKeyBytes(
    #[schemars(with = "Base64")]
    #[serde_as(as = "Readable<Base64, Bytes>")]
    pub [u8; AuthorityPublicKey::LENGTH],
);

impl AuthorityPublicKeyBytes {
    /// Internal formatting implementation used by both Debug and Display traits
    ///
    /// Formats the public key bytes with a 'k#' prefix followed by the hex-encoded bytes
    fn fmt_impl(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        let s = Hex::encode(self.0);
        write!(f, "k#{}", s)?;
        Ok(())
    }
}

impl TryFrom<AuthorityPublicKeyBytes> for AuthorityPublicKey {
    type Error = FastCryptoError;

    /// Converts compressed public key bytes to a full AuthorityPublicKey
    ///
    /// This conversion is necessary when the full public key is needed for
    /// cryptographic operations like signature verification.
    fn try_from(bytes: AuthorityPublicKeyBytes) -> Result<AuthorityPublicKey, Self::Error> {
        AuthorityPublicKey::from_bytes(bytes.as_ref())
    }
}

impl From<&AuthorityPublicKey> for AuthorityPublicKeyBytes {
    /// Converts a full AuthorityPublicKey to its compressed bytes representation
    ///
    /// This conversion is used when storing or transmitting public keys in a
    /// space-efficient format.
    fn from(pk: &AuthorityPublicKey) -> AuthorityPublicKeyBytes {
        // This unwrap is safe because we're converting from a valid public key
        AuthorityPublicKeyBytes::from_bytes(pk.as_ref()).unwrap()
    }
}

impl Debug for AuthorityPublicKeyBytes {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.fmt_impl(f)
    }
}

impl Display for AuthorityPublicKeyBytes {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.fmt_impl(f)
    }
}

impl ToFromBytes for AuthorityPublicKeyBytes {
    fn from_bytes(bytes: &[u8]) -> Result<Self, fastcrypto::error::FastCryptoError> {
        let bytes: [u8; AuthorityPublicKey::LENGTH] =
            bytes.try_into().map_err(|_| fastcrypto::error::FastCryptoError::InvalidInput)?;
        Ok(AuthorityPublicKeyBytes(bytes))
    }
}

impl AuthorityPublicKeyBytes {
    /// Constant representing a zero-initialized public key
    ///
    /// Used as a default value and in testing
    pub const ZERO: Self = Self::new([0u8; AuthorityPublicKey::LENGTH]);

    /// Creates a new AuthorityPublicKeyBytes from raw bytes
    ///
    /// This constructor ensures type safety by requiring the exact byte length
    /// expected for an authority public key.
    ///
    /// ## Arguments
    /// * `bytes` - Raw byte array of exactly AuthorityPublicKey::LENGTH size
    pub const fn new(bytes: [u8; AuthorityPublicKey::LENGTH]) -> AuthorityPublicKeyBytes
where {
        AuthorityPublicKeyBytes(bytes)
    }
}

impl FromStr for AuthorityPublicKeyBytes {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value = Hex::decode(s).map_err(|e| anyhow!(e))?;
        Self::from_bytes(&value[..]).map_err(|e| anyhow!(e))
    }
}

impl Default for AuthorityPublicKeyBytes {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<'a> ConciseableName<'a> for AuthorityPublicKeyBytes {
    type ConciseTypeRef = ConciseAuthorityPublicKeyBytesRef<'a>;
    type ConciseType = ConciseAuthorityPublicKeyBytes;

    /// Get a ConciseAuthorityPublicKeyBytesRef. Usage:
    ///
    ///   debug!(name = ?authority.concise());
    ///   format!("{:?}", authority.concise());
    fn concise(&'a self) -> ConciseAuthorityPublicKeyBytesRef<'a> {
        ConciseAuthorityPublicKeyBytesRef(self)
    }

    fn concise_owned(&self) -> ConciseAuthorityPublicKeyBytes {
        ConciseAuthorityPublicKeyBytes(*self)
    }
}

/// A wrapper around AuthorityPublicKeyBytes that provides a concise Debug impl.
pub struct ConciseAuthorityPublicKeyBytesRef<'a>(&'a AuthorityPublicKeyBytes);

impl Debug for ConciseAuthorityPublicKeyBytesRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        let s = Hex::encode(self.0.0.get(0..4).ok_or(std::fmt::Error)?);
        write!(f, "k#{}..", s)
    }
}

impl Display for ConciseAuthorityPublicKeyBytesRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        Debug::fmt(self, f)
    }
}

/// A wrapper around AuthorityPublicKeyBytes but owns it.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ConciseAuthorityPublicKeyBytes(AuthorityPublicKeyBytes);

impl Debug for ConciseAuthorityPublicKeyBytes {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        let s = Hex::encode(self.0.0.get(0..4).ok_or(std::fmt::Error)?);
        write!(f, "k#{}..", s)
    }
}

impl Display for ConciseAuthorityPublicKeyBytes {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        Debug::fmt(self, f)
    }
}

/// Empty signature information placeholder
///
/// Used in contexts where signature information might be optional
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EmptySignInfo {}

/// Signature information from a single authority
///
/// ## Purpose
/// Contains all information needed to verify a signature from a specific authority,
/// including the authority's identity, the epoch in which the signature was created,
/// and the signature itself.
///
/// ## Lifecycle
/// Created when an authority signs a message, and used during verification to
/// authenticate the message and its source.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
#[derive(Clone, Debug, Eq, Serialize, Deserialize)]
pub struct AuthoritySignInfo {
    /// Epoch in which the signature was created
    pub epoch: EpochId,

    /// Identity of the authority that created the signature
    pub authority: AuthorityName,

    /// The cryptographic signature
    pub signature: AuthoritySignature,
}

impl AuthoritySignInfo {
    /// Creates a new AuthoritySignInfo by signing the provided value
    ///
    /// ## Behavior
    /// Constructs an intent message from the provided value and intent,
    /// signs it using the provided signer, and creates a new AuthoritySignInfo
    /// with the resulting signature and authority information.
    ///
    /// ## Arguments
    /// * `epoch` - Current epoch ID
    /// * `value` - The value to sign
    /// * `intent` - The intent context for the signature
    /// * `name` - The authority's name (public key)
    /// * `secret` - The authority's signing key
    ///
    /// ## Returns
    /// A new AuthoritySignInfo containing the signature and metadata
    pub fn new<T>(
        epoch: EpochId,
        value: &T,
        intent: Intent,
        name: AuthorityName,
        secret: &dyn Signer<AuthoritySignature>,
    ) -> Self
    where
        T: Serialize,
    {
        Self {
            epoch,
            authority: name,
            signature: AuthoritySignature::new_secure(
                &IntentMessage::new(intent, value),
                &epoch,
                secret,
            ),
        }
    }
}

mod private {
    pub trait SealedAuthoritySignInfoTrait {}
    impl SealedAuthoritySignInfoTrait for super::EmptySignInfo {}
    impl SealedAuthoritySignInfoTrait for super::AuthoritySignInfo {}
    impl<const S: bool> SealedAuthoritySignInfoTrait for super::AuthorityQuorumSignInfo<S> {}
}

/// AuthoritySignInfoTrait is a trait used specifically for a few structs in messages.rs
/// to template on whether the struct is signed by an authority. We want to limit how
/// those structs can be instantiated on, hence the sealed trait.
/// TODO: We could also add the aggregated signature as another impl of the trait.
///       This will make CertifiedTransaction also an instance of the same struct.
pub trait AuthoritySignInfoTrait: private::SealedAuthoritySignInfoTrait {
    fn verify_secure<T: Serialize>(
        &self,
        data: &T,
        intent: Intent,
        committee: &Committee,
    ) -> SomaResult;

    fn add_to_verification_obligation<'a>(
        &self,
        committee: &'a Committee,
        obligation: &mut VerificationObligation<'a>,
        message_index: usize,
    ) -> SomaResult<()>;
}

impl AuthoritySignInfoTrait for AuthoritySignInfo {
    fn verify_secure<T: Serialize>(
        &self,
        data: &T,
        intent: Intent,
        committee: &Committee,
    ) -> SomaResult<()> {
        let mut obligation = VerificationObligation::default();
        let idx = obligation.add_message(data, self.epoch, intent);
        self.add_to_verification_obligation(committee, &mut obligation, idx)?;
        obligation.verify_all()?;
        Ok(())
    }

    fn add_to_verification_obligation<'a>(
        &self,
        committee: &'a Committee,
        obligation: &mut VerificationObligation<'a>,
        message_index: usize,
    ) -> SomaResult<()> {
        if self.epoch != committee.epoch() {
            return Err(SomaError::WrongEpoch {
                expected_epoch: committee.epoch(),
                actual_epoch: self.epoch,
            });
        }
        let weight = committee.weight(&self.authority);
        if weight == 0 {
            return Err(SomaError::UnknownSigner {
                signer: Some(self.authority.concise().to_string()),
                index: None,
                committee: Box::new(committee.clone()),
            });
        }

        obligation
            .public_keys
            .get_mut(message_index)
            .ok_or(SomaError::InvalidAddress)?
            .push(committee.public_key(&self.authority)?);
        obligation
            .signatures
            .get_mut(message_index)
            .ok_or(SomaError::InvalidAddress)?
            .add_signature(self.signature.clone())
            .map_err(|_| SomaError::InvalidSignature {
                error: "Fail to aggregator auth sig".to_string(),
            })?;
        Ok(())
    }
}

impl Display for AuthoritySignInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "AuthoritySignInfo {{ epoch: {:?}, authority: {} }}", self.epoch, self.authority,)
    }
}

impl PartialEq for AuthoritySignInfo {
    fn eq(&self, other: &Self) -> bool {
        // We do not compare the signature, because there can be multiple
        // valid signatures for the same epoch and authority.
        self.epoch == other.epoch && self.authority == other.authority
    }
}

/// Represents at least a quorum (could be more) of authority signatures
///
/// ## Purpose
/// Provides an efficient representation of multiple authority signatures that
/// together form a quorum certificate. Instead of storing individual signatures,
/// this structure stores an aggregated signature and a bitmap indicating which
/// authorities contributed to the signature.
///
/// ## Threshold Behavior
/// The STRONG_THRESHOLD generic parameter determines the quorum threshold:
/// - When STRONG_THRESHOLD is true: requires 2f+1 stake (strong quorum)
/// - When STRONG_THRESHOLD is false: requires f+1 stake (weak quorum)
///
/// Where f is the maximum number of Byzantine (faulty) nodes the system can tolerate.
///
/// ## Thread Safety
/// This type is immutable and can be safely shared across threads.
///
/// ## Examples
/// ```ignore
/// // Create a strong quorum signature (2f+1)
/// let strong_quorum = AuthorityQuorumSignInfo::<true>::new_from_auth_sign_infos(
///     signatures,
///     committee
/// )?;
///
/// // Verify the quorum signature
/// strong_quorum.verify_secure(&data, Intent::SomaTransaction, committee)?;
/// ```
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AuthorityQuorumSignInfo<const STRONG_THRESHOLD: bool> {
    pub epoch: EpochId,
    #[schemars(with = "Base64")]
    pub signature: AggregateAuthoritySignature,
    #[schemars(with = "Base64")]
    #[serde_as(as = "SomaBitmap")]
    pub signers_map: RoaringBitmap,
}

pub type AuthorityStrongQuorumSignInfo = AuthorityQuorumSignInfo<true>;

impl<const STRONG_THRESHOLD: bool> AuthoritySignInfoTrait
    for AuthorityQuorumSignInfo<STRONG_THRESHOLD>
{
    fn verify_secure<T: Serialize>(
        &self,
        data: &T,
        intent: Intent,
        committee: &Committee,
    ) -> SomaResult {
        let mut obligation = VerificationObligation::default();
        let idx = obligation.add_message(data, self.epoch, intent);
        self.add_to_verification_obligation(committee, &mut obligation, idx)?;
        obligation.verify_all()?;
        Ok(())
    }

    fn add_to_verification_obligation<'a>(
        &self,
        committee: &'a Committee,
        obligation: &mut VerificationObligation<'a>,
        message_index: usize,
    ) -> SomaResult<()> {
        if self.epoch != committee.epoch() {
            return Err(SomaError::WrongEpoch {
                expected_epoch: committee.epoch(),
                actual_epoch: self.epoch,
            });
        }

        let mut weight = 0;

        // Create obligations for the committee signatures
        obligation
            .signatures
            .get_mut(message_index)
            .ok_or(SomaError::InvalidAuthenticator)?
            .add_aggregate(self.signature.clone())
            .map_err(|_| SomaError::InvalidSignature {
                error: "Signature Aggregation failed".to_string(),
            })?;

        let selected_public_keys =
            obligation.public_keys.get_mut(message_index).ok_or(SomaError::InvalidAuthenticator)?;

        for authority_index in self.signers_map.iter() {
            let authority = committee.authority_by_index(authority_index).ok_or_else(|| {
                SomaError::UnknownSigner {
                    signer: None,
                    index: Some(authority_index),
                    committee: Box::new(committee.clone()),
                }
            })?;

            // Update weight.
            let voting_rights = committee.weight(authority);

            if voting_rights == 0 {
                return Err(SomaError::UnknownSigner {
                    signer: Some(authority.concise().to_string()),
                    index: Some(authority_index),
                    committee: Box::new(committee.clone()),
                });
            }

            weight += voting_rights;

            selected_public_keys.push(committee.public_key(authority)?);
        }

        if weight < Self::quorum_threshold(committee) {
            return Err(SomaError::CertificateRequiresQuorum);
        }

        Ok(())
    }
}

impl<const STRONG_THRESHOLD: bool> AuthorityQuorumSignInfo<STRONG_THRESHOLD> {
    /// Creates a new AuthorityQuorumSignInfo from individual authority signatures
    ///
    /// ## Behavior
    /// Validates that all signatures are from the same epoch, checks that they have
    /// sufficient stake to meet the quorum threshold, aggregates the signatures,
    /// and creates a bitmap of signers.
    ///
    /// ## Arguments
    /// * `auth_sign_infos` - Vector of individual authority signatures
    /// * `committee` - The committee configuration for the current epoch
    ///
    /// ## Returns
    /// A new AuthorityQuorumSignInfo if the signatures form a valid quorum
    ///
    /// ## Errors
    /// - Returns error if signatures are from different epochs
    /// - Returns error if total stake is below the quorum threshold
    /// - Returns error if any signer is not in the committee
    /// - Returns error if signature aggregation fails
    pub fn new_from_auth_sign_infos(
        auth_sign_infos: Vec<AuthoritySignInfo>,
        committee: &Committee,
    ) -> SomaResult<Self> {
        // Verify all signatures are from the same epoch as the committee
        if !(auth_sign_infos.iter().all(|a| a.epoch == committee.epoch)) {
            return Err(SomaError::InvalidSignature {
                error: "All signatures must be from the same epoch as the committee".to_string(),
            });
        }

        // Calculate total stake and verify it meets the quorum threshold
        let total_stake: VotingPower =
            auth_sign_infos.iter().map(|a| committee.weight(&a.authority)).sum();
        if total_stake < Self::quorum_threshold(committee) {
            return Err(SomaError::InvalidSignature {
                error: "Signatures don't have enough stake to form a quorum".to_string(),
            });
        }

        // Organize signatures by authority
        let signatures: BTreeMap<_, _> =
            auth_sign_infos.into_iter().map(|a| (a.authority, a.signature)).collect();

        // Create bitmap of signers using their committee indices
        let mut map = RoaringBitmap::new();
        for pk in signatures.keys() {
            map.insert(committee.authority_index(pk).ok_or_else(|| SomaError::UnknownSigner {
                signer: Some(pk.concise().to_string()),
                index: None,
                committee: Box::new(committee.clone()),
            })?);
        }

        // Extract signatures for aggregation
        let sigs: Vec<AuthoritySignature> = signatures.into_values().collect();

        // Create the quorum signature info with aggregated signature
        Ok(AuthorityQuorumSignInfo {
            epoch: committee.epoch,
            signature: AggregateAuthoritySignature::aggregate(&sigs)
                .map_err(|e| SomaError::InvalidSignature { error: e.to_string() })?,
            signers_map: map,
        })
    }

    /// Gets the appropriate quorum threshold based on STRONG_THRESHOLD
    ///
    /// ## Arguments
    /// * `committee` - The committee configuration
    ///
    /// ## Returns
    /// The voting power threshold required for a quorum
    pub fn quorum_threshold(committee: &Committee) -> VotingPower {
        committee.threshold::<STRONG_THRESHOLD>()
    }
}
/// Trait for authority signatures in the SOMA blockchain
///
/// ## Purpose
/// Defines the interface for creating and verifying authority signatures
/// with epoch and intent context. This trait abstracts the specific signature
/// scheme implementation details.
///
/// ## Usage
/// Implemented by signature types used for authority operations, providing
/// a consistent interface for secure signature creation and verification.
pub trait SomaAuthoritySignature {
    /// Verifies a signature against a message, epoch, and author
    ///
    /// ## Arguments
    /// * `value` - The intent message that was signed
    /// * `epoch_id` - The epoch in which the signature was created
    /// * `author` - The public key of the purported signer
    ///
    /// ## Returns
    /// Ok(()) if the signature is valid, or an error if verification fails
    fn verify_secure<T>(
        &self,
        value: &IntentMessage<T>,
        epoch_id: EpochId,
        author: AuthorityPublicKeyBytes,
    ) -> Result<(), SomaError>
    where
        T: Serialize;

    /// Creates a new signature for a message with epoch context
    ///
    /// ## Arguments
    /// * `value` - The intent message to sign
    /// * `epoch_id` - The current epoch ID
    /// * `secret` - The signing key
    ///
    /// ## Returns
    /// A new signature over the message and epoch
    fn new_secure<T>(
        value: &IntentMessage<T>,
        epoch_id: &EpochId,
        secret: &dyn Signer<Self>,
    ) -> Self
    where
        T: Serialize;
}

impl SomaAuthoritySignature for AuthoritySignature {
    /// Creates a new authority signature with epoch context
    ///
    /// ## Behavior
    /// Serializes the intent message, appends the epoch information,
    /// and signs the combined data.
    ///
    /// ## Arguments
    /// * `value` - The intent message to sign
    /// * `epoch` - The current epoch ID
    /// * `secret` - The authority's signing key
    ///
    /// ## Returns
    /// A new authority signature over the message and epoch
    #[instrument(level = "trace", skip_all)]
    fn new_secure<T>(value: &IntentMessage<T>, epoch: &EpochId, secret: &dyn Signer<Self>) -> Self
    where
        T: Serialize,
    {
        // Serialize the intent message
        let mut intent_msg_bytes =
            bcs::to_bytes(&value).expect("Message serialization should not fail");

        // Append epoch information to the serialized message
        epoch.write(&mut intent_msg_bytes);

        // Sign the combined data
        secret.sign(&intent_msg_bytes)
    }

    /// Verifies an authority signature against a message, epoch, and author
    ///
    /// ## Behavior
    /// Serializes the intent message, appends the epoch information,
    /// converts the author's public key bytes to a full public key,
    /// and verifies the signature against the combined data.
    ///
    /// ## Arguments
    /// * `value` - The intent message that was signed
    /// * `epoch` - The epoch in which the signature was created
    /// * `author` - The public key bytes of the purported signer
    ///
    /// ## Returns
    /// Ok(()) if the signature is valid, or an error if verification fails
    ///
    /// ## Errors
    /// - Returns error if public key conversion fails
    /// - Returns error if signature verification fails
    #[instrument(level = "trace", skip_all)]
    fn verify_secure<T>(
        &self,
        value: &IntentMessage<T>,
        epoch: EpochId,
        author: AuthorityPublicKeyBytes,
    ) -> Result<(), SomaError>
    where
        T: Serialize,
    {
        // Serialize the intent message
        let mut message = bcs::to_bytes(&value).expect("Message serialization should not fail");

        // Append epoch information to the serialized message
        epoch.write(&mut message);

        // Convert public key bytes to full public key
        let public_key = AuthorityPublicKey::try_from(author).map_err(|_| {
            SomaError::KeyConversionError(
                "Failed to serialize public key bytes to valid public key".to_string(),
            )
        })?;

        // Verify the signature
        public_key.verify(&message[..], self).map_err(|e| SomaError::InvalidSignature {
            error: format!(
                "Fail to verify auth sig {} epoch: {} author: {}",
                e,
                epoch,
                author.concise()
            ),
        })
    }
}

/// Default epoch ID used for proof of possession signatures.
/// PoP is epoch-independent, so we use epoch 0.
pub const DEFAULT_EPOCH_ID: EpochId = 0;

/// Creates a proof of possession: a BLS12-381 authority signature over
/// `protocol_pubkey_bytes || soma_address`, wrapped in an IntentMessage
/// with IntentScope::ProofOfPossession. This proves the holder of the
/// authority protocol key also controls the validator's account address.
pub fn generate_proof_of_possession(
    keypair: &AuthorityKeyPair,
    address: SomaAddress,
) -> AuthoritySignature {
    let mut msg: Vec<u8> = Vec::new();
    msg.extend_from_slice(keypair.public().as_bytes());
    msg.extend_from_slice(address.as_ref());
    AuthoritySignature::new_secure(
        &IntentMessage::new(Intent::soma_app(IntentScope::ProofOfPossession), msg),
        &DEFAULT_EPOCH_ID,
        keypair,
    )
}

/// Verifies a proof of possession against the expected protocol public key
/// and validator account address.
pub fn verify_proof_of_possession(
    pop: &AuthoritySignature,
    protocol_pubkey: &AuthorityPublicKey,
    soma_address: SomaAddress,
) -> Result<(), SomaError> {
    let mut msg = protocol_pubkey.as_bytes().to_vec();
    msg.extend_from_slice(soma_address.as_ref());
    pop.verify_secure(
        &IntentMessage::new(Intent::soma_app(IntentScope::ProofOfPossession), msg),
        DEFAULT_EPOCH_ID,
        protocol_pubkey.into(),
    )
}

pub fn get_authority_key_pair() -> (SomaAddress, AuthorityKeyPair) {
    get_key_pair()
}

pub fn get_key_pair_from_rng<KP: KeypairTraits, R>(csprng: &mut R) -> (SomaAddress, KP)
where
    R: rand::CryptoRng + rand::RngCore,
    <KP as KeypairTraits>::PubKey: SomaPublicKey,
{
    let kp = KP::generate(&mut StdRng::from_rng(csprng).unwrap());
    (kp.public().into(), kp)
}

pub fn get_key_pair<KP: KeypairTraits>() -> (SomaAddress, KP)
where
    <KP as KeypairTraits>::PubKey: SomaPublicKey,
{
    get_key_pair_from_rng(&mut OsRng)
}

pub fn random_committee_key_pairs_of_size(size: usize) -> Vec<AuthorityKeyPair> {
    let mut rng = StdRng::from_seed([0; 32]);
    (0..size)
        .map(|_| {
            // TODO: We are generating the keys 4 times to match exactly as how we generate
            // keys in ConfigBuilder::build (soma-config/src/network_config_builder). This is because
            // we are using these key generation functions as fixtures and we call them
            // independently in different paths and exact the results to be the same.
            // We should eliminate them.
            let key_pair = get_key_pair_from_rng::<AuthorityKeyPair, _>(&mut rng);
            get_key_pair_from_rng::<AuthorityKeyPair, _>(&mut rng);
            get_key_pair_from_rng::<Ed25519KeyPair, _>(&mut rng);
            get_key_pair_from_rng::<Ed25519KeyPair, _>(&mut rng);
            key_pair.1
        })
        .collect()
}

#[enum_dispatch(AuthenticatorTrait)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericSignature {
    MultiSig,
    Signature,
}

impl GenericSignature {
    pub fn verify_authenticator<T>(
        &self,
        value: &IntentMessage<T>,
        author: SomaAddress,
    ) -> SomaResult
    where
        T: Serialize,
    {
        self.verify_claims(value, author)
    }

    /// Parse [enum CompressedSignature] from trait SomaSignature `flag || sig || pk`.
    /// This is useful for the MultiSig to combine partial signature into a MultiSig public key.
    pub fn to_compressed(&self) -> Result<CompressedSignature, SomaError> {
        match self {
            GenericSignature::Signature(s) => {
                let bytes = s.signature_bytes();
                match s.scheme() {
                    SignatureScheme::ED25519 => Ok(CompressedSignature::Ed25519(
                        (&Ed25519Signature::from_bytes(bytes).map_err(|_| {
                            SomaError::InvalidSignature {
                                error: "Cannot parse ed25519 sig".to_string(),
                            }
                        })?)
                            .into(),
                    )),

                    _ => Err(SomaError::UnsupportedFeatureError {
                        error: "Unsupported signature scheme".to_string(),
                    }),
                }
            }

            _ => Err(SomaError::UnsupportedFeatureError {
                error: "Unsupported signature scheme".to_string(),
            }),
        }
    }

    /// Parse [struct PublicKey] from trait SomaSignature `flag || sig || pk`.
    /// This is useful for the MultiSig to construct the bitmap in [struct MultiPublicKey].
    pub fn to_public_key(&self) -> Result<PublicKey, SomaError> {
        match self {
            GenericSignature::Signature(s) => {
                let bytes = s.public_key_bytes();
                match s.scheme() {
                    SignatureScheme::ED25519 => Ok(PublicKey::Ed25519(
                        (&Ed25519PublicKey::from_bytes(bytes).map_err(|_| {
                            SomaError::KeyConversionError("Cannot parse ed25519 pk".to_string())
                        })?)
                            .into(),
                    )),

                    _ => Err(SomaError::UnsupportedFeatureError {
                        error: "Unsupported signature scheme in MultiSig".to_string(),
                    }),
                }
            }

            _ => Err(SomaError::UnsupportedFeatureError {
                error: "Unsupported signature scheme".to_string(),
            }),
        }
    }
}

impl ToFromBytes for GenericSignature {
    fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        match SignatureScheme::from_flag_byte(
            bytes.first().ok_or(FastCryptoError::InputTooShort(0))?,
        ) {
            Ok(x) => match x {
                SignatureScheme::ED25519 => Ok(GenericSignature::Signature(
                    Signature::from_bytes(bytes).map_err(|_| FastCryptoError::InvalidSignature)?,
                )),
                SignatureScheme::MultiSig => match MultiSig::from_bytes(bytes) {
                    Ok(multisig) => Ok(GenericSignature::MultiSig(multisig)),
                    _ => Err(FastCryptoError::InvalidInput),
                },
                _ => Err(FastCryptoError::InvalidInput),
            },
            Err(_) => Err(FastCryptoError::InvalidInput),
        }
    }
}

/// Trait useful to get the bytes reference for [enum GenericSignature].
impl AsRef<[u8]> for GenericSignature {
    fn as_ref(&self) -> &[u8] {
        match self {
            GenericSignature::Signature(s) => s.as_ref(),
            GenericSignature::MultiSig(s) => s.as_ref(),
        }
    }
}

impl ::serde::Serialize for GenericSignature {
    fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            #[derive(serde::Serialize)]
            struct GenericSignature(String);
            GenericSignature(self.encode_base64()).serialize(serializer)
        } else {
            #[derive(serde::Serialize)]
            struct GenericSignature<'a>(&'a [u8]);
            GenericSignature(self.as_ref()).serialize(serializer)
        }
    }
}

impl<'de> ::serde::Deserialize<'de> for GenericSignature {
    fn deserialize<D: ::serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        if deserializer.is_human_readable() {
            #[derive(serde::Deserialize)]
            struct GenericSignature(String);
            let s = GenericSignature::deserialize(deserializer)?;
            Self::decode_base64(&s.0).map_err(::serde::de::Error::custom)
        } else {
            #[derive(serde::Deserialize)]
            struct GenericSignature(Vec<u8>);

            let data = GenericSignature::deserialize(deserializer)?;
            Self::from_bytes(&data.0).map_err(|e| Error::custom(e.to_string()))
        }
    }
}

#[enum_dispatch]
pub trait AuthenticatorTrait {
    fn verify_claims<T>(&self, value: &IntentMessage<T>, author: SomaAddress) -> SomaResult
    where
        T: Serialize;
}

impl AuthenticatorTrait for Signature {
    fn verify_claims<T>(&self, value: &IntentMessage<T>, author: SomaAddress) -> SomaResult
    where
        T: Serialize,
    {
        self.verify_secure(value, author, self.scheme())
    }
}

/// Something that we know how to hash and sign.
pub trait Signable<W> {
    fn write(&self, writer: &mut W);
}

fn hash<S: Signable<H>, H: HashFunction<DIGEST_SIZE>, const DIGEST_SIZE: usize>(
    signable: &S,
) -> [u8; DIGEST_SIZE] {
    let mut digest = H::default();
    signable.write(&mut digest);
    let hash = digest.finalize();
    hash.into()
}

pub fn default_hash<S: Signable<DefaultHash>>(signable: &S) -> [u8; 32] {
    hash::<S, DefaultHash, 32>(signable)
}

/// Activate the blanket implementation of `Signable` based on serde and BCS.
/// * We use `serde_name` to extract a seed from the name of structs and enums.
/// * We use `BCS` to generate canonical bytes suitable for hashing and signing.
///
/// # Safety
/// We protect the access to this marker trait through a "sealed trait" pattern:
/// impls must be add added here (nowehre else) which lets us note those impls
/// MUST be on types that comply with the `serde_name` machinery
/// for the below implementations not to panic. One way to check they work is to write
/// a unit test for serialization to / deserialization from signable bytes.
mod bcs_signable {

    pub trait BcsSignable: serde::Serialize + serde::de::DeserializeOwned {}
    impl BcsSignable for crate::transaction::TransactionData {}
    impl BcsSignable for crate::transaction::SenderSignedData {}
    impl BcsSignable for crate::committee::Committee {}
    impl BcsSignable for crate::effects::TransactionEffects {}
    impl BcsSignable for crate::object::ObjectInner {}
    impl BcsSignable for crate::checkpoints::CheckpointSummary {}
    impl BcsSignable for crate::checkpoints::CheckpointContents {}
}

impl<T, W> Signable<W> for T
where
    T: bcs_signable::BcsSignable,
    W: std::io::Write,
{
    fn write(&self, writer: &mut W) {
        let name = serde_name::trace_name::<Self>().expect("Self must be a struct or an enum");
        // Note: This assumes that names never contain the separator `::`.
        write!(writer, "{}::", name).expect("Hasher should not fail");
        bcs::serialize_into(writer, &self).expect("Message serialization should not fail");
    }
}

impl<W> Signable<W> for EpochId
where
    W: std::io::Write,
{
    fn write(&self, writer: &mut W) {
        bcs::serialize_into(writer, &self).expect("Message serialization should not fail");
    }
}

#[derive(
    Clone,
    Copy,
    Deserialize,
    Serialize,
    JsonSchema,
    Debug,
    EnumString,
    strum_macros::Display,
    PartialEq,
    Eq
)]
#[strum(serialize_all = "lowercase")]
pub enum SignatureScheme {
    ED25519,
    BLS12381,
    MultiSig,
}

impl SignatureScheme {
    pub fn flag(&self) -> u8 {
        match self {
            SignatureScheme::ED25519 => 0x00,
            SignatureScheme::BLS12381 => 0x01,
            SignatureScheme::MultiSig => 0x02,
        }
    }

    pub fn from_flag_byte(byte_int: &u8) -> Result<SignatureScheme, SomaError> {
        match byte_int {
            0x00 => Ok(SignatureScheme::ED25519),
            0x01 => Ok(SignatureScheme::BLS12381),
            0x02 => Ok(SignatureScheme::MultiSig),
            _ => Err(SomaError::KeyConversionError("Invalid key scheme".to_string())),
        }
    }
}

/// Unlike [enum Signature], [enum CompressedSignature] does not contain public key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub enum CompressedSignature {
    Ed25519(Ed25519SignatureAsBytes),
}

impl AsRef<[u8]> for CompressedSignature {
    fn as_ref(&self) -> &[u8] {
        match self {
            CompressedSignature::Ed25519(sig) => &sig.0,
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, From, PartialEq, Eq)]
pub enum SomaKeyPair {
    Ed25519(Ed25519KeyPair),
}

impl SomaKeyPair {
    pub fn public(&self) -> PublicKey {
        match self {
            SomaKeyPair::Ed25519(kp) => PublicKey::Ed25519(kp.public().into()),
        }
    }

    pub fn copy(&self) -> Self {
        match self {
            SomaKeyPair::Ed25519(kp) => kp.copy().into(),
        }
    }

    pub fn inner(&self) -> Ed25519KeyPair {
        match self {
            SomaKeyPair::Ed25519(kp) => kp.copy(),
        }
    }
}

impl Signer<Signature> for SomaKeyPair {
    fn sign(&self, msg: &[u8]) -> Signature {
        match self {
            SomaKeyPair::Ed25519(kp) => kp.sign(msg),
        }
    }
}

impl EncodeDecodeBase64 for SomaKeyPair {
    fn encode_base64(&self) -> String {
        Base64::encode(self.to_bytes())
    }

    fn decode_base64(value: &str) -> FastCryptoResult<Self> {
        let bytes = Base64::decode(value)?;
        Self::from_bytes(&bytes).map_err(|_| FastCryptoError::InvalidInput)
    }
}

impl SomaKeyPair {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.push(self.public().flag());

        match self {
            SomaKeyPair::Ed25519(kp) => {
                bytes.extend_from_slice(kp.as_bytes());
            }
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, eyre::Report> {
        match SignatureScheme::from_flag_byte(bytes.first().ok_or_else(|| eyre!("Invalid length"))?)
        {
            Ok(x) => match x {
                SignatureScheme::ED25519 => Ok(SomaKeyPair::Ed25519(Ed25519KeyPair::from_bytes(
                    bytes.get(1..).ok_or_else(|| eyre!("Invalid length"))?,
                )?)),
                _ => Err(eyre!("Invalid flag byte")),
            },
            _ => Err(eyre!("Invalid bytes")),
        }
    }

    pub fn to_bytes_no_flag(&self) -> Vec<u8> {
        match self {
            SomaKeyPair::Ed25519(kp) => kp.as_bytes().to_vec(),
        }
    }

    /// Encode a SomaKeyPair as `flag || privkey` in Bech32 starting with "somaprivkey" to a string. Note that the pubkey is not encoded.
    pub fn encode(&self) -> Result<String, eyre::Report> {
        Bech32::encode(self.to_bytes(), SOMA_PRIV_KEY_PREFIX).map_err(|e| eyre!(e))
    }

    /// Decode a SomaKeyPair from `flag || privkey` in Bech32 starting with "somaprivkey" to SomaKeyPair. The public key is computed directly from the private key bytes.
    pub fn decode(value: &str) -> Result<Self, eyre::Report> {
        let bytes = Bech32::decode(value, SOMA_PRIV_KEY_PREFIX)?;
        Self::from_bytes(&bytes)
    }
}

impl Serialize for SomaKeyPair {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = self.encode_base64();
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for SomaKeyPair {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let s = String::deserialize(deserializer)?;
        SomaKeyPair::decode_base64(&s).map_err(|e| Error::custom(e.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublicKey {
    Ed25519(Ed25519PublicKeyAsBytes),
}

impl PublicKey {
    pub fn flag(&self) -> u8 {
        self.scheme().flag()
    }

    pub fn scheme(&self) -> SignatureScheme {
        match self {
            PublicKey::Ed25519(_) => Ed25519SomaSignature::SCHEME,
        }
    }

    pub fn try_from_bytes(
        curve: SignatureScheme,
        key_bytes: &[u8],
    ) -> Result<PublicKey, eyre::Report> {
        match curve {
            SignatureScheme::ED25519 => {
                Ok(PublicKey::Ed25519((&Ed25519PublicKey::from_bytes(key_bytes)?).into()))
            }
            _ => Err(eyre!("Unsupported curve")),
        }
    }
}

impl EncodeDecodeBase64 for PublicKey {
    fn encode_base64(&self) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&[self.flag()]);
        bytes.extend_from_slice(self.as_ref());
        Base64::encode(&bytes[..])
    }

    fn decode_base64(value: &str) -> FastCryptoResult<Self> {
        let bytes = Base64::decode(value)?;
        match bytes.first() {
            Some(x) => {
                if x == &SignatureScheme::ED25519.flag() {
                    let pk: Ed25519PublicKey =
                        Ed25519PublicKey::from_bytes(bytes.get(1..).ok_or(
                            FastCryptoError::InputLengthWrong(Ed25519PublicKey::LENGTH + 1),
                        )?)?;
                    Ok(PublicKey::Ed25519((&pk).into()))
                } else {
                    Err(FastCryptoError::InvalidInput)
                }
            }
            _ => Err(FastCryptoError::InvalidInput),
        }
    }
}

impl AsRef<[u8]> for PublicKey {
    fn as_ref(&self) -> &[u8] {
        match self {
            PublicKey::Ed25519(pk) => &pk.0,
        }
    }
}

// Enums for signature scheme signatures
#[enum_dispatch]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Signature {
    Ed25519SomaSignature,
}

impl Serialize for Signature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = self.as_ref();

        if serializer.is_human_readable() {
            let s = Base64::encode(bytes);
            serializer.serialize_str(&s)
        } else {
            serializer.serialize_bytes(bytes)
        }
    }
}

impl<'de> Deserialize<'de> for Signature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        let bytes = if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            Base64::decode(&s).map_err(|e| Error::custom(e.to_string()))?
        } else {
            let data: Vec<u8> = Vec::deserialize(deserializer)?;
            data
        };

        Self::from_bytes(&bytes).map_err(|e| Error::custom(e.to_string()))
    }
}

impl Signature {
    /// The messaged passed in is already hashed form.
    pub fn new_hashed(hashed_msg: &[u8], secret: &dyn Signer<Signature>) -> Self {
        Signer::sign(secret, hashed_msg)
    }

    pub fn new_secure<T>(value: &IntentMessage<T>, secret: &dyn Signer<Signature>) -> Self
    where
        T: Serialize,
    {
        // Compute the BCS hash of the value in intent message. In the case of transaction data,
        // this is the BCS hash of `struct TransactionData`, different from the transaction digest
        // itself that computes the BCS hash of the Rust type prefix and `struct TransactionData`.
        // (See `fn digest` in `impl Message for SenderSignedData`).
        let mut hasher = DefaultHash::default();
        hasher.update(bcs::to_bytes(&value).expect("Message serialization should not fail"));

        Signer::sign(secret, &hasher.finalize().digest)
    }
}

impl AsRef<[u8]> for Signature {
    fn as_ref(&self) -> &[u8] {
        match self {
            Signature::Ed25519SomaSignature(sig) => sig.as_ref(),
        }
    }
}

impl AsMut<[u8]> for Signature {
    fn as_mut(&mut self) -> &mut [u8] {
        match self {
            Signature::Ed25519SomaSignature(sig) => sig.as_mut(),
        }
    }
}

impl ToFromBytes for Signature {
    fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        match bytes.first() {
            Some(x) => {
                if x == &Ed25519SomaSignature::SCHEME.flag() {
                    Ok(<Ed25519SomaSignature as ToFromBytes>::from_bytes(bytes)?.into())
                } else {
                    Err(FastCryptoError::InvalidInput)
                }
            }
            _ => Err(FastCryptoError::InvalidInput),
        }
    }
}

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash, AsRef, AsMut)]
#[as_ref(forward)]
#[as_mut(forward)]
pub struct Ed25519SomaSignature(
    #[schemars(with = "Base64")]
    #[serde_as(as = "Readable<Base64, Bytes>")]
    [u8; Ed25519PublicKey::LENGTH + Ed25519Signature::LENGTH + 1],
);

// Implementation useful for simplify testing when mock signature is needed
impl Default for Ed25519SomaSignature {
    fn default() -> Self {
        Self([0; Ed25519PublicKey::LENGTH + Ed25519Signature::LENGTH + 1])
    }
}

impl SomaSignatureInner for Ed25519SomaSignature {
    type Sig = Ed25519Signature;
    type PubKey = Ed25519PublicKey;
    type KeyPair = Ed25519KeyPair;
    const LENGTH: usize = Ed25519PublicKey::LENGTH + Ed25519Signature::LENGTH + 1;
}

impl SomaSignatureInner for Signature {
    type Sig = Ed25519Signature;
    type PubKey = Ed25519PublicKey;
    type KeyPair = Ed25519KeyPair;
    const LENGTH: usize = Ed25519PublicKey::LENGTH + Ed25519Signature::LENGTH + 1;
}

impl SomaPublicKey for Ed25519PublicKey {
    const SIGNATURE_SCHEME: SignatureScheme = SignatureScheme::ED25519;
}

impl ToFromBytes for Ed25519SomaSignature {
    fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        if bytes.len() != Self::LENGTH {
            return Err(FastCryptoError::InputLengthWrong(Self::LENGTH));
        }
        let mut sig_bytes = [0; Self::LENGTH];
        sig_bytes.copy_from_slice(bytes);
        Ok(Self(sig_bytes))
    }
}

impl Signer<Signature> for Ed25519KeyPair {
    fn sign(&self, msg: &[u8]) -> Signature {
        Ed25519SomaSignature::new(self, msg).into()
    }
}

impl From<Ed25519SomaSignature> for Signature {
    fn from(sig: Ed25519SomaSignature) -> Self {
        Signature::Ed25519SomaSignature(sig)
    }
}

//
// This struct exists due to the limitations of the `enum_dispatch` library.
//
pub trait SomaSignatureInner: Sized + ToFromBytes + PartialEq + Eq + Hash {
    type Sig: Authenticator<PubKey = Self::PubKey>;
    type PubKey: VerifyingKey<Sig = Self::Sig> + SomaPublicKey;
    type KeyPair: KeypairTraits<PubKey = Self::PubKey, Sig = Self::Sig>;

    const LENGTH: usize = Self::Sig::LENGTH + Self::PubKey::LENGTH + 1;
    const SCHEME: SignatureScheme = Self::PubKey::SIGNATURE_SCHEME;

    /// Returns the deserialized signature and deserialized pubkey.
    fn get_verification_inputs(&self) -> SomaResult<(Self::Sig, Self::PubKey)> {
        let pk = Self::PubKey::from_bytes(self.public_key_bytes())
            .map_err(|_| SomaError::KeyConversionError("Invalid public key".to_string()))?;

        // deserialize the signature
        let signature = Self::Sig::from_bytes(self.signature_bytes()).map_err(|_| {
            SomaError::InvalidSignature { error: "Fail to get pubkey and sig".to_string() }
        })?;

        Ok((signature, pk))
    }

    fn new(kp: &Self::KeyPair, message: &[u8]) -> Self {
        let sig = Signer::sign(kp, message);

        let mut signature_bytes: Vec<u8> = Vec::new();
        signature_bytes
            .extend_from_slice(&[<Self::PubKey as SomaPublicKey>::SIGNATURE_SCHEME.flag()]);
        signature_bytes.extend_from_slice(sig.as_ref());
        signature_bytes.extend_from_slice(kp.public().as_ref());
        Self::from_bytes(&signature_bytes[..])
            .expect("Serialized signature did not have expected size")
    }
}

pub trait SomaPublicKey: VerifyingKey {
    const SIGNATURE_SCHEME: SignatureScheme;
}

impl SomaPublicKey for BLS12381PublicKey {
    const SIGNATURE_SCHEME: SignatureScheme = SignatureScheme::BLS12381;
}

pub trait SomaSignature: Sized + ToFromBytes {
    fn signature_bytes(&self) -> &[u8];
    fn public_key_bytes(&self) -> &[u8];
    fn scheme(&self) -> SignatureScheme;

    fn verify_secure<T>(
        &self,
        value: &IntentMessage<T>,
        author: SomaAddress,
        scheme: SignatureScheme,
    ) -> SomaResult<()>
    where
        T: Serialize;
}

impl<S: SomaSignatureInner + Sized> SomaSignature for S {
    fn signature_bytes(&self) -> &[u8] {
        // Access array slice is safe because the array bytes is initialized as
        // flag || signature || pubkey with its defined length.
        &self.as_ref()[1..1 + S::Sig::LENGTH]
    }

    fn public_key_bytes(&self) -> &[u8] {
        // Access array slice is safe because the array bytes is initialized as
        // flag || signature || pubkey with its defined length.
        &self.as_ref()[S::Sig::LENGTH + 1..]
    }

    fn scheme(&self) -> SignatureScheme {
        S::PubKey::SIGNATURE_SCHEME
    }

    fn verify_secure<T>(
        &self,
        value: &IntentMessage<T>,
        author: SomaAddress,
        scheme: SignatureScheme,
    ) -> Result<(), SomaError>
    where
        T: Serialize,
    {
        let mut hasher = DefaultHash::default();
        hasher.update(bcs::to_bytes(&value).expect("Message serialization should not fail"));
        let digest = hasher.finalize().digest;

        let (sig, pk) = &self.get_verification_inputs()?;
        let address = SomaAddress::from(pk);
        if author != address {
            return Err(SomaError::IncorrectSigner {
                error: format!("Incorrect signer, expected {:?}, got {:?}", author, address),
            });
        }
        pk.verify(&digest, sig).map_err(|e| SomaError::InvalidSignature {
            error: format!("Fail to verify user sig {}", e),
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NetworkPublicKey(ed25519::Ed25519PublicKey);
pub struct NetworkPrivateKey(ed25519::Ed25519PrivateKey);
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkSignature(ed25519::Ed25519Signature);

#[derive(Serialize, Debug, Deserialize)]
pub struct NetworkKeyPair(ed25519::Ed25519KeyPair);

impl NetworkPublicKey {
    pub fn new(key: ed25519::Ed25519PublicKey) -> Self {
        Self(key)
    }

    pub fn into_inner(self) -> ed25519::Ed25519PublicKey {
        self.0
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.0.to_bytes()
    }
    pub fn verify(&self, msg: &[u8], signature: &NetworkSignature) -> Result<(), FastCryptoError> {
        self.0.verify(msg, &signature.0)
    }
}

impl NetworkPrivateKey {
    pub fn into_inner(self) -> ed25519::Ed25519PrivateKey {
        self.0
    }
}

impl NetworkKeyPair {
    pub fn new(keypair: ed25519::Ed25519KeyPair) -> Self {
        Self(keypair)
    }

    pub fn generate<R: rand::Rng + fastcrypto::traits::AllowedRng>(rng: &mut R) -> Self {
        Self(ed25519::Ed25519KeyPair::generate(rng))
    }

    pub fn public(&self) -> NetworkPublicKey {
        NetworkPublicKey(self.0.public().clone())
    }

    pub fn private_key(self) -> NetworkPrivateKey {
        NetworkPrivateKey(self.0.copy().private())
    }

    pub fn private_key_bytes(self) -> [u8; 32] {
        self.0.private().0.to_bytes()
    }
    pub fn into_inner(self) -> ed25519::Ed25519KeyPair {
        self.0
    }

    pub fn sign(&self, msg: &[u8]) -> NetworkSignature {
        NetworkSignature(self.0.sign(msg))
    }
}

impl Clone for NetworkKeyPair {
    fn clone(&self) -> Self {
        Self(self.0.copy())
    }
}

/// Protocol key is used for signing blocks and verifying block signatures.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProtocolPublicKey(ed25519::Ed25519PublicKey);
pub struct ProtocolKeyPair(ed25519::Ed25519KeyPair);
pub struct ProtocolKeySignature(ed25519::Ed25519Signature);

impl ProtocolPublicKey {
    pub fn new(key: ed25519::Ed25519PublicKey) -> Self {
        Self(key)
    }

    pub fn verify(
        &self,
        message: &[u8],
        signature: &ProtocolKeySignature,
    ) -> Result<(), FastCryptoError> {
        self.0.verify(message, &signature.0)
    }

    pub fn to_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    pub fn inner(&self) -> ed25519::Ed25519PublicKey {
        self.0.clone()
    }
}

impl ProtocolKeyPair {
    pub fn new(keypair: ed25519::Ed25519KeyPair) -> Self {
        Self(keypair)
    }

    pub fn generate<R: rand::Rng + fastcrypto::traits::AllowedRng>(rng: &mut R) -> Self {
        Self(ed25519::Ed25519KeyPair::generate(rng))
    }

    pub fn public(&self) -> ProtocolPublicKey {
        ProtocolPublicKey(self.0.public().clone())
    }

    pub fn sign(&self, message: &[u8]) -> ProtocolKeySignature {
        ProtocolKeySignature(self.0.sign(message))
    }
}

impl Clone for ProtocolKeyPair {
    fn clone(&self) -> Self {
        Self(self.0.copy())
    }
}

impl ProtocolKeySignature {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        Ok(Self(ed25519::Ed25519Signature::from_bytes(bytes)?))
    }

    pub fn to_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

#[derive(Default)]
pub struct VerificationObligation<'a> {
    pub messages: Vec<Vec<u8>>,
    pub signatures: Vec<AggregateAuthoritySignature>,
    pub public_keys: Vec<Vec<&'a AuthorityPublicKey>>,
}

impl<'a> VerificationObligation<'a> {
    pub fn new() -> VerificationObligation<'a> {
        VerificationObligation::default()
    }

    /// Add a new message to the list of messages to be verified.
    /// Returns the index of the message.
    pub fn add_message<T>(&mut self, message_value: &T, epoch: EpochId, intent: Intent) -> usize
    where
        T: Serialize,
    {
        let intent_msg = IntentMessage::new(intent, message_value);
        let mut intent_msg_bytes =
            bcs::to_bytes(&intent_msg).expect("Message serialization should not fail");
        epoch.write(&mut intent_msg_bytes);
        self.signatures.push(AggregateAuthoritySignature::default());
        self.public_keys.push(Vec::new());
        self.messages.push(intent_msg_bytes);
        self.messages.len() - 1
    }

    // Attempts to add signature and public key to the obligation. If this fails, ensure to call `verify` manually.
    pub fn add_signature_and_public_key(
        &mut self,
        signature: &AuthoritySignature,
        public_key: &'a AuthorityPublicKey,
        idx: usize,
    ) -> SomaResult<()> {
        self.public_keys.get_mut(idx).ok_or(SomaError::InvalidAuthenticator)?.push(public_key);
        self.signatures
            .get_mut(idx)
            .ok_or(SomaError::InvalidAuthenticator)?
            .add_signature(signature.clone())
            .map_err(|_| SomaError::InvalidSignature {
                error: "Failed to add signature to obligation".to_string(),
            })?;
        Ok(())
    }

    pub fn verify_all(self) -> SomaResult<()> {
        let mut pks = Vec::with_capacity(self.public_keys.len());
        for pk in self.public_keys.clone() {
            pks.push(pk.into_iter());
        }
        AggregateAuthoritySignature::batch_verify(
            &self.signatures.iter().collect::<Vec<_>>()[..],
            pks,
            &self.messages.iter().map(|x| &x[..]).collect::<Vec<_>>()[..],
        )
        .map_err(|e| {
            let message = format!(
                "pks: {:?}, messages: {:?}, sigs: {:?}",
                &self.public_keys,
                self.messages.iter().map(Base64::encode).collect::<Vec<String>>(),
                &self
                    .signatures
                    .iter()
                    .map(|s| Base64::encode(s.as_ref()))
                    .collect::<Vec<String>>()
            );

            let chunk_size = 2048;

            // This error message may be very long, so we print out the error in chunks of to avoid
            // hitting a max log line length on the system.
            for (i, chunk) in
                message.as_bytes().chunks(chunk_size).map(std::str::from_utf8).enumerate()
            {
                warn!(
                    "Failed to batch verify aggregated auth sig: {} (chunk {}): {}",
                    e,
                    i,
                    chunk.unwrap()
                );
            }

            SomaError::InvalidSignature {
                error: format!("Failed to batch verify aggregated auth sig: {}", e),
            }
        })?;
        Ok(())
    }
}

impl FromStr for Signature {
    type Err = eyre::Report;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::decode_base64(s).map_err(|e| eyre!("Fail to decode base64 {}", e.to_string()))
    }
}

impl FromStr for PublicKey {
    type Err = eyre::Report;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::decode_base64(s).map_err(|e| eyre!("Fail to decode base64 {}", e.to_string()))
    }
}

impl FromStr for GenericSignature {
    type Err = eyre::Report;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::decode_base64(s).map_err(|e| eyre!("Fail to decode base64 {}", e.to_string()))
    }
}

/// AES-256 symmetric decryption key for encrypted model weights.
/// 32 bytes. Used with AES-256-CTR for actual encryption/decryption
/// of model weights in the CLI/inference-engine.
#[serde_as]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DecryptionKey(#[serde_as(as = "Bytes")] [u8; 32]);

impl DecryptionKey {
    pub fn new(key: [u8; 32]) -> Self {
        Self(key)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Debug for DecryptionKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "DecryptionKey(<redacted>)")
    }
}
