use super::{block::SignedBlockHeader, transaction::SignedTransaction};
use crate::crypto::keys::ProtocolKeySignature;
use crate::crypto::{DefaultHashFunction, DIGEST_LENGTH};
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{
    fmt,
    hash::{Hash, Hasher},
};

/// Certificate is a generic transaction certificate that contains a signed
/// transaction, a signed block_header, and a threshold signature. The verification
/// process consists of:
///
/// 1. Compute the digest of the block_header and verify against the threshold signature for the corresponding epoch
/// 2. Compute the TransactionRef of the signed_transaction and verify inclusion in the block
/// 3. Verify the transaction details are required e.g. payment amount is correct
#[derive(Clone, Deserialize, Serialize)]
#[enum_dispatch(TransactionCertificateAPI)]
pub(crate) enum TransactionCertificate {
    V1(TransactionCertificateV1),
}

/// Version 1 of the transaction certificate. Adding versioning here because while not currently sent over the wire,
/// it is reasonable to assume that it may be.
#[derive(Clone, Deserialize, Serialize)]
struct TransactionCertificateV1 {
    /// a transaction with its corresponding signature.
    signed_transaction: SignedTransaction,
    /// a block header with its corresponding signature
    signed_block_header: SignedBlockHeader,
    // TODO: replace this with a proper threshold signature
    /// threshold signature
    certificate_signature: ProtocolKeySignature,
}

impl TransactionCertificate {
    /// new constructs a new transaction certificate
    pub(crate) const fn new_v1(
        signed_transaction: SignedTransaction,
        signed_block_header: SignedBlockHeader,
        certificate_signature: ProtocolKeySignature,
    ) -> TransactionCertificateV1 {
        TransactionCertificateV1 {
            signed_transaction,
            signed_block_header,
            certificate_signature,
        }
    }
}

/// [`TransactionCertificate`] API exposes common fields. Functionality can be added, but should not be removed.
#[enum_dispatch]
trait TransactionCertificateAPI {
    /// Returns the signed transaction
    fn signed_transaction(&self) -> &SignedTransaction;
    /// Returns the round
    fn signed_block_header(&self) -> &SignedBlockHeader;
    /// Returns the author authority index. Dependent on epoch.
    fn certificate_signature(&self) -> &ProtocolKeySignature;
}

impl TransactionCertificateAPI for TransactionCertificateV1 {
    fn signed_transaction(&self) -> &SignedTransaction {
        &self.signed_transaction
    }
    fn signed_block_header(&self) -> &SignedBlockHeader {
        &self.signed_block_header
    }
    fn certificate_signature(&self) -> &ProtocolKeySignature {
        &self.certificate_signature
    }
}

/// Note: clone() is relatively cheap with most underlying data refcounted.
#[derive(Clone)]
struct VerifiedTransactionCertificate {
    /// arc reference to the full block data
    transaction_certificate: Arc<TransactionCertificate>,
    digest: TransactionCertificateDigest,
    serialized: Bytes,
}

impl VerifiedTransactionCertificate {
    pub(crate) fn digest(&self) -> TransactionCertificateDigest {
        self.digest
    }

    /// Returns the serialized block with signature.
    pub(crate) fn serialized(&self) -> &Bytes {
        &self.serialized
    }
}

/// Identifies a transaction certificate which includes the certificate signature
/// the signature algorithm is assumed to be non-malleable, so it is impossible for another
/// party to create an altered but valid signature, producing an equivocating Digest.
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TransactionCertificateDigest([u8; DIGEST_LENGTH]);

impl TransactionCertificateDigest {
    pub const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    pub const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}

impl Hash for TransactionCertificateDigest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<TransactionCertificateDigest> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: TransactionCertificateDigest) -> Self {
        Digest::new(hd.0)
    }
}

impl fmt::Display for TransactionCertificateDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
                .get(0..4)
                .ok_or(fmt::Error)?
        )
    }
}

impl fmt::Debug for TransactionCertificateDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for TransactionCertificateDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
