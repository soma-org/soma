use serde::{Deserialize, Serialize};

use crate::{
    crypto::{AuthoritySignInfo, AuthorityStrongQuorumSignInfo},
    transaction::CertifiedTransaction,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TransactionStatus {
    /// Signature over the transaction.
    Signed(AuthoritySignInfo),
    /// For executed transaction, we could return an optional certificate signature on the transaction
    /// (i.e. the signature part of the CertifiedTransaction), as well as the signed effects.
    /// The certificate signature is optional because for transactions executed in previous
    /// epochs, we won't keep around the certificate signatures.
    Executed(Option<AuthorityStrongQuorumSignInfo>),
}

impl TransactionStatus {
    pub fn into_signed_for_testing(self) -> AuthoritySignInfo {
        match self {
            Self::Signed(s) => s,
            _ => unreachable!("Incorrect response type"),
        }
    }
}

impl PartialEq for TransactionStatus {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::Signed(s1) => match other {
                Self::Signed(s2) => s1.epoch == s2.epoch,
                _ => false,
            },
            Self::Executed(c1) => match other {
                Self::Executed(c2) => c1.as_ref().map(|a| a.epoch) == c2.as_ref().map(|a| a.epoch),
                _ => false,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HandleTransactionResponse {
    pub status: TransactionStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandleCertificateResponse {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandleCertificateRequest {
    pub certificate: CertifiedTransaction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitCertificateResponse {
    /// If transaction is already executed, return same result as handle_certificate
    pub executed: Option<HandleCertificateResponse>,
}
