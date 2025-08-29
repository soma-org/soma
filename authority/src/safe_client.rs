use std::{net::SocketAddr, sync::Arc};

use crate::client::AuthorityAPI;
use tracing::debug;
use types::crypto::AuthoritySignInfoTrait;
use types::finality::SignedConsensusFinality;
use types::intent::Intent;
use types::storage::committee_store::CommitteeStore;
use types::{
    committee::{Committee, EpochId},
    crypto::AuthorityPublicKeyBytes,
    digests::{TransactionDigest, TransactionEffectsDigest},
    effects::{self, SignedTransactionEffects, TransactionEffectsAPI},
    error::{SomaError, SomaResult},
    grpc::{HandleCertificateRequest, HandleCertificateResponse, TransactionStatus},
    quorum_driver::PlainTransactionInfoResponse,
    transaction::{CertifiedTransaction, SignedTransaction, Transaction},
};

#[derive(Clone)]
pub struct SafeClient<C>
where
    C: Clone,
{
    authority_client: C,
    committee_store: Arc<CommitteeStore>,
    address: AuthorityPublicKeyBytes,
}

impl<C: Clone> SafeClient<C> {
    pub fn new(
        authority_client: C,
        committee_store: Arc<CommitteeStore>,
        address: AuthorityPublicKeyBytes,
    ) -> Self {
        Self {
            authority_client,
            committee_store,
            address,
        }
    }
}

impl<C: Clone> SafeClient<C> {
    pub fn authority_client(&self) -> &C {
        &self.authority_client
    }

    #[cfg(test)]
    pub fn authority_client_mut(&mut self) -> &mut C {
        &mut self.authority_client
    }

    fn get_committee(&self, epoch_id: &EpochId) -> SomaResult<Arc<Committee>> {
        self.committee_store
            .get_committee(epoch_id)?
            .ok_or(SomaError::MissingCommitteeAtEpoch(*epoch_id))
    }

    fn check_signed_effects_plain(
        &self,
        digest: &TransactionDigest,
        signed_effects: SignedTransactionEffects,
        expected_effects_digest: Option<&TransactionEffectsDigest>,
    ) -> SomaResult<SignedTransactionEffects> {
        // Check it has the right signer
        if !(signed_effects.auth_sig().authority == self.address) {
            return Err(SomaError::ByzantineAuthoritySuspicion {
                authority: self.address,
                reason: format!(
                    "Unexpected validator address in the signed effects signature: {:?}",
                    signed_effects.auth_sig().authority
                ),
            });
        }

        // Checks it concerns the right tx
        if !(signed_effects.data().transaction_digest() == digest) {
            return Err(SomaError::ByzantineAuthoritySuspicion {
                authority: self.address,
                reason: "Unexpected tx digest in the signed effects".to_string(),
            });
        }
        // check that the effects digest is correct.
        if let Some(effects_digest) = expected_effects_digest {
            if !(signed_effects.digest() == effects_digest) {
                return Err(SomaError::ByzantineAuthoritySuspicion {
                    authority: self.address,
                    reason: "Effects digest does not match with expected digest".to_string(),
                });
            }
        }
        self.get_committee(&signed_effects.epoch())?;
        Ok(signed_effects)
    }

    fn check_signed_finality_plain(
        &self,
        digest: &TransactionDigest,
        signed_finality: SignedConsensusFinality,
    ) -> SomaResult<SignedConsensusFinality> {
        // Check it has the right signer
        if signed_finality.auth_sig().authority != self.address {
            return Err(SomaError::ByzantineAuthoritySuspicion {
                authority: self.address,
                reason: format!(
                    "Unexpected validator address in the signed finality signature: {:?}",
                    signed_finality.auth_sig().authority
                ),
            });
        }

        // Check it concerns the right tx
        if signed_finality.data().tx_digest != *digest {
            return Err(SomaError::ByzantineAuthoritySuspicion {
                authority: self.address,
                reason: "Unexpected tx digest in the signed consensus finality".to_string(),
            });
        }

        // Verify the signature is valid for this epoch
        let committee = self.get_committee(&signed_finality.epoch())?;

        // Verify the signature
        signed_finality
            .auth_sig()
            .verify_secure(
                signed_finality.data(),
                Intent::soma_transaction(),
                &committee,
            )
            .map_err(|e| SomaError::ByzantineAuthoritySuspicion {
                authority: self.address,
                reason: format!("Invalid consensus finality signature: {}", e),
            })?;

        Ok(signed_finality)
    }

    fn check_transaction_info(
        &self,
        digest: &TransactionDigest,
        transaction: Transaction,
        status: TransactionStatus,
    ) -> SomaResult<PlainTransactionInfoResponse> {
        if !(digest == transaction.digest()) {
            return Err(SomaError::ByzantineAuthoritySuspicion {
                authority: self.address,
                reason: "Signed transaction digest does not match with expected digest".to_string(),
            });
        }
        match status {
            TransactionStatus::Signed(signed) => {
                self.get_committee(&signed.epoch)?;
                Ok(PlainTransactionInfoResponse::Signed(
                    SignedTransaction::new_from_data_and_sig(transaction.into_data(), signed),
                ))
            }
            TransactionStatus::Executed(cert_opt, effects, finality) => {
                let signed_effects = self.check_signed_effects_plain(digest, effects, None)?;
                // Verify finality if present
                let signed_finality = finality
                    .map(|f| self.check_signed_finality_plain(digest, f))
                    .transpose()?;
                match cert_opt {
                    Some(cert) => {
                        let committee = self.get_committee(&cert.epoch)?;
                        let ct = CertifiedTransaction::new_from_data_and_sig(
                            transaction.into_data(),
                            cert,
                        );
                        ct.verify_committee_sigs_only(&committee).map_err(|e| {
                            SomaError::FailedToVerifyTxCertWithExecutedEffects {
                                validator_name: self.address,
                                error: e.to_string(),
                            }
                        })?;
                        Ok(PlainTransactionInfoResponse::ExecutedWithCert(
                            ct,
                            signed_effects,
                            signed_finality,
                        ))
                    }
                    None => Ok(PlainTransactionInfoResponse::ExecutedWithoutCert(
                        transaction,
                        signed_effects,
                        signed_finality,
                    )),
                }
            }
        }
    }

    pub fn address(&self) -> &AuthorityPublicKeyBytes {
        &self.address
    }

    fn verify_certificate_response(
        &self,
        digest: &TransactionDigest,
        HandleCertificateResponse {
            signed_effects,
            signed_finality,
        }: HandleCertificateResponse,
    ) -> SomaResult<HandleCertificateResponse> {
        let signed_effects = self.check_signed_effects_plain(digest, signed_effects, None)?;

        let signed_finality = signed_finality
            .map(|finality| self.check_signed_finality_plain(digest, finality))
            .transpose()?;

        debug!(
            "Verified certificate response: effects={:?}, has_finality={}",
            signed_effects,
            signed_finality.is_some()
        );

        Ok(HandleCertificateResponse {
            signed_effects,
            signed_finality,
        })
    }
}

impl<C> SafeClient<C>
where
    C: AuthorityAPI + Send + Sync + Clone + 'static,
{
    /// Initiate a new transfer to a Sui or Primary account.
    pub async fn handle_transaction(
        &self,
        transaction: Transaction,
        client_addr: Option<SocketAddr>,
    ) -> Result<PlainTransactionInfoResponse, SomaError> {
        let digest = *transaction.digest();
        let response = self
            .authority_client
            .handle_transaction(transaction.clone(), client_addr)
            .await?;
        self.check_transaction_info(&digest, transaction, response.status)
    }

    /// Execute a certificate.
    pub async fn handle_certificate(
        &self,
        request: HandleCertificateRequest,
        client_addr: Option<SocketAddr>,
    ) -> Result<HandleCertificateResponse, SomaError> {
        let digest = *request.certificate.digest();
        let response = self
            .authority_client
            .handle_certificate(request, client_addr)
            .await?;

        self.verify_certificate_response(&digest, response)
    }
}
