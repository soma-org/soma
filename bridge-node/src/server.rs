//! gRPC signature exchange server.
//!
//! Each validator's bridge node runs this server to collect and share
//! ECDSA signatures for bridge actions (deposits, withdrawals, emergency ops).

use dashmap::DashMap;
use std::collections::BTreeMap;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};
use types::bridge::BridgeCommittee;

use crate::proto::bridge_node_server::BridgeNode;
use crate::proto::{
    CollectedSignature, GetSignaturesRequest, GetSignaturesResponse, SignatureRequest,
    SignatureResponse,
};

/// Collected signatures for a single bridge action.
#[derive(Debug, Default)]
struct ActionSignatures {
    /// Map from signer_index to 65-byte signature.
    signatures: BTreeMap<u32, Vec<u8>>,
}

/// The gRPC server implementation for bridge signature exchange.
pub struct BridgeServer {
    /// Collected signatures, keyed by action digest (keccak256 hash).
    signatures: Arc<DashMap<Vec<u8>, ActionSignatures>>,
    /// The current bridge committee (for stake calculation).
    committee: Arc<tokio::sync::RwLock<BridgeCommittee>>,
}

impl BridgeServer {
    pub fn new(committee: BridgeCommittee) -> Self {
        Self {
            signatures: Arc::new(DashMap::new()),
            committee: Arc::new(tokio::sync::RwLock::new(committee)),
        }
    }

    /// Get the total voting power of collected signatures for an action.
    async fn total_stake_for_action(&self, action_digest: &[u8]) -> u64 {
        let committee = self.committee.read().await;
        let members: Vec<_> = committee.members.values().collect();

        self.signatures
            .get(action_digest)
            .map(|entry| {
                entry
                    .signatures
                    .keys()
                    .filter_map(|idx| members.get(*idx as usize))
                    .map(|m| m.voting_power)
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Check if we have enough signatures for a given threshold.
    pub async fn has_quorum(&self, action_digest: &[u8], threshold: u64) -> bool {
        self.total_stake_for_action(action_digest).await >= threshold
    }

    /// Get all collected signatures and bitmap for an action.
    /// Returns (aggregated_signatures, signer_bitmap) ready for on-chain submission.
    pub async fn get_aggregated_signatures(
        &self,
        action_digest: &[u8],
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        let entry = self.signatures.get(action_digest)?;
        let committee = self.committee.read().await;
        let member_count = committee.members.len();

        // Build aggregated signature (concatenated 65-byte sigs in order)
        // and bitmap (one bit per committee member)
        let bitmap_len = (member_count + 7) / 8;
        let mut bitmap = vec![0u8; bitmap_len];
        let mut agg_sig = Vec::new();

        for (&idx, sig) in &entry.signatures {
            if (idx as usize) < member_count {
                bitmap[idx as usize / 8] |= 1 << (idx as usize % 8);
                agg_sig.extend_from_slice(sig);
            }
        }

        Some((agg_sig, bitmap))
    }

    /// Update the committee (e.g., after epoch boundary).
    pub async fn update_committee(&self, new_committee: BridgeCommittee) {
        let mut committee = self.committee.write().await;
        *committee = new_committee;
        // Clear stale signatures from previous committee
        self.signatures.clear();
        info!("Bridge committee updated, cleared pending signatures");
    }
}

#[tonic::async_trait]
impl BridgeNode for BridgeServer {
    async fn submit_signature(
        &self,
        request: Request<SignatureRequest>,
    ) -> Result<Response<SignatureResponse>, Status> {
        let req = request.into_inner();

        if req.action_digest.len() != 32 {
            return Err(Status::invalid_argument("action_digest must be 32 bytes"));
        }
        if req.signature.len() != 65 {
            return Err(Status::invalid_argument(
                "signature must be 65 bytes (recoverable ECDSA)",
            ));
        }

        // Validate signer_index is within committee bounds
        {
            let committee = self.committee.read().await;
            if req.signer_index as usize >= committee.members.len() {
                return Err(Status::invalid_argument(format!(
                    "signer_index {} out of range (committee size: {})",
                    req.signer_index,
                    committee.members.len()
                )));
            }
        }

        // Store the signature
        let mut entry = self
            .signatures
            .entry(req.action_digest.clone())
            .or_default();

        if entry.signatures.contains_key(&req.signer_index) {
            debug!(
                signer = req.signer_index,
                "Duplicate signature, ignoring"
            );
        } else {
            entry
                .signatures
                .insert(req.signer_index, req.signature);
            debug!(
                signer = req.signer_index,
                total = entry.signatures.len(),
                "Signature collected"
            );
        }

        let total = entry.signatures.len() as u32;

        Ok(Response::new(SignatureResponse {
            accepted: true,
            total_signatures: total,
        }))
    }

    async fn get_signatures(
        &self,
        request: Request<GetSignaturesRequest>,
    ) -> Result<Response<GetSignaturesResponse>, Status> {
        let req = request.into_inner();

        let (sigs, total_stake) = match self.signatures.get(&req.action_digest) {
            Some(entry) => {
                let committee = self.committee.read().await;
                let members: Vec<_> = committee.members.values().collect();

                let sigs: Vec<CollectedSignature> = entry
                    .signatures
                    .iter()
                    .map(|(&idx, sig)| CollectedSignature {
                        signer_index: idx,
                        signature: sig.clone(),
                    })
                    .collect();

                let total_stake: u64 = entry
                    .signatures
                    .keys()
                    .filter_map(|idx| members.get(*idx as usize))
                    .map(|m| m.voting_power)
                    .sum();

                (sigs, total_stake)
            }
            None => (vec![], 0),
        };

        Ok(Response::new(GetSignaturesResponse {
            signatures: sigs,
            total_stake,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::base::SomaAddress;
    use types::bridge::{BridgeCommittee, BridgeMember};

    fn test_committee(n: usize) -> BridgeCommittee {
        let mut members = BTreeMap::new();
        for i in 0..n {
            let addr = SomaAddress::from([i as u8; 32]);
            members.insert(
                addr,
                BridgeMember {
                    ecdsa_pubkey: vec![0x02; 33],
                    voting_power: 10000 / n as u64,
                },
            );
        }
        BridgeCommittee {
            members,
            threshold_deposit: 3334,
            threshold_withdraw: 3334,
            threshold_pause: 450,
            threshold_unpause: 6667,
        }
    }

    #[tokio::test]
    async fn test_submit_and_get_signatures() {
        let server = BridgeServer::new(test_committee(4));
        let digest = vec![0xAB; 32];
        let sig = vec![0x01; 65];

        // Submit signature
        let req = Request::new(SignatureRequest {
            action_digest: digest.clone(),
            signature: sig.clone(),
            signer_index: 0,
        });
        let resp = server.submit_signature(req).await.unwrap();
        assert!(resp.get_ref().accepted);
        assert_eq!(resp.get_ref().total_signatures, 1);

        // Get signatures
        let req = Request::new(GetSignaturesRequest {
            action_digest: digest.clone(),
        });
        let resp = server.get_signatures(req).await.unwrap();
        assert_eq!(resp.get_ref().signatures.len(), 1);
        assert_eq!(resp.get_ref().total_stake, 2500); // 10000/4
    }

    #[tokio::test]
    async fn test_quorum_detection() {
        let server = BridgeServer::new(test_committee(4));
        let digest = vec![0xAB; 32];

        // 1 signature = 2500 stake (below deposit threshold of 3334)
        let req = Request::new(SignatureRequest {
            action_digest: digest.clone(),
            signature: vec![0x01; 65],
            signer_index: 0,
        });
        server.submit_signature(req).await.unwrap();
        assert!(!server.has_quorum(&digest, 3334).await);

        // 2 signatures = 5000 stake (above deposit threshold)
        let req = Request::new(SignatureRequest {
            action_digest: digest.clone(),
            signature: vec![0x02; 65],
            signer_index: 1,
        });
        server.submit_signature(req).await.unwrap();
        assert!(server.has_quorum(&digest, 3334).await);
    }

    #[tokio::test]
    async fn test_aggregated_signatures() {
        let server = BridgeServer::new(test_committee(4));
        let digest = vec![0xAB; 32];

        // Submit from signers 0 and 2
        for (idx, sig_byte) in [(0u32, 0x01u8), (2, 0x03)] {
            let req = Request::new(SignatureRequest {
                action_digest: digest.clone(),
                signature: vec![sig_byte; 65],
                signer_index: idx,
            });
            server.submit_signature(req).await.unwrap();
        }

        let (agg_sig, bitmap) = server.get_aggregated_signatures(&digest).await.unwrap();

        // 2 signatures × 65 bytes = 130 bytes
        assert_eq!(agg_sig.len(), 130);
        // Bitmap: bit 0 and bit 2 set = 0b00000101 = 5
        assert_eq!(bitmap, vec![5]);
    }

    #[tokio::test]
    async fn test_invalid_signer_index() {
        let server = BridgeServer::new(test_committee(4));
        let req = Request::new(SignatureRequest {
            action_digest: vec![0xAB; 32],
            signature: vec![0x01; 65],
            signer_index: 99, // out of bounds
        });
        let result = server.submit_signature(req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_duplicate_signature_ignored() {
        let server = BridgeServer::new(test_committee(4));
        let digest = vec![0xAB; 32];

        // Submit twice from same signer
        for _ in 0..2 {
            let req = Request::new(SignatureRequest {
                action_digest: digest.clone(),
                signature: vec![0x01; 65],
                signer_index: 0,
            });
            server.submit_signature(req).await.unwrap();
        }

        // Should still be 1 signature
        let req = Request::new(GetSignaturesRequest {
            action_digest: digest,
        });
        let resp = server.get_signatures(req).await.unwrap();
        assert_eq!(resp.get_ref().signatures.len(), 1);
    }
}
