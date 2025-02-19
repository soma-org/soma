use serde::{Deserialize, Serialize};

use crate::{
    authority_committee::{AuthorityBitSet, AuthorityCommittee},
    block::{BlockRef, Epoch},
    crypto::keys::{AuthorityAggregateSignature, AuthorityPublicKey},
    error::{SharedError, SharedResult},
    transaction::SignedTransaction,
};

/// In the future this will likely operate at the block level rather than individual level.
/// Instead of a single tx, the authorities will sign off on a merkle root of the txs in a block.
/// The proof for a specific tx would then be the signed root, along with the neccessary leaves.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BlockClaim {
    /// Epoch links to the correct authority committee
    epoch: Epoch,
    /// BlockRef links to the block that is being proven
    block_ref: BlockRef,
    /// The tx claim
    transaction: SignedTransaction,
}

impl BlockClaim {
    pub fn new(epoch: Epoch, block_ref: BlockRef, transaction: SignedTransaction) -> Self {
        Self {
            epoch,
            block_ref,
            transaction,
        }
    }
}

///Finality proof
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FinalityProof {
    /// details of the block (epoch, block_ref, tx or block merkle root in the future)
    claim: BlockClaim,
    /// which authorities signed off (for the specified epoch in the claim)
    authorities: AuthorityBitSet,
    /// agg signature
    signature: AuthorityAggregateSignature,
    // TODO: add merkle leaves
}

impl FinalityProof {
    pub fn new(
        claim: BlockClaim,
        authorities: AuthorityBitSet,
        signature: AuthorityAggregateSignature,
    ) -> Self {
        Self {
            claim,
            authorities,
            signature,
        }
    }
    pub fn epoch(&self) -> Epoch {
        self.claim.epoch
    }
    pub fn block_ref(&self) -> BlockRef {
        self.claim.block_ref
    }
    pub fn transaction(&self) -> SignedTransaction {
        self.claim.transaction.clone()
    }
    pub fn verify(&self, committee: &AuthorityCommittee) -> SharedResult<()> {
        if self.claim.epoch != committee.epoch() {
            return Err(SharedError::WrongEpoch);
        }
        let authorities = self.authorities.get_indices();
        let pks: Vec<AuthorityPublicKey> = authorities
            .iter()
            .map(|authority| committee.authority(*authority).authority_key.clone())
            .collect();

        let aggregate_stake: u64 = authorities
            .iter()
            .map(|authority| committee.authority(*authority).stake)
            .sum();

        if aggregate_stake < committee.quorum_threshold() {
            return Err(SharedError::QuorumFailed);
        }

        // TODO: should probably enforce scoping to the BLS / Authority aggregation system similar to ED25519
        let message = bcs::to_bytes(&self.claim).map_err(SharedError::SerializationFailure)?;
        self.signature
            .verify(&pks, &message)
            .map_err(SharedError::MalformedSignature)
    }
}

#[cfg(test)]
mod tests {
    use crate::authority_committee::AuthorityIndex;
    use crate::crypto::address::Address;
    use crate::crypto::keys::{AuthoritySignature, ProtocolKeySignature};
    use crate::digest::Digest;
    use crate::scope::{Scope, ScopedMessage};
    use crate::transaction::{
        ShardTransaction, TransactionData, TransactionDataV1, TransactionExpiration,
        TransactionKind,
    };

    use super::*;

    const STARTING_PORT: u16 = 8000;

    fn mock_tx() -> SignedTransaction {
        let sig = ProtocolKeySignature::from_bytes(&[1u8; 64]).unwrap();
        let tx = ShardTransaction::new(Digest::new_from_bytes(b"test"), 100);
        let tx_kind = TransactionKind::ShardTransaction(tx);
        SignedTransaction {
            scoped_message: ScopedMessage::new(
                Scope::TransactionData,
                TransactionData::new_v1(tx_kind, Address::default(), TransactionExpiration::None),
            ),
            tx_signatures: vec![sig],
        }
    }

    fn diff_tx() -> SignedTransaction {
        let sig = ProtocolKeySignature::from_bytes(&[2u8; 64]).unwrap();
        let tx = ShardTransaction::new(Digest::new_from_bytes(b"test"), 100);
        let tx_kind = TransactionKind::ShardTransaction(tx);
        SignedTransaction {
            scoped_message: ScopedMessage::new(
                Scope::TransactionData,
                TransactionData::new_v1(tx_kind, Address::default(), TransactionExpiration::None),
            ),
            tx_signatures: vec![sig],
        }
    }
    #[test]
    fn test_finality_proof_successful_verification() {
        // Create committee with 4 authorities, each with stake 1
        let stakes = vec![1u64; 4];
        let (committee, keypairs) =
            AuthorityCommittee::local_test_committee(0, stakes, STARTING_PORT);

        // Create a test claim
        let claim = BlockClaim {
            epoch: 0,
            block_ref: BlockRef::default(),
            transaction: mock_tx(),
        };

        // Sign with 3 out of 4 authorities (meeting 2f+1 threshold)
        let message = bcs::to_bytes(&claim).unwrap();
        let signatures: Vec<AuthoritySignature> = keypairs[..3]
            .iter()
            .map(|(_, _, authority_kp)| authority_kp.sign(&message))
            .collect();

        // Create authority bitset for the signing authorities
        let authorities = AuthorityBitSet::new(
            &(0..3)
                .map(|i| AuthorityIndex::new_for_test(i))
                .collect::<Vec<_>>(),
        );

        // Create and verify the proof
        let proof = FinalityProof {
            claim,
            authorities,
            signature: AuthorityAggregateSignature::new(&signatures).unwrap(),
        };

        assert!(proof.verify(&committee).is_ok());
    }

    #[test]
    fn test_finality_proof_wrong_bitset() {
        // Create committee with 4 authorities, each with stake 1
        let stakes = vec![1u64; 4];
        let (committee, keypairs) =
            AuthorityCommittee::local_test_committee(0, stakes, STARTING_PORT);

        // Create a test claim
        let claim = BlockClaim {
            epoch: 0,
            block_ref: BlockRef::default(),
            transaction: mock_tx(),
        };

        // Sign with 3 out of 4 authorities (meeting 2f+1 threshold)
        let message = bcs::to_bytes(&claim).unwrap();
        let signatures: Vec<AuthoritySignature> = keypairs[..3]
            .iter()
            .map(|(_, _, authority_kp)| authority_kp.sign(&message))
            .collect();

        // Create authority bitset for the signing authorities
        let authorities = AuthorityBitSet::new(
            &(0..4)
                .map(|i| AuthorityIndex::new_for_test(i))
                .collect::<Vec<_>>(),
        );

        // Create and verify the proof
        let proof = FinalityProof {
            claim,
            authorities,
            signature: AuthorityAggregateSignature::new(&signatures).unwrap(),
        };

        assert!(proof.verify(&committee).is_err());
    }
    #[test]
    fn test_finality_proof_insufficient_quorum() {
        // Create committee with 4 authorities, each with stake 1
        let stakes = vec![1u64; 4];
        let (committee, keypairs) =
            AuthorityCommittee::local_test_committee(0, stakes, STARTING_PORT);

        let claim = BlockClaim {
            epoch: 0,
            block_ref: BlockRef::default(),
            transaction: mock_tx(),
        };

        // Sign with only 2 out of 4 authorities (not meeting 2f+1 threshold)
        let message = bcs::to_bytes(&claim).unwrap();
        let signatures: Vec<AuthoritySignature> = keypairs[..2]
            .iter()
            .map(|(_, _, authority_kp)| authority_kp.sign(&message))
            .collect();

        let authorities = AuthorityBitSet::new(
            &(0..2)
                .map(|i| AuthorityIndex::new_for_test(i))
                .collect::<Vec<_>>(),
        );

        let proof = FinalityProof {
            claim,
            authorities,
            signature: AuthorityAggregateSignature::new(&signatures).unwrap(),
        };

        assert_eq!(proof.verify(&committee).is_err(), true);
    }

    #[test]
    fn test_finality_proof_wrong_epoch() {
        // Create committee with 4 authorities, each with stake 1
        let stakes = vec![1u64; 4];
        let (committee, keypairs) =
            AuthorityCommittee::local_test_committee(0, stakes, STARTING_PORT);

        let claim = BlockClaim {
            epoch: 1, // Wrong epoch
            block_ref: BlockRef::default(),
            transaction: mock_tx(),
        };

        let message = bcs::to_bytes(&claim).unwrap();
        let signatures: Vec<AuthoritySignature> = keypairs[..3]
            .iter()
            .map(|(_, _, authority_kp)| authority_kp.sign(&message))
            .collect();

        let authorities = AuthorityBitSet::new(
            &(0..3)
                .map(|i| AuthorityIndex::new_for_test(i))
                .collect::<Vec<_>>(),
        );

        let proof = FinalityProof {
            claim,
            authorities,
            signature: AuthorityAggregateSignature::new(&signatures).unwrap(),
        };

        // This should fail signature verification since the epoch doesn't match
        assert!(proof.verify(&committee).is_err());
    }

    #[test]
    fn test_finality_proof_tampered_transaction() {
        // Create committee with 4 authorities, each with stake 1
        let stakes = vec![1u64; 4];
        let (committee, keypairs) =
            AuthorityCommittee::local_test_committee(0, stakes, STARTING_PORT);

        let original_claim = BlockClaim {
            epoch: 0,
            block_ref: BlockRef::default(),
            transaction: mock_tx(),
        };

        // Sign the original claim
        let message = bcs::to_bytes(&original_claim).unwrap();
        let signatures: Vec<AuthoritySignature> = keypairs[..3]
            .iter()
            .map(|(_, _, authority_kp)| authority_kp.sign(&message))
            .collect();

        // Create a proof with a modified transaction
        let tampered_claim = BlockClaim {
            epoch: 0,
            block_ref: BlockRef::default(),
            transaction: diff_tx(), // Different transaction
        };

        let authorities = AuthorityBitSet::new(
            &(0..3)
                .map(|i| AuthorityIndex::new_for_test(i))
                .collect::<Vec<_>>(),
        );

        let proof = FinalityProof {
            claim: tampered_claim,
            authorities,
            signature: AuthorityAggregateSignature::new(&signatures).unwrap(),
        };

        // Should fail signature verification
        assert!(matches!(
            proof.verify(&committee),
            Err(SharedError::MalformedSignature(_))
        ));
    }
}
