use serde::{Deserialize, Serialize};

use crate::{
    authority_committee::{AuthorityBitSet, AuthorityCommittee, AuthorityIndex},
    block::{BlockHeader, BlockHeaderAPI, BlockHeaderRef, SignedBlockHeader, VerifiedBlockHeader},
    crypto::keys::{AuthorityAggregateSignature, AuthorityPublicKey},
    digest::Digest,
    error::SharedResult,
    transaction::SignedTransaction,
};

pub struct FinalityProof {
    verified_block_header: VerifiedBlockHeader,
    signed_transaction: SignedTransaction,
    // First round
    first_round_digests: Vec<[u8; 32]>,
    first_round_authorities: AuthorityBitSet,
    first_round_agg_sig: AuthorityAggregateSignature,
    // Second round, one BitSet per first round digest
    second_round_authorities: Vec<AuthorityBitSet>,
    second_round_agg_sigs: Vec<AuthorityAggregateSignature>,
}

impl FinalityProof {
    pub fn verify(&self, authority_committee: AuthorityCommittee) -> SharedResult<()> {
        let round = self.verified_block_header.round();
        let author = self.verified_block_header.author();
        let digest = self.verified_block_header.digest();
        let bref = BlockHeaderRef::new(round, author, digest);

        let authorities = self.first_round_authorities.get_indices();
        let pks: Vec<AuthorityPublicKey> = authorities
            .iter()
            .map(|authority| {
                authority_committee
                    .authority(*authority)
                    .authority_key
                    .clone()
            })
            .collect();

        // self.first_round_agg_sig.verify(&pks, message);

        Ok(())
    }
}
