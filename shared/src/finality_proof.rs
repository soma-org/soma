use serde::{Deserialize, Serialize};

use crate::{
    authority_committee::{AuthorityBitSet, AuthorityCommittee, AuthorityIndex},
    block::{BlockHeader, BlockHeaderAPI, BlockRef},
    crypto::keys::{AuthorityAggregateSignature, AuthorityPublicKey},
    digest::Digest,
    error::SharedResult,
    signed::Signed,
    transaction::SignedTransaction,
    verified::Verified,
};

pub struct FinalityProof {
    signed_block_header: Verified<Signed<BlockHeader>>,
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
        let epoch = self.signed_block_header.epoch();
        let round = self.signed_block_header.round();
        let author = self.signed_block_header.author();
        let digest = self.signed_block_header.digest();
        let bref = BlockRef::new(round, author, digest);

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
