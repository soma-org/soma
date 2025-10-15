use serde::{Deserialize, Serialize};

use crate::{
    committee::{
        AuthorityIndex, Committee, CommitteeWithNetworkMetadata, EpochId, NetworkingCommittee,
    },
    consensus::{
        block::{Block, BlockDigest, BlockRef, SignedBlock, VerifiedBlock, GENESIS_ROUND},
        commit::{CommitDigest, CommitRef, CommittedSubDag},
        ConsensusTransaction,
    },
    effects::{self, TransactionEffects},
    encoder_committee::EncoderCommittee,
    error::SomaResult,
    object::{Object, ObjectID},
    system_state::{get_system_state, SystemState, SystemStateTrait},
    transaction::{CertifiedTransaction, Transaction},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genesis {
    transaction: CertifiedTransaction,
    effects: TransactionEffects,
    objects: Vec<Object>,
}

impl Genesis {
    pub fn new_with_certified_tx(
        transaction: CertifiedTransaction,
        effects: TransactionEffects,
        objects: Vec<Object>,
    ) -> Self {
        Self {
            transaction,
            effects,
            objects,
        }
    }

    pub fn encoder_committee(&self) -> EncoderCommittee {
        self.system_object().get_current_epoch_encoder_committee()
    }

    pub fn networking_committee(&self) -> NetworkingCommittee {
        self.system_object()
            .get_current_epoch_networking_committee()
    }

    pub fn committee_with_network(&self) -> CommitteeWithNetworkMetadata {
        self.system_object().get_current_epoch_committee()
    }

    pub fn committee(&self) -> SomaResult<Committee> {
        Ok(self.committee_with_network().committee().clone())
    }

    pub fn transaction(&self) -> Transaction {
        self.transaction.clone().into_unsigned()
    }

    pub fn effects(&self) -> &TransactionEffects {
        &self.effects
    }

    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    pub fn object(&self, id: ObjectID) -> Option<Object> {
        self.objects.iter().find(|o| o.id() == id).cloned()
    }

    pub fn epoch(&self) -> EpochId {
        0
    }

    pub fn commit(&self) -> CommittedSubDag {
        // Create a genesis block that contains the genesis transaction
        let genesis_block = self.create_genesis_block();

        CommittedSubDag::new(
            genesis_block.reference(), // Use the genesis block's reference as leader
            vec![genesis_block],       // Include the genesis block in blocks
            0,                         // Genesis timestamp
            CommitRef::new(0, CommitDigest::default()),
            CommitDigest::MIN,
        )
    }

    /// Creates a synthetic genesis block containing the genesis transaction and objects
    fn create_genesis_block(&self) -> VerifiedBlock {
        let consensus_tx = ConsensusTransaction::new_certificate_message(self.transaction.clone());

        // Serialize the consensus transaction
        let tx_bytes =
            bcs::to_bytes(&consensus_tx).expect("Serializing consensus transaction cannot fail");

        // Create a Transaction for the block (wrapper around bytes)
        let block_tx = crate::consensus::block::Transaction::new(tx_bytes);

        // Create the genesis block
        let block = Block::new(
            0,                 // epoch
            GENESIS_ROUND,     // round 0
            AuthorityIndex(0), // Use authority 0 as genesis author
            0,                 // timestamp_ms
            vec![],            // No ancestors for genesis
            vec![block_tx],    // Single transaction containing the consensus tx
            vec![],            // No commit votes for genesis
            None,              // No end of epoch data for genesis
        );

        // Create a signed block without actual signature (genesis doesn't need signature)
        let signed_block = SignedBlock::new_genesis(block);

        // Serialize and create verified block
        let serialized = signed_block
            .serialize()
            .expect("Genesis block serialization should not fail");

        VerifiedBlock::new_verified(signed_block, serialized)
    }

    pub fn system_object(&self) -> SystemState {
        get_system_state(&self.objects()).expect("System State object must always exist")
    }
}
