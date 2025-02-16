use serde::{Deserialize, Serialize};

use crate::{
    committee::{AuthorityIndex, Committee, CommitteeWithNetworkMetadata, EpochId},
    consensus::{
        block::{BlockDigest, BlockRef},
        commit::{CommitDigest, CommitRef, CommittedSubDag},
    },
    effects::{self, TransactionEffects},
    error::SomaResult,
    object::{Object, ObjectID},
    system_state::{get_system_state, SystemState, SystemStateTrait},
    transaction::Transaction,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genesis {
    transaction: Transaction,
    effects: TransactionEffects,
    objects: Vec<Object>,
}

impl Genesis {
    pub fn new(
        transaction: Transaction,
        effects: TransactionEffects,
        objects: Vec<Object>,
    ) -> Self {
        Self {
            transaction,
            effects,
            objects,
        }
    }

    pub fn committee_with_network(&self) -> CommitteeWithNetworkMetadata {
        self.system_object().get_current_epoch_committee()
    }

    pub fn committee(&self) -> SomaResult<Committee> {
        Ok(self.committee_with_network().committee().clone())
    }

    pub fn transaction(&self) -> &Transaction {
        &self.transaction
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
        CommittedSubDag::new(
            BlockRef::new(0, AuthorityIndex(0), BlockDigest::default(), 0),
            vec![],
            0,
            CommitRef::new(0, CommitDigest::default()),
            CommitDigest::MIN,
        )
    }

    pub fn system_object(&self) -> SystemState {
        get_system_state(&self.objects()).expect("System State object must always exist")
    }
}
