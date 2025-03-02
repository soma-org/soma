use std::sync::Arc;

use crate::{
    actors::{ActorMessage, Processor},
    storage::datastore::Store,
    types::shard_votes::{CommitRound, ShardVotes},
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{signed::Signed, verified::Verified};

pub(crate) struct CommitVotesProcessor {
    store: Arc<dyn Store>,
}

impl CommitVotesProcessor {
    pub(crate) fn new(store: Arc<dyn Store>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Processor for CommitVotesProcessor {
    type Input = Verified<Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature>>;
    type Output = ();

    async fn process(&self, _msg: ActorMessage<Self>) {
        // write to the store
        ()
    }

    fn shutdown(&mut self) {}
}
