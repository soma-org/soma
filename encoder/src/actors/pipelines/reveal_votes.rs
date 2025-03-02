use std::sync::Arc;

use crate::{
    actors::{ActorMessage, Processor},
    storage::datastore::Store,
    types::shard_votes::{RevealRound, ShardVotes},
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{signed::Signed, verified::Verified};

pub(crate) struct RevealVotesProcessor {
    store: Arc<dyn Store>,
}

impl RevealVotesProcessor {
    pub(crate) fn new(store: Arc<dyn Store>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Processor for RevealVotesProcessor {
    type Input = Verified<Signed<ShardVotes<RevealRound>, min_sig::BLS12381Signature>>;
    type Output = ();

    async fn process(&self, _msg: ActorMessage<Self>) {
        //gets shard, and votes
        // for each accept vote for a slot: add voter to hashset
        // for each reject vote for a slot: add voter to hashset
        // once there is f+1 rejects: finalize the slot as rejected
        // once there is 2f+1 accepts: finalize the slot as accepted f
        ()
    }

    fn shutdown(&mut self) {}
}
