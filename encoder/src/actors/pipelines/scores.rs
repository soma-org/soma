use std::{ops::Deref, sync::Arc};

use crate::{
    actors::{ActorMessage, Processor},
    error::{ShardError, ShardResult},
    storage::datastore::Store,
    types::{encoder_committee::EncoderIndex, shard::Shard, shard_scores::ShardScores},
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{digest::Digest, signed::Signed, verified::Verified};

pub(crate) struct ScoresProcessor {
    store: Arc<dyn Store>,
    own_index: EncoderIndex,
}

impl ScoresProcessor {
    pub(crate) fn new(store: Arc<dyn Store>, own_index: EncoderIndex) -> Self {
        Self { store, own_index }
    }
}

#[async_trait]
impl Processor for ScoresProcessor {
    type Input = (
        Shard,
        Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, votes) = msg.input;
            let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
            let epoch = shard.epoch();

            // TODO
            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
