use std::sync::Arc;

use crate::{
    actors::{ActorMessage, Processor},
    error::{ShardError, ShardResult},
    storage::datastore::Store,
    types::{
        certified::Certified,
        encoder_committee::EncoderIndex,
        encoder_context::EncoderContext,
        shard::Shard,
        shard_scores::{ShardScores, ShardScoresAPI},
    },
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::{EncoderAggregateSignature, EncoderSignature},
    digest::Digest,
    signed::Signed,
    verified::Verified,
};

pub(crate) struct ScoresProcessor {
    context: Arc<EncoderContext>,
    store: Arc<dyn Store>,
    own_index: EncoderIndex,
}

impl ScoresProcessor {
    pub(crate) fn new(
        context: Arc<EncoderContext>,
        store: Arc<dyn Store>,
        own_index: EncoderIndex,
    ) -> Self {
        Self {
            context,
            store,
            own_index,
        }
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
            let (shard, shard_scores) = msg.input;
            let epoch = shard.epoch();
            let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
            let signed_score_set = shard_scores.signed_score_set();

            let evaluator = shard_scores.evaluator();

            let matching_scores =
                self.store
                    .add_scores(epoch, shard_ref, evaluator, signed_score_set.clone())?;

            if matching_scores.len() >= shard.evaluation_quorum_threshold() as usize {
                let (signatures, evaluator_indices): (Vec<EncoderSignature>, Vec<EncoderIndex>) = {
                    let mut sigs = Vec::new();
                    let mut indices = Vec::new();

                    for (evaluator_index, signed_scores) in matching_scores.iter() {
                        let sig = EncoderSignature::from_bytes(&signed_scores.raw_signature())
                            .map_err(ShardError::SignatureAggregationFailure)?;
                        sigs.push(sig);
                        indices.push(*evaluator_index);
                    }
                    (sigs, indices)
                };

                let agg = EncoderAggregateSignature::new(&signatures)
                    .map_err(ShardError::SignatureAggregationFailure)?;

                let cert = Certified::new_v1(signed_score_set.into_inner(), evaluator_indices, agg);
                println!("{:?}", cert);
            }
            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
