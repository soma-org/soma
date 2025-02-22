use crate::{
    actors::{ActorMessage, Processor},
    types::shard_reveal::ShardReveal,
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{signed::Signed, verified::Verified};

pub(crate) struct RevealProcessor {}

impl RevealProcessor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl Processor for RevealProcessor {
    type Input = Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>;
    type Output = ();

    async fn process(&self, _msg: ActorMessage<Self>) {
        ()
    }

    fn shutdown(&mut self) {}
}
