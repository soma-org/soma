use crate::{
    actors::{ActorMessage, Processor},
    types::{certified::Certified, shard_commit::ShardCommit},
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{signed::Signed, verified::Verified};

pub(crate) struct CertifiedCommitProcessor {}

impl CertifiedCommitProcessor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl Processor for CertifiedCommitProcessor {
    type Input = Verified<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>;
    type Output = ();

    async fn process(&self, _msg: ActorMessage<Self>) {
        ()
    }

    fn shutdown(&mut self) {}
}
