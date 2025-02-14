use async_trait::async_trait;
use bytes::Bytes;
use shared::{
    crypto::keys::ProtocolKeyPair,
    scope::Scope,
    serialized::Serialized,
    signed::{Signature, Signed},
    verified::Verified,
};
use std::sync::Arc;

use crate::{
    error::{ShardError, ShardResult},
    networking::messaging::EncoderInternalNetworkService,
    storage::datastore::Store,
    types::{
        certified::Certified,
        encoder_committee::EncoderIndex,
        encoder_context::EncoderContext,
        shard::ShardRef,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_input::ShardInput,
        shard_reveal::{ShardReveal, ShardRevealAPI},
        shard_slots::{ShardSlots, ShardSlotsAPI},
    },
};

use super::pipeline_dispatcher::PipelineDispatcher;

pub(crate) struct EncoderInternalService<PD: PipelineDispatcher> {
    context: Arc<EncoderContext>,
    pipeline_dispatcher: Arc<PD>, //TODO: confirm this needs an arc?
    store: Arc<dyn Store>,
    protocol_keypair: Arc<ProtocolKeyPair>,
}

impl<PD: PipelineDispatcher> EncoderInternalService<PD> {
    pub(crate) fn new(
        context: Arc<EncoderContext>,
        pipeline_dispatcher: Arc<PD>,
        store: Arc<dyn Store>,
        protocol_keypair: Arc<ProtocolKeyPair>,
    ) -> Self {
        println!("configured core thread");
        Self {
            context,
            pipeline_dispatcher,
            store,
            protocol_keypair,
        }
    }
}

fn unverified<T>(input: &T) -> ShardResult<()> {
    Ok(())
}

#[async_trait]
impl<PD: PipelineDispatcher> EncoderInternalNetworkService for EncoderInternalService<PD> {
    async fn handle_send_commit(
        &self,
        peer: EncoderIndex,
        commit: Bytes,
    ) -> ShardResult<Serialized<Signature<Signed<ShardCommit>>>> {
        unimplemented!()
    }
    async fn handle_send_certified_commit(
        &self,
        peer: EncoderIndex,
        certified_commit: Bytes,
    ) -> ShardResult<()> {
        unimplemented!()
    }
    async fn handle_send_commit_votes(&self, peer: EncoderIndex, votes: Bytes) -> ShardResult<()> {
        unimplemented!()
    }
    async fn handle_send_reveal(&self, peer: EncoderIndex, reveal: Bytes) -> ShardResult<()> {
        unimplemented!()
    }
    async fn handle_send_reveal_votes(&self, peer: EncoderIndex, votes: Bytes) -> ShardResult<()> {
        unimplemented!()
    }
}
