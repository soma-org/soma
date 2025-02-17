use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::ProtocolKeyPair,
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
        shard_commit::ShardCommit,
        shard_reveal::ShardReveal,
        shard_votes::{CommitRound, RevealRound, ShardVotes},
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

#[async_trait]
impl<PD: PipelineDispatcher> EncoderInternalNetworkService for EncoderInternalService<PD> {
    async fn handle_send_commit(
        &self,
        peer: EncoderIndex,
        commit_bytes: Bytes,
    ) -> ShardResult<
        Serialized<
            Signature<Signed<ShardCommit, min_sig::BLS12381Signature>, min_sig::BLS12381Signature>,
        >,
    > {
        // convert into correct type
        let signed_commit: Signed<ShardCommit, min_sig::BLS12381Signature> =
            bcs::from_bytes(&commit_bytes).map_err(ShardError::MalformedType)?;
        // perform verification on type and auth including signature checks
        let verified_commit = Verified::new(signed_commit, commit_bytes, |c| Ok(()))
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        // check store for conflicts, handle accordingly
        // issue signature if there are no conflicts
        unimplemented!()
    }
    async fn handle_send_certified_commit(
        &self,
        peer: EncoderIndex,
        certified_commit_bytes: Bytes,
    ) -> ShardResult<()> {
        // convert into correct type
        let certified_commit: Certified<Signed<ShardCommit, min_sig::BLS12381Signature>> =
            bcs::from_bytes(&certified_commit_bytes).map_err(ShardError::MalformedType)?;
        // perform verification on type and auth including signature checks
        let verified_commit = Verified::new(certified_commit, certified_commit_bytes, |c| Ok(()))
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        // send to orchestrator
        unimplemented!()
    }
    async fn handle_send_commit_votes(
        &self,
        peer: EncoderIndex,
        votes_bytes: Bytes,
    ) -> ShardResult<()> {
        // convert into correct type
        let votes: Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature> =
            bcs::from_bytes(&votes_bytes).map_err(ShardError::MalformedType)?;
        // perform verification on type and auth including signature checks
        let verified_commit = Verified::new(votes, votes_bytes, |v| Ok(()))
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        // send to orchestrator
        unimplemented!()
    }
    async fn handle_send_reveal(&self, peer: EncoderIndex, reveal_bytes: Bytes) -> ShardResult<()> {
        // convert into correct type
        let reveal: Signed<ShardReveal, min_sig::BLS12381Signature> =
            bcs::from_bytes(&reveal_bytes).map_err(ShardError::MalformedType)?;
        // perform verification on type and auth including signature checks
        let verified_commit = Verified::new(reveal, reveal_bytes, |r| Ok(()))
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        // send to orchestrator
        unimplemented!()
    }
    async fn handle_send_reveal_votes(
        &self,
        peer: EncoderIndex,
        votes_bytes: Bytes,
    ) -> ShardResult<()> {
        // convert into correct type
        let votes: Signed<ShardVotes<RevealRound>, min_sig::BLS12381Signature> =
            bcs::from_bytes(&votes_bytes).map_err(ShardError::MalformedType)?;
        // perform verification on type and auth including signature checks
        let verified_commit = Verified::new(votes, votes_bytes, |v| Ok(()))
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        // send to orchestrator
        unimplemented!()
    }
}
