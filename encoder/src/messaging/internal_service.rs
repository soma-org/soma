use crate::{
    core::pipeline_dispatcher::InternalDispatcher,
    datastore::Store,
    messaging::EncoderInternalNetworkService,
    types::{
        commit::{verify_signed_commit, Commit, CommitAPI},
        commit_votes::{verify_commit_votes, CommitVotes, CommitVotesAPI},
        context::Context,
        reveal::{verify_signed_reveal, Reveal, RevealAPI},
        score_vote::{verify_signed_score_vote, ScoreVote, ScoreVoteAPI},
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::EncoderPublicKey,
    error::{ShardError, ShardResult},
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::error;
use types::{shard::ShardAuthToken, shard_verifier::ShardVerifier};

pub(crate) struct EncoderInternalService<D: InternalDispatcher> {
    context: Context,
    store: Arc<dyn Store>,
    dispatcher: D,
    shard_verifier: Arc<ShardVerifier>,
}

impl<D: InternalDispatcher> EncoderInternalService<D> {
    pub(crate) fn new(
        context: Context,
        store: Arc<dyn Store>,
        dispatcher: D,
        shard_verifier: Arc<ShardVerifier>,
    ) -> Self {
        Self {
            context,
            store,
            dispatcher,
            shard_verifier,
        }
    }

    fn shard_verification(
        &self,
        auth_token: &ShardAuthToken,
        peer: &EncoderPublicKey,
    ) -> ShardResult<(Shard, CancellationToken)> {
        let inner_context = self.context.inner();
        let committees = inner_context.committees(auth_token.epoch())?;

        let (shard, cancellation) = self.shard_verifier.verify(
            committees.authority_committee.clone(),
            committees.encoder_committee.clone(),
            committees.vdf_iterations,
            &auth_token,
        )?;

        if !shard.contains(peer) {
            return Err(ShardError::UnauthorizedPeer);
        }
        if !shard.contains(&self.context.own_encoder_key()) {
            return Err(ShardError::UnauthorizedPeer);
        }

        Ok((shard, cancellation))
    }
}

#[async_trait]
impl<D: InternalDispatcher> EncoderInternalNetworkService for EncoderInternalService<D> {
    async fn handle_send_commit(
        &self,
        peer: &EncoderPublicKey,
        commit_bytes: Bytes,
    ) -> ShardResult<()> {
        let result: ShardResult<()> = {
            let signed_commit: Signed<Commit, min_sig::BLS12381Signature> =
                bcs::from_bytes(&commit_bytes).map_err(ShardError::MalformedType)?;
            let (shard, cancellation) =
                self.shard_verification(signed_commit.auth_token(), peer)?;
            let verified_commit =
                Verified::new(signed_commit.clone(), commit_bytes, |signed_commit| {
                    verify_signed_commit(signed_commit, peer, &shard)
                })
                .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

            self.dispatcher
                .dispatch_commit(shard, verified_commit, cancellation)
                .await
        };

        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                error!("{}", e.to_string());
                Err(e)
            }
        }
    }

    async fn handle_send_commit_votes(
        &self,
        peer: &EncoderPublicKey,
        commit_votes_bytes: Bytes,
    ) -> ShardResult<()> {
        let result: ShardResult<()> = {
            let commit_votes: Signed<CommitVotes, min_sig::BLS12381Signature> =
                bcs::from_bytes(&commit_votes_bytes).map_err(ShardError::MalformedType)?;

            let (shard, cancellation) = self.shard_verification(commit_votes.auth_token(), peer)?;

            let verified_commit_votes = Verified::new(commit_votes, commit_votes_bytes, |votes| {
                verify_commit_votes(votes, peer, &shard)
            })
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

            self.dispatcher
                .dispatch_commit_votes(shard, verified_commit_votes, cancellation)
                .await
        };
        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                error!("{}", e.to_string());
                Err(e)
            }
        }
    }

    async fn handle_send_reveal(
        &self,
        peer: &EncoderPublicKey,
        reveal_bytes: Bytes,
    ) -> ShardResult<()> {
        let result: ShardResult<()> = {
            let reveal: Signed<Reveal, min_sig::BLS12381Signature> =
                bcs::from_bytes(&reveal_bytes).map_err(ShardError::MalformedType)?;

            let (shard, cancellation) = self.shard_verification(reveal.auth_token(), peer)?;

            let verified_reveal = Verified::new(reveal, reveal_bytes, |reveal| {
                verify_signed_reveal(reveal, peer, &shard)
            })
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

            self.dispatcher
                .dispatch_reveal(shard, verified_reveal, cancellation)
                .await
        };
        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                error!("{}", e.to_string());
                Err(e)
            }
        }
    }

    async fn handle_send_score_vote(
        &self,
        peer: &EncoderPublicKey,
        score_vote_bytes: Bytes,
    ) -> ShardResult<()> {
        let result: ShardResult<()> = {
            let score_vote: Signed<ScoreVote, min_sig::BLS12381Signature> =
                bcs::from_bytes(&score_vote_bytes).map_err(ShardError::MalformedType)?;

            let (shard, cancellation) = self.shard_verification(score_vote.auth_token(), peer)?;

            let verified_score_vote = Verified::new(score_vote, score_vote_bytes, |score_vote| {
                verify_signed_score_vote(&score_vote, peer, &shard)
            })
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

            self.dispatcher
                .dispatch_score_vote(shard, verified_score_vote, cancellation)
                .await
        };
        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                error!("{}", e.to_string());
                Err(e)
            }
        }
    }
}
