use crate::{
    core::pipeline_dispatcher::InternalDispatcher,
    messaging::EncoderInternalNetworkService,
    types::{
        commit::{verify_commit, Commit, CommitAPI},
        commit_votes::{verify_commit_votes, CommitVotes, CommitVotesAPI},
        context::Context,
        report_vote::{verify_report_vote, ReportVote, ReportVoteAPI},
        reveal::{verify_reveal, Reveal, RevealAPI},
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::error;
use types::{
    error::{ShardError, ShardResult},
    shard::Shard,
    shard_crypto::keys::EncoderPublicKey,
    shard_crypto::verified::Verified,
};
use types::{shard::ShardAuthToken, shard_verifier::ShardVerifier};

pub(crate) struct EncoderInternalService<D: InternalDispatcher> {
    context: Context,
    dispatcher: D,
    shard_verifier: Arc<ShardVerifier>,
}

impl<D: InternalDispatcher> EncoderInternalService<D> {
    pub(crate) fn new(context: Context, dispatcher: D, shard_verifier: Arc<ShardVerifier>) -> Self {
        Self {
            context,
            dispatcher,
            shard_verifier,
        }
    }

    fn shard_verification(
        &self,
        auth_token: &ShardAuthToken,
        peer: &EncoderPublicKey,
    ) -> ShardResult<(Shard, CancellationToken)> {
        // All internal shard requests
        // - allowed communication (matches tls key associated with valid encoder - handled in tonic service)
        // - shard auth token is valid
        // - peer is in shard
        // - own key is in shard
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
            let commit: Commit =
                bcs::from_bytes(&commit_bytes).map_err(ShardError::MalformedType)?;
            let (shard, cancellation) = self.shard_verification(commit.auth_token(), peer)?;

            // checks that message author matches peer inside type verification function
            let verified_commit = Verified::new(commit.clone(), commit_bytes, |signed_commit| {
                verify_commit(signed_commit, peer, &shard)
            })
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

            // dispatcher handles repeated/conflicting messages from peers
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
            let commit_votes: CommitVotes =
                bcs::from_bytes(&commit_votes_bytes).map_err(ShardError::MalformedType)?;

            let (shard, cancellation) = self.shard_verification(commit_votes.auth_token(), peer)?;

            // checks that message author matches peer inside type verification function
            let verified_commit_votes = Verified::new(commit_votes, commit_votes_bytes, |votes| {
                verify_commit_votes(votes, peer, &shard)
            })
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

            // dispatcher handles repeated/conflicting messages from peers
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
            let reveal: Reveal =
                bcs::from_bytes(&reveal_bytes).map_err(ShardError::MalformedType)?;

            let (shard, cancellation) = self.shard_verification(reveal.auth_token(), peer)?;
            // TODO: should make this more efficient without needing to clone the encoder committee
            let encoder_committee = self
                .context
                .inner()
                .committees(shard.epoch())?
                .encoder_committee
                .clone();

            // checks that message author matches peer inside type verification function
            let verified_reveal = Verified::new(reveal, reveal_bytes, |reveal| {
                verify_reveal(reveal, peer, &shard, &encoder_committee)
            })
            .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

            // dispatcher handles repeated/conflicting messages from peers
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

    async fn handle_send_report_vote(
        &self,
        peer: &EncoderPublicKey,
        report_vote_bytes: Bytes,
    ) -> ShardResult<()> {
        let result: ShardResult<()> = {
            let report_vote: ReportVote =
                bcs::from_bytes(&report_vote_bytes).map_err(ShardError::MalformedType)?;

            let (shard, cancellation) = self.shard_verification(report_vote.auth_token(), peer)?;
            // TODO: should make this more efficient without needing to clone the encoder committee
            let encoder_committee = self
                .context
                .inner()
                .committees(shard.epoch())?
                .encoder_committee
                .clone();

            // checks that message author matches peer inside type verification function
            let verified_report_vote =
                Verified::new(report_vote, report_vote_bytes, |report_vote| {
                    verify_report_vote(&report_vote, peer, &shard, &encoder_committee)
                })
                .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

            // dispatcher handles repeated/conflicting messages from peers
            self.dispatcher
                .dispatch_report_vote(shard, verified_report_vote, cancellation)
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
