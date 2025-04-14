use crate::{
    core::pipeline_dispatcher::InternalDispatcher,
    error::{ShardError, ShardResult},
    messaging::EncoderInternalNetworkService,
    types::{
        context::Context,
        shard_commit::{verify_signed_shard_commit, ShardCommit, ShardCommitAPI},
        shard_commit_votes::{verify_shard_commit_votes, ShardCommitVotes, ShardCommitVotesAPI},
        shard_reveal::{verify_signed_shard_reveal, ShardReveal, ShardRevealAPI},
        shard_reveal_votes::{verify_shard_reveal_votes, ShardRevealVotes, ShardRevealVotesAPI},
        shard_scores::{verify_signed_scores, ShardScores, ShardScoresAPI},
        shard_verifier::ShardVerifier,
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{crypto::keys::EncoderPublicKey, signed::Signed, verified::Verified};
use std::ops::Deref;

pub(crate) struct EncoderInternalService<D: InternalDispatcher> {
    context: Context,
    dispatcher: D,
    shard_verifier: ShardVerifier,
}

impl<D: InternalDispatcher> EncoderInternalService<D> {
    pub(crate) fn new(context: Context, dispatcher: D, shard_verifier: ShardVerifier) -> Self {
        Self {
            context,
            dispatcher,
            shard_verifier,
        }
    }
}

#[async_trait]
impl<D: InternalDispatcher> EncoderInternalNetworkService for EncoderInternalService<D> {
    async fn handle_send_commit(
        &self,
        peer: &EncoderPublicKey,
        commit_bytes: Bytes,
    ) -> ShardResult<()> {
        let signed_commit: Signed<ShardCommit, min_sig::BLS12381Signature> =
            bcs::from_bytes(&commit_bytes).map_err(ShardError::MalformedType)?;
        if peer != signed_commit.committer() {
            return Err(ShardError::FailedTypeVerification(
                "sender must be committer".to_string(),
            ));
        }
        let shard = self
            .shard_verifier
            .verify(&self.context, signed_commit.auth_token())
            .await?;

        let verified_commit = Verified::new(signed_commit.clone(), commit_bytes, |signed_commit| {
            verify_signed_shard_commit(&signed_commit, &shard)?;
            Ok(())
        })
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

        let _ = self
            .dispatcher
            .dispatch_commit(shard, verified_commit)
            .await?;

        Ok(())
    }
    async fn handle_send_commit_votes(
        &self,
        peer: &EncoderPublicKey,
        votes_bytes: Bytes,
    ) -> ShardResult<()> {
        let votes: Signed<ShardCommitVotes, min_sig::BLS12381Signature> =
            bcs::from_bytes(&votes_bytes).map_err(ShardError::MalformedType)?;
        if peer != votes.voter().deref() {
            return Err(ShardError::FailedTypeVerification(
                "sender must be voter".to_string(),
            ));
        }
        let shard = self
            .shard_verifier
            .verify(&self.context, votes.auth_token())
            .await?;

        let verified_commit_votes = Verified::new(votes, votes_bytes, |votes| {
            verify_shard_commit_votes(votes, &shard)
        })
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        let _ = self
            .dispatcher
            .dispatch_commit_votes(shard, verified_commit_votes)
            .await?;

        Ok(())
    }
    async fn handle_send_reveal(
        &self,
        peer: &EncoderPublicKey,
        reveal_bytes: Bytes,
    ) -> ShardResult<()> {
        let reveal: Signed<ShardReveal, min_sig::BLS12381Signature> =
            bcs::from_bytes(&reveal_bytes).map_err(ShardError::MalformedType)?;
        if peer != reveal.inference_encoder().deref() {
            return Err(ShardError::FailedTypeVerification(
                "sender must be inference encoder for reveal".to_string(),
            ));
        }
        let shard = self
            .shard_verifier
            .verify(&self.context, reveal.auth_token())
            .await?;

        let verified_reveal = Verified::new(reveal.clone(), reveal_bytes, |reveal| {
            verify_signed_shard_reveal(reveal, &shard)
        })
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;

        let _ = self
            .dispatcher
            .dispatch_reveal(shard, verified_reveal)
            .await?;
        Ok(())
    }
    async fn handle_send_reveal_votes(
        &self,
        peer: &EncoderPublicKey,
        votes_bytes: Bytes,
    ) -> ShardResult<()> {
        let votes: Signed<ShardRevealVotes, min_sig::BLS12381Signature> =
            bcs::from_bytes(&votes_bytes).map_err(ShardError::MalformedType)?;
        if peer != votes.voter().deref() {
            return Err(ShardError::FailedTypeVerification(
                "sender must be voter".to_string(),
            ));
        }
        let shard = self
            .shard_verifier
            .verify(&self.context, votes.auth_token())
            .await?;

        let verified_reveal_votes = Verified::new(votes, votes_bytes, |votes| {
            verify_shard_reveal_votes(votes, &shard)
        })
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        let _ = self
            .dispatcher
            .dispatch_reveal_votes(shard, verified_reveal_votes)
            .await?;

        Ok(())
    }
    async fn handle_send_scores(
        &self,
        peer: &EncoderPublicKey,
        scores_bytes: Bytes,
    ) -> ShardResult<()> {
        let scores: Signed<ShardScores, min_sig::BLS12381Signature> =
            bcs::from_bytes(&scores_bytes).map_err(ShardError::MalformedType)?;
        if peer != scores.evaluator().deref() {
            return Err(ShardError::FailedTypeVerification(
                "sender must be score producer".to_string(),
            ));
        }
        let shard = self
            .shard_verifier
            .verify(&self.context, scores.auth_token())
            .await?;

        let verified_scores = Verified::new(scores, scores_bytes, |scores| {
            verify_signed_scores(scores, &shard)
        })
        .map_err(|e| ShardError::FailedTypeVerification(e.to_string()))?;
        let _ = self
            .dispatcher
            .dispatch_scores(shard, verified_scores)
            .await?;
        Ok(())
    }
}
