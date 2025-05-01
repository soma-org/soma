use crate::{
    core::pipeline_dispatcher::InternalDispatcher,
    datastore::Store,
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
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

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
}

#[async_trait]
impl<D: InternalDispatcher> EncoderInternalNetworkService for EncoderInternalService<D> {
    async fn handle_send_commit(
        &self,
        peer: &EncoderPublicKey,
        commit_bytes: Bytes,
    ) -> ShardResult<()> {
        info!("Handling send commit from peer: {:?}", peer);
        debug!("Commit bytes size: {} bytes", commit_bytes.len());

        trace!("Deserializing commit bytes to signed commit");
        let signed_commit: Signed<ShardCommit, min_sig::BLS12381Signature> =
            match bcs::from_bytes(&commit_bytes) {
                Ok(commit) => {
                    debug!("Successfully deserialized commit");
                    commit
                }
                Err(e) => {
                    error!("Failed to deserialize commit: {:?}", e);
                    return Err(ShardError::MalformedType(e));
                }
            };

        debug!("Checking if peer is the committer");
        if peer != signed_commit.committer() {
            warn!(
                "Sender must be committer. Got: {:?}, expected: {:?}",
                peer,
                signed_commit.committer()
            );
            return Err(ShardError::FailedTypeVerification(
                "sender must be committer".to_string(),
            ));
        }

        debug!("Verifying shard with auth token");
        let shard = match self
            .shard_verifier
            .verify(&self.context, signed_commit.auth_token())
            .await
        {
            Ok(s) => {
                debug!("Shard verification succeeded");
                s
            }
            Err(e) => {
                error!("Shard verification failed: {:?}", e);
                return Err(e);
            }
        };

        debug!("Creating verified commit object");
        let verified_commit =
            match Verified::new(signed_commit.clone(), commit_bytes, |signed_commit| {
                debug!("Verifying signed shard commit");
                verify_signed_shard_commit(signed_commit, &shard)?;
                Ok(())
            }) {
                Ok(v) => {
                    debug!("Verified commit created successfully");
                    v
                }
                Err(e) => {
                    error!("Failed to create verified commit: {:?}", e);
                    return Err(ShardError::FailedTypeVerification(e.to_string()));
                }
            };

        debug!("Locking signed commit in store");
        if let Err(e) = self.store.lock_signed_commit(&shard, &signed_commit) {
            error!("Failed to lock signed commit: {:?}", e);
            return Err(e);
        }

        debug!("Looking up object server for peer: {:?}", peer);
        if let Some((peer, address)) = self.context.inner().object_server(peer) {
            let probe_metadata = self
                .context
                .probe_metadata(shard.epoch(), signed_commit.committer())?;
            debug!("Found object server at address: {:?}", address);
            debug!("Dispatching commit to object server");
            match self
                .dispatcher
                .dispatch_commit(shard, verified_commit, probe_metadata, peer, address)
                .await
            {
                Ok(_) => {
                    info!("Successfully dispatched commit");
                }
                Err(e) => {
                    error!("Failed to dispatch commit: {:?}", e);
                    return Err(e);
                }
            }
        } else {
            error!("Object server not found for peer: {:?}", peer);
            return Err(ShardError::NotFound("object server not found".to_string()));
        }

        info!("handle_send_commit completed successfully");
        Ok(())
    }

    async fn handle_send_commit_votes(
        &self,
        peer: &EncoderPublicKey,
        votes_bytes: Bytes,
    ) -> ShardResult<()> {
        info!("Handling send commit votes from peer: {:?}", peer);
        debug!("Votes bytes size: {} bytes", votes_bytes.len());

        trace!("Deserializing votes bytes");
        let votes: Signed<ShardCommitVotes, min_sig::BLS12381Signature> =
            match bcs::from_bytes(&votes_bytes) {
                Ok(v) => {
                    debug!("Successfully deserialized votes");
                    v
                }
                Err(e) => {
                    error!("Failed to deserialize votes: {:?}", e);
                    return Err(ShardError::MalformedType(e));
                }
            };

        debug!("Checking if peer is the voter");
        if peer != votes.voter() {
            warn!(
                "Sender must be voter. Got: {:?}, expected: {:?}",
                peer,
                votes.voter()
            );
            return Err(ShardError::FailedTypeVerification(
                "sender must be voter".to_string(),
            ));
        }

        debug!("Verifying shard with auth token");
        let shard = match self
            .shard_verifier
            .verify(&self.context, votes.auth_token())
            .await
        {
            Ok(s) => {
                debug!("Shard verification succeeded");
                s
            }
            Err(e) => {
                error!("Shard verification failed: {:?}", e);
                return Err(e);
            }
        };

        debug!("Creating verified commit votes object");
        let verified_commit_votes = match Verified::new(votes, votes_bytes, |votes| {
            debug!("Verifying shard commit votes");
            verify_shard_commit_votes(votes, &shard)
        }) {
            Ok(v) => {
                debug!("Verified commit votes created successfully");
                v
            }
            Err(e) => {
                error!("Failed to create verified commit votes: {:?}", e);
                return Err(ShardError::FailedTypeVerification(e.to_string()));
            }
        };

        debug!("Dispatching commit votes");
        match self
            .dispatcher
            .dispatch_commit_votes(shard, verified_commit_votes)
            .await
        {
            Ok(_) => {
                info!("Successfully dispatched commit votes");
            }
            Err(e) => {
                error!("Failed to dispatch commit votes: {:?}", e);
                return Err(e);
            }
        }

        info!("handle_send_commit_votes completed successfully");
        Ok(())
    }

    async fn handle_send_reveal(
        &self,
        peer: &EncoderPublicKey,
        reveal_bytes: Bytes,
    ) -> ShardResult<()> {
        info!("Handling send reveal from peer: {:?}", peer);
        debug!("Reveal bytes size: {} bytes", reveal_bytes.len());

        trace!("Deserializing reveal bytes");
        let reveal: Signed<ShardReveal, min_sig::BLS12381Signature> =
            match bcs::from_bytes(&reveal_bytes) {
                Ok(r) => {
                    debug!("Successfully deserialized reveal");
                    r
                }
                Err(e) => {
                    error!("Failed to deserialize reveal: {:?}", e);
                    return Err(ShardError::MalformedType(e));
                }
            };

        debug!("Checking if peer is the encoder");
        if peer != reveal.encoder() {
            warn!(
                "Sender must be inference encoder for reveal. Got: {:?}, expected: {:?}",
                peer,
                reveal.encoder()
            );
            return Err(ShardError::FailedTypeVerification(
                "sender must be inference encoder for reveal".to_string(),
            ));
        }

        debug!("Verifying shard with auth token");
        let shard = match self
            .shard_verifier
            .verify(&self.context, reveal.auth_token())
            .await
        {
            Ok(s) => {
                debug!("Shard verification succeeded");
                s
            }
            Err(e) => {
                error!("Shard verification failed: {:?}", e);
                return Err(e);
            }
        };

        debug!("Creating verified reveal object");
        let verified_reveal = match Verified::new(reveal.clone(), reveal_bytes, |reveal| {
            debug!("Verifying signed shard reveal");
            verify_signed_shard_reveal(reveal, &shard)
        }) {
            Ok(v) => {
                debug!("Verified reveal created successfully");
                v
            }
            Err(e) => {
                error!("Failed to create verified reveal: {:?}", e);
                return Err(ShardError::FailedTypeVerification(e.to_string()));
            }
        };

        debug!("Checking reveal key in store");
        if let Err(e) = self.store.check_reveal_key(&shard, &reveal) {
            error!("Failed to check reveal key: {:?}", e);
            return Err(e);
        }

        debug!("Dispatching reveal");
        match self
            .dispatcher
            .dispatch_reveal(shard, verified_reveal)
            .await
        {
            Ok(_) => {
                info!("Successfully dispatched reveal");
            }
            Err(e) => {
                error!("Failed to dispatch reveal: {:?}", e);
                return Err(e);
            }
        }

        info!("handle_send_reveal completed successfully");
        Ok(())
    }

    async fn handle_send_reveal_votes(
        &self,
        peer: &EncoderPublicKey,
        votes_bytes: Bytes,
    ) -> ShardResult<()> {
        info!("Handling send reveal votes from peer: {:?}", peer);
        debug!("Votes bytes size: {} bytes", votes_bytes.len());

        trace!("Deserializing votes bytes");
        let votes: Signed<ShardRevealVotes, min_sig::BLS12381Signature> =
            match bcs::from_bytes(&votes_bytes) {
                Ok(v) => {
                    debug!("Successfully deserialized votes");
                    v
                }
                Err(e) => {
                    error!("Failed to deserialize votes: {:?}", e);
                    return Err(ShardError::MalformedType(e));
                }
            };

        debug!("Checking if peer is the voter");
        if peer != votes.voter() {
            warn!(
                "Sender must be voter. Got: {:?}, expected: {:?}",
                peer,
                votes.voter()
            );
            return Err(ShardError::FailedTypeVerification(
                "sender must be voter".to_string(),
            ));
        }

        debug!("Verifying shard with auth token");
        let shard = match self
            .shard_verifier
            .verify(&self.context, votes.auth_token())
            .await
        {
            Ok(s) => {
                debug!("Shard verification succeeded");
                s
            }
            Err(e) => {
                error!("Shard verification failed: {:?}", e);
                return Err(e);
            }
        };

        debug!("Creating verified reveal votes object");
        let verified_reveal_votes = match Verified::new(votes, votes_bytes, |votes| {
            debug!("Verifying shard reveal votes");
            verify_shard_reveal_votes(votes, &shard)
        }) {
            Ok(v) => {
                debug!("Verified reveal votes created successfully");
                v
            }
            Err(e) => {
                error!("Failed to create verified reveal votes: {:?}", e);
                return Err(ShardError::FailedTypeVerification(e.to_string()));
            }
        };

        debug!("Dispatching reveal votes");
        match self
            .dispatcher
            .dispatch_reveal_votes(shard, verified_reveal_votes)
            .await
        {
            Ok(_) => {
                info!("Successfully dispatched reveal votes");
            }
            Err(e) => {
                error!("Failed to dispatch reveal votes: {:?}", e);
                return Err(e);
            }
        }

        info!("handle_send_reveal_votes completed successfully");
        Ok(())
    }

    async fn handle_send_scores(
        &self,
        peer: &EncoderPublicKey,
        scores_bytes: Bytes,
    ) -> ShardResult<()> {
        info!("Handling send scores from peer: {:?}", peer);
        debug!("Scores bytes size: {} bytes", scores_bytes.len());

        trace!("Deserializing scores bytes");
        let scores: Signed<ShardScores, min_sig::BLS12381Signature> =
            match bcs::from_bytes(&scores_bytes) {
                Ok(s) => {
                    debug!("Successfully deserialized scores");
                    s
                }
                Err(e) => {
                    error!("Failed to deserialize scores: {:?}", e);
                    return Err(ShardError::MalformedType(e));
                }
            };

        debug!("Checking if peer is the evaluator");
        if peer != scores.evaluator() {
            warn!(
                "Sender must be score producer. Got: {:?}, expected: {:?}",
                peer,
                scores.evaluator()
            );
            return Err(ShardError::FailedTypeVerification(
                "sender must be score producer".to_string(),
            ));
        }

        debug!("Verifying shard with auth token");
        let shard = match self
            .shard_verifier
            .verify(&self.context, scores.auth_token())
            .await
        {
            Ok(s) => {
                debug!("Shard verification succeeded");
                s
            }
            Err(e) => {
                error!("Shard verification failed: {:?}", e);
                return Err(e);
            }
        };

        debug!("Creating verified scores object");
        let verified_scores = match Verified::new(scores, scores_bytes, |scores| {
            debug!("Verifying signed scores");
            verify_signed_scores(scores, &shard)
        }) {
            Ok(v) => {
                debug!("Verified scores created successfully");
                v
            }
            Err(e) => {
                error!("Failed to create verified scores: {:?}", e);
                return Err(ShardError::FailedTypeVerification(e.to_string()));
            }
        };

        debug!("Dispatching scores");
        match self
            .dispatcher
            .dispatch_scores(shard, verified_scores)
            .await
        {
            Ok(_) => {
                info!("Successfully dispatched scores");
            }
            Err(e) => {
                error!("Failed to dispatch scores: {:?}", e);
                return Err(e);
            }
        }

        info!("handle_send_scores completed successfully");
        Ok(())
    }
}
