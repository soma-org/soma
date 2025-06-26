use crate::{
    core::pipeline_dispatcher::InternalDispatcher,
    datastore::Store,
    messaging::EncoderInternalNetworkService,
    types::{
        commit::{verify_signed_commit, Commit, CommitAPI},
        commit_votes::{verify_commit_votes, CommitVotes, CommitVotesAPI},
        context::Context,
        finality::{verify_signed_finality, Finality, FinalityAPI},
        reveal::{verify_signed_reveal, Reveal, RevealAPI},
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::EncoderPublicKey,
    error::{ShardError, ShardResult},
    shard_verifier::ShardVerifier,
    signed::Signed,
    verified::Verified,
};
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};
use types::shard_score::{verify_signed_score, ShardScore, ShardScoreAPI};

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
        let signed_commit: Signed<Commit, min_sig::BLS12381Signature> =
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
        if peer != signed_commit.author() {
            warn!(
                "Sender must be committer. Got: {:?}, expected: {:?}",
                peer,
                signed_commit.author()
            );
            return Err(ShardError::FailedTypeVerification(
                "sender must be committer".to_string(),
            ));
        }

        debug!("Verifying shard with auth token");
        let inner_context = self.context.inner();

        let committees = match inner_context.committees(signed_commit.auth_token().epoch()) {
            Ok(c) => {
                tracing::debug!("Successfully retrieved committees");
                c
            }
            Err(e) => {
                tracing::error!("Failed to get committees: {:?}", e);
                return Err(e);
            }
        };

        let (shard, cancellation) = match self.shard_verifier.verify(
            committees.authority_committee.clone(),
            committees.encoder_committee.clone(),
            committees.vdf_iterations,
            signed_commit.auth_token(),
        ) {
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
                verify_signed_commit(signed_commit, &shard)?;
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
            match self
                .dispatcher
                .dispatch_commit(shard, verified_commit, cancellation)
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
            debug!("Own key: {:?}", self.context.inner().own_encoder_key());
            debug!(
                "Num object servers: {}",
                self.context.inner().encoder_object_servers.len()
            );
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
        let votes: Signed<CommitVotes, min_sig::BLS12381Signature> =
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
        if peer != votes.author() {
            warn!(
                "Sender must be voter. Got: {:?}, expected: {:?}",
                peer,
                votes.author()
            );
            return Err(ShardError::FailedTypeVerification(
                "sender must be voter".to_string(),
            ));
        }

        debug!("Verifying shard with auth token");
        let inner_context = self.context.inner();

        let committees = match inner_context.committees(votes.auth_token().epoch()) {
            Ok(c) => {
                tracing::debug!("Successfully retrieved committees");
                c
            }
            Err(e) => {
                tracing::error!("Failed to get committees: {:?}", e);
                return Err(e);
            }
        };

        let (shard, cancellation) = match self.shard_verifier.verify(
            committees.authority_committee.clone(),
            committees.encoder_committee.clone(),
            committees.vdf_iterations,
            votes.auth_token(),
        ) {
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
            verify_commit_votes(votes, &shard)
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
            .dispatch_commit_votes(shard, verified_commit_votes, cancellation)
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
        let reveal: Signed<Reveal, min_sig::BLS12381Signature> =
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
        if peer != reveal.author() {
            warn!(
                "Sender must be inference encoder for reveal. Got: {:?}, expected: {:?}",
                peer,
                reveal.author()
            );
            return Err(ShardError::FailedTypeVerification(
                "sender must be inference encoder for reveal".to_string(),
            ));
        }

        debug!("Verifying shard with auth token");
        let inner_context = self.context.inner();

        let committees = match inner_context.committees(reveal.auth_token().epoch()) {
            Ok(c) => {
                tracing::debug!("Successfully retrieved committees");
                c
            }
            Err(e) => {
                tracing::error!("Failed to get committees: {:?}", e);
                return Err(e);
            }
        };

        let (shard, cancellation) = match self.shard_verifier.verify(
            committees.authority_committee.clone(),
            committees.encoder_committee.clone(),
            committees.vdf_iterations,
            reveal.auth_token(),
        ) {
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
            verify_signed_reveal(reveal, &shard)
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

        debug!("Dispatching reveal");
        match self
            .dispatcher
            .dispatch_reveal(shard, verified_reveal, cancellation)
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

    async fn handle_send_scores(
        &self,
        peer: &EncoderPublicKey,
        scores_bytes: Bytes,
    ) -> ShardResult<()> {
        info!("Handling send scores from peer: {:?}", peer);
        debug!("Scores bytes size: {} bytes", scores_bytes.len());

        trace!("Deserializing scores bytes");
        let scores: Signed<ShardScore, min_sig::BLS12381Signature> =
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
        if peer != scores.author() {
            warn!(
                "Sender must be score producer. Got: {:?}, expected: {:?}",
                peer,
                scores.author()
            );
            return Err(ShardError::FailedTypeVerification(
                "sender must be score producer".to_string(),
            ));
        }

        debug!("Verifying shard with auth token");
        let inner_context = self.context.inner();

        let committees = match inner_context.committees(scores.auth_token().epoch()) {
            Ok(c) => {
                tracing::debug!("Successfully retrieved committees");
                c
            }
            Err(e) => {
                tracing::error!("Failed to get committees: {:?}", e);
                return Err(e);
            }
        };

        let (shard, cancellation) = match self.shard_verifier.verify(
            committees.authority_committee.clone(),
            committees.encoder_committee.clone(),
            committees.vdf_iterations,
            scores.auth_token(),
        ) {
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
            verify_signed_score(scores, &shard)
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
            .dispatch_scores(shard, verified_scores, cancellation)
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
    async fn handle_send_finality(
        &self,
        peer: &EncoderPublicKey,
        finality_bytes: Bytes,
    ) -> ShardResult<()> {
        info!("Handling send finality from peer: {:?}", peer);
        debug!("Finality bytes size: {} bytes", finality_bytes.len());
        trace!("Deserializing finality bytes");
        let finality: Signed<Finality, min_sig::BLS12381Signature> =
            match bcs::from_bytes(&finality_bytes) {
                Ok(s) => {
                    debug!("Successfully deserialized");
                    s
                }
                Err(e) => {
                    error!("Failed to deserialize: {:?}", e);
                    return Err(ShardError::MalformedType(e));
                }
            };
        debug!("Checking if peer is the encoder");
        if peer != finality.encoder() {
            warn!(
                "Sender must be encoder. Got: {:?}, expected: {:?}",
                peer,
                finality.encoder()
            );
            return Err(ShardError::FailedTypeVerification(
                "sender must be score producer".to_string(),
            ));
        }
        debug!("Verifying shard with auth token");
        let inner_context = self.context.inner();

        let committees = match inner_context.committees(finality.auth_token().epoch()) {
            Ok(c) => {
                tracing::debug!("Successfully retrieved committees");
                c
            }
            Err(e) => {
                tracing::error!("Failed to get committees: {:?}", e);
                return Err(e);
            }
        };

        let (shard, cancellation) = match self.shard_verifier.verify(
            committees.authority_committee.clone(),
            committees.encoder_committee.clone(),
            committees.vdf_iterations,
            finality.auth_token(),
        ) {
            Ok(s) => {
                debug!("Shard verification succeeded");
                s
            }
            Err(e) => {
                error!("Shard verification failed: {:?}", e);
                return Err(e);
            }
        };
        debug!("Creating verified finality object");
        let verified_finality = match Verified::new(finality, finality_bytes, |finality| {
            debug!("Verifying signed finality");
            verify_signed_finality(finality, &shard)
        }) {
            Ok(v) => {
                debug!("Verified finality created successfully");
                v
            }
            Err(e) => {
                error!("Failed to create verified finality: {:?}", e);
                return Err(ShardError::FailedTypeVerification(e.to_string()));
            }
        };
        debug!("Dispatching finality");
        match self
            .dispatcher
            .dispatch_finality(shard, verified_finality, cancellation)
            .await
        {
            Ok(_) => {
                info!("Successfully dispatched finality");
            }
            Err(e) => {
                error!("Failed to dispatch finality: {:?}", e);
                return Err(e);
            }
        }

        info!("handle_send_finality completed successfully");
        Ok(())
    }
}
