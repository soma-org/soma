use core::panic;
use dashmap::DashMap;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use objects::storage::ObjectStorage;
use parking_lot::RwLock;
use probe::messaging::ProbeClient;
use shared::{
    crypto::{
        keys::{EncoderAggregateSignature, EncoderKeyPair, EncoderPublicKey, EncoderSignature},
        Aes256IV, Aes256Key, EncryptionKey,
    },
    digest::Digest,
    metadata::Metadata,
    scope::Scope,
    signed::Signed,
    verified::Verified,
};
use std::{collections::{HashMap, HashSet}, future::Future, sync::Arc, time::Duration};
use tokio::{
    sync::{oneshot, Semaphore},
    time::sleep,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::{
    actors::{
        pipelines::{
            broadcast::{self, BroadcastAction, BroadcastProcessor},
            evaluation::{self, EvaluationProcessor},
        },
        ActorHandle,
    },
    datastore::Store,
    error::{ShardError, ShardResult},
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        shard::Shard,
        shard_commit::{Route, ShardCommit, ShardCommitAPI},
        shard_commit_votes::{ShardCommitVotes, ShardCommitVotesAPI, ShardCommitVotesV1},
        shard_reveal::{ShardReveal, ShardRevealAPI, ShardRevealV1},
        shard_reveal_votes::{ShardRevealVotes, ShardRevealVotesAPI, ShardRevealVotesV1},
        shard_scores::{Score, ScoreSet, ScoreSetAPI, ShardScores, ShardScoresAPI, ShardScoresV1},
        shard_verifier::ShardAuthToken,
    },
};

use super::internal_broadcaster::Broadcaster;

#[derive(Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
enum ShardStage {
    Commit,
    Reveal,
}

#[derive(Eq, PartialEq, PartialOrd, Ord, Hash, Clone, Copy)]
enum Finality {
    Accepted,
    Rejected,
}

pub(crate) struct ShardTracker<C: EncoderInternalNetworkClient, S: ObjectStorage, P: ProbeClient> {
    #[allow(clippy::type_complexity)]
    oneshots: Arc<DashMap<(Digest<Shard>, ShardStage), oneshot::Sender<()>>>,
    max_requests: Arc<Semaphore>, // Limits concurrent tasks
    broadcast_handle: RwLock<Option<ActorHandle<BroadcastProcessor<C, S, P>>>>,
    store: Arc<dyn Store>,
    encoder_keypair: Arc<EncoderKeyPair>,
    evaluation_handle: RwLock<Option<ActorHandle<EvaluationProcessor<C, S, P>>>>,
}

impl<C: EncoderInternalNetworkClient, S: ObjectStorage, P: ProbeClient> ShardTracker<C, S, P> {
    pub(crate) fn new(
        max_requests: Arc<Semaphore>,
        store: Arc<dyn Store>,
        encoder_keypair: Arc<EncoderKeyPair>,
    ) -> Self {
        Self {
            oneshots: Arc::new(DashMap::new()),
            max_requests,
            store,
            encoder_keypair,
            broadcast_handle: RwLock::new(None),
            evaluation_handle: RwLock::new(None),
        }
    }

    pub(crate) fn set_broadcast_handle(&self, handle: ActorHandle<BroadcastProcessor<C, S, P>>) {
        *self.broadcast_handle.write() = Some(handle);
    }

    pub(crate) fn set_evaluation_handle(&self, handle: ActorHandle<EvaluationProcessor<C, S, P>>) {
        *self.evaluation_handle.write() = Some(handle);
    }

    pub(crate) async fn track_valid_commit(
        &self,
        shard: Shard,
        signed_commit: Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let quorum_threshold = shard.quorum_threshold() as usize;
        let max_size = shard.size();
        let count = self.store.count_signed_commits(&shard)?;
        let shard_digest = shard.digest()?;

        info!(
            "Tracking commit from: {:?}",
            signed_commit.clone().committer()
        );

        match count {
            1 => {
                self.store.add_first_commit_time(&shard)?;
            }
            count if count == quorum_threshold => {
                let mut duration = self
                    .store
                    .get_first_commit_time(&shard)
                    .map_or(Duration::from_secs(60), |first_commit_time| {
                        first_commit_time.elapsed()
                    });
                duration = std::cmp::max(duration, Duration::from_secs(5));

                let broadcast_handle_option = self.broadcast_handle.read().clone();

                if let Some(broadcast_handle) = broadcast_handle_option {
                    info!("Starting commit timer with duration: {} ms", duration.as_millis());
                    self.start_timer(
                        shard_digest,
                        ShardStage::Commit,
                        duration,
                        async move || {
                            info!("On trigger - sending commit vote to BroadcastProcessor");             
                            match broadcast_handle
                                .background_process(
                                    BroadcastAction::CommitVote(
                                        signed_commit.auth_token().clone(),
                                        shard,
                                    ),
                                    CancellationToken::new(),
                                )
                                .await
                            {
                                Ok(_) => info!("Broadcasting commit vote succeeded"),
                                Err(e) => error!("Broadcasting commit vote failed: {}", e),
                            }; 
                        }
                    )
                    .await;
                } else {
                    panic!("Broadcast handle not set for ShardTracker!");
                }
            }
            count if count == max_size => {
                let key = (shard_digest, ShardStage::Commit);
                if let Some((_key, tx)) = self.oneshots.remove(&key) {
                    let _ = tx.send(());
                }
            }
            _ => {}
        };
        Ok(())
    }

    pub(crate) async fn track_valid_commit_votes(
        &self,
        shard: Shard,
        commit_votes: Verified<Signed<ShardCommitVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        info!("Starting track_valid_commit_votes for voter: {:?}", commit_votes.voter());
    
        let mut finalized_encoders: HashMap<EncoderPublicKey, Finality> = HashMap::new();
        let num_votes = self.store.count_commit_votes(&shard)?;
        debug!("Current commit vote count: {}", num_votes);
    
        let accepts_keys: HashSet<_> = commit_votes
            .accepts()
            .into_iter()
            .map(|(key, _)| key.clone())
            .collect();
        let encoders_set: HashSet<_> = shard.encoders().into_iter().collect();
        let rejects: Vec<EncoderPublicKey> =
            encoders_set.difference(&accepts_keys).cloned().collect();
    
        let remaining_votes = shard.size() - num_votes;
        debug!("Processing votes breakdown - accepts: {}, rejects: {}, remaining: {}", 
               accepts_keys.len(), rejects.len(), remaining_votes);
    
        for (encoder, digest) in commit_votes.accepts() {
            let vote_counts =
                self.store
                    .get_commit_votes_for_encoder(&shard, encoder, Some(digest))?;
    
            debug!("Evaluating encoder commit votes - encoder: {:?}, accept_count: {:?}, reject_count: {}, highest: {}, quorum_threshold: {}", 
                   encoder, vote_counts.accept_count().unwrap_or(0_usize), vote_counts.reject_count(), 
                   vote_counts.highest(), shard.quorum_threshold());
    
            if vote_counts.accept_count().unwrap_or(0_usize) >= shard.quorum_threshold() as usize {
                info!("Encoder commit ACCEPTED - reached quorum threshold for encoder: {:?}", encoder);
                finalized_encoders.insert(encoder.clone(), Finality::Accepted);
            } else if vote_counts.reject_count() >= shard.rejection_threshold() as usize
                || vote_counts.highest() + remaining_votes < shard.quorum_threshold() as usize
            {
                info!("Encoder commit REJECTED - either reject count >= quorum or can't reach quorum for encoder: {:?}", encoder);
                finalized_encoders.insert(encoder.clone(), Finality::Rejected);
            } else {
                debug!("Encoder commit still PENDING - not enough votes yet for encoder: {:?}", encoder);
            }
        }
    
        for encoder in rejects {
            let vote_counts = self
                .store
                .get_commit_votes_for_encoder(&shard, &encoder, None)?;
    
            debug!("Evaluating rejected encoder - encoder: {:?}, reject_count: {}, highest: {}, quorum_threshold: {}", 
                   encoder, vote_counts.reject_count(), vote_counts.highest(), shard.quorum_threshold());
    
            if vote_counts.reject_count() >= shard.rejection_threshold() as usize
                || vote_counts.highest() + remaining_votes < shard.quorum_threshold() as usize
            {
                info!("Rejected encoder commit REJECTED - either reject count >= quorum or can't reach quorum for encoder: {:?}", encoder);
                finalized_encoders.insert(encoder.clone(), Finality::Rejected);
            } else {
                debug!("Rejected encoder still PENDING - not enough votes yet for encoder: {:?}", encoder);
            }
        }
    
        let total_finalized = finalized_encoders.len();
        let total_accepted = finalized_encoders.values()
            .filter(|&&f| f == Finality::Accepted)
            .count();
    
        info!("Finality status for commit votes - total_finalized: {}, total_accepted: {}, shard_size: {}, quorum_threshold: {}", 
              total_finalized, total_accepted, shard.size(), shard.quorum_threshold());
    
        if total_finalized == shard.size() {
            if total_accepted >= shard.quorum_threshold() as usize {
                let already_revealing = self.store.count_signed_reveal(&shard).unwrap_or(0) > 0;
                if already_revealing {
                    info!("Skipping trigger of BroadcastAction::Reveal as reveal phase has already started");
                    return Ok(());
                }

                info!("ALL COMMIT VOTES FINALIZED - Proceeding to REVEAL phase");
                let broadcast_handle_option = self.broadcast_handle.read().clone();
                if let Some(broadcast_handle) = broadcast_handle_option {
                    info!("Triggering BroadcastAction::Reveal");
                    match broadcast_handle
                        .background_process(
                            BroadcastAction::Reveal(commit_votes.auth_token().clone(), shard),
                            CancellationToken::new(),
                        )
                        .await {
                            Ok(_) => info!("BroadcastAction::Reveal successfully triggered"),
                            Err(e) => error!("BroadcastAction::Reveal failed: {:?}", e)
                        }
                    
                } else {
                    error!("Broadcast handle not set for ShardTracker!");
                    panic!("Broadcast handle not set for ShardTracker!");
                }
            } else {
                warn!("ALL COMMIT VOTES FINALIZED - But failed to reach quorum, shard will be terminated. Total accepted: {}, quorum_threshold: {}", 
                     total_accepted, shard.quorum_threshold());
                // should clean up and shutdown the shard since it will not be able to complete
            }
        } else {
            debug!("Not all commits finalized yet - continuing to collect votes. Total finalized: {}, shard_size: {}", 
                  total_finalized, shard.size());
        }
    
        info!("Completed track_valid_commit_votes");
        Ok(())
    }
    
    pub(crate) async fn track_valid_reveal(
        &self,
        shard: Shard,
        signed_reveal: Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        info!("Starting track_valid_reveal for revealer: {:?}", signed_reveal.encoder());
    
        let quorum_threshold = shard.quorum_threshold() as usize;
        let max_size = shard.size();
        let count = self.store.count_signed_reveal(&shard)?;
        let shard_digest = shard.digest()?;
    
        info!("Reveal count statistics - current_reveal_count: {}, quorum_threshold: {}, max_size: {}", 
              count, quorum_threshold, max_size);
    
        match count {
            1 => {
                info!("First reveal received - recording timestamp");
                self.store.add_first_reveal_time(&shard)?;
            }
            count if count == quorum_threshold => {
                let mut duration = self
                    .store
                    .get_first_reveal_time(&shard)
                    .map_or(Duration::from_secs(60), |first_reveal_time| {
                        first_reveal_time.elapsed()
                    });
                duration = std::cmp::max(duration, Duration::from_secs(5));
    
                info!("Quorum of reveals reached - starting vote timer. Duration since first (ms): {}", 
                      duration.as_millis());
    
                let broadcast_handle_option = self.broadcast_handle.read().clone();
    
                if let Some(broadcast_handle) = broadcast_handle_option {
                    debug!("Broadcasting reveal vote after timer");
                    self.start_timer(
                        shard_digest,
                        ShardStage::Reveal,
                        duration,
                        async move || {
                            info!("Reveal timer triggered - broadcasting reveal vote");
                            match broadcast_handle
                                .background_process(
                                    BroadcastAction::RevealVote(
                                        signed_reveal.auth_token().clone(),
                                        shard,
                                    ),
                                    CancellationToken::new(),
                                )
                                .await
                            {
                                Ok(_) => info!("Broadcasting reveal vote succeeded"),
                                Err(e) => error!("Broadcasting reveal vote failed: {}", e),
                            }
                        },
                    )
                    .await;
                } else {
                    error!("Broadcast handle not set for ShardTracker!");
                    panic!("Broadcast handle not set for ShardTracker!");
                }
            }
            count if count == max_size => {
                info!("Maximum reveals received - triggering immediate vote");
                let key = (shard_digest, ShardStage::Reveal);
                if let Some((_key, tx)) = self.oneshots.remove(&key) {
                    debug!("Sending oneshot to trigger immediate vote");
                    let _ = tx.send(());
                    info!("Successfully triggered immediate vote through oneshot");
                } else {
                    debug!("No timer found to cancel - vote may have already started");
                }
            }
            _ => {
                debug!("Collecting more reveals before taking action. Current count: {}, quorum_threshold: {}", 
                       count, quorum_threshold);
            }
        };
    
        info!("Completed track_valid_reveal");
        Ok(())
    }
    
    pub(crate) async fn track_valid_reveal_votes(
        &self,
        shard: Shard,
        reveal_votes: Verified<Signed<ShardRevealVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        info!("Starting track_valid_reveal_votes for voter: {:?}", reveal_votes.voter());
    
        let mut finalized_encoders: HashMap<EncoderPublicKey, Finality> = HashMap::new();
    
        // Log each encoder's vote status
        for encoder in shard.encoders() {
            let agg_votes = self.store.get_reveal_votes_for_encoder(&shard, &encoder)?;
    
            debug!("Evaluating reveal votes for encoder: {:?}, accept_count: {}, reject_count: {}, quorum_threshold: {}, rejection_threshold: {}", 
                   encoder, agg_votes.accept_count(), agg_votes.reject_count(), 
                   shard.quorum_threshold(), shard.rejection_threshold());
    
            if agg_votes.accept_count() >= shard.quorum_threshold() as usize {
                info!("Encoder reveal ACCEPTED - reached quorum threshold for encoder: {:?}", encoder);
                finalized_encoders.insert(encoder.clone(), Finality::Accepted);
            } else if agg_votes.reject_count() >= shard.rejection_threshold() as usize {
                info!("Encoder reveal REJECTED - reached rejection threshold for encoder: {:?}", encoder);
                finalized_encoders.insert(encoder.clone(), Finality::Rejected);
            } else {
                debug!("Encoder reveal still PENDING - not enough votes yet for encoder: {:?}", encoder);
            }
        }
    
        let total_finalized = finalized_encoders.len();
        let total_accepted = finalized_encoders
            .values()
            .filter(|&&f| f == Finality::Accepted)
            .count();
    
        info!("Finality status for reveal votes - total_finalized: {}, total_accepted: {}, shard_size: {}, quorum_threshold: {}", 
              total_finalized, total_accepted, shard.size(), shard.quorum_threshold());
    
        if total_finalized == shard.size() {
            if total_accepted >= shard.quorum_threshold() as usize {
                info!("ALL REVEAL VOTES FINALIZED - Proceeding to EVALUATION phase");
    
                let input = (reveal_votes.auth_token().to_owned(), shard.clone());
                let evaluation_handle_option = self.evaluation_handle.read().clone();
    
                if let Some(evaluation_handle) = evaluation_handle_option {
                    info!("Triggering evaluation process");
                    evaluation_handle
                        .process(input, CancellationToken::new())
                        .await?;
                    info!("Evaluation process successfully triggered");
                } else {
                    error!("Evaluation handle not set in ShardTracker!");
                    panic!("Evaluation handle not set in ShardTracker!");
                }
            } else {
                warn!("ALL REVEAL VOTES FINALIZED - But failed to reach quorum, shard will be terminated. Total accepted: {}, quorum_threshold: {}", 
                     total_accepted, shard.quorum_threshold());
                // should clean up and shutdown the shard since it will not be able to complete
            }
        } else {
            debug!("Not all reveals finalized yet - continuing to collect votes. Total finalized: {}, shard_size: {}", 
                  total_finalized, shard.size());
        }
    
        info!("Completed track_valid_reveal_votes");
        Ok(())
    }
    
    pub(crate) async fn track_valid_scores(
        &self,
        shard: Shard,
        scores: Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        info!("Starting track_valid_scores for scorer: {:?}", scores.evaluator());
    
        let all_scores = self.store.get_signed_scores(&shard)?;
        debug!("Current score count: {}, quorum_threshold: {}", 
               all_scores.len(), shard.quorum_threshold());
    
        let matching_scores: Vec<Signed<ShardScores, min_sig::BLS12381Signature>> = all_scores
            .iter()
            .filter(|score| {
                score.signed_score_set().into_inner() == scores.signed_score_set().into_inner()
            })
            .cloned()
            .collect();
    
        info!("Found matching scores: {}, quorum_threshold: {}", 
              matching_scores.len(), shard.quorum_threshold());

        info!(
            "Matching scores: {:?}", matching_scores.iter().map(|s| s.clone().into_inner().signed_score_set().into_inner().scores()).collect::<Vec<Vec<Score>>>()
        );
    
        if matching_scores.len() >= shard.quorum_threshold() as usize {
            info!("QUORUM OF MATCHING SCORES - Aggregating signatures");
    
            let (signatures, evaluators): (Vec<EncoderSignature>, Vec<EncoderPublicKey>) = {
                let mut sigs = Vec::new();
                let mut evaluators = Vec::new();
                for signed_scores in matching_scores.iter() {
                    let sig = EncoderSignature::from_bytes(&signed_scores.raw_signature())
                        .map_err(ShardError::SignatureAggregationFailure)?;
                    sigs.push(sig);
                    evaluators.push(signed_scores.evaluator().clone());
                }
                (sigs, evaluators)
            };
    
            debug!("Creating aggregate signature with {} signatures from {} evaluators", 
                   signatures.len(), evaluators.len());
    
            let agg = EncoderAggregateSignature::new(&signatures)
                .map_err(ShardError::SignatureAggregationFailure)?;
    
            info!("Successfully created aggregate score with {} evaluators", evaluators.len());
    
            self.store.add_aggregate_score(&shard, (agg.clone(), evaluators))?;
    
            info!("SHARD CONSENSUS COMPLETE - Aggregate score stored: {:?}", agg);
        } else {
            debug!("Not enough matching scores yet - waiting for more scores. Matching scores: {}, quorum_threshold: {}", 
                   matching_scores.len(), shard.quorum_threshold());
        }
    
        info!("Completed track_valid_scores");
        Ok(())
    }

    pub async fn start_timer<F, Fut>(
        &self,
        shard_digest: Digest<Shard>,
        stage: ShardStage,
        timeout: Duration,
        on_trigger: F,
    ) where
        F: FnOnce() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let key = (shard_digest, stage);
        let (tx, rx) = oneshot::channel();
        self.oneshots.insert(key, tx);
        let oneshots = self.oneshots.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = sleep(timeout) => {
                    on_trigger().await; 
                }
                _ = rx => {
                    on_trigger().await;
                }
            }
            oneshots.remove(&key); // Clean up
        });
    }
}
