use dashmap::DashMap;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::{EncoderAggregateSignature, EncoderKeyPair, EncoderPublicKey, EncoderSignature},
    digest::Digest,
    signed::Signed,
    verified::Verified,
};
use std::{collections::HashSet, sync::Arc, time::Duration};
use tokio::{
    sync::{oneshot, Semaphore},
    time::sleep,
};

use crate::{
    datastore::Store,
    error::{ShardError, ShardResult},
    messaging::{
        internal_broadcasts::{broadcast_commit_vote, broadcast_reveal, broadcast_reveal_vote},
        EncoderInternalNetworkClient,
    },
    types::{
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_commit_votes::ShardCommitVotes,
        shard_reveal::{ShardReveal, ShardRevealAPI},
        shard_reveal_votes::{ShardRevealVotes, ShardRevealVotesAPI},
        shard_scores::{ShardScores, ShardScoresAPI},
    },
};

use super::internal_broadcaster::Broadcaster;

#[derive(Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
enum ShardStage {
    Commit,
    Reveal,
}

#[derive(Clone)]
pub(crate) struct ShardTracker<C: EncoderInternalNetworkClient> {
    #[allow(clippy::type_complexity)]
    oneshots: Arc<DashMap<(Digest<Shard>, ShardStage), oneshot::Sender<()>>>,
    max_requests: Arc<Semaphore>, // Limits concurrent tasks
    broadcaster: Arc<Broadcaster<C>>,
    store: Arc<dyn Store>,
    encoder_keypair: Arc<EncoderKeyPair>,
}

impl<C: EncoderInternalNetworkClient> ShardTracker<C> {
    pub(crate) fn new(
        max_requests: Arc<Semaphore>,
        broadcaster: Arc<Broadcaster<C>>,
        store: Arc<dyn Store>,
        encoder_keypair: Arc<EncoderKeyPair>,
    ) -> Self {
        Self {
            oneshots: Arc::new(DashMap::new()),
            max_requests,
            broadcaster,
            store,
            encoder_keypair,
        }
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

        match count {
            1 => {
                self.store.add_first_commit_time(&shard)?;
            }
            count if count == quorum_threshold => {
                let duration = self
                    .store
                    .get_first_commit_time(&shard)
                    .map_or(Duration::from_secs(60), |first_commit_time| {
                        first_commit_time.elapsed()
                    });
                let peers = shard.encoders();
                self.start_timer(
                    shard_digest,
                    ShardStage::Commit,
                    duration,
                    broadcast_commit_vote(
                        peers,
                        self.broadcaster.clone(),
                        self.store.clone(),
                        signed_commit.auth_token().clone(),
                        shard,
                        self.encoder_keypair.clone(),
                    ),
                );
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
        for encoder in shard.encoders() {
            let vote_counts = self.store.get_reveal_votes_for_encoder(&shard, &encoder)?;
            if vote_counts.accept_count() >= shard.quorum_threshold() as usize {
                // finalized as an accept
            } else if vote_counts.reject_count() >= shard.rejection_threshold() as usize {
                // finalized as a reject
            }
        }

        Ok(())
    }

    pub(crate) async fn track_valid_reveal(
        &self,
        shard: Shard,
        signed_reveal: Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let quorum_threshold = shard.quorum_threshold() as usize;
        let max_size = shard.size();
        let count = self.store.count_signed_reveal(&shard)?;
        let shard_digest = shard.digest()?;

        match count {
            1 => {
                self.store.add_first_reveal_time(&shard)?;
            }
            count if count == quorum_threshold => {
                let duration = self
                    .store
                    .get_first_reveal_time(&shard)
                    .map_or(Duration::from_secs(60), |first_reveal_time| {
                        first_reveal_time.elapsed()
                    });
                let peers = shard.encoders();
                self.start_timer(
                    shard_digest,
                    ShardStage::Reveal,
                    duration,
                    broadcast_reveal_vote(
                        peers,
                        self.broadcaster.clone(),
                        self.store.clone(),
                        signed_reveal.auth_token().clone(),
                        shard,
                        self.encoder_keypair.clone(),
                    ),
                );
            }
            count if count == max_size => {
                let key = (shard_digest, ShardStage::Reveal);
                if let Some((_key, tx)) = self.oneshots.remove(&key) {
                    let _ = tx.send(());
                }
            }
            _ => {}
        };
        Ok(())
    }

    pub(crate) async fn track_valid_reveal_votes(
        &self,
        shard: Shard,
        reveal_votes: Verified<Signed<ShardRevealVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        #[derive(Eq, PartialEq, PartialOrd, Ord, Hash, Clone, Copy)]
        enum Finality {
            Accepted,
            Rejected,
        }
        let mut finalized: HashSet<Finality> = HashSet::new();
        for encoder in shard.encoders() {
            if reveal_votes.accepts().contains(&encoder) {
                finalized.insert(Finality::Accepted);
            } else {
                finalized.insert(Finality::Rejected);
            }
        }
        let total_finalized = finalized.len();
        let total_accepted = finalized
            .iter()
            .filter(|&&f| f == Finality::Accepted)
            .count();

        if total_finalized == shard.size() {
            if total_accepted >= shard.quorum_threshold() as usize {
                let peers = shard.encoders();
                broadcast_reveal(
                    peers,
                    self.broadcaster.clone(),
                    reveal_votes.auth_token().clone(),
                    shard,
                    self.encoder_keypair.clone(),
                )()
                // yay safely finalized all reveals
            } else {
                // should clean up and shutdown the shard since it will not be able to complete
            }
        }
        Ok(())
    }

    pub(crate) async fn track_valid_scores(
        &self,
        shard: Shard,
        scores: Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let all_scores = self.store.get_signed_scores(&shard)?;

        let matching_scores: Vec<Signed<ShardScores, min_sig::BLS12381Signature>> = all_scores
            .iter()
            .filter(|score| {
                score.signed_score_set().into_inner() == scores.signed_score_set().into_inner()
            })
            .cloned()
            .collect();
        if matching_scores.len() >= shard.quorum_threshold() as usize {
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

            let agg = EncoderAggregateSignature::new(&signatures)
                .map_err(ShardError::SignatureAggregationFailure)?;

            println!("{:?} {:?}", agg, evaluators);
        }

        Ok(())
    }
    pub async fn start_timer<F>(
        &self,
        shard_digest: Digest<Shard>,
        stage: ShardStage,
        timeout: Duration,
        on_trigger: F,
    ) where
        F: FnOnce() + Send + 'static,
    {
        let key = (shard_digest, stage);
        let (tx, rx) = oneshot::channel();
        self.oneshots.insert(key, tx);
        let oneshots = self.oneshots.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = sleep(timeout) => {
                    on_trigger();
                }
                _ = rx => {
                    on_trigger();
                }
            }
            oneshots.remove(&key); // Clean up
        });
    }
}
