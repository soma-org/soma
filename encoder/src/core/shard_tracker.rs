use dashmap::DashMap;
use fastcrypto::bls12381::min_sig;
use shared::{crypto::keys::EncoderKeyPair, digest::Digest, signed::Signed, verified::Verified};
use std::{sync::Arc, time::Duration};
use tokio::{
    sync::{oneshot, Semaphore},
    time::sleep,
};

use crate::{
    datastore::Store,
    error::ShardResult,
    messaging::{
        internal_broadcasts::{broadcast_commit_vote, broadcast_reveal_vote},
        EncoderInternalNetworkClient,
    },
    types::{
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_commit_votes::ShardCommitVotes,
        shard_reveal::{ShardReveal, ShardRevealAPI},
        shard_reveal_votes::ShardRevealVotes,
        shard_scores::ShardScores,
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
        let max_size = shard.size() as usize;
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
        // let (total_finalized_slots, total_accepted_slots) = self.store.add_commit_vote(
        //     epoch,
        //     shard_ref,
        //     shard.clone(),
        //     votes.deref().to_owned().deref().to_owned(),
        // )?;

        // if total_finalized_slots == shard.inference_size() {
        //     if total_accepted_slots >= shard.minimum_inference_size() as usize {
        //         if shard.inference_set_contains(&self.own_encoder_key) {
        //             let peers = shard.shard_set();
        //             let _ = self
        //                 .broadcaster
        //                 .process(
        //                     (
        //                         auth_token,
        //                         shard,
        //                         BroadcastType::RevealKey(epoch, shard_ref),
        //                         peers,
        //                     ),
        //                     msg.cancellation.clone(),
        //                 )
        //                 .await;
        //             // trigger reveal if member of inference set
        //         }
        //         // TODO: figure out how rejections are accounted for and whether eval set needs to do anything
        //     } else {
        //         // trigger cancellation, this shard cannot proceed
        //     }
        // }
        //
        Ok(())
    }

    pub(crate) async fn track_valid_reveal(
        &self,
        shard: Shard,
        signed_reveal: Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let quorum_threshold = shard.quorum_threshold() as usize;
        let max_size = shard.size() as usize;
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
        // // TODO: need to ensure that a person can only vote once with a locked in digest
        // let (total_finalized_slots, total_accepted_slots) = self.store.add_reveal_vote(
        //     epoch,
        //     shard_ref,
        //     shard.clone(),
        //     votes.deref().to_owned().deref().to_owned(),
        // )?;

        // if total_finalized_slots == shard.inference_size() {
        //     if total_accepted_slots >= shard.minimum_inference_size() as usize {
        //         if shard.evaluation_set().contains(&self.own_index) {
        //             self.evaluation_handle
        //                 .process((auth_token, shard), msg.cancellation.clone())
        //                 .await?;
        //         }
        //     } else {
        //         // trigger cancellation, this shard cannot proceed
        //     }
        // }
        Ok(())
    }

    pub(crate) async fn track_valid_scores(
        &self,
        shard: Shard,
        scores: Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // let (shard, shard_scores) = msg.input;
        // let epoch = shard.epoch();
        // let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
        // let signed_score_set = shard_scores.signed_score_set();

        // let evaluator = shard_scores.evaluator();

        // let matching_scores =
        //     self.store
        //         .add_scores(epoch, shard_ref, evaluator, signed_score_set.clone())?;

        // if matching_scores.len() >= shard.evaluation_quorum_threshold() as usize {
        //     let (signatures, evaluator_indices): (Vec<EncoderSignature>, Vec<EncoderIndex>) = {
        //         let mut sigs = Vec::new();
        //         let mut indices = Vec::new();

        //         for (evaluator_index, signed_scores) in matching_scores.iter() {
        //             let sig = EncoderSignature::from_bytes(&signed_scores.raw_signature())
        //                 .map_err(ShardError::SignatureAggregationFailure)?;
        //             sigs.push(sig);
        //             indices.push(*evaluator_index);
        //         }
        //         (sigs, indices)
        //     };

        //     let agg = EncoderAggregateSignature::new(&signatures)
        //         .map_err(ShardError::SignatureAggregationFailure)?;

        //     let cert = Certified::new_v1(signed_score_set.into_inner(), evaluator_indices, agg);
        //     println!("{:?}", cert);
        // }

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
