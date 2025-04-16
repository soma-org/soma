use dashmap::DashMap;
use fastcrypto::bls12381::min_sig;
use shared::{digest::Digest, signed::Signed, verified::Verified};
use std::{sync::Arc, time::Duration};
use tokio::{
    sync::{oneshot, Semaphore},
    time::sleep,
};

use crate::{
    error::ShardResult,
    types::{
        shard::Shard, shard_commit::ShardCommit, shard_commit_votes::ShardCommitVotes,
        shard_reveal::ShardReveal, shard_reveal_votes::ShardRevealVotes, shard_scores::ShardScores,
    },
};

#[derive(Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
enum SlotType {
    Commit,
    Reveal,
}
#[derive(Clone)]
pub(crate) struct ShardTracker {
    #[allow(clippy::type_complexity)]
    slots: Arc<DashMap<(Digest<Shard>, SlotType), oneshot::Sender<()>>>,
    semaphore: Arc<Semaphore>, // Limits concurrent tasks
}

impl ShardTracker {
    pub(crate) fn new(max_concurrent_tasks: usize) -> Self {
        Self {
            slots: Arc::new(DashMap::new()),
            semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
        }
    }

    pub(crate) async fn track_valid_commit(
        &self,
        shard: Shard,
        signed_commit: Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // if count == shard.minimum_inference_size() as usize {
        //     let epoch = verified_signed_commit.auth_token().epoch();
        //     let duration = self
        //         .store
        //         .time_since_first_certified_commit(epoch, shard_ref)
        //         .unwrap_or(Duration::from_secs(60));

        //     let peers = shard.shard_set();

        //     let broadcaster = self.broadcaster.clone();
        //     let shard_clone = shard.clone();
        //     self.slot_tracker
        //         .start_commit_vote_timer(shard_ref, duration, move || async move {
        //             let _ = broadcaster
        //                 .process(
        //                     (
        //                         verified_signed_commit.auth_token(),
        //                         shard_clone,
        //                         BroadcastType::CommitVote(epoch, shard_ref),
        //                         peers,
        //                     ),
        //                     msg.cancellation.clone(),
        //                 )
        //                 .await;
        //         })
        //         .await;
        // }
        // if count == shard.inference_size() {
        //     self.slot_tracker.trigger_commit_vote(shard_ref).await;
        // }

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
        // if count == shard.minimum_inference_size() as usize {
        //     let duration = self
        //         .store
        //         .time_since_first_reveal(epoch, shard_ref)
        //         .unwrap_or(Duration::from_secs(60));
        //     // TODO: make this cleaner should be built into the shard
        //     let inference_set = shard.inference_set(); // Vec<EncoderIndex>
        //     let evaluation_set = shard.evaluation_set(); // Vec<EncoderIndex>

        //     // Combine into a HashSet to deduplicate
        //     let mut peers_set: HashSet<EncoderIndex> = inference_set.into_iter().collect();
        //     peers_set.extend(evaluation_set);

        //     // Convert back to Vec
        //     let peers: Vec<EncoderIndex> = peers_set.into_iter().collect();
        //     let shard_clone = shard.clone();
        //     let broadcaster = self.broadcaster.clone();
        //     self.slot_tracker
        //         .start_reveal_vote_timer(shard_ref, duration, move || async move {
        //             let _ = broadcaster
        //                 .process(
        //                     (
        //                         auth_token,
        //                         shard_clone,
        //                         BroadcastType::RevealVote(epoch, shard_ref),
        //                         peers,
        //                     ),
        //                     msg.cancellation.clone(),
        //                 )
        //                 .await;
        //         })
        //         .await;
        // }
        // if count == shard.inference_size() {
        //     self.slot_tracker.trigger_reveal_vote(shard_ref).await;
        // }
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
    pub(crate) async fn start_commit_vote_timer<F, Fut>(
        &self,
        shard_ref: Digest<Shard>,
        timeout: Duration,
        on_trigger: F,
    ) where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
        // Acquire a permit, blocking if the limit is reached
        let permit = self.semaphore.clone().acquire_owned().await.unwrap();
        let (tx, rx) = oneshot::channel();

        let slot_key = (shard_ref, SlotType::Commit);
        self.slots.insert(slot_key, tx);

        let slots = self.slots.clone();

        tokio::spawn(async move {
            tokio::select! {
                _ = sleep(timeout) => {
                    on_trigger().await;
                    // Timer hits, trigger work
                }
                _ = rx => {
                    on_trigger().await;
                    // Oneshot receives, trigger work
                }
            }
            slots.remove(&slot_key); // Clean up
            drop(permit); // Release the permit when the task completes
        });
    }

    pub(crate) async fn trigger_commit_vote(&self, shard_ref: Digest<Shard>) {
        let slot_key = (shard_ref, SlotType::Commit);
        if let Some((_, tx)) = self.slots.remove(&slot_key) {
            let _ = tx.send(());
        }
    }
    pub(crate) async fn start_reveal_vote_timer<F, Fut>(
        &self,
        shard_ref: Digest<Shard>,
        timeout: Duration,
        on_trigger: F,
    ) where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
        // Acquire a permit, blocking if the limit is reached
        let permit = self.semaphore.clone().acquire_owned().await.unwrap();
        let (tx, rx) = oneshot::channel();

        let slot_key = (shard_ref, SlotType::Reveal);
        self.slots.insert(slot_key, tx);

        let slots = self.slots.clone();

        tokio::spawn(async move {
            tokio::select! {
                _ = sleep(timeout) => {
                    on_trigger().await;
                    // Timer hits, trigger work
                }
                _ = rx => {
                    on_trigger().await;
                    // Oneshot receives, trigger work
                }
            }
            slots.remove(&slot_key); // Clean up
            drop(permit); // Release the permit when the task completes
        });
    }

    pub(crate) async fn trigger_reveal_vote(&self, shard_ref: Digest<Shard>) {
        let slot_key = (shard_ref, SlotType::Reveal);
        if let Some((_, tx)) = self.slots.remove(&slot_key) {
            let _ = tx.send(());
        }
    }
}
