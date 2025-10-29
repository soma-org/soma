use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::{ShardStage, Store},
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        commit::{Commit, CommitAPI},
        commit_votes::{CommitVotes, CommitVotesV1},
    },
};
use async_trait::async_trait;
use dashmap::DashMap;
use intelligence::evaluation::messaging::EvaluationClient;
use object_store::ObjectStore;
use std::{future::Future, sync::Arc, time::Duration};
use tokio::{sync::oneshot, time::sleep};
use tracing::info;
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::ShardResult,
    shard::Shard,
    shard_crypto::{digest::Digest, keys::EncoderKeyPair, verified::Verified},
};

use super::commit_votes::CommitVotesProcessor;

pub(crate) struct CommitProcessor<C: EncoderInternalNetworkClient, E: EvaluationClient> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<C>>,
    commit_vote_handle: ActorHandle<CommitVotesProcessor<C, E>>,
    oneshots: Arc<DashMap<Digest<Shard>, oneshot::Sender<()>>>,
    encoder_keypair: Arc<EncoderKeyPair>,
}

impl<C: EncoderInternalNetworkClient, E: EvaluationClient> CommitProcessor<C, E> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<C>>,
        commit_vote_handle: ActorHandle<CommitVotesProcessor<C, E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
    ) -> Self {
        Self {
            store,
            broadcaster,
            commit_vote_handle,
            oneshots: Arc::new(DashMap::new()),
            encoder_keypair,
        }
    }

    pub async fn start_timer<F, Fut>(
        &self,
        shard_digest: Digest<Shard>,
        timeout: Duration,
        on_trigger: F,
    ) where
        F: FnOnce() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ShardResult<()>> + Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        self.oneshots.insert(shard_digest, tx);
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
            oneshots.remove(&shard_digest); // Clean up
        });
    }
}

#[async_trait]
impl<C: EncoderInternalNetworkClient, E: EvaluationClient> Processor for CommitProcessor<C, E> {
    type Input = (Shard, Verified<Commit>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, verified_commit) = msg.input;

            let _ = self.store.add_shard_stage_message(
                &shard,
                ShardStage::Commit,
                verified_commit.author(),
            )?;

            let _ = self.store.add_submission_digest(
                &shard,
                verified_commit.author(),
                verified_commit.submission_digest().clone(),
            )?;

            let quorum_threshold = shard.quorum_threshold() as usize;
            let max_size = shard.size();
            let shard_digest = shard.digest()?;

            let all_submission_digests = self.store.get_all_submission_digests(&shard)?;
            let count = all_submission_digests.len();

            match count {
                count if count == quorum_threshold => {
                    let _ = self
                        .store
                        .add_shard_stage_dispatch(&shard, ShardStage::CommitVote)?;

                    let earliest = all_submission_digests
                        .iter()
                        .map(|(_, _, t)| t)
                        .min()
                        .unwrap();

                    let duration = std::cmp::max(earliest.elapsed(), Duration::from_secs(5));

                    let broadcaster = self.broadcaster.clone();

                    let store = self.store.clone();
                    info!(
                        "Starting commit timer with duration: {} ms",
                        duration.as_millis()
                    );
                    let encoder_keypair = self.encoder_keypair.clone();
                    let commit_vote_handle = self.commit_vote_handle.clone();
                    self.start_timer(shard_digest, duration, async move || {
                        info!("On trigger - sending commit vote to BroadcastProcessor");
                        let all_submission_digests = store.get_all_submission_digests(&shard)?;

                        let accepted_submission_digests = all_submission_digests
                            .iter()
                            .map(|(encoder, submission_digest, _)| {
                                (encoder.clone(), submission_digest.clone())
                            })
                            .collect();

                        let commit_votes = CommitVotes::V1(CommitVotesV1::new(
                            verified_commit.auth_token().clone(),
                            encoder_keypair.public(),
                            accepted_submission_digests,
                        ));

                        let verified_commit_votes = Verified::from_trusted(commit_votes).unwrap();

                        info!(
                            "Broadcasting commit vote to other encoders: {:?}",
                            shard.encoders()
                        );

                        commit_vote_handle
                            .process(
                                (shard.clone(), verified_commit_votes.clone()),
                                msg.cancellation.clone(),
                            )
                            .await?;
                        // Broadcast to other encoders
                        broadcaster
                            .broadcast(
                                verified_commit_votes.clone(),
                                shard.encoders(),
                                |client, peer, verified_type| async move {
                                    client
                                        .send_commit_votes(&peer, &verified_type, MESSAGE_TIMEOUT)
                                        .await?;
                                    Ok(())
                                },
                            )
                            .await?;

                        Ok(())
                    })
                    .await;
                }
                count if count == max_size => {
                    let key = shard_digest;
                    if let Some((_key, tx)) = self.oneshots.remove(&key) {
                        let _ = tx.send(());
                    }
                }
                _ => {}
            };
            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
