use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        commit::{Commit, CommitAPI},
        commit_votes::{CommitVotes, CommitVotesV1},
    },
};
use async_trait::async_trait;
use dashmap::DashMap;
use evaluation::messaging::EvaluationClient;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use objects::{networking::ObjectNetworkClient, storage::ObjectStorage};
use quick_cache::sync::{Cache, GuardResult};
use std::{future::Future, sync::Arc, time::Duration};
use tokio::{sync::oneshot, time::sleep};
use tracing::info;
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
    shard::Shard,
    shard_crypto::{
        digest::Digest,
        keys::{EncoderKeyPair, EncoderPublicKey},
        scope::Scope,
        signed::Signed,
        verified::Verified,
    },
};

use super::commit_votes::CommitVotesProcessor;

pub(crate) struct CommitProcessor<
    O: ObjectNetworkClient,
    E: EncoderInternalNetworkClient,
    S: ObjectStorage,
    P: EvaluationClient,
> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<E>>,
    commit_vote_handle: ActorHandle<CommitVotesProcessor<O, E, S, P>>,
    oneshots: Arc<DashMap<Digest<Shard>, oneshot::Sender<()>>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    recv_dedup: Cache<(Digest<Shard>, EncoderPublicKey), ()>,
    send_dedup: Cache<Digest<Shard>, ()>,
}

impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > CommitProcessor<O, E, S, P>
{
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<E>>,
        commit_vote_handle: ActorHandle<CommitVotesProcessor<O, E, S, P>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        recv_cache_capacity: usize,
        send_cache_capacity: usize,
    ) -> Self {
        Self {
            store,
            broadcaster,
            commit_vote_handle,
            oneshots: Arc::new(DashMap::new()),
            encoder_keypair,
            recv_dedup: Cache::new(recv_cache_capacity),
            send_dedup: Cache::new(send_cache_capacity),
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
impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > Processor for CommitProcessor<O, E, S, P>
{
    type Input = (Shard, Verified<Commit>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, verified_commit) = msg.input;

            let shard_digest = shard.digest()?;

            match self.recv_dedup.get_value_or_guard(
                &(shard_digest, verified_commit.author().clone()),
                Some(Duration::from_secs(5)),
            ) {
                GuardResult::Value(_) => return Err(ShardError::RecvDuplicate),
                GuardResult::Guard(placeholder) => {
                    placeholder.insert(());
                }
                GuardResult::Timeout => (),
            }

            let _ = self.store.add_commit(&shard, &verified_commit)?;

            let quorum_threshold = shard.quorum_threshold() as usize;
            let max_size = shard.size();
            let count = self.store.count_commits(&shard)?;
            let shard_digest = shard.digest()?;

            info!(
                "Tracking commit from: {:?}",
                verified_commit.clone().author()
            );

            match count {
                1 => {
                    self.store.add_first_commit_time(&shard)?;
                }
                count if count == quorum_threshold => {
                    match self
                        .send_dedup
                        .get_value_or_guard(&(shard_digest), Some(Duration::from_secs(5)))
                    {
                        GuardResult::Value(_) => return Ok(()),
                        GuardResult::Guard(placeholder) => {
                            placeholder.insert(());
                        }
                        GuardResult::Timeout => (),
                    };
                    let duration = self.store.get_first_commit_time(&shard).map_or(
                        Duration::from_secs(60),
                        |first_commit_time| {
                            // TODO: add min to bound the timeout to gurantee reasonable clean up
                            std::cmp::max(first_commit_time.elapsed(), Duration::from_secs(5))
                        },
                    );

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
                        let commits = match store.get_commits(&shard) {
                            Ok(commits) => commits,
                            Err(e) => {
                                tracing::error!("Error getting signed commits: {:?}", e);
                                return Err(ShardError::MissingData);
                            }
                        };

                        // Format commits for votes
                        let commits = commits
                            .iter()
                            .map(|commit| (commit.author().clone(), commit.reveal_digest().clone()))
                            .collect();

                        // Create votes
                        let votes = CommitVotes::V1(CommitVotesV1::new(
                            verified_commit.auth_token().clone(),
                            encoder_keypair.public(),
                            commits,
                        ));

                        let verified = Verified::from_trusted(votes).unwrap();

                        info!(
                            "Broadcasting commit vote to other encoders: {:?}",
                            shard.encoders()
                        );

                        commit_vote_handle
                            .process((shard.clone(), verified.clone()), msg.cancellation.clone())
                            .await?;
                        // Broadcast to other encoders
                        broadcaster
                            .broadcast(
                                verified.clone(),
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
