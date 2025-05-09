use crate::{
    actors::{
        workers::downloader::{Downloader, DownloaderInput},
        ActorHandle, ActorMessage, Processor,
    },
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    error::{ShardError, ShardResult},
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        shard::Shard,
        shard_commit::{ShardCommit, ShardCommitAPI},
        shard_commit_votes::{ShardCommitVotes, ShardCommitVotesV1},
    },
};
use async_trait::async_trait;
use dashmap::DashMap;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use objects::{networking::ObjectNetworkClient, storage::ObjectStorage};
use probe::messaging::ProbeClient;
use quick_cache::sync::{Cache, GuardResult};
use shared::{
    crypto::keys::{EncoderKeyPair, EncoderPublicKey, PeerPublicKey},
    digest::Digest,
    probe::ProbeMetadata,
    scope::Scope,
    signed::Signed,
    verified::Verified,
};
use soma_network::multiaddr::Multiaddr;
use std::{future::Future, sync::Arc, time::Duration};
use tokio::{sync::oneshot, time::sleep};
use tracing::info;

use super::commit_votes::CommitVotesProcessor;

pub(crate) struct CommitProcessor<
    E: EncoderInternalNetworkClient,
    C: ObjectNetworkClient,
    S: ObjectStorage,
    P: ProbeClient,
> {
    store: Arc<dyn Store>,
    downloader: ActorHandle<Downloader<C, S>>,
    broadcaster: Arc<Broadcaster<E>>,
    commit_vote_handle: ActorHandle<CommitVotesProcessor<E, S, P>>,
    oneshots: Arc<DashMap<Digest<Shard>, oneshot::Sender<()>>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    recv_dedup: Cache<(Digest<Shard>, EncoderPublicKey), ()>,
    send_dedup: Cache<Digest<Shard>, ()>,
}

impl<E: EncoderInternalNetworkClient, C: ObjectNetworkClient, S: ObjectStorage, P: ProbeClient>
    CommitProcessor<E, C, S, P>
{
    pub(crate) fn new(
        store: Arc<dyn Store>,
        downloader: ActorHandle<Downloader<C, S>>,
        broadcaster: Arc<Broadcaster<E>>,
        commit_vote_handle: ActorHandle<CommitVotesProcessor<E, S, P>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        recv_cache_capacity: usize,
        send_cache_capacity: usize,
    ) -> Self {
        Self {
            store,
            downloader,
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
impl<E: EncoderInternalNetworkClient, O: ObjectNetworkClient, S: ObjectStorage, P: ProbeClient>
    Processor for CommitProcessor<E, O, S, P>
{
    type Input = (
        Shard,
        Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        ProbeMetadata,
        PeerPublicKey,
        Multiaddr,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, verified_signed_commit, probe_metadata, peer, address) = msg.input;

            let shard_digest = shard.digest()?;

            match self.recv_dedup.get_value_or_guard(
                &(shard_digest, verified_signed_commit.encoder().clone()),
                Some(Duration::from_secs(5)),
            ) {
                GuardResult::Value(_) => return Err(ShardError::RecvDuplicate),
                GuardResult::Guard(placeholder) => {
                    placeholder.insert(());
                }
                GuardResult::Timeout => (),
            }

            let commit_input = DownloaderInput::new(
                peer.clone(),
                address.clone(),
                verified_signed_commit.commit_metadata().clone(),
            );

            self.downloader
                .process(commit_input, msg.cancellation.clone())
                .await?;

            // let probe_input = DownloaderInput::new(peer, address, probe_metadata.metadata());

            // self.downloader
            //     .process(probe_input, msg.cancellation.clone())
            //     .await?;

            let _ = self
                .store
                .add_signed_commit(&shard, &verified_signed_commit)?;

            let quorum_threshold = shard.quorum_threshold() as usize;
            let max_size = shard.size();
            let count = self.store.count_signed_commits(&shard)?;
            let shard_digest = shard.digest()?;

            info!(
                "Tracking commit from: {:?}",
                verified_signed_commit.clone().committer()
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
                        let commits = match store.get_signed_commits(&shard) {
                            Ok(commits) => commits,
                            Err(e) => {
                                tracing::error!("Error getting signed commits: {:?}", e);
                                return Err(ShardError::MissingData);
                            }
                        };

                        // Format commits for votes
                        let commits = commits
                            .iter()
                            .map(|commit| (commit.encoder().clone(), Digest::new(commit).unwrap()))
                            .collect();

                        // Create votes
                        let votes = ShardCommitVotes::V1(ShardCommitVotesV1::new(
                            verified_signed_commit.auth_token().clone(),
                            encoder_keypair.public(),
                            commits,
                        ));

                        // Sign votes
                        let inner_keypair = encoder_keypair.inner().copy();
                        let signed_votes =
                            Signed::new(votes, Scope::ShardCommitVotes, &inner_keypair.private())
                                .unwrap();
                        let verified = Verified::from_trusted(signed_votes).unwrap();

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
