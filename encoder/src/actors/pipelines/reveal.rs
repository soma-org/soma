use crate::{
    actors::{ActorHandle, ActorMessage, Processor},
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    error::{ShardError, ShardResult},
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        shard::Shard,
        shard_reveal::{ShardReveal, ShardRevealAPI},
        shard_reveal_votes::{ShardRevealVotes, ShardRevealVotesV1},
    },
};
use async_trait::async_trait;
use dashmap::DashMap;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use objects::storage::ObjectStorage;
use probe::messaging::ProbeClient;
use quick_cache::sync::{Cache, GuardResult};
use shared::{
    crypto::keys::{EncoderKeyPair, EncoderPublicKey},
    digest::Digest,
    scope::Scope,
    signed::Signed,
    verified::Verified,
};
use std::{future::Future, sync::Arc, time::Duration};
use tokio::{sync::oneshot, time::sleep};
use tracing::{debug, info};

use super::reveal_votes::RevealVotesProcessor;

pub(crate) struct RevealProcessor<E: EncoderInternalNetworkClient, S: ObjectStorage, P: ProbeClient>
{
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<E>>,
    oneshots: Arc<DashMap<Digest<Shard>, oneshot::Sender<()>>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    reveal_votes_pipeline: ActorHandle<RevealVotesProcessor<E, S, P>>,
    recv_dedup: Cache<(Digest<Shard>, EncoderPublicKey), ()>,
    send_dedup: Cache<Digest<Shard>, ()>,
}

impl<E: EncoderInternalNetworkClient, S: ObjectStorage, P: ProbeClient> RevealProcessor<E, S, P> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        reveal_votes_pipeline: ActorHandle<RevealVotesProcessor<E, S, P>>,
        recv_cache_capacity: usize,
        send_cache_capacity: usize,
    ) -> Self {
        Self {
            store,
            broadcaster,
            oneshots: Arc::new(DashMap::new()),
            encoder_keypair,
            reveal_votes_pipeline,
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
impl<E: EncoderInternalNetworkClient, S: ObjectStorage, P: ProbeClient> Processor
    for RevealProcessor<E, S, P>
{
    type Input = (
        Shard,
        Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, verified_reveal) = msg.input;
            let shard_digest = shard.digest()?;

            match self.recv_dedup.get_value_or_guard(
                &(shard_digest, verified_reveal.encoder().clone()),
                Some(Duration::from_secs(5)),
            ) {
                GuardResult::Value(_) => return Err(ShardError::RecvDuplicate),
                GuardResult::Guard(placeholder) => {
                    placeholder.insert(());
                }
                GuardResult::Timeout => (),
            }

            self.store.add_signed_reveal(&shard, &verified_reveal)?;

            info!(
                "Starting track_valid_reveal for revealer: {:?}",
                verified_reveal.encoder()
            );

            let quorum_threshold = shard.quorum_threshold() as usize;
            let max_size = shard.size();
            let count = self.store.count_signed_reveal(&shard)?;
            let shard_digest = shard.digest()?;

            info!(
            "Reveal count statistics - current_reveal_count: {}, quorum_threshold: {}, max_size: \
             {}",
            count, quorum_threshold, max_size
        );

            match count {
                1 => {
                    info!("First reveal received - recording timestamp");
                    self.store.add_first_reveal_time(&shard)?;
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
                    let mut duration = self
                        .store
                        .get_first_reveal_time(&shard)
                        .map_or(Duration::from_secs(60), |first_reveal_time| {
                            first_reveal_time.elapsed()
                        });
                    duration = std::cmp::max(duration, Duration::from_secs(5));

                    info!(
                    "Quorum of reveals reached - starting vote timer. Duration since first (ms): \
                     {}",
                    duration.as_millis()
                    );

                    let broadcaster = self.broadcaster.clone();
                    let store = self.store.clone();
                    let encoder_keypair = self.encoder_keypair.clone();
                    let reveal_vote_pipeline = self.reveal_votes_pipeline.clone();
                    debug!("Broadcasting reveal vote after timer");
                    self.start_timer(shard_digest, duration, async move || {
                        // Get reveals from store
                        let reveals = match store.get_signed_reveals(&shard) {
                            Ok(reveals) => reveals,
                            Err(e) => {
                                tracing::error!("Error getting signed reveals: {:?}", e);
                                return Err(ShardError::MissingData);
                            }
                        };

                        // Format reveals for votes
                        let reveals = reveals
                            .iter()
                            .map(|reveal| reveal.encoder().clone())
                            .collect();

                        // Create votes
                        let votes = ShardRevealVotes::V1(ShardRevealVotesV1::new(
                            verified_reveal.auth_token().clone(),
                            encoder_keypair.public(),
                            reveals,
                        ));

                        // Sign votes
                        let inner_keypair = encoder_keypair.inner().copy();
                        let signed_votes =
                            Signed::new(votes, Scope::ShardRevealVotes, &inner_keypair.private())
                                .unwrap();
                        let verified = Verified::from_trusted(signed_votes).unwrap();

                        // send to pipeline
                        reveal_vote_pipeline
                            .process((shard.clone(), verified.clone()), msg.cancellation.clone())
                            .await?;

                        // Broadcast to other encoders
                        let res = broadcaster
                            .broadcast(
                                verified.clone(),
                                shard.encoders(),
                                |client, peer, verified_type| async move {
                                    client
                                        .send_reveal_votes(&peer, &verified_type, MESSAGE_TIMEOUT)
                                        .await?;
                                    Ok(())
                                },
                            )
                            .await;

                        Ok(())
                    })
                    .await;
                }
                count if count == max_size => {
                    info!("Maximum reveals received - triggering immediate vote");
                    let key = shard_digest;
                    if let Some((_key, tx)) = self.oneshots.remove(&key) {
                        debug!("Sending oneshot to trigger immediate vote");
                        let _ = tx.send(());
                        info!("Successfully triggered immediate vote through oneshot");
                    } else {
                        debug!("No timer found to cancel - vote may have already started");
                    }
                }
                _ => {
                    debug!(
                        "Collecting more reveals before taking action. Current count: {}, \
                     quorum_threshold: {}",
                        count, quorum_threshold
                    );
                }
            };

            info!("Completed track_valid_reveal");
            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
