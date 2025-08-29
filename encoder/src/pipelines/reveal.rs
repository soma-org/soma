use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::reveal::{Reveal, RevealAPI},
};
use async_trait::async_trait;
use dashmap::DashMap;
use evaluation::messaging::EvaluationClient;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use objects::{networking::ObjectNetworkClient, storage::ObjectStorage};
use quick_cache::sync::{Cache, GuardResult};
use shared::{
    actors::{ActorHandle, ActorMessage, Processor},
    crypto::keys::{EncoderKeyPair, EncoderPublicKey},
    digest::Digest,
    error::{ShardError, ShardResult},
    scope::Scope,
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use std::{future::Future, sync::Arc, time::Duration};
use tokio::{sync::oneshot, time::sleep};
use tracing::{debug, info};

use super::evaluation::EvaluationProcessor;

pub(crate) struct RevealProcessor<
    O: ObjectNetworkClient,
    E: EncoderInternalNetworkClient,
    S: ObjectStorage,
    P: EvaluationClient,
> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<E>>,
    oneshots: Arc<DashMap<Digest<Shard>, oneshot::Sender<()>>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    evaluation_pipeline: ActorHandle<EvaluationProcessor<O, E, S, P>>,
    recv_dedup: Cache<(Digest<Shard>, EncoderPublicKey), ()>,
    send_dedup: Cache<Digest<Shard>, ()>,
}

impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > RevealProcessor<O, E, S, P>
{
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        evaluation_pipeline: ActorHandle<EvaluationProcessor<O, E, S, P>>,
        recv_cache_capacity: usize,
        send_cache_capacity: usize,
    ) -> Self {
        Self {
            store,
            broadcaster,
            oneshots: Arc::new(DashMap::new()),
            encoder_keypair,
            evaluation_pipeline,
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
    > Processor for RevealProcessor<O, E, S, P>
{
    type Input = (Shard, Verified<Signed<Reveal, min_sig::BLS12381Signature>>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, verified_reveal) = msg.input;
            let shard_digest = shard.digest()?;

            match self.recv_dedup.get_value_or_guard(
                &(shard_digest, verified_reveal.author().clone()),
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
                verified_reveal.author()
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
                    let evaluation_pipeline = self.evaluation_pipeline.clone();
                    debug!("Broadcasting reveal vote after timer");
                    let input = (verified_reveal.auth_token().to_owned(), shard.clone());
                    self.start_timer(shard_digest, duration, async move || {
                        evaluation_pipeline
                            .process(input, msg.cancellation.clone())
                            .await?;

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
