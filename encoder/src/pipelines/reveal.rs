use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::{ShardStage, Store},
    messaging::EncoderInternalNetworkClient,
    types::reveal::{Reveal, RevealAPI},
};
use async_trait::async_trait;
use dashmap::DashMap;
use intelligence::evaluation::networking::EvaluationClient;
use object_store::ObjectStore;
use std::{collections::HashMap, future::Future, sync::Arc, time::Duration};
use tokio::{sync::oneshot, time::sleep};
use tracing::{debug, info};
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::ShardResult,
    shard::Shard,
    shard_crypto::{
        digest::Digest,
        keys::{EncoderKeyPair, EncoderPublicKey},
        verified::Verified,
    },
    submission::{Submission, SubmissionAPI},
};

use super::evaluation::EvaluationProcessor;

pub(crate) struct RevealProcessor<C: EncoderInternalNetworkClient, E: EvaluationClient> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<C>>,
    oneshots: Arc<DashMap<Digest<Shard>, oneshot::Sender<()>>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    evaluation_pipeline: ActorHandle<EvaluationProcessor<C, E>>,
}

impl<C: EncoderInternalNetworkClient, E: EvaluationClient> RevealProcessor<C, E> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<C>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        evaluation_pipeline: ActorHandle<EvaluationProcessor<C, E>>,
    ) -> Self {
        Self {
            store,
            broadcaster,
            oneshots: Arc::new(DashMap::new()),
            encoder_keypair,
            evaluation_pipeline,
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
impl<C: EncoderInternalNetworkClient, E: EvaluationClient> Processor for RevealProcessor<C, E> {
    type Input = (Shard, Verified<Reveal>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, verified_reveal) = msg.input;
            let _ = self.store.add_shard_stage_message(
                &shard,
                ShardStage::Reveal,
                verified_reveal.author(),
            )?;

            let _ = self
                .store
                .add_submission(&shard, verified_reveal.submission().clone())?;

            let quorum_threshold = shard.quorum_threshold() as usize;
            let shard_digest = shard.digest()?;

            let all_submissions = self.store.get_all_submissions(&shard)?;
            let all_accepted_commits = self.store.get_all_accepted_commits(&shard)?;

            let max_size = all_accepted_commits.len();

            let accepted_lookup: HashMap<EncoderPublicKey, Digest<Submission>> =
                all_accepted_commits
                    .into_iter()
                    .map(|(encoder, digest)| (encoder, digest))
                    .collect();

            let count = all_submissions
                .iter()
                .filter(|(submission, _instant)| {
                    accepted_lookup
                        .get(submission.encoder())
                        .map_or(false, |accepted_digest| {
                            accepted_digest == &Digest::new(submission).unwrap()
                        })
                })
                .count();

            info!(
            "Reveal count statistics - current_reveal_count: {}, quorum_threshold: {}, max_size: \
             {}",
            count, quorum_threshold, max_size
        );

            match count {
                count if count == quorum_threshold => {
                    let _ = self
                        .store
                        .add_shard_stage_dispatch(&shard, ShardStage::Evaluation)?;
                    let earliest = all_submissions.iter().map(|(_, t)| t).min().unwrap();

                    let duration = std::cmp::max(earliest.elapsed(), Duration::from_secs(5));

                    info!(
                    "Quorum of reveals reached - starting vote timer. Duration since first (ms): \
                     {}",
                    duration.as_millis()
                    );

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
