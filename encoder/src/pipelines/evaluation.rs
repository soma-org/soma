use std::{sync::Arc, time::Duration};

use crate::{
    // actors::workers::storage::StorageProcessor,
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        reveal::{verify_reveal_score_matches, Reveal, RevealAPI},
        score_vote::{ScoreVote, ScoreVoteV1},
    },
};
use async_trait::async_trait;
use evaluation::{
    messaging::EvaluationClient, EvaluationInput, EvaluationInputV1, EvaluationOutputAPI,
    EvaluationScoreAPI, ProbeSetAPI, ProbeWeightAPI,
};
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use objects::{
    networking::{downloader::Downloader, ObjectNetworkClient},
    storage::ObjectStorage,
};
use quick_cache::sync::{Cache, GuardResult};
use shared::{
    actors::{ActorHandle, ActorMessage, Processor},
    crypto::keys::EncoderKeyPair,
    digest::Digest,
    error::{ShardError, ShardResult},
    metadata::{DownloadableMetadataAPI, Metadata},
    scope::Scope,
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use tokio_util::sync::CancellationToken;
use types::{
    score_set::{ScoreSet, ScoreSetV1},
    shard::ShardAuthToken,
};

use super::score_vote::ScoreVoteProcessor;

// use super::broadcast::{BroadcastAction, BroadcastProcessor};

pub(crate) struct EvaluationProcessor<
    O: ObjectNetworkClient,
    E: EncoderInternalNetworkClient,
    S: ObjectStorage,
    P: EvaluationClient,
> {
    store: Arc<dyn Store>,
    downloader: ActorHandle<Downloader<O, S>>,
    broadcaster: Arc<Broadcaster<E>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    storage: Arc<S>,
    score_pipeline: ActorHandle<ScoreVoteProcessor<E>>,
    evaluation_client: Arc<P>,
    recv_dedup: Cache<Digest<Shard>, ()>,
}

impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > EvaluationProcessor<O, E, S, P>
{
    pub(crate) fn new(
        store: Arc<dyn Store>,
        downloader: ActorHandle<Downloader<O, S>>,
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        storage: Arc<S>,
        score_pipeline: ActorHandle<ScoreVoteProcessor<E>>,
        evaluation_client: Arc<P>,
        recv_cache_capacity: usize,
    ) -> Self {
        Self {
            store,
            downloader,
            broadcaster,
            encoder_keypair,
            storage,
            score_pipeline,
            evaluation_client,
            recv_dedup: Cache::new(recv_cache_capacity),
        }
    }
}

#[async_trait]
impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > Processor for EvaluationProcessor<O, E, S, P>
{
    type Input = (ShardAuthToken, Shard);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (auth_token, shard) = msg.input;
            let shard_digest = shard.digest()?;
            match self
                .recv_dedup
                .get_value_or_guard(&(shard_digest), Some(Duration::from_secs(5)))
            {
                GuardResult::Value(_) => return Err(ShardError::RecvDuplicate),
                GuardResult::Guard(placeholder) => {
                    placeholder.insert(());
                }
                GuardResult::Timeout => (),
            }

            let mut reveals = self.store.get_reveals(&shard)?;
            reveals.sort_by(|a, b| {
                a.score()
                    .value()
                    .partial_cmp(&b.score().value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_reveal = self
                .process_reveals(
                    reveals,
                    auth_token
                        .metadata_commitment()
                        .downloadable_metadata()
                        .metadata(),
                    msg.cancellation.clone(),
                )
                .await?;

            let score_set = ScoreSet::V1(ScoreSetV1::new(
                best_reveal.author().clone(),
                best_reveal.score().clone(),
                best_reveal.summary_embedding(),
                best_reveal.probe_set().clone(),
                shard.digest()?,
            ));

            let inner_keypair = self.encoder_keypair.inner().copy();

            // Sign score set
            let signed_score_set =
                Signed::new(score_set, Scope::ScoreSet, &inner_keypair.copy().private()).unwrap();

            // Create scores
            let score_vote = ScoreVote::V1(ScoreVoteV1::new(
                auth_token,
                self.encoder_keypair.public(),
                signed_score_set,
            ));

            let verified = Verified::from_trusted(score_vote).unwrap();

            self.score_pipeline
                .process((shard.clone(), verified.clone()), msg.cancellation.clone())
                .await?;

            // Broadcast to other encoders
            self.broadcaster
                .broadcast(
                    verified.clone(),
                    shard.encoders(),
                    |client, peer, verified_type| async move {
                        client
                            .send_score_vote(&peer, &verified_type, MESSAGE_TIMEOUT)
                            .await?;
                        Ok(())
                    },
                )
                .await?;
            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}

impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > EvaluationProcessor<O, E, S, P>
{
    async fn process_reveals(
        &self,
        reveals: Vec<Reveal>,
        data_metadata: Metadata,
        cancellation: CancellationToken,
    ) -> ShardResult<Reveal> {
        for reveal in reveals {
            // TODO: skip early if your own representations
            // download the representations
            self.downloader
                .process(reveal.tensors().clone(), cancellation.clone())
                .await?;

            for probe in reveal.probe_set().probe_weights() {
                // download probes
                self.downloader
                    .process(probe.downloadable_metadata(), cancellation.clone())
                    .await?;
            }

            let evaluation_input = EvaluationInput::V1(EvaluationInputV1::new(
                data_metadata.clone(),
                reveal.tensors().metadata(),
                reveal.probe_set().clone(),
            ));
            let evaluation_timeout = Duration::from_secs(1);

            // pass into the evaluation step
            let evaluation_output = self
                .evaluation_client
                .evaluation(evaluation_input, evaluation_timeout)
                .await
                .map_err(ShardError::EvaluationError)?;

            if verify_reveal_score_matches(
                evaluation_output.score(),
                evaluation_output.summary_embedding(),
                &reveal,
            )
            .is_ok()
            {
                return Ok(reveal); // Short-circuit and return the first valid reveal
            }
        }

        // Return an error if no reveal passes verification
        Err(ShardError::ShardFailure(
            "no reveals were valid".to_string(),
        ))
    }
}
