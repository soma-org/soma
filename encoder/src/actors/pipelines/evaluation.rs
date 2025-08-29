use std::{sync::Arc, time::Duration};

use crate::{
    actors::workers::storage::StorageProcessor,
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
};
use async_trait::async_trait;
use fastcrypto::traits::KeyPair;
use objects::storage::ObjectStorage;
use probe::{messaging::ProbeClient, EmbeddingV1, ProbeInputV1, ProbeOutputAPI, ScoreAPI};
use quick_cache::sync::{Cache, GuardResult};
use shared::{
    actors::{ActorHandle, ActorMessage, Processor},
    crypto::{
        keys::{EncoderKeyPair, EncoderPublicKey},
        EncryptionKey,
    },
    digest::Digest,
    error::{ShardError, ShardResult},
    metadata::Metadata,
    scope::Scope,
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use types::shard::ShardAuthToken;
use types::shard_scores::{ScoreSet, ScoreSetV1, ScoreV1, ShardScores, ShardScoresV1};

use super::scores::ScoresProcessor;

// use super::broadcast::{BroadcastAction, BroadcastProcessor};

pub(crate) struct EvaluationProcessor<
    E: EncoderInternalNetworkClient,
    S: ObjectStorage,
    P: ProbeClient,
> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<E>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    storage: ActorHandle<StorageProcessor<S>>,
    score_pipeline: ActorHandle<ScoresProcessor<E>>,
    probe_client: Arc<P>,
    recv_dedup: Cache<Digest<Shard>, ()>,
}

impl<E: EncoderInternalNetworkClient, S: ObjectStorage, P: ProbeClient>
    EvaluationProcessor<E, S, P>
{
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        storage: ActorHandle<StorageProcessor<S>>,
        score_pipeline: ActorHandle<ScoresProcessor<E>>,
        probe_client: Arc<P>,
        recv_cache_capacity: usize,
    ) -> Self {
        Self {
            store,
            broadcaster,
            encoder_keypair,
            storage,
            score_pipeline,
            probe_client,
            recv_dedup: Cache::new(recv_cache_capacity),
        }
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient, S: ObjectStorage, P: ProbeClient> Processor
    for EvaluationProcessor<E, S, P>
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

            let embeddings: Vec<EmbeddingV1> = shard
                .encoders()
                .iter()
                .map(|e| {
                    EmbeddingV1::new(
                        e.clone(),
                        e.clone(),
                        Metadata::default(),
                        Metadata::default(),
                        EncryptionKey::default(),
                    )
                })
                .collect();

            let timeout = Duration::from_secs(1);
            let probe_input = probe::ProbeInput::V1(ProbeInputV1::new(embeddings));
            let probe_output = self
                .probe_client
                .probe(probe_input, timeout)
                .await
                .map_err(ShardError::ProbeError)?;

            let scores = probe_output.scores();

            let scores: Vec<ScoreV1> = scores
                .iter()
                .map(|s| ScoreV1::new(s.encoder().clone(), s.rank()))
                .collect();
            let score_set = ScoreSet::V1(ScoreSetV1::new(shard.epoch(), shard.digest()?, scores));

            let inner_keypair = self.encoder_keypair.inner().copy();

            // Sign score set
            let signed_score_set = Signed::new(
                score_set,
                Scope::ShardScores,
                &inner_keypair.copy().private(),
            )
            .unwrap();

            // Create scores
            let scores = ShardScores::V1(ShardScoresV1::new(
                auth_token,
                self.encoder_keypair.public(),
                signed_score_set,
            ));

            // Sign scores
            let signed_scores =
                Signed::new(scores, Scope::ShardScores, &inner_keypair.private()).unwrap();
            let verified = Verified::from_trusted(signed_scores).unwrap();

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
                            .send_scores(&peer, &verified_type, MESSAGE_TIMEOUT)
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
