use std::sync::Arc;

use crate::{
    actors::{workers::storage::StorageProcessor, ActorHandle, ActorMessage, Processor},
    core::{internal_broadcaster::Broadcaster, shard_tracker::ShardTracker},
    datastore::Store,
    error::ShardResult,
    messaging::EncoderInternalNetworkClient,
    types::{
        shard::Shard,
        shard_scores::{ScoreSet, ScoreSetV1, ScoreV1, ShardScores},
        shard_verifier::ShardAuthToken,
    },
};
use async_trait::async_trait;
use objects::storage::ObjectStorage;
use shared::crypto::keys::EncoderKeyPair;

use super::broadcast::{BroadcastAction, BroadcastProcessor};

pub(crate) struct EvaluationProcessor<C: EncoderInternalNetworkClient, S: ObjectStorage> {
    store: Arc<dyn Store>,
    broadcast_handle: ActorHandle<BroadcastProcessor<C, S>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    storage: ActorHandle<StorageProcessor<S>>,
}

impl<C: EncoderInternalNetworkClient, S: ObjectStorage> EvaluationProcessor<C, S> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcast_handle: ActorHandle<BroadcastProcessor<C, S>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        storage: ActorHandle<StorageProcessor<S>>,
    ) -> Self {
        Self {
            store,
            broadcast_handle,
            encoder_keypair,
            storage,
        }
    }
}

#[async_trait]
impl<C: EncoderInternalNetworkClient, S: ObjectStorage> Processor for EvaluationProcessor<C, S> {
    type Input = (ShardAuthToken, Shard);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (auth_token, shard) = msg.input;

            let mut encoders = shard.encoders();
            encoders.sort();

            let scores = encoders
                .iter()
                .enumerate()
                .map(|(i, e)| ScoreV1::new(e.to_owned(), i as u8))
                .collect();

            let score_set = ScoreSet::V1(ScoreSetV1::new(shard.epoch(), shard.digest()?, scores));

            // create score set
            self.broadcast_handle
                .process(
                    BroadcastAction::Scores(shard, auth_token, score_set),
                    msg.cancellation.clone(),
                )
                .await?;
            // let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
            // let epoch = auth_token.epoch();

            // let accepted_slots = self.store.get_filled_reveal_slots(epoch, shard_ref);

            // // let device = NdArrayDevice::Cpu;
            // for slot in accepted_slots {
            //     let certified_commit = self.store.get_certified_commit(epoch, shard_ref, slot)?;
            //     let committer = certified_commit.committer();
            // let probe_metadata = self
            //     .context
            //     .inner()
            //     .current_committees()
            //     .encoder_committee
            //     .encoder(committer)
            //     .probe
            //     .clone();
            // let probe_path = ObjectPath::from_checksum(probe_metadata.checksum());
            // let (_, embedding_checksum) = self.store.get_reveal(epoch, shard_ref, slot)?;
            // let embedding_path = ObjectPath::from_checksum(embedding_checksum);

            // let probe_bytes = match self
            //     .storage
            //     .process(
            //         StorageProcessorInput::Get(probe_path),
            //         msg.cancellation.clone(),
            //     )
            //     .await?
            // {
            //     StorageProcessorOutput::Get(bytes) => bytes,
            //     _ => return Err(ShardError::MissingData),
            // };

            // let serialized_probe: SerializedProbe =
            //     bcs::from_bytes(&probe_bytes).map_err(ShardError::MalformedType)?;
            // let probe: Probe<NdArray> = serialized_probe.to_probe(&device);

            // let embedding_bytes = match self
            //     .storage
            //     .process(
            //         StorageProcessorInput::Get(embedding_path),
            //         msg.cancellation.clone(),
            //     )
            //     .await?
            // {
            //     StorageProcessorOutput::Get(bytes) => bytes,
            //     _ => return Err(ShardError::MissingData),
            // };

            // let embeddings: Array2<f32> =
            //     bcs::from_bytes(&embedding_bytes).map_err(ShardError::MalformedType)?;

            // let _i = probe.reconstruction(&device, embeddings);
            // }
            // self.store.fi

            // get all finalized slots that are accepted from the store
            // get the checksum for the revealed data from the store and get that object
            // get the probe checksum for the corresponding slots
            // load the revealed embeddings from object storage
            // load the probe from object storage
            // run the embeddings through the probe
            // get a final score using a codified loss function applied to the original piece of data and probe outputs
            // do this for all valid embeddings
            // package scores
            // broadcast scores
            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
