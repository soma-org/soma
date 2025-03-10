use std::sync::Arc;

use crate::{
    actors::{
        workers::{
            broadcaster::BroadcasterProcessor,
            storage::{StorageProcessor, StorageProcessorInput, StorageProcessorOutput},
        },
        ActorHandle, ActorMessage, Processor,
    },
    error::{ShardError, ShardResult},
    networking::messaging::EncoderInternalNetworkClient,
    storage::{
        datastore::Store,
        object::{ObjectPath, ObjectStorage},
    },
    types::{
        encoder_context::EncoderContext, shard::Shard, shard_commit::ShardCommitAPI,
        shard_verifier::ShardAuthToken,
    },
};
use async_trait::async_trait;
use burn::backend::{ndarray::NdArrayDevice, NdArray};
use ndarray::Array2;
use probe::{Probe, ProbeAPI, SerializedProbe, SerializedProbeAPI};
use shared::{digest::Digest, metadata::MetadataAPI};

pub(crate) struct EvaluationProcessor<E: EncoderInternalNetworkClient, S: ObjectStorage> {
    context: Arc<EncoderContext>,
    store: Arc<dyn Store>,
    broadcaster: ActorHandle<BroadcasterProcessor<E>>,
    storage: ActorHandle<StorageProcessor<S>>,
}

impl<E: EncoderInternalNetworkClient, S: ObjectStorage> EvaluationProcessor<E, S> {
    pub(crate) fn new(
        context: Arc<EncoderContext>,
        store: Arc<dyn Store>,
        broadcaster: ActorHandle<BroadcasterProcessor<E>>,
        storage: ActorHandle<StorageProcessor<S>>,
    ) -> Self {
        Self {
            context,
            store,
            broadcaster,
            storage,
        }
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient, S: ObjectStorage> Processor for EvaluationProcessor<E, S> {
    type Input = (ShardAuthToken, Shard);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (auth_token, shard) = msg.input;
            let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
            let epoch = shard.epoch();

            let accepted_slots = self.store.get_filled_reveal_slots(epoch, shard_ref);

            // let device = NdArrayDevice::Cpu;
            for slot in accepted_slots {
                let certified_commit = self.store.get_certified_commit(epoch, shard_ref, slot)?;
                let committer = certified_commit.committer();
                let probe_metadata = self
                    .context
                    .encoder_committee
                    .encoder(committer)
                    .probe
                    .clone();
                let probe_path = ObjectPath::from_checksum(probe_metadata.checksum());
                let (_, embedding_checksum) = self.store.get_reveal(epoch, shard_ref, slot)?;
                let embedding_path = ObjectPath::from_checksum(embedding_checksum);

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
            }
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
