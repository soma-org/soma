use async_trait::async_trait;
use safetensors::SafeTensors;
use std::sync::Arc;

use objects::{
    networking::{downloader::Downloader, ObjectNetworkClient},
    storage::{ObjectPath, ObjectStorage},
};
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::ShardResult,
    evaluation::{
        EvaluationInput, EvaluationInputAPI, EvaluationOutput, ProbeSetAPI, ProbeWeightAPI,
    },
    metadata::{DownloadableMetadata, DownloadableMetadataV1, MetadataAPI},
};

use super::safetensor_buffer::SafetensorBuffer;

pub(crate) struct CoreProcessor<O: ObjectNetworkClient, S: ObjectStorage + SafetensorBuffer> {
    downloader: ActorHandle<Downloader<O, S>>,
    storage: Arc<S>,
}

impl<O: ObjectNetworkClient, S: ObjectStorage + SafetensorBuffer> CoreProcessor<O, S> {
    pub(crate) fn new(downloader: ActorHandle<Downloader<O, S>>, storage: Arc<S>) -> Self {
        Self {
            downloader,
            storage,
        }
    }
}

#[async_trait]
impl<O: ObjectNetworkClient, S: ObjectStorage + SafetensorBuffer> Processor
    for CoreProcessor<O, S>
{
    type Input = EvaluationInput;
    type Output = EvaluationOutput;

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<Self::Output> = async {
            let evaluation_input = msg.input;

            let _ = self
                .downloader
                .process(
                    DownloadableMetadata::V1(DownloadableMetadataV1::new(
                        evaluation_input.tls_key(),
                        evaluation_input.address(),
                        evaluation_input.data(),
                    )),
                    msg.cancellation.clone(),
                )
                .await?;

            let _ = self
                .downloader
                .process(
                    DownloadableMetadata::V1(DownloadableMetadataV1::new(
                        evaluation_input.tls_key(),
                        evaluation_input.address(),
                        evaluation_input.embeddings(),
                    )),
                    msg.cancellation.clone(),
                )
                .await?;

            for p in evaluation_input.probe_set().probe_weights() {
                let _ = self
                    .downloader
                    .process(
                        DownloadableMetadata::V1(DownloadableMetadataV1::new(
                            evaluation_input.tls_key(),
                            evaluation_input.address(),
                            p.metadata(),
                        )),
                        msg.cancellation.clone(),
                    )
                    .await?;
            }

            let embedding_buffer = self
                .storage
                .safetensor_buffer(ObjectPath::from_checksum(
                    evaluation_input.embeddings().checksum(),
                ))
                .unwrap();

            let embedding_tensors = SafeTensors::deserialize(embedding_buffer.as_ref()).unwrap();

            // ensure all of this data is downloaded to the connected storage

            unimplemented!();
        }
        .await;
        let _ = msg.sender.send(result);
    }
    fn shutdown(&mut self) {}
}
