use async_trait::async_trait;
use std::sync::Arc;

use objects::{
    networking::{downloader::Downloader, internal_service::InternalClientPool},
    storage::ObjectStorage,
};
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::ShardResult,
    evaluation::{EvaluationInput, EvaluationInputAPI, EvaluationOutput},
    metadata::{DownloadableMetadata, DownloadableMetadataV1},
};

use super::safetensor_buffer::SafetensorBuffer;

pub(crate) struct CoreProcessor<S: ObjectStorage + SafetensorBuffer> {
    downloader: ActorHandle<Downloader<InternalClientPool, S>>,
    storage: Arc<S>,
}

impl<S: ObjectStorage + SafetensorBuffer> CoreProcessor<S> {
    pub(crate) fn new(
        downloader: ActorHandle<Downloader<InternalClientPool, S>>,
        storage: Arc<S>,
    ) -> Self {
        Self {
            downloader,
            storage,
        }
    }
}

#[async_trait]
impl<S: ObjectStorage + SafetensorBuffer> Processor for CoreProcessor<S> {
    type Input = EvaluationInput;
    type Output = EvaluationOutput;

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<Self::Output> = async {
            let evaluation_input = msg.input;

            let _ = self
                .downloader
                .process(
                    DownloadableMetadata::V1(DownloadableMetadataV1::new(
                        None,
                        None,
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
                        None,
                        None,
                        evaluation_input.address(),
                        evaluation_input.embeddings(),
                    )),
                    msg.cancellation.clone(),
                )
                .await?;

            // for p in evaluation_input.probe_set().probe_weights() {
            //     let _ = self
            //         .downloader
            //         .process(
            //             DownloadableMetadata::V1(DownloadableMetadataV1::new(
            //                 evaluation_input.tls_key(),
            //                 evaluation_input.address(),
            //             )),
            //             msg.cancellation.clone(),
            //         )
            //         .await?;
            // }

            // let embedding_buffer = self
            //     .storage
            //     .safetensor_buffer(ObjectPath::from_checksum(
            //         evaluation_input.embeddings().checksum(),
            //     ))
            //     .unwrap();

            // let embedding_tensors = SafeTensors::deserialize(embedding_buffer.as_ref()).unwrap();

            // load probe in burn (in safe tensor format)
            // ensure all of this data is downloaded to the connected storage

            unimplemented!();
        }
        .await;
        let _ = msg.sender.send(result);
    }
    fn shutdown(&mut self) {}
}
