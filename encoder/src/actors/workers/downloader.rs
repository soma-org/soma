use std::{path, sync::Arc};

use bytes::Bytes;
use shared::{
    checksum::Checksum,
    metadata::{Metadata, MetadataAPI, MetadataCommitment},
    network_committee::NetworkingIndex,
};
use tokio::sync::Semaphore;

use crate::{
    error::{ShardError, ShardResult},
    networking::object::{http_network::ObjectHttpClient, ObjectNetworkClient, GET_OBJECT_TIMEOUT},
    storage::object::ObjectPath,
    types::encoder_committee::{EncoderIndex, Epoch},
};
use async_trait::async_trait;

use crate::actors::{ActorMessage, Processor};

pub(crate) struct DownloaderInput {
    epoch: Epoch,
    peer: EncoderIndex,
    metadata: Metadata,
}

impl DownloaderInput {
    pub(crate) fn new(epoch: Epoch, peer: EncoderIndex, metadata: Metadata) -> Self {
        Self {
            epoch,
            peer,
            metadata,
        }
    }
}

pub(crate) struct Downloader<B: ObjectNetworkClient> {
    semaphore: Arc<Semaphore>,
    client: Arc<B>,
}

impl<B: ObjectNetworkClient> Downloader<B> {
    pub(crate) fn new(concurrency: usize, client: Arc<B>) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(concurrency)),
            client,
        }
    }
}

#[async_trait]
impl<B: ObjectNetworkClient> Processor for Downloader<B> {
    type Input = DownloaderInput;
    type Output = Bytes;

    async fn process(&self, msg: ActorMessage<Self>) {
        let client = self.client.clone();
        if let Ok(permit) = self.semaphore.clone().acquire_owned().await {
            tokio::spawn(async move {
                let result: ShardResult<Bytes> = async {
                    let input = msg.input;
                    let object = client
                        .get_object(input.epoch, input.peer, &input.metadata, GET_OBJECT_TIMEOUT)
                        .await?;
                    if Checksum::new_from_bytes(&object) != input.metadata.checksum() {
                        return Err(ShardError::ObjectValidation(
                            "object checksum does not match metadata".to_string(),
                        ));
                    };

                    if object.len() != input.metadata.size() {
                        return Err(ShardError::ObjectValidation(
                            "object size does not match metadata".to_string(),
                        ));
                    }

                    Ok(object)
                }
                .await;
                let _ = msg.sender.send(result);
                drop(permit);
            });
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
