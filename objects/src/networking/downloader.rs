use std::sync::Arc;

use crate::{
    networking::ObjectNetworkClient,
    storage::{ObjectPath, ObjectStorage},
};
use async_trait::async_trait;
use shared::{
    actors::{ActorMessage, Processor},
    metadata::DownloadableMetadata,
};
use shared::{
    crypto::keys::PeerPublicKey,
    metadata::{Metadata, MetadataAPI},
};
use shared::{
    error::{ShardError, ShardResult},
    metadata::DownloadableMetadataAPI,
};
use soma_network::multiaddr::Multiaddr;
use tokio::sync::Semaphore;

pub struct Downloader<C: ObjectNetworkClient, S: ObjectStorage> {
    client: Arc<C>,
    storage: Arc<S>,
    semaphore: Arc<Semaphore>,
}

impl<C: ObjectNetworkClient, S: ObjectStorage> Downloader<C, S> {
    pub fn new(concurrency: usize, client: Arc<C>, storage: Arc<S>) -> Self {
        let semaphore = Arc::new(Semaphore::new(concurrency));
        Self {
            client,
            storage,
            semaphore,
        }
    }
}

#[async_trait]
impl<C: ObjectNetworkClient, S: ObjectStorage> Processor for Downloader<C, S> {
    type Input = DownloadableMetadata;
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        if let Ok(permit) = self.semaphore.clone().acquire_owned().await {
            let storage = self.storage.clone();
            let client = self.client.clone();
            tokio::spawn(async move {
                let result: ShardResult<()> = async {
                    let downloadable_metadata = msg.input;
                    let object_path =
                        ObjectPath::from_checksum(downloadable_metadata.metadata().checksum());

                    // check if the object exists
                    // TODO: explicitly match on the not found error only
                    if storage.exists(&object_path).await.is_ok() {
                        // if it exists, no need to download

                        return Ok(());
                    }

                    // get an object writer for the storage backend
                    let mut writer = storage
                        .get_object_writer(&object_path)
                        .await
                        .map_err(ShardError::ObjectError)?;

                    // download the object, streaming it directly into storage
                    if let Err(e) = client
                        .download_object(&mut writer, &downloadable_metadata)
                        .await
                    {
                        tracing::error!("Error downloading object! Delete and abort.");
                        // if there is an error, delete the object
                        storage
                            .delete_object(&object_path)
                            .await
                            .map_err(ShardError::ObjectError)?;
                        return Err(ShardError::ObjectError(e));
                    }

                    Ok(())
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
