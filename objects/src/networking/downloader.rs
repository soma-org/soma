use std::sync::Arc;

use crate::{
    networking::{ClientPool, ObjectClient},
    storage::ObjectStorage,
};
use async_trait::async_trait;
use tokio::sync::Semaphore;
use types::actors::{ActorMessage, Processor};
use types::error::{ShardError, ShardResult};
use types::metadata::{DownloadableMetadata, DownloadableMetadataAPI, MetadataAPI};

pub struct Downloader<P: ClientPool, S: ObjectStorage> {
    client: Arc<ObjectClient<P>>,
    storage: Arc<S>,
    semaphore: Arc<Semaphore>,
}

impl<P: ClientPool, S: ObjectStorage> Downloader<P, S> {
    pub fn new(concurrency: usize, client: Arc<ObjectClient<P>>, storage: Arc<S>) -> Self {
        let semaphore = Arc::new(Semaphore::new(concurrency));
        Self {
            client,
            storage,
            semaphore,
        }
    }
}

#[async_trait]
impl<P: ClientPool, S: ObjectStorage> Processor for Downloader<P, S> {
    type Input = DownloadableMetadata;
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        if let Ok(permit) = self.semaphore.clone().acquire_owned().await {
            let storage = self.storage.clone();
            let client = self.client.clone();
            tokio::spawn(async move {
                let result: ShardResult<()> = async {
                    let downloadable_metadata = msg.input;
                    let object_path = downloadable_metadata.metadata().path();

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
