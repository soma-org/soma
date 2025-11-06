use std::sync::Arc;

use async_trait::async_trait;
use object_store::ObjectStore;
use tokio::sync::Semaphore;
use types::actors::{ActorMessage, Processor};
use types::error::{ShardError, ShardResult};
use types::metadata::{DownloadMetadata, ObjectPath};

use crate::networking::DownloadClient;

pub struct Downloader {
    client: Arc<DownloadClient>,
    semaphore: Arc<Semaphore>,
}

impl Downloader {
    pub fn new(concurrency: usize, client: Arc<DownloadClient>) -> Self {
        let semaphore = Arc::new(Semaphore::new(concurrency));
        Self { client, semaphore }
    }
}

#[async_trait]
impl Processor for Downloader {
    type Input = (DownloadMetadata, ObjectPath, Arc<dyn ObjectStore>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        if let Ok(permit) = self.semaphore.clone().acquire_owned().await {
            let client = self.client.clone();
            tokio::spawn(async move {
                let result: ShardResult<()> = async {
                    let (download_metadata, path, storage) = msg.input;
                    if let Err(e) = client.download(storage, path, &download_metadata).await {
                        tracing::error!("Error downloading object! Delete and abort.");
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
