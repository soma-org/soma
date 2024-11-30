use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::storage::blob::{BlobPath, BlobStorage};
use async_trait::async_trait;

use super::{ActorMessage, Processor};

pub(crate) struct StorageProcessor<B: BlobStorage> {
    store: Arc<B>,
    semaphore: Option<Arc<Semaphore>>,
}

impl<B: BlobStorage> StorageProcessor<B> {
    pub fn new(store: B, concurrency: Option<usize>) -> Self {
        let semaphore = concurrency.map(|n| Arc::new(Semaphore::new(n)));
        Self {
            store: Arc::new(store),
            semaphore,
        }
    }
}

pub(crate) enum StorageProcessorInput {
    Store(BlobPath, Bytes),
}

#[async_trait]
impl<B: BlobStorage> Processor for StorageProcessor<B> {
    type Input = StorageProcessorInput;
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        match msg.input {
            StorageProcessorInput::Store(path, contents) => {
                if let Some(sem) = &self.semaphore {
                    if let Ok(permit) = sem.clone().acquire_owned().await {
                        let store: Arc<B> = self.store.clone();
                        tokio::spawn(async move {
                            // TODO: improve by deriving the path from the bytes or pass this in
                            let _ = msg.sender.send(store.put_object(&path, contents).await);
                            drop(permit);
                        });
                    }
                } else {
                    let store: Arc<B> = self.store.clone();
                    let _ = msg.sender.send(store.put_object(&path, contents).await);
                }
            }
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
