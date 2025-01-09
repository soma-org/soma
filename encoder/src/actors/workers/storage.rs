use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::storage::object::{ObjectPath, ObjectStorage};
use async_trait::async_trait;

use crate::actors::{ActorMessage, Processor};

pub(crate) struct StorageProcessor<B: ObjectStorage> {
    store: Arc<B>,
    semaphore: Option<Arc<Semaphore>>,
}

impl<B: ObjectStorage> StorageProcessor<B> {
    pub fn new(store: Arc<B>, concurrency: Option<usize>) -> Self {
        let semaphore = concurrency.map(|n| Arc::new(Semaphore::new(n)));
        Self { store, semaphore }
    }
}

pub(crate) enum StorageProcessorInput {
    Store(ObjectPath, Bytes),
    Get(ObjectPath),
}

pub(crate) enum StorageProcessorOutput {
    Store(),
    Get(Bytes),
}
#[async_trait]
impl<B: ObjectStorage> Processor for StorageProcessor<B> {
    type Input = StorageProcessorInput;
    type Output = StorageProcessorOutput;

    async fn process(&self, msg: ActorMessage<Self>) {
        if let Some(sem) = &self.semaphore {
            if let Ok(permit) = sem.clone().acquire_owned().await {
                let store: Arc<B> = self.store.clone();
                tokio::spawn(async move {
                    // match msg.input {
                    //     StorageProcessorInput::Store(path, contents) => {
                    //     },
                    //     StorageProcessorInput::Get(path) => {

                    //     },
                    // }
                    drop(permit);
                });
            }
        } else {
            match msg.input {
                StorageProcessorInput::Store(path, contents) => {
                    let result = self
                        .store
                        .put_object(&path, contents)
                        .await
                        .map(|_| StorageProcessorOutput::Store());

                    let _ = msg.sender.send(result);
                }
                StorageProcessorInput::Get(path) => {
                    let result = self
                        .store
                        .get_object(&path)
                        .await
                        .map(|bytes| StorageProcessorOutput::Get(bytes));
                    let _ = msg.sender.send(result);
                }
            }
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
