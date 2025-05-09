use std::sync::Arc;

use bytes::Bytes;
use objects::storage::{ObjectPath, ObjectStorage};
use tokio::sync::Semaphore;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
};

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
                    match msg.input {
                        StorageProcessorInput::Store(path, contents) => {
                            let result = store
                                .put_object(&path, contents)
                                .await
                                .map(|_| StorageProcessorOutput::Store())
                                .map_err(ShardError::ObjectError);

                            let _ = msg.sender.send(result);
                        }
                        StorageProcessorInput::Get(path) => {
                            let result = store
                                .get_object(&path)
                                .await
                                .map(StorageProcessorOutput::Get)
                                .map_err(ShardError::ObjectError);
                            let _ = msg.sender.send(result);
                        }
                    }
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
                        .map(|_| StorageProcessorOutput::Store())
                        .map_err(ShardError::ObjectError);

                    let _ = msg.sender.send(result);
                }
                StorageProcessorInput::Get(path) => {
                    let result = self
                        .store
                        .get_object(&path)
                        .await
                        .map(StorageProcessorOutput::Get)
                        .map_err(ShardError::ObjectError);
                    let _ = msg.sender.send(result);
                }
            }
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}

impl<B: ObjectStorage> ActorHandle<StorageProcessor<B>> {
    pub(crate) async fn store(
        &self,
        object_path: ObjectPath,
        bytes: Bytes,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        let input = StorageProcessorInput::Store(object_path, bytes);
        let x = self.process(input, cancellation).await?;
        Ok(())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::actors::ActorManager;
//     use crate::error::{ShardError, ShardResult};
//     use async_trait::async_trait;
//     use bytes::Bytes;
//     use std::sync::Arc;
//     use std::time::Duration;
//     use tokio::time::sleep;
//     use tokio_util::sync::CancellationToken;

//     // Mock ObjectStorage implementation for testing
//     struct MockStorage {
//         should_fail: bool,
//     }

//     #[async_trait]
//     impl ObjectStorage for MockStorage {
//         async fn put_object(&self, path: &ObjectPath, contents: Bytes) -> ShardResult<()> {
//             if self.should_fail {
//                 Err(ShardError::ObjectStorage("tt".to_string()))
//             } else {
//                 Ok(())
//             }
//         }

//         async fn get_object(&self, path: &ObjectPath) -> ShardResult<Bytes> {
//             if self.should_fail {
//                 Err(ShardError::ObjectStorage("tt".to_string()))
//             } else {
//                 Ok(Bytes::from("test data"))
//             }
//         }
//         async fn delete_object(&self, path: &ObjectPath) -> ShardResult<()> {
//             Ok(())
//         }
//     }

//     #[tokio::test]
//     async fn test_storage_processor_basic() {
//         // Test basic store and get operations without concurrency limit
//         let storage = Arc::new(MockStorage { should_fail: false });
//         let processor = StorageProcessor::new(storage, None);
//         let manager = ActorManager::new(1, processor);
//         let handle = manager.handle();
//         let cancellation_token = CancellationToken::new();

//         let test_path = ObjectPath::new("test/path".to_string()).unwrap();
//         let test_data = Bytes::from("test data");

//         // Test store operation
//         let store_result = handle
//             .process(
//                 StorageProcessorInput::Store(test_path.clone(), test_data.clone()),
//                 cancellation_token.clone(),
//             )
//             .await;
//         assert!(store_result.is_ok());
//         if let Ok(StorageProcessorOutput::Store()) = store_result {
//             // Success case
//         } else {
//             panic!("Expected Store output");
//         }

//         // Test get operation
//         let get_result = handle
//             .process(
//                 StorageProcessorInput::Get(test_path.clone()),
//                 cancellation_token.clone(),
//             )
//             .await;
//         assert!(get_result.is_ok());
//         if let Ok(StorageProcessorOutput::Get(data)) = get_result {
//             assert_eq!(data, Bytes::from("test data"));
//         } else {
//             panic!("Expected Get output");
//         }

//         manager.shutdown();
//         sleep(Duration::from_millis(100)).await;
//     }

//     #[tokio::test]
//     async fn test_storage_processor_concurrent() {
//         // Test concurrent operations with semaphore
//         let storage = Arc::new(MockStorage { should_fail: false });
//         let processor = StorageProcessor::new(storage, Some(2)); // Limit to 2 concurrent operations
//         let manager = ActorManager::new(1, processor);
//         let handle = manager.handle();
//         let cancellation_token = CancellationToken::new();

//         let test_path1 = ObjectPath::new("test/path1".to_string()).unwrap();
//         let test_path2 = ObjectPath::new("test/path2".to_string()).unwrap();
//         let test_data = Bytes::from("test data");

//         // Launch multiple store operations
//         let store1 = handle.process(
//             StorageProcessorInput::Store(test_path1, test_data.clone()),
//             cancellation_token.clone(),
//         );
//         let store2 = handle.process(
//             StorageProcessorInput::Store(test_path2, test_data.clone()),
//             cancellation_token.clone(),
//         );

//         // Wait for both to complete
//         let (result1, result2) = tokio::join!(store1, store2);
//         assert!(result1.is_ok());
//         assert!(result2.is_ok());

//         manager.shutdown();
//         sleep(Duration::from_millis(100)).await;
//     }

//     #[tokio::test]
//     async fn test_storage_processor_error_handling() {
//         // Test error cases
//         let storage = Arc::new(MockStorage { should_fail: true });
//         let processor = StorageProcessor::new(storage, None);
//         let manager = ActorManager::new(1, processor);
//         let handle = manager.handle();
//         let cancellation_token = CancellationToken::new();

//         let test_path = ObjectPath::new("test/path".to_string()).unwrap();
//         let test_data = Bytes::from("test data");

//         // Test store operation with failure
//         let store_result = handle
//             .process(
//                 StorageProcessorInput::Store(test_path.clone(), test_data),
//                 cancellation_token.clone(),
//             )
//             .await;
//         assert!(store_result.is_err());

//         // Test get operation with failure
//         let get_result = handle
//             .process(
//                 StorageProcessorInput::Get(test_path),
//                 cancellation_token.clone(),
//             )
//             .await;
//         assert!(get_result.is_err());

//         manager.shutdown();
//         sleep(Duration::from_millis(100)).await;
//     }
// }
