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

#[derive(Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{actors::ActorManager, error::ShardError};
    use async_trait::async_trait;
    use bytes::Bytes;
    use shared::checksum::Checksum;
    use std::{sync::Arc, time::Duration};
    use tokio::sync::{oneshot, Semaphore};
    use tokio_util::sync::CancellationToken;

    // Mock ObjectNetworkClient
    struct MockClient {
        return_bytes: Bytes,
        should_fail: bool,
    }

    #[async_trait]
    impl ObjectNetworkClient for MockClient {
        async fn get_object(
            &self,
            _epoch: Epoch,
            _peer: EncoderIndex,
            _metadata: &Metadata,
            _timeout: Duration,
        ) -> ShardResult<Bytes> {
            if self.should_fail {
                Err(ShardError::NetworkRequest("mock failure".to_string()))
            } else {
                Ok(self.return_bytes.clone())
            }
        }
    }

    // Helper to create test Downloader with ActorManager
    fn setup_downloader(
        return_bytes: Bytes,
        should_fail: bool,
    ) -> (ActorManager<Downloader<MockClient>>, Bytes) {
        let client = Arc::new(MockClient {
            return_bytes: return_bytes.clone(),
            should_fail,
        });
        let downloader = Downloader::new(1, client);
        let manager = ActorManager::new(10, downloader);
        (manager, return_bytes)
    }

    #[tokio::test]
    async fn test_successful_download_and_validation() {
        // Setup
        let test_bytes = Bytes::from("test data");
        let checksum = Checksum::new_from_bytes(&test_bytes);
        let metadata = Metadata::new_v1(
            None,
            None,
            checksum,
            vec![1], // Non-empty shape required
            test_bytes.len(),
        );

        let peer = EncoderIndex::new_for_test(2);

        let (manager, expected_bytes) = setup_downloader(test_bytes, false);
        let handle = manager.handle();
        let input = DownloaderInput::new(1, peer, metadata);

        // Test
        let result = handle.process(input, CancellationToken::new()).await;

        // Verify
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expected_bytes);
    }

    #[tokio::test]
    async fn test_checksum_mismatch() {
        // Setup
        let test_bytes = Bytes::from("test data");
        let wrong_checksum = Checksum::new_from_bytes(&Bytes::from("different"));
        let metadata = Metadata::new_v1(None, None, wrong_checksum, vec![1], test_bytes.len());

        let peer = EncoderIndex::new_for_test(2);

        let (manager, _) = setup_downloader(test_bytes, false);
        let handle = manager.handle();
        let input = DownloaderInput::new(1, peer, metadata);

        // Test
        let result = handle.process(input, CancellationToken::new()).await;

        // Verify
        assert!(matches!(
            result,
            Err(ShardError::ObjectValidation(msg)) if msg == "object checksum does not match metadata"
        ));
    }

    #[tokio::test]
    async fn test_size_mismatch() {
        // Setup
        let test_bytes = Bytes::from("test data");
        let checksum = Checksum::new_from_bytes(&test_bytes);
        let metadata = Metadata::new_v1(
            None,
            None,
            checksum,
            vec![1],
            test_bytes.len() + 1, // Wrong size
        );

        let peer = EncoderIndex::new_for_test(2);
        let (manager, _) = setup_downloader(test_bytes, false);
        let handle = manager.handle();
        let input = DownloaderInput::new(1, peer, metadata);

        // Test
        let result = handle.process(input, CancellationToken::new()).await;

        // Verify
        assert!(matches!(
            result,
            Err(ShardError::ObjectValidation(msg)) if msg == "object size does not match metadata"
        ));
    }

    #[tokio::test]
    async fn test_client_failure() {
        // Setup
        let test_bytes = Bytes::from("test data");
        let checksum = Checksum::new_from_bytes(&test_bytes);
        let metadata = Metadata::new_v1(None, None, checksum, vec![1], test_bytes.len());

        let peer = EncoderIndex::new_for_test(2);
        let (manager, _) = setup_downloader(test_bytes, true); // Should fail
        let handle = manager.handle();
        let input = DownloaderInput::new(1, peer, metadata);

        // Test
        let result = handle.process(input, CancellationToken::new()).await;

        // Verify
        assert!(matches!(
            result,
            Err(ShardError::NetworkRequest(msg)) if msg == "mock failure"
        ));
    }

    #[tokio::test]
    async fn test_cancellation() {
        // Setup
        let test_bytes = Bytes::from("test data");
        let checksum = Checksum::new_from_bytes(&test_bytes);
        let metadata = Metadata::new_v1(None, None, checksum, vec![1], test_bytes.len());

        let peer = EncoderIndex::new_for_test(2);
        let (manager, _) = setup_downloader(test_bytes, false);
        let handle = manager.handle();
        let input = DownloaderInput::new(1, peer, metadata);

        let token = CancellationToken::new();
        token.cancel();

        // Test
        let result = handle.process(input, token).await;

        // Verify
        assert!(matches!(
            result,
            Err(ShardError::ActorError(msg)) if msg == "task cancelled"
        ));
    }

    #[tokio::test]
    async fn test_concurrency_limit() {
        // Setup
        let test_bytes = Bytes::from("test data");
        let checksum = Checksum::new_from_bytes(&test_bytes);
        let metadata = Metadata::new_v1(None, None, checksum, vec![1], test_bytes.len());
        let peer = EncoderIndex::new_for_test(2);

        let client = Arc::new(MockClient {
            return_bytes: test_bytes,
            should_fail: false,
        });
        let downloader = Downloader::new(1, client); // Only 1 concurrent task
        let manager = ActorManager::new(10, downloader);
        let handle = manager.handle();

        let input = DownloaderInput::new(1, peer, metadata);

        // Test - Send two requests
        let fut1 = handle.process(input.clone(), CancellationToken::new());
        let fut2 = handle.process(input, CancellationToken::new());

        // Verify both complete successfully (second waits for first)
        let (res1, res2) = tokio::join!(fut1, fut2);
        assert!(res1.is_ok());
        assert!(res2.is_ok());
    }

    #[tokio::test]
    async fn test_shutdown() {
        // Setup
        let test_bytes = Bytes::from("test data");
        let checksum = Checksum::new_from_bytes(&test_bytes);
        let metadata = Metadata::new_v1(None, None, checksum, vec![1], test_bytes.len());

        let (manager, _) = setup_downloader(test_bytes, false);
        let handle = manager.handle();

        let peer = EncoderIndex::new_for_test(2);
        // Process one successful request
        let input = DownloaderInput::new(1, peer, metadata.clone());
        let res = handle.process(input, CancellationToken::new()).await;
        assert!(res.is_ok());

        // Shutdown and verify subsequent requests fail
        manager.shutdown();
        tokio::time::sleep(Duration::from_millis(100)).await; // Wait for shutdown

        let peer = EncoderIndex::new_for_test(2);
        let input = DownloaderInput::new(1, peer, metadata);
        let res = handle.process(input, CancellationToken::new()).await;
        assert!(matches!(
            res,
            Err(ShardError::ActorError(msg)) if msg == "channel closed"
        ));
    }
}
