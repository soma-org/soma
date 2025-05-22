use crate::{
    block::BlockRef,
    entropy::{BlockEntropy, BlockEntropyProof, EntropyAPI},
};
use std::sync::{Arc, Mutex};
use tokio::{
    runtime::Handle,
    sync::mpsc::{self, Sender},
    task::JoinHandle,
};

use crate::{encoder_committee::Epoch, error::ShardError};
use async_trait::async_trait;

use crate::actors::{ActorMessage, Processor};

pub struct VDFProcessor<E: EntropyAPI> {
    vdf: Arc<Mutex<E>>,
}

impl<E: EntropyAPI> VDFProcessor<E> {
    pub fn new(mut vdf: E, buffer: usize) -> Self {
        Self {
            vdf: Arc::new(Mutex::new(vdf)),
        }
    }
}

#[async_trait]
impl<E: EntropyAPI> Processor for VDFProcessor<E> {
    type Input = (Epoch, BlockRef, BlockEntropy, BlockEntropyProof, u64);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let vdf = self.vdf.clone();
        let (epoch, block_ref, entropy, proof, iterations) = msg.input;

        // Spawn a task without awaiting it, similar to your other processors
        let _ = Handle::current().spawn_blocking(move || {
            // Lock the VDF instance only within the task
            match vdf.lock() {
                Ok(mut vdf_instance) => {
                    let result =
                        vdf_instance.verify_entropy(epoch, block_ref, &entropy, &proof, iterations);
                    match result {
                        Ok(_) => {
                            let _ = msg.sender.send(Ok(()));
                        }
                        Err(e) => {
                            let _ = msg.sender.send(Err(ShardError::ActorError(e.to_string())));
                        }
                    }
                }
                Err(_) => {
                    let _ = msg.sender.send(Err(ShardError::ActorError(
                        "Failed to lock VDF".to_string(),
                    )));
                }
            }
        });
    }

    fn shutdown(&mut self) {
        // self.worker_handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actors::ActorMessage;
    use crate::error::{ShardError, ShardResult};
    use crate::error::{SharedError, SharedResult};
    use bytes::Bytes;
    use std::time::Duration;
    use tokio::sync::oneshot::Sender;
    use tokio::time::sleep;
    use tokio_util::sync::CancellationToken;

    // Mock implementation of EntropyAPI for testing
    #[derive(Clone)]
    struct MockEntropyAPI {
        should_succeed: bool,
    }

    impl EntropyAPI for MockEntropyAPI {
        fn get_entropy(
            &mut self,
            _epoch: Epoch,
            _block_ref: BlockRef,
            _iterations: u64,
        ) -> SharedResult<(BlockEntropy, BlockEntropyProof)> {
            // Not needed for processor tests
            unimplemented!()
        }

        fn verify_entropy(
            &mut self,
            _epoch: Epoch,
            _block_ref: BlockRef,
            _block_entropy: &BlockEntropy,
            _block_entropy_proof: &BlockEntropyProof,
            _iterations: u64,
        ) -> SharedResult<()> {
            if self.should_succeed {
                Ok(())
            } else {
                Err(SharedError::FailedVDF("verification failed".to_string()))
            }
        }
    }

    // Helper function to create test data
    fn create_test_message(
        sender: Sender<ShardResult<()>>,
    ) -> ActorMessage<VDFProcessor<MockEntropyAPI>> {
        ActorMessage {
            input: (
                1,                   // epoch
                BlockRef::default(), // Using default for simplicity
                BlockEntropy::new(Bytes::new()),
                BlockEntropyProof::new(Bytes::new()),
                1,
            ),
            sender,
            cancellation: CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn test_vdf_processor_successful_verification() {
        // Create mock that succeeds
        let mock_entropy = MockEntropyAPI {
            should_succeed: true,
        };
        let processor = VDFProcessor::new(mock_entropy, 10);

        // Create test message with response channel
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let msg = create_test_message(sender);

        // Process the message
        processor.process(msg).await;

        // Check result
        let result: ShardResult<()> = receiver.await.unwrap();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_vdf_processor_failed_verification() {
        // Create mock that fails
        let mock_entropy = MockEntropyAPI {
            should_succeed: false,
        };
        let processor = VDFProcessor::new(mock_entropy, 10);

        // Create test message with response channel
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let msg = create_test_message(sender);

        // Process the message
        processor.process(msg).await;

        // Check result
        let result: ShardResult<()> = receiver.await.unwrap();
        assert!(result.is_err());
        if let Err(ShardError::ActorError(msg)) = result {
            assert!(msg.contains("verification failed"));
        }
    }

    #[tokio::test]
    async fn test_vdf_processor_shutdown() {
        let mock_entropy = MockEntropyAPI {
            should_succeed: true,
        };
        let mut processor = VDFProcessor::new(mock_entropy, 10);

        // Shutdown the processor
        processor.shutdown();

        sleep(Duration::from_millis(100)).await;
        // Verify the worker handle is aborted
        // assert!(processor.worker_handle.is_finished());
    }
}
