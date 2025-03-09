use shared::{
    block::BlockRef,
    entropy::{BlockEntropyOutput, BlockEntropyProof, EntropyAPI, EntropyVDF},
    error::SharedError,
};
use tokio::{
    sync::mpsc::{self, Sender},
    task::JoinHandle,
};

use crate::{error::ShardError, types::encoder_committee::Epoch};
use async_trait::async_trait;

use crate::actors::{ActorMessage, Processor};

pub(crate) struct VDFProcessor<E: EntropyAPI> {
    tx: Sender<ActorMessage<Self>>,
    worker_handle: JoinHandle<()>,
}

impl<E: EntropyAPI> VDFProcessor<E> {
    pub(crate) fn new(vdf: E, buffer: usize) -> Self {
        let (tx, mut rx) = mpsc::channel::<ActorMessage<Self>>(buffer);
        let worker_handle = tokio::task::spawn_blocking(move || {
            while let Some(msg) = rx.blocking_recv() {
                let (epoch, block_ref, entropy, proof) = msg.input;
                match vdf.verify_entropy(epoch, block_ref, &entropy, &proof) {
                    Ok(_) => {
                        let _ = msg.sender.send(Ok(()));
                    }
                    Err(e) => {
                        let _ = msg.sender.send(Err(ShardError::ActorError(e.to_string())));
                    }
                }
            }
        });

        Self { tx, worker_handle }
    }
}

#[async_trait]
impl<E: EntropyAPI> Processor for VDFProcessor<E> {
    type Input = (Epoch, BlockRef, BlockEntropyOutput, BlockEntropyProof);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        if let Err(e) = self.tx.send(msg).await {
            let _ =
                e.0.sender
                    .send(Err(ShardError::ActorError("task cancelled".to_string())));
        }
    }

    fn shutdown(&mut self) {
        self.worker_handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actors::ActorMessage;
    use crate::error::{ShardError, ShardResult};
    use bytes::Bytes;
    use shared::error::SharedResult;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tokio::time::sleep;
    use tokio_util::sync::CancellationToken;

    // Mock implementation of EntropyAPI for testing
    #[derive(Clone)]
    struct MockEntropyAPI {
        should_succeed: bool,
    }

    impl EntropyAPI for MockEntropyAPI {
        fn get_entropy(
            &self,
            _epoch: Epoch,
            _block_ref: BlockRef,
        ) -> SharedResult<(BlockEntropyOutput, BlockEntropyProof)> {
            // Not needed for processor tests
            unimplemented!()
        }

        fn verify_entropy(
            &self,
            _epoch: Epoch,
            _block_ref: BlockRef,
            _tx_entropy: &BlockEntropyOutput,
            _tx_entropy_proof: &BlockEntropyProof,
        ) -> SharedResult<()> {
            if self.should_succeed {
                Ok(())
            } else {
                Err(SharedError::FailedVDF("verification failed".to_string()))
            }
        }
    }

    // Helper function to create test data
    fn create_test_message() -> ActorMessage<VDFProcessor<MockEntropyAPI>> {
        let (sender, _receiver) = tokio::sync::oneshot::channel();
        ActorMessage {
            input: (
                1,                   // epoch
                BlockRef::default(), // Using default for simplicity
                BlockEntropyOutput::new(Bytes::new()),
                BlockEntropyProof::new(Bytes::new()),
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
        let msg = create_test_message();

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
        let msg = create_test_message();

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
        assert!(processor.worker_handle.is_finished());
    }
}
