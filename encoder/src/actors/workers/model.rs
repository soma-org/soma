use crate::intelligence::model::Model;
use async_trait::async_trait;
use numpy::ndarray::ArrayD;
use shared::actors::{ActorMessage, Processor};
use shared::error::{ShardError, ShardResult};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// ModelProcessor takes a generic model that implements the 'Model' trait and an optional semaphore.
#[derive(Clone)]
pub(crate) struct ModelProcessor<M: Model> {
    model: Arc<M>,
    semaphore: Option<Arc<Semaphore>>,
}

impl<M: Model> ModelProcessor<M> {
    /// New takes a model that implements the 'Model' trait along with an optional semaphore to limit concurrency.
    pub fn new(model: M, concurrency: Option<usize>) -> Self {
        let semaphore = concurrency.map(|n| Arc::new(Semaphore::new(n)));
        Self {
            model: Arc::new(model),
            semaphore,
        }
    }
}

/// ModelProcessor adjusts concurrency depending on user settings. If concurrency is allowed, the processor will concurrently
/// process data via the model up to the semaphore limit. Otherwise, the processor assumes synchronous operation.
#[async_trait]
impl<M: Model> Processor for ModelProcessor<M> {
    type Input = ArrayD<f32>;
    type Output = ArrayD<f32>;

    async fn process(&self, msg: ActorMessage<Self>) {
        // if semaphore not none
        if let Some(sem) = &self.semaphore {
            // await until semaphore permits spawning a new tokio task using the model
            if let Ok(permit) = sem.clone().acquire_owned().await {
                let model = self.model.clone();
                tokio::spawn(async move {
                    if let Ok(embeddings) = model.call(&msg.input).await {
                        let _ = msg.sender.send(Ok(embeddings));
                    }
                    drop(permit)
                });
            }
        } else {
            // await for the result before processing any new data
            if let Ok(embeddings) = self.model.call(&msg.input).await {
                let _ = msg.sender.send(Ok(embeddings));
            }
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use numpy::ndarray::{Array, IxDyn};
    use shared::error::ShardError;
    use std::time::Duration;
    use tokio::sync::oneshot;
    use tokio_util::sync::CancellationToken;

    // Only testing the processor logic not model functionality

    // Mock model that simply returns the input array
    #[derive(Clone)]
    struct MockModel;

    #[async_trait]
    impl Model for MockModel {
        async fn call(&self, input: &ArrayD<f32>) -> ShardResult<ArrayD<f32>> {
            Ok(input.clone())
        }
    }

    #[tokio::test]
    async fn test_sync_processing() {
        let processor = ModelProcessor::new(MockModel, None);
        let input = Array::zeros(IxDyn(&[2, 2]));
        let (tx, rx) = oneshot::channel();

        let message = ActorMessage {
            input: input.clone(),
            sender: tx,
            cancellation: CancellationToken::new(),
        };

        processor.process(message).await;

        let result = rx.await.unwrap().unwrap();
        assert_eq!(result, input);
    }

    #[tokio::test]
    async fn test_concurrent_processing() {
        let processor = ModelProcessor::new(MockModel, Some(2));
        let input = Array::zeros(IxDyn(&[2, 2]));

        // Create multiple concurrent requests
        let mut handles = vec![];
        for _ in 0..3 {
            let (tx, rx) = oneshot::channel();
            let message = ActorMessage {
                input: input.clone(),
                sender: tx,
                cancellation: CancellationToken::new(),
            };

            let proc = processor.clone();
            handles.push(tokio::spawn(async move {
                proc.process(message).await;
                rx.await.unwrap()
            }));
        }

        // Wait for all results
        for handle in handles {
            let result = handle.await.unwrap().unwrap();
            assert_eq!(result, input);
        }
    }

    #[tokio::test]
    async fn test_semaphore_limit() {
        let processor = ModelProcessor::new(MockModel, Some(1));
        let input = Array::zeros(IxDyn(&[2, 2]));

        // Create two concurrent requests
        let (tx1, rx1) = oneshot::channel();
        let (tx2, rx2) = oneshot::channel();

        let message1 = ActorMessage {
            input: input.clone(),
            sender: tx1,
            cancellation: CancellationToken::new(),
        };

        let message2 = ActorMessage {
            input: input.clone(),
            sender: tx2,
            cancellation: CancellationToken::new(),
        };

        // Process both messages concurrently
        let proc = processor.clone();
        tokio::spawn(async move {
            proc.process(message1).await;
        });

        // Small delay to ensure first message starts processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        processor.process(message2).await;

        // Both should complete successfully despite the semaphore limit
        let result1 = rx1.await.unwrap().unwrap();
        let result2 = rx2.await.unwrap().unwrap();

        assert_eq!(result1, input);
        assert_eq!(result2, input);
    }
}
