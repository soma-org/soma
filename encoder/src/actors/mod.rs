#![doc = include_str!("README.md")]

pub(crate) mod orchestrators;
pub(crate) mod workers;

use async_trait::async_trait;
use tokio::{
    select,
    sync::{mpsc, oneshot},
};
use tokio_util::sync::CancellationToken;
use crate::error::{ShardError, ShardResult};

#[async_trait]
pub(crate) trait Processor: Send + Sync + Sized + 'static {
    type Input: Send + Sync + Sized + 'static;
    type Output: Send + Sync + Sized + 'static;
    // Tried returning a result here, except it seemed to be more confusing than beneficial.
    // in most cases error handling was being passed back via the sender in a spawned task, and
    // error handling can still work inside the process but outside of a spawned task via the sender.
    // Keeping this fn without a return type makes it more obvious that the return should occur via the sender.
    async fn process(&self, input: ActorMessage<Self>);
    fn shutdown(&mut self);
}

pub(crate) struct Actor<P: Processor> {
    receiver: mpsc::Receiver<ActorMessage<P>>,
    shutdown_rx: oneshot::Receiver<()>,
    processor: P,
}

pub(crate) struct ActorMessage<P: Processor> {
    input: P::Input,
    sender: oneshot::Sender<ShardResult<P::Output>>,
    cancellation: CancellationToken,
}

impl<P: Processor> Actor<P> {
    // intentionally not a reference, run consumes the actor
    async fn run(mut self) {
        loop {
            select! {
                Ok(()) = &mut self.shutdown_rx => {
                    // TODO: potentially process whatever messages are remaining?
                    // Shutdown the processor
                    self.processor.shutdown();
                    break;
                }

                msg = self.receiver.recv() => {
                    match msg {
                        Some(msg) => {
                            if msg.cancellation.is_cancelled() {
                                let _ = msg.sender.send(Err(ShardError::ActorError("task cancelled".to_string())));
                                continue;
                            }
                            self.processor.process(msg).await;
                        }
                        None => {
                            // Channel closed, cleanup processor
                            self.processor.shutdown();
                            break;
                        }
                    }
                }
            }
        }
    }
}

pub(crate) struct ActorHandle<P: Processor> {
    sender: mpsc::Sender<ActorMessage<P>>,
}

impl<P: Processor> Clone for ActorHandle<P> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
        }
    }
}

impl<P: Processor> ActorHandle<P> {
    pub async fn process(
        &self,
        input: P::Input,
        cancellation: CancellationToken,
    ) -> ShardResult<P::Output> {
        // TODO: make this more explicitly the actor response
        let (sender, receiver) = oneshot::channel();
        let msg = ActorMessage {
            input,
            sender,
            cancellation,
        };
        match self.sender.send(msg).await {
            Ok(_) => match receiver.await {
                Ok(res) => res,
                Err(_) => Err(ShardError::ActorError("channel closed".to_string())),
            },
            Err(_) => Err(ShardError::ActorError("channel closed".to_string())),
        }
    }

    pub async fn background_process(
        &self,
        input: P::Input,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        // TODO: tokio spawn a task that listens on the receiver and handles any error or times out
        let (sender, _receiver) = oneshot::channel();
        let msg = ActorMessage {
            input,
            sender,
            cancellation,
        };
        match self.sender.send(msg).await {
            Ok(_) => Ok(()),
            Err(_) => Err(ShardError::ActorError("channel closed".to_string())),
        }
    }
}

pub(crate) struct ActorManager<P: Processor> {
    sender: mpsc::Sender<ActorMessage<P>>,
    shutdown_tx: oneshot::Sender<()>,
}

impl<P: Processor> ActorManager<P> {
    pub fn new(buffer: usize, processor: P) -> Self {
        let (sender, receiver) = mpsc::channel(buffer);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let actor = Actor {
            receiver,
            shutdown_rx,
            processor,
        };
        tokio::spawn(actor.run());
        Self {
            sender,
            shutdown_tx,
        }
    }

    pub fn handle(&self) -> ActorHandle<P> {
        ActorHandle {
            sender: self.sender.clone(),
        }
    }

    pub fn shutdown(self) {
        self.shutdown_tx.send(());
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::time::{sleep, Duration};

    // A simple test processor that counts how many times it processes messages
    struct TestProcessor {
        counter: Arc<Mutex<i32>>,
        shutdown_called: Arc<Mutex<bool>>,
    }

    #[async_trait]
    impl Processor for TestProcessor {
        type Input = i32;
        type Output = i32;

        async fn process(&self, msg: ActorMessage<Self>) {
            let mut counter = self.counter.lock().unwrap();
            *counter += 1;
            let result = msg.input * 2; // Simple multiplication operation
            let _ = msg.sender.send(Ok(result));
        }

        fn shutdown(&mut self) {
            let mut shutdown = self.shutdown_called.lock().unwrap();
            *shutdown = true;
        }
    }

    #[tokio::test]
    async fn test_basic_processing() {
        let counter = Arc::new(Mutex::new(0));
        let shutdown_called = Arc::new(Mutex::new(false));
        
        let processor = TestProcessor {
            counter: counter.clone(),
            shutdown_called: shutdown_called.clone(),
        };
        
        let manager = ActorManager::new(10, processor);
        let handle = manager.handle();
        
        let result = handle.process(5, CancellationToken::new()).await.unwrap();
        assert_eq!(result, 10);
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_multiple_messages() {
        let counter = Arc::new(Mutex::new(0));
        let shutdown_called = Arc::new(Mutex::new(false));
        
        let processor = TestProcessor {
            counter: counter.clone(),
            shutdown_called: shutdown_called.clone(),
        };
        
        let manager = ActorManager::new(10, processor);
        let handle = manager.handle();
        
        let mut results = Vec::new();
        for i in 0..5 {
            let result = handle.process(i, CancellationToken::new()).await.unwrap();
            results.push(result);
        }
        
        assert_eq!(results, vec![0, 2, 4, 6, 8]);
        assert_eq!(*counter.lock().unwrap(), 5);
    }

    #[tokio::test]
    async fn test_cancellation() {
        let counter = Arc::new(Mutex::new(0));
        let shutdown_called = Arc::new(Mutex::new(false));
        
        let processor = TestProcessor {
            counter: counter.clone(),
            shutdown_called: shutdown_called.clone(),
        };
        
        let manager = ActorManager::new(10, processor);
        let handle = manager.handle();
        
        let token = CancellationToken::new();
        token.cancel();
        
        let result = handle.process(5, token).await;
        assert!(matches!(result, Err(ShardError::ActorError(_))));
        assert_eq!(*counter.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_shutdown() {
        let counter = Arc::new(Mutex::new(0));
        let shutdown_called = Arc::new(Mutex::new(false));
        
        let processor = TestProcessor {
            counter: counter.clone(),
            shutdown_called: shutdown_called.clone(),
        };
        
        let manager = ActorManager::new(10, processor);
        let handle = manager.handle();
        
        // Send a message before shutdown
        let result = handle.process(5, CancellationToken::new()).await.unwrap();
        assert_eq!(result, 10);
        
        // Initiate shutdown
        manager.shutdown();
        
        // Wait a bit to ensure shutdown is processed
        sleep(Duration::from_millis(100)).await;
        
        assert!(*shutdown_called.lock().unwrap());
    }

    #[tokio::test]
    async fn test_background_processing() {
        let counter = Arc::new(Mutex::new(0));
        let shutdown_called = Arc::new(Mutex::new(false));
        
        let processor = TestProcessor {
            counter: counter.clone(),
            shutdown_called: shutdown_called.clone(),
        };
        
        let manager = ActorManager::new(10, processor);
        let handle = manager.handle();
        
        // Send background message
        handle.background_process(5, CancellationToken::new()).await.unwrap();
        
        // Wait a bit to ensure processing completes
        sleep(Duration::from_millis(100)).await;
        
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_handle_clone() {
        let counter = Arc::new(Mutex::new(0));
        let shutdown_called = Arc::new(Mutex::new(false));
        
        let processor = TestProcessor {
            counter: counter.clone(),
            shutdown_called: shutdown_called.clone(),
        };
        
        let manager = ActorManager::new(10, processor);
        let handle = manager.handle();
        let handle_clone = handle.clone();
        
        let result1 = handle.process(5, CancellationToken::new()).await.unwrap();
        let result2 = handle_clone.process(6, CancellationToken::new()).await.unwrap();
        
        assert_eq!(result1, 10);
        assert_eq!(result2, 12);
        assert_eq!(*counter.lock().unwrap(), 2);
    }
}