pub(crate) mod blob;
pub(crate) mod compression;
pub(crate) mod downloader;
pub(crate) mod encryption;
pub(crate) mod model;

use async_trait::async_trait;
use tokio::{
    select,
    sync::{mpsc, oneshot},
};
use tokio_util::sync::CancellationToken;

use crate::error::{ShardError, ShardResult};

#[async_trait]
trait Processor: Send + Sync + Sized + 'static {
    type Input: Send + Sync + Sized + 'static;
    type Output: Send + Sync + Sized + 'static;
    // Tried returning a result here, except it seemed to be more confusing than beneficial.
    // in most cases error handling was being passed back via the sender in a spawned task, and
    // error handling can still work inside the process but outside of a spawned task via the sender.
    // Keeping this fn without a return type makes it more obvious that the return should occur via the sender.
    async fn process(&self, input: ActorMessage<Self>);
    fn shutdown(&mut self);
}

struct Actor<P: Processor> {
    receiver: mpsc::Receiver<ActorMessage<P>>,
    shutdown_rx: oneshot::Receiver<()>,
    processor: P,
}

struct ActorMessage<P: Processor> {
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

pub struct ActorHandle<P: Processor> {
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
    pub async fn send(
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
}

pub struct ActorManager<P: Processor> {
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
