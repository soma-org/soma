mod downloader;
mod model;

use async_trait::async_trait;
use tokio::{
    select,
    sync::{mpsc, oneshot},
};
use tokio_util::sync::CancellationToken;

#[async_trait]
trait Processor: Send + Sync + Sized + 'static {
    type Input: Send + Sync + Sized + 'static;
    type Output: Send + Sync + Sized + 'static;
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
    sender: oneshot::Sender<P::Output>,
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

#[derive(Clone)]
pub struct ActorHandle<P: Processor> {
    sender: mpsc::Sender<ActorMessage<P>>,
}

impl<P: Processor> ActorHandle<P> {
    pub async fn send(&self, input: P::Input, cancellation: CancellationToken) -> P::Output {
        let (sender, receiver) = oneshot::channel();
        let msg = ActorMessage {
            input,
            sender,
            cancellation,
        };
        let _ = self.sender.send(msg).await;
        receiver.await.expect("Actor task has been killed")
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
