use shared::{
    block::BlockRef,
    entropy::{BlockEntropyOutput, BlockEntropyProof, EntropyAPI, EntropyVDF},
};
use tokio::{
    sync::mpsc::{self, Sender},
    task::JoinHandle,
};

use crate::{error::ShardError, types::encoder_committee::Epoch};
use async_trait::async_trait;

use crate::actors::{ActorMessage, Processor};

pub(crate) struct VDFProcessor {
    tx: Sender<ActorMessage<VDFProcessor>>,
    worker_handle: JoinHandle<()>,
}

impl VDFProcessor {
    pub(crate) fn new(vdf: EntropyVDF, buffer: usize) -> Self {
        let (tx, mut rx) = mpsc::channel::<ActorMessage<VDFProcessor>>(buffer);
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
impl Processor for VDFProcessor {
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
