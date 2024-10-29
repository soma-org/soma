use async_trait::async_trait;

use crate::{
    core::encoder_core::EncoderCore,
    error::{ShardError, ShardResult},
    networking::messaging::EncoderNetworkClient,
    types::{shard_input::ShardInput, signed::Signed, verified::Verified},
};
use tokio::sync::mpsc;

// TODO: make this configurable
const CORE_THREAD_COMMANDS_CHANNEL_SIZE: usize = 1000;

enum EncoderCoreThreadCommand {
    ProcessShardInput(Verified<Signed<ShardInput>>),
}

#[async_trait]
pub trait EncoderCoreThreadDispatcher: Sync + Send + 'static {
    async fn process_shard_input(
        &self,
        shard_input: Verified<Signed<ShardInput>>,
    ) -> ShardResult<()>;
}

pub struct EncoderCoreThreadHandle {
    sender: mpsc::Sender<EncoderCoreThreadCommand>,
    join_handle: tokio::task::JoinHandle<()>,
}

impl EncoderCoreThreadHandle {
    pub async fn stop(self) {
        drop(self.sender);
        self.join_handle.await.ok();
    }
}

struct EncoderCoreThread<C: EncoderNetworkClient> {
    core: EncoderCore<C>,
    receiver: mpsc::Receiver<EncoderCoreThreadCommand>,
}

impl<C: EncoderNetworkClient> EncoderCoreThread<C> {
    pub async fn run(mut self) {
        // tracing::debug!("Started core thread");

        loop {
            tokio::select! {
                command = self.receiver.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    match command {
                        EncoderCoreThreadCommand::ProcessShardInput(shard_input) => {
                            self.core.process_shard_input(shard_input).await;
                        }
                    }
                }
            }
            // add any important listeners here
        }
    }
}

#[derive(Clone)]
pub(crate) struct EncoderChannelCoreThreadDispatcher {
    sender: mpsc::Sender<EncoderCoreThreadCommand>,
}

impl EncoderChannelCoreThreadDispatcher {
    pub(crate) fn start<C: EncoderNetworkClient>(
        core: EncoderCore<C>,
    ) -> (Self, EncoderCoreThreadHandle) {
        let (sender, receiver) = mpsc::channel(CORE_THREAD_COMMANDS_CHANNEL_SIZE);
        let core_thread = EncoderCoreThread { core, receiver };

        let join_handle = tokio::spawn(async move { core_thread.run().await });

        let dispatcher = EncoderChannelCoreThreadDispatcher {
            sender: sender.clone(),
        };
        let handle = EncoderCoreThreadHandle {
            join_handle,
            sender,
        };
        println!("started core thread");
        (dispatcher, handle)
    }

    async fn send(&self, command: EncoderCoreThreadCommand) -> ShardResult<()> {
        self.sender
            .send(command)
            .await
            .map_err(|e| ShardError::FailedToSendToCoreThread(e.to_string()))?;

        Ok(())
    }
}

#[async_trait]
impl EncoderCoreThreadDispatcher for EncoderChannelCoreThreadDispatcher {
    async fn process_shard_input(
        &self,
        shard_input: Verified<Signed<ShardInput>>,
    ) -> ShardResult<()> {
        self.send(EncoderCoreThreadCommand::ProcessShardInput(shard_input))
            .await?;
        Ok(())
    }
}
