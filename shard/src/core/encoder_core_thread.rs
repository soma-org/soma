use async_trait::async_trait;

use crate::{
    core::encoder_core::EncoderCore,
    error::{ShardError, ShardResult},
    networking::messaging::LeaderNetworkClient,
    types::{shard_input::VerifiedSignedShardInput, shard_selection::VerifiedSignedShardSelection},
};
use tokio::sync::mpsc;

// TODO: make this configurable
const CORE_THREAD_COMMANDS_CHANNEL_SIZE: usize = 1000;

enum EncoderCoreThreadCommand {
    ProcessInput(VerifiedSignedShardInput),
    ProcessSelection(VerifiedSignedShardSelection),
}

#[async_trait]
pub trait EncoderCoreThreadDispatcher: Sync + Send + 'static {
    async fn process_input(&self, input: VerifiedSignedShardInput) -> ShardResult<()>;
    async fn process_selection(&self, selection: VerifiedSignedShardSelection) -> ShardResult<()>;
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

struct EncoderCoreThread<LNC: LeaderNetworkClient> {
    core: EncoderCore<LNC>,
    receiver: mpsc::Receiver<EncoderCoreThreadCommand>,
}

impl<LNC: LeaderNetworkClient> EncoderCoreThread<LNC> {
    pub async fn run(mut self) {
        // tracing::debug!("Started core thread");

        loop {
            tokio::select! {
                command = self.receiver.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    match command {
                        EncoderCoreThreadCommand::ProcessInput(input) => {
                            self.core.process_input(input).await;
                        }
                        EncoderCoreThreadCommand::ProcessSelection(selection) => {
                            self.core.process_selection(selection).await;
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
    pub(crate) fn start<LNC: LeaderNetworkClient>(
        core: EncoderCore<LNC>,
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
    async fn process_input(&self, input: VerifiedSignedShardInput) -> ShardResult<()> {
        // TODO: better error handling
        self.send(EncoderCoreThreadCommand::ProcessInput(input))
            .await?;
        Ok(())
    }
    async fn process_selection(&self, selection: VerifiedSignedShardSelection) -> ShardResult<()> {
        // TODO: better error handling
        self.send(EncoderCoreThreadCommand::ProcessSelection(selection))
            .await?;
        Ok(())
    }
}
