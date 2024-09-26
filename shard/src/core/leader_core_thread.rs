use async_trait::async_trait;

use crate::{
    core::leader_core::LeaderCore,
    error::{ShardError, ShardResult},
    networking::messaging::EncoderNetworkClient,
    types::{
        shard_commit::VerifiedSignedShardCommit, shard_endorsement::VerifiedSignedShardEndorsement,
    },
};
use tokio::sync::mpsc;

// TODO: make this configurable
const CORE_THREAD_COMMANDS_CHANNEL_SIZE: usize = 1000;

enum LeaderCoreThreadCommand {
    ProcessCommit(VerifiedSignedShardCommit),
    ProcessEndorsement(VerifiedSignedShardEndorsement),
}

#[async_trait]
pub trait LeaderCoreThreadDispatcher: Sync + Send + 'static {
    async fn process_commit(&self, input: VerifiedSignedShardCommit) -> ShardResult<()>;
    async fn process_endorsement(
        &self,
        selection: VerifiedSignedShardEndorsement,
    ) -> ShardResult<()>;
}

pub struct LeaderCoreThreadHandle {
    sender: mpsc::Sender<LeaderCoreThreadCommand>,
    join_handle: tokio::task::JoinHandle<()>,
}

impl LeaderCoreThreadHandle {
    pub async fn stop(self) {
        drop(self.sender);
        self.join_handle.await.ok();
    }
}

struct LeaderCoreThread<ENC: EncoderNetworkClient> {
    core: LeaderCore<ENC>,
    receiver: mpsc::Receiver<LeaderCoreThreadCommand>,
}

impl<ENC: EncoderNetworkClient> LeaderCoreThread<ENC> {
    pub async fn run(mut self) {
        // tracing::debug!("Started core thread");

        loop {
            tokio::select! {
                command = self.receiver.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    match command {
                        LeaderCoreThreadCommand::ProcessCommit(input) => {
                            self.core.process_commit(input).await;
                        }
                        LeaderCoreThreadCommand::ProcessEndorsement(selection) => {
                            self.core.process_endorsement(selection).await;
                        }
                    }
                }
            }
            // add any important listeners here
        }
    }
}

#[derive(Clone)]
pub(crate) struct LeaderChannelCoreThreadDispatcher {
    sender: mpsc::Sender<LeaderCoreThreadCommand>,
}

impl LeaderChannelCoreThreadDispatcher {
    pub(crate) fn start<ENC: EncoderNetworkClient>(
        core: LeaderCore<ENC>,
    ) -> (Self, LeaderCoreThreadHandle) {
        let (sender, receiver) = mpsc::channel(CORE_THREAD_COMMANDS_CHANNEL_SIZE);
        let core_thread = LeaderCoreThread { core, receiver };

        let join_handle = tokio::spawn(async move { core_thread.run().await });

        let dispatcher = LeaderChannelCoreThreadDispatcher {
            sender: sender.clone(),
        };
        let handle = LeaderCoreThreadHandle {
            join_handle,
            sender,
        };
        println!("started core thread");
        (dispatcher, handle)
    }

    async fn send(&self, command: LeaderCoreThreadCommand) -> ShardResult<()> {
        self.sender
            .send(command)
            .await
            .map_err(|e| ShardError::FailedToSendToCoreThread(e.to_string()))?;

        Ok(())
    }
}

#[async_trait]
impl LeaderCoreThreadDispatcher for LeaderChannelCoreThreadDispatcher {
    async fn process_commit(&self, input: VerifiedSignedShardCommit) -> ShardResult<()> {
        // TODO: better error handling
        self.send(LeaderCoreThreadCommand::ProcessCommit(input))
            .await?;
        Ok(())
    }
    async fn process_endorsement(
        &self,
        selection: VerifiedSignedShardEndorsement,
    ) -> ShardResult<()> {
        // TODO: better error handling
        self.send(LeaderCoreThreadCommand::ProcessEndorsement(selection))
            .await?;
        Ok(())
    }
}
