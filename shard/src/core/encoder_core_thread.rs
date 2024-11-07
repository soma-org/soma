use async_trait::async_trait;
use tracing::warn;

use crate::{
    core::encoder_core::EncoderCore,
    error::{ShardError, ShardResult},
    networking::messaging::EncoderNetworkClient,
    types::{shard_input::ShardInput, signed::Signed, verified::Verified},
};
use tokio::{sync::mpsc, task::JoinSet};

// TODO: make this configurable
const CORE_THREAD_COMMANDS_CHANNEL_SIZE: usize = 1000;

#[async_trait]
pub trait TaskDispatcher: Sync + Send + 'static {
    async fn process_shard_input(
        &self,
        shard_input: Verified<Signed<ShardInput>>,
    ) -> ShardResult<()>;
}

pub struct TaskManagerHandle {
    shard_input_sender: mpsc::Sender<Verified<Signed<ShardInput>>>,
    join_set: JoinSet<()>,
}

impl TaskManagerHandle {
    pub async fn stop(self) {
        drop(self.shard_input_sender);
        self.join_set.join_all().await;
    }
}

struct TaskManager<C: EncoderNetworkClient> {
    core: EncoderCore<C>,
    shard_input_receiver: mpsc::Receiver<Verified<Signed<ShardInput>>>,
}

impl<C: EncoderNetworkClient> TaskManager<C> {
    pub async fn run_shard_input_receiver(mut self) {
        loop {
            tokio::select! {
                shard_input= self.shard_input_receiver.recv() => {
                    let Some(shard_input) = shard_input else {
                        break;
                    };
                    self.core.process_shard_input(shard_input).await;
                }
                // TODO: select on other important things here if needed
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct ChannelTaskDispatcher {
    shard_input_sender: mpsc::WeakSender<Verified<Signed<ShardInput>>>,
}

impl ChannelTaskDispatcher {
    pub(crate) fn start<C: EncoderNetworkClient>(
        core: EncoderCore<C>,
    ) -> (Self, TaskManagerHandle) {
        let (shard_input_sender, shard_input_receiver) =
            mpsc::channel(CORE_THREAD_COMMANDS_CHANNEL_SIZE);
        let task_manager = TaskManager {
            core,
            shard_input_receiver,
        };

        let mut join_set = JoinSet::new();

        join_set.spawn(async move { task_manager.run_shard_input_receiver().await });

        let dispatcher = ChannelTaskDispatcher {
            shard_input_sender: shard_input_sender.downgrade(),
        };
        let handle = TaskManagerHandle {
            join_set,
            shard_input_sender,
        };
        (dispatcher, handle)
    }

    async fn send<T: Sync + Send>(&self, sender: &mpsc::WeakSender<T>, value: T) {
        if let Some(sender) = sender.upgrade() {
            if let Err(err) = sender.send(value).await {
                warn!(
                    "Couldn't send to task manager thread, probably is shutting down: {}",
                    err
                );
            }
        }
    }
}

#[async_trait]
impl TaskDispatcher for ChannelTaskDispatcher {
    async fn process_shard_input(
        &self,
        shard_input: Verified<Signed<ShardInput>>,
    ) -> ShardResult<()> {
        self.send(&self.shard_input_sender, shard_input).await;
        Ok(())
    }
}
