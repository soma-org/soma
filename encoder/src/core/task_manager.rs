use async_trait::async_trait;
use tracing::warn;

use crate::{
    core::encoder_core::EncoderCore,
    error::ShardResult,
    intelligence::model::Model,
    networking::{blob::ObjectNetworkClient, messaging::EncoderNetworkClient},
    storage::object::ObjectStorage,
    types::{
        certificate::ShardCertificate, shard_commit::ShardCommit,
        shard_completion_proof::ShardCompletionProof, shard_endorsement::ShardEndorsement,
        shard_input::ShardInput, shard_removal::ShardRemoval, shard_reveal::ShardReveal,
        signed::Signed, verified::Verified,
    },
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

    async fn process_shard_commit_certificate(
        &self,
        shard_commit_certificate: Verified<ShardCertificate<Signed<ShardCommit>>>,
    ) -> ShardResult<()>;

    async fn process_shard_reveal_certificate(
        &self,
        shard_reveal_certificate: Verified<ShardCertificate<Signed<ShardReveal>>>,
    ) -> ShardResult<()>;

    async fn process_shard_removal_certificate(
        &self,
        shard_removal_certificate: Verified<ShardCertificate<ShardRemoval>>,
    ) -> ShardResult<()>;

    async fn process_shard_endorsement_certificate(
        &self,
        shard_endorsement_certificate: Verified<ShardCertificate<Signed<ShardEndorsement>>>,
    ) -> ShardResult<()>;

    async fn process_shard_completion_proof(
        &self,
        shard_completion_proof: Verified<ShardCompletionProof>,
    ) -> ShardResult<()>;
}

pub struct TaskManagerHandle {
    shard_input_sender: mpsc::Sender<Verified<Signed<ShardInput>>>,
    shard_commit_certificate_sender: mpsc::Sender<Verified<ShardCertificate<Signed<ShardCommit>>>>,
    shard_reveal_certificate_sender: mpsc::Sender<Verified<ShardCertificate<Signed<ShardReveal>>>>,
    shard_removal_certificate_sender: mpsc::Sender<Verified<ShardCertificate<ShardRemoval>>>,
    shard_endorsement_certificate_sender:
        mpsc::Sender<Verified<ShardCertificate<Signed<ShardEndorsement>>>>,
    shard_completion_proof_sender: mpsc::Sender<Verified<ShardCompletionProof>>,
    join_set: JoinSet<()>,
}

impl TaskManagerHandle {
    pub async fn stop(self) {
        drop(self.shard_input_sender);
        self.join_set.join_all().await;
    }
}

struct TaskManager<C: EncoderNetworkClient, M: Model, B: ObjectStorage, BC: ObjectNetworkClient> {
    core: EncoderCore<C, M, B, BC>,
    shard_input_receiver: mpsc::Receiver<Verified<Signed<ShardInput>>>,
    shard_commit_certificate_receiver:
        mpsc::Receiver<Verified<ShardCertificate<Signed<ShardCommit>>>>,
    shard_reveal_certificate_receiver:
        mpsc::Receiver<Verified<ShardCertificate<Signed<ShardReveal>>>>,
    shard_removal_certificate_receiver: mpsc::Receiver<Verified<ShardCertificate<ShardRemoval>>>,
    shard_endorsement_certificate_receiver:
        mpsc::Receiver<Verified<ShardCertificate<Signed<ShardEndorsement>>>>,
    shard_completion_proof_receiver: mpsc::Receiver<Verified<ShardCompletionProof>>,
}

impl<C: EncoderNetworkClient, M: Model, B: ObjectStorage, BC: ObjectNetworkClient>
    TaskManager<C, M, B, BC>
{
    fn spawn_tasks(self, join_set: &mut JoinSet<()>) {
        let TaskManager {
            core,
            mut shard_input_receiver,
            mut shard_commit_certificate_receiver,
            mut shard_reveal_certificate_receiver,
            mut shard_removal_certificate_receiver,
            mut shard_endorsement_certificate_receiver,
            mut shard_completion_proof_receiver,
        } = self;

        // Spawn task for shard input
        join_set.spawn({
            let core = core.clone();
            async move {
                loop {
                    tokio::select! {
                        shard_input = shard_input_receiver.recv() => {
                            let Some(shard_input) = shard_input else { break };
                            // core.process_shard_input(shard_input).await;
                        }
                    }
                }
            }
        });

        // Spawn task for commit certificates
        join_set.spawn({
            let core = core.clone();
            async move {
                loop {
                    tokio::select! {
                        cert = shard_commit_certificate_receiver.recv() => {
                            let Some(cert) = cert else { break };
                            // core.process_shard_commit_certificate(cert).await;
                        }
                    }
                }
            }
        });

        // Spawn task for reveal certificates
        join_set.spawn({
            let core = core.clone();
            async move {
                loop {
                    tokio::select! {
                        cert = shard_reveal_certificate_receiver.recv() => {
                            let Some(cert) = cert else { break };
                            // core.process_shard_reveal_certificate(cert).await;
                        }
                    }
                }
            }
        });

        // Spawn task for removal certificates
        join_set.spawn({
            let core = core.clone();
            async move {
                loop {
                    tokio::select! {
                        cert = shard_removal_certificate_receiver.recv() => {
                            let Some(cert) = cert else { break };
                            core.process_shard_removal_certificate(cert).await;
                        }
                    }
                }
            }
        });

        // Spawn task for endorsement certificates
        join_set.spawn({
            let core = core.clone();
            async move {
                loop {
                    tokio::select! {
                        cert = shard_endorsement_certificate_receiver.recv() => {
                            let Some(cert) = cert else { break };
                            core.process_shard_endorsement_certificate(cert).await;
                        }
                    }
                }
            }
        });

        // Spawn task for completion proofs
        join_set.spawn({
            let core = core.clone();
            async move {
                loop {
                    tokio::select! {
                        proof = shard_completion_proof_receiver.recv() => {
                            let Some(proof) = proof else { break };
                            core.process_shard_completion_proof(proof).await;
                        }
                    }
                }
            }
        });
    }
}

#[derive(Clone)]
pub(crate) struct ChannelTaskDispatcher {
    shard_input_sender: mpsc::WeakSender<Verified<Signed<ShardInput>>>,
    shard_commit_certificate_sender:
        mpsc::WeakSender<Verified<ShardCertificate<Signed<ShardCommit>>>>,
    shard_reveal_certificate_sender:
        mpsc::WeakSender<Verified<ShardCertificate<Signed<ShardReveal>>>>,
    shard_removal_certificate_sender: mpsc::WeakSender<Verified<ShardCertificate<ShardRemoval>>>,
    shard_endorsement_certificate_sender:
        mpsc::WeakSender<Verified<ShardCertificate<Signed<ShardEndorsement>>>>,
    shard_completion_proof_sender: mpsc::WeakSender<Verified<ShardCompletionProof>>,
}

impl ChannelTaskDispatcher {
    pub(crate) fn start<
        C: EncoderNetworkClient,
        M: Model,
        B: ObjectStorage,
        BC: ObjectNetworkClient,
    >(
        core: EncoderCore<C, M, B, BC>,
    ) -> (Self, TaskManagerHandle) {
        let (shard_input_sender, shard_input_receiver) =
            mpsc::channel(CORE_THREAD_COMMANDS_CHANNEL_SIZE);

        let (shard_commit_certificate_sender, shard_commit_certificate_receiver) =
            mpsc::channel(CORE_THREAD_COMMANDS_CHANNEL_SIZE);

        let (shard_reveal_certificate_sender, shard_reveal_certificate_receiver) =
            mpsc::channel(CORE_THREAD_COMMANDS_CHANNEL_SIZE);

        let (shard_removal_certificate_sender, shard_removal_certificate_receiver) =
            mpsc::channel(CORE_THREAD_COMMANDS_CHANNEL_SIZE);

        let (shard_endorsement_certificate_sender, shard_endorsement_certificate_receiver) =
            mpsc::channel(CORE_THREAD_COMMANDS_CHANNEL_SIZE);

        let (shard_completion_proof_sender, shard_completion_proof_receiver) =
            mpsc::channel(CORE_THREAD_COMMANDS_CHANNEL_SIZE);

        let task_manager = TaskManager {
            core,
            shard_input_receiver,
            shard_commit_certificate_receiver,
            shard_reveal_certificate_receiver,
            shard_removal_certificate_receiver,
            shard_endorsement_certificate_receiver,
            shard_completion_proof_receiver,
        };

        let mut join_set: JoinSet<()> = JoinSet::new();
        task_manager.spawn_tasks(&mut join_set);

        let dispatcher = ChannelTaskDispatcher {
            shard_input_sender: shard_input_sender.downgrade(),
            shard_commit_certificate_sender: shard_commit_certificate_sender.downgrade(),
            shard_reveal_certificate_sender: shard_reveal_certificate_sender.downgrade(),
            shard_removal_certificate_sender: shard_removal_certificate_sender.downgrade(),
            shard_endorsement_certificate_sender: shard_endorsement_certificate_sender.downgrade(),
            shard_completion_proof_sender: shard_completion_proof_sender.downgrade(),
        };
        let handle = TaskManagerHandle {
            join_set,
            shard_input_sender,
            shard_commit_certificate_sender,
            shard_reveal_certificate_sender,
            shard_removal_certificate_sender,
            shard_endorsement_certificate_sender,
            shard_completion_proof_sender,
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

    async fn process_shard_commit_certificate(
        &self,
        shard_commit_certificate: Verified<ShardCertificate<Signed<ShardCommit>>>,
    ) -> ShardResult<()> {
        self.send(
            &self.shard_commit_certificate_sender,
            shard_commit_certificate,
        )
        .await;
        Ok(())
    }

    async fn process_shard_reveal_certificate(
        &self,
        shard_reveal_certificate: Verified<ShardCertificate<Signed<ShardReveal>>>,
    ) -> ShardResult<()> {
        self.send(
            &self.shard_reveal_certificate_sender,
            shard_reveal_certificate,
        )
        .await;
        Ok(())
    }

    async fn process_shard_removal_certificate(
        &self,
        shard_removal_certificate: Verified<ShardCertificate<ShardRemoval>>,
    ) -> ShardResult<()> {
        self.send(
            &self.shard_removal_certificate_sender,
            shard_removal_certificate,
        )
        .await;
        Ok(())
    }

    async fn process_shard_endorsement_certificate(
        &self,
        shard_endorsement_certificate: Verified<ShardCertificate<Signed<ShardEndorsement>>>,
    ) -> ShardResult<()> {
        self.send(
            &self.shard_endorsement_certificate_sender,
            shard_endorsement_certificate,
        )
        .await;
        Ok(())
    }

    async fn process_shard_completion_proof(
        &self,
        shard_completion_proof: Verified<ShardCompletionProof>,
    ) -> ShardResult<()> {
        self.send(&self.shard_completion_proof_sender, shard_completion_proof)
            .await;
        Ok(())
    }
}
