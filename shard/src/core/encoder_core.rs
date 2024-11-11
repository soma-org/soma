// use crate::networking::messaging::MESSAGE_TIMEOUT;
use crate::{
    networking::messaging::EncoderNetworkClient,
    types::{
        certificate::ShardCertificate, shard_commit::ShardCommit,
        shard_completion_proof::ShardCompletionProof, shard_endorsement::ShardEndorsement,
        shard_input::ShardInput, shard_removal::ShardRemoval, shard_reveal::ShardReveal,
        signed::Signed, verified::Verified,
    },
};
use std::sync::Arc;
use tokio::sync::Semaphore;

pub struct EncoderCore<C: EncoderNetworkClient> {
    shard_input_semaphore: Arc<Semaphore>,
    shard_commit_certificate_semaphore: Arc<Semaphore>,
    shard_reveal_certificate_semaphore: Arc<Semaphore>,
    shard_removal_certificate_semaphore: Arc<Semaphore>,
    shard_endorsement_certificate_semaphore: Arc<Semaphore>,
    shard_completion_proof_semaphore: Arc<Semaphore>,
    client: Arc<C>,
}

impl<C> EncoderCore<C>
where
    C: EncoderNetworkClient,
{
    pub fn new(max_concurrent_tasks: usize, client: Arc<C>) -> Self {
        Self {
            shard_input_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            shard_commit_certificate_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            shard_reveal_certificate_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            shard_removal_certificate_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            shard_endorsement_certificate_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            shard_completion_proof_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            client,
        }
    }

    pub async fn process_shard_input(&self, shard_input: Verified<Signed<ShardInput>>) {
        if let Ok(permit) = self.shard_input_semaphore.clone().acquire_owned().await {
            tokio::spawn(async move {
                println!("{:?}", shard_input);
                drop(permit);
            });
        }
    }

    pub async fn process_shard_commit_certificate(
        &self,
        shard_commit_certificate: Verified<ShardCertificate<Signed<ShardCommit>>>,
    ) {
        if let Ok(permit) = self
            .shard_commit_certificate_semaphore
            .clone()
            .acquire_owned()
            .await
        {
            tokio::spawn(async move {
                println!("{:?}", shard_commit_certificate);
                drop(permit);
            });
        }
    }

    pub async fn process_shard_reveal_certificate(
        &self,
        shard_reveal_certificate: Verified<ShardCertificate<Signed<ShardReveal>>>,
    ) {
        if let Ok(permit) = self
            .shard_reveal_certificate_semaphore
            .clone()
            .acquire_owned()
            .await
        {
            tokio::spawn(async move {
                println!("{:?}", shard_reveal_certificate);
                drop(permit);
            });
        }
    }

    pub async fn process_shard_removal_certificate(
        &self,
        shard_removal_certificate: Verified<ShardCertificate<ShardRemoval>>,
    ) {
        if let Ok(permit) = self
            .shard_removal_certificate_semaphore
            .clone()
            .acquire_owned()
            .await
        {
            tokio::spawn(async move {
                println!("{:?}", shard_removal_certificate);
                drop(permit);
            });
        }
    }

    pub async fn process_shard_endorsement_certificate(
        &self,
        shard_endorsement_certificate: Verified<ShardCertificate<Signed<ShardEndorsement>>>,
    ) {
        if let Ok(permit) = self
            .shard_endorsement_certificate_semaphore
            .clone()
            .acquire_owned()
            .await
        {
            tokio::spawn(async move {
                println!("{:?}", shard_endorsement_certificate);
                drop(permit);
            });
        }
    }

    pub async fn process_shard_completion_proof(
        &self,
        shard_completion_proof: Verified<ShardCompletionProof>,
    ) {
        if let Ok(permit) = self
            .shard_completion_proof_semaphore
            .clone()
            .acquire_owned()
            .await
        {
            tokio::spawn(async move {
                println!("{:?}", shard_completion_proof);
                drop(permit);
            });
        }
    }
}

impl<C: EncoderNetworkClient> Clone for EncoderCore<C> {
    fn clone(&self) -> Self {
        Self {
            shard_input_semaphore: Arc::clone(&self.shard_input_semaphore),
            shard_commit_certificate_semaphore: Arc::clone(
                &self.shard_commit_certificate_semaphore,
            ),
            shard_reveal_certificate_semaphore: Arc::clone(
                &self.shard_reveal_certificate_semaphore,
            ),
            shard_removal_certificate_semaphore: Arc::clone(
                &self.shard_removal_certificate_semaphore,
            ),
            shard_endorsement_certificate_semaphore: Arc::clone(
                &self.shard_endorsement_certificate_semaphore,
            ),
            shard_completion_proof_semaphore: Arc::clone(&self.shard_completion_proof_semaphore),
            client: Arc::clone(&self.client),
        }
    }
}
