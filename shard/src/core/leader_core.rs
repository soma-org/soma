// use crate::networking::messaging::MESSAGE_TIMEOUT;
use crate::types::shard_endorsement::VerifiedSignedShardEndorsement;
use crate::{
    networking::messaging::EncoderNetworkClient, types::shard_commit::VerifiedSignedShardCommit,
};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// The core process that handles executing the underlying jobs for
pub struct LeaderCore<ENC: EncoderNetworkClient /*S: BlobStorage */> {
    semaphore: Arc<Semaphore>,
    encoder_client: Arc<ENC>,
}

impl<ENC> LeaderCore<ENC>
where
    ENC: EncoderNetworkClient, // S: BlobStorage,
{
    pub fn new(max_concurrent_tasks: usize, encoder_network_client: ENC) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            encoder_client: Arc::new(encoder_network_client),
            // blob_client: Arc::new(blob_client),
        }
    }

    pub async fn process_commit(&self, commit: VerifiedSignedShardCommit) {
        let permit = self.semaphore.clone().acquire_owned().await.unwrap();

        tokio::spawn(async move {
            drop(permit);
        });
    }

    pub async fn process_endorsement(&self, endorsement: VerifiedSignedShardEndorsement) {
        let permit = self.semaphore.clone().acquire_owned().await.unwrap();

        tokio::spawn(async move {
            drop(permit);
        });
    }
}
