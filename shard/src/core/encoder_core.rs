// use crate::networking::messaging::MESSAGE_TIMEOUT;
use crate::{
    networking::messaging::LeaderNetworkClient,
    types::{shard_input::VerifiedSignedShardInput, shard_selection::VerifiedSignedShardSelection},
};
use std::sync::Arc;
use tokio::sync::Semaphore;

pub struct EncoderCore<LNC: LeaderNetworkClient> {
    semaphore: Arc<Semaphore>,
    leader_client: Arc<LNC>,
}

impl<LNC> EncoderCore<LNC>
where
    LNC: LeaderNetworkClient,
{
    pub fn new(
        max_concurrent_tasks: usize,
        leader_network_client: LNC,
        /*blob_client: BlobClient<S>*/
    ) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            leader_client: Arc::new(leader_network_client),
            // blob_client: Arc::new(blob_client),
        }
    }

    pub async fn process_input(&self, input: VerifiedSignedShardInput) {
        let permit = self.semaphore.clone().acquire_owned().await.unwrap();

        tokio::spawn(async move {
            drop(permit);
        });
    }

    pub async fn process_selection(&self, selection: VerifiedSignedShardSelection) {
        let permit = self.semaphore.clone().acquire_owned().await.unwrap();

        tokio::spawn(async move {
            drop(permit);
        });
    }
}
