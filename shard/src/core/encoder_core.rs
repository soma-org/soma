// use crate::networking::messaging::MESSAGE_TIMEOUT;
use crate::{
    networking::messaging::EncoderNetworkClient,
    types::{shard_input::ShardInput, signed::Signed, verified::Verified},
};
use std::sync::Arc;
use tokio::sync::Semaphore;

pub struct EncoderCore<C: EncoderNetworkClient> {
    semaphore: Arc<Semaphore>,
    client: Arc<C>,
}

impl<C> EncoderCore<C>
where
    C: EncoderNetworkClient,
{
    pub fn new(max_concurrent_tasks: usize, client: Arc<C>) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            client,
            // blob_client: Arc::new(blob_client),
        }
    }

    pub async fn process_shard_input(&self, shard_input: Verified<Signed<ShardInput>>) {
        let permit = self.semaphore.clone().acquire_owned().await.unwrap();

        tokio::spawn(async move {
            drop(permit);
        });
    }

    // pub async fn process_selection(&self, selection: VerifiedSignedShardSelection) {
    //     let permit = self.semaphore.clone().acquire_owned().await.unwrap();

    //     tokio::spawn(async move {
    //         drop(permit);
    //     });
    // }
}
