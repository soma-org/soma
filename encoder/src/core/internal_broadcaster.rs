use serde::Serialize;
use shared::{crypto::keys::EncoderPublicKey, verified::Verified};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tokio::{sync::Semaphore, task::JoinSet};

use crate::{error::ShardResult, messaging::EncoderInternalNetworkClient};

const MAX_RETRY_INTERVAL: Duration = Duration::from_secs(10);

pub(crate) struct Broadcaster<C: EncoderInternalNetworkClient> {
    client: Arc<C>,
    semaphore: Arc<Semaphore>,
}

impl<C: EncoderInternalNetworkClient> Broadcaster<C> {
    pub(crate) fn new(client: Arc<C>, semaphore: Arc<Semaphore>) -> Self {
        Self { client, semaphore }
    }

    pub(crate) async fn broadcast<T, F, Fut>(
        &self,
        input: Verified<T>,
        peers: Vec<EncoderPublicKey>,
        network_fn: F,
    ) -> ShardResult<()>
    where
        T: Serialize + Send + Sync + 'static,
        F: FnOnce(Arc<C>, EncoderPublicKey, Verified<T>) -> Fut + Copy + Send + 'static,
        Fut: Future<Output = ShardResult<()>> + Send + 'static,
    {
        struct NetworkingResult {
            peer: EncoderPublicKey,
            result: ShardResult<()>,
            retries: u64,
        }

        let mut join_set: JoinSet<NetworkingResult> = JoinSet::new();

        // for each shard member
        for peer in peers {
            let client = self.client.clone();
            let cloned_input = input.clone();
            let sema_clone = self.semaphore.clone();
            join_set.spawn(async move {
                if let Ok(permit) = sema_clone.acquire_owned().await {
                    let result = network_fn(client, peer.clone(), cloned_input).await;
                    drop(permit);
                    NetworkingResult {
                        peer,
                        result,
                        retries: 0,
                    }
                } else {
                    NetworkingResult {
                        peer,
                        result: Err(crate::error::ShardError::ConcurrencyError(
                            "could not acquire semaphore".to_string(),
                        )),
                        retries: 0,
                    }
                }
            });
        }

        while let Some(result) = join_set.join_next().await {
            if let Ok(result) = result {
                match result.result {
                    Ok(_) => {}

                    Err(e) => {
                        // TODO: deal with the semaphore error?
                        let client = self.client.clone();
                        let peer = result.peer;
                        let retries = result.retries + 1;
                        // TODO: potentially change retry increase to be exponential backoff
                        let retry_interval = Duration::from_secs(retries);
                        if retry_interval <= MAX_RETRY_INTERVAL {
                            let cloned_input = input.clone();

                            let sema_clone = self.semaphore.clone();
                            join_set.spawn(async move {
                                sleep(retry_interval).await;
                                if let Ok(permit) = sema_clone.acquire_owned().await {
                                    let result =
                                        network_fn(client, peer.clone(), cloned_input).await;
                                    drop(permit);
                                    NetworkingResult {
                                        peer,
                                        result,
                                        retries,
                                    }
                                } else {
                                    NetworkingResult {
                                        peer,
                                        result: Err(crate::error::ShardError::ConcurrencyError(
                                            "could not acquire semaphore".to_string(),
                                        )),
                                        retries,
                                    }
                                }
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
