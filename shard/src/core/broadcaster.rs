use serde::Serialize;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinSet;
use tokio::time::sleep;

use crate::{
    error::{ShardError, ShardResult},
    networking::messaging::EncoderNetworkClient,
    types::{context::EncoderContext, network_committee::NetworkingIndex, signed::Signature},
};

const MAX_RETRY_INTERVAL: Duration = Duration::from_secs(10);

pub(crate) struct Broadcaster<C: EncoderNetworkClient> {
    context: Arc<EncoderContext>,
    network_client: Arc<C>,
}

impl<C: EncoderNetworkClient> Broadcaster<C> {
    pub(crate) fn new(context: Arc<EncoderContext>, network_client: Arc<C>) -> Self {
        Self {
            context,
            network_client,
        }
    }

    pub(crate) async fn collect_signatures<T, F, Fut>(
        &self,
        input: T,
        peers: Vec<NetworkingIndex>,
        network_fn: F,
    ) -> ShardResult<Vec<Signature<T>>>
    where
        T: Serialize + Send + Sync + 'static,
        F: FnOnce(Arc<C>, NetworkingIndex) -> Fut + Copy + Send + Sync + 'static,
        Fut: Future<Output = ShardResult<Signature<T>>> + Send + Sync + 'static,
    {
        struct NetworkingResult<T: Serialize + Send + 'static> {
            peer: NetworkingIndex,
            result: ShardResult<Signature<T>>,
            retries: u64,
        }

        let mut join_set: JoinSet<NetworkingResult<T>> = JoinSet::new();
        let mut valid_signatures = Vec::new();

        // for each shard member
        for peer in peers {
            let client = self.network_client.clone();
            join_set.spawn(async move {
                let result = network_fn(client, peer).await;
                NetworkingResult {
                    peer,
                    result,
                    retries: 0,
                }
            });
        }

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(result) => {
                    match result.result {
                        Ok(signature) => {
                            // assume signature validation occurs inside of the generic function
                            valid_signatures.push(signature);
                            if valid_signatures.len() >= self.context.quorum {
                                // We hit quorum, can exit early
                                join_set.abort_all();
                                return Ok(valid_signatures);
                            }
                        }

                        Err(e) => {
                            match e {
                                ShardError::ConflictingRequest => {
                                    // TODO: do not retry
                                    // add count to conflicts
                                    // abort if unique conflicts > (shard_size - quorum)
                                }
                                _ => {
                                    let client = self.network_client.clone();
                                    let peer = result.peer;
                                    let retries = result.retries + 1;
                                    // TODO: potentially change retry increase to be exponential backoff
                                    let retry_interval = Duration::from_secs(retries);
                                    if retry_interval <= MAX_RETRY_INTERVAL {
                                        join_set.spawn(async move {
                                            sleep(retry_interval).await;
                                            let result = network_fn(client, peer).await;
                                            NetworkingResult {
                                                peer,
                                                result,
                                                retries,
                                            }
                                        });
                                    } else {
                                        // add to count of conflicts because we can stop early if
                                        // abort if unique conflicts > (shard_size - quorum)
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => {}
            }
        }

        Err(ShardError::QuorumFailed)
    }

    pub(crate) async fn broadcast<T, F, Fut>(
        &self,
        input: T,
        peers: Vec<NetworkingIndex>,
        network_fn: F,
    ) -> ShardResult<()>
    where
        T: Serialize + Send + 'static,
        F: FnOnce(Arc<C>, NetworkingIndex) -> Fut + Copy + Send + 'static,
        Fut: Future<Output = ShardResult<()>> + Send + 'static,
    {
        struct NetworkingResult {
            peer: NetworkingIndex,
            result: ShardResult<()>,
            retries: u64,
        }

        let mut join_set: JoinSet<NetworkingResult> = JoinSet::new();

        // for each shard member
        for peer in peers {
            let client = self.network_client.clone();
            join_set.spawn(async move {
                let result = network_fn(client, peer).await;
                NetworkingResult {
                    peer,
                    result,
                    retries: 0,
                }
            });
        }

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(result) => {
                    match result.result {
                        Ok(_) => {}

                        Err(e) => {
                            let client = self.network_client.clone();
                            let peer = result.peer;
                            let retries = result.retries + 1;
                            // TODO: potentially change retry increase to be exponential backoff
                            let retry_interval = Duration::from_secs(retries);
                            if retry_interval <= MAX_RETRY_INTERVAL {
                                join_set.spawn(async move {
                                    sleep(retry_interval).await;
                                    let result = network_fn(client, peer).await;
                                    NetworkingResult {
                                        peer,
                                        result,
                                        retries,
                                    }
                                });
                            }
                        }
                    }
                }
                Err(_) => {}
            }
        }
        Ok(())
    }
}

// given some shard and a message, broadcast the message to the entire shard. Go ahead and
