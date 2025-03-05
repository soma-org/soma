use fastcrypto::bls12381::min_sig;
use serde::Serialize;
use shared::{signed::Signature, verified::Verified};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinSet;
use tokio::time::sleep;

use crate::{
    error::ShardResult,
    networking::messaging::EncoderInternalNetworkClient,
    types::{encoder_committee::EncoderIndex, encoder_context::EncoderContext},
};

const MAX_RETRY_INTERVAL: Duration = Duration::from_secs(10);

pub(crate) struct Broadcaster<C: EncoderInternalNetworkClient> {
    context: Arc<EncoderContext>,
    client: Arc<C>,
}

impl<C: EncoderInternalNetworkClient> Broadcaster<C> {
    pub(crate) fn new(context: Arc<EncoderContext>, client: Arc<C>) -> Self {
        Self { context, client }
    }

    pub(crate) async fn broadcast<T, F, Fut>(
        &self,
        input: Verified<T>,
        peers: Vec<EncoderIndex>,
        network_fn: F,
    ) -> ShardResult<()>
    where
        T: Serialize + Send + Sync + 'static,
        F: FnOnce(Arc<C>, EncoderIndex, Verified<T>) -> Fut + Copy + Send + 'static,
        Fut: Future<Output = ShardResult<()>> + Send + 'static,
    {
        struct NetworkingResult {
            peer: EncoderIndex,
            result: ShardResult<()>,
            retries: u64,
        }

        let mut join_set: JoinSet<NetworkingResult> = JoinSet::new();

        // for each shard member
        for peer in peers {
            let client = self.client.clone();
            let cloned_input = input.clone();
            join_set.spawn(async move {
                let result = network_fn(client, peer, cloned_input).await;
                NetworkingResult {
                    peer,
                    result,
                    retries: 0,
                }
            });
        }

        while let Some(result) = join_set.join_next().await {
            if let Ok(result) = result {
                match result.result {
                    Ok(_) => {}

                    Err(e) => {
                        let client = self.client.clone();
                        let peer = result.peer;
                        let retries = result.retries + 1;
                        // TODO: potentially change retry increase to be exponential backoff
                        let retry_interval = Duration::from_secs(retries);
                        if retry_interval <= MAX_RETRY_INTERVAL {
                            let cloned_input = input.clone();

                            join_set.spawn(async move {
                                sleep(retry_interval).await;
                                let result = network_fn(client, peer, cloned_input).await;
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
        }
        Ok(())
    }
}
