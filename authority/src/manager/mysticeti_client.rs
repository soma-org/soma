use arc_swap::{ArcSwapOption, Guard};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tap::tap::TapFallible;
use tokio::time::sleep;
use tracing::warn;
use types::{
    consensus::{ConsensusTransaction, ConsensusTransactionKind},
    error::{SomaError, SomaResult},
};

use crate::{
    adapter::SubmitToConsensus, epoch_store::AuthorityPerEpochStore,
    handler::SequencedConsensusTransactionKey,
};
use consensus::TransactionClient;

/// Gets a client to submit transactions to Mysticeti, or waits for one to be available.
/// This hides the complexities of async consensus initialization and submitting to different
/// instances of consensus across epochs.
#[derive(Default, Clone)]
pub struct LazyMysticetiClient {
    client: Arc<ArcSwapOption<TransactionClient>>,
}

impl LazyMysticetiClient {
    pub fn new() -> Self {
        Self {
            client: Arc::new(ArcSwapOption::empty()),
        }
    }

    async fn get(&self) -> Guard<Option<Arc<TransactionClient>>> {
        let client = self.client.load();
        if client.is_some() {
            return client;
        }

        // Consensus client is initialized after validators or epoch starts, and cleared after an epoch ends.
        // But calls to get() can happen during validator startup or epoch change, before consensus finished
        // initializations.
        // TODO: maybe listen to updates from consensus manager instead of polling.
        let mut count = 0;
        let start = Instant::now();
        const RETRY_INTERVAL: Duration = Duration::from_millis(100);
        loop {
            let client = self.client.load();
            if client.is_some() {
                return client;
            } else {
                sleep(RETRY_INTERVAL).await;
                count += 1;
                if count % 100 == 0 {
                    warn!(
                        "Waiting for consensus to initialize after {:?}",
                        Instant::now() - start
                    );
                }
            }
        }
    }

    pub fn set(&self, client: Arc<TransactionClient>) {
        self.client.store(Some(client));
    }

    pub fn clear(&self) {
        self.client.store(None);
    }
}

#[async_trait::async_trait]
impl SubmitToConsensus for LazyMysticetiClient {
    async fn submit_to_consensus(
        &self,
        transactions: &[ConsensusTransaction],
        _epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult {
        // TODO(mysticeti): confirm comment is still true
        // The retrieved TransactionClient can be from the past epoch. Submit would fail after
        // Mysticeti shuts down, so there should be no correctness issue.
        let client = self.get().await;
        let transactions_bytes = transactions
            .iter()
            .map(|t| bcs::to_bytes(t).expect("Serializing consensus transaction cannot fail"))
            .collect::<Vec<_>>();
        let block_ref = client
            .as_ref()
            .expect("Client should always be returned")
            .submit(transactions_bytes)
            .await
            .tap_err(|r| {
                // Will be logged by caller as well.
                warn!("Submit transactions failed with: {:?}", r);
            })
            .map_err(|err| SomaError::FailedToSubmitToConsensus(err.to_string()))?;

        let is_soft_bundle = transactions.len() > 1;

        if !is_soft_bundle
            && matches!(
                transactions[0].kind,
                ConsensusTransactionKind::EndOfPublish(_) // | ConsensusTransactionKind::CapabilityNotification(_)
                                                          // | ConsensusTransactionKind::CapabilityNotificationV2(_)
                                                          // | ConsensusTransactionKind::RandomnessDkgMessage(_, _)
                                                          // | ConsensusTransactionKind::RandomnessDkgConfirmation(_, _)
            )
        {
            let transaction_key = SequencedConsensusTransactionKey::External(transactions[0].key());
            tracing::info!("Transaction {transaction_key:?} was included in {block_ref}",)
        };
        Ok(())
    }
}
