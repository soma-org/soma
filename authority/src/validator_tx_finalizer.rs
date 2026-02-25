// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::authority_aggregator::AuthorityAggregator;
use crate::authority_client::AuthorityAPI;
use crate::authority_per_epoch_store::AuthorityPerEpochStore;
use crate::cache::TransactionCacheRead;
use arc_swap::ArcSwap;
use std::cmp::min;
use std::ops::Add;
use std::sync::Arc;
#[cfg(any(msim, test))]
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;
use tokio::select;
use tokio::time::Instant;
use tracing::{debug, error, trace};
use types::base::AuthorityName;
use types::digests::TransactionDigest;
use types::transaction::VerifiedSignedTransaction;

pub struct ValidatorTxFinalizerConfig {
    pub tx_finalization_delay: Duration,
    pub tx_finalization_timeout: Duration,
    /// Incremental delay for validators to wake up to finalize a transaction.
    pub validator_delay_increments_sec: u64,
    pub validator_max_delay: Duration,
}

#[cfg(not(any(msim, test)))]
impl Default for ValidatorTxFinalizerConfig {
    fn default() -> Self {
        Self {
            // Only wake up the transaction finalization task for a given transaction
            // after 1 mins of seeing it. This gives plenty of time for the transaction
            // to become final in the normal way. We also don't want this delay to be too long
            // to reduce memory usage held up by the finalizer threads.
            tx_finalization_delay: Duration::from_secs(60),
            // If a transaction can not be finalized within 1 min of being woken up, give up.
            tx_finalization_timeout: Duration::from_secs(60),
            validator_delay_increments_sec: 10,
            validator_max_delay: Duration::from_secs(180),
        }
    }
}

#[cfg(any(msim, test))]
impl Default for ValidatorTxFinalizerConfig {
    fn default() -> Self {
        Self {
            tx_finalization_delay: Duration::from_secs(5),
            tx_finalization_timeout: Duration::from_secs(60),
            validator_delay_increments_sec: 1,
            validator_max_delay: Duration::from_secs(15),
        }
    }
}

/// The `ValidatorTxFinalizer` is responsible for finalizing transactions that
/// have been signed by the validator. It does this by waiting for a delay
/// after the transaction has been signed, and then attempting to finalize
/// the transaction if it has not yet been done by a fullnode.
pub struct ValidatorTxFinalizer<C: Clone> {
    agg: Arc<ArcSwap<AuthorityAggregator<C>>>,
    name: AuthorityName,
    config: Arc<ValidatorTxFinalizerConfig>,
}

impl<C: Clone> ValidatorTxFinalizer<C> {
    pub fn new(agg: Arc<ArcSwap<AuthorityAggregator<C>>>, name: AuthorityName) -> Self {
        Self { agg, name, config: Arc::new(ValidatorTxFinalizerConfig::default()) }
    }

    #[cfg(test)]
    pub(crate) fn new_for_testing(
        agg: Arc<ArcSwap<AuthorityAggregator<C>>>,
        name: AuthorityName,
    ) -> Self {
        Self::new(agg, name)
    }

    #[cfg(test)]
    pub(crate) fn auth_agg(&self) -> &Arc<ArcSwap<AuthorityAggregator<C>>> {
        &self.agg
    }
}

impl<C> ValidatorTxFinalizer<C>
where
    C: Clone + AuthorityAPI + Send + Sync + 'static,
{
    pub async fn track_signed_tx(
        &self,
        cache_read: Arc<dyn TransactionCacheRead>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        tx: VerifiedSignedTransaction,
    ) {
        let tx_digest = *tx.digest();
        trace!(?tx_digest, "Tracking signed transaction");
        match self.delay_and_finalize_tx(cache_read, epoch_store, tx).await {
            Ok(did_run) => {
                if did_run {
                    debug!(?tx_digest, "Transaction finalized");
                }
            }
            Err(err) => {
                debug!(?tx_digest, "Failed to finalize transaction: {err}");
            }
        }
    }

    async fn delay_and_finalize_tx(
        &self,
        cache_read: Arc<dyn TransactionCacheRead>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        tx: VerifiedSignedTransaction,
    ) -> anyhow::Result<bool> {
        let tx_digest = *tx.digest();
        let Some(tx_finalization_delay) = self.determine_finalization_delay(&tx_digest) else {
            return Ok(false);
        };
        let digests = [tx_digest];
        select! {
            _ = tokio::time::sleep(tx_finalization_delay) => {
                trace!(?tx_digest, "Waking up to finalize transaction");
            }
            _ = cache_read.notify_read_executed_effects_digests(
                &digests,
            ) => {
                trace!(?tx_digest, "Transaction already finalized");
                return Ok(false);
            }
        }

        if epoch_store.is_pending_consensus_certificate(&tx_digest) {
            trace!(
                ?tx_digest,
                "Transaction has been submitted to consensus, no need to help drive finality"
            );
            return Ok(false);
        }

        debug!(?tx_digest, "Invoking authority aggregator to finalize transaction");
        tokio::time::timeout(
            self.config.tx_finalization_timeout,
            self.agg.load().execute_transaction_block(tx.into_unsigned().inner(), None),
        )
        .await??;

        Ok(true)
    }

    // We want to avoid all validators waking up at the same time to finalize the same transaction.
    // That can lead to a waste of resource and flood the network unnecessarily.
    // Here we use the transaction digest to determine an order of all validators.
    // Validators will wake up one by one with incremental delays to finalize the transaction.
    // The hope is that the first few should be able to finalize the transaction,
    // and the rest will see it already executed and do not need to do anything.
    fn determine_finalization_delay(&self, tx_digest: &TransactionDigest) -> Option<Duration> {
        let agg = self.agg.load();
        let order = agg.committee.shuffle_by_stake_from_tx_digest(tx_digest);
        let Some(position) = order.iter().position(|&name| name == self.name) else {
            // Somehow the validator is not found in the committee. This should never happen.
            // TODO: This is where we should report system invariant violation.
            error!("Validator {} not found in the committee", self.name);
            return None;
        };
        // TODO: As an optimization, we could also limit the number of validators that would do this.
        let extra_delay = position as u64 * self.config.validator_delay_increments_sec;
        let delay = self.config.tx_finalization_delay.add(Duration::from_secs(extra_delay));
        Some(min(delay, self.config.validator_max_delay))
    }
}
