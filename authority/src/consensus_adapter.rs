// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::future::Future;
use std::net::IpAddr;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

use arc_swap::{ArcSwap, ArcSwapOption};
use consensus::BlockStatus;
use dashmap::DashMap;
use dashmap::try_result::TryResult;
use futures::FutureExt;
use futures::future::{self, Either, select};
use futures::stream::FuturesUnordered;
use futures::{StreamExt, pin_mut};
use itertools::Itertools;
use parking_lot::RwLockReadGuard;
use protocol_config::ProtocolConfig;
use tokio::sync::{Semaphore, SemaphorePermit, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Duration;
use tokio::time::{self};
use tracing::{Instrument, debug, debug_span, info, instrument, trace, warn};
use types::base::AuthorityName;
use types::committee::Committee;
use types::consensus::ConsensusPosition;
use types::consensus::ConsensusTransactionKind;
use types::consensus::{ConsensusTransaction, ConsensusTransactionKey};
use types::digests::TransactionDigest;
use types::error::{SomaError, SomaResult};
use types::peer_id::ConnectionStatus;

use crate::authority_per_epoch_store::AuthorityPerEpochStore;
use crate::checkpoints::CheckpointStore;
use crate::consensus_handler::{SequencedConsensusTransactionKey, classify};
use crate::reconfiguration::{ReconfigState, ReconfigurationInitiator};

pub type BlockStatusReceiver = oneshot::Receiver<BlockStatus>;

#[mockall::automock]
pub trait SubmitToConsensus: Sync + Send + 'static {
    fn submit_to_consensus(
        &self,
        transactions: &[ConsensusTransaction],
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult;

    fn submit_best_effort(
        &self,
        transaction: &ConsensusTransaction,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        timeout: Duration,
    ) -> SomaResult;
}

#[mockall::automock]
#[async_trait::async_trait]
pub trait ConsensusClient: Sync + Send + 'static {
    async fn submit(
        &self,
        transactions: &[ConsensusTransaction],
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<(Vec<ConsensusPosition>, BlockStatusReceiver)>;
}

/// Submit Sui certificates to the consensus.
pub struct ConsensusAdapter {
    /// The network client connecting to the consensus node of this authority.
    consensus_client: Arc<dyn ConsensusClient>,
    /// The checkpoint store for the validator
    checkpoint_store: Arc<CheckpointStore>,
    /// Authority pubkey.
    authority: AuthorityName,
    /// The limit to number of inflight transactions at this node.
    max_pending_transactions: usize,
    /// Number of submitted transactions still inflight at this node.
    num_inflight_transactions: AtomicU64,
    /// Dictates the maximum position  from which will submit to consensus. Even if the is elected to
    /// submit from a higher position than this, it will "reset" to the max_submit_position.
    max_submit_position: Option<usize>,
    /// When provided it will override the current back off logic and will use this value instead
    /// as delay step.
    submit_delay_step_override: Option<Duration>,
    /// A structure to check the reputation scores populated by Consensus
    low_scoring_authorities: ArcSwap<Arc<ArcSwap<HashMap<AuthorityName, u64>>>>,

    /// Semaphore limiting parallel submissions to consensus
    submit_semaphore: Arc<Semaphore>,

    protocol_config: ProtocolConfig,
}

pub struct ConnectionMonitorStatusForTests {}

impl ConsensusAdapter {
    /// Make a new Consensus adapter instance.
    pub fn new(
        consensus_client: Arc<dyn ConsensusClient>,
        checkpoint_store: Arc<CheckpointStore>,
        authority: AuthorityName,
        max_pending_transactions: usize,
        max_pending_local_submissions: usize,
        max_submit_position: Option<usize>,
        submit_delay_step_override: Option<Duration>,
        protocol_config: ProtocolConfig,
    ) -> Self {
        let num_inflight_transactions = Default::default();
        let low_scoring_authorities =
            ArcSwap::from_pointee(Arc::new(ArcSwap::from_pointee(HashMap::new())));
        Self {
            consensus_client,
            checkpoint_store,
            authority,
            max_pending_transactions,
            max_submit_position,
            submit_delay_step_override,
            num_inflight_transactions,
            low_scoring_authorities,
            submit_semaphore: Arc::new(Semaphore::new(max_pending_local_submissions)),
            protocol_config,
        }
    }

    pub fn swap_low_scoring_authorities(
        &self,
        new_low_scoring: Arc<ArcSwap<HashMap<AuthorityName, u64>>>,
    ) {
        self.low_scoring_authorities.swap(Arc::new(new_low_scoring));
    }

    /// Get the current number of in-flight transactions
    pub fn num_inflight_transactions(&self) -> u64 {
        self.num_inflight_transactions.load(Ordering::Relaxed)
    }

    pub fn submit_recovered(self: &Arc<Self>, epoch_store: &Arc<AuthorityPerEpochStore>) {
        // Transactions being sent to consensus can be dropped on crash, before included in a proposed block.
        // System transactions do not have clients to retry them. They need to be resubmitted to consensus on restart.
        // get_all_pending_consensus_transactions() can return both system and certified transactions though.
        //
        // todo - get_all_pending_consensus_transactions is called twice when
        // initializing AuthorityPerEpochStore and here, should not be a big deal but can be optimized
        let mut recovered = epoch_store.get_all_pending_consensus_transactions();

        #[allow(clippy::collapsible_if)] // This if can be collapsed but it will be ugly
        if epoch_store.should_send_end_of_publish() {
            if !recovered.iter().any(ConsensusTransaction::is_end_of_publish) {
                // There are two cases when this is needed
                // (1) We send EndOfPublish message after removing pending certificates in submit_and_wait_inner
                // It is possible that node will crash between those two steps, in which case we might need to
                // re-introduce EndOfPublish message on restart
                // (2) If node crashed inside ConsensusAdapter::close_epoch,
                // after reconfig lock state was written to DB and before we persisted EndOfPublish message
                recovered.push(ConsensusTransaction::new_end_of_publish(self.authority));
            }
        }
        debug!(
            "Submitting {:?} recovered pending consensus transactions to consensus",
            recovered.len()
        );
        for transaction in recovered {
            if transaction.is_end_of_publish() {
                info!(epoch=?epoch_store.epoch(), "Submitting EndOfPublish message to consensus");
            }
            self.submit_unchecked(&[transaction], epoch_store, None, None);
        }
    }

    fn await_submit_delay(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        transactions: &[ConsensusTransaction],
    ) -> (impl Future<Output = ()>, usize, usize, usize) {
        if transactions.iter().any(|tx| tx.is_mfp_transaction()) {
            // UserTransactions are generally sent to just one validator and should
            // be submitted to consensus without delay.
            return (tokio::time::sleep(Duration::ZERO), 0, 0, 0);
        }

        // Use the minimum digest to compute submit delay.
        let min_digest = transactions
            .iter()
            .filter_map(|tx| match &tx.kind {
                ConsensusTransactionKind::CertifiedTransaction(certificate) => {
                    Some(certificate.digest())
                }
                ConsensusTransactionKind::UserTransaction(transaction) => {
                    Some(transaction.digest())
                }
                _ => None,
            })
            .min();
        let mut amplification_factor = 0;

        let (duration, position, positions_moved, preceding_disconnected) = match min_digest {
            Some(digest) => {
                self.await_submit_delay_user_transaction(epoch_store.committee(), digest)
            }
            _ => (Duration::ZERO, 0, 0, 0),
        };
        (tokio::time::sleep(duration), position, positions_moved, preceding_disconnected)
    }

    fn await_submit_delay_user_transaction(
        &self,
        committee: &Committee,
        tx_digest: &TransactionDigest,
    ) -> (Duration, usize, usize, usize) {
        let (mut position, positions_moved, preceding_disconnected) =
            self.submission_position(committee, tx_digest);

        let (delay_step, position) = self.override_by_max_submit_position_settings(position);

        (delay_step * position as u32, position, positions_moved, preceding_disconnected)
    }

    /// Overrides the latency and the position if there are defined settings for `max_submit_position` and
    /// `submit_delay_step_override`. If the `max_submit_position` has defined, then that will always be used
    /// irrespective of any so far decision. Same for the `submit_delay_step_override`.
    fn override_by_max_submit_position_settings(&self, mut position: usize) -> (Duration, usize) {
        // Respect any manual override for position and latency from the settings
        if let Some(max_submit_position) = self.max_submit_position {
            position = std::cmp::min(position, max_submit_position);
        }

        let delay_step = self.submit_delay_step_override.unwrap_or(Duration::from_secs(1));
        (delay_step, position)
    }

    /// Check when this authority should submit the certificate to consensus.
    /// This sorts all authorities based on pseudo-random distribution derived from transaction hash.
    ///
    /// The function targets having 1 consensus transaction submitted per user transaction
    /// when system operates normally.
    ///
    /// The function returns the position of this authority when it is their turn to submit the transaction to consensus.
    fn submission_position(
        &self,
        committee: &Committee,
        tx_digest: &TransactionDigest,
    ) -> (usize, usize, usize) {
        let positions = committee.shuffle_by_stake_from_tx_digest(tx_digest);

        self.check_submission_wrt_connectivity_and_scores(positions)
    }

    /// This function runs the following algorithm to decide whether or not to submit a transaction
    /// to consensus.
    ///
    /// It takes in a deterministic list that represents positions of all the authorities.
    /// The authority in the first position will be responsible for submitting to consensus, and
    /// so we check if we are this validator, and if so, return true.
    ///
    /// If we are not in that position, we check our connectivity to the authority in that position.
    /// If we are connected to them, we can assume that they are operational and will submit the transaction.
    /// If we are not connected to them, we assume that they are not operational and we will not rely
    /// on that authority to submit the transaction. So we shift them out of the first position, and
    /// run this algorithm again on the new set of positions.
    ///
    /// This can possibly result in a transaction being submitted twice if an authority sees a false
    /// negative in connectivity to another, such as in the case of a network partition.
    ///
    /// Recursively, if the authority further ahead of us in the positions is a low performing authority, we will
    /// move our positions up one, and submit the transaction. This allows maintaining performance
    /// overall. We will only do this part for authorities that are not low performers themselves to
    /// prevent extra amplification in the case that the positions look like [low_scoring_a1, low_scoring_a2, a3]
    fn check_submission_wrt_connectivity_and_scores(
        &self,
        positions: Vec<AuthorityName>,
    ) -> (usize, usize, usize) {
        let low_scoring_authorities = self.low_scoring_authorities.load().load_full();
        if low_scoring_authorities.get(&self.authority).is_some() {
            return (positions.len(), 0, 0);
        }
        let initial_position = get_position_in_list(self.authority, positions.clone());
        let mut preceding_disconnected = 0;
        let mut before_our_position = true;

        let filtered_positions: Vec<_> = positions
            .into_iter()
            .filter(|authority| {
                let keep = self.authority == *authority; // don't filter ourself out
                if keep {
                    before_our_position = false;
                }

                // Filter out low scoring nodes
                let high_scoring = low_scoring_authorities.get(authority).is_none();

                keep || (high_scoring)
            })
            .collect();

        let position = get_position_in_list(self.authority, filtered_positions);

        (position, initial_position - position, preceding_disconnected)
    }

    /// This method blocks until transaction is persisted in local database
    /// It then returns handle to async task, user can join this handle to await while transaction is processed by consensus
    ///
    /// This method guarantees that once submit(but not returned async handle) returns,
    /// transaction is persisted and will eventually be sent to consensus even after restart
    ///
    /// When submitting a certificate caller **must** provide a ReconfigState lock guard
    pub fn submit(
        self: &Arc<Self>,
        transaction: ConsensusTransaction,
        lock: Option<&RwLockReadGuard<ReconfigState>>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        tx_consensus_position: Option<oneshot::Sender<Vec<ConsensusPosition>>>,
        submitter_client_addr: Option<IpAddr>,
    ) -> SomaResult<JoinHandle<()>> {
        self.submit_batch(
            &[transaction],
            lock,
            epoch_store,
            tx_consensus_position,
            submitter_client_addr,
        )
    }

    // Submits the provided transactions to consensus in a batched fashion. The `transactions` vector can be also empty in case of a ping check.
    // In this case the system will simulate a transaction submission to consensus and return the consensus position.
    pub fn submit_batch(
        self: &Arc<Self>,
        transactions: &[ConsensusTransaction],
        lock: Option<&RwLockReadGuard<ReconfigState>>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        tx_consensus_position: Option<oneshot::Sender<Vec<ConsensusPosition>>>,
        submitter_client_addr: Option<IpAddr>,
    ) -> SomaResult<JoinHandle<()>> {
        if !transactions.is_empty() {
            epoch_store.insert_pending_consensus_transactions(transactions, lock)?;
        }

        Ok(self.submit_unchecked(
            transactions,
            epoch_store,
            tx_consensus_position,
            submitter_client_addr,
        ))
    }

    /// Performs weakly consistent checks on internal buffers to quickly
    /// discard transactions if we are overloaded
    fn check_limits(&self) -> bool {
        // First check total transactions (waiting and in submission)
        if self.num_inflight_transactions.load(Ordering::Relaxed) as usize
            > self.max_pending_transactions
        {
            return false;
        }
        // Then check if submit_semaphore has permits
        self.submit_semaphore.available_permits() > 0
    }

    fn submit_unchecked(
        self: &Arc<Self>,
        transactions: &[ConsensusTransaction],
        epoch_store: &Arc<AuthorityPerEpochStore>,
        tx_consensus_position: Option<oneshot::Sender<Vec<ConsensusPosition>>>,
        submitter_client_addr: Option<IpAddr>,
    ) -> JoinHandle<()> {
        // Reconfiguration lock is dropped when pending_consensus_transactions is persisted, before it is handled by consensus
        let async_stage = self
            .clone()
            .submit_and_wait(
                transactions.to_vec(),
                epoch_store.clone(),
                tx_consensus_position,
                submitter_client_addr,
            )
            .in_current_span();
        // Number of these tasks is weakly limited based on `num_inflight_transactions`.
        // (Limit is not applied atomically, and only to user transactions.)
        tokio::spawn(async_stage)
    }

    async fn submit_and_wait(
        self: Arc<Self>,
        transactions: Vec<ConsensusTransaction>,
        epoch_store: Arc<AuthorityPerEpochStore>,
        tx_consensus_position: Option<oneshot::Sender<Vec<ConsensusPosition>>>,
        submitter_client_addr: Option<IpAddr>,
    ) {
        // When epoch_terminated signal is received all pending submit_and_wait_inner are dropped.
        //
        // This is needed because submit_and_wait_inner waits on read_notify for consensus message to be processed,
        // which may never happen on epoch boundary.
        //
        // In addition to that, within_alive_epoch ensures that all pending consensus
        // adapter tasks are stopped before reconfiguration can proceed.
        //
        // This is essential because narwhal workers reuse same ports when narwhal restarts,
        // this means we might be sending transactions from previous epochs to narwhal of
        // new epoch if we have not had this barrier.
        epoch_store
            .within_alive_epoch(self.submit_and_wait_inner(
                transactions,
                &epoch_store,
                tx_consensus_position,
                submitter_client_addr,
            ))
            .await
            .ok(); // result here indicates if epoch ended earlier, we don't care about it
    }

    #[allow(clippy::option_map_unit_fn)]
    #[instrument(name="ConsensusAdapter::submit_and_wait_inner", level="trace", skip_all, fields(tx_count = ?transactions.len(), tx_type = tracing::field::Empty, tx_keys = tracing::field::Empty, submit_status = tracing::field::Empty, consensus_positions = tracing::field::Empty))]
    async fn submit_and_wait_inner(
        self: Arc<Self>,
        transactions: Vec<ConsensusTransaction>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        mut tx_consensus_positions: Option<oneshot::Sender<Vec<ConsensusPosition>>>,
        submitter_client_addr: Option<IpAddr>,
    ) {
        if transactions.is_empty() {
            // If transactions are empty, then we attempt to ping consensus and simulate a transaction submission to consensus.
            // We intentionally do not wait for the block status, as we are only interested in the consensus position and return it immediately.
            debug!(
                "Performing a ping check, pinging consensus to get a consensus position in next block"
            );
            let (consensus_positions, _status_waiter) =
                self.submit_inner(&transactions, epoch_store, &[], "ping", false).await;

            if let Some(tx_consensus_positions) = tx_consensus_positions.take() {
                let _ = tx_consensus_positions.send(consensus_positions);
            } else {
                debug!("Ping check must have a consensus position channel");
            }
            return;
        }

        // Record submitted transactions early for DoS protection
        for transaction in &transactions {
            if let ConsensusTransactionKind::UserTransaction(tx) = &transaction.kind {
                epoch_store
                    .submitted_transaction_cache
                    .record_submitted_tx(tx.digest(), submitter_client_addr);
            }
        }

        // If tx_consensus_positions channel is provided, the caller is looking for a
        // consensus position for mfp. Therefore we will skip shortcutting submission
        // if txes have already been processed.
        let skip_processed_checks = tx_consensus_positions.is_some();

        // Current code path ensures:
        // - If transactions.len() > 1, it is a soft bundle. System transactions should have been submitted individually.
        // - If is_soft_bundle, then all transactions are of CertifiedTransaction or UserTransaction kind.
        // - If not is_soft_bundle, then transactions must contain exactly 1 tx, and transactions[0] can be of any kind.
        let is_soft_bundle = transactions.len() > 1;

        let mut transaction_keys = Vec::new();
        let mut tx_consensus_positions = tx_consensus_positions;

        for transaction in &transactions {
            if matches!(transaction.kind, ConsensusTransactionKind::EndOfPublish(..)) {
                info!(epoch=?epoch_store.epoch(), "Submitting EndOfPublish message to consensus");
            }

            let transaction_key = SequencedConsensusTransactionKey::External(transaction.key());
            transaction_keys.push(transaction_key);
        }
        let tx_type = if is_soft_bundle { "soft_bundle" } else { classify(&transactions[0]) };
        tracing::Span::current().record("tx_type", tx_type);
        tracing::Span::current().record("tx_keys", tracing::field::debug(&transaction_keys));

        let mut guard = InflightDropGuard::acquire(&self, tx_type);

        // Create the waiter until the node's turn comes to submit to consensus
        let (await_submit, position, positions_moved, preceding_disconnected) =
            self.await_submit_delay(epoch_store, &transactions[..]);

        let processed_via_consensus_or_checkpoint = if skip_processed_checks {
            // If we need to get consensus position, don't bypass consensus submission
            // for tx digest returned from consensus/checkpoint processing
            future::pending().boxed()
        } else {
            self.await_consensus_or_checkpoint(transaction_keys.clone(), epoch_store).boxed()
        };
        pin_mut!(processed_via_consensus_or_checkpoint);

        let processed_waiter = tokio::select! {
            // We need to wait for some delay until we submit transaction to the consensus
            _ = await_submit => Some(processed_via_consensus_or_checkpoint),

            // If epoch ends, don't wait for submit delay
            _ = epoch_store.user_certs_closed_notify() => {
                warn!(epoch = ?epoch_store.epoch(), "Epoch ended, skipping submission delay");
                Some(processed_via_consensus_or_checkpoint)
            }

            // If transaction is received by consensus or checkpoint while we wait, we are done.
            _ = &mut processed_via_consensus_or_checkpoint => {
                None
            }
        };

        // Log warnings for administrative transactions that fail to get sequenced
        let _monitor = if matches!(transactions[0].kind, ConsensusTransactionKind::EndOfPublish(_))
        {
            assert!(!is_soft_bundle, "System transactions should have been submitted individually");
            let transaction_keys = transaction_keys.clone();
            Some(CancelOnDrop(tokio::spawn(async move {
                let mut i = 0u64;
                loop {
                    i += 1;
                    const WARN_DELAY_S: u64 = 30;
                    tokio::time::sleep(Duration::from_secs(WARN_DELAY_S)).await;
                    let total_wait = i * WARN_DELAY_S;
                    warn!(
                        "Still waiting {} seconds for transactions {:?} to commit in consensus",
                        total_wait, transaction_keys
                    );
                }
            })))
        } else {
            None
        };

        if let Some(processed_waiter) = processed_waiter {
            debug!("Submitting {:?} to consensus", transaction_keys);

            // populate the position only when this authority submits the transaction
            // to consensus
            guard.position = Some(position);
            guard.positions_moved = Some(positions_moved);
            guard.preceding_disconnected = Some(preceding_disconnected);

            let _permit: SemaphorePermit = self
                .submit_semaphore
                .acquire()
                .await
                .expect("Consensus adapter does not close semaphore");

            // We enter this branch when in select above await_submit completed and processed_waiter is pending
            // This means it is time for us to submit transaction to consensus
            let submit_inner = async {
                const RETRY_DELAY_STEP: Duration = Duration::from_secs(1);

                loop {
                    // Submit the transaction to consensus and return the submit result with a status waiter
                    let (consensus_positions, status_waiter) = self
                        .submit_inner(
                            &transactions,
                            epoch_store,
                            &transaction_keys,
                            tx_type,
                            is_soft_bundle,
                        )
                        .await;

                    if let Some(tx_consensus_positions) = tx_consensus_positions.take() {
                        tracing::Span::current().record(
                            "consensus_positions",
                            tracing::field::debug(&consensus_positions),
                        );
                        // We send the first consensus position returned by consensus
                        // to the submitting client even if it is retried internally within
                        // consensus adapter due to an error or GC. They can handle retries
                        // as needed if the consensus position does not return the desired
                        // results (e.g. not sequenced due to garbage collection).
                        let _ = tx_consensus_positions.send(consensus_positions);
                    }

                    match status_waiter.await {
                        Ok(status @ BlockStatus::Sequenced(_)) => {
                            tracing::Span::current()
                                .record("status", tracing::field::debug(&status));

                            // Block has been sequenced. Nothing more to do, we do have guarantees that the transaction will appear in consensus output.
                            trace!(
                                "Transaction {transaction_keys:?} has been sequenced by consensus."
                            );
                            break;
                        }
                        Ok(status @ BlockStatus::GarbageCollected(_)) => {
                            tracing::Span::current()
                                .record("status", tracing::field::debug(&status));

                            // Block has been garbage collected and we have no guarantees that the transaction will appear in consensus output. We'll
                            // resubmit the transaction to consensus. If the transaction has been already "processed", then probably someone else has submitted
                            // the transaction and managed to get sequenced. Then this future will have been cancelled anyways so no need to check here on the processed output.
                            debug!(
                                "Transaction {transaction_keys:?} was garbage collected before being sequenced. Will be retried."
                            );
                            time::sleep(RETRY_DELAY_STEP).await;
                            continue;
                        }
                        Err(err) => {
                            warn!(
                                "Error while waiting for status from consensus for transactions {transaction_keys:?}, with error {:?}. Will be retried.",
                                err
                            );
                            time::sleep(RETRY_DELAY_STEP).await;
                            continue;
                        }
                    }
                }
            };

            guard.processed_method = if skip_processed_checks {
                // When getting consensus positions, we only care about submit_inner completing
                submit_inner.await;
                ProcessedMethod::Consensus
            } else {
                match select(processed_waiter, submit_inner.boxed()).await {
                    Either::Left((observed_via_consensus, _submit_inner)) => observed_via_consensus,
                    Either::Right(((), processed_waiter)) => {
                        debug!("Submitted {transaction_keys:?} to consensus");
                        processed_waiter.await
                    }
                }
            };
        }
        debug!("{transaction_keys:?} processed by consensus");

        let consensus_keys: Vec<_> = transactions
            .iter()
            .filter_map(|t| {
                if t.is_mfp_transaction() {
                    // UserTransaction is not inserted into the pending consensus transactions table.
                    // Also UserTransaction shares the same key as CertifiedTransaction, so removing
                    // the key here can have unexpected effects.
                    None
                } else {
                    Some(t.key())
                }
            })
            .collect();
        epoch_store
            .remove_pending_consensus_transactions(&consensus_keys)
            .expect("Storage error when removing consensus transaction");

        let is_user_tx = is_soft_bundle
            || matches!(transactions[0].kind, ConsensusTransactionKind::CertifiedTransaction(_))
            || matches!(transactions[0].kind, ConsensusTransactionKind::UserTransaction(_));
        if is_user_tx && epoch_store.should_send_end_of_publish() {
            // sending message outside of any locks scope
            if let Err(err) = self.submit(
                ConsensusTransaction::new_end_of_publish(self.authority),
                None,
                epoch_store,
                None,
                None,
            ) {
                warn!("Error when sending end of publish message: {:?}", err);
            } else {
                info!(epoch=?epoch_store.epoch(), "Sending EndOfPublish message to consensus");
            }
        }
    }

    #[instrument(name = "ConsensusAdapter::submit_inner", level = "trace", skip_all)]
    async fn submit_inner(
        self: &Arc<Self>,
        transactions: &[ConsensusTransaction],
        epoch_store: &Arc<AuthorityPerEpochStore>,
        transaction_keys: &[SequencedConsensusTransactionKey],
        tx_type: &str,
        is_soft_bundle: bool,
    ) -> (Vec<ConsensusPosition>, BlockStatusReceiver) {
        let ack_start = Instant::now();
        let mut retries: u32 = 0;

        let (consensus_positions, status_waiter) = loop {
            match self.consensus_client.submit(transactions, epoch_store).await {
                Err(err) => {
                    // This can happen during reconfig, or when consensus has full internal buffers
                    // and needs to back pressure, so retry a few times before logging warnings.
                    if retries > 30 || (retries > 3 && (is_soft_bundle)) {
                        warn!(
                            "Failed to submit transactions {transaction_keys:?} to consensus: {err:?}. Retry #{retries}"
                        );
                    }

                    retries += 1;

                    time::sleep(Duration::from_secs(10)).await;
                }
                Ok((consensus_positions, status_waiter)) => {
                    break (consensus_positions, status_waiter);
                }
            }
        };

        // we want to record the num of retries when reporting latency but to avoid label
        // cardinality we do some simple bucketing to give us a good enough idea of how
        // many retries happened associated with the latency.
        let bucket = match retries {
            0..=10 => retries.to_string(), // just report the retry count as is
            11..=20 => "between_10_and_20".to_string(),
            21..=50 => "between_20_and_50".to_string(),
            51..=100 => "between_50_and_100".to_string(),
            _ => "over_100".to_string(),
        };

        (consensus_positions, status_waiter)
    }

    /// Waits for transactions to appear either to consensus output or been executed via a checkpoint (state sync).
    /// Returns the processed method, whether the transactions have been processed via consensus, or have been synced via checkpoint.
    async fn await_consensus_or_checkpoint(
        self: &Arc<Self>,
        transaction_keys: Vec<SequencedConsensusTransactionKey>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> ProcessedMethod {
        let notifications = FuturesUnordered::new();
        for transaction_key in transaction_keys {
            let transaction_digests = match transaction_key {
                SequencedConsensusTransactionKey::External(
                    ConsensusTransactionKey::Certificate(digest),
                ) => vec![digest],
                _ => vec![],
            };

            let checkpoint_synced_future = if let SequencedConsensusTransactionKey::External(
                ConsensusTransactionKey::CheckpointSignature(_, checkpoint_sequence_number, _),
            ) = transaction_key
            {
                // If the transaction is a checkpoint signature, we can also wait to get notified when a checkpoint with equal or higher sequence
                // number has been already synced. This way we don't try to unnecessarily sequence the signature for an already verified checkpoint.
                Either::Left(
                    self.checkpoint_store.notify_read_synced_checkpoint(checkpoint_sequence_number),
                )
            } else {
                Either::Right(future::pending())
            };

            // We wait for each transaction individually to be processed by consensus or executed in a checkpoint. We could equally just
            // get notified in aggregate when all transactions are processed, but with this approach can get notified in a more fine-grained way
            // as transactions can be marked as processed in different ways. This is mostly a concern for the soft-bundle transactions.
            notifications.push(async move {
                tokio::select! {
                    processed = epoch_store.consensus_messages_processed_notify(vec![transaction_key]) => {
                        processed.expect("Storage error when waiting for consensus message processed");

                        return ProcessedMethod::Consensus;
                    },
                    processed = epoch_store.transactions_executed_in_checkpoint_notify(transaction_digests), if !transaction_digests.is_empty() => {
                        processed.expect("Storage error when waiting for transaction executed in checkpoint");

                    }
                    _ = checkpoint_synced_future => {

                    }
                }
                ProcessedMethod::Checkpoint
            });
        }

        let processed_methods = notifications.collect::<Vec<ProcessedMethod>>().await;
        for method in processed_methods {
            if method == ProcessedMethod::Checkpoint {
                return ProcessedMethod::Checkpoint;
            }
        }
        ProcessedMethod::Consensus
    }
}

pub fn get_position_in_list(
    search_authority: AuthorityName,
    positions: Vec<AuthorityName>,
) -> usize {
    positions
        .into_iter()
        .find_position(|authority| *authority == search_authority)
        .expect("Couldn't find ourselves in shuffled committee")
        .0
}

impl ReconfigurationInitiator for Arc<ConsensusAdapter> {
    /// This method is called externally to begin reconfiguration
    /// It sets reconfig state to reject new certificates from user.
    /// ConsensusAdapter will send EndOfPublish message once pending certificate queue is drained.
    fn close_epoch(&self, epoch_store: &Arc<AuthorityPerEpochStore>) {
        {
            let reconfig_guard = epoch_store.get_reconfig_state_write_lock_guard();
            if !reconfig_guard.should_accept_user_certs() {
                // Allow caller to call this method multiple times
                return;
            }
            epoch_store.close_user_certs(reconfig_guard);
        }
        if epoch_store.should_send_end_of_publish() {
            if let Err(err) = self.submit(
                ConsensusTransaction::new_end_of_publish(self.authority),
                None,
                epoch_store,
                None,
                None,
            ) {
                warn!("Error when sending end of publish message: {:?}", err);
            } else {
                info!(epoch=?epoch_store.epoch(), "Sending EndOfPublish message to consensus");
            }
        }
    }
}

struct CancelOnDrop<T>(JoinHandle<T>);

impl<T> Deref for CancelOnDrop<T> {
    type Target = JoinHandle<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Drop for CancelOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Tracks number of inflight consensus requests and relevant metrics
struct InflightDropGuard<'a> {
    adapter: &'a ConsensusAdapter,
    start: Instant,
    position: Option<usize>,
    positions_moved: Option<usize>,
    preceding_disconnected: Option<usize>,
    tx_type: &'static str,
    processed_method: ProcessedMethod,
}

#[derive(PartialEq, Eq)]
enum ProcessedMethod {
    Consensus,
    Checkpoint,
}

impl<'a> InflightDropGuard<'a> {
    pub fn acquire(adapter: &'a ConsensusAdapter, tx_type: &'static str) -> Self {
        adapter.num_inflight_transactions.fetch_add(1, Ordering::SeqCst);

        Self {
            adapter,
            start: Instant::now(),
            position: None,
            positions_moved: None,
            preceding_disconnected: None,

            tx_type,
            processed_method: ProcessedMethod::Consensus,
        }
    }
}

impl Drop for InflightDropGuard<'_> {
    fn drop(&mut self) {
        self.adapter.num_inflight_transactions.fetch_sub(1, Ordering::SeqCst);

        let position = if let Some(position) = self.position {
            position.to_string()
        } else {
            "not_submitted".to_string()
        };

        let latency = self.start.elapsed();
        let processed_method = match self.processed_method {
            ProcessedMethod::Consensus => "processed_via_consensus",
            ProcessedMethod::Checkpoint => "processed_via_checkpoint",
        };

        // Only sample latency after consensus quorum is up. Otherwise, the wait for consensus
        // quorum at the beginning of an epoch can distort the sampled latencies.
        // Technically there are more system transaction types that can be included in samples
        // after the first consensus commit, but this set of types should be enough.
        if self.position == Some(0) {
            // Transaction types below require quorum existed in the current epoch.
            // TODO: refactor tx_type to enum.
            let sampled = matches!(
                self.tx_type,
                "shared_certificate" | "owned_certificate" | "checkpoint_signature" | "soft_bundle"
            );
        }
    }
}

impl SubmitToConsensus for Arc<ConsensusAdapter> {
    fn submit_to_consensus(
        &self,
        transactions: &[ConsensusTransaction],
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult {
        self.submit_batch(transactions, None, epoch_store, None, None).map(|_| ())
    }

    fn submit_best_effort(
        &self,
        transaction: &ConsensusTransaction,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        // timeout is required, or the spawned task can run forever
        timeout: Duration,
    ) -> SomaResult {
        let permit = match self.submit_semaphore.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                return Err(SomaError::TooManyTransactionsPendingConsensus.into());
            }
        };

        let key = SequencedConsensusTransactionKey::External(transaction.key());
        let tx_type = classify(transaction);

        let async_stage = {
            let transaction = transaction.clone();
            let epoch_store = epoch_store.clone();
            let this = self.clone();

            async move {
                let _permit = permit; // Hold permit for lifetime of task

                let result = tokio::time::timeout(
                    timeout,
                    this.submit_inner(&[transaction], &epoch_store, &[key], tx_type, false),
                )
                .await;

                if let Err(e) = result {
                    warn!("Consensus submission timed out: {e:?}");
                }
            }
        };

        let epoch_store = epoch_store.clone();
        tokio::spawn(async move { epoch_store.within_alive_epoch(async_stage).await });
        Ok(())
    }
}

pub fn position_submit_certificate(
    committee: &Committee,
    ourselves: &AuthorityName,
    tx_digest: &TransactionDigest,
) -> usize {
    let validators = committee.shuffle_by_stake_from_tx_digest(tx_digest);
    get_position_in_list(*ourselves, validators)
}
