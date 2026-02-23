// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use lru::LruCache;
use parking_lot::RwLock;
use std::collections::BTreeSet;
use std::net::IpAddr;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tracing::debug;
use types::digests::TransactionDigest;
use types::traffic_control::Weight;

pub(crate) const DEFAULT_CACHE_CAPACITY: usize = 100_000;

/// Cache for tracking submitted transactions to prevent DoS through excessive resubmissions.
/// Uses LRU eviction to automatically remove least recently used entries when at capacity.
/// Tracks submission counts and enforces gas-price-based amplification limits.
pub(crate) struct SubmittedTransactionCache {
    inner: RwLock<Inner>,
}

struct Inner {
    transactions: LruCache<TransactionDigest, SubmissionMetadata>,
}

#[derive(Debug, Clone)]
struct SubmissionMetadata {
    /// Number of times this transaction has been submitted
    submission_count: u32,
    /// Maximum allowed submissions
    max_allowed_submissions: u32,
    /// Set of client IP addresses that have submitted this transaction
    submitter_client_addrs: BTreeSet<IpAddr>,
}

impl SubmittedTransactionCache {
    pub(crate) fn new(cache_capacity: Option<usize>) -> Self {
        let capacity = cache_capacity
            .and_then(NonZeroUsize::new)
            .unwrap_or_else(|| NonZeroUsize::new(DEFAULT_CACHE_CAPACITY).unwrap());

        Self { inner: RwLock::new(Inner { transactions: LruCache::new(capacity) }) }
    }

    pub(crate) fn record_submitted_tx(
        &self,
        digest: &TransactionDigest,
        submitter_client_addr: Option<IpAddr>,
    ) {
        let mut inner = self.inner.write();

        let max_allowed_submissions = 1;

        if let Some(metadata) = inner.transactions.get_mut(digest) {
            // Track additional client addresses for resubmissions
            if let Some(addr) = submitter_client_addr {
                if metadata.submitter_client_addrs.insert(addr) {
                    debug!("Added new client address {addr} for transaction {digest}");
                }
            }
            debug!("Transaction {digest} already tracked in submission cache");
        } else {
            // First time we're submitting this transaction, however we will wait till
            // we see the transaction in consensus output to increment the submission count.
            let submitter_client_addrs = submitter_client_addr.into_iter().collect();
            let metadata = SubmissionMetadata {
                submission_count: 0,
                max_allowed_submissions,
                submitter_client_addrs,
            };

            inner.transactions.put(*digest, metadata);

            debug!(
                "First submission of transaction {digest} (max_allowed: {max_allowed_submissions})",
            );
        }
    }

    /// Increments the submission count when we see a transaction in consensus output.
    /// This tracks how many times the transaction has appeared in consensus (from any validator).
    /// Returns the spam weight and set of submitter client addresses if the transaction exceeds allowed submissions.
    pub(crate) fn increment_submission_count(
        &self,
        digest: &TransactionDigest,
    ) -> Option<(Weight, BTreeSet<IpAddr>)> {
        let mut inner = self.inner.write();

        if let Some(metadata) = inner.transactions.get_mut(digest) {
            metadata.submission_count += 1;

            if metadata.submission_count > metadata.max_allowed_submissions {
                let spam_weight = Weight::one();

                debug!(
                    "Transaction {} seen in consensus {} times, exceeds limit {} (spam_weight: {:?})",
                    digest,
                    metadata.submission_count,
                    metadata.max_allowed_submissions,
                    spam_weight
                );

                return Some((spam_weight, metadata.submitter_client_addrs.clone()));
            }
        }
        // If we don't know about this transaction, it was submitted by another validator
        // We don't track spam weight for transactions we didn't submit
        None
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, digest: &TransactionDigest) -> bool {
        self.inner.read().transactions.contains(digest)
    }

    #[cfg(test)]
    pub(crate) fn get_submission_count(&self, digest: &TransactionDigest) -> Option<u32> {
        self.inner.read().transactions.peek(digest).map(|m| m.submission_count)
    }
}
