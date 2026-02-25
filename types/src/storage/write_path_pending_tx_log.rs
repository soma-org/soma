// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! WritePathPendingTransactionLog is used in TransactionOrchestrator
//! to deduplicate transaction submission processing. It helps to achieve:
//! 1. At one time, a transaction is only processed once.
//! 2. When Fullnode crashes and restarts, the pending transaction will be loaded and retried.

use std::collections::HashSet;
use std::path::PathBuf;

use crate::crypto::EmptySignInfo;
use crate::digests::TransactionDigest;
use crate::envelope::TrustedEnvelope;
use crate::error::{SomaError, SomaResult};
use crate::transaction::{SenderSignedData, VerifiedTransaction};
use parking_lot::Mutex;
use store::DBMapUtils;
use store::{rocks::DBMap, traits::Map};

#[derive(DBMapUtils)]
struct WritePathPendingTransactionTable {
    logs: DBMap<TransactionDigest, TrustedEnvelope<SenderSignedData, EmptySignInfo>>,
}

pub struct WritePathPendingTransactionLog {
    // Disk storage for pending transactions.
    pending_transactions: WritePathPendingTransactionTable,
    // In-memory set of pending transactions.
    transactions_set: Mutex<HashSet<TransactionDigest>>,
}

impl WritePathPendingTransactionLog {
    pub fn new(path: PathBuf) -> Self {
        let pending_transactions =
            WritePathPendingTransactionTable::open_tables_read_write(path, None, None);
        Self { pending_transactions, transactions_set: Mutex::new(HashSet::new()) }
    }

    // Returns whether the table currently has this transaction in record.
    // If not, write the transaction and return true; otherwise return false.
    pub fn write_pending_transaction_maybe(&self, tx: &VerifiedTransaction) -> bool {
        let tx_digest = tx.digest();
        let mut transactions_set = self.transactions_set.lock();
        if transactions_set.contains(tx_digest) {
            return false;
        }
        // Hold the lock while inserting into the logs to avoid race conditions.
        self.pending_transactions.logs.insert(tx_digest, tx.serializable_ref()).unwrap();
        transactions_set.insert(*tx_digest);
        true
    }

    pub fn finish_transaction(&self, tx: &TransactionDigest) -> SomaResult {
        let mut transactions_set = self.transactions_set.lock();
        // Hold the lock while removing from the logs to avoid race conditions.
        let mut write_batch = self.pending_transactions.logs.batch();
        write_batch.delete_batch(&self.pending_transactions.logs, std::iter::once(tx))?;
        write_batch.write().map_err(SomaError::from)?;
        transactions_set.remove(tx);
        Ok(())
    }

    pub fn load_all_pending_transactions(&self) -> SomaResult<Vec<VerifiedTransaction>> {
        let mut transactions_set = self.transactions_set.lock();
        let transactions = self
            .pending_transactions
            .logs
            .safe_iter()
            .map(|item| item.map(|(_tx_digest, tx)| VerifiedTransaction::from(tx)))
            .collect::<Result<Vec<_>, _>>()?;
        transactions_set.extend(transactions.iter().map(|t| *t.digest()));
        Ok(transactions)
    }
}
