// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::RwLock;

use bytes::Bytes;
use indexer_framework::pipeline::Processor;
use indexer_framework::pipeline::concurrent::BatchStatus;
use indexer_framework::pipeline::concurrent::Handler;
use indexer_store_traits::Store;
use types::full_checkpoint_content::Checkpoint;

use crate::bigtable::client::PartialWriteError;
use crate::bigtable::proto::bigtable::v2::mutate_rows_request::Entry;
use crate::bigtable::store::BigTableStore;
use crate::config::ConcurrentLayer;
use crate::rate_limiter::CompositeRateLimiter;

/// BigTable's hard limit is 100k mutations per MutateRows request.
const MAX_MUTATIONS_PER_BATCH: usize = 50_000;

pub const DEFAULT_MAX_ROWS: usize = 100;

/// Extension of `Processor` that specifies a BigTable table name.
pub trait BigTableProcessor: Processor<Value = Entry> {
    const TABLE: &'static str;
    const MIN_EAGER_ROWS: usize = 50;
    const MAX_PENDING_ROWS: usize = 1000;
}

/// Generic wrapper that implements `concurrent::Handler` for any `BigTableProcessor`.
pub struct BigTableHandler<P> {
    processor: P,
    max_rows: usize,
    rate_limiter: Arc<CompositeRateLimiter>,
}

/// Batch of BigTable entries.
#[derive(Default)]
pub struct BigTableBatch {
    inner: RwLock<BigTableBatchInner>,
}

#[derive(Default)]
struct BigTableBatchInner {
    entries: BTreeMap<Bytes, Entry>,
    total_mutations: usize,
}

impl<P> BigTableHandler<P>
where
    P: BigTableProcessor,
{
    pub(crate) fn new(
        processor: P,
        config: &ConcurrentLayer,
        rate_limiter: Arc<CompositeRateLimiter>,
    ) -> Self {
        Self {
            processor,
            max_rows: config.max_rows.unwrap_or(DEFAULT_MAX_ROWS),
            rate_limiter,
        }
    }
}

#[async_trait::async_trait]
impl<P> Processor for BigTableHandler<P>
where
    P: BigTableProcessor + Send + Sync,
{
    const NAME: &'static str = P::NAME;
    type Value = Entry;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
        self.processor.process(checkpoint).await
    }
}

#[async_trait::async_trait]
impl<P> Handler for BigTableHandler<P>
where
    P: BigTableProcessor + Send + Sync,
{
    type Store = BigTableStore;
    type Batch = BigTableBatch;

    const MIN_EAGER_ROWS: usize = P::MIN_EAGER_ROWS;
    const MAX_PENDING_ROWS: usize = P::MAX_PENDING_ROWS;

    fn batch(
        &self,
        batch: &mut Self::Batch,
        values: &mut std::vec::IntoIter<Self::Value>,
    ) -> BatchStatus {
        let mut inner = batch.inner.write().unwrap();

        for entry in values {
            inner.total_mutations += entry.mutations.len();
            inner.entries.insert(entry.row_key.clone(), entry);

            if inner.entries.len() >= self.max_rows
                || inner.total_mutations >= MAX_MUTATIONS_PER_BATCH
            {
                return BatchStatus::Ready;
            }
        }

        BatchStatus::Pending
    }

    async fn commit<'a>(
        &self,
        batch: &Self::Batch,
        conn: &mut <Self::Store as Store>::Connection<'a>,
    ) -> anyhow::Result<usize> {
        let entries_to_write: Vec<Entry> = {
            let inner = batch.inner.read().unwrap();
            if inner.entries.is_empty() {
                return Ok(0);
            }
            inner.entries.values().cloned().collect()
        };
        let count = entries_to_write.len();

        self.rate_limiter.acquire(count).await;

        match conn
            .client()
            .write_entries(P::TABLE, entries_to_write)
            .await
        {
            Ok(()) => Ok(count),
            Err(e) => {
                if let Some(partial) = e.downcast_ref::<PartialWriteError>() {
                    let mut inner = batch.inner.write().unwrap();
                    let failed: std::collections::BTreeSet<&Bytes> =
                        partial.failed_keys.iter().map(|f| &f.key).collect();
                    inner.entries.retain(|key, _| failed.contains(key));
                }
                Err(e)
            }
        }
    }
}
