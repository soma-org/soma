// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! BigTable Store implementation for the indexer framework.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use indexer_store_traits::CommitterWatermark;
use indexer_store_traits::Connection;
use indexer_store_traits::PrunerWatermark;
use indexer_store_traits::ReaderWatermark;
use indexer_store_traits::Store;

use crate::Watermark;
use crate::bigtable::client::BigTableClient;
use crate::bigtable::legacy_watermark::LegacyWatermarkTracker;
use crate::tables;
use crate::write_legacy_data;

/// A Store implementation backed by BigTable.
#[derive(Clone)]
pub struct BigTableStore {
    client: BigTableClient,
    legacy_watermark_tracker: Option<Arc<Mutex<LegacyWatermarkTracker>>>,
}

/// A connection to BigTable for watermark operations and data writes.
pub struct BigTableConnection<'a> {
    client: BigTableClient,
    legacy_watermark_tracker: Option<Arc<Mutex<LegacyWatermarkTracker>>>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl BigTableStore {
    pub fn new(client: BigTableClient) -> Self {
        let legacy_watermark_tracker = if write_legacy_data() {
            Some(Arc::new(Mutex::new(LegacyWatermarkTracker::new())))
        } else {
            None
        };
        Self {
            client,
            legacy_watermark_tracker,
        }
    }
}

impl BigTableConnection<'_> {
    /// Returns a mutable reference to the underlying BigTable client.
    pub fn client(&mut self) -> &mut BigTableClient {
        &mut self.client
    }
}

#[async_trait]
impl Store for BigTableStore {
    type Connection<'c> = BigTableConnection<'c>;

    async fn connect<'c>(&'c self) -> Result<Self::Connection<'c>> {
        Ok(BigTableConnection {
            client: self.client.clone(),
            legacy_watermark_tracker: self.legacy_watermark_tracker.clone(),
            _marker: std::marker::PhantomData,
        })
    }
}

#[async_trait]
impl Connection for BigTableConnection<'_> {
    async fn init_watermark(
        &mut self,
        pipeline_task: &str,
        _default_next_checkpoint: u64,
    ) -> Result<Option<u64>> {
        Ok(self
            .client
            .get_pipeline_watermark(pipeline_task)
            .await?
            .map(|wm| wm.checkpoint_hi_inclusive))
    }

    async fn committer_watermark(
        &mut self,
        pipeline_task: &str,
    ) -> Result<Option<CommitterWatermark>> {
        Ok(self
            .client
            .get_pipeline_watermark(pipeline_task)
            .await?
            .map(Into::into))
    }

    async fn set_committer_watermark(
        &mut self,
        pipeline_task: &str,
        watermark: CommitterWatermark,
    ) -> Result<bool> {
        let pipeline_watermark: Watermark = watermark.into();
        self.client
            .set_pipeline_watermark(pipeline_task, &pipeline_watermark)
            .await?;

        // Legacy dual-write support
        if let Some(ref tracker) = self.legacy_watermark_tracker {
            let pipeline_name = pipeline_task
                .split_once('@')
                .map_or(pipeline_task, |(name, _)| name);

            let maybe_update = {
                let mut guard = tracker.lock().expect("legacy tracker lock poisoned");
                guard.update(pipeline_name, pipeline_watermark.checkpoint_hi_inclusive)
            };

            if let Some((min, prev)) = maybe_update {
                let next_checkpoint = min + 1;
                let entry = tables::make_entry(
                    vec![0u8],
                    [(
                        tables::DEFAULT_COLUMN,
                        Bytes::from(next_checkpoint.to_be_bytes().to_vec()),
                    )],
                    Some(next_checkpoint),
                );
                if let Err(e) = self
                    .client
                    .write_entries(tables::watermark_alt_legacy::NAME, [entry])
                    .await
                {
                    tracker
                        .lock()
                        .expect("legacy tracker lock poisoned")
                        .rollback(min, prev);
                    return Err(e);
                }
            }
        }

        Ok(true)
    }

    async fn reader_watermark(
        &mut self,
        _pipeline: &'static str,
    ) -> Result<Option<ReaderWatermark>> {
        Ok(None)
    }

    async fn pruner_watermark(
        &mut self,
        _pipeline: &'static str,
        _delay: Duration,
    ) -> Result<Option<PrunerWatermark>> {
        Ok(None)
    }

    async fn set_reader_watermark(
        &mut self,
        _pipeline: &'static str,
        _reader_lo: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    async fn set_pruner_watermark(
        &mut self,
        _pipeline: &'static str,
        _pruner_hi: u64,
    ) -> Result<bool> {
        Ok(false)
    }
}
