// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use chrono::DateTime;
use chrono::Utc;
use scoped_futures::ScopedBoxFuture;

/// Represents a database connection that can be used by the indexer framework to manage watermark
/// operations, agnostic of the underlying store implementation.
#[async_trait]
pub trait Connection: Send {
    /// If no existing watermark record exists, initializes it with `default_next_checkpoint`.
    /// Returns the committer watermark `checkpoint_hi_inclusive`.
    async fn init_watermark(
        &mut self,
        pipeline_task: &str,
        default_next_checkpoint: u64,
    ) -> anyhow::Result<Option<u64>>;

    /// Given a `pipeline_task`, return the committer watermark from the Store.
    async fn committer_watermark(
        &mut self,
        pipeline_task: &str,
    ) -> anyhow::Result<Option<CommitterWatermark>>;

    /// Get the reader watermark for a pipeline.
    async fn reader_watermark(
        &mut self,
        pipeline: &'static str,
    ) -> anyhow::Result<Option<ReaderWatermark>>;

    /// Get the pruner watermark for a pipeline.
    async fn pruner_watermark(
        &mut self,
        pipeline: &'static str,
        delay: Duration,
    ) -> anyhow::Result<Option<PrunerWatermark>>;

    /// Atomically set the committer watermark. Returns true if the watermark was updated.
    async fn set_committer_watermark(
        &mut self,
        pipeline_task: &str,
        watermark: CommitterWatermark,
    ) -> anyhow::Result<bool>;

    /// Set the reader low watermark for a pipeline.
    async fn set_reader_watermark(
        &mut self,
        pipeline: &'static str,
        reader_lo: u64,
    ) -> anyhow::Result<bool>;

    /// Set the pruner high watermark for a pipeline.
    async fn set_pruner_watermark(
        &mut self,
        pipeline: &'static str,
        pruner_hi: u64,
    ) -> anyhow::Result<bool>;
}

/// Provides connections to the underlying store.
#[async_trait]
pub trait Store: Send + Sync + 'static + Clone {
    type Connection<'c>: Connection
    where
        Self: 'c;

    /// The delimiter used to separate pipeline names from task names.
    const DELIMITER: &'static str = "@";

    async fn connect<'c>(&'c self) -> Result<Self::Connection<'c>>;
}

/// A store that supports transactional operations.
#[async_trait]
pub trait TransactionalStore: Store {
    async fn transaction<'a, T, F>(&self, f: F) -> anyhow::Result<T>
    where
        T: Send + 'a,
        F: for<'r> FnOnce(
                &'r mut Self::Connection<'_>,
            ) -> ScopedBoxFuture<'a, 'r, anyhow::Result<T>>
            + Send
            + 'a;
}

/// Watermark tracking for the committer phase.
#[derive(Debug, Clone, Copy)]
pub struct CommitterWatermark {
    pub epoch: u64,
    pub checkpoint_hi_inclusive: u64,
    pub tx_hi: u64,
    pub timestamp_ms_hi_inclusive: u64,
}

/// Watermark tracking for the reader phase.
#[derive(Debug, Clone, Copy)]
pub struct ReaderWatermark {
    pub checkpoint_hi_inclusive: u64,
    pub reader_lo: u64,
}

/// Watermark tracking for the pruner phase.
#[derive(Debug, Clone)]
pub struct PrunerWatermark {
    pub wait_for: Duration,
    pub reader_lo: u64,
    pub pruner_hi: u64,
}

impl CommitterWatermark {
    pub fn new(
        epoch: u64,
        checkpoint_hi_inclusive: u64,
        tx_hi: u64,
        timestamp_ms_hi_inclusive: u64,
    ) -> Self {
        Self { epoch, checkpoint_hi_inclusive, tx_hi, timestamp_ms_hi_inclusive }
    }

    pub fn timestamp(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp_ms_hi_inclusive as i64).unwrap_or_default()
    }
}

impl PrunerWatermark {
    /// Returns the range of checkpoints that can be pruned [pruner_hi, reader_lo).
    pub fn prunable_range(&self) -> Option<std::ops::Range<u64>> {
        if self.pruner_hi < self.reader_lo { Some(self.pruner_hi..self.reader_lo) } else { None }
    }
}

/// Construct a pipeline-task key from a pipeline name and optional task name.
pub fn pipeline_task(pipeline: &str, task: Option<&str>, delimiter: &str) -> String {
    match task {
        Some(task) => format!("{pipeline}{delimiter}{task}"),
        None => pipeline.to_string(),
    }
}
