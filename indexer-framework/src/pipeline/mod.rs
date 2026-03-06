// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

pub use crate::config::ConcurrencyConfig;
use crate::store::CommitterWatermark;
pub use processor::Processor;
use rand::Rng;
use serde::Deserialize;
use serde::Serialize;

pub mod concurrent;
mod logging;
mod processor;
pub mod sequential;

/// Issue a warning every time the number of pending watermarks exceeds this number.
const WARN_PENDING_WATERMARKS: usize = 10000;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommitterConfig {
    /// Number of concurrent writers per pipeline.
    pub write_concurrency: usize,

    /// The collector will check for pending data at least this often, in milliseconds.
    pub collect_interval_ms: u64,

    /// Watermark task will check for pending watermarks this often, in milliseconds.
    pub watermark_interval_ms: u64,

    /// Maximum random jitter to add to the watermark interval, in milliseconds.
    pub watermark_interval_jitter_ms: u64,
}

/// Processed values associated with a single checkpoint. This is an internal type used to
/// communicate between the processor and the collector parts of the pipeline.
pub(crate) struct IndexedCheckpoint<P: Processor> {
    /// Values to be inserted into the database from this checkpoint
    pub values: Vec<P::Value>,
    /// The watermark associated with this checkpoint
    pub watermark: CommitterWatermark,
}

/// A representation of the proportion of a watermark.
#[derive(Debug, Clone)]
pub(crate) struct WatermarkPart {
    /// The watermark itself
    pub watermark: CommitterWatermark,
    /// The number of rows from this watermark that are in this part
    pub batch_rows: usize,
    /// The total number of rows from this watermark
    pub total_rows: usize,
}

impl CommitterConfig {
    pub fn collect_interval(&self) -> Duration {
        Duration::from_millis(self.collect_interval_ms)
    }

    pub fn watermark_interval(&self) -> Duration {
        Duration::from_millis(self.watermark_interval_ms)
    }

    /// Returns the next watermark update instant with a random jitter added.
    pub fn watermark_interval_with_jitter(&self) -> tokio::time::Instant {
        let jitter = if self.watermark_interval_jitter_ms == 0 {
            0
        } else {
            rand::thread_rng().gen_range(0..=self.watermark_interval_jitter_ms)
        };
        tokio::time::Instant::now() + Duration::from_millis(self.watermark_interval_ms + jitter)
    }
}

impl<P: Processor> IndexedCheckpoint<P> {
    pub(crate) fn new(
        epoch: u64,
        cp_sequence_number: u64,
        tx_hi: u64,
        timestamp_ms: u64,
        values: Vec<P::Value>,
    ) -> Self {
        Self {
            watermark: CommitterWatermark {
                epoch,
                checkpoint_hi_inclusive: cp_sequence_number,
                tx_hi,
                timestamp_ms_hi_inclusive: timestamp_ms,
            },
            values,
        }
    }

    /// Number of rows from this checkpoint
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    /// The checkpoint sequence number that this data is from
    pub(crate) fn checkpoint(&self) -> u64 {
        self.watermark.checkpoint_hi_inclusive
    }
}

impl WatermarkPart {
    pub(crate) fn checkpoint(&self) -> u64 {
        self.watermark.checkpoint_hi_inclusive
    }

    pub(crate) fn timestamp_ms(&self) -> u64 {
        self.watermark.timestamp_ms_hi_inclusive
    }

    /// Check if all the rows from this watermark are represented in this part.
    pub(crate) fn is_complete(&self) -> bool {
        self.batch_rows == self.total_rows
    }

    /// Add the rows from `other` to this part.
    pub(crate) fn add(&mut self, other: WatermarkPart) {
        debug_assert_eq!(self.checkpoint(), other.checkpoint());
        self.batch_rows += other.batch_rows;
    }

    /// Record that `rows` have been taken from this part.
    pub(crate) fn take(&mut self, rows: usize) -> WatermarkPart {
        debug_assert!(
            self.batch_rows >= rows,
            "Can't take more rows than are available"
        );

        self.batch_rows -= rows;
        WatermarkPart {
            watermark: self.watermark,
            batch_rows: rows,
            total_rows: self.total_rows,
        }
    }
}

impl Default for CommitterConfig {
    fn default() -> Self {
        Self {
            write_concurrency: 5,
            collect_interval_ms: 500,
            watermark_interval_ms: 500,
            watermark_interval_jitter_ms: 0,
        }
    }
}
