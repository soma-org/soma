// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::ALL_PIPELINE_NAMES;

/// Tracks per-pipeline watermarks and computes a min across all pipelines.
/// Used to dual-write the legacy `[0]` row in `watermark_alt`.
pub(crate) struct LegacyWatermarkTracker {
    watermarks: HashMap<String, u64>,
    last_written: Option<u64>,
}

impl LegacyWatermarkTracker {
    pub fn new() -> Self {
        Self {
            watermarks: HashMap::with_capacity(ALL_PIPELINE_NAMES.len()),
            last_written: None,
        }
    }

    /// Record a pipeline's latest checkpoint_hi_inclusive.
    /// Returns `Some((min, prev_last_written))` when all pipelines have reported
    /// AND the min has advanced past `last_written`.
    pub fn update(&mut self, pipeline: &str, checkpoint_hi: u64) -> Option<(u64, Option<u64>)> {
        self.watermarks.insert(pipeline.to_owned(), checkpoint_hi);

        if self.watermarks.len() < ALL_PIPELINE_NAMES.len() {
            return None;
        }

        let min = *self.watermarks.values().min()?;

        if self.last_written.is_some_and(|prev| min <= prev) {
            return None;
        }

        let prev = self.last_written;
        self.last_written = Some(min);
        Some((min, prev))
    }

    /// Roll back after a failed write.
    pub fn rollback(&mut self, failed_min: u64, prev: Option<u64>) {
        if self.last_written == Some(failed_min) {
            self.last_written = prev;
        }
    }
}
