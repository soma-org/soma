// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

/// Reports progress for a single download operation.
pub trait DownloadProgress: Send + Sync {
    /// Called periodically with the total number of bytes downloaded so far.
    fn update(&self, downloaded_bytes: u64);
    /// Called once when the download completes.
    fn finish(&self);
}

/// Factory for creating per-download progress reporters.
///
/// Used by the runtime to create a separate progress reporter for each
/// concurrent download (data, model weights, etc.).
pub trait ProgressFactory: Send + Sync {
    fn create(&self, label: &str, total_bytes: u64) -> Arc<dyn DownloadProgress>;
}
