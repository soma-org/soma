// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Progress bar helpers for CLI download commands and the scoring server.
//!
//! Provides two things:
//! 1. Simple callback-based progress for `ProxyClient` streaming downloads (CLI commands).
//! 2. An `IndicatifProgressFactory` implementing `blobs::progress::ProgressFactory`
//!    for the scoring server's parallel blob downloads.

use std::io::IsTerminal;
use std::sync::Arc;
use std::time::Instant;

use blobs::progress::{DownloadProgress, ProgressFactory};
use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressStyle};

// =============================================================================
// Part 1: CLI download commands (ProxyClient callback)
// =============================================================================

const BAR_TEMPLATE: &str =
    "{spinner:.green} {prefix:>8} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}";
const PROGRESS_CHARS: &str = "\u{2588}\u{2589}\u{258a}\u{258b}\u{258c}\u{258d}\u{258e}\u{258f} ";

fn bar_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(BAR_TEMPLATE)
        .expect("valid template")
        .progress_chars(PROGRESS_CHARS)
}

/// Create a TTY-aware progress bar for a single download.
pub fn create_progress_bar() -> ProgressBar {
    if !std::io::stderr().is_terminal() {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(0);
    pb.set_style(bar_style());
    pb
}

/// Build a callback suitable for `ProxyClient::fetch_*_with_progress()`.
///
/// Clones the given `ProgressBar` (cheap, Arc-based) and returns a closure
/// that updates position, length, and speed on every chunk.
pub fn make_progress_callback(pb: &ProgressBar) -> impl FnMut(u64, Option<u64>) + Send {
    let pb = pb.clone();
    let start = Instant::now();
    move |downloaded: u64, total: Option<u64>| {
        if let Some(total) = total {
            pb.set_length(total);
        }
        pb.set_position(downloaded);
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let speed = downloaded as f64 / elapsed;
            pb.set_message(format!("{}/s", HumanBytes(speed as u64)));
        }
    }
}

// =============================================================================
// Part 2: Scoring server (ProgressFactory for BlobEngine)
// =============================================================================

/// Create a `ProgressFactory` for the scoring server, or `None` when not on a TTY.
pub fn scoring_progress_factory() -> Option<Arc<dyn ProgressFactory>> {
    if !std::io::stderr().is_terminal() {
        return None;
    }
    Some(Arc::new(IndicatifProgressFactory { multi: MultiProgress::new() }))
}

struct IndicatifProgressFactory {
    multi: MultiProgress,
}

impl ProgressFactory for IndicatifProgressFactory {
    fn create(&self, label: &str, total_bytes: u64) -> Arc<dyn DownloadProgress> {
        let pb = self.multi.add(ProgressBar::new(total_bytes));
        pb.set_style(bar_style());
        pb.set_prefix(label.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        Arc::new(IndicatifProgress { pb })
    }
}

struct IndicatifProgress {
    pb: ProgressBar,
}

impl DownloadProgress for IndicatifProgress {
    fn update(&self, downloaded_bytes: u64) {
        self.pb.set_position(downloaded_bytes);
    }

    fn finish(&self) {
        self.pb.finish();
    }
}
