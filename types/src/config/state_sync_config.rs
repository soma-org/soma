use crate::checkpoints::CheckpointSequenceNumber;
use crate::digests::CheckpointDigest;
use serde::{Deserialize, Serialize};
use std::{num::NonZeroU32, time::Duration};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StateSyncConfig {
    /// List of "known-good" checkpoints that state sync will be forced to use. State sync will
    /// skip verification of pinned checkpoints, and reject checkpoints with digests that don't
    /// match pinned values for a given sequence number.
    ///
    /// This can be used:
    /// - in case of a fork, to prevent the node from syncing to the wrong chain.
    /// - in case of a network stall, to force the node to proceed with a manually-injected
    ///   checkpoint.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub pinned_checkpoints: Vec<(CheckpointSequenceNumber, CheckpointDigest)>,

    /// Query peers for their latest checkpoint every interval period.
    ///
    /// If unspecified, this will default to `5,000` milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval_period_ms: Option<u64>,

    /// Size of the StateSync actor's mailbox.
    ///
    /// If unspecified, this will default to `1,024`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mailbox_capacity: Option<usize>,

    /// Size of the broadcast channel use for notifying other systems of newly sync'ed checkpoints.
    ///
    /// If unspecified, this will default to `1,024`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synced_checkpoint_broadcast_channel_capacity: Option<usize>,

    /// Set the upper bound on the number of checkpoint headers to be downloaded concurrently.
    ///
    /// If unspecified, this will default to `400`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_header_download_concurrency: Option<usize>,

    /// Set the upper bound on the number of checkpoint contents to be downloaded concurrently.
    ///
    /// If unspecified, this will default to `400`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_content_download_concurrency: Option<usize>,

    /// Set the upper bound on the number of individual transactions contained in checkpoint
    /// contents to be downloaded concurrently. If both this value and
    /// `checkpoint_content_download_concurrency` are set, the lower of the two will apply.
    ///
    /// If unspecified, this will default to `50,000`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_content_download_tx_concurrency: Option<u64>,

    /// Set the timeout that should be used when sending most state-sync RPC requests.
    ///
    /// If unspecified, this will default to `10,000` milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,

    /// Set the timeout that should be used when sending RPC requests to sync checkpoint contents.
    ///
    /// If unspecified, this will default to `10,000` milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_content_timeout_ms: Option<u64>,

    /// Per-peer rate-limit (in requests/sec) for the PushCheckpointSummary RPC.
    ///
    /// If unspecified, this will default to no limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub push_checkpoint_summary_rate_limit: Option<NonZeroU32>,

    /// Per-peer rate-limit (in requests/sec) for the GetCheckpointSummary RPC.
    ///
    /// If unspecified, this will default to no limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub get_checkpoint_summary_rate_limit: Option<NonZeroU32>,

    /// Per-peer rate-limit (in requests/sec) for the GetCheckpointContents RPC.
    ///
    /// If unspecified, this will default to no limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub get_checkpoint_contents_rate_limit: Option<NonZeroU32>,

    /// Per-peer inflight limit for the GetCheckpointContents RPC.
    ///
    /// If unspecified, this will default to no limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub get_checkpoint_contents_inflight_limit: Option<usize>,

    /// Per-checkpoint inflight limit for the GetCheckpointContents RPC. This is enforced globally
    /// across all peers.
    ///
    /// If unspecified, this will default to no limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub get_checkpoint_contents_per_checkpoint_limit: Option<usize>,

    /// The amount of time to wait before retry if there are no peers to sync content from.
    /// If unspecified, this will set to default value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wait_interval_when_no_peer_to_sync_content_ms: Option<u64>,
}

impl StateSyncConfig {
    pub fn interval_period(&self) -> Duration {
        const INTERVAL_PERIOD_MS: u64 = 5_000; // 5 seconds

        Duration::from_millis(self.interval_period_ms.unwrap_or(INTERVAL_PERIOD_MS))
    }

    pub fn mailbox_capacity(&self) -> usize {
        const MAILBOX_CAPACITY: usize = 1_024;

        self.mailbox_capacity.unwrap_or(MAILBOX_CAPACITY)
    }

    pub fn synced_checkpoint_broadcast_channel_capacity(&self) -> usize {
        const SYNCED_CHECKPOINT_BROADCAST_CHANNEL_CAPACITY: usize = 1_024;

        self.synced_checkpoint_broadcast_channel_capacity
            .unwrap_or(SYNCED_CHECKPOINT_BROADCAST_CHANNEL_CAPACITY)
    }

    pub fn checkpoint_header_download_concurrency(&self) -> usize {
        const CHECKPOINT_HEADER_DOWNLOAD_CONCURRENCY: usize = 400;

        self.checkpoint_header_download_concurrency
            .unwrap_or(CHECKPOINT_HEADER_DOWNLOAD_CONCURRENCY)
    }

    pub fn checkpoint_content_download_concurrency(&self) -> usize {
        const CHECKPOINT_CONTENT_DOWNLOAD_CONCURRENCY: usize = 400;

        self.checkpoint_content_download_concurrency
            .unwrap_or(CHECKPOINT_CONTENT_DOWNLOAD_CONCURRENCY)
    }

    pub fn checkpoint_content_download_tx_concurrency(&self) -> u64 {
        const CHECKPOINT_CONTENT_DOWNLOAD_TX_CONCURRENCY: u64 = 50_000;

        self.checkpoint_content_download_tx_concurrency
            .unwrap_or(CHECKPOINT_CONTENT_DOWNLOAD_TX_CONCURRENCY)
    }

    pub fn timeout(&self) -> Duration {
        const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

        self.timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_TIMEOUT)
    }

    pub fn checkpoint_content_timeout(&self) -> Duration {
        const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

        self.checkpoint_content_timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_TIMEOUT)
    }

    pub fn wait_interval_when_no_peer_to_sync_content(&self) -> Duration {
        self.wait_interval_when_no_peer_to_sync_content_ms
            .map(Duration::from_millis)
            .unwrap_or(self.default_wait_interval_when_no_peer_to_sync_content())
    }

    fn default_wait_interval_when_no_peer_to_sync_content(&self) -> Duration {
        if cfg!(msim) {
            Duration::from_secs(5)
        } else {
            Duration::from_secs(10)
        }
    }
}
