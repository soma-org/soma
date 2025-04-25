use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StateSyncConfig {
    /// Query peers for their latest commit every interval period.
    ///
    /// If unspecified, this will default to `5,000` milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval_period_ms: Option<u64>,

    /// Size of the StateSync actor's mailbox.
    ///
    /// If unspecified, this will default to `1,024`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mailbox_capacity: Option<usize>,

    /// Size of the broadcast channel use for notifying other systems of newly sync'ed commits.
    ///
    /// If unspecified, this will default to `1,024`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synced_commit_broadcast_channel_capacity: Option<u64>,

    /// Set the upper bound on the number of commit contents to be downloaded concurrently.
    ///
    /// If unspecified, this will default to `400`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_content_download_concurrency: Option<u64>,

    /// Set the timeout that should be used when sending most state-sync RPC requests.
    ///
    /// If unspecified, this will default to `10,000` milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,

    /// Set the timeout that should be used when sending RPC requests to sync commit contents.
    ///
    /// If unspecified, this will default to `10,000` milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit_content_timeout_ms: Option<u64>,

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

    pub fn synced_commit_broadcast_channel_capacity(&self) -> u64 {
        const SYNCED_CHECKPOINT_BROADCAST_CHANNEL_CAPACITY: u64 = 1_024;

        self.synced_commit_broadcast_channel_capacity
            .unwrap_or(SYNCED_CHECKPOINT_BROADCAST_CHANNEL_CAPACITY)
    }

    pub fn commit_content_download_concurrency(&self) -> u64 {
        const CHECKPOINT_CONTENT_DOWNLOAD_CONCURRENCY: u64 = 400;

        self.commit_content_download_concurrency
            .unwrap_or(CHECKPOINT_CONTENT_DOWNLOAD_CONCURRENCY)
    }

    pub fn timeout(&self) -> Duration {
        const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

        self.timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_TIMEOUT)
    }

    pub fn commit_content_timeout(&self) -> Duration {
        const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

        self.commit_content_timeout_ms
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
