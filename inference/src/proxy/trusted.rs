// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! somacode default-trust set.
//!
//! Periodically polls the points service's
//! `GET <base>/v1/providers/authorized` endpoint and caches the
//! returned address set. The router consults the cache as a filter
//! step before scoring candidates — when the cache is empty or the
//! fetch ever fails after at least one success, the filter is
//! reported as "unknown" and the router defaults to **open** (no
//! filter) to avoid silently routing nothing during a points-service
//! outage.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use types::base::SomaAddress;

#[derive(Debug, Clone)]
struct State {
    /// `Some(set)` after at least one successful fetch. `None` when
    /// no fetch has yet succeeded — callers treat as "open / no
    /// filter" so the proxy is usable from cold-start before the
    /// first poll lands.
    set: Option<HashSet<SomaAddress>>,
}

#[derive(Debug, Clone)]
pub struct TrustedProviders {
    state: Arc<RwLock<State>>,
    base_url: String,
    refresh: Duration,
    http: reqwest::Client,
}

impl TrustedProviders {
    pub fn new(base_url: String, refresh_secs: u64) -> Self {
        Self {
            state: Arc::new(RwLock::new(State { set: None })),
            base_url,
            refresh: Duration::from_secs(refresh_secs.max(60)),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("reqwest client"),
        }
    }

    /// Returns `true` if the candidate is allowed under the current
    /// snapshot, or no snapshot has been fetched yet (open default).
    pub async fn is_allowed(&self, address: &SomaAddress) -> bool {
        let g = self.state.read().await;
        match &g.set {
            None => true,
            Some(s) => s.contains(address),
        }
    }

    /// One-shot fetch + parse + replace. Returns the new size on
    /// success or an error message on failure. Callers wire this into
    /// a background tick.
    pub async fn refresh(&self) -> anyhow::Result<usize> {
        #[derive(serde::Deserialize)]
        struct Row {
            provider_address: String,
            #[serde(default)]
            authorized: bool,
        }
        let url = format!("{}/v1/providers/authorized", self.base_url.trim_end_matches('/'));
        let resp = self.http.get(&url).send().await?.error_for_status()?;
        let rows: Vec<Row> = resp.json().await?;
        let mut set = HashSet::with_capacity(rows.len());
        for r in rows {
            if !r.authorized {
                continue;
            }
            // Parse `0x...` or raw hex into a SomaAddress.
            let stripped = r.provider_address.strip_prefix("0x").unwrap_or(&r.provider_address);
            if let Ok(bytes) = hex::decode(stripped) {
                if bytes.len() == SomaAddress::LENGTH {
                    let mut arr = [0u8; SomaAddress::LENGTH];
                    arr.copy_from_slice(&bytes);
                    set.insert(SomaAddress::new(arr));
                }
            }
        }
        let size = set.len();
        let mut g = self.state.write().await;
        g.set = Some(set);
        Ok(size)
    }

    /// Spawn a background task that periodically refreshes the cache.
    /// Returns immediately; the first fetch happens inside the task.
    pub fn spawn_refresher(self: &Arc<Self>) {
        let me = self.clone();
        tokio::spawn(async move {
            loop {
                match me.refresh().await {
                    Ok(n) => tracing::info!(count = n, "trusted_providers refresh OK"),
                    Err(e) => tracing::warn!(err = %e, "trusted_providers refresh failed"),
                }
                tokio::time::sleep(me.refresh).await;
            }
        });
    }
}
