// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-provider liveness cache.
//!
//! Probes a provider's `GET <endpoint>/health` with a short timeout and
//! caches the result for a brief TTL. The router consults the cache as a
//! filter step before scoring candidates, so providers whose registered
//! endpoint is unreachable get skipped at routing time rather than
//! producing a downstream `OpenChannel` / connection-refused failure
//! that surfaces to the user as an opaque `channel_error`.
//!
//! Why this lives in the proxy and not a sidecar: somacode used to run a
//! Bun-side trusted-providers server that did its own liveness probe and
//! exposed the filtered set via `--trusted-providers-url`. That works
//! when somacode is the only entry point, but anyone running
//! `soma proxy` directly gets the unfiltered indexer list. Folding the
//! probe into the proxy makes the filter unconditional.
//!
//! Cache semantics:
//! - Entries are keyed by `SomaAddress`.
//! - `is_live` returns the cached value when the entry's age is below
//!   `ttl`; otherwise it probes synchronously, updates the cache, and
//!   returns the fresh value.
//! - `mark_dead` evicts an entry so the next call re-probes. Callers
//!   wire this on `OpenChannel` / connection-refused failures so a
//!   provider that flapped after the last refresh isn't re-picked
//!   immediately.
//! - A background refresher (`spawn_refresher`) walks every known
//!   provider every `refresh` and bulk-probes with bounded concurrency.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::{RwLock, Semaphore};
use types::base::SomaAddress;

use crate::health::ProviderHealth;
use crate::transport::{EndpointAddr, IrohBuyer, WireRequestMeta};

/// Maximum concurrent `/health` probes inside `probe_all`. Unbounded
/// `Promise.all`-style fan-out can starve the tokio runtime and the
/// proxy's other HTTP work; 8 is fast enough for typical provider
/// populations and keeps the runtime responsive.
const PROBE_CONCURRENCY: usize = 8;

#[derive(Debug, Clone, Copy)]
struct Entry {
    last_check: Instant,
    is_live: bool,
    /// Last `saturation` the provider published on its `/health`
    /// payload (`None` if unreachable or not reported).
    saturation: Option<f64>,
}

pub struct LivenessCache {
    entries: RwLock<HashMap<SomaAddress, Entry>>,
    /// iroh endpoint used to probe providers' `/health` over iroh.
    iroh: Arc<IrohBuyer>,
    /// Per-probe timeout.
    timeout: Duration,
    /// How long a probe result stays fresh in the cache.
    ttl: Duration,
    /// How often the background refresher walks the full set.
    /// On proxy restart, the cache starts empty — the first chat
    /// triggers a synchronous probe via `is_live`. No persistence is
    /// needed because the chain mirror is the source of truth for
    /// the provider population.
    refresh: Duration,
}

impl LivenessCache {
    pub fn new(iroh: Arc<IrohBuyer>, timeout_ms: u64, refresh_secs: u64) -> Self {
        let timeout = Duration::from_millis(timeout_ms.max(100));
        let refresh = Duration::from_secs(refresh_secs.max(5));
        // A probe's result is "stale" when older than the refresh
        // interval — bounding TTL by refresh keeps the on-demand path
        // self-healing even without a background refresher.
        let ttl = refresh;
        Self { entries: RwLock::new(HashMap::new()), iroh, timeout, ttl, refresh }
    }

    /// Probe a provider's `/health` over iroh. Returns liveness plus the
    /// provider's published saturation. A provider with no iroh address is
    /// unreachable (iroh-only), so it probes as not-live.
    async fn iroh_health(
        iroh: &IrohBuyer,
        iroh_addr: Option<&EndpointAddr>,
        timeout: Duration,
    ) -> (bool, Option<f64>) {
        let Some(addr) = iroh_addr else { return (false, None) };
        let meta = WireRequestMeta {
            method: "GET".to_string(),
            path: "/health".to_string(),
            headers: Vec::new(),
        };
        let fut = iroh.post(addr.clone(), meta, Bytes::new());
        match tokio::time::timeout(timeout, fut).await {
            Ok(Ok(resp)) if (200..300).contains(&resp.status) => {
                let body = resp.read_to_end().await.unwrap_or_default();
                let sat =
                    serde_json::from_slice::<ProviderHealth>(&body).ok().and_then(|h| h.saturation);
                (true, sat)
            }
            _ => (false, None),
        }
    }

    /// Returns `true` if the provider responded to its `/health`
    /// endpoint within `timeout` on the most recent probe (cached for
    /// `ttl`). Probes synchronously on cache miss / stale entry.
    pub async fn is_live(&self, addr: &SomaAddress, iroh_addr: Option<&EndpointAddr>) -> bool {
        if let Some(e) = self.entries.read().await.get(addr).copied() {
            if e.last_check.elapsed() < self.ttl {
                return e.is_live;
            }
        }
        let (live, saturation) = Self::iroh_health(&self.iroh, iroh_addr, self.timeout).await;
        self.entries
            .write()
            .await
            .insert(*addr, Entry { last_check: Instant::now(), is_live: live, saturation });
        live
    }

    /// The provider's last-published device saturation in `[0,1]`, as
    /// seen on its most recent `/health` probe. `None` if the provider
    /// hasn't been probed, is unreachable, or doesn't report it. The
    /// router uses this to steer demand away from saturated devices.
    pub async fn saturation(&self, addr: &SomaAddress) -> Option<f64> {
        self.entries.read().await.get(addr).and_then(|e| e.saturation)
    }

    /// Drop the cache entry for `addr` so the next `is_live` call
    /// re-probes. Use this when a downstream RPC against the provider
    /// failed in a way that suggests its endpoint went down between
    /// the last probe and the call (connection refused, channel-open
    /// failure caused by no upstream, etc.).
    pub async fn mark_dead(&self, addr: &SomaAddress) {
        self.entries.write().await.remove(addr);
    }

    /// Concurrently probe a batch and update the cache. Returns the
    /// number of providers that responded.
    pub async fn probe_all(&self, providers: &[(SomaAddress, Option<EndpointAddr>)]) -> usize {
        if providers.is_empty() {
            return 0;
        }
        let sem = Arc::new(Semaphore::new(PROBE_CONCURRENCY));
        let now = Instant::now();
        let mut handles = Vec::with_capacity(providers.len());
        for (addr, iroh_addr) in providers {
            let permit = sem.clone().acquire_owned().await.expect("semaphore not closed");
            let iroh = self.iroh.clone();
            let timeout = self.timeout;
            let addr = *addr;
            let iroh_addr = iroh_addr.clone();
            handles.push(tokio::spawn(async move {
                let (live, saturation) =
                    Self::iroh_health(&iroh, iroh_addr.as_ref(), timeout).await;
                drop(permit);
                (addr, live, saturation)
            }));
        }
        let mut alive = 0;
        let mut entries = self.entries.write().await;
        for h in handles {
            if let Ok((addr, live, saturation)) = h.await {
                if live {
                    alive += 1;
                }
                entries.insert(addr, Entry { last_check: now, is_live: live, saturation });
            }
        }
        alive
    }

    /// Refresh cadence — exposed so callers driving their own refresh
    /// loop know how long to sleep between ticks. Bounded to a 5 s
    /// floor at construction.
    pub fn refresh_interval(&self) -> Duration {
        self.refresh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(b: u8) -> SomaAddress {
        let mut a = [0u8; SomaAddress::LENGTH];
        a[0] = b;
        SomaAddress::new(a)
    }

    /// `mark_dead` evicts the entry; the next `is_live` call must
    /// re-probe (which fails for a provider with no iroh address).
    #[tokio::test]
    async fn mark_dead_forces_reprobe() {
        let iroh = Arc::new(crate::transport::IrohBuyer::bind_local().await.unwrap());
        let cache = LivenessCache::new(iroh, 50, 30);
        // Seed a "live" entry directly so we can observe eviction.
        cache
            .entries
            .write()
            .await
            .insert(addr(1), Entry { last_check: Instant::now(), is_live: true, saturation: None });
        assert!(cache.is_live(&addr(1), None).await); // hits cache
        cache.mark_dead(&addr(1)).await;
        // No cache → probes for real → fails (no iroh address to reach).
        assert!(!cache.is_live(&addr(1), None).await);
    }

    /// A fresh cache hit returns the stored value without probing,
    /// even when the endpoint string is bogus — proves we don't
    /// re-probe on every call.
    #[tokio::test]
    async fn fresh_cache_hit_skips_probe() {
        let iroh = Arc::new(crate::transport::IrohBuyer::bind_local().await.unwrap());
        let cache = LivenessCache::new(iroh, 50, 30);
        cache
            .entries
            .write()
            .await
            .insert(addr(2), Entry { last_check: Instant::now(), is_live: true, saturation: None });
        // No iroh address would probe false; the cache hit must shortcut that.
        assert!(cache.is_live(&addr(2), None).await);
    }

    /// An entry older than `ttl` is treated as stale and re-probed.
    #[tokio::test]
    async fn stale_entry_reprobes() {
        let iroh = Arc::new(crate::transport::IrohBuyer::bind_local().await.unwrap());
        let cache = LivenessCache::new(iroh, 50, 5);
        // Force a stale entry (last_check 1 hour ago).
        cache.entries.write().await.insert(
            addr(3),
            Entry {
                last_check: Instant::now() - Duration::from_secs(3600),
                is_live: true,
                saturation: None,
            },
        );
        // Stale → re-probe → fails (no iroh address to reach).
        assert!(!cache.is_live(&addr(3), None).await);
    }
}
