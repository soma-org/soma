// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::hash::Hash;

use lru::LruCache;
use parking_lot::RwLock;

use crate::error::SomaResult;

// Cache up to this many verified certs. We will need to tune this number in the future - a decent
// guess to start with is that it should be 10-20 times larger than peak transactions per second,
// on the assumption that we should see most certs twice within about 10-20 seconds at most:
// Once via RPC, once via consensus.
const VERIFIED_CERTIFICATE_CACHE_SIZE: usize = 100_000;

pub struct VerifiedDigestCache<D> {
    inner: RwLock<LruCache<D, ()>>,
}

impl<D: Hash + Eq + Copy> Default for VerifiedDigestCache<D> {
    fn default() -> Self {
        Self {
            inner: RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(VERIFIED_CERTIFICATE_CACHE_SIZE).unwrap(),
            )),
        }
    }
}

impl<D: Hash + Eq + Copy> VerifiedDigestCache<D> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_cached(&self, digest: &D) -> bool {
        let inner = self.inner.read();
        inner.contains(digest)
    }

    pub fn cache_digest(&self, digest: D) {
        let mut inner = self.inner.write();
        if let Some(old) = inner.push(digest, ()) {}
    }

    pub fn cache_digests(&self, digests: Vec<D>) {
        let mut inner = self.inner.write();
        digests.into_iter().for_each(|d| if let Some(old) = inner.push(d, ()) {});
    }

    pub fn is_verified<F, G>(&self, digest: D, verify_callback: F, uncached_checks: G) -> SomaResult
    where
        F: FnOnce() -> SomaResult,
        G: FnOnce() -> SomaResult,
    {
        if !self.is_cached(&digest) {
            verify_callback()?;
            self.cache_digest(digest);
        } else {
            // Checks that are required to be performed outside the cache.
            uncached_checks()?;
        }
        Ok(())
    }

    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.clear();
    }

    // Initialize an empty cache when the cache is not needed (in testing scenarios, graphql and rosetta initialization).
    pub fn new_empty() -> Self {
        Self::new()
    }
}
