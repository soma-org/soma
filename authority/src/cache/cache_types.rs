// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-core/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use moka::sync::SegmentedCache as MokaCache;
use parking_lot::Mutex;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::hash::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use tracing::debug;
use types::object::Version;
pub enum CacheResult<T> {
    /// Entry is in the cache
    Hit(T),
    /// Entry is not in the cache and is known to not exist
    NegativeHit,
    /// Entry is not in the cache and may or may not exist in the store
    Miss,
}

/// CachedVersionMap is a map from version to value, with the additional contraints:
/// - The key (SequenceNumber) must be monotonically increasing for each insert. If
///   a key is inserted that is less than the previous key, it results in an assertion
///   failure.
/// - Similarly, only the item with the least key can be removed.
/// - The intent of these constraints is to ensure that there are never gaps in the collection,
///   so that membership in the map can be tested by comparing to both the highest and lowest
///   (first and last) entries.
#[derive(Debug)]
pub struct CachedVersionMap<V> {
    values: VecDeque<(Version, V)>,
}

impl<V> Default for CachedVersionMap<V> {
    fn default() -> Self {
        Self { values: VecDeque::new() }
    }
}

impl<V> CachedVersionMap<V> {
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn insert(&mut self, version: Version, value: V) {
        if !self.values.is_empty() {
            let back = self.values.back().unwrap().0;
            if back >= version {
                panic!("version must be monotonically increasing ({:?} < {:?})", back, version);
            }
        }
        self.values.push_back((version, value));
    }

    pub fn all_versions_lt_or_eq_descending<'a>(
        &'a self,
        version: &'a Version,
    ) -> impl Iterator<Item = &'a (Version, V)> {
        self.values.iter().rev().filter(move |(v, _)| v <= version)
    }

    pub fn get(&self, version: &Version) -> Option<&V> {
        for (v, value) in self.values.iter().rev() {
            match v.cmp(version) {
                Ordering::Less => return None,
                Ordering::Equal => return Some(value),
                Ordering::Greater => (),
            }
        }

        None
    }

    pub fn get_prior_to(&self, version: &Version) -> Option<(Version, &V)> {
        for (v, value) in self.values.iter().rev() {
            if v < version {
                return Some((*v, value));
            }
        }

        None
    }

    /// returns the newest (highest) version in the map
    pub fn get_highest(&self) -> Option<&(Version, V)> {
        self.values.back()
    }

    /// returns the oldest (lowest) version in the map
    pub fn get_least(&self) -> Option<&(Version, V)> {
        self.values.front()
    }

    // pop items from the front of the collection until the size is <= limit
    pub fn truncate_to(&mut self, limit: usize) {
        while self.values.len() > limit {
            self.values.pop_front();
        }
    }

    // remove the value if it is the first element in values.
    pub fn pop_oldest(&mut self, version: &Version) -> Option<V> {
        let oldest = self.values.pop_front()?;
        // if this assert fails it indicates we are committing transaction data out
        // of causal order
        assert_eq!(oldest.0, *version, "version must be the oldest in the map");
        Some(oldest.1)
    }
}

// an iterator adapter that asserts that the wrapped iterator yields elements in order
#[allow(dead_code)]
pub(super) struct AssertOrdered<I: Iterator> {
    iter: I,
    last: Option<I::Item>,
}

impl<I: Iterator> AssertOrdered<I> {
    #[allow(dead_code)]
    fn new(iter: I) -> Self {
        Self { iter, last: None }
    }
}

impl<I: IntoIterator> From<I> for AssertOrdered<I::IntoIter> {
    fn from(iter: I) -> Self {
        Self::new(iter.into_iter())
    }
}

impl<I: Iterator> Iterator for AssertOrdered<I>
where
    I::Item: Ord + Copy,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.iter.next();
        if let Some(next) = next {
            if let Some(last) = &self.last {
                assert!(*last < next, "iterator must yield elements in order");
            }
            self.last = Some(next);
        }
        next
    }
}

// Could just use the Ord trait but I think it would be confusing to overload it
// in that way.
pub trait IsNewer {
    fn is_newer_than(&self, other: &Self) -> bool;
}

pub struct MonotonicCache<K, V> {
    cache: MokaCache<K, Arc<Mutex<V>>>,
    // When inserting a possibly stale value, we prove that it is not stale by
    // ensuring that no fresh value was inserted since we began reading the value
    // we are inserting. We do this by hashing the key to an element in this array,
    // reading the current value, and then passing that value to insert(). If the
    // value is out of date, then there may have been an intervening write, so we
    // discard the insert attempt.
    key_generation: Vec<AtomicU64>,
}

#[derive(Copy, Clone)]
pub enum Ticket {
    // Read tickets are used when caching the result of a read from the db.
    // They are only valid if the generation number matches the current generation.
    // Used to ensure that no write occurred while we were reading from the db.
    Read(u64),
    // Write tickets are always valid. Used when caching writes, which cannot be stale.
    Write,
}

// key_generation should be big enough to make false positives unlikely. If, on
// average, there is one millisecond between acquiring the ticket and calling insert(),
// then even at 1 million inserts per second, there will be 1000 inserts between acquiring
// the ticket and calling insert(), so about 1/16th of the entries will be invalidated,
// so valid inserts will succeed with probability 15/16.
const KEY_GENERATION_SIZE: usize = 1024 * 16;

impl<K, V> MonotonicCache<K, V>
where
    K: Hash + Eq + Send + Sync + Copy + std::fmt::Debug + 'static,
    V: IsNewer + Clone + Send + Sync + 'static,
{
    pub fn new(cache_size: u64) -> Self {
        Self {
            cache: MokaCache::builder(8).max_capacity(cache_size).build(),
            key_generation: (0..KEY_GENERATION_SIZE).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    pub fn get(&self, key: &K) -> Option<Arc<Mutex<V>>> {
        self.cache.get(key)
    }

    fn generation(&self, key: &K) -> &AtomicU64 {
        let mut state = DefaultHasher::new();
        key.hash(&mut state);
        let hash = state.finish();
        &self.key_generation[(hash % KEY_GENERATION_SIZE as u64) as usize]
    }

    /// Get a ticket for caching the result of a read operation. The ticket will be
    /// expired if a writer writes a new version of the value.
    /// The caller must obtain the ticket BEFORE checking the dirty set and db. By
    /// obeying this rule, the caller can be sure that if their ticket remains valid
    /// at insert time, they either are inserting the most recent value, or a concurrent
    /// writer will shortly overwrite their value.
    pub fn get_ticket_for_read(&self, key: &K) -> Ticket {
        let r#gen = self.generation(key);
        Ticket::Read(r#gen.load(std::sync::atomic::Ordering::Acquire))
    }

    // Update the cache with guaranteed monotonicity. That is, if there are N
    // calls to the this function from N threads, the write with the newest value will
    // win the race regardless of what ordering the writes occur in.
    //
    // Caller should log the insert with trace! and increment the appropriate metric.
    pub fn insert(&self, key: &K, value: V, ticket: Ticket) -> Result<(), ()> {
        let r#gen = self.generation(key);

        // invalidate other readers as early as possible. If a reader acquires a
        // new ticket after this point, then it will read the new value from
        // the dirty set (or db).
        if matches!(ticket, Ticket::Write) {
            r#gen.fetch_add(1, std::sync::atomic::Ordering::Release);
        }

        let check_ticket = || -> Result<(), ()> {
            match ticket {
                Ticket::Read(ticket) => {
                    if ticket != r#gen.load(std::sync::atomic::Ordering::Acquire) {
                        return Err(());
                    }
                    Ok(())
                }
                Ticket::Write => Ok(()),
            }
        };

        // Warning: tricky code!
        let entry = self
            .cache
            .entry(*key)
            // Suppose there is a reader (who has an old version) and a writer (who has
            // the newest version by definition) both trying to insert when the cache has
            // no entry. Here are the possible outcomes:
            //
            // 1. Race in `or_optionally_insert_with`:
            //    1. Reader wins race, ticket is valid, and reader inserts old version.
            //       Writer will overwrite the old version after the !is_fresh check.
            //    2. Writer wins race. Reader will enter is_fresh check, lock entry, and
            //       find that its ticket is expired.
            //
            // 2. No race on `or_optionally_insert_with`:
            //    1. Reader inserts first (via `or_optionally_insert_with`), writer enters !is_fresh
            //       check and overwrites entry.
            //       1. There are two sub-cases here because the reader's entry could be evicted,
            //          but in either case the writer obviously overwrites it.
            //    2. Writer inserts first (via `or_optionally_insert_with`), invalidates ticket.
            //       Then, two cases can follow:
            //       1. Reader skips `or_optionally_insert_with` (because entry is present), enters
            //          !is_fresh check, and does not insert because its ticket is expired.
            //       2. The writer's cache entry is evicted already, so reader enters
            //          `or_optionally_insert_with`. The ticket is expired so we do not insert.
            //
            // The other cases are where there is already an entry. In this case neither reader
            // nor writer will enter `or_optionally_insert_with` callback. Instead they will both enter
            // the !is_fresh check and lock the entry:
            // 1. If the reader locks first, it will insert its old version. Then the writer
            //    will lock and overwrite it with the newer version.
            // 2. If the writer locks first, it will have already expired the ticket, and the
            //    reader will not insert anything.
            //
            // There may also be more than one concurrent reader. However, the only way the two
            // readers can have different versions is if there is concurrently a writer that wrote
            // a new version. In this case all stale readers will fail the ticket check, and only
            // up-to-date readers will remain. So we cannot have a bad insert caused by two readers
            // racing to insert, both with valid tickets.
            .or_optionally_insert_with(|| {
                check_ticket().ok()?;
                Some(Arc::new(Mutex::new(value.clone())))
            })
            // Note: Ticket::Write cannot expire, but an insert can still fail, in the case where
            // a writer and reader are racing to call `or_optionally_insert_with`, the reader wins,
            // but then fails to insert because its ticket is expired. Then no entry at all is inserted.
            .ok_or(())?;

        // !is_fresh means we did not insert a new entry in or_optionally_insert_with above.
        if !entry.is_fresh() {
            let mut entry = entry.value().lock();
            check_ticket()?;

            // Ticket expiry should make this assert impossible.
            if entry.is_newer_than(&value) {
                debug!("entry is newer than value {:?}", key);
            } else {
                *entry = value;
            }
        }

        Ok(())
    }

    pub fn invalidate(&self, key: &K) {
        self.cache.invalidate(key);
    }

    #[cfg(test)]
    pub fn contains_key(&self, key: &K) -> bool {
        self.cache.contains_key(key)
    }

    pub fn invalidate_all(&self) {
        self.cache.invalidate_all();
    }

    pub fn is_empty(&self) -> bool {
        self.cache.iter().next().is_none()
    }
}
