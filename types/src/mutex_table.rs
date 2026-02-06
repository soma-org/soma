use parking_lot::{ArcMutexGuard, ArcRwLockReadGuard, ArcRwLockWriteGuard, Mutex, RwLock};
use std::collections::HashMap;
use std::collections::hash_map::{DefaultHasher, RandomState};
use std::error::Error;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::info;

type OwnedMutexGuard<T> = ArcMutexGuard<parking_lot::RawMutex, T>;
type OwnedRwLockReadGuard<T> = ArcRwLockReadGuard<parking_lot::RawRwLock, T>;
type OwnedRwLockWriteGuard<T> = ArcRwLockWriteGuard<parking_lot::RawRwLock, T>;

pub trait Lock: Send + Sync + Default {
    type Guard;
    type ReadGuard;
    fn lock_owned(self: Arc<Self>) -> Self::Guard;
    fn try_lock_owned(self: Arc<Self>) -> Option<Self::Guard>;
    fn read_lock_owned(self: Arc<Self>) -> Self::ReadGuard;
}

impl Lock for Mutex<()> {
    type Guard = OwnedMutexGuard<()>;
    type ReadGuard = Self::Guard;

    fn lock_owned(self: Arc<Self>) -> Self::Guard {
        self.lock_arc()
    }

    fn try_lock_owned(self: Arc<Self>) -> Option<Self::Guard> {
        self.try_lock_arc()
    }

    fn read_lock_owned(self: Arc<Self>) -> Self::ReadGuard {
        self.lock_arc()
    }
}

impl Lock for RwLock<()> {
    type Guard = OwnedRwLockWriteGuard<()>;
    type ReadGuard = OwnedRwLockReadGuard<()>;

    fn lock_owned(self: Arc<Self>) -> Self::Guard {
        self.write_arc()
    }

    fn try_lock_owned(self: Arc<Self>) -> Option<Self::Guard> {
        self.try_write_arc()
    }

    fn read_lock_owned(self: Arc<Self>) -> Self::ReadGuard {
        self.read_arc()
    }
}

type InnerLockTable<K, L> = HashMap<K, Arc<L>>;
// MutexTable supports mutual exclusion on keys such as TransactionDigest or ObjectDigest
pub struct LockTable<K: Hash, L: Lock> {
    random_state: RandomState,
    lock_table: Arc<Vec<RwLock<InnerLockTable<K, L>>>>,
    _k: std::marker::PhantomData<K>,
    _cleaner: JoinHandle<()>,
    stop: Arc<AtomicBool>,
    size: Arc<AtomicUsize>,
}

pub type MutexTable<K> = LockTable<K, Mutex<()>>;
pub type RwLockTable<K> = LockTable<K, RwLock<()>>;

#[derive(Debug)]
pub enum TryAcquireLockError {
    LockTableLocked,
    LockEntryLocked,
}

impl fmt::Display for TryAcquireLockError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "operation would block")
    }
}

impl Error for TryAcquireLockError {}
pub type MutexGuard = OwnedMutexGuard<()>;
pub type RwLockGuard = OwnedRwLockReadGuard<()>;

impl<K: Hash + Eq + Send + Sync + 'static, L: Lock + 'static> LockTable<K, L> {
    pub fn new_with_cleanup(
        num_shards: usize,
        cleanup_period: Duration,
        cleanup_initial_delay: Duration,
        cleanup_entries_threshold: usize,
    ) -> Self {
        let num_shards = if cfg!(msim) { 4 } else { num_shards };

        let lock_table: Arc<Vec<RwLock<InnerLockTable<K, L>>>> =
            Arc::new((0..num_shards).map(|_| RwLock::new(HashMap::new())).collect());
        let cloned = lock_table.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_cloned = stop.clone();
        let size: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
        let size_cloned = size.clone();
        Self {
            random_state: RandomState::new(),
            lock_table,
            _k: std::marker::PhantomData {},
            _cleaner: tokio::spawn(async move {
                tokio::time::sleep(cleanup_initial_delay).await;
                let mut previous_cleanup_instant = Instant::now();
                while !stop_cloned.load(Ordering::SeqCst) {
                    if size_cloned.load(Ordering::SeqCst) >= cleanup_entries_threshold
                        || previous_cleanup_instant.elapsed() >= cleanup_period
                    {
                        let num_removed = Self::cleanup(cloned.clone());
                        size_cloned.fetch_sub(num_removed, Ordering::SeqCst);
                        previous_cleanup_instant = Instant::now();
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                info!("Stopping mutex table cleanup!");
            }),
            stop,
            size,
        }
    }

    pub fn new(num_shards: usize) -> Self {
        Self::new_with_cleanup(num_shards, Duration::from_secs(10), Duration::from_secs(10), 10_000)
    }

    pub fn size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    pub fn cleanup(lock_table: Arc<Vec<RwLock<InnerLockTable<K, L>>>>) -> usize {
        let mut num_removed: usize = 0;
        for shard in lock_table.iter() {
            let map = shard.try_write();
            if map.is_none() {
                continue;
            }
            map.unwrap().retain(|_k, v| {
                // MutexMap::(try_|)acquire_locks will lock the map and call Arc::clone on the entry
                // This check ensures that we only drop entry from the map if this is the only mutex copy
                // This check is also likely sufficient e.g. you don't even need try_lock below, but keeping it just in case
                if Arc::strong_count(v) == 1 {
                    num_removed += 1;
                    false
                } else {
                    true
                }
            });
        }
        num_removed
    }

    fn get_lock_idx(&self, key: &K) -> usize {
        let mut hasher = if !cfg!(test) {
            self.random_state.build_hasher()
        } else {
            // be deterministic for tests
            DefaultHasher::new()
        };

        key.hash(&mut hasher);
        // unwrap ok - converting u64 -> usize
        let hash: usize = hasher.finish().try_into().unwrap();
        hash % self.lock_table.len()
    }

    pub fn acquire_locks<I>(&self, object_iter: I) -> Vec<L::Guard>
    where
        I: Iterator<Item = K>,
        K: Ord,
    {
        let mut objects: Vec<K> = object_iter.into_iter().collect();
        objects.sort_unstable();
        objects.dedup();

        let mut guards = Vec::with_capacity(objects.len());
        for object in objects.into_iter() {
            guards.push(self.acquire_lock(object));
        }
        guards
    }

    pub fn acquire_read_locks(&self, mut objects: Vec<K>) -> Vec<L::ReadGuard>
    where
        K: Ord,
    {
        objects.sort_unstable();
        objects.dedup();
        let mut guards = Vec::with_capacity(objects.len());
        for object in objects.into_iter() {
            guards.push(self.get_lock(object).read_lock_owned());
        }
        guards
    }

    pub fn get_lock(&self, k: K) -> Arc<L> {
        let lock_idx = self.get_lock_idx(&k);
        let element = {
            let map = self.lock_table[lock_idx].read();
            map.get(&k).cloned()
        };
        if let Some(element) = element {
            element
        } else {
            // element doesn't exist

            {
                let mut map = self.lock_table[lock_idx].write();
                map.entry(k)
                    .or_insert_with(|| {
                        self.size.fetch_add(1, Ordering::SeqCst);
                        Arc::new(L::default())
                    })
                    .clone()
            }
        }
    }

    pub fn acquire_lock(&self, k: K) -> L::Guard {
        self.get_lock(k).lock_owned()
    }

    pub fn try_acquire_lock(&self, k: K) -> Result<L::Guard, TryAcquireLockError> {
        let lock_idx = self.get_lock_idx(&k);
        let element = {
            let map =
                self.lock_table[lock_idx].try_read().ok_or(TryAcquireLockError::LockTableLocked)?;
            map.get(&k).cloned()
        };
        if let Some(element) = element {
            let lock = element.try_lock_owned();
            lock.ok_or(TryAcquireLockError::LockEntryLocked)
        } else {
            // element doesn't exist
            let element = {
                let mut map = self.lock_table[lock_idx]
                    .try_write()
                    .ok_or(TryAcquireLockError::LockTableLocked)?;
                map.entry(k)
                    .or_insert_with(|| {
                        self.size.fetch_add(1, Ordering::SeqCst);
                        Arc::new(L::default())
                    })
                    .clone()
            };
            let lock = element.try_lock_owned();
            lock.ok_or(TryAcquireLockError::LockEntryLocked)
        }
    }
}

impl<K: Hash, L: Lock> Drop for LockTable<K, L> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}
