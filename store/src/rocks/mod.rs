// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

pub mod errors;
mod options;
mod rocks_util;
pub(crate) mod safe_iter;
use std::borrow::Borrow;
use std::collections::HashSet;
use std::ffi::CStr;
use std::marker::PhantomData;
use std::ops::{Bound, Deref, RangeBounds};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use backoff::backoff::Backoff;
use fastcrypto::hash::{Digest, HashFunction};
use rocksdb::checkpoint::Checkpoint;
use rocksdb::{
    AsColumnFamilyRef, ColumnFamilyDescriptor, DBPinnableSlice, Error, LiveFile, MultiThreaded,
    ReadOptions, WriteBatch,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tracing::{debug, instrument};

use crate::memstore::{InMemoryBatch, InMemoryDB};
use crate::rocks::errors::{typed_store_err_from_bcs_err, typed_store_err_from_rocks_err};
pub use crate::rocks::options::{
    DBMapTableConfigMap, DBOptions, ReadWriteOptions, default_db_options, read_size_from_env,
};
use crate::rocks::safe_iter::{SafeIter, SafeRevIter};
use crate::util::{iterator_bounds, iterator_bounds_with_range};
use crate::{DbIterator, Map, TypedStoreError, be_fix_int_ser, nondeterministic};

#[allow(dead_code, unsafe_code)]
const ROCKSDB_PROPERTY_TOTAL_BLOB_FILES_SIZE: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked("rocksdb.total-blob-file-size\0".as_bytes()) };

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct RocksDB {
    pub underlying: rocksdb::DBWithThreadMode<MultiThreaded>,
}

impl Drop for RocksDB {
    fn drop(&mut self) {
        self.underlying.cancel_all_background_work(/* wait */ true);
    }
}

#[derive(Clone)]
pub enum ColumnFamily {
    Rocks(String),
    InMemory(String),
}

impl std::fmt::Debug for ColumnFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColumnFamily::Rocks(name) => write!(f, "RocksDB cf: {}", name),
            ColumnFamily::InMemory(name) => write!(f, "InMemory cf: {}", name),
        }
    }
}

impl ColumnFamily {
    fn rocks_cf<'a>(&self, rocks_db: &'a RocksDB) -> Arc<rocksdb::BoundColumnFamily<'a>> {
        match &self {
            ColumnFamily::Rocks(name) => rocks_db
                .underlying
                .cf_handle(name)
                .expect("Map-keying column family should have been checked at DB creation"),
            _ => unreachable!("invariant is checked by the caller"),
        }
    }
}

pub enum Storage {
    Rocks(RocksDB),
    InMemory(InMemoryDB),
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::Rocks(db) => write!(f, "RocksDB Storage {:?}", db),
            Storage::InMemory(db) => write!(f, "InMemoryDB Storage {:?}", db),
        }
    }
}

#[derive(Debug)]
pub struct Database {
    storage: Storage,
}

enum GetResult<'a> {
    Rocks(DBPinnableSlice<'a>),
    InMemory(Vec<u8>),
}

impl Deref for GetResult<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            GetResult::Rocks(d) => d.deref(),
            GetResult::InMemory(d) => d.deref(),
        }
    }
}

impl Database {
    pub fn new(storage: Storage) -> Self {
        Self { storage }
    }

    /// Flush all memtables to SST files on disk.
    pub fn flush(&self) -> Result<(), TypedStoreError> {
        match &self.storage {
            Storage::Rocks(rocks_db) => rocks_db.underlying.flush().map_err(|e| {
                TypedStoreError::RocksDBError(format!("Failed to flush database: {}", e))
            }),
            Storage::InMemory(_) => {
                // InMemory databases don't need flushing
                Ok(())
            }
        }
    }

    fn get<K: AsRef<[u8]>>(
        &self,
        cf: &ColumnFamily,
        key: K,
        readopts: &ReadOptions,
    ) -> Result<Option<GetResult<'_>>, TypedStoreError> {
        match (&self.storage, cf) {
            (Storage::Rocks(db), ColumnFamily::Rocks(_)) => Ok(db
                .underlying
                .get_pinned_cf_opt(&cf.rocks_cf(db), key, readopts)
                .map_err(typed_store_err_from_rocks_err)?
                .map(GetResult::Rocks)),
            (Storage::InMemory(db), ColumnFamily::InMemory(cf_name)) => {
                Ok(db.get(cf_name, key).map(GetResult::InMemory))
            }

            _ => Err(TypedStoreError::RocksDBError("typed store invariant violation".to_string())),
        }
    }

    fn multi_get<I, K>(
        &self,
        cf: &ColumnFamily,
        keys: I,
        readopts: &ReadOptions,
    ) -> Vec<Result<Option<GetResult<'_>>, TypedStoreError>>
    where
        I: IntoIterator<Item = K>,
        K: AsRef<[u8]>,
    {
        match (&self.storage, cf) {
            (Storage::Rocks(db), ColumnFamily::Rocks(_)) => {
                let keys_vec: Vec<K> = keys.into_iter().collect();
                let res = db.underlying.batched_multi_get_cf_opt(
                    &cf.rocks_cf(db),
                    keys_vec.iter(),
                    /* sorted_input */ false,
                    readopts,
                );
                res.into_iter()
                    .map(|r| {
                        r.map_err(typed_store_err_from_rocks_err)
                            .map(|item| item.map(GetResult::Rocks))
                    })
                    .collect()
            }
            (Storage::InMemory(db), ColumnFamily::InMemory(cf_name)) => db
                .multi_get(cf_name, keys)
                .into_iter()
                .map(|r| Ok(r.map(GetResult::InMemory)))
                .collect(),

            _ => unreachable!("typed store invariant violation"),
        }
    }

    pub fn drop_cf(&self, name: &str) -> Result<(), rocksdb::Error> {
        match &self.storage {
            Storage::Rocks(db) => db.underlying.drop_cf(name),
            Storage::InMemory(db) => {
                db.drop_cf(name);
                Ok(())
            }
        }
    }

    pub fn delete_file_in_range<K: AsRef<[u8]>>(
        &self,
        cf: &impl AsColumnFamilyRef,
        from: K,
        to: K,
    ) -> Result<(), rocksdb::Error> {
        match &self.storage {
            Storage::Rocks(rocks) => rocks.underlying.delete_file_in_range_cf(cf, from, to),
            _ => unimplemented!("delete_file_in_range is only supported for rocksdb backend"),
        }
    }

    fn delete_cf<K: AsRef<[u8]>>(&self, cf: &ColumnFamily, key: K) -> Result<(), TypedStoreError> {
        let ret = match (&self.storage, cf) {
            (Storage::Rocks(db), ColumnFamily::Rocks(_)) => db
                .underlying
                .delete_cf(&cf.rocks_cf(db), key)
                .map_err(typed_store_err_from_rocks_err),
            (Storage::InMemory(db), ColumnFamily::InMemory(cf_name)) => {
                db.delete(cf_name, key.as_ref());
                Ok(())
            }
            _ => Err(TypedStoreError::RocksDBError("typed store invariant violation".to_string())),
        };
        #[allow(clippy::let_and_return)]
        ret
    }

    pub fn path_for_pruning(&self) -> &Path {
        match &self.storage {
            Storage::Rocks(rocks) => rocks.underlying.path(),
            _ => unimplemented!("method is only supported for rocksdb backend"),
        }
    }

    fn put_cf(
        &self,
        cf: &ColumnFamily,
        key: Vec<u8>,
        value: Vec<u8>,
    ) -> Result<(), TypedStoreError> {
        let ret = match (&self.storage, cf) {
            (Storage::Rocks(db), ColumnFamily::Rocks(_)) => db
                .underlying
                .put_cf(&cf.rocks_cf(db), key, value)
                .map_err(typed_store_err_from_rocks_err),
            (Storage::InMemory(db), ColumnFamily::InMemory(cf_name)) => {
                db.put(cf_name, key, value);
                Ok(())
            }
            _ => Err(TypedStoreError::RocksDBError("typed store invariant violation".to_string())),
        };
        #[allow(clippy::let_and_return)]
        ret
    }

    pub fn key_may_exist_cf<K: AsRef<[u8]>>(
        &self,
        cf_name: &str,
        key: K,
        readopts: &ReadOptions,
    ) -> bool {
        match &self.storage {
            // [`rocksdb::DBWithThreadMode::key_may_exist_cf`] can have false positives,
            // but no false negatives. We use it to short-circuit the absent case
            Storage::Rocks(rocks) => {
                rocks.underlying.key_may_exist_cf_opt(&rocks_cf(rocks, cf_name), key, readopts)
            }
            _ => true,
        }
    }

    pub fn write(&self, batch: StorageWriteBatch) -> Result<(), TypedStoreError> {
        self.write_opt(batch, &rocksdb::WriteOptions::default())
    }

    pub fn write_opt(
        &self,
        batch: StorageWriteBatch,
        write_options: &rocksdb::WriteOptions,
    ) -> Result<(), TypedStoreError> {
        let ret = match (&self.storage, batch) {
            (Storage::Rocks(rocks), StorageWriteBatch::Rocks(batch)) => rocks
                .underlying
                .write_opt(batch, write_options)
                .map_err(typed_store_err_from_rocks_err),
            (Storage::InMemory(db), StorageWriteBatch::InMemory(batch)) => {
                // InMemory doesn't support write options
                db.write(batch);
                Ok(())
            }
            _ => Err(TypedStoreError::RocksDBError(
                "using invalid batch type for the database".to_string(),
            )),
        };
        #[allow(clippy::let_and_return)]
        ret
    }

    pub fn compact_range_cf<K: AsRef<[u8]>>(
        &self,
        cf_name: &str,
        start: Option<K>,
        end: Option<K>,
    ) {
        if let Storage::Rocks(rocksdb) = &self.storage {
            rocksdb.underlying.compact_range_cf(&rocks_cf(rocksdb, cf_name), start, end);
        }
    }

    pub fn checkpoint(&self, path: &Path) -> Result<(), TypedStoreError> {
        // TODO: implement for other storage types
        if let Storage::Rocks(rocks) = &self.storage {
            let checkpoint =
                Checkpoint::new(&rocks.underlying).map_err(typed_store_err_from_rocks_err)?;
            checkpoint
                .create_checkpoint(path)
                .map_err(|e| TypedStoreError::RocksDBError(e.to_string()))?;
        }
        Ok(())
    }

    fn db_name(&self) -> String {
        "default".to_string()
    }

    pub fn live_files(&self) -> Result<Vec<LiveFile>, Error> {
        match &self.storage {
            Storage::Rocks(rocks) => rocks.underlying.live_files(),
            _ => Ok(vec![]),
        }
    }
}

fn rocks_cf<'a>(rocks_db: &'a RocksDB, cf_name: &str) -> Arc<rocksdb::BoundColumnFamily<'a>> {
    rocks_db
        .underlying
        .cf_handle(cf_name)
        .expect("Map-keying column family should have been checked at DB creation")
}

fn rocks_cf_from_db<'a>(
    db: &'a Database,
    cf_name: &str,
) -> Result<Arc<rocksdb::BoundColumnFamily<'a>>, TypedStoreError> {
    match &db.storage {
        Storage::Rocks(rocksdb) => Ok(rocksdb
            .underlying
            .cf_handle(cf_name)
            .expect("Map-keying column family should have been checked at DB creation")),
        _ => Err(TypedStoreError::RocksDBError(
            "using invalid batch type for the database".to_string(),
        )),
    }
}

/// An interface to a rocksDB database, keyed by a columnfamily
#[derive(Clone, Debug)]
pub struct DBMap<K, V> {
    pub db: Arc<Database>,
    _phantom: PhantomData<fn(K) -> V>,
    column_family: ColumnFamily,
    // the column family under which the map is stored
    cf: String,
    pub opts: ReadWriteOptions,
}

#[allow(unsafe_code)]
unsafe impl<K: Send, V: Send> Send for DBMap<K, V> {}

impl<K, V> DBMap<K, V> {
    pub(crate) fn new(
        db: Arc<Database>,
        opts: &ReadWriteOptions,
        opt_cf: &str,
        column_family: ColumnFamily,
        _is_deprecated: bool,
    ) -> Self {
        let _db_cloned = Arc::downgrade(&db.clone());
        let _cf = opt_cf.to_string();

        DBMap {
            db: db.clone(),
            opts: opts.clone(),
            _phantom: PhantomData,
            column_family,
            cf: opt_cf.to_string(),
        }
    }

    /// Reopens an open database as a typed map operating under a specific column family.
    /// if no column family is passed, the default column family is used.
    #[instrument(level = "debug", skip(db), err)]
    pub fn reopen(
        db: &Arc<Database>,
        opt_cf: Option<&str>,
        rw_options: &ReadWriteOptions,
        is_deprecated: bool,
    ) -> Result<Self, TypedStoreError> {
        let cf_key = opt_cf.unwrap_or(rocksdb::DEFAULT_COLUMN_FAMILY_NAME).to_owned();
        Ok(DBMap::new(
            db.clone(),
            rw_options,
            &cf_key,
            ColumnFamily::Rocks(cf_key.to_string()),
            is_deprecated,
        ))
    }

    pub fn cf_name(&self) -> &str {
        &self.cf
    }

    pub fn batch(&self) -> DBBatch {
        let batch = match &self.db.storage {
            Storage::Rocks(_) => StorageWriteBatch::Rocks(WriteBatch::default()),
            Storage::InMemory(_) => StorageWriteBatch::InMemory(InMemoryBatch::default()),
        };
        DBBatch::new(&self.db, batch)
    }

    pub fn flush(&self) -> Result<(), TypedStoreError> {
        self.db.flush()
    }

    pub fn compact_range<J: Serialize>(&self, start: &J, end: &J) -> Result<(), TypedStoreError> {
        let from_buf = be_fix_int_ser(start);
        let to_buf = be_fix_int_ser(end);
        self.db.compact_range_cf(&self.cf, Some(from_buf), Some(to_buf));
        Ok(())
    }

    pub fn compact_range_raw(
        &self,
        cf_name: &str,
        start: Vec<u8>,
        end: Vec<u8>,
    ) -> Result<(), TypedStoreError> {
        self.db.compact_range_cf(cf_name, Some(start), Some(end));
        Ok(())
    }

    /// Returns a vector of raw values corresponding to the keys provided.
    fn multi_get_pinned<J>(
        &self,
        keys: impl IntoIterator<Item = J>,
    ) -> Result<Vec<Option<GetResult<'_>>>, TypedStoreError>
    where
        J: Borrow<K>,
        K: Serialize,
    {
        let keys_bytes = keys.into_iter().map(|k| be_fix_int_ser(k.borrow()));
        let results: Result<Vec<_>, TypedStoreError> = self
            .db
            .multi_get(&self.column_family, keys_bytes, &self.opts.readopts())
            .into_iter()
            .collect();
        let entries = results?;
        let _entry_size = entries.iter().flatten().map(|entry| entry.len()).sum::<usize>();

        Ok(entries)
    }

    #[allow(dead_code)]
    fn get_rocksdb_int_property(
        rocksdb: &RocksDB,
        cf: &impl AsColumnFamilyRef,
        property_name: &std::ffi::CStr,
    ) -> Result<i64, TypedStoreError> {
        match rocksdb.underlying.property_int_value_cf(cf, property_name) {
            Ok(Some(value)) => Ok(value.min(i64::MAX as u64).try_into().unwrap_or_default()),
            Ok(None) => Ok(0),
            Err(e) => Err(TypedStoreError::RocksDBError(e.into_string())),
        }
    }

    pub fn checkpoint_db(&self, path: &Path) -> Result<(), TypedStoreError> {
        self.db.checkpoint(path)
    }

    /// Creates a safe reversed iterator with optional bounds.
    /// Both upper bound and lower bound are included.
    #[allow(clippy::complexity)]
    pub fn reversed_safe_iter_with_bounds(
        &self,
        lower_bound: Option<K>,
        upper_bound: Option<K>,
    ) -> Result<DbIterator<'_, (K, V)>, TypedStoreError>
    where
        K: Serialize + DeserializeOwned,
        V: Serialize + DeserializeOwned,
    {
        let (it_lower_bound, it_upper_bound) = iterator_bounds_with_range::<K>((
            lower_bound.as_ref().map(Bound::Included).unwrap_or(Bound::Unbounded),
            upper_bound.as_ref().map(Bound::Included).unwrap_or(Bound::Unbounded),
        ));
        match &self.db.storage {
            Storage::Rocks(db) => {
                let readopts = rocks_util::apply_range_bounds(
                    self.opts.readopts(),
                    it_lower_bound,
                    it_upper_bound,
                );
                let upper_bound_key = upper_bound.as_ref().map(|k| be_fix_int_ser(&k));
                let db_iter = db.underlying.raw_iterator_cf_opt(&rocks_cf(db, &self.cf), readopts);

                let iter = SafeIter::new(self.cf.clone(), db_iter);
                Ok(Box::new(SafeRevIter::new(iter, upper_bound_key)))
            }
            Storage::InMemory(db) => {
                Ok(db.iterator(&self.cf, it_lower_bound, it_upper_bound, true))
            }
        }
    }
}

pub enum StorageWriteBatch {
    Rocks(rocksdb::WriteBatch),
    InMemory(InMemoryBatch),
}

/// Provides a mutable struct to form a collection of database write operations, and execute them.
///
/// Batching write and delete operations is faster than performing them one by one and ensures their atomicity,
///  ie. they are all written or none is.
/// This is also true of operations across column families in the same database.
///
/// Serializations / Deserialization, and naming of column families is performed by passing a DBMap<K,V>
/// with each operation.
pub struct DBBatch {
    database: Arc<Database>,
    batch: StorageWriteBatch,
}

impl DBBatch {
    /// Create a new batch associated with a DB reference.
    ///
    /// Use `open_cf` to get the DB reference or an existing open database.
    pub fn new(dbref: &Arc<Database>, batch: StorageWriteBatch) -> Self {
        DBBatch { database: dbref.clone(), batch }
    }

    /// Consume the batch and write its operations to the database
    #[instrument(level = "trace", skip_all, err)]
    pub fn write(self) -> Result<(), TypedStoreError> {
        self.write_opt(&rocksdb::WriteOptions::default())
    }

    /// Consume the batch and write its operations to the database with custom write options
    #[instrument(level = "trace", skip_all, err)]
    pub fn write_opt(self, write_options: &rocksdb::WriteOptions) -> Result<(), TypedStoreError> {
        let _db_name = self.database.db_name();

        self.database.write_opt(self.batch, write_options)?;

        Ok(())
    }

    pub fn size_in_bytes(&self) -> usize {
        match self.batch {
            StorageWriteBatch::Rocks(ref b) => b.size_in_bytes(),
            StorageWriteBatch::InMemory(_) => 0,
        }
    }

    pub fn delete_batch<J: Borrow<K>, K: Serialize, V>(
        &mut self,
        db: &DBMap<K, V>,
        purged_vals: impl IntoIterator<Item = J>,
    ) -> Result<(), TypedStoreError> {
        if !Arc::ptr_eq(&db.db, &self.database) {
            return Err(TypedStoreError::CrossDBBatch);
        }

        purged_vals.into_iter().try_for_each::<_, Result<_, TypedStoreError>>(|k| {
            let k_buf = be_fix_int_ser(k.borrow());
            match (&mut self.batch, &db.column_family) {
                (StorageWriteBatch::Rocks(b), ColumnFamily::Rocks(name)) => {
                    b.delete_cf(&rocks_cf_from_db(&self.database, name)?, k_buf)
                }
                (StorageWriteBatch::InMemory(b), ColumnFamily::InMemory(name)) => {
                    b.delete_cf(name, k_buf)
                }

                _ => Err(TypedStoreError::RocksDBError(
                    "typed store invariant violation".to_string(),
                ))?,
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Deletes a range of keys between `from` (inclusive) and `to` (non-inclusive)
    /// by writing a range delete tombstone in the db map
    /// If the DBMap is configured with ignore_range_deletions set to false,
    /// the effect of this write will be visible immediately i.e. you won't
    /// see old values when you do a lookup or scan. But if it is configured
    /// with ignore_range_deletions set to true, the old value are visible until
    /// compaction actually deletes them which will happen sometime after. By
    /// default ignore_range_deletions is set to true on a DBMap (unless it is
    /// overridden in the config), so please use this function with caution
    pub fn schedule_delete_range<K: Serialize, V>(
        &mut self,
        db: &DBMap<K, V>,
        from: &K,
        to: &K,
    ) -> Result<(), TypedStoreError> {
        if !Arc::ptr_eq(&db.db, &self.database) {
            return Err(TypedStoreError::CrossDBBatch);
        }

        let from_buf = be_fix_int_ser(from);
        let to_buf = be_fix_int_ser(to);

        if let StorageWriteBatch::Rocks(b) = &mut self.batch {
            b.delete_range_cf(&rocks_cf_from_db(&self.database, db.cf_name())?, from_buf, to_buf);
        }
        Ok(())
    }

    /// inserts a range of (key, value) pairs given as an iterator
    pub fn insert_batch<J: Borrow<K>, K: Serialize, U: Borrow<V>, V: Serialize>(
        &mut self,
        db: &DBMap<K, V>,
        new_vals: impl IntoIterator<Item = (J, U)>,
    ) -> Result<&mut Self, TypedStoreError> {
        if !Arc::ptr_eq(&db.db, &self.database) {
            return Err(TypedStoreError::CrossDBBatch);
        }
        let mut total = 0usize;
        new_vals.into_iter().try_for_each::<_, Result<_, TypedStoreError>>(|(k, v)| {
            let k_buf = be_fix_int_ser(k.borrow());
            let v_buf = bcs::to_bytes(v.borrow()).map_err(typed_store_err_from_bcs_err)?;
            total += k_buf.len() + v_buf.len();
            if db.opts.log_value_hash {
                let key_hash = default_hash(&k_buf);
                let value_hash = default_hash(&v_buf);
                debug!(
                    "Insert to DB table: {:?}, key_hash: {:?}, value_hash: {:?}",
                    db.cf_name(),
                    key_hash,
                    value_hash
                );
            }
            match (&mut self.batch, &db.column_family) {
                (StorageWriteBatch::Rocks(b), ColumnFamily::Rocks(name)) => {
                    b.put_cf(&rocks_cf_from_db(&self.database, name)?, k_buf, v_buf)
                }
                (StorageWriteBatch::InMemory(b), ColumnFamily::InMemory(name)) => {
                    b.put_cf(name, k_buf, v_buf)
                }

                _ => Err(TypedStoreError::RocksDBError(
                    "typed store invariant violation".to_string(),
                ))?,
            }
            Ok(())
        })?;

        Ok(self)
    }

    pub fn partial_merge_batch<J: Borrow<K>, K: Serialize, U: Borrow<V>, V: Serialize>(
        &mut self,
        db: &DBMap<K, V>,
        new_vals: impl IntoIterator<Item = (J, U)>,
    ) -> Result<&mut Self, TypedStoreError> {
        if !Arc::ptr_eq(&db.db, &self.database) {
            return Err(TypedStoreError::CrossDBBatch);
        }
        new_vals.into_iter().try_for_each::<_, Result<_, TypedStoreError>>(|(k, v)| {
            let k_buf = be_fix_int_ser(k.borrow());
            let v_buf = bcs::to_bytes(v.borrow()).map_err(typed_store_err_from_bcs_err)?;
            match &mut self.batch {
                StorageWriteBatch::Rocks(b) => {
                    b.merge_cf(&rocks_cf_from_db(&self.database, db.cf_name())?, k_buf, v_buf)
                }
                _ => unimplemented!("merge operator is only implemented for RocksDB"),
            }
            Ok(())
        })?;
        Ok(self)
    }
}

impl<'a, K, V> Map<'a, K, V> for DBMap<K, V>
where
    K: Serialize + DeserializeOwned,
    V: Serialize + DeserializeOwned,
{
    type Error = TypedStoreError;

    #[instrument(level = "trace", skip_all, err)]
    fn contains_key(&self, key: &K) -> Result<bool, TypedStoreError> {
        let key_buf = be_fix_int_ser(key);
        let readopts = self.opts.readopts();
        Ok(self.db.key_may_exist_cf(&self.cf, &key_buf, &readopts)
            && self.db.get(&self.column_family, &key_buf, &readopts)?.is_some())
    }

    #[instrument(level = "trace", skip_all, err)]
    fn multi_contains_keys<J>(
        &self,
        keys: impl IntoIterator<Item = J>,
    ) -> Result<Vec<bool>, Self::Error>
    where
        J: Borrow<K>,
    {
        let values = self.multi_get_pinned(keys)?;
        Ok(values.into_iter().map(|v| v.is_some()).collect())
    }

    #[instrument(level = "trace", skip_all, err)]
    fn get(&self, key: &K) -> Result<Option<V>, TypedStoreError> {
        let key_buf = be_fix_int_ser(key);
        let res = self.db.get(&self.column_family, &key_buf, &self.opts.readopts())?;

        match res {
            Some(data) => {
                let value = bcs::from_bytes(&data).map_err(typed_store_err_from_bcs_err);
                if value.is_err() {
                    let key_hash = default_hash(&key_buf);
                    let value_hash = default_hash(&data);
                    tracing::error!(
                        "Failed to deserialize value from DB table {:?}, key_hash: {:?}, value_hash: {:?}, error: {:?}",
                        self.cf_name(),
                        key_hash,
                        value_hash,
                        value.as_ref().err().unwrap()
                    );
                    // TODO: debug-fatal
                }
                Ok(Some(value?))
            }
            None => Ok(None),
        }
    }

    #[instrument(level = "trace", skip_all, err)]
    fn insert(&self, key: &K, value: &V) -> Result<(), TypedStoreError> {
        let key_buf = be_fix_int_ser(key);
        let value_buf = bcs::to_bytes(value).map_err(typed_store_err_from_bcs_err)?;

        self.db.put_cf(&self.column_family, key_buf, value_buf)?;

        Ok(())
    }

    #[instrument(level = "trace", skip_all, err)]
    fn remove(&self, key: &K) -> Result<(), TypedStoreError> {
        let key_buf = be_fix_int_ser(key);
        self.db.delete_cf(&self.column_family, key_buf)?;

        Ok(())
    }

    /// Writes a range delete tombstone to delete all entries in the db map
    /// If the DBMap is configured with ignore_range_deletions set to false,
    /// the effect of this write will be visible immediately i.e. you won't
    /// see old values when you do a lookup or scan. But if it is configured
    /// with ignore_range_deletions set to true, the old value are visible until
    /// compaction actually deletes them which will happen sometime after. By
    /// default ignore_range_deletions is set to true on a DBMap (unless it is
    /// overridden in the config), so please use this function with caution
    #[instrument(level = "trace", skip_all, err)]
    fn schedule_delete_all(&self) -> Result<(), TypedStoreError> {
        let first_key = self.safe_iter().next().transpose()?.map(|(k, _v)| k);
        let last_key =
            self.reversed_safe_iter_with_bounds(None, None)?.next().transpose()?.map(|(k, _v)| k);
        if let Some((first_key, last_key)) = first_key.zip(last_key) {
            let mut batch = self.batch();
            batch.schedule_delete_range(self, &first_key, &last_key)?;
            batch.write()?;
        }
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.safe_iter().next().is_none()
    }

    fn safe_iter(&'a self) -> DbIterator<'a, (K, V)> {
        match &self.db.storage {
            Storage::Rocks(db) => {
                let db_iter = db
                    .underlying
                    .raw_iterator_cf_opt(&rocks_cf(db, &self.cf), self.opts.readopts());
                Box::new(SafeIter::new(self.cf.clone(), db_iter))
            }
            Storage::InMemory(db) => db.iterator(&self.cf, None, None, false),
        }
    }

    fn safe_iter_with_bounds(
        &'a self,
        lower_bound: Option<K>,
        upper_bound: Option<K>,
    ) -> DbIterator<'a, (K, V)> {
        let (lower_bound, upper_bound) = iterator_bounds(lower_bound, upper_bound);
        match &self.db.storage {
            Storage::Rocks(db) => {
                let readopts =
                    rocks_util::apply_range_bounds(self.opts.readopts(), lower_bound, upper_bound);
                let db_iter = db.underlying.raw_iterator_cf_opt(&rocks_cf(db, &self.cf), readopts);

                Box::new(SafeIter::new(self.cf.clone(), db_iter))
            }
            Storage::InMemory(db) => db.iterator(&self.cf, lower_bound, upper_bound, false),
        }
    }

    fn safe_range_iter(&'a self, range: impl RangeBounds<K>) -> DbIterator<'a, (K, V)> {
        let (lower_bound, upper_bound) = iterator_bounds_with_range(range);
        match &self.db.storage {
            Storage::Rocks(db) => {
                let readopts =
                    rocks_util::apply_range_bounds(self.opts.readopts(), lower_bound, upper_bound);
                let db_iter = db.underlying.raw_iterator_cf_opt(&rocks_cf(db, &self.cf), readopts);
                Box::new(SafeIter::new(self.cf.clone(), db_iter))
            }
            Storage::InMemory(db) => db.iterator(&self.cf, lower_bound, upper_bound, false),
        }
    }

    /// Returns a vector of values corresponding to the keys provided.
    #[instrument(level = "trace", skip_all, err)]
    fn multi_get<J>(
        &self,
        keys: impl IntoIterator<Item = J>,
    ) -> Result<Vec<Option<V>>, TypedStoreError>
    where
        J: Borrow<K>,
    {
        let results = self.multi_get_pinned(keys)?;
        let values_parsed: Result<Vec<_>, TypedStoreError> = results
            .into_iter()
            .map(|value_byte| match value_byte {
                Some(data) => {
                    Ok(Some(bcs::from_bytes(&data).map_err(typed_store_err_from_bcs_err)?))
                }
                None => Ok(None),
            })
            .collect();

        values_parsed
    }

    /// Convenience method for batch insertion
    #[instrument(level = "trace", skip_all, err)]
    fn multi_insert<J, U>(
        &self,
        key_val_pairs: impl IntoIterator<Item = (J, U)>,
    ) -> Result<(), Self::Error>
    where
        J: Borrow<K>,
        U: Borrow<V>,
    {
        let mut batch = self.batch();
        batch.insert_batch(self, key_val_pairs)?;
        batch.write()
    }

    /// Convenience method for batch removal
    #[instrument(level = "trace", skip_all, err)]
    fn multi_remove<J>(&self, keys: impl IntoIterator<Item = J>) -> Result<(), Self::Error>
    where
        J: Borrow<K>,
    {
        let mut batch = self.batch();
        batch.delete_batch(self, keys)?;
        batch.write()
    }

    /// Try to catch up with primary when running as secondary
    #[instrument(level = "trace", skip_all, err)]
    fn try_catch_up_with_primary(&self) -> Result<(), Self::Error> {
        if let Storage::Rocks(rocks) = &self.db.storage {
            rocks.underlying.try_catch_up_with_primary().map_err(typed_store_err_from_rocks_err)?;
        }
        Ok(())
    }
}

/// Opens a database with options, and a number of column families with individual options that are created if they do not exist.
#[instrument(level="debug", skip_all, fields(path = ?path.as_ref()), err)]
pub fn open_cf_opts<P: AsRef<Path>>(
    path: P,
    db_options: Option<rocksdb::Options>,
    opt_cfs: &[(&str, rocksdb::Options)],
) -> Result<Arc<Database>, TypedStoreError> {
    let path = path.as_ref();
    // Ensure parent directories exist before RocksDB tries to create the DB directory.
    // RocksDB's create_if_missing only creates the leaf directory, not parents.
    std::fs::create_dir_all(path).map_err(|e| {
        TypedStoreError::RocksDBError(format!("Failed to create directory {path:?}: {e}"))
    })?;
    // In the simulator, we intercept the wall clock in the test thread only. This causes problems
    // because rocksdb uses the simulated clock when creating its background threads, but then
    // those threads see the real wall clock (because they are not the test thread), which causes
    // rocksdb to panic. The `nondeterministic` macro evaluates expressions in new threads, which
    // resolves the issue.
    //
    // This is a no-op in non-simulator builds.

    let cfs = populate_missing_cfs(opt_cfs, path).map_err(typed_store_err_from_rocks_err)?;
    nondeterministic!({
        let mut options = db_options.unwrap_or_else(|| default_db_options().options);
        options.create_if_missing(true);
        options.create_missing_column_families(true);
        let rocksdb = {
            rocksdb::DBWithThreadMode::<MultiThreaded>::open_cf_descriptors(
                &options,
                path,
                cfs.into_iter().map(|(name, opts)| ColumnFamilyDescriptor::new(name, opts)),
            )
            .map_err(typed_store_err_from_rocks_err)?
        };
        Ok(Arc::new(Database::new(Storage::Rocks(RocksDB { underlying: rocksdb }))))
    })
}

/// Opens a database with options, and a number of column families with individual options that are created if they do not exist.
pub fn open_cf_opts_secondary<P: AsRef<Path>>(
    primary_path: P,
    secondary_path: Option<P>,
    db_options: Option<rocksdb::Options>,
    opt_cfs: &[(&str, rocksdb::Options)],
) -> Result<Arc<Database>, TypedStoreError> {
    let primary_path = primary_path.as_ref();
    let secondary_path = secondary_path.as_ref().map(|p| p.as_ref());
    // See comment above for explanation of why nondeterministic is necessary here.
    nondeterministic!({
        // Customize database options
        let mut options = db_options.unwrap_or_else(|| default_db_options().options);

        fdlimit::raise_fd_limit();
        // This is a requirement by RocksDB when opening as secondary
        options.set_max_open_files(-1);

        let mut opt_cfs: std::collections::HashMap<_, _> = opt_cfs.iter().cloned().collect();
        let cfs = rocksdb::DBWithThreadMode::<MultiThreaded>::list_cf(&options, primary_path)
            .ok()
            .unwrap_or_default();

        let default_db_options = default_db_options();
        // Add CFs not explicitly listed
        for cf_key in cfs.iter() {
            if !opt_cfs.contains_key(&cf_key[..]) {
                opt_cfs.insert(cf_key, default_db_options.options.clone());
            }
        }

        let primary_path = primary_path.to_path_buf();
        let secondary_path = secondary_path.map(|q| q.to_path_buf()).unwrap_or_else(|| {
            let mut s = primary_path.clone();
            s.pop();
            s.push("SECONDARY");
            s.as_path().to_path_buf()
        });

        let rocksdb = {
            options.create_if_missing(true);
            options.create_missing_column_families(true);
            let db = rocksdb::DBWithThreadMode::<MultiThreaded>::open_cf_descriptors_as_secondary(
                &options,
                &primary_path,
                &secondary_path,
                opt_cfs
                    .iter()
                    .map(|(name, opts)| ColumnFamilyDescriptor::new(*name, (*opts).clone())),
            )
            .map_err(typed_store_err_from_rocks_err)?;
            db.try_catch_up_with_primary().map_err(typed_store_err_from_rocks_err)?;
            db
        };
        Ok(Arc::new(Database::new(Storage::Rocks(RocksDB { underlying: rocksdb }))))
    })
}

// Drops a database if there is no other handle to it, with retries and timeout.
pub async fn safe_drop_db(path: PathBuf, timeout: Duration) -> Result<(), rocksdb::Error> {
    let mut backoff =
        backoff::ExponentialBackoff { max_elapsed_time: Some(timeout), ..Default::default() };
    loop {
        match rocksdb::DB::destroy(&rocksdb::Options::default(), path.clone()) {
            Ok(()) => return Ok(()),
            Err(err) => match backoff.next_backoff() {
                Some(duration) => tokio::time::sleep(duration).await,
                None => return Err(err),
            },
        }
    }
}

fn populate_missing_cfs(
    input_cfs: &[(&str, rocksdb::Options)],
    path: &Path,
) -> Result<Vec<(String, rocksdb::Options)>, rocksdb::Error> {
    let mut cfs = vec![];
    let input_cf_index: HashSet<_> = input_cfs.iter().map(|(name, _)| *name).collect();
    let existing_cfs =
        rocksdb::DBWithThreadMode::<MultiThreaded>::list_cf(&rocksdb::Options::default(), path)
            .ok()
            .unwrap_or_default();

    for cf_name in existing_cfs {
        if !input_cf_index.contains(&cf_name[..]) {
            cfs.push((cf_name, rocksdb::Options::default()));
        }
    }
    cfs.extend(input_cfs.iter().map(|(name, opts)| (name.to_string(), (*opts).clone())));
    Ok(cfs)
}

fn default_hash(value: &[u8]) -> Digest<32> {
    let mut hasher = fastcrypto::hash::Blake2b256::default();
    hasher.update(value);
    hasher.finalize()
}

/// Evaluates an expression in a new thread which will not be subject to interception of
/// getrandom(), clock_gettime(), etc.
#[cfg(msim)]
#[macro_export]
macro_rules! nondeterministic {
    ($expr: expr) => {
        std::thread::scope(move |s| s.spawn(move || $expr).join().unwrap())
    };
}

/// Simply evaluates expr.
#[cfg(not(msim))]
#[macro_export]
macro_rules! nondeterministic {
    ($expr: expr) => {
        $expr
    };
}
