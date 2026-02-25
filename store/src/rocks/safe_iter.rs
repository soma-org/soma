// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;

use bincode::Options;
use rocksdb::{DBWithThreadMode, Direction, MultiThreaded};
use serde::de::DeserializeOwned;

use super::TypedStoreError;

/// An iterator over all key-value pairs in a data map.
pub struct SafeIter<'a, K, V> {
    _cf_name: String,
    db_iter: rocksdb::DBRawIteratorWithThreadMode<'a, DBWithThreadMode<MultiThreaded>>,
    _phantom: PhantomData<(K, V)>,
    direction: Direction,
    is_initialized: bool,
}

impl<'a, K: DeserializeOwned, V: DeserializeOwned> SafeIter<'a, K, V> {
    pub(super) fn new(
        cf_name: String,
        db_iter: rocksdb::DBRawIteratorWithThreadMode<'a, DBWithThreadMode<MultiThreaded>>,
    ) -> Self {
        Self {
            _cf_name: cf_name,
            db_iter,
            _phantom: PhantomData,
            direction: Direction::Forward,
            is_initialized: false,
        }
    }
}

impl<K: DeserializeOwned, V: DeserializeOwned> Iterator for SafeIter<'_, K, V> {
    type Item = Result<(K, V), TypedStoreError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Implicitly set iterator to the first entry in the column family if it hasn't been initialized
        // used for backward compatibility
        if !self.is_initialized {
            self.db_iter.seek_to_first();
            self.is_initialized = true;
        }
        if self.db_iter.valid() {
            let config = bincode::DefaultOptions::new().with_big_endian().with_fixint_encoding();
            let raw_key = self.db_iter.key().expect("Valid iterator failed to get key");
            let raw_value = self.db_iter.value().expect("Valid iterator failed to get value");

            let key = config.deserialize(raw_key).ok();
            let value = bcs::from_bytes(raw_value).ok();
            match self.direction {
                Direction::Forward => self.db_iter.next(),
                Direction::Reverse => self.db_iter.prev(),
            }
            key.and_then(|k| value.map(|v| Ok((k, v))))
        } else {
            match self.db_iter.status() {
                Ok(_) => None,
                Err(err) => Some(Err(TypedStoreError::RocksDBError(format!("{err}")))),
            }
        }
    }
}

/// An iterator with a reverted direction to the original. The `RevIter`
/// is hosting an iteration which is consuming in the opposing direction.
/// It's not possible to do further manipulation (ex re-reverse) to the
/// iterator.
pub struct SafeRevIter<'a, K, V> {
    iter: SafeIter<'a, K, V>,
}

impl<'a, K, V> SafeRevIter<'a, K, V> {
    pub(crate) fn new(mut iter: SafeIter<'a, K, V>, upper_bound: Option<Vec<u8>>) -> Self {
        iter.is_initialized = true;
        iter.direction = Direction::Reverse;
        match upper_bound {
            None => iter.db_iter.seek_to_last(),
            Some(key) => iter.db_iter.seek_for_prev(&key),
        }
        Self { iter }
    }
}

impl<K: DeserializeOwned, V: DeserializeOwned> Iterator for SafeRevIter<'_, K, V> {
    type Item = Result<(K, V), TypedStoreError>;

    /// Will give the next item backwards
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
