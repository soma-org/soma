use bincode::Options;
use std::cmp::{max, min};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::AtomicU64;
use std::sync::{Mutex, Weak};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{sync::Arc, time::Duration};
use store::rocksdb::compaction_filter::Decision;
use store::rocksdb::LiveFile;
use store::{Map, TypedStoreError};
use types::storage::ObjectKey;

use crate::store_tables::{AuthorityPrunerTables, StoreObject};

#[derive(Clone)]
pub struct ObjectsCompactionFilter {
    db: Weak<AuthorityPrunerTables>,
}

impl ObjectsCompactionFilter {
    pub fn new(db: Arc<AuthorityPrunerTables>) -> Self {
        Self {
            db: Arc::downgrade(&db),
        }
    }
    pub fn filter(&mut self, key: &[u8], value: &[u8]) -> anyhow::Result<Decision> {
        let ObjectKey(object_id, version) = bincode::DefaultOptions::new()
            .with_big_endian()
            .with_fixint_encoding()
            .deserialize(key)?;
        let object: StoreObject = bcs::from_bytes(value)?;
        if matches!(object, StoreObject::Value(_)) {
            if let Some(db) = self.db.upgrade() {
                match db.object_tombstones.get(&object_id)? {
                    Some(gc_version) => {
                        if version <= gc_version {
                            return Ok(Decision::Remove);
                        }
                    }
                    None => {}
                }
            }
        }
        Ok(Decision::Keep)
    }
}
