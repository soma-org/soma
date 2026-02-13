// Tests for the WritebackCache and CachedVersionMap.
// These exercise the cache's object read/write paths, versioned lookups,
// negative caching, genesis insertion, and flush-to-store behavior.

use std::sync::Arc;

use tempfile::tempdir;
use types::object::{Object, ObjectID, Owner, Version};
use types::storage::ObjectKey;
use types::base::dbg_addr;

use crate::{
    authority_store::AuthorityStore,
    authority_store_tables::AuthorityPerpetualTables,
    cache::{ExecutionCacheReconfigAPI, ExecutionCacheWrite, ObjectCacheRead},
    cache::writeback_cache::WritebackCache,
};

use super::cache_types::CachedVersionMap;

/// Helper: create a WritebackCache backed by a temporary RocksDB store.
fn make_test_cache() -> WritebackCache {
    let dir = tempdir().unwrap();
    let perpetual_tables = Arc::new(AuthorityPerpetualTables::open(dir.path(), None));
    let store = AuthorityStore::open_no_genesis(perpetual_tables).unwrap();
    WritebackCache::new_for_tests(store)
}

// =============================================================================
// CachedVersionMap unit tests
// =============================================================================

#[test]
fn test_cached_version_map_insert_and_get() {
    let mut map = CachedVersionMap::<u64>::default();
    assert!(map.is_empty());

    map.insert(Version::from(1), 100);
    map.insert(Version::from(3), 300);
    map.insert(Version::from(5), 500);

    assert!(!map.is_empty());
    assert_eq!(map.get(&Version::from(1)), Some(&100));
    assert_eq!(map.get(&Version::from(3)), Some(&300));
    assert_eq!(map.get(&Version::from(5)), Some(&500));
    // Version 2 does not exist
    assert_eq!(map.get(&Version::from(2)), None);
}

#[test]
fn test_cached_version_map_highest_and_least() {
    let mut map = CachedVersionMap::<u64>::default();
    map.insert(Version::from(10), 1);
    map.insert(Version::from(20), 2);
    map.insert(Version::from(30), 3);

    let highest = map.get_highest().unwrap();
    assert_eq!(highest.0, Version::from(30));
    assert_eq!(highest.1, 3);

    let least = map.get_least().unwrap();
    assert_eq!(least.0, Version::from(10));
    assert_eq!(least.1, 1);
}

#[test]
fn test_cached_version_map_truncate() {
    let mut map = CachedVersionMap::<u64>::default();
    for i in 1..=10u64 {
        map.insert(Version::from(i), i * 100);
    }

    // Truncate to 3 -- only the highest 3 should remain
    map.truncate_to(3);

    assert_eq!(map.get(&Version::from(8)), Some(&800));
    assert_eq!(map.get(&Version::from(9)), Some(&900));
    assert_eq!(map.get(&Version::from(10)), Some(&1000));
    // Earlier versions should be gone
    assert_eq!(map.get(&Version::from(7)), None);
    assert_eq!(map.get(&Version::from(1)), None);
}

#[test]
fn test_cached_version_map_pop_oldest() {
    let mut map = CachedVersionMap::<u64>::default();
    map.insert(Version::from(1), 10);
    map.insert(Version::from(2), 20);
    map.insert(Version::from(3), 30);

    let popped = map.pop_oldest(&Version::from(1));
    assert_eq!(popped, Some(10));

    // After popping, version 1 is gone, version 2 is now the least
    assert_eq!(map.get(&Version::from(1)), None);
    let least = map.get_least().unwrap();
    assert_eq!(least.0, Version::from(2));
}

// =============================================================================
// WritebackCache object read/write tests
// =============================================================================

#[tokio::test]
async fn test_cache_write_and_read_object() {
    // Writing an object via write_object_entry_for_test should make it
    // readable via get_object and get_object_by_key.
    let cache = make_test_cache();
    let sender = dbg_addr(1);
    let obj_id = ObjectID::random();
    let obj = Object::with_id_owner_for_testing(obj_id, sender);

    cache.write_object_entry_for_test(obj.clone());

    // Read by ID (latest)
    let read = cache.get_object(&obj_id);
    assert!(read.is_some(), "Object should be readable after write");
    assert_eq!(read.unwrap().id(), obj_id);

    // Read by key (ID + version)
    let read_by_key = cache.get_object_by_key(&obj_id, obj.version());
    assert!(read_by_key.is_some(), "Object should be readable by key");
    assert_eq!(read_by_key.unwrap().version(), obj.version());
}

#[tokio::test]
async fn test_cache_read_nonexistent_object() {
    // Reading an object that was never written should return None.
    let cache = make_test_cache();
    let nonexistent_id = ObjectID::random();

    assert!(cache.get_object(&nonexistent_id).is_none());
    assert!(cache.get_object_by_key(&nonexistent_id, Version::from(1)).is_none());
}

#[tokio::test]
async fn test_cache_versioned_reads() {
    // Write multiple versions of the same object (simulating successive mutations).
    // The cache should return the correct version for each key query.
    let cache = make_test_cache();
    let sender = dbg_addr(1);
    let obj_id = ObjectID::random();

    // Version 1
    let obj_v1 = Object::with_id_owner_version_for_testing(
        obj_id,
        Version::from(1),
        Owner::AddressOwner(sender),
    );
    cache.write_object_entry_for_test(obj_v1.clone());

    // Version 5
    let obj_v5 = Object::with_id_owner_version_for_testing(
        obj_id,
        Version::from(5),
        Owner::AddressOwner(sender),
    );
    cache.write_object_entry_for_test(obj_v5.clone());

    // get_object should return latest (version 5)
    let latest = cache.get_object(&obj_id).unwrap();
    assert_eq!(latest.version(), Version::from(5));

    // get_object_by_key for version 1 should return v1
    let v1 = cache.get_object_by_key(&obj_id, Version::from(1));
    assert!(v1.is_some(), "Version 1 should still be in dirty set");
    assert_eq!(v1.unwrap().version(), Version::from(1));

    // get_object_by_key for version 5 should return v5
    let v5 = cache.get_object_by_key(&obj_id, Version::from(5));
    assert!(v5.is_some());
    assert_eq!(v5.unwrap().version(), Version::from(5));

    // A version that was never written should not be found
    let v3 = cache.get_object_by_key(&obj_id, Version::from(3));
    assert!(v3.is_none(), "Version 3 was never written");
}

#[tokio::test]
async fn test_cache_genesis_object_flush_to_store() {
    // insert_genesis_object writes directly to the DB (store), bypassing dirty cache.
    // After insertion, the object should be readable through the cache (via DB fallback).
    let cache = make_test_cache();
    let sender = dbg_addr(1);
    let obj_id = ObjectID::random();
    let obj = Object::with_id_owner_for_testing(obj_id, sender);

    cache.insert_genesis_object(obj.clone());

    // The object should be readable (cache falls back to store on miss)
    let read = cache.get_object(&obj_id);
    assert!(read.is_some(), "Genesis object should be readable via store fallback");
    assert_eq!(read.unwrap().id(), obj_id);

    // Also readable by key
    let read_by_key = cache.get_object_by_key(&obj_id, obj.version());
    assert!(read_by_key.is_some(), "Genesis object should be readable by key via store fallback");
}

#[tokio::test]
async fn test_cache_object_exists_by_key() {
    // object_exists_by_key should return true for written objects and false otherwise.
    let cache = make_test_cache();
    let sender = dbg_addr(1);
    let obj_id = ObjectID::random();
    let obj = Object::with_id_owner_for_testing(obj_id, sender);

    assert!(
        !cache.object_exists_by_key(&obj_id, obj.version()),
        "Should not exist before write"
    );

    cache.write_object_entry_for_test(obj.clone());

    assert!(
        cache.object_exists_by_key(&obj_id, obj.version()),
        "Should exist after write"
    );

    assert!(
        !cache.object_exists_by_key(&obj_id, Version::from(999)),
        "Should not exist at unwritten version"
    );
}

#[tokio::test]
async fn test_cache_latest_object_ref_or_tombstone() {
    // get_latest_object_ref_or_tombstone should return the latest ref.
    let cache = make_test_cache();
    let sender = dbg_addr(1);
    let obj_id = ObjectID::random();
    let obj = Object::with_id_owner_for_testing(obj_id, sender);

    assert!(
        cache.get_latest_object_ref_or_tombstone(obj_id).is_none(),
        "Should be None for unknown object"
    );

    cache.write_object_entry_for_test(obj.clone());

    let obj_ref = cache.get_latest_object_ref_or_tombstone(obj_id);
    assert!(obj_ref.is_some(), "Should return ref after write");
    let obj_ref = obj_ref.unwrap();
    assert_eq!(obj_ref.0, obj_id);
    assert_eq!(obj_ref.1, obj.version());
}

#[tokio::test]
async fn test_cache_multi_object_exists_by_key() {
    // multi_object_exists_by_key should correctly report existence for a batch.
    let cache = make_test_cache();
    let sender = dbg_addr(1);

    let id1 = ObjectID::random();
    let obj1 = Object::with_id_owner_for_testing(id1, sender);
    cache.write_object_entry_for_test(obj1.clone());

    let id2 = ObjectID::random();
    // id2 is never written

    let keys = vec![
        ObjectKey(id1, obj1.version()),
        ObjectKey(id2, Version::from(1)),
    ];

    let results = cache.multi_object_exists_by_key(&keys);
    assert_eq!(results.len(), 2);
    assert!(results[0], "obj1 should exist");
    assert!(!results[1], "obj2 should not exist");
}

#[tokio::test]
async fn test_cache_find_object_lt_or_eq_version() {
    // find_object_lt_or_eq_version should return the highest version <= bound.
    let cache = make_test_cache();
    let sender = dbg_addr(1);
    let obj_id = ObjectID::random();

    let obj_v2 = Object::with_id_owner_version_for_testing(
        obj_id,
        Version::from(2),
        Owner::AddressOwner(sender),
    );
    cache.write_object_entry_for_test(obj_v2.clone());

    let obj_v5 = Object::with_id_owner_version_for_testing(
        obj_id,
        Version::from(5),
        Owner::AddressOwner(sender),
    );
    cache.write_object_entry_for_test(obj_v5.clone());

    // Bound at version 5 should return v5
    let found = cache.find_object_lt_or_eq_version(obj_id, Version::from(5));
    assert!(found.is_some());
    assert_eq!(found.unwrap().version(), Version::from(5));

    // Bound at version 10 should also return v5 (highest <= 10)
    let found = cache.find_object_lt_or_eq_version(obj_id, Version::from(10));
    assert!(found.is_some());
    assert_eq!(found.unwrap().version(), Version::from(5));

    // Bound at version 3 should return v2 (highest <= 3)
    let found = cache.find_object_lt_or_eq_version(obj_id, Version::from(3));
    assert!(found.is_some());
    assert_eq!(found.unwrap().version(), Version::from(2));

    // Bound at version 1 should return None (no version <= 1)
    let found = cache.find_object_lt_or_eq_version(obj_id, Version::from(1));
    assert!(found.is_none());
}
