use std::{cmp::Ordering, collections::VecDeque};

use types::object::Version;

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
        Self {
            values: VecDeque::new(),
        }
    }
}

impl<V> CachedVersionMap<V> {
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn insert(&mut self, version: Version, value: V) {
        if !self.values.is_empty() {
            let back = self.values.back().unwrap().0;
            assert!(
                back < version,
                "version must be monotonically increasing ({:?} < {:?})",
                back,
                version
            );
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
