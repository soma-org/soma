use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    sync::Arc,
};

use crate::{
    committee::{Committee, EpochId},
    error::{SomaError, SomaResult},
};
use parking_lot::RwLock;

pub struct CommitteeStore {
    tables: RwLock<CommitteeStoreTables>,
    cache: RwLock<HashMap<EpochId, Arc<Committee>>>,
}

// #[derive(DBMapUtils)]
pub struct CommitteeStoreTables {
    // Map from each epoch ID to the committee information.
    // #[default_options_override_fn = "committee_table_default_config"]
    committee_map: BTreeMap<EpochId, Committee>, //TODO: use DBMap / RocksDB
}

impl CommitteeStore {
    pub fn new(path: PathBuf, genesis_committee: &Committee) -> Self {
        // let tables = CommitteeStoreTables::open_tables_read_write(
        //     path,
        //     MetricConf::new("committee"),
        //     db_options,
        //     None,
        // );
        let mut store = Self {
            tables: RwLock::new(CommitteeStoreTables {
                committee_map: BTreeMap::new(),
            }),
            cache: RwLock::new(HashMap::new()),
        };
        if store.database_is_empty() {
            store
                .init_genesis_committee(genesis_committee.clone())
                .expect("Init genesis committee data must not fail");
        }
        store
    }

    // pub fn new_for_testing(genesis_committee: &Committee) -> Self {
    //     let dir = std::env::temp_dir();
    //     let path = dir.join(format!("DB_{:?}", nondeterministic!(ObjectID::random())));
    //     Self::new(path, genesis_committee, None)
    // }

    pub fn init_genesis_committee(&self, genesis_committee: Committee) -> SomaResult {
        assert_eq!(genesis_committee.epoch, 0);
        self.tables
            .write()
            .committee_map
            .insert(0, genesis_committee.clone());
        self.cache.write().insert(0, Arc::new(genesis_committee));
        Ok(())
    }

    pub fn insert_new_committee(&self, new_committee: Committee) -> SomaResult {
        if let Some(old_committee) = self.get_committee(&new_committee.epoch)? {
            // If somehow we already have this committee in the store, they must be the same.
            assert_eq!(*old_committee, new_committee);
        } else {
            let committee = new_committee.clone();
            self.tables
                .write()
                .committee_map
                .insert(new_committee.epoch, new_committee);
            self.cache
                .write()
                .insert(committee.epoch, Arc::new(committee.clone()));
        }
        Ok(())
    }

    pub fn get_committee(&self, epoch_id: &EpochId) -> SomaResult<Option<Arc<Committee>>> {
        if let Some(committee) = self.cache.read().get(epoch_id) {
            return Ok(Some(Arc::clone(committee)));
        }
        let committee = self
            .tables
            .read()
            .committee_map
            .get(epoch_id)
            .map(|committee| Arc::new(committee.clone()));
        if let Some(ref committee) = committee {
            self.cache.write().insert(*epoch_id, Arc::clone(committee));
        }
        Ok(committee)
    }
    // todo - make use of cache or remove this method

    pub fn get_latest_committee(&self) -> Committee {
        self.tables
            .read()
            .committee_map
            .iter()
            .next_back()
            .map(|(_, committee)| committee.clone())
            .expect("Committee map should not be empty")
    }
    /// Return the committee specified by `epoch`. If `epoch` is `None`, return the latest committee.
    // todo - make use of cache or remove this method
    pub fn get_or_latest_committee(&self, epoch: Option<EpochId>) -> SomaResult<Committee> {
        Ok(match epoch {
            Some(epoch) => self
                .get_committee(&epoch)?
                .ok_or(SomaError::MissingCommitteeAtEpoch(epoch))
                .map(|c| Committee::clone(&*c))?,
            None => self.get_latest_committee(),
        })
    }

    fn database_is_empty(&self) -> bool {
        self.tables.read().committee_map.iter().next().is_none()
    }
}
