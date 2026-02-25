// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    sync::Arc,
};

use crate::{
    committee::{Committee, EpochId},
    error::{SomaError, SomaResult},
    object::ObjectID,
};
use parking_lot::RwLock;
use store::{
    DBMapUtils, Map as _, nondeterministic,
    rocks::{DBMap, DBOptions, default_db_options},
    rocksdb::Options,
};

pub struct CommitteeStore {
    tables: CommitteeStoreTables,
    cache: RwLock<HashMap<EpochId, Arc<Committee>>>,
}

#[derive(DBMapUtils)]
pub struct CommitteeStoreTables {
    // Map from each epoch ID to the committee information.
    // #[default_options_override_fn = "committee_table_default_config"]
    pub committee_map: DBMap<EpochId, Committee>,
}

impl CommitteeStore {
    pub fn new(path: PathBuf, genesis_committee: &Committee, db_options: Option<Options>) -> Self {
        let tables = CommitteeStoreTables::open_tables_read_write(path, db_options, None);
        // );
        let mut store = Self { tables, cache: RwLock::new(HashMap::new()) };
        if store.database_is_empty().expect("CommitteeStore initialization failed") {
            store
                .init_genesis_committee(genesis_committee.clone())
                .expect("Init genesis committee data must not fail");
        }
        store
    }

    pub fn new_for_testing(genesis_committee: &Committee) -> Self {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("DB_{:?}", nondeterministic!(ObjectID::random())));
        Self::new(path, genesis_committee, None)
    }

    pub fn init_genesis_committee(&self, genesis_committee: Committee) -> SomaResult {
        assert_eq!(genesis_committee.epoch, 0);
        self.tables.committee_map.insert(&0, &genesis_committee)?;
        self.cache.write().insert(0, Arc::new(genesis_committee));
        Ok(())
    }

    pub fn insert_new_committee(&self, new_committee: &Committee) -> SomaResult {
        if let Some(old_committee) = self.get_committee(&new_committee.epoch)? {
            // If somehow we already have this committee in the store, they must be the same.
            assert_eq!(&*old_committee, new_committee);
        } else {
            self.tables.committee_map.insert(&new_committee.epoch, new_committee)?;
            self.cache.write().insert(new_committee.epoch, Arc::new(new_committee.clone()));
        }
        Ok(())
    }

    pub fn get_committee(&self, epoch_id: &EpochId) -> SomaResult<Option<Arc<Committee>>> {
        if let Some(committee) = self.cache.read().get(epoch_id) {
            return Ok(Some(Arc::clone(committee)));
        }
        let committee = self.tables.committee_map.get(epoch_id)?;
        let committee = committee.map(Arc::new);
        if let Some(ref committee) = committee {
            self.cache.write().insert(*epoch_id, committee.clone());
        }
        Ok(committee)
    }
    // // todo - make use of cache or remove this method

    // pub fn get_latest_committee(&self) -> Committee {
    //     self.tables
    //         .read()
    //         .committee_map
    //         .iter()
    //         .next_back()
    //         .map(|(_, committee)| committee.clone())
    //         .expect("Committee map should not be empty")
    // }
    // /// Return the committee specified by `epoch`. If `epoch` is `None`, return the latest committee.
    // // todo - make use of cache or remove this method
    // pub fn get_or_latest_committee(&self, epoch: Option<EpochId>) -> SomaResult<Committee> {
    //     Ok(match epoch {
    //         Some(epoch) => self
    //             .get_committee(&epoch)?
    //             .ok_or(SomaError::MissingCommitteeAtEpoch(epoch))
    //             .map(|c| Committee::clone(&*c))?,
    //         None => self.get_latest_committee(),
    //     })
    // }

    fn database_is_empty(&self) -> SomaResult<bool> {
        Ok(self.tables.committee_map.safe_iter().next().transpose()?.is_none())
    }
}
