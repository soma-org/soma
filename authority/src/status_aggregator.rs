// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::{collections::BTreeMap, sync::Arc};

use tracing::debug;
use types::{
    base::AuthorityName,
    committee::{AuthorityIndex, Committee, StakeUnit},
};

/// Aggregates various types of statuses from different authorities,
/// and the total stake of authorities that have inserted statuses.
/// Only keeps the latest status for each authority.
pub(crate) struct StatusAggregator<T> {
    committee: Arc<Committee>,
    total_votes: StakeUnit,
    statuses: BTreeMap<AuthorityName, T>,
}

impl<T> StatusAggregator<T> {
    pub(crate) fn new(committee: Arc<Committee>) -> Self {
        Self { committee, total_votes: 0, statuses: BTreeMap::new() }
    }

    /// Returns true if the status is inserted the first time for the authority.
    pub(crate) fn insert(&mut self, authority: AuthorityName, status: T) -> bool {
        let Some(index) = self.committee.authority_index(&authority) else {
            debug!("Authority {} not found in committee", authority);
            return false;
        };
        if self.statuses.insert(authority, status).is_some() {
            return false;
        }
        self.total_votes += self.committee.stake_by_index(AuthorityIndex(index));
        true
    }

    /// Returns the total stake of authorities that have inserted statuses.
    pub(crate) fn total_votes(&self) -> StakeUnit {
        self.total_votes
    }

    /// Returns all authorities that have inserted statuses.
    pub(crate) fn authorities(&self) -> Vec<AuthorityName> {
        self.statuses.keys().copied().collect()
    }

    /// Returns the status of each authority.
    #[cfg(test)]
    pub(crate) fn statuses(&self) -> &BTreeMap<AuthorityName, T> {
        &self.statuses
    }

    /// Returns the status of each authority.
    pub(crate) fn status_by_authority(&self) -> Vec<(AuthorityName, StakeUnit, T)>
    where
        T: Clone,
    {
        self.statuses
            .iter()
            .map(|(name, status)| {
                (
                    *name,
                    self.committee.stake_by_index(AuthorityIndex(
                        self.committee.authority_index(name).unwrap(),
                    )),
                    status.clone(),
                )
            })
            .collect()
    }

    pub(crate) fn reached_validity_threshold(&self) -> bool {
        self.total_votes >= self.committee.validity_threshold()
    }

    pub(crate) fn reached_quorum_threshold(&self) -> bool {
        self.total_votes >= self.committee.quorum_threshold()
    }
}
