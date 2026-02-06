use std::{collections::BTreeSet, marker::PhantomData};

use crate::committee::{AuthorityIndex, Committee, Stake};

pub trait CommitteeThreshold {
    fn is_threshold(committee: &Committee, amount: Stake) -> bool;
    fn threshold(committee: &Committee) -> Stake;
}

#[derive(Default)]
pub struct QuorumThreshold;

#[cfg(test)]
#[derive(Default)]
pub struct ValidityThreshold;

impl CommitteeThreshold for QuorumThreshold {
    fn is_threshold(committee: &Committee, amount: Stake) -> bool {
        committee.reached_quorum(amount)
    }
    fn threshold(committee: &Committee) -> Stake {
        committee.quorum_threshold()
    }
}

#[cfg(test)]
impl CommitteeThreshold for ValidityThreshold {
    fn is_threshold(committee: &Committee, amount: Stake) -> bool {
        committee.reached_validity(amount)
    }
    fn threshold(committee: &Committee) -> Stake {
        committee.validity_threshold()
    }
}

#[derive(Default)]
pub struct StakeAggregator<T> {
    votes: BTreeSet<AuthorityIndex>,
    stake: Stake,
    _phantom: PhantomData<T>,
}

impl<T: CommitteeThreshold> StakeAggregator<T> {
    pub fn new() -> Self {
        Self { votes: Default::default(), stake: 0, _phantom: Default::default() }
    }

    /// Adds a vote for the specified authority index to the aggregator. It is guaranteed to count
    /// the vote only once for an authority. The method returns true when the required threshold has
    /// been reached.
    pub fn add(&mut self, vote: AuthorityIndex, committee: &Committee) -> bool {
        if self.votes.insert(vote) {
            self.stake += committee.stake_by_index(vote);
        }
        T::is_threshold(committee, self.stake)
    }

    /// Adds a vote for the specified authority index to the aggregator. It is guaranteed to count
    /// the vote only once for an authority.
    /// The method returns true when the vote comes from a new authority and is counted.
    pub fn add_unique(&mut self, vote: AuthorityIndex, committee: &Committee) -> bool {
        if self.votes.insert(vote) {
            self.stake += committee.stake_by_index(vote);
            return true;
        }
        false
    }

    pub fn stake(&self) -> Stake {
        self.stake
    }

    pub fn reached_threshold(&self, committee: &Committee) -> bool {
        T::is_threshold(committee, self.stake)
    }

    pub fn threshold(&self, committee: &Committee) -> Stake {
        T::threshold(committee)
    }

    pub fn clear(&mut self) {
        self.votes.clear();
        self.stake = 0;
    }
}
