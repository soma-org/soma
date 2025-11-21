use std::{
    collections::BTreeMap,
    fmt::{Debug, Formatter},
    sync::Arc,
};

use parking_lot::RwLock;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use types::committee::{AuthorityIndex, Stake};

use types::consensus::{
    block::Round,
    commit::{CommitIndex, CommitRange},
    context::Context,
    leader_scoring::ReputationScores,
};

use crate::dag_state::DagState;

/// The `LeaderSchedule` is responsible for producing the leader schedule across
/// an epoch. The leader schedule is subject to change periodically based on
/// calculated `ReputationScores` of the authorities.
#[derive(Clone)]
pub(crate) struct LeaderSchedule {
    pub leader_swap_table: Arc<RwLock<LeaderSwapTable>>,
    context: Arc<Context>,
    num_commits_per_schedule: u64,
}

impl LeaderSchedule {
    /// The window where the schedule change takes place in consensus. It represents
    /// number of committed sub dags.
    /// TODO: move this to protocol config
    #[cfg(not(msim))]
    const CONSENSUS_COMMITS_PER_SCHEDULE: u64 = 300;
    #[cfg(msim)]
    const CONSENSUS_COMMITS_PER_SCHEDULE: u64 = 10;

    pub(crate) fn new(context: Arc<Context>, leader_swap_table: LeaderSwapTable) -> Self {
        Self {
            context,
            num_commits_per_schedule: Self::CONSENSUS_COMMITS_PER_SCHEDULE,
            leader_swap_table: Arc::new(RwLock::new(leader_swap_table)),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_num_commits_per_schedule(mut self, num_commits_per_schedule: u64) -> Self {
        self.num_commits_per_schedule = num_commits_per_schedule;
        self
    }

    /// Restores the `LeaderSchedule` from storage. It will attempt to retrieve the
    /// last stored `ReputationScores` and use them to build a `LeaderSwapTable`.
    pub(crate) fn from_store(context: Arc<Context>, dag_state: Arc<RwLock<DagState>>) -> Self {
        let leader_swap_table = dag_state.read().recover_last_commit_info().map_or(
            LeaderSwapTable::default(),
            |(last_commit_ref, last_commit_info)| {
                LeaderSwapTable::new(
                    context.clone(),
                    last_commit_ref.index,
                    last_commit_info.reputation_scores,
                )
            },
        );

        tracing::info!(
            "LeaderSchedule recovered using {leader_swap_table:?}. There are {} committed subdags scored in DagState.",
            dag_state.read().scoring_subdags_count(),
        );

        // create the schedule
        Self::new(context, leader_swap_table)
    }

    pub(crate) fn commits_until_leader_schedule_update(
        &self,
        dag_state: Arc<RwLock<DagState>>,
    ) -> usize {
        let subdag_count = dag_state.read().scoring_subdags_count() as u64;

        assert!(
            subdag_count <= self.num_commits_per_schedule,
            "Committed subdags count exceeds the number of commits per schedule"
        );
        self.num_commits_per_schedule
            .checked_sub(subdag_count)
            .unwrap() as usize
    }

    /// Checks whether the dag state sub dags list is empty. If yes then that means that
    /// either (1) the system has just started and there is no unscored sub dag available (2) the
    /// schedule has updated - new scores have been calculated. Both cases we consider as valid cases
    /// where the schedule has been updated.
    pub(crate) fn leader_schedule_updated(&self, dag_state: &RwLock<DagState>) -> bool {
        dag_state.read().is_scoring_subdag_empty()
    }

    pub(crate) fn update_leader_schedule_v2(&self, dag_state: &RwLock<DagState>) {
        let (reputation_scores, last_commit_index) = {
            let dag_state = dag_state.read();
            let reputation_scores = dag_state.calculate_scoring_subdag_scores();

            let last_commit_index = dag_state.scoring_subdag_commit_range();

            (reputation_scores, last_commit_index)
        };

        {
            let mut dag_state = dag_state.write();
            // Clear scoring subdag as we have updated the leader schedule
            dag_state.clear_scoring_subdag();
            // Buffer score and last commit rounds in dag state to be persisted later
            dag_state.add_commit_info(reputation_scores.clone());
        }

        self.update_leader_swap_table(LeaderSwapTable::new(
            self.context.clone(),
            last_commit_index,
            reputation_scores.clone(),
        ));
    }

    pub(crate) fn elect_leader(&self, round: u32, leader_offset: u32) -> AuthorityIndex {
        cfg_if::cfg_if! {
            // TODO: we need to differentiate the leader strategy in tests, so for
            // some type of testing (ex sim tests) we can use the staked approach.
            if #[cfg(test)] {
                let leader = AuthorityIndex::new_for_test((round + leader_offset) % self.context.committee.size() as u32);
                let table = self.leader_swap_table.read();
                table.swap(leader, round, leader_offset).unwrap_or(leader)
            } else {
                let leader = self.elect_leader_stake_based(round, leader_offset);
                let table = self.leader_swap_table.read();
                table.swap(leader, round, leader_offset).unwrap_or(leader)
            }
        }
    }

    pub(crate) fn elect_leader_stake_based(&self, round: u32, offset: u32) -> AuthorityIndex {
        assert!((offset as usize) < self.context.committee.size());

        // To ensure that we elect different leaders for the same round (using
        // different offset) we are using the round number as seed to shuffle in
        // a weighted way the results, but skip based on the offset.
        // TODO: use a cache in case this proves to be computationally expensive
        let mut seed_bytes = [0u8; 32];
        seed_bytes[32 - 4..].copy_from_slice(&(round).to_le_bytes());
        let mut rng = StdRng::from_seed(seed_bytes);

        let choices = self
            .context
            .committee
            .authorities()
            .map(|(index, authority)| (index, authority.stake as f32))
            .collect::<Vec<_>>();

        *choices
            .choose_multiple_weighted(&mut rng, self.context.committee.size(), |item| item.1)
            .expect("Weighted choice error: stake values incorrect!")
            .skip(offset as usize)
            .map(|(index, _)| index)
            .next()
            .unwrap()
    }

    /// Atomically updates the `LeaderSwapTable` with the new provided one. Any
    /// leader queried from now on will get calculated according to this swap
    /// table until a new one is provided again.
    fn update_leader_swap_table(&self, table: LeaderSwapTable) {
        let read = self.leader_swap_table.read();
        let old_commit_range = &read.reputation_scores.commit_range;
        let new_commit_range = &table.reputation_scores.commit_range;

        // Unless LeaderSchedule is brand new and using the default commit range
        // of CommitRange(0..0) all future LeaderSwapTables should be calculated
        // from a CommitRange of equal length and immediately following the
        // preceding commit range of the old swap table.
        if *old_commit_range != CommitRange::default() {
            assert!(
                old_commit_range.is_next_range(new_commit_range)
                    && old_commit_range.is_equal_size(new_commit_range),
                "The new LeaderSwapTable has an invalid CommitRange. Old LeaderSwapTable {old_commit_range:?} vs new LeaderSwapTable {new_commit_range:?}",
            );
        }
        drop(read);

        tracing::trace!("Updating {table:?}");

        let mut write = self.leader_swap_table.write();
        *write = table;
    }
}

#[derive(Default, Clone)]
pub(crate) struct LeaderSwapTable {
    /// The list of `f` (by configurable stake) authorities with best scores as
    /// those defined by the provided `ReputationScores`. Those authorities will
    /// be used in the position of the `bad_nodes` on the final leader schedule.
    /// Storing the hostname & stake along side the authority index for debugging.
    pub(crate) good_nodes: Vec<(AuthorityIndex, String, Stake)>,

    /// The set of `f` (by configurable stake) authorities with the worst scores
    /// as those defined by the provided `ReputationScores`. Every time where such
    /// authority is elected as leader on the schedule, it will swapped by one of
    /// the authorities of the `good_nodes`.
    /// Storing the hostname & stake along side the authority index for debugging.
    pub(crate) bad_nodes: BTreeMap<AuthorityIndex, (String, Stake)>,

    /// Scores by authority in descending order, needed by other parts of the system
    /// for a consistent view on how each validator performs in consensus.
    pub(crate) reputation_scores_desc: Vec<(AuthorityIndex, u64)>,

    // The scores for which the leader swap table was built from. This struct is
    // used for debugging purposes. Once `good_nodes` & `bad_nodes` are identified
    // the `reputation_scores` are no longer needed functionally for the swap table.
    pub(crate) reputation_scores: ReputationScores,
}

impl LeaderSwapTable {
    // Constructs a new table based on the provided reputation scores. The
    // `swap_stake_threshold` designates the total (by stake) nodes that will be
    // considered as "bad" based on their scores and will be replaced by good nodes.
    // The `swap_stake_threshold` should be in the range of [0 - 33].
    pub(crate) fn new(
        context: Arc<Context>,
        commit_index: CommitIndex,
        reputation_scores: ReputationScores,
    ) -> Self {
        let swap_stake_threshold = context
            .protocol_config
            .consensus_bad_nodes_stake_threshold();
        Self::new_inner(
            context,
            swap_stake_threshold,
            commit_index,
            reputation_scores,
        )
    }

    fn new_inner(
        context: Arc<Context>,
        // Ignore linter warning in simtests.
        // TODO: maybe override protocol configs in tests for swap_stake_threshold, and call new().
        #[allow(unused_variables)] swap_stake_threshold: u64,
        commit_index: CommitIndex,
        reputation_scores: ReputationScores,
    ) -> Self {
        #[cfg(msim)]
        let swap_stake_threshold = 33;

        assert!(
            (0..=33).contains(&swap_stake_threshold),
            "The swap_stake_threshold ({swap_stake_threshold}) should be in range [0 - 33], out of bounds parameter detected"
        );

        // When reputation scores are disabled or at genesis, use the default value.
        if reputation_scores.scores_per_authority.is_empty() {
            return Self::default();
        }

        // Randomize order of authorities when they have the same score,
        // to avoid bias in the selection of the good and bad nodes.
        let mut seed_bytes = [0u8; 32];
        seed_bytes[28..32].copy_from_slice(&commit_index.to_le_bytes());
        let mut rng = StdRng::from_seed(seed_bytes);
        let mut authorities_by_score = reputation_scores.authorities_by_score(context.clone());
        assert_eq!(authorities_by_score.len(), context.committee.size());
        authorities_by_score.shuffle(&mut rng);
        // Stable sort the authorities by score descending. Order of authorities with the same score is preserved.
        authorities_by_score.sort_by(|a1, a2| a2.1.cmp(&a1.1));

        // Calculating the good nodes
        let good_nodes = Self::retrieve_first_nodes(
            context.clone(),
            authorities_by_score.iter(),
            swap_stake_threshold,
        )
        .into_iter()
        .collect::<Vec<(AuthorityIndex, String, Stake)>>();

        // Calculating the bad nodes
        // Reverse the sorted authorities to score ascending so we get the first
        // low scorers up to the provided stake threshold.
        let bad_nodes = Self::retrieve_first_nodes(
            context.clone(),
            authorities_by_score.iter().rev(),
            swap_stake_threshold,
        )
        .into_iter()
        .map(|(idx, hostname, stake)| (idx, (hostname, stake)))
        .collect::<BTreeMap<AuthorityIndex, (String, Stake)>>();

        good_nodes.iter().for_each(|(idx, hostname, stake)| {
            tracing::debug!(
                "Good node {hostname} with stake {stake} has score {} for {:?}",
                reputation_scores.scores_per_authority[idx.to_owned()],
                reputation_scores.commit_range,
            );
        });

        bad_nodes.iter().for_each(|(idx, (hostname, stake))| {
            tracing::debug!(
                "Bad node {hostname} with stake {stake} has score {} for {:?}",
                reputation_scores.scores_per_authority[idx.to_owned()],
                reputation_scores.commit_range,
            );
        });

        tracing::info!("Scores used for new LeaderSwapTable: {reputation_scores:?}");

        Self {
            good_nodes,
            bad_nodes,
            reputation_scores_desc: authorities_by_score,
            reputation_scores,
        }
    }

    /// Checks whether the provided leader is a bad performer and needs to be
    /// swapped in the schedule with a good performer. If not, then the method
    /// returns None. Otherwise the leader to swap with is returned instead. The
    /// `leader_round` & `leader_offset` represents the DAG slot on which the
    /// provided `AuthorityIndex` is a leader on and is used as a seed to random
    /// function in order to calculate the good node that will swap in that round
    /// with the bad node. We are intentionally not doing weighted randomness as
    /// we want to give to all the good nodes equal opportunity to get swapped
    /// with bad nodes and nothave one node with enough stake end up swapping
    /// bad nodes more frequently than the others on the final schedule.
    pub(crate) fn swap(
        &self,
        leader: AuthorityIndex,
        leader_round: Round,
        leader_offset: u32,
    ) -> Option<AuthorityIndex> {
        if self.bad_nodes.contains_key(&leader) {
            // TODO: Re-work swap for the multileader case
            assert!(
                leader_offset == 0,
                "Swap for multi-leader case not implemented yet."
            );
            let mut seed_bytes = [0u8; 32];
            seed_bytes[24..28].copy_from_slice(&leader_round.to_le_bytes());
            seed_bytes[28..32].copy_from_slice(&leader_offset.to_le_bytes());
            let mut rng = StdRng::from_seed(seed_bytes);

            let (idx, _hostname, _stake) = self
                .good_nodes
                .choose(&mut rng)
                .expect("There should be at least one good node available");

            tracing::trace!(
                "Swapping bad leader {} -> {} for round {}",
                leader,
                idx,
                leader_round
            );

            return Some(*idx);
        }
        None
    }

    /// Retrieves the first nodes provided by the iterator `authorities` until the
    /// `stake_threshold` has been reached. The `stake_threshold` should be between
    /// [0, 100] and expresses the percentage of stake that is considered the cutoff.
    /// It's the caller's responsibility to ensure that the elements of the `authorities`
    /// input is already sorted.
    fn retrieve_first_nodes<'a>(
        context: Arc<Context>,
        authorities: impl Iterator<Item = &'a (AuthorityIndex, u64)>,
        stake_threshold: u64,
    ) -> Vec<(AuthorityIndex, String, Stake)> {
        let mut filtered_authorities = Vec::new();

        let mut stake = 0;
        for &(authority_idx, _score) in authorities {
            stake += context.committee.stake_by_index(authority_idx);

            // If the total accumulated stake has surpassed the stake threshold
            // then we omit this last authority and we exit the loop. Important to
            // note that this means if the threshold is too low we may not have
            // any nodes returned.
            if stake > (stake_threshold * context.committee.total_stake()) / 100 as Stake {
                break;
            }

            let authority = context
                .committee
                .authority_by_authority_index(authority_idx)
                .expect("Index should be available");
            filtered_authorities.push((authority_idx, authority.hostname.clone(), authority.stake));
        }

        filtered_authorities
    }
}

impl Debug for LeaderSwapTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!(
            "LeaderSwapTable for {:?}, good_nodes: {:?} with stake: {}, bad_nodes: {:?} with stake: {}",
            self.reputation_scores.commit_range,
            self.good_nodes
                .iter()
                .map(|(idx, _hostname, _stake)| idx.to_owned())
                .collect::<Vec<AuthorityIndex>>(),
            self.good_nodes
                .iter()
                .map(|(_idx, _hostname, stake)| stake)
                .sum::<Stake>(),
            self.bad_nodes.keys().map(|idx| idx.to_owned()),
            self.bad_nodes
                .values()
                .map(|(_hostname, stake)| stake)
                .sum::<Stake>(),
        ))
    }
}
