use std::sync::Arc;

use crate::{dag_state::DagState, round_tracker::QuorumRound};
use parking_lot::RwLock;
use tracing::{debug, info};
use types::committee::{AuthorityIndex, Stake};
use types::consensus::{context::Context, leader_scoring::ReputationScores};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum AncestorState {
    Include,
    // Exclusion score is the value stored in this state
    Exclude(u64),
}

#[derive(Clone)]
struct AncestorInfo {
    state: AncestorState,
    // This will be set to the future clock round for which this ancestor state
    // will be locked.
    lock_until_round: u32,
}

impl AncestorInfo {
    fn new() -> Self {
        Self {
            state: AncestorState::Include,
            lock_until_round: 0,
        }
    }

    fn is_locked(&self, current_clock_round: u32) -> bool {
        self.lock_until_round >= current_clock_round
    }

    fn set_lock(&mut self, lock_until_round: u32) {
        self.lock_until_round = lock_until_round;
    }
}

#[derive(Debug)]
struct StateTransition {
    authority_id: AuthorityIndex,
    // The authority propagation score taken from leader scoring.
    score: u64,
    // The stake of the authority that is transitioning state.
    stake: u64,
    // The authority high quorum round is the lowest round higher or equal to rounds
    // from a quorum of authorities
    high_quorum_round: u32,
}

pub(crate) struct AncestorStateManager {
    context: Arc<Context>,
    dag_state: Arc<RwLock<DagState>>,
    state_map: Vec<AncestorInfo>,
    excluded_nodes_stake_threshold: u64,
    // This is the running total of ancestors by stake that have been marked
    // as excluded. This cannot exceed the excluded_nodes_stake_threshold
    total_excluded_stake: Stake,
    // This is the reputation scores that we use for leader election but we are
    // using it here as a signal for high quality block propagation as well.
    pub(crate) propagation_scores: ReputationScores,
}

impl AncestorStateManager {
    // This value is based on the production round rates of between 10-15 rounds per second
    // which means we will be locking state between 30-45 seconds.
    #[cfg(not(test))]
    const STATE_LOCK_CLOCK_ROUNDS: u32 = 450;
    #[cfg(test)]
    const STATE_LOCK_CLOCK_ROUNDS: u32 = 5;

    // Exclusion threshold is based on propagation (reputation) scores
    const SCORE_EXCLUSION_THRESHOLD_PERCENTAGE: u64 = 20;

    pub(crate) fn new(context: Arc<Context>, dag_state: Arc<RwLock<DagState>>) -> Self {
        let state_map = vec![AncestorInfo::new(); context.committee.size()];

        // Note: this value cannot be greater than the threshold used in leader
        // schedule to identify bad nodes.
        let excluded_nodes_stake_threshold_percentage = 2 * context
            .protocol_config
            .consensus_bad_nodes_stake_threshold()
            / 3;

        let excluded_nodes_stake_threshold = (excluded_nodes_stake_threshold_percentage
            * context.committee.total_stake())
            / 100 as Stake;

        Self {
            context,
            dag_state,
            state_map,
            excluded_nodes_stake_threshold,
            // All ancestors start in the include state.
            total_excluded_stake: 0,
            propagation_scores: ReputationScores::default(),
        }
    }

    pub(crate) fn set_propagation_scores(&mut self, scores: ReputationScores) {
        self.propagation_scores = scores;
    }

    pub(crate) fn get_ancestor_states(&self) -> Vec<AncestorState> {
        self.state_map.iter().map(|info| info.state).collect()
    }

    /// Updates the state of all ancestors based on the latest scores and quorum rounds
    pub(crate) fn update_all_ancestors_state(&mut self, accepted_quorum_rounds: &[QuorumRound]) {
        // If round prober has not run yet and we don't have network quorum round,
        // it is okay because network_high_quorum_round will be zero and we will
        // include all ancestors until we get more information.
        let network_high_quorum_round =
            self.calculate_network_high_quorum_round(accepted_quorum_rounds);

        let current_clock_round = self.dag_state.read().threshold_clock_round();
        let low_score_threshold = (self.propagation_scores.highest_score()
            * Self::SCORE_EXCLUSION_THRESHOLD_PERCENTAGE)
            / 100;

        debug!(
            "Updating all ancestor state at round {current_clock_round} using network high quorum round of {network_high_quorum_round}, low score threshold of {low_score_threshold}, and exclude stake threshold of {}",
            self.excluded_nodes_stake_threshold
        );

        // We will first collect all potential state transitions as we need to ensure
        // we do not move more ancestors to EXCLUDE state than the excluded_nodes_stake_threshold
        // allows
        let mut exclude_to_include = Vec::new();
        let mut include_to_exclude = Vec::new();

        // If propagation scores are not ready because the first 300 commits have not
        // happened, this is okay as we will only start excluding ancestors after that
        // point in time.
        for (idx, score) in self
            .propagation_scores
            .scores_per_authority
            .iter()
            .enumerate()
        {
            let authority_id = self
                .context
                .committee
                .to_authority_index(idx)
                .expect("Index should be valid");
            let ancestor_info = &self.state_map[idx];
            let (_low, authority_high_quorum_round) = accepted_quorum_rounds[idx];
            let stake = self
                .context
                .committee
                .authority_by_authority_index(authority_id)
                .expect("Index should be valid")
                .stake;

            // Skip if locked
            if ancestor_info.is_locked(current_clock_round) {
                continue;
            }

            match ancestor_info.state {
                AncestorState::Include => {
                    if *score <= low_score_threshold {
                        include_to_exclude.push(StateTransition {
                            authority_id,
                            score: *score,
                            stake,
                            high_quorum_round: authority_high_quorum_round,
                        });
                    }
                }
                AncestorState::Exclude(_) => {
                    if *score > low_score_threshold
                        || authority_high_quorum_round >= network_high_quorum_round
                    {
                        exclude_to_include.push(StateTransition {
                            authority_id,
                            score: *score,
                            stake,
                            high_quorum_round: authority_high_quorum_round,
                        });
                    }
                }
            }
        }

        // We can apply the state change for all ancestors that are moving to the
        // include state as that will never cause us to exceed the excluded_nodes_stake_threshold
        for transition in exclude_to_include {
            self.apply_state_change(transition, AncestorState::Include, current_clock_round);
        }

        // Sort include_to_exclude by worst scores first as these should take priority
        // to be excluded if we can't exclude them all due to the excluded_nodes_stake_threshold
        include_to_exclude.sort_by_key(|t| t.score);

        // We can now apply state change for all ancestors that are moving to the exclude
        // state as we know there is no new stake that will be freed up by ancestor
        // state transition to include.
        for transition in include_to_exclude {
            // If the stake of this ancestor would cause us to exceed the threshold
            // we do nothing. The lock will continue to be unlocked meaning we can
            // try again immediately on the next call to update_all_ancestors_state
            if self.total_excluded_stake + transition.stake <= self.excluded_nodes_stake_threshold {
                let new_state = AncestorState::Exclude(transition.score);
                self.apply_state_change(transition, new_state, current_clock_round);
            } else {
                info!(
                    "Authority {} would have moved to {:?} state with score {} & quorum_round {} but we would have exceeded total excluded stake threshold. current_excluded_stake {} + authority_stake {} > exclude_stake_threshold {}",
                    transition.authority_id,
                    AncestorState::Exclude(transition.score),
                    transition.score,
                    transition.high_quorum_round,
                    self.total_excluded_stake,
                    transition.stake,
                    self.excluded_nodes_stake_threshold
                );
            }
        }
    }

    fn apply_state_change(
        &mut self,
        transition: StateTransition,
        new_state: AncestorState,
        current_clock_round: u32,
    ) {
        let block_hostname = &self
            .context
            .committee
            .authority_by_authority_index(transition.authority_id)
            .expect("Index should be valid")
            .hostname;
        let ancestor_info = &mut self.state_map[transition.authority_id.value()];

        match (ancestor_info.state, new_state) {
            (AncestorState::Exclude(_), AncestorState::Include) => {
                self.total_excluded_stake = self.total_excluded_stake
                    .checked_sub(transition.stake)
                    .expect("total_excluded_stake underflow - trying to subtract more stake than we're tracking as excluded");
            }
            (AncestorState::Include, AncestorState::Exclude(_)) => {
                self.total_excluded_stake += transition.stake;
            }
            _ => {
                panic!("Calls to this function should only be made for state transition.")
            }
        }

        ancestor_info.state = new_state;
        let lock_until_round = current_clock_round + Self::STATE_LOCK_CLOCK_ROUNDS;
        ancestor_info.set_lock(lock_until_round);

        info!(
            "Authority {} moved to {new_state:?} state with score {} & quorum_round {} and locked until round {lock_until_round}. Total excluded stake: {}",
            transition.authority_id,
            transition.score,
            transition.high_quorum_round,
            self.total_excluded_stake
        );
    }

    /// Calculate the network's high quorum round based on accepted rounds via
    /// RoundTracker.
    ///
    /// The authority high quorum round is the lowest round higher or equal to rounds  
    /// from a quorum of authorities. The network high quorum round is using the high
    /// quorum round of each authority as tracked by the [`RoundTracker`] and then
    /// finding the high quroum round of those high quorum rounds.
    fn calculate_network_high_quorum_round(&self, accepted_quorum_rounds: &[QuorumRound]) -> u32 {
        let committee = &self.context.committee;

        let mut high_quorum_rounds_with_stake = accepted_quorum_rounds
            .iter()
            .zip(committee.authorities())
            .map(|((_low, high), (_, authority))| (*high, authority.stake))
            .collect::<Vec<_>>();
        high_quorum_rounds_with_stake.sort();

        let mut total_stake = 0;
        let mut network_high_quorum_round = 0;

        for (round, stake) in high_quorum_rounds_with_stake.iter() {
            total_stake += stake;
            if total_stake >= self.context.committee.quorum_threshold() {
                network_high_quorum_round = *round;
                break;
            }
        }

        network_high_quorum_round
    }
}
