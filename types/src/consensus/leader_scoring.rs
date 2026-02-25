// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::block::{BlockAPI, BlockRef};
use super::commit::{CommitRange, CommittedSubDag};
use super::context::Context;
use super::stake_aggregator::{QuorumThreshold, StakeAggregator};
use crate::committee::AuthorityIndex;

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReputationScores {
    /// Score per authority. Vec index is the `AuthorityIndex`.
    pub scores_per_authority: Vec<u64>,
    // The range of commits these scores were calculated from.
    pub commit_range: CommitRange,
}

impl ReputationScores {
    pub fn new(commit_range: CommitRange, scores_per_authority: Vec<u64>) -> Self {
        Self { scores_per_authority, commit_range }
    }

    pub fn highest_score(&self) -> u64 {
        *self.scores_per_authority.iter().max().unwrap_or(&0)
    }

    // Returns the authorities index with score tuples.
    pub fn authorities_by_score(&self, context: Arc<Context>) -> Vec<(AuthorityIndex, u64)> {
        self.scores_per_authority
            .iter()
            .enumerate()
            .map(|(index, score)| {
                (
                    context
                        .committee
                        .to_authority_index(index)
                        .expect("Should be a valid AuthorityIndex"),
                    *score,
                )
            })
            .collect()
    }
}

/// ScoringSubdag represents the scoring votes in a collection of subdags across
/// multiple commits.
/// These subdags are "scoring" for the purposes of leader schedule change. As
/// new subdags are added, the DAG is traversed and votes for leaders are recorded
/// and scored along with stake. On a leader schedule change, finalized reputation
/// scores will be calculated based on the votes & stake collected in this struct.
pub struct ScoringSubdag {
    pub context: Arc<Context>,
    pub commit_range: Option<CommitRange>,
    // Only includes committed leaders for now.
    // TODO: Include skipped leaders as well
    pub leaders: HashSet<BlockRef>,
    // A map of votes to the stake of strongly linked blocks that include that vote
    // Note: Including stake aggregator so that we can quickly check if it exceeds
    // quourum threshold and only include those scores for certain scoring strategies.
    pub votes: BTreeMap<BlockRef, StakeAggregator<QuorumThreshold>>,
}

impl ScoringSubdag {
    pub fn new(context: Arc<Context>) -> Self {
        Self { context, commit_range: None, leaders: HashSet::new(), votes: BTreeMap::new() }
    }

    pub fn add_subdags(&mut self, committed_subdags: Vec<CommittedSubDag>) {
        for subdag in committed_subdags {
            // If the commit range is not set, then set it to the range of the first
            // committed subdag index.
            if let Some(commit_range) = &mut self.commit_range {
                commit_range.extend_to(subdag.commit_ref.index);
            } else {
                self.commit_range =
                    Some(CommitRange::new(subdag.commit_ref.index..=subdag.commit_ref.index));
            }

            // Add the committed leader to the list of leaders we will be scoring.
            tracing::trace!("Adding new committed leader {} for scoring", subdag.leader);
            self.leaders.insert(subdag.leader);

            // Check each block in subdag. Blocks are in order so we should traverse the
            // oldest blocks first
            for block in subdag.blocks {
                for ancestor in block.ancestors() {
                    // Weak links may point to blocks with lower round numbers
                    // than strong links.
                    if ancestor.round != block.round().saturating_sub(1) {
                        continue;
                    }

                    // If a blocks strong linked ancestor is in leaders, then
                    // it's a vote for leader.
                    if self.leaders.contains(ancestor) {
                        // There should never be duplicate references to blocks
                        // with strong linked ancestors to leader.
                        tracing::trace!(
                            "Found a vote {} for leader {ancestor} from authority {}",
                            block.reference(),
                            block.author()
                        );
                        assert!(
                            self.votes.insert(block.reference(), StakeAggregator::new()).is_none(),
                            "Vote {block} already exists. Duplicate vote found for leader {ancestor}"
                        );
                    }

                    if let Some(stake) = self.votes.get_mut(ancestor) {
                        // Vote is strongly linked to a future block, so we
                        // consider this a distributed vote.
                        tracing::trace!(
                            "Found a distributed vote {ancestor} from authority {}",
                            ancestor.author
                        );
                        stake.add(block.author(), &self.context.committee);
                    }
                }
            }
        }
    }

    // Iterate through votes and calculate scores for each authority based on
    // distributed vote scoring strategy.
    pub fn calculate_distributed_vote_scores(&self) -> ReputationScores {
        let scores_per_authority = self.distributed_votes_scores();

        // TODO: Normalize scores
        ReputationScores::new(
            self.commit_range
                .clone()
                .expect("CommitRange should be set if calculate_scores is called."),
            scores_per_authority,
        )
    }

    /// This scoring strategy aims to give scores based on overall vote distribution.
    /// Instead of only giving one point for each vote that is included in 2f+1
    /// blocks. We give a score equal to the amount of stake of all blocks that
    /// included the vote.
    fn distributed_votes_scores(&self) -> Vec<u64> {
        let num_authorities = self.context.committee.size();
        let mut scores_per_authority = vec![0_u64; num_authorities];

        for (vote, stake_agg) in self.votes.iter() {
            let authority = vote.author;
            let stake = stake_agg.stake();
            tracing::trace!(
                "[{}] scores +{stake} reputation for {authority}!",
                self.context.own_index,
            );
            scores_per_authority[authority.value()] += stake;
        }
        scores_per_authority
    }

    pub fn scored_subdags_count(&self) -> usize {
        if let Some(commit_range) = &self.commit_range { commit_range.size() } else { 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.leaders.is_empty() && self.votes.is_empty() && self.commit_range.is_none()
    }

    pub fn clear(&mut self) {
        self.leaders.clear();
        self.votes.clear();
        self.commit_range = None;
    }
}
